"""
PDF exam-paper ingestion.

Two engines, because real exam PDFs are messy:

* "text"   -- pull the text layer with PyMuPDF and regex out numbered questions
              with (A)/(B)/(C) options. Free and instant. Works on plain
              practice-test PDFs; falls apart on anything with real notation.
* "vision" -- render each page to a PNG and ask a vision model to transcribe it
              into strict JSON. Costs a handful of API calls per page, but it is
              the only thing that copes with AP papers, whose figures are vector
              drawings (invisible to get_images()) and whose equations extract as
              gibberish -- the real 2026 AP Physics 1 paper gives "h h > 2 1"
              where the page says "h2 > h1".
* "auto"   -- text first; fall back to vision for any page the text parser fails
              on or that looks figure-heavy.

Every extraction is cached under `data/cache/`, keyed by (pdf, engine, pages,
dpi), so you pay the vision cost for a given paper exactly once. Rendered page
PNGs are kept beside the cache and attached to questions that need a figure.

Answer keys: exam PDFs usually ship without one. Extract first
(`--extract-only`), fill in the key by hand or point `--answer-key` at a file,
then run the benchmark. Questions with no key are still asked and logged, but
they are reported as `ungraded` instead of silently counting as wrong.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import ModelConfig, PDFConfig, QuestionSetConfig, RunConfig
from .schemas import LETTERS, MCQuestion

# ---------------------------------------------------------------------------- #
# Regexes for the text engine
# ---------------------------------------------------------------------------- #
# "12. " / "12) " at the start of a line -> a new question
_QUESTION_START = re.compile(r"(?m)^[ \t]*(\d{1,3})[.)][ \t]+")
# "(A) foo" / "A. foo" / "A) foo" -> an option, captured up to the next option
_OPTION = re.compile(
    r"(?m)^[ \t]*[\(\[]?([A-E])[\)\].][ \t]+(.*?)(?=(?:^[ \t]*[\(\[]?[A-E][\)\].][ \t])|\Z)",
    re.DOTALL,
)
# Junk lines that show up on every page of a real exam paper.
_BOILERPLATE = re.compile(
    r"(?im)^\s*(?:visit college board.*|©.*|page \d+|\d+\s*$|GO ON TO THE NEXT PAGE.*|"
    r"STOP\s*$|END OF (?:EXAM|SECTION).*|AP .*(?:FREE-RESPONSE|MULTIPLE-CHOICE) QUESTIONS.*)$"
)
# Wording that marks a free-response prompt rather than a multiple-choice item.
_FRQ_HINT = re.compile(r"\b(derive|justify|sketch|explain your|show your work|calculate and)\b", re.I)

# The transcription contract for the vision engine.
_VISION_PROMPT = """You are digitizing ONE page of a printed exam paper.

Return STRICT JSON, nothing else:
{"questions": [{"number": 12, "type": "mcq", "question": "full question text",
"choices": ["first option text", "second option text"], "answer": "B",
"has_figure": true}]}

Rules:
- Transcribe every complete question on this page. Preserve units and notation;
  write maths as readable plain text (e.g. "h2 > h1", "v = 3.0 m/s", "theta_0").
- "choices": the option TEXT only, in order, with no "(A)" prefixes.
  Use [] for a free-response question.
- "type": "mcq" if it offers lettered options, otherwise "frq".
- "answer": the correct option letter ONLY if this page literally prints it
  (an answer key or a marked solution). Otherwise null. Never guess.
- "has_figure": true if answering needs a diagram, graph or table on the page.
- Ignore instructions, directions, headers, footers, copyright lines, formula
  sheets and reference tables.
- A question continued from the previous page: transcribe the part you can see.
- No questions on the page -> {"questions": []}
"""


# ---------------------------------------------------------------------------- #
# Public entry point
# ---------------------------------------------------------------------------- #
def load_pdf_questions(
    pdf_path: str | Path,
    *,
    set_name: str,
    domain: str,
    pdf_cfg: PDFConfig,
    run_cfg: RunConfig,
    pages: Optional[str] = None,
    engine: Optional[str] = None,
    answer_key: Optional[str] = None,
    include_frq: bool = False,
    refresh: bool = False,
) -> List[MCQuestion]:
    """
    Turn a PDF exam paper into MCQuestion objects.

    `pages` is a 1-based spec like "1-12" or "2,5,7-9" (None = every page).
    `engine` overrides `pdf_cfg.engine`. `refresh=True` ignores the cache.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    engine = (engine or pdf_cfg.engine).lower()
    if engine not in {"auto", "text", "vision"}:
        raise ValueError(f"unknown pdf engine '{engine}' (auto | text | vision)")

    doc = _open_pdf(pdf_path)
    try:
        page_indexes = _resolve_pages(pages, doc.page_count)
        cache_path = _cache_path(pdf_path, engine, page_indexes, pdf_cfg)

        if cache_path.exists() and not refresh:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            print(f"  [pdf] reusing cached extraction: {cache_path}")
        else:
            raw = _extract(doc, page_indexes, pdf_path, engine, pdf_cfg, run_cfg)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
            print(f"  [pdf] cached extraction -> {cache_path}")
    finally:
        doc.close()

    questions = _to_questions(raw, set_name=set_name, domain=domain, source=str(pdf_path))

    if answer_key:
        key = load_answer_key(answer_key)
        applied = apply_answer_key(questions, key)
        print(f"  [pdf] answer key matched {applied}/{len(questions)} questions")

    if not include_frq:
        frq = [q for q in questions if q.question_type != "mcq"]
        if frq:
            print(f"  [pdf] dropping {len(frq)} free-response question(s) "
                  f"-- pass --include-frq to keep them (they stay ungraded)")
        questions = [q for q in questions if q.question_type == "mcq"]

    ungraded = [q for q in questions if not q.is_gradeable]
    if ungraded:
        print(f"  [pdf] {len(ungraded)}/{len(questions)} question(s) have no answer key "
              f"-- they will be asked but reported as 'ungraded'")
    return questions


def extract_to_json(
    pdf_path: str | Path,
    out_path: str | Path,
    *,
    set_name: str,
    domain: str,
    pdf_cfg: PDFConfig,
    run_cfg: RunConfig,
    pages: Optional[str] = None,
    engine: Optional[str] = None,
    refresh: bool = False,
) -> Path:
    """
    Extract and write an editable JSON question file (the `--extract-only` flow).

    Hand-edit the file to fix transcription slips and fill in `"answer"`, then
    feed it back with `--questions-file <that file> --subject "<domain>"`.
    """
    questions = load_pdf_questions(
        pdf_path, set_name=set_name, domain=domain, pdf_cfg=pdf_cfg, run_cfg=run_cfg,
        pages=pages, engine=engine, include_frq=True, refresh=refresh,
    )
    payload = [
        {
            # the printed question number (q012 -> 12), not the page number
            "number": _printed_number(q.question_id),
            "page": q.page_number,
            "type": q.question_type,
            "question": q.question,
            "choices": q.choices,
            # Letter or null -- fill this in by hand for papers with no key.
            "answer": q.correct_letter or None,
            "has_figure": q.needs_figure,
            "image_paths": q.image_paths,
            "source": q.source,
        }
        for q in questions
    ]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    n_mcq = sum(1 for q in questions if q.question_type == "mcq")
    print(f"\nwrote {len(payload)} question(s) ({n_mcq} multiple-choice) -> {out}")
    print('fill in each "answer" field, then rerun with '
          f'--questions-file {out} --subject "{domain}"')
    return out


# ---------------------------------------------------------------------------- #
# PDF plumbing
# ---------------------------------------------------------------------------- #
def _open_pdf(path: Path):
    """Open with PyMuPDF, handling the empty-password encryption exam PDFs use."""
    try:
        import pymupdf  # PyMuPDF >= 1.24 exposes the modern module name
    except ImportError:  # pragma: no cover -- older wheels only ship `fitz`
        try:
            import fitz as pymupdf  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PDF support needs PyMuPDF: pip install pymupdf"
            ) from exc

    doc = pymupdf.open(path)
    if doc.needs_pass and not doc.authenticate(""):
        raise ValueError(f"{path.name} is password-protected; cannot read it")
    return doc


def _resolve_pages(spec: Optional[str], page_count: int) -> List[int]:
    """'1-3,7' -> [0, 1, 2, 6]. None -> every page. Raises on nonsense."""
    if not spec:
        return list(range(page_count))
    out: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_s, hi_s = chunk.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
        else:
            lo = hi = int(chunk)
        if lo < 1 or hi < lo:
            raise ValueError(f"bad page range '{chunk}'")
        out.extend(range(lo - 1, min(hi, page_count)))
    if not out:
        raise ValueError(f"page spec '{spec}' selected no pages")
    return sorted(set(out))


def _cache_path(pdf_path: Path, engine: str, page_indexes: Sequence[int], cfg: PDFConfig) -> Path:
    """Cache file identified by pdf content hash + engine + pages + dpi."""
    digest = hashlib.sha1(
        f"{pdf_path.resolve()}|{pdf_path.stat().st_mtime_ns}|{engine}|"
        f"{cfg.dpi}|{min(page_indexes)}-{max(page_indexes)}|{len(page_indexes)}".encode()
    ).hexdigest()[:10]
    return Path(cfg.cache_dir) / f"{pdf_path.stem}__{engine}__{digest}.json"


def _figure_dir(pdf_path: Path, cfg: PDFConfig) -> Path:
    d = Path(cfg.cache_dir) / f"{pdf_path.stem}_pages"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _render_page(page, page_no: int, pdf_path: Path, cfg: PDFConfig) -> str:
    """Render one page to a cached PNG and return its path."""
    out = _figure_dir(pdf_path, cfg) / f"page_{page_no:03d}_{cfg.dpi}dpi.png"
    if not out.exists():
        page.get_pixmap(dpi=cfg.dpi).save(out)
    return str(out)


def _page_has_figure(page, cfg: PDFConfig) -> bool:
    """
    Exam figures are usually vector drawings, not embedded rasters, so count
    both. The threshold is deliberately generous -- a few stray vectors are just
    table rules or underlines.
    """
    return (
        len(page.get_images(full=True)) > 0
        or len(page.get_drawings()) >= cfg.vector_figure_threshold
    )


# ---------------------------------------------------------------------------- #
# Extraction
# ---------------------------------------------------------------------------- #
def _extract(
    doc,
    page_indexes: Sequence[int],
    pdf_path: Path,
    engine: str,
    pdf_cfg: PDFConfig,
    run_cfg: RunConfig,
) -> List[Dict[str, Any]]:
    """Run the chosen engine page by page and return raw question dicts."""
    vision_client = None
    raw: List[Dict[str, Any]] = []

    for idx in page_indexes:
        page = doc[idx]
        page_no = idx + 1
        has_figure = _page_has_figure(page, pdf_cfg)
        found: List[Dict[str, Any]] = []
        used = engine

        if engine in {"text", "auto"}:
            found = _extract_text_engine(page.get_text())
            if engine == "auto" and not found:
                used = "vision"     # text layer gave us nothing usable
            else:
                used = "text"

        if used == "vision" or engine == "vision":
            if vision_client is None:
                vision_client = _build_vision_client(pdf_cfg, run_cfg)
            found = _extract_vision_engine(
                page, page_no, pdf_path, pdf_cfg, vision_client
            )
            used = "vision"

        # Attach the page render where a figure is involved.
        attach = pdf_cfg.attach_figures
        for item in found:
            item.setdefault("has_figure", has_figure)
            wants_image = attach == "always" or (
                attach == "auto" and (item.get("has_figure") or has_figure)
            )
            item["image_paths"] = (
                [_render_page(page, page_no, pdf_path, pdf_cfg)] if wants_image else []
            )
            item["page"] = page_no
            item["engine"] = used

        print(f"  [pdf] page {page_no:>3}: {len(found)} question(s) via {used}"
              f"{' (figure)' if has_figure else ''}")
        raw.extend(found)

    return raw


def _extract_text_engine(page_text: str) -> List[Dict[str, Any]]:
    """Regex out numbered multiple-choice questions from a page's text layer."""
    text = _BOILERPLATE.sub("", page_text)
    starts = list(_QUESTION_START.finditer(text))
    if not starts:
        return []

    out: List[Dict[str, Any]] = []
    for i, match in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        block = text[match.end(): end].strip()
        if not block:
            continue

        options = _OPTION.findall(block)
        stem = _OPTION.split(block)[0].strip() if options else block
        stem = re.sub(r"\s*\n\s*", " ", stem).strip()
        if not stem:
            continue

        choices = [re.sub(r"\s+", " ", txt).strip() for _, txt in options]
        # No lettered options means it cannot be multiple choice, whatever the
        # wording looks like. _FRQ_HINT only sharpens the log message.
        is_mcq = len(choices) >= 2 and not _FRQ_HINT.search(block[:400])
        out.append({
            "number": int(match.group(1)),
            "type": "mcq" if is_mcq else "frq",
            "question": stem,
            "choices": choices,
            "answer": None,          # a plain question paper never prints the key
        })
    return out


def _extract_vision_engine(
    page, page_no: int, pdf_path: Path, pdf_cfg: PDFConfig, client
) -> List[Dict[str, Any]]:
    """Render the page and have a vision model transcribe it into JSON."""
    from .prompts import _data_url  # reuse the base64 data-URL helper

    png_path = _render_page(page, page_no, pdf_path, pdf_cfg)
    url = _data_url(png_path)
    if not url:
        print(f"  [pdf] page {page_no}: could not read rendered PNG, skipping")
        return []

    messages = [
        {"role": "system", "content": _VISION_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": f"Page {page_no} of {pdf_path.name}. Transcribe it."},
            {"type": "image_url", "image_url": {"url": url}},
        ]},
    ]
    result = client.complete(messages)
    if not result.ok:
        print(f"  [pdf] page {page_no}: extraction failed ({result.error})")
        return []

    payload = _loads_json_object(result.text)
    if not payload or not isinstance(payload.get("questions"), list):
        print(f"  [pdf] page {page_no}: extractor returned unusable JSON")
        return []

    out: List[Dict[str, Any]] = []
    for item in payload["questions"]:
        if not isinstance(item, dict) or not str(item.get("question", "")).strip():
            continue
        choices = [str(c).strip() for c in (item.get("choices") or []) if str(c).strip()]
        out.append({
            "number": item.get("number"),
            "type": (item.get("type") or ("mcq" if len(choices) >= 2 else "frq")).lower(),
            "question": str(item["question"]).strip(),
            "choices": choices,
            "answer": item.get("answer"),
            "has_figure": bool(item.get("has_figure")),
        })
    return out


def _build_vision_client(pdf_cfg: PDFConfig, run_cfg: RunConfig):
    """An LLMClient dedicated to transcription, so it gets retries too."""
    from .llm_client import LLMClient

    extractor = ModelConfig(
        name="pdf-extractor",
        litellm_model=pdf_cfg.extractor_model,
        provider="extractor",
        api_key_env=pdf_cfg.extractor_api_key_env,
        api_base=pdf_cfg.extractor_api_base,
        temperature=0.0,
        max_tokens=pdf_cfg.extractor_max_tokens,
        supports_json_mode=True,
        supports_vision=True,
    )
    print(f"  [pdf] vision extractor: {extractor.litellm_model}")
    return LLMClient(model_cfg=extractor, run_cfg=run_cfg)


def _loads_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object out of a reply that may be fenced or padded."""
    text = text.strip()
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1).strip())
    if "{" in text and "}" in text:
        candidates.append(text[text.index("{"): text.rindex("}") + 1])
    for candidate in candidates:
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded
        if isinstance(loaded, list):
            return {"questions": loaded}
    return None


# ---------------------------------------------------------------------------- #
# Raw dicts -> MCQuestion
# ---------------------------------------------------------------------------- #
def _to_questions(
    raw: Iterable[Dict[str, Any]], *, set_name: str, domain: str, source: str
) -> List[MCQuestion]:
    out: List[MCQuestion] = []
    seen: Dict[str, int] = {}
    for i, item in enumerate(raw):
        choices = [str(c) for c in (item.get("choices") or [])]
        correct = _letter_to_index(item.get("answer"), len(choices))
        # Papers repeat numbering ("2." inside question 4's text), and duplicate
        # ids would collide in the resume cache, so make them unique.
        label = _display_number(item, i)
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            label = f"{label}_{seen[label]}"
        try:
            out.append(MCQuestion(
                question_id=f"{set_name}::{label}",
                set_name=set_name,
                domain=domain,
                question=str(item.get("question", "")).strip(),
                choices=choices,
                correct_index=correct,
                source=f"{source}#p{item.get('page', '?')}",
                question_type=(item.get("type") or "mcq").lower(),
                needs_figure=bool(item.get("has_figure")),
                image_paths=[str(p) for p in (item.get("image_paths") or [])],
                page_number=item.get("page"),
            ))
        except Exception as exc:
            print(f"  [pdf] skipping extracted item {i}: {exc}")
    return out


def _printed_number(question_id: str) -> Optional[int]:
    """'ap_physics::q012' -> 12; 'ap_physics::i003' (no printed number) -> None."""
    tail = question_id.split("::")[-1]
    return int(tail[1:]) if tail.startswith("q") and tail[1:].isdigit() else None


def _display_number(item: Dict[str, Any], fallback: int) -> str:
    """Prefer the printed question number; fall back to extraction order."""
    number = item.get("number")
    if isinstance(number, int) and number > 0:
        return f"q{number:03d}"
    if isinstance(number, str) and number.strip().isdigit():
        return f"q{int(number):03d}"
    return f"i{fallback:03d}"


def _letter_to_index(answer: Any, n_choices: int) -> Optional[int]:
    """'B' or 1 -> 1. Returns None when there is no usable answer."""
    if answer is None or answer == "" or n_choices == 0:
        return None
    if isinstance(answer, bool):
        return None
    if isinstance(answer, int):
        return answer if 0 <= answer < n_choices else None
    match = re.search(r"[A-Ta-t]", str(answer))
    if not match:
        return None
    idx = LETTERS.index(match.group(0).upper())
    return idx if idx < n_choices else None


# ---------------------------------------------------------------------------- #
# Answer keys
# ---------------------------------------------------------------------------- #
def load_answer_key(spec: str) -> Dict[str, str]:
    """
    Accepts, in order of convenience:
      * inline:  "1:B,2:C,3:A"
      * .json:   {"1": "B", "2": "C"}  or  [{"number": 1, "answer": "B"}, ...]
      * .csv:    number,answer  (a header row is tolerated)
      * .txt:    one letter per line, in question order
    Returns {question_number_as_str: LETTER}.
    """
    path = Path(spec)
    if not path.exists():
        # Inline form.
        key: Dict[str, str] = {}
        for pair in re.split(r"[,\s]+", spec.strip()):
            if not pair:
                continue
            if ":" not in pair and "=" not in pair:
                raise ValueError(f"cannot read answer key '{spec}' (no such file, "
                                 f"and '{pair}' is not number:LETTER)")
            num, letter = re.split(r"[:=]", pair, maxsplit=1)
            key[num.strip().lstrip("qQ0") or num.strip()] = letter.strip().upper()[:1]
        return key

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            return {str(k).strip(): str(v).strip().upper()[:1] for k, v in data.items()}
        out: Dict[str, str] = {}
        for i, item in enumerate(data, start=1):
            if isinstance(item, dict):
                num = str(item.get("number", i))
                ans = item.get("answer")
                if ans:
                    out[num] = str(ans).strip().upper()[:1]
            elif isinstance(item, str) and item.strip():
                out[str(i)] = item.strip().upper()[:1]
        return out

    if path.suffix.lower() == ".csv":
        out = {}
        for row in csv.reader(text.splitlines()):
            if len(row) < 2:
                continue
            num, ans = row[0].strip(), row[1].strip()
            if not num or not re.fullmatch(r"\d+", num):
                continue        # header or blank line
            out[num] = ans.upper()[:1]
        return out

    # Plain text: one letter per line, in order.
    letters = [l.strip().upper()[:1] for l in text.splitlines() if l.strip()]
    return {str(i): l for i, l in enumerate(letters, start=1) if l in LETTERS}


def apply_answer_key(questions: Sequence[MCQuestion], key: Dict[str, str]) -> int:
    """
    Fill in `correct_index` from a key, matching on the printed question number
    (`sat_math::q012` -> "12"). Returns how many questions were matched.
    """
    normalised = {str(int(k)) if k.isdigit() else k.upper(): v for k, v in key.items()}
    matched = 0
    for pos, q in enumerate(questions, start=1):
        tail = q.question_id.split("::")[-1]
        digits = re.sub(r"\D", "", tail)
        candidates = [digits.lstrip("0") or digits, tail.upper(), str(pos)]
        letter = next((normalised[c] for c in candidates if c in normalised), None)
        if not letter or letter not in LETTERS:
            continue
        idx = LETTERS.index(letter)
        if idx < len(q.choices):
            q.correct_index = idx
            matched += 1
    return matched


# ---------------------------------------------------------------------------- #
# Config-driven convenience wrapper
# ---------------------------------------------------------------------------- #
def load_from_set_config(
    cfg: QuestionSetConfig, pdf_cfg: PDFConfig, run_cfg: RunConfig, refresh: bool = False
) -> List[MCQuestion]:
    """Used when a `"format": "pdf"` question set is declared in config.json."""
    return load_pdf_questions(
        cfg.local_path or "",
        set_name=cfg.name,
        domain=cfg.domain,
        pdf_cfg=pdf_cfg,
        run_cfg=run_cfg,
        pages=cfg.pages,
        engine=cfg.pdf_engine,
        answer_key=cfg.answer_key,
        include_frq=cfg.include_frq,
        refresh=refresh,
    )
