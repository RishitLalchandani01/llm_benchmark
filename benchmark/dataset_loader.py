"""
Dataset engine.

Fetches multiple-choice questions from Hugging Face (or a local file) and
normalises every row into an `MCQuestion`, so the rest of the pipeline never
has to care that MMLU and AGIEval store their answers differently.

Supported `format` values
-------------------------
mmlu      : question:str, choices:list[str], answer:int          (cais/mmlu)
agieval   : query:str,    choices:list["(A) x"], gold:list[int]   (hails/agieval-*)
generic   : arbitrary columns, mapped through QuestionSetConfig.columns
local     : a .json / .jsonl file (plain shape, or the richer shape written by
            `--extract-only`, which may carry figures and null answers)
pdf       : an exam paper -- delegated to benchmark/pdf_loader.py

`build_cli_question_set()` turns the --questions-file/--subject command-line
options into a QuestionSetConfig, so a one-off file needs no config.json edit.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import PDFConfig, QuestionSetConfig, RunConfig
from .schemas import LETTERS, MCQuestion

# "(A) 42" / "A. 42" / "A) 42"  ->  "42"
_OPTION_PREFIX = re.compile(r"^\s*[\(\[]?([A-T])[\)\].:]\s*")
# AGIEval crams the options into the query text; strip that trailing block.
_ANSWER_CHOICES_BLOCK = re.compile(r"\n?\s*Answer Choices:.*\Z", re.IGNORECASE | re.DOTALL)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_question_set(
    cfg: QuestionSetConfig,
    limit_override: Optional[int] = None,
    seed: int = 1234,
    shuffle: bool = True,
    pdf_cfg: Optional[PDFConfig] = None,
    run_cfg: Optional[RunConfig] = None,
    refresh_extract: bool = False,
) -> List[MCQuestion]:
    """Load, normalise, shuffle and truncate one question set."""
    # PDFs have their own (cached, possibly vision-based) pipeline.
    if cfg.format == "pdf" or str(cfg.local_path or "").lower().endswith(".pdf"):
        from .pdf_loader import load_from_set_config

        questions = load_from_set_config(
            cfg,
            pdf_cfg or PDFConfig(),
            run_cfg or RunConfig(),
            refresh=refresh_extract,
        )
        return _shuffle_and_limit(questions, cfg, limit_override, seed, shuffle)

    rows, source = _fetch_rows(cfg)

    questions: List[MCQuestion] = []
    skipped = 0
    for i, row in enumerate(rows):
        try:
            q = _normalise_row(row, cfg, index=i, source=source)
        except Exception as exc:  # a malformed row should never kill a run
            skipped += 1
            if skipped <= 3:
                print(f"  [loader] skipping row {i} of {cfg.name}: {exc}")
            continue
        questions.append(q)

    # A JSON file may also take an external answer key and carry free-response
    # items (both happen when the file came out of `--extract-only`).
    if cfg.answer_key:
        from .pdf_loader import apply_answer_key, load_answer_key

        matched = apply_answer_key(questions, load_answer_key(cfg.answer_key))
        print(f"  [loader] answer key matched {matched}/{len(questions)} questions")

    if not cfg.include_frq:
        dropped = [q for q in questions if q.question_type != "mcq"]
        if dropped:
            print(f"  [loader] dropping {len(dropped)} free-response question(s) "
                  f"-- pass --include-frq to keep them")
        questions = [q for q in questions if q.question_type == "mcq"]

    ungraded = sum(1 for q in questions if not q.is_gradeable)
    if ungraded:
        print(f"  [loader] {ungraded}/{len(questions)} question(s) have no answer key "
              f"-- they will be asked but reported as 'ungraded'")

    print(
        f"  [loader] {cfg.name}: {len(questions)} questions parsed "
        f"({cfg.domain}, source={source}, skipped={skipped})"
    )
    return _shuffle_and_limit(questions, cfg, limit_override, seed, shuffle)


def _shuffle_and_limit(
    questions: List[MCQuestion],
    cfg: QuestionSetConfig,
    limit_override: Optional[int],
    seed: int,
    shuffle: bool,
) -> List[MCQuestion]:
    """Deterministic sampling so every model sees the SAME questions."""
    if shuffle:
        random.Random(seed).shuffle(questions)
    limit = limit_override if limit_override is not None else cfg.limit
    if limit is not None:
        questions = questions[:limit]
    print(f"  [loader] {cfg.name}: using {len(questions)} question(s)")
    return questions


def load_all(
    cfgs: Iterable[QuestionSetConfig],
    limit_override: Optional[int] = None,
    seed: int = 1234,
    shuffle: bool = True,
    pdf_cfg: Optional[PDFConfig] = None,
    run_cfg: Optional[RunConfig] = None,
    refresh_extract: bool = False,
) -> Dict[str, List[MCQuestion]]:
    """{set_name: [MCQuestion, ...]} for every requested set."""
    out: Dict[str, List[MCQuestion]] = {}
    for cfg in cfgs:
        out[cfg.name] = load_question_set(
            cfg, limit_override, seed, shuffle, pdf_cfg, run_cfg, refresh_extract
        )
    return out


def build_cli_question_set(
    path: str,
    subject: str,
    *,
    input_format: str = "auto",
    pages: Optional[str] = None,
    pdf_engine: Optional[str] = None,
    answer_key: Optional[str] = None,
    include_frq: bool = False,
    limit: Optional[int] = None,
    set_name: Optional[str] = None,
) -> QuestionSetConfig:
    """
    Build a question set from --questions-file / --subject, with no config.json
    entry needed. `input_format="auto"` decides from the file extension.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"questions file not found: {p}")

    fmt = input_format.lower()
    if fmt == "auto":
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            fmt = "pdf"
        elif suffix in {".json", ".jsonl"}:
            fmt = "local"
        else:
            raise ValueError(
                f"cannot tell what '{p.name}' is -- pass --input-format json|pdf"
            )
    elif fmt == "json":
        fmt = "local"
    if fmt not in {"local", "pdf"}:
        raise ValueError(f"--input-format must be auto, json or pdf (got '{input_format}')")

    return QuestionSetConfig(
        name=set_name or _slug(p.stem),
        domain=subject,
        format=fmt,
        local_path=str(p),
        limit=limit,
        pages=pages,
        pdf_engine=pdf_engine,
        answer_key=answer_key,
        include_frq=include_frq,
        enabled=True,
    )


def _slug(text: str) -> str:
    """'ap-physics-1 (2026)' -> 'ap_physics_1_2026' for use as a set name."""
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", text)).strip("_").lower() or "cli_set"


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def _fetch_rows(cfg: QuestionSetConfig) -> tuple[List[Dict[str, Any]], str]:
    """Return (rows, human-readable source string)."""
    if cfg.format == "local" or cfg.local_path:
        path = Path(cfg.local_path or "")
        if not path.exists():
            raise FileNotFoundError(f"local dataset not found: {path}")
        return _read_local(path), str(path)

    if not cfg.hf_path:
        raise ValueError(f"question set '{cfg.name}' needs hf_path or local_path")

    try:
        from datasets import load_dataset  # imported lazily: heavy dependency
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the `datasets` package is required for Hugging Face sets "
            "(pip install datasets)"
        ) from exc

    ds = load_dataset(cfg.hf_path, cfg.hf_subset, split=cfg.split)
    source = f"{cfg.hf_path}/{cfg.hf_subset or 'default'}:{cfg.split}"
    return [dict(row) for row in ds], source


def _read_local(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    return data["questions"] if isinstance(data, dict) else data


# --------------------------------------------------------------------------- #
# Normalising
#
# Each adapter returns MCQuestion kwargs (not a tuple), so the richer fields a
# PDF extraction carries -- figures, question type, a missing answer key -- can
# travel through the plain JSON path untouched.
# --------------------------------------------------------------------------- #
def _normalise_row(
    row: Dict[str, Any], cfg: QuestionSetConfig, index: int, source: str
) -> MCQuestion:
    fmt = cfg.format if cfg.format != "local" else (cfg.columns.get("format") or "generic")
    if fmt == "mmlu":
        fields = _from_mmlu(row)
    elif fmt == "agieval":
        fields = _from_agieval(row)
    else:
        fields = _from_generic(row, cfg.columns)

    number = row.get("number")
    if isinstance(number, int) and number > 0:
        qid = f"{cfg.name}::q{number:03d}"
    elif isinstance(number, str) and number.strip().isdigit():
        qid = f"{cfg.name}::q{int(number):03d}"
    else:
        qid = f"{cfg.name}::{index:05d}"

    return MCQuestion(
        question_id=qid,
        set_name=cfg.name,
        domain=cfg.domain,
        source=source,
        **fields,
    )


def _from_mmlu(row: Dict[str, Any]) -> Dict[str, Any]:
    """cais/mmlu: answer is already a 0-based integer index."""
    return {
        "question": str(row["question"]),
        "choices": [_strip_option_prefix(str(c)) for c in row["choices"]],
        "correct_index": int(row["answer"]),
    }


def _from_agieval(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    AGIEval SAT sets: the stem lives in `query` (with the options appended)
    and the answer is `gold`, a list of 0-based indices.
    """
    stem = _ANSWER_CHOICES_BLOCK.sub("", str(row.get("query", row.get("question", ""))))
    stem = re.sub(r"^\s*Question:\s*", "", stem).strip()

    gold = row.get("gold", row.get("answer"))
    if isinstance(gold, (list, tuple)):
        if not gold:
            raise ValueError("empty `gold` field")
        gold = gold[0]
    correct = int(gold) if not isinstance(gold, str) else LETTERS.index(gold.strip().upper())
    return {
        "question": stem,
        "choices": [_strip_option_prefix(str(c)) for c in row["choices"]],
        "correct_index": correct,
    }


def _from_generic(row: Dict[str, Any], columns: Dict[str, str]) -> Dict[str, Any]:
    """
    Map arbitrary column names, e.g.
      "columns": {"question": "prompt", "choices": "options", "answer": "label"}

    Also the shape written by `--extract-only`: a null/missing "answer" simply
    means "no key yet", and "type" / "has_figure" / "image_paths" pass through.
    """
    q_col = columns.get("question", "question")
    c_col = columns.get("choices", "choices")
    a_col = columns.get("answer", "answer")

    raw_choices = row.get(c_col) or []
    if isinstance(raw_choices, str):  # "A) x | B) y" style
        raw_choices = [p for p in re.split(r"\s*\|\s*", raw_choices) if p.strip()]
    choices = [_strip_option_prefix(str(c)) for c in raw_choices]

    answer = row.get(a_col)
    correct: Optional[int] = None
    if answer is not None and str(answer).strip() != "":
        if isinstance(answer, bool):
            correct = None
        elif isinstance(answer, int):
            correct = answer
        elif str(answer).strip().isdigit():
            correct = int(str(answer).strip())
        else:
            letter = str(answer).strip().upper()[0]
            correct = LETTERS.index(letter) if letter in LETTERS else None
    if correct is not None and not (0 <= correct < len(choices)):
        correct = None      # a key that does not fit the options is no key at all

    declared_type = str(row.get("type") or "").lower()
    question_type = declared_type if declared_type in {"mcq", "frq"} else (
        "mcq" if len(choices) >= 2 else "frq"
    )

    return {
        "question": str(row[q_col]),
        "choices": choices,
        "correct_index": correct,
        "question_type": question_type,
        "needs_figure": bool(row.get("has_figure") or row.get("needs_figure")),
        "image_paths": [str(x) for x in (row.get("image_paths") or [])],
        "page_number": row.get("page") if isinstance(row.get("page"), int) else None,
    }


def _strip_option_prefix(text: str) -> str:
    """'(B) 17' -> '17'. Leaves already-clean option text untouched."""
    return _OPTION_PREFIX.sub("", text).strip()
