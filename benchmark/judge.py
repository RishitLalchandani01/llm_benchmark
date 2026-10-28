"""
LLM-as-judge grading for free-response questions.

Why this exists
---------------
A released AP paper is mostly free response: "derive an expression for...",
"justify your answer...". There is no letter to compare, so the only way to
score it automatically is to have a strong model read the response against the
official scoring guidelines and award points.

Treat the output as what it is -- a second model's opinion. It is reported in
its own columns (`frq_score_pct`, never folded into MCQ `accuracy_pct`), the
judge's reasoning is written into the raw CSV so every score is auditable by
hand, and the judge should be a different model from the ones under test.

Rubrics
-------
`--rubric` accepts the scoring guidelines as .pdf, .txt/.md, or .json. A PDF or
text file is split on "Question N" headings so each question is judged against
its own section; if no headings are found the whole text is used for every
question. JSON is a plain {"1": "rubric text", ...} mapping.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .config import JudgeConfig, ModelConfig, RunConfig
from .prompts import build_judge_messages
from .schemas import MCQuestion

# "Question 3" / "QUESTION 3:" / "Question 3 (Version J)" at the start of a line
_RUBRIC_SPLIT = re.compile(r"(?im)^\s*question\s+(\d{1,2})\b.*$")


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
class JudgeVerdict:
    """What the judge decided about one response."""

    __slots__ = ("score", "max_score", "reasoning", "error", "latency_s", "attempts")

    def __init__(
        self,
        score: Optional[float] = None,
        max_score: Optional[float] = None,
        reasoning: Optional[str] = None,
        error: Optional[str] = None,
        latency_s: float = 0.0,
        attempts: int = 0,
    ) -> None:
        self.score = score
        self.max_score = max_score
        self.reasoning = reasoning
        self.error = error
        self.latency_s = latency_s
        self.attempts = attempts

    @property
    def ok(self) -> bool:
        return self.error is None and self.score is not None

    @property
    def fraction(self) -> Optional[float]:
        if self.score is None or not self.max_score:
            return None
        return max(0.0, min(1.0, self.score / self.max_score))


# --------------------------------------------------------------------------- #
# The judge
# --------------------------------------------------------------------------- #
class Judge:
    """Wraps one LLMClient plus the rubric lookup. Safe to share across threads."""

    def __init__(self, judge_cfg: JudgeConfig, run_cfg: RunConfig) -> None:
        from .llm_client import LLMClient

        self.cfg = judge_cfg
        self.model_cfg = ModelConfig(
            name="judge",
            litellm_model=judge_cfg.model,
            provider="judge",
            api_key_env=judge_cfg.api_key_env,
            api_base=judge_cfg.api_base,
            temperature=judge_cfg.temperature,
            max_tokens=judge_cfg.max_tokens,
            supports_json_mode=True,
            supports_vision=judge_cfg.supports_vision,
        )
        self.client = LLMClient(model_cfg=self.model_cfg, run_cfg=run_cfg)
        self.rubrics: Dict[str, str] = (
            load_rubric(judge_cfg.rubric_path, judge_cfg.rubric_max_chars)
            if judge_cfg.rubric_path else {}
        )
        print(f"  [judge] {judge_cfg.model}"
              f"{f' | rubric sections: {sorted(self.rubrics)}' if self.rubrics else ' | no rubric'}")

    # ---------------------------------------------------------------- #
    def rubric_for(self, question: MCQuestion) -> Optional[str]:
        """Rubric section for this question number, else the catch-all, else None."""
        if not self.rubrics:
            return None
        # 'q012' -> '12'; 'q002_2' (a duplicate printed number) -> '2'
        tail = question.question_id.split("::")[-1]
        match = re.match(r"[a-zA-Z]?0*(\d+)", tail)
        number = match.group(1) if match else tail
        return self.rubrics.get(number) or self.rubrics.get("*")

    def max_score_for(self, question: MCQuestion, rubric: Optional[str]) -> float:
        """
        Points available: read "N points" out of the rubric if it says so,
        otherwise fall back to the configured default.
        """
        if rubric:
            match = re.search(r"(\d{1,2})\s*points?\s*(?:total|possible)", rubric, re.I)
            if match:
                return float(match.group(1))
        return float(self.cfg.default_max_score)

    # ---------------------------------------------------------------- #
    def score(self, question: MCQuestion, answer_text: str) -> JudgeVerdict:
        """Score one free-response answer. Never raises."""
        rubric = self.rubric_for(question)
        max_score = self.max_score_for(question, rubric)
        attach = bool(self.model_cfg.supports_vision and question.image_paths)
        last: Optional[str] = None

        for reask in (False, True):
            result = self.client.complete(build_judge_messages(
                question, answer_text, rubric=rubric, max_score=max_score,
                attach_images=attach, reask=reask,
            ))
            if not result.ok:
                return JudgeVerdict(error=result.error or "judge returned nothing",
                                    latency_s=result.latency_s, attempts=result.attempts)

            score, reported_max, reasoning, err = parse_verdict(result.text, max_score)
            if score is not None:
                return JudgeVerdict(
                    score=score, max_score=reported_max or max_score,
                    reasoning=reasoning, latency_s=result.latency_s,
                    attempts=result.attempts,
                )
            last = err
        return JudgeVerdict(error=last or "unparseable judge verdict")


# --------------------------------------------------------------------------- #
# Verdict parsing
# --------------------------------------------------------------------------- #
def parse_verdict(
    raw: str, expected_max: float
) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    """
    (score, max_score, reasoning, error) out of the judge's reply.
    Clamped to [0, max_score] so a hallucinated 12/1 cannot poison the average.
    """
    payload = _loads_json(raw)
    if payload is None:
        match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", raw or "")
        if not match:
            return None, None, None, f"unparseable judge reply: {(raw or '')[:120]!r}"
        payload = {"score": float(match.group(1)), "max_score": float(match.group(2))}

    try:
        score = float(payload["score"])
    except (KeyError, TypeError, ValueError):
        return None, None, None, f"judge reply had no numeric score: {str(payload)[:120]}"

    try:
        max_score = float(payload.get("max_score") or expected_max)
    except (TypeError, ValueError):
        max_score = expected_max
    if max_score <= 0:
        max_score = expected_max or 1.0

    reasoning = payload.get("reasoning")
    per_part = payload.get("per_part")
    if isinstance(per_part, list) and per_part:
        bits = [
            f"{p.get('part', '?')}:{p.get('awarded', '?')}"
            for p in per_part if isinstance(p, dict)
        ]
        reasoning = f"[{' '.join(bits)}] {reasoning or ''}".strip()

    clamped = max(0.0, min(float(score), max_score))
    note = str(reasoning)[:1500] if reasoning else None
    return clamped, max_score, note, None


def _loads_json(text: str) -> Optional[Dict[str, Any]]:
    """Tolerant JSON object parse (raw, fenced, or padded with prose)."""
    if not text:
        return None
    candidates = [text.strip()]
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
    return None


# --------------------------------------------------------------------------- #
# Rubric loading
# --------------------------------------------------------------------------- #
def load_rubric(path: str | Path, max_chars: int = 24000) -> Dict[str, str]:
    """
    {question_number: rubric_text}, plus "*" as a catch-all.

    .json -> used as the mapping directly
    .pdf  -> text extracted with PyMuPDF, then split on "Question N" headings
    .txt / .md -> same splitting
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"rubric not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("a JSON rubric must be an object: {\"1\": \"...\"}")
        return {str(k).strip(): str(v)[:max_chars] for k, v in data.items()}

    if path.suffix.lower() == ".pdf":
        from .pdf_loader import _open_pdf

        doc = _open_pdf(path)
        try:
            text = "\n".join(page.get_text() for page in doc)
        finally:
            doc.close()
    else:
        text = path.read_text(encoding="utf-8", errors="replace")

    sections = _split_rubric(text, max_chars)
    print(f"  [judge] rubric: {path.name} -> "
          f"{len(sections)} section(s) ({len(text)} chars of text)")
    return sections


def _split_rubric(text: str, max_chars: int) -> Dict[str, str]:
    """Split scoring guidelines on 'Question N' headings; fall back to one blob."""
    matches = list(_RUBRIC_SPLIT.finditer(text))
    if not matches:
        return {"*": text[:max_chars]}

    out: Dict[str, str] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        number = match.group(1).lstrip("0") or match.group(1)
        chunk = text[match.start(): end].strip()[:max_chars]
        # A question's guidelines can be spread over several headings; keep the
        # longest block we see for each number.
        if len(chunk) > len(out.get(number, "")):
            out[number] = chunk
    out["*"] = text[:max_chars]
    return out
