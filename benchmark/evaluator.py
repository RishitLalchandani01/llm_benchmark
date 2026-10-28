"""
Answer parsing, grading and aggregation.

Parsing is deliberately layered: strict JSON first, then progressively looser
salvage attempts. Small models (Llama 3.2 1B) often return
```json\n{...}\n``` or "The answer is (C)." -- throwing those away would
under-report their accuracy and bias the whole scaling comparison.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

import pandas as pd
from pydantic import ValidationError

from .schemas import AnswerResponse, MCQuestion, QuestionResult

# ```json ... ``` fences
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
# the first {...} block in a blob of text
_FIRST_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)
# "answer": "B"  /  'answer': B
_ANSWER_FIELD = re.compile(r"""["']?answer["']?\s*[:=]\s*["']?\(?([A-T])\)?""", re.IGNORECASE)
# "The answer is C" / "Final answer: (C)"
_PROSE_ANSWER = re.compile(
    r"(?:answer|option|choice)\s*(?:is|:|=)?\s*\(?([A-T])\)?\b", re.IGNORECASE
)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_answer(raw: str, valid_letters: List[str]) -> Tuple[Optional[AnswerResponse], Optional[str]]:
    """
    Turn a raw completion into an AnswerResponse.
    Returns (answer, None) on success or (None, reason) on failure.
    """
    if not raw or not raw.strip():
        return None, "empty response"

    text = raw.strip()
    allowed = set(valid_letters)

    for candidate in _candidate_payloads(text):
        try:
            answer = AnswerResponse.model_validate(candidate)
        except (ValidationError, ValueError):
            continue
        if answer.answer in allowed:
            return answer, None
        return None, f"letter '{answer.answer}' not in {sorted(allowed)}"

    # Last resort: pull a letter straight out of the prose.
    for pattern in (_ANSWER_FIELD, _PROSE_ANSWER):
        match = pattern.search(text)
        if match and match.group(1).upper() in allowed:
            return AnswerResponse(answer=match.group(1).upper()), None

    # A bare single letter reply ("C") is also acceptable.
    stripped = text.strip().strip(".()[] ").upper()
    if len(stripped) == 1 and stripped in allowed:
        return AnswerResponse(answer=stripped), None

    return None, f"unparseable response: {text[:120]!r}"


def _candidate_payloads(text: str):
    """Yield dicts that might be the answer object, cheapest guess first."""
    # 1. the whole thing is valid JSON
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            yield loaded
    except json.JSONDecodeError:
        pass

    # 2. inside a markdown fence
    for block in _FENCE.findall(text):
        try:
            loaded = json.loads(block.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            yield loaded

    # 3. the first {...} substring anywhere in the reply
    for block in _FIRST_OBJECT.findall(text):
        try:
            loaded = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            yield loaded


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #
def grade(
    question: MCQuestion,
    predicted_letter: Optional[str],
    *,
    model_name: str,
    provider: str = "unknown",
    params_b: Optional[float] = None,
    latency_s: float = 0.0,
    attempts: int = 1,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    status: str = "ok",
    error: Optional[str] = None,
    raw_response: Optional[str] = None,
    saw_image: bool = False,
    confidence: Optional[float] = None,
    answer_text_chars: int = 4000,
    raw_response_chars: int = 600,
    score: Optional[float] = None,
    max_score: Optional[float] = None,
    judge_model: Optional[str] = None,
    judge_reasoning: Optional[str] = None,
    answer_text: Optional[str] = None,
) -> QuestionResult:
    """
    Build one result row.

    A failed API call counts as an incorrect answer (so flaky models cannot win
    by silence). A multiple-choice question with no answer key, or a
    free-response question the judge could not score, is marked `ungraded`: it
    is still asked and logged, but excluded from the scoring denominators.
    """
    if question.is_frq:
        gradeable = score is not None
        graded_by = "judge" if gradeable else "none"
    else:
        gradeable = question.is_gradeable
        graded_by = "key" if gradeable else "none"
    if status == "ok" and not gradeable:
        status = "ungraded"
    return QuestionResult(
        model_name=model_name,
        params_b=params_b,
        provider=provider,
        set_name=question.set_name,
        domain=question.domain,
        question_id=question.question_id,
        predicted_letter=predicted_letter,
        correct_letter=question.correct_letter,
        # Only meaningful for multiple choice; free-response rows carry `score`.
        is_correct=gradeable and not question.is_frq and bool(predicted_letter)
        and predicted_letter == question.correct_letter,
        latency_s=round(latency_s, 3),
        attempts=attempts,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        confidence=confidence,
        parse_ok=predicted_letter is not None,
        graded=gradeable,
        graded_by=graded_by,
        question_type=question.question_type,
        score=score,
        max_score=max_score,
        judge_model=judge_model,
        judge_reasoning=judge_reasoning,
        answer_text=(answer_text or "")[:answer_text_chars] or None,
        saw_image=saw_image,
        needs_figure=question.needs_figure,
        status=status,
        error=error,
        raw_response=(raw_response or "")[:raw_response_chars] or None,
    )


# --------------------------------------------------------------------------- #
# Aggregation / reporting
# --------------------------------------------------------------------------- #
def results_to_dataframe(results: List[QuestionResult]) -> pd.DataFrame:
    """Raw, one row per (model, question)."""
    return pd.DataFrame([r.model_dump() for r in results])


def _pct(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    100 * num / den, rounded, with a 0 denominator giving NaN rather than an
    error. to_numeric keeps pandas from choking on pd.NA in an integer column.
    """
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce").replace(0, float("nan"))
    return (num / den * 100).round(2)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Per (model, domain) summary: accuracy %, latency, failure counts.

    Accuracy is `n_correct / n_graded`. API failures stay in the denominator so
    a flaky model cannot look good by only answering what it managed to reach,
    but questions with no answer key (common with PDF papers) are excluded and
    counted in `n_ungraded` instead.
    """
    if raw.empty:
        return pd.DataFrame()

    # Free-response rows are scored 0..max_score by the judge; multiple-choice
    # rows are right or wrong. Keep the two denominators apart.
    raw = raw.copy()
    if "question_type" not in raw.columns:
        raw["question_type"] = "mcq"
    for col in ("score", "max_score"):
        if col not in raw.columns:
            raw[col] = pd.NA
    is_frq = raw["question_type"].astype(str).eq("frq")
    raw["_is_mcq_graded"] = (~is_frq) & raw["graded"].astype(bool)
    raw["_is_frq_graded"] = is_frq & raw["graded"].astype(bool)
    # "could not parse the answer letter" is meaningless for free response.
    raw["_mcq_parse_fail"] = (~is_frq) & (~raw["parse_ok"].astype(bool))
    frac = pd.to_numeric(raw["score"], errors="coerce") / pd.to_numeric(
        raw["max_score"], errors="coerce"
    ).replace(0, pd.NA)
    raw["_credit"] = frac.clip(0, 1).where(raw["_is_frq_graded"])

    grouped = raw.groupby(["model_name", "provider", "params_b", "domain"], dropna=False)
    summary = grouped.agg(
        n_questions=("is_correct", "size"),
        n_graded=("graded", lambda s: int(s.astype(bool).sum())),
        n_mcq_graded=("_is_mcq_graded", lambda s: int(s.sum())),
        n_correct=("is_correct", "sum"),
        n_frq_graded=("_is_frq_graded", lambda s: int(s.sum())),
        frq_points=("score", lambda s: pd.to_numeric(s, errors="coerce").sum()),
        frq_points_available=("max_score", lambda s: pd.to_numeric(s, errors="coerce").sum()),
        frq_credit=("_credit", lambda s: pd.to_numeric(s, errors="coerce").sum()),
        avg_latency_s=("latency_s", "mean"),
        p95_latency_s=("latency_s", lambda s: s.quantile(0.95)),
        avg_completion_tokens=("completion_tokens", "mean"),
        parse_failures=("_mcq_parse_fail", lambda s: int(s.sum())),
        api_errors=("status", lambda s: int((s == "api_error").sum())),
        figure_questions=("needs_figure", lambda s: int(s.astype(bool).sum())),
        figures_sent=("saw_image", lambda s: int(s.astype(bool).sum())),
        avg_attempts=("attempts", "mean"),
    ).reset_index()

    summary["n_ungraded"] = summary["n_questions"] - summary["n_graded"]
    # Guard against a set with nothing scoreable: 0/0 -> NaN, not a crash.
    summary["accuracy_pct"] = _pct(summary["n_correct"], summary["n_mcq_graded"])
    summary["frq_score_pct"] = _pct(summary["frq_points"], summary["frq_points_available"])
    # Headline number across both question types: a free-response answer worth
    # 3 of 4 points counts as 0.75 of an item.
    credit = summary["n_correct"].fillna(0) + summary["frq_credit"].fillna(0)
    summary["credit_pct"] = _pct(credit, summary["n_graded"])
    summary["avg_latency_s"] = summary["avg_latency_s"].round(3)
    summary["p95_latency_s"] = summary["p95_latency_s"].round(3)
    summary["avg_completion_tokens"] = summary["avg_completion_tokens"].round(1)
    summary["avg_attempts"] = summary["avg_attempts"].round(2)

    ordered = [
        "model_name", "provider", "params_b", "domain", "n_questions", "n_graded",
        "n_ungraded", "n_mcq_graded", "n_correct", "accuracy_pct",
        "n_frq_graded", "frq_points", "frq_points_available", "frq_score_pct",
        "credit_pct", "avg_latency_s", "p95_latency_s", "avg_completion_tokens",
        "parse_failures", "api_errors", "figure_questions", "figures_sent",
        "avg_attempts",
    ]
    return summary[ordered].sort_values(["domain", "credit_pct"], ascending=[True, False])


def overall_by_model(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Macro-average across domains -- this is the table that answers
    "does parameter count buy utility?" in one glance.
    """
    if summary.empty:
        return pd.DataFrame()
    out = (
        summary.groupby(["model_name", "params_b"], dropna=False)
        .agg(
            domains=("domain", "nunique"),
            questions=("n_questions", "sum"),
            graded=("n_graded", "sum"),
            mean_accuracy_pct=("accuracy_pct", "mean"),
            mean_frq_score_pct=("frq_score_pct", "mean"),
            mean_credit_pct=("credit_pct", "mean"),
            avg_latency_s=("avg_latency_s", "mean"),
        )
        .reset_index()
    )
    for col in ("mean_accuracy_pct", "mean_frq_score_pct", "mean_credit_pct"):
        out[col] = pd.to_numeric(out[col], errors="coerce").round(2)
    out["avg_latency_s"] = out["avg_latency_s"].round(3)
    # Credit points per billion parameters: a crude "bang per parameter" stat.
    # to_numeric keeps undisclosed sizes (params_b = null) as NaN instead of raising.
    params = pd.to_numeric(out["params_b"], errors="coerce")
    out["credit_per_b_params"] = (out["mean_credit_pct"] / params).round(3)
    return out.sort_values("mean_credit_pct", ascending=False)


def accuracy_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    """
    models x domains pivot of credit %, ready to paste into a paper.
    `credit_pct` equals plain MCQ accuracy for a run with no free response.
    """
    if summary.empty:
        return pd.DataFrame()
    return summary.pivot_table(
        index="model_name", columns="domain", values="credit_pct", aggfunc="first"
    ).round(1)
