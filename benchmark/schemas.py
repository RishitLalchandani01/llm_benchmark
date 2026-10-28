"""
Typed data structures shared by every stage of the pipeline.

Keeping these in one place means the dataset loader, the LLM client and the
evaluator can never disagree about what a "question" or a "result" looks like.
"""

from __future__ import annotations

import re
import string
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# A..T is plenty: MMLU uses 4 options, AGIEval SAT sometimes uses 5.
LETTERS: str = string.ascii_uppercase


class MCQuestion(BaseModel):
    """One normalised multiple-choice question, whatever dataset it came from."""

    question_id: str            # stable id, e.g. "ap_physics::0042"
    set_name: str               # which question set config produced it
    domain: str                 # subject domain, e.g. "AP Physics"
    question: str               # the question stem
    choices: List[str] = Field(default_factory=list)  # option text, no "(A)" prefix
    correct_index: Optional[int] = None  # 0-based index; None = no answer key yet
    source: str = "unknown"     # dataset path/split/pdf page for provenance

    # --- PDF extras -------------------------------------------------------- #
    question_type: str = "mcq"            # "mcq" (gradeable) | "frq" (free response)
    needs_figure: bool = False            # answering requires a diagram/graph
    image_paths: List[str] = Field(default_factory=list)  # rendered page/figure PNGs
    page_number: Optional[int] = None     # 1-based page it was extracted from

    @field_validator("choices")
    @classmethod
    def _choice_count(cls, v: List[str]) -> List[str]:
        # Free-response questions legitimately have zero choices, so only the
        # upper bound is a hard error here; `is_gradeable` filters the rest.
        if len(v) > len(LETTERS):
            raise ValueError(f"too many choices ({len(v)}) to label with letters")
        return v

    @field_validator("correct_index")
    @classmethod
    def _index_non_negative(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("correct_index must be >= 0")
        return v

    def model_post_init(self, __context) -> None:  # pydantic v2 hook
        if self.correct_index is not None and self.correct_index >= len(self.choices):
            raise ValueError(
                f"correct_index {self.correct_index} out of range "
                f"for {len(self.choices)} choices ({self.question_id})"
            )

    @property
    def letters(self) -> List[str]:
        """['A', 'B', 'C', ...] sized to this question."""
        return list(LETTERS[: len(self.choices)])

    @property
    def correct_letter(self) -> str:
        """'' when no answer key is known -- such rows are reported, not scored."""
        return "" if self.correct_index is None else LETTERS[self.correct_index]

    @property
    def is_gradeable(self) -> bool:
        """A multiple-choice question is scoreable only with >=2 options and a key."""
        return (
            self.question_type == "mcq"
            and len(self.choices) >= 2
            and self.correct_index is not None
        )

    @property
    def is_frq(self) -> bool:
        """Free response: graded by the LLM judge, not by letter comparison."""
        return self.question_type == "frq"

    def render_choices(self) -> str:
        """Human/LLM readable option block."""
        return "\n".join(f"({letter}) {text}" for letter, text in zip(self.letters, self.choices))


class AnswerResponse(BaseModel):
    """
    The strict schema we demand back from the model.

    `extra="ignore"` so a chatty model that adds fields we did not ask for is
    still accepted -- we only actually need `answer`.
    """

    model_config = ConfigDict(extra="ignore")

    answer: str = Field(..., description="A single option letter, e.g. 'B'")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reasoning: Optional[str] = Field(default=None, max_length=4000)

    @field_validator("answer", mode="before")
    @classmethod
    def _normalise_letter(cls, v) -> str:
        """
        Accept the many ways models say 'B':
        'B', 'b', '(B)', 'B.', 'Answer: B', or even the integer 1 (0-based).
        """
        if isinstance(v, int):
            return LETTERS[v] if 0 <= v < len(LETTERS) else str(v)
        text = str(v).strip().upper()
        match = re.search(r"\b([A-T])\b", text) or re.search(r"([A-T])", text)
        if not match:
            raise ValueError(f"could not read an option letter out of {v!r}")
        return match.group(1)


class QuestionResult(BaseModel):
    """One graded (model, question) pair -- this becomes one row of the raw CSV."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    params_b: Optional[float] = None     # parameter count, for the scaling analysis
    provider: str = "unknown"
    set_name: str = ""
    domain: str = ""
    question_id: str = ""
    predicted_letter: Optional[str] = None
    correct_letter: str = ""
    is_correct: bool = False
    # Self-reported confidence, kept because calibration -- not accuracy -- is what
    # decides whether a compact model can be trusted to answer without escalation.
    confidence: Optional[float] = None
    latency_s: float = 0.0
    attempts: int = 0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    parse_ok: bool = False
    graded: bool = True                  # False when nothing could score this row

    # --- free-response scoring (empty for multiple-choice rows) ------------ #
    question_type: str = "mcq"
    score: Optional[float] = None        # points awarded by the judge
    max_score: Optional[float] = None    # points available
    judge_model: Optional[str] = None
    judge_reasoning: Optional[str] = None
    answer_text: Optional[str] = None    # the model's prose answer, truncated

    graded_by: str = "key"               # key | judge | none
    saw_image: bool = False              # was a figure actually sent to the model?
    needs_figure: bool = False           # did the question require a figure?
    status: str = "ok"                   # ok | parse_error | api_error | ungraded
    error: Optional[str] = None
    raw_response: Optional[str] = None   # truncated, for eyeballing failures
