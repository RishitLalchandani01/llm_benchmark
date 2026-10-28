"""
Configuration loading.

Design rules
------------
1. Nothing secret lives in config.json. A model entry names an *environment
   variable* (`api_key_env`) and the key is read from the process environment
   at call time. `${VAR}` placeholders anywhere in the JSON are also expanded,
   so `"api_base": "${OLLAMA_HOST}"` works too.
2. Everything is validated by pydantic, so a typo in config.json fails loudly
   at startup instead of halfway through a 2,000-call benchmark run.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# Matches ${SOME_ENV_VAR} inside any string value in config.json.
_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


# --------------------------------------------------------------------------- #
# Model / provider configuration
# --------------------------------------------------------------------------- #
class ModelConfig(BaseModel):
    """One model under test. `litellm_model` is the string litellm routes on."""

    model_config = ConfigDict(protected_namespaces=())

    name: str                                   # friendly label used in reports
    litellm_model: str                          # e.g. "gemini/gemini-2.5-flash"
    provider: str = "unknown"                   # gemini | xai | ollama | groq | together
    params_b: Optional[float] = None            # billions of params (None = undisclosed)
    api_key_env: Optional[str] = None           # NAME of the env var holding the key
    api_base: Optional[str] = None              # for self-hosted / Ollama
    temperature: float = 0.0                    # deterministic-ish grading
    max_tokens: int = 700
    supports_json_mode: bool = True             # send response_format={"type":"json_object"}
    supports_vision: bool = False               # can it be sent figure images?
    requests_per_minute: Optional[float] = None # client-side throttle (None = unlimited)
    extra_params: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True

    def resolved_api_key(self) -> Optional[str]:
        """Read the key from the environment. Never stored in config.json."""
        if not self.api_key_env:
            return None          # local Ollama needs no key
        key = os.getenv(self.api_key_env)
        if not key:
            warnings.warn(
                f"[config] env var {self.api_key_env} is not set -- "
                f"calls to '{self.name}' will probably fail with an auth error.",
                stacklevel=2,
            )
        return key


# --------------------------------------------------------------------------- #
# Question-set configuration
# --------------------------------------------------------------------------- #
class QuestionSetConfig(BaseModel):
    """
    Where a batch of questions comes from.

    format:
      "mmlu"    -> columns: question (str), choices (list[str]), answer (int)
      "agieval" -> columns: query (str), choices (list["(A) ..."]), gold (list[int])
      "local"   -> a .json/.jsonl file on disk (same shape as "generic")
      "pdf"     -> a PDF exam paper, parsed by benchmark/pdf_loader.py
      "generic" -> use `columns` to map arbitrary HF column names
    """

    name: str
    domain: str                                  # e.g. "AP Physics" -> report column
    format: str = "mmlu"
    hf_path: Optional[str] = None                # e.g. "cais/mmlu"
    hf_subset: Optional[str] = None              # e.g. "high_school_physics"
    split: str = "test"
    limit: Optional[int] = 50                    # questions to sample (None = all)
    local_path: Optional[str] = None             # .json/.jsonl (format "local") or .pdf
    columns: Dict[str, str] = Field(default_factory=dict)  # generic column mapping
    enabled: bool = True

    # --- PDF-only options (ignored by the other formats) ------------------- #
    pages: Optional[str] = None                  # "1-12" or "2,5,7-9"; None = all
    pdf_engine: Optional[str] = None             # auto | text | vision (None = pdf.engine)
    answer_key: Optional[str] = None             # path to a key file, or "1:B,2:C"
    include_frq: bool = False                    # keep free-response items (ungraded)


# --------------------------------------------------------------------------- #
# Run-wide settings
# --------------------------------------------------------------------------- #
class RunConfig(BaseModel):
    output_dir: str = "results"
    raw_csv: str = "raw_responses.csv"
    summary_csv: str = "summary.csv"
    cache_jsonl: str = "raw_cache.jsonl"   # append-only log enabling --resume
    max_workers: int = 4                   # threads PER MODEL
    max_retries: int = 5                   # attempts per request (incl. the first)
    initial_backoff_s: float = 2.0
    max_backoff_s: float = 60.0
    request_timeout_s: float = 120.0
    reask_on_parse_failure: bool = True    # one stricter retry if JSON is unreadable
    shuffle_questions: bool = True
    seed: int = 1234
    save_raw_text: bool = True             # keep a truncated copy of each completion
    answer_text_chars: int = 4000          # free-response answers kept for comparison
    raw_response_chars: int = 600          # truncated completion, for eyeballing failures


class PDFConfig(BaseModel):
    """
    How PDF exam papers are turned into questions.

    "vision" renders each page to a PNG and asks a vision model to transcribe it.
    That costs a few API calls per page but it is the only thing that survives
    real exam papers: AP PDFs draw their figures as vector graphics and their
    equations extract as gibberish ("h h > 2 1" for "h2 > h1").
    Extractions are cached, so you pay for a given PDF once.
    """

    engine: str = "auto"                      # auto | text | vision
    dpi: int = 150                            # page render resolution for vision/figures
    cache_dir: str = "data/cache"             # extracted JSON + rendered page PNGs
    attach_figures: str = "auto"              # auto | always | never
    vector_figure_threshold: int = 12         # >= this many vector ops => page has a figure
    extractor_model: str = "gemini/gemini-2.5-flash"
    extractor_api_key_env: Optional[str] = "GEMINI_API_KEY"
    extractor_api_base: Optional[str] = None
    extractor_max_tokens: int = 4000


class JudgeConfig(BaseModel):
    """
    Grading for free-response questions.

    Free-response answers cannot be checked with string equality, so a strong
    "judge" model scores each answer against the official scoring guidelines.
    That score is another model's opinion, not ground truth, so it is reported
    in its own columns (`frq_score_pct`) and never mixed into MCQ accuracy.
    Keep `model` different from the models under test to limit self-preference.
    """

    enabled: bool = False
    model: str = "gemini/gemini-2.5-pro"       # the grader; use a strong one
    api_key_env: Optional[str] = "GEMINI_API_KEY"
    api_base: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1200
    supports_vision: bool = True               # so it can see the figure too
    default_max_score: float = 1.0             # used when the rubric gives no total
    rubric_path: Optional[str] = None          # scoring guidelines: .pdf/.txt/.md/.json
    rubric_max_chars: int = 24000              # per-question rubric budget


class BenchmarkConfig(BaseModel):
    models: List[ModelConfig]
    question_sets: List[QuestionSetConfig]
    run: RunConfig = Field(default_factory=RunConfig)
    pdf: PDFConfig = Field(default_factory=PDFConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)

    # ---------------------------- loading ---------------------------------- #
    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(_expand_env(raw))

    # ---------------------------- selection -------------------------------- #
    def select_models(self, names: Optional[List[str]]) -> List[ModelConfig]:
        """`names=None` -> every enabled model. Unknown names raise."""
        pool = {m.name: m for m in self.models}
        if not names:
            return [m for m in self.models if m.enabled]
        chosen: List[ModelConfig] = []
        for n in names:
            if n not in pool:
                raise KeyError(f"unknown model '{n}'. available: {sorted(pool)}")
            chosen.append(pool[n])
        return chosen

    def select_question_sets(self, names: Optional[List[str]]) -> List[QuestionSetConfig]:
        """`names=None` -> every enabled question set. Unknown names raise."""
        pool = {q.name: q for q in self.question_sets}
        if not names:
            return [q for q in self.question_sets if q.enabled]
        chosen: List[QuestionSetConfig] = []
        for n in names:
            if n not in pool:
                raise KeyError(f"unknown question set '{n}'. available: {sorted(pool)}")
            chosen.append(pool[n])
        return chosen


def _expand_env(node: Any) -> Any:
    """Recursively replace ${VAR} placeholders with environment values."""
    if isinstance(node, dict):
        return {k: _expand_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env(v) for v in node]
    if isinstance(node, str):
        def sub(match: "re.Match[str]") -> str:
            var = match.group(1)
            value = os.getenv(var)
            if value is None:
                warnings.warn(f"[config] ${{{var}}} is not set; leaving it empty", stacklevel=2)
                return ""
            return value
        return _ENV_PLACEHOLDER.sub(sub, node)
    return node
