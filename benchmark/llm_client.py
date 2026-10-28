"""
Unified LLM client.

litellm gives us one `completion()` signature for Gemini, Grok (xAI), and Llama
served through Ollama / Groq / Together. This module adds the things litellm
does not do for us:

* a client-side requests-per-minute throttle (thread-safe),
* exponential backoff with jitter on retryable failures,
* automatic downgrade when a provider rejects `response_format` (JSON mode),
* a small, uniform return type so the evaluator does not import litellm.
"""

from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import ModelConfig, RunConfig

try:
    import litellm
    from litellm import completion as litellm_completion
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "litellm is required. Install the project deps: pip install -r requirements.txt"
    ) from exc

# Keep litellm quiet and predictable.
litellm.drop_params = True        # silently drop params a provider doesn't support
litellm.suppress_debug_info = True


# --------------------------------------------------------------------------- #
# Return type
# --------------------------------------------------------------------------- #
@dataclass
class CompletionResult:
    """Everything the evaluator needs from one API call."""

    text: str
    latency_s: float
    attempts: int
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
class RateLimiter:
    """
    Dead-simple thread-safe throttle: guarantees at least `60 / rpm` seconds
    between the starts of two requests. Good enough for free-tier limits and
    far easier to reason about than a token bucket.
    """

    def __init__(self, requests_per_minute: Optional[float]) -> None:
        self._interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            self._next_allowed = max(now, self._next_allowed) + self._interval
        if wait > 0:
            time.sleep(wait)


# --------------------------------------------------------------------------- #
# Error classification
# --------------------------------------------------------------------------- #
def _exception_names() -> Tuple[tuple, tuple]:
    """
    (retryable, fatal) exception tuples, resolved defensively because litellm
    renames/moves exception classes between versions.
    """
    def grab(*names: str) -> tuple:
        found = []
        for n in names:
            cls = getattr(litellm, n, None) or getattr(
                getattr(litellm, "exceptions", object()), n, None
            )
            if isinstance(cls, type):
                found.append(cls)
        return tuple(found)

    retryable = grab(
        "RateLimitError", "Timeout", "APIConnectionError", "ServiceUnavailableError",
        "InternalServerError", "APIError",
    )
    fatal = grab(
        "AuthenticationError", "PermissionDeniedError", "NotFoundError",
        "ContextWindowExceededError", "BadRequestError",
    )
    return retryable, fatal


_RETRYABLE, _FATAL = _exception_names()


def _is_retryable(exc: BaseException) -> bool:
    """Fatal wins over retryable (BadRequestError subclasses APIError sometimes)."""
    if _FATAL and isinstance(exc, _FATAL):
        # A 429 disguised as a BadRequest still deserves a retry.
        return "rate" in str(exc).lower() or "429" in str(exc)
    if _RETRYABLE and isinstance(exc, _RETRYABLE):
        return True
    text = str(exc).lower()
    return any(s in text for s in ("timeout", "timed out", "overloaded", "429", "503", "502"))


# --------------------------------------------------------------------------- #
# The client
# --------------------------------------------------------------------------- #
@dataclass
class LLMClient:
    """One client per model under test; safe to share across worker threads."""

    model_cfg: ModelConfig
    run_cfg: RunConfig
    limiter: RateLimiter = field(init=False)
    _json_mode: bool = field(init=False)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.limiter = RateLimiter(self.model_cfg.requests_per_minute)
        self._json_mode = self.model_cfg.supports_json_mode

    # ------------------------------------------------------------------ #
    def _call_kwargs(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assemble litellm kwargs, pulling the API key from the environment."""
        kwargs: Dict[str, Any] = {
            "model": self.model_cfg.litellm_model,
            "messages": messages,
            "temperature": self.model_cfg.temperature,
            "max_tokens": self.model_cfg.max_tokens,
            "timeout": self.run_cfg.request_timeout_s,
            **self.model_cfg.extra_params,
        }
        api_key = self.model_cfg.resolved_api_key()
        if api_key:
            kwargs["api_key"] = api_key
        if self.model_cfg.api_base:
            kwargs["api_base"] = self.model_cfg.api_base
        if self._json_mode:
            # Native structured output where the provider supports it.
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    # ------------------------------------------------------------------ #
    def complete(self, messages: List[Dict[str, Any]]) -> CompletionResult:
        """
        Call the model with retries. Never raises: failures come back as a
        CompletionResult with `error` set, so one bad question cannot abort a run.
        """
        backoff = self.run_cfg.initial_backoff_s
        last_error = "unknown error"
        started_total = time.perf_counter()

        for attempt in range(1, self.run_cfg.max_retries + 1):
            self.limiter.acquire()
            t0 = time.perf_counter()
            try:
                response = litellm_completion(**self._call_kwargs(messages))
                latency = time.perf_counter() - t0
                text = (response.choices[0].message.content or "").strip()
                usage = getattr(response, "usage", None)
                return CompletionResult(
                    text=text,
                    latency_s=latency,
                    attempts=attempt,
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                )

            except Exception as exc:  # noqa: BLE001 -- deliberate catch-all
                last_error = f"{type(exc).__name__}: {exc}"[:400]

                # Self-heal: some providers 400 on response_format. Drop it once
                # and keep going in plain-text mode for the rest of the run.
                if self._json_mode and "response_format" in str(exc).lower():
                    with self._lock:
                        self._json_mode = False
                    print(f"  [{self.model_cfg.name}] JSON mode unsupported -> "
                          f"falling back to prompt-only JSON")
                    continue

                if attempt >= self.run_cfg.max_retries or not _is_retryable(exc):
                    break

                # Exponential backoff with jitter so parallel workers desynchronise.
                sleep_s = min(backoff, self.run_cfg.max_backoff_s) * (0.5 + random.random())
                print(f"  [{self.model_cfg.name}] attempt {attempt} failed "
                      f"({last_error[:90]}) -- retrying in {sleep_s:.1f}s")
                time.sleep(sleep_s)
                backoff = min(backoff * 2, self.run_cfg.max_backoff_s)

        return CompletionResult(
            text="",
            latency_s=time.perf_counter() - started_total,
            attempts=self.run_cfg.max_retries,
            error=last_error,
        )


def preflight(model_cfg: ModelConfig) -> Optional[str]:
    """
    Cheap sanity check before burning a whole run: returns a warning string,
    or None if the model looks callable.
    """
    if model_cfg.api_key_env and not os.getenv(model_cfg.api_key_env):
        return f"env var {model_cfg.api_key_env} is empty"
    if model_cfg.provider == "ollama" and not model_cfg.api_base:
        return "ollama models usually need an api_base (e.g. http://localhost:11434)"
    return None


def describe_vision(model_cfg: ModelConfig) -> str:
    """One-line note for the run header: will this model be shown figures?"""
    return "figures attached" if model_cfg.supports_vision else "text only (no figures)"
