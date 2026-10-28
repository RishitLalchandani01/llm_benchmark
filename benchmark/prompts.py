"""
Prompt construction.

The whole framework depends on getting machine-readable answers back, so the
system prompt is deliberately blunt and repetitive about the output contract.
Small models (Llama 3.2 1B especially) drift into prose unless you keep saying
"JSON only, no markdown".

Questions pulled out of a PDF may also carry a rendered page image. When the
model under test is vision-capable the image is attached as a data URL; when it
is not, we say so in the prompt and flag the row, because "got it wrong because
it could not see the diagram" is a different finding from "got it wrong".
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import MCQuestion

SYSTEM_PROMPT = """You are an expert test-taker answering a standardized \
multiple-choice exam question (SAT / AP level).

Think through the problem internally, then reply with ONE JSON object and \
nothing else.

Required JSON shape:
{"answer": "<single option letter>", "confidence": <number between 0 and 1>, \
"reasoning": "<at most two short sentences>"}

Hard rules:
- Output raw JSON only. No markdown fences, no ```json, no text before or after.
- "answer" must be exactly one letter from the options given (for example "C").
- Never answer with the option text, only the letter.
- Always pick a single best option, even if you are unsure. Never refuse.
"""

# Appended to the user turn when the first attempt came back unparseable.
REASK_SUFFIX = """
Your previous reply could not be parsed. Reply with ONLY this, filled in:
{"answer": "A", "confidence": 0.5, "reasoning": "short"}
No other text.
"""

# Said to text-only models when the question depends on a figure they cannot see.
NO_VISION_NOTE = (
    "\n\nNote: this question refers to a figure that is not included in this "
    "text prompt. Reason from the wording alone and still choose one letter."
)

# --------------------------------------------------------------------------- #
# Free-response prompts
# --------------------------------------------------------------------------- #
FRQ_SYSTEM_PROMPT = """You are taking a written AP-level exam (free response).

Answer the question fully and in order, labelling each part exactly as the
question labels it (Part A, Part A i, Part B, ...).

- Show the physics or mathematics: state the principle you start from, then the
  algebra, then the result with units.
- Derivations must end in a symbolic expression in the requested variables.
- Where the question asks you to justify or compare, give the reasoning in
  words, not just an equation.
- If a part asks for a sketch or graph, describe the shape precisely instead
  (start value, end value, curvature, intercepts).
- Be complete but do not pad. No preamble, no restating the question.
"""

JUDGE_SYSTEM_PROMPT = """You are an experienced AP exam reader scoring one
student response against the official scoring guidelines.

Reply with ONE JSON object and nothing else:
{"score": <points awarded>, "max_score": <points available>,
 "per_part": [{"part": "A i", "awarded": 1, "note": "short"}],
 "reasoning": "two or three sentences"}

Scoring rules:
- Award points strictly by the guidelines supplied. If no guidelines are
  supplied, award credit for physical correctness, correct reasoning and correct
  final expressions, and use the max_score stated in the task.
- Partial credit is expected: score each part independently.
- Do not reward confident writing, length, or restating the question.
- A correct answer reached by wrong reasoning earns the reasoning points only.
- Never award more than max_score. Output raw JSON, no markdown fences.
"""

JUDGE_REASK = ('\nYour previous reply was not parseable. Reply with ONLY: '
               '{"score": 0, "max_score": 1, "reasoning": "short"}')

_MAX_IMAGES_PER_QUESTION = 2  # a page render plus, at most, one continuation page


def build_messages(
    question: MCQuestion,
    reask: bool = False,
    attach_images: bool = False,
) -> List[Dict[str, Any]]:
    """
    Return the litellm `messages` list for one question.

    attach_images=True sends the question's rendered page PNG(s) alongside the
    text (only meaningful for vision-capable models).
    """
    allowed = ", ".join(question.letters)
    text = (
        f"Subject: {question.domain}\n\n"
        f"Question:\n{question.question.strip()}\n\n"
        f"Options:\n{question.render_choices()}\n\n"
        f"Valid letters: {allowed}. Respond with JSON only."
    )

    images = question.image_paths[:_MAX_IMAGES_PER_QUESTION] if attach_images else []
    if question.needs_figure and not images:
        text += NO_VISION_NOTE
    if reask:
        text += REASK_SUFFIX

    if not images:
        user_content: Any = text
    else:
        # litellm's OpenAI-style multimodal content blocks; data URLs avoid
        # needing anywhere to host the images.
        user_content = [{"type": "text", "text": text}]
        for path in images:
            url = _data_url(path)
            if url:
                user_content.append({"type": "image_url", "image_url": {"url": url}})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _data_url(path: str) -> str | None:
    """Read an image off disk into a base64 data URL, or None if unreadable."""
    p = Path(path)
    if not p.exists():
        return None
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode('ascii')}"


def build_frq_messages(
    question: MCQuestion, attach_images: bool = False
) -> List[Dict[str, Any]]:
    """Ask the model under test to actually write a free-response answer."""
    text = (
        f"Subject: {question.domain}\n\n"
        f"Question:\n{question.question.strip()}\n\n"
        "Write your full response now."
    )
    images = question.image_paths[:_MAX_IMAGES_PER_QUESTION] if attach_images else []
    if question.needs_figure and not images:
        text += (
            "\n\nNote: the figure referenced above is not included here. State any "
            "assumption you make about it and answer as fully as you can."
        )

    if not images:
        content: Any = text
    else:
        content = [{"type": "text", "text": text}]
        for path in images:
            url = _data_url(path)
            if url:
                content.append({"type": "image_url", "image_url": {"url": url}})

    return [
        {"role": "system", "content": FRQ_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def build_judge_messages(
    question: MCQuestion,
    answer_text: str,
    *,
    rubric: Optional[str] = None,
    max_score: float = 1.0,
    attach_images: bool = False,
    reask: bool = False,
) -> List[Dict[str, Any]]:
    """Hand the judge the question, the official rubric (if any), and the answer."""
    parts = [
        f"Subject: {question.domain}",
        f"Points available (max_score): {max_score}",
        f"\nQUESTION\n{question.question.strip()}",
    ]
    if rubric:
        parts.append(f"\nOFFICIAL SCORING GUIDELINES\n{rubric.strip()}")
    else:
        parts.append(
            "\nNo official scoring guidelines were supplied. Score on physical "
            "correctness, quality of reasoning and correctness of final expressions."
        )
    parts.append(f"\nSTUDENT RESPONSE\n{answer_text.strip() or '(the model returned nothing)'}")
    parts.append("\nScore it now. JSON only.")
    text = "\n".join(parts)
    if reask:
        text += JUDGE_REASK

    images = question.image_paths[:_MAX_IMAGES_PER_QUESTION] if attach_images else []
    if not images:
        content: Any = text
    else:
        content = [{"type": "text", "text": text}]
        for path in images:
            url = _data_url(path)
            if url:
                content.append({"type": "image_url", "image_url": {"url": url}})

    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
