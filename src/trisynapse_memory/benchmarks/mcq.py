"""Deterministic 4-way MCQ prompts and exact-match scoring.

PersonaMem-v2 end-to-end retrieves with the raw user query, then asks the
completion model to pick A–D. Official MCQ tables use exact option match, not
the generic benchmark judge. Option order is a stable hash of the question id
so A is not always the gold answer.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from trisynapse_memory.adapters.benchmarks.base import BenchmarkQuestion

_MCQ_LETTERS = "ABCD"
_LETTER = re.compile(r"^\s*([A-Da-d])(?:\b|[.)\s:])")


def labeled_mcq_options(question: BenchmarkQuestion) -> tuple[list[tuple[str, str]], str]:
    """Return ``(letter, text)`` pairs and the letter of ``question.gold``."""

    options = [question.gold, *[item for item in question.choices if item and item != question.gold]]
    options = options[: len(_MCQ_LETTERS)]
    ordered = sorted(
        options,
        key=lambda text: hashlib.sha256(f"{question.id}\0{text}".encode()).hexdigest(),
    )
    labeled = list(zip(_MCQ_LETTERS, ordered))
    correct = next(letter for letter, text in labeled if text == question.gold)
    return labeled, correct


def mcq_answer_question(question: BenchmarkQuestion) -> str:
    """User prompt for answer generation. Does not replace the retrieval query."""

    labeled, _ = labeled_mcq_options(question)
    lines = [
        question.question.strip(),
        "",
        "Choose the best personalized response. Reply with the letter only (A, B, C, or D).",
        "",
    ]
    lines.extend(f"{letter}. {text}" for letter, text in labeled)
    return "\n".join(lines)


def score_mcq_prediction(prediction: str, question: BenchmarkQuestion) -> dict[str, Any]:
    """Exact-match MCQ fields for a result row. Sets ``accuracy`` to 0 or 1."""

    labeled, correct_letter = labeled_mcq_options(question)
    choice = parse_mcq_choice(prediction, labeled)
    return {
        "scoring": "mcq",
        "mcq_choice": choice,
        "mcq_correct_letter": correct_letter,
        "accuracy": 1.0 if choice == correct_letter else 0.0,
    }


def parse_mcq_choice(prediction: str, labeled: list[tuple[str, str]]) -> str | None:
    """Read A–D from a letter prefix or an exact option string."""

    text = (prediction or "").strip()
    if not text:
        return None
    match = _LETTER.match(text)
    if match:
        return match.group(1).upper()
    for letter, option in labeled:
        if text == option.strip():
            return letter
    return None
