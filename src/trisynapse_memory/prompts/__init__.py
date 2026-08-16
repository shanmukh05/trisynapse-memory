"""Versioned production prompts and reproducibility metadata."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Iterable


@dataclass(frozen=True)
class PromptSpec:
    """One immutable view of a packaged prompt."""

    name: str
    version: str
    text: str
    sha256: str

    def provenance(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version, "sha256": self.sha256}


_PROMPTS: dict[str, tuple[str, str]] = {
    "extraction": ("extraction-v1", "extraction.md"),
    "episode_recall": ("episode-recall-v1", "episode_recall.md"),
    "answer": ("answer-v1", "answer.md"),
    "benchmark_judge": ("benchmark-judge-v1", "benchmark_judge.md"),
    "image_extraction": ("image-extraction-v1", "image_extraction.md"),
}


@lru_cache(maxsize=None)
def load_prompt(name: str) -> PromptSpec:
    """Load a named packaged prompt and calculate its content identity."""

    try:
        version, filename = _PROMPTS[name]
    except KeyError as exc:
        raise KeyError(f"unknown production prompt: {name}") from exc
    text = files(__package__).joinpath(filename).read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"packaged prompt is empty: {filename}")
    return PromptSpec(
        name=name,
        version=version,
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def prompt_provenance(names: Iterable[str]) -> list[dict[str, str]]:
    """Return stable provenance in caller-specified execution order."""

    return [load_prompt(name).provenance() for name in names]


__all__ = ["PromptSpec", "load_prompt", "prompt_provenance"]
