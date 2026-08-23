"""Lexical tokenization and answer-context token accounting."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Protocol


_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_CJK_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x3040, 0x30FF),
    (0xAC00, 0xD7AF),
)
LEXICAL_TOKENIZER_VERSION = "unicode-code-v2"


def lexical_tokens(text: str) -> list[str]:
    """Tokenize Unicode prose and code identifiers without one-language assumptions.

    Original ASCII words are retained. A small suffix variant is added for
    common English inflections, while code identifiers are split on snake/camel
    boundaries. CJK runs emit characters and adjacent bigrams so they remain
    searchable without requiring a language-specific segmenter.
    """

    result: list[str] = []
    for run in _unicode_runs(text or ""):
        if all(_is_cjk(character) for character in run):
            characters = list(run)
            result.extend(characters)
            result.extend("".join(characters[index:index + 2]) for index in range(len(characters) - 1))
            continue
        for underscore_part in run.replace("$", "_").split("_"):
            for part in _CAMEL_BOUNDARY.split(underscore_part):
                token = part.casefold().strip("'’-_")
                if not token:
                    continue
                result.append(token)
                variant = _english_suffix_variant(token)
                if variant and variant != token:
                    result.append(variant)
    return result


def _unicode_runs(text: str) -> list[str]:
    runs: list[str] = []
    current: list[str] = []
    current_is_cjk: bool | None = None
    for character in text:
        category = unicodedata.category(character)
        accepted = category[0] in {"L", "N", "M"} or character in {"_", "$", "'", "’", "-"}
        if not accepted:
            if current:
                runs.append("".join(current))
                current = []
                current_is_cjk = None
            continue
        character_is_cjk = _is_cjk(character)
        if current and character_is_cjk != current_is_cjk and (character_is_cjk or current_is_cjk):
            runs.append("".join(current))
            current = []
        current.append(character)
        current_is_cjk = character_is_cjk
    if current:
        runs.append("".join(current))
    return runs


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return any(start <= codepoint <= end for start, end in _CJK_RANGES)


def _english_suffix_variant(token: str) -> str | None:
    if not token.isascii() or not token.isalpha():
        return None
    if len(token) > 6 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]
    return None


class TokenCounter(Protocol):
    name: str
    exact: bool

    def count(self, text: str) -> int: ...


@dataclass(frozen=True)
class ApproximateTokenCounter:
    """Offline fallback calibrated by provider family, never presented as exact."""

    provider: str = "generic"
    model: str | None = None
    exact: bool = False

    @property
    def name(self) -> str:
        return f"unicode-byte-estimate-v1:{self.provider}:{self.model or 'default'}"

    def count(self, text: str) -> int:
        if not text:
            return 0
        # UTF-8 bytes handle CJK/code more safely than len(text)/4. Provider
        # families use slightly different average bytes per token.
        bytes_per_token = {
            "anthropic": 3.6,
            "gemini": 3.8,
            "openai": 4.0,
            "openrouter": 3.8,
            "deepinfra": 3.8,
            "deepseek": 3.7,
            "kimi": 3.5,
        }.get(self.provider, 3.8)
        lexical_floor = len(lexical_tokens(text))
        byte_estimate = round(len(text.encode("utf-8")) / bytes_per_token)
        return max(1, lexical_floor, byte_estimate)


class TiktokenCounter:
    """Local OpenAI tokenizer; constructed only when tiktoken is installed."""

    exact = True

    def __init__(self, model: str) -> None:
        import tiktoken

        try:
            self._encoding = tiktoken.encoding_for_model(model)
            encoding_name = self._encoding.name
        except KeyError:
            self._encoding = tiktoken.get_encoding("o200k_base")
            encoding_name = "o200k_base-fallback"
            self.exact = False
        self.name = f"tiktoken:{encoding_name}:{model}"

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))


class CallableTokenCounter:
    """Adapter for an application/provider supplied local ``count_tokens`` method."""

    exact = True

    def __init__(self, callback: Any, name: str) -> None:
        self._callback = callback
        self.name = name

    def count(self, text: str) -> int:
        value = self._callback(text)
        if isinstance(value, dict):
            value = value.get("total_tokens") or value.get("input_tokens") or value.get("tokens")
        return max(0, int(value))


def token_counter_for(completion: Any | None) -> TokenCounter:
    """Select the best local counter for the active completion provider."""

    if completion is not None and callable(getattr(completion, "count_tokens", None)):
        return CallableTokenCounter(
            completion.count_tokens,
            f"provider:{getattr(completion, 'model', completion.__class__.__name__)}",
        )
    settings = getattr(completion, "settings", None)
    provider = str(getattr(settings, "provider", "none") or "none").casefold()
    model = str(getattr(completion, "model", "") or "")
    if provider == "openai" and model:
        try:
            return TiktokenCounter(model)
        except ImportError:
            pass
    return ApproximateTokenCounter(provider=provider, model=model or None)
