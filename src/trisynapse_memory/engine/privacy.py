"""Pre-Trace privacy filtering.

Filtering happens before hashing and persistence. The filter intentionally
targets high-confidence credentials by default; optional PII filtering can be
enabled by applications that do not need email addresses or phone numbers in
memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RedactionResult:
    text: str
    categories: list[str] = field(default_factory=list)


class PrivacyFilter:
    def __init__(self, *, redact_pii: bool = False) -> None:
        patterns: list[tuple[str, re.Pattern[str]]] = [
            ("private_block", re.compile(r"<private>.*?</private>", re.I | re.S)),
            ("private_block", re.compile(r"\[private\].*?\[/private\]", re.I | re.S)),
            ("pem_private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", re.S)),
            ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
            ("github_token", re.compile(r"\bgh[opusr]_[A-Za-z0-9]{20,}\b")),
            ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
            ("bearer_token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{16,}={0,2}")),
            (
                "assigned_secret",
                re.compile(
                    r"(?i)\b(api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*['\"]?([^\s'\"]{8,})"
                ),
            ),
        ]
        if redact_pii:
            patterns.extend([
                ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
                ("phone", re.compile(r"(?<!\w)(?:\+?\d[\d .()-]{8,}\d)(?!\w)")),
            ])
        self._patterns = patterns

    def redact(self, text: str) -> RedactionResult:
        value = text
        found: list[str] = []
        for category, pattern in self._patterns:
            if pattern.search(value):
                found.append(category)
                if category == "assigned_secret":
                    value = pattern.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)
                else:
                    value = pattern.sub(f"[REDACTED:{category}]", value)
        return RedactionResult(text=value, categories=list(dict.fromkeys(found)))

    def redact_value(self, value: Any) -> tuple[Any, list[str]]:
        """Recursively filter string values before structured metadata is persisted."""

        if isinstance(value, str):
            result = self.redact(value)
            return result.text, result.categories
        if isinstance(value, dict):
            filtered: dict[Any, Any] = {}
            categories: list[str] = []
            for key, item in value.items():
                if isinstance(key, str) and re.search(
                    r"(?i)(?:api[_-]?key|access[_-]?token|authorization|client[_-]?secret|password)", key
                ):
                    filtered[key], found = "[REDACTED:assigned_secret]", ["assigned_secret"]
                else:
                    filtered[key], found = self.redact_value(item)
                categories.extend(found)
            return filtered, list(dict.fromkeys(categories))
        if isinstance(value, list):
            filtered_items: list[Any] = []
            categories = []
            for item in value:
                filtered, found = self.redact_value(item)
                filtered_items.append(filtered)
                categories.extend(found)
            return filtered_items, list(dict.fromkeys(categories))
        if isinstance(value, tuple):
            filtered, categories = self.redact_value(list(value))
            return tuple(filtered), categories
        return value, []
