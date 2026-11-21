"""PersonaMem-v2 text/32k dataset adapter.

Owns CSV question rows and 32k chat-history flattening for PersonaMem.
The shared runner still ingests through ``MemoryEngine``. Cell literals and
history-link resolution must stay faithful to the published CSV so evidence
prefix hits remain comparable. ``incorrect_answers`` become MCQ distractors
for end-to-end; retrieval still searches the raw ``user_query``.
"""

from __future__ import annotations

import ast
import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from trisynapse_memory.adapters.benchmarks.base import (
    BenchmarkAdapter,
    BenchmarkCase,
    BenchmarkQuestion,
    PreparedCase,
)
from trisynapse_memory.engine import MemoryEngine
from trisynapse_memory.engine.models import MemoryNamespace, MemoryQueryResult, SearchHit


class PersonaMemAdapter(BenchmarkAdapter):
    name = "personamem"
    default_filename = "benchmark/text/benchmark.csv"

    def load_cases(self, path: Path) -> Iterable[BenchmarkCase]:
        csv_path = self.resolve_dataset(path)
        suite_root = _suite_root(csv_path)
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        with csv_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                persona_id = str(row.get("persona_id") or "").strip()
                if not persona_id:
                    continue
                grouped[persona_id].append(row)
        for persona_id, rows in grouped.items():
            questions = tuple(_question_from_row(persona_id, index, row) for index, row in enumerate(rows))
            history_link = next(
                (str(row.get("chat_history_32k_link") or "").strip() for row in rows if row.get("chat_history_32k_link")),
                "",
            )
            yield BenchmarkCase(
                id=persona_id,
                payload={"persona_id": persona_id, "history_link": history_link, "suite_root": str(suite_root)},
                questions=questions,
            )

    def ingest_case(self, engine: MemoryEngine, case: BenchmarkCase) -> PreparedCase:
        namespace = MemoryNamespace(project_id=f"benchmark:personamem:{case.id}")
        suite_root = Path(case.payload["suite_root"])
        history_path = _resolve_history_path(suite_root, str(case.payload.get("history_link") or ""))
        sessions = _sessions_from_history(_load_history(history_path))
        prefix = f"persona:{case.id}:"
        group_ids: list[str] = []
        for session_index, messages in enumerate(sessions):
            if not messages:
                continue
            group_id = f"{prefix}s{session_index:04d}"
            group_ids.append(group_id)
            engine.ingest_messages(
                [
                    {
                        "id": f"{session_index}:{turn_index}",
                        "role": message.get("role", "speaker"),
                        "content": message.get("content") or message.get("text") or "",
                    }
                    for turn_index, message in enumerate(messages)
                ],
                group_id=group_id,
                namespace=namespace,
            )
        return PreparedCase(tuple(group_ids), namespace, group_prefix=prefix)

    def result_metadata(
        self,
        question: BenchmarkQuestion,
        result: MemoryQueryResult,
        *,
        retrieval_hits: list[SearchHit] | None = None,
    ) -> dict[str, Any]:
        retrieved = [hit.text for hit in retrieval_hits or []]
        evidence_text = question.evidence_text.strip()
        hit = _prefix_evidence_hit(evidence_text, retrieved)
        cited = " ".join(citation.excerpt for citation in result.citations)
        cited_hit = bool(evidence_text) and evidence_text[:80] in cited
        return _persona_overlap_fields(question, hit, cited_hit, bool(result.citations))


def _suite_root(csv_path: Path) -> Path:
    if csv_path.parent.name == "text" and csv_path.parent.parent.name == "benchmark":
        return csv_path.parent.parent.parent
    return csv_path.parent


def _resolve_history_path(suite_root: Path, link: str) -> Path:
    if not link:
        raise FileNotFoundError(f"PersonaMem case is missing chat_history_32k_link under {suite_root}")
    candidate = Path(link)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    resolved = (suite_root / candidate).resolve()
    if resolved.is_file():
        return resolved
    nested = (suite_root / "data" / "chat_history_32k" / candidate.name).resolve()
    if nested.is_file():
        return nested
    raise FileNotFoundError(resolved)


def _question_from_row(persona_id: str, index: int, row: dict[str, str]) -> BenchmarkQuestion:
    query = _cell_text(row.get("user_query"))
    gold = _cell_text(row.get("correct_answer"))
    snippet = _cell_text(row.get("related_conversation_snippet") or row.get("related_conversation"))
    return BenchmarkQuestion(
        id=f"{persona_id}:q{index}",
        question=query,
        gold=gold,
        evidence=(snippet,) if snippet else (),
        evidence_text=snippet,
        choices=_incorrect_answers(row.get("incorrect_answers")),
        metadata={
            "topic_query": row.get("topic_query") or "",
            "preference": _cell_text(row.get("preference")),
            "pref_type": row.get("pref_type") or "",
            "conversation_scenario": row.get("conversation_scenario") or "",
            "updated": row.get("updated") or "",
        },
    )


def _incorrect_answers(value: object) -> tuple[str, ...]:
    parsed = _parse_incorrect_answers(value)
    return tuple(item for item in (_cell_text(part) for part in parsed) if item)


def _parse_incorrect_answers(value: object) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    if isinstance(parsed, list):
        return parsed
    return []


def _persona_overlap_fields(
    question: BenchmarkQuestion,
    hit: bool,
    cited_hit: bool,
    has_citations: bool,
) -> dict[str, Any]:
    return {
        **question.metadata,
        "gold_evidence_ids": list(question.evidence or ()),
        "evidence_hit": hit,
        "evidence_hit_at_k": hit,
        "evidence_recall_at_k": 1.0 if hit else 0.0,
        "all_evidence_retrieved": hit,
        "citation_evidence_recall": 1.0 if cited_hit else 0.0,
        "citation_precision": 1.0 if cited_hit and has_citations else None,
    }


def _prefix_evidence_hit(evidence_text: str, retrieved: list[str]) -> bool:
    return bool(evidence_text) and any(evidence_text[:80] in text for text in retrieved)


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text[0] not in "{\"'[":
        return text.strip("\"'")
    return _parsed_cell_text(text)


def _parsed_cell_text(text: str) -> str:
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text
    if isinstance(parsed, dict):
        return str(parsed.get("content") or parsed.get("text") or text).strip()
    if isinstance(parsed, list):
        parts = [_cell_text(item) for item in parsed]
        return " ".join(part for part in parts if part)
    if isinstance(parsed, str):
        return parsed.strip()
    return text.strip("\"'")


def _load_history(path: Path) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _sessions_from_history(payload: Any) -> list[list[dict[str, Any]]]:
    messages = _flatten_messages(payload)
    if not messages:
        return []
    chunk_size = 8
    return [messages[index:index + chunk_size] for index in range(0, len(messages), chunk_size)]


def _flatten_messages(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return _flatten_message_list(payload)
    if isinstance(payload, dict):
        return _flatten_message_dict(payload)
    return []


def _flatten_message_list(payload: list[Any]) -> list[dict[str, Any]]:
    if payload and isinstance(payload[0], list):
        return [item for session in payload for item in _flatten_messages(session)]
    collected: list[dict[str, Any]] = []
    for item in payload:
        collected.extend(_flatten_messages(item))
    return collected


def _flatten_message_dict(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "role" in payload and ("content" in payload or "text" in payload):
        return [payload]
    for key in ("chat_history", "conversations", "messages"):
        if key in payload:
            return _flatten_messages(payload[key])
    collected: list[dict[str, Any]] = []
    for value in payload.values():
        if isinstance(value, (list, dict)):
            collected.extend(_flatten_messages(value))
    return collected
