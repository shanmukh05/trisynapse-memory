"""HaluMem dataset adapter.

Owns JSONL user/session flattening and timestamp normalization for HaluMem.
The shared runner still ingests through ``MemoryEngine``. Question ids keep the
``uuid:s{session}:q{index}`` shape so smoke artifacts stay comparable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trisynapse_memory.adapters.benchmarks.base import (
    BenchmarkAdapter,
    BenchmarkCase,
    BenchmarkQuestion,
    PreparedCase,
    overlap,
)
from trisynapse_memory.engine import MemoryEngine
from trisynapse_memory.engine.models import MemoryNamespace, MemoryQueryResult, SearchHit


class HaluMemAdapter(BenchmarkAdapter):
    name = "halumem"
    default_filename = "HaluMem-Medium.jsonl"

    def load_cases(self, path: Path) -> Iterable[BenchmarkCase]:
        with path.open(encoding="utf-8") as handle:
            for line_index, raw in enumerate(handle):
                if not raw.strip():
                    continue
                yield _halumem_case(line_index, json.loads(raw))

    def ingest_case(self, engine: MemoryEngine, case: BenchmarkCase) -> PreparedCase:
        namespace = MemoryNamespace(project_id=f"benchmark:halumem:{case.id}")
        group_ids: list[str] = []
        for session_index, session in enumerate(case.payload.get("sessions") or []):
            group_id = f"halu:{case.id}:s{session_index:04d}"
            group_ids.append(group_id)
            messages = [
                {
                    "id": f"{session_index}:{turn_index}",
                    "role": turn.get("role", "speaker"),
                    "content": turn.get("content", ""),
                    "timestamp": _normalize_timestamp(turn.get("timestamp")),
                }
                for turn_index, turn in enumerate(session.get("dialogue") or [])
            ]
            if messages:
                engine.ingest_messages(messages, group_id=group_id, namespace=namespace)
        return PreparedCase(tuple(group_ids), namespace, group_prefix=f"halu:{case.id}:")

    def result_metadata(
        self,
        question: BenchmarkQuestion,
        result: MemoryQueryResult,
        *,
        retrieval_hits: list[SearchHit] | None = None,
    ) -> dict[str, Any]:
        snippets = _evidence_snippets(question.evidence)
        context = " ".join(hit.text for hit in retrieval_hits or [] if hit.text)
        cited = " ".join(citation.excerpt for citation in result.citations if citation.excerpt)
        retrieved_hits = [snippet for snippet in snippets if overlap(snippet, context)]
        cited_hits = [snippet for snippet in snippets if overlap(snippet, cited)]
        hit = bool(retrieved_hits)
        return {
            **question.metadata,
            "gold_evidence_ids": snippets,
            "evidence_hit": hit,
            "evidence_hit_at_k": hit,
            "evidence_recall_at_k": len(retrieved_hits) / len(snippets) if snippets else None,
            "all_evidence_retrieved": bool(snippets) and len(retrieved_hits) == len(snippets),
            "citation_evidence_recall": len(cited_hits) / len(snippets) if snippets else None,
            "citation_precision": len(cited_hits) / len(result.citations) if result.citations else None,
            "accuracy": (1.0 if hit else 0.0) if snippets else None,
        }


def _halumem_case(line_index: int, user: dict[str, Any]) -> BenchmarkCase:
    user_id = str(user.get("uuid") or line_index)
    questions: list[BenchmarkQuestion] = []
    for session_index, session in enumerate(user.get("sessions") or []):
        questions.extend(_halumem_session_questions(user_id, session_index, session))
    return BenchmarkCase(id=user_id, payload=user, questions=tuple(questions))


def _halumem_session_questions(
    user_id: str,
    session_index: int,
    session: dict[str, Any],
) -> list[BenchmarkQuestion]:
    questions: list[BenchmarkQuestion] = []
    for question_index, item in enumerate(session.get("questions") or []):
        evidence_text = " ".join(
            str(value.get("memory_content") if isinstance(value, dict) else value)
            for value in item.get("evidence") or []
        )
        question_type = str(item.get("question_type") or "").strip()
        questions.append(BenchmarkQuestion(
            id=f"{user_id}:s{session_index}:q{question_index}",
            question=str(item.get("question") or ""),
            gold=str(item.get("answer") or ""),
            evidence=item.get("evidence") or [],
            evidence_text=evidence_text,
            metadata={
                "category": question_type or "uncategorized",
                "question_type": question_type,
                "difficulty": item.get("difficulty"),
            },
        ))
    return questions


def _evidence_snippets(evidence: Any) -> list[str]:
    snippets: list[str] = []
    for item in evidence or []:
        if isinstance(item, dict):
            text = str(item.get("memory_content") or item.get("content") or "").strip()
        else:
            text = str(item).strip()
        if text:
            snippets.append(text)
    return snippets


def _normalize_timestamp(value: object) -> str | None:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%b %d, %Y, %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise ValueError(f"unsupported HaluMem timestamp: {text}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat()
