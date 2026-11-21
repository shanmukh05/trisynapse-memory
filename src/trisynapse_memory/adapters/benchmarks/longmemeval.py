"""LongMemEval dataset adapter.

Owns haystack-session flattening and session-id evidence scoring for
LongMemEval JSON lists. The shared runner still ingests through
``MemoryEngine``. ``lme:`` group prefixes and source-ref ids must stay stable
so published recall metrics remain comparable.
"""

from __future__ import annotations

import json
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


class LongMemEvalAdapter(BenchmarkAdapter):
    name = "longmemeval"
    default_filename = "longmemeval_s_cleaned.json"

    def load_cases(self, path: Path) -> Iterable[BenchmarkCase]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("LongMemEval dataset root must be a JSON list")
        for index, item in enumerate(payload):
            question_id = str(item.get("question_id") or index)
            question = BenchmarkQuestion(
                id=question_id,
                question=str(item.get("question") or ""),
                gold=str(item.get("answer") or ""),
                evidence=tuple(str(value) for value in item.get("answer_session_ids") or []),
                evidence_text=str(item.get("answer") or ""),
                metadata={"question_type": item.get("question_type")},
            )
            yield BenchmarkCase(id=question_id, payload=item, questions=(question,))

    def ingest_case(self, engine: MemoryEngine, case: BenchmarkCase) -> PreparedCase:
        namespace = MemoryNamespace(project_id=f"benchmark:longmemeval:{case.id}")
        group_ids: list[str] = []
        session_ids = case.payload.get("haystack_session_ids") or []
        dates = case.payload.get("haystack_dates") or []
        for session_index, turns in enumerate(case.payload.get("haystack_sessions") or []):
            session_id = str(session_ids[session_index] if session_index < len(session_ids) else session_index)
            group_id = f"lme:{case.id}:{session_id}"
            group_ids.append(group_id)
            date = dates[session_index] if session_index < len(dates) else None
            messages = []
            for turn_index, turn in enumerate(turns):
                content = str(turn.get("content") or "")
                if date:
                    content = f"[{date}] {content}"
                messages.append({
                    "id": f"{session_id}:{turn_index}",
                    "role": turn.get("role"),
                    "content": content,
                })
            engine.ingest_messages(messages, group_id=group_id, namespace=namespace)
        return PreparedCase(tuple(group_ids), namespace, group_prefix=f"lme:{case.id}:")

    def result_metadata(
        self,
        question: BenchmarkQuestion,
        result: MemoryQueryResult,
        *,
        retrieval_hits: list[SearchHit] | None = None,
    ) -> dict[str, Any]:
        prefix = f"lme:{question.id}:"
        cited_sessions = _source_session_ids(result.citations, prefix)
        retrieved_sessions = _source_session_ids(retrieval_hits or [], prefix)
        evidence = set(question.evidence or ())
        return _session_overlap_metadata(question.metadata, evidence, retrieved_sessions, cited_sessions)


def _source_session_ids(items: Iterable[Any], prefix: str) -> set[str]:
    return {
        str(item.source_ref.get("id", "")).removeprefix(prefix)
        for item in items
        if isinstance(item.source_ref, dict)
    }


def _session_overlap_metadata(
    metadata: dict[str, Any],
    evidence: set[str],
    retrieved_sessions: set[str],
    cited_sessions: set[str],
) -> dict[str, Any]:
    retrieved_evidence = evidence & retrieved_sessions
    cited_evidence = evidence & cited_sessions
    return {
        **metadata,
        "gold_evidence_ids": sorted(evidence),
        "retrieved_ids": sorted(retrieved_sessions),
        "cited_ids": sorted(cited_sessions),
        "evidence_hit": bool(retrieved_evidence),
        "evidence_hit_at_k": bool(retrieved_evidence),
        "evidence_recall_at_k": len(retrieved_evidence) / len(evidence) if evidence else None,
        "all_evidence_retrieved": bool(evidence) and evidence <= retrieved_sessions,
        "citation_evidence_recall": len(cited_evidence) / len(evidence) if evidence else None,
        "citation_precision": len(cited_evidence) / len(cited_sessions) if cited_sessions else None,
    }
