"""LongMemEval dataset adapter."""

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
        episode_ids: list[str] = []
        session_ids = case.payload.get("haystack_session_ids") or []
        dates = case.payload.get("haystack_dates") or []
        for session_index, turns in enumerate(case.payload.get("haystack_sessions") or []):
            session_id = str(session_ids[session_index] if session_index < len(session_ids) else session_index)
            episode_id = f"lme:{case.id}:{session_id}"
            episode_ids.append(episode_id)
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
            engine.ingest_messages(messages, episode_id=episode_id, namespace=namespace)
        return PreparedCase(tuple(episode_ids), namespace, episode_prefix=f"lme:{case.id}:")

    def result_metadata(
        self,
        question: BenchmarkQuestion,
        result: MemoryQueryResult,
        *,
        retrieval_hits: list[SearchHit] | None = None,
    ) -> dict[str, Any]:
        retrieval_hits = retrieval_hits or []
        prefix = f"lme:{question.id}:"
        cited_sessions = {
            str(citation.source_ref.get("id", "")).removeprefix(prefix)
            for citation in result.citations
            if isinstance(citation.source_ref, dict)
        }
        retrieved_sessions = {
            str(hit.source_ref.get("id", "")).removeprefix(prefix)
            for hit in retrieval_hits
            if isinstance(hit.source_ref, dict)
        }
        evidence = set(question.evidence or ())
        retrieved_evidence = evidence & retrieved_sessions
        cited_evidence = evidence & cited_sessions
        return {
            **question.metadata,
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
