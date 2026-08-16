"""LoCoMo dataset adapter."""

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
from trisynapse_memory.engine.models import MemoryNamespace, MemoryQueryResult


class LoCoMoAdapter(BenchmarkAdapter):
    name = "locomo"
    default_filename = "locomo10.json"

    def load_cases(self, path: Path) -> Iterable[BenchmarkCase]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("LoCoMo dataset root must be a JSON list")
        for sample_index, sample in enumerate(payload):
            sample_id = str(sample.get("sample_id") or sample_index)
            questions = tuple(
                BenchmarkQuestion(
                    id=f"{sample_id}:q{index}",
                    question=str(item.get("question") or ""),
                    gold=str(item.get("answer") or ""),
                    evidence=tuple(str(value) for value in item.get("evidence") or []),
                    metadata={"category": item.get("category")},
                )
                for index, item in enumerate(sample.get("qa") or [])
            )
            yield BenchmarkCase(id=sample_id, payload=sample, questions=questions)

    def ingest_case(self, engine: MemoryEngine, case: BenchmarkCase) -> PreparedCase:
        namespace = MemoryNamespace(project_id=f"benchmark:locomo:{case.id}")
        conversation = case.payload.get("conversation") or {}
        episode_ids: list[str] = []
        for key, turns in conversation.items():
            if not key.startswith("session_") or key.endswith("_date_time") or not isinstance(turns, list):
                continue
            episode_id = f"locomo:{case.id}:{key}"
            episode_ids.append(episode_id)
            observed = conversation.get(f"{key}_date_time")
            for index, turn in enumerate(turns):
                text = f"{turn.get('speaker', 'speaker')}: {turn.get('text', '')}"
                if observed:
                    text = f"[{observed}] {text}"
                engine.ingest_observation(
                    text,
                    episode_id=episode_id,
                    source_ref={"type": "locomo", "sample_id": case.id},
                    locator={"dia_id": turn.get("dia_id"), "turn_index": index},
                    external_key=f"locomo:{case.id}:{turn.get('dia_id') or key + ':' + str(index)}",
                    namespace=namespace,
                    process=False,
                    schedule=False,
                )
        return PreparedCase(tuple(episode_ids), namespace)

    def result_metadata(
        self, question: BenchmarkQuestion, result: MemoryQueryResult
    ) -> dict[str, Any]:
        cited = {
            str(citation.locator.get("dia_id"))
            for citation in result.citations
            if isinstance(citation.locator, dict) and citation.locator.get("dia_id")
        }
        evidence = set(question.evidence or ())
        return {
            **question.metadata,
            "evidence_hit": bool(evidence & cited),
            "cited_ids": sorted(cited),
        }
