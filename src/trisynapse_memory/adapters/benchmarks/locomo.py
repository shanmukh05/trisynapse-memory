"""LoCoMo dataset adapter."""

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
)
from trisynapse_memory.engine import MemoryEngine
from trisynapse_memory.engine.models import MemoryNamespace, MemoryQueryResult, SearchHit


class LoCoMoAdapter(BenchmarkAdapter):
    name = "locomo"
    default_filename = "locomo10.json"

    def load_cases(self, path: Path) -> Iterable[BenchmarkCase]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("LoCoMo dataset root must be a JSON list")
        for sample_index, sample in enumerate(payload):
            sample_id = str(sample.get("sample_id") or sample_index)
            evidence_text_by_id: dict[str, str] = {}
            for key, turns in (sample.get("conversation") or {}).items():
                if not key.startswith("session_") or key.endswith("_date_time") or not isinstance(turns, list):
                    continue
                for turn in turns:
                    dialog_id = str(turn.get("dia_id") or "")
                    if not dialog_id:
                        continue
                    parts = [str(turn.get("text") or "").strip()]
                    if turn.get("blip_caption"):
                        parts.append(str(turn["blip_caption"]).strip())
                    evidence_text_by_id[dialog_id] = " ".join(part for part in parts if part)
            questions = tuple(
                BenchmarkQuestion(
                    id=f"{sample_id}:q{index}",
                    question=str(item.get("question") or ""),
                    gold=str(item.get("answer") or ""),
                    evidence=tuple(str(value) for value in item.get("evidence") or []),
                    evidence_text=" ".join(
                        evidence_text_by_id.get(str(value), "")
                        for value in item.get("evidence") or []
                    ).strip(),
                    metadata={"category": item.get("category")},
                )
                for index, item in enumerate(sample.get("qa") or [])
            )
            yield BenchmarkCase(id=sample_id, payload=sample, questions=questions)

    def ingest_case(self, engine: MemoryEngine, case: BenchmarkCase) -> PreparedCase:
        namespace = MemoryNamespace(project_id=f"benchmark:locomo:{case.id}")
        conversation = case.payload.get("conversation") or {}
        episode_ids: list[str] = []
        session_keys = sorted(
            (
                key for key, value in conversation.items()
                if key.startswith("session_") and not key.endswith("_date_time") and isinstance(value, list)
            ),
            key=_session_number,
        )
        for key in session_keys:
            turns = conversation[key]
            episode_id = f"locomo:{case.id}:{key}"
            episode_ids.append(episode_id)
            raw_observed = conversation.get(f"{key}_date_time")
            observed = _normalize_locomo_timestamp(raw_observed)
            for index, turn in enumerate(turns):
                text = f"{turn.get('speaker', 'speaker')}: {turn.get('text', '')}"
                dialog_id = turn.get("dia_id")
                source_ref = {
                    "type": "locomo",
                    "sample_id": case.id,
                    "session": key,
                    "raw_timestamp": raw_observed,
                }
                engine.ingest_observation(
                    text,
                    episode_id=episode_id,
                    observed_at=observed,
                    source_ref=source_ref,
                    locator={"kind": "dialog", "dia_id": dialog_id, "turn_index": index},
                    external_key=f"locomo:{case.id}:{dialog_id or key + ':' + str(index)}:dialog",
                    modality="conversation",
                    source_type="conversation",
                    retrieval_fields={"speaker": turn.get("speaker", "speaker"), "message": turn.get("text", "")},
                    namespace=namespace,
                    process=False,
                    schedule=False,
                )
                caption = str(turn.get("blip_caption") or "").strip()
                if caption:
                    engine.ingest_observation(
                        f"Image shared by {turn.get('speaker', 'speaker')}: {caption}",
                        episode_id=episode_id,
                        observed_at=observed,
                        source_ref=source_ref,
                        locator={"kind": "image_caption", "dia_id": dialog_id, "turn_index": index},
                        external_key=f"locomo:{case.id}:{dialog_id or key + ':' + str(index)}:caption",
                        modality="image",
                        source_type="image_caption",
                        retrieval_fields={"description": caption, "visible_text": caption},
                        namespace=namespace,
                        process=False,
                        schedule=False,
                    )
        return PreparedCase(tuple(episode_ids), namespace)

    def result_metadata(
        self,
        question: BenchmarkQuestion,
        result: MemoryQueryResult,
        *,
        retrieval_hits: list[SearchHit] | None = None,
    ) -> dict[str, Any]:
        cited = {
            str(citation.locator.get("dia_id"))
            for citation in result.citations
            if isinstance(citation.locator, dict) and citation.locator.get("dia_id")
        }
        evidence = set(question.evidence or ())
        retrieved = {
            str(hit.locator.get("dia_id"))
            for hit in retrieval_hits or []
            if isinstance(hit.locator, dict) and hit.locator.get("dia_id")
        }
        retrieved_evidence = evidence & retrieved
        cited_evidence = evidence & cited
        return {
            **question.metadata,
            "gold_evidence_ids": sorted(evidence),
            "retrieved_ids": sorted(retrieved),
            "cited_ids": sorted(cited),
            "evidence_hit": bool(retrieved_evidence),
            "evidence_hit_at_k": bool(retrieved_evidence),
            "evidence_recall_at_k": len(retrieved_evidence) / len(evidence) if evidence else None,
            "all_evidence_retrieved": bool(evidence) and evidence <= retrieved,
            "citation_evidence_recall": len(cited_evidence) / len(evidence) if evidence else None,
            "citation_precision": len(cited_evidence) / len(cited) if cited else None,
        }


def _session_number(value: str) -> tuple[int, str]:
    try:
        return int(value.removeprefix("session_")), value
    except ValueError:
        return 10**9, value


def _normalize_locomo_timestamp(value: object) -> datetime | None:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    formats = (
        "%I:%M %p on %d %B, %Y",
        "%d %B %Y",
        "%d %B, %Y",
    )
    for pattern in formats:
        try:
            return datetime.strptime(text, pattern).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"unsupported LoCoMo timestamp: {text}") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
