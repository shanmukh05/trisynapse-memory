"""LoCoMo dataset adapter.

Owns conversation/session flattening and dialog-id evidence scoring for the
LoCoMo JSON list. The shared runner still ingests through ``MemoryEngine``.
Session keys and ``dia_id`` locators must stay byte-compatible with published
LoCoMo artifacts so recall metrics remain comparable.
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
            yield _locomo_case(sample_index, sample)

    def ingest_case(self, engine: MemoryEngine, case: BenchmarkCase) -> PreparedCase:
        namespace = MemoryNamespace(project_id=f"benchmark:locomo:{case.id}")
        conversation = case.payload.get("conversation") or {}
        group_ids = [
            _ingest_locomo_session(engine, case, conversation, key, namespace)
            for key in _locomo_session_keys(conversation)
        ]
        return PreparedCase(tuple(group_ids), namespace)

    def result_metadata(
        self,
        question: BenchmarkQuestion,
        result: MemoryQueryResult,
        *,
        retrieval_hits: list[SearchHit] | None = None,
    ) -> dict[str, Any]:
        cited = _locator_dia_ids(result.citations)
        retrieved = _locator_dia_ids(retrieval_hits or [])
        evidence = set(question.evidence or ())
        return _dia_overlap_metadata(question.metadata, evidence, retrieved, cited)


def _locomo_case(sample_index: int, sample: dict[str, Any]) -> BenchmarkCase:
    sample_id = str(sample.get("sample_id") or sample_index)
    evidence_text_by_id = _dialog_evidence_texts(sample)
    return BenchmarkCase(
        id=sample_id,
        payload=sample,
        questions=_qa_questions(sample_id, sample, evidence_text_by_id),
    )


def _dialog_evidence_texts(sample: dict[str, Any]) -> dict[str, str]:
    evidence_text_by_id: dict[str, str] = {}
    for key, turns in (sample.get("conversation") or {}).items():
        if not _is_locomo_session(key, turns):
            continue
        for turn in turns:
            dialog_id = str(turn.get("dia_id") or "")
            if not dialog_id:
                continue
            evidence_text_by_id[dialog_id] = _turn_evidence_text(turn)
    return evidence_text_by_id


def _is_locomo_session(key: str, turns: Any) -> bool:
    return key.startswith("session_") and not key.endswith("_date_time") and isinstance(turns, list)


def _turn_evidence_text(turn: dict[str, Any]) -> str:
    parts = [str(turn.get("text") or "").strip()]
    if turn.get("blip_caption"):
        parts.append(str(turn["blip_caption"]).strip())
    return " ".join(part for part in parts if part)


def _qa_questions(
    sample_id: str,
    sample: dict[str, Any],
    evidence_text_by_id: dict[str, str],
) -> tuple[BenchmarkQuestion, ...]:
    return tuple(
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


def _locomo_session_keys(conversation: dict[str, Any]) -> list[str]:
    return sorted(
        (
            key for key, value in conversation.items()
            if key.startswith("session_") and not key.endswith("_date_time") and isinstance(value, list)
        ),
        key=_session_number,
    )


def _ingest_locomo_session(
    engine: MemoryEngine,
    case: BenchmarkCase,
    conversation: dict[str, Any],
    key: str,
    namespace: MemoryNamespace,
) -> str:
    group_id = f"locomo:{case.id}:{key}"
    raw_observed = conversation.get(f"{key}_date_time")
    observed = _normalize_locomo_timestamp(raw_observed)
    for index, turn in enumerate(conversation[key]):
        _ingest_locomo_turn(engine, case, key, index, turn, group_id, raw_observed, observed, namespace)
    return group_id


def _ingest_locomo_turn(
    engine: MemoryEngine,
    case: BenchmarkCase,
    key: str,
    index: int,
    turn: dict[str, Any],
    group_id: str,
    raw_observed: Any,
    observed: datetime | None,
    namespace: MemoryNamespace,
) -> None:
    dialog_id = turn.get("dia_id")
    source_ref = {
        "type": "locomo",
        "sample_id": case.id,
        "session": key,
        "raw_timestamp": raw_observed,
    }
    engine.ingest_raw_source(
        f"{turn.get('speaker', 'speaker')}: {turn.get('text', '')}",
        group_id=group_id,
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
    if not caption:
        return
    engine.ingest_raw_source(
        f"Image shared by {turn.get('speaker', 'speaker')}: {caption}",
        group_id=group_id,
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


def _locator_dia_ids(items: Iterable[Any]) -> set[str]:
    return {
        str(item.locator.get("dia_id"))
        for item in items
        if isinstance(item.locator, dict) and item.locator.get("dia_id")
    }


def _dia_overlap_metadata(
    metadata: dict[str, Any],
    evidence: set[str],
    retrieved: set[str],
    cited: set[str],
) -> dict[str, Any]:
    retrieved_evidence = evidence & retrieved
    cited_evidence = evidence & cited
    return {
        **metadata,
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
    parsed = _parse_locomo_timestamp(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _parse_locomo_timestamp(text: str) -> datetime:
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
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"unsupported LoCoMo timestamp: {text}") from exc
