"""Append-only ingest helpers for raw sources, document chunks, and facts.

Owns observation writes, document splitting, and extraction persistence.
``MemoryEngine`` schedules these calls; this module does not choose retrieval
semantics. Evidence refs stay limited to known raw-source ids so invented
completion ids cannot become citations.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

from trisynapse_memory.engine.models import Actor, MemoryRecord, MemoryNamespace
from trisynapse_memory.engine.memory_log.store import SQLiteMemoryLogStore
from trisynapse_memory.engine.providers.registry import (
    DEFAULT_COMPLETION_PARALLEL_WORKERS,
    complete_json_parallel,
    skippable_completion_error,
    timeout_completion_error,
)
from trisynapse_memory.prompts import load_prompt


def ingest_raw_source(
    store: SQLiteMemoryLogStore,
    text: str,
    *,
    group_id: str | None,
    observed_at: datetime | str | None = None,
    source_ref: dict[str, Any] | str | None = None,
    locator: dict[str, Any] | str | None = None,
    scope: dict[str, Any] | None = None,
    external_key: str | None = None,
    actor: Actor | dict[str, Any] | None = None,
    modality: str = "text",
    namespace: MemoryNamespace | dict[str, Any] | None = None,
    payload_extra: dict[str, Any] | None = None,
) -> MemoryRecord:
    if not text or not text.strip():
        raise ValueError("observation text must not be empty")
    content = text.strip()
    payload = {"content_hash": hashlib.sha256(content.encode()).hexdigest(), "modality": modality}
    payload.update(payload_extra or {})
    return store.append(
        kind="raw_source",
        text=content,
        group_id=group_id,
        observed_at=observed_at,
        source_ref=source_ref,
        locator=locator,
        scope=scope,
        external_key=external_key,
        actor=actor,
        namespace=namespace,
        payload=payload,
        confidence=1.0,
    )


def ingest_document(
    store: SQLiteMemoryLogStore,
    text: str,
    *,
    document_id: str,
    title: str | None = None,
    chunk_chars: int = 3500,
    scope: dict[str, Any] | None = None,
    observed_at: datetime | str | None = None,
    namespace: MemoryNamespace | dict[str, Any] | None = None,
    payload_extra: dict[str, Any] | None = None,
) -> list[MemoryRecord]:
    if chunk_chars < 256:
        raise ValueError("chunk_chars must be at least 256")
    chunks = chunk_document(text, chunk_chars=chunk_chars)
    group_id = f"item:{document_id}"
    source_ref = {"type": "item", "id": document_id, "title": title or document_id}
    return [
        ingest_raw_source(
            store,
            chunk,
            group_id=group_id,
            observed_at=observed_at,
            source_ref=source_ref,
            locator={"kind": "chunk", "index": index},
            scope=scope,
            external_key=f"document:{document_id}:chunk:{index}:{hashlib.sha256(chunk.encode()).hexdigest()[:16]}",
            modality="document",
            namespace=namespace,
            payload_extra={
                **(payload_extra or {}),
                "source_type": "document",
                "retrieval_fields": {
                    "title": title or document_id,
                    "section": f"chunk {index}",
                    "content": chunk,
                },
            },
        )
        for index, chunk in enumerate(chunks)
    ]


def extract_facts(
    store: SQLiteMemoryLogStore,
    group_id: str,
    complete_json: Callable[[str, str], dict[str, Any]],
    *,
    namespace: MemoryNamespace | dict[str, Any] | None = None,
) -> list[MemoryRecord]:
    return extract_facts_many(store, [group_id], complete_json, namespace=namespace)


def extract_facts_many(
    store: SQLiteMemoryLogStore,
    group_ids: Sequence[str],
    complete_json: Callable[[str, str], dict[str, Any]],
    *,
    namespace: MemoryNamespace | dict[str, Any] | None = None,
    max_workers: int = DEFAULT_COMPLETION_PARALLEL_WORKERS,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[MemoryRecord]:
    prepared = _prepared_extraction_groups(store, group_ids, namespace)
    facts_prompt = load_prompt("facts")
    result, pending = _reuse_or_queue_extractions(store, prepared, namespace)
    already = len(prepared) - len(pending)
    if already and on_progress is not None:
        on_progress(already, len(prepared))
    if not pending:
        return result
    _run_fact_extractions(
        store,
        complete_json,
        facts_prompt,
        prepared,
        pending,
        result,
        already,
        max_workers=max_workers,
        on_progress=on_progress,
    )
    return result


def _prepared_extraction_groups(
    store: SQLiteMemoryLogStore,
    group_ids: Sequence[str],
    namespace: MemoryNamespace | dict[str, Any] | None,
) -> list[tuple[str, list[MemoryRecord], str]]:
    prepared: list[tuple[str, list[MemoryRecord], str]] = []
    for group_id in group_ids:
        observations = [
            item for item in store.list_records(kinds=["raw_source"], group_prefix=group_id, namespace=namespace)
            if item.group_id == group_id
        ]
        if not observations:
            raise KeyError(f"record group has no raw sources: {group_id}")
        prepared.append((group_id, observations, _extraction_prompt(group_id, observations)))
    return prepared


def _reuse_or_queue_extractions(
    store: SQLiteMemoryLogStore,
    prepared: list[tuple[str, list[MemoryRecord], str]],
    namespace: MemoryNamespace | dict[str, Any] | None,
) -> tuple[list[MemoryRecord], list[tuple[str, list[MemoryRecord], str]]]:
    result: list[MemoryRecord] = []
    pending: list[tuple[str, list[MemoryRecord], str]] = []
    for group_id, observations, prompt in prepared:
        if _extraction_already_done(store, group_id, namespace):
            result.extend(
                item for item in store.list_records(kinds=["fact"], group_prefix=group_id, namespace=namespace)
                if item.group_id == group_id
            )
            continue
        pending.append((group_id, observations, prompt))
    return result, pending


def _run_fact_extractions(
    store: SQLiteMemoryLogStore,
    complete_json: Callable[[str, str], dict[str, Any]],
    facts_prompt: Any,
    prepared: list[tuple[str, list[MemoryRecord], str]],
    pending: list[tuple[str, list[MemoryRecord], str]],
    result: list[MemoryRecord],
    already: int,
    *,
    max_workers: int,
    on_progress: Callable[[int, int], None] | None,
) -> None:
    completed = already

    def persist(index: int, payload: dict[str, Any]) -> None:
        nonlocal completed
        group_id, observations, _ = pending[index]
        skipped = bool(payload.get("_skipped_invalid_json"))
        reason = str(payload.get("_skip_reason") or "invalid_json")
        facts = _append_extracted_facts(store, group_id, observations, payload, complete_json, facts_prompt)
        result.extend(facts)
        if skipped or not facts:
            _mark_extraction_done(store, group_id, observations, skipped=skipped, reason=reason)
        completed += 1
        if on_progress is not None:
            on_progress(completed, len(prepared))

    def recover(_index: int, exc: BaseException) -> dict[str, Any] | None:
        if not skippable_completion_error(exc):
            return None
        reason = "timeout" if timeout_completion_error(exc) else "invalid_json"
        return {"facts": [], "_skipped_invalid_json": True, "_skip_reason": reason}

    complete_json_parallel(
        complete_json,
        [(facts_prompt.text, prompt) for _, _, prompt in pending],
        max_workers=max_workers,
        on_item=persist,
        on_error=recover,
    )


def _extraction_prompt(group_id: str, observations: list[MemoryRecord]) -> str:
    timestamp = next((item.observed_at for item in observations if item.observed_at), None)
    prompt_lines = [
        "[observation_id={identifier} observed_at={observed} locator={locator}] {text}".format(
            identifier=item.id,
            observed=item.observed_at or "unknown",
            locator=item.locator or "unknown",
            text=item.text,
        )
        for item in observations
    ]
    return f"Batch: {group_id}\nBatch timestamp: {timestamp or 'unknown'}\n\n" + "\n".join(prompt_lines)


def _extraction_done_key(group_id: str) -> str:
    return f"extraction:{group_id}:completed"


def _extraction_already_done(
    store: SQLiteMemoryLogStore,
    group_id: str,
    namespace: MemoryNamespace | dict[str, Any] | None,
) -> bool:
    existing = [
        item for item in store.list_records(kinds=["fact"], group_prefix=group_id, namespace=namespace)
        if item.group_id == group_id
    ]
    return bool(existing) or store.get_by_external_key(_extraction_done_key(group_id)) is not None


def _mark_extraction_done(
    store: SQLiteMemoryLogStore,
    group_id: str,
    observations: list[MemoryRecord],
    *,
    skipped: bool,
    reason: str = "invalid_json",
) -> None:
    status = f"skipped_{reason}" if skipped else "completed"
    texts = {
        "skipped_invalid_json": "Fact extraction skipped: invalid model JSON.",
        "skipped_timeout": "Fact extraction skipped: provider timed out.",
    }
    store.append(
        kind="note",
        text=texts.get(status, "Fact extraction completed."),
        group_id=group_id,
        namespace=observations[0].namespace,
        scope=observations[0].scope,
        actor=Actor(type="ingest_pipeline", id="extraction"),
        confidence=1.0,
        payload={"extraction_status": status},
        external_key=_extraction_done_key(group_id),
    )


def _append_extracted_facts(
    store: SQLiteMemoryLogStore,
    group_id: str,
    observations: list[MemoryRecord],
    payload: dict[str, Any],
    complete_json: Callable[[str, str], dict[str, Any]],
    facts_prompt: Any,
) -> list[MemoryRecord]:
    facts = payload.get("facts") or []
    if not isinstance(facts, list):
        raise ValueError("extraction response field 'facts' must be a list")
    timestamp = next((item.observed_at for item in observations if item.observed_at), None)
    result: list[MemoryRecord] = []
    for index, fact in enumerate(facts):
        if not isinstance(fact, dict):
            continue
        record = _append_one_fact(
            store, group_id, observations, fact, index, timestamp, complete_json, facts_prompt
        )
        if record is not None:
            result.append(record)
    return result


def _append_one_fact(
    store: SQLiteMemoryLogStore,
    group_id: str,
    observations: list[MemoryRecord],
    fact: dict[str, Any],
    index: int,
    timestamp: datetime | None,
    complete_json: Callable[[str, str], dict[str, Any]],
    facts_prompt: Any,
) -> MemoryRecord | None:
    text = str(fact.get("text") or "").strip()
    if not text:
        return None
    evidence = _fact_evidence(fact, observations)
    evidence_time = next((item.observed_at for item in evidence if item.observed_at), timestamp)
    primary = evidence[0]
    return store.append(
        kind="fact",
        text=text,
        group_id=group_id,
        observed_at=evidence_time,
        evidence_refs=[item.id for item in evidence],
        subject=_optional(fact.get("subject")),
        relation=_optional(fact.get("relation")),
        object=_optional(fact.get("object")),
        temporal_anchor=_resolve_temporal(fact.get("temporal_expression"), evidence_time),
        confidence=_fact_confidence(fact.get("confidence")),
        actor=Actor(
            type="ingest_pipeline",
            id="extraction",
            model=getattr(complete_json, "model", None),
            prompt_version=facts_prompt.version,
        ),
        namespace=observations[0].namespace,
        scope=observations[0].scope,
        source_ref=primary.source_ref,
        locator=primary.locator,
        payload=_fact_payload(fact, primary, complete_json, facts_prompt),
        external_key=f"extraction:{group_id}:{index}:{hashlib.sha256(text.encode()).hexdigest()[:16]}",
    )


def _fact_payload(
    fact: dict[str, Any],
    primary: MemoryRecord,
    complete_json: Callable[[str, str], dict[str, Any]],
    facts_prompt: Any,
) -> dict[str, Any]:
    return {
        "modality": primary.payload.get("modality", "text"),
        "source_type": primary.payload.get("source_type", primary.payload.get("modality", "text")),
        "retrieval_fields": {
            **(primary.payload.get("retrieval_fields") or {}),
            "subject": _optional(fact.get("subject")) or "",
            "relation": _optional(fact.get("relation")) or "",
            "object": _optional(fact.get("object")) or "",
        },
        "generation": {
            "provider": getattr(getattr(complete_json, "settings", None), "provider", "custom"),
            "model": getattr(complete_json, "model", None),
            "prompt": facts_prompt.provenance(),
        },
    }


def _fact_evidence(fact: dict[str, Any], observations: list[MemoryRecord]) -> list[MemoryRecord]:
    """Resolve model evidence IDs to raw sources.

    Facts prompts require immutable source IDs. Known IDs are kept. Unknown IDs
    are dropped because completions invent them. If none remain, select the
    smallest set of raw sources with direct lexical support instead of assigning
    an entire record group to every fact.
    """

    supplied = _supplied_evidence(fact, observations)
    if supplied is not None:
        return supplied
    return _lexical_evidence(fact, observations)


def _supplied_evidence(fact: dict[str, Any], observations: list[MemoryRecord]) -> list[MemoryRecord] | None:
    supplied = fact.get("evidence_ids")
    if supplied is None:
        return None
    if not isinstance(supplied, list):
        raise ValueError("extraction fact evidence_ids must be a list")
    by_id = {item.id: item for item in observations}
    selected = [by_id[str(value)] for value in supplied if str(value) in by_id]
    if not selected:
        return None
    return list({item.id: item for item in selected}.values())


def _lexical_evidence(fact: dict[str, Any], observations: list[MemoryRecord]) -> list[MemoryRecord]:
    fact_text = " ".join(
        str(fact.get(key) or "")
        for key in ("subject", "relation", "object", "text", "temporal_expression")
    )
    wanted = _evidence_tokens(fact_text)
    ranked = [
        (len(wanted & _evidence_tokens(observation.text)) / max(len(wanted), 1), observation)
        for observation in observations
    ]
    ranked.sort(key=lambda item: (item[0], -item[1].seq), reverse=True)
    best = ranked[0][0]
    # Weak overlap is below the 0.12 cutoff, so the slice would be empty.
    if best < 0.12:
        return [ranked[0][1]]
    return [item for score, item in ranked[:3] if score >= max(0.12, best * 0.72)]


def _evidence_tokens(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for",
        "is", "was", "are", "it", "that", "this", "with", "from", "has",
        "had", "have",
    }
    return {
        token for token in re.findall(r"[a-z0-9']+", text.lower())
        if len(token) > 2 and token not in stop
    }


def chunk_document(text: str, *, chunk_chars: int) -> list[str]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        current = _add_paragraph_chunk(chunks, current, paragraph, chunk_chars)
    if current:
        chunks.append(current)
    return chunks or ([text.strip()] if text.strip() else [])


def _add_paragraph_chunk(
    chunks: list[str],
    current: str,
    paragraph: str,
    chunk_chars: int,
) -> str:
    if len(paragraph) > chunk_chars:
        if current:
            chunks.append(current)
        chunks.extend(paragraph[start : start + chunk_chars] for start in range(0, len(paragraph), chunk_chars))
        return ""
    candidate = f"{current}\n\n{paragraph}".strip()
    if current and len(candidate) > chunk_chars:
        chunks.append(current)
        return paragraph
    return candidate


def _resolve_temporal(expression: Any, observed_at: datetime | None) -> str | None:
    if expression is None:
        return observed_at.isoformat() if observed_at else None
    value = str(expression).strip()
    if observed_at and value.lower() in {"today", "now", "this week"}:
        return observed_at.isoformat()
    return value or None


def _optional(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _fact_confidence(value: Any) -> float:
    if value is None or value == "":
        return 0.7
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.7
    if confidence != confidence:
        return 0.7
    return max(0.0, min(1.0, confidence))
