"""Append-only formation helpers."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any, Callable

from trisynapse_memory.engine.models import Actor, MemoryDelta, MemoryNamespace
from trisynapse_memory.engine.trace import SQLiteTraceStore
from trisynapse_memory.prompts import load_prompt


def ingest_observation(
    store: SQLiteTraceStore,
    text: str,
    *,
    episode_id: str | None,
    observed_at: datetime | str | None = None,
    source_ref: dict[str, Any] | str | None = None,
    locator: dict[str, Any] | str | None = None,
    scope: dict[str, Any] | None = None,
    privacy_scope: dict[str, Any] | None = None,
    external_key: str | None = None,
    actor: Actor | dict[str, Any] | None = None,
    modality: str = "text",
    namespace: MemoryNamespace | dict[str, Any] | None = None,
    payload_extra: dict[str, Any] | None = None,
) -> MemoryDelta:
    if not text or not text.strip():
        raise ValueError("observation text must not be empty")
    content = text.strip()
    payload = {"content_hash": hashlib.sha256(content.encode()).hexdigest(), "modality": modality}
    payload.update(payload_extra or {})
    return store.append(
        kind="observation",
        text=content,
        episode_id=episode_id,
        observed_at=observed_at,
        source_ref=source_ref,
        locator=locator,
        scope=scope,
        privacy_scope=privacy_scope,
        external_key=external_key,
        actor=actor,
        namespace=namespace,
        payload=payload,
        confidence=1.0,
    )


def ingest_document(
    store: SQLiteTraceStore,
    text: str,
    *,
    document_id: str,
    title: str | None = None,
    chunk_chars: int = 3500,
    scope: dict[str, Any] | None = None,
    observed_at: datetime | str | None = None,
    namespace: MemoryNamespace | dict[str, Any] | None = None,
    payload_extra: dict[str, Any] | None = None,
) -> list[MemoryDelta]:
    if chunk_chars < 256:
        raise ValueError("chunk_chars must be at least 256")
    chunks = chunk_document(text, chunk_chars=chunk_chars)
    episode_id = f"item:{document_id}"
    source_ref = {"type": "item", "id": document_id, "title": title or document_id}
    return [
        ingest_observation(
            store,
            chunk,
            episode_id=episode_id,
            observed_at=observed_at,
            source_ref=source_ref,
            locator={"kind": "chunk", "index": index},
            scope=scope,
            external_key=f"document:{document_id}:chunk:{index}:{hashlib.sha256(chunk.encode()).hexdigest()[:16]}",
            modality="document",
            namespace=namespace,
            payload_extra=payload_extra,
        )
        for index, chunk in enumerate(chunks)
    ]


def extract_episode(
    store: SQLiteTraceStore,
    episode_id: str,
    complete_json: Callable[[str, str], dict[str, Any]],
    *,
    namespace: MemoryNamespace | dict[str, Any] | None = None,
) -> list[MemoryDelta]:
    observations = [
        item for item in store.list_deltas(kinds=["observation"], episode_prefix=episode_id, namespace=namespace)
        if item.episode_id == episode_id
    ]
    if not observations:
        raise KeyError(f"episode has no observations: {episode_id}")
    timestamp = next((item.observed_at for item in observations if item.observed_at), None)
    prompt = f"Episode: {episode_id}\nTimestamp: {timestamp or 'unknown'}\n\n" + "\n".join(item.text for item in observations)
    extraction_prompt = load_prompt("extraction")
    payload = complete_json(extraction_prompt.text, prompt)
    facts = payload.get("facts") or []
    if not isinstance(facts, list):
        raise ValueError("extraction response field 'facts' must be a list")
    result: list[MemoryDelta] = []
    for index, fact in enumerate(facts):
        text = str(fact.get("text") or "").strip()
        if not text:
            continue
        temporal = _resolve_temporal(fact.get("temporal_expression"), timestamp)
        result.append(
            store.append(
                kind="extraction",
                text=text,
                episode_id=episode_id,
                observed_at=timestamp,
                evidence_refs=[item.id for item in observations],
                subject=_optional(fact.get("subject")),
                relation=_optional(fact.get("relation")),
                object=_optional(fact.get("object")),
                temporal_anchor=temporal,
                confidence=float(fact.get("confidence", 0.7)),
                actor=Actor(
                    type="formation_pipeline",
                    id="extraction",
                    model=getattr(complete_json, "model", None),
                    prompt_version=extraction_prompt.version,
                ),
                namespace=observations[0].namespace,
                scope=observations[0].scope,
                payload={
                    "generation": {
                        "provider": getattr(getattr(complete_json, "settings", None), "provider", "custom"),
                        "model": getattr(complete_json, "model", None),
                        "prompt": extraction_prompt.provenance(),
                    }
                },
                external_key=f"extraction:{episode_id}:{index}:{hashlib.sha256(text.encode()).hexdigest()[:16]}",
            )
        )
    return result


def chunk_document(text: str, *, chunk_chars: int) -> list[str]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_chars:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(paragraph), chunk_chars):
                chunks.append(paragraph[start : start + chunk_chars])
            continue
        candidate = f"{current}\n\n{paragraph}".strip()
        if current and len(candidate) > chunk_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or ([text.strip()] if text.strip() else [])


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
