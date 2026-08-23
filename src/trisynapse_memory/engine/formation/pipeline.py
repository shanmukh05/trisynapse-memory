"""Append-only formation helpers."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any, Callable

from trisynapse_memory.engine.models import Actor, MemoryDelta, MemoryNamespace
from trisynapse_memory.engine.trace.store import SQLiteTraceStore
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
    prompt_lines = []
    for item in observations:
        prompt_lines.append(
            "[observation_id={identifier} observed_at={observed} locator={locator}] {text}".format(
                identifier=item.id,
                observed=item.observed_at or "unknown",
                locator=item.locator or "unknown",
                text=item.text,
            )
        )
    prompt = f"Episode: {episode_id}\nEpisode timestamp: {timestamp or 'unknown'}\n\n" + "\n".join(prompt_lines)
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
        evidence = _fact_evidence(fact, observations)
        evidence_time = next((item.observed_at for item in evidence if item.observed_at), timestamp)
        temporal = _resolve_temporal(fact.get("temporal_expression"), evidence_time)
        primary = evidence[0]
        result.append(
            store.append(
                kind="extraction",
                text=text,
                episode_id=episode_id,
                observed_at=evidence_time,
                evidence_refs=[item.id for item in evidence],
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
                source_ref=primary.source_ref,
                locator=primary.locator,
                payload={
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
                        "prompt": extraction_prompt.provenance(),
                    }
                },
                external_key=f"extraction:{episode_id}:{index}:{hashlib.sha256(text.encode()).hexdigest()[:16]}",
            )
        )
    return result


def _fact_evidence(fact: dict[str, Any], observations: list[MemoryDelta]) -> list[MemoryDelta]:
    """Resolve model evidence IDs and provide a narrow legacy fallback.

    New extraction prompts require immutable observation IDs. Older/custom
    completion providers may omit them, so we select the smallest set of
    observations with direct lexical support instead of assigning an entire
    episode to every fact.
    """

    by_id = {item.id: item for item in observations}
    supplied = fact.get("evidence_ids")
    if supplied is not None:
        if not isinstance(supplied, list):
            raise ValueError("extraction fact evidence_ids must be a list")
        unknown = [str(value) for value in supplied if str(value) not in by_id]
        if unknown:
            raise ValueError(f"extraction fact referenced unknown observation IDs: {unknown}")
        selected = [by_id[str(value)] for value in supplied]
        if selected:
            return list({item.id: item for item in selected}.values())

    fact_text = " ".join(
        str(fact.get(key) or "")
        for key in ("subject", "relation", "object", "text", "temporal_expression")
    )
    wanted = _evidence_tokens(fact_text)
    ranked: list[tuple[float, MemoryDelta]] = []
    for observation in observations:
        available = _evidence_tokens(observation.text)
        overlap = len(wanted & available) / max(len(wanted), 1)
        ranked.append((overlap, observation))
    ranked.sort(key=lambda item: (item[0], -item[1].seq), reverse=True)
    best = ranked[0][0]
    if best <= 0:
        return [observations[0]]
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
