"""Compile disposable claims and episode-summary views over the memory log.

Owns claim grouping (subject/relation buckets) and routing-only group cards.
Callers persist views through ``MemoryEngine``; this module does not write the
store. LLM failures that are skippable fall back to a deterministic card so
ingest can finish; other provider errors still propagate.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from trisynapse_memory.engine.models import Claim, GroupSummaryView, MemoryRecord
from trisynapse_memory.engine.providers.registry import (
    DEFAULT_COMPLETION_PARALLEL_WORKERS,
    complete_json_parallel,
    skippable_completion_error,
)
from trisynapse_memory.prompts import load_prompt


def compile_claims(
    extractions: list[MemoryRecord],
    observations_by_id: dict[str, MemoryRecord] | None = None,
    *,
    half_life_days: float = 90,
) -> list[Claim]:
    """Compile extraction evidence into reproducible, disposable claims.

    Claims group by subject/relation. Distinct objects are retained as a
    contested claim instead of selecting a winner silently.
    """

    buckets = _claim_buckets(extractions)
    now = datetime.now(timezone.utc)
    observations_by_id = observations_by_id or {}
    return [
        _compile_claim_group(key, group, observations_by_id, now, half_life_days)
        for key, group in sorted(buckets.items())
    ]


def _claim_buckets(extractions: list[MemoryRecord]) -> dict[str, list[MemoryRecord]]:
    buckets: dict[str, list[MemoryRecord]] = defaultdict(list)
    for delta in extractions:
        if not delta.text:
            continue
        subject = _normal(delta.subject or _subject_from_text(delta.text))
        relation = _normal(delta.relation or "states")
        buckets[f"{subject}|{relation}"].append(delta)
    return buckets


def _compile_claim_group(
    key: str,
    group: list[MemoryRecord],
    observations: dict[str, MemoryRecord],
    now: datetime,
    half_life_days: float,
) -> Claim:
    ordered = sorted(group, key=_claim_sort_key)
    latest = ordered[-1]
    objects = _claim_objects(ordered)
    status = "CONTESTED" if len(objects) > 1 else "ACTIVE"
    groups = {item.group_id for item in ordered if item.group_id}
    claim_id = "claim_" + hashlib.sha256(
        (key + "|" + "|".join(item.id for item in ordered)).encode()
    ).hexdigest()[:16]
    return Claim(
        id=claim_id,
        claim_key=key,
        text=latest.text,
        status=status,
        source_record_ids=[item.id for item in ordered],
        raw_source_ids=_claim_observation_ids(ordered, observations),
        temporal_anchor=latest.temporal_anchor,
        confidence=_claim_confidence(ordered, groups, latest, now, half_life_days, status),
        subject=latest.subject,
        relation=latest.relation,
        object=latest.object,
        objects=objects,
    )


def _claim_sort_key(item: MemoryRecord) -> tuple[str, datetime, int]:
    return (item.temporal_anchor or "", item.observed_at or item.written_at, item.seq)


def _claim_objects(ordered: list[MemoryRecord]) -> list[str]:
    return sorted({_normal(item.object or item.text) for item in ordered} - {""})


def _claim_confidence(
    ordered: list[MemoryRecord],
    groups: set[str | None],
    latest: MemoryRecord,
    now: datetime,
    half_life_days: float,
    status: str,
) -> float:
    avg_written_confidence = sum(item.confidence for item in ordered) / len(ordered)
    newest_time = latest.observed_at or latest.written_at
    age_days = max(0.0, (now - newest_time).total_seconds() / 86400)
    recency = math.exp(-math.log(2) * age_days / max(half_life_days, 1))
    support = min(1.0, math.log1p(len(groups) or 1) / math.log(5))
    confidence = 0.45 * avg_written_confidence + 0.35 * support + 0.20 * recency
    if status == "CONTESTED":
        confidence *= 0.72
    return max(0.0, min(1.0, confidence))


def _claim_observation_ids(
    ordered: list[MemoryRecord],
    observations: dict[str, MemoryRecord],
) -> list[str]:
    observation_ids: list[str] = []
    for extraction in ordered:
        for ref in extraction.evidence_refs:
            if ref in observations and ref not in observation_ids:
                observation_ids.append(ref)
    return observation_ids


def build_group_summary_views(
    deltas: list[MemoryRecord],
    *,
    complete_json: Callable[[str, str], dict[str, Any]] | None = None,
    group_ids: list[str] | None = None,
    build_version: int = 1,
    max_workers: int = DEFAULT_COMPLETION_PARALLEL_WORKERS,
    on_progress: Callable[[int, int], None] | None = None,
    on_view: Callable[[GroupSummaryView], None] | None = None,
) -> list[GroupSummaryView]:
    """Build one routing-only view per episode.

    If a completion provider is supplied, it is called once per episode.
    Independent episode calls run concurrently. Sessions whose fact extraction
    was skipped, or whose summary completion times out or returns invalid JSON,
    keep a local deterministic card so ingest can finish. Other provider
    failures still propagate. Completed views are handed to ``on_view`` as each
    episode finishes so a later retry can skip them.
    """

    skipped_groups = _skipped_extraction_groups(deltas)
    grouped = _grouped_summary_sources(deltas, group_ids)
    group_prompt = load_prompt("group_summary")
    views: list[GroupSummaryView | None] = [None] * len(grouped)
    finished = 0

    def persist(index: int, payload: dict[str, Any] | None) -> None:
        nonlocal finished
        views[index] = _persist_group_summary_view(
            grouped[index],
            payload,
            complete_json=complete_json,
            group_prompt=group_prompt,
            build_version=build_version,
            on_view=on_view,
        )
        finished += 1
        if on_progress is not None:
            on_progress(finished, len(grouped))

    if complete_json is None:
        for index, _item in enumerate(grouped):
            persist(index, None)
        return [item for item in views if item is not None]

    _complete_group_summaries(
        complete_json,
        grouped,
        skipped_groups,
        persist,
        max_workers=max_workers,
        group_prompt=group_prompt,
    )
    return [item for item in views if item is not None]


def _grouped_summary_sources(
    deltas: list[MemoryRecord],
    group_ids: list[str] | None,
) -> list[tuple[str, list[MemoryRecord]]]:
    by_group: dict[str, list[MemoryRecord]] = defaultdict(list)
    allowed = set(group_ids or [])
    for delta in deltas:
        if not _include_in_group_summary(delta, allowed):
            continue
        by_group[delta.group_id].append(delta)
    return [
        (group_id, sorted(group, key=lambda item: item.seq))
        for group_id, group in sorted(by_group.items())
    ]


def _include_in_group_summary(delta: MemoryRecord, allowed: set[str]) -> bool:
    if not delta.group_id or delta.kind not in {"raw_source", "fact", "note"}:
        return False
    if _is_extraction_marker(delta):
        return False
    return not (allowed and delta.group_id not in allowed)


def _persist_group_summary_view(
    item: tuple[str, list[MemoryRecord]],
    payload: dict[str, Any] | None,
    *,
    complete_json: Callable[[str, str], dict[str, Any]] | None,
    group_prompt: Any,
    build_version: int,
    on_view: Callable[[GroupSummaryView], None] | None,
) -> GroupSummaryView:
    if payload and payload.get("_use_deterministic"):
        payload = None
    group_id, ordered = item
    view = _group_summary_view(
        group_id,
        ordered,
        payload,
        complete_json=None if payload is None else complete_json,
        group_prompt=group_prompt,
        build_version=build_version,
    )
    if on_view is not None:
        on_view(view)
    return view


def _complete_group_summaries(
    complete_json: Callable[[str, str], dict[str, Any]],
    grouped: list[tuple[str, list[MemoryRecord]]],
    skipped_groups: set[str],
    persist: Callable[[int, dict[str, Any] | None], None],
    *,
    max_workers: int,
    group_prompt: Any,
) -> None:
    llm_indices: list[int] = []
    requests: list[tuple[str, str]] = []
    for index, (group_id, ordered) in enumerate(grouped):
        if group_id in skipped_groups:
            persist(index, None)
            continue
        llm_indices.append(index)
        requests.append((group_prompt.text, _group_prompt(group_id, ordered)))

    def recover(_index: int, exc: BaseException) -> dict[str, Any] | None:
        if skippable_completion_error(exc):
            return {"_use_deterministic": True}
        return None

    complete_json_parallel(
        complete_json,
        requests,
        max_workers=max_workers,
        on_item=lambda req_index, payload: persist(llm_indices[req_index], payload),
        on_error=recover,
    )


def _group_summary_view(
    group_id: str,
    ordered: list[MemoryRecord],
    payload: dict[str, Any] | None,
    *,
    complete_json: Callable[[str, str], dict[str, Any]] | None,
    group_prompt: Any,
    build_version: int,
) -> GroupSummaryView:
    concept, summary, alt_phrasings = _group_summary_fields(group_id, ordered, payload)
    source_ids = [item.id for item in ordered]
    evidence_hash = hashlib.sha256(
        json.dumps([item.id for item in ordered], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_key = hashlib.sha256(f"group|{group_id}|{evidence_hash}|v{build_version}".encode()).hexdigest()
    observed = next((item.observed_at for item in ordered if item.observed_at), None)
    return GroupSummaryView(
        id=f"group_summary_{hashlib.sha256(group_id.encode()).hexdigest()[:12]}",
        group_id=group_id,
        concept_or_topic=concept,
        summary=summary,
        alt_phrasings=alt_phrasings,
        source_log_ids=source_ids,
        observed_at=observed,
        stale=False,
        build_version=build_version,
        cache_key=cache_key,
        generation_provenance=_group_summary_provenance(complete_json, group_prompt),
    )


def _group_summary_fields(
    group_id: str,
    ordered: list[MemoryRecord],
    payload: dict[str, Any] | None,
) -> tuple[str, str, list[str]]:
    if payload is None:
        return _deterministic_group_summary(group_id, ordered)
    concept = str(payload.get("concept_or_topic") or group_id).strip()
    summary = str(payload.get("summary") or "").strip()
    alt_phrasings = [
        str(item).strip() for item in payload.get("alt_phrasings", []) if str(item).strip()
    ][:3]
    return concept, summary, alt_phrasings


def _group_summary_provenance(
    complete_json: Callable[[str, str], dict[str, Any]] | None,
    group_prompt: Any,
) -> dict[str, Any]:
    if complete_json is None:
        return {"provider": "none", "model": None, "prompt": group_prompt.provenance()}
    return {
        "provider": getattr(getattr(complete_json, "settings", None), "provider", "none"),
        "model": getattr(complete_json, "model", None),
        "prompt": group_prompt.provenance(),
    }


def _group_prompt(group_id: str, deltas: list[MemoryRecord]) -> str:
    lines = [
        f"[{item.kind}] {item.text}"
        for item in deltas
        if item.text and not _is_extraction_marker(item)
    ]
    return f"Group: {group_id}\n\n" + "\n".join(lines)


def _is_extraction_marker(item: MemoryRecord) -> bool:
    return item.kind == "note" and bool((item.payload or {}).get("extraction_status"))


def _skipped_extraction_groups(deltas: list[MemoryRecord]) -> set[str]:
    return {
        item.group_id
        for item in deltas
        if item.group_id
        and item.kind == "note"
        and str((item.payload or {}).get("extraction_status") or "").startswith("skipped")
    }


def _deterministic_group_summary(group_id: str, deltas: list[MemoryRecord]) -> tuple[str, str, list[str]]:
    observations = _texts_of_kind(deltas, "raw_source")
    extractions = _texts_of_kind(deltas, "fact")
    source = extractions[:4] or observations[:4]
    terms = _top_terms(" ".join(observations + extractions), limit=4)
    concept = " ".join(terms) if terms else group_id
    return concept, _bounded_summary(" ".join(source)), _group_alt_phrasings(group_id, terms)


def _texts_of_kind(deltas: list[MemoryRecord], kind: str) -> list[str]:
    return [item.text.strip() for item in deltas if item.kind == kind and item.text.strip()]


def _bounded_summary(summary: str) -> str:
    if len(summary) <= 900:
        return summary
    return summary[:897].rstrip() + "..."


def _group_alt_phrasings(group_id: str, terms: list[str]) -> list[str]:
    recall = f"Recall {' '.join(terms[:3])}" if terms else f"Find {group_id}"
    return [f"What happened in {group_id}?", recall]


def _normal(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", value.lower()))


def _subject_from_text(text: str) -> str:
    return " ".join(text.split()[:4])


def _top_terms(text: str, *, limit: int) -> list[str]:
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "was", "are", "it", "that", "this", "with", "from", "user"}
    counts: dict[str, int] = defaultdict(int)
    for token in re.findall(r"[a-z0-9']+", text.lower()):
        if len(token) > 2 and token not in stop:
            counts[token] += 1
    return [item[0] for item in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]]
