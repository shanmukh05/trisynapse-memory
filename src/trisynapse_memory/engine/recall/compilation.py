"""Recall compilers for claims and episode routing views."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from trisynapse_memory.engine.models import CompiledClaim, EpisodeRecallView, MemoryDelta
from trisynapse_memory.prompts import load_prompt


def compile_claims(
    extractions: list[MemoryDelta],
    observations_by_id: dict[str, MemoryDelta] | None = None,
    *,
    half_life_days: float = 90,
) -> list[CompiledClaim]:
    """Compile extraction evidence into reproducible, disposable claims.

    Claims group by subject/relation. Distinct objects are retained as a
    contested claim instead of selecting a winner silently.
    """

    buckets: dict[str, list[MemoryDelta]] = defaultdict(list)
    for delta in extractions:
        if not delta.text:
            continue
        subject = _normal(delta.subject or _subject_from_text(delta.text))
        relation = _normal(delta.relation or "states")
        buckets[f"{subject}|{relation}"].append(delta)

    now = datetime.now(timezone.utc)
    result: list[CompiledClaim] = []
    observations_by_id = observations_by_id or {}
    for key, group in sorted(buckets.items()):
        ordered = sorted(group, key=lambda item: (item.temporal_anchor or "", item.observed_at or item.written_at, item.seq))
        latest = ordered[-1]
        objects = {_normal(item.object or item.text) for item in ordered}
        status = "CONTESTED" if len(objects) > 1 else "ACTIVE"
        episodes = {item.episode_id for item in ordered if item.episode_id}
        avg_written_confidence = sum(item.confidence for item in ordered) / len(ordered)
        newest_time = latest.observed_at or latest.written_at
        age_days = max(0.0, (now - newest_time).total_seconds() / 86400)
        recency = math.exp(-math.log(2) * age_days / max(half_life_days, 1))
        support = min(1.0, math.log1p(len(episodes) or 1) / math.log(5))
        confidence = 0.45 * avg_written_confidence + 0.35 * support + 0.20 * recency
        if status == "CONTESTED":
            confidence *= 0.72
        observation_ids: list[str] = []
        for extraction in ordered:
            for ref in extraction.evidence_refs:
                if ref in observations_by_id and ref not in observation_ids:
                    observation_ids.append(ref)
        claim_id = "claim_" + hashlib.sha256((key + "|" + "|".join(item.id for item in ordered)).encode()).hexdigest()[:16]
        result.append(
            CompiledClaim(
                id=claim_id,
                claim_key=key,
                text=latest.text,
                status=status,
                source_delta_ids=[item.id for item in ordered],
                observation_delta_ids=observation_ids,
                temporal_anchor=latest.temporal_anchor,
                confidence=max(0.0, min(1.0, confidence)),
                subject=latest.subject,
                relation=latest.relation,
                object=latest.object,
            )
        )
    return result


def build_episode_recall_views(
    deltas: list[MemoryDelta],
    *,
    complete_json: Callable[[str, str], dict[str, Any]] | None = None,
    episode_ids: list[str] | None = None,
    build_version: int = 1,
) -> list[EpisodeRecallView]:
    """Build one routing-only view per episode.

    If a completion provider is supplied, it is called once per episode. The
    deterministic summarizer is explicit local behavior, not a substitute for
    a failed provider call: provider failures are allowed to propagate.
    """

    by_episode: dict[str, list[MemoryDelta]] = defaultdict(list)
    allowed = set(episode_ids or [])
    for delta in deltas:
        if not delta.episode_id or delta.kind not in {"observation", "extraction", "annotation"}:
            continue
        if allowed and delta.episode_id not in allowed:
            continue
        by_episode[delta.episode_id].append(delta)

    views: list[EpisodeRecallView] = []
    for episode_id, group in sorted(by_episode.items()):
        ordered = sorted(group, key=lambda item: item.seq)
        episode_prompt = load_prompt("episode_recall")
        if complete_json is not None:
            payload = complete_json(episode_prompt.text, _episode_prompt(episode_id, ordered))
            concept = str(payload.get("concept_or_topic") or episode_id).strip()
            summary = str(payload.get("summary") or "").strip()
            alt_phrasings = [str(item).strip() for item in payload.get("alt_phrasings", []) if str(item).strip()][:3]
        else:
            concept, summary, alt_phrasings = _deterministic_episode_summary(episode_id, ordered)
        source_ids = [item.id for item in ordered]
        evidence_hash = hashlib.sha256(
            json.dumps([item.id for item in ordered], separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        cache_key = hashlib.sha256(f"episode|{episode_id}|{evidence_hash}|v{build_version}".encode()).hexdigest()
        observed = next((item.observed_at for item in ordered if item.observed_at), None)
        views.append(
            EpisodeRecallView(
                id=f"episode_recall_{hashlib.sha256(episode_id.encode()).hexdigest()[:12]}",
                episode_id=episode_id,
                concept_or_topic=concept,
                summary=summary,
                alt_phrasings=alt_phrasings,
                source_trace_ids=source_ids,
                observed_at=observed,
                stale=False,
                build_version=build_version,
                cache_key=cache_key,
                generation_provenance={
                    "provider": getattr(getattr(complete_json, "settings", None), "provider", "none") if complete_json else "none",
                    "model": getattr(complete_json, "model", None) if complete_json else None,
                    "prompt": episode_prompt.provenance(),
                },
            )
        )
    return views

def _episode_prompt(episode_id: str, deltas: list[MemoryDelta]) -> str:
    lines = [f"[{item.kind}] {item.text}" for item in deltas if item.text]
    return f"Episode: {episode_id}\n\n" + "\n".join(lines)


def _deterministic_episode_summary(episode_id: str, deltas: list[MemoryDelta]) -> tuple[str, str, list[str]]:
    observations = [item.text.strip() for item in deltas if item.kind == "observation" and item.text.strip()]
    extractions = [item.text.strip() for item in deltas if item.kind == "extraction" and item.text.strip()]
    source = extractions[:4] or observations[:4]
    summary = " ".join(source)
    if len(summary) > 900:
        summary = summary[:897].rstrip() + "..."
    terms = _top_terms(" ".join(observations + extractions), limit=4)
    concept = " ".join(terms) if terms else episode_id
    alt = [f"What happened in {episode_id}?", f"Recall {' '.join(terms[:3])}" if terms else f"Find {episode_id}"]
    return concept, summary, alt


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
