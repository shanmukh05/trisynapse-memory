"""Staged hybrid retrieval for Ledger & Lens + Pulse & Atlas (benchmark reference).

Implements the doc-specified pipeline without dataset-specific shortcuts:
  - SBERT semantic search
  - BM25 lexical search
  - Graph walk (similar_to, followed_by, about_same_entity)
  - Temporal boost on resolved anchors
  - RRF fusion + rerank
  - Cue refinement (pattern completion)
  - Confidence gating
  - QueryLens escalation (cold / on-demand recompile)
  - Atlas-as-router drill-down (Atlas search → expand source_trace_ids → Trace-only answer context)
"""

from __future__ import annotations

MODULE_VERSION = "20260802-benchmark-harness"

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9']+")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
MONTH_NAMES = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


GROUNDED_KINDS = frozenset({"observation", "extraction", "compiled"})
TRACE_KINDS = frozenset({"observation", "extraction"})
ROUTING_KINDS = frozenset({"atlas", "compiled"})

INFERENCE_RE = re.compile(
    r"\b(would|likely|probably|might|could|should|do you think|want to|still)\b",
    re.I,
)
TEMPORAL_RE = re.compile(r"\b(when|what year|what date|how long ago|how many times)\b", re.I)
LIST_RE = re.compile(
    r"\b(how many|what .+ (books|activities|instruments|pets|types|changes|symbols|fields|plans))\b",
    re.I,
)


def classify_query(query: str) -> str:
    """Dataset-agnostic query shape for answer/retrieval policy."""
    if INFERENCE_RE.search(query):
        return "inference"
    if TEMPORAL_RE.search(query):
        return "temporal"
    if LIST_RE.search(query) or re.search(r"\bwhat .+ and\b", query, re.I):
        return "list"
    return "fact"


def answer_policy_for_query(query: str) -> dict[str, str]:
    """Instructions appended to the answer prompt based on query shape."""
    kind = classify_query(query)
    policies = {
        "fact": (
            "Give one short factual answer grounded in [observation] or [extraction] items. "
            "Prefer raw trace over compiled summaries."
        ),
        "temporal": (
            "Resolve relative dates using timestamps in context ('next month', 'last year', "
            "'week before X' means seven days prior to X, not X itself). Give an absolute date or year."
        ),
        "list": (
            "List ALL distinct items supported by context. Do not stop after the first match."
        ),
        "inference": (
            "This is an inference question. Reason from multiple [observation]/[extraction] facts. "
            "Give your best supported judgment (yes/no/likely/unlikely) even if no single sentence "
            "states the answer outright. Never abstain when related facts exist. "
            "If evidence points both ways, say so briefly and pick the better-supported side."
        ),
    }
    return {"query_kind": kind, "instruction": policies[kind]}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


@dataclass
class TraceEdge:
    target_id: str
    kind: str  # similar_to | followed_by | about_same_entity | caused_by
    weight: float = 1.0


@dataclass
class AtlasEntry:
    id: str
    concept_or_topic: str
    summary: str
    alt_phrasings: list[str] = field(default_factory=list)
    source_trace_ids: list[str] = field(default_factory=list)
    observed_at: str | None = None
    stale: bool = False
    build_version: int = 1


@dataclass
class RetrievableItem:
    item_id: str
    kind: str  # observation | extraction | compiled | atlas
    text: str
    index_texts: list[str]
    meta: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None
    strength_hot: float = 1.0
    strength_cold: float = 1.0
    confidence: float = 0.7
    access_count: int = 0


@dataclass
class SearchHit:
    item_id: str
    kind: str
    text: str
    score: float
    meta: dict[str, Any] = field(default_factory=dict)
    route: str = "fused"


@dataclass
class RetrievalResult:
    hits: list[SearchHit]
    confident: bool
    stage: str
    refined_query: str | None = None
    escalated: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


class DeltaLike(Protocol):
    id: str
    seq: int
    kind: str
    episode_id: str
    text: str
    observed_at: str | None
    subject: str | None
    relation: str | None
    obj: str | None
    temporal_anchor: str | None
    confidence: float


class ClaimLike(Protocol):
    claim_key: str
    text: str
    status: str
    source_delta_ids: list[str]
    temporal_anchor: str | None
    confidence: float


@dataclass
class MemoryIndex:
    items: dict[str, RetrievableItem]
    adjacency: dict[str, list[TraceEdge]]
    bm25_df: Counter
    bm25_avgdl: float
    bm25_n: int
    embed_model_name: str

    def get(self, item_id: str) -> RetrievableItem | None:
        return self.items.get(item_id)


def bm25_score(
    query: str,
    doc: str,
    avgdl: float,
    df: Counter,
    n: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    q_tokens = tokenize(query)
    d_tokens = tokenize(doc)
    if not d_tokens:
        return 0.0
    dl = len(d_tokens)
    tf = Counter(d_tokens)
    score = 0.0
    for term in set(q_tokens):
        if term not in tf:
            continue
        n_term = df.get(term, 0)
        idf = math.log(1 + (n - n_term + 0.5) / (n_term + 0.5))
        freq = tf[term]
        denom = freq + k1 * (1 - b + b * dl / max(avgdl, 1))
        score += idf * (freq * (k1 + 1)) / denom
    return score


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] += 1.0 / (k + rank + 1)
    return dict(scores)


def extract_temporal_cues(query: str) -> list[str]:
    cues: list[str] = []
    q = query.lower()
    for m in YEAR_RE.finditer(q):
        cues.append(m.group(0))
    for name in MONTH_NAMES:
        if name in q:
            cues.append(name)
    for token in ("when", "date", "year", "month", "ago", "last", "next"):
        if token in q:
            cues.append(token)
    return cues


def temporal_boost(text: str, cues: list[str]) -> float:
    if not cues or not text:
        return 0.0
    lower = text.lower()
    hits = sum(1 for c in cues if c in lower)
    return hits / max(len(cues), 1)


def extract_key_terms(hits: Iterable[SearchHit], k: int = 5) -> list[str]:
    freq: Counter = Counter()
    for hit in hits:
        freq.update(tokenize(hit.text))
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "is",
        "was",
        "are",
        "it",
        "that",
        "this",
        "with",
        "at",
        "by",
        "from",
        "as",
        "be",
        "have",
        "has",
        "had",
        "i",
        "you",
        "we",
        "they",
        "he",
        "she",
    }
    terms = [t for t, _ in freq.most_common(30) if t not in stop and len(t) > 2]
    return terms[:k]


def evidence_agrees(top_hits: list[SearchHit]) -> bool:
    if len(top_hits) < 2:
        return True
    top_tokens = set(tokenize(top_hits[0].text))
    if not top_tokens:
        return False
    overlaps = []
    for hit in top_hits[1:5]:
        ht = set(tokenize(hit.text))
        if not ht:
            continue
        overlaps.append(len(top_tokens & ht) / max(len(top_tokens | ht), 1))
    if not overlaps:
        return True
    return sum(overlaps) / len(overlaps) >= 0.08


def build_graph_edges(
    deltas: list[DeltaLike],
    embeddings: dict[str, np.ndarray],
    *,
    similar_threshold: float = 0.55,
    similar_top_k: int = 5,
) -> dict[str, list[TraceEdge]]:
    adj: dict[str, list[TraceEdge]] = defaultdict(list)
    by_episode: dict[str, list[DeltaLike]] = defaultdict(list)
    by_entity: dict[str, list[str]] = defaultdict(list)

    for d in deltas:
        by_episode[d.episode_id].append(d)
        if d.subject:
            by_entity[d.subject.lower().strip()].append(d.id)
        if d.obj:
            by_entity[d.obj.lower().strip()].append(d.id)

    for episode_deltas in by_episode.values():
        ordered = sorted(episode_deltas, key=lambda x: x.seq)
        for prev, nxt in zip(ordered, ordered[1:]):
            adj[prev.id].append(TraceEdge(nxt.id, "followed_by", 1.0))
            adj[nxt.id].append(TraceEdge(prev.id, "followed_by", 0.8))

    for ids in by_entity.values():
        unique = list(dict.fromkeys(ids))
        for i, a in enumerate(unique):
            for b in unique[i + 1 : i + 6]:
                if a != b:
                    adj[a].append(TraceEdge(b, "about_same_entity", 0.9))
                    adj[b].append(TraceEdge(a, "about_same_entity", 0.9))

    ids = [d.id for d in deltas if d.id in embeddings]
    for i, a_id in enumerate(ids):
        sims: list[tuple[str, float]] = []
        va = embeddings[a_id]
        for b_id in ids:
            if a_id == b_id:
                continue
            sim = cosine(va, embeddings[b_id])
            if sim >= similar_threshold:
                sims.append((b_id, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        for b_id, sim in sims[:similar_top_k]:
            adj[a_id].append(TraceEdge(b_id, "similar_to", sim))

    return dict(adj)


def graph_walk(
    seeds: list[str],
    adjacency: dict[str, list[TraceEdge]],
    *,
    hops: int = 2,
    max_nodes: int = 40,
) -> list[tuple[str, float, str]]:
    """Return (node_id, score, edge_kind) from multi-hop walk."""
    visited: dict[str, tuple[float, str]] = {}
    frontier: list[tuple[str, float, str]] = [(s, 1.0, "seed") for s in seeds]

    for _ in range(hops):
        next_frontier: list[tuple[str, float, str]] = []
        for node, score, _ in frontier:
            if node in visited:
                continue
            visited[node] = (score, _)
            for edge in adjacency.get(node, []):
                if edge.target_id not in visited:
                    next_score = score * edge.weight
                    next_frontier.append((edge.target_id, next_score, edge.kind))
        frontier = sorted(next_frontier, key=lambda x: x[1], reverse=True)[:max_nodes]

    ranked = sorted(visited.items(), key=lambda x: x[1][0], reverse=True)
    return [(node, sc, kind) for node, (sc, kind) in ranked[:max_nodes]]


def embed_texts(model: Any, texts: list[str], batch_size: int = 64) -> list[np.ndarray]:
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    return [np.asarray(v, dtype=np.float32) for v in vectors]


def build_memory_index(
    *,
    deltas: list[DeltaLike],
    claims: list[ClaimLike],
    atlas_entries: list[AtlasEntry],
    embed_model: Any,
    embed_model_name: str,
) -> MemoryIndex:
    items: dict[str, RetrievableItem] = {}

    for d in deltas:
        if not d.text:
            continue
        items[d.id] = RetrievableItem(
            item_id=d.id,
            kind=d.kind,
            text=d.text,
            index_texts=[d.text],
            meta={
                "episode_id": d.episode_id,
                "observed_at": d.observed_at,
                "temporal_anchor": d.temporal_anchor,
                "subject": d.subject,
                "relation": d.relation,
                "object": d.obj,
            },
            strength_hot=1.0,
            strength_cold=1.0,
            confidence=d.confidence,
        )

    for i, claim in enumerate(claims):
        cid = f"claim_{i}"
        items[cid] = RetrievableItem(
            item_id=cid,
            kind="compiled",
            text=claim.text,
            index_texts=[claim.text],
            meta={
                "status": claim.status,
                "temporal_anchor": claim.temporal_anchor,
                "claim_key": claim.claim_key,
                "source_delta_ids": claim.source_delta_ids,
            },
            confidence=claim.confidence,
        )

    for entry in atlas_entries:
        index_texts = [entry.summary, *entry.alt_phrasings]
        items[entry.id] = RetrievableItem(
            item_id=entry.id,
            kind="atlas",
            text=entry.summary,
            index_texts=[t for t in index_texts if t],
            meta={
                "concept_or_topic": entry.concept_or_topic,
                "observed_at": entry.observed_at,
                "source_trace_ids": entry.source_trace_ids,
                "stale": entry.stale,
            },
            confidence=0.95,
        )

    all_texts: list[str] = []
    text_owner: list[str] = []
    for item in items.values():
        for text in item.index_texts:
            if not text:
                continue
            all_texts.append(text)
            text_owner.append(item.item_id)

    embeddings = embed_texts(embed_model, all_texts)
    vec_by_item: dict[str, list[np.ndarray]] = defaultdict(list)
    for item_id, vec in zip(text_owner, embeddings):
        vec_by_item[item_id].append(vec)

    for item_id, vecs in vec_by_item.items():
        if len(vecs) == 1:
            items[item_id].embedding = vecs[0]
        else:
            stacked = np.stack(vecs)
            items[item_id].embedding = stacked.mean(axis=0)
            norm = np.linalg.norm(items[item_id].embedding)
            if norm > 0:
                items[item_id].embedding = items[item_id].embedding / norm

    delta_embeddings = {d.id: items[d.id].embedding for d in deltas if d.id in items and items[d.id].embedding is not None}
    adjacency = build_graph_edges(deltas, delta_embeddings)

    docs = [it.index_texts[0] for it in items.values()]
    df: Counter = Counter()
    for doc in docs:
        df.update(set(tokenize(doc)))
    avgdl = sum(len(tokenize(d)) for d in docs) / max(len(docs), 1)

    return MemoryIndex(
        items=items,
        adjacency=adjacency,
        bm25_df=df,
        bm25_avgdl=avgdl,
        bm25_n=len(docs),
        embed_model_name=embed_model_name,
    )


def _rank_bm25(query: str, index: MemoryIndex, candidates: set[str] | None = None) -> list[str]:
    ranked: list[tuple[str, float]] = []
    for item_id, item in index.items.items():
        if candidates is not None and item_id not in candidates:
            continue
        score = bm25_score(query, item.index_texts[0], index.bm25_avgdl, index.bm25_df, index.bm25_n)
        if score > 0:
            ranked.append((item_id, score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked]


def _rank_semantic(query: str, index: MemoryIndex, query_vec: np.ndarray, candidates: set[str] | None = None) -> list[str]:
    ranked: list[tuple[str, float]] = []
    for item_id, item in index.items.items():
        if candidates is not None and item_id not in candidates:
            continue
        if item.embedding is None:
            continue
        sim = cosine(query_vec, item.embedding)
        if sim > 0:
            ranked.append((item_id, sim))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked]


def _rank_temporal(query: str, index: MemoryIndex, candidates: set[str] | None = None) -> list[str]:
    cues = extract_temporal_cues(query)
    if not cues:
        return []
    ranked: list[tuple[str, float]] = []
    for item_id, item in index.items.items():
        if candidates is not None and item_id not in candidates:
            continue
        anchor = item.meta.get("temporal_anchor") or item.meta.get("observed_at") or ""
        score = temporal_boost(str(anchor) + " " + item.text, cues)
        if score > 0:
            ranked.append((item_id, score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked]


def _item_in_scope(item: RetrievableItem, episode_prefix: str | None) -> bool:
    if not episode_prefix:
        return True
    episode_id = str(item.meta.get("episode_id") or "")
    concept = str(item.meta.get("concept_or_topic") or "")
    return episode_id.startswith(episode_prefix) or concept.startswith(episode_prefix)


def _filter_pool(pool: set[str], index: MemoryIndex, episode_prefix: str | None) -> set[str]:
    if not episode_prefix:
        return pool
    return {i for i in pool if i in index.items and _item_in_scope(index.items[i], episode_prefix)}


def _candidate_pool(index: MemoryIndex, *, hot_only: bool, episode_prefix: str | None = None) -> set[str]:
    if not hot_only:
        pool = set(index.items.keys())
    else:
        floor = 0.3
        pool = {i for i, it in index.items.items() if it.strength_hot >= floor}
    return _filter_pool(pool, index, episode_prefix)


def _rerank(
    item_ids: list[str],
    index: MemoryIndex,
    query_vec: np.ndarray,
    fused_scores: dict[str, float],
    *,
    r_sim: float = 0.45,
    r_recency: float = 0.15,
    r_reliable: float = 0.2,
    r_access: float = 0.1,
    r_fused: float = 0.1,
) -> list[tuple[str, float]]:
    max_fused = max(fused_scores.values()) if fused_scores else 1.0
    scored: list[tuple[str, float]] = []
    for item_id in item_ids:
        item = index.items[item_id]
        sim = cosine(query_vec, item.embedding) if item.embedding is not None else 0.0
        recency = 1.0 if item.meta.get("observed_at") or item.meta.get("temporal_anchor") else 0.5
        reliable = item.confidence * (0.5 if item.meta.get("stale") else 1.0)
        access = math.log1p(item.access_count)
        fused = fused_scores.get(item_id, 0.0) / max(max_fused, 1e-9)
        score = (
            r_sim * sim
            + r_recency * recency
            + r_reliable * reliable
            + r_access * min(access, 2.0)
            + r_fused * fused
        )
        scored.append((item_id, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _to_hits(ranked: list[tuple[str, float]], index: MemoryIndex, route: str) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for item_id, score in ranked:
        item = index.items[item_id]
        hits.append(
            SearchHit(
                item_id=item_id,
                kind=item.kind,
                text=item.text,
                score=score,
                meta=dict(item.meta),
                route=route,
            )
        )
    return hits


def _is_confident(ranked: list[tuple[str, float]], hits: list[SearchHit], margin: float) -> bool:
    if not ranked:
        return False
    top_score = ranked[0][1]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin_ok = (top_score - second_score) >= margin or len(ranked) == 1
    agreement_ok = evidence_agrees(hits)
    top_hit = hits[0] if hits else None
    stale = bool(top_hit and top_hit.kind == "atlas" and top_hit.meta.get("stale"))
    no_stale_conflict = not stale
    return margin_ok and agreement_ok and no_stale_conflict and top_score > 0.05


def query_lens_recompile(
    query: str,
    extractions: list[DeltaLike],
    compile_fn: Callable[[list[DeltaLike]], list[ClaimLike]],
) -> list[ClaimLike]:
    """On-demand compilation scoped to entities/terms in the query."""
    q_tokens = set(tokenize(query))
    scoped: list[DeltaLike] = []
    for ext in extractions:
        blob = " ".join(
            filter(
                None,
                [ext.text, ext.subject, ext.relation, ext.obj],
            )
        ).lower()
        blob_tokens = set(tokenize(blob))
        if q_tokens & blob_tokens:
            scoped.append(ext)
    if not scoped:
        scoped = extractions
    return compile_fn(scoped)


def _trace_ids_from_item(item: RetrievableItem) -> set[str]:
    ids: set[str] = set()
    if item.kind in TRACE_KINDS:
        ids.add(item.item_id)
    elif item.kind == "atlas":
        ids.update(item.meta.get("source_trace_ids") or [])
    elif item.kind == "compiled":
        ids.update(item.meta.get("source_delta_ids") or [])
    return ids


def _episodes_for_ids(item_ids: Iterable[str], index: MemoryIndex) -> set[str]:
    episodes: set[str] = set()
    for item_id in item_ids:
        item = index.items.get(item_id)
        if item and item.meta.get("episode_id"):
            episodes.add(str(item.meta["episode_id"]))
    return episodes


def _trace_ids_in_episodes(episodes: set[str], index: MemoryIndex) -> set[str]:
    if not episodes:
        return set()
    return {
        item_id
        for item_id, item in index.items.items()
        if item.kind in TRACE_KINDS and item.meta.get("episode_id") in episodes
    }


def _expand_trace_neighborhood(
    seed_ids: set[str],
    index: MemoryIndex,
    *,
    graph_hops: int = 1,
    max_nodes: int = 60,
) -> set[str]:
    grounded: set[str] = set()
    for seed_id in seed_ids:
        item = index.items.get(seed_id)
        if not item:
            continue
        grounded |= _trace_ids_from_item(item)

    trace_seeds = [i for i in grounded if index.items[i].kind in TRACE_KINDS]
    walked = graph_walk(trace_seeds, index.adjacency, hops=graph_hops, max_nodes=max_nodes)
    for node_id, _, _ in walked:
        item = index.items.get(node_id)
        if item and item.kind in TRACE_KINDS:
            grounded.add(node_id)
    return grounded


def _collect_routing_seeds(
    ordered_ids: list[str],
    index: MemoryIndex,
    *,
    top_n: int = 5,
) -> list[str]:
    seeds: list[str] = []
    for item_id in ordered_ids:
        item = index.items.get(item_id)
        if not item:
            continue
        if item.kind in ROUTING_KINDS:
            seeds.append(item_id)
        if len(seeds) >= top_n:
            break
    return seeds


def ground_hits_via_atlas(
    query: str,
    index: MemoryIndex,
    query_vec: np.ndarray,
    fused: dict[str, float],
    reranked: list[tuple[str, float]],
    *,
    max_context_items: int,
    graph_hops: int = 2,
) -> tuple[list[SearchHit], dict[str, Any]]:
    """Atlas/compiled route; answer context is Trace-only (observations + extractions).

    Compiled claims may appear as supporting slots but Atlas summaries never
    appear in the returned answer context.
    """
    query_kind = classify_query(query)
    ordered_ids = [i for i, _ in reranked] if reranked else sorted(fused.keys(), key=lambda i: fused[i], reverse=True)

    direct_grounded = [
        item_id
        for item_id in ordered_ids
        if index.items.get(item_id) and index.items[item_id].kind in GROUNDED_KINDS
    ]

    routing_seeds = _collect_routing_seeds(ordered_ids, index)
    seed_trace_ids: set[str] = set()
    for seed_id in routing_seeds:
        seed_trace_ids |= _trace_ids_from_item(index.items[seed_id])

    for item_id in ordered_ids[:8]:
        item = index.items.get(item_id)
        if item and item.kind in TRACE_KINDS:
            seed_trace_ids.add(item_id)

    drill_hops = graph_hops + (1 if query_kind == "inference" else 0)
    drilled = _expand_trace_neighborhood(seed_trace_ids, index, graph_hops=drill_hops)

    if query_kind == "inference":
        episodes = _episodes_for_ids(drilled, index)
        drilled |= _trace_ids_in_episodes(episodes, index)
        q_tokens = set(tokenize(query))
        for item_id, item in index.items.items():
            if item.kind != "compiled":
                continue
            if item.meta.get("status") == "CONTESTED" or query_kind == "inference":
                blob = set(tokenize(item.text))
                if q_tokens & blob:
                    drilled |= _trace_ids_from_item(item)

    pool_ids = set(direct_grounded) | drilled
    pool_ids = {i for i in pool_ids if index.items.get(i) and index.items[i].kind in GROUNDED_KINDS}

    if not pool_ids:
        pool_ids = {i for i, _ in reranked if index.items.get(i) and index.items[i].kind != "atlas"}

    reranked_grounded = _rerank(list(pool_ids), index, query_vec, fused)
    boosted: list[tuple[str, float]] = []
    for item_id, score in reranked_grounded:
        kind = index.items[item_id].kind
        if kind == "observation":
            score += 0.06
        elif kind == "extraction":
            score += 0.04
        elif kind == "compiled":
            score += 0.01
        boosted.append((item_id, score))
    boosted.sort(key=lambda x: x[1], reverse=True)

    hits = _to_hits(boosted[:max_context_items], index, route="grounded_trace")
    diagnostics = {
        "query_kind": query_kind,
        "routing_seeds": routing_seeds,
        "drilled_trace_count": len(drilled),
        "atlas_in_answer_context": 0,
        "grounded_observation_count": sum(1 for h in hits if h.kind == "observation"),
        "grounded_extraction_count": sum(1 for h in hits if h.kind == "extraction"),
    }
    for hit in hits:
        hit.meta["grounded"] = True
    return hits, diagnostics


@dataclass
class RetrieverConfig:
    top_k: int = 12
    max_context_items: int = 24
    rrf_k: int = 60
    graph_hops: int = 2
    graph_seed_top: int = 5
    margin_threshold: float = 0.04
    max_refinement_rounds: int = 2
    hot_only_stage1: bool = True
    ground_via_atlas: bool = True


class MemoryRetriever:
    def __init__(
        self,
        index: MemoryIndex,
        embed_model: Any,
        *,
        extractions: list[DeltaLike] | None = None,
        compile_fn: Callable[[list[DeltaLike]], list[ClaimLike]] | None = None,
        config: RetrieverConfig | None = None,
    ) -> None:
        self.index = index
        self.embed_model = embed_model
        self.extractions = extractions or []
        self.compile_fn = compile_fn
        self.config = config or RetrieverConfig()

    def _search_once(
        self,
        query: str,
        *,
        hot_only: bool,
        extra_items: dict[str, RetrievableItem] | None = None,
        episode_prefix: str | None = None,
    ) -> tuple[list[SearchHit], dict[str, float], list[tuple[str, float]], dict[str, Any]]:
        cfg = self.config
        working_index = self.index
        search_diag: dict[str, Any] = {}
        if extra_items:
            merged_items = dict(self.index.items)
            merged_items.update(extra_items)
            docs = [it.index_texts[0] for it in merged_items.values()]
            df: Counter = Counter()
            for doc in docs:
                df.update(set(tokenize(doc)))
            avgdl = sum(len(tokenize(d)) for d in docs) / max(len(docs), 1)
            working_index = MemoryIndex(
                items=merged_items,
                adjacency=self.index.adjacency,
                bm25_df=df,
                bm25_avgdl=avgdl,
                bm25_n=len(docs),
                embed_model_name=self.index.embed_model_name,
            )

        pool = _candidate_pool(working_index, hot_only=hot_only, episode_prefix=episode_prefix)
        query_vec = np.asarray(
            self.embed_model.encode([query], normalize_embeddings=True)[0],
            dtype=np.float32,
        )

        bm25_rank = _rank_bm25(query, working_index, pool)
        sem_rank = _rank_semantic(query, working_index, query_vec, pool)
        temp_rank = _rank_temporal(query, working_index, pool)

        seed_ids = bm25_rank[: cfg.graph_seed_top] + sem_rank[: cfg.graph_seed_top]
        seed_ids = list(dict.fromkeys(seed_ids))
        walked = graph_walk(seed_ids, working_index.adjacency, hops=cfg.graph_hops)
        graph_rank = [node for node, _, _ in walked if node in pool]

        fused = rrf_fuse([bm25_rank, sem_rank, temp_rank, graph_rank], k=cfg.rrf_k)
        if not fused:
            return [], {}, [], {}

        ordered_ids = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)
        reranked = _rerank(ordered_ids[: cfg.top_k * 4], working_index, query_vec, fused)

        if cfg.ground_via_atlas:
            hits, search_diag = ground_hits_via_atlas(
                query,
                working_index,
                query_vec,
                fused,
                reranked,
                max_context_items=cfg.max_context_items,
                graph_hops=cfg.graph_hops,
            )
        else:
            hits = _to_hits(reranked[: cfg.max_context_items], working_index, route="fused")
            search_diag = {"query_kind": classify_query(query)}

        return hits, fused, reranked, search_diag

    def retrieve(self, query: str, *, episode_prefix: str | None = None) -> RetrievalResult:
        cfg = self.config
        stage = "fast"
        escalated = False
        refined_query: str | None = None
        current_query = query
        search_diag: dict[str, Any] = {}

        hits, fused, reranked, search_diag = self._search_once(
            current_query, hot_only=cfg.hot_only_stage1, episode_prefix=episode_prefix
        )
        confident = _is_confident(reranked, hits, cfg.margin_threshold)

        for round_no in range(cfg.max_refinement_rounds):
            if confident:
                break
            terms = extract_key_terms(hits)
            if not terms:
                break
            refined_query = f"{query} {' '.join(terms)}"
            current_query = refined_query
            stage = f"refine_{round_no + 1}"
            hits, fused, reranked, search_diag = self._search_once(
            current_query, hot_only=cfg.hot_only_stage1, episode_prefix=episode_prefix
        )
            confident = _is_confident(reranked, hits, cfg.margin_threshold)

        if not confident and self.compile_fn and self.extractions:
            stage = "query_lens"
            escalated = True
            extra_claims = query_lens_recompile(query, self.extractions, self.compile_fn)
            extra_items: dict[str, RetrievableItem] = {}
            base = len(self.index.items)
            for i, claim in enumerate(extra_claims):
                cid = f"qlens_{base + i}"
                vec = np.asarray(
                    self.embed_model.encode([claim.text], normalize_embeddings=True)[0],
                    dtype=np.float32,
                )
                extra_items[cid] = RetrievableItem(
                    item_id=cid,
                    kind="compiled",
                    text=claim.text,
                    index_texts=[claim.text],
                    meta={
                        "status": claim.status,
                        "temporal_anchor": claim.temporal_anchor,
                        "query_lens": True,
                        "source_delta_ids": claim.source_delta_ids if hasattr(claim, "source_delta_ids") else [],
                    },
                    embedding=vec,
                    confidence=claim.confidence,
                    strength_hot=0.5,
                    strength_cold=1.0,
                )
            hits, fused, reranked, search_diag = self._search_once(
                query, hot_only=False, extra_items=extra_items, episode_prefix=episode_prefix
            )
            confident = _is_confident(reranked, hits, cfg.margin_threshold * 0.5)

        if not confident:
            stage = "cold"
            hits, fused, reranked, search_diag = self._search_once(
                query, hot_only=False, episode_prefix=episode_prefix
            )
            confident = _is_confident(reranked, hits, cfg.margin_threshold * 0.5)

        top_score = reranked[0][1] if reranked else 0.0
        diagnostics = {
            "top_score": round(top_score, 4),
            "result_count": len(hits),
            "margin": round(reranked[0][1] - reranked[1][1], 4) if len(reranked) > 1 else None,
            **search_diag,
        }
        return RetrievalResult(
            hits=hits,
            confident=confident,
            stage=stage,
            refined_query=refined_query,
            escalated=escalated,
            diagnostics=diagnostics,
        )
