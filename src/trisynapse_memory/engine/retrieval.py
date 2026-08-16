"""Staged BM25 + SBERT + graph + temporal retrieval.

Episode Recall views are indexed as routing hints. The final hit list is rebuilt
from observation/extraction deltas only, enforcing "Episode Recall routes;
Trace grounds" at the data boundary rather than relying on prompt wording.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from trisynapse_memory.engine.compilation import compile_claims
from trisynapse_memory.engine.embedding import Embedder
from trisynapse_memory.engine.models import (
    CompiledClaim,
    EpisodeRecallView,
    MemoryDelta,
    MemoryNamespace,
    MemorySearchResult,
    QueryCandidateSnapshot,
    QueryStep,
    RetrievalTrace,
    SearchHit,
)
from trisynapse_memory.engine.trace import SQLiteTraceStore
from trisynapse_memory.engine.vector_cache import VectorCache

TOKEN_RE = re.compile(r"[a-z0-9']+")
INFERENCE_RE = re.compile(r"\b(would|likely|probably|might|could|should|do you think|want to|still)\b", re.I)
TEMPORAL_RE = re.compile(r"\b(when|what year|what date|how long ago|how many times|before|after)\b", re.I)
LIST_RE = re.compile(r"\b(list|enumerate|which|how many|what .+ (books|activities|items|instruments|pets|types|changes|plans))\b", re.I)


def classify_query(query: str) -> str:
    if INFERENCE_RE.search(query):
        return "inference"
    if TEMPORAL_RE.search(query):
        return "temporal"
    if LIST_RE.search(query) or re.search(r"\bwhat .+ and\b", query, re.I):
        return "list"
    return "fact"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def bm25_score(query: str, doc: str, *, avgdl: float, df: Counter[str], n: int, k1: float = 1.5, b: float = 0.75) -> float:
    terms = tokenize(query)
    tokens = tokenize(doc)
    if not tokens:
        return 0.0
    tf = Counter(tokens)
    score = 0.0
    for term in set(terms):
        if term not in tf:
            continue
        idf = math.log(1 + (n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
        frequency = tf[term]
        denominator = frequency + k1 * (1 - b + b * len(tokens) / max(avgdl, 1))
        score += idf * (frequency * (k1 + 1)) / denominator
    return score


def cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    nl = math.sqrt(sum(a * a for a in left))
    nr = math.sqrt(sum(b * b for b in right))
    return dot / (nl * nr) if nl and nr else 0.0


def rrf_fuse(rankings: Iterable[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] += 1 / (k + rank + 1)
    return dict(scores)


@dataclass
class _Item:
    id: str
    kind: str
    text: str
    index_text: str
    confidence: float
    episode_id: str | None = None
    observed_at: Any = None
    temporal_anchor: str | None = None
    source_delta_ids: list[str] = field(default_factory=list)
    source_ref: Any = None
    locator: Any = None
    stale: bool = False
    embedding: list[float] = field(default_factory=list)


@dataclass
class _Index:
    items: dict[str, _Item]
    adjacency: dict[str, list[tuple[str, float, str]]]
    df: Counter[str]
    avgdl: float


@dataclass
class RetrieverConfig:
    top_k: int = 12
    max_context_items: int = 24
    graph_hops: int = 2
    graph_seed_top: int = 5
    margin_threshold: float = 0.018
    max_refinement_rounds: int = 2
    similar_threshold: float = 0.58
    deep_recall_enabled: bool = True


class HybridRetriever:
    def __init__(self, store: SQLiteTraceStore, embedder: Embedder, vector_cache: VectorCache, config: RetrieverConfig | None = None) -> None:
        self.store = store
        self.embedder = embedder
        self.vector_cache = vector_cache
        self.config = config or RetrieverConfig()

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        episode_prefix: str | None = None,
        scope: dict[str, Any] | None = None,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
        query_id: str,
        on_step: Callable[[QueryStep], None] | None = None,
    ) -> MemorySearchResult:
        if not query.strip():
            raise ValueError("search query must not be empty")
        memory_namespace = (
            namespace if isinstance(namespace, MemoryNamespace)
            else MemoryNamespace.model_validate(namespace or {})
        )
        sequence = 1

        def emit(
            phase: str,
            label: str,
            *,
            input: dict[str, Any] | None = None,
            output: dict[str, Any] | None = None,
            metrics: dict[str, float | int | str | bool | None] | None = None,
            candidates: list[QueryCandidateSnapshot] | None = None,
            duration_ms: float | None = None,
            parent_ids: list[str] | None = None,
        ) -> str:
            nonlocal sequence
            identifier = f"{query_id}:{sequence}:{phase}"
            step = QueryStep(
                id=identifier,
                phase=phase,
                label=label,
                sequence=sequence,
                input=input or {},
                output=output or {},
                metrics=metrics or {},
                candidates=candidates or [],
                duration_ms=duration_ms,
                parent_ids=parent_ids or [],
            )
            sequence += 1
            if on_step:
                on_step(step)
            return identifier

        index_started = time.perf_counter()
        cutoff = self.store.active_seq_cutoff()
        deltas = self.store.list_deltas(
            kinds=["observation", "extraction", "annotation"], episode_prefix=episode_prefix,
            scope=scope, seq_cutoff=cutoff,
            namespace=namespace,
        )
        observations = {item.id: item for item in deltas if item.kind == "observation"}
        extractions = [item for item in deltas if item.kind == "extraction"]
        claims = compile_claims(extractions, observations)
        episodes = [
            view for view in self.store.episode_recall_views(namespace=namespace)
            if (not episode_prefix or view.episode_id.startswith(episode_prefix))
        ]
        index = self._build_index(deltas, claims, episodes)
        classification_id = emit(
            "classification",
            "Classify query and prepare index",
            input={"query": query},
            output={
                "query_kind": classify_query(query),
                "trace_items": len(deltas),
                "compiled_claims": len(claims),
                "episode_recall_views": len(episodes),
                "index_items": len(index.items),
                "seq_cutoff": cutoff,
            },
            duration_ms=(time.perf_counter() - index_started) * 1000,
        )
        if not index.items:
            trace = RetrievalTrace(
                query_id=query_id, query=query, namespace=memory_namespace,
                query_kind=classify_query(query), stage="cold", confident=False,
            )
            self.store.write_retrieval_trace(trace)
            emit("confidence", "No searchable evidence", output={"confident": False, "stage": "cold"}, parent_ids=[classification_id])
            return MemorySearchResult(query_id=query_id, hits=[], stage="cold", confident=False, retrieval_trace=trace)

        limit = top_k or self.config.top_k
        current_query = query
        refined: str | None = None
        stage = "fast"
        escalated = False
        search_started = time.perf_counter()
        hits, diagnostics = self._search_once(current_query, index, limit=limit, graph_hops=self.config.graph_hops)
        route_id = emit(
            "routes",
            "Run hybrid retrieval routes",
            input={"query": current_query, "stage": stage},
            output={"routes": diagnostics.get("routes", {})},
            candidates=diagnostics.get("candidate_snapshots", []),
            duration_ms=(time.perf_counter() - search_started) * 1000,
            parent_ids=[classification_id],
        )
        confident = self._is_confident(diagnostics, hits)
        confidence_id = emit(
            "confidence",
            "Evaluate retrieval confidence",
            output={"confident": confident, "stage": stage},
            metrics={"top_score": diagnostics.get("top_score", 0), "margin": diagnostics.get("margin")},
            parent_ids=[route_id],
        )
        for round_number in range(self.config.max_refinement_rounds):
            if confident:
                break
            terms = _key_terms(hits)
            if not terms:
                break
            refined = f"{query} {' '.join(terms)}"
            stage = f"refine_{round_number + 1}"
            refine_started = time.perf_counter()
            hits, diagnostics = self._search_once(refined, index, limit=limit, graph_hops=self.config.graph_hops)
            confident = self._is_confident(diagnostics, hits)
            confidence_id = emit(
                "refinement",
                f"Refine query · round {round_number + 1}",
                input={"query": query, "terms": terms},
                output={"refined_query": refined, "confident": confident, "routes": diagnostics.get("routes", {})},
                metrics={"top_score": diagnostics.get("top_score", 0), "margin": diagnostics.get("margin")},
                candidates=diagnostics.get("candidate_snapshots", []),
                duration_ms=(time.perf_counter() - refine_started) * 1000,
                parent_ids=[confidence_id],
            )

        if not confident and self.config.deep_recall_enabled:
            stage = "deep_recall"
            escalated = True
            deep_limit = max(limit, self.config.max_context_items if classify_query(query) in {"list", "inference"} else limit * 2)
            deep_started = time.perf_counter()
            hits, diagnostics = self._search_once(query, index, limit=deep_limit, graph_hops=self.config.graph_hops + 1, cold=True)
            confident = self._is_confident(diagnostics, hits, relaxed=True)
            confidence_id = emit(
                "deep_recall",
                "Escalate to Deep Recall",
                input={"query": query, "limit": deep_limit, "graph_hops": self.config.graph_hops + 1},
                output={"confident": confident, "routes": diagnostics.get("routes", {})},
                metrics={"top_score": diagnostics.get("top_score", 0), "margin": diagnostics.get("margin")},
                candidates=diagnostics.get("candidate_snapshots", []),
                duration_ms=(time.perf_counter() - deep_started) * 1000,
                parent_ids=[confidence_id],
            )

        emit(
            "grounding",
            "Ground results in Trace",
            output={
                "hit_ids": [hit.item_id for hit in hits[:limit]],
                "routing_seeds": diagnostics.get("routing_seeds", []),
                "drilled_trace_count": diagnostics.get("drilled_trace_count", 0),
            },
            candidates=[_snapshot(hit, "grounded_trace", rank) for rank, hit in enumerate(hits[:20], 1)],
            parent_ids=[confidence_id],
        )

        trace = RetrievalTrace(
            query_id=query_id,
            query=query,
            namespace=memory_namespace,
            query_kind=classify_query(query),
            stage=stage,
            confident=confident,
            escalated=escalated,
            refined_query=refined,
            routes=diagnostics.get("routes", {}),
            routing_seeds=diagnostics.get("routing_seeds", []),
            drilled_trace_count=diagnostics.get("drilled_trace_count", 0),
            episode_recall_in_answer_context=sum(1 for hit in hits if hit.kind == "episode_recall"),
            top_score=diagnostics.get("top_score", 0),
            margin=diagnostics.get("margin"),
        )
        self.store.write_retrieval_trace(trace)
        return MemorySearchResult(query_id=query_id, hits=hits[:limit], stage=stage, confident=confident, retrieval_trace=trace)

    def _build_index(
        self,
        deltas: list[MemoryDelta],
        claims: list[CompiledClaim],
        episode_views: list[EpisodeRecallView],
    ) -> _Index:
        items: dict[str, _Item] = {}
        for delta in deltas:
            if delta.kind not in {"observation", "extraction"} or not delta.text:
                continue
            items[delta.id] = _Item(
                id=delta.id, kind=delta.kind, text=delta.text, index_text=delta.text, confidence=delta.confidence,
                episode_id=delta.episode_id, observed_at=delta.observed_at, temporal_anchor=delta.temporal_anchor,
                source_delta_ids=[delta.id], source_ref=delta.source_ref, locator=delta.locator,
            )
        for claim in claims:
            items[claim.id] = _Item(
                id=claim.id, kind="compiled", text=claim.text, index_text=claim.text, confidence=claim.confidence,
                temporal_anchor=claim.temporal_anchor,
                source_delta_ids=list(dict.fromkeys(claim.observation_delta_ids + claim.source_delta_ids)),
            )
        for view in episode_views:
            items[view.id] = _Item(
                id=view.id, kind="episode_recall", text=view.summary,
                index_text=" ".join([view.summary, view.concept_or_topic, *view.alt_phrasings]),
                confidence=0.95 if not view.stale else 0.55, episode_id=view.episode_id, observed_at=view.observed_at,
                source_delta_ids=view.source_trace_ids, stale=view.stale,
            )

        self._embed_items(items)
        adjacency = _build_graph(items, similar_threshold=self.config.similar_threshold)
        docs = [item.index_text for item in items.values()]
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(set(tokenize(doc)))
        avgdl = sum(len(tokenize(doc)) for doc in docs) / max(len(docs), 1)
        return _Index(items=items, adjacency=adjacency, df=df, avgdl=avgdl)

    def _embed_items(self, items: dict[str, _Item]) -> None:
        hashes = {item.id: hashlib.sha256(item.index_text.encode()).hexdigest() for item in items.values()}
        cache_key = str(getattr(self.embedder, "cache_key", self.embedder.model_name))
        cached = self.vector_cache.get(list(hashes.values()), cache_key)
        missing_ids = [item_id for item_id, text_hash in hashes.items() if text_hash not in cached]
        if missing_ids:
            vectors = self.embedder.encode([items[item_id].index_text for item_id in missing_ids])
            if len(vectors) != len(missing_ids):
                raise RuntimeError("embedding provider returned the wrong number of vectors")
            new_values = {hashes[item_id]: vector for item_id, vector in zip(missing_ids, vectors)}
            self.vector_cache.put(new_values, cache_key)
            cached.update(new_values)
        for item_id, text_hash in hashes.items():
            items[item_id].embedding = cached[text_hash]

    def _search_once(self, query: str, index: _Index, *, limit: int, graph_hops: int, cold: bool = False) -> tuple[list[SearchHit], dict[str, Any]]:
        query_vector = self.embedder.encode([query])[0]
        bm25 = sorted(
            ((item.id, bm25_score(query, item.index_text, avgdl=index.avgdl, df=index.df, n=len(index.items))) for item in index.items.values()),
            key=lambda pair: pair[1], reverse=True,
        )
        semantic = sorted(((item.id, cosine(query_vector, item.embedding)) for item in index.items.values()), key=lambda pair: pair[1], reverse=True)
        temporal = _temporal_ranking(query, index.items)
        bm25_ids = [item_id for item_id, score in bm25 if score > 0]
        semantic_ids = [item_id for item_id, score in semantic if score > 0]
        seeds = list(dict.fromkeys(bm25_ids[: self.config.graph_seed_top] + semantic_ids[: self.config.graph_seed_top]))
        graph = _graph_walk(seeds, index.adjacency, hops=graph_hops)
        graph_ids = [item_id for item_id, _, _ in graph]
        fused = rrf_fuse([bm25_ids, semantic_ids, [item_id for item_id, _ in temporal], graph_ids])
        reranked: list[tuple[str, float]] = []
        max_fused = max(fused.values(), default=1)
        for item_id in fused:
            item = index.items[item_id]
            semantic_score = cosine(query_vector, item.embedding)
            reliability = item.confidence * (0.55 if item.stale else 1)
            recency = 1 if item.observed_at or item.temporal_anchor else 0.5
            score = 0.52 * semantic_score + 0.18 * reliability + 0.10 * recency + 0.20 * fused[item_id] / max_fused
            reranked.append((item_id, score))
        reranked.sort(key=lambda pair: pair[1], reverse=True)
        grounded, routing_seeds, drilled_count = _ground_trace(query, reranked, index, graph_hops=graph_hops, cold=cold)
        scored: list[tuple[str, float]] = []
        for item_id in grounded:
            item = index.items[item_id]
            score = next((value for candidate, value in reranked if candidate == item_id), 0)
            score += 0.06 if item.kind == "observation" else 0.04
            scored.append((item_id, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        hits = [
            SearchHit(
                item_id=item_id, kind=index.items[item_id].kind, text=index.items[item_id].text, score=score,
                route="grounded_trace", episode_id=index.items[item_id].episode_id,
                observed_at=index.items[item_id].observed_at, temporal_anchor=index.items[item_id].temporal_anchor,
                source_delta_ids=index.items[item_id].source_delta_ids, source_ref=index.items[item_id].source_ref,
                locator=index.items[item_id].locator, confidence=index.items[item_id].confidence,
            )
            for item_id, score in scored[: max(limit, self.config.max_context_items)]
        ]
        top = reranked[0][1] if reranked else 0
        second = reranked[1][1] if len(reranked) > 1 else 0
        diagnostics = {
            "routes": {"bm25": bm25_ids[:10], "semantic": semantic_ids[:10], "temporal": [item[0] for item in temporal[:10]], "graph": graph_ids[:10]},
            "routing_seeds": routing_seeds,
            "drilled_trace_count": drilled_count,
            "top_score": round(top, 6),
            "margin": round(top - second, 6) if reranked else None,
            "candidate_snapshots": [
                *_route_snapshots("bm25", bm25, index),
                *_route_snapshots("semantic", semantic, index),
                *_route_snapshots("temporal", temporal, index),
                *_route_snapshots("graph", [(item_id, score) for item_id, score, _ in graph], index),
            ],
        }
        return hits, diagnostics

    def _is_confident(self, diagnostics: dict[str, Any], hits: list[SearchHit], *, relaxed: bool = False) -> bool:
        if not hits:
            return False
        threshold = self.config.margin_threshold * (0.45 if relaxed else 1)
        margin = diagnostics.get("margin") or 0
        top = diagnostics.get("top_score") or 0
        agreement = _evidence_agrees(hits[:5])
        return top > 0.12 and (margin >= threshold or agreement or len(hits) == 1) and all(hit.kind != "episode_recall" for hit in hits)


def _snapshot(hit: SearchHit, route: str, rank: int) -> QueryCandidateSnapshot:
    return QueryCandidateSnapshot(
        item_id=hit.item_id,
        kind=hit.kind,
        route=route,
        rank=rank,
        score=round(hit.score, 6),
        excerpt=hit.text[:600],
        source_delta_ids=hit.source_delta_ids,
        source_ref=hit.source_ref,
        locator=hit.locator,
    )


def _route_snapshots(
    route: str,
    ranking: Iterable[tuple[str, float]],
    index: _Index,
) -> list[QueryCandidateSnapshot]:
    snapshots: list[QueryCandidateSnapshot] = []
    for rank, (item_id, score) in enumerate(list(ranking)[:20], 1):
        item = index.items.get(item_id)
        if item is None:
            continue
        snapshots.append(QueryCandidateSnapshot(
            item_id=item.id,
            kind=item.kind,
            route=route,
            rank=rank,
            score=round(float(score), 6),
            excerpt=item.text[:600],
            source_delta_ids=item.source_delta_ids,
            source_ref=item.source_ref,
            locator=item.locator,
        ))
    return snapshots


def _build_graph(items: dict[str, _Item], *, similar_threshold: float) -> dict[str, list[tuple[str, float, str]]]:
    adjacency: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
    trace_items = [item for item in items.values() if item.kind in {"observation", "extraction"}]
    by_episode: dict[str, list[_Item]] = defaultdict(list)
    for item in trace_items:
        if item.episode_id:
            by_episode[item.episode_id].append(item)
    for group in by_episode.values():
        for left, right in zip(group, group[1:]):
            adjacency[left.id].append((right.id, 1.0, "followed_by"))
            adjacency[right.id].append((left.id, 0.8, "followed_by"))
    _add_similarity_edges(
        adjacency,
        trace_items,
        similar_threshold=similar_threshold,
        neighbors_per_item=5,
    )
    return dict(adjacency)


def _add_similarity_edges(
    adjacency: dict[str, list[tuple[str, float, str]]],
    trace_items: list[_Item],
    *,
    similar_threshold: float,
    neighbors_per_item: int,
    block_size: int = 256,
) -> None:
    """Add bounded-memory semantic neighbors using vectorized cosine search."""

    if len(trace_items) < 2:
        return
    matrix = np.asarray([item.embedding for item in trace_items], dtype=np.float32)
    if matrix.ndim != 2 or not matrix.shape[1]:
        return
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    neighbor_count = min(neighbors_per_item, len(trace_items) - 1)
    for start in range(0, len(trace_items), block_size):
        stop = min(start + block_size, len(trace_items))
        similarities = normalized[start:stop] @ normalized.T
        local_rows = np.arange(stop - start)
        similarities[local_rows, np.arange(start, stop)] = -np.inf
        candidates = np.argpartition(-similarities, neighbor_count - 1, axis=1)[:, :neighbor_count]
        for local_index, candidate_indices in enumerate(candidates):
            row = similarities[local_index]
            left = trace_items[start + local_index]
            for right_index in sorted(candidate_indices, key=lambda index: float(row[index]), reverse=True):
                score = float(row[right_index])
                if score >= similar_threshold:
                    adjacency[left.id].append((trace_items[right_index].id, score, "similar_to"))


def _graph_walk(seeds: list[str], adjacency: dict[str, list[tuple[str, float, str]]], *, hops: int) -> list[tuple[str, float, str]]:
    visited: dict[str, tuple[float, str]] = {}
    frontier = [(seed, 1.0, "seed") for seed in seeds]
    for _ in range(hops + 1):
        next_frontier: list[tuple[str, float, str]] = []
        for node, score, edge_kind in frontier:
            if node in visited and visited[node][0] >= score:
                continue
            visited[node] = (score, edge_kind)
            for target, weight, kind in adjacency.get(node, []):
                next_frontier.append((target, score * weight, kind))
        frontier = sorted(next_frontier, key=lambda item: item[1], reverse=True)[:80]
    return [(node, score, kind) for node, (score, kind) in sorted(visited.items(), key=lambda item: item[1][0], reverse=True)]


def _ground_trace(query: str, reranked: list[tuple[str, float]], index: _Index, *, graph_hops: int, cold: bool) -> tuple[list[str], list[str], int]:
    ordered = [item_id for item_id, _ in reranked]
    routing_seeds = [item_id for item_id in ordered if index.items[item_id].kind in {"episode_recall", "compiled"}][:8]
    trace_ids = {item_id for item_id in ordered[:12] if index.items[item_id].kind in {"observation", "extraction"}}
    for seed in routing_seeds:
        trace_ids.update(item_id for item_id in index.items[seed].source_delta_ids if item_id in index.items and index.items[item_id].kind in {"observation", "extraction"})
    walked = _graph_walk(list(trace_ids), index.adjacency, hops=graph_hops)
    trace_ids.update(item_id for item_id, _, _ in walked if item_id in index.items and index.items[item_id].kind in {"observation", "extraction"})
    if classify_query(query) == "inference" or cold:
        episodes = {index.items[item_id].episode_id for item_id in trace_ids if index.items[item_id].episode_id}
        trace_ids.update(item.id for item in index.items.values() if item.kind in {"observation", "extraction"} and item.episode_id in episodes)
    return [item_id for item_id in ordered if item_id in trace_ids and index.items[item_id].kind in {"observation", "extraction"}], routing_seeds, len(trace_ids)


def _temporal_ranking(query: str, items: dict[str, _Item]) -> list[tuple[str, float]]:
    cues = re.findall(r"\b(?:19|20)\d{2}\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december|when|before|after|last|next)\b", query.lower())
    if not cues:
        return []
    ranked = []
    for item in items.values():
        blob = f"{item.temporal_anchor or ''} {item.observed_at or ''} {item.text}".lower()
        score = sum(cue in blob for cue in cues) / len(cues)
        if score:
            ranked.append((item.id, score))
    return sorted(ranked, key=lambda pair: pair[1], reverse=True)


def _key_terms(hits: list[SearchHit], limit: int = 4) -> list[str]:
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "was", "are", "it", "that", "this", "with", "from"}
    counts: Counter[str] = Counter()
    for hit in hits[:5]:
        counts.update(tokenize(hit.text))
    return [token for token, _ in counts.most_common(20) if token not in stop and len(token) > 2][:limit]


def _evidence_agrees(hits: list[SearchHit]) -> bool:
    if len(hits) < 2:
        return True
    first = set(tokenize(hits[0].text))
    overlaps = []
    for hit in hits[1:]:
        other = set(tokenize(hit.text))
        if first and other:
            overlaps.append(len(first & other) / len(first | other))
    return bool(overlaps) and sum(overlaps) / len(overlaps) >= 0.07
