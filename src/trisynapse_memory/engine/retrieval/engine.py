"""Staged BM25 + SBERT + graph + temporal retrieval.

Episode Recall views are indexed as routing hints. The final hit list is rebuilt
from observation/extraction deltas only, enforcing "Episode Recall routes;
Trace grounds" at the data boundary rather than relying on prompt wording.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from trisynapse_memory.engine.recall.compilation import compile_claims
from trisynapse_memory.engine.providers.embedding import Embedder
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
from trisynapse_memory.engine.trace.store import SQLiteTraceStore
from trisynapse_memory.engine.recall.vector_cache import VectorCache
from trisynapse_memory.engine.retrieval.contracts import (
    HeuristicQueryPlanner,
    QueryPlanner,
    RetrievalDocument,
    RouteContext,
    RouteRegistry,
    route_weights,
)
from trisynapse_memory.engine.utils import (
    bm25_document_score,
    cosine_similarity,
    normalize_rows,
    reciprocal_rank_fusion,
)
from trisynapse_memory.engine.retrieval.tokenization import (
    ApproximateTokenCounter,
    TokenCounter,
    lexical_tokens,
)

def classify_query(query: str) -> str:
    """Compatibility helper backed by the replaceable production planner."""

    return HeuristicQueryPlanner().classify(query)


def tokenize(text: str) -> list[str]:
    return lexical_tokens(text)


def bm25_score(query: str, doc: str, *, avgdl: float, df: Counter[str], n: int, k1: float = 1.5, b: float = 0.75) -> float:
    return bm25_document_score(
        tokenize(query),
        tokenize(doc),
        average_document_length=avgdl,
        document_frequencies=df,
        document_count=n,
        k1=k1,
        b=b,
    )


def rrf_fuse(rankings: Iterable[list[str]], k: int = 60) -> dict[str, float]:
    return reciprocal_rank_fusion(
        ((str(index), ranking) for index, ranking in enumerate(rankings)),
        rank_constant=k,
    )


def weighted_rrf_fuse(
    rankings: Iterable[tuple[str, list[str]]],
    weights: dict[str, float],
    k: int = 60,
) -> dict[str, float]:
    return reciprocal_rank_fusion(rankings, weights=weights, rank_constant=k)


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
    subject: str | None = None
    relation: str | None = None
    object: str | None = None
    modality: str = "text"
    source_type: str = "text"
    fields: dict[str, str] = field(default_factory=dict)
    seq: int = 0
    stale: bool = False
    embedding: list[float] = field(default_factory=list)


@dataclass
class _Index:
    items: dict[str, _Item]
    adjacency: dict[str, list[tuple[str, float, str]]]
    df: Counter[str]
    avgdl: float
    namespace: MemoryNamespace = field(default_factory=MemoryNamespace)


@dataclass
class RetrieverConfig:
    top_k: int = 12
    max_context_items: int = 24
    max_context_tokens: int = 6000
    per_source_context_tokens: int = 2000
    graph_hops: int = 2
    graph_seed_top: int = 5
    margin_threshold: float = 0.018
    max_refinement_rounds: int = 2
    similar_threshold: float = 0.58
    deep_recall_enabled: bool = True
    retrieval_profile: str = "auto"
    enabled_routes: tuple[str, ...] = (
        "bm25", "semantic", "temporal", "graph", "code", "table", "image",
        "document", "conversation",
    )
    route_weights: dict[str, float] = field(default_factory=dict)


class HybridRetriever:
    def __init__(
        self,
        store: SQLiteTraceStore,
        embedder: Embedder,
        vector_cache: VectorCache,
        config: RetrieverConfig | None = None,
        *,
        planner: QueryPlanner | None = None,
        routes: RouteRegistry | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.vector_cache = vector_cache
        self.config = config or RetrieverConfig()
        self.planner = planner or HeuristicQueryPlanner()
        self.routes = routes or RouteRegistry()
        self.token_counter = token_counter or ApproximateTokenCounter()

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
        index = self._build_index(deltas, claims, episodes, memory_namespace)
        initial_plan = self.planner.plan(
            query,
            available_modalities={item.modality for item in index.items.values()},
            enabled_routes=self.config.enabled_routes,
            requested_profile=self.config.retrieval_profile,
        )
        classification_id = emit(
            "classification",
            "Classify query and prepare index",
            input={"query": query},
            output={
                "query_kind": initial_plan.query_kind,
                "retrieval_profile": initial_plan.profile,
                "modalities": list(initial_plan.modalities),
                "routes": list(initial_plan.routes),
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
                query_kind=initial_plan.query_kind, stage="cold", confident=False,
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
            output={
                "routes": diagnostics.get("routes", {}),
                "query_plan": diagnostics.get("query_plan", {}),
            },
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
                output={
                    "refined_query": refined,
                    "confident": confident,
                    "routes": diagnostics.get("routes", {}),
                    "query_plan": diagnostics.get("query_plan", {}),
                },
                metrics={"top_score": diagnostics.get("top_score", 0), "margin": diagnostics.get("margin")},
                candidates=diagnostics.get("candidate_snapshots", []),
                duration_ms=(time.perf_counter() - refine_started) * 1000,
                parent_ids=[confidence_id],
            )

        force_bridge_recall = initial_plan.query_kind == "multi_hop"
        if (not confident or force_bridge_recall) and self.config.deep_recall_enabled:
            stage = "deep_recall"
            escalated = True
            deep_limit = max(
                limit,
                self.config.max_context_items
                if initial_plan.query_kind in {"list", "inference", "multi_hop"}
                else limit * 2,
            )
            query_lens_terms = _query_lens_terms(query, index)
            deep_query = f"{query} {' '.join(query_lens_terms)}".strip()
            deep_started = time.perf_counter()
            hits, diagnostics = self._search_once(deep_query, index, limit=deep_limit, graph_hops=self.config.graph_hops + 1, cold=True)
            diagnostics["query_lens_terms"] = query_lens_terms
            confident = self._is_confident(diagnostics, hits, relaxed=True)
            confidence_id = emit(
                "deep_recall",
                "Escalate to Deep Recall",
                input={"query": query, "limit": deep_limit, "graph_hops": self.config.graph_hops + 1},
                output={
                    "confident": confident,
                    "routes": diagnostics.get("routes", {}),
                    "query_lens_terms": query_lens_terms,
                },
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
                "context_tokens": diagnostics.get("context_tokens", 0),
                "token_counter": diagnostics.get("token_counter", {}),
            },
            candidates=[_snapshot(hit, "grounded_trace", rank) for rank, hit in enumerate(hits[:20], 1)],
            parent_ids=[confidence_id],
        )

        trace = RetrievalTrace(
            query_id=query_id,
            query=query,
            namespace=memory_namespace,
            query_kind=initial_plan.query_kind,
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
        namespace: MemoryNamespace,
    ) -> _Index:
        items: dict[str, _Item] = {}
        for delta in deltas:
            if delta.kind not in {"observation", "extraction"} or not delta.text:
                continue
            raw_fields = delta.payload.get("retrieval_fields") or {}
            fields = {
                str(name): _field_text(value)
                for name, value in raw_fields.items()
                if _field_text(value)
            } if isinstance(raw_fields, dict) else {}
            modality = str(delta.payload.get("modality") or "text")
            source_type = str(delta.payload.get("source_type") or modality)
            document = RetrievalDocument(
                id=delta.id, text=delta.text, modality=modality,
                source_type=source_type, fields=fields,
            )
            items[delta.id] = _Item(
                id=delta.id, kind=delta.kind, text=delta.text, index_text=document.index_text, confidence=delta.confidence,
                episode_id=delta.episode_id, observed_at=delta.observed_at, temporal_anchor=delta.temporal_anchor,
                source_delta_ids=list(dict.fromkeys([delta.id, *delta.evidence_refs])),
                source_ref=delta.source_ref, locator=delta.locator,
                subject=delta.subject, relation=delta.relation, object=delta.object,
                modality=modality, source_type=source_type, fields=fields,
                seq=delta.seq,
            )
        for claim in claims:
            items[claim.id] = _Item(
                id=claim.id, kind="compiled", text=claim.text, index_text=claim.text, confidence=claim.confidence,
                temporal_anchor=claim.temporal_anchor,
                source_delta_ids=list(dict.fromkeys(claim.observation_delta_ids + claim.source_delta_ids)),
                subject=claim.subject, relation=claim.relation, object=claim.object,
            )
        for view in episode_views:
            items[view.id] = _Item(
                id=view.id, kind="episode_recall", text=view.summary,
                index_text=" ".join([view.summary, view.concept_or_topic, *view.alt_phrasings]),
                confidence=0.95 if not view.stale else 0.55, episode_id=view.episode_id, observed_at=view.observed_at,
                source_delta_ids=view.source_trace_ids, stale=view.stale,
            )

        self._embed_items(items)
        adjacency = self.store.retrieval_graph(namespace, item_ids=items)
        self._ensure_similarity_edges(items, namespace, adjacency)
        adjacency = self.store.retrieval_graph(namespace, item_ids=items)
        # Compiled claims and Recall views are routing hints, not persisted
        # Trace documents, so merge their statistics into the durable index.
        df, persisted_avgdl, persisted_n = self.store.retrieval_index_statistics(namespace)
        dynamic = [item for item in items.values() if item.kind in {"compiled", "episode_recall"}]
        for item in dynamic:
            df.update(set(tokenize(item.index_text)))
        total_tokens = persisted_avgdl * persisted_n + sum(len(tokenize(item.index_text)) for item in dynamic)
        avgdl = total_tokens / max(persisted_n + len(dynamic), 1)
        return _Index(items=items, adjacency=adjacency, df=df, avgdl=avgdl, namespace=namespace)

    def _ensure_similarity_edges(
        self,
        items: dict[str, _Item],
        namespace: MemoryNamespace,
        adjacency: dict[str, list[tuple[str, float, str]]],
    ) -> None:
        trace_items = [item for item in items.values() if item.kind in {"observation", "extraction"}]
        if len(trace_items) < 2:
            return
        existing = {
            source
            for source, edges in adjacency.items()
            if any(kind == "similar_to" for _, _, kind in edges)
        }
        pending = [item for item in trace_items if item.id not in existing]
        if not pending:
            return
        normalized = normalize_rows([item.embedding for item in trace_items])
        by_id = {item.id: index for index, item in enumerate(trace_items)}
        edges: list[tuple[str, str, str, float]] = []
        for item in pending:
            row_index = by_id[item.id]
            scores = normalized[row_index] @ normalized.T
            scores[row_index] = -np.inf
            neighbor_count = min(5, len(trace_items) - 1)
            candidates = np.argpartition(-scores, neighbor_count - 1)[:neighbor_count]
            for candidate in sorted(candidates, key=lambda value: float(scores[value]), reverse=True):
                score = float(scores[candidate])
                if score >= self.config.similar_threshold:
                    edges.append((item.id, trace_items[candidate].id, "similar_to", score))
        if edges:
            self.store.put_retrieval_edges(namespace, edges)

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
        plan = self.planner.plan(
            query,
            available_modalities={item.modality for item in index.items.values()},
            enabled_routes=self.config.enabled_routes,
            requested_profile=self.config.retrieval_profile,
        )
        semantic_candidates: dict[str, float] | None = None
        nearest = getattr(self.vector_cache, "nearest", None)
        if callable(nearest):
            cache_key = str(getattr(self.embedder, "cache_key", self.embedder.model_name))
            nearest_hashes = nearest(
                query_vector,
                cache_key,
                max(200, self.config.max_context_items * 8),
            )
            hash_scores = dict(nearest_hashes)
            semantic_candidates = {
                item.id: hash_scores[text_hash]
                for item in index.items.values()
                if (text_hash := hashlib.sha256(item.index_text.encode()).hexdigest()) in hash_scores
            }
        context = RouteContext(
            items=index.items,
            query_vector=query_vector,
            bm25=lambda route_query, document: bm25_score(
                route_query, document, avgdl=index.avgdl, df=index.df, n=len(index.items)
            ),
            semantic=cosine_similarity,
            graph_walk=lambda seeds: _graph_walk(seeds, index.adjacency, hops=graph_hops),
            lexical_candidates={
                item_id: score
                for item_id, score in self.store.lexical_candidates(plan.query, index.namespace)
                if item_id in index.items
            },
            semantic_candidates=semantic_candidates,
        )
        enabled = self.routes.enabled(plan.routes)
        rankings: dict[str, list[tuple[str, float]]] = {}
        for route in enabled:
            if route.name == "graph":
                continue
            rankings[route.name] = route.rank(plan, context)
        context.seed_ids = list(dict.fromkeys(
            [item_id for item_id, _ in rankings.get("bm25", [])[: self.config.graph_seed_top]]
            + [item_id for item_id, _ in rankings.get("semantic", [])[: self.config.graph_seed_top]]
        ))
        for route in enabled:
            if route.name == "graph":
                rankings[route.name] = route.rank(plan, context)
        weights = route_weights(plan.profile, self.config.route_weights)
        fused = weighted_rrf_fuse(
            (
                (name, [item_id for item_id, score in ranking if score > 0])
                for name, ranking in rankings.items()
            ),
            weights,
        )
        reranked: list[tuple[str, float]] = []
        max_fused = max(fused.values(), default=1)
        for item_id in fused:
            item = index.items[item_id]
            semantic_score = cosine_similarity(query_vector, item.embedding)
            reliability = item.confidence * (0.55 if item.stale else 1)
            recency = 1 if item.observed_at or item.temporal_anchor else 0.5
            score = 0.52 * semantic_score + 0.18 * reliability + 0.10 * recency + 0.20 * fused[item_id] / max_fused
            if item.modality in plan.modalities:
                score += 0.04
            reranked.append((item_id, score))
        reranked.sort(key=lambda pair: pair[1], reverse=True)
        grounded, routing_seeds, drilled_count = _ground_trace(
            query,
            query_vector,
            reranked,
            fused,
            index,
            graph_hops=graph_hops,
            cold=cold,
            context_limit=max(limit, self.config.max_context_items),
            max_context_tokens=self.config.max_context_tokens,
            per_source_context_tokens=self.config.per_source_context_tokens,
            token_counter=self.token_counter,
        )
        scored: list[tuple[str, float]] = []
        for grounded_rank, item_id in enumerate(grounded):
            item = index.items[item_id]
            prior = next((value for candidate, value in reranked if candidate == item_id), 0)
            grounding_score = 1.0 - grounded_rank / max(len(grounded), 1)
            score = 0.65 * prior + 0.35 * grounding_score
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
                metadata={
                    "modality": index.items[item_id].modality,
                    "source_type": index.items[item_id].source_type,
                    "retrieval_fields": index.items[item_id].fields,
                    "estimated_tokens": self.token_counter.count(index.items[item_id].text),
                    "token_counter": self.token_counter.name,
                },
            )
            for item_id, score in scored[: max(limit, self.config.max_context_items)]
        ]
        top = reranked[0][1] if reranked else 0
        second = reranked[1][1] if len(reranked) > 1 else 0
        diagnostics = {
            "routes": {
                name: [item_id for item_id, _ in ranking[:10]]
                for name, ranking in rankings.items()
            },
            "query_plan": {
                "query_kind": plan.query_kind,
                "modalities": list(plan.modalities),
                "profile": plan.profile,
                "routes": list(plan.routes),
                "route_weights": weights,
            },
            "routing_seeds": routing_seeds,
            "drilled_trace_count": drilled_count,
            "top_score": round(top, 6),
            "margin": round(top - second, 6) if reranked else None,
            "context_tokens": sum(self.token_counter.count(hit.text) for hit in hits),
            "token_counter": {
                "name": self.token_counter.name,
                "exact": self.token_counter.exact,
            },
            "candidate_snapshots": [
                snapshot
                for name, ranking in rankings.items()
                for snapshot in _route_snapshots(name, ranking, index)
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


def _graph_walk(seeds: list[str], adjacency: dict[str, list[tuple[str, float, str]]], *, hops: int) -> list[tuple[str, float, str]]:
    """Bounded max-product traversal over persisted weighted graph edges.

    The best score seen for a node wins, cycles cannot increase work forever,
    and each hop retains at most 80 strongest frontier entries.
    """

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


def _ground_trace(
    query: str,
    query_vector: list[float],
    reranked: list[tuple[str, float]],
    fused: dict[str, float],
    index: _Index,
    *,
    graph_hops: int,
    cold: bool,
    context_limit: int,
    max_context_tokens: int,
    per_source_context_tokens: int,
    token_counter: TokenCounter,
) -> tuple[list[str], list[str], int]:
    ordered = [item_id for item_id, _ in reranked]
    routing_seeds = [item_id for item_id in ordered if index.items[item_id].kind in {"episode_recall", "compiled"}][:5]
    trace_ids = {
        item_id
        for item_id in ordered[: max(12, context_limit)]
        if index.items[item_id].kind in {"observation", "extraction"}
    }
    query_terms = set(tokenize(query))
    for seed in routing_seeds:
        routing_item = index.items[seed]
        source_items = [
            index.items[item_id]
            for item_id in routing_item.source_delta_ids
            if item_id in index.items and index.items[item_id].kind in {"observation", "extraction"}
        ]
        if routing_item.kind == "compiled":
            trace_ids.update(item.id for item in source_items)
        else:
            # Episode Recall chooses a region; only its best matching Trace
            # records seed graph expansion. This avoids flooding grounding with
            # several complete episodes.
            source_items.sort(
                key=lambda item: (
                    0.72 * cosine_similarity(query_vector, item.embedding)
                    + 0.28 * len(query_terms & set(tokenize(item.text))) / max(len(query_terms), 1)
                ),
                reverse=True,
            )
            trace_ids.update(item.id for item in source_items[:8])
    walked = _graph_walk(list(trace_ids), index.adjacency, hops=graph_hops)
    walk_scores = {item_id: score for item_id, score, _ in walked}
    trace_ids.update(item_id for item_id, _, _ in walked if item_id in index.items and index.items[item_id].kind in {"observation", "extraction"})
    query_kind = classify_query(query)
    if cold:
        episodes = {index.items[item_id].episode_id for item_id in trace_ids if index.items[item_id].episode_id}
        trace_ids.update(
            item.id
            for item in index.items.values()
            if item.kind in {"observation", "extraction"} and item.episode_id in episodes
        )
    if query_kind in {"inference", "multi_hop"}:
        relevant_claims = [
            item for item in index.items.values()
            if item.kind == "compiled" and set(tokenize(item.index_text)) & set(tokenize(query))
        ]
        for claim in relevant_claims:
            trace_ids.update(
                item_id for item_id in claim.source_delta_ids
                if item_id in index.items and index.items[item_id].kind in {"observation", "extraction"}
            )

    max_fused = max(fused.values(), default=1.0)
    prior_scores = dict(reranked)
    grounded_scores: list[tuple[str, float]] = []
    for item_id in trace_ids:
        item = index.items.get(item_id)
        if item is None or item.kind not in {"observation", "extraction"}:
            continue
        item_terms = set(tokenize(item.index_text))
        lexical = len(query_terms & item_terms) / max(len(query_terms), 1)
        score = (
            0.56 * cosine_similarity(query_vector, item.embedding)
            + 0.14 * lexical
            + 0.12 * (fused.get(item_id, 0.0) / max_fused)
            + 0.10 * prior_scores.get(item_id, 0.0)
            + 0.05 * item.confidence
            + 0.03 * walk_scores.get(item_id, 0.0)
        )
        if item.kind == "observation":
            score += 0.05
        grounded_scores.append((item_id, score))
    grounded_scores.sort(key=lambda pair: pair[1], reverse=True)
    selected = _diverse_grounded(
        grounded_scores,
        index,
        context_limit=context_limit,
        max_context_tokens=max_context_tokens,
        per_source_context_tokens=per_source_context_tokens,
        token_counter=token_counter,
    )
    return selected, routing_seeds, len(trace_ids)


def _diverse_grounded(
    ranked: list[tuple[str, float]],
    index: _Index,
    *,
    context_limit: int,
    max_context_tokens: int = 6000,
    per_source_context_tokens: int = 2000,
    token_counter: TokenCounter | None = None,
) -> list[str]:
    episode_count = len({index.items[item_id].episode_id for item_id, _ in ranked if index.items[item_id].episode_id})
    per_episode = max(4, context_limit // 2) if episode_count > 1 else context_limit
    selected: list[str] = []
    episode_usage: Counter[str] = Counter()
    normalized_texts: set[str] = set()
    deferred: list[str] = []
    total_tokens = 0
    source_tokens: Counter[str] = Counter()
    counter = token_counter or ApproximateTokenCounter()

    def accept(item_id: str) -> bool:
        nonlocal total_tokens
        item = index.items[item_id]
        estimated_tokens = max(1, counter.count(item.text))
        source = _source_bucket(item)
        if selected and total_tokens + estimated_tokens > max_context_tokens:
            return False
        if selected and source_tokens[source] + estimated_tokens > per_source_context_tokens:
            return False
        selected.append(item_id)
        total_tokens += estimated_tokens
        source_tokens[source] += estimated_tokens
        return True

    for item_id, _ in ranked:
        item = index.items[item_id]
        normalized = " ".join(tokenize(item.text))
        if normalized in normalized_texts:
            continue
        episode = item.episode_id or ""
        if episode and episode_usage[episode] >= per_episode:
            deferred.append(item_id)
            continue
        if not accept(item_id):
            deferred.append(item_id)
            continue
        normalized_texts.add(normalized)
        episode_usage[episode] += 1
        if len(selected) >= context_limit:
            return selected
    for item_id in deferred:
        normalized = " ".join(tokenize(index.items[item_id].text))
        if normalized in normalized_texts:
            continue
        if not accept(item_id):
            continue
        normalized_texts.add(normalized)
        if len(selected) >= context_limit:
            break
    return selected


def _source_bucket(item: _Item) -> str:
    if isinstance(item.source_ref, dict):
        identifier = item.source_ref.get("id") or item.source_ref.get("source_key")
        if identifier:
            return f"source:{identifier}"
    if item.episode_id:
        return f"episode:{item.episode_id}"
    return f"item:{item.id}"


def _query_lens_terms(query: str, index: _Index, *, limit: int = 8) -> list[str]:
    """Find entity/relation bridge terms for a difficult query.

    This is a query-scoped, disposable lens over extraction/claim structure. It
    never writes benchmark-specific information or mutates Trace.
    """

    query_terms = set(tokenize(query))
    candidates: Counter[str] = Counter()
    for item in index.items.values():
        if item.kind not in {"extraction", "compiled"}:
            continue
        fields = [item.subject, item.relation, item.object, item.text]
        blob_terms = set(tokenize(" ".join(value for value in fields if value)))
        overlap = query_terms & blob_terms
        if not overlap:
            continue
        for value in (item.subject, item.relation, item.object):
            for token in tokenize(value or ""):
                if len(token) > 2 and token not in query_terms:
                    candidates[token] += 1 + len(overlap)
    return [token for token, _ in candidates.most_common(limit)]


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


def _field_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        return " ".join(part for item in value if (part := _field_text(item)))
    if isinstance(value, dict):
        return " ".join(f"{key} {_field_text(item)}" for key, item in value.items())
    return str(value)
