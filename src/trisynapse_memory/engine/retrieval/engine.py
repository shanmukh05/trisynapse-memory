"""Hybrid retrieval: staged BM25, embeddings, graph, and temporal routes.

This module owns `HybridRetriever`. Indexes and Episode Recall views only
nominate candidates; grounding rebuilds the hit list from observation and
extraction log evidence. Query steps, fusion weights, and grounding scores
are part of the public search contract.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from trisynapse_memory.engine.indexes.claims import compile_claims
from trisynapse_memory.engine.providers.embedding import Embedder
from trisynapse_memory.engine.models import (
    Claim,
    GroupSummaryView,
    MemoryRecord,
    MemoryNamespace,
    MemorySearchResult,
    QueryCandidateSnapshot,
    QueryStep,
    RetrievalTrace,
    SearchHit,
)
from trisynapse_memory.engine.memory_log.store import SQLiteMemoryLogStore
from trisynapse_memory.engine.indexes.embedding_index import EmbeddingIndex
from trisynapse_memory.engine.retrieval.contracts import (
    HeuristicQueryPlanner,
    QueryPlan,
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
from trisynapse_memory.engine.extensions import (
    BranchResult,
    CandidateReranker,
    DefaultCandidateReranker,
    FusionStrategy,
    RetrievalBranch,
    RetrievalBranchRegistry,
    RetrievalCandidate,
    RetrievalContext,
    IndexReader,
    RerankContext,
    WeightedRRFFusion,
)

_Emit = Callable[..., str]
_Rankings = dict[str, list[tuple[str, float]]]

_TRACE_KINDS = frozenset({"raw_source", "fact"})
_EXPORTABLE_CANDIDATES = _TRACE_KINDS | {"claim", "group_summary"}
_FRESH_GROUP_CONFIDENCE = 0.95
_STALE_GROUP_CONFIDENCE = 0.55
_SIMILAR_NEIGHBOR_CAP = 5
_MIN_PER_EPISODE = 4
_ROUTING_SEED_CAP = 5
_MIN_ORDERED_TRACE = 12
_SEED_SOURCE_CAP = 8

_HIT_PRIOR_WEIGHT = 0.65
_HIT_GROUNDING_WEIGHT = 0.35
_HIT_RAW_SOURCE_BONUS = 0.06
_HIT_OTHER_BONUS = 0.04

_SEED_COSINE_WEIGHT = 0.72
_SEED_LEXICAL_WEIGHT = 0.28

_GROUND_COSINE_WEIGHT = 0.56
_GROUND_LEXICAL_WEIGHT = 0.14
_GROUND_FUSED_WEIGHT = 0.12
_GROUND_PRIOR_WEIGHT = 0.10
_GROUND_CONFIDENCE_WEIGHT = 0.05
_GROUND_WALK_WEIGHT = 0.03
_GROUND_RAW_SOURCE_BONUS = 0.05


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
    group_id: str | None = None
    observed_at: Any = None
    temporal_anchor: str | None = None
    source_record_ids: list[str] = field(default_factory=list)
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
class _PreparedIndex:
    """Loaded Trace snapshot plus the first query plan."""

    index: _Index
    plan: QueryPlan
    delta_count: int
    claim_count: int
    group_count: int
    seq_cutoff: Any
    started: float


@dataclass
class RetrieverConfig:
    top_k: int = 12
    max_context_items: int = 24
    max_context_tokens: int = 6000
    per_source_context_tokens: int = 2000
    graph_hops: int = 2
    graph_seed_top: int = 5
    margin_threshold: float = 0.018
    confidence_floor: float = 0.12
    max_refinement_rounds: int = 2
    similar_threshold: float = 0.58
    deep_summary_enabled: bool = True
    claims_enabled: bool = True
    group_summaries_enabled: bool = True
    reranker_enabled: bool = True
    raw_candidates_only: bool = False
    readout: str = "grounded"
    fusion: str = "weighted_rrf"
    rrf_rank_constant: int = 60
    retrieval_profile: str = "auto"
    enabled_routes: tuple[str, ...] = (
        "bm25", "semantic", "temporal", "graph", "code", "table", "image",
        "document", "conversation",
    )
    route_weights: dict[str, float] = field(default_factory=dict)


class HybridRetriever:
    """Fuse planner routes and custom branches, then ground hits in Trace."""

    def __init__(
        self,
        store: SQLiteMemoryLogStore,
        embedder: Embedder,
        embedding_index: EmbeddingIndex,
        config: RetrieverConfig | None = None,
        *,
        planner: QueryPlanner | None = None,
        routes: RouteRegistry | None = None,
        branches: RetrievalBranchRegistry | None = None,
        fusion_strategy: FusionStrategy | None = None,
        reranker: CandidateReranker | None = None,
        unavailable_branches: set[str] | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.embedding_index = embedding_index
        self.config = config or RetrieverConfig()
        self.planner = planner or HeuristicQueryPlanner()
        self.routes = routes or RouteRegistry()
        self.branches = branches or RetrievalBranchRegistry()
        if fusion_strategy is None or (
            type(fusion_strategy) is WeightedRRFFusion and self.config.rrf_rank_constant != 60
        ):
            fusion_strategy = WeightedRRFFusion(rank_constant=self.config.rrf_rank_constant)
        self.fusion_strategy = fusion_strategy
        self.reranker = reranker or DefaultCandidateReranker()
        self.unavailable_branches = unavailable_branches or set()
        self.token_counter = token_counter or ApproximateTokenCounter()

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        group_prefix: str | None = None,
        scope: dict[str, Any] | None = None,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
        query_id: str,
        on_step: Callable[[QueryStep], None] | None = None,
    ) -> MemorySearchResult:
        """Run hybrid retrieval and return Trace-grounded hits for one query."""

        if not query.strip():
            raise ValueError("search query must not be empty")
        memory_namespace = (
            namespace if isinstance(namespace, MemoryNamespace)
            else MemoryNamespace.model_validate(namespace or {})
        )
        emit = _emit_factory(query_id, on_step)
        prepared = self._prepare_search_index(
            query,
            group_prefix=group_prefix,
            scope=scope,
            namespace=namespace,
            memory_namespace=memory_namespace,
        )
        classification_id = emit(
            "classification",
            "Classify query and prepare index",
            input={"query": query},
            output={
                "query_kind": prepared.plan.query_kind,
                "retrieval_profile": prepared.plan.profile,
                "modalities": list(prepared.plan.modalities),
                "routes": list(prepared.plan.routes),
                "trace_items": prepared.delta_count,
                "compiled_claims": prepared.claim_count,
                "group_summary_views": prepared.group_count,
                "index_items": len(prepared.index.items),
                "seq_cutoff": prepared.seq_cutoff,
            },
            duration_ms=(time.perf_counter() - prepared.started) * 1000,
        )
        if not prepared.index.items:
            return self._empty_search_result(
                query_id, query, memory_namespace, prepared.plan, emit, classification_id,
            )

        limit = top_k or self.config.top_k
        search_started = time.perf_counter()
        hits, diagnostics = self._search_once(
            query, prepared.index, limit=limit, graph_hops=self.config.graph_hops,
        )
        route_id = _emit_route_diagnostics(
            emit,
            "routes",
            "Run hybrid retrieval routes",
            diagnostics=diagnostics,
            input={"query": query, "stage": "fast"},
            duration_ms=(time.perf_counter() - search_started) * 1000,
            parent_ids=[classification_id],
        )
        confident = self._is_confident(diagnostics, hits)
        confidence_id = emit(
            "confidence",
            "Evaluate retrieval confidence",
            output={"confident": confident, "stage": "fast"},
            metrics={"top_score": diagnostics.get("top_score", 0), "margin": diagnostics.get("margin")},
            parent_ids=[route_id],
        )
        hits, diagnostics, refined, stage, confident, confidence_id = self._maybe_refine(
            query, prepared.index, limit, hits, diagnostics, confident, emit, confidence_id,
        )
        hits, diagnostics, stage, escalated, confident, confidence_id = self._maybe_deep_summary(
            query, prepared.index, limit, prepared.plan, hits, diagnostics,
            stage, confident, emit, confidence_id,
        )
        emit(
            "grounding",
            "Pack ranked candidates" if self.config.readout == "ranked" else "Ground results in Trace",
            output={
                "hit_ids": [hit.item_id for hit in hits[:limit]],
                "routing_seeds": diagnostics.get("routing_seeds", []),
                "drilled_log_count": diagnostics.get("drilled_log_count", 0),
                "context_tokens": diagnostics.get("context_tokens", 0),
                "token_counter": diagnostics.get("token_counter", {}),
                "readout": self.config.readout,
            },
            candidates=[_snapshot(hit, hit.route, rank) for rank, hit in enumerate(hits[:20], 1)],
            parent_ids=[confidence_id],
        )
        return self._finish_search(
            query_id, query, memory_namespace, prepared.plan, stage, confident,
            escalated, refined, hits, diagnostics, limit,
        )

    def _prepare_search_index(
        self,
        query: str,
        *,
        group_prefix: str | None,
        scope: dict[str, Any] | None,
        namespace: MemoryNamespace | dict[str, Any] | None,
        memory_namespace: MemoryNamespace,
    ) -> _PreparedIndex:
        started = time.perf_counter()
        cutoff = self.store.active_seq_cutoff()
        deltas = self.store.list_records(
            kinds=["raw_source"] if self.config.raw_candidates_only else ["raw_source", "fact", "note"],
            group_prefix=group_prefix,
            scope=scope, seq_cutoff=cutoff,
            namespace=namespace,
        )
        if not self.config.raw_candidates_only:
            retracted = self.store.retracted_ids(include_dependents=True)
            deltas = [item for item in deltas if item.id not in retracted]
        claims = []
        if self.config.claims_enabled and not self.config.raw_candidates_only:
            observations = {item.id: item for item in deltas if item.kind == "raw_source"}
            extractions = [item for item in deltas if item.kind == "fact"]
            claims = compile_claims(extractions, observations)
        groups = []
        if self.config.group_summaries_enabled and not self.config.raw_candidates_only:
            groups = [
                view for view in self.store.group_summary_views(namespace=namespace)
                if not group_prefix or view.group_id.startswith(group_prefix)
            ]
        index = self._build_index(deltas, claims, groups, memory_namespace)
        plan = self.planner.plan(
            query,
            available_modalities={item.modality for item in index.items.values()},
            enabled_routes=self.config.enabled_routes,
            requested_profile=self.config.retrieval_profile,
        )
        return _PreparedIndex(
            index=index,
            plan=plan,
            delta_count=len(deltas),
            claim_count=len(claims),
            group_count=len(groups),
            seq_cutoff=cutoff,
            started=started,
        )

    def _empty_search_result(
        self,
        query_id: str,
        query: str,
        namespace: MemoryNamespace,
        plan: QueryPlan,
        emit: _Emit,
        classification_id: str,
    ) -> MemorySearchResult:
        trace = RetrievalTrace(
            query_id=query_id, query=query, namespace=namespace,
            query_kind=plan.query_kind, stage="cold", confident=False,
        )
        self.store.write_retrieval_trace(trace)
        emit(
            "confidence",
            "No searchable evidence",
            output={"confident": False, "stage": "cold"},
            parent_ids=[classification_id],
        )
        return MemorySearchResult(
            query_id=query_id, hits=[], stage="cold", confident=False, retrieval_trace=trace,
        )

    def _maybe_refine(
        self,
        query: str,
        index: _Index,
        limit: int,
        hits: list[SearchHit],
        diagnostics: dict[str, Any],
        confident: bool,
        emit: _Emit,
        confidence_id: str,
    ) -> tuple[list[SearchHit], dict[str, Any], str | None, str, bool, str]:
        refined: str | None = None
        stage = "fast"
        for round_number in range(self.config.max_refinement_rounds):
            if confident:
                break
            terms = _key_terms(hits)
            if not terms:
                break
            refined = f"{query} {' '.join(terms)}"
            stage = f"refine_{round_number + 1}"
            refine_started = time.perf_counter()
            hits, diagnostics = self._search_once(
                refined, index, limit=limit, graph_hops=self.config.graph_hops,
            )
            confident = self._is_confident(diagnostics, hits)
            confidence_id = _emit_route_diagnostics(
                emit,
                "refinement",
                f"Refine query · round {round_number + 1}",
                diagnostics=diagnostics,
                input={"query": query, "terms": terms},
                extra_output={"refined_query": refined, "confident": confident},
                metrics={"top_score": diagnostics.get("top_score", 0), "margin": diagnostics.get("margin")},
                duration_ms=(time.perf_counter() - refine_started) * 1000,
                parent_ids=[confidence_id],
            )
        return hits, diagnostics, refined, stage, confident, confidence_id

    def _maybe_deep_summary(
        self,
        query: str,
        index: _Index,
        limit: int,
        plan: QueryPlan,
        hits: list[SearchHit],
        diagnostics: dict[str, Any],
        stage: str,
        confident: bool,
        emit: _Emit,
        confidence_id: str,
    ) -> tuple[list[SearchHit], dict[str, Any], str, bool, bool, str]:
        force_bridge_recall = plan.query_kind == "multi_hop"
        if not self.config.deep_summary_enabled or not self.config.group_summaries_enabled or self.config.raw_candidates_only:
            return hits, diagnostics, stage, False, confident, confidence_id
        if confident and not force_bridge_recall:
            return hits, diagnostics, stage, False, confident, confidence_id
        deep_limit = max(
            limit,
            self.config.max_context_items
            if plan.query_kind in {"list", "inference", "multi_hop"}
            else limit * 2,
        )
        query_lens_terms = _query_lens_terms(query, index)
        deep_query = f"{query} {' '.join(query_lens_terms)}".strip()
        deep_started = time.perf_counter()
        hits, diagnostics = self._search_once(
            deep_query, index, limit=deep_limit,
            graph_hops=self.config.graph_hops + 1 if self.config.graph_hops else 0, cold=True,
        )
        diagnostics["query_lens_terms"] = query_lens_terms
        confident = self._is_confident(diagnostics, hits, relaxed=True)
        confidence_id = _emit_route_diagnostics(
            emit,
            "deep_summary",
            "Escalate to Deep Recall",
            diagnostics=diagnostics,
            input={"query": query, "limit": deep_limit,
                   "graph_hops": self.config.graph_hops + 1 if self.config.graph_hops else 0},
            extra_output={"confident": confident, "query_lens_terms": query_lens_terms},
            include_query_plan=False,
            metrics={"top_score": diagnostics.get("top_score", 0), "margin": diagnostics.get("margin")},
            duration_ms=(time.perf_counter() - deep_started) * 1000,
            parent_ids=[confidence_id],
        )
        return hits, diagnostics, "deep_summary", True, confident, confidence_id

    def _finish_search(
        self,
        query_id: str,
        query: str,
        namespace: MemoryNamespace,
        plan: QueryPlan,
        stage: str,
        confident: bool,
        escalated: bool,
        refined: str | None,
        hits: list[SearchHit],
        diagnostics: dict[str, Any],
        limit: int,
    ) -> MemorySearchResult:
        trace = RetrievalTrace(
            query_id=query_id,
            query=query,
            namespace=namespace,
            query_kind=plan.query_kind,
            stage=stage,
            confident=confident,
            escalated=escalated,
            refined_query=refined,
            routes=diagnostics.get("routes", {}),
            routing_seeds=diagnostics.get("routing_seeds", []),
            drilled_log_count=diagnostics.get("drilled_log_count", 0),
            group_summary_in_answer_context=sum(1 for hit in hits if hit.kind == "group_summary"),
            top_score=diagnostics.get("top_score", 0),
            margin=diagnostics.get("margin"),
        )
        self.store.write_retrieval_trace(trace)
        return MemorySearchResult(
            query_id=query_id, hits=hits[:limit], stage=stage, confident=confident, retrieval_trace=trace,
        )

    def _build_index(
        self,
        deltas: list[MemoryRecord],
        claims: list[Claim],
        group_views: list[GroupSummaryView],
        namespace: MemoryNamespace,
    ) -> _Index:
        items: dict[str, _Item] = {}
        for delta in deltas:
            item = _item_from_delta(delta)
            if item is None:
                continue
            if self.config.raw_candidates_only:
                item.index_text = item.text
            items[item.id] = item
        for claim in claims:
            items[claim.id] = _item_from_claim(claim)
        for view in group_views:
            items[view.id] = _item_from_group_view(view)

        if not self._lexical_baseline():
            self._embed_items(items)
        adjacency = {}
        if self.config.graph_hops:
            adjacency = self.store.retrieval_graph(namespace, item_ids=items)
            self._ensure_similarity_edges(items, namespace, adjacency)
            adjacency = self.store.retrieval_graph(namespace, item_ids=items)
        # Compiled claims and Recall views are routing hints, not persisted
        # Trace documents, so merge their statistics into the durable index.
        if self.config.raw_candidates_only:
            tokens = [tokenize(item.index_text) for item in items.values()]
            df = Counter(term for document in tokens for term in set(document))
            avgdl = sum(map(len, tokens)) / max(len(tokens), 1)
        else:
            df, avgdl = _merge_dynamic_index_stats(
                items, *self.store.retrieval_index_statistics(namespace)
            )
        return _Index(items=items, adjacency=adjacency, df=df, avgdl=avgdl, namespace=namespace)

    def _lexical_baseline(self) -> bool:
        return (
            self.config.raw_candidates_only and self.config.readout == "ranked"
            and not self.config.reranker_enabled and not self.config.graph_hops
            and self.config.enabled_routes == ("bm25",)
        )

    def _ensure_similarity_edges(
        self,
        items: dict[str, _Item],
        namespace: MemoryNamespace,
        adjacency: dict[str, list[tuple[str, float, str]]],
    ) -> None:
        trace_items = [item for item in items.values() if item.kind in _TRACE_KINDS]
        if len(trace_items) < 2:
            return
        existing = _sources_with_similar_edges(adjacency)
        pending = [item for item in trace_items if item.id not in existing]
        if not pending:
            return
        edges = _similar_edges(trace_items, pending, self.config.similar_threshold)
        if not edges:
            return
        self.store.put_retrieval_edges(namespace, edges)

    def _embed_items(self, items: dict[str, _Item]) -> None:
        hashes = {item.id: hashlib.sha256(item.index_text.encode()).hexdigest() for item in items.values()}
        cache_key = str(getattr(self.embedder, "cache_key", self.embedder.model_name))
        cached = self.embedding_index.get(list(hashes.values()), cache_key)
        missing_ids = [item_id for item_id, text_hash in hashes.items() if text_hash not in cached]
        if missing_ids:
            vectors = self.embedder.encode([items[item_id].index_text for item_id in missing_ids])
            if len(vectors) != len(missing_ids):
                raise RuntimeError("embedding provider returned the wrong number of vectors")
            new_values = {hashes[item_id]: vector for item_id, vector in zip(missing_ids, vectors)}
            self.embedding_index.put(new_values, cache_key)
            cached.update(new_values)
        for item_id, text_hash in hashes.items():
            items[item_id].embedding = cached[text_hash]

    def _search_once(
        self, query: str, index: _Index, *, limit: int, graph_hops: int, cold: bool = False,
    ) -> tuple[list[SearchHit], dict[str, Any]]:
        query_vector = [] if self._lexical_baseline() else self.embedder.encode([query])[0]
        plan = self.planner.plan(
            query,
            available_modalities={item.modality for item in index.items.values()},
            enabled_routes=self.config.enabled_routes,
            requested_profile=self.config.retrieval_profile,
        )
        context = RouteContext(
            items=index.items,
            query_vector=query_vector,
            bm25=lambda route_query, document: bm25_score(
                route_query, document, avgdl=index.avgdl, df=index.df, n=len(index.items)
            ),
            semantic=cosine_similarity,
            graph_walk=lambda seeds: _graph_walk(seeds, index.adjacency, hops=graph_hops),
            lexical_candidates=None if self.config.raw_candidates_only else {
                item_id: score
                for item_id, score in self.store.lexical_candidates(plan.query, index.namespace)
                if item_id in index.items
            },
            semantic_candidates=(
                None if self.config.raw_candidates_only else self._semantic_candidates(query_vector, index)
            ),
        )
        rankings = self._rank_enabled_routes(plan, context)
        branch_errors = self._run_retrieval_branches(plan, rankings, index)
        if any(item.kind == "index_extension" and not item.embedding for item in index.items.values()):
            self._embed_items(index.items)
        weights, fused, reranked = self._fuse_and_rerank(plan, rankings, index, query_vector)
        if self.config.readout == "ranked":
            hits = self._packed_ranked_hits(reranked, index, limit)
            return hits, self._search_diagnostics(plan, rankings, weights, reranked, [], 0, hits, branch_errors, index)
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
            raw_sources_only=self.config.readout == "raw_sources",
        )
        hits = self._score_grounded_hits(grounded, reranked, index, limit)
        return hits, self._search_diagnostics(
            plan, rankings, weights, reranked, routing_seeds, drilled_count,
            hits, branch_errors, index,
        )

    def _semantic_candidates(
        self, query_vector: list[float], index: _Index,
    ) -> dict[str, float] | None:
        nearest = getattr(self.embedding_index, "nearest", None)
        if not callable(nearest):
            return None
        cache_key = str(getattr(self.embedder, "cache_key", self.embedder.model_name))
        nearest_hashes = nearest(
            query_vector,
            cache_key,
            max(200, self.config.max_context_items * 8),
        )
        hash_scores = dict(nearest_hashes)
        return {
            item.id: hash_scores[text_hash]
            for item in index.items.values()
            if (text_hash := hashlib.sha256(item.index_text.encode()).hexdigest()) in hash_scores
        }

    def _rank_enabled_routes(self, plan: QueryPlan, context: RouteContext) -> _Rankings:
        rankings: _Rankings = {}
        pending_routes = list(self.routes.enabled(plan.routes))
        while pending_routes:
            ready = [
                route for route in pending_routes
                if set(getattr(route, "depends_on", ())).issubset(rankings)
            ]
            if not ready:
                # A disabled dependency produces an empty expansion rather than
                # silently changing the remaining independent rankings.
                ready = list(pending_routes)
            context.seed_ids = list(dict.fromkeys(
                [item_id for item_id, _ in rankings.get("bm25", [])[: self.config.graph_seed_top]]
                + [item_id for item_id, _ in rankings.get("semantic", [])[: self.config.graph_seed_top]]
            ))
            for route in ready:
                rankings[route.name] = route.rank(plan, context)
                pending_routes.remove(route)
        return rankings

    def _run_retrieval_branches(
        self,
        plan: QueryPlan,
        rankings: _Rankings,
        index: _Index,
    ) -> dict[str, str]:
        """Run optional custom branches; mutates ``rankings`` with their results."""

        branch_results = _seed_route_branch_results(rankings, index)
        branch_errors: dict[str, str] = {}
        selected = self._selected_branch_names(plan)
        if not selected:
            return branch_errors
        for level in self.branches.levels(selected, already_resolved=rankings):
            for branch in level:
                self._run_one_branch(branch, plan, index, rankings, branch_results, branch_errors)
        return branch_errors

    def _selected_branch_names(self, plan: QueryPlan) -> tuple[str, ...]:
        selected = tuple(
            name for name in plan.routes
            if name in self.branches.names and name not in self.unavailable_branches
        )
        return self._with_branch_dependencies(selected)

    def _run_one_branch(
        self,
        branch: RetrievalBranch,
        plan: QueryPlan,
        index: _Index,
        rankings: _Rankings,
        branch_results: dict[str, BranchResult],
        branch_errors: dict[str, str],
    ) -> None:
        try:
            result = self._invoke_branch(branch, plan, index, branch_results)
            candidates = _validate_branch_candidates(
                branch.spec.name,
                result.candidates[: branch.spec.max_candidates],
                index,
                self.store,
            )
            branch_results[branch.spec.name] = BranchResult(
                branch=branch.spec.name,
                candidates=tuple(candidates),
                diagnostics=result.diagnostics,
            )
            rankings[branch.spec.name] = [
                (_materialize_candidate(candidate, index), candidate.score)
                for candidate in candidates
            ]
        except Exception as exc:
            branch_errors[branch.spec.name] = str(exc)[:1000]
            rankings[branch.spec.name] = []
            branch_results[branch.spec.name] = BranchResult(
                branch=branch.spec.name,
                candidates=(),
                diagnostics={"error": str(exc)[:1000]},
            )

    def _invoke_branch(
        self,
        branch: RetrievalBranch,
        plan: QueryPlan,
        index: _Index,
        branch_results: dict[str, BranchResult],
    ) -> BranchResult:
        deadline = time.monotonic() + branch.spec.timeout_ms / 1000
        branch_context = RetrievalContext(
            namespace=index.namespace,
            seq_cutoff=self.store.active_seq_cutoff(),
            items=index.items,
            indexes=IndexReader(self.store, index.namespace),
            embedder=self.embedder,
            token_counter=self.token_counter,
            prior_results=dict(branch_results),
            deadline=deadline,
        )
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"retrieval-{branch.spec.name}",
        )
        future = executor.submit(branch.retrieve, plan, branch_context)
        try:
            returned = future.result(timeout=branch.spec.timeout_ms / 1000)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"retrieval branch timed out after {branch.spec.timeout_ms} ms"
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        if isinstance(returned, BranchResult):
            return returned
        return BranchResult(branch=branch.spec.name, candidates=tuple(returned))

    def _fuse_and_rerank(
        self,
        plan: QueryPlan,
        rankings: _Rankings,
        index: _Index,
        query_vector: list[float],
    ) -> tuple[dict[str, float], dict[str, float], list[tuple[str, float]]]:
        weights = route_weights(plan.profile, self.config.route_weights)
        for branch in self.branches.branches:
            profile_weight = branch.spec.profile_weights.get(plan.profile)
            weights.setdefault(
                branch.spec.name,
                float(profile_weight if profile_weight is not None else branch.spec.default_weight),
            )
        candidates = {
            name: [item_id for item_id, score in ranking if score > 0]
            for name, ranking in rankings.items()
        }
        if self.config.fusion == "round_robin":
            ordered = dict.fromkeys(
                item for row in zip_longest(*candidates.values()) for item in row if item is not None
            )
            fused = {item: 1.0 / rank for rank, item in enumerate(ordered, 1)}
        else:
            fused = dict(self.fusion_strategy.fuse(candidates, weights))
        if not self.config.reranker_enabled:
            return weights, fused, list(fused.items())
        reranked = self.reranker.rerank(
            plan,
            list(fused.items()),
            RerankContext(
                items=index.items,
                query_vector=query_vector,
                semantic=cosine_similarity,
            ),
        )
        return weights, fused, reranked

    def _packed_ranked_hits(
        self, ranking: list[tuple[str, float]], index: _Index, limit: int,
    ) -> list[SearchHit]:
        eligible = [(item_id, score) for item_id, score in ranking if index.items[item_id].kind in _EXPORTABLE_CANDIDATES]
        selected = _diverse_grounded(
            eligible, index, context_limit=limit, max_context_tokens=self.config.max_context_tokens,
            per_source_context_tokens=self.config.per_source_context_tokens, token_counter=self.token_counter,
        )
        scores = dict(eligible)
        return [self._hit_from_item(index.items[item_id], scores[item_id], route="ranked") for item_id in selected]

    def _score_grounded_hits(
        self,
        grounded: list[str],
        reranked: list[tuple[str, float]],
        index: _Index,
        limit: int,
    ) -> list[SearchHit]:
        scored: list[tuple[str, float]] = []
        for grounded_rank, item_id in enumerate(grounded):
            item = index.items[item_id]
            prior = next((value for candidate, value in reranked if candidate == item_id), 0)
            grounding_score = 1.0 - grounded_rank / max(len(grounded), 1)
            score = _HIT_PRIOR_WEIGHT * prior + _HIT_GROUNDING_WEIGHT * grounding_score
            score += _HIT_RAW_SOURCE_BONUS if item.kind == "raw_source" else _HIT_OTHER_BONUS
            scored.append((item_id, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [
            self._hit_from_item(index.items[item_id], score)
            for item_id, score in scored[: max(limit, self.config.max_context_items)]
        ]

    def _hit_from_item(self, item: _Item, score: float, *, route: str = "grounded_trace") -> SearchHit:
        return SearchHit(
            item_id=item.id, kind=item.kind, text=item.text, score=score,
            route=route, group_id=item.group_id,
            observed_at=item.observed_at, temporal_anchor=item.temporal_anchor,
            source_record_ids=item.source_record_ids, source_ref=item.source_ref,
            locator=item.locator, confidence=item.confidence,
            metadata={
                "modality": item.modality,
                "source_type": item.source_type,
                "retrieval_fields": item.fields,
                "estimated_tokens": self.token_counter.count(item.text),
                "token_counter": self.token_counter.name,
            },
        )

    def _search_diagnostics(
        self,
        plan: QueryPlan,
        rankings: _Rankings,
        weights: dict[str, float],
        reranked: list[tuple[str, float]],
        routing_seeds: list[str],
        drilled_count: int,
        hits: list[SearchHit],
        branch_errors: dict[str, str],
        index: _Index,
    ) -> dict[str, Any]:
        top = reranked[0][1] if reranked else 0
        second = reranked[1][1] if len(reranked) > 1 else 0
        return {
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
            "drilled_log_count": drilled_count,
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
            "branch_errors": branch_errors,
            "unavailable_branches": sorted(
                set(plan.routes) & self.unavailable_branches
            ),
            "fusion_strategy": "round_robin" if self.config.fusion == "round_robin" else self.fusion_strategy.name,
            "reranker": self.reranker.name if self.config.reranker_enabled else None,
        }

    def _with_branch_dependencies(self, selected: Iterable[str]) -> tuple[str, ...]:
        names = list(dict.fromkeys(selected))
        seen = set(names)
        cursor = 0
        while cursor < len(names):
            branch = self.branches.get(names[cursor])
            cursor += 1
            if branch is None:
                continue
            for dependency in branch.spec.depends_on:
                if (
                    dependency in self.branches.names
                    and dependency not in self.unavailable_branches
                    and dependency not in seen
                ):
                    seen.add(dependency)
                    names.append(dependency)
        return tuple(names)

    def _is_confident(self, diagnostics: dict[str, Any], hits: list[SearchHit], *, relaxed: bool = False) -> bool:
        if not hits:
            return False
        threshold = self.config.margin_threshold * (0.45 if relaxed else 1)
        margin = diagnostics.get("margin") or 0
        top = diagnostics.get("top_score") or 0
        agreement = _evidence_agrees(hits[:5])
        return (
            top > self.config.confidence_floor
            and (margin >= threshold or agreement or len(hits) == 1)
            and all(hit.kind != "group_summary" for hit in hits)
        )


def _emit_factory(query_id: str, on_step: Callable[[QueryStep], None] | None) -> _Emit:
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

    return emit


def _emit_route_diagnostics(
    emit: _Emit,
    phase: str,
    label: str,
    *,
    diagnostics: dict[str, Any],
    input: dict[str, Any] | None = None,
    extra_output: dict[str, Any] | None = None,
    include_query_plan: bool = True,
    metrics: dict[str, float | int | str | bool | None] | None = None,
    duration_ms: float | None = None,
    parent_ids: list[str] | None = None,
) -> str:
    output: dict[str, Any] = dict(extra_output or {})
    output["routes"] = diagnostics.get("routes", {})
    if include_query_plan:
        output["query_plan"] = diagnostics.get("query_plan", {})
    output["branch_errors"] = diagnostics.get("branch_errors", {})
    output["unavailable_branches"] = diagnostics.get("unavailable_branches", [])
    output["fusion_strategy"] = diagnostics.get("fusion_strategy")
    output["reranker"] = diagnostics.get("reranker")
    return emit(
        phase,
        label,
        input=input,
        output=output,
        metrics=metrics,
        candidates=diagnostics.get("candidate_snapshots", []),
        duration_ms=duration_ms,
        parent_ids=parent_ids,
    )


def _seed_route_branch_results(rankings: _Rankings, index: _Index) -> dict[str, BranchResult]:
    return {
        name: BranchResult(
            branch=name,
            candidates=tuple(
                _candidate_from_index(name, item_id, score, index)
                for item_id, score in ranking
                if item_id in index.items
            ),
        )
        for name, ranking in rankings.items()
    }


def _candidate_from_index(
    branch: str,
    item_id: str,
    score: float,
    index: _Index,
) -> RetrievalCandidate:
    item = index.items[item_id]
    is_trace = item.kind in _TRACE_KINDS
    return RetrievalCandidate(
        id=item_id,
        branch=branch,
        score=float(score),
        kind="log" if is_trace else "index",
        evidence_record_ids=(item_id,) if is_trace else tuple(item.source_record_ids),
        text=item.text,
        source_ref=item.source_ref,
        locator=item.locator,
        metadata={"kind": item.kind, "modality": item.modality},
    )


def _validate_branch_candidates(
    branch: str,
    candidates: Sequence[RetrievalCandidate],
    index: _Index,
    store: SQLiteMemoryLogStore,
) -> list[RetrievalCandidate]:
    del store  # Index is already scoped to the active Trace snapshot.
    result: list[RetrievalCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        canonical = _checked_branch_candidate(branch, candidate, index)
        if canonical is None or canonical in seen:
            continue
        seen.add(canonical)
        result.append(candidate)
    return result


def _checked_branch_candidate(
    branch: str,
    candidate: RetrievalCandidate,
    index: _Index,
) -> str | None:
    if not isinstance(candidate, RetrievalCandidate):
        raise TypeError(f"retrieval branch {branch} returned a non-RetrievalCandidate value")
    if candidate.branch != branch:
        raise ValueError(f"retrieval branch {branch} returned a candidate for {candidate.branch}")
    if candidate.score <= 0:
        return None
    _require_valid_evidence(branch, candidate, index)
    return candidate.canonical_id


def _require_valid_evidence(
    branch: str,
    candidate: RetrievalCandidate,
    index: _Index,
) -> None:
    evidence = tuple(dict.fromkeys(candidate.evidence_record_ids))
    if not evidence or _invalid_active_evidence(evidence, index):
        raise ValueError(f"retrieval branch {branch} returned invalid active evidence")
    if candidate.kind == "log" and len(evidence) != 1:
        raise ValueError("Log candidates must identify exactly one evidence record")
    if candidate.kind == "index" and not candidate.text:
        raise ValueError("Index candidates require routing text")


def _invalid_active_evidence(evidence: Sequence[str], index: _Index) -> bool:
    return any(
        item_id not in index.items or index.items[item_id].kind not in _TRACE_KINDS
        for item_id in evidence
    )


def _materialize_candidate(candidate: RetrievalCandidate, index: _Index) -> str:
    if candidate.kind == "log":
        return candidate.evidence_record_ids[0]
    item_id = candidate.canonical_id
    metadata = dict(candidate.metadata)
    fields = {
        str(key): str(value)
        for key, value in (metadata.get("fields") or {}).items()
        if value is not None
    }
    index.items[item_id] = _Item(
        id=item_id,
        kind="index_extension",
        text=candidate.text or "",
        index_text=candidate.text or "",
        confidence=float(metadata.get("confidence", 0.8)),
        source_record_ids=list(candidate.evidence_record_ids),
        source_ref=candidate.source_ref,
        locator=candidate.locator,
        modality=str(metadata.get("modality") or "text"),
        source_type=str(metadata.get("source_type") or candidate.channel_id or "recall"),
        fields=fields,
        stale=bool(metadata.get("stale", False)),
    )
    return item_id


def _snapshot(hit: SearchHit, route: str, rank: int) -> QueryCandidateSnapshot:
    return QueryCandidateSnapshot(
        item_id=hit.item_id,
        kind=hit.kind,
        route=route,
        rank=rank,
        score=round(hit.score, 6),
        excerpt=hit.text[:600],
        source_record_ids=hit.source_record_ids,
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
            source_record_ids=item.source_record_ids,
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
    raw_sources_only: bool = False,
) -> tuple[list[str], list[str], int]:
    """Rebuild the hit list from Trace log evidence nominated by index routes."""

    ordered = [item_id for item_id, _ in reranked]
    routing_seeds = [
        item_id for item_id in ordered
        if index.items[item_id].kind in {"group_summary", "claim", "index_extension"}
    ][:_ROUTING_SEED_CAP]
    query_terms = set(tokenize(query))
    walk_scores, trace_ids = _collect_trace_ids(
        query,
        query_vector,
        query_terms,
        ordered,
        routing_seeds,
        index,
        graph_hops=graph_hops,
        cold=cold,
        context_limit=context_limit,
    )
    grounded_scores = _score_grounded_items(
        query_vector, query_terms, trace_ids, fused, reranked, walk_scores, index,
    )
    if raw_sources_only:
        grounded_scores = [pair for pair in grounded_scores if index.items[pair[0]].kind == "raw_source"]
    selected = _diverse_grounded(
        grounded_scores,
        index,
        context_limit=context_limit,
        max_context_tokens=max_context_tokens,
        per_source_context_tokens=per_source_context_tokens,
        token_counter=token_counter,
    )
    return selected, routing_seeds, len(trace_ids)


def _collect_trace_ids(
    query: str,
    query_vector: list[float],
    query_terms: set[str],
    ordered: list[str],
    routing_seeds: list[str],
    index: _Index,
    *,
    graph_hops: int,
    cold: bool,
    context_limit: int,
) -> tuple[dict[str, float], set[str]]:
    trace_ids = {
        item_id
        for item_id in ordered[: max(_MIN_ORDERED_TRACE, context_limit)]
        if index.items[item_id].kind in _TRACE_KINDS
    }
    _expand_from_routing_seeds(routing_seeds, index, query_vector, query_terms, trace_ids)
    walked = _graph_walk(list(trace_ids), index.adjacency, hops=graph_hops)
    walk_scores = {item_id: score for item_id, score, _ in walked}
    trace_ids.update(
        item_id
        for item_id, _, _ in walked
        if item_id in index.items and index.items[item_id].kind in _TRACE_KINDS
    )
    if cold:
        _expand_cold_groups(trace_ids, index)
    if classify_query(query) in {"inference", "multi_hop"}:
        _expand_multihop_claims(query, trace_ids, index)
    return walk_scores, trace_ids


def _expand_from_routing_seeds(
    routing_seeds: list[str],
    index: _Index,
    query_vector: list[float],
    query_terms: set[str],
    trace_ids: set[str],
) -> None:
    for seed in routing_seeds:
        routing_item = index.items[seed]
        source_items = [
            index.items[item_id]
            for item_id in routing_item.source_record_ids
            if item_id in index.items and index.items[item_id].kind in _TRACE_KINDS
        ]
        if routing_item.kind in {"claim", "index_extension"}:
            trace_ids.update(item.id for item in source_items)
            continue
        # Episode Recall chooses a region; only its best matching Trace
        # records seed graph expansion. This avoids flooding grounding with
        # several complete episodes.
        source_items.sort(
            key=lambda item: (
                _SEED_COSINE_WEIGHT * cosine_similarity(query_vector, item.embedding)
                + _SEED_LEXICAL_WEIGHT * len(query_terms & set(tokenize(item.text))) / max(len(query_terms), 1)
            ),
            reverse=True,
        )
        trace_ids.update(item.id for item in source_items[:_SEED_SOURCE_CAP])


def _expand_cold_groups(trace_ids: set[str], index: _Index) -> None:
    groups = {index.items[item_id].group_id for item_id in trace_ids if index.items[item_id].group_id}
    trace_ids.update(
        item.id
        for item in index.items.values()
        if item.kind in _TRACE_KINDS and item.group_id in groups
    )


def _expand_multihop_claims(query: str, trace_ids: set[str], index: _Index) -> None:
    relevant_claims = [
        item for item in index.items.values()
        if item.kind == "claim" and set(tokenize(item.index_text)) & set(tokenize(query))
    ]
    for claim in relevant_claims:
        trace_ids.update(
            item_id for item_id in claim.source_record_ids
            if item_id in index.items and index.items[item_id].kind in _TRACE_KINDS
        )


def _score_grounded_items(
    query_vector: list[float],
    query_terms: set[str],
    trace_ids: set[str],
    fused: dict[str, float],
    reranked: list[tuple[str, float]],
    walk_scores: dict[str, float],
    index: _Index,
) -> list[tuple[str, float]]:
    max_fused = max(fused.values(), default=1.0)
    prior_scores = dict(reranked)
    grounded_scores: list[tuple[str, float]] = []
    for item_id in trace_ids:
        item = index.items.get(item_id)
        if item is None or item.kind not in _TRACE_KINDS:
            continue
        item_terms = set(tokenize(item.index_text))
        lexical = len(query_terms & item_terms) / max(len(query_terms), 1)
        score = (
            _GROUND_COSINE_WEIGHT * cosine_similarity(query_vector, item.embedding)
            + _GROUND_LEXICAL_WEIGHT * lexical
            + _GROUND_FUSED_WEIGHT * (fused.get(item_id, 0.0) / max_fused)
            + _GROUND_PRIOR_WEIGHT * prior_scores.get(item_id, 0.0)
            + _GROUND_CONFIDENCE_WEIGHT * item.confidence
            + _GROUND_WALK_WEIGHT * walk_scores.get(item_id, 0.0)
        )
        if item.kind == "raw_source":
            score += _GROUND_RAW_SOURCE_BONUS
        grounded_scores.append((item_id, score))
    grounded_scores.sort(key=lambda pair: pair[1], reverse=True)
    return grounded_scores


@dataclass
class _ContextBudget:
    """Running token and source budgets while packing grounded context."""

    selected: list[str] = field(default_factory=list)
    total_tokens: int = 0
    source_tokens: Counter[str] = field(default_factory=Counter)
    max_context_tokens: int = 6000
    per_source_context_tokens: int = 2000
    token_counter: TokenCounter = field(default_factory=ApproximateTokenCounter)

    def try_accept(self, item: _Item) -> bool:
        estimated_tokens = max(1, self.token_counter.count(item.text))
        source = _source_bucket(item)
        if self.selected and self.total_tokens + estimated_tokens > self.max_context_tokens:
            return False
        if self.selected and self.source_tokens[source] + estimated_tokens > self.per_source_context_tokens:
            return False
        self.selected.append(item.id)
        self.total_tokens += estimated_tokens
        self.source_tokens[source] += estimated_tokens
        return True


def _diverse_grounded(
    ranked: list[tuple[str, float]],
    index: _Index,
    *,
    context_limit: int,
    max_context_tokens: int = 6000,
    per_source_context_tokens: int = 2000,
    token_counter: TokenCounter | None = None,
) -> list[str]:
    """Pack ranked Trace items under episode and token budgets."""

    group_count = len({index.items[item_id].group_id for item_id, _ in ranked if index.items[item_id].group_id})
    per_episode = max(_MIN_PER_EPISODE, context_limit // 2) if group_count > 1 else context_limit
    budget = _ContextBudget(
        max_context_tokens=max_context_tokens,
        per_source_context_tokens=per_source_context_tokens,
        token_counter=token_counter or ApproximateTokenCounter(),
    )
    normalized_texts: set[str] = set()
    deferred = _collect_primary_grounded(
        ranked, index, budget, normalized_texts, per_episode, context_limit,
    )
    if len(budget.selected) >= context_limit:
        return budget.selected
    _fill_deferred_grounded(deferred, index, budget, normalized_texts, context_limit)
    return budget.selected


def _collect_primary_grounded(
    ranked: list[tuple[str, float]],
    index: _Index,
    budget: _ContextBudget,
    normalized_texts: set[str],
    per_episode: int,
    context_limit: int,
) -> list[str]:
    episode_usage: Counter[str] = Counter()
    deferred: list[str] = []
    for item_id, _ in ranked:
        item = index.items[item_id]
        normalized = " ".join(tokenize(item.text))
        if normalized in normalized_texts:
            continue
        episode = item.group_id or ""
        if episode and episode_usage[episode] >= per_episode:
            deferred.append(item_id)
            continue
        if not budget.try_accept(item):
            deferred.append(item_id)
            continue
        normalized_texts.add(normalized)
        episode_usage[episode] += 1
        if len(budget.selected) >= context_limit:
            break
    return deferred


def _fill_deferred_grounded(
    deferred: list[str],
    index: _Index,
    budget: _ContextBudget,
    normalized_texts: set[str],
    context_limit: int,
) -> None:
    for item_id in deferred:
        normalized = " ".join(tokenize(index.items[item_id].text))
        if normalized in normalized_texts:
            continue
        if not budget.try_accept(index.items[item_id]):
            continue
        normalized_texts.add(normalized)
        if len(budget.selected) >= context_limit:
            break


def _source_bucket(item: _Item) -> str:
    if isinstance(item.source_ref, dict):
        identifier = item.source_ref.get("id") or item.source_ref.get("source_key")
        if identifier:
            return f"source:{identifier}"
    if item.group_id:
        return f"group:{item.group_id}"
    return f"item:{item.id}"


def _query_lens_terms(query: str, index: _Index, *, limit: int = 8) -> list[str]:
    """Find entity/relation bridge terms for a difficult query.

    This is a query-scoped, disposable lens over extraction/claim structure. It
    never writes benchmark-specific information or mutates Trace.
    """

    query_terms = set(tokenize(query))
    candidates: Counter[str] = Counter()
    for item in index.items.values():
        _count_lens_terms(item, query_terms, candidates)
    return [token for token, _ in candidates.most_common(limit)]


def _count_lens_terms(item: _Item, query_terms: set[str], candidates: Counter[str]) -> None:
    if item.kind not in {"fact", "claim"}:
        return
    overlap = _field_term_overlap(item, query_terms)
    if not overlap:
        return
    _add_bridge_tokens(item, query_terms, candidates, len(overlap))


def _field_term_overlap(item: _Item, query_terms: set[str]) -> set[str]:
    fields = [item.subject, item.relation, item.object, item.text]
    blob_terms = set(tokenize(" ".join(value for value in fields if value)))
    return query_terms & blob_terms


def _add_bridge_tokens(
    item: _Item, query_terms: set[str], candidates: Counter[str], overlap_count: int,
) -> None:
    for value in (item.subject, item.relation, item.object):
        for token in tokenize(value or ""):
            if len(token) <= 2 or token in query_terms:
                continue
            candidates[token] += 1 + overlap_count


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


def _item_from_delta(delta: MemoryRecord) -> _Item | None:
    if delta.kind not in _TRACE_KINDS or not delta.text:
        return None
    raw_fields = delta.payload.get("retrieval_fields") or {}
    fields: dict[str, str] = {}
    if isinstance(raw_fields, dict):
        fields = {
            str(name): _field_text(value)
            for name, value in raw_fields.items()
            if _field_text(value)
        }
    modality = str(delta.payload.get("modality") or "text")
    source_type = str(delta.payload.get("source_type") or modality)
    document = RetrievalDocument(
        id=delta.id, text=delta.text, modality=modality,
        source_type=source_type, fields=fields,
    )
    return _Item(
        id=delta.id, kind=delta.kind, text=delta.text, index_text=document.index_text, confidence=delta.confidence,
        group_id=delta.group_id, observed_at=delta.observed_at, temporal_anchor=delta.temporal_anchor,
        source_record_ids=list(dict.fromkeys([delta.id, *delta.evidence_refs])),
        source_ref=delta.source_ref, locator=delta.locator,
        subject=delta.subject, relation=delta.relation, object=delta.object,
        modality=modality, source_type=source_type, fields=fields,
        seq=delta.seq,
    )


def _item_from_claim(claim: Claim) -> _Item:
    return _Item(
        id=claim.id, kind="claim", text=claim.text, index_text=claim.text, confidence=claim.confidence,
        temporal_anchor=claim.temporal_anchor,
        source_record_ids=list(dict.fromkeys(claim.raw_source_ids + claim.source_record_ids)),
        subject=claim.subject, relation=claim.relation, object=claim.object,
    )


def _item_from_group_view(view: GroupSummaryView) -> _Item:
    return _Item(
        id=view.id, kind="group_summary", text=view.summary,
        index_text=" ".join([view.summary, view.concept_or_topic, *view.alt_phrasings]),
        confidence=_STALE_GROUP_CONFIDENCE if view.stale else _FRESH_GROUP_CONFIDENCE,
        group_id=view.group_id, observed_at=view.observed_at,
        source_record_ids=view.source_log_ids, stale=view.stale,
    )


def _merge_dynamic_index_stats(
    items: dict[str, _Item],
    df: Counter[str],
    persisted_avgdl: float,
    persisted_n: int,
) -> tuple[Counter[str], float]:
    dynamic = [item for item in items.values() if item.kind in {"claim", "group_summary"}]
    for item in dynamic:
        df.update(set(tokenize(item.index_text)))
    total_tokens = persisted_avgdl * persisted_n + sum(len(tokenize(item.index_text)) for item in dynamic)
    avgdl = total_tokens / max(persisted_n + len(dynamic), 1)
    return df, avgdl


def _sources_with_similar_edges(
    adjacency: dict[str, list[tuple[str, float, str]]],
) -> set[str]:
    return {
        source
        for source, edges in adjacency.items()
        if any(kind == "similar_to" for _, _, kind in edges)
    }


def _similar_edges(
    trace_items: list[_Item],
    pending: list[_Item],
    threshold: float,
) -> list[tuple[str, str, str, float]]:
    normalized = normalize_rows([item.embedding for item in trace_items])
    by_id = {item.id: index for index, item in enumerate(trace_items)}
    edges: list[tuple[str, str, str, float]] = []
    for item in pending:
        edges.extend(_neighbors_for_item(item, normalized, by_id, trace_items, threshold))
    return edges


def _neighbors_for_item(
    item: _Item,
    normalized: Any,
    by_id: dict[str, int],
    trace_items: list[_Item],
    threshold: float,
) -> list[tuple[str, str, str, float]]:
    row_index = by_id[item.id]
    scores = normalized[row_index] @ normalized.T
    scores[row_index] = -np.inf
    neighbor_count = min(_SIMILAR_NEIGHBOR_CAP, len(trace_items) - 1)
    candidates = np.argpartition(-scores, neighbor_count - 1)[:neighbor_count]
    edges: list[tuple[str, str, str, float]] = []
    for candidate in sorted(candidates, key=lambda value: float(scores[value]), reverse=True):
        score = float(scores[candidate])
        if score < threshold:
            continue
        edges.append((item.id, trace_items[candidate].id, "similar_to", score))
    return edges
