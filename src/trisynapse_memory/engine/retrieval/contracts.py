"""Generic query planning and pluggable retrieval-route contracts.

This module deliberately knows nothing about benchmarks.  A route consumes a
canonical, multi-field retrieval document and a query plan; the production
retriever decides how route results are fused and grounded back into Trace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from trisynapse_memory.engine.retrieval.tokenization import lexical_tokens


@dataclass(frozen=True)
class RetrievalDocument:
    """Canonical searchable representation of one Trace-backed item."""

    id: str
    text: str
    modality: str = "text"
    source_type: str = "text"
    fields: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def index_text(self) -> str:
        values = [self.text]
        for name, value in self.fields.items():
            value = str(value).strip()
            if value and value != self.text:
                values.append(f"{name}: {value}")
        return "\n".join(values)


@dataclass(frozen=True)
class QueryPlan:
    """Planner output consumed by every retrieval route."""

    query: str
    query_kind: str
    terms: tuple[str, ...]
    modalities: tuple[str, ...] = ()
    routes: tuple[str, ...] = ()
    profile: str = "balanced"
    metadata: dict[str, Any] = field(default_factory=dict)


class QueryPlanner(Protocol):
    def plan(
        self,
        query: str,
        *,
        available_modalities: set[str] | None = None,
        enabled_routes: tuple[str, ...] = (),
        requested_profile: str = "auto",
    ) -> QueryPlan: ...


class HeuristicQueryPlanner:
    """Replaceable default planner with broad source and intent cues."""

    _inference = re.compile(r"\b(would|likely|probably|might|could|should|do you think|want to|still)\b", re.I)
    _temporal = re.compile(r"\b(when|what year|what date|how long ago|how many times|before|after)\b", re.I)
    _listing = re.compile(r"\b(list|enumerate|which|how many|what .+ (books|activities|items|instruments|pets|types|changes|plans))\b", re.I)
    _multi_hop = re.compile(
        r"\b(because|lead to|result(?:ed)? in|relationship between|connect|both|in common|"
        r"how did .+ (?:affect|influence|change)|what happened(?: .+)? after|why did)\b",
        re.I,
    )
    _modality_cues = {
        "code": re.compile(r"\b(code|function|method|class|symbol|repository|repo|module|import|line|bug|api)\b", re.I),
        "table": re.compile(r"\b(table|row|column|cell|sheet|spreadsheet|csv|xlsx|aggregate|total)\b", re.I),
        "image": re.compile(r"\b(image|photo|picture|screenshot|diagram|chart|figure|visible)\b", re.I),
        "document": re.compile(r"\b(document|page|section|paragraph|pdf|slide|chapter|article)\b", re.I),
        "conversation": re.compile(r"\b(said|told|asked|conversation|message|chat|speaker|discussed)\b", re.I),
    }

    def classify(self, query: str) -> str:
        if self._multi_hop.search(query):
            return "multi_hop"
        if self._inference.search(query):
            return "inference"
        if self._temporal.search(query):
            return "temporal"
        if self._listing.search(query) or re.search(r"\bwhat .+ and\b", query, re.I):
            return "list"
        return "fact"

    def plan(
        self,
        query: str,
        *,
        available_modalities: set[str] | None = None,
        enabled_routes: tuple[str, ...] = (),
        requested_profile: str = "auto",
    ) -> QueryPlan:
        available = available_modalities or set()
        modalities = tuple(
            name for name, pattern in self._modality_cues.items()
            if pattern.search(query) and (not available or name in available)
        )
        if requested_profile == "auto":
            profile = modalities[0] if len(modalities) == 1 else "mixed" if modalities else "balanced"
        else:
            profile = requested_profile
        return QueryPlan(
            query=query,
            query_kind=self.classify(query),
            terms=tuple(lexical_tokens(query)),
            modalities=modalities,
            routes=enabled_routes,
            profile=profile,
        )


@dataclass
class RouteContext:
    items: dict[str, Any]
    query_vector: list[float]
    bm25: Callable[[str, str], float]
    semantic: Callable[[list[float], list[float]], float]
    graph_walk: Callable[[list[str]], list[tuple[str, float, str]]]
    seed_ids: list[str] = field(default_factory=list)
    lexical_candidates: dict[str, float] | None = None
    semantic_candidates: dict[str, float] | None = None


class RetrievalRoute(Protocol):
    name: str

    def rank(self, plan: QueryPlan, context: RouteContext) -> list[tuple[str, float]]: ...


class BM25Route:
    name = "bm25"

    def rank(self, plan: QueryPlan, context: RouteContext) -> list[tuple[str, float]]:
        if context.semantic_candidates is not None:
            return _sorted_scores(context.semantic_candidates.items())
        return _sorted_scores(
            (item.id, context.bm25(plan.query, item.index_text))
            for item in context.items.values()
        )


class SemanticRoute:
    name = "semantic"

    def rank(self, plan: QueryPlan, context: RouteContext) -> list[tuple[str, float]]:
        if context.lexical_candidates is not None:
            scores = dict(context.lexical_candidates)
            for item in context.items.values():
                if item.kind in {"compiled", "episode_recall"}:
                    scores[item.id] = context.bm25(plan.query, item.index_text)
            return _sorted_scores(scores.items())
        return _sorted_scores(
            (item.id, context.semantic(context.query_vector, item.embedding))
            for item in context.items.values()
        )


class TemporalRoute:
    name = "temporal"

    def rank(self, plan: QueryPlan, context: RouteContext) -> list[tuple[str, float]]:
        cues = re.findall(
            r"\b(?:19|20)\d{2}\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december|before|after|last|next)\b",
            plan.query.casefold(),
        )
        asks_when = plan.query_kind == "temporal"
        if not cues and not asks_when:
            return []
        event_terms = set(plan.terms) - {"when", "what", "year", "date", "did", "does", "was", "were", "before", "after", "last", "next"}
        scores = []
        for item in context.items.values():
            blob = f"{item.temporal_anchor or ''} {item.observed_at or ''} {item.index_text}".casefold()
            cue_score = sum(cue in blob for cue in cues) / max(len(cues), 1)
            lexical = len(event_terms & set(lexical_tokens(item.index_text))) / max(len(event_terms), 1)
            has_time = bool(item.temporal_anchor or item.observed_at)
            score = 0.55 * cue_score + 0.35 * lexical + (0.10 if has_time else 0.0)
            if asks_when and not has_time:
                score *= 0.45
            scores.append((item.id, score))
        return _sorted_scores(scores)


class GraphRoute:
    name = "graph"

    def rank(self, plan: QueryPlan, context: RouteContext) -> list[tuple[str, float]]:
        return [(item_id, score) for item_id, score, _ in context.graph_walk(context.seed_ids)]


class ModalityRoute:
    """Field-aware route specialized for one source modality."""

    modality = "text"
    name = "text"
    field_weights: dict[str, float] = {}

    def rank(self, plan: QueryPlan, context: RouteContext) -> list[tuple[str, float]]:
        preferred = self.modality in plan.modalities
        scores: list[tuple[str, float]] = []
        for item in context.items.values():
            if item.modality != self.modality:
                continue
            lexical = context.bm25(plan.query, item.index_text)
            semantic = context.semantic(context.query_vector, item.embedding)
            field_overlap = 0.0
            wanted = set(plan.terms)
            for field_name, weight in self.field_weights.items():
                field_value = item.fields.get(field_name, "")
                if field_value:
                    field_overlap += weight * len(wanted & set(lexical_tokens(field_value))) / max(len(wanted), 1)
            intent_boost = 0.12 if preferred else 0.0
            scores.append((item.id, 0.56 * semantic + 0.30 * _bounded(lexical) + 0.14 * field_overlap + intent_boost))
        return _sorted_scores(scores)


class CodeRoute(ModalityRoute):
    modality = "code"
    name = "code"
    field_weights = {"symbol": 1.0, "path": 0.8, "language": 0.5, "imports": 0.6}


class TableRoute(ModalityRoute):
    modality = "table"
    name = "table"
    field_weights = {"headers": 1.0, "sheet": 0.8, "row": 0.5, "path": 0.4}


class ImageRoute(ModalityRoute):
    modality = "image"
    name = "image"
    field_weights = {"visible_text": 1.0, "description": 0.8, "filename": 0.4}


class DocumentRoute(ModalityRoute):
    modality = "document"
    name = "document"
    field_weights = {"title": 0.8, "section": 1.0, "page": 0.7, "path": 0.5}


class ConversationRoute(ModalityRoute):
    modality = "conversation"
    name = "conversation"
    field_weights = {"speaker": 1.0, "message": 0.8}


class RouteRegistry:
    """Ordered route collection; callers may replace or extend it."""

    def __init__(self, routes: list[RetrievalRoute] | None = None) -> None:
        self._routes = list(routes) if routes is not None else [
            BM25Route(), SemanticRoute(), TemporalRoute(), GraphRoute(),
            CodeRoute(), TableRoute(), ImageRoute(), DocumentRoute(), ConversationRoute(),
        ]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(route.name for route in self._routes)

    def register(self, route: RetrievalRoute, *, replace: bool = False) -> None:
        existing = next((index for index, value in enumerate(self._routes) if value.name == route.name), None)
        if existing is not None:
            if not replace:
                raise ValueError(f"retrieval route already registered: {route.name}")
            self._routes[existing] = route
        else:
            self._routes.append(route)

    def enabled(self, names: tuple[str, ...]) -> list[RetrievalRoute]:
        allowed = set(names)
        return [route for route in self._routes if not allowed or route.name in allowed]


DEFAULT_ROUTE_WEIGHTS = {
    "bm25": 1.0,
    "semantic": 1.0,
    "temporal": 0.85,
    "graph": 0.85,
    "code": 1.0,
    "table": 1.0,
    "image": 0.9,
    "document": 0.95,
    "conversation": 1.0,
}


PROFILE_ROUTE_WEIGHTS = {
    "balanced": {},
    "mixed": {},
    "precise": {"bm25": 1.25, "semantic": 0.9, "graph": 0.65},
    "broad": {"semantic": 1.2, "graph": 1.15, "bm25": 0.85},
    "code": {"code": 1.8, "graph": 1.15, "document": 0.65},
    "table": {"table": 1.8, "semantic": 0.9},
    "image": {"image": 1.8, "semantic": 1.05},
    "document": {"document": 1.55, "bm25": 1.1},
    "conversation": {"conversation": 1.5, "temporal": 1.1},
}


def route_weights(profile: str, overrides: dict[str, float] | None = None) -> dict[str, float]:
    weights = dict(DEFAULT_ROUTE_WEIGHTS)
    weights.update(PROFILE_ROUTE_WEIGHTS.get(profile, {}))
    weights.update(overrides or {})
    return weights


def _sorted_scores(values: Any) -> list[tuple[str, float]]:
    return sorted(
        ((item_id, float(score)) for item_id, score in values if score > 0),
        key=lambda pair: pair[1],
        reverse=True,
    )


def _bounded(value: float) -> float:
    return value / (1.0 + max(value, 0.0))
