"""Versioned in-process extension contracts for Formation, Recall, and retrieval.

Extensions are trusted Python components registered before an engine starts.
They receive bounded service objects from the engine rather than owning Trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Iterable, Literal, Mapping, Protocol, Sequence


EXTENSION_API_VERSION = 1


@dataclass(frozen=True)
class ExtensionSpec:
    id: str
    version: str
    engine_api: str = "1"
    storage_revision: int = 0
    description: str = ""


@dataclass(frozen=True)
class SourcePreparationContext:
    completion: Any | None = None


class SourceHandler(Protocol):
    name: str
    kinds: tuple[str, ...]

    def accepts(self, source: Any) -> bool: ...

    def prepare(self, source: Any, context: SourcePreparationContext) -> Any: ...


class SourceHandlerRegistry:
    def __init__(self, handlers: Iterable[SourceHandler] = ()) -> None:
        self._handlers: list[SourceHandler] = []
        self._frozen = False
        for handler in handlers:
            self.register(handler)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(handler.name for handler in self._handlers)

    def register(self, handler: SourceHandler, *, replace: bool = False) -> None:
        self._ensure_mutable()
        _validate_name(handler.name, "source handler")
        existing = next((i for i, value in enumerate(self._handlers) if value.name == handler.name), None)
        if existing is not None:
            if not replace:
                raise ValueError(f"source handler already registered: {handler.name}")
            self._handlers[existing] = handler
        else:
            self._handlers.append(handler)

    def prepare(self, source: Any, context: SourcePreparationContext) -> Any:
        matches = [handler for handler in self._handlers if handler.accepts(source)]
        if not matches:
            raise ValueError(f"no source handler accepts source kind: {getattr(source, 'kind', 'unknown')}")
        if len(matches) > 1:
            names = ", ".join(handler.name for handler in matches)
            raise ValueError(f"multiple source handlers accept the source: {names}")
        return matches[0].prepare(source, context)

    def freeze(self) -> None:
        self._frozen = True

    def _ensure_mutable(self) -> None:
        if self._frozen:
            raise RuntimeError("source handler registry is frozen")


@dataclass(frozen=True)
class FormationEvent:
    name: str
    episode_id: str | None
    namespace: Any
    evidence_version: int
    delta: Any | None = None


@dataclass(frozen=True)
class ProposedDelta:
    kind: Literal["extraction", "annotation"]
    text: str
    evidence_refs: tuple[str, ...]
    subject: str | None = None
    relation: str | None = None
    object: str | None = None
    temporal_anchor: str | None = None
    confidence: float = 0.7
    payload: Mapping[str, Any] = field(default_factory=dict)
    source_ref: Any = None
    locator: Any = None
    external_key: str | None = None


@dataclass(frozen=True)
class FormationContext:
    namespace: Any
    episode_id: str
    evidence_version: int
    completion: Any | None = None


@dataclass(frozen=True)
class FormationProcessorSpec:
    name: str
    events: tuple[str, ...] = ("episode_committed",)
    input_kinds: tuple[str, ...] = ("observation",)
    output_kinds: tuple[str, ...] = ("extraction",)
    max_attempts: int = 3


class FormationProcessor(Protocol):
    spec: FormationProcessorSpec

    def should_schedule(self, event: FormationEvent) -> bool: ...

    def process(
        self, episode: Sequence[Any], context: FormationContext
    ) -> Sequence[ProposedDelta]: ...


class FormationProcessorRegistry:
    def __init__(self) -> None:
        self._processors: dict[str, FormationProcessor] = {}
        self._frozen = False

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._processors)

    def register(self, processor: FormationProcessor, *, replace: bool = False) -> None:
        self._ensure_mutable()
        name = processor.spec.name
        _validate_name(name, "formation processor")
        if name in self._processors and not replace:
            raise ValueError(f"formation processor already registered: {name}")
        self._processors[name] = processor

    def get(self, name: str) -> FormationProcessor | None:
        return self._processors.get(name)

    def scheduled(self, event: FormationEvent) -> list[FormationProcessor]:
        scheduled: list[FormationProcessor] = []
        for processor in self._processors.values():
            if event.name not in processor.spec.events:
                continue
            try:
                if processor.should_schedule(event):
                    scheduled.append(processor)
            except Exception:
                # Optional processor scheduling must not invalidate committed
                # Trace. Runtime failures remain visible on durable jobs.
                continue
        return scheduled

    def freeze(self) -> None:
        self._frozen = True

    def _ensure_mutable(self) -> None:
        if self._frozen:
            raise RuntimeError("formation processor registry is frozen")


@dataclass(frozen=True)
class RecallChannelSpec:
    id: str
    title: str
    kind: str = "cards"
    producer_version: str = "1"
    persistent: bool = True
    async_projection: bool = True
    playground_seed: str | None = "excerpt"


class RecallChannel(Protocol):
    spec: RecallChannelSpec


@dataclass(frozen=True)
class RecallRecord:
    channel_id: str
    record_id: str
    namespace: Any
    evidence_version: int
    text: str
    evidence_refs: tuple[str, ...]
    fields: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    active: bool = True
    stale: bool = False
    producer_version: str = "1"


@dataclass(frozen=True)
class TraceChangeBatch:
    deltas: tuple[Any, ...]
    namespace: Any
    evidence_version: int
    event: str = "episode_committed"
    episode_id: str | None = None


class RecallWriter:
    """Namespace- and channel-scoped writer for core-owned Recall storage."""

    def __init__(self, store: Any, spec: RecallChannelSpec, namespace: Any) -> None:
        self._store = store
        self._spec = spec
        self._namespace = namespace

    def put(
        self,
        record_id: str,
        text: str,
        evidence_refs: Iterable[str],
        *,
        evidence_version: int,
        fields: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        stale: bool = False,
    ) -> RecallRecord:
        record = RecallRecord(
            channel_id=self._spec.id,
            record_id=record_id,
            namespace=self._namespace,
            evidence_version=evidence_version,
            text=text,
            evidence_refs=tuple(evidence_refs),
            fields=fields or {},
            metadata=metadata or {},
            stale=stale,
            producer_version=self._spec.producer_version,
        )
        self._store.put_recall_record(record)
        return record

    def records(
        self, *, search: str | None = None, cursor: int = 0, limit: int = 200
    ) -> tuple[list[RecallRecord], int]:
        return self._store.recall_records(
            self._spec.id, self._namespace, search=search, cursor=cursor, limit=limit
        )

    def replace(self, records: Iterable[RecallRecord]) -> None:
        """Atomically replace this channel's records in the scoped namespace."""
        values = list(records)
        for record in values:
            if record.channel_id != self._spec.id:
                raise ValueError("replacement record channel does not match writer channel")
            if record.namespace != self._namespace:
                raise ValueError("replacement record namespace does not match writer namespace")
        self._store.replace_recall_records(self._spec.id, self._namespace, values)


class TraceReader:
    """Read-only, namespace-scoped view of active Trace deltas."""

    def __init__(self, store: Any, namespace: Any, *, seq_cutoff: int | None = None) -> None:
        self._store = store
        self.namespace = namespace
        self.seq_cutoff = seq_cutoff

    def deltas(
        self,
        *,
        kinds: Iterable[str] | None = None,
        episode_prefix: str | None = None,
    ) -> tuple[Any, ...]:
        return tuple(self._store.list_deltas(
            kinds=kinds,
            namespace=self.namespace,
            episode_prefix=episode_prefix,
            seq_cutoff=self.seq_cutoff,
            include_retracted=False,
        ))


class RecallReader:
    """Read-only, namespace-scoped view of generic Recall records."""

    def __init__(self, store: Any, namespace: Any) -> None:
        self._store = store
        self.namespace = namespace

    def records(
        self,
        channel_id: str,
        *,
        search: str | None = None,
        cursor: int = 0,
        limit: int = 200,
        active_only: bool = True,
    ) -> tuple[list[RecallRecord], int]:
        return self._store.recall_records(
            channel_id,
            self.namespace,
            search=search,
            cursor=cursor,
            limit=limit,
            active_only=active_only,
        )

    def channel_counts(self) -> dict[str, dict[str, int]]:
        return self._store.recall_channel_counts(self.namespace)


@dataclass(frozen=True)
class InspectRequest:
    search: str | None = None
    cursor: str | None = None
    limit: int = 50


@dataclass(frozen=True)
class ExtensionState:
    extension_id: str
    installed_version: str
    engine_api: str
    storage_revision: int
    last_projected_seq: int = 0
    status: str = "available"
    last_error: str | None = None


class RecallChannelRegistry:
    def __init__(self) -> None:
        self._channels: dict[str, RecallChannel] = {}
        self._frozen = False

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._channels)

    @property
    def channels(self) -> tuple[RecallChannel, ...]:
        return tuple(self._channels.values())

    def register(self, channel: RecallChannel, *, replace: bool = False) -> None:
        self._ensure_mutable()
        channel_id = channel.spec.id
        _validate_name(channel_id, "recall channel")
        if channel_id in self._channels and not replace:
            raise ValueError(f"recall channel already registered: {channel_id}")
        if channel.spec.kind not in {"timeline", "table", "postings", "embedding", "cards", "graph"}:
            raise ValueError(f"unsupported recall helper kind: {channel.spec.kind}")
        self._channels[channel_id] = channel

    def get(self, channel_id: str) -> RecallChannel | None:
        return self._channels.get(channel_id)

    def freeze(self) -> None:
        self._frozen = True

    def _ensure_mutable(self) -> None:
        if self._frozen:
            raise RuntimeError("recall channel registry is frozen")


@dataclass(frozen=True)
class RetrievalCandidate:
    id: str
    branch: str
    score: float
    kind: Literal["trace", "recall"]
    evidence_delta_ids: tuple[str, ...]
    text: str | None = None
    channel_id: str | None = None
    source_ref: Any = None
    locator: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def canonical_id(self) -> str:
        if self.kind == "trace":
            trace_id = self.evidence_delta_ids[0] if self.evidence_delta_ids else self.id
            return f"trace:{trace_id}"
        channel = self.channel_id or self.branch
        return f"recall:{channel}:{self.id}"


@dataclass(frozen=True)
class BranchResult:
    branch: str
    candidates: tuple[RetrievalCandidate, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalBranchSpec:
    name: str
    title: str | None = None
    default_weight: float = 1.0
    profile_weights: Mapping[str, float] = field(default_factory=dict)
    supported_modalities: tuple[str, ...] = ()
    supported_query_kinds: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    cost_tier: Literal["local", "model", "remote"] = "local"
    max_candidates: int = 200
    timeout_ms: int = 5_000


@dataclass
class RetrievalContext:
    namespace: Any
    trace_cutoff: int
    items: Mapping[str, Any]
    recall: Any
    embedder: Any
    token_counter: Any
    prior_results: Mapping[str, BranchResult] = field(default_factory=dict)
    deadline: float | None = None

    def __post_init__(self) -> None:
        self.items = MappingProxyType(dict(self.items))
        self.prior_results = MappingProxyType(dict(self.prior_results))


class RetrievalBranch(Protocol):
    spec: RetrievalBranchSpec

    def retrieve(self, plan: Any, context: RetrievalContext) -> BranchResult | Sequence[RetrievalCandidate]: ...


class RetrievalBranchRegistry:
    def __init__(self) -> None:
        self._branches: dict[str, RetrievalBranch] = {}
        self._frozen = False

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._branches)

    @property
    def branches(self) -> tuple[RetrievalBranch, ...]:
        return tuple(self._branches.values())

    def register(self, branch: RetrievalBranch, *, replace: bool = False) -> None:
        self._ensure_mutable()
        spec = branch.spec
        _validate_name(spec.name, "retrieval branch")
        if not 0 <= spec.default_weight <= 10:
            raise ValueError("retrieval branch default weight must be between 0 and 10")
        if spec.max_candidates < 1 or spec.max_candidates > 10_000:
            raise ValueError("retrieval branch max_candidates must be between 1 and 10000")
        if spec.timeout_ms < 1 or spec.timeout_ms > 300_000:
            raise ValueError("retrieval branch timeout_ms must be between 1 and 300000")
        if spec.name in self._branches and not replace:
            raise ValueError(f"retrieval branch already registered: {spec.name}")
        self._branches[spec.name] = branch

    def get(self, name: str) -> RetrievalBranch | None:
        return self._branches.get(name)

    def enabled(self, names: Iterable[str]) -> list[RetrievalBranch]:
        allowed = set(names)
        return [branch for name, branch in self._branches.items() if not allowed or name in allowed]

    def levels(
        self, names: Iterable[str], *, already_resolved: Iterable[str] = ()
    ) -> list[list[RetrievalBranch]]:
        selected = {branch.spec.name: branch for branch in self.enabled(names)}
        levels: list[list[RetrievalBranch]] = []
        resolved: set[str] = set(already_resolved)
        completed: set[str] = set()
        while len(completed) < len(selected):
            level = [
                branch for name, branch in selected.items()
                if name not in completed and set(branch.spec.depends_on).issubset(resolved)
            ]
            if not level:
                pending = sorted(set(selected) - completed)
                raise ValueError(f"cyclic or missing retrieval branch dependencies: {pending}")
            levels.append(level)
            completed.update(branch.spec.name for branch in level)
            resolved.update(branch.spec.name for branch in level)
        return levels

    def validate(self, available_names: Iterable[str] = ()) -> None:
        available = set(available_names) | set(self._branches)
        for branch in self._branches.values():
            missing = set(branch.spec.depends_on) - available
            if missing:
                raise ValueError(
                    f"retrieval branch {branch.spec.name} has missing dependencies: {sorted(missing)}"
                )
        self.levels(self._branches, already_resolved=set(available_names))

    def freeze(self) -> None:
        self._frozen = True

    def _ensure_mutable(self) -> None:
        if self._frozen:
            raise RuntimeError("retrieval branch registry is frozen")


class FusionStrategy(Protocol):
    name: str

    def fuse(
        self,
        rankings: Mapping[str, Sequence[str]],
        weights: Mapping[str, float],
        limit: int | None = None,
    ) -> list[tuple[str, float]]: ...


class WeightedRRFFusion:
    name = "weighted_rrf"

    def __init__(self, rank_constant: int = 60) -> None:
        self.rank_constant = rank_constant

    def fuse(
        self,
        rankings: Mapping[str, Sequence[str]],
        weights: Mapping[str, float],
        limit: int | None = None,
    ) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for name, item_ids in rankings.items():
            weight = float(weights.get(name, 1.0))
            for rank, item_id in enumerate(item_ids, 1):
                scores[item_id] = scores.get(item_id, 0.0) + weight / (self.rank_constant + rank)
        ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return ordered[:limit] if limit is not None else ordered


@dataclass(frozen=True)
class RerankContext:
    items: Mapping[str, Any]
    query_vector: Sequence[float]
    semantic: Callable[[Sequence[float], Sequence[float]], float]


class CandidateReranker(Protocol):
    name: str

    def rerank(
        self,
        plan: Any,
        fused: Sequence[tuple[str, float]],
        context: RerankContext,
    ) -> list[tuple[str, float]]: ...


class DefaultCandidateReranker:
    name = "semantic_reliability_recency_v1"

    def rerank(
        self,
        plan: Any,
        fused: Sequence[tuple[str, float]],
        context: RerankContext,
    ) -> list[tuple[str, float]]:
        max_fused = max((score for _, score in fused), default=1.0)
        values: list[tuple[str, float]] = []
        for item_id, fused_score in fused:
            item = context.items[item_id]
            semantic_score = context.semantic(context.query_vector, item.embedding)
            reliability = item.confidence * (0.55 if item.stale else 1)
            recency = 1 if item.observed_at or item.temporal_anchor else 0.5
            score = (
                0.52 * semantic_score
                + 0.18 * reliability
                + 0.10 * recency
                + 0.20 * fused_score / max_fused
            )
            if item.modality in plan.modalities:
                score += 0.04
            values.append((item_id, score))
        return sorted(values, key=lambda pair: (-pair[1], pair[0]))


class JobHandler(Protocol):
    kind: str

    def run(self, engine: Any, job: Any) -> Any: ...


class JobHandlerRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, JobHandler | Callable[[Any, Any], Any]] = {}
        self._frozen = False

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._handlers)

    def register(
        self,
        handler: JobHandler | Callable[[Any, Any], Any],
        *,
        kind: str | None = None,
        replace: bool = False,
    ) -> None:
        self._ensure_mutable()
        job_kind = kind or getattr(handler, "kind", "")
        _validate_name(job_kind, "job handler")
        if job_kind in self._handlers and not replace:
            raise ValueError(f"job handler already registered: {job_kind}")
        self._handlers[job_kind] = handler

    def get(self, kind: str) -> JobHandler | Callable[[Any, Any], Any] | None:
        return self._handlers.get(kind)

    def run(self, engine: Any, job: Any) -> Any:
        handler = self.get(job.kind)
        if handler is None:
            raise RuntimeError(f"no job handler is registered for {job.kind}")
        run = getattr(handler, "run", handler)
        return run(engine, job)

    def freeze(self) -> None:
        self._frozen = True

    def _ensure_mutable(self) -> None:
        if self._frozen:
            raise RuntimeError("job handler registry is frozen")


class MemoryExtension(Protocol):
    spec: ExtensionSpec

    def register(self, registry: "EngineExtensionRegistry") -> None: ...


class EngineExtensionRegistry:
    """All extension capabilities, validated and frozen as one graph."""

    def __init__(self) -> None:
        self.source_handlers = SourceHandlerRegistry()
        self.formation_processors = FormationProcessorRegistry()
        self.recall_channels = RecallChannelRegistry()
        self.retrieval_branches = RetrievalBranchRegistry()
        self.job_handlers = JobHandlerRegistry()
        self.extensions: dict[str, ExtensionSpec] = {}
        self.component_owners: dict[tuple[str, str], str] = {}
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    def register_extension(self, extension: MemoryExtension) -> None:
        if self._frozen:
            raise RuntimeError("extension registry is frozen")
        spec = extension.spec
        _validate_name(spec.id, "extension")
        if spec.id in self.extensions:
            raise ValueError(f"extension already registered: {spec.id}")
        _validate_engine_api(spec.engine_api)
        snapshots = (
            list(self.source_handlers._handlers),
            dict(self.formation_processors._processors),
            dict(self.recall_channels._channels),
            dict(self.retrieval_branches._branches),
            dict(self.job_handlers._handlers),
        )
        self.extensions[spec.id] = spec
        before = {
            "source": set(self.source_handlers.names),
            "formation": set(self.formation_processors.names),
            "recall": set(self.recall_channels.names),
            "retrieval": set(self.retrieval_branches.names),
            "job": set(self.job_handlers.names),
        }
        try:
            extension.register(self)
        except Exception:
            self.extensions.pop(spec.id, None)
            self.source_handlers._handlers = snapshots[0]
            self.formation_processors._processors = snapshots[1]
            self.recall_channels._channels = snapshots[2]
            self.retrieval_branches._branches = snapshots[3]
            self.job_handlers._handlers = snapshots[4]
            raise
        after = {
            "source": set(self.source_handlers.names),
            "formation": set(self.formation_processors.names),
            "recall": set(self.recall_channels.names),
            "retrieval": set(self.retrieval_branches.names),
            "job": set(self.job_handlers.names),
        }
        for capability, names in after.items():
            for name in names - before[capability]:
                self.component_owners[(capability, name)] = spec.id

    def owner(self, capability: str, name: str) -> str | None:
        return self.component_owners.get((capability, name))

    def validate_and_freeze(self, legacy_route_names: Iterable[str] = ()) -> None:
        self.retrieval_branches.validate(legacy_route_names)
        self._validate_formation_graph()
        for channel in self.recall_channels.channels:
            if channel.spec.persistent and not callable(getattr(channel, "rebuild", None)):
                raise ValueError(
                    f"persistent recall channel {channel.spec.id} must implement rebuild"
                )
        self.source_handlers.freeze()
        self.formation_processors.freeze()
        self.recall_channels.freeze()
        self.retrieval_branches.freeze()
        self.job_handlers.freeze()
        self._frozen = True

    def _validate_formation_graph(self) -> None:
        processors = list(self.formation_processors._processors.values())
        for processor in processors:
            overlap = set(processor.spec.input_kinds) & set(processor.spec.output_kinds)
            if overlap:
                raise ValueError(
                    f"formation processor {processor.spec.name} consumes its own output kinds: {sorted(overlap)}"
                )


def _validate_name(value: str, label: str) -> None:
    if not value or not value.strip() or any(char.isspace() for char in value):
        raise ValueError(f"{label} name must be non-empty and contain no whitespace")


def _validate_engine_api(value: str) -> None:
    accepted = {"1", ">=1,<2", ">=1.0,<2.0"}
    if value not in accepted:
        raise ValueError(
            f"extension engine_api {value!r} is incompatible with API {EXTENSION_API_VERSION}"
        )
