"""Data contracts for the append-only Trace & Recall engine."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EngineModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderRole(str, Enum):
    COMPLETION = "completion"
    EMBEDDING = "embedding"


class ProviderSelection(EngineModel):
    """A non-secret provider choice persisted in one memory store."""

    provider: str
    model: str | None = None
    base_url: str | None = None

    @model_validator(mode="after")
    def _valid_selection(self) -> "ProviderSelection":
        self.provider = self.provider.strip().lower()
        self.model = self.model.strip() if self.model else None
        self.base_url = self.base_url.strip().rstrip("/") if self.base_url else None
        if not self.provider:
            raise ValueError("provider must not be empty")
        if self.provider != "none" and not self.model:
            raise ValueError("model is required when a provider is selected")
        if self.provider == "none" and (self.model or self.base_url):
            raise ValueError("the none provider cannot have a model or base URL")
        return self


class ModelConfiguration(EngineModel):
    completion: ProviderSelection = Field(
        default_factory=lambda: ProviderSelection(provider="none")
    )
    embedding: ProviderSelection = Field(
        default_factory=lambda: ProviderSelection(
            provider="sentence-transformers", model="all-MiniLM-L6-v2"
        )
    )
    revision: int = Field(default=0, ge=0)
    updated_at: datetime = Field(default_factory=utc_now)


class ProviderDescriptor(EngineModel):
    id: str
    display_name: str
    roles: list[ProviderRole]
    credential_env: str | None = None
    credential_configured: bool = False
    default_base_url: str | None = None
    native_protocol: bool = False
    notes: str | None = None


class ModelDescriptor(EngineModel):
    provider: str
    id: str
    display_name: str
    roles: list[ProviderRole]
    vision: bool | None = None
    structured_output: bool | None = None
    context_length: int | None = None
    source: Literal["live", "curated", "custom"] = "live"
    capability_status: Literal["verified", "unknown"] = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelConfigurationChange(EngineModel):
    status: Literal["applied", "rebuild_pending", "rebuild_failed"]
    configuration: ModelConfiguration
    pending_configuration: ModelConfiguration | None = None
    job_id: str | None = None
    rebuild_required: bool = False
    message: str | None = None


class ConnectionTestResult(EngineModel):
    ok: bool
    role: ProviderRole
    provider: str
    model: str | None = None
    message: str
    billed_request: bool = True
    vision_supported: bool | None = None


class Actor(EngineModel):
    type: Literal["user", "formation_pipeline", "compilation_job", "external_api"] = "external_api"
    id: str = "local"
    model: str | None = None
    prompt_version: str | None = None


class MemoryNamespace(EngineModel):
    """Stable isolation boundary shared by every public interface.

    ``project_id`` defaults to ``default`` so existing local-only callers stay
    isolated in one explicit namespace instead of writing unscoped records.
    """

    user_id: str | None = None
    agent_id: str | None = None
    project_id: str = "default"
    session_id: str | None = None

    @model_validator(mode="after")
    def _non_empty_values(self) -> "MemoryNamespace":
        for field_name in ("user_id", "agent_id", "project_id", "session_id"):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        return self

    def as_scope(self) -> dict[str, str]:
        return {
            key: value
            for key, value in self.model_dump().items()
            if value is not None
        }


class MemoryDelta(EngineModel):
    """One immutable, hash-chained piece of evidence."""

    id: str
    seq: int
    prev_hash: str
    hash: str
    written_at: datetime = Field(default_factory=utc_now)
    observed_at: datetime | None = None
    kind: Literal["observation", "extraction", "annotation", "access", "retraction"]
    actor: Actor = Field(default_factory=Actor)
    namespace: MemoryNamespace = Field(default_factory=MemoryNamespace)
    episode_id: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0, le=1)
    privacy_scope: dict[str, Any] = Field(default_factory=dict)
    scope: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    text: str = ""
    subject: str | None = None
    relation: str | None = None
    object: str | None = None
    temporal_anchor: str | None = None
    source_ref: dict[str, Any] | str | None = None
    locator: dict[str, Any] | str | None = None
    external_key: str | None = None


class CompiledClaim(EngineModel):
    id: str
    claim_key: str
    text: str
    status: Literal["ACTIVE", "SUPERSEDED", "CONTESTED"] = "ACTIVE"
    source_delta_ids: list[str] = Field(default_factory=list)
    observation_delta_ids: list[str] = Field(default_factory=list)
    temporal_anchor: str | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)
    subject: str | None = None
    relation: str | None = None
    object: str | None = None


class EpisodeRecallView(EngineModel):
    id: str
    episode_id: str
    concept_or_topic: str
    summary: str
    alt_phrasings: list[str] = Field(default_factory=list)
    source_trace_ids: list[str] = Field(default_factory=list)
    observed_at: datetime | None = None
    stale: bool = False
    build_version: int = 1
    cache_key: str
    generated_at: datetime = Field(default_factory=utc_now)
    namespace: MemoryNamespace = Field(default_factory=MemoryNamespace)
    generation_provenance: dict[str, Any] = Field(default_factory=dict)


class SearchHit(EngineModel):
    item_id: str
    kind: Literal["observation", "extraction", "compiled", "episode_recall"]
    text: str
    score: float
    route: str = "fused"
    episode_id: str | None = None
    observed_at: datetime | None = None
    temporal_anchor: str | None = None
    source_delta_ids: list[str] = Field(default_factory=list)
    source_ref: dict[str, Any] | str | None = None
    locator: dict[str, Any] | str | None = None
    confidence: float = 0.7
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalTrace(EngineModel):
    query_id: str
    query: str
    namespace: MemoryNamespace = Field(default_factory=MemoryNamespace)
    query_kind: Literal["fact", "temporal", "list", "inference"]
    stage: Literal["fast", "refine_1", "refine_2", "deep_recall", "cold"]
    confident: bool
    escalated: bool = False
    refined_query: str | None = None
    routes: dict[str, list[str]] = Field(default_factory=dict)
    routing_seeds: list[str] = Field(default_factory=list)
    drilled_trace_count: int = 0
    episode_recall_in_answer_context: int = 0
    top_score: float = 0
    margin: float | None = None
    created_at: datetime = Field(default_factory=utc_now)


class QueryCandidateSnapshot(EngineModel):
    """Bounded, safe evidence retained to explain one retrieval route."""

    item_id: str
    kind: str
    route: str
    rank: int = Field(ge=1)
    score: float
    excerpt: str
    source_delta_ids: list[str] = Field(default_factory=list)
    source_ref: dict[str, Any] | str | None = None
    locator: dict[str, Any] | str | None = None


class QueryStep(EngineModel):
    id: str
    phase: str
    label: str
    sequence: int = Field(ge=0)
    status: Literal["pending", "running", "succeeded", "failed", "skipped"] = "succeeded"
    parent_ids: list[str] = Field(default_factory=list)
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    candidates: list[QueryCandidateSnapshot] = Field(default_factory=list, max_length=80)
    duration_ms: float | None = Field(default=None, ge=0)
    created_at: datetime = Field(default_factory=utc_now)


class RetrievalConfiguration(EngineModel):
    default_top_k: int = Field(default=12, ge=1, le=100)
    max_context_items: int = Field(default=24, ge=1, le=100)
    max_refinement_rounds: int = Field(default=2, ge=0, le=2)
    graph_hops: int = Field(default=2, ge=0, le=4)
    confidence_margin: float = Field(default=0.018, ge=0, le=1)
    deep_recall_enabled: bool = True
    answer_abstain_threshold: float = Field(default=0.10, ge=0, le=1)
    revision: int = Field(default=0, ge=0)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _context_covers_results(self) -> "RetrievalConfiguration":
        if self.max_context_items < self.default_top_k:
            raise ValueError("max_context_items must be at least default_top_k")
        return self


class QueryRun(EngineModel):
    id: str
    mode: Literal["query", "search"] = "query"
    status: Literal["pending", "running", "completed", "failed", "interrupted"] = "pending"
    namespace: MemoryNamespace = Field(default_factory=MemoryNamespace)
    query: str
    answer: str | None = None
    abstain: bool | None = None
    citations: list["Citation"] = Field(default_factory=list)
    steps: list[QueryStep] = Field(default_factory=list)
    retrieval_trace: RetrievalTrace | None = None
    retrieval_configuration: RetrievalConfiguration = Field(default_factory=RetrievalConfiguration)
    generation_provenance: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    attempt: int = Field(default=0, ge=0)
    partial: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0)


class QueryRunPage(EngineModel):
    runs: list[QueryRun] = Field(default_factory=list)
    next_cursor: str | None = None


class QueryRunRemoveRequest(EngineModel):
    query_ids: list[str] = Field(default_factory=list, max_length=1000)
    before: datetime | None = None
    all_in_namespace: bool = False
    confirm: bool = False


class MemorySearchResult(EngineModel):
    query_id: str
    hits: list[SearchHit] = Field(default_factory=list)
    stage: str
    confident: bool
    retrieval_trace: RetrievalTrace


class Citation(EngineModel):
    delta_id: str
    source_ref: dict[str, Any] | str | None = None
    locator: dict[str, Any] | str | None = None
    excerpt: str
    observed_at: datetime | None = None


class MemoryQueryResult(EngineModel):
    query_id: str
    question: str
    answer: str
    abstain: bool
    citations: list[Citation] = Field(default_factory=list)
    retrieval_trace: RetrievalTrace


class EpisodeInfo(EngineModel):
    episode_id: str
    delta_count: int
    observation_count: int
    extraction_count: int
    first_observed_at: datetime | None = None
    last_observed_at: datetime | None = None
    stale: bool = True


class TraceVerification(EngineModel):
    valid: bool
    delta_count: int
    broken_at_seq: int | None = None
    reason: str | None = None


class RecallSnapshot(EngineModel):
    id: str
    label: str | None = None
    seq_cutoff: int
    evidence_hash: str
    created_at: datetime = Field(default_factory=utc_now)
    active: bool = False


class SnapshotDiff(EngineModel):
    from_snapshot: str
    to_snapshot: str
    added_delta_ids: list[str] = Field(default_factory=list)
    removed_delta_ids: list[str] = Field(default_factory=list)
    from_seq: int
    to_seq: int


class MemoryPage(EngineModel):
    items: list[MemoryDelta] = Field(default_factory=list)
    next_cursor: int | None = None


class MemoryHistory(EngineModel):
    memory_id: str
    events: list[MemoryDelta] = Field(default_factory=list)


class RemoveResult(EngineModel):
    remove_id: str
    removed_delta_ids: list[str] = Field(default_factory=list)
    old_root_hash: str
    new_root_hash: str
    requested_by: str
    created_at: datetime = Field(default_factory=utc_now)


class RemoveRequest(EngineModel):
    delta_ids: list[str] = Field(min_length=1, max_length=1000)
    reason: str
    requested_by: str = "user"
    confirm: bool = False
    namespace: MemoryNamespace | None = None


SourceKind = Literal["text", "file", "directory", "archive", "git", "url", "image"]


class SourceInput(EngineModel):
    """One source accepted by the unified ingestion pipeline."""

    kind: SourceKind = "file"
    source_key: str | None = None
    path: str | None = None
    url: str | None = None
    text: str | None = None
    content_base64: str | None = None
    filename: str | None = None
    title: str | None = None
    ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    scope: dict[str, Any] = Field(default_factory=dict)
    namespace: MemoryNamespace | None = None

    @model_validator(mode="after")
    def _has_payload(self) -> "SourceInput":
        choices = [self.path, self.url, self.text, self.content_base64]
        if sum(value is not None for value in choices) != 1:
            raise ValueError("source requires exactly one of path, url, text, or content_base64")
        if self.kind == "git" and self.url is None:
            raise ValueError("git sources require url")
        if self.kind == "url" and self.url is None:
            raise ValueError("url sources require url")
        if self.kind in {"directory", "archive"} and self.path is None and self.content_base64 is None:
            raise ValueError(f"{self.kind} sources require path or content_base64")
        return self


class SourceRecord(EngineModel):
    id: str
    source_key: str
    kind: SourceKind
    title: str
    uri: str | None = None
    content_hash: str
    blob_path: str
    media_type: str
    filename: str | None = None
    byte_size: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=0, ge=0)
    ingestion_run_id: str | None = None
    preview_type: str | None = None
    skipped_count: int = Field(default=0, ge=0)
    version: int = 1
    status: Literal["active", "superseded", "removed"] = "active"
    namespace: MemoryNamespace = Field(default_factory=MemoryNamespace)
    metadata: dict[str, Any] = Field(default_factory=dict)
    delta_ids: list[str] = Field(default_factory=list)
    previous_source_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    removed_at: datetime | None = None


class SourceIngestionResult(EngineModel):
    index: int
    source_id: str | None = None
    source_key: str | None = None
    kind: SourceKind
    status: Literal["success", "skipped", "failed"]
    episode_id: str | None = None
    delta_ids: list[str] = Field(default_factory=list)
    skipped_paths: list[str] = Field(default_factory=list)
    error: str | None = None


class IngestionRun(EngineModel):
    id: str
    status: Literal["pending", "running", "completed", "partial", "failed"] = "pending"
    namespace: MemoryNamespace = Field(default_factory=MemoryNamespace)
    inputs: list[SourceInput] = Field(default_factory=list)
    results: list[SourceIngestionResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SourcePreviewItem(EngineModel):
    delta_id: str
    kind: str
    text: str
    locator: dict[str, Any] | str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourcePreview(EngineModel):
    source_id: str
    preview_type: str
    media_type: str
    items: list[SourcePreviewItem] = Field(default_factory=list)
    next_cursor: int | None = None
    manifest: list[str] = Field(default_factory=list)


class MemoryGraphNode(EngineModel):
    id: str
    type: Literal["source", "trace", "episode", "recall", "claim", "concept"]
    label: str
    subtitle: str | None = None
    status: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class MemoryGraphEdge(EngineModel):
    id: str
    source: str
    target: str
    type: str
    label: str | None = None
    weight: float = 1.0
    data: dict[str, Any] = Field(default_factory=dict)


class MemoryGraphPage(EngineModel):
    view: Literal["knowledge", "lineage", "trace"]
    nodes: list[MemoryGraphNode] = Field(default_factory=list)
    edges: list[MemoryGraphEdge] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    truncated: bool = False
    next_cursor: str | None = None


class MemoryJob(EngineModel):
    id: str
    kind: Literal["extract_episode", "compile_episode", "rebuild_embeddings", "execute_query"]
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    payload: dict[str, Any] = Field(default_factory=dict)
    attempts: int = 0
    max_attempts: int = 3
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
