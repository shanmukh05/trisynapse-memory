"""Public ingest, memory-log, and index memory engine.

MemoryEngine is the orchestration facade. Graph assembly, provider
bootstrap, ingestion commits, and job-handler details live in private
helpers in this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import time
import zipfile
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from trisynapse_memory.engine.indexes.claims import (
    build_group_summary_views,
    compile_claims,
)
from trisynapse_memory.engine.indexes import inspect as index_inspect
from trisynapse_memory.engine.providers.embedding import Embedder, SentenceTransformerEmbedder, UnavailableEmbedder
from trisynapse_memory.engine.ingest.pipeline import extract_facts_many, ingest_document, ingest_raw_source
from trisynapse_memory.engine.providers.registry import DEFAULT_COMPLETION_PARALLEL_WORKERS
from trisynapse_memory.engine.models import (
    Actor,
    Citation,
    Claim,
    RecordGroupInfo,
    GroupSummaryView,
    MemoryCatalog,
    MemoryRecord,
    MemoryDocumentPage,
    MemoryHelperPage,
    MemoryHistory,
    MemoryGraphEdge,
    MemoryGraphNode,
    MemoryGraphPage,
    MemoryJob,
    MemoryNamespace,
    ModelConfiguration,
    ModelConfigurationChange,
    ModelDescriptor,
    ConnectionTestResult,
    ProviderDescriptor,
    ProviderRole,
    ProviderSelection,
    QueryRun,
    QueryRunPage,
    QueryStep,
    RetrievalConfiguration,
    MemoryPage,
    IngestionRun,
    RemoveResult,
    MemoryQueryResult,
    MemorySearchResult,
    MemoryTermPage,
    IndexSnapshot,
    SnapshotDiff,
    SourceIngestionResult,
    SourceInput,
    SourcePreview,
    SourcePreviewItem,
    SourceRecord,
    StoreValidation,
    VectorNeighbors,
    VectorProjection,
)
from trisynapse_memory.engine.providers.registry import (
    EmbeddingRebuildRequired,
    ProviderError,
    ProviderSettings,
    complete_json_parallel,
    completion_from_settings,
    embedder_from_settings,
    embedding_cache_key,
    fetch_model_catalog,
    list_provider_descriptors,
    provider_descriptor,
    provider_provenance,
    settings_from_selection,
    skippable_completion_error,
    validate_selection,
)
from trisynapse_memory.engine.retrieval.engine import HybridRetriever, RetrieverConfig, classify_query
from trisynapse_memory.engine.retrieval.contracts import QueryPlanner, RouteRegistry
from trisynapse_memory.engine.memory_log.store import SQLiteMemoryLogStore
from trisynapse_memory.engine.retrieval.tokenization import TokenCounter, token_counter_for
from trisynapse_memory.engine.ingest.sources import BuiltinSourceHandler, PreparedChunk, PreparedSource, store_blob
from trisynapse_memory.engine.indexes.embedding_index import EmbeddingIndex, preferred_embedding_index
from trisynapse_memory.prompts import load_prompt
from trisynapse_memory.engine.extensions import (
    EngineExtensionRegistry,
    ExtensionState,
    CandidateReranker,
    IngestContext,
    IngestEvent,
    FusionStrategy,
    MemoryExtension,
    ProposedRecord,
    IndexWriter,
    SourcePreparationContext,
    MemoryLogReader,
    MemoryLogChangeBatch,
    WeightedRRFFusion,
)
from trisynapse_memory.engine.utils import __version__

CompletionJSON = Callable[[str, str], dict[str, Any]]


class SnapshotManager:
    """Recall-window snapshots; rollback never mutates the underlying trace."""

    def __init__(self, store: SQLiteMemoryLogStore) -> None:
        self._store = store

    def create(self, label: str | None = None) -> IndexSnapshot:
        return self._store.create_snapshot(label)

    def list(self) -> list[IndexSnapshot]:
        return self._store.list_snapshots()

    def diff(self, a: str, b: str) -> SnapshotDiff:
        return self._store.snapshot_diff(a, b)

    def rollback(self, snapshot_id: str) -> IndexSnapshot:
        """Activate a historical evidence cutoff without deleting newer deltas."""

        return self._store.activate_snapshot(snapshot_id)


class MemoryEngine:
    """Local-first append-only memory with disposable recall products."""

    VERSION = __version__

    def __init__(
        self,
        store: SQLiteMemoryLogStore,
        *,
        embedder: Embedder | None = None,
        completion: CompletionJSON | None = None,
        retriever_config: RetrieverConfig | None = None,
        query_planner: QueryPlanner | None = None,
        retrieval_routes: RouteRegistry | None = None,
        token_counter: TokenCounter | None = None,
        embedding_index: EmbeddingIndex | None = None,
        extension_registry: EngineExtensionRegistry | None = None,
        fusion_strategy: FusionStrategy | None = None,
        candidate_reranker: CandidateReranker | None = None,
        default_namespace: MemoryNamespace | dict[str, Any] | None = None,
        auto_process: bool = True,
        _managed_completion: bool = False,
        _managed_embedding: bool = False,
        _provider_errors: dict[str, str] | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.completion = completion
        self._retriever_override = retriever_config is not None
        self.retriever_config = retriever_config or _retriever_config(
            self.store.get_retrieval_configuration()
        )
        self.query_planner = query_planner
        self.retrieval_routes = retrieval_routes
        self._token_counter_override = token_counter is not None
        self.token_counter = token_counter or token_counter_for(completion)
        self.embedding_index = embedding_index or preferred_embedding_index(store)
        self.extensions = extension_registry or _build_extension_registry(())
        self.fusion_strategy = fusion_strategy or WeightedRRFFusion()
        self.candidate_reranker = candidate_reranker
        self.snapshot = SnapshotManager(store)
        self.default_namespace = _namespace(default_namespace)
        self.auto_process = auto_process
        self._managed_completion = _managed_completion
        self._managed_embedding = _managed_embedding
        self._provider_errors = dict(_provider_errors or {})
        self._config_revision = self.store.get_model_configuration().revision

    @classmethod
    def open(
        cls,
        path: str | Path = "~/.trisynapse-memory/store",
        *,
        embedder: Embedder | None = None,
        completion: CompletionJSON | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        local_files_only: bool = False,
        embedding_index: EmbeddingIndex | None = None,
        extensions: Iterable[MemoryExtension] = (),
        extension_registry: EngineExtensionRegistry | None = None,
        fusion_strategy: FusionStrategy | None = None,
        candidate_reranker: CandidateReranker | None = None,
        query_planner: QueryPlanner | None = None,
        retrieval_routes: RouteRegistry | None = None,
        token_counter: TokenCounter | None = None,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
        completion_provider: ProviderSettings | None = None,
        embedding_provider: ProviderSettings | None = None,
        auto_process: bool = True,
    ) -> "MemoryEngine":
        store = SQLiteMemoryLogStore(path)
        configuration = store.get_model_configuration()
        managed_embedding = embedder is None and embedding_provider is None and embedding_model == "all-MiniLM-L6-v2"
        managed_completion = completion is None and completion_provider is None
        provider_errors: dict[str, str] = {}
        provider = _resolve_open_embedder(
            embedder,
            embedding_provider,
            embedding_model,
            configuration,
            local_files_only,
            provider_errors,
        )
        completion_callable = _resolve_open_completion(
            completion, completion_provider, configuration, provider_errors
        )
        registry = _resolve_open_registry(extensions, extension_registry, retrieval_routes)
        engine = cls(
            store,
            embedder=provider,
            completion=completion_callable,
            embedding_index=embedding_index,
            extension_registry=registry,
            fusion_strategy=fusion_strategy,
            candidate_reranker=candidate_reranker,
            query_planner=query_planner,
            retrieval_routes=retrieval_routes,
            token_counter=token_counter,
            default_namespace=namespace,
            auto_process=auto_process,
            _managed_completion=managed_completion,
            _managed_embedding=managed_embedding,
            _provider_errors=provider_errors,
        )
        engine._sync_model_configuration(force=True)
        engine._sync_extension_state()
        return engine

    @classmethod
    def from_env(cls, path: str | Path = "~/.trisynapse-memory/store", **kwargs: Any) -> "MemoryEngine":
        """Open an engine; environment variables provide credentials, never model choices."""

        return cls.open(path, **kwargs)

    @classmethod
    def open_vault(
        cls,
        path: str | Path,
        *,
        enabled: bool | None = None,
        **kwargs: Any,
    ) -> "MemoryEngine":
        flag = enabled if enabled is not None else os.getenv("TRISYNAPSE_MEMORY_V2_ENABLED", "").lower() in {"1", "true", "yes"}
        if not flag:
            raise RuntimeError("Trace & Recall vault integration is disabled; set TRISYNAPSE_MEMORY_V2_ENABLED=true")
        vault = Path(path).expanduser()
        return cls.open(vault / ".trisynapse" / "memory", **kwargs)

    @property
    def store_path(self) -> str:
        return str(self.store.root)

    def close(self) -> None:
        self.store.close()

    # Model configuration
    def list_providers(self) -> list[ProviderDescriptor]:
        return list_provider_descriptors()

    def list_models(
        self,
        role: ProviderRole | str,
        provider: str,
        *,
        refresh: bool = False,
        base_url: str | None = None,
    ) -> list[ModelDescriptor]:
        provider_role = role if isinstance(role, ProviderRole) else ProviderRole(role)
        endpoint = (base_url or provider_descriptor(provider).default_base_url or "").rstrip("/")
        cache_key = hashlib.sha256(
            f"{provider.lower()}|{provider_role.value}|{endpoint.lower()}".encode("utf-8")
        ).hexdigest()
        cached = self.store.get_cached_models(cache_key)
        if cached and not refresh and datetime.now(timezone.utc) - cached[0] < timedelta(minutes=15):
            return cached[1]
        try:
            models = fetch_model_catalog(
                provider, provider_role, base_url=base_url
            )
            self.store.put_cached_models(cache_key, models)
            return models
        except ProviderError:
            if cached:
                return cached[1]
            raise

    def get_model_configuration(self) -> ModelConfiguration:
        return self.store.get_model_configuration()

    def get_model_configuration_status(self) -> ModelConfigurationChange:
        current, pending, job_id, last_error = self.store.get_model_configuration_state()
        if pending is not None:
            return ModelConfigurationChange(
                status="rebuild_pending",
                configuration=current,
                pending_configuration=pending,
                job_id=job_id,
                rebuild_required=True,
                message="The current embedding index remains active while the replacement is built.",
            )
        if last_error:
            return ModelConfigurationChange(
                status="rebuild_failed",
                configuration=current,
                rebuild_required=False,
                message=last_error,
            )
        return ModelConfigurationChange(status="applied", configuration=current)

    def set_model_configuration(
        self,
        configuration: ModelConfiguration | dict[str, Any],
        *,
        confirm_embedding_rebuild: bool = False,
        wait: bool = False,
        expected_revision: int | None = None,
    ) -> ModelConfigurationChange:
        requested = _coerce_model_configuration(configuration)
        validate_selection(requested.completion, ProviderRole.COMPLETION)
        validate_selection(requested.embedding, ProviderRole.EMBEDDING)
        selection_warnings = _model_selection_warnings(self.store, requested)
        current, pending, _, _ = self.store.get_model_configuration_state()
        expected = expected_revision if expected_revision is not None else requested.revision
        if expected != current.revision:
            raise ValueError(
                f"model configuration revision conflict: expected {expected}, current {current.revision}"
            )
        embedding_changed = requested.embedding != current.embedding
        completion_changed = requested.completion != current.completion
        if pending is not None and embedding_changed:
            raise ValueError("an embedding rebuild is already pending")
        if completion_changed:
            current = self._save_completion_selection(current, requested.completion)
        if not embedding_changed:
            return self._applied_model_change(current, selection_warnings)
        return self._apply_embedding_configuration(
            current,
            requested,
            selection_warnings,
            confirm_embedding_rebuild=confirm_embedding_rebuild,
            wait=wait,
        )

    def _save_completion_selection(
        self, current: ModelConfiguration, completion: ProviderSelection
    ) -> ModelConfiguration:
        active = current.model_copy(deep=True)
        active.completion = completion
        return self.store.save_model_configuration(
            active, expected_revision=current.revision
        )

    def _applied_model_change(
        self,
        current: ModelConfiguration,
        warnings: list[str],
        extra_messages: tuple[str, ...] = (),
    ) -> ModelConfigurationChange:
        self._sync_model_configuration(force=True)
        parts = [*extra_messages, *warnings]
        return ModelConfigurationChange(
            status="applied",
            configuration=current,
            message="; ".join(parts) or None,
        )

    def _apply_embedding_configuration(
        self,
        current: ModelConfiguration,
        requested: ModelConfiguration,
        warnings: list[str],
        *,
        confirm_embedding_rebuild: bool,
        wait: bool,
    ) -> ModelConfigurationChange:
        if not self.store.has_searchable_content():
            active = current.model_copy(deep=True)
            active.embedding = requested.embedding
            current = self.store.save_model_configuration(
                active, expected_revision=current.revision
            )
            return self._applied_model_change(
                current,
                warnings,
                extra_messages=(
                    "Embedding configuration changed immediately because the store is empty.",
                ),
            )
        if not confirm_embedding_rebuild:
            raise EmbeddingRebuildRequired(
                "changing the embedding provider, endpoint, or model requires a complete vector rebuild; "
                "retry with confirm_embedding_rebuild=True"
            )
        target = current.model_copy(deep=True)
        target.embedding = requested.embedding
        job = self.store.enqueue_job(
            "rebuild_embeddings",
            {"embedding": requested.embedding.model_dump(mode="json")},
            dedup_key=(
                f"rebuild-embeddings:{current.revision}:{requested.embedding.provider}:"
                f"{requested.embedding.base_url}:{requested.embedding.model}"
            ),
            max_attempts=3,
        )
        self.store.stage_embedding_configuration(target, job.id)
        change = ModelConfigurationChange(
            status="rebuild_pending",
            configuration=current,
            pending_configuration=target,
            job_id=job.id,
            rebuild_required=True,
            message="; ".join([
                "The old embedding index remains active until the rebuild succeeds.",
                *warnings,
            ]),
        )
        if not wait:
            return change
        self._wait_for_job(job)
        self._sync_model_configuration(force=True)
        return self.get_model_configuration_status()

    def _wait_for_job(self, job: MemoryJob) -> MemoryJob | None:
        for _ in range(1000):
            state = self.store.get_job(job.id)
            if state is None or state.status in {"completed", "failed"}:
                break
            self.run_jobs(max_jobs=1)
        return self.store.get_job(job.id)

    def test_model_connection(
        self,
        role: ProviderRole | str,
        selection: ProviderSelection | dict[str, Any] | None = None,
    ) -> ConnectionTestResult:
        provider_role = role if isinstance(role, ProviderRole) else ProviderRole(role)
        current = self.get_model_configuration()
        selected = _coerce_connection_selection(provider_role, current, selection)
        validate_selection(selected, provider_role)
        if selected.provider == "none":
            return ConnectionTestResult(
                ok=True, role=provider_role, provider="none", model=None,
                message="No completion provider is selected.", billed_request=False,
            )
        try:
            vision = None
            if provider_role == ProviderRole.COMPLETION:
                vision = _probe_completion(selected)
            else:
                _probe_embedding(selected)
            return ConnectionTestResult(
                ok=True, role=provider_role, provider=selected.provider,
                model=selected.model, message="Connection succeeded.",
                billed_request=True, vision_supported=vision,
            )
        except Exception as exc:
            return ConnectionTestResult(
                ok=False, role=provider_role, provider=selected.provider,
                model=selected.model, message=str(exc), billed_request=True,
            )

    def get_retrieval_configuration(self) -> RetrievalConfiguration:
        return self.store.get_retrieval_configuration()

    def set_retrieval_configuration(
        self,
        configuration: RetrievalConfiguration | dict[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> RetrievalConfiguration:
        requested = (
            configuration
            if isinstance(configuration, RetrievalConfiguration)
            else RetrievalConfiguration.model_validate(configuration)
        )
        expected = requested.revision if expected_revision is None else expected_revision
        value = self.store.save_retrieval_configuration(
            requested, expected_revision=expected
        )
        if not self._retriever_override:
            self.retriever_config = _retriever_config(value)
        return value

    def check(self) -> dict[str, Any]:
        """Return local installation, provider, store, and job diagnostics."""

        import shutil as system_shutil

        self._sync_model_configuration(force=True)
        path = Path(self.store_path)
        mode = path.stat().st_mode & 0o777 if path.exists() else None
        status = self.get_model_configuration_status()
        providers = {item.id: item for item in self.list_providers()}
        missing = _missing_provider_credentials(
            providers, status.configuration.completion, status.configuration.embedding
        )
        storage = self.validate_store()
        runs = self.list_ingestion_runs(namespace=self.default_namespace)
        return {
            "ok": storage.ok and os.access(path, os.W_OK) and not missing,
            "version": self.VERSION,
            "command": system_shutil.which("trisynapse-memory"),
            "store": str(path),
            "store_writable": os.access(path, os.W_OK),
            "store_permissions": oct(mode) if mode is not None else None,
            "store_encrypted": False,
            "storage": storage.model_dump(mode="json"),
            "model_configuration": status.model_dump(mode="json"),
            "missing_credentials": missing,
            "provider_errors": self._provider_errors,
            "vision_interface": hasattr(self.completion, "complete_multimodal"),
            "token_counter": {
                "name": self.token_counter.name,
                "exact": self.token_counter.exact,
            },
            "source_extras": _source_extra_availability(),
            "port_8765_available": _port_8765_free(),
            "pending_jobs": len(self.list_jobs(status="pending")),
            "failed_jobs": len(self.list_jobs(status="failed")),
            "pending_runs": sum(run.status in {"pending", "running"} for run in runs),
            "failed_runs": sum(run.status in {"failed", "partial"} for run in runs),
            "warning": "Retained originals are permission-restricted but not encrypted.",
        }

    # Ingest
    def ingest_raw_source(
        self,
        text: str,
        *,
        group_id: str | None = None,
        source_ref: dict[str, Any] | str | None = None,
        locator: dict[str, Any] | str | None = None,
        scope: dict[str, Any] | None = None,
        observed_at: datetime | str | None = None,
        external_key: str | None = None,
        actor: Actor | dict[str, Any] | None = None,
        modality: str = "text",
        source_type: str | None = None,
        retrieval_fields: dict[str, Any] | None = None,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
        process: bool | None = None,
        schedule: bool = True,
    ) -> MemoryRecord:
        memory_namespace = self._resolve_namespace(namespace)
        delta = ingest_raw_source(
            self.store,
            text,
            group_id=group_id,
            source_ref=source_ref,
            locator=locator,
            scope=self._scope(memory_namespace, scope),
            observed_at=observed_at,
            external_key=external_key,
            actor=actor,
            modality=modality,
            namespace=memory_namespace,
            payload_extra={
                "source_type": source_type or modality,
                "retrieval_fields": retrieval_fields or {},
            },
        )
        if group_id and schedule:
            self._schedule_group(group_id, memory_namespace, process=process)
        return delta

    def ingest_messages(
        self,
        messages: Iterable[dict[str, Any] | str],
        *,
        group_id: str,
        source_ref: dict[str, Any] | str | None = None,
        scope: dict[str, Any] | None = None,
        run_fact_extraction: bool = False,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        memory_namespace = self._resolve_namespace(namespace, session_id=group_id)
        result: list[MemoryRecord] = []
        for index, message in enumerate(messages):
            if isinstance(message, str):
                text = message
                role = None
                observed_at = None
                message_id = None
            else:
                text = str(message.get("content") or message.get("text") or "")
                role = message.get("role")
                observed_at = message.get("observed_at") or message.get("timestamp")
                message_id = message.get("id")
            prefix = f"{role}: " if role else ""
            result.append(
                self.ingest_raw_source(
                    prefix + text,
                    group_id=group_id,
                    source_ref=source_ref or {"type": "chat", "id": group_id},
                    locator={"kind": "message_index", "message_index": index, "message_id": message_id},
                    scope=scope,
                    namespace=memory_namespace,
                    observed_at=observed_at,
                    external_key=f"message:{group_id}:{message_id or index}",
                    modality="conversation",
                    source_type="conversation",
                    retrieval_fields={"speaker": role or "", "message": text},
                    process=False,
                    schedule=False,
                )
            )
        self._schedule_group(group_id, memory_namespace, force_extraction=run_fact_extraction)
        return result

    def ingest_document(
        self,
        text: str,
        *,
        document_id: str,
        title: str | None = None,
        chunk_chars: int = 3500,
        scope: dict[str, Any] | None = None,
        observed_at: datetime | str | None = None,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        memory_namespace = self._resolve_namespace(namespace, session_id=f"item:{document_id}")
        deltas = ingest_document(
            self.store, text, document_id=document_id, title=title, chunk_chars=chunk_chars,
            scope=self._scope(memory_namespace, scope), observed_at=observed_at,
            namespace=memory_namespace,
            payload_extra={"document_metadata": metadata or {}},
        )
        self._schedule_group(f"item:{document_id}", memory_namespace)
        return deltas

    def ingest_file(
        self,
        path: str | Path,
        *,
        document_id: str | None = None,
        title: str | None = None,
        chunk_chars: int = 3500,
        scope: dict[str, Any] | None = None,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        file_path = Path(path)
        result = self.ingest(
            SourceInput(
                kind="directory" if file_path.is_dir() else "file",
                path=str(file_path),
                source_key=document_id or str(file_path.expanduser().resolve()),
                title=title,
                scope=scope or {},
            ),
            namespace=namespace,
        )
        return [self.get_record(delta_id) for delta_id in result.record_ids]

    def ingest(
        self,
        source: SourceInput | dict[str, Any],
        *,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> SourceIngestionResult:
        run = self.ingest_many([source], namespace=namespace, on_progress=on_progress)
        result = run.results[0]
        if result.status == "failed":
            raise RuntimeError(result.error or "source ingestion failed")
        return result

    def ingest_many(
        self,
        sources: Iterable[SourceInput | dict[str, Any]],
        *,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
        max_workers: int = 4,
        on_progress: Callable[[str], None] | None = None,
    ) -> IngestionRun:
        if on_progress:
            on_progress("Creating durable ingestion run")
        run = self.create_ingestion_run(sources, namespace=namespace)
        return self.process_ingestion_run(
            run.id,
            max_workers=max_workers,
            on_progress=on_progress,
        )

    def create_ingestion_run(
        self,
        sources: Iterable[SourceInput | dict[str, Any]],
        *,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> IngestionRun:
        inputs = [item if isinstance(item, SourceInput) else SourceInput.model_validate(item) for item in sources]
        if not inputs:
            raise ValueError("ingest_many requires at least one source")
        if len(inputs) > 100:
            raise ValueError("ingest_many accepts at most 100 sources")
        memory_namespace = self._resolve_namespace(namespace)
        for item in inputs:
            if item.namespace is not None and item.namespace != memory_namespace:
                raise ValueError("every source in an ingestion run must use the run namespace")
        run = IngestionRun(id=f"run_{secrets.token_hex(10)}", status="pending", namespace=memory_namespace, inputs=inputs)
        self.store.put_ingestion_run(run)
        return run

    def process_ingestion_run(
        self,
        run_id: str,
        *,
        max_workers: int = 4,
        on_progress: Callable[[str], None] | None = None,
    ) -> IngestionRun:
        if on_progress:
            on_progress("Loading source and model configuration")
        self._sync_model_configuration()
        run = self.get_ingestion_run(run_id)
        run.status = "running"
        run.updated_at = datetime.now(timezone.utc)
        self.store.put_ingestion_run(run)
        prepared, failures = self._prepare_ingestion_sources(
            run.inputs, max_workers, on_progress
        )
        _enforce_retained_source_limit(run.inputs, prepared, failures)
        results = self._commit_ingestion_results(
            run.inputs, prepared, failures, run.namespace, run.id, on_progress
        )
        return self._finalize_ingestion_run(run, results, on_progress)

    def _prepare_ingestion_sources(
        self,
        inputs: list[SourceInput],
        max_workers: int,
        on_progress: Callable[[str], None] | None,
    ) -> tuple[dict[int, PreparedSource], dict[int, str]]:
        prepared: dict[int, PreparedSource] = {}
        failures: dict[int, str] = {}
        source_context = SourcePreparationContext(completion=self.completion)
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, 4, len(inputs)))) as pool:
            futures = {
                pool.submit(self.extensions.source_handlers.prepare, item, source_context): index
                for index, item in enumerate(inputs)
            }
            for prepared_count, future in enumerate(as_completed(futures), 1):
                index = futures[future]
                try:
                    prepared[index] = future.result()
                except Exception as exc:
                    failures[index] = str(exc)
                if on_progress:
                    on_progress(
                        f"Preprocessed source {prepared_count} of {len(inputs)}"
                    )
        return prepared, failures

    def _commit_ingestion_results(
        self,
        inputs: list[SourceInput],
        prepared: dict[int, PreparedSource],
        failures: dict[int, str],
        namespace: MemoryNamespace,
        run_id: str,
        on_progress: Callable[[str], None] | None,
    ) -> list[SourceIngestionResult]:
        results: list[SourceIngestionResult] = []
        for index, item in enumerate(inputs):
            if on_progress:
                on_progress(f"Writing source {index + 1} of {len(inputs)} to Trace")
            if index in failures:
                results.append(SourceIngestionResult(
                    index=index, source_key=item.source_key, kind=item.kind,
                    status="failed", error=failures[index],
                ))
                continue
            try:
                results.append(self._commit_source(
                    index, prepared[index], namespace, run_id=run_id
                ))
            except Exception as exc:
                results.append(SourceIngestionResult(
                    index=index, source_key=item.source_key, kind=item.kind,
                    status="failed", error=str(exc),
                ))
        return results

    def _finalize_ingestion_run(
        self,
        run: IngestionRun,
        results: list[SourceIngestionResult],
        on_progress: Callable[[str], None] | None,
    ) -> IngestionRun:
        run.results = results
        run.status = _ingestion_run_status(results)
        run.updated_at = datetime.now(timezone.utc)
        self.store.put_ingestion_run(run)
        if on_progress:
            on_progress("Finalizing ingestion run")
        return run

    def _commit_source(
        self,
        index: int,
        prepared: PreparedSource,
        namespace: MemoryNamespace,
        *,
        run_id: str | None = None,
    ) -> SourceIngestionResult:
        source_key = _prepared_source_key(prepared)
        previous = self.store.latest_source(source_key, namespace)
        if _is_unchanged_source(previous, prepared):
            # A no-op ingestion still verifies and repairs its content-addressed
            # retained blob before returning the existing source version.
            store_blob(self.store.root, prepared.original, filename=prepared.filename)
            return SourceIngestionResult(
                index=index, source_id=previous.id, source_key=source_key, kind=previous.kind,
                status="skipped", group_id=f"source:{previous.id}:v{previous.version}", record_ids=previous.record_ids,
                skipped_paths=prepared.skipped_paths,
            )
        source_id = _source_identity_id(namespace, source_key, prepared.content_hash)
        version = (previous.version + 1) if previous else 1
        group_id = f"source:{source_id}:v{version}"
        blob_path = store_blob(self.store.root, prepared.original, filename=prepared.filename)
        source_ref = {
            "type": prepared.kind, "id": source_id, "source_key": source_key, "title": prepared.title,
            "version": version, "content_hash": prepared.content_hash, "uri": prepared.uri,
        }
        deltas = self._ingest_prepared_chunks(
            prepared, source_id, source_ref, group_id, namespace
        )
        if not deltas:
            raise ValueError("source produced no usable text chunks")
        record = SourceRecord(
            id=source_id,
            source_key=source_key,
            kind=prepared.kind,
            title=prepared.title,
            uri=prepared.uri,
            content_hash=prepared.content_hash,
            blob_path=blob_path,
            media_type=prepared.media_type,
            filename=prepared.filename,
            byte_size=len(prepared.original),
            chunk_count=len(deltas),
            ingestion_run_id=run_id,
            preview_type=str(prepared.metadata.get("source_type") or prepared.kind),
            skipped_count=len(prepared.skipped_paths),
            version=version,
            namespace=namespace,
            metadata={**prepared.metadata, "skipped_paths": prepared.skipped_paths},
            record_ids=[item.id for item in deltas],
            previous_source_id=previous.id if previous else None,
        )
        self.store.put_source(record)
        self._supersede_previous_source(previous, source_id, namespace)
        self._schedule_group(group_id, namespace)
        return SourceIngestionResult(
            index=index, source_id=source_id, source_key=source_key, kind=prepared.kind,
            status="success", group_id=group_id, record_ids=record.record_ids, skipped_paths=prepared.skipped_paths,
        )

    def _ingest_prepared_chunks(
        self,
        prepared: PreparedSource,
        source_id: str,
        source_ref: dict[str, Any],
        group_id: str,
        namespace: MemoryNamespace,
    ) -> list[MemoryRecord]:
        deltas: list[MemoryRecord] = []
        for chunk_index, chunk in enumerate(prepared.chunks):
            if not chunk.text.strip():
                continue
            modality, source_type, retrieval_fields = _source_retrieval_descriptor(
                prepared, chunk
            )
            deltas.append(self.ingest_raw_source(
                chunk.text,
                group_id=group_id,
                source_ref=source_ref,
                locator={**chunk.locator, "chunk_index": chunk_index, "metadata": chunk.metadata},
                scope={**prepared.source.scope, "source_id": source_id, "source_kind": prepared.kind},
                external_key=f"source:{source_id}:chunk:{chunk_index}:{hashlib.sha256(chunk.text.encode()).hexdigest()[:16]}",
                namespace=namespace,
                modality=modality,
                source_type=source_type,
                retrieval_fields=retrieval_fields,
                process=False,
                schedule=False,
            ))
        return deltas

    def _supersede_previous_source(
        self,
        previous: SourceRecord | None,
        source_id: str,
        namespace: MemoryNamespace,
    ) -> None:
        if previous is None or previous.status != "active":
            return
        retracted = self.store.retracted_ids()
        for delta_id in previous.record_ids:
            if delta_id in retracted:
                continue
            self.retract(
                record_id=delta_id,
                reason=f"source replaced by {source_id}",
                requested_by="source-ingestion",
                namespace=namespace,
            )
        previous.status = "superseded"
        previous.removed_at = datetime.now(timezone.utc)
        self.store.put_source(previous)

    def retry_ingestion(self, run_id: str) -> IngestionRun:
        retry = self.create_retry_ingestion(run_id)
        return self.process_ingestion_run(retry.id)

    def create_retry_ingestion(self, run_id: str) -> IngestionRun:
        """Create a failed-only retry, or return an interrupted run to resume."""

        run = self.get_ingestion_run(run_id)
        if run.status in {"pending", "running"}:
            return run
        failed = [run.inputs[result.index] for result in run.results if result.status == "failed"]
        if not failed:
            raise ValueError("ingestion run has no failed sources")
        return self.create_ingestion_run(failed, namespace=run.namespace)

    def get_ingestion_run(self, run_id: str) -> IngestionRun:
        run = self.store.get_ingestion_run(run_id)
        if run is None:
            raise KeyError(f"unknown ingestion run: {run_id}")
        return run

    def list_ingestion_runs(
        self, *, namespace: MemoryNamespace | dict[str, Any] | None = None, limit: int = 100
    ) -> list[IngestionRun]:
        return self.store.list_ingestion_runs(self._resolve_namespace(namespace), limit=limit)

    def list_sources(
        self,
        *,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
        include_removed: bool = False,
        search: str | None = None,
        kinds: Iterable[str] | None = None,
        status: str | None = None,
        sort: str = "newest",
        limit: int | None = None,
        cursor: int = 0,
    ) -> list[SourceRecord]:
        values = self.store.list_sources(
            self._resolve_namespace(namespace), include_removed=include_removed
        )
        values = [self._hydrate_source(item) for item in values]
        values = _filter_sources(values, search=search, kinds=kinds, status=status)
        _sort_sources(values, sort)
        if limit is not None:
            values = values[cursor:cursor + limit]
        return values

    def get_source(
        self, source_id: str, *, namespace: MemoryNamespace | dict[str, Any] | None = None
    ) -> SourceRecord:
        source = self.store.get_source(source_id)
        requested = self._resolve_namespace(namespace)
        if source is None or source.namespace != requested:
            raise KeyError(f"unknown source: {source_id}")
        return self._hydrate_source(source)

    def _hydrate_source(self, source: SourceRecord) -> SourceRecord:
        update = _source_hydrate_updates(self.store, source)
        if not update:
            return source
        hydrated = source.model_copy(update=update)
        self.store.put_source(hydrated)
        return hydrated

    def source_preview(
        self,
        source_id: str,
        *,
        cursor: int = 0,
        limit: int = 50,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> SourcePreview:
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")
        source = self.get_source(source_id, namespace=namespace)
        if source.status == "removed":
            raise KeyError(f"unknown source: {source_id}")
        selected_ids = source.record_ids[cursor:cursor + limit + 1]
        values, manifest = _source_preview_items(self.store, selected_ids[:limit])
        return SourcePreview(
            source_id=source.id,
            preview_type=source.preview_type or source.kind,
            media_type=source.media_type,
            items=values,
            next_cursor=cursor + limit if len(selected_ids) > limit else None,
            manifest=manifest,
        )

    def source_content_path(
        self,
        source_id: str,
        *,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> tuple[SourceRecord, Path]:
        source = self.get_source(source_id, namespace=namespace)
        if source.status == "removed":
            raise KeyError(f"unknown source: {source_id}")
        path = (self.store.root / source.blob_path).resolve()
        source_root = (self.store.root / "sources").resolve()
        if source_root not in path.parents or not path.is_file():
            raise FileNotFoundError(source_id)
        return source, path

    def run_fact_extraction(
        self,
        *,
        group_id: str,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        return self.run_fact_extraction_many([group_id], namespace=namespace)

    def run_fact_extraction_many(
        self,
        group_ids: Sequence[str],
        *,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
        max_workers: int = DEFAULT_COMPLETION_PARALLEL_WORKERS,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[MemoryRecord]:
        self._sync_model_configuration()
        if self.completion is None:
            raise RuntimeError("run_fact_extraction requires a configured completion provider")
        return extract_facts_many(
            self.store,
            group_ids,
            self.completion,
            namespace=self._resolve_namespace(namespace),
            max_workers=max_workers,
            on_progress=on_progress,
        )

    def build_group_summary(
        self,
        group_ids: list[str] | None = None,
        *,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
        max_workers: int = DEFAULT_COMPLETION_PARALLEL_WORKERS,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[GroupSummaryView]:
        self._sync_model_configuration()
        memory_namespace = self._resolve_namespace(namespace)
        deltas = self.store.list_records(
            kinds=["raw_source", "fact", "note"], namespace=memory_namespace
        )
        existing, targets, pending_ids = _group_summary_plan(
            self.store, memory_namespace, deltas, group_ids
        )
        if not pending_ids:
            return _cached_group_summaries(targets, existing, on_progress)
        already = len(targets) - len(pending_ids)
        views = build_group_summary_views(
            deltas,
            complete_json=self.completion,
            group_ids=pending_ids,
            max_workers=max_workers,
            on_progress=_group_summary_progress(on_progress, already, len(targets)),
            on_view=lambda view: _persist_group_summary(
                self.store, view, memory_namespace
            ),
        )
        return _merge_group_summaries(targets, existing, views)

    def retract(
        self,
        *,
        record_id: str,
        reason: str,
        requested_by: str = "user",
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> MemoryRecord:
        memory_namespace = self._resolve_namespace(namespace)
        target = self.get(record_id, namespace=memory_namespace)
        if target is None:
            raise KeyError(f"unknown record: {record_id}")
        return self.store.append(
            kind="retraction",
            text=f"Retracted {record_id}: {reason}",
            evidence_refs=[record_id],
            payload={"target_record_ids": [record_id], "reason": reason, "requested_by": requested_by},
            actor=Actor(type="user", id=requested_by),
            confidence=1.0,
            namespace=memory_namespace,
            scope=self._scope(memory_namespace, None),
        )

    forget = retract

    def correct(
        self,
        *,
        record_id: str,
        text: str,
        reason: str = "user correction",
        requested_by: str = "user",
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> MemoryRecord:
        memory_namespace = self._resolve_namespace(namespace)
        target = self.get(record_id, namespace=memory_namespace)
        return self.store.append(
            kind="fact",
            text=text,
            group_id=target.group_id,
            evidence_refs=[record_id],
            payload={"note_type": "correction", "target_record_ids": [record_id], "reason": reason},
            actor=Actor(type="user", id=requested_by),
            confidence=1.0,
            namespace=memory_namespace,
            scope=target.scope,
        )

    def remove(
        self,
        *,
        record_ids: list[str],
        reason: str,
        requested_by: str = "user",
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> RemoveResult:
        memory_namespace = self._resolve_namespace(namespace)
        for delta_id in record_ids:
            self.get(delta_id, namespace=memory_namespace, include_retracted=True)
        for delta_id in record_ids:
            if delta_id not in self.store.retracted_ids():
                self.retract(record_id=delta_id, reason=reason, requested_by=requested_by, namespace=memory_namespace)
        result = self.store.hard_remove(record_ids, requested_by=requested_by, reason=reason)
        self.embedding_index.clear()
        return result

    def remove_source(
        self,
        source_id: str,
        *,
        reason: str,
        requested_by: str = "user",
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> RemoveResult:
        source = self.get_source(source_id, namespace=namespace)
        result = self.remove(record_ids=source.record_ids, reason=reason, requested_by=requested_by, namespace=source.namespace)
        source.status = "removed"
        source.removed_at = datetime.now(timezone.utc)
        self.store.put_source(source)
        if not self.store.blob_is_referenced(source.blob_path, excluding_source_id=source.id):
            blob = self.store.root / source.blob_path
            blob.unlink(missing_ok=True)
        return result

    def add(self, text: str, **kwargs: Any) -> MemoryRecord:
        return self.ingest_raw_source(text, **kwargs)

    def add_many(self, items: Iterable[dict[str, Any] | str], **kwargs: Any) -> list[MemoryRecord]:
        result: list[MemoryRecord] = []
        scheduled: dict[tuple[str, str], tuple[str, MemoryNamespace, bool | None]] = {}
        for item in items:
            payload = {**kwargs, **({"text": item} if isinstance(item, str) else item)}
            group_id = payload.get("group_id")
            if group_id:
                namespace = self._resolve_namespace(payload.get("namespace"))
                process = payload.get("process")
                payload["schedule"] = False
                scheduled[(_namespace_key(namespace), str(group_id))] = (str(group_id), namespace, process)
            result.append(self.add(**payload))
        for group_id, namespace, process in scheduled.values():
            self._schedule_group(group_id, namespace, process=process)
        return result

    def append_deltas(self, deltas: Iterable[dict[str, Any]]) -> list[MemoryRecord]:
        return [self.store.append(**delta) for delta in deltas]

    # Retrieval
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        group_prefix: str | None = None,
        scope: dict[str, Any] | None = None,
        query_id: str | None = None,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
        on_step: Callable[[QueryStep], None] | None = None,
        persist: bool = True,
    ) -> MemorySearchResult:
        memory_namespace = self._resolve_namespace(namespace)
        identifier = query_id or f"q_{secrets.token_hex(8)}"
        run = self._new_query_run(identifier, "search", query, memory_namespace, persist=persist)
        if on_step and run.steps:
            on_step(run.steps[0])
        return self._execute_search(
            run,
            top_k=top_k,
            group_prefix=group_prefix,
            scope=scope,
            finish=True,
            on_step=on_step,
            persist=persist,
        )

    def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        group_prefix: str | None = None,
        scope: dict[str, Any] | None = None,
        abstain_threshold: float | None = None,
        history: list[dict[str, Any]] | None = None,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
        on_step: Callable[[QueryStep], None] | None = None,
        answer_question: str | None = None,
    ) -> MemoryQueryResult:
        """Retrieve with ``question``; generate from ``answer_question`` when set.

        Benchmark MCQ options belong in ``answer_question`` so they do not
        enter the retriever query.
        """

        del history
        memory_namespace = self._resolve_namespace(namespace)
        run = self._new_query_run(
            f"q_{secrets.token_hex(8)}", "query", question, memory_namespace
        )
        if on_step and run.steps:
            on_step(run.steps[0])
        return self._execute_query(
            run,
            original_question=question,
            top_k=top_k,
            group_prefix=group_prefix,
            scope=scope,
            abstain_threshold=abstain_threshold,
            on_step=on_step,
            answer_question=answer_question,
        )

    def _new_query_run(
        self,
        identifier: str,
        mode: str,
        query: str,
        namespace: MemoryNamespace,
        *,
        status: str = "running",
        persist: bool = True,
    ) -> QueryRun:
        configuration = self.get_retrieval_configuration()
        run = QueryRun(
            id=identifier,
            mode=mode,
            status=status,
            namespace=namespace,
            query=query,
            retrieval_configuration=configuration,
            steps=[QueryStep(
                id=f"{identifier}:0:input",
                phase="input",
                label="Record query input",
                sequence=0,
                output={"query": query},
            )] if status == "running" else [],
            attempt=1 if status == "running" else 0,
        )
        if persist:
            self.store.put_query_run(run)
        return run

    def _append_query_step(
        self,
        run: QueryRun,
        step: QueryStep,
        on_step: Callable[[QueryStep], None] | None = None,
        *,
        persist: bool = True,
    ) -> None:
        run.steps = [item for item in run.steps if item.id != step.id]
        run.steps.append(step)
        run.steps.sort(key=lambda item: item.sequence)
        run.updated_at = datetime.now(timezone.utc)
        if persist:
            self.store.put_query_run(run)
        if on_step:
            on_step(step)

    def _execute_search(
        self,
        run: QueryRun,
        *,
        top_k: int | None,
        group_prefix: str | None,
        scope: dict[str, Any] | None,
        finish: bool,
        on_step: Callable[[QueryStep], None] | None = None,
        persist: bool = True,
    ) -> MemorySearchResult:
        self._sync_model_configuration()
        retriever = self._hybrid_retriever()
        effective_top_k = top_k or run.retrieval_configuration.default_top_k
        started = time.perf_counter()
        result = retriever.search(
            run.query,
            top_k=effective_top_k,
            group_prefix=group_prefix,
            scope=self._scope(run.namespace, scope),
            namespace=run.namespace,
            query_id=run.id,
            on_step=lambda step: self._append_query_step(run, step, on_step, persist=persist),
        )
        run.retrieval_trace = result.retrieval_trace
        if finish:
            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc)
            run.updated_at = run.completed_at
            run.duration_ms = (time.perf_counter() - started) * 1000
            if persist:
                self.store.put_query_run(run)
        return result

    def _hybrid_retriever(self) -> HybridRetriever:
        if not self._retriever_override:
            self.retriever_config = _retriever_config(self.get_retrieval_configuration())
        return HybridRetriever(
            self.store,
            self.embedder,
            self.embedding_index,
            self.retriever_config,
            planner=self.query_planner,
            routes=self.retrieval_routes,
            branches=self.extensions.retrieval_branches,
            fusion_strategy=self.fusion_strategy,
            reranker=self.candidate_reranker,
            unavailable_branches=self._unavailable_retrieval_branches(),
            token_counter=self.token_counter,
        )

    def _unavailable_retrieval_branches(self) -> set[str]:
        unavailable: set[str] = set()
        states = {item.extension_id: item for item in self.store.extension_states()}
        for name in self.extensions.retrieval_branches.names:
            owner = self.extensions.owner("retrieval", name)
            if owner is None:
                continue
            state = states.get(owner)
            if state is None or state.status == "available":
                continue
            unavailable.add(name)
        return unavailable

    def _execute_query(
        self,
        run: QueryRun,
        *,
        original_question: str,
        top_k: int | None,
        group_prefix: str | None,
        scope: dict[str, Any] | None,
        abstain_threshold: float | None,
        on_step: Callable[[QueryStep], None] | None = None,
        answer_question: str | None = None,
    ) -> MemoryQueryResult:
        started = time.perf_counter()
        search_result = self._execute_search(
            run,
            # ``default_top_k`` controls ordinary search results. Answer
            # generation gets the larger grounded context window unless the
            # caller explicitly requests a limit.
            top_k=_query_context_top_k(top_k, run),
            group_prefix=group_prefix,
            scope=scope,
            finish=False,
            on_step=on_step,
        )
        hits = search_result.hits
        threshold = _query_abstain_threshold(run, abstain_threshold)
        relevant = _hits_are_relevant(hits, threshold)
        generation_started = time.perf_counter()
        answer, abstain, cited_hits = _query_answer(
            self.completion, answer_question or original_question, hits, relevant
        )
        citations = _citations([] if abstain else cited_hits)
        provenance = _query_generation_provenance(self.completion)
        self._append_query_step(run, QueryStep(
            id=f"{run.id}:{len(run.steps)}:answer",
            phase="answer",
            label=_query_answer_label(relevant),
            sequence=len(run.steps),
            input={"evidence_ids": [hit.item_id for hit in hits], "threshold": threshold},
            output={"answer": answer, "abstain": abstain, "citation_ids": [item.record_id for item in citations]},
            metrics={"evidence_count": len(hits), "citation_count": len(citations)},
            duration_ms=(time.perf_counter() - generation_started) * 1000,
        ), on_step)
        self.store.append(
            kind="access",
            text=f"Query access {search_result.query_id}",
            evidence_refs=[citation.record_id for citation in citations],
            payload={
                "query_id": search_result.query_id,
                "compiled_view_ids_used": [],
                "was_helpful": None,
                "generation": provenance,
            },
            confidence=1.0,
            namespace=run.namespace,
            scope=self._scope(run.namespace, scope),
        )
        self._append_query_step(run, QueryStep(
            id=f"{run.id}:{len(run.steps)}:audit",
            phase="audit",
            label="Record citation access",
            sequence=len(run.steps),
            output={"query_id": run.id, "citation_ids": [item.record_id for item in citations]},
        ), on_step)
        run.answer = answer
        run.abstain = abstain
        run.citations = citations
        run.generation_provenance = provenance
        run.status = "completed"
        run.completed_at = datetime.now(timezone.utc)
        run.updated_at = run.completed_at
        run.duration_ms = (time.perf_counter() - started) * 1000
        self.store.put_query_run(run)
        return MemoryQueryResult(
            query_id=search_result.query_id,
            question=original_question,
            answer=answer,
            abstain=abstain,
            citations=citations,
            retrieval_hits=hits,
            retrieval_trace=search_result.retrieval_trace,
        )

    def create_query_run(
        self,
        query: str,
        *,
        mode: str = "query",
        top_k: int | None = None,
        group_prefix: str | None = None,
        scope: dict[str, Any] | None = None,
        abstain_threshold: float | None = None,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> QueryRun:
        if mode not in {"query", "search"}:
            raise ValueError("mode must be query or search")
        memory_namespace = self._resolve_namespace(namespace)
        run = self._new_query_run(
            f"q_{secrets.token_hex(8)}", mode, query, memory_namespace, status="pending"
        )
        job = self.store.enqueue_job(
            "execute_query",
            {
                "query_id": run.id,
                "query": query,
                "mode": mode,
                "top_k": top_k,
                "group_prefix": group_prefix,
                "scope": scope or {},
                "abstain_threshold": abstain_threshold,
                "namespace": memory_namespace.model_dump(mode="json"),
            },
            dedup_key=f"execute-query:{run.id}",
            max_attempts=1,
        )
        run.generation_provenance = {"job_id": job.id}
        self.store.put_query_run(run)
        return run

    def execute_query_run(self, query_id: str) -> QueryRun:
        run = self.get_query_run(query_id)
        if run.status == "completed":
            return run
        payload_job = next(
            (job for job in self.list_jobs(limit=500) if job.payload.get("query_id") == query_id),
            None,
        )
        payload = payload_job.payload if payload_job else {}
        run.status = "running"
        run.attempt += 1
        run.updated_at = datetime.now(timezone.utc)
        run.steps = [QueryStep(
            id=f"{run.id}:0:input",
            phase="input",
            label="Record query input",
            sequence=0,
            output={"query": run.query},
        )]
        self.store.put_query_run(run)
        try:
            if run.mode == "search":
                self._execute_search(
                    run,
                    top_k=payload.get("top_k"),
                    group_prefix=payload.get("group_prefix"),
                    scope=payload.get("scope"),
                    finish=True,
                )
            else:
                self._execute_query(
                    run,
                    original_question=run.query,
                    top_k=payload.get("top_k"),
                    group_prefix=payload.get("group_prefix"),
                    scope=payload.get("scope"),
                    abstain_threshold=payload.get("abstain_threshold"),
                )
        except Exception as exc:
            run.status = "failed"
            run.error = str(exc)[:1000]
            run.updated_at = datetime.now(timezone.utc)
            run.completed_at = run.updated_at
            self.store.put_query_run(run)
            raise
        return run

    def get_query_run(
        self,
        query_id: str,
        *,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> QueryRun:
        run = self.store.get_query_run(query_id)
        requested = self._resolve_namespace(namespace)
        if run is None or run.namespace != requested:
            raise KeyError(f"unknown query run: {query_id}")
        return run

    def list_query_runs(
        self,
        *,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
        limit: int = 50,
        cursor: str | None = None,
        search: str | None = None,
        mode: str | None = None,
        status: str | None = None,
        stage: str | None = None,
        before: datetime | None = None,
    ) -> QueryRunPage:
        return self.store.list_query_runs(
            self._resolve_namespace(namespace), limit=limit, cursor=cursor,
            search=search, mode=mode, status=status, stage=stage, before=before,
        )

    def remove_query_runs(
        self,
        *,
        query_ids: list[str] | None = None,
        before: datetime | None = None,
        all_in_namespace: bool = False,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> list[str]:
        return self.store.remove_query_runs(
            self._resolve_namespace(namespace), query_ids=query_ids,
            before=before, all_in_namespace=all_in_namespace,
        )

    # Introspection
    def get_record(self, delta_id: str) -> MemoryRecord:
        delta = self.store.get(delta_id)
        if delta is None:
            raise KeyError(f"unknown delta: {delta_id}")
        return delta

    def get(
        self,
        delta_id: str,
        *,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
        include_retracted: bool = False,
    ) -> MemoryRecord:
        delta = self.get_record(delta_id)
        requested = self._resolve_namespace(namespace)
        requested_values = requested.model_dump(exclude_none=True)
        if any(getattr(delta.namespace, key) != value for key, value in requested_values.items()):
            raise KeyError(f"unknown delta: {delta_id}")
        if not include_retracted and delta_id in self.store.retracted_ids():
            raise KeyError(f"unknown delta: {delta_id}")
        return delta

    def list(
        self,
        *,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
        kinds: Iterable[str] | None = None,
        scope: dict[str, Any] | None = None,
        cursor: int | None = None,
        limit: int = 50,
        include_retracted: bool = False,
    ) -> MemoryPage:
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        memory_namespace = self._resolve_namespace(namespace)
        items = self.store.list_records(
            kinds=kinds,
            namespace=memory_namespace,
            scope=self._scope(memory_namespace, scope),
            after_seq=cursor,
            limit=limit + 1,
            include_retracted=include_retracted,
        )
        next_cursor = items[limit - 1].seq if len(items) > limit else None
        return MemoryPage(items=items[:limit], next_cursor=next_cursor)

    def history(
        self,
        delta_id: str,
        *,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> MemoryHistory:
        self.get(delta_id, namespace=namespace, include_retracted=True)
        return MemoryHistory(memory_id=delta_id, events=self.store.history(delta_id))

    def export(
        self,
        *,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
        include_retracted: bool = True,
    ) -> dict[str, Any]:
        memory_namespace = self._resolve_namespace(namespace)
        deltas = self.store.list_records(namespace=memory_namespace, include_retracted=include_retracted)
        return {
            "schema_version": 2,
            "engine_version": self.VERSION,
            "namespace": memory_namespace.model_dump(mode="json"),
            "deltas": [item.model_dump(mode="json") for item in deltas],
        }

    def export_to(self, path: str | Path, **kwargs: Any) -> Path:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.export(**kwargs), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return output

    def backup(self, destination: str | Path) -> Path:
        validation = self.validate_store()
        if not validation.ok:
            raise RuntimeError(
                "memory store failed validation before backup: "
                + "; ".join(validation.issues)
            )
        output = Path(destination).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.store._connection.execute("PRAGMA wal_checkpoint(FULL)")
        archive = shutil.make_archive(str(output.with_suffix("")), "zip", root_dir=self.store.root)
        return Path(archive)

    @classmethod
    def restore_backup(cls, archive: str | Path, destination: str | Path) -> "MemoryEngine":
        source = Path(archive).expanduser().resolve()
        target = Path(destination).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(f"restore destination must be empty: {target}")
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source) as bundle:
            for member in bundle.infolist():
                resolved = (target / member.filename).resolve()
                if target not in resolved.parents and resolved != target:
                    raise ValueError(f"unsafe backup member: {member.filename}")
            bundle.extractall(target)
        engine = cls.open(target)
        validation = engine.validate_store()
        if not validation.ok:
            engine.close()
            raise RuntimeError(
                "restored memory store failed validation: " + "; ".join(validation.issues)
            )
        return engine

    def get_retrieval_trace(
        self,
        query_id: str,
        *,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> Any:
        trace = self.store.get_retrieval_trace(query_id)
        if trace is None:
            return None
        requested = self._resolve_namespace(namespace)
        requested_values = requested.model_dump(exclude_none=True)
        if any(getattr(trace.namespace, key) != value for key, value in requested_values.items()):
            return None
        return trace

    def list_groups(
        self,
        *,
        scope: dict[str, Any] | None = None,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> list[RecordGroupInfo]:
        memory_namespace = self._resolve_namespace(namespace)
        grouped: dict[str, list[MemoryRecord]] = defaultdict(list)
        for delta in self.store.list_records(scope=self._scope(memory_namespace, scope), namespace=memory_namespace):
            if not delta.group_id:
                continue
            grouped[delta.group_id].append(delta)
        fresh = {view.group_id: not view.stale for view in self.store.group_summary_views(namespace=memory_namespace)}
        return [
            _record_group_info(group_id, deltas, fresh)
            for group_id, deltas in sorted(grouped.items())
        ]

    def export_graph(
        self,
        format: str = "json",
        *,
        scope: dict[str, Any] | None = None,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> dict[str, Any] | str:
        memory_namespace = self._resolve_namespace(namespace)
        graph = _claim_export_graph(
            _namespace_claims(
                self.store, memory_namespace, self._scope(memory_namespace, scope)
            )
        )
        if format == "json":
            return graph
        if format != "graphml":
            raise ValueError("format must be 'json' or 'graphml'")
        return _graphml(graph)

    def memory_graph(
        self,
        view: str = "knowledge",
        *,
        search: str | None = None,
        node_types: Iterable[str] | None = None,
        source_id: str | None = None,
        group_id: str | None = None,
        limit: int = 500,
        cursor: str | None = None,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> MemoryGraphPage:
        _validate_memory_graph_args(view, limit)
        memory_namespace = self._resolve_namespace(namespace)
        if view == "retrieval":
            return index_inspect.retrieval_graph_page(
                self, memory_namespace, seed_id=source_id or group_id, limit=limit
            )
        deltas = _filter_graph_deltas(
            self.store.list_records(namespace=memory_namespace, include_retracted=True),
            group_id=group_id,
            source_id=source_id,
        )
        claims = _memory_graph_claims(deltas)
        nodes: dict[str, MemoryGraphNode] = {}
        edges: dict[str, MemoryGraphEdge] = {}
        if view in {"lineage", "log"}:
            _add_source_lineage_nodes(
                nodes,
                edges,
                self.list_sources(namespace=memory_namespace, include_removed=True),
                source_id,
            )
            _add_log_delta_nodes(
                view, nodes, edges, deltas, self.store.retracted_ids()
            )
        if view in {"knowledge", "lineage"}:
            _add_knowledge_claim_graph(
                view,
                nodes,
                edges,
                claims,
                self.store.group_summary_views(namespace=memory_namespace),
            )
        return _paginate_memory_graph(
            view, nodes, edges, search, node_types, limit, cursor
        )

    def memory_graph_neighbors(
        self,
        node_id: str,
        *,
        view: str = "lineage",
        limit: int = 500,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> MemoryGraphPage:
        if view == "retrieval":
            memory_namespace = self._resolve_namespace(namespace)
            return index_inspect.retrieval_graph_page(
                self, memory_namespace, seed_id=node_id, limit=limit
            )
        graph = self.memory_graph(view, limit=2000, namespace=namespace)
        return _neighbor_subgraph(graph, node_id, limit)

    def memory_catalog(
        self,
        *,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> MemoryCatalog:
        return index_inspect.memory_catalog(self, self._resolve_namespace(namespace))

    def memory_helper_items(
        self,
        helper_id: str,
        *,
        search: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> MemoryHelperPage:
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        return index_inspect.helper_items(
            self, helper_id, self._resolve_namespace(namespace),
            search=search, cursor=cursor, limit=limit,
        )

    def memory_documents(
        self,
        *,
        search: str | None = None,
        modality: str | None = None,
        cursor: int = 0,
        limit: int = 50,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> MemoryDocumentPage:
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        return index_inspect.document_page(
            self, self._resolve_namespace(namespace),
            search=search, modality=modality, cursor=cursor, limit=limit,
        )

    def memory_terms(
        self,
        *,
        search: str | None = None,
        cursor: int = 0,
        limit: int = 40,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> MemoryTermPage:
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")
        return index_inspect.term_page(
            self, self._resolve_namespace(namespace), search=search, cursor=cursor, limit=limit,
        )

    def memory_claims(
        self,
        *,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> list[Claim]:
        return index_inspect.compiled_claims(self, self._resolve_namespace(namespace))

    def memory_vector_projection(
        self,
        *,
        sample: int = 500,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> VectorProjection:
        sample = max(10, min(sample, 1000))
        return index_inspect.vector_projection(self, self._resolve_namespace(namespace), sample=sample)

    def memory_vector_neighbors(
        self,
        record_id: str,
        *,
        limit: int = 12,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> VectorNeighbors:
        return index_inspect.vector_neighbors(
            self, self._resolve_namespace(namespace), record_id, limit=max(1, min(limit, 50))
        )

    def memory_retrieval_graph(
        self,
        *,
        seed_id: str | None = None,
        edge_kind: str | None = None,
        limit: int = 400,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> MemoryGraphPage:
        return index_inspect.retrieval_graph_page(
            self, self._resolve_namespace(namespace),
            seed_id=seed_id, edge_kind=edge_kind, limit=max(1, min(limit, 2000)),
        )

    def compile_profile(
        self,
        *,
        scope: dict[str, Any] | None = None,
        query: str | None = None,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        memory_namespace = self._resolve_namespace(namespace)
        profile = _profile_from_claims(
            _namespace_claims(
                self.store, memory_namespace, self._scope(memory_namespace, scope)
            )
        )
        if query:
            profile["search_results"] = [
                hit.model_dump(mode="json")
                for hit in self.search(query, scope=scope, namespace=memory_namespace).hits
            ]
        return profile

    def record_feedback(
        self,
        query_id: str,
        *,
        helpful: bool,
        comment: str | None = None,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> MemoryRecord:
        memory_namespace = self._resolve_namespace(namespace)
        trace = self.get_retrieval_trace(query_id, namespace=memory_namespace)
        if trace is None:
            raise KeyError(f"unknown query: {query_id}")
        return self.store.append(
            kind="access",
            text=f"Feedback for query {query_id}: {'helpful' if helpful else 'not helpful'}",
            payload={"query_id": query_id, "was_helpful": helpful, "comment": comment},
            namespace=memory_namespace,
            scope=self._scope(memory_namespace, None),
            confidence=1.0,
        )

    def validate_store(self, *, check_source_blobs: bool = True) -> StoreValidation:
        return self.store.validate(check_source_blobs=check_source_blobs)

    def list_jobs(self, *, status: str | None = None, limit: int = 100) -> list[MemoryJob]:
        return self.store.list_jobs(status=status, limit=limit)

    def rebuild_extension(self, extension_id: str, *, wait: bool = False) -> MemoryJob:
        spec = self.extensions.extensions.get(extension_id)
        if spec is None:
            raise KeyError(f"extension is not installed: {extension_id}")
        kind = f"extension:{extension_id}:rebuild"
        if self.extensions.job_handlers.get(kind) is None:
            raise ValueError(f"extension has no persistent Recall channels: {extension_id}")
        evidence_version = self.store.max_seq()
        job = self.store.enqueue_job(
            kind,
            {"extension_id": extension_id, "evidence_version": evidence_version},
            dedup_key=(
                f"extension-rebuild:{extension_id}:{spec.storage_revision}:"
                f"{evidence_version}"
            ),
        )
        previous = next(
            (item for item in self.store.extension_states() if item.extension_id == extension_id),
            None,
        )
        self.store.put_extension_state(ExtensionState(
            extension_id=extension_id,
            installed_version=spec.version,
            engine_api=spec.engine_api,
            storage_revision=spec.storage_revision,
            last_projected_seq=previous.last_projected_seq if previous else 0,
            status="rebuild_required",
        ))
        if not wait:
            return job
        current = self._wait_for_job(job)
        if current is not None:
            return current
        return job

    def run_jobs(self, *, max_jobs: int = 100) -> list[MemoryJob]:
        self._sync_model_configuration()
        completed: list[MemoryJob] = []
        for _ in range(max_jobs):
            job = self.store.claim_job()
            if job is None:
                break
            try:
                self.extensions.job_handlers.run(self, job)
                completed.append(self.store.finish_job(job.id))
            except Exception as exc:
                finished = self.store.finish_job(job.id, error=str(exc))
                self._record_extension_job_failure(job, str(exc))
                if job.kind == "rebuild_embeddings" and finished.status == "failed":
                    self.store.fail_pending_embedding(job.id, str(exc))
                if job.kind == "execute_query":
                    run = self.store.get_query_run(str(job.payload.get("query_id") or ""))
                    if run is not None and run.status not in {"completed", "failed"}:
                        run.status = "failed"
                        run.error = str(exc)[:1000]
                        run.updated_at = datetime.now(timezone.utc)
                        self.store.put_query_run(run)
                completed.append(finished)
        return completed

    def _record_extension_job_failure(self, job: MemoryJob, error: str) -> None:
        owner = self._extension_job_owner(job)
        if owner is None:
            return
        self._mark_extension_degraded(owner, error)

    def _extension_job_owner(self, job: MemoryJob) -> str | None:
        kind = job.kind
        if kind.startswith("ingest:"):
            return self.extensions.owner("ingest", kind.removeprefix("ingest:"))
        if kind.startswith("index:") and kind.endswith(":project"):
            return self.extensions.owner(
                "index", kind.removeprefix("index:").removesuffix(":project")
            )
        if kind.startswith("extension:") and kind.endswith(":rebuild"):
            return kind.removeprefix("extension:").removesuffix(":rebuild")
        return None

    def _mark_extension_degraded(self, extension_id: str, error: str) -> None:
        spec = self.extensions.extensions.get(extension_id)
        if spec is None:
            return
        previous = next(
            (state for state in self.store.extension_states() if state.extension_id == extension_id),
            None,
        )
        self.store.put_extension_state(ExtensionState(
            extension_id=extension_id,
            installed_version=spec.version,
            engine_api=spec.engine_api,
            storage_revision=spec.storage_revision,
            last_projected_seq=previous.last_projected_seq if previous else 0,
            status="degraded",
            last_error=error[:1000],
        ))

    def _record_extension_success(
        self, capability: str, name: str, evidence_version: int
    ) -> None:
        owner = self.extensions.owner(capability, name)
        if owner is None:
            return
        spec = self.extensions.extensions[owner]
        self.store.put_extension_state(ExtensionState(
            extension_id=owner,
            installed_version=spec.version,
            engine_api=spec.engine_api,
            storage_revision=spec.storage_revision,
            last_projected_seq=evidence_version,
            status="available",
        ))

    def _sync_extension_state(self) -> None:
        existing = {state.extension_id: state for state in self.store.extension_states()}
        for extension_id, spec in self.extensions.extensions.items():
            previous = existing.get(extension_id)
            revision_changed = bool(
                previous
                and previous.storage_revision != spec.storage_revision
                and _extension_has_persistent_recall(self.extensions, extension_id)
            )
            self.store.put_extension_state(
                _extension_state_for_sync(extension_id, spec, previous, revision_changed)
            )
            if revision_changed:
                self.rebuild_extension(extension_id)

    def _sync_model_configuration(self, *, force: bool = False) -> None:
        configuration = self.store.get_model_configuration()
        if not force and configuration.revision == self._config_revision:
            return
        if self._managed_completion:
            self._sync_managed_completion(configuration)
        if self._managed_embedding:
            self._sync_managed_embedding(configuration)
        if not self._token_counter_override:
            self.token_counter = token_counter_for(self.completion)
        self._config_revision = configuration.revision

    def _sync_managed_completion(self, configuration: ModelConfiguration) -> None:
        try:
            self.completion = completion_from_settings(
                settings_from_selection(configuration.completion)
            )
            descriptor = self.store.find_cached_model(
                configuration.completion.provider,
                configuration.completion.model or "",
            )
            if self.completion is not None and descriptor is not None:
                self.completion.vision_supported = descriptor.vision
            self._provider_errors.pop("completion", None)
        except ProviderError as exc:
            self.completion = None
            self._provider_errors["completion"] = str(exc)

    def _sync_managed_embedding(self, configuration: ModelConfiguration) -> None:
        try:
            self.embedder = embedder_from_settings(
                settings_from_selection(configuration.embedding)
            )
            self._provider_errors.pop("embedding", None)
        except ProviderError as exc:
            selection = configuration.embedding
            self.embedder = UnavailableEmbedder(
                selection.model or "unknown",
                embedding_cache_key(
                    selection.provider, selection.base_url, selection.model or "unknown"
                ),
                str(exc),
            )
            self._provider_errors["embedding"] = str(exc)

    def _prewarm_embedding_cache(self, embedder: Embedder) -> None:
        deltas = self.store.list_records(
            kinds=["raw_source", "fact", "note"],
            include_retracted=False,
        )
        unique = {
            hashlib.sha256(text.encode("utf-8")).hexdigest(): text
            for text in _embedding_prewarm_texts(self.store, deltas)
        }
        cache_key = str(getattr(embedder, "cache_key", embedder.model_name))
        existing = self.embedding_index.get(list(unique), cache_key)
        missing = [
            (text_hash, text)
            for text_hash, text in unique.items()
            if text_hash not in existing
        ]
        self._put_missing_embeddings(embedder, missing, cache_key)

    def _put_missing_embeddings(
        self,
        embedder: Embedder,
        missing: list[tuple[str, str]],
        cache_key: str,
    ) -> None:
        for offset in range(0, len(missing), 64):
            batch = missing[offset : offset + 64]
            vectors = embedder.encode([text for _, text in batch])
            if len(vectors) != len(batch):
                raise RuntimeError("embedding provider returned the wrong number of vectors")
            self.embedding_index.put(
                {text_hash: vector for (text_hash, _), vector in zip(batch, vectors)},
                cache_key,
            )

    def _resolve_namespace(
        self,
        namespace: MemoryNamespace | dict[str, Any] | None,
        *,
        session_id: str | None = None,
    ) -> MemoryNamespace:
        value = _namespace(namespace) if namespace is not None else self.default_namespace.model_copy()
        if session_id is not None and value.session_id is None:
            value.session_id = session_id
        return value

    @staticmethod
    def _scope(namespace: MemoryNamespace, scope: dict[str, Any] | None) -> dict[str, Any]:
        return {**namespace.as_scope(), **(scope or {})}

    def _schedule_group(
        self,
        group_id: str,
        namespace: MemoryNamespace,
        *,
        force_extraction: bool = False,
        process: bool | None = None,
    ) -> None:
        payload = {"group_id": group_id, "namespace": namespace.model_dump(mode="json")}
        namespace_key = _namespace_key(namespace)
        evidence_version = self.store.max_seq()
        if self.completion is not None or force_extraction:
            self.store.enqueue_job(
                "extract_facts", payload, dedup_key=f"extract:{namespace_key}:{group_id}:{evidence_version}"
            )
        self.store.enqueue_job(
            "summarize_group", payload, dedup_key=f"compile:{namespace_key}:{group_id}:{evidence_version}"
        )
        event = IngestEvent(
            name="group_committed",
            group_id=group_id,
            namespace=namespace,
            evidence_version=evidence_version,
        )
        for processor in self.extensions.ingest_processors.scheduled(event):
            self.store.enqueue_job(
                f"ingest:{processor.spec.name}",
                {**payload, "processor": processor.spec.name, "evidence_version": evidence_version},
                dedup_key=f"ingest:{processor.spec.name}:{namespace_key}:{group_id}:{evidence_version}",
                max_attempts=processor.spec.max_attempts,
            )
        for channel in self.extensions.index_channels.channels:
            if not callable(getattr(channel, "project", None)):
                continue
            self.store.enqueue_job(
                f"index:{channel.spec.id}:project",
                {**payload, "channel_id": channel.spec.id, "evidence_version": evidence_version},
                dedup_key=f"index:{channel.spec.id}:{namespace_key}:{group_id}:{evidence_version}",
            )
        should_process = process if process is not None else self.auto_process
        if should_process:
            self.run_jobs()

def _build_extension_registry(
    extensions: Iterable[MemoryExtension],
    *,
    legacy_route_names: Iterable[str] = (),
) -> EngineExtensionRegistry:
    registry = EngineExtensionRegistry()
    for extension in extensions:
        registry.register_extension(extension)
    return _finalize_extension_registry(registry, legacy_route_names)


def _finalize_extension_registry(
    registry: EngineExtensionRegistry,
    legacy_route_names: Iterable[str],
) -> EngineExtensionRegistry:
    if registry.frozen:
        return registry
    _ensure_builtin_source_handler(registry)
    _register_builtin_job_handlers(registry)
    _register_ingest_job_handlers(registry)
    _register_index_job_handlers(registry)
    _register_extension_rebuild_handlers(registry)
    registry.validate_and_freeze(legacy_route_names)
    return registry


def _ensure_builtin_source_handler(registry: EngineExtensionRegistry) -> None:
    if "builtin.sources" not in registry.source_handlers.names:
        registry.source_handlers.register(BuiltinSourceHandler())


def _register_builtin_job_handlers(registry: EngineExtensionRegistry) -> None:
    for kind, handler in (
        ("extract_facts", _job_extract_facts),
        ("summarize_group", _job_summarize_group),
        ("rebuild_embeddings", _job_rebuild_embeddings),
        ("execute_query", _job_execute_query),
    ):
        if kind not in registry.job_handlers.names:
            registry.job_handlers.register(handler, kind=kind)


def _register_ingest_job_handlers(registry: EngineExtensionRegistry) -> None:
    for name in registry.ingest_processors.names:
        kind = f"ingest:{name}"
        if kind not in registry.job_handlers.names:
            registry.job_handlers.register(_job_ingest_processor, kind=kind)


def _register_index_job_handlers(registry: EngineExtensionRegistry) -> None:
    for channel_id in registry.index_channels.names:
        channel = registry.index_channels.get(channel_id)
        if channel is None or not callable(getattr(channel, "project", None)):
            continue
        kind = f"index:{channel_id}:project"
        if kind not in registry.job_handlers.names:
            registry.job_handlers.register(_job_index_projection, kind=kind)


def _register_extension_rebuild_handlers(registry: EngineExtensionRegistry) -> None:
    for extension_id in registry.extensions:
        if not _extension_has_persistent_recall(registry, extension_id):
            continue
        registry.job_handlers.register(
            _job_extension_rebuild,
            kind=f"extension:{extension_id}:rebuild",
        )


def _job_extract_facts(engine: MemoryEngine, job: MemoryJob) -> None:
    if engine.completion is None:
        raise RuntimeError("no completion provider is configured")
    namespace = MemoryNamespace.model_validate(job.payload.get("namespace") or {})
    engine.run_fact_extraction(
        group_id=str(job.payload.get("group_id") or ""), namespace=namespace
    )


def _job_summarize_group(engine: MemoryEngine, job: MemoryJob) -> None:
    namespace = MemoryNamespace.model_validate(job.payload.get("namespace") or {})
    engine.build_group_summary(
        [str(job.payload.get("group_id") or "")], namespace=namespace
    )


def _job_rebuild_embeddings(engine: MemoryEngine, job: MemoryJob) -> None:
    selection = ProviderSelection.model_validate(job.payload["embedding"])
    target_embedder = embedder_from_settings(settings_from_selection(selection))
    engine._prewarm_embedding_cache(target_embedder)
    engine.store.activate_pending_embedding(job.id)
    engine._sync_model_configuration(force=True)


def _job_execute_query(engine: MemoryEngine, job: MemoryJob) -> None:
    engine.execute_query_run(str(job.payload["query_id"]))


def _job_namespace(job: MemoryJob) -> MemoryNamespace:
    return MemoryNamespace.model_validate(job.payload.get("namespace") or {})


def _job_evidence_version(engine: MemoryEngine, job: MemoryJob) -> int:
    return int(job.payload.get("evidence_version") or engine.store.max_seq())


def _job_ingest_processor(engine: MemoryEngine, job: MemoryJob) -> None:
    name = str(job.payload.get("processor") or job.kind.removeprefix("ingest:"))
    processor = engine.extensions.ingest_processors.get(name)
    if processor is None:
        raise RuntimeError(f"ingest processor is unavailable: {name}")
    namespace = _job_namespace(job)
    group_id = str(job.payload.get("group_id") or "")
    evidence = engine.store.list_records(
        kinds=processor.spec.input_kinds,
        namespace=namespace,
        group_prefix=group_id,
        include_retracted=False,
    )
    proposed = processor.process(evidence, IngestContext(
        namespace=namespace,
        group_id=group_id,
        evidence_version=_job_evidence_version(engine, job),
        completion=engine.completion,
    ))
    available = {item.id: item for item in evidence}
    for item in proposed:
        _append_proposed_ingest_record(
            engine, name, processor, item, available, group_id, namespace
        )
    engine._record_extension_success(
        "ingest", name, _job_evidence_version(engine, job)
    )


def _job_index_projection(engine: MemoryEngine, job: MemoryJob) -> None:
    channel_id = str(job.payload.get("channel_id") or "")
    channel = engine.extensions.index_channels.get(channel_id)
    if channel is None:
        raise RuntimeError(f"index channel is unavailable: {channel_id}")
    namespace = MemoryNamespace.model_validate(job.payload.get("namespace") or {})
    group_id = str(job.payload.get("group_id") or "")
    evidence_version = int(job.payload.get("evidence_version") or engine.store.max_seq())
    deltas = engine.store.list_records(
        namespace=namespace,
        group_prefix=group_id or None,
        seq_cutoff=evidence_version,
        include_retracted=False,
    )
    batch = MemoryLogChangeBatch(
        records=tuple(deltas),
        namespace=namespace,
        evidence_version=evidence_version,
        group_id=group_id or None,
    )
    writer = IndexWriter(engine.store, channel.spec, namespace)
    channel.project(batch, writer)
    engine._record_extension_success("index", channel_id, evidence_version)


def _job_extension_rebuild(engine: MemoryEngine, job: MemoryJob) -> None:
    extension_id = str(job.payload.get("extension_id") or "")
    spec = engine.extensions.extensions.get(extension_id)
    if spec is None:
        raise RuntimeError(f"extension is unavailable: {extension_id}")
    evidence_version = int(job.payload.get("evidence_version") or engine.store.max_seq())
    channels = [
        channel
        for channel in engine.extensions.index_channels.channels
        if channel.spec.persistent
        and engine.extensions.owner("index", channel.spec.id) == extension_id
    ]
    namespaces = engine.store.namespaces() or [engine.default_namespace]
    for namespace in namespaces:
        reader = MemoryLogReader(engine.store, namespace, seq_cutoff=evidence_version)
        for channel in channels:
            channel.rebuild(reader, IndexWriter(engine.store, channel.spec, namespace))
    engine.store.put_extension_state(ExtensionState(
        extension_id=extension_id,
        installed_version=spec.version,
        engine_api=spec.engine_api,
        storage_revision=spec.storage_revision,
        last_projected_seq=evidence_version,
        status="available",
    ))


def _answer_prompt(question: str, hits: list[Any]) -> str:
    query_kind = classify_query(question)
    policies = {
        "fact": "Return the shortest directly supported factual answer.",
        "temporal": "Resolve the event time from observed_at and temporal anchors; prefer an absolute date.",
        "list": "Aggregate and deduplicate every supported item across the supplied records.",
        "inference": "Combine all related records and make the best supported judgment; mention uncertainty only when evidence conflicts.",
        "multi_hop": "Connect the relevant records step by step, then return the concise conclusion supported by that chain.",
    }
    lines = [
        (
            f"[{hit.kind} id={hit.item_id} locator={hit.locator or 'unknown'}] "
            f"({hit.temporal_anchor or hit.observed_at or 'unknown date'}) {hit.text}"
        )
        for hit in hits
        if hit.kind in {"raw_source", "fact"}
    ]
    return (
        f"Question type: {query_kind}\n"
        f"Answer policy: {policies[query_kind]}\n"
        f"Question: {question}\n\nTrace context:\n" + "\n".join(lines)
    )


_CHUNK_KIND_MODALITY = {
    "code_symbol": "code",
    "code_lines": "code",
    "row": "table",
    "cell": "table",
    "image": "image",
    "message": "conversation",
    "message_index": "conversation",
    "dialog": "conversation",
}


def _source_retrieval_descriptor(
    prepared: PreparedSource,
    chunk: PreparedChunk,
) -> tuple[str, str, dict[str, Any]]:
    """Map every accepted source to generic modality and searchable fields."""

    if chunk.modality:
        return (
            chunk.modality,
            chunk.source_type or prepared.kind,
            dict(chunk.retrieval_fields),
        )
    locator = chunk.locator
    source_type = str(
        chunk.metadata.get("source_type")
        or prepared.metadata.get("source_type")
        or prepared.kind
    )
    modality = _chunk_modality(str(locator.get("kind") or ""), source_type, prepared, locator)
    return modality, source_type, _chunk_retrieval_fields(prepared, chunk, locator, modality)


def _retriever_config(value: RetrievalConfiguration) -> RetrieverConfig:
    return RetrieverConfig(
        top_k=value.default_top_k,
        max_context_items=value.max_context_items,
        max_context_tokens=value.max_context_tokens,
        per_source_context_tokens=value.per_source_context_tokens,
        graph_hops=value.graph_hops,
        margin_threshold=value.confidence_margin,
        confidence_floor=value.confidence_floor,
        max_refinement_rounds=value.max_refinement_rounds,
        deep_summary_enabled=value.deep_summary_enabled,
        claims_enabled=value.claims_enabled,
        group_summaries_enabled=value.group_summaries_enabled,
        reranker_enabled=value.reranker_enabled,
        raw_candidates_only=value.raw_candidates_only,
        readout=value.readout,
        fusion=value.fusion,
        rrf_rank_constant=value.rrf_rank_constant,
        retrieval_profile=value.retrieval_profile,
        enabled_routes=tuple(value.enabled_routes),
        route_weights=dict(value.route_weights),
    )


def _extractive_answer(question: str, hits: list[Any]) -> str:
    kind = classify_query(question)
    ranked_hits = list(hits)
    if kind == "temporal":
        date_pattern = re.compile(
            r"\b(?:19|20)\d{2}\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
            re.I,
        )
        ranked_hits.sort(key=lambda hit: (bool(hit.temporal_anchor or date_pattern.search(hit.text)), hit.score), reverse=True)
    else:
        query_terms = set(re.findall(r"[a-z0-9']+", question.lower()))
        ranked_hits.sort(
            key=lambda hit: (len(query_terms & set(re.findall(r"[a-z0-9']+", hit.text.lower()))), hit.score),
            reverse=True,
        )
    texts = list(dict.fromkeys(hit.text.strip() for hit in ranked_hits if hit.text.strip()))
    if not texts:
        return ""
    if kind == "list":
        return "\n".join(f"- {text}" for text in texts[:12])
    if kind == "inference" and len(texts) > 1:
        return " ".join(texts[:3])
    return texts[0]


def _citations(hits: list[Any]) -> list[Citation]:
    result: list[Citation] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.item_id in seen or hit.kind not in {"raw_source", "fact"}:
            continue
        seen.add(hit.item_id)
        result.append(
            Citation(
                record_id=hit.item_id,
                source_ref=hit.source_ref,
                locator=hit.locator,
                excerpt=hit.text[:500],
                observed_at=hit.observed_at,
            )
        )
    return result


def _graphml(graph: dict[str, Any]) -> str:
    def esc(value: Any) -> str:
        return str(value).replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
    nodes = "".join(f'<node id="{esc(item["id"])}"/>' for item in graph["nodes"])
    edges = "".join(
        f'<edge source="{esc(item["source"])}" target="{esc(item["target"])}"><data key="relation">{esc(item.get("relation") or "related")}</data></edge>'
        for item in graph["edges"]
    )
    return f'<?xml version="1.0"?><graphml xmlns="http://graphml.graphdrawing.org/xmlns"><graph edgedefault="directed">{nodes}{edges}</graph></graphml>'


def _namespace(value: MemoryNamespace | dict[str, Any] | None) -> MemoryNamespace:
    if isinstance(value, MemoryNamespace):
        return value.model_copy()
    return MemoryNamespace.model_validate(value or {})


def _namespace_key(namespace: MemoryNamespace) -> str:
    serialized = json.dumps(namespace.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _open_embedding_settings(
    embedding_provider: ProviderSettings | None,
    embedding_model: str,
    configuration: ModelConfiguration,
) -> ProviderSettings:
    if embedding_provider is not None:
        return embedding_provider
    if embedding_model != "all-MiniLM-L6-v2":
        return ProviderSettings(provider="sentence-transformers", model=embedding_model)
    return settings_from_selection(configuration.embedding)


def _unavailable_embedder(configuration: ModelConfiguration, error: str) -> UnavailableEmbedder:
    selection = configuration.embedding
    model = selection.model or "unknown"
    return UnavailableEmbedder(
        model,
        embedding_cache_key(selection.provider, selection.base_url, model),
        error,
    )


def _resolve_open_embedder(
    embedder: Embedder | None,
    embedding_provider: ProviderSettings | None,
    embedding_model: str,
    configuration: ModelConfiguration,
    local_files_only: bool,
    provider_errors: dict[str, str],
) -> Embedder:
    if embedder is not None:
        return embedder
    try:
        provider = embedder_from_settings(
            _open_embedding_settings(embedding_provider, embedding_model, configuration)
        )
        if isinstance(provider, SentenceTransformerEmbedder):
            provider.local_files_only = local_files_only
        return provider
    except ProviderError as exc:
        provider_errors["embedding"] = str(exc)
        return _unavailable_embedder(configuration, str(exc))


def _resolve_open_completion(
    completion: CompletionJSON | None,
    completion_provider: ProviderSettings | None,
    configuration: ModelConfiguration,
    provider_errors: dict[str, str],
) -> CompletionJSON | None:
    if completion is not None:
        return completion
    completion_settings = completion_provider or settings_from_selection(configuration.completion)
    try:
        return completion_from_settings(completion_settings)
    except ProviderError as exc:
        provider_errors["completion"] = str(exc)
        return None


def _resolve_open_registry(
    extensions: Iterable[MemoryExtension],
    extension_registry: EngineExtensionRegistry | None,
    retrieval_routes: RouteRegistry | None,
) -> EngineExtensionRegistry:
    legacy_names = retrieval_routes.names if retrieval_routes else RouteRegistry().names
    if extension_registry is None:
        return _build_extension_registry(extensions, legacy_route_names=legacy_names)
    if tuple(extensions):
        raise ValueError("pass extensions or extension_registry, not both")
    return _finalize_extension_registry(extension_registry, legacy_names)


def _coerce_model_configuration(
    configuration: ModelConfiguration | dict[str, Any],
) -> ModelConfiguration:
    if isinstance(configuration, ModelConfiguration):
        return configuration
    return ModelConfiguration.model_validate(configuration)


def _model_selection_warnings(
    store: SQLiteMemoryLogStore, requested: ModelConfiguration
) -> list[str]:
    warnings: list[str] = []
    for role, selection in (
        (ProviderRole.COMPLETION, requested.completion),
        (ProviderRole.EMBEDDING, requested.embedding),
    ):
        warning = _selection_warning(store, role, selection)
        if warning:
            warnings.append(warning)
    return warnings


def _selection_warning(
    store: SQLiteMemoryLogStore, role: ProviderRole, selection: ProviderSelection
) -> str | None:
    if selection.provider == "none":
        return None
    descriptor = store.find_cached_model(selection.provider, selection.model or "")
    if descriptor is not None and role not in descriptor.roles:
        raise ProviderError(
            f"model {selection.model} is not cataloged for {role.value}"
        )
    if descriptor is not None:
        return None
    if selection.provider == "sentence-transformers" and selection.model == "all-MiniLM-L6-v2":
        return None
    return (
        f"{selection.provider}/{selection.model} was accepted as an unverified custom model ID"
    )


def _coerce_connection_selection(
    role: ProviderRole,
    current: ModelConfiguration,
    selection: ProviderSelection | dict[str, Any] | None,
) -> ProviderSelection:
    if selection is None:
        if role == ProviderRole.COMPLETION:
            return current.completion
        return current.embedding
    if isinstance(selection, ProviderSelection):
        return selection
    return ProviderSelection.model_validate(selection)


def _probe_completion(selected: ProviderSelection) -> bool:
    completion = completion_from_settings(settings_from_selection(selected))
    assert completion is not None
    completion.complete_json(
        "Return only JSON.",
        'Reply with exactly {"status":"ok"}.',
    )
    return hasattr(completion, "complete_multimodal")


def _probe_embedding(selected: ProviderSelection) -> None:
    embedder = embedder_from_settings(settings_from_selection(selected))
    vectors = embedder.encode(["Trisynapse connection test"])
    if len(vectors) != 1 or not vectors[0]:
        raise ProviderError("embedding provider returned an empty vector")


def _missing_provider_credentials(
    providers: dict[str, ProviderDescriptor],
    completion: ProviderSelection,
    embedding: ProviderSelection,
) -> list[str]:
    required = [
        item for item in (completion, embedding)
        if item.provider not in {"none", "sentence-transformers", "openai-compatible"}
    ]
    return [
        providers[item.provider].credential_env for item in required
        if not providers[item.provider].credential_configured
    ]


def _source_extra_availability() -> dict[str, bool]:
    import importlib.util

    return {
        name: importlib.util.find_spec(module) is not None
        for name, module in {
            "pdf": "pypdf", "office": "docx", "spreadsheets": "openpyxl",
            "tree_sitter": "tree_sitter_language_pack", "tokens": "tiktoken",
        }.items()
    }


def _port_8765_free() -> bool:
    import socket

    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", 8765))
        except OSError:
            return False
    return True


def _enforce_retained_source_limit(
    inputs: Sequence[SourceInput],
    prepared: dict[int, PreparedSource],
    failures: dict[int, str],
) -> None:
    retained_bytes = 0
    for index in range(len(inputs)):
        if index not in prepared:
            continue
        candidate_bytes = retained_bytes + len(prepared[index].original)
        if candidate_bytes <= 250 * 1024 * 1024:
            retained_bytes = candidate_bytes
            continue
        failures[index] = "ingestion run exceeds the 250 MiB retained-source limit"
        del prepared[index]


def _ingestion_run_status(results: list[SourceIngestionResult]) -> str:
    succeeded = sum(item.status in {"success", "skipped"} for item in results)
    failed = sum(item.status == "failed" for item in results)
    if failed and not succeeded:
        return "failed"
    if failed:
        return "partial"
    return "completed"


def _prepared_source_key(prepared: PreparedSource) -> str:
    return (
        prepared.source.source_key
        or prepared.uri
        or f"upload:{prepared.filename}:{prepared.content_hash}"
    )


def _is_unchanged_source(
    previous: SourceRecord | None, prepared: PreparedSource
) -> bool:
    return (
        previous is not None
        and previous.status == "active"
        and previous.content_hash == prepared.content_hash
    )


def _source_identity_id(
    namespace: MemoryNamespace, source_key: str, content_hash: str
) -> str:
    identity = json.dumps(
        {
            "namespace": namespace.model_dump(mode="json", exclude_none=True),
            "source_key": source_key,
            "content_hash": content_hash,
        },
        sort_keys=True,
    )
    return f"src_{hashlib.sha256(identity.encode()).hexdigest()[:20]}"


def _source_search_blob(item: SourceRecord) -> str:
    return " ".join(
        filter(None, [item.title, item.source_key, item.uri, item.filename])
    ).lower()


def _filter_sources(
    values: list[SourceRecord],
    *,
    search: str | None,
    kinds: Iterable[str] | None,
    status: str | None,
) -> list[SourceRecord]:
    values = _filter_sources_by_search(values, search)
    values = _filter_sources_by_kind(values, kinds)
    return _filter_sources_by_status(values, status)


def _filter_sources_by_search(
    values: list[SourceRecord], search: str | None
) -> list[SourceRecord]:
    if not search:
        return values
    term = search.lower()
    return [item for item in values if term in _source_search_blob(item)]


def _filter_sources_by_kind(
    values: list[SourceRecord], kinds: Iterable[str] | None
) -> list[SourceRecord]:
    allowed = set(kinds or [])
    if not allowed:
        return values
    return [item for item in values if item.kind in allowed]


def _filter_sources_by_status(
    values: list[SourceRecord], status: str | None
) -> list[SourceRecord]:
    if not status:
        return values
    return [item for item in values if item.status == status]


def _sort_sources(values: list[SourceRecord], sort: str) -> None:
    if sort == "oldest":
        values.sort(key=lambda item: item.created_at)
        return
    if sort == "title":
        values.sort(key=lambda item: item.title.lower())
        return
    if sort != "newest":
        raise ValueError("sort must be newest, oldest, or title")


def _derived_source_filename(
    store: SQLiteMemoryLogStore, source: SourceRecord
) -> str | None:
    if source.filename:
        return None
    derived = source.metadata.get("filename")
    if derived:
        return str(derived)
    for delta_id in source.record_ids:
        delta = store.get(delta_id)
        if delta is None or not isinstance(delta.locator, dict):
            continue
        path = delta.locator.get("path")
        if path:
            return Path(str(path)).name
    return str(source.title)


def _source_ingestion_run_id(
    store: SQLiteMemoryLogStore, source: SourceRecord
) -> str | None:
    if source.ingestion_run_id:
        return None
    for run in store.list_ingestion_runs(source.namespace, limit=10_000):
        if any(result.source_id == source.id for result in run.results):
            return run.id
    return None


def _source_size_updates(source: SourceRecord, path: Path) -> dict[str, Any]:
    update: dict[str, Any] = {}
    if not source.byte_size and path.is_file():
        update["byte_size"] = path.stat().st_size
    if not source.chunk_count:
        update["chunk_count"] = len(source.record_ids)
    preview = _derived_preview_type(source)
    if preview:
        update["preview_type"] = preview
    skipped = _derived_skipped_count(source)
    if skipped:
        update["skipped_count"] = skipped
    return update


def _derived_preview_type(source: SourceRecord) -> str | None:
    if source.preview_type:
        return None
    return str(source.metadata.get("source_type") or source.kind)


def _derived_skipped_count(source: SourceRecord) -> int | None:
    if source.skipped_count:
        return None
    skipped_paths = source.metadata.get("skipped_paths") or []
    if not skipped_paths:
        return None
    return len(skipped_paths)


def _source_hydrate_updates(
    store: SQLiteMemoryLogStore, source: SourceRecord
) -> dict[str, Any]:
    update: dict[str, Any] = {}
    filename = _derived_source_filename(store, source)
    if filename is not None:
        update["filename"] = filename
    update.update(_source_size_updates(source, store.root / source.blob_path))
    run_id = _source_ingestion_run_id(store, source)
    if run_id:
        update["ingestion_run_id"] = run_id
    return update


def _source_preview_items(
    store: SQLiteMemoryLogStore, record_ids: list[str]
) -> tuple[list[SourcePreviewItem], list[str]]:
    values: list[SourcePreviewItem] = []
    manifest: list[str] = []
    for delta_id in record_ids:
        item = _preview_item_from_delta(store.get(delta_id))
        if item is None:
            continue
        preview, item_manifest = item
        if item_manifest is not None:
            manifest = item_manifest
        values.append(preview)
    return values, manifest


def _preview_item_from_delta(
    delta: MemoryRecord | None,
) -> tuple[SourcePreviewItem, list[str] | None] | None:
    if delta is None:
        return None
    locator = delta.locator
    locator_metadata = locator.get("metadata", {}) if isinstance(locator, dict) else {}
    kind = (
        str(locator.get("kind") or delta.kind)
        if isinstance(locator, dict) else delta.kind
    )
    manifest = None
    if kind == "manifest":
        manifest = [line for line in delta.text.splitlines()[1:] if line.strip()]
    return SourcePreviewItem(
        record_id=delta.id,
        kind=kind,
        text=delta.text,
        locator=locator,
        metadata=locator_metadata,
    ), manifest


def _group_summary_plan(
    store: SQLiteMemoryLogStore,
    namespace: MemoryNamespace,
    deltas: list[MemoryRecord],
    group_ids: list[str] | None,
) -> tuple[dict[str, GroupSummaryView], list[str], list[str]]:
    requested = list(group_ids or [])
    existing = _existing_group_summaries(store, namespace, requested)
    targets = requested or _group_summary_targets(deltas)
    pending_ids = [group_id for group_id in targets if group_id not in existing]
    return existing, targets, pending_ids


def _group_summary_progress(
    on_progress: Callable[[int, int], None] | None,
    already: int,
    total: int,
) -> Callable[[int, int], None] | None:
    if on_progress is None:
        return None
    return lambda done, count: _report_group_summary_progress(
        on_progress, already, total, done, count
    )


def _merge_group_summaries(
    targets: list[str],
    existing: dict[str, GroupSummaryView],
    views: list[GroupSummaryView],
) -> list[GroupSummaryView]:
    return [existing[group_id] for group_id in targets if group_id in existing] + views


def _existing_group_summaries(
    store: SQLiteMemoryLogStore,
    namespace: MemoryNamespace,
    requested: list[str],
) -> dict[str, GroupSummaryView]:
    return {
        view.group_id: view
        for view in store.group_summary_views(namespace=namespace, include_stale=False)
        if not requested or view.group_id in requested
    }


def _group_summary_targets(deltas: list[MemoryRecord]) -> list[str]:
    return list(dict.fromkeys(
        item.group_id
        for item in deltas
        if item.group_id and item.kind in {"raw_source", "fact", "note"}
    ))


def _cached_group_summaries(
    targets: list[str],
    existing: dict[str, GroupSummaryView],
    on_progress: Callable[[int, int], None] | None,
) -> list[GroupSummaryView]:
    if on_progress is not None:
        on_progress(len(targets), len(targets))
    return [existing[group_id] for group_id in targets if group_id in existing]


def _persist_group_summary(
    store: SQLiteMemoryLogStore, view: GroupSummaryView, namespace: MemoryNamespace
) -> None:
    view.namespace = namespace
    view.cache_key = f"{view.cache_key}:{_namespace_key(namespace)}"
    store.put_group_summary(view)


def _report_group_summary_progress(
    on_progress: Callable[[int, int], None],
    already: int,
    total: int,
    done: int,
    count: int,
) -> None:
    del count
    on_progress(already + done, total)


def _query_context_top_k(top_k: int | None, run: QueryRun) -> int:
    if top_k is not None:
        return top_k
    return run.retrieval_configuration.max_context_items


def _query_abstain_threshold(run: QueryRun, abstain_threshold: float | None) -> float:
    if abstain_threshold is None:
        return run.retrieval_configuration.answer_abstain_threshold
    return abstain_threshold


def _hits_are_relevant(hits: list[Any], threshold: float) -> bool:
    return bool(hits) and hits[0].score >= threshold


def _query_generation_provenance(completion: CompletionJSON | None) -> dict[str, Any]:
    prompt = None
    if completion is not None:
        prompt = load_prompt("answer").provenance()
    return {**provider_provenance(completion, kind="completion"), "prompt": prompt}


def _query_answer_label(relevant: bool) -> str:
    if relevant:
        return "Generate grounded answer"
    return "Abstain without enough evidence"


def _query_answer(
    completion: CompletionJSON | None,
    question: str,
    hits: list[Any],
    relevant: bool,
) -> tuple[str, bool, list[Any]]:
    if completion is not None and relevant:
        return _completion_query_answer(completion, question, hits)
    if relevant:
        answer = _extractive_answer(question, hits)
        return answer, not answer, hits
    return "I don't have enough grounded evidence to answer that.", True, hits


def _completion_query_answer(
    completion: CompletionJSON,
    question: str,
    hits: list[Any],
) -> tuple[str, bool, list[Any]]:
    payload = complete_json_parallel(
        completion,
        [(load_prompt("answer").text, _answer_prompt(question, hits))],
        on_error=_skippable_answer_payload,
    )[0]
    answer, abstain = _answer_from_payload(payload)
    cited_hits = _cited_hits_from_payload(payload, hits)
    if payload.get("citation_ids") is None:
        return answer, abstain, cited_hits
    if not answer or abstain or cited_hits:
        return answer, abstain, cited_hits
    return "I don't have enough grounded evidence to answer that.", True, cited_hits


def _skippable_answer_payload(_index: int, exc: BaseException) -> dict[str, Any] | None:
    if not isinstance(exc, Exception) or not skippable_completion_error(exc):
        return None
    return {
        "answer": "I don't have enough grounded evidence to answer that.",
        "abstain": True,
    }


def _answer_from_payload(payload: dict[str, Any]) -> tuple[str, bool]:
    raw_answer = payload.get("answer")
    answer = str(raw_answer).strip() if raw_answer else ""
    abstain = bool(payload.get("abstain", False)) or not answer
    return answer, abstain


def _cited_hits_from_payload(payload: dict[str, Any], hits: list[Any]) -> list[Any]:
    requested = payload.get("citation_ids")
    if requested is None:
        return hits
    if not isinstance(requested, list):
        raise ValueError("answer citation_ids must be a list")
    requested_ids = {str(value) for value in requested}
    return [hit for hit in hits if hit.item_id in requested_ids]


def _record_group_info(
    group_id: str,
    deltas: list[MemoryRecord],
    fresh: dict[str, bool],
) -> RecordGroupInfo:
    observed = [item.observed_at for item in deltas if item.observed_at]
    return RecordGroupInfo(
        group_id=group_id,
        record_count=len(deltas),
        raw_source_count=sum(item.kind == "raw_source" for item in deltas),
        fact_count=sum(item.kind == "fact" for item in deltas),
        first_observed_at=min(observed) if observed else None,
        last_observed_at=max(observed) if observed else None,
        stale=not fresh.get(group_id, False),
    )


def _namespace_claims(
    store: SQLiteMemoryLogStore,
    namespace: MemoryNamespace,
    scope: dict[str, Any],
) -> list[Claim]:
    deltas = store.list_records(
        kinds=["raw_source", "fact"],
        scope=scope,
        namespace=namespace,
    )
    observations = {item.id: item for item in deltas if item.kind == "raw_source"}
    return compile_claims([item for item in deltas if item.kind == "fact"], observations)


def _claim_export_graph(claims: list[Claim]) -> dict[str, Any]:
    nodes: list[dict[str, str]] = []
    edges: list[dict[str, Any]] = []
    seen: set[str] = set()
    for claim in claims:
        _append_claim_export_nodes(nodes, seen, claim)
        if not claim.subject or not claim.object:
            continue
        edges.append({
            "source": claim.subject,
            "target": claim.object,
            "relation": claim.relation,
            "confidence": claim.confidence,
            "status": claim.status,
        })
    return {"nodes": nodes, "edges": edges}


def _append_claim_export_nodes(
    nodes: list[dict[str, str]], seen: set[str], claim: Claim
) -> None:
    for label in (claim.subject, claim.object):
        if not label or label in seen:
            continue
        seen.add(label)
        nodes.append({"id": label, "label": label})


def _validate_memory_graph_args(view: str, limit: int) -> None:
    if view not in {"knowledge", "lineage", "log", "retrieval"}:
        raise ValueError("view must be knowledge, lineage, log, or retrieval")
    if limit < 1 or limit > 2000:
        raise ValueError("limit must be between 1 and 2000")


def _filter_graph_deltas(
    deltas: list[MemoryRecord],
    *,
    group_id: str | None,
    source_id: str | None,
) -> list[MemoryRecord]:
    if group_id:
        deltas = [item for item in deltas if item.group_id == group_id]
    if source_id:
        deltas = [item for item in deltas if item.scope.get("source_id") == source_id]
    return deltas


def _memory_graph_claims(deltas: list[MemoryRecord]) -> list[Claim]:
    observations = {item.id: item for item in deltas if item.kind == "raw_source"}
    return compile_claims([item for item in deltas if item.kind == "fact"], observations)


def _graph_add_node(nodes: dict[str, MemoryGraphNode], node: MemoryGraphNode) -> None:
    nodes.setdefault(node.id, node)


def _graph_add_edge(edges: dict[str, MemoryGraphEdge], edge: MemoryGraphEdge) -> None:
    edges.setdefault(edge.id, edge)


def _add_source_lineage_nodes(
    nodes: dict[str, MemoryGraphNode],
    edges: dict[str, MemoryGraphEdge],
    sources: list[SourceRecord],
    source_id: str | None,
) -> None:
    if source_id:
        sources = [item for item in sources if item.id == source_id]
    for source in sources:
        _graph_add_node(nodes, MemoryGraphNode(
            id=source.id, type="source", label=source.title,
            subtitle=source.kind, status=source.status,
            data={"source_key": source.source_key, "version": source.version, "media_type": source.media_type},
        ))
        if not source.previous_source_id:
            continue
        _graph_add_edge(edges, MemoryGraphEdge(
            id=f"source-version:{source.id}:{source.previous_source_id}",
            source=source.id, target=source.previous_source_id,
            type="supersedes", label="supersedes",
        ))


def _add_log_delta_nodes(
    view: str,
    nodes: dict[str, MemoryGraphNode],
    edges: dict[str, MemoryGraphEdge],
    deltas: list[MemoryRecord],
    retracted_ids: set[str],
) -> None:
    previous_by_group: dict[str, str] = {}
    for delta in deltas:
        _graph_add_node(nodes, MemoryGraphNode(
            id=delta.id, type="log", label=(delta.text or delta.kind)[:90],
            subtitle=delta.kind, status="retracted" if delta.id in retracted_ids else "active",
            data={
                "kind": delta.kind, "seq": delta.seq, "group_id": delta.group_id,
                "observed_at": delta.observed_at.isoformat() if delta.observed_at else None,
                "source_ref": delta.source_ref, "locator": delta.locator,
            },
        ))
        _link_delta_to_source(edges, delta)
        _link_delta_to_group(view, nodes, edges, delta, previous_by_group)
        for evidence_id in delta.evidence_refs:
            _graph_add_edge(edges, MemoryGraphEdge(
                id=f"evidence:{evidence_id}:{delta.id}", source=evidence_id,
                target=delta.id, type="supports", label="supports",
            ))


def _link_delta_to_source(
    edges: dict[str, MemoryGraphEdge], delta: MemoryRecord
) -> None:
    current_source = str(delta.scope.get("source_id") or "")
    if not current_source:
        return
    _graph_add_edge(edges, MemoryGraphEdge(
        id=f"source-trace:{current_source}:{delta.id}", source=current_source,
        target=delta.id, type="produced", label="produced",
    ))


def _link_delta_to_group(
    view: str,
    nodes: dict[str, MemoryGraphNode],
    edges: dict[str, MemoryGraphEdge],
    delta: MemoryRecord,
    previous_by_group: dict[str, str],
) -> None:
    if not delta.group_id:
        return
    group_node = f"group:{delta.group_id}"
    _graph_add_node(nodes, MemoryGraphNode(
        id=group_node, type="group", label=delta.group_id,
        subtitle="record group",
    ))
    _graph_add_edge(edges, MemoryGraphEdge(
        id=f"episode-trace:{group_node}:{delta.id}", source=group_node,
        target=delta.id, type="contains", label="contains",
    ))
    previous = previous_by_group.get(delta.group_id)
    if view == "log" and previous:
        _graph_add_edge(edges, MemoryGraphEdge(
            id=f"follows:{previous}:{delta.id}", source=previous,
            target=delta.id, type="followed_by", label="then",
        ))
    previous_by_group[delta.group_id] = delta.id


def _add_knowledge_claim_graph(
    view: str,
    nodes: dict[str, MemoryGraphNode],
    edges: dict[str, MemoryGraphEdge],
    claims: list[Claim],
    summaries: list[GroupSummaryView],
) -> None:
    for claim in claims:
        _add_claim_graph_nodes(view, nodes, edges, claim)
    if view != "lineage":
        return
    for recall in summaries:
        _graph_add_node(nodes, MemoryGraphNode(
            id=recall.id, type="index", label=recall.concept_or_topic,
            subtitle="Batch summary", status="stale" if recall.stale else "fresh",
            data={"group_id": recall.group_id, "summary": recall.summary},
        ))
        for trace_id in recall.source_log_ids:
            _graph_add_edge(edges, MemoryGraphEdge(
                id=f"recall-ground:{trace_id}:{recall.id}", source=trace_id,
                target=recall.id, type="summarizes", label="summarizes",
            ))


def _add_claim_graph_nodes(
    view: str,
    nodes: dict[str, MemoryGraphNode],
    edges: dict[str, MemoryGraphEdge],
    claim: Claim,
) -> None:
    claim_id = f"claim:{claim.id}"
    _graph_add_node(nodes, MemoryGraphNode(
        id=claim_id, type="claim", label=claim.text[:100],
        subtitle=claim.relation, status=claim.status,
        data={"confidence": claim.confidence, "source_record_ids": claim.source_record_ids},
    ))
    _add_claim_endpoint(nodes, edges, claim_id, "subject", claim.subject, claim)
    _add_claim_endpoint(nodes, edges, claim_id, "object", claim.object, claim)
    if view != "lineage":
        return
    for evidence_id in dict.fromkeys(claim.source_record_ids + claim.raw_source_ids):
        _graph_add_edge(edges, MemoryGraphEdge(
            id=f"claim-ground:{evidence_id}:{claim_id}", source=evidence_id,
            target=claim_id, type="grounds", label="grounds",
        ))


def _add_claim_endpoint(
    nodes: dict[str, MemoryGraphNode],
    edges: dict[str, MemoryGraphEdge],
    claim_id: str,
    role: str,
    label: str | None,
    claim: Claim,
) -> None:
    if not label:
        return
    concept_id = f"concept:{hashlib.sha256(label.lower().encode()).hexdigest()[:16]}"
    _graph_add_node(nodes, MemoryGraphNode(id=concept_id, type="concept", label=label, subtitle=role))
    if role == "subject":
        _graph_add_edge(edges, MemoryGraphEdge(
            id=f"claim-subject:{concept_id}:{claim_id}", source=concept_id,
            target=claim_id, type="subject", label=claim.relation or "claim",
            weight=claim.confidence,
        ))
        return
    _graph_add_edge(edges, MemoryGraphEdge(
        id=f"claim-object:{claim_id}:{concept_id}", source=claim_id,
        target=concept_id, type="object", label=claim.relation or "relates to",
        weight=claim.confidence,
    ))


def _filter_graph_nodes(
    all_nodes: list[MemoryGraphNode],
    edges: dict[str, MemoryGraphEdge],
    search: str | None,
    node_types: Iterable[str] | None,
) -> list[MemoryGraphNode]:
    allowed = set(node_types or [])
    if allowed:
        all_nodes = [item for item in all_nodes if item.type in allowed]
    matched = _graph_search_ids(all_nodes, edges, search)
    if matched is None:
        return all_nodes
    return [item for item in all_nodes if item.id in matched]


def _graph_search_ids(
    all_nodes: list[MemoryGraphNode],
    edges: dict[str, MemoryGraphEdge],
    search: str | None,
) -> set[str] | None:
    if not search:
        return None
    term = search.lower()
    matched = {
        item.id for item in all_nodes
        if term in f"{item.label} {item.subtitle or ''}".lower()
    }
    return matched | _matched_neighbor_ids(edges, matched)


def _matched_neighbor_ids(
    edges: dict[str, MemoryGraphEdge], matched: set[str]
) -> set[str]:
    neighbor_ids: set[str] = set()
    for edge in edges.values():
        if edge.source not in matched and edge.target not in matched:
            continue
        neighbor_ids.update((edge.source, edge.target))
    return neighbor_ids


def _paginate_memory_graph(
    view: str,
    nodes: dict[str, MemoryGraphNode],
    edges: dict[str, MemoryGraphEdge],
    search: str | None,
    node_types: Iterable[str] | None,
    limit: int,
    cursor: str | None,
) -> MemoryGraphPage:
    all_nodes = _filter_graph_nodes(list(nodes.values()), edges, search, node_types)
    counts: dict[str, int] = defaultdict(int)
    for item in all_nodes:
        counts[item.type] += 1
    offset = int(cursor or 0)
    page_nodes = all_nodes[offset:offset + limit]
    page_ids = {item.id for item in page_nodes}
    page_edges = [
        edge for edge in edges.values()
        if edge.source in page_ids and edge.target in page_ids
    ]
    truncated = offset + limit < len(all_nodes)
    return MemoryGraphPage(
        view=view,
        nodes=page_nodes,
        edges=page_edges,
        counts=dict(counts),
        truncated=truncated,
        next_cursor=str(offset + limit) if truncated else None,
    )


def _neighbor_subgraph(
    graph: MemoryGraphPage, node_id: str, limit: int
) -> MemoryGraphPage:
    edges = _incident_edges(graph.edges, node_id, limit)
    ids = {node_id}
    for edge in edges:
        ids.update((edge.source, edge.target))
    nodes = [node for node in graph.nodes if node.id in ids]
    return MemoryGraphPage(
        view=graph.view,
        nodes=nodes,
        edges=edges,
        counts=_node_type_counts(nodes),
        truncated=len(graph.edges) > len(edges),
    )


def _incident_edges(
    edges: list[MemoryGraphEdge], node_id: str, limit: int
) -> list[MemoryGraphEdge]:
    return [edge for edge in edges if node_id in {edge.source, edge.target}][:limit]


def _node_type_counts(nodes: list[MemoryGraphNode]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for node in nodes:
        counts[node.type] += 1
    return dict(counts)


def _profile_from_claims(claims: list[Claim]) -> dict[str, Any]:
    ordered = sorted(claims, key=lambda item: item.confidence, reverse=True)
    return {
        "static": [item.text for item in ordered if item.confidence >= 0.7][:12],
        "dynamic": [item.text for item in ordered if item.confidence < 0.7][:12],
    }


def _extension_has_persistent_recall(
    extensions: EngineExtensionRegistry, extension_id: str
) -> bool:
    return any(
        channel.spec.persistent
        and extensions.owner("index", channel.spec.id) == extension_id
        for channel in extensions.index_channels.channels
    )


def _extension_state_for_sync(
    extension_id: str,
    spec: Any,
    previous: ExtensionState | None,
    revision_changed: bool,
) -> ExtensionState:
    status = "rebuild_required" if revision_changed else "available"
    last_error = None
    if previous is not None and status != "available":
        last_error = previous.last_error
    return ExtensionState(
        extension_id=extension_id,
        installed_version=spec.version,
        engine_api=spec.engine_api,
        storage_revision=spec.storage_revision,
        last_projected_seq=previous.last_projected_seq if previous else 0,
        status=status,
        last_error=last_error,
    )


def _embedding_prewarm_texts(
    store: SQLiteMemoryLogStore, deltas: list[MemoryRecord]
) -> list[str]:
    texts = _record_prewarm_texts(deltas)
    texts.extend(_claim_prewarm_texts(deltas))
    texts.extend(_summary_prewarm_texts(store))
    return texts


def _record_prewarm_texts(deltas: list[MemoryRecord]) -> list[str]:
    return [
        item.text for item in deltas
        if item.kind in {"raw_source", "fact"} and item.text.strip()
    ]


def _claim_prewarm_texts(deltas: list[MemoryRecord]) -> list[str]:
    observations = {item.id: item for item in deltas if item.kind == "raw_source"}
    claims = compile_claims([item for item in deltas if item.kind == "fact"], observations)
    return [item.text for item in claims if item.text.strip()]


def _summary_prewarm_texts(store: SQLiteMemoryLogStore) -> list[str]:
    return [
        " ".join([view.summary, view.concept_or_topic, *view.alt_phrasings])
        for view in store.group_summary_views()
        if view.summary.strip()
    ]


def _append_proposed_ingest_record(
    engine: MemoryEngine,
    name: str,
    processor: Any,
    item: Any,
    available: dict[str, MemoryRecord],
    group_id: str,
    namespace: MemoryNamespace,
) -> None:
    if not isinstance(item, ProposedRecord):
        raise TypeError(f"ingest processor {name} returned a non-ProposedRecord value")
    if item.kind not in processor.spec.output_kinds:
        raise ValueError(f"ingest processor {name} returned disallowed kind {item.kind}")
    if not item.evidence_refs or any(ref not in available for ref in item.evidence_refs):
        raise ValueError(f"ingest processor {name} returned invalid evidence references")
    primary = available[item.evidence_refs[0]]
    engine.store.append(
        kind=item.kind,
        text=item.text,
        group_id=group_id,
        evidence_refs=item.evidence_refs,
        confidence=item.confidence,
        subject=item.subject,
        relation=item.relation,
        object=item.object,
        temporal_anchor=item.temporal_anchor,
        actor=Actor(type="ingest_pipeline", id=name),
        namespace=namespace,
        scope=primary.scope,
        payload={**dict(item.payload), "producer": name},
        source_ref=item.source_ref if item.source_ref is not None else primary.source_ref,
        locator=item.locator if item.locator is not None else primary.locator,
        external_key=item.external_key,
    )


def _chunk_modality(
    chunk_kind: str,
    source_type: str,
    prepared: PreparedSource,
    locator: dict[str, Any],
) -> str:
    if chunk_kind == "notebook_cell":
        return "code" if locator.get("cell_type") == "code" else "document"
    mapped = _CHUNK_KIND_MODALITY.get(chunk_kind)
    if mapped:
        return mapped
    if source_type in {"csv", "xlsx", "spreadsheet"}:
        return "table"
    if source_type == "image" or prepared.kind == "image":
        return "image"
    return "document"


_TEXT_FIELD_BY_MODALITY = {
    "image": ("visible_text", "description"),
    "table": ("record",),
    "code": ("code",),
}


def _chunk_retrieval_fields(
    prepared: PreparedSource,
    chunk: PreparedChunk,
    locator: dict[str, Any],
    modality: str,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "title": prepared.title,
        "filename": prepared.filename,
        "path": locator.get("path") or prepared.filename,
        "section": locator.get("section"),
        "page": locator.get("page"),
        "slide": locator.get("slide"),
        "sheet": locator.get("sheet"),
        "row": locator.get("row"),
        "symbol": locator.get("symbol"),
        "symbol_kind": locator.get("symbol_kind"),
        "language": _chunk_language(locator, chunk, prepared),
        "imports": chunk.metadata.get("imports"),
        "headers": chunk.metadata.get("headers") or prepared.metadata.get("headers"),
    }
    for key in _TEXT_FIELD_BY_MODALITY.get(modality, ("content",)):
        fields[key] = chunk.text
    return _drop_empty_fields(fields)


def _chunk_language(
    locator: dict[str, Any], chunk: PreparedChunk, prepared: PreparedSource
) -> Any:
    return (
        locator.get("language")
        or chunk.metadata.get("language")
        or prepared.metadata.get("language")
    )


def _drop_empty_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in fields.items() if value is not None and value != ""
    }
