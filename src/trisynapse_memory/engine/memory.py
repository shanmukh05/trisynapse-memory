"""Public Trace & Recall memory engine."""

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from trisynapse_memory.engine.recall.compilation import (
    build_episode_recall_views,
    compile_claims,
)
from trisynapse_memory.engine.providers.embedding import Embedder, SentenceTransformerEmbedder, UnavailableEmbedder
from trisynapse_memory.engine.formation.pipeline import extract_episode, ingest_document, ingest_observation
from trisynapse_memory.engine.models import (
    Actor,
    Citation,
    EpisodeInfo,
    EpisodeRecallView,
    MemoryDelta,
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
    RecallSnapshot,
    SnapshotDiff,
    SourceIngestionResult,
    SourceInput,
    SourcePreview,
    SourcePreviewItem,
    SourceRecord,
    StoreValidation,
)
from trisynapse_memory.engine.providers.registry import (
    EmbeddingRebuildRequired,
    ProviderError,
    ProviderSettings,
    completion_from_settings,
    embedder_from_settings,
    embedding_cache_key,
    fetch_model_catalog,
    list_provider_descriptors,
    provider_descriptor,
    provider_provenance,
    settings_from_selection,
    validate_selection,
)
from trisynapse_memory.engine.retrieval.engine import HybridRetriever, RetrieverConfig, classify_query
from trisynapse_memory.engine.retrieval.contracts import QueryPlanner, RouteRegistry
from trisynapse_memory.engine.trace.store import SQLiteTraceStore
from trisynapse_memory.engine.retrieval.tokenization import TokenCounter, token_counter_for
from trisynapse_memory.engine.formation.sources import PreparedChunk, PreparedSource, prepare_source, store_blob
from trisynapse_memory.engine.recall.vector_cache import VectorCache, preferred_vector_cache
from trisynapse_memory.prompts import load_prompt
from trisynapse_memory.engine.utils import __version__

CompletionJSON = Callable[[str, str], dict[str, Any]]


class SnapshotManager:
    """Recall-window snapshots; rollback never mutates the underlying trace."""

    def __init__(self, store: SQLiteTraceStore) -> None:
        self._store = store

    def create(self, label: str | None = None) -> RecallSnapshot:
        return self._store.create_snapshot(label)

    def list(self) -> list[RecallSnapshot]:
        return self._store.list_snapshots()

    def diff(self, a: str, b: str) -> SnapshotDiff:
        return self._store.snapshot_diff(a, b)

    def rollback(self, snapshot_id: str) -> RecallSnapshot:
        """Activate a historical evidence cutoff without deleting newer deltas."""

        return self._store.activate_snapshot(snapshot_id)


class MemoryEngine:
    """Local-first append-only memory with disposable recall products."""

    VERSION = __version__

    def __init__(
        self,
        store: SQLiteTraceStore,
        *,
        embedder: Embedder | None = None,
        completion: CompletionJSON | None = None,
        retriever_config: RetrieverConfig | None = None,
        query_planner: QueryPlanner | None = None,
        retrieval_routes: RouteRegistry | None = None,
        token_counter: TokenCounter | None = None,
        vector_cache: VectorCache | None = None,
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
        self.vector_cache = vector_cache or preferred_vector_cache(store)
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
        vector_cache: VectorCache | None = None,
        query_planner: QueryPlanner | None = None,
        retrieval_routes: RouteRegistry | None = None,
        token_counter: TokenCounter | None = None,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
        completion_provider: ProviderSettings | None = None,
        embedding_provider: ProviderSettings | None = None,
        auto_process: bool = True,
    ) -> "MemoryEngine":
        store = SQLiteTraceStore(path)
        configuration = store.get_model_configuration()
        managed_embedding = embedder is None and embedding_provider is None and embedding_model == "all-MiniLM-L6-v2"
        managed_completion = completion is None and completion_provider is None
        provider_errors: dict[str, str] = {}
        if embedder is not None:
            provider = embedder
        else:
            embedding_settings = embedding_provider or settings_from_selection(configuration.embedding)
            if embedding_provider is None and embedding_model != "all-MiniLM-L6-v2":
                embedding_settings = ProviderSettings(
                    provider="sentence-transformers", model=embedding_model
                )
            try:
                provider = embedder_from_settings(embedding_settings)
                if isinstance(provider, SentenceTransformerEmbedder):
                    provider.local_files_only = local_files_only
            except ProviderError as exc:
                provider_errors["embedding"] = str(exc)
                selection = configuration.embedding
                provider = UnavailableEmbedder(
                    selection.model or "unknown",
                    embedding_cache_key(
                        selection.provider, selection.base_url, selection.model or "unknown"
                    ),
                    str(exc),
                )
        if completion is not None:
            completion_callable = completion
        else:
            completion_settings = completion_provider or settings_from_selection(configuration.completion)
            try:
                completion_callable = completion_from_settings(completion_settings)
            except ProviderError as exc:
                provider_errors["completion"] = str(exc)
                completion_callable = None
        engine = cls(
            store,
            embedder=provider,
            completion=completion_callable,
            vector_cache=vector_cache,
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
        requested = (
            configuration
            if isinstance(configuration, ModelConfiguration)
            else ModelConfiguration.model_validate(configuration)
        )
        validate_selection(requested.completion, ProviderRole.COMPLETION)
        validate_selection(requested.embedding, ProviderRole.EMBEDDING)
        selection_warnings: list[str] = []
        for role, selection in (
            (ProviderRole.COMPLETION, requested.completion),
            (ProviderRole.EMBEDDING, requested.embedding),
        ):
            if selection.provider == "none":
                continue
            descriptor = self.store.find_cached_model(selection.provider, selection.model or "")
            if descriptor is not None and role not in descriptor.roles:
                raise ProviderError(
                    f"model {selection.model} is not cataloged for {role.value}"
                )
            if descriptor is None and not (
                selection.provider == "sentence-transformers"
                and selection.model == "all-MiniLM-L6-v2"
            ):
                selection_warnings.append(
                    f"{selection.provider}/{selection.model} was accepted as an unverified custom model ID"
                )
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
            active = current.model_copy(deep=True)
            active.completion = requested.completion
            current = self.store.save_model_configuration(
                active, expected_revision=current.revision
            )

        if not embedding_changed:
            self._sync_model_configuration(force=True)
            return ModelConfigurationChange(
                status="applied",
                configuration=current,
                message="; ".join(selection_warnings) or None,
            )

        if not self.store.has_searchable_content():
            active = current.model_copy(deep=True)
            active.embedding = requested.embedding
            current = self.store.save_model_configuration(
                active, expected_revision=current.revision
            )
            self._sync_model_configuration(force=True)
            return ModelConfigurationChange(
                status="applied",
                configuration=current,
                message="; ".join([
                    "Embedding configuration changed immediately because the store is empty.",
                    *selection_warnings,
                ]),
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
            dedup_key=f"rebuild-embeddings:{current.revision}:{requested.embedding.provider}:{requested.embedding.base_url}:{requested.embedding.model}",
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
                *selection_warnings,
            ]),
        )
        if wait:
            for _ in range(1000):
                state = self.store.get_job(job.id)
                if state is None or state.status in {"completed", "failed"}:
                    break
                self.run_jobs(max_jobs=1)
            self._sync_model_configuration(force=True)
            return self.get_model_configuration_status()
        return change

    def test_model_connection(
        self,
        role: ProviderRole | str,
        selection: ProviderSelection | dict[str, Any] | None = None,
    ) -> ConnectionTestResult:
        provider_role = role if isinstance(role, ProviderRole) else ProviderRole(role)
        current = self.get_model_configuration()
        selected = (
            current.completion if provider_role == ProviderRole.COMPLETION else current.embedding
        ) if selection is None else (
            selection if isinstance(selection, ProviderSelection) else ProviderSelection.model_validate(selection)
        )
        validate_selection(selected, provider_role)
        if selected.provider == "none":
            return ConnectionTestResult(
                ok=True, role=provider_role, provider="none", model=None,
                message="No completion provider is selected.", billed_request=False,
            )
        try:
            if provider_role == ProviderRole.COMPLETION:
                completion = completion_from_settings(settings_from_selection(selected))
                assert completion is not None
                completion.complete_json(
                    "Return only JSON.",
                    'Reply with exactly {"status":"ok"}.',
                )
                vision = hasattr(completion, "complete_multimodal")
            else:
                embedder = embedder_from_settings(settings_from_selection(selected))
                vectors = embedder.encode(["Trisynapse connection test"])
                if len(vectors) != 1 or not vectors[0]:
                    raise ProviderError("embedding provider returned an empty vector")
                vision = None
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

        import importlib.util
        import shutil as system_shutil
        import socket

        self._sync_model_configuration(force=True)
        path = Path(self.store_path)
        mode = path.stat().st_mode & 0o777 if path.exists() else None
        port_free = True
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", 8765))
            except OSError:
                port_free = False
        status = self.get_model_configuration_status()
        providers = {item.id: item for item in self.list_providers()}
        completion = status.configuration.completion
        embedding = status.configuration.embedding
        required = [
            item for item in (completion, embedding)
            if item.provider not in {"none", "sentence-transformers", "openai-compatible"}
        ]
        missing = [
            providers[item.provider].credential_env for item in required
            if not providers[item.provider].credential_configured
        ]
        storage = self.validate_store()
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
            "source_extras": {
                name: importlib.util.find_spec(module) is not None
                for name, module in {
                    "pdf": "pypdf", "office": "docx", "spreadsheets": "openpyxl",
                    "tree_sitter": "tree_sitter_language_pack", "tokens": "tiktoken",
                }.items()
            },
            "port_8765_available": port_free,
            "pending_jobs": len(self.list_jobs(status="pending")),
            "failed_jobs": len(self.list_jobs(status="failed")),
            "pending_runs": sum(
                run.status in {"pending", "running"}
                for run in self.list_ingestion_runs(namespace=self.default_namespace)
            ),
            "failed_runs": sum(
                run.status in {"failed", "partial"}
                for run in self.list_ingestion_runs(namespace=self.default_namespace)
            ),
            "warning": "Retained originals are permission-restricted but not encrypted.",
        }

    # Formation
    def ingest_observation(
        self,
        text: str,
        *,
        episode_id: str | None = None,
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
    ) -> MemoryDelta:
        memory_namespace = self._resolve_namespace(namespace)
        delta = ingest_observation(
            self.store,
            text,
            episode_id=episode_id,
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
        if episode_id and schedule:
            self._schedule_episode(episode_id, memory_namespace, process=process)
        return delta

    def ingest_messages(
        self,
        messages: Iterable[dict[str, Any] | str],
        *,
        episode_id: str,
        source_ref: dict[str, Any] | str | None = None,
        scope: dict[str, Any] | None = None,
        run_extraction: bool = False,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> list[MemoryDelta]:
        memory_namespace = self._resolve_namespace(namespace, session_id=episode_id)
        result: list[MemoryDelta] = []
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
                self.ingest_observation(
                    prefix + text,
                    episode_id=episode_id,
                    source_ref=source_ref or {"type": "chat", "id": episode_id},
                    locator={"kind": "message_index", "message_index": index, "message_id": message_id},
                    scope=scope,
                    namespace=memory_namespace,
                    observed_at=observed_at,
                    external_key=f"message:{episode_id}:{message_id or index}",
                    modality="conversation",
                    source_type="conversation",
                    retrieval_fields={"speaker": role or "", "message": text},
                    process=False,
                    schedule=False,
                )
            )
        self._schedule_episode(episode_id, memory_namespace, force_extraction=run_extraction)
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
    ) -> list[MemoryDelta]:
        memory_namespace = self._resolve_namespace(namespace, session_id=f"item:{document_id}")
        deltas = ingest_document(
            self.store, text, document_id=document_id, title=title, chunk_chars=chunk_chars,
            scope=self._scope(memory_namespace, scope), observed_at=observed_at,
            namespace=memory_namespace,
            payload_extra={"document_metadata": metadata or {}},
        )
        self._schedule_episode(f"item:{document_id}", memory_namespace)
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
    ) -> list[MemoryDelta]:
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
        return [self.get_delta(delta_id) for delta_id in result.delta_ids]

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
        inputs = run.inputs
        memory_namespace = run.namespace
        run.status = "running"
        run.updated_at = datetime.now(timezone.utc)
        self.store.put_ingestion_run(run)
        prepared: dict[int, PreparedSource] = {}
        failures: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, 4, len(inputs)))) as pool:
            futures = {pool.submit(prepare_source, item, self.completion): index for index, item in enumerate(inputs)}
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
        retained_bytes = 0
        for index in range(len(inputs)):
            if index not in prepared:
                continue
            candidate_bytes = retained_bytes + len(prepared[index].original)
            if candidate_bytes > 250 * 1024 * 1024:
                failures[index] = "ingestion run exceeds the 250 MiB retained-source limit"
                del prepared[index]
            else:
                retained_bytes = candidate_bytes
        results: list[SourceIngestionResult] = []
        for index, item in enumerate(inputs):
            if on_progress:
                on_progress(f"Writing source {index + 1} of {len(inputs)} to Trace")
            if index in failures:
                results.append(SourceIngestionResult(index=index, source_key=item.source_key, kind=item.kind, status="failed", error=failures[index]))
                continue
            try:
                results.append(self._commit_source(index, prepared[index], memory_namespace, run_id=run.id))
            except Exception as exc:
                results.append(SourceIngestionResult(index=index, source_key=item.source_key, kind=item.kind, status="failed", error=str(exc)))
        succeeded = sum(item.status in {"success", "skipped"} for item in results)
        failed = sum(item.status == "failed" for item in results)
        run.results = results
        run.status = "failed" if failed and not succeeded else "partial" if failed else "completed"
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
        source_key = prepared.source.source_key or prepared.uri or f"upload:{prepared.filename}:{prepared.content_hash}"
        previous = self.store.latest_source(source_key, namespace)
        if previous is not None and previous.status == "active" and previous.content_hash == prepared.content_hash:
            # A no-op ingestion still verifies and repairs its content-addressed
            # retained blob before returning the existing source version.
            store_blob(self.store.root, prepared.original, filename=prepared.filename)
            return SourceIngestionResult(
                index=index, source_id=previous.id, source_key=source_key, kind=previous.kind,
                status="skipped", episode_id=f"source:{previous.id}:v{previous.version}", delta_ids=previous.delta_ids,
                skipped_paths=prepared.skipped_paths,
            )
        identity = json.dumps(
            {
                "namespace": namespace.model_dump(mode="json", exclude_none=True),
                "source_key": source_key,
                "content_hash": prepared.content_hash,
            },
            sort_keys=True,
        )
        source_id = f"src_{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
        version = (previous.version + 1) if previous else 1
        episode_id = f"source:{source_id}:v{version}"
        blob_path = store_blob(self.store.root, prepared.original, filename=prepared.filename)
        source_ref = {
            "type": prepared.kind, "id": source_id, "source_key": source_key, "title": prepared.title,
            "version": version, "content_hash": prepared.content_hash, "uri": prepared.uri,
        }
        deltas: list[MemoryDelta] = []
        for chunk_index, chunk in enumerate(prepared.chunks):
            if not chunk.text.strip():
                continue
            modality, source_type, retrieval_fields = _source_retrieval_descriptor(
                prepared, chunk
            )
            delta = self.ingest_observation(
                chunk.text,
                episode_id=episode_id,
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
            )
            deltas.append(delta)
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
            delta_ids=[item.id for item in deltas],
            previous_source_id=previous.id if previous else None,
        )
        self.store.put_source(record)
        if previous is not None and previous.status == "active":
            for delta_id in previous.delta_ids:
                if delta_id not in self.store.retracted_ids():
                    self.retract(delta_id=delta_id, reason=f"source replaced by {source_id}", requested_by="source-ingestion", namespace=namespace)
            previous.status = "superseded"
            previous.removed_at = datetime.now(timezone.utc)
            self.store.put_source(previous)
        self._schedule_episode(episode_id, namespace)
        return SourceIngestionResult(
            index=index, source_id=source_id, source_key=source_key, kind=prepared.kind,
            status="success", episode_id=episode_id, delta_ids=record.delta_ids, skipped_paths=prepared.skipped_paths,
        )

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
        if search:
            term = search.lower()
            values = [
                item for item in values
                if term in " ".join(filter(None, [item.title, item.source_key, item.uri, item.filename])).lower()
            ]
        allowed_kinds = set(kinds or [])
        if allowed_kinds:
            values = [item for item in values if item.kind in allowed_kinds]
        if status:
            values = [item for item in values if item.status == status]
        if sort == "oldest":
            values.sort(key=lambda item: item.created_at)
        elif sort == "title":
            values.sort(key=lambda item: item.title.lower())
        elif sort != "newest":
            raise ValueError("sort must be newest, oldest, or title")
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
        path = self.store.root / source.blob_path
        update: dict[str, Any] = {}
        if not source.filename:
            derived_filename = source.metadata.get("filename")
            if not derived_filename:
                for delta_id in source.delta_ids:
                    delta = self.store.get(delta_id)
                    if delta is not None and isinstance(delta.locator, dict) and delta.locator.get("path"):
                        derived_filename = Path(str(delta.locator["path"])).name
                        break
            update["filename"] = str(derived_filename or source.title)
        if not source.byte_size and path.is_file():
            update["byte_size"] = path.stat().st_size
        if not source.chunk_count:
            update["chunk_count"] = len(source.delta_ids)
        if not source.preview_type:
            update["preview_type"] = str(source.metadata.get("source_type") or source.kind)
        skipped_paths = source.metadata.get("skipped_paths") or []
        if not source.skipped_count and skipped_paths:
            update["skipped_count"] = len(skipped_paths)
        if not source.ingestion_run_id:
            for run in self.store.list_ingestion_runs(source.namespace, limit=10_000):
                if any(result.source_id == source.id for result in run.results):
                    update["ingestion_run_id"] = run.id
                    break
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
        values: list[SourcePreviewItem] = []
        manifest: list[str] = []
        selected_ids = source.delta_ids[cursor:cursor + limit + 1]
        for delta_id in selected_ids[:limit]:
            delta = self.store.get(delta_id)
            if delta is None:
                continue
            locator = delta.locator
            locator_metadata = (
                locator.get("metadata", {}) if isinstance(locator, dict) else {}
            )
            kind = (
                str(locator.get("kind") or delta.kind)
                if isinstance(locator, dict) else delta.kind
            )
            if kind == "manifest":
                manifest = [line for line in delta.text.splitlines()[1:] if line.strip()]
            values.append(SourcePreviewItem(
                delta_id=delta.id,
                kind=kind,
                text=delta.text,
                locator=locator,
                metadata=locator_metadata,
            ))
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

    def run_extraction(
        self,
        *,
        episode_id: str,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> list[MemoryDelta]:
        self._sync_model_configuration()
        if self.completion is None:
            raise RuntimeError("run_extraction requires a configured completion provider")
        return extract_episode(self.store, episode_id, self.completion, namespace=self._resolve_namespace(namespace))

    def build_episode_recall(
        self,
        episode_ids: list[str] | None = None,
        *,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> list[EpisodeRecallView]:
        self._sync_model_configuration()
        memory_namespace = self._resolve_namespace(namespace)
        deltas = self.store.list_deltas(
            kinds=["observation", "extraction", "annotation"], namespace=memory_namespace
        )
        views = build_episode_recall_views(deltas, complete_json=self.completion, episode_ids=episode_ids)
        for view in views:
            view.namespace = memory_namespace
            view.cache_key = f"{view.cache_key}:{_namespace_key(memory_namespace)}"
            self.store.put_episode_recall(view)
        return views

    def retract(
        self,
        *,
        delta_id: str,
        reason: str,
        requested_by: str = "user",
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> MemoryDelta:
        memory_namespace = self._resolve_namespace(namespace)
        target = self.get(delta_id, namespace=memory_namespace)
        if target is None:
            raise KeyError(f"unknown delta: {delta_id}")
        return self.store.append(
            kind="retraction",
            text=f"Retracted {delta_id}: {reason}",
            evidence_refs=[delta_id],
            payload={"target_delta_ids": [delta_id], "reason": reason, "requested_by": requested_by},
            actor=Actor(type="user", id=requested_by),
            confidence=1.0,
            namespace=memory_namespace,
            scope=self._scope(memory_namespace, None),
        )

    forget = retract

    def correct(
        self,
        *,
        delta_id: str,
        text: str,
        reason: str = "user correction",
        requested_by: str = "user",
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> MemoryDelta:
        memory_namespace = self._resolve_namespace(namespace)
        target = self.get(delta_id, namespace=memory_namespace)
        return self.store.append(
            kind="extraction",
            text=text,
            episode_id=target.episode_id,
            evidence_refs=[delta_id],
            payload={"annotation_type": "correction", "target_delta_ids": [delta_id], "reason": reason},
            actor=Actor(type="user", id=requested_by),
            confidence=1.0,
            namespace=memory_namespace,
            scope=target.scope,
        )

    def remove(
        self,
        *,
        delta_ids: list[str],
        reason: str,
        requested_by: str = "user",
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> RemoveResult:
        memory_namespace = self._resolve_namespace(namespace)
        for delta_id in delta_ids:
            self.get(delta_id, namespace=memory_namespace, include_retracted=True)
        for delta_id in delta_ids:
            if delta_id not in self.store.retracted_ids():
                self.retract(delta_id=delta_id, reason=reason, requested_by=requested_by, namespace=memory_namespace)
        result = self.store.hard_remove(delta_ids, requested_by=requested_by, reason=reason)
        self.vector_cache.clear()
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
        result = self.remove(delta_ids=source.delta_ids, reason=reason, requested_by=requested_by, namespace=source.namespace)
        source.status = "removed"
        source.removed_at = datetime.now(timezone.utc)
        self.store.put_source(source)
        if not self.store.blob_is_referenced(source.blob_path, excluding_source_id=source.id):
            blob = self.store.root / source.blob_path
            blob.unlink(missing_ok=True)
        return result

    def add(self, text: str, **kwargs: Any) -> MemoryDelta:
        return self.ingest_observation(text, **kwargs)

    def add_batch(self, items: Iterable[dict[str, Any] | str], **kwargs: Any) -> list[MemoryDelta]:
        result: list[MemoryDelta] = []
        scheduled: dict[tuple[str, str], tuple[str, MemoryNamespace, bool | None]] = {}
        for item in items:
            payload = {**kwargs, **({"text": item} if isinstance(item, str) else item)}
            episode_id = payload.get("episode_id")
            if episode_id:
                namespace = self._resolve_namespace(payload.get("namespace"))
                process = payload.get("process")
                payload["schedule"] = False
                scheduled[(_namespace_key(namespace), str(episode_id))] = (str(episode_id), namespace, process)
            result.append(self.add(**payload))
        for episode_id, namespace, process in scheduled.values():
            self._schedule_episode(episode_id, namespace, process=process)
        return result

    def append_deltas(self, deltas: Iterable[dict[str, Any]]) -> list[MemoryDelta]:
        return [self.store.append(**delta) for delta in deltas]

    # Retrieval
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        episode_prefix: str | None = None,
        scope: dict[str, Any] | None = None,
        query_id: str | None = None,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
        on_step: Callable[[QueryStep], None] | None = None,
    ) -> MemorySearchResult:
        memory_namespace = self._resolve_namespace(namespace)
        identifier = query_id or f"q_{secrets.token_hex(8)}"
        run = self._new_query_run(identifier, "search", query, memory_namespace)
        if on_step and run.steps:
            on_step(run.steps[0])
        return self._execute_search(
            run,
            top_k=top_k,
            episode_prefix=episode_prefix,
            scope=scope,
            finish=True,
            on_step=on_step,
        )

    def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        episode_prefix: str | None = None,
        scope: dict[str, Any] | None = None,
        abstain_threshold: float | None = None,
        history: list[dict[str, Any]] | None = None,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
        on_step: Callable[[QueryStep], None] | None = None,
    ) -> MemoryQueryResult:
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
            episode_prefix=episode_prefix,
            scope=scope,
            abstain_threshold=abstain_threshold,
            on_step=on_step,
        )

    def _new_query_run(
        self,
        identifier: str,
        mode: str,
        query: str,
        namespace: MemoryNamespace,
        *,
        status: str = "running",
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
        self.store.put_query_run(run)
        return run

    def _append_query_step(
        self,
        run: QueryRun,
        step: QueryStep,
        on_step: Callable[[QueryStep], None] | None = None,
    ) -> None:
        run.steps = [item for item in run.steps if item.id != step.id]
        run.steps.append(step)
        run.steps.sort(key=lambda item: item.sequence)
        run.updated_at = datetime.now(timezone.utc)
        self.store.put_query_run(run)
        if on_step:
            on_step(step)

    def _execute_search(
        self,
        run: QueryRun,
        *,
        top_k: int | None,
        episode_prefix: str | None,
        scope: dict[str, Any] | None,
        finish: bool,
        on_step: Callable[[QueryStep], None] | None = None,
    ) -> MemorySearchResult:
        self._sync_model_configuration()
        if not self._retriever_override:
            self.retriever_config = _retriever_config(self.get_retrieval_configuration())
        effective_top_k = top_k or run.retrieval_configuration.default_top_k
        retriever = HybridRetriever(
            self.store,
            self.embedder,
            self.vector_cache,
            self.retriever_config,
            planner=self.query_planner,
            routes=self.retrieval_routes,
            token_counter=self.token_counter,
        )
        started = time.perf_counter()
        result = retriever.search(
            run.query,
            top_k=effective_top_k,
            episode_prefix=episode_prefix,
            scope=self._scope(run.namespace, scope),
            namespace=run.namespace,
            query_id=run.id,
            on_step=lambda step: self._append_query_step(run, step, on_step),
        )
        run.retrieval_trace = result.retrieval_trace
        if finish:
            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc)
            run.updated_at = run.completed_at
            run.duration_ms = (time.perf_counter() - started) * 1000
            self.store.put_query_run(run)
        return result

    def _execute_query(
        self,
        run: QueryRun,
        *,
        original_question: str,
        top_k: int | None,
        episode_prefix: str | None,
        scope: dict[str, Any] | None,
        abstain_threshold: float | None,
        on_step: Callable[[QueryStep], None] | None = None,
    ) -> MemoryQueryResult:
        started = time.perf_counter()
        search_result = self._execute_search(
            run,
            # ``default_top_k`` controls ordinary search results. Answer
            # generation gets the larger grounded context window unless the
            # caller explicitly requests a limit.
            top_k=(
                top_k
                if top_k is not None
                else run.retrieval_configuration.max_context_items
            ),
            episode_prefix=episode_prefix,
            scope=scope,
            finish=False,
            on_step=on_step,
        )
        hits = search_result.hits
        threshold = (
            run.retrieval_configuration.answer_abstain_threshold
            if abstain_threshold is None else abstain_threshold
        )
        relevant = bool(hits and hits[0].score >= threshold)
        generation_started = time.perf_counter()
        cited_hits = hits
        if self.completion is not None and relevant:
            payload = self.completion(
                load_prompt("answer").text,
                _answer_prompt(original_question, hits),
            )
            answer = str(payload.get("answer") or "").strip()
            abstain = bool(payload.get("abstain", False)) or not answer
            requested_citations = payload.get("citation_ids")
            if requested_citations is not None:
                if not isinstance(requested_citations, list):
                    raise ValueError("answer citation_ids must be a list")
                requested_ids = {str(value) for value in requested_citations}
                cited_hits = [hit for hit in hits if hit.item_id in requested_ids]
                if answer and not abstain and not cited_hits:
                    answer = "I don't have enough grounded evidence to answer that."
                    abstain = True
        elif relevant:
            answer = _extractive_answer(original_question, hits)
            abstain = not answer
        else:
            answer = "I don't have enough grounded evidence to answer that."
            abstain = True
        citations = _citations(cited_hits if not abstain else [])
        provenance = {
            **provider_provenance(self.completion, kind="completion"),
            "prompt": load_prompt("answer").provenance() if self.completion is not None else None,
        }
        self._append_query_step(run, QueryStep(
            id=f"{run.id}:{len(run.steps)}:answer",
            phase="answer",
            label="Generate grounded answer" if relevant else "Abstain without enough evidence",
            sequence=len(run.steps),
            input={"evidence_ids": [hit.item_id for hit in hits], "threshold": threshold},
            output={"answer": answer, "abstain": abstain, "citation_ids": [item.delta_id for item in citations]},
            metrics={"evidence_count": len(hits), "citation_count": len(citations)},
            duration_ms=(time.perf_counter() - generation_started) * 1000,
        ), on_step)
        self.store.append(
            kind="access",
            text=f"Query access {search_result.query_id}",
            evidence_refs=[citation.delta_id for citation in citations],
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
            output={"query_id": run.id, "citation_ids": [item.delta_id for item in citations]},
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
        episode_prefix: str | None = None,
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
                "episode_prefix": episode_prefix,
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
                    episode_prefix=payload.get("episode_prefix"),
                    scope=payload.get("scope"),
                    finish=True,
                )
            else:
                self._execute_query(
                    run,
                    original_question=run.query,
                    top_k=payload.get("top_k"),
                    episode_prefix=payload.get("episode_prefix"),
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
    def get_delta(self, delta_id: str) -> MemoryDelta:
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
    ) -> MemoryDelta:
        delta = self.get_delta(delta_id)
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
        items = self.store.list_deltas(
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
        deltas = self.store.list_deltas(namespace=memory_namespace, include_retracted=include_retracted)
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

    def list_episodes(
        self,
        *,
        scope: dict[str, Any] | None = None,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> list[EpisodeInfo]:
        memory_namespace = self._resolve_namespace(namespace)
        grouped: dict[str, list[MemoryDelta]] = defaultdict(list)
        for delta in self.store.list_deltas(scope=self._scope(memory_namespace, scope), namespace=memory_namespace):
            if delta.episode_id:
                grouped[delta.episode_id].append(delta)
        fresh = {view.episode_id: not view.stale for view in self.store.episode_recall_views(namespace=memory_namespace)}
        result = []
        for episode_id, deltas in sorted(grouped.items()):
            observed = [item.observed_at for item in deltas if item.observed_at]
            result.append(
                EpisodeInfo(
                    episode_id=episode_id,
                    delta_count=len(deltas),
                    observation_count=sum(item.kind == "observation" for item in deltas),
                    extraction_count=sum(item.kind == "extraction" for item in deltas),
                    first_observed_at=min(observed) if observed else None,
                    last_observed_at=max(observed) if observed else None,
                    stale=not fresh.get(episode_id, False),
                )
            )
        return result

    def export_graph(
        self,
        format: str = "json",
        *,
        scope: dict[str, Any] | None = None,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> dict[str, Any] | str:
        memory_namespace = self._resolve_namespace(namespace)
        deltas = self.store.list_deltas(
            kinds=["observation", "extraction"],
            scope=self._scope(memory_namespace, scope),
            namespace=memory_namespace,
        )
        observations = {item.id: item for item in deltas if item.kind == "observation"}
        claims = compile_claims([item for item in deltas if item.kind == "extraction"], observations)
        nodes = []
        edges = []
        seen: set[str] = set()
        for claim in claims:
            for label in (claim.subject, claim.object):
                if label and label not in seen:
                    seen.add(label)
                    nodes.append({"id": label, "label": label})
            if claim.subject and claim.object:
                edges.append({"source": claim.subject, "target": claim.object, "relation": claim.relation, "confidence": claim.confidence, "status": claim.status})
        graph = {"nodes": nodes, "edges": edges}
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
        episode_id: str | None = None,
        limit: int = 500,
        cursor: str | None = None,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> MemoryGraphPage:
        if view not in {"knowledge", "lineage", "trace"}:
            raise ValueError("view must be knowledge, lineage, or trace")
        if limit < 1 or limit > 2000:
            raise ValueError("limit must be between 1 and 2000")
        memory_namespace = self._resolve_namespace(namespace)
        deltas = self.store.list_deltas(
            namespace=memory_namespace, include_retracted=True
        )
        if episode_id:
            deltas = [item for item in deltas if item.episode_id == episode_id]
        if source_id:
            deltas = [item for item in deltas if item.scope.get("source_id") == source_id]
        observations = {item.id: item for item in deltas if item.kind == "observation"}
        claims = compile_claims(
            [item for item in deltas if item.kind == "extraction"], observations
        )
        nodes: dict[str, MemoryGraphNode] = {}
        edges: dict[str, MemoryGraphEdge] = {}

        def add_node(node: MemoryGraphNode) -> None:
            nodes.setdefault(node.id, node)

        def add_edge(edge: MemoryGraphEdge) -> None:
            edges.setdefault(edge.id, edge)

        if view in {"lineage", "trace"}:
            source_values = self.list_sources(
                namespace=memory_namespace, include_removed=True
            )
            if source_id:
                source_values = [item for item in source_values if item.id == source_id]
            for source in source_values:
                add_node(MemoryGraphNode(
                    id=source.id, type="source", label=source.title,
                    subtitle=source.kind, status=source.status,
                    data={"source_key": source.source_key, "version": source.version, "media_type": source.media_type},
                ))
                if source.previous_source_id:
                    add_edge(MemoryGraphEdge(
                        id=f"source-version:{source.id}:{source.previous_source_id}",
                        source=source.id, target=source.previous_source_id,
                        type="supersedes", label="supersedes",
                    ))
            previous_by_episode: dict[str, str] = {}
            for delta in deltas:
                add_node(MemoryGraphNode(
                    id=delta.id, type="trace", label=(delta.text or delta.kind)[:90],
                    subtitle=delta.kind, status="retracted" if delta.id in self.store.retracted_ids() else "active",
                    data={
                        "kind": delta.kind, "seq": delta.seq, "episode_id": delta.episode_id,
                        "observed_at": delta.observed_at.isoformat() if delta.observed_at else None,
                        "source_ref": delta.source_ref, "locator": delta.locator,
                    },
                ))
                current_source = str(delta.scope.get("source_id") or "")
                if current_source:
                    add_edge(MemoryGraphEdge(
                        id=f"source-trace:{current_source}:{delta.id}", source=current_source,
                        target=delta.id, type="produced", label="produced",
                    ))
                if delta.episode_id:
                    episode_node = f"episode:{delta.episode_id}"
                    add_node(MemoryGraphNode(
                        id=episode_node, type="episode", label=delta.episode_id,
                        subtitle="episode",
                    ))
                    add_edge(MemoryGraphEdge(
                        id=f"episode-trace:{episode_node}:{delta.id}", source=episode_node,
                        target=delta.id, type="contains", label="contains",
                    ))
                    previous = previous_by_episode.get(delta.episode_id)
                    if view == "trace" and previous:
                        add_edge(MemoryGraphEdge(
                            id=f"follows:{previous}:{delta.id}", source=previous,
                            target=delta.id, type="followed_by", label="then",
                        ))
                    previous_by_episode[delta.episode_id] = delta.id
                for evidence_id in delta.evidence_refs:
                    add_edge(MemoryGraphEdge(
                        id=f"evidence:{evidence_id}:{delta.id}", source=evidence_id,
                        target=delta.id, type="supports", label="supports",
                    ))

        if view in {"knowledge", "lineage"}:
            for claim in claims:
                claim_id = f"claim:{claim.id}"
                add_node(MemoryGraphNode(
                    id=claim_id, type="claim", label=claim.text[:100],
                    subtitle=claim.relation, status=claim.status,
                    data={"confidence": claim.confidence, "source_delta_ids": claim.source_delta_ids},
                ))
                for role, label in (("subject", claim.subject), ("object", claim.object)):
                    if not label:
                        continue
                    concept_id = f"concept:{hashlib.sha256(label.lower().encode()).hexdigest()[:16]}"
                    add_node(MemoryGraphNode(id=concept_id, type="concept", label=label, subtitle=role))
                    if role == "subject":
                        add_edge(MemoryGraphEdge(
                            id=f"claim-subject:{concept_id}:{claim_id}", source=concept_id,
                            target=claim_id, type="subject", label=claim.relation or "claim",
                            weight=claim.confidence,
                        ))
                    else:
                        add_edge(MemoryGraphEdge(
                            id=f"claim-object:{claim_id}:{concept_id}", source=claim_id,
                            target=concept_id, type="object", label=claim.relation or "relates to",
                            weight=claim.confidence,
                        ))
                if view == "lineage":
                    for evidence_id in dict.fromkeys(claim.source_delta_ids + claim.observation_delta_ids):
                        add_edge(MemoryGraphEdge(
                            id=f"claim-ground:{evidence_id}:{claim_id}", source=evidence_id,
                            target=claim_id, type="grounds", label="grounds",
                        ))
            if view == "lineage":
                for recall in self.store.episode_recall_views(namespace=memory_namespace):
                    add_node(MemoryGraphNode(
                        id=recall.id, type="recall", label=recall.concept_or_topic,
                        subtitle="Episode Recall", status="stale" if recall.stale else "fresh",
                        data={"episode_id": recall.episode_id, "summary": recall.summary},
                    ))
                    for trace_id in recall.source_trace_ids:
                        add_edge(MemoryGraphEdge(
                            id=f"recall-ground:{trace_id}:{recall.id}", source=trace_id,
                            target=recall.id, type="summarizes", label="summarizes",
                        ))

        all_nodes = list(nodes.values())
        allowed = set(node_types or [])
        if allowed:
            all_nodes = [item for item in all_nodes if item.type in allowed]
        if search:
            term = search.lower()
            matched = {item.id for item in all_nodes if term in f"{item.label} {item.subtitle or ''}".lower()}
            neighbor_ids = {
                endpoint for edge in edges.values()
                if edge.source in matched or edge.target in matched
                for endpoint in (edge.source, edge.target)
            }
            all_nodes = [item for item in all_nodes if item.id in matched | neighbor_ids]
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

    def memory_graph_neighbors(
        self,
        node_id: str,
        *,
        view: str = "lineage",
        limit: int = 500,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> MemoryGraphPage:
        graph = self.memory_graph(view, limit=2000, namespace=namespace)
        edges = [edge for edge in graph.edges if node_id in {edge.source, edge.target}][:limit]
        ids = {node_id}
        for edge in edges:
            ids.update((edge.source, edge.target))
        return MemoryGraphPage(
            view=graph.view,
            nodes=[node for node in graph.nodes if node.id in ids],
            edges=edges,
            counts=graph.counts,
            truncated=len(graph.edges) > len(edges),
        )

    def compile_profile(
        self,
        *,
        scope: dict[str, Any] | None = None,
        query: str | None = None,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        memory_namespace = self._resolve_namespace(namespace)
        deltas = self.store.list_deltas(
            kinds=["observation", "extraction"],
            scope=self._scope(memory_namespace, scope),
            namespace=memory_namespace,
        )
        observations = {item.id: item for item in deltas if item.kind == "observation"}
        claims = compile_claims([item for item in deltas if item.kind == "extraction"], observations)
        ordered = sorted(claims, key=lambda item: item.confidence, reverse=True)
        profile = {
            "static": [item.text for item in ordered if item.confidence >= 0.7][:12],
            "dynamic": [item.text for item in ordered if item.confidence < 0.7][:12],
        }
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
    ) -> MemoryDelta:
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

    def run_jobs(self, *, max_jobs: int = 100) -> list[MemoryJob]:
        self._sync_model_configuration()
        completed: list[MemoryJob] = []
        for _ in range(max_jobs):
            job = self.store.claim_job()
            if job is None:
                break
            try:
                namespace = MemoryNamespace.model_validate(job.payload.get("namespace") or {})
                episode_id = str(job.payload.get("episode_id") or "")
                if job.kind == "extract_episode":
                    if self.completion is None:
                        raise RuntimeError("no completion provider is configured")
                    self.run_extraction(episode_id=episode_id, namespace=namespace)
                elif job.kind == "compile_episode":
                    self.build_episode_recall([episode_id], namespace=namespace)
                elif job.kind == "rebuild_embeddings":
                    selection = ProviderSelection.model_validate(job.payload["embedding"])
                    target_embedder = embedder_from_settings(settings_from_selection(selection))
                    self._prewarm_embedding_cache(target_embedder)
                    self.store.activate_pending_embedding(job.id)
                    self._sync_model_configuration(force=True)
                elif job.kind == "execute_query":
                    self.execute_query_run(str(job.payload["query_id"]))
                completed.append(self.store.finish_job(job.id))
            except Exception as exc:
                finished = self.store.finish_job(job.id, error=str(exc))
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

    def _sync_model_configuration(self, *, force: bool = False) -> None:
        configuration = self.store.get_model_configuration()
        if not force and configuration.revision == self._config_revision:
            return
        if self._managed_completion:
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
        if self._managed_embedding:
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
        if not self._token_counter_override:
            self.token_counter = token_counter_for(self.completion)
        self._config_revision = configuration.revision

    def _prewarm_embedding_cache(self, embedder: Embedder) -> None:
        deltas = self.store.list_deltas(
            kinds=["observation", "extraction", "annotation"],
            include_retracted=False,
        )
        observations = {item.id: item for item in deltas if item.kind == "observation"}
        claims = compile_claims(
            [item for item in deltas if item.kind == "extraction"], observations
        )
        texts = [
            item.text for item in deltas
            if item.kind in {"observation", "extraction"} and item.text.strip()
        ]
        texts.extend(item.text for item in claims if item.text.strip())
        texts.extend(
            " ".join([view.summary, view.concept_or_topic, *view.alt_phrasings])
            for view in self.store.episode_recall_views()
            if view.summary.strip()
        )
        unique = {
            hashlib.sha256(text.encode("utf-8")).hexdigest(): text
            for text in texts
        }
        cache_key = str(getattr(embedder, "cache_key", embedder.model_name))
        existing = self.vector_cache.get(list(unique), cache_key)
        missing = [(text_hash, text) for text_hash, text in unique.items() if text_hash not in existing]
        for offset in range(0, len(missing), 64):
            batch = missing[offset : offset + 64]
            vectors = embedder.encode([text for _, text in batch])
            if len(vectors) != len(batch):
                raise RuntimeError("embedding provider returned the wrong number of vectors")
            self.vector_cache.put(
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

    def _schedule_episode(
        self,
        episode_id: str,
        namespace: MemoryNamespace,
        *,
        force_extraction: bool = False,
        process: bool | None = None,
    ) -> None:
        payload = {"episode_id": episode_id, "namespace": namespace.model_dump(mode="json")}
        namespace_key = _namespace_key(namespace)
        evidence_version = self.store.max_seq()
        if self.completion is not None or force_extraction:
            self.store.enqueue_job(
                "extract_episode", payload, dedup_key=f"extract:{namespace_key}:{episode_id}:{evidence_version}"
            )
        self.store.enqueue_job(
            "compile_episode", payload, dedup_key=f"compile:{namespace_key}:{episode_id}:{evidence_version}"
        )
        should_process = process if process is not None else self.auto_process
        if should_process:
            self.run_jobs()

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
        if hit.kind in {"observation", "extraction"}
    ]
    return (
        f"Question type: {query_kind}\n"
        f"Answer policy: {policies[query_kind]}\n"
        f"Question: {question}\n\nTrace context:\n" + "\n".join(lines)
    )


def _source_retrieval_descriptor(
    prepared: PreparedSource,
    chunk: PreparedChunk,
) -> tuple[str, str, dict[str, Any]]:
    """Map every accepted source to generic modality and searchable fields."""

    locator = chunk.locator
    chunk_kind = str(locator.get("kind") or "")
    source_type = str(
        chunk.metadata.get("source_type")
        or prepared.metadata.get("source_type")
        or prepared.kind
    )
    if chunk_kind in {"code_symbol", "code_lines"}:
        modality = "code"
    elif chunk_kind == "notebook_cell":
        modality = "code" if locator.get("cell_type") == "code" else "document"
    elif chunk_kind in {"row", "cell"} or source_type in {"csv", "xlsx", "spreadsheet"}:
        modality = "table"
    elif chunk_kind == "image" or source_type == "image" or prepared.kind == "image":
        modality = "image"
    elif chunk_kind in {"message", "message_index", "dialog"}:
        modality = "conversation"
    else:
        modality = "document"

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
        "language": locator.get("language") or chunk.metadata.get("language") or prepared.metadata.get("language"),
        "imports": chunk.metadata.get("imports"),
        "headers": chunk.metadata.get("headers") or prepared.metadata.get("headers"),
    }
    if modality == "image":
        fields["visible_text"] = chunk.text
        fields["description"] = chunk.text
    elif modality == "table":
        fields["record"] = chunk.text
    elif modality == "code":
        fields["code"] = chunk.text
    else:
        fields["content"] = chunk.text
    return modality, source_type, {
        key: value for key, value in fields.items() if value is not None and value != ""
    }


def _retriever_config(value: RetrievalConfiguration) -> RetrieverConfig:
    return RetrieverConfig(
        top_k=value.default_top_k,
        max_context_items=value.max_context_items,
        max_context_tokens=value.max_context_tokens,
        per_source_context_tokens=value.per_source_context_tokens,
        graph_hops=value.graph_hops,
        margin_threshold=value.confidence_margin,
        max_refinement_rounds=value.max_refinement_rounds,
        deep_recall_enabled=value.deep_recall_enabled,
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
        if hit.item_id in seen or hit.kind not in {"observation", "extraction"}:
            continue
        seen.add(hit.item_id)
        result.append(
            Citation(
                delta_id=hit.item_id,
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
