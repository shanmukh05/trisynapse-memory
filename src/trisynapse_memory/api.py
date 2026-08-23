"""FastAPI surface for Trace & Recall and benchmark artifacts."""

from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from trisynapse_memory.adapters.agent_events import AgentEvent, capture_agent_event
from trisynapse_memory.benchmarks import (
    discover_benchmark_runs,
    read_benchmark_run,
    run_trace_recall_benchmark,
    run_trace_recall_smoke,
)
from trisynapse_memory.engine import EmbeddingRebuildRequired, MemoryEngine, SourceInput
from trisynapse_memory.engine.models import (
    MemoryNamespace,
    ModelConfiguration,
    ProviderRole,
    ProviderSelection,
    QueryRunRemoveRequest,
    RetrievalConfiguration,
    RemoveRequest,
)

MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_RUN_BYTES = 250 * 1024 * 1024


class ObservationRequest(BaseModel):
    text: str
    episode_id: str | None = None
    observed_at: str | None = None
    source_ref: dict[str, Any] | str | None = None
    locator: dict[str, Any] | str | None = None
    scope: dict[str, Any] | None = None
    external_key: str | None = None
    modality: str = "text"
    source_type: str | None = None
    retrieval_fields: dict[str, Any] | None = None
    namespace: MemoryNamespace | None = None


class MessagesRequest(BaseModel):
    episode_id: str
    messages: list[dict[str, Any] | str]
    scope: dict[str, Any] | None = None
    run_extraction: bool = False
    namespace: MemoryNamespace | None = None


class DocumentRequest(BaseModel):
    document_id: str
    title: str | None = None
    text: str
    chunk_chars: int = Field(default=3500, ge=256)
    scope: dict[str, Any] | None = None
    namespace: MemoryNamespace | None = None


class FileRequest(BaseModel):
    filename: str
    content_base64: str
    document_id: str | None = None
    title: str | None = None
    chunk_chars: int = Field(default=3500, ge=256)
    scope: dict[str, Any] | None = None
    namespace: MemoryNamespace | None = None


class BatchMemoryRequest(BaseModel):
    items: list[ObservationRequest] = Field(min_length=1, max_length=1000)


class CorrectionRequest(BaseModel):
    text: str
    reason: str = "user correction"
    requested_by: str = "user"
    namespace: MemoryNamespace | None = None


class ForgetRequest(BaseModel):
    reason: str
    requested_by: str = "user"
    namespace: MemoryNamespace | None = None


class SourceIngestRequest(BaseModel):
    sources: list[SourceInput] = Field(min_length=1, max_length=100)
    namespace: MemoryNamespace | None = None


class SourceRemoveRequest(BaseModel):
    reason: str
    requested_by: str = "user"
    namespace: MemoryNamespace | None = None


class FeedbackRequest(BaseModel):
    query_id: str
    helpful: bool
    comment: str | None = None
    namespace: MemoryNamespace | None = None


class BackupRequest(BaseModel):
    destination: str


class AgentEventRequest(AgentEvent):
    pass


class ExtractRequest(BaseModel):
    episode_id: str
    namespace: MemoryNamespace | None = None


class CompileEpisodesRequest(BaseModel):
    episode_ids: list[str] | None = None
    namespace: MemoryNamespace | None = None


class RetractRequest(BaseModel):
    delta_id: str
    reason: str
    namespace: MemoryNamespace | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = Field(default=None, ge=1, le=100)
    episode_prefix: str | None = None
    scope: dict[str, Any] | None = None
    include_diagnostics: bool = False
    namespace: MemoryNamespace | None = None


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, ge=1, le=100)
    episode_prefix: str | None = None
    scope: dict[str, Any] | None = None
    abstain_threshold: float | None = Field(default=None, ge=0, le=1)
    namespace: MemoryNamespace | None = None


class ProfileRequest(BaseModel):
    scope: dict[str, Any] | None = None
    query: str | None = None
    namespace: MemoryNamespace | None = None


class QueryRunRequest(BaseModel):
    query: str = Field(min_length=1, max_length=100_000)
    mode: Literal["query", "search"] = "query"
    top_k: int | None = Field(default=None, ge=1, le=100)
    episode_prefix: str | None = None
    scope: dict[str, Any] | None = None
    abstain_threshold: float | None = Field(default=None, ge=0, le=1)
    namespace: MemoryNamespace | None = None


class QueryRunRemoveBody(QueryRunRemoveRequest):
    namespace: MemoryNamespace | None = None


class SnapshotRequest(BaseModel):
    label: str | None = None


class ModelConfigurationRequest(BaseModel):
    completion: ProviderSelection
    embedding: ProviderSelection
    revision: int = Field(ge=0)
    confirm_embedding_rebuild: bool = False


class ModelConnectionTestRequest(BaseModel):
    role: ProviderRole
    selection: ProviderSelection | None = None


class BenchmarkRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark: Literal["locomo", "longmemeval", "halumem", "memorydoc"]
    mode: Literal["retrieval", "end-to-end"] = "retrieval"
    raw_data_root: str | None = None
    limit: int | None = Field(default=None, ge=1)
    sampling: Literal["auto", "stratified", "sequential"] = "auto"
    judge: ProviderSelection | None = None


def create_app(
    storage_root: str | Path = "~/.trisynapse-memory/store",
    *,
    engine: MemoryEngine | None = None,
    api_key: str | None = None,
    api_keys: dict[str, MemoryNamespace | dict[str, Any]] | None = None,
    studio: bool = False,
) -> FastAPI:
    root = Path(storage_root).expanduser()
    memory = engine or MemoryEngine.from_env(root)
    jobs: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()
    worker_stop = threading.Event()

    def worker_loop() -> None:
        while not worker_stop.wait(1.0):
            memory.run_jobs(max_jobs=10)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        worker = threading.Thread(target=worker_loop, name="trisynapse-memory-worker", daemon=True)
        worker.start()
        try:
            yield
        finally:
            worker_stop.set()
            worker.join(timeout=2)

    app = FastAPI(title="Trisynapse Memory API", version=MemoryEngine.VERSION, lifespan=lifespan)
    app.state.memory_engine = memory
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "OPTIONS"],
        allow_headers=["*"],
    )

    scoped_keys = {
        key: value if isinstance(value, MemoryNamespace) else MemoryNamespace.model_validate(value)
        for key, value in (api_keys or {}).items()
    }

    def authorize(authorization: str | None = Header(default=None)) -> MemoryNamespace | None:
        if api_key is None and not scoped_keys:
            return None
        token = authorization.removeprefix("Bearer ") if authorization and authorization.startswith("Bearer ") else None
        if api_key is not None and token == api_key:
            return None
        if token and token in scoped_keys:
            return scoped_keys[token]
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    auth = [Depends(authorize)]

    def authorize_admin(authorization: str | None = Header(default=None)) -> None:
        if api_key is None and not scoped_keys:
            return
        token = authorization.removeprefix("Bearer ") if authorization and authorization.startswith("Bearer ") else None
        if api_key is not None and token == api_key:
            return
        if token and token in scoped_keys:
            raise HTTPException(status_code=403, detail="store-wide administration requires the administrator API key")
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    admin_auth = [Depends(authorize_admin)]

    def requested_namespace(
        requested: MemoryNamespace | None,
        allowed: MemoryNamespace | None,
    ) -> MemoryNamespace | None:
        if allowed is None:
            return requested
        if requested is None:
            return allowed
        allowed_values = allowed.model_dump(exclude_none=True)
        for key, value in allowed_values.items():
            current = getattr(requested, key)
            if current is None:
                setattr(requested, key, value)
            elif current != value:
                raise HTTPException(status_code=403, detail="API key is not authorized for this namespace")
        return requested

    studio_root = Path(__file__).with_name("studio")
    studio_dist = studio_root / "dist"
    studio_assets = studio_dist if studio_dist.is_dir() else studio_root
    if studio and studio_assets.is_dir():
        app.mount("/studio-assets", StaticFiles(directory=studio_assets), name="studio-assets")

    studio_index = studio_assets / "index.html"

    def studio_unavailable() -> HTMLResponse:
        return HTMLResponse(
            content="""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Trisynapse Memory Studio</title>
    <style>
      :root { color-scheme: light; font-family: ui-sans-serif, system-ui, sans-serif; }
      body { margin: 0; background: #f7f7f5; color: #2f3437; }
      main { max-width: 42rem; margin: 12vh auto; padding: 2rem; }
      section { background: #fff; border: 1px solid #e3e2e0; border-radius: 12px; padding: 2rem; }
      h1 { margin-top: 0; font-size: 1.55rem; }
      p { line-height: 1.6; }
      code { background: #f1f1ef; border-radius: 4px; padding: .15rem .35rem; }
      a { color: #2f6f9f; }
    </style>
  </head>
  <body>
    <main>
      <section>
        <h1>Trisynapse Memory Studio</h1>
        <p>The API is running, but the Studio web assets are not installed.</p>
        <p>For a source checkout, run <code>pnpm --filter @trisynapse/studio build</code>
        and restart the server. For an installed release, reinstall
        <code>trisynapse-memory[all]</code>.</p>
        <p><a href="/api/v1/health">API health</a> · <a href="/docs">OpenAPI documentation</a></p>
      </section>
    </main>
  </body>
</html>
""",
            status_code=200,
        )

    @app.get("/", response_model=None)
    def root_info() -> dict[str, Any] | HTMLResponse | RedirectResponse:
        if studio:
            if studio_index.is_file():
                return RedirectResponse("/studio/", status_code=307)
            return studio_unavailable()
        return {
            "name": "trisynapse-memory",
            "version": MemoryEngine.VERSION,
            "studio": "disabled",
            "api": "/api/v1/health",
            "openapi": "/openapi.json",
        }

    @app.get("/studio", response_model=None)
    def studio_redirect() -> HTMLResponse | RedirectResponse:
        if not studio:
            raise HTTPException(status_code=404, detail="Studio is not enabled")
        if not studio_index.is_file():
            return studio_unavailable()
        return RedirectResponse("/studio/", status_code=307)

    @app.get("/studio/{studio_path:path}", response_model=None)
    def studio_app(studio_path: str = "") -> FileResponse | HTMLResponse:
        del studio_path
        if not studio:
            raise HTTPException(status_code=404, detail="Studio is not enabled")
        if not studio_index.is_file():
            return studio_unavailable()
        return FileResponse(studio_index)

    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        storage_ready = memory.store.is_ready()
        return {
            "status": "ready" if storage_ready else "degraded",
            "version": MemoryEngine.VERSION,
            "store_path": memory.store_path,
            "storage_ready": storage_ready,
            "pending_jobs": len(memory.list_jobs(status="pending")),
        }

    @app.get("/api/health")
    def legacy_health() -> dict[str, Any]:
        return {"status": "ready", "storage_root": str(root.resolve())}

    @app.get("/api/v1/config", dependencies=admin_auth)
    def config() -> dict[str, Any]:
        model_status = memory.get_model_configuration_status()
        return {
            "version": MemoryEngine.VERSION,
            "store_path": memory.store_path,
            "embedding_model": model_status.configuration.embedding.model,
            "completion_configured": memory.completion is not None,
            "model_configuration": model_status.model_dump(mode="json"),
            "storage_ready": memory.store.is_ready(),
        }

    @app.get("/api/v1/check", dependencies=admin_auth)
    def check() -> dict[str, Any]:
        return memory.check()

    @app.get("/api/v1/session", dependencies=auth)
    def session(allowed: MemoryNamespace | None = Depends(authorize)) -> dict[str, Any]:
        if allowed is not None:
            role = "scoped"
            effective = allowed
        elif api_key is None and not scoped_keys:
            role = "open"
            effective = memory.default_namespace
        else:
            role = "admin"
            effective = memory.default_namespace
        return {
            "role": role,
            "effective_namespace": effective.model_dump(mode="json"),
            "capabilities": {
                "admin": role in {"admin", "open"},
                "source_download": True,
                "query_history": True,
                "model_configuration": role in {"admin", "open"},
            },
        }

    @app.get("/api/v1/retrieval-configuration", dependencies=admin_auth)
    def retrieval_configuration() -> dict[str, Any]:
        return memory.get_retrieval_configuration().model_dump(mode="json")

    @app.put("/api/v1/retrieval-configuration", dependencies=admin_auth)
    def update_retrieval_configuration(
        request: RetrievalConfiguration,
    ) -> dict[str, Any]:
        try:
            return memory.set_retrieval_configuration(
                request, expected_revision=request.revision
            ).model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/v1/providers", dependencies=admin_auth)
    def providers() -> dict[str, Any]:
        return {
            "providers": [item.model_dump(mode="json") for item in memory.list_providers()]
        }

    @app.get("/api/v1/providers/{provider}/models", dependencies=admin_auth)
    def provider_models(
        provider: str,
        role: ProviderRole,
        refresh: bool = False,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        try:
            values = memory.list_models(role, provider, refresh=refresh, base_url=base_url)
            return {"models": [item.model_dump(mode="json") for item in values]}
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/v1/model-configuration", dependencies=admin_auth)
    def model_configuration() -> dict[str, Any]:
        return memory.get_model_configuration_status().model_dump(mode="json")

    @app.put("/api/v1/model-configuration", dependencies=admin_auth)
    def update_model_configuration(
        request: ModelConfigurationRequest,
        response: Response,
    ) -> dict[str, Any]:
        configuration = ModelConfiguration(
            completion=request.completion,
            embedding=request.embedding,
            revision=request.revision,
        )
        try:
            result = memory.set_model_configuration(
                configuration,
                confirm_embedding_rebuild=request.confirm_embedding_rebuild,
                expected_revision=request.revision,
            )
        except EmbeddingRebuildRequired as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            status = 409 if "revision conflict" in str(exc) or "already pending" in str(exc) else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        if result.status == "rebuild_pending":
            response.status_code = 202
        return result.model_dump(mode="json")

    @app.post("/api/v1/model-configuration/test", dependencies=admin_auth)
    def test_model_connection(request: ModelConnectionTestRequest) -> dict[str, Any]:
        try:
            return memory.test_model_connection(
                request.role, request.selection
            ).model_dump(mode="json")
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/v1/metrics", dependencies=admin_auth)
    def metrics() -> dict[str, Any]:
        job_counts = {
            status: len(memory.list_jobs(status=status, limit=500))
            for status in ("pending", "running", "completed", "failed")
        }
        return {
            "trace_deltas": memory.store.delta_count(),
            "storage_ready": memory.store.is_ready(),
            "episode_recall_views": len(memory.store.episode_recall_views()),
            "jobs": job_counts,
        }

    @app.get("/api/v1/openapi.json")
    def openapi_alias() -> dict[str, Any]:
        return app.openapi()

    @app.post("/api/v1/memory/observations", dependencies=auth)
    def add_observation(
        request: ObservationRequest,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        payload = request.model_dump()
        payload["namespace"] = requested_namespace(request.namespace, allowed)
        delta = memory.ingest_observation(**payload)
        return {"delta_id": delta.id}

    @app.post("/api/v1/memory/messages", dependencies=auth)
    def add_messages(
        request: MessagesRequest,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        payload = request.model_dump()
        payload["namespace"] = requested_namespace(request.namespace, allowed)
        deltas = memory.ingest_messages(**payload)
        return {"delta_ids": [item.id for item in deltas], "extraction_status": "completed" if request.run_extraction else "not_requested"}

    @app.post("/api/v1/memory/documents", dependencies=auth)
    def add_document(
        request: DocumentRequest,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        namespace = requested_namespace(request.namespace, allowed)
        result = memory.ingest(SourceInput(
            kind="text", text=request.text, filename=f"{request.document_id}.txt",
            source_key=request.document_id, title=request.title, scope=request.scope or {},
        ), namespace=namespace)
        return {"episode_id": result.episode_id, "source_id": result.source_id, "chunk_count": len(result.delta_ids), "delta_ids": result.delta_ids}

    @app.post("/api/v1/memory/files", dependencies=auth)
    def add_file(
        request: FileRequest,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        if len(request.content_base64) > (MAX_FILE_BYTES * 4 // 3) + 8:
            raise HTTPException(status_code=413, detail="file exceeds the 25 MiB v0 upload limit")
        try:
            content = base64.b64decode(request.content_base64, validate=True)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="content_base64 is invalid") from exc
        if len(content) > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail="file exceeds the 25 MiB v0 upload limit")
        try:
            result = memory.ingest(SourceInput(
                kind="image" if Path(request.filename).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} else "file",
                content_base64=request.content_base64,
                filename=request.filename,
                source_key=request.document_id or request.filename,
                title=request.title or request.filename,
                scope=request.scope or {},
            ), namespace=requested_namespace(request.namespace, allowed))
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"document_id": request.document_id or Path(request.filename).stem, "source_id": result.source_id, "chunk_count": len(result.delta_ids), "delta_ids": result.delta_ids}

    @app.post("/api/v1/memories/batch", dependencies=auth)
    def add_batch(
        request: BatchMemoryRequest,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        payloads = []
        for item in request.items:
            payload = item.model_dump()
            payload["namespace"] = requested_namespace(item.namespace, allowed)
            payloads.append(payload)
        deltas = memory.add_batch(payloads)
        return {"delta_ids": [item.id for item in deltas]}

    @app.get("/api/v1/memories", dependencies=auth)
    def list_memories(
        user_id: str | None = None,
        agent_id: str | None = None,
        project_id: str = "default",
        session_id: str | None = None,
        kind: list[str] | None = Query(default=None),
        cursor: int | None = None,
        limit: int = Query(default=50, ge=1, le=500),
        include_retracted: bool = False,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        namespace = MemoryNamespace(user_id=user_id, agent_id=agent_id, project_id=project_id, session_id=session_id)
        page = memory.list(
            namespace=requested_namespace(namespace, allowed), kinds=kind, cursor=cursor,
            limit=limit, include_retracted=include_retracted,
        )
        return page.model_dump(mode="json")

    @app.get("/api/v1/memories/{delta_id}", dependencies=auth)
    def get_memory(
        delta_id: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        project_id: str = "default",
        session_id: str | None = None,
        include_retracted: bool = False,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        namespace = MemoryNamespace(user_id=user_id, agent_id=agent_id, project_id=project_id, session_id=session_id)
        try:
            return memory.get(
                delta_id,
                namespace=requested_namespace(namespace, allowed),
                include_retracted=include_retracted,
            ).model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/memories/{delta_id}/history", dependencies=auth)
    def memory_history(
        delta_id: str,
        project_id: str = "default",
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        try:
            return memory.history(
                delta_id,
                namespace=requested_namespace(MemoryNamespace(project_id=project_id), allowed),
            ).model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/v1/memories/{delta_id}/corrections", dependencies=auth)
    def correct_memory(
        delta_id: str,
        request: CorrectionRequest,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        try:
            delta = memory.correct(
                delta_id=delta_id,
                text=request.text,
                reason=request.reason,
                requested_by=request.requested_by,
                namespace=requested_namespace(request.namespace, allowed),
            )
            return {"correction_id": delta.id}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/v1/memories/{delta_id}/forget", dependencies=auth)
    def forget_memory(
        delta_id: str,
        request: ForgetRequest,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        try:
            delta = memory.forget(
                delta_id=delta_id,
                reason=request.reason,
                requested_by=request.requested_by,
                namespace=requested_namespace(request.namespace, allowed),
            )
            return {"retraction_id": delta.id}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/v1/memory/remove", dependencies=auth)
    def remove_memory(
        request: RemoveRequest,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        if not request.confirm:
            raise HTTPException(status_code=400, detail="remove requires confirm=true")
        try:
            return memory.remove(
                delta_ids=request.delta_ids,
                reason=request.reason,
                requested_by=request.requested_by,
                namespace=requested_namespace(request.namespace, allowed),
            ).model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/v1/sources/ingest", status_code=202, dependencies=auth)
    def ingest_sources(
        request: SourceIngestRequest,
        background_tasks: BackgroundTasks,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        if any(item.path is not None for item in request.sources):
            raise HTTPException(status_code=400, detail="REST source ingestion does not accept server filesystem paths")
        estimated_bytes = 0
        for item in request.sources:
            if item.content_base64 is not None:
                if len(item.content_base64) > (MAX_FILE_BYTES * 4 // 3) + 8:
                    raise HTTPException(status_code=413, detail="uploaded source exceeds the 25 MiB file limit")
                estimated_bytes += len(item.content_base64) * 3 // 4
            elif item.text is not None:
                size = len(item.text.encode("utf-8"))
                if size > MAX_FILE_BYTES:
                    raise HTTPException(status_code=413, detail="text source exceeds the 25 MiB file limit")
                estimated_bytes += size
        if estimated_bytes > MAX_RUN_BYTES:
            raise HTTPException(status_code=413, detail="ingestion run exceeds the 250 MiB retained-source limit")
        namespace = requested_namespace(request.namespace, allowed)
        try:
            run = memory.create_ingestion_run(request.sources, namespace=namespace)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        background_tasks.add_task(memory.process_ingestion_run, run.id)
        return run.model_dump(mode="json")

    @app.get("/api/v1/sources", dependencies=auth)
    def list_sources(
        include_removed: bool = False,
        q: str | None = None,
        kind: list[str] = Query(default=[]),
        status: str | None = None,
        sort: str = "newest",
        cursor: int = Query(default=0, ge=0),
        limit: int = Query(default=48, ge=1, le=200),
        created_from: str | None = None,
        created_to: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        project_id: str = "default",
        session_id: str | None = None,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        namespace = requested_namespace(MemoryNamespace(user_id=user_id, agent_id=agent_id, project_id=project_id, session_id=session_id), allowed)
        try:
            values = memory.list_sources(
                namespace=namespace, include_removed=include_removed, search=q,
                kinds=kind, status=status, sort=sort,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if created_from:
            values = [item for item in values if item.created_at.isoformat() >= created_from]
        if created_to:
            values = [item for item in values if item.created_at.isoformat() <= created_to]
        facets: dict[str, int] = {}
        for item in values:
            facets[item.kind] = facets.get(item.kind, 0) + 1
        page = values[cursor:cursor + limit]
        return {
            "sources": [item.model_dump(mode="json") for item in page],
            "next_cursor": cursor + limit if cursor + limit < len(values) else None,
            "total": len(values),
            "facets": facets,
        }

    @app.get("/api/v1/ingestion-runs", dependencies=auth)
    def list_ingestion_runs(
        limit: int = Query(default=100, ge=1, le=500),
        user_id: str | None = None,
        agent_id: str | None = None,
        project_id: str = "default",
        session_id: str | None = None,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        namespace = requested_namespace(
            MemoryNamespace(user_id=user_id, agent_id=agent_id, project_id=project_id, session_id=session_id),
            allowed,
        )
        return {"runs": [item.model_dump(mode="json") for item in memory.list_ingestion_runs(namespace=namespace, limit=limit)]}

    @app.get("/api/v1/sources/{source_id}", dependencies=auth)
    def get_source(
        source_id: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        project_id: str = "default",
        session_id: str | None = None,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        namespace = requested_namespace(MemoryNamespace(user_id=user_id, agent_id=agent_id, project_id=project_id, session_id=session_id), allowed)
        try:
            return memory.get_source(source_id, namespace=namespace).model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/sources/{source_id}/preview", dependencies=auth)
    def preview_source(
        source_id: str,
        cursor: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200),
        user_id: str | None = None,
        agent_id: str | None = None,
        project_id: str = "default",
        session_id: str | None = None,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        namespace = requested_namespace(
            MemoryNamespace(user_id=user_id, agent_id=agent_id, project_id=project_id, session_id=session_id),
            allowed,
        )
        try:
            return memory.source_preview(
                source_id, cursor=cursor, limit=limit, namespace=namespace
            ).model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/sources/{source_id}/content", dependencies=auth, response_model=None)
    def source_content(
        source_id: str,
        disposition: Literal["inline", "attachment"] = "attachment",
        user_id: str | None = None,
        agent_id: str | None = None,
        project_id: str = "default",
        session_id: str | None = None,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> FileResponse:
        namespace = requested_namespace(
            MemoryNamespace(user_id=user_id, agent_id=agent_id, project_id=project_id, session_id=session_id),
            allowed,
        )
        try:
            source, path = memory.source_content_path(source_id, namespace=namespace)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        safe_inline = source.media_type.startswith("image/") or source.media_type == "application/pdf" or (
            source.media_type.startswith("text/") and source.media_type not in {"text/html", "application/xhtml+xml"}
        )
        content_disposition = "inline" if disposition == "inline" and safe_inline else "attachment"
        return FileResponse(
            path,
            media_type=source.media_type,
            filename=source.filename or source.title,
            content_disposition_type=content_disposition,
            headers={
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "private, no-store",
                "Content-Security-Policy": "sandbox; default-src 'none'",
            },
        )

    @app.get("/api/v1/ingestion-runs/{run_id}", dependencies=auth)
    def get_ingestion_run(run_id: str, allowed: MemoryNamespace | None = Depends(authorize)) -> dict[str, Any]:
        try:
            run = memory.get_ingestion_run(run_id)
            if allowed is not None and run.namespace != allowed:
                raise KeyError(run_id)
            return run.model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/v1/ingestion-runs/{run_id}/retry", status_code=202, dependencies=auth)
    def retry_ingestion_run(
        run_id: str,
        background_tasks: BackgroundTasks,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        try:
            original = memory.get_ingestion_run(run_id)
            if allowed is not None and original.namespace != allowed:
                raise KeyError(run_id)
            retry = memory.create_retry_ingestion(run_id)
            background_tasks.add_task(memory.process_ingestion_run, retry.id)
            return retry.model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/sources/{source_id}/remove", dependencies=auth)
    def remove_source(
        source_id: str,
        request: SourceRemoveRequest,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        namespace = requested_namespace(request.namespace, allowed)
        try:
            return memory.remove_source(source_id, reason=request.reason, requested_by=request.requested_by, namespace=namespace).model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/memory/export", dependencies=auth)
    def export_memory(
        project_id: str = "default",
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        return memory.export(namespace=requested_namespace(MemoryNamespace(project_id=project_id), allowed))

    @app.post("/api/v1/memory/extract", dependencies=auth)
    def extract(
        request: ExtractRequest,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        try:
            deltas = memory.run_extraction(
                episode_id=request.episode_id,
                namespace=requested_namespace(request.namespace, allowed),
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"extraction_count": len(deltas), "delta_ids": [item.id for item in deltas]}

    @app.post("/api/v1/memory/episodes/compile", dependencies=auth)
    def compile_episodes(
        request: CompileEpisodesRequest,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        views = memory.build_episode_recall(
            request.episode_ids,
            namespace=requested_namespace(request.namespace, allowed),
        )
        return {"episode_recall_entries": [item.model_dump(mode="json") for item in views]}

    @app.post("/api/v1/memory/retract", dependencies=auth)
    def retract(
        request: RetractRequest,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        try:
            payload = request.model_dump()
            payload["namespace"] = requested_namespace(request.namespace, allowed)
            delta = memory.retract(**payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"retraction_id": delta.id}

    @app.post("/api/v1/search", dependencies=auth)
    def search(
        request: SearchRequest,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        result = memory.search(
            request.query,
            top_k=request.top_k,
            episode_prefix=request.episode_prefix,
            scope=request.scope,
            namespace=requested_namespace(request.namespace, allowed),
        )
        payload = result.model_dump(mode="json")
        if not request.include_diagnostics:
            payload["retrieval_trace"].pop("routes", None)
        return payload

    @app.post("/api/v1/query", dependencies=auth)
    def query(
        request: QueryRequest,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        payload = request.model_dump()
        payload["namespace"] = requested_namespace(request.namespace, allowed)
        return memory.query(**payload).model_dump(mode="json")

    @app.post("/api/v1/query-runs", status_code=202, dependencies=auth)
    def create_query_run(
        request: QueryRunRequest,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        namespace = requested_namespace(request.namespace, allowed)
        return memory.create_query_run(
            request.query,
            mode=request.mode,
            top_k=request.top_k,
            episode_prefix=request.episode_prefix,
            scope=request.scope,
            abstain_threshold=request.abstain_threshold,
            namespace=namespace,
        ).model_dump(mode="json")

    @app.get("/api/v1/query-runs", dependencies=auth)
    def list_query_runs(
        limit: int = Query(default=50, ge=1, le=200),
        cursor: str | None = None,
        q: str | None = None,
        mode: Literal["query", "search"] | None = None,
        status: str | None = None,
        stage: str | None = None,
        before: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        project_id: str = "default",
        session_id: str | None = None,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        namespace = requested_namespace(
            MemoryNamespace(user_id=user_id, agent_id=agent_id, project_id=project_id, session_id=session_id),
            allowed,
        )
        parsed_before = datetime.fromisoformat(before.replace("Z", "+00:00")) if before else None
        return memory.list_query_runs(
            namespace=namespace, limit=limit, cursor=cursor, search=q,
            mode=mode, status=status, stage=stage, before=parsed_before,
        ).model_dump(mode="json")

    @app.post("/api/v1/query-runs/remove", dependencies=auth)
    def remove_query_runs(
        request: QueryRunRemoveBody,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        if not request.confirm:
            raise HTTPException(status_code=400, detail="query history removal requires confirm=true")
        if not request.query_ids and request.before is None and not request.all_in_namespace:
            raise HTTPException(status_code=400, detail="select query IDs, a before date, or all_in_namespace")
        removed = memory.remove_query_runs(
            query_ids=request.query_ids,
            before=request.before,
            all_in_namespace=request.all_in_namespace,
            namespace=requested_namespace(request.namespace, allowed),
        )
        return {"removed_query_ids": removed}

    @app.get("/api/v1/query-runs/{query_id}", dependencies=auth)
    def get_query_run(
        query_id: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        project_id: str = "default",
        session_id: str | None = None,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        namespace = requested_namespace(
            MemoryNamespace(user_id=user_id, agent_id=agent_id, project_id=project_id, session_id=session_id),
            allowed,
        )
        try:
            return memory.get_query_run(query_id, namespace=namespace).model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/query-runs/{query_id}/events", dependencies=auth, response_model=None)
    def query_run_events(
        query_id: str,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
        user_id: str | None = None,
        agent_id: str | None = None,
        project_id: str = "default",
        session_id: str | None = None,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> StreamingResponse:
        namespace = requested_namespace(
            MemoryNamespace(user_id=user_id, agent_id=agent_id, project_id=project_id, session_id=session_id),
            allowed,
        )
        try:
            memory.get_query_run(query_id, namespace=namespace)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        def events():
            previous = last_event_id
            heartbeat_at = time.monotonic()
            for _ in range(3600):
                try:
                    run = memory.get_query_run(query_id, namespace=namespace)
                except KeyError:
                    return
                event_id = f"{run.updated_at.isoformat()}:{len(run.steps)}:{run.status}"
                if event_id != previous:
                    payload = json.dumps(run.model_dump(mode="json"), ensure_ascii=False)
                    yield f"id: {event_id}\nevent: run\ndata: {payload}\n\n"
                    previous = event_id
                    heartbeat_at = time.monotonic()
                if run.status in {"completed", "failed", "interrupted"}:
                    yield f"event: complete\ndata: {json.dumps({'status': run.status})}\n\n"
                    return
                if time.monotonic() - heartbeat_at >= 15:
                    yield ": heartbeat\n\n"
                    heartbeat_at = time.monotonic()
                time.sleep(0.25)

        return StreamingResponse(
            events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/v1/query-runs/{query_id}/remove", dependencies=auth)
    def remove_query_run(
        query_id: str,
        request: QueryRunRemoveBody,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        if not request.confirm:
            raise HTTPException(status_code=400, detail="query history removal requires confirm=true")
        removed = memory.remove_query_runs(
            query_ids=[query_id], namespace=requested_namespace(request.namespace, allowed)
        )
        if not removed:
            raise HTTPException(status_code=404, detail=f"unknown query run: {query_id}")
        return {"removed_query_ids": removed}

    @app.post("/api/v1/profile", dependencies=auth)
    def profile(
        request: ProfileRequest,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        payload = request.model_dump()
        payload["namespace"] = requested_namespace(request.namespace, allowed)
        return memory.compile_profile(**payload)

    @app.post("/api/v1/feedback", dependencies=auth)
    def feedback(
        request: FeedbackRequest,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        try:
            delta = memory.record_feedback(
                request.query_id,
                helpful=request.helpful,
                comment=request.comment,
                namespace=requested_namespace(request.namespace, allowed),
            )
            return {"feedback_id": delta.id}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/v1/integrations/events", dependencies=auth)
    def agent_event(
        request: AgentEventRequest,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        event_namespace = MemoryNamespace(
            user_id=request.user_id,
            agent_id=request.agent_id,
            project_id=request.project_id,
            session_id=request.session_id,
        )
        requested_namespace(event_namespace, allowed)
        return capture_agent_event(memory, request)

    @app.get("/api/v1/deltas/{delta_id}", dependencies=auth)
    def delta(
        delta_id: str,
        project_id: str = "default",
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        try:
            namespace = requested_namespace(MemoryNamespace(project_id=project_id), allowed)
            return memory.get(delta_id, namespace=namespace, include_retracted=True).model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/episodes", dependencies=auth)
    def episodes(
        project_id: str = "default",
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        namespace = requested_namespace(MemoryNamespace(project_id=project_id), allowed)
        return {"episodes": [item.model_dump(mode="json") for item in memory.list_episodes(namespace=namespace)]}

    @app.get("/api/v1/graph", dependencies=auth)
    def graph(
        project_id: str = "default",
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        namespace = requested_namespace(MemoryNamespace(project_id=project_id), allowed)
        return memory.export_graph(format="json", namespace=namespace)

    @app.get("/api/v1/memory-graph", dependencies=auth)
    def memory_graph(
        view: Literal["knowledge", "lineage", "trace"] = "knowledge",
        q: str | None = None,
        node_type: list[str] = Query(default=[]),
        source_id: str | None = None,
        episode_id: str | None = None,
        limit: int = Query(default=500, ge=1, le=2000),
        cursor: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        project_id: str = "default",
        session_id: str | None = None,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        namespace = requested_namespace(
            MemoryNamespace(user_id=user_id, agent_id=agent_id, project_id=project_id, session_id=session_id),
            allowed,
        )
        try:
            return memory.memory_graph(
                view, search=q, node_types=node_type, source_id=source_id,
                episode_id=episode_id, limit=limit, cursor=cursor, namespace=namespace,
            ).model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/v1/memory-graph/nodes/{node_id}/neighbors", dependencies=auth)
    def memory_graph_neighbors(
        node_id: str,
        view: Literal["knowledge", "lineage", "trace"] = "lineage",
        limit: int = Query(default=500, ge=1, le=2000),
        user_id: str | None = None,
        agent_id: str | None = None,
        project_id: str = "default",
        session_id: str | None = None,
        allowed: MemoryNamespace | None = Depends(authorize),
    ) -> dict[str, Any]:
        namespace = requested_namespace(
            MemoryNamespace(user_id=user_id, agent_id=agent_id, project_id=project_id, session_id=session_id),
            allowed,
        )
        return memory.memory_graph_neighbors(
            node_id, view=view, limit=limit, namespace=namespace
        ).model_dump(mode="json")

    @app.get("/api/v1/jobs", dependencies=admin_auth)
    def memory_jobs(status: str | None = None, limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
        return {"jobs": [item.model_dump(mode="json") for item in memory.list_jobs(status=status, limit=limit)]}

    @app.post("/api/v1/jobs/run", dependencies=admin_auth)
    def run_jobs(limit: int = Query(default=100, ge=1, le=1000)) -> dict[str, Any]:
        values = memory.run_jobs(max_jobs=limit)
        return {"jobs": [item.model_dump(mode="json") for item in values]}

    @app.post("/api/v1/admin/backup", dependencies=admin_auth)
    def backup(request: BackupRequest) -> dict[str, Any]:
        return {"path": str(memory.backup(request.destination))}

    @app.get("/api/v1/snapshots", dependencies=admin_auth)
    def snapshots() -> dict[str, Any]:
        return {"snapshots": [item.model_dump(mode="json") for item in memory.snapshot.list()]}

    @app.post("/api/v1/snapshots", dependencies=admin_auth)
    def create_snapshot(request: SnapshotRequest) -> dict[str, Any]:
        return memory.snapshot.create(request.label).model_dump(mode="json")

    @app.get("/api/v1/snapshots/{snapshot_id}/diff", dependencies=admin_auth)
    def snapshot_diff(snapshot_id: str, against: str) -> dict[str, Any]:
        try:
            return memory.snapshot.diff(snapshot_id, against).model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    _add_benchmark_routes(app, root, jobs, lock, admin_auth, memory)
    return app


def _add_benchmark_routes(
    app: FastAPI,
    root: Path,
    jobs: dict[str, dict[str, Any]],
    lock: threading.Lock,
    admin_auth: list[Any],
    memory: MemoryEngine,
) -> None:
    @app.get("/api/benchmarks/runs", dependencies=admin_auth)
    @app.get("/api/v1/benchmarks/runs", dependencies=admin_auth)
    def benchmark_runs() -> dict[str, Any]:
        return {"runs": discover_benchmark_runs(root)}

    @app.get("/api/benchmarks/catalog", dependencies=admin_auth)
    def benchmark_catalog() -> dict[str, Any]:
        runs = discover_benchmark_runs(root)
        return {
            "schema_version": 1,
            "benchmarks": [
                {"name": name, "runs": [item for item in runs if item.get("benchmark") == name]}
                for name in sorted({str(item.get("benchmark")) for item in runs if item.get("benchmark")})
            ],
        }

    @app.get("/api/benchmarks/runs/{benchmark}/{run_id}", dependencies=admin_auth)
    @app.get("/api/v1/benchmarks/runs/{benchmark}/{run_id}", dependencies=admin_auth)
    def benchmark_run(benchmark: str, run_id: str) -> dict[str, Any]:
        if not _safe_segment(benchmark) or not _safe_segment(run_id):
            raise HTTPException(status_code=400, detail="invalid benchmark or run id")
        try:
            return read_benchmark_run(root, benchmark, run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="run not found")

    @app.post("/api/benchmarks/runs", status_code=202, dependencies=admin_auth)
    @app.post("/api/v1/benchmarks/runs", status_code=202, dependencies=admin_auth)
    def start_benchmark(request: BenchmarkRunRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        with lock:
            jobs[job_id] = {"job_id": job_id, "status": "queued", "request": request.model_dump()}
        background_tasks.add_task(
            _execute_benchmark, job_id, request, root, jobs, lock,
            memory.get_model_configuration(),
        )
        return jobs[job_id]

    @app.get("/api/jobs/{job_id}", dependencies=admin_auth)
    def job(job_id: str) -> dict[str, Any]:
        with lock:
            payload = jobs.get(job_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="job not found")
        return payload


def _execute_benchmark(
    job_id: str,
    request: BenchmarkRunRequest,
    storage_root: Path,
    jobs: dict[str, dict[str, Any]],
    lock: threading.Lock,
    model_configuration: ModelConfiguration,
) -> None:
    with lock:
        jobs[job_id]["status"] = "running"
    try:
        if request.benchmark in {"locomo", "longmemeval"}:
            raw_data = request.raw_data_root or f"data/{request.benchmark}"
            summary = run_trace_recall_benchmark(
                request.benchmark,
                raw_data,
                max_questions=request.limit or 25,
                output_root=storage_root / request.benchmark / "runs",
                mode=request.mode,
                model_configuration=model_configuration,
                sampling=request.sampling,
                judge_selection=request.judge,
            )
            with lock:
                jobs[job_id].update(status="completed", summary=summary)
            return
        raw_data = request.raw_data_root or f"data/{request.benchmark}"
        summary = run_trace_recall_smoke(
            request.benchmark,
            raw_data,
            max_questions=request.limit or 25,
            output_root=storage_root / request.benchmark / "runs",
            mode=request.mode,
            model_configuration=model_configuration,
            sampling=request.sampling,
            judge_selection=request.judge,
        )
        with lock:
            jobs[job_id].update(status="completed", summary=summary)
    except Exception as exc:
        with lock:
            jobs[job_id].update(status="failed", error=str(exc))


def _safe_segment(value: str) -> bool:
    return bool(value) and value not in {".", ".."} and "/" not in value and "\\" not in value
