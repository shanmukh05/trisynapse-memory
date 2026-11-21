"""Trisynapse's memory backend for the pinned OmniMemEval checkout."""

from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

from trisynapse_memory.engine import MemoryEngine
from trisynapse_memory.engine.providers.embedding import SentenceTransformerEmbedder
from trisynapse_memory.engine.providers.registry import (
    ProviderSettings,
    openai_json_chat_payload,
    parse_json_response,
)

DEFAULT_MEMORY_MODEL = "gpt-4.1-mini-2025-04-14"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_MESSAGES_PER_GROUP = 8
_WRITE_LOCK = threading.Lock()
_EMBEDDING_LOAD_LOCK = threading.Lock()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with _WRITE_LOCK, path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformerEmbedder:
    embedder = SentenceTransformerEmbedder(os.environ.get("TRISYNAPSE_OMNI_EMBEDDING", DEFAULT_EMBEDDING_MODEL))
    # Warm the model once before the harness starts concurrent searches.
    embedder.encode(["Trisynapse"])
    return embedder


class _RecordedCompletion:
    def __init__(self, model: str, path: Path) -> None:
        from openai import OpenAI

        base_url = os.environ["ANSWER_BASE_URL"]
        self.settings = ProviderSettings(provider="openai", model=model, base_url=base_url)
        self.model = model
        self.path = path
        self.client = OpenAI(api_key=os.environ["ANSWER_API_KEY"], base_url=base_url)

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        request = openai_json_chat_payload(
            self.model, [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        started = perf_counter()
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            _append_jsonl(self.path, {
                "time": datetime.now(timezone.utc).isoformat(),
                "duration_ms": (perf_counter() - started) * 1000,
                "request": request,
                "error_type": type(exc).__name__,
            })
            raise
        content = response.choices[0].message.content
        _append_jsonl(self.path, {
            "time": datetime.now(timezone.utc).isoformat(),
            "duration_ms": (perf_counter() - started) * 1000,
            "request": request,
            "response_id": response.id,
            "model": response.model,
            "usage": response.usage.model_dump(mode="json") if response.usage else None,
            "response": content,
        })
        return parse_json_response(content or "")

    __call__ = complete_json

    def close(self) -> None:
        self.client.close()


class TrisynapseClient:
    """Store each benchmark user in its own Trace and return ranked evidence text."""

    def __init__(self) -> None:
        self.root = Path(os.environ["TRISYNAPSE_OMNI_RUN_DIR"]).resolve()
        self.run_version = os.environ["TRISYNAPSE_OMNI_RUN_VERSION"]
        self.memory_version = os.environ.get("TRISYNAPSE_OMNI_MEMORY_VERSION", self.run_version)
        self.model = os.environ.get("TRISYNAPSE_OMNI_MEMORY_MODEL", DEFAULT_MEMORY_MODEL)
        self.retrieval = json.loads((self.root / "retrieval-config.json").read_text(encoding="utf-8"))
        self.memory = self.root / "memory"
        self.memory.mkdir(parents=True, exist_ok=True)

    def _user_store(self, user_id: str) -> Path:
        if self.memory_version != self.run_version:
            suffix = f"_{self.run_version}"
            if not user_id.endswith(suffix):
                raise ValueError(f"Unrecognized LoCoMo user identifier: {user_id}")
            user_id = f"{user_id.removesuffix(suffix)}_{self.memory_version}"
        return self.memory / sha256(user_id.encode()).hexdigest()[:24]

    @contextmanager
    def _engine(self, user_id: str, *, ingest: bool = False) -> Iterator[MemoryEngine]:
        root = self._user_store(user_id)
        root.mkdir(parents=True, exist_ok=True)
        with _EMBEDDING_LOAD_LOCK:
            embedder = _embedder()
        completion = (
            _RecordedCompletion(self.model, root / "memory_model_calls.jsonl")
            if ingest and self.model != "none" else None
        )
        try:
            engine = MemoryEngine.open(root, embedder=embedder, completion=completion, auto_process=False)
            try:
                if ingest:
                    current = engine.get_retrieval_configuration()
                    if current.model_dump(mode="json", exclude={"revision", "updated_at"}) != self.retrieval:
                        engine.set_retrieval_configuration(self.retrieval, expected_revision=current.revision)
                yield engine
            finally:
                engine.close()
        finally:
            if completion is not None:
                completion.close()

    def add(
        self,
        messages: list[dict[str, Any]],
        user_id: str,
        conv_id: str | None = None,
        *,
        session_key: str | None = None,
    ) -> None:
        session = conv_id or session_key
        if not session:
            raise ValueError("OmniMemEval must supply a stable session identifier")
        if not messages:
            return
        with self._engine(user_id, ingest=True) as engine:
            groups: list[str] = []
            for index, message in enumerate(messages):
                content = str(message.get("content") or "")
                if not content.strip():
                    continue
                group = f"{session}:{index // _MESSAGES_PER_GROUP:05d}"
                role = message.get("role") or ""
                locator = {
                    "kind": "message_index",
                    "session_id": session,
                    "message_index": index,
                    "message_id": message.get("id"),
                    "role": role,
                }
                engine.ingest_raw_source(
                    content,
                    group_id=group,
                    source_ref={"type": "chat", "id": session, "user_id": user_id},
                    locator=locator,
                    observed_at=message.get("chat_time"),
                    external_key=f"omnimemeval:{session}:{index}",
                    modality="conversation",
                    source_type="conversation",
                    retrieval_fields={"speaker": message.get("name") or role, "message": content},
                    schedule=False,
                )
                if not groups or group != groups[-1]:
                    groups.append(group)
            if self.model != "none" and groups:
                engine.run_fact_extraction_many(groups)
            if groups:
                engine.build_group_summary(groups)

    def search(self, query: str, user_id: str, top_k: int) -> list[str]:
        if not (self._user_store(user_id) / "trace.sqlite3").is_file():
            raise FileNotFoundError(f"OmniMemEval user was not ingested: {user_id}")
        with self._engine(user_id) as engine:
            result = engine.search(query, top_k=top_k)
        hits = result.hits[:top_k]
        _append_jsonl(self._user_store(user_id) / "searches.jsonl", {
            "user_id": user_id,
            "query": query,
            "query_id": result.query_id,
            "top_k": top_k,
            "returned": [hit.model_dump(mode="json") for hit in hits],
        })
        return [hit.text for hit in hits]

    def delete(self, user_id: str) -> None:
        """Remove a user from the active backend while retaining the run evidence."""
        active = self._user_store(user_id)
        if active.exists():
            retired = self.root / "retired-memory"
            retired.mkdir(exist_ok=True)
            active.rename(retired / f"{active.name}-{uuid.uuid4().hex[:12]}")
