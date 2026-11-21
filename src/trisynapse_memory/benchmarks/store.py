"""Persistent per-model benchmark stores for ingest-once, query-many runs."""

from __future__ import annotations

import json
import re
from hashlib import sha256
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trisynapse_memory.adapters.benchmarks.base import PreparedCase
from trisynapse_memory.engine import MemoryEngine
from trisynapse_memory.engine.models import MemoryNamespace, ModelConfiguration

MANIFEST_FILENAME = "benchmark_store.json"
_STORE_SEGMENT = re.compile(r"[^A-Za-z0-9._+-]+")


def query_checkpoint_path(store_path: str | Path, mode: str) -> Path:
    return Path(store_path) / f"query_checkpoint_{mode}.json"


def load_query_checkpoint(store_path: str | Path, mode: str) -> list[dict[str, Any]]:
    path = query_checkpoint_path(store_path, mode)
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and row.get("question_id")]


def save_query_checkpoint(store_path: str | Path, mode: str, rows: list[dict[str, Any]]) -> None:
    root = Path(store_path)
    root.mkdir(parents=True, exist_ok=True)
    path = query_checkpoint_path(root, mode)
    tmp = path.with_suffix(".json.tmp")
    payload = {
        "schema_version": 1,
        "mode": mode,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def clear_query_checkpoint(store_path: str | Path, mode: str) -> None:
    path = query_checkpoint_path(store_path, mode)
    path.unlink(missing_ok=True)


def dataset_root(data_root: str | Path, dataset_path: Path) -> Path:
    original = Path(data_root)
    return original.parent if original.is_file() else original


def store_segment(value: str | None, default: str = "none") -> str:
    text = (value or default).strip() or default
    cleaned = _STORE_SEGMENT.sub("_", text).strip("._")
    return cleaned or default


def completion_identity(configuration: ModelConfiguration | None) -> tuple[str, str]:
    if configuration is None or configuration.completion.provider == "none":
        return "none", "none"
    return configuration.completion.provider, configuration.completion.model or "none"


def embedding_identity(configuration: ModelConfiguration | None) -> tuple[str, str]:
    if configuration is None:
        return "sentence-transformers", "all-MiniLM-L6-v2"
    return (
        configuration.embedding.provider or "sentence-transformers",
        configuration.embedding.model or "all-MiniLM-L6-v2",
    )


def benchmark_store_path(
    data_root: str | Path,
    dataset_path: Path,
    configuration: ModelConfiguration | None,
    *,
    isolate_dataset: bool = False,
) -> Path:
    """Return ``<data>/stores/<completion-provider>/<completion-model>/<embedding-provider>/<embedding-model>``."""

    completion_provider, completion_model = completion_identity(configuration)
    embedding_provider, embedding_model = embedding_identity(configuration)
    stores = dataset_root(data_root, dataset_path) / "stores"
    if isolate_dataset:
        digest = sha256(str(dataset_path.resolve()).encode()).hexdigest()[:12]
        stores = stores / f"{store_segment(dataset_path.stem)}-{digest}"
    return (
        stores
        / store_segment(completion_provider)
        / store_segment(completion_model)
        / store_segment(embedding_provider)
        / store_segment(embedding_model)
    )


def empty_manifest(
    suite: str,
    configuration: ModelConfiguration | None,
) -> dict[str, Any]:
    completion_provider, completion_model = completion_identity(configuration)
    embedding_provider, embedding_model = embedding_identity(configuration)
    return {
        "schema_version": 1,
        "suite": suite,
        "engine_version": MemoryEngine.VERSION,
        "completion": {"provider": completion_provider, "model": completion_model},
        "embedding": {"provider": embedding_provider, "model": embedding_model},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "cases": {},
    }


def load_manifest(store_path: str | Path) -> dict[str, Any] | None:
    path = Path(store_path) / MANIFEST_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid benchmark store manifest: {path}")
    payload.setdefault("cases", {})
    return payload


def save_manifest(store_path: str | Path, manifest: dict[str, Any]) -> None:
    root = Path(store_path)
    root.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    path = root / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepared_case_from_manifest(entry: dict[str, Any]) -> PreparedCase:
    return PreparedCase(
        tuple(entry["group_ids"]),
        MemoryNamespace.model_validate(entry["namespace"]),
        entry.get("group_prefix"),
    )


def manifest_case_entry(
    prepared: PreparedCase,
    *,
    doc_count: int,
    ingest_time_ms: float,
) -> dict[str, Any]:
    return {
        "group_ids": list(prepared.group_ids),
        "namespace": prepared.namespace.model_dump(mode="json"),
        "group_prefix": prepared.group_prefix,
        "doc_count": doc_count,
        "ingest_time_ms": ingest_time_ms,
        "ingest_time_per_doc_ms": ingest_time_ms / doc_count if doc_count else 0.0,
    }


def embedding_matches_store(manifest: dict[str, Any], configuration: ModelConfiguration | None) -> bool:
    provider, model = embedding_identity(configuration)
    recorded = manifest.get("embedding") or {}
    return recorded.get("provider") == provider and recorded.get("model") == model
