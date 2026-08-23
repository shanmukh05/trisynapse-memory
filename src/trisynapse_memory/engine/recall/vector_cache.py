"""Disposable vector cache with LanceDB as the preferred local backend."""

from __future__ import annotations

import hashlib
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np

from trisynapse_memory.engine.utils import cosine_scores
from trisynapse_memory.engine.trace.store import SQLiteTraceStore


class VectorCache(Protocol):
    def get(self, text_hashes: list[str], model: str) -> dict[str, list[float]]: ...
    def put(self, values: dict[str, list[float]], model: str) -> None: ...
    def nearest(self, vector: list[float], model: str, limit: int) -> list[tuple[str, float]]: ...
    def clear(self) -> None: ...


class SQLiteVectorCache:
    """Portable test/safe-mode cache; the trace remains independent of it."""

    def __init__(self, store: SQLiteTraceStore) -> None:
        self.store = store

    def get(self, text_hashes: list[str], model: str) -> dict[str, list[float]]:
        return self.store.get_embeddings(text_hashes, model)

    def put(self, values: dict[str, list[float]], model: str) -> None:
        self.store.put_embeddings(values, model)

    def clear(self) -> None:
        self.store.clear_embeddings()

    def nearest(self, vector: list[float], model: str, limit: int) -> list[tuple[str, float]]:
        values = self.store.list_embeddings(model)
        compatible = [
            (text_hash, embedding)
            for text_hash, embedding in values.items()
            if len(embedding) == len(vector)
        ]
        if not compatible or not vector or limit <= 0:
            return []
        scores = cosine_scores(vector, [embedding for _, embedding in compatible])
        count = min(limit, len(compatible))
        candidate_indices = (
            np.arange(len(compatible))
            if count == len(compatible)
            else np.argpartition(-scores, count - 1)[:count]
        )
        ordered = sorted(candidate_indices, key=lambda index: float(scores[index]), reverse=True)
        return [(compatible[index][0], float(scores[index])) for index in ordered]


class LanceVectorCache:
    """LanceDB cache partitioned by embedding model and vector dimension."""

    def __init__(self, path: str | Path) -> None:
        try:
            import lancedb
        except ImportError as exc:
            raise RuntimeError("LanceDB is not installed; install the base package dependencies") from exc
        self._db = lancedb.connect(str(Path(path)))

    def get(self, text_hashes: list[str], model: str) -> dict[str, list[float]]:
        if not text_hashes:
            return {}
        table_name = _table_name(model)
        if table_name not in self._table_names():
            return {}
        requested = set(text_hashes)
        rows = self._db.open_table(table_name).to_arrow().to_pylist()
        return {str(row["text_hash"]): list(map(float, row["vector"])) for row in rows if row["text_hash"] in requested}

    def put(self, values: dict[str, list[float]], model: str) -> None:
        if not values:
            return
        table_name = _table_name(model)
        rows = [
            {"text_hash": key, "model": model, "vector": vector, "created_at": datetime.now(timezone.utc).isoformat()}
            for key, vector in values.items()
        ]
        if table_name in self._table_names():
            existing = self.get(list(values), model)
            missing = [row for row in rows if row["text_hash"] not in existing]
            if missing:
                self._db.open_table(table_name).add(missing)
        else:
            self._db.create_table(table_name, data=rows)

    def clear(self) -> None:
        for table_name in self._table_names():
            self._db.drop_table(table_name)

    def nearest(self, vector: list[float], model: str, limit: int) -> list[tuple[str, float]]:
        table_name = _table_name(model)
        if table_name not in self._table_names() or not vector:
            return []
        rows = (
            self._db.open_table(table_name)
            .search(vector)
            .distance_type("cosine")
            .limit(limit)
            .to_list()
        )
        return [
            (str(row["text_hash"]), 1.0 - float(row.get("_distance", 1.0)))
            for row in rows
        ]

    def _table_names(self) -> set[str]:
        response = self._db.list_tables()
        return {str(item) for item in response.tables}


def preferred_vector_cache(store: SQLiteTraceStore) -> VectorCache:
    try:
        return LanceVectorCache(store.root / "vectors.lance")
    except RuntimeError:
        warnings.warn(
            "LanceDB is unavailable; using the explicit SQLite disposable-vector-cache mode. "
            "Embeddings still come from the configured real model.",
            RuntimeWarning,
            stacklevel=2,
        )
        return SQLiteVectorCache(store)


def _table_name(model: str) -> str:
    readable = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")[:30]
    digest = hashlib.sha256(model.encode()).hexdigest()[:10]
    return f"emb_{readable}_{digest}"
