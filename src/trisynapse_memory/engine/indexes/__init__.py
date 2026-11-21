"""Rebuildable search helpers over the memory log.

Re-exports claim compilation and the embedding index. Inspection pages live in
``inspect.py``; the helper catalog lives in ``catalog.py``. Indexes must remain
disposable: they may be rebuilt, but they are not evidence.
"""

from trisynapse_memory.engine.indexes.claims import (
    build_group_summary_views,
    compile_claims,
)
from trisynapse_memory.engine.indexes.embedding_index import (
    EmbeddingIndex,
    LanceEmbeddingIndex,
    SQLiteEmbeddingIndex,
    preferred_embedding_index,
)

__all__ = [
    "EmbeddingIndex",
    "LanceEmbeddingIndex",
    "SQLiteEmbeddingIndex",
    "build_group_summary_views",
    "compile_claims",
    "preferred_embedding_index",
]
