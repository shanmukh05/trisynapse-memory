"""Registered index helpers that Studio inspects without a closed tab enum."""

from __future__ import annotations

from dataclasses import dataclass

from trisynapse_memory.engine.models import MemoryHelperKind


@dataclass(frozen=True)
class IndexHelperSpec:
    id: str
    title: str
    kind: MemoryHelperKind
    inspect_path: str
    playground_seed: str | None = None


BUILTIN_INDEX_HELPERS: tuple[IndexHelperSpec, ...] = (
    IndexHelperSpec(
        id="log",
        title="Memory log",
        kind="timeline",
        inspect_path="/api/v1/memories",
        playground_seed="excerpt",
    ),
    IndexHelperSpec(
        id="documents",
        title="Documents",
        kind="table",
        inspect_path="/api/v1/memory/documents",
        playground_seed="excerpt",
    ),
    IndexHelperSpec(
        id="bm25",
        title="BM25",
        kind="postings",
        inspect_path="/api/v1/memory/terms",
        playground_seed="term",
    ),
    IndexHelperSpec(
        id="vectors",
        title="Embedding index",
        kind="embedding",
        inspect_path="/api/v1/memory/vectors/projection",
        playground_seed="excerpt",
    ),
    IndexHelperSpec(
        id="groups",
        title="Group summaries",
        kind="cards",
        inspect_path="/api/v1/groups",
        playground_seed="summary",
    ),
    IndexHelperSpec(
        id="claims",
        title="Claims",
        kind="table",
        inspect_path="/api/v1/memory/claims",
        playground_seed="text",
    ),
    IndexHelperSpec(
        id="graph",
        title="Graph",
        kind="graph",
        inspect_path="/api/v1/memory/retrieval-graph",
        playground_seed="label",
    ),
)


def builtin_helper_ids() -> set[str]:
    return {item.id for item in BUILTIN_INDEX_HELPERS}


def helper_spec(helper_id: str) -> IndexHelperSpec | None:
    return next((item for item in BUILTIN_INDEX_HELPERS if item.id == helper_id), None)
