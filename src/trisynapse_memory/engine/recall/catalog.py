"""Registered Recall helpers that Studio inspects without a closed tab enum."""

from __future__ import annotations

from dataclasses import dataclass

from trisynapse_memory.engine.models import MemoryHelperKind


@dataclass(frozen=True)
class RecallHelperSpec:
    id: str
    title: str
    kind: MemoryHelperKind
    inspect_path: str
    playground_seed: str | None = None


BUILTIN_RECALL_HELPERS: tuple[RecallHelperSpec, ...] = (
    RecallHelperSpec(
        id="trace",
        title="Trace",
        kind="timeline",
        inspect_path="/api/v1/memories",
        playground_seed="excerpt",
    ),
    RecallHelperSpec(
        id="documents",
        title="Documents",
        kind="table",
        inspect_path="/api/v1/memory/documents",
        playground_seed="excerpt",
    ),
    RecallHelperSpec(
        id="bm25",
        title="BM25",
        kind="postings",
        inspect_path="/api/v1/memory/terms",
        playground_seed="term",
    ),
    RecallHelperSpec(
        id="vectors",
        title="Vectors",
        kind="embedding",
        inspect_path="/api/v1/memory/vectors/projection",
        playground_seed="excerpt",
    ),
    RecallHelperSpec(
        id="episodes",
        title="Episode Recall",
        kind="cards",
        inspect_path="/api/v1/episodes",
        playground_seed="summary",
    ),
    RecallHelperSpec(
        id="claims",
        title="Claims",
        kind="table",
        inspect_path="/api/v1/memory/claims",
        playground_seed="text",
    ),
    RecallHelperSpec(
        id="graph",
        title="Graph",
        kind="graph",
        inspect_path="/api/v1/memory/retrieval-graph",
        playground_seed="label",
    ),
)


def builtin_helper_ids() -> set[str]:
    return {item.id for item in BUILTIN_RECALL_HELPERS}


def helper_spec(helper_id: str) -> RecallHelperSpec | None:
    return next((item for item in BUILTIN_RECALL_HELPERS if item.id == helper_id), None)
