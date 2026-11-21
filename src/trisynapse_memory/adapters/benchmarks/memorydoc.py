"""MemoryDoc micro-world dataset adapter.

Owns document-chunk and conversation flattening for MemoryDoc JSON fixtures.
The shared runner still ingests through ``MemoryEngine``. World id and
``mwd:`` group prefixes must stay stable so smoke metrics stay comparable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from trisynapse_memory.adapters.benchmarks.base import (
    BenchmarkAdapter,
    BenchmarkCase,
    BenchmarkQuestion,
    PreparedCase,
)
from trisynapse_memory.engine import MemoryEngine
from trisynapse_memory.engine.ingest.pipeline import chunk_document
from trisynapse_memory.engine.models import MemoryNamespace


class MemoryDocAdapter(BenchmarkAdapter):
    name = "memorydoc"
    default_filename = "fixtures/smoke_micro_world.json"

    def load_cases(self, path: Path) -> Iterable[BenchmarkCase]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for world_index, world in enumerate(_memorydoc_worlds(payload)):
            yield _memorydoc_case(world_index, world)

    def ingest_case(self, engine: MemoryEngine, case: BenchmarkCase) -> PreparedCase:
        namespace = MemoryNamespace(project_id=f"benchmark:memorydoc:{case.id}")
        prefix = f"mwd:{case.id}:"
        group_ids = [
            *_ingest_memorydoc_documents(engine, case, prefix, namespace),
            *_ingest_memorydoc_conversations(engine, case, prefix, namespace),
        ]
        return PreparedCase(tuple(group_ids), namespace, group_prefix=prefix)


def _memorydoc_worlds(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    return payload.get("micro_worlds") or payload.get("worlds") or [payload]


def _memorydoc_case(world_index: int, world: dict[str, Any]) -> BenchmarkCase:
    world_id = str(world.get("micro_world_id") or world.get("world_id") or world.get("id") or world_index)
    return BenchmarkCase(id=world_id, payload=world, questions=_memorydoc_questions(world_id, world))


def _memorydoc_questions(world_id: str, world: dict[str, Any]) -> tuple[BenchmarkQuestion, ...]:
    items = world.get("qa_pairs") or world.get("questions") or []
    return tuple(_memorydoc_question(world_id, index, item) for index, item in enumerate(items))


def _memorydoc_question(world_id: str, index: int, item: dict[str, Any]) -> BenchmarkQuestion:
    evidence = item.get("evidence_references") or item.get("evidence") or []
    return BenchmarkQuestion(
        id=f"{world_id}:q{index}",
        question=str(item.get("question") or ""),
        gold=str(item.get("gold_answer") or item.get("answer") or ""),
        evidence=evidence,
        evidence_text=_memorydoc_evidence_text(evidence),
    )


def _memorydoc_evidence_text(evidence: Any) -> str:
    return " ".join(_evidence_ref_text(ref) for ref in evidence)


def _evidence_ref_text(ref: Any) -> str:
    if isinstance(ref, dict):
        return str(ref.get("passage_span") or ref.get("excerpt") or "")
    return str(ref)


def _ingest_memorydoc_documents(
    engine: MemoryEngine,
    case: BenchmarkCase,
    prefix: str,
    namespace: MemoryNamespace,
) -> list[str]:
    group_ids: list[str] = []
    for document in case.payload.get("documents") or []:
        document_id = str(document.get("document_id") or document.get("id"))
        group_id = f"{prefix}doc:{document_id}"
        group_ids.append(group_id)
        text = str(document.get("text") or document.get("content") or "")
        for chunk_index, chunk in enumerate(chunk_document(text, chunk_chars=3500)):
            engine.ingest_raw_source(
                chunk,
                group_id=group_id,
                source_ref={"type": "document", "id": document_id},
                locator={"kind": "chunk", "index": chunk_index},
                external_key=f"mwd:{case.id}:doc:{document_id}:{chunk_index}",
                namespace=namespace,
                process=False,
                schedule=False,
            )
    return group_ids


def _ingest_memorydoc_conversations(
    engine: MemoryEngine,
    case: BenchmarkCase,
    prefix: str,
    namespace: MemoryNamespace,
) -> list[str]:
    group_ids: list[str] = []
    for session in case.payload.get("conversations") or case.payload.get("sessions") or []:
        session_id = str(session.get("session_id") or session.get("event_id") or session.get("id"))
        group_id = f"{prefix}chat:{session_id}"
        group_ids.append(group_id)
        utterances = session.get("utterances") or session.get("messages") or []
        engine.ingest_messages(
            [_memorydoc_utterance(index, utterance) for index, utterance in enumerate(utterances)],
            group_id=group_id,
            namespace=namespace,
        )
    return group_ids


def _memorydoc_utterance(index: int, utterance: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": utterance.get("utterance_id") or utterance.get("id") or index,
        "role": utterance.get("speaker") or utterance.get("persona") or utterance.get("role") or "speaker",
        "content": utterance.get("text") or utterance.get("content") or "",
    }
