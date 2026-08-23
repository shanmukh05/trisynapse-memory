"""MemoryDoc micro-world dataset adapter."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from trisynapse_memory.adapters.benchmarks.base import (
    BenchmarkAdapter,
    BenchmarkCase,
    BenchmarkQuestion,
    PreparedCase,
)
from trisynapse_memory.engine import MemoryEngine
from trisynapse_memory.engine.formation.pipeline import chunk_document
from trisynapse_memory.engine.models import MemoryNamespace


class MemoryDocAdapter(BenchmarkAdapter):
    name = "memorydoc"
    default_filename = "fixtures/smoke_micro_world.json"

    def load_cases(self, path: Path) -> Iterable[BenchmarkCase]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        worlds = payload if isinstance(payload, list) else payload.get("micro_worlds") or payload.get("worlds") or [payload]
        for world_index, world in enumerate(worlds):
            world_id = str(world.get("micro_world_id") or world.get("world_id") or world.get("id") or world_index)
            questions = []
            for index, item in enumerate(world.get("qa_pairs") or world.get("questions") or []):
                evidence = item.get("evidence_references") or item.get("evidence") or []
                evidence_text = " ".join(
                    str(ref.get("passage_span") or ref.get("excerpt") or "")
                    if isinstance(ref, dict) else str(ref)
                    for ref in evidence
                )
                questions.append(BenchmarkQuestion(
                    id=f"{world_id}:q{index}",
                    question=str(item.get("question") or ""),
                    gold=str(item.get("gold_answer") or item.get("answer") or ""),
                    evidence=evidence,
                    evidence_text=evidence_text,
                ))
            yield BenchmarkCase(id=world_id, payload=world, questions=tuple(questions))

    def ingest_case(self, engine: MemoryEngine, case: BenchmarkCase) -> PreparedCase:
        namespace = MemoryNamespace(project_id=f"benchmark:memorydoc:{case.id}")
        prefix = f"mwd:{case.id}:"
        episode_ids: list[str] = []
        for document in case.payload.get("documents") or []:
            document_id = str(document.get("document_id") or document.get("id"))
            episode_id = f"{prefix}doc:{document_id}"
            episode_ids.append(episode_id)
            text = str(document.get("text") or document.get("content") or "")
            for chunk_index, chunk in enumerate(chunk_document(text, chunk_chars=3500)):
                engine.ingest_observation(
                    chunk,
                    episode_id=episode_id,
                    source_ref={"type": "document", "id": document_id},
                    locator={"kind": "chunk", "index": chunk_index},
                    external_key=f"mwd:{case.id}:doc:{document_id}:{chunk_index}",
                    namespace=namespace,
                    process=False,
                    schedule=False,
                )
        for session in case.payload.get("conversations") or case.payload.get("sessions") or []:
            session_id = str(session.get("session_id") or session.get("event_id") or session.get("id"))
            episode_id = f"{prefix}chat:{session_id}"
            episode_ids.append(episode_id)
            utterances = session.get("utterances") or session.get("messages") or []
            engine.ingest_messages(
                [
                    {
                        "id": utterance.get("utterance_id") or utterance.get("id") or index,
                        "role": utterance.get("speaker") or utterance.get("persona") or utterance.get("role") or "speaker",
                        "content": utterance.get("text") or utterance.get("content") or "",
                    }
                    for index, utterance in enumerate(utterances)
                ],
                episode_id=episode_id,
                namespace=namespace,
            )
        return PreparedCase(tuple(episode_ids), namespace, episode_prefix=prefix)
