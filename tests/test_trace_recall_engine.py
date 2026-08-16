from __future__ import annotations

import hashlib

from fastapi.testclient import TestClient

from trisynapse_memory.api import create_app
from trisynapse_memory.engine import MemoryEngine


class DeterministicEmbedder:
    model_name = "deterministic-test-v1"

    def encode(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            vector = [0.0] * 64
            for token in text.lower().split():
                vector[int(hashlib.sha256(token.encode()).hexdigest(), 16) % len(vector)] += 1
            norm = sum(value * value for value in vector) ** 0.5 or 1
            result.append([value / norm for value in vector])
        return result


def make_engine(tmp_path) -> MemoryEngine:
    return MemoryEngine.open(tmp_path, embedder=DeterministicEmbedder())


def test_trace_is_hash_chained_idempotent_and_retractable(tmp_path) -> None:
    engine = make_engine(tmp_path)
    first = engine.ingest_observation(
        "Caroline attended an LGBTQ support group on May 7, 2023.",
        episode_id="chat:one",
        external_key="message:one",
    )
    duplicate = engine.ingest_observation(
        "Caroline attended an LGBTQ support group on May 7, 2023.",
        episode_id="chat:one",
        external_key="message:one",
    )
    second = engine.ingest_observation("The meeting was helpful.", episode_id="chat:one")

    assert duplicate.id == first.id
    assert second.seq == first.seq + 1
    assert second.prev_hash == first.hash
    assert engine.verify_trace().valid is True

    tombstone = engine.retract(delta_id=first.id, reason="remove imported message")
    assert tombstone.kind == "retraction"
    assert first.id not in {item.id for item in engine.store.list_deltas()}
    assert first.id in {item.id for item in engine.store.list_deltas(include_retracted=True)}
    assert engine.verify_trace().valid is True


def test_episode_recall_routes_but_never_grounds_answer(tmp_path) -> None:
    engine = make_engine(tmp_path)
    engine.ingest_messages(
        [
            {"id": "m1", "role": "user", "content": "Caroline attended an LGBTQ support group on May 7, 2023."},
            {"id": "m2", "role": "assistant", "content": "She described the meeting as helpful."},
        ],
        episode_id="chat:caroline",
    )
    views = engine.build_episode_recall()
    result = engine.query("When did Caroline attend the support group?")

    assert len(views) == 1
    assert result.abstain is False
    assert "May 7, 2023" in result.answer
    assert result.citations
    assert result.retrieval_trace.routing_seeds
    assert result.retrieval_trace.episode_recall_in_answer_context == 0
    assert all(engine.get_delta(item.delta_id).kind in {"observation", "extraction"} for item in result.citations)


def test_snapshots_are_recall_windows_not_trace_mutations(tmp_path) -> None:
    engine = make_engine(tmp_path)
    first = engine.ingest_observation("The project uses SQLite.", episode_id="project:one")
    before = engine.snapshot.create("before vector store")
    second = engine.ingest_observation("The vector cache uses LanceDB.", episode_id="project:one")
    after = engine.snapshot.create("after vector store")

    diff = engine.snapshot.diff(before.id, after.id)
    rolled_back = engine.snapshot.rollback(before.id)

    assert diff.added_delta_ids == [second.id]
    assert rolled_back.seq_cutoff == first.seq
    assert engine.store.max_seq() == second.seq
    assert engine.verify_trace().valid is True


def test_rest_smoke_ingest_compile_search_and_query(tmp_path) -> None:
    engine = make_engine(tmp_path)
    client = TestClient(create_app(tmp_path, engine=engine))

    add = client.post("/api/v1/memory/observations", json={
        "text": "Melanie read The Left Hand of Darkness.",
        "episode_id": "chat:books",
        "source_ref": {"type": "chat", "id": "books"},
    })
    compile_response = client.post("/api/v1/memory/episodes/compile", json={})
    search = client.post("/api/v1/search", json={"query": "What book did Melanie read?", "top_k": 3})
    query = client.post("/api/v1/query", json={"question": "What book did Melanie read?", "top_k": 3})

    assert add.status_code == 200
    assert compile_response.status_code == 200
    assert search.status_code == 200
    assert search.json()["hits"]
    assert query.status_code == 200
    assert query.json()["abstain"] is False
    assert query.json()["citations"]
