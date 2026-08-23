from __future__ import annotations

import hashlib
import re

from fastapi.testclient import TestClient

from trisynapse_memory.api import create_app
from trisynapse_memory.engine import MemoryEngine
from trisynapse_memory.engine.models import MemoryNamespace
from trisynapse_memory.engine.retrieval.engine import (
    RetrieverConfig,
    _Index,
    _Item,
    _diverse_grounded,
    classify_query,
)
from trisynapse_memory.engine.retrieval.contracts import (
    HeuristicQueryPlanner,
    QueryPlan,
    RouteRegistry,
    route_weights,
)


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


def test_trace_is_ordered_idempotent_and_retractable(tmp_path) -> None:
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
    assert second.seq == 2
    assert "hash" not in second.model_dump()
    assert "prev_hash" not in second.model_dump()
    assert engine.validate_store().ok is True

    tombstone = engine.retract(delta_id=first.id, reason="remove imported message")
    assert tombstone.kind == "retraction"
    assert first.id not in {item.id for item in engine.store.list_deltas()}
    assert first.id in {item.id for item in engine.store.list_deltas(include_retracted=True)}
    assert engine.validate_store().ok is True


def test_store_validation_reports_sequence_gaps_and_broken_evidence_links(tmp_path) -> None:
    engine = make_engine(tmp_path)
    first = engine.ingest_observation("First observation")
    second = engine.ingest_observation("Second observation")
    engine.store._connection.execute(
        "UPDATE deltas SET evidence_refs_json=? WHERE id=?",
        ('["missing-delta"]', first.id),
    )
    engine.store._connection.execute(
        "UPDATE deltas SET seq=3 WHERE id=?",
        (second.id,),
    )
    engine.store._connection.commit()

    validation = engine.validate_store(check_source_blobs=False)
    assert validation.ok is False
    assert validation.sequence_contiguous is False
    assert validation.broken_evidence_refs == [f"{first.id}->missing-delta"]


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


def test_query_reports_durable_retrieval_steps_as_they_complete(tmp_path) -> None:
    engine = make_engine(tmp_path)
    engine.add("Project Atlas launches on Monday.")
    steps = []

    result = engine.query("When does Project Atlas launch?", on_step=steps.append)

    phases = [step.phase for step in steps]
    assert result.answer
    assert phases[0] == "input"
    assert "classification" in phases
    assert "routes" in phases
    assert "grounding" in phases
    assert phases[-2:] == ["answer", "audit"]


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
    assert engine.validate_store().ok is True


def test_rest_smoke_ingest_compile_search_and_query(tmp_path) -> None:
    engine = make_engine(tmp_path)
    client = TestClient(create_app(tmp_path, engine=engine))

    add = client.post("/api/v1/memory/observations", json={
        "text": "Melanie read The Left Hand of Darkness.",
        "episode_id": "chat:books",
        "source_ref": {"type": "chat", "id": "books"},
        "modality": "conversation",
        "source_type": "chat",
        "retrieval_fields": {"speaker": "Melanie", "message": "read The Left Hand of Darkness"},
    })
    compile_response = client.post("/api/v1/memory/episodes/compile", json={})
    search = client.post("/api/v1/search", json={"query": "What book did Melanie read?", "top_k": 3})
    query = client.post("/api/v1/query", json={"question": "What book did Melanie read?", "top_k": 3})

    assert add.status_code == 200
    added_delta = engine.get(add.json()["delta_id"])
    assert added_delta.payload["modality"] == "conversation"
    assert added_delta.payload["retrieval_fields"]["speaker"] == "Melanie"
    assert compile_response.status_code == 200
    assert search.status_code == 200
    assert search.json()["hits"]
    assert query.status_code == 200
    assert query.json()["abstain"] is False
    assert query.json()["citations"]


def test_entity_graph_restores_multi_hop_bridges(tmp_path) -> None:
    engine = make_engine(tmp_path)
    one = engine.store.append(
        kind="extraction", text="Maya adopted Comet", subject="Maya", relation="adopted", object="Comet"
    )
    two = engine.store.append(
        kind="extraction", text="Comet needs daily medication", subject="Comet", relation="needs", object="medication"
    )
    graph = engine.store.retrieval_graph(MemoryNamespace())

    assert any(target == two.id and kind == "about_same_entity" for target, _, kind in graph[one.id])
    assert classify_query("What happened after Maya adopted Comet?") == "multi_hop"


def test_extraction_records_only_model_selected_observation_ids(tmp_path) -> None:
    class EvidenceCompletion:
        model = "evidence-test"

        def __call__(self, system: str, user: str) -> dict:
            identifier = re.search(r"observation_id=([^ ]+).*Maya adopted Comet", user).group(1)
            return {"facts": [{
                "subject": "Maya",
                "relation": "adopted",
                "object": "Comet",
                "text": "Maya adopted Comet.",
                "temporal_expression": None,
                "confidence": 0.95,
                "evidence_ids": [identifier],
            }]}

    engine = MemoryEngine.open(
        tmp_path,
        embedder=DeterministicEmbedder(),
        completion=EvidenceCompletion(),
        auto_process=False,
    )
    selected = engine.ingest_observation(
        "Maya adopted Comet.", episode_id="chat:pets", locator={"dia_id": "D1:1"}, schedule=False,
    )
    engine.ingest_observation(
        "Jordan bought a bicycle.", episode_id="chat:pets", locator={"dia_id": "D1:2"}, schedule=False,
    )

    extraction = engine.run_extraction(episode_id="chat:pets")[0]

    assert extraction.evidence_refs == [selected.id]
    assert extraction.locator == {"dia_id": "D1:1"}


def test_answer_model_selects_the_smallest_citation_set(tmp_path) -> None:
    class CitationCompletion:
        model = "citation-test"

        def __call__(self, system: str, user: str) -> dict:
            identifier = re.search(r"id=([^ ]+).*Maya prefers tea", user).group(1)
            return {"answer": "Tea", "abstain": False, "citation_ids": [identifier]}

    engine = MemoryEngine.open(
        tmp_path,
        embedder=DeterministicEmbedder(),
        completion=CitationCompletion(),
        auto_process=False,
    )
    selected = engine.ingest_observation("Maya prefers tea.", schedule=False)
    engine.ingest_observation("Maya owns a bicycle.", schedule=False)

    answer = engine.query("What does Maya prefer?")

    assert [citation.delta_id for citation in answer.citations] == [selected.id]
    assert len(answer.retrieval_hits) > len(answer.citations)


def test_query_planner_and_route_registry_are_replaceable(tmp_path) -> None:
    class Planner:
        def plan(self, query, **kwargs):
            return QueryPlan(
                query=query,
                query_kind="fact",
                terms=("needle",),
                modalities=("code",),
                routes=("custom",),
                profile="code",
            )

    class CustomRoute:
        name = "custom"

        def rank(self, plan, context):
            return [(item.id, 1.0) for item in context.items.values() if "needle" in item.index_text]

    registry = RouteRegistry([])
    registry.register(CustomRoute())
    engine = MemoryEngine.open(
        tmp_path,
        embedder=DeterministicEmbedder(),
        query_planner=Planner(),
        retrieval_routes=registry,
        auto_process=False,
    )
    engine.retriever_config = RetrieverConfig(
        enabled_routes=("custom",), max_refinement_rounds=0, deep_recall_enabled=False
    )
    engine._retriever_override = True
    wanted = engine.ingest_observation(
        "needle implementation", modality="code", retrieval_fields={"symbol": "needle"}, schedule=False
    )
    engine.ingest_observation("unrelated prose", schedule=False)

    result = engine.search("anything", top_k=1)

    assert result.hits[0].item_id == wanted.id
    route_step = next(step for step in engine.get_query_run(result.query_id).steps if step.phase == "routes")
    assert route_step.output["query_plan"]["profile"] == "code"
    assert list(route_step.output["routes"]) == ["custom"]


def test_source_aware_routes_and_multi_field_index(tmp_path) -> None:
    engine = make_engine(tmp_path)
    code = engine.ingest_observation(
        "def authenticate(request): pass",
        modality="code",
        source_type="python",
        retrieval_fields={"symbol": "authenticate", "path": "api/security.py", "language": "python"},
        schedule=False,
    )
    engine.ingest_observation(
        "Authentication guidance for operators.",
        modality="document",
        retrieval_fields={"title": "Operations", "section": "Security"},
        schedule=False,
    )

    result = engine.search("Which function in api/security.py authenticates requests?", top_k=2)
    indexed = {item.id: item for item in engine.store.retrieval_documents(MemoryNamespace())}

    assert result.hits[0].item_id == code.id
    assert indexed[code.id].modality == "code"
    assert indexed[code.id].fields["symbol"] == "authenticate"
    assert "code" in result.retrieval_trace.routes
    assert "document" in result.retrieval_trace.routes


def test_lexical_and_graph_indexes_persist_and_update_incrementally(tmp_path) -> None:
    engine = make_engine(tmp_path)
    first = engine.store.append(
        kind="extraction", text="Comet is Maya's dog", subject="Comet", relation="owner", object="Maya"
    )
    second = engine.store.append(
        kind="extraction", text="Comet needs medication", subject="Comet", relation="needs", object="medication"
    )
    stats_before = engine.store.retrieval_index_statistics(MemoryNamespace())
    graph_before = engine.store.retrieval_graph(MemoryNamespace())
    engine.close()

    reopened = make_engine(tmp_path)
    stats_after = reopened.store.retrieval_index_statistics(MemoryNamespace())
    graph_after = reopened.store.retrieval_graph(MemoryNamespace())

    assert stats_before == stats_after
    assert stats_after[2] == 2
    assert any(target == second.id and kind == "about_same_entity" for target, _, kind in graph_after[first.id])
    assert graph_before == graph_after


def test_token_budgeted_context_balances_sources() -> None:
    items = {
        "a1": _Item(id="a1", kind="observation", text="a" * 120, index_text="a", confidence=1, source_ref={"id": "a"}),
        "a2": _Item(id="a2", kind="observation", text="b" * 120, index_text="b", confidence=1, source_ref={"id": "a"}),
        "b1": _Item(id="b1", kind="observation", text="c" * 120, index_text="c", confidence=1, source_ref={"id": "b"}),
    }
    index = _Index(items=items, adjacency={}, df={}, avgdl=1)

    selected = _diverse_grounded(
        [("a1", 1.0), ("a2", 0.9), ("b1", 0.8)],
        index,
        context_limit=3,
        max_context_tokens=70,
        per_source_context_tokens=35,
    )

    assert selected == ["a1", "b1"]


def test_query_records_the_effective_token_counter(tmp_path) -> None:
    class WordCounter:
        name = "word-counter-test"
        exact = True

        @staticmethod
        def count(text):
            return len(text.split())

    engine = MemoryEngine.open(
        tmp_path,
        embedder=DeterministicEmbedder(),
        token_counter=WordCounter(),
        auto_process=False,
    )
    engine.ingest_observation("Maya prefers green tea", schedule=False)

    result = engine.search("What tea does Maya prefer?", top_k=1)
    grounding = next(
        step for step in engine.get_query_run(result.query_id).steps if step.phase == "grounding"
    )

    assert result.hits[0].metadata["token_counter"] == "word-counter-test"
    assert grounding.output["token_counter"] == {"name": "word-counter-test", "exact": True}
    assert grounding.output["context_tokens"] == 4


def test_default_planner_selects_source_profiles_without_benchmark_rules() -> None:
    planner = HeuristicQueryPlanner()

    assert planner.plan("Which function parses the archive?", available_modalities={"code"}).profile == "code"
    assert planner.plan("What is shown in the diagram?", available_modalities={"image"}).modalities == ("image",)
    assert planner.plan("Total the spreadsheet column", available_modalities={"table"}).profile == "table"
    assert route_weights("code")["code"] > route_weights("balanced")["code"]
    assert route_weights("code", {"semantic": 0.25})["semantic"] == 0.25
