from __future__ import annotations

import hashlib

from fastapi.testclient import TestClient

from trisynapse_memory.api import create_app
from trisynapse_memory.engine import MemoryEngine
from trisynapse_memory.engine.models import MemoryNamespace
from trisynapse_memory.engine.retrieval.contracts import RouteRegistry


class DeterministicEmbedder:
    model_name = "inspect-test-v1"
    cache_key = "inspect-test-v1"

    def encode(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            vector = [0.0] * 32
            for token in text.lower().split():
                vector[int(hashlib.sha256(token.encode()).hexdigest(), 16) % len(vector)] += 1
            norm = sum(value * value for value in vector) ** 0.5 or 1
            result.append([value / norm for value in vector])
        return result


def make_engine(tmp_path) -> MemoryEngine:
    return MemoryEngine.open(tmp_path, embedder=DeterministicEmbedder())


def seeded_engine(tmp_path) -> MemoryEngine:
    engine = make_engine(tmp_path)
    namespace = MemoryNamespace(project_id="default")
    first = engine.ingest_observation(
        "Project Atlas launched on 14 May 2026.",
        episode_id="source:atlas:v1",
        namespace=namespace,
    )
    second = engine.ingest_observation(
        "Maya Chen is the owner of Project Atlas.",
        episode_id="source:atlas:v1",
        namespace=namespace,
    )
    engine.store.append(
        kind="extraction",
        text="Project Atlas launched on 14 May 2026.",
        subject="Project Atlas",
        relation="launched_on",
        object="14 May 2026",
        evidence_refs=[first.id],
        episode_id="source:atlas:v1",
        namespace=namespace,
    )
    engine.store.append(
        kind="extraction",
        text="Maya Chen owns Project Atlas.",
        subject="Project Atlas",
        relation="owned_by",
        object="Maya Chen",
        evidence_refs=[second.id],
        episode_id="source:atlas:v1",
        namespace=namespace,
    )
    engine.build_episode_recall(["source:atlas:v1"], namespace=namespace)
    engine._prewarm_embedding_cache(engine.embedder)
    return engine


def test_catalog_lists_helpers_and_routes(tmp_path) -> None:
    engine = seeded_engine(tmp_path)
    catalog = engine.memory_catalog()
    ids = [item.id for item in catalog.helpers]
    assert ids[:7] == ["trace", "documents", "bm25", "vectors", "episodes", "claims", "graph"]
    assert {route.name for route in catalog.retrieval_routes} >= {"bm25", "semantic", "graph"}
    claims = next(item for item in catalog.helpers if item.id == "claims")
    assert claims.count >= 2
    bm25 = next(item for item in catalog.helpers if item.id == "bm25")
    assert bm25.count >= 1
    assert any(item.count >= 1 for item in catalog.helpers if item.id == "graph")


def test_empty_store_catalog_is_safe(tmp_path) -> None:
    catalog = make_engine(tmp_path).memory_catalog()
    assert [item.id for item in catalog.helpers][:7] == ["trace", "documents", "bm25", "vectors", "episodes", "claims", "graph"]
    assert all(item.count == 0 for item in catalog.helpers)


def test_documents_terms_claims_and_generic_helpers(tmp_path) -> None:
    engine = seeded_engine(tmp_path)
    documents = engine.memory_documents()
    assert documents.total >= 2
    assert documents.documents[0].modality
    terms = engine.memory_terms(search="atlas")
    assert terms.total >= 1
    assert terms.terms[0].postings
    claims = engine.memory_claims()
    assert {item.relation for item in claims} >= {"launched_on", "owned_by"}
    assert all(isinstance(item.objects, list) for item in claims)
    page = engine.memory_helper_items("bm25", search="atlas")
    assert page.items and page.kind == "postings"


def test_vector_projection_omits_raw_vectors(tmp_path) -> None:
    engine = seeded_engine(tmp_path)
    projection = engine.memory_vector_projection(sample=50)
    assert projection.searchable >= 2
    assert projection.points
    dumped = projection.model_dump()
    assert "vector" not in dumped
    seed = projection.points[0].id
    neighbors = engine.memory_vector_neighbors(seed, limit=5)
    assert neighbors.delta_id == seed
    assert all(item.id != seed for item in neighbors.neighbors)


def test_retrieval_graph_and_neighbors_are_bounded(tmp_path) -> None:
    engine = seeded_engine(tmp_path)
    graph = engine.memory_retrieval_graph()
    assert graph.view == "retrieval"
    assert graph.edges
    seed = graph.nodes[0].id
    neighbors = engine.memory_graph_neighbors(seed, view="retrieval", limit=20)
    assert neighbors.view == "retrieval"
    assert seed in {node.id for node in neighbors.nodes}


def test_search_persist_false_does_not_write_query_runs(tmp_path) -> None:
    engine = seeded_engine(tmp_path)
    namespace = MemoryNamespace(project_id="default")
    before = engine.list_query_runs(namespace=namespace)
    result = engine.search("Who owns Project Atlas?", persist=False, namespace=namespace)
    after = engine.list_query_runs(namespace=namespace)
    assert result.hits
    assert len(after.runs) == len(before.runs)
    persisted = engine.search("Who owns Project Atlas?", persist=True, namespace=namespace)
    final = engine.list_query_runs(namespace=namespace)
    assert persisted.query_id
    assert len(final.runs) == len(before.runs) + 1


def test_api_catalog_and_persist_false_search(tmp_path) -> None:
    engine = seeded_engine(tmp_path)
    client = TestClient(create_app(tmp_path, engine=engine, api_key=None))
    catalog = client.get("/api/v1/memory/catalog")
    assert catalog.status_code == 200
    body = catalog.json()
    assert [item["id"] for item in body["helpers"]][:3] == ["trace", "documents", "bm25"]
    terms = client.get("/api/v1/memory/terms", params={"q": "atlas"})
    assert terms.status_code == 200 and terms.json()["terms"]
    search = client.post("/api/v1/search", json={"query": "Atlas owner", "persist": False, "include_diagnostics": True})
    assert search.status_code == 200
    history = client.get("/api/v1/query-runs")
    assert history.json()["runs"] == []


def test_unknown_helper_falls_back_to_generic_items(tmp_path) -> None:
    engine = make_engine(tmp_path)
    page = engine.memory_helper_items("future_index")
    assert page.helper_id == "future_index"
    assert page.items == []


def test_custom_route_appears_in_catalog(tmp_path) -> None:
    class ExtraRoute:
        name = "custom_lex"

        def rank(self, plan, context):
            return []

    registry = RouteRegistry()
    registry.register(ExtraRoute())
    engine = MemoryEngine.open(tmp_path, embedder=DeterministicEmbedder(), retrieval_routes=registry)
    names = [item.name for item in engine.memory_catalog().retrieval_routes]
    assert "custom_lex" in names
