from __future__ import annotations

import hashlib
import time

import pytest

from trisynapse_memory import (
    BranchResult,
    EngineExtensionRegistry,
    ExtensionSpec,
    FormationProcessorSpec,
    MemoryEngine,
    ProposedDelta,
    RecallChannelSpec,
    RecallRecord,
    RetrievalConfiguration,
    RetrievalBranchSpec,
    RetrievalCandidate,
    SourceInput,
)
from trisynapse_memory.engine.formation.sources import PreparedChunk, PreparedSource
from trisynapse_memory.engine.retrieval.contracts import BM25Route, QueryPlan, RouteContext, SemanticRoute
from trisynapse_memory.engine.retrieval.engine import RetrieverConfig


class DeterministicEmbedder:
    model_name = "extension-test-v1"
    cache_key = "extension-test-v1"

    def encode(self, texts: list[str]) -> list[list[float]]:
        values = []
        for text in texts:
            vector = [0.0] * 24
            for token in text.casefold().split():
                vector[int(hashlib.sha256(token.encode()).hexdigest(), 16) % len(vector)] += 1
            norm = sum(value * value for value in vector) ** 0.5 or 1
            values.append([value / norm for value in vector])
        return values


class EntityChannel:
    spec = RecallChannelSpec(id="test.entities", title="Entity profiles")

    def project(self, batch, writer):
        observations = [item for item in batch.deltas if item.kind == "observation"]
        if observations:
            writer.put(
                "atlas",
                "Project Atlas profile: launch ownership and status",
                [item.id for item in observations],
                evidence_version=batch.evidence_version,
                fields={"entity": "Project Atlas"},
            )

    def rebuild(self, reader, writer):
        observations = reader.deltas(kinds=("observation",))
        records = []
        if observations:
            records.append(RecallRecord(
                channel_id=self.spec.id,
                record_id="atlas",
                namespace=reader.namespace,
                evidence_version=reader.seq_cutoff or 0,
                text="Project Atlas profile: launch ownership and status",
                evidence_refs=tuple(item.id for item in observations),
                fields={"entity": "Project Atlas"},
            ))
        writer.replace(records)


class EntityBranch:
    spec = RetrievalBranchSpec(
        name="test.entity_profiles",
        title="Entity profiles",
        default_weight=1.4,
        depends_on=("bm25",),
    )

    def retrieve(self, plan, context):
        records, _ = context.recall.records("test.entities", search="atlas", limit=20)
        return BranchResult(
            branch=self.spec.name,
            candidates=tuple(
                RetrievalCandidate(
                    id=record.record_id,
                    branch=self.spec.name,
                    channel_id=record.channel_id,
                    kind="recall",
                    score=1.0,
                    text=record.text,
                    evidence_delta_ids=record.evidence_refs,
                    metadata={"fields": record.fields},
                )
                for record in records
            ),
        )


class EntityExtension:
    spec = ExtensionSpec(
        id="test.entity_extension",
        version="0.1.0",
        engine_api=">=1,<2",
        storage_revision=1,
    )

    def register(self, registry):
        registry.recall_channels.register(EntityChannel())
        registry.retrieval_branches.register(EntityBranch())


class AcronymSourceHandler:
    name = "test.acronym_source"
    kinds = ("acronym",)

    def accepts(self, source):
        return source.kind == "acronym"

    def prepare(self, source, context):
        del context
        raw = (source.text or "").encode()
        return PreparedSource(
            source=source,
            kind="acronym",
            title=source.title or "Acronyms",
            uri=None,
            filename="acronyms.txt",
            media_type="text/plain",
            original=raw,
            chunks=[PreparedChunk(
                source.text or "",
                {"kind": "acronym_entry"},
                modality="document",
                source_type="acronym",
                retrieval_fields={"acronym": "RRF"},
            )],
        )


class AcronymExtension:
    spec = ExtensionSpec(id="test.acronym_extension", version="1.0.0")

    def register(self, registry):
        registry.source_handlers.register(AcronymSourceHandler())


class FactProcessor:
    spec = FormationProcessorSpec(name="test.fact_processor")

    def should_schedule(self, event):
        return event.episode_id is not None

    def process(self, episode, context):
        del context
        first = episode[0]
        return [ProposedDelta(
            kind="extraction",
            text="Project Atlas is active.",
            evidence_refs=(first.id,),
            subject="Project Atlas",
            relation="status",
            object="active",
            external_key=f"test-fact:{first.id}",
        )]


class FormationExtension:
    spec = ExtensionSpec(id="test.formation_extension", version="1.0.0")

    def register(self, registry):
        registry.formation_processors.register(FactProcessor())


def test_extension_registry_rejects_duplicates_and_freezes() -> None:
    registry = EngineExtensionRegistry()
    registry.register_extension(EntityExtension())
    with pytest.raises(ValueError, match="already registered"):
        registry.register_extension(EntityExtension())
    registry.validate_and_freeze(("bm25",))
    with pytest.raises(RuntimeError, match="frozen"):
        registry.recall_channels.register(EntityChannel())


def test_custom_source_handler_controls_retrieval_fields(tmp_path) -> None:
    engine = MemoryEngine.open(
        tmp_path,
        embedder=DeterministicEmbedder(),
        extensions=[AcronymExtension()],
        auto_process=False,
    )
    result = engine.ingest(SourceInput(
        kind="acronym", text="RRF means reciprocal rank fusion.", source_key="rrf"
    ))
    delta = engine.get_delta(result.delta_ids[0])

    assert delta.payload["modality"] == "document"
    assert delta.payload["source_type"] == "acronym"
    assert delta.payload["retrieval_fields"] == {"acronym": "RRF"}


def test_formation_processor_runs_through_durable_registry(tmp_path) -> None:
    engine = MemoryEngine.open(
        tmp_path,
        embedder=DeterministicEmbedder(),
        extensions=[FormationExtension()],
    )
    observation = engine.ingest_observation(
        "Project Atlas launched.", episode_id="atlas"
    )
    extractions = engine.store.list_deltas(kinds=["extraction"])

    assert len(extractions) == 1
    assert extractions[0].evidence_refs == [observation.id]
    assert extractions[0].actor.id == "test.fact_processor"
    assert any(job.kind == "formation:test.fact_processor" for job in engine.list_jobs())


def test_recall_channel_and_branch_ground_back_to_trace(tmp_path) -> None:
    engine = MemoryEngine.open(
        tmp_path,
        embedder=DeterministicEmbedder(),
        extensions=[EntityExtension()],
    )
    evidence = engine.ingest_observation(
        "Project Atlas launched on Monday.", episode_id="atlas"
    )
    engine.retriever_config = RetrieverConfig(
        enabled_routes=("bm25", "test.entity_profiles"),
        max_refinement_rounds=0,
        deep_recall_enabled=False,
    )
    engine._retriever_override = True

    result = engine.search("What is the Project Atlas status?", top_k=2)
    catalog = engine.memory_catalog()
    page = engine.memory_helper_items("test.entities")

    assert result.hits and result.hits[0].item_id == evidence.id
    assert "test.entity_profiles" in result.retrieval_trace.routes
    assert any(item.id == "test.entities" for item in catalog.helpers)
    assert any(item.name == "test.entity_profiles" for item in catalog.retrieval_routes)
    assert any(item.id == "test.entity_extension" for item in catalog.extensions)
    assert page.items[0].data["evidence_refs"] == [evidence.id]

    engine.remove(delta_ids=[evidence.id], reason="privacy request")
    records, total = engine.store.recall_records("test.entities", engine.default_namespace)
    assert records == [] and total == 0


def test_failed_custom_branch_does_not_break_healthy_routes(tmp_path) -> None:
    class BrokenBranch:
        spec = RetrievalBranchSpec(name="test.broken")

        def retrieve(self, plan, context):
            raise RuntimeError("branch exploded")

    class BrokenExtension:
        spec = ExtensionSpec(id="test.broken_extension", version="1")

        def register(self, registry):
            registry.retrieval_branches.register(BrokenBranch())

    engine = MemoryEngine.open(
        tmp_path,
        embedder=DeterministicEmbedder(),
        extensions=[BrokenExtension()],
        auto_process=False,
    )
    wanted = engine.ingest_observation("Maya owns Project Atlas.", schedule=False)
    engine.retriever_config = RetrieverConfig(
        enabled_routes=("bm25", "test.broken"),
        max_refinement_rounds=0,
        deep_recall_enabled=False,
    )
    engine._retriever_override = True

    result = engine.search("Who owns Project Atlas?", top_k=1)
    route_step = next(
        step for step in engine.get_query_run(result.query_id).steps if step.phase == "routes"
    )

    assert result.hits[0].item_id == wanted.id
    assert route_step.output["branch_errors"] == {"test.broken": "branch exploded"}


def test_slow_custom_branch_times_out_without_breaking_healthy_routes(tmp_path) -> None:
    class SlowBranch:
        spec = RetrievalBranchSpec(name="test.slow", timeout_ms=1)

        def retrieve(self, plan, context):
            time.sleep(0.05)
            return BranchResult(branch=self.spec.name, candidates=())

    class SlowExtension:
        spec = ExtensionSpec(id="test.slow_extension", version="1")

        def register(self, registry):
            registry.retrieval_branches.register(SlowBranch())

    engine = MemoryEngine.open(
        tmp_path,
        embedder=DeterministicEmbedder(),
        extensions=[SlowExtension()],
        auto_process=False,
    )
    wanted = engine.ingest_observation("Maya owns Project Atlas.", schedule=False)
    engine.retriever_config = RetrieverConfig(
        enabled_routes=("bm25", "test.slow"),
        max_refinement_rounds=0,
        deep_recall_enabled=False,
    )
    engine._retriever_override = True

    result = engine.search("Who owns Project Atlas?", top_k=1)
    route_step = next(
        step for step in engine.get_query_run(result.query_id).steps if step.phase == "routes"
    )

    assert result.hits[0].item_id == wanted.id
    assert "timed out after 1 ms" in route_step.output["branch_errors"]["test.slow"]


def test_missing_extension_is_inert_and_visible_after_reopen(tmp_path) -> None:
    engine = MemoryEngine.open(
        tmp_path,
        embedder=DeterministicEmbedder(),
        extensions=[EntityExtension()],
        auto_process=False,
    )
    assert engine.memory_catalog().extensions[0].status == "available"
    configuration = engine.get_retrieval_configuration()
    engine.set_retrieval_configuration(RetrievalConfiguration(
        enabled_routes=(*configuration.enabled_routes, "test.entity_profiles"),
        route_weights={**configuration.route_weights, "test.entity_profiles": 1.4},
        retrieval_profile=configuration.retrieval_profile,
        revision=configuration.revision,
    ))
    engine.close()

    reopened = MemoryEngine.open(
        tmp_path,
        embedder=DeterministicEmbedder(),
        auto_process=False,
    )
    extension = next(
        item for item in reopened.memory_catalog().extensions
        if item.id == "test.entity_extension"
    )

    assert extension.status == "unavailable"
    missing_route = next(
        item for item in reopened.memory_catalog().retrieval_routes
        if item.name == "test.entity_profiles"
    )
    assert missing_route.enabled is True
    assert missing_route.available is False


def test_storage_revision_schedules_and_completes_rebuild(tmp_path) -> None:
    engine = MemoryEngine.open(
        tmp_path,
        embedder=DeterministicEmbedder(),
        extensions=[EntityExtension()],
    )
    engine.ingest_observation("Project Atlas launched.", episode_id="atlas")
    engine.close()

    class EntityExtensionV2(EntityExtension):
        spec = ExtensionSpec(
            id="test.entity_extension",
            version="0.2.0",
            engine_api=">=1,<2",
            storage_revision=2,
        )

    reopened = MemoryEngine.open(
        tmp_path,
        embedder=DeterministicEmbedder(),
        extensions=[EntityExtensionV2()],
        auto_process=False,
    )
    assert reopened.memory_catalog().extensions[0].status == "rebuild_required"

    completed = reopened.run_jobs()
    assert any(
        job.kind == "extension:test.entity_extension:rebuild"
        and job.status == "completed"
        for job in completed
    )
    assert reopened.memory_catalog().extensions[0].status == "available"


def test_recall_lifecycle_follows_transitive_evidence_dependencies(tmp_path) -> None:
    engine = MemoryEngine.open(
        tmp_path, embedder=DeterministicEmbedder(), auto_process=False
    )
    observation = engine.ingest_observation("Sensitive source fact.", schedule=False)
    extraction = engine.store.append(
        kind="extraction",
        text="Derived sensitive fact.",
        evidence_refs=[observation.id],
        namespace=engine.default_namespace,
    )
    engine.store.put_recall_record(RecallRecord(
        channel_id="test.transitive",
        record_id="derived",
        namespace=engine.default_namespace,
        evidence_version=extraction.seq,
        text="Derived profile text.",
        evidence_refs=(extraction.id,),
    ))

    engine.retract(delta_id=observation.id, reason="expired")
    records, total = engine.store.recall_records(
        "test.transitive", engine.default_namespace
    )

    assert records == [] and total == 0


def test_bm25_and_semantic_route_names_match_candidate_sources() -> None:
    plan = QueryPlan(query="atlas", query_kind="fact", terms=("atlas",))
    context = RouteContext(
        items={},
        query_vector=[],
        bm25=lambda query, text: 0.0,
        semantic=lambda left, right: 0.0,
        graph_walk=lambda seeds: [],
        lexical_candidates={"lexical": 0.9},
        semantic_candidates={"vector": 0.8},
    )

    assert BM25Route().rank(plan, context) == [("lexical", 0.9)]
    assert SemanticRoute().rank(plan, context) == [("vector", 0.8)]
