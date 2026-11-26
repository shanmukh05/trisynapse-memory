from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from trisynapse_memory.benchmarks import runner
from trisynapse_memory.adapters.benchmarks.base import BenchmarkCase, BenchmarkQuestion
from trisynapse_memory.adapters.benchmarks import adapter_names, get_adapter
from trisynapse_memory.engine import MemoryEngine


class DeterministicEmbedder:
    model_name = "adapter-test-v1"
    provider_name = "test"

    def encode(self, texts: list[str]) -> list[list[float]]:
        values = []
        for text in texts:
            vector = [0.0] * 32
            for token in text.lower().split():
                vector[int(hashlib.sha256(token.encode()).hexdigest(), 16) % len(vector)] += 1
            norm = sum(item * item for item in vector) ** 0.5 or 1
            values.append([item / norm for item in vector])
        return values


def _open_test_engine(store_path, mode):
    assert mode == "retrieval"
    return MemoryEngine.open(
        store_path, embedder=DeterministicEmbedder(), auto_process=False
    )


def _write_dataset(tmp_path, suite: str):
    if suite == "locomo":
        path = tmp_path / "locomo.json"
        payload = [{
            "sample_id": "l1",
            "conversation": {
                "session_1": [{
                    "speaker": "A",
                    "text": "The launch is Monday.",
                    "dia_id": "D1:1",
                    "blip_caption": "A blue launch poster",
                }],
                "session_1_date_time": "1:56 pm on 8 May, 2023",
            },
            "qa": [{"question": "When is the launch?", "answer": "Monday", "evidence": ["D1:1"]}],
        }]
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path
    if suite == "longmemeval":
        path = tmp_path / "longmemeval.json"
        payload = [{
            "question_id": "q1",
            "question": "What color is the bicycle?",
            "answer": "Blue",
            "question_type": "single-session-user",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2026-08-01"],
            "haystack_sessions": [[{"role": "user", "content": "My bicycle is blue."}]],
            "answer_session_ids": ["s1"],
        }]
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path
    if suite == "halumem":
        path = tmp_path / "halumem.jsonl"
        payload = {
            "uuid": "u1",
            "sessions": [{
                "dialogue": [{
                    "role": "user", "content": "I prefer tea.",
                    "timestamp": "Sep 04, 2025, 18:42:18",
                }],
                "questions": [{
                    "question": "What drink do I prefer?",
                    "answer": "Tea",
                    "evidence": [{"memory_content": "I prefer tea."}],
                }],
            }],
        }
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return path
    if suite == "personamem":
        root = tmp_path / "personamem"
        history_dir = root / "data" / "chat_history_32k"
        history_dir.mkdir(parents=True)
        csv_dir = root / "benchmark" / "text"
        csv_dir.mkdir(parents=True)
        (history_dir / "h1.json").write_text(
            json.dumps([
                {"role": "user", "content": "I prefer green tea in the morning."},
                {"role": "assistant", "content": "I will remember that."},
            ]),
            encoding="utf-8",
        )
        csv_path = csv_dir / "benchmark.csv"
        csv_path.write_text(
            "persona_id,user_query,correct_answer,chat_history_32k_link,related_conversation_snippet,topic_query\n"
            "p1,What tea do I like?,Green tea,data/chat_history_32k/h1.json,I prefer green tea in the morning.,Health\n",
            encoding="utf-8",
        )
        return csv_path
    path = tmp_path / "memorydoc.json"
    payload = {
        "micro_world_id": "w1",
        "documents": [{"document_id": "d1", "text": "Project Zephyr uses SQLite for storage."}],
        "qa_pairs": [{
            "question": "What storage does Project Zephyr use?",
            "gold_answer": "SQLite",
            "evidence_references": [{"passage_span": "uses SQLite for storage"}],
        }],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_adapter_registry_is_explicit() -> None:
    assert adapter_names() == ("beam100k", "beam10m", "halumem", "locomo", "longmemeval", "memorydoc", "personamem")
    with pytest.raises(ValueError, match="unsupported benchmark suite"):
        get_adapter("mystery-benchmark")


@pytest.mark.parametrize("suite", ["locomo", "longmemeval", "halumem", "memorydoc", "personamem"])
def test_every_adapter_runs_through_shared_runner(tmp_path, monkeypatch, suite) -> None:
    path = _write_dataset(tmp_path, suite)
    output = tmp_path / "runs" / suite
    monkeypatch.setattr(runner, "_open_benchmark_engine", _open_test_engine)

    summary = runner.run_benchmark(
        suite, path, max_questions=1, output_root=output, mode="retrieval"
    )
    artifact = json.loads(
        (output / f"trace_recall_{suite}_{summary['run_id']}.json").read_text(encoding="utf-8")
    )

    assert summary["questions"] == 1
    assert "accuracy" in summary
    assert "mean_recall_time_ms" in summary
    assert "recall_average_time_ms" in summary
    assert "mean_context_tokens" in summary
    assert "mean_ingest_time_per_doc_ms" in summary
    assert "ingest_time_per_doc_ms" in summary
    assert artifact["benchmark"] == suite
    assert artifact["mode"] == "retrieval"
    assert artifact["store_validation"]["ok"] is True
    assert artifact["results"][0]["question"]
    assert artifact["results"][0]["accuracy"] is not None
    assert artifact["results"][0]["recall_time_ms"] >= 0
    assert artifact["results"][0]["context_tokens"] >= 0
    assert artifact["results"][0]["ingest_time_per_doc_ms"] >= 0
    assert artifact["artifact_schema_version"] == 4
    assert artifact["results"][0]["retrieval_hits"]
    assert artifact["results"][0]["query_steps"]
    assert "abstain" in artifact["results"][0]


def test_runner_reports_structured_progress(tmp_path, monkeypatch) -> None:
    path = _write_dataset(tmp_path, "locomo")
    events: list[runner.BenchmarkProgress] = []
    monkeypatch.setattr(runner, "_open_benchmark_engine", _open_test_engine)

    summary = runner.run_benchmark(
        "locomo",
        path,
        max_questions=1,
        output_root=tmp_path / "runs",
        mode="retrieval",
        on_progress=events.append,
    )

    assert summary["questions"] == 1
    assert events[0].stage == "dataset"
    assert any(event.stage == "ingestion" and event.case_id == "l1" for event in events)
    assert any(event.stage == "query" and event.question_id for event in events)
    assert events[-1] == runner.BenchmarkProgress(
        stage="complete",
        description="Completed 1 questions",
        completed=1,
        total=1,
    )


def test_stratified_sampling_spreads_partial_runs_across_cases_and_categories() -> None:
    cases = [
        BenchmarkCase(
            id=f"case-{case_index}",
            payload={},
            questions=tuple(
                BenchmarkQuestion(
                    id=f"case-{case_index}:cat-{category}",
                    question="q",
                    gold="a",
                    metadata={"category": category},
                )
                for category in (1, 2, 3, 4)
            ),
        )
        for case_index in range(5)
    ]

    selected = runner._select_questions(cases, max_questions=8, sampling="stratified")

    assert len({case_index for case_index, _ in selected}) == 5
    assert {question.metadata["category"] for _, question in selected} == {1, 2, 3, 4}


def test_select_questions_selects_all_questions_when_max_questions_is_none() -> None:
    cases = [
        BenchmarkCase(
            id=f"case-{case_index}",
            payload={},
            questions=tuple(
                BenchmarkQuestion(
                    id=f"case-{case_index}:cat-{category}",
                    question="q",
                    gold="a",
                    metadata={"category": category},
                )
                for category in (1, 2, 3, 4)
            ),
        )
        for case_index in range(5)
    ]

    selected_stratified = runner._select_questions(cases, max_questions=None, sampling="stratified")
    selected_sequential = runner._select_questions(cases, max_questions=None, sampling="sequential")

    assert len(selected_stratified) == 20
    assert len(selected_sequential) == 20
    assert len({q.id for _, q in selected_stratified}) == 20
    assert len({q.id for _, q in selected_sequential}) == 20


def test_runner_defaults_to_full_dataset_when_max_questions_omitted(tmp_path, monkeypatch) -> None:
    path = _write_dataset(tmp_path, "locomo")
    monkeypatch.setattr(runner, "_open_benchmark_engine", _open_test_engine)

    summary = runner.run_benchmark(
        "locomo",
        path,
        output_root=tmp_path / "runs",
        mode="retrieval",
    )

    assert summary["questions"] == 1
    artifact = json.loads(
        (tmp_path / "runs" / f"trace_recall_locomo_{summary['run_id']}.json").read_text(encoding="utf-8")
    )
    assert artifact["run_configuration"]["max_questions"] is None


def test_personamem_chunks_history_into_sessions_of_eight(tmp_path) -> None:
    root = tmp_path / "personamem"
    history_dir = root / "data" / "chat_history_32k"
    csv_dir = root / "benchmark" / "text"
    history_dir.mkdir(parents=True)
    csv_dir.mkdir(parents=True)
    messages = [{"role": "user" if index % 2 == 0 else "assistant", "content": f"turn {index}"} for index in range(9)]
    (history_dir / "h1.json").write_text(json.dumps({"metadata": {}, "chat_history": messages}), encoding="utf-8")
    csv_path = csv_dir / "benchmark.csv"
    csv_path.write_text(
        "persona_id,user_query,correct_answer,chat_history_32k_link,related_conversation_snippet,topic_query\n"
        "p1,What tea do I like?,Green tea,data/chat_history_32k/h1.json,turn 0,Health\n",
        encoding="utf-8",
    )
    engine = MemoryEngine.open(tmp_path / "store-personamem-chunks", embedder=DeterministicEmbedder(), auto_process=False)
    prepared = get_adapter("personamem").ingest_case(engine, next(iter(get_adapter("personamem").load_cases(csv_path))))
    assert prepared.group_ids == ("persona:p1:s0000", "persona:p1:s0001")


@pytest.mark.parametrize(
    ("suite", "expected_namespace"),
    [
        ("locomo", "benchmark:locomo:l1"),
        ("longmemeval", "benchmark:longmemeval:q1"),
        ("halumem", "benchmark:halumem:u1"),
        ("memorydoc", "benchmark:memorydoc:w1"),
        ("personamem", "benchmark:personamem:p1"),
    ],
)
def test_adapters_isolate_each_case_in_a_namespace(tmp_path, suite, expected_namespace) -> None:
    path = _write_dataset(tmp_path, suite)
    adapter = get_adapter(suite)
    case = next(iter(adapter.load_cases(path)))
    engine = MemoryEngine.open(tmp_path / f"store-{suite}", embedder=DeterministicEmbedder(), auto_process=False)

    prepared = adapter.ingest_case(engine, case)

    assert prepared.namespace.project_id == expected_namespace
    assert prepared.group_ids


def test_locomo_preserves_time_and_indexes_captions_separately(tmp_path) -> None:
    path = _write_dataset(tmp_path, "locomo")
    adapter = get_adapter("locomo")
    case = next(iter(adapter.load_cases(path)))
    engine = MemoryEngine.open(tmp_path / "store", embedder=DeterministicEmbedder(), auto_process=False)

    prepared = adapter.ingest_case(engine, case)
    deltas = engine.store.list_records(kinds=["raw_source"], namespace=prepared.namespace)

    assert len(deltas) == 2
    assert all(item.observed_at and item.observed_at.year == 2023 for item in deltas)
    assert {item.payload["modality"] for item in deltas} == {"conversation", "image"}
    assert {item.payload["source_type"] for item in deltas} == {"conversation", "image_caption"}
    assert all(item.locator["dia_id"] == "D1:1" for item in deltas)
    assert case.questions[0].evidence_text == "The launch is Monday. A blue launch poster"


def _persisted_store(tmp_path) -> Path:
    return (
        tmp_path / "stores" / "none" / "none" / "sentence-transformers" / "all-MiniLM-L6-v2"
    )


def test_benchmark_store_path_nests_provider_and_model() -> None:
    from trisynapse_memory.benchmarks.store import benchmark_store_path, store_segment
    from trisynapse_memory.engine.models import ModelConfiguration, ProviderSelection

    assert store_segment("openai/gpt-4o-mini") == "openai_gpt-4o-mini"
    configuration = ModelConfiguration(
        completion=ProviderSelection(provider="gemini", model="gemini-3.5-flash"),
        embedding=ProviderSelection(provider="sentence-transformers", model="all-MiniLM-L6-v2"),
    )
    path = benchmark_store_path("data/locomo", Path("data/locomo/locomo10.json"), configuration)
    assert path == Path(
        "data/locomo/stores/gemini/gemini-3.5-flash/sentence-transformers/all-MiniLM-L6-v2"
    )


def test_benchmark_persists_ingested_store_under_the_dataset_root(tmp_path, monkeypatch) -> None:
    path = _write_dataset(tmp_path, "locomo")
    monkeypatch.setattr(runner, "_open_benchmark_engine", _open_test_engine)

    summary = runner.run_benchmark(
        "locomo",
        path,
        max_questions=1,
        output_root=tmp_path / "runs",
        mode="retrieval",
        stage="ingest",
    )
    store = Path(summary["store"])
    manifest = json.loads((store / "benchmark_store.json").read_text(encoding="utf-8"))

    assert summary["questions"] == 0
    assert summary["stage"] == "ingest"
    assert store == _persisted_store(tmp_path)
    assert "l1" in manifest["cases"]
    assert manifest["completion"] == {"provider": "none", "model": "none"}


def test_query_stage_reuses_ingested_cases(tmp_path, monkeypatch) -> None:
    path = _write_dataset(tmp_path, "locomo")
    monkeypatch.setattr(runner, "_open_benchmark_engine", _open_test_engine)
    runner.run_benchmark(
        "locomo", path, max_questions=1, output_root=tmp_path / "runs", mode="retrieval", stage="ingest"
    )
    events: list[runner.BenchmarkProgress] = []

    summary = runner.run_benchmark(
        "locomo",
        path,
        max_questions=1,
        output_root=tmp_path / "runs",
        mode="retrieval",
        stage="query",
        on_progress=events.append,
    )

    assert summary["questions"] == 1
    assert any(event.stage == "reuse" and event.case_id == "l1" for event in events)
    assert all(event.stage != "ingestion" for event in events)


def test_query_stage_requires_a_persisted_store(tmp_path) -> None:
    path = _write_dataset(tmp_path, "locomo")
    with pytest.raises(FileNotFoundError, match="no ingested benchmark store"):
        runner.run_benchmark("locomo", path, mode="retrieval", stage="query")


def test_rebuild_store_reingests_cases(tmp_path, monkeypatch) -> None:
    path = _write_dataset(tmp_path, "locomo")
    monkeypatch.setattr(runner, "_open_benchmark_engine", _open_test_engine)
    runner.run_benchmark(
        "locomo", path, max_questions=1, output_root=tmp_path / "runs", mode="retrieval", stage="ingest"
    )
    events: list[runner.BenchmarkProgress] = []

    runner.run_benchmark(
        "locomo",
        path,
        max_questions=1,
        output_root=tmp_path / "runs",
        mode="retrieval",
        stage="auto",
        rebuild_store=True,
        on_progress=events.append,
    )

    assert any(event.stage == "ingestion" and event.case_id == "l1" for event in events)


def test_ingest_resumes_remaining_sessions_after_a_provider_failure(tmp_path, monkeypatch) -> None:
    import threading

    from trisynapse_memory.engine.providers.registry import ProviderError
    from trisynapse_memory.prompts import load_prompt

    path = tmp_path / "locomo.json"
    path.write_text(
        json.dumps([{
            "sample_id": "l1",
            "conversation": {
                "session_1": [{"speaker": "A", "text": "The launch is Monday.", "dia_id": "D1:1"}],
                "session_1_date_time": "8 May, 2023",
                "session_2": [{"speaker": "A", "text": "I prefer tea.", "dia_id": "D2:1"}],
                "session_2_date_time": "9 May, 2023",
            },
            "qa": [{"question": "When is the launch?", "answer": "Monday", "evidence": ["D1:1"]}],
        }]),
        encoding="utf-8",
    )

    class FlakyCompletion:
        model = "resume-test"
        provider_name = "test"

        def __init__(self) -> None:
            self.facts = 0
            self.fail_at = 2
            self._lock = threading.Lock()

        def __call__(self, system: str, user: str) -> dict:
            if system == load_prompt("facts").text:
                with self._lock:
                    self.facts += 1
                    count = self.facts
                if count == self.fail_at:
                    raise ProviderError("provider HTTP 429: rate limited")
                return {
                    "facts": [{
                        "text": "A stated a fact.",
                        "subject": "A",
                        "relation": "stated",
                        "object": "a fact",
                        "confidence": 0.9,
                        "evidence_ids": [],
                    }]
                }
            if system == load_prompt("group_summary").text:
                return {"concept_or_topic": "chat", "summary": "A stated a fact.", "alt_phrasings": []}
            return {"answer": "Monday", "abstain": False, "citation_ids": []}

    completion = FlakyCompletion()

    def open_engine(store_path, mode, configuration=None):
        return MemoryEngine.open(
            store_path, embedder=DeterministicEmbedder(), completion=completion, auto_process=False
        )

    monkeypatch.setattr(runner, "_open_benchmark_engine", open_engine)
    with pytest.raises(ProviderError, match="429"):
        runner.run_benchmark("locomo", path, output_root=tmp_path / "runs", mode="end-to-end", stage="ingest")
    first = completion.facts
    completion.fail_at = 10_000
    summary = runner.run_benchmark("locomo", path, output_root=tmp_path / "runs", mode="end-to-end", stage="ingest")
    assert summary["stage"] == "ingest"
    assert completion.facts == first + 1


def test_query_resumes_remaining_questions_after_a_failure(tmp_path, monkeypatch) -> None:
    path = tmp_path / "locomo.json"
    path.write_text(
        json.dumps([{
            "sample_id": "l1",
            "conversation": {
                "session_1": [{"speaker": "A", "text": "The launch is Monday.", "dia_id": "D1:1"}],
                "session_1_date_time": "8 May, 2023",
            },
            "qa": [
                {"question": "When is the launch?", "answer": "Monday", "evidence": ["D1:1"]},
                {"question": "What is launching?", "answer": "The launch", "evidence": ["D1:1"]},
            ],
        }]),
        encoding="utf-8",
    )
    queries = {"n": 0}

    def open_engine(store_path, mode, configuration=None):
        engine = MemoryEngine.open(store_path, embedder=DeterministicEmbedder(), auto_process=False)
        original = engine.query

        def wrapped(*args, **kwargs):
            queries["n"] += 1
            if queries["n"] == 2:
                raise RuntimeError("interrupted")
            return original(*args, **kwargs)

        engine.query = wrapped  # type: ignore[method-assign]
        return engine

    monkeypatch.setattr(runner, "_open_benchmark_engine", open_engine)
    ingested = runner.run_benchmark(
        "locomo", path, output_root=tmp_path / "runs", mode="retrieval", stage="ingest"
    )
    store = Path(ingested["store"])
    with pytest.raises(RuntimeError, match="interrupted"):
        runner.run_benchmark("locomo", path, output_root=tmp_path / "runs", mode="retrieval", stage="query")
    assert (store / "query_checkpoint_retrieval.json").is_file()
    first = queries["n"]
    summary = runner.run_benchmark(
        "locomo", path, output_root=tmp_path / "runs", mode="retrieval", stage="query"
    )
    assert summary["questions"] == 2
    assert queries["n"] == first + 1
    assert not (store / "query_checkpoint_retrieval.json").exists()
