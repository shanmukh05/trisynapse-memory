from __future__ import annotations

import hashlib
import json

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
    assert adapter_names() == ("halumem", "locomo", "longmemeval", "memorydoc")
    with pytest.raises(ValueError, match="unsupported benchmark suite"):
        get_adapter("mystery-benchmark")


@pytest.mark.parametrize("suite", ["locomo", "longmemeval", "halumem", "memorydoc"])
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
    assert artifact["benchmark"] == suite
    assert artifact["mode"] == "retrieval"
    assert artifact["store_validation"]["ok"] is True
    assert artifact["results"][0]["question"]
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


@pytest.mark.parametrize(
    ("suite", "expected_namespace"),
    [
        ("locomo", "benchmark:locomo:l1"),
        ("longmemeval", "benchmark:longmemeval:q1"),
        ("halumem", "benchmark:halumem:u1"),
        ("memorydoc", "benchmark:memorydoc:w1"),
    ],
)
def test_adapters_isolate_each_case_in_a_namespace(tmp_path, suite, expected_namespace) -> None:
    path = _write_dataset(tmp_path, suite)
    adapter = get_adapter(suite)
    case = next(iter(adapter.load_cases(path)))
    engine = MemoryEngine.open(tmp_path / f"store-{suite}", embedder=DeterministicEmbedder(), auto_process=False)

    prepared = adapter.ingest_case(engine, case)

    assert prepared.namespace.project_id == expected_namespace
    assert prepared.episode_ids


def test_locomo_preserves_time_and_indexes_captions_separately(tmp_path) -> None:
    path = _write_dataset(tmp_path, "locomo")
    adapter = get_adapter("locomo")
    case = next(iter(adapter.load_cases(path)))
    engine = MemoryEngine.open(tmp_path / "store", embedder=DeterministicEmbedder(), auto_process=False)

    prepared = adapter.ingest_case(engine, case)
    deltas = engine.store.list_deltas(kinds=["observation"], namespace=prepared.namespace)

    assert len(deltas) == 2
    assert all(item.observed_at and item.observed_at.year == 2023 for item in deltas)
    assert {item.payload["modality"] for item in deltas} == {"conversation", "image"}
    assert {item.payload["source_type"] for item in deltas} == {"conversation", "image_caption"}
    assert all(item.locator["dia_id"] == "D1:1" for item in deltas)
    assert case.questions[0].evidence_text == "The launch is Monday. A blue launch poster"
