from __future__ import annotations

import hashlib
import json
import threading
import time

import pytest

from trisynapse_memory.benchmarks import runner as benchmark_runner
from trisynapse_memory.benchmarks.release import evaluate_release_gate
from trisynapse_memory.engine import MemoryEngine
from trisynapse_memory.prompts import load_prompt, prompt_provenance


class DeterministicEmbedder:
    model_name = "benchmark-test-embedding-v1"
    provider_name = "test"

    def encode(self, texts: list[str]) -> list[list[float]]:
        result = []
        for value in texts:
            vector = [0.0] * 32
            for token in value.lower().split():
                vector[int(hashlib.sha256(token.encode()).hexdigest(), 16) % len(vector)] += 1
            norm = sum(item * item for item in vector) ** 0.5 or 1
            result.append([item / norm for item in vector])
        return result


class RecordingCompletion:
    model = "recording-completion-v1"
    provider_name = "test"

    def __init__(self) -> None:
        self.prompt_names: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, system: str, user: str) -> dict:
        if system == load_prompt("facts").text:
            with self._lock:
                self.prompt_names.append("facts")
            return {
                "facts": [{
                    "subject": "Caroline",
                    "relation": "attended",
                    "object": "support group",
                    "text": "Caroline attended the support group on 7 May 2023.",
                    "temporal_expression": "7 May 2023",
                    "confidence": 0.95,
                }]
            }
        if system == load_prompt("group_summary").text:
            with self._lock:
                self.prompt_names.append("group_summary")
            return {
                "concept_or_topic": "Caroline support group",
                "summary": "Caroline attended a support group on 7 May 2023.",
                "alt_phrasings": ["When was the support group?"],
            }
        if system.startswith(load_prompt("answer").text):
            with self._lock:
                self.prompt_names.append("answer")
            return {"answer": "7 May 2023", "abstain": False}
        if system == load_prompt("benchmark_judge").text:
            with self._lock:
                self.prompt_names.append("benchmark_judge")
            return {"correct": True, "score": 1.0, "reason": "Matches the reference."}
        raise AssertionError(f"unexpected prompt: {system[:80]}")


class ConcurrentRecordingCompletion(RecordingCompletion):
    def __init__(self) -> None:
        super().__init__()
        self.max_in_flight = 0
        self._in_flight = 0

    def __call__(self, system: str, user: str) -> dict:
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            time.sleep(0.15)
            return super().__call__(system, user)
        finally:
            with self._lock:
                self._in_flight -= 1


class AbstainingCompletion(RecordingCompletion):
    def __call__(self, system: str, user: str) -> dict:
        if system.startswith(load_prompt("answer").text):
            self.prompt_names.append("answer")
            return {"answer": "", "abstain": True}
        if system == load_prompt("benchmark_judge").text:
            self.prompt_names.append("benchmark_judge")
            return {"correct": False, "score": 0.0, "reason": "No answer."}
        return super().__call__(system, user)


def _locomo_fixture(path, sessions: int = 1) -> None:
    conversation: dict[str, object] = {}
    for index in range(1, sessions + 1):
        conversation[f"session_{index}"] = [{
            "speaker": "Caroline",
            "text": "I attended the support group.",
            "dia_id": f"D{index}:1",
        }]
        conversation[f"session_{index}_date_time"] = "7 May 2023"
    path.write_text(json.dumps([{
        "sample_id": "sample-1",
        "conversation": conversation,
        "qa": [{
            "question": "When did Caroline attend the support group?",
            "answer": "7 May 2023",
            "evidence": ["D1:1"],
            "category": 2,
        }],
    }]), encoding="utf-8")


def test_packaged_prompts_are_versioned_and_hashed() -> None:
    names = ["facts", "group_summary", "answer", "benchmark_judge"]
    provenance = prompt_provenance(names)

    assert [item["name"] for item in provenance] == names
    assert [item["version"] for item in provenance] == [
        "extraction-v2",
        "group-summary-v1",
        "answer-v2",
        "benchmark-judge-v1",
    ]
    assert all(len(item["sha256"]) == 64 for item in provenance)
    with pytest.raises(KeyError, match="unknown production prompt"):
        load_prompt("removed-snapshot-prompt")


def test_engine_uses_packaged_prompts_and_records_extraction_version(tmp_path) -> None:
    completion = RecordingCompletion()
    engine = MemoryEngine.open(
        tmp_path, embedder=DeterministicEmbedder(), completion=completion, auto_process=False
    )
    engine.ingest_raw_source(
        "Caroline attended the support group on 7 May 2023.",
        group_id="chat:one",
        schedule=False,
    )

    extraction = engine.run_fact_extraction(group_id="chat:one")
    engine.build_group_summary(["chat:one"])
    answer = engine.query("When did Caroline attend the support group?")

    assert extraction[0].actor.prompt_version == load_prompt("facts").version
    assert answer.answer == "7 May 2023"
    assert completion.prompt_names == ["facts", "group_summary", "answer"]


def test_fact_extraction_keeps_facts_with_null_confidence(tmp_path) -> None:
    class NullConfidenceCompletion:
        model = "gpt-5.6-luna"

        def complete_json(self, system: str, user: str) -> dict:
            return {
                "facts": [
                    {
                        "subject": "Ada",
                        "relation": "likes",
                        "object": "cats",
                        "text": "Ada likes cats.",
                        "confidence": None,
                    },
                    "not-a-fact",
                ]
            }

        __call__ = complete_json

    engine = MemoryEngine.open(
        tmp_path, embedder=DeterministicEmbedder(), completion=NullConfidenceCompletion(), auto_process=False
    )
    engine.ingest_raw_source("Ada likes cats.", group_id="chat:pets", schedule=False)
    extraction = engine.run_fact_extraction(group_id="chat:pets")
    assert len(extraction) == 1
    assert extraction[0].text == "Ada likes cats."
    assert extraction[0].confidence == 0.7


def test_fact_extraction_keeps_facts_with_weak_lexical_evidence(tmp_path) -> None:
    class WeakEvidenceCompletion:
        model = "gpt-5.6-luna"

        def complete_json(self, system: str, user: str) -> dict:
            return {
                "facts": [
                    {
                        "subject": "Ada",
                        "relation": "prefers",
                        "object": "quantum chromodynamics gauge field theory",
                        "text": (
                            "Ada prefers quantum chromodynamics gauge field "
                            "theory renormalization group flow."
                        ),
                        "confidence": 0.9,
                        "evidence_ids": ["invented-id"],
                    }
                ]
            }

        __call__ = complete_json

    engine = MemoryEngine.open(
        tmp_path, embedder=DeterministicEmbedder(), completion=WeakEvidenceCompletion(), auto_process=False
    )
    engine.ingest_raw_source("Ada likes cats.", group_id="chat:pets", schedule=False)
    extraction = engine.run_fact_extraction(group_id="chat:pets")
    assert len(extraction) == 1
    assert extraction[0].text.startswith("Ada prefers")
    assert extraction[0].evidence_refs


@pytest.mark.parametrize("mode", ["retrieval", "end-to-end"])
def test_benchmark_modes_record_provider_and_prompt_provenance(tmp_path, monkeypatch, mode) -> None:
    dataset = tmp_path / "locomo.json"
    output = tmp_path / "runs"
    _locomo_fixture(dataset)
    completion = RecordingCompletion() if mode == "end-to-end" else None

    def open_test_engine(store_path, requested_mode):
        assert requested_mode == mode
        return MemoryEngine.open(
            store_path,
            embedder=DeterministicEmbedder(),
            completion=completion,
            auto_process=False,
        )

    monkeypatch.setattr(benchmark_runner, "_open_benchmark_engine", open_test_engine)
    result = benchmark_runner.run_trace_recall_benchmark(
        "locomo", dataset, max_questions=1, output_root=output, mode=mode
    )
    artifact = json.loads((output / f"trace_recall_locomo_{result['run_id']}.json").read_text(encoding="utf-8"))

    assert artifact["artifact_schema_version"] == 4
    assert artifact["mode"] == mode
    assert artifact["providers"]["embedding"]["provider"] == "test"
    assert "accuracy" in artifact["summary"]
    assert "mean_recall_time_ms" in artifact["summary"]
    assert "recall_average_time_ms" in artifact["summary"]
    assert "mean_context_tokens" in artifact["summary"]
    assert "mean_ingest_time_per_doc_ms" in artifact["summary"]
    assert "ingest_time_per_doc_ms" in artifact["summary"]
    if mode == "retrieval":
        assert artifact["providers"]["completion"]["provider"] == "none"
        assert artifact["prompts"] == []
        assert "mean_judge_score" not in artifact["summary"]
    else:
        assert artifact["providers"]["completion"]["model"] == completion.model
        assert [item["name"] for item in artifact["prompts"]] == [
            "facts", "group_summary", "answer", "benchmark_judge"
        ]
        assert artifact["summary"]["mean_judge_score"] == 1.0
        assert artifact["summary"]["accuracy"] == 1.0
        assert set(completion.prompt_names) == {
            "facts", "group_summary", "answer", "benchmark_judge"
        }


def test_end_to_end_mode_requires_completion_configuration(tmp_path, monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="end-to-end benchmark mode requires"):
        benchmark_runner._open_benchmark_engine(tmp_path, "end-to-end")


def test_end_to_end_prepares_independent_sessions_concurrently(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "locomo.json"
    _locomo_fixture(dataset, sessions=3)
    completion = ConcurrentRecordingCompletion()
    events: list[benchmark_runner.BenchmarkProgress] = []

    def open_test_engine(store_path, requested_mode):
        return MemoryEngine.open(
            store_path,
            embedder=DeterministicEmbedder(),
            completion=completion,
            auto_process=False,
        )

    monkeypatch.setattr(benchmark_runner, "_open_benchmark_engine", open_test_engine)
    result = benchmark_runner.run_benchmark(
        "locomo",
        dataset,
        max_questions=1,
        output_root=tmp_path / "runs",
        mode="end-to-end",
        on_progress=events.append,
    )

    assert result["questions"] == 1
    assert completion.max_in_flight >= 2
    assert completion.prompt_names.count("facts") == 3
    assert completion.prompt_names.count("group_summary") == 3
    assert any(
        event.stage == "extraction" and "3/3 sessions" in event.description
        for event in events
    )
    assert any(
        event.stage == "summaries" and "3/3 sessions" in event.description
        for event in events
    )


def test_retrieval_recall_is_not_erased_when_answer_abstains(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "locomo.json"
    _locomo_fixture(dataset)
    completion = AbstainingCompletion()

    def open_test_engine(store_path, requested_mode):
        return MemoryEngine.open(
            store_path,
            embedder=DeterministicEmbedder(),
            completion=completion,
            auto_process=False,
        )

    monkeypatch.setattr(benchmark_runner, "_open_benchmark_engine", open_test_engine)
    result = benchmark_runner.run_benchmark(
        "locomo",
        dataset,
        max_questions=1,
        output_root=tmp_path / "runs",
        mode="end-to-end",
    )
    artifact = json.loads(
        (tmp_path / "runs" / f"trace_recall_locomo_{result['run_id']}.json").read_text()
    )
    row = artifact["results"][0]

    assert row["abstain"] is True
    assert row["cited_ids"] == []
    assert row["evidence_hit_at_k"] is True
    assert row["evidence_recall_at_k"] == 1.0


def test_judge_row_treats_invalid_json_as_incorrect() -> None:
    def broken(system: str, user: str) -> dict:
        raise ValueError("completion returned invalid JSON: Unterminated string starting at")

    row = {"question": "q", "gold": "a", "prediction": "b"}
    benchmark_runner._judge_row(broken, row, "end-to-end")
    assert row["accuracy"] == 0.0
    assert row["judge"]["correct"] is False
    assert row["judge"]["reason"] == "invalid_json"


def test_benchmark_can_use_an_independent_judge(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "locomo.json"
    _locomo_fixture(dataset)
    completion = RecordingCompletion()
    judge = RecordingCompletion()
    judge.model = "independent-judge-v1"

    def open_test_engine(store_path, requested_mode):
        return MemoryEngine.open(
            store_path,
            embedder=DeterministicEmbedder(),
            completion=completion,
            auto_process=False,
        )

    monkeypatch.setattr(benchmark_runner, "_open_benchmark_engine", open_test_engine)
    result = benchmark_runner.run_benchmark(
        "locomo",
        dataset,
        max_questions=1,
        output_root=tmp_path / "runs",
        mode="end-to-end",
        judge_completion=judge,
    )
    artifact = json.loads(
        (tmp_path / "runs" / f"trace_recall_locomo_{result['run_id']}.json").read_text()
    )

    assert artifact["providers"]["completion"]["model"] == completion.model
    assert artifact["providers"]["judge"]["model"] == "independent-judge-v1"
    assert "benchmark_judge" not in completion.prompt_names
    assert judge.prompt_names == ["benchmark_judge"]


def test_release_gate_does_not_mix_benchmark_modes(tmp_path) -> None:
    for suite, questions, recall in [("locomo", 100, 0.55), ("longmemeval", 25, 0.80)]:
        runs = tmp_path / suite / "runs"
        runs.mkdir(parents=True)
        common = {
            "benchmark": suite,
            "artifact_schema_version": 4,
            "engine_version": "0.3.0",
            "architecture": "trace-and-recall-production",
            "store_validation": {"ok": True},
            "results": [],
        }
        retrieval = {
            **common,
            "run_id": "20260804T000000000000Z",
            "mode": "retrieval",
            "providers": {
                "completion": {"provider": "none"},
                "embedding": {"provider": "test", "model": "test-v1"},
            },
            "prompts": [],
            "summary": {
                "questions": questions,
                "evidence_recall_at_k": recall,
                "mean_token_f1": 0.20,
            },
        }
        (runs / f"trace_recall_{suite}_{retrieval['run_id']}.json").write_text(
            json.dumps(retrieval), encoding="utf-8"
        )

    assert evaluate_release_gate(tmp_path, mode="retrieval")["passed"] is True
    end_to_end = evaluate_release_gate(tmp_path, mode="end-to-end")
    assert end_to_end["passed"] is False
    assert all("no production end-to-end artifact" in value["error"] for value in end_to_end["suites"].values())
