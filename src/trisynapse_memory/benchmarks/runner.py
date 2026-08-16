"""Shared production benchmark orchestration."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from trisynapse_memory.adapters.benchmarks import get_adapter
from trisynapse_memory.adapters.benchmarks.base import BenchmarkAdapter, PreparedCase
from trisynapse_memory.benchmarks.evaluation import (
    BenchmarkMode,
    result_row,
    summarize,
    write_benchmark_artifact,
)
from trisynapse_memory.engine import MemoryEngine
from trisynapse_memory.engine.models import ModelConfiguration
from trisynapse_memory.engine.providers import provider_provenance
from trisynapse_memory.prompts import load_prompt, prompt_provenance

_END_TO_END_PROMPTS = ("extraction", "episode_recall", "answer", "benchmark_judge")


def run_benchmark(
    suite: str,
    data_root: str | Path,
    *,
    max_questions: int = 25,
    output_root: str | Path | None = None,
    mode: BenchmarkMode = "retrieval",
    model_configuration: ModelConfiguration | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run any registered dataset adapter through the same production engine."""

    if max_questions < 1:
        raise ValueError("max_questions must be at least 1")
    mode = validate_mode(mode)
    adapter = get_adapter(suite)
    dataset_path = adapter.resolve_dataset(data_root)
    with tempfile.TemporaryDirectory(prefix=f"trisynapse-{suite}-") as store_path:
        engine = (
            _open_benchmark_engine(store_path, mode)
            if model_configuration is None
            else _open_benchmark_engine(store_path, mode, model_configuration)
        )
        try:
            rows = _execute_cases(engine, adapter, dataset_path, max_questions, mode)
            summary = summarize(rows)
            verification = engine.verify_trace().model_dump(mode="json")
            providers = _provider_provenance(engine)
        finally:
            engine.close()
    root = Path(output_root) if output_root is not None else _default_output_root(data_root, dataset_path)
    run_id, output = write_benchmark_artifact(
        root,
        suite=suite,
        mode=mode,
        providers=providers,
        prompts=prompt_provenance(_END_TO_END_PROMPTS if mode == "end-to-end" else ()),
        summary=summary,
        trace_verification=verification,
        results=rows,
    )
    return {**summary, "run_id": run_id, "mode": mode, "artifact": str(output)}


def run_trace_recall_benchmark(
    suite: str,
    data_root: str | Path,
    *,
    max_questions: int = 25,
    output_root: str | Path | None = None,
    mode: BenchmarkMode = "retrieval",
    model_configuration: ModelConfiguration | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility entry point for release-gated LoCoMo/LongMemEval runs."""

    if suite not in {"locomo", "longmemeval"}:
        raise ValueError("production benchmark runner supports locomo and longmemeval")
    return run_benchmark(
        suite, data_root, max_questions=max_questions, output_root=output_root,
        mode=mode, model_configuration=model_configuration,
    )


def run_trace_recall_smoke(
    suite: str,
    data_root: str | Path,
    *,
    max_questions: int = 25,
    output_root: str | Path | None = None,
    mode: BenchmarkMode = "retrieval",
    model_configuration: ModelConfiguration | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility entry point for HaluMem and MemoryDoc smoke runs."""

    if suite not in {"halumem", "memorydoc"}:
        raise ValueError("trace recall smoke runner supports halumem and memorydoc")
    return run_benchmark(
        suite, data_root, max_questions=max_questions, output_root=output_root,
        mode=mode, model_configuration=model_configuration,
    )


def _execute_cases(
    engine: MemoryEngine,
    adapter: BenchmarkAdapter,
    dataset_path: Path,
    max_questions: int,
    mode: BenchmarkMode,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in adapter.load_cases(dataset_path):
        if len(rows) >= max_questions:
            break
        prepared = adapter.ingest_case(engine, case)
        _prepare_case(engine, prepared, mode)
        for question in case.questions:
            if len(rows) >= max_questions:
                break
            result = engine.query(
                question.question,
                episode_prefix=prepared.episode_prefix,
                namespace=prepared.namespace,
            )
            context = " ".join(citation.excerpt for citation in result.citations)
            row = result_row(
                question.id,
                question.question,
                question.gold,
                result.answer,
                context,
                question.evidence_text,
            )
            row.update(adapter.result_metadata(question, result))
            _judge_row(engine, row, mode)
            rows.append(row)
    return rows


def validate_mode(mode: str) -> BenchmarkMode:
    if mode not in {"retrieval", "end-to-end"}:
        raise ValueError("benchmark mode must be 'retrieval' or 'end-to-end'")
    return mode


def _open_benchmark_engine(
    store_path: str | Path,
    mode: BenchmarkMode,
    model_configuration: ModelConfiguration | dict[str, Any] | None = None,
) -> MemoryEngine:
    if mode == "retrieval":
        return MemoryEngine.open(store_path, local_files_only=True, auto_process=False)
    engine = MemoryEngine.from_env(store_path, auto_process=False)
    if model_configuration is not None:
        configuration = (
            model_configuration
            if isinstance(model_configuration, ModelConfiguration)
            else ModelConfiguration.model_validate(model_configuration)
        )
        configuration.revision = engine.get_model_configuration().revision
        engine.set_model_configuration(configuration)
    if engine.completion is None:
        engine.close()
        raise RuntimeError(
            "end-to-end benchmark mode requires a saved completion model and its provider credential"
        )
    return engine


def _prepare_case(engine: MemoryEngine, prepared: PreparedCase, mode: BenchmarkMode) -> None:
    if mode == "end-to-end":
        for episode_id in prepared.episode_ids:
            engine.run_extraction(episode_id=episode_id, namespace=prepared.namespace)
    engine.build_episode_recall(list(prepared.episode_ids), namespace=prepared.namespace)


def _provider_provenance(engine: MemoryEngine) -> dict[str, dict[str, Any]]:
    return {
        "completion": provider_provenance(engine.completion, kind="completion"),
        "embedding": provider_provenance(engine.embedder, kind="embedding"),
    }


def _judge_row(engine: MemoryEngine, row: dict[str, Any], mode: BenchmarkMode) -> None:
    if mode != "end-to-end":
        return
    if engine.completion is None:
        raise RuntimeError("end-to-end benchmark mode requires a completion provider")
    payload = engine.completion(
        load_prompt("benchmark_judge").text,
        json.dumps(
            {
                "question": row["question"],
                "reference_answer": row["gold"],
                "system_answer": row["prediction"],
            },
            ensure_ascii=False,
        ),
    )
    score = max(0.0, min(1.0, float(payload.get("score", 1.0 if payload.get("correct") else 0.0))))
    row["judge"] = {
        "correct": bool(payload.get("correct", score >= 0.5)),
        "score": score,
        "reason": str(payload.get("reason") or "").strip(),
    }


def _default_output_root(data_root: str | Path, dataset_path: Path) -> Path:
    original = Path(data_root)
    return original.parent / "runs" if original.is_file() else original / "runs"
