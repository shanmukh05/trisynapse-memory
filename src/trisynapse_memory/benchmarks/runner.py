"""Shared production benchmark orchestration."""

from __future__ import annotations

import json
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from trisynapse_memory.adapters.benchmarks import get_adapter
from trisynapse_memory.adapters.benchmarks.base import BenchmarkAdapter, PreparedCase
from trisynapse_memory.benchmarks.evaluation import (
    BenchmarkMode,
    result_row,
    summarize,
    write_benchmark_artifact,
)
from trisynapse_memory.engine import MemoryEngine
from trisynapse_memory.engine.models import ModelConfiguration, ProviderSelection
from trisynapse_memory.engine.providers.registry import (
    completion_from_settings,
    provider_provenance,
    settings_from_selection,
)
from trisynapse_memory.prompts import load_prompt, prompt_provenance

_END_TO_END_PROMPTS = ("extraction", "episode_recall", "answer", "benchmark_judge")


@dataclass(frozen=True)
class BenchmarkProgress:
    """A provider-neutral benchmark progress update.

    Library callers can use these events in notebooks, web applications, or their
    own terminal UI without depending on Rich. The CLI renders the same events as
    a progress bar on stderr.
    """

    stage: str
    description: str
    completed: int
    total: int
    case_id: str | None = None
    question_id: str | None = None


BenchmarkProgressCallback = Callable[[BenchmarkProgress], None]
BenchmarkSampling = Literal["auto", "sequential", "stratified"]


def run_benchmark(
    suite: str,
    data_root: str | Path,
    *,
    max_questions: int = 25,
    output_root: str | Path | None = None,
    mode: BenchmarkMode = "retrieval",
    model_configuration: ModelConfiguration | dict[str, Any] | None = None,
    on_progress: BenchmarkProgressCallback | None = None,
    sampling: BenchmarkSampling = "auto",
    judge_completion: Callable[[str, str], dict[str, Any]] | None = None,
    judge_selection: ProviderSelection | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run any registered dataset adapter through the same production engine."""

    if max_questions < 1:
        raise ValueError("max_questions must be at least 1")
    mode = validate_mode(mode)
    if judge_completion is not None and judge_selection is not None:
        raise ValueError("pass judge_completion or judge_selection, not both")
    if judge_selection is not None:
        selection = (
            judge_selection
            if isinstance(judge_selection, ProviderSelection)
            else ProviderSelection.model_validate(judge_selection)
        )
        judge_completion = completion_from_settings(settings_from_selection(selection))
    _notify(on_progress, "dataset", f"Loading {suite} dataset", 0, max_questions)
    adapter = get_adapter(suite)
    dataset_path = adapter.resolve_dataset(data_root)
    _notify(on_progress, "store", "Opening isolated benchmark store", 0, max_questions)
    with tempfile.TemporaryDirectory(prefix=f"trisynapse-{suite}-") as store_path:
        engine = (
            _open_benchmark_engine(store_path, mode)
            if model_configuration is None
            else _open_benchmark_engine(store_path, mode, model_configuration)
        )
        try:
            effective_sampling = "stratified" if sampling == "auto" and suite == "locomo" else (
                "sequential" if sampling == "auto" else sampling
            )
            rows = _execute_cases(
                engine,
                adapter,
                dataset_path,
                max_questions,
                mode,
                on_progress=on_progress,
                sampling=effective_sampling,
                judge_completion=judge_completion,
            )
            completed = len(rows)
            _notify(on_progress, "summary", "Calculating benchmark metrics", completed, max_questions)
            summary = summarize(rows)
            _notify(on_progress, "validation", "Validating benchmark store", completed, max_questions)
            validation = engine.validate_store().model_dump(mode="json")
            providers = _provider_provenance(engine, judge_completion=judge_completion)
        finally:
            engine.close()
    root = Path(output_root) if output_root is not None else _default_output_root(data_root, dataset_path)
    _notify(on_progress, "artifact", "Writing benchmark artifact", completed, max_questions)
    run_id, output = write_benchmark_artifact(
        root,
        suite=suite,
        mode=mode,
        providers=providers,
        prompts=prompt_provenance(_END_TO_END_PROMPTS if mode == "end-to-end" else ()),
        summary=summary,
        store_validation=validation,
        results=rows,
        run_configuration={
            "max_questions": max_questions,
            "sampling": effective_sampling,
            "category_counts": dict(Counter(str(row.get("category")) for row in rows)),
        },
    )
    _notify(on_progress, "complete", f"Completed {completed} questions", completed, completed)
    return {**summary, "run_id": run_id, "mode": mode, "artifact": str(output)}


def run_trace_recall_benchmark(
    suite: str,
    data_root: str | Path,
    *,
    max_questions: int = 25,
    output_root: str | Path | None = None,
    mode: BenchmarkMode = "retrieval",
    model_configuration: ModelConfiguration | dict[str, Any] | None = None,
    on_progress: BenchmarkProgressCallback | None = None,
    sampling: BenchmarkSampling = "auto",
    judge_completion: Callable[[str, str], dict[str, Any]] | None = None,
    judge_selection: ProviderSelection | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility entry point for release-gated LoCoMo/LongMemEval runs."""

    if suite not in {"locomo", "longmemeval"}:
        raise ValueError("production benchmark runner supports locomo and longmemeval")
    return run_benchmark(
        suite, data_root, max_questions=max_questions, output_root=output_root,
        mode=mode, model_configuration=model_configuration, on_progress=on_progress,
        sampling=sampling, judge_completion=judge_completion, judge_selection=judge_selection,
    )


def run_trace_recall_smoke(
    suite: str,
    data_root: str | Path,
    *,
    max_questions: int = 25,
    output_root: str | Path | None = None,
    mode: BenchmarkMode = "retrieval",
    model_configuration: ModelConfiguration | dict[str, Any] | None = None,
    on_progress: BenchmarkProgressCallback | None = None,
    sampling: BenchmarkSampling = "auto",
    judge_completion: Callable[[str, str], dict[str, Any]] | None = None,
    judge_selection: ProviderSelection | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility entry point for HaluMem and MemoryDoc smoke runs."""

    if suite not in {"halumem", "memorydoc"}:
        raise ValueError("trace recall smoke runner supports halumem and memorydoc")
    return run_benchmark(
        suite, data_root, max_questions=max_questions, output_root=output_root,
        mode=mode, model_configuration=model_configuration, on_progress=on_progress,
        sampling=sampling, judge_completion=judge_completion, judge_selection=judge_selection,
    )


def _execute_cases(
    engine: MemoryEngine,
    adapter: BenchmarkAdapter,
    dataset_path: Path,
    max_questions: int,
    mode: BenchmarkMode,
    *,
    on_progress: BenchmarkProgressCallback | None = None,
    sampling: Literal["sequential", "stratified"] = "sequential",
    judge_completion: Callable[[str, str], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = list(adapter.load_cases(dataset_path))
    selected = _select_questions(cases, max_questions=max_questions, sampling=sampling)
    by_case: dict[int, list[Any]] = defaultdict(list)
    for case_index, question in selected:
        by_case[case_index].append(question)
    for case_index, case in enumerate(cases):
        selected_questions = by_case.get(case_index, [])
        if not selected_questions:
            continue
        _notify(
            on_progress,
            "ingestion",
            f"Ingesting benchmark case {case.id}",
            len(rows),
            max_questions,
            case_id=case.id,
        )
        prepared = adapter.ingest_case(engine, case)
        _notify(
            on_progress,
            "preparation",
            f"Preparing Recall for case {case.id}",
            len(rows),
            max_questions,
            case_id=case.id,
        )
        _prepare_case(engine, prepared, mode)
        for question in selected_questions:
            _notify(
                on_progress,
                "query",
                f"Running question {len(rows) + 1} of {max_questions}",
                len(rows),
                max_questions,
                case_id=case.id,
                question_id=question.id,
            )
            result = engine.query(
                question.question,
                episode_prefix=prepared.episode_prefix,
                namespace=prepared.namespace,
            )
            context = " ".join(hit.text for hit in result.retrieval_hits)
            row = result_row(
                question.id,
                question.question,
                question.gold,
                result.answer,
                context,
                question.evidence_text,
            )
            row.update({
                "abstain": result.abstain,
                "retrieval_trace": result.retrieval_trace.model_dump(mode="json"),
                "retrieval_hits": [hit.model_dump(mode="json") for hit in result.retrieval_hits],
            })
            query_run = engine.get_query_run(result.query_id, namespace=prepared.namespace)
            row["query_steps"] = [step.model_dump(mode="json") for step in query_run.steps]
            row.update(adapter.result_metadata(
                question,
                result,
                retrieval_hits=result.retrieval_hits,
            ))
            _judge_row(judge_completion or engine.completion, row, mode)
            rows.append(row)
            _notify(
                on_progress,
                "questions",
                f"Completed question {len(rows)} of {max_questions}",
                len(rows),
                max_questions,
                case_id=case.id,
                question_id=question.id,
            )
    return rows


def _select_questions(
    cases: list[Any],
    *,
    max_questions: int,
    sampling: Literal["sequential", "stratified"],
) -> list[tuple[int, Any]]:
    if sampling == "sequential":
        return [
            (case_index, question)
            for case_index, case in enumerate(cases)
            for question in case.questions
        ][:max_questions]

    # Deterministic round-robin over category and conversation. Partial LoCoMo
    # runs therefore do not silently measure only the first conversation.
    buckets: dict[str, dict[int, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for case_index, case in enumerate(cases):
        for question in case.questions:
            category = str(question.metadata.get("category", "uncategorized"))
            buckets[category][case_index].append(question)
    selected: list[tuple[int, Any]] = []
    categories = sorted(buckets)
    case_cursors = {
        category: position % max(len(cases), 1)
        for position, category in enumerate(categories)
    }
    category_cursor = 0
    stalled = 0
    while len(selected) < max_questions and categories:
        category = categories[category_cursor % len(categories)]
        category_cursor += 1
        found = False
        for offset in range(len(cases)):
            case_index = (case_cursors[category] + offset) % len(cases)
            values = buckets[category].get(case_index)
            if values:
                selected.append((case_index, values.pop(0)))
                case_cursors[category] = (case_index + 1) % len(cases)
                found = True
                stalled = 0
                break
        if not found:
            stalled += 1
            if stalled >= len(categories):
                break
    return selected


def _notify(
    callback: BenchmarkProgressCallback | None,
    stage: str,
    description: str,
    completed: int,
    total: int,
    *,
    case_id: str | None = None,
    question_id: str | None = None,
) -> None:
    if callback is not None:
        callback(BenchmarkProgress(stage, description, completed, total, case_id, question_id))


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


def _provider_provenance(
    engine: MemoryEngine,
    *,
    judge_completion: Callable[[str, str], dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        "completion": provider_provenance(engine.completion, kind="completion"),
        "embedding": provider_provenance(engine.embedder, kind="embedding"),
        "judge": provider_provenance(judge_completion or engine.completion, kind="judge"),
    }


def _judge_row(
    completion: Callable[[str, str], dict[str, Any]] | None,
    row: dict[str, Any],
    mode: BenchmarkMode,
) -> None:
    if mode != "end-to-end":
        return
    if completion is None:
        raise RuntimeError("end-to-end benchmark mode requires a completion provider")
    payload = completion(
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
