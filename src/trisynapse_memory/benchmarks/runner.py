"""Shared production benchmark orchestration.

Owns ingest-once / query-many execution, sampling, and checkpoint reuse for
every registered dataset adapter. Adapters parse datasets; this module does
not. Sampling and store reuse must stay deterministic so a resumed run matches
a clean run with the same model and question cap.
"""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Literal

from trisynapse_memory.adapters.benchmarks import get_adapter
from trisynapse_memory.adapters.benchmarks.base import BenchmarkAdapter, BenchmarkCase, BenchmarkQuestion, PreparedCase
from trisynapse_memory.benchmarks.evaluation import (
    BenchmarkMode,
    result_row,
    summarize,
    write_benchmark_artifact,
)
from trisynapse_memory.benchmarks.mcq import mcq_answer_question, score_mcq_prediction
from trisynapse_memory.benchmarks.store import (
    benchmark_store_path,
    clear_query_checkpoint,
    embedding_matches_store,
    empty_manifest,
    load_manifest,
    load_query_checkpoint,
    manifest_case_entry,
    prepared_case_from_manifest,
    save_manifest,
    save_query_checkpoint,
)
from trisynapse_memory.engine import MemoryEngine
from trisynapse_memory.engine.models import ModelConfiguration, ProviderSelection
from trisynapse_memory.engine.providers.registry import (
    completion_from_settings,
    provider_provenance,
    settings_from_selection,
    skippable_completion_error,
)
from trisynapse_memory.engine.retrieval.tokenization import TokenCounter, token_counter_for
from trisynapse_memory.prompts import load_prompt, prompt_provenance

_END_TO_END_PROMPTS = ("facts", "group_summary", "answer", "benchmark_judge")


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
BenchmarkStage = Literal["auto", "ingest", "query"]


def run_benchmark(
    suite: str,
    data_root: str | Path,
    *,
    dataset: str | Path | None = None,
    max_questions: int | None = None,
    output_root: str | Path | None = None,
    mode: BenchmarkMode = "retrieval",
    model_configuration: ModelConfiguration | dict[str, Any] | None = None,
    on_progress: BenchmarkProgressCallback | None = None,
    sampling: BenchmarkSampling = "auto",
    stage: BenchmarkStage = "auto",
    rebuild_store: bool = False,
    judge_completion: Callable[[str, str], dict[str, Any]] | None = None,
    judge_selection: ProviderSelection | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run any registered dataset adapter through the same production engine."""

    if max_questions is not None and max_questions < 1:
        raise ValueError("max_questions must be at least 1")
    mode = validate_mode(mode)
    stage = validate_stage(stage)
    configuration = _model_configuration(model_configuration)
    judge_completion = _resolve_judge_completion(judge_completion, judge_selection)
    adapter, dataset_path, store_path = _prepare_benchmark_store(
        suite, data_root, configuration, rebuild_store, stage, max_questions, on_progress, dataset=dataset
    )
    _notify(on_progress, "store", f"Opening benchmark store {store_path}", 0, max_questions or 0)
    engine = (
        _open_benchmark_engine(store_path, mode)
        if configuration is None
        else _open_benchmark_engine(store_path, mode, configuration)
    )
    effective_sampling = _effective_sampling(sampling, suite)
    try:
        rows, ingest_metrics = _execute_cases(
            engine,
            adapter,
            dataset_path,
            max_questions,
            mode,
            store_path=store_path,
            stage=stage,
            configuration=configuration,
            suite=suite,
            on_progress=on_progress,
            sampling=effective_sampling,
            judge_completion=judge_completion,
        )
        completed = len(rows)
        _notify(on_progress, "summary", "Calculating benchmark metrics", completed, completed)
        summary = summarize(rows, ingest_metrics=ingest_metrics)
        _notify(on_progress, "validation", "Validating benchmark store", completed, completed)
        validation = engine.validate_store().model_dump(mode="json")
        providers = _provider_provenance(engine, judge_completion=judge_completion)
    finally:
        engine.close()
    return _write_benchmark_output(
        suite,
        data_root,
        dataset_path,
        store_path,
        mode,
        stage,
        max_questions,
        effective_sampling,
        output_root,
        rows,
        summary,
        validation,
        providers,
        completed,
        on_progress,
    )


def run_trace_recall_benchmark(
    suite: str,
    data_root: str | Path,
    *,
    max_questions: int | None = None,
    output_root: str | Path | None = None,
    mode: BenchmarkMode = "retrieval",
    model_configuration: ModelConfiguration | dict[str, Any] | None = None,
    on_progress: BenchmarkProgressCallback | None = None,
    sampling: BenchmarkSampling = "auto",
    stage: BenchmarkStage = "auto",
    rebuild_store: bool = False,
    judge_completion: Callable[[str, str], dict[str, Any]] | None = None,
    judge_selection: ProviderSelection | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility entry point for release-gated LoCoMo/LongMemEval runs."""

    if suite not in {"locomo", "longmemeval", "personamem", "beam100k", "beam10m"}:
        raise ValueError("production benchmark runner supports locomo, longmemeval, personamem, beam100k, and beam10m")
    return run_benchmark(
        suite, data_root, max_questions=max_questions, output_root=output_root,
        mode=mode, model_configuration=model_configuration, on_progress=on_progress,
        sampling=sampling, stage=stage, rebuild_store=rebuild_store,
        judge_completion=judge_completion, judge_selection=judge_selection,
    )


def run_trace_recall_smoke(
    suite: str,
    data_root: str | Path,
    *,
    max_questions: int | None = None,
    output_root: str | Path | None = None,
    mode: BenchmarkMode = "retrieval",
    model_configuration: ModelConfiguration | dict[str, Any] | None = None,
    on_progress: BenchmarkProgressCallback | None = None,
    sampling: BenchmarkSampling = "auto",
    stage: BenchmarkStage = "auto",
    rebuild_store: bool = False,
    judge_completion: Callable[[str, str], dict[str, Any]] | None = None,
    judge_selection: ProviderSelection | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility entry point for HaluMem and MemoryDoc smoke runs."""

    if suite not in {"halumem", "memorydoc"}:
        raise ValueError("trace recall smoke runner supports halumem and memorydoc")
    return run_benchmark(
        suite, data_root, max_questions=max_questions, output_root=output_root,
        mode=mode, model_configuration=model_configuration, on_progress=on_progress,
        sampling=sampling, stage=stage, rebuild_store=rebuild_store,
        judge_completion=judge_completion, judge_selection=judge_selection,
    )


def _execute_cases(
    engine: MemoryEngine,
    adapter: BenchmarkAdapter,
    dataset_path: Path,
    max_questions: int | None,
    mode: BenchmarkMode,
    *,
    store_path: Path,
    stage: BenchmarkStage,
    configuration: ModelConfiguration | None,
    suite: str,
    on_progress: BenchmarkProgressCallback | None = None,
    sampling: Literal["sequential", "stratified"] = "sequential",
    judge_completion: Callable[[str, str], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = list(adapter.load_cases(dataset_path))
    selected = [] if stage == "ingest" else _select_questions(
        cases, max_questions=max_questions, sampling=sampling
    )
    by_case: dict[int, list[BenchmarkQuestion]] = defaultdict(list)
    for case_index, question in selected:
        by_case[case_index].append(question)
    needed_indexes = list(range(len(cases))) if stage == "ingest" else sorted(by_case)
    token_counter = token_counter_for(engine.completion)
    ingest_totals = {"ms": 0.0, "docs": 0}
    manifest = load_manifest(store_path) or empty_manifest(suite, configuration)
    checkpoint_rows = load_query_checkpoint(store_path, mode) if stage != "ingest" else []
    checkpoint_by_id = {str(row["question_id"]): row for row in checkpoint_rows}
    for case_index in needed_indexes:
        case = cases[case_index]
        prepared, ingest_time_per_doc_ms = _prepared_case_for_run(
            engine,
            adapter,
            case,
            manifest=manifest,
            store_path=store_path,
            stage=stage,
            on_progress=on_progress,
            completed=len(rows),
            total_questions=len(selected),
            ingest_totals=ingest_totals,
        )
        if stage == "ingest":
            continue
        _query_case_questions(
            engine,
            adapter,
            case,
            prepared,
            by_case[case_index],
            case_ingest_time_per_doc_ms=ingest_time_per_doc_ms,
            token_counter=token_counter,
            mode=mode,
            store_path=store_path,
            on_progress=on_progress,
            rows=rows,
            total_questions=len(selected),
            checkpoint_by_id=checkpoint_by_id,
            judge_completion=judge_completion,
        )
    return rows, _ingest_metrics(ingest_totals)


def _prepared_case_for_run(
    engine: MemoryEngine,
    adapter: BenchmarkAdapter,
    case: BenchmarkCase,
    *,
    manifest: dict[str, Any],
    store_path: Path,
    stage: BenchmarkStage,
    on_progress: BenchmarkProgressCallback | None,
    completed: int,
    total_questions: int,
    ingest_totals: dict[str, float],
) -> tuple[PreparedCase, float]:
    cached = manifest["cases"].get(case.id)
    if cached is not None:
        _notify(
            on_progress,
            "reuse",
            f"Reusing ingested case {case.id}",
            completed,
            total_questions,
            case_id=case.id,
        )
        return prepared_case_from_manifest(cached), float(cached.get("ingest_time_per_doc_ms") or 0.0)
    if stage == "query":
        raise FileNotFoundError(
            f"benchmark case {case.id} is not in {store_path}. "
            "Run --stage ingest for this model, or --stage auto."
        )
    return _ingest_one_case(
        engine,
        adapter,
        case,
        manifest=manifest,
        store_path=store_path,
        on_progress=on_progress,
        completed=completed,
        total_questions=total_questions,
        ingest_totals=ingest_totals,
    )


def _ingest_one_case(
    engine: MemoryEngine,
    adapter: BenchmarkAdapter,
    case: BenchmarkCase,
    *,
    manifest: dict[str, Any],
    store_path: Path,
    on_progress: BenchmarkProgressCallback | None,
    completed: int,
    total_questions: int,
    ingest_totals: dict[str, float],
) -> tuple[PreparedCase, float]:
    _notify(
        on_progress,
        "ingestion",
        f"Ingesting benchmark case {case.id}",
        completed,
        total_questions,
        case_id=case.id,
    )
    ingest_start = time.perf_counter()
    prepared = adapter.ingest_case(engine, case)
    _prepare_case(
        engine,
        prepared,
        on_progress=on_progress,
        completed=completed,
        total=total_questions,
        case_id=case.id,
    )
    case_duration_ms = (time.perf_counter() - ingest_start) * 1000
    doc_count = max(len(prepared.group_ids), 1)
    case_ingest_time_per_doc_ms = case_duration_ms / doc_count
    ingest_totals["ms"] += case_duration_ms
    ingest_totals["docs"] += doc_count
    manifest["cases"][case.id] = manifest_case_entry(
        prepared, doc_count=doc_count, ingest_time_ms=case_duration_ms
    )
    save_manifest(store_path, manifest)
    return prepared, case_ingest_time_per_doc_ms


def _query_case_questions(
    engine: MemoryEngine,
    adapter: BenchmarkAdapter,
    case: BenchmarkCase,
    prepared: PreparedCase,
    selected_questions: list[BenchmarkQuestion],
    *,
    case_ingest_time_per_doc_ms: float,
    token_counter: TokenCounter,
    mode: BenchmarkMode,
    store_path: Path,
    on_progress: BenchmarkProgressCallback | None,
    rows: list[dict[str, Any]],
    total_questions: int,
    checkpoint_by_id: dict[str, dict[str, Any]],
    judge_completion: Callable[[str, str], dict[str, Any]] | None,
) -> None:
    query_completion = engine.completion
    if mode == "retrieval":
        engine.completion = None
    try:
        for question in selected_questions:
            row = checkpoint_by_id.get(question.id)
            reused = row is not None
            if row is None:
                _notify(
                    on_progress,
                    "query",
                    f"Running question {len(rows) + 1} of {total_questions}",
                    len(rows),
                    total_questions,
                    case_id=case.id,
                    question_id=question.id,
                )
                row = _query_one_question(
                    engine,
                    adapter,
                    prepared,
                    question,
                    case_ingest_time_per_doc_ms=case_ingest_time_per_doc_ms,
                    token_counter=token_counter,
                    mode=mode,
                    query_completion=query_completion,
                    judge_completion=judge_completion,
                )
                checkpoint_by_id[question.id] = row
                save_query_checkpoint(store_path, mode, list(checkpoint_by_id.values()))
            rows.append(row)
            action = "Reusing" if reused else "Completed"
            _notify(
                on_progress,
                "questions",
                f"{action} question {len(rows)} of {total_questions}",
                len(rows),
                total_questions,
                case_id=case.id,
                question_id=question.id,
            )
    finally:
        engine.completion = query_completion


def _query_one_question(
    engine: MemoryEngine,
    adapter: BenchmarkAdapter,
    prepared: PreparedCase,
    question: BenchmarkQuestion,
    *,
    case_ingest_time_per_doc_ms: float,
    token_counter: TokenCounter,
    mode: BenchmarkMode,
    query_completion: Any,
    judge_completion: Callable[[str, str], dict[str, Any]] | None,
) -> dict[str, Any]:
    query_start = time.perf_counter()
    answer_question = mcq_answer_question(question) if mode == "end-to-end" and question.choices else None
    result = engine.query(
        question.question,
        group_prefix=prepared.group_prefix,
        namespace=prepared.namespace,
        answer_question=answer_question,
    )
    recall_time_ms = (time.perf_counter() - query_start) * 1000
    context = " ".join(hit.text for hit in result.retrieval_hits)
    row = result_row(
        question.id,
        question.question,
        question.gold,
        result.answer,
        context,
        question.evidence_text,
        context_tokens=token_counter.count(context),
        recall_time_ms=recall_time_ms,
        ingest_time_per_doc_ms=case_ingest_time_per_doc_ms,
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
    if mode == "end-to-end" and question.choices:
        row.update(score_mcq_prediction(result.answer, question))
    else:
        _judge_row(judge_completion or query_completion, row, mode)
    if "accuracy" not in row:
        row["accuracy"] = 1.0 if row.get("evidence_hit_at_k") or row.get("evidence_hit") else 0.0
    return row


def _ingest_metrics(ingest_totals: dict[str, float]) -> dict[str, Any]:
    total_ingest_ms = ingest_totals["ms"]
    total_docs = ingest_totals["docs"]
    per_doc = total_ingest_ms / total_docs if total_docs else 0.0
    return {
        "total_ingest_time_ms": total_ingest_ms,
        "total_docs_ingested": total_docs,
        "mean_ingest_time_per_doc_ms": per_doc,
        "ingest_time_per_doc_ms": per_doc,
    }


def _select_questions(
    cases: list[BenchmarkCase],
    *,
    max_questions: int | None,
    sampling: Literal["sequential", "stratified"],
) -> list[tuple[int, BenchmarkQuestion]]:
    total_available = sum(len(case.questions) for case in cases)
    target_count = total_available if max_questions is None else min(max_questions, total_available)
    if sampling == "sequential":
        return _sequential_question_sample(cases, target_count)
    return _stratified_question_sample(cases, target_count)


def _sequential_question_sample(
    cases: list[BenchmarkCase],
    target_count: int,
) -> list[tuple[int, BenchmarkQuestion]]:
    questions = (
        (case_index, question)
        for case_index, case in enumerate(cases)
        for question in case.questions
    )
    return list(islice(questions, target_count))


def _stratified_question_sample(
    cases: list[BenchmarkCase], target_count: int
) -> list[tuple[int, BenchmarkQuestion]]:
    # Deterministic round-robin over category and conversation. Partial LoCoMo
    # runs therefore do not silently measure only the first conversation.
    buckets = _category_case_buckets(cases)
    selected: list[tuple[int, BenchmarkQuestion]] = []
    categories = sorted(buckets)
    case_cursors = {
        category: position % max(len(cases), 1)
        for position, category in enumerate(categories)
    }
    category_cursor = 0
    stalled = 0
    while len(selected) < target_count and categories:
        category = categories[category_cursor % len(categories)]
        category_cursor += 1
        if _take_next_in_category(buckets, category, len(cases), case_cursors, selected):
            stalled = 0
            continue
        stalled += 1
        if stalled >= len(categories):
            break
    return selected


def _category_case_buckets(cases: list[BenchmarkCase]) -> dict[str, dict[int, deque[BenchmarkQuestion]]]:
    buckets: dict[str, dict[int, deque[BenchmarkQuestion]]] = defaultdict(lambda: defaultdict(deque))
    for case_index, case in enumerate(cases):
        for question in case.questions:
            category = str(question.metadata.get("category", "uncategorized"))
            buckets[category][case_index].append(question)
    return buckets


def _take_next_in_category(
    buckets: dict[str, dict[int, deque[BenchmarkQuestion]]],
    category: str,
    case_count: int,
    case_cursors: dict[str, int],
    selected: list[tuple[int, BenchmarkQuestion]],
) -> bool:
    for offset in range(case_count):
        case_index = (case_cursors[category] + offset) % case_count
        values = buckets[category].get(case_index)
        if values:
            selected.append((case_index, values.popleft()))
            case_cursors[category] = (case_index + 1) % case_count
            return True
    return False


def _resolve_judge_completion(
    judge_completion: Callable[[str, str], dict[str, Any]] | None,
    judge_selection: ProviderSelection | dict[str, Any] | None,
) -> Callable[[str, str], dict[str, Any]] | None:
    if judge_completion is not None and judge_selection is not None:
        raise ValueError("pass judge_completion or judge_selection, not both")
    if judge_selection is None:
        return judge_completion
    selection = (
        judge_selection
        if isinstance(judge_selection, ProviderSelection)
        else ProviderSelection.model_validate(judge_selection)
    )
    return completion_from_settings(settings_from_selection(selection))


def _effective_sampling(sampling: BenchmarkSampling, suite: str) -> Literal["sequential", "stratified"]:
    if sampling == "auto" and suite == "locomo":
        return "stratified"
    if sampling == "auto":
        return "sequential"
    return sampling


def _prepare_benchmark_store(
    suite: str,
    data_root: str | Path,
    configuration: ModelConfiguration | None,
    rebuild_store: bool,
    stage: BenchmarkStage,
    max_questions: int | None,
    on_progress: BenchmarkProgressCallback | None,
    *,
    dataset: str | Path | None = None,
) -> tuple[BenchmarkAdapter, Path, Path]:
    _notify(on_progress, "dataset", f"Loading {suite} dataset", 0, max_questions or 0)
    adapter = get_adapter(suite)
    dataset_path = adapter.resolve_dataset(data_root if dataset is None else dataset)
    suite_root = Path(data_root).parent if Path(data_root).is_file() else Path(data_root)
    custom_dataset = dataset is not None and dataset_path.resolve() != (suite_root / adapter.default_filename).resolve()
    store_path = benchmark_store_path(data_root, dataset_path, configuration, isolate_dataset=custom_dataset)
    if rebuild_store and store_path.exists():
        shutil.rmtree(store_path)
    manifest = load_manifest(store_path)
    if stage == "query" and manifest is None:
        raise FileNotFoundError(
            f"no ingested benchmark store at {store_path}. Run --stage ingest for this model first."
        )
    if manifest is not None and not embedding_matches_store(manifest, configuration):
        raise ValueError(
            f"store at {store_path} was built with embedding {manifest.get('embedding')}; "
            "use the same embedding model or pass rebuild_store=True"
        )
    return adapter, dataset_path, store_path


def _write_benchmark_output(
    suite: str,
    data_root: str | Path,
    dataset_path: Path,
    store_path: Path,
    mode: BenchmarkMode,
    stage: BenchmarkStage,
    max_questions: int | None,
    effective_sampling: str,
    output_root: str | Path | None,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    validation: dict[str, Any],
    providers: dict[str, dict[str, Any]],
    completed: int,
    on_progress: BenchmarkProgressCallback | None,
) -> dict[str, Any]:
    root = Path(output_root) if output_root is not None else _default_output_root(data_root, dataset_path)
    _notify(on_progress, "artifact", "Writing benchmark artifact", completed, completed)
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
            "stage": stage,
            "dataset": str(dataset_path.resolve()),
            "store": str(store_path),
            "category_counts": dict(Counter(str(row.get("category")) for row in rows)),
        },
    )
    if stage != "ingest":
        clear_query_checkpoint(store_path, mode)
    _notify(on_progress, "complete", f"Completed {completed} questions", completed, completed)
    return {
        **summary,
        "run_id": run_id,
        "mode": mode,
        "stage": stage,
        "store": str(store_path),
        "artifact": str(output),
    }


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


def validate_stage(stage: str) -> BenchmarkStage:
    if stage not in {"auto", "ingest", "query"}:
        raise ValueError("benchmark stage must be 'auto', 'ingest', or 'query'")
    return stage


def _model_configuration(
    model_configuration: ModelConfiguration | dict[str, Any] | None,
) -> ModelConfiguration | None:
    if model_configuration is None:
        return None
    if isinstance(model_configuration, ModelConfiguration):
        return model_configuration
    return ModelConfiguration.model_validate(model_configuration)


def _open_benchmark_engine(
    store_path: str | Path,
    mode: BenchmarkMode,
    model_configuration: ModelConfiguration | dict[str, Any] | None = None,
) -> MemoryEngine:
    engine = MemoryEngine.from_env(store_path, auto_process=False)
    try:
        configuration = _model_configuration(model_configuration)
        if configuration is not None:
            engine.set_model_configuration(configuration.model_copy(
                update={"revision": engine.get_model_configuration().revision},
            ))
        if mode == "end-to-end" and engine.completion is None:
            raise RuntimeError(
                "end-to-end benchmark mode requires a saved completion model and its provider credential"
            )
    except BaseException:
        engine.close()
        raise
    return engine


def _prepare_case(
    engine: MemoryEngine,
    prepared: PreparedCase,
    *,
    on_progress: BenchmarkProgressCallback | None = None,
    completed: int = 0,
    total: int = 0,
    case_id: str | None = None,
) -> None:
    group_ids = list(prepared.group_ids)
    session_count = len(group_ids)
    _notify(
        on_progress,
        "preparation",
        f"Preparing Recall for case {case_id} ({session_count} sessions)",
        completed,
        total,
        case_id=case_id,
    )
    if engine.completion is not None:
        engine.run_fact_extraction_many(
            group_ids,
            namespace=prepared.namespace,
            on_progress=lambda done, count: _notify(
                on_progress,
                "extraction",
                f"Extracting facts for case {case_id} ({done}/{count} sessions)",
                completed,
                total,
                case_id=case_id,
            ),
        )
    engine.build_group_summary(
        group_ids,
        namespace=prepared.namespace,
        on_progress=lambda done, count: _notify(
            on_progress,
            "summaries",
            f"Building group summaries for case {case_id} ({done}/{count} sessions)",
            completed,
            total,
            case_id=case_id,
        ),
    )


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
    try:
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
    except Exception as exc:
        if not skippable_completion_error(exc):
            raise
        row["judge"] = {"correct": False, "score": 0.0, "reason": "invalid_json"}
        row["accuracy"] = 0.0
        return
    score = max(0.0, min(1.0, float(payload.get("score", 1.0 if payload.get("correct") else 0.0))))
    correct = bool(payload.get("correct", score >= 0.5))
    row["judge"] = {
        "correct": correct,
        "score": score,
        "reason": str(payload.get("reason") or "").strip(),
    }
    row["accuracy"] = 1.0 if correct else 0.0


def _default_output_root(data_root: str | Path, dataset_path: Path) -> Path:
    original = Path(data_root)
    return original.parent / "runs" if original.is_file() else original / "runs"
