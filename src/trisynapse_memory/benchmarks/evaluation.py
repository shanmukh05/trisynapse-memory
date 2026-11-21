"""Shared benchmark scoring and schema-v4 artifact handling.

This module owns result-row construction, summary metrics, token overlap
scoring, and run-artifact I/O for production Trace & Recall benches. Runners
write rows through `result_row` and `summarize`, then persist them with
`write_benchmark_artifact`. Summary keys are part of the public artifact
contract and must stay stable.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from trisynapse_memory.adapters.benchmarks.base import benchmark_tokens, overlap
from trisynapse_memory.engine import MemoryEngine

BenchmarkMode = Literal["retrieval", "end-to-end"]
BENCHMARK_ARTIFACT_SCHEMA_VERSION = 4
_SAFE_COMPONENT = re.compile(r"^[a-zA-Z0-9_-]+$")


def result_row(
    question_id: str,
    question: str,
    gold: str,
    prediction: str,
    context: str,
    evidence: str,
    *,
    context_tokens: int | None = None,
    recall_time_ms: float | None = None,
    ingest_time_per_doc_ms: float | None = None,
) -> dict[str, Any]:
    row = {
        "question_id": question_id,
        "question": question,
        "gold": gold,
        "prediction": prediction,
        "token_f1": token_f1(prediction, gold),
        "evidence_in_context": overlap(evidence, context) if evidence else overlap(gold, context),
    }
    if context_tokens is not None:
        row["context_tokens"] = context_tokens
    if recall_time_ms is not None:
        row["recall_time_ms"] = recall_time_ms
    if ingest_time_per_doc_ms is not None:
        row["ingest_time_per_doc_ms"] = ingest_time_per_doc_ms
    return row


def _mean(values: list[float]) -> float:
    if not values:
        return 0
    return sum(values) / len(values)


def _float_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _int_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(int(row[key])) for row in rows if row.get(key) is not None]


def _rows_with_value(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get(key) is not None]


def _every_row_has(rows: list[dict[str, Any]], key: str) -> bool:
    if not rows:
        return False
    return all(key in row for row in rows)


def _flag_mean(rows: list[dict[str, Any]], key: str) -> float:
    return _mean([float(bool(row.get(key))) for row in rows])


def _judge_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    judged = [row["judge"] for row in rows if "judge" in row]
    if not judged:
        return {}
    judge_accuracy = _mean([float(bool(item["correct"])) for item in judged])
    return {
        "mean_judge_score": _mean([item["score"] for item in judged]),
        "judge_accuracy": judge_accuracy,
        "accuracy": judge_accuracy,
    }


def _evidence_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_rows = _rows_with_value(rows, "evidence_recall_at_k")
    if evidence_rows:
        return {
            "evidence_recall_at_k": _mean(_float_values(evidence_rows, "evidence_recall_at_k")),
            "evidence_hit_at_k": _flag_mean(evidence_rows, "evidence_hit_at_k"),
            "all_evidence_retrieved_rate": _flag_mean(evidence_rows, "all_evidence_retrieved"),
        }
    if not _every_row_has(rows, "evidence_hit"):
        return {}
    # Compatibility for adapters whose datasets provide evidence text but no
    # stable evidence identifiers.
    return {"evidence_recall_at_k": _flag_mean(rows, "evidence_hit")}


def _citation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recall = _float_values(rows, "citation_evidence_recall")
    precision = _float_values(rows, "citation_precision")
    metrics: dict[str, Any] = {}
    if recall:
        metrics["citation_evidence_recall"] = _mean(recall)
    if precision:
        metrics["citation_precision"] = _mean(precision)
    return metrics


def _accuracy_metrics(rows: list[dict[str, Any]], partial_result: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if _every_row_has(rows, "abstain"):
        metrics["abstention_rate"] = _flag_mean(rows, "abstain")

    judged = _judge_summary(rows)
    if judged:
        metrics.update(judged)
        return metrics

    scored = _float_values(rows, "accuracy")
    if scored:
        metrics["accuracy"] = _mean(scored)
        return metrics

    if "evidence_hit_at_k" in partial_result:
        metrics["accuracy"] = partial_result["evidence_hit_at_k"]
        return metrics
    if "evidence_recall_at_k" in partial_result:
        metrics["accuracy"] = partial_result["evidence_recall_at_k"]
    return metrics


def _timing_metrics(rows: list[dict[str, Any]], ingest_metrics: dict[str, Any] | None) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    recall_times = _float_values(rows, "recall_time_ms")
    if recall_times:
        mean_recall = _mean(recall_times)
        metrics["mean_recall_time_ms"] = mean_recall
        metrics["recall_average_time_ms"] = mean_recall

    context_tokens = _int_values(rows, "context_tokens")
    if context_tokens:
        metrics["mean_context_tokens"] = _mean(context_tokens)

    ingest_times = _float_values(rows, "ingest_time_per_doc_ms")
    if ingest_times:
        mean_ingest = _mean(ingest_times)
        metrics["mean_ingest_time_per_doc_ms"] = mean_ingest
        metrics["ingest_time_per_doc_ms"] = mean_ingest

    if ingest_metrics:
        metrics.update(ingest_metrics)
    return metrics


def _category_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not _rows_with_value(rows, "category"):
        return {}
    categories: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        categories.setdefault(str(row.get("category", "uncategorized")), []).append(row)
    return {
        "by_category": {
            category: summarize(values, include_categories=False)
            for category, values in sorted(categories.items())
        }
    }


def summarize(
    rows: list[dict[str, Any]],
    *,
    include_categories: bool = True,
    ingest_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "questions": len(rows),
        "mean_token_f1": _mean([row["token_f1"] for row in rows]),
        "evidence_in_context_rate": _mean([row["evidence_in_context"] for row in rows]),
    }
    result.update(_evidence_metrics(rows))
    result.update(_citation_metrics(rows))
    result.update(_accuracy_metrics(rows, result))
    result.update(_timing_metrics(rows, ingest_metrics))
    if include_categories:
        result.update(_category_breakdown(rows))
    return result


def token_f1(prediction: str, gold: str) -> float:
    predicted = benchmark_tokens(prediction)
    target = benchmark_tokens(gold)
    if not predicted or not target:
        return 0
    common = sum((Counter(predicted) & Counter(target)).values())
    if not common:
        return 0
    precision = common / len(predicted)
    recall = common / len(target)
    return 2 * precision * recall / (precision + recall)


def write_benchmark_artifact(
    output_root: str | Path,
    *,
    suite: str,
    mode: BenchmarkMode,
    providers: dict[str, Any],
    prompts: list[dict[str, str]],
    summary: dict[str, Any],
    store_validation: dict[str, Any],
    results: list[dict[str, Any]],
    run_configuration: dict[str, Any] | None = None,
) -> tuple[str, Path]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / f"trace_recall_{suite}_{run_id}.json"
    artifact = {
        "run_id": run_id,
        "benchmark": suite,
        "artifact_schema_version": BENCHMARK_ARTIFACT_SCHEMA_VERSION,
        "engine_version": MemoryEngine.VERSION,
        "architecture": "trace-and-recall-production",
        "mode": mode,
        "providers": providers,
        "prompts": prompts,
        "run_configuration": run_configuration or {},
        "summary": summary,
        "store_validation": store_validation,
        "results": results,
    }
    output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return run_id, output


def discover_benchmark_runs(storage_root: str | Path) -> list[dict[str, Any]]:
    root = Path(storage_root)
    runs: list[dict[str, Any]] = []
    for path in root.glob("*/runs/trace_recall_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("architecture") != "trace-and-recall-production":
            continue
        runs.append({
            "run_id": payload.get("run_id"),
            "benchmark": payload.get("benchmark"),
            "engine_version": payload.get("engine_version"),
            "architecture": payload.get("architecture"),
            "mode": payload.get("mode", "retrieval"),
            "artifact_schema_version": payload.get("artifact_schema_version", 1),
            "status": "completed",
            "summary": payload.get("summary") or {},
            "artifact": str(path),
        })
    return sorted(runs, key=lambda item: str(item.get("run_id") or ""), reverse=True)


def read_benchmark_run(storage_root: str | Path, benchmark: str, run_id: str) -> dict[str, Any]:
    if not _SAFE_COMPONENT.fullmatch(benchmark) or not _SAFE_COMPONENT.fullmatch(run_id):
        raise ValueError("benchmark and run id must be safe path components")
    path = Path(storage_root) / benchmark / "runs" / f"trace_recall_{benchmark}_{run_id}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))
