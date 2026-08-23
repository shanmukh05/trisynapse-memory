"""Shared benchmark scoring and schema-v4 artifact handling."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

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
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "question": question,
        "gold": gold,
        "prediction": prediction,
        "token_f1": token_f1(prediction, gold),
        "evidence_in_context": overlap(evidence, context) if evidence else overlap(gold, context),
    }


def summarize(
    rows: list[dict[str, Any]],
    *,
    include_categories: bool = True,
) -> dict[str, Any]:
    count = len(rows)
    result = {
        "questions": count,
        "mean_token_f1": sum(row["token_f1"] for row in rows) / count if count else 0,
        "evidence_in_context_rate": sum(row["evidence_in_context"] for row in rows) / count if count else 0,
    }
    evidence_rows = [row for row in rows if row.get("evidence_recall_at_k") is not None]
    if evidence_rows:
        result["evidence_recall_at_k"] = sum(float(row["evidence_recall_at_k"]) for row in evidence_rows) / len(evidence_rows)
        result["evidence_hit_at_k"] = sum(bool(row.get("evidence_hit_at_k")) for row in evidence_rows) / len(evidence_rows)
        result["all_evidence_retrieved_rate"] = sum(bool(row.get("all_evidence_retrieved")) for row in evidence_rows) / len(evidence_rows)
    elif rows and all("evidence_hit" in row for row in rows):
        # Compatibility for adapters whose datasets provide evidence text but no
        # stable evidence identifiers.
        result["evidence_recall_at_k"] = sum(bool(row["evidence_hit"]) for row in rows) / count
    citation_recall = [row["citation_evidence_recall"] for row in rows if row.get("citation_evidence_recall") is not None]
    citation_precision = [row["citation_precision"] for row in rows if row.get("citation_precision") is not None]
    if citation_recall:
        result["citation_evidence_recall"] = sum(citation_recall) / len(citation_recall)
    if citation_precision:
        result["citation_precision"] = sum(citation_precision) / len(citation_precision)
    if rows and all("abstain" in row for row in rows):
        result["abstention_rate"] = sum(bool(row["abstain"]) for row in rows) / count
    judged = [row["judge"] for row in rows if "judge" in row]
    if judged:
        result["mean_judge_score"] = sum(item["score"] for item in judged) / len(judged)
        result["judge_accuracy"] = sum(bool(item["correct"]) for item in judged) / len(judged)
    if include_categories and any(row.get("category") is not None for row in rows):
        categories: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            categories.setdefault(str(row.get("category", "uncategorized")), []).append(row)
        result["by_category"] = {
            category: summarize(values, include_categories=False)
            for category, values in sorted(categories.items())
        }
    return result


def token_f1(prediction: str, gold: str) -> float:
    predicted = _tokens(prediction)
    target = _tokens(gold)
    if not predicted or not target:
        return 0
    common = sum((Counter(predicted) & Counter(target)).values())
    if not common:
        return 0
    precision = common / len(predicted)
    recall = common / len(target)
    return 2 * precision * recall / (precision + recall)


def overlap(left: str, right: str) -> bool:
    left_tokens = {item for item in _tokens(left) if len(item) > 3}
    right_tokens = set(_tokens(right))
    return bool(left_tokens) and len(left_tokens & right_tokens) / len(left_tokens) >= 0.5


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


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", value.lower())
