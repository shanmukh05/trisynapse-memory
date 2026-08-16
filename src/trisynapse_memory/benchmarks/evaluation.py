"""Shared benchmark scoring and schema-v2 artifact handling."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from trisynapse_memory.engine import MemoryEngine

BenchmarkMode = Literal["retrieval", "end-to-end"]
BENCHMARK_ARTIFACT_SCHEMA_VERSION = 2
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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    result = {
        "questions": count,
        "mean_token_f1": sum(row["token_f1"] for row in rows) / count if count else 0,
        "evidence_in_context_rate": sum(row["evidence_in_context"] for row in rows) / count if count else 0,
    }
    if rows and all("evidence_hit" in row for row in rows):
        result["evidence_recall_at_k"] = sum(bool(row["evidence_hit"]) for row in rows) / count
    judged = [row["judge"] for row in rows if "judge" in row]
    if judged:
        result["mean_judge_score"] = sum(item["score"] for item in judged) / len(judged)
        result["judge_accuracy"] = sum(bool(item["correct"]) for item in judged) / len(judged)
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
    trace_verification: dict[str, Any],
    results: list[dict[str, Any]],
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
        "summary": summary,
        "trace_verification": trace_verification,
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
