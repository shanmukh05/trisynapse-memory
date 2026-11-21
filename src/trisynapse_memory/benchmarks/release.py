"""Reproducible benchmark requirements for a v0 release tag.

Owns the gate that reads the latest production artifact per suite and compares
it to the published recall/judge floors. Runners write artifacts; this module
only scores them. A missing or non-production artifact fails the suite — it
does not invent a passing score.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trisynapse_memory.benchmarks.evaluation import BENCHMARK_ARTIFACT_SCHEMA_VERSION, BenchmarkMode


RELEASE_REQUIREMENTS = {
    "locomo": {"questions": 100, "evidence_recall_at_k": 0.55},
    "longmemeval": {"questions": 25, "evidence_recall_at_k": 0.80},
}
END_TO_END_REQUIREMENTS = {
    suite: {**requirements, "mean_token_f1": 0.20, "judge_accuracy": 0.50}
    for suite, requirements in RELEASE_REQUIREMENTS.items()
}


def evaluate_release_gate(
    data_root: str | Path = "data", *, mode: BenchmarkMode = "retrieval"
) -> dict[str, Any]:
    if mode not in {"retrieval", "end-to-end"}:
        raise ValueError("release gate mode must be 'retrieval' or 'end-to-end'")
    root = Path(data_root)
    selected_requirements = END_TO_END_REQUIREMENTS if mode == "end-to-end" else RELEASE_REQUIREMENTS
    suites: dict[str, Any] = {}
    passed = True
    for suite, requirements in selected_requirements.items():
        report, suite_passed = _evaluate_suite_gate(root, suite, requirements, mode)
        suites[suite] = report
        passed = passed and suite_passed
    return {"passed": passed, "mode": mode, "requirements": selected_requirements, "suites": suites}


def _evaluate_suite_gate(
    root: Path,
    suite: str,
    requirements: dict[str, float],
    mode: BenchmarkMode,
) -> tuple[dict[str, Any], bool]:
    candidates = _mode_artifacts(root, suite, mode)
    if not candidates:
        return {"passed": False, "error": f"no production {mode} artifact"}, False
    artifact, payload = candidates[0]
    checks = _requirement_checks(payload.get("summary") or {}, requirements)
    flags = _artifact_quality_flags(payload, mode)
    suite_passed = _suite_gate_passed(flags, checks)
    return {
        "passed": suite_passed,
        "artifact": str(artifact),
        "production_artifact": flags["production_artifact"],
        "artifact_schema_current": flags["artifact_schema_current"],
        "provenance_present": flags["provenance_present"],
        "workflow_provenance_valid": flags["workflow_provenance_valid"],
        "mode": mode,
        "engine_version": flags["engine_version"],
        "storage_ok": flags["storage_ok"],
        "checks": checks,
    }, suite_passed


def _mode_artifacts(root: Path, suite: str, mode: BenchmarkMode) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted((root / suite / "runs").glob(f"trace_recall_{suite}_*.json"), reverse=True)
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        payload = _read_artifact(path)
        if payload is None:
            continue
        if payload.get("mode") == mode:
            candidates.append((path, payload))
    return candidates


def _read_artifact(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _requirement_checks(summary: dict[str, Any], requirements: dict[str, float]) -> dict[str, Any]:
    return {
        key: {
            "actual": summary.get(key),
            "minimum": minimum,
            "passed": summary.get(key) is not None and float(summary[key]) >= minimum,
        }
        for key, minimum in requirements.items()
    }


def _artifact_quality_flags(payload: dict[str, Any], mode: BenchmarkMode) -> dict[str, Any]:
    return {
        "production_artifact": payload.get("architecture") == "trace-and-recall-production",
        "artifact_schema_current": payload.get("artifact_schema_version") == BENCHMARK_ARTIFACT_SCHEMA_VERSION,
        "provenance_present": isinstance(payload.get("providers"), dict)
        and isinstance(payload.get("prompts"), list),
        "workflow_provenance_valid": _workflow_provenance_valid(payload, mode),
        "engine_version": payload.get("engine_version"),
        "storage_ok": bool((payload.get("store_validation") or {}).get("ok")),
    }


def _suite_gate_passed(flags: dict[str, Any], checks: dict[str, Any]) -> bool:
    required = (
        flags["production_artifact"],
        flags["artifact_schema_current"],
        flags["provenance_present"],
        flags["workflow_provenance_valid"],
        flags["engine_version"],
        flags["storage_ok"],
        all(item["passed"] for item in checks.values()),
    )
    return all(required)


def _workflow_provenance_valid(payload: dict[str, Any], mode: BenchmarkMode) -> bool:
    providers = payload.get("providers") or {}
    prompt_names = [item.get("name") for item in payload.get("prompts") or [] if isinstance(item, dict)]
    completion_provider = providers.get("completion", {}).get("provider")
    if mode == "end-to-end":
        return (
            completion_provider not in {None, "none"}
            and prompt_names == ["facts", "group_summary", "answer", "benchmark_judge"]
        )
    return completion_provider == "none" and prompt_names == []
