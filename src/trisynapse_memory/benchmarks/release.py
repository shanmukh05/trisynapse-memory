"""Reproducible benchmark requirements for a v0 release tag."""

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
        paths = sorted((root / suite / "runs").glob(f"trace_recall_{suite}_*.json"), reverse=True)
        candidates: list[tuple[Path, dict[str, Any]]] = []
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("mode") == mode:
                candidates.append((path, payload))
        if not candidates:
            suites[suite] = {"passed": False, "error": f"no production {mode} artifact"}
            passed = False
            continue
        artifact, payload = candidates[0]
        summary = payload.get("summary") or {}
        production_artifact = payload.get("architecture") == "trace-and-recall-production"
        schema_current = payload.get("artifact_schema_version") == BENCHMARK_ARTIFACT_SCHEMA_VERSION
        provenance_present = isinstance(payload.get("providers"), dict) and isinstance(payload.get("prompts"), list)
        providers = payload.get("providers") or {}
        prompt_names = [item.get("name") for item in payload.get("prompts") or [] if isinstance(item, dict)]
        workflow_provenance_valid = (
            providers.get("completion", {}).get("provider") not in {None, "none"}
            and prompt_names == ["extraction", "episode_recall", "answer", "benchmark_judge"]
            if mode == "end-to-end"
            else providers.get("completion", {}).get("provider") == "none" and prompt_names == []
        )
        engine_version = payload.get("engine_version")
        checks = {
            key: {
                "actual": summary.get(key),
                "minimum": minimum,
                "passed": summary.get(key) is not None and float(summary[key]) >= minimum,
            }
            for key, minimum in requirements.items()
        }
        storage_ok = bool((payload.get("store_validation") or {}).get("ok"))
        suite_passed = bool(
            production_artifact and schema_current and provenance_present and workflow_provenance_valid and engine_version
            and storage_ok and all(item["passed"] for item in checks.values())
        )
        suites[suite] = {
            "passed": suite_passed,
            "artifact": str(artifact),
            "production_artifact": production_artifact,
            "artifact_schema_current": schema_current,
            "provenance_present": provenance_present,
            "workflow_provenance_valid": workflow_provenance_valid,
            "mode": mode,
            "engine_version": engine_version,
            "storage_ok": storage_ok,
            "checks": checks,
        }
        passed = passed and suite_passed
    return {"passed": passed, "mode": mode, "requirements": selected_requirements, "suites": suites}
