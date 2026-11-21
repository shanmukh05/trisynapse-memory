"""YAML benchmark configs for ingest-once, query-many runs.

Owns path resolution, profile merge, and model/judge parsing for
``bench.yaml``. The shared runner executes the resolved config; this module
does not ingest. A named profile overlays only non-null fields so a CLI
override can still win.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trisynapse_memory.adapters.benchmarks import adapter_names
from trisynapse_memory.benchmarks.runner import (
    BenchmarkProgressCallback,
    run_benchmark,
)
from trisynapse_memory.engine.models import ModelConfiguration, ProviderSelection

_PACKAGED_CONFIGS = Path(__file__).resolve().parent / "configs"


class BenchmarkYamlModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BenchmarkProviderConfig(BenchmarkYamlModel):
    provider: str
    model: str | None = None
    base_url: str | None = None

    def to_selection(self) -> ProviderSelection:
        return ProviderSelection(provider=self.provider, model=self.model, base_url=self.base_url)


class BenchmarkModelsConfig(BenchmarkYamlModel):
    completion: BenchmarkProviderConfig | None = None
    embedding: BenchmarkProviderConfig | None = None

    def to_model_configuration(self) -> ModelConfiguration | None:
        if self.completion is None and self.embedding is None:
            return None
        kwargs: dict[str, Any] = {}
        if self.completion is not None:
            kwargs["completion"] = self.completion.to_selection()
        if self.embedding is not None:
            kwargs["embedding"] = self.embedding.to_selection()
        return ModelConfiguration(**kwargs)


class BenchmarkProfileConfig(BenchmarkYamlModel):
    stage: Literal["auto", "ingest", "query"] | None = None
    mode: Literal["retrieval", "end-to-end"] | None = None
    sampling: Literal["auto", "stratified", "sequential"] | None = None
    max_questions: int | None = Field(default=None, ge=1)
    rebuild_store: bool | None = None
    judge: BenchmarkProviderConfig | None = None


class BenchmarkFileConfig(BenchmarkProfileConfig):
    suite: str | None = None
    data_root: str | None = None
    dataset: str | None = None
    output_root: str | None = None
    models: BenchmarkModelsConfig | None = None
    profiles: dict[str, BenchmarkProfileConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _known_suite(self) -> "BenchmarkFileConfig":
        if self.suite is not None and self.suite not in adapter_names():
            supported = ", ".join(adapter_names())
            raise ValueError(f"unsupported benchmark suite '{self.suite}'; choose one of: {supported}")
        return self


class ResolvedBenchmarkConfig(BenchmarkYamlModel):
    suite: str
    data_root: Path
    dataset: str | None = None
    dataset_path: Path | None = None
    output_root: Path | None = None
    stage: Literal["auto", "ingest", "query"] = "auto"
    mode: Literal["retrieval", "end-to-end"] = "retrieval"
    sampling: Literal["auto", "stratified", "sequential"] = "auto"
    max_questions: int | None = None
    rebuild_store: bool = False
    models: ModelConfiguration | None = None
    judge: ProviderSelection | None = None
    profile: str | None = None
    config_path: Path


def resolve_config_path(value: str | Path) -> Path:
    """Resolve a YAML file, a dataset directory, a packaged suite name, or ``data/<suite>/bench.yaml``."""

    raw = Path(value).expanduser()
    if raw.is_file():
        return raw.resolve()
    if raw.is_dir():
        local = raw / "bench.yaml"
        if local.is_file():
            return local.resolve()
    name = str(value).strip()
    packaged = _PACKAGED_CONFIGS / f"{name}.yaml"
    if packaged.is_file():
        return packaged
    local = Path("data") / name / "bench.yaml"
    if local.is_file():
        return local.resolve()
    raise FileNotFoundError(
        f"benchmark config not found: {value}. Pass a YAML path, a dataset directory, "
        f"a suite name ({', '.join(adapter_names())}), or data/<suite>/bench.yaml"
    )


def load_benchmark_config(
    path: str | Path,
    *,
    profile: str | None = None,
    **overrides: Any,
) -> ResolvedBenchmarkConfig:
    config_path = resolve_config_path(path)
    parsed = BenchmarkFileConfig.model_validate(_read_yaml(config_path))
    merged = _merged_benchmark_mapping(parsed, profile, overrides)
    suite = _required_suite(merged)
    data_root, dataset, dataset_path = _resolved_dataset_paths(merged, config_path, suite)
    output = merged.get("output_root")
    return ResolvedBenchmarkConfig(
        suite=suite,
        data_root=data_root,
        dataset=dataset,
        dataset_path=dataset_path,
        output_root=_resolve_dir(output, config_path, default=None) if output else None,
        stage=merged.get("stage") or "auto",
        mode=merged.get("mode") or "retrieval",
        sampling=merged.get("sampling") or "auto",
        max_questions=_resolved_max_questions(merged),
        rebuild_store=bool(merged.get("rebuild_store")),
        models=_parsed_models(merged),
        judge=_parsed_judge(merged),
        profile=profile,
        config_path=config_path,
    )


def _merged_benchmark_mapping(
    parsed: BenchmarkFileConfig,
    profile: str | None,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    merged = parsed.model_dump()
    profiles = merged.pop("profiles") or {}
    if profile:
        if profile not in profiles:
            names = ", ".join(sorted(profiles)) or "(none)"
            raise KeyError(f"unknown benchmark profile '{profile}'; choose one of: {names}")
        _overlay_non_null(merged, profiles[profile])
    _overlay_non_null(merged, overrides)
    return merged


def _overlay_non_null(merged: dict[str, Any], values: dict[str, Any]) -> None:
    for key, value in values.items():
        if value is not None:
            merged[key] = value


def _required_suite(merged: dict[str, Any]) -> str:
    suite = merged.get("suite")
    if not suite:
        raise ValueError("benchmark config must set suite")
    if suite not in adapter_names():
        supported = ", ".join(adapter_names())
        raise ValueError(f"unsupported benchmark suite '{suite}'; choose one of: {supported}")
    return suite


def _resolved_dataset_paths(
    merged: dict[str, Any],
    config_path: Path,
    suite: str,
) -> tuple[Path, str | None, Path | None]:
    data_root = _resolve_dir(merged.get("data_root"), config_path, default=Path("data") / suite)
    dataset = merged.get("dataset")
    if not dataset:
        return data_root, None, None
    resolved_file = _resolve_dataset_file(dataset, data_root, config_path)
    if resolved_file.parent == data_root.resolve():
        data_root = resolved_file
    return data_root, dataset, resolved_file


def _resolved_max_questions(merged: dict[str, Any]) -> int | None:
    stage = merged.get("stage") or "auto"
    max_questions = merged.get("max_questions")
    if stage == "ingest" or max_questions == 0:
        return None
    return max_questions


def _parsed_models(merged: dict[str, Any]) -> ModelConfiguration | None:
    if not merged.get("models"):
        return None
    return BenchmarkModelsConfig.model_validate(merged["models"]).to_model_configuration()


def _parsed_judge(merged: dict[str, Any]) -> ProviderSelection | None:
    if not merged.get("judge"):
        return None
    return BenchmarkProviderConfig.model_validate(merged["judge"]).to_selection()


def run_benchmark_from_config(
    path: str | Path,
    *,
    profile: str | None = None,
    on_progress: BenchmarkProgressCallback | None = None,
    fallback_model_configuration: ModelConfiguration | dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Load a dataset YAML config and run the shared benchmark runner."""

    resolved = load_benchmark_config(path, profile=profile, **overrides)
    return run_resolved_benchmark(
        resolved,
        on_progress=on_progress,
        fallback_model_configuration=fallback_model_configuration,
    )


def run_resolved_benchmark(
    resolved: ResolvedBenchmarkConfig,
    *,
    on_progress: BenchmarkProgressCallback | None = None,
    fallback_model_configuration: ModelConfiguration | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a previously resolved config. Its model settings take precedence."""

    return run_benchmark(
        resolved.suite,
        resolved.data_root,
        dataset=resolved.dataset_path,
        max_questions=resolved.max_questions,
        output_root=resolved.output_root,
        mode=resolved.mode,
        model_configuration=resolved.models or fallback_model_configuration,
        on_progress=on_progress,
        sampling=resolved.sampling,
        stage=resolved.stage,
        rebuild_store=resolved.rebuild_store,
        judge_selection=resolved.judge,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m trisynapse_memory.benchmarks",
        description="Run a LoCoMo-style ingest/query benchmark from a YAML config.",
    )
    parser.add_argument(
        "config",
        help=f"YAML path, dataset directory, or suite name ({', '.join(adapter_names())})",
    )
    parser.add_argument("--profile", help="Named profile from the YAML file (ingest, retrieval, end-to-end)")
    parser.add_argument("--stage", choices=["auto", "ingest", "query"])
    parser.add_argument("--mode", choices=["retrieval", "end-to-end"])
    parser.add_argument("--sampling", choices=["auto", "stratified", "sequential"])
    parser.add_argument(
        "--max-questions",
        type=int,
        help="Cap evaluated questions. 0 means the full dataset (overrides a YAML cap).",
    )
    parser.add_argument("--output-root", help="Directory for benchmark artifacts")
    parser.add_argument("--rebuild-store", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    overrides: dict[str, Any] = {
        "stage": args.stage,
        "mode": args.mode,
        "sampling": args.sampling,
        "max_questions": args.max_questions,
        "output_root": args.output_root,
        "rebuild_store": args.rebuild_store,
    }

    def report(event: Any) -> None:
        print(f"{event.stage}: {event.description}", file=sys.stderr)

    summary = run_benchmark_from_config(
        args.config,
        profile=args.profile,
        on_progress=report,
        **overrides,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "loading benchmark YAML requires PyYAML. Install with: pip install 'trisynapse-memory[sources]'"
        ) from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark config must be a YAML mapping: {path}")
    return payload


def _resolve_dataset_file(dataset: str, data_root: Path, config_path: Path) -> Path:
    candidate = Path(dataset).expanduser()
    if candidate.is_absolute():
        return candidate
    if data_root.is_file():
        return (data_root.parent / candidate).resolve()
    from_root = (data_root / candidate).resolve()
    from_file = (config_path.parent / candidate).resolve()
    if from_root.exists() or not from_file.exists():
        return from_root
    return from_file


def _resolve_dir(value: str | Path | None, config_path: Path, *, default: Path | None) -> Path:
    if value is None or value == "":
        if default is None:
            raise ValueError("directory path is required")
        return default if default.is_absolute() else (Path.cwd() / default).resolve()
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    from_file = (config_path.parent / candidate).resolve()
    if str(value) in {".", "./"} or from_file.exists():
        return from_file
    return (Path.cwd() / candidate).resolve()
