from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from trisynapse_memory.benchmarks.config import (
    load_benchmark_config,
    main,
    resolve_config_path,
    run_benchmark_from_config,
)
from trisynapse_memory.cli import app
from trisynapse_memory.engine.models import ModelConfiguration


class _FakeEngine:
    def get_model_configuration(self) -> ModelConfiguration:
        return ModelConfiguration()

    def close(self) -> None:
        return None


def _write_config(path: Path) -> Path:
    path.write_text(
        """
suite: locomo
data_root: .
sampling: stratified
max_questions: 50
models:
  completion:
    provider: gemini
    model: gemini-3.5-flash
  embedding:
    provider: sentence-transformers
    model: all-MiniLM-L6-v2
profiles:
  ingest:
    stage: ingest
    mode: end-to-end
  retrieval:
    stage: query
    mode: retrieval
    max_questions: 25
  end-to-end:
    stage: query
    mode: end-to-end
    max_questions: 25
    judge:
      provider: gemini
      model: gemini-3.8-flash
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_load_benchmark_config_applies_named_profiles(tmp_path) -> None:
    path = _write_config(tmp_path / "bench.yaml")
    ingest = load_benchmark_config(path, profile="ingest")
    retrieval = load_benchmark_config(path, profile="retrieval")

    assert ingest.suite == "locomo"
    assert ingest.data_root == tmp_path
    assert ingest.stage == "ingest"
    assert ingest.mode == "end-to-end"
    assert ingest.sampling == "stratified"
    assert ingest.models is not None
    assert ingest.models.completion.provider == "gemini"
    assert ingest.max_questions is None
    assert ingest.judge is None
    assert retrieval.stage == "query"
    assert retrieval.mode == "retrieval"
    assert retrieval.max_questions == 25
    end_to_end = load_benchmark_config(path, profile="end-to-end")
    assert end_to_end.max_questions == 25
    assert end_to_end.judge is not None
    assert end_to_end.judge.provider == "gemini"
    assert end_to_end.judge.model == "gemini-3.8-flash"


def test_load_benchmark_config_rejects_unknown_keys(tmp_path) -> None:
    path = tmp_path / "bench.yaml"
    path.write_text("suite: locomo\nconfig_path: old.yaml\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_benchmark_config(path)


def test_resolve_config_path_accepts_packaged_suite_name() -> None:
    path = resolve_config_path("locomo")
    assert path.name == "locomo.yaml"
    loaded = load_benchmark_config(path, profile="ingest")
    assert loaded.suite == "locomo"
    assert loaded.stage == "ingest"


def test_longmemeval_config_pins_s_split(tmp_path) -> None:
    packaged = load_benchmark_config("longmemeval", profile="ingest")
    assert packaged.dataset == "longmemeval_s_cleaned.json"
    assert packaged.data_root.name == "longmemeval_s_cleaned.json"

    path = tmp_path / "bench.yaml"
    path.write_text(
        "suite: longmemeval\ndata_root: .\ndataset: longmemeval_s_cleaned.json\n",
        encoding="utf-8",
    )
    (tmp_path / "longmemeval_s_cleaned.json").write_text("[]", encoding="utf-8")
    loaded = load_benchmark_config(path, profile=None)
    assert loaded.data_root == (tmp_path / "longmemeval_s_cleaned.json").resolve()


def test_personamem_config_keeps_suite_directory_for_nested_csv() -> None:
    loaded = load_benchmark_config("personamem", profile="ingest")
    assert loaded.suite == "personamem"
    assert loaded.dataset == "benchmark/text/benchmark.csv"
    assert loaded.data_root.name == "personamem"
    assert loaded.stage == "ingest"


def test_resolve_config_path_accepts_dataset_directory(tmp_path) -> None:
    dataset = tmp_path / "locomo"
    dataset.mkdir()
    _write_config(dataset / "bench.yaml")
    assert resolve_config_path(dataset) == (dataset / "bench.yaml").resolve()


def test_run_benchmark_from_config_uses_resolved_settings(tmp_path, monkeypatch) -> None:
    path = _write_config(tmp_path / "bench.yaml")
    captured: dict[str, object] = {}

    def fake_run(suite, data_root, **kwargs):
        captured["suite"] = suite
        captured["data_root"] = Path(data_root)
        captured["kwargs"] = kwargs
        return {"questions": 0, "suite": suite}

    monkeypatch.setattr("trisynapse_memory.benchmarks.config.run_benchmark", fake_run)
    result = run_benchmark_from_config(path, profile="ingest")

    assert result["suite"] == "locomo"
    assert captured["suite"] == "locomo"
    assert captured["data_root"] == tmp_path
    assert captured["kwargs"]["stage"] == "ingest"
    assert captured["kwargs"]["mode"] == "end-to-end"
    assert captured["kwargs"]["max_questions"] is None


def test_cli_and_module_flags_override_yaml_question_cap(tmp_path, monkeypatch) -> None:
    path = _write_config(tmp_path / "bench.yaml")
    captured: dict[str, object] = {}

    def fake_run(suite, data_root, **kwargs):
        captured["kwargs"] = kwargs
        return {"questions": 0, "suite": suite, "stage": kwargs["stage"]}

    monkeypatch.setattr("trisynapse_memory.benchmarks.config.run_benchmark", fake_run)
    assert main([str(path), "--profile", "retrieval", "--max-questions", "10"]) == 0
    assert captured["kwargs"]["max_questions"] == 10
    assert main([str(path), "--profile", "retrieval", "--max-questions", "0"]) == 0
    assert captured["kwargs"]["max_questions"] is None
    assert captured["kwargs"]["judge_selection"] is None

    monkeypatch.setattr("trisynapse_memory.cli.MemoryEngine.from_env", lambda *args, **kwargs: _FakeEngine())
    result = CliRunner().invoke(
        app,
        ["--json", "bench", "run", "--config", str(path), "--profile", "end-to-end", "--max-questions", "100"],
    )
    assert result.exit_code == 0, result.output
    assert captured["kwargs"]["max_questions"] == 100
    assert captured["kwargs"]["judge_selection"].model == "gemini-3.8-flash"


def test_module_main_runs_packaged_suite_profile(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_run(suite, data_root, **kwargs):
        captured["suite"] = suite
        captured["kwargs"] = kwargs
        return {"questions": 0, "suite": suite, "stage": kwargs["stage"]}

    monkeypatch.setattr("trisynapse_memory.benchmarks.config.run_benchmark", fake_run)
    assert main(["locomo", "--profile", "ingest"]) == 0
    assert captured["suite"] == "locomo"
    assert captured["kwargs"]["stage"] == "ingest"
    assert captured["kwargs"]["mode"] == "end-to-end"
    assert '"stage": "ingest"' in capsys.readouterr().out


def test_cli_benchmark_config_flag_runs_yaml_profile(tmp_path, monkeypatch) -> None:
    path = _write_config(tmp_path / "bench.yaml")
    monkeypatch.setattr("trisynapse_memory.cli.MemoryEngine.from_env", lambda *args, **kwargs: _FakeEngine())
    monkeypatch.setattr(
        "trisynapse_memory.benchmarks.config.run_benchmark",
        lambda suite, data_root, **kwargs: {"questions": 0, "store": str(data_root), "stage": kwargs["stage"]},
    )
    result = CliRunner().invoke(
        app,
        ["--json", "bench", "run", "--config", str(path), "--profile", "ingest"],
    )
    assert result.exit_code == 0, result.output
    assert "ingest" in result.output


def test_cli_suite_profile_loads_packaged_yaml(monkeypatch) -> None:
    monkeypatch.setattr("trisynapse_memory.cli.MemoryEngine.from_env", lambda *args, **kwargs: _FakeEngine())
    monkeypatch.setattr(
        "trisynapse_memory.benchmarks.config.run_benchmark",
        lambda suite, data_root, **kwargs: {"questions": 0, "suite": suite, "stage": kwargs["stage"]},
    )
    result = CliRunner().invoke(app, ["--json", "bench", "run", "locomo", "--profile", "ingest"])
    assert result.exit_code == 0, result.output
    assert "ingest" in result.output
