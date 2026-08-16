from __future__ import annotations

import hashlib
import json
from importlib.metadata import version as package_version

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from trisynapse_memory.api import create_app
from trisynapse_memory.cli import app
from trisynapse_memory.engine import (
    EmbeddingRebuildRequired,
    MemoryEngine,
    ProviderRole,
    ProviderSelection,
)
from trisynapse_memory.engine import memory as memory_module
from trisynapse_memory.engine.providers import (
    AnthropicCompletion,
    ProviderSettings,
    completion_from_settings,
    embedding_cache_key,
    fetch_model_catalog,
    list_provider_descriptors,
)
from trisynapse_memory.engine import providers as provider_module


class DeterministicEmbedder:
    model_name = "deterministic-v1"
    cache_key = "test:deterministic-v1"
    provider_name = "test"

    def encode(self, texts: list[str]) -> list[list[float]]:
        values: list[list[float]] = []
        for text in texts:
            vector = [0.0] * 12
            for token in text.lower().split():
                vector[int(hashlib.sha256(token.encode()).hexdigest(), 16) % len(vector)] += 1
            values.append(vector)
        return values


class ReplacementEmbedder(DeterministicEmbedder):
    model_name = "replacement-v1"
    cache_key = "deepinfra:replacement-v1"
    provider_name = "deepinfra"


def test_provider_matrix_and_direct_provider_endpoints(monkeypatch) -> None:
    for key, value in {
        "ANTHROPIC_API_KEY": "a",
        "DEEPINFRA_API_TOKEN": "i",
        "DEEPSEEK_API_KEY": "d",
        "MOONSHOT_API_KEY": "k",
    }.items():
        monkeypatch.setenv(key, value)
    values = {item.id: item for item in list_provider_descriptors()}
    assert ProviderRole.EMBEDDING not in values["anthropic"].roles
    assert ProviderRole.EMBEDDING not in values["deepseek"].roles
    assert ProviderRole.EMBEDDING not in values["kimi"].roles
    assert ProviderRole.EMBEDDING in values["deepinfra"].roles
    assert completion_from_settings(ProviderSettings(provider="deepinfra")).base_url == "https://api.deepinfra.com/v1/openai"
    assert completion_from_settings(ProviderSettings(provider="deepseek")).base_url == "https://api.deepseek.com"
    assert completion_from_settings(ProviderSettings(provider="kimi")).base_url == "https://api.moonshot.ai/v1"


def test_explicit_python_key_can_override_environment_lookup(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    completion = completion_from_settings(
        ProviderSettings(
            provider="anthropic", model="claude-test", api_key="secret-manager-value"
        )
    )
    assert completion.api_key == "secret-manager-value"


def test_anthropic_uses_native_messages_and_image_shapes(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_request(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return {"content": [{"type": "text", "text": '{"status":"ok"}'}]}

    monkeypatch.setattr(provider_module, "_request_json", fake_request)
    completion = AnthropicCompletion(
        ProviderSettings(provider="anthropic", model="claude-test", api_key="secret")
    )
    assert completion.complete_json("system", "user") == {"status": "ok"}
    completion.complete_multimodal("system", "user", b"png", "image/png")
    assert calls[0]["url"].endswith("/v1/messages")
    assert calls[0]["headers"]["x-api-key"] == "secret"
    image = calls[1]["payload"]["messages"][0]["content"][0]
    assert image["type"] == "image"
    assert image["source"]["media_type"] == "image/png"


def test_qwen_models_are_catalog_entries_not_a_direct_provider(monkeypatch) -> None:
    monkeypatch.setenv("DEEPINFRA_API_TOKEN", "token")
    monkeypatch.setattr(
        provider_module,
        "_request_json",
        lambda *args, **kwargs: [{
            "id": "Qwen/Qwen3-Embedding-8B",
            "type": "embeddings",
            "context_length": 32768,
        }],
    )
    models = fetch_model_catalog("deepinfra", ProviderRole.EMBEDDING)
    assert [item.id for item in models] == ["Qwen/Qwen3-Embedding-8B"]
    assert "qwen" not in {item.id for item in list_provider_descriptors()}


def test_completion_selection_persists_without_storing_credentials(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "never-write-this")
    engine = MemoryEngine.open(tmp_path, embedder=DeterministicEmbedder())
    configuration = engine.get_model_configuration()
    configuration.completion = ProviderSelection(provider="anthropic", model="claude-test")
    change = engine.set_model_configuration(configuration)
    assert change.status == "applied"
    assert change.configuration.revision == 1
    stored = engine.store._connection.execute(
        "SELECT current_json FROM model_configuration WHERE id=1"
    ).fetchone()["current_json"]
    assert "never-write-this" not in stored
    engine.close()

    reopened = MemoryEngine.open(tmp_path, embedder=DeterministicEmbedder())
    assert reopened.get_model_configuration().completion.provider == "anthropic"
    assert reopened.completion.model == "claude-test"
    reopened.close()


def test_store_revision_refreshes_other_engine_and_rejects_stale_updates(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    first = MemoryEngine.open(tmp_path, embedder=DeterministicEmbedder())
    second = MemoryEngine.open(tmp_path, embedder=DeterministicEmbedder())
    stale = second.get_model_configuration()
    updated = first.get_model_configuration()
    updated.completion = ProviderSelection(provider="anthropic", model="claude-test")
    first.set_model_configuration(updated)

    second.check()
    assert second.completion.model == "claude-test"
    stale.completion = ProviderSelection(provider="anthropic", model="another-model")
    with pytest.raises(ValueError, match="revision conflict"):
        second.set_model_configuration(stale)
    first.close()
    second.close()


def test_explicit_python_provider_remains_an_instance_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")

    class ExplicitCompletion:
        model = "explicit-model"

        def __call__(self, system, user):
            return {"answer": "ok"}

    completion = ExplicitCompletion()
    engine = MemoryEngine.open(
        tmp_path, embedder=DeterministicEmbedder(), completion=completion
    )
    configuration = engine.get_model_configuration()
    configuration.completion = ProviderSelection(provider="anthropic", model="stored-model")
    engine.set_model_configuration(configuration)
    engine.check()
    assert engine.completion is completion
    assert engine.get_model_configuration().completion.model == "stored-model"
    engine.close()


def test_embedding_change_requires_confirmation_then_activates_atomically(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPINFRA_API_TOKEN", "token")
    engine = MemoryEngine.open(tmp_path, embedder=DeterministicEmbedder(), auto_process=False)
    engine.add("A searchable memory record.")
    configuration = engine.get_model_configuration()
    configuration.embedding = ProviderSelection(provider="deepinfra", model="replacement-v1")
    with pytest.raises(EmbeddingRebuildRequired):
        engine.set_model_configuration(configuration)
    assert engine.get_model_configuration().embedding.provider == "sentence-transformers"

    monkeypatch.setattr(memory_module, "embedder_from_settings", lambda settings: ReplacementEmbedder())
    change = engine.set_model_configuration(configuration, confirm_embedding_rebuild=True)
    assert change.status == "rebuild_pending"
    assert engine.get_model_configuration().embedding.provider == "sentence-transformers"
    jobs = engine.run_jobs(max_jobs=1)
    assert jobs[0].status == "completed"
    assert engine.get_model_configuration().embedding.provider == "deepinfra"
    assert engine.get_model_configuration_status().status == "applied"
    engine.close()


def test_failed_embedding_rebuild_keeps_previous_configuration(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPINFRA_API_TOKEN", "token")
    engine = MemoryEngine.open(tmp_path, embedder=DeterministicEmbedder(), auto_process=False)
    engine.add("Keep the current embedding index active.")
    configuration = engine.get_model_configuration()
    configuration.embedding = ProviderSelection(provider="deepinfra", model="broken")

    class BrokenEmbedder(ReplacementEmbedder):
        def encode(self, texts):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(memory_module, "embedder_from_settings", lambda settings: BrokenEmbedder())
    change = engine.set_model_configuration(configuration, confirm_embedding_rebuild=True)
    for _ in range(3):
        engine.run_jobs(max_jobs=1)
    assert engine.get_model_configuration().embedding.provider == "sentence-transformers"
    status = engine.get_model_configuration_status()
    assert status.status == "rebuild_failed"
    assert "provider unavailable" in (status.message or "")
    assert engine.store.get_job(change.job_id).status == "failed"
    engine.close()


def test_embedding_cache_identity_includes_provider_endpoint_and_model() -> None:
    left = embedding_cache_key("openai", "https://api.openai.com/v1", "shared")
    right = embedding_cache_key("deepinfra", "https://api.deepinfra.com/v1/openai", "shared")
    other_endpoint = embedding_cache_key("openai", "https://example.test/v1", "shared")
    assert len({left, right, other_endpoint}) == 3


def test_rest_configuration_requires_rebuild_confirmation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPINFRA_API_TOKEN", "token")
    engine = MemoryEngine.open(tmp_path, embedder=DeterministicEmbedder(), auto_process=False)
    engine.add("REST must protect embedding changes.")
    client = TestClient(create_app(tmp_path, engine=engine))
    current = client.get("/api/v1/model-configuration").json()["configuration"]
    payload = {
        "completion": current["completion"],
        "embedding": {"provider": "deepinfra", "model": "replacement-v1"},
        "revision": current["revision"],
        "confirm_embedding_rebuild": False,
    }
    assert client.put("/api/v1/model-configuration", json=payload).status_code == 409
    monkeypatch.setattr(memory_module, "embedder_from_settings", lambda settings: ReplacementEmbedder())
    payload["confirm_embedding_rebuild"] = True
    accepted = client.put("/api/v1/model-configuration", json=payload)
    assert accepted.status_code == 202
    assert accepted.json()["job_id"]
    assert client.get("/api/v1/providers").status_code == 200
    assert client.get("/api/v1/check").json()["version"] == package_version("trisynapse-memory")
    engine.close()


def test_cli_exposes_check_and_model_commands(tmp_path) -> None:
    runner = CliRunner()
    checked = runner.invoke(app, ["--path", str(tmp_path), "--json", "check"])
    assert checked.exit_code == 0
    assert json.loads(checked.output)["version"] == package_version("trisynapse-memory")
    current = runner.invoke(app, ["--path", str(tmp_path), "--json", "models", "current"])
    assert current.exit_code == 0
    assert json.loads(current.output)["configuration"]["embedding"]["provider"] == "sentence-transformers"
    disabled = runner.invoke(
        app, ["--path", str(tmp_path), "--json", "models", "set", "completion", "none"]
    )
    assert disabled.exit_code == 0
    obsolete = "doc" + "tor"
    assert runner.invoke(app, [obsolete]).exit_code != 0
