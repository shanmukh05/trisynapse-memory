from __future__ import annotations

import hashlib

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from trisynapse_memory.adapters.agent_events import capture_agent_event
from trisynapse_memory.api import create_app
from trisynapse_memory.cli import app
from trisynapse_memory.engine import MemoryEngine
from trisynapse_memory.engine.formation.sources import load_document
from trisynapse_memory.engine.models import MemoryNamespace
from trisynapse_memory.engine.providers.registry import ProviderSettings, completion_from_settings, embedder_from_settings


class DeterministicEmbedder:
    model_name = "v0-product-test"

    def encode(self, texts: list[str]) -> list[list[float]]:
        values = []
        for text in texts:
            vector = [0.0] * 32
            for token in text.lower().split():
                vector[int(hashlib.sha256(token.encode()).hexdigest(), 16) % len(vector)] += 1
            norm = sum(item * item for item in vector) ** 0.5 or 1
            values.append([item / norm for item in vector])
        return values


def engine_at(path, *, auto_process=True):
    return MemoryEngine.open(path, embedder=DeterministicEmbedder(), auto_process=auto_process)


def test_legacy_config_constructor_and_cli_flag_are_removed() -> None:
    assert not hasattr(MemoryEngine, "from_config")
    result = CliRunner().invoke(app, ["bench", "run", "locomo", "--config", "old.yaml"])
    assert result.exit_code != 0


def test_openrouter_uses_its_own_key_and_endpoint_for_both_provider_roles(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    settings = ProviderSettings(provider="openrouter", model="vendor/model")

    completion = completion_from_settings(settings)
    embedder = embedder_from_settings(settings)

    assert completion.api_key == "openrouter-key"
    assert completion.base_url == "https://openrouter.ai/api/v1"
    assert embedder.api_key == "openrouter-key"
    assert embedder.base_url == "https://openrouter.ai/api/v1"


def test_namespaces_isolate_reads_and_search(tmp_path) -> None:
    engine = engine_at(tmp_path)
    alice = MemoryNamespace(user_id="alice", project_id="support")
    bob = MemoryNamespace(user_id="bob", project_id="support")
    alice_delta = engine.add("Alice prefers email updates.", namespace=alice, episode_id="chat:alice")
    engine.add("Bob prefers phone calls.", namespace=bob, episode_id="chat:bob")

    assert [item.id for item in engine.list(namespace=alice).items] == [alice_delta.id]
    assert all("Bob" not in hit.text for hit in engine.search("preferred contact", namespace=alice).hits)
    try:
        engine.get(alice_delta.id, namespace=bob)
    except KeyError:
        pass
    else:
        raise AssertionError("cross-namespace get must be hidden")


def test_lifecycle_and_remove_remain_consistent_without_content_filtering(tmp_path) -> None:
    engine = engine_at(tmp_path)
    namespace = MemoryNamespace(project_id="private")
    original = engine.add(
        "Use api_key=super-secret-value and <private>never store this</private>.",
        namespace=namespace,
        episode_id="manual:secret",
        source_ref={"authorization": "Bearer abcdefghijklmnopqrstuvwxyz"},
        scope={"metadata": {"client_secret": "also-super-secret"}},
    )
    assert "super-secret-value" in original.text
    assert "never store this" in original.text
    assert original.privacy_scope == {}
    serialized = original.model_dump_json()
    assert "abcdefghijklmnopqrstuvwxyz" in serialized
    assert "also-super-secret" in serialized

    correction = engine.correct(delta_id=original.id, text="Use the credential vault instead.", namespace=namespace)
    history = engine.history(original.id, namespace=namespace)
    assert correction.id in {item.id for item in history.events}
    engine.search("credential vault", namespace=namespace)

    result = engine.remove(delta_ids=[original.id], reason="credential incident", namespace=namespace)
    removed = engine.get(original.id, namespace=namespace, include_retracted=True)
    assert removed.text == "[REMOVED]"
    assert result.removed_delta_ids == [original.id]
    assert engine.store.episode_recall_views(namespace=namespace) == []
    assert engine.validate_store().ok is True


def test_jobs_survive_reopen_and_process(tmp_path) -> None:
    namespace = MemoryNamespace(project_id="jobs")
    engine = engine_at(tmp_path, auto_process=False)
    engine.add("The project uses SQLite.", episode_id="project:one", namespace=namespace)
    assert engine.list_jobs(status="pending")
    claimed = engine.store.claim_job()
    assert claimed is not None and claimed.status == "running"
    engine.store._connection.execute(
        "UPDATE jobs SET updated_at='2000-01-01T00:00:00+00:00' WHERE id=?", (claimed.id,)
    )
    engine.store._connection.commit()
    engine.close()

    reopened = engine_at(tmp_path, auto_process=False)
    completed = reopened.run_jobs()
    assert completed and completed[-1].status == "completed"
    assert reopened.list_jobs(status="pending") == []


def test_message_batch_schedules_one_compile_job(tmp_path) -> None:
    engine = engine_at(tmp_path, auto_process=False)
    engine.ingest_messages(
        [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}],
        episode_id="chat:batch",
    )
    jobs = engine.list_jobs(status="pending")
    assert len(jobs) == 1
    assert jobs[0].kind == "compile_episode"


def test_html_and_code_file_loaders(tmp_path) -> None:
    html = tmp_path / "guide.html"
    html.write_text("<h1>Guide</h1><script>secretNoise()</script><p>Use Trace first.</p>", encoding="utf-8")
    source = tmp_path / "sample.py"
    source.write_text("def answer():\n    return 42\n", encoding="utf-8")

    loaded_html = load_document(html)
    loaded_code = load_document(source)
    assert "Use Trace first" in loaded_html.text
    assert "secretNoise" not in loaded_html.text
    assert "000001: def answer" in loaded_code.text
    assert loaded_code.metadata["language"] == "py"


def test_pdf_loader_and_backup_restore(tmp_path) -> None:
    from pypdf import PdfWriter

    source = tmp_path / "source"
    engine = engine_at(source)
    namespace = MemoryNamespace(project_id="archive")
    engine.add("The restore check keeps this evidence.", namespace=namespace)

    pdf = tmp_path / "report.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_metadata({"/Title": "Archive report"})
    with pdf.open("wb") as handle:
        writer.write(handle)
    loaded = load_document(pdf)
    assert loaded.title == "Archive report"
    assert loaded.metadata["page_count"] == 1
    assert "[Page 1]" in loaded.text

    archive = engine.backup(tmp_path / "memory-backup.zip")
    engine.close()
    restored = MemoryEngine.restore_backup(archive, tmp_path / "restored")
    assert restored.validate_store().ok
    assert restored.list(namespace=namespace).items[0].text == "The restore check keeps this evidence."


def test_agent_events_capture_and_return_context(tmp_path) -> None:
    engine = engine_at(tmp_path)
    event = {
        "type": "user_prompt",
        "session_id": "s1",
        "agent_id": "codex",
        "project_id": "demo",
        "user_id": "alice",
        "content": "Keep release notes concise.",
    }
    captured = capture_agent_event(engine, event)
    started = capture_agent_event(engine, {**event, "type": "session_start"})
    assert captured["captured"] is True
    assert started["captured"] is False
    assert "context" in started


def test_rest_lifecycle_scoped_key_and_studio(tmp_path) -> None:
    engine = engine_at(tmp_path)
    client = TestClient(create_app(
        tmp_path,
        engine=engine,
        api_keys={"alice-token": {"user_id": "alice", "project_id": "support"}},
        studio=True,
    ))
    auth = {"Authorization": "Bearer alice-token"}
    add = client.post(
        "/api/v1/memory/observations",
        headers=auth,
        json={"text": "Alice prefers email.", "namespace": {"user_id": "alice", "project_id": "support"}},
    )
    assert add.status_code == 200
    memory_id = add.json()["delta_id"]
    listed = client.get("/api/v1/memories?project_id=support", headers=auth)
    history = client.get(f"/api/v1/memories/{memory_id}/history?project_id=support", headers=auth)
    forbidden = client.post(
        "/api/v1/search",
        headers=auth,
        json={"query": "email", "namespace": {"user_id": "bob", "project_id": "support"}},
    )
    studio = client.get("/")
    admin_forbidden = client.get("/api/v1/metrics", headers=auth)

    bob = MemoryNamespace(user_id="bob", project_id="support")
    bob_memory = engine.add("Bob prefers phone calls.", namespace=bob)
    legacy_forbidden = client.get(f"/api/v1/deltas/{bob_memory.id}?project_id=support", headers=auth)
    bob_query = engine.search("phone calls", namespace=bob)
    feedback_forbidden = client.post(
        "/api/v1/feedback",
        headers=auth,
        json={"query_id": bob_query.query_id, "helpful": True},
    )

    assert listed.status_code == 200 and len(listed.json()["items"]) == 1
    assert history.status_code == 200
    assert forbidden.status_code == 403
    assert admin_forbidden.status_code == 403
    assert legacy_forbidden.status_code == 404
    assert feedback_forbidden.status_code == 404
    assert studio.status_code == 200 and "Memory Studio" in studio.text
