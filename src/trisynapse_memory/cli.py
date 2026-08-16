"""Typer/Rich command line and Textual entry point."""

from __future__ import annotations

import json
import os
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from trisynapse_memory.engine import MemoryEngine, SourceInput
from trisynapse_memory.engine.models import MemoryNamespace, ProviderRole, ProviderSelection


CONTEXT = {"help_option_names": ["-h", "--help"]}
app = typer.Typer(name="trisynapse-memory", help="Store traces. Recall meaning.", invoke_without_command=True, no_args_is_help=False, context_settings=CONTEXT)
add_app = typer.Typer(help="Append normalized observations or compatibility documents.")
jobs_app = typer.Typer(help="Inspect or run persistent memory jobs.")
sources_app = typer.Typer(help="List, inspect, and remove retained sources.")
runs_app = typer.Typer(help="Inspect and retry durable ingestion runs.")
remove_app = typer.Typer(help="Physically redact memory or a complete source.")
episodes_app = typer.Typer(help="Inspect and compile episodes.")
graph_app = typer.Typer(help="Export the derived knowledge graph.")
snapshot_app = typer.Typer(help="Manage recall-window snapshots.")
bench_app = typer.Typer(help="Run benchmark adapters and release gates.")
models_app = typer.Typer(help="Discover and select completion and embedding models.")
app.add_typer(add_app, name="add")
app.add_typer(jobs_app, name="jobs")
app.add_typer(sources_app, name="sources")
app.add_typer(runs_app, name="runs")
app.add_typer(remove_app, name="remove")
app.add_typer(episodes_app, name="episodes")
app.add_typer(graph_app, name="graph")
app.add_typer(snapshot_app, name="snapshot")
app.add_typer(bench_app, name="bench")
app.add_typer(models_app, name="models")


@dataclass
class State:
    path: Path
    namespace: MemoryNamespace
    json_output: bool = False
    quiet: bool = False
    no_color: bool = False
    yes: bool = False

    def engine(self) -> MemoryEngine:
        return MemoryEngine.from_env(self.path, namespace=self.namespace)


def _version(value: bool) -> None:
    if value:
        typer.echo(MemoryEngine.VERSION)
        raise typer.Exit()


@app.callback()
def root(
    ctx: typer.Context,
    path: Path = typer.Option(Path("~/.trisynapse-memory/store"), "--path", envvar="TRISYNAPSE_MEMORY_PATH", help="Memory store directory."),
    user_id: str | None = typer.Option(None, "--user-id"),
    agent_id: str | None = typer.Option(None, "--agent-id"),
    project_id: str = typer.Option("default", "--project-id"),
    session_id: str | None = typer.Option(None, "--session-id"),
    json_output: bool = typer.Option(False, "--json", help="Write only machine-readable JSON to stdout."),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress non-essential output."),
    no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI color."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm irreversible operations."),
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True, help="Show the version and exit."),
) -> None:
    """Open the full terminal when no command is provided in a TTY."""

    del version
    state = State(path.expanduser(), MemoryNamespace(user_id=user_id, agent_id=agent_id, project_id=project_id, session_id=session_id), json_output, quiet, no_color, yes)
    ctx.obj = state
    if ctx.invoked_subcommand is None:
        if sys.stdin.isatty() and sys.stdout.isatty() and not json_output:
            from trisynapse_memory.terminal import run_terminal

            run_terminal(state.path, state.namespace)
        else:
            typer.echo(ctx.get_help())


def _state(ctx: typer.Context) -> State:
    return ctx.find_root().obj


def _normalize(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _emit(ctx: typer.Context, value: Any, *, table: Table | None = None) -> None:
    state = _state(ctx)
    if state.quiet:
        return
    normalized = _normalize(value)
    if state.json_output or table is None:
        typer.echo(json.dumps(normalized, indent=None if state.json_output else 2, ensure_ascii=False, default=str))
    else:
        Console(no_color=state.no_color).print(table)


@app.command("init")
def initialize(ctx: typer.Context, force_key: bool = typer.Option(False, "--force-key")) -> None:
    """Initialize a local store and permission-restricted API key."""

    state = _state(ctx)
    engine = MemoryEngine.open(state.path)
    key_path = state.path / ".api-key"
    if force_key or not key_path.exists():
        key_path.write_text(secrets.token_urlsafe(32) + "\n", encoding="utf-8")
        key_path.chmod(0o600)
    _emit(ctx, {"status": "initialized", "store_path": str(state.path.resolve()), "api_key_path": str(key_path), "trace": engine.verify_trace()})
    engine.close()


@add_app.command("observation")
def add_observation(ctx: typer.Context, text: str, episode: str = "manual:default", topic: str | None = None, source: str | None = None) -> None:
    state = _state(ctx)
    engine = state.engine()
    delta = engine.ingest_observation(text, episode_id=episode, source_ref={"type": "manual", **({"id": source} if source else {})}, scope={"topic_ids": [topic]} if topic else None, namespace=state.namespace)
    _emit(ctx, {"delta_id": delta.id, "seq": delta.seq, "hash": delta.hash})
    engine.close()


@add_app.command("document")
def add_document(ctx: typer.Context, file: Path, document_id: str | None = None, title: str | None = None) -> None:
    """Compatibility wrapper over unified source ingestion."""

    state = _state(ctx)
    engine = state.engine()
    result = engine.ingest(SourceInput(kind="file", path=str(file), source_key=document_id, title=title), namespace=state.namespace)
    _emit(ctx, result)
    engine.close()


@app.command("ingest")
def ingest_sources(ctx: typer.Context, source: list[str] = typer.Argument(None), manifest: Path | None = typer.Option(None, "--manifest")) -> None:
    """Ingest mixed local paths, public web pages, and public Git repositories."""

    descriptors = [_source_argument(value) for value in source]
    if manifest is not None:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        values = payload.get("sources", []) if isinstance(payload, dict) else payload
        descriptors.extend(SourceInput.model_validate(item) for item in values)
    if not descriptors:
        raise typer.BadParameter("provide at least one SOURCE or --manifest")
    state = _state(ctx)
    engine = state.engine()
    _emit(ctx, engine.ingest_many(descriptors, namespace=state.namespace))
    engine.close()


@sources_app.command("list")
def sources_list(ctx: typer.Context, include_removed: bool = False) -> None:
    state = _state(ctx)
    engine = state.engine()
    values = engine.list_sources(namespace=state.namespace, include_removed=include_removed)
    table = Table("ID", "Kind", "Version", "Status", "Title")
    for item in values:
        table.add_row(item.id, item.kind, str(item.version), item.status, item.title)
    _emit(ctx, {"sources": values}, table=table)
    engine.close()


@sources_app.command("show")
def sources_show(ctx: typer.Context, source_id: str) -> None:
    state = _state(ctx)
    engine = state.engine()
    _emit(ctx, engine.get_source(source_id, namespace=state.namespace))
    engine.close()


@sources_app.command("remove")
def sources_remove(ctx: typer.Context, source_id: str, reason: str = typer.Option("source removed", "--reason")) -> None:
    _remove_source(ctx, source_id, reason)


@runs_app.command("list")
def runs_list(ctx: typer.Context, limit: int = 100) -> None:
    state = _state(ctx)
    engine = state.engine()
    values = engine.list_ingestion_runs(namespace=state.namespace, limit=limit)
    table = Table("Run", "Status", "Sources", "Updated")
    for item in values:
        table.add_row(item.id, item.status, str(len(item.inputs)), str(item.updated_at))
    _emit(ctx, {"runs": values}, table=table)
    engine.close()


@runs_app.command("show")
def runs_show(ctx: typer.Context, run_id: str) -> None:
    state = _state(ctx)
    engine = state.engine()
    run = engine.get_ingestion_run(run_id)
    if run.namespace != state.namespace:
        raise typer.BadParameter("run does not belong to the active namespace")
    _emit(ctx, run)
    engine.close()


@runs_app.command("retry")
def runs_retry(ctx: typer.Context, run_id: str) -> None:
    state = _state(ctx)
    engine = state.engine()
    _emit(ctx, engine.retry_ingestion(run_id))
    engine.close()


@remove_app.command("memory")
def remove_memory(ctx: typer.Context, memory_id: list[str] = typer.Argument(...), reason: str = typer.Option(..., "--reason")) -> None:
    state = _state(ctx)
    if not state.yes and not typer.confirm("Physically redact these memory records? This cannot be undone"):
        raise typer.Abort()
    engine = state.engine()
    _emit(ctx, engine.remove(delta_ids=memory_id, reason=reason, namespace=state.namespace))
    engine.close()


@remove_app.command("source")
def remove_source(ctx: typer.Context, source_id: str, reason: str = typer.Option("source removed", "--reason")) -> None:
    _remove_source(ctx, source_id, reason)


def _remove_source(ctx: typer.Context, source_id: str, reason: str) -> None:
    state = _state(ctx)
    if not state.yes and not typer.confirm("Remove the original source and all derived memory? This cannot be undone"):
        raise typer.Abort()
    engine = state.engine()
    _emit(ctx, engine.remove_source(source_id, reason=reason, namespace=state.namespace))
    engine.close()


@app.command("list")
def list_memories(ctx: typer.Context, kind: list[str] = typer.Option(None, "--kind"), cursor: int | None = None, limit: int = 50, include_retracted: bool = False) -> None:
    state = _state(ctx)
    engine = state.engine()
    _emit(ctx, engine.list(namespace=state.namespace, kinds=kind, cursor=cursor, limit=limit, include_retracted=include_retracted))
    engine.close()


@app.command("get")
def get_memory(ctx: typer.Context, memory_id: str, include_retracted: bool = False) -> None:
    state = _state(ctx)
    engine = state.engine()
    _emit(ctx, engine.get(memory_id, namespace=state.namespace, include_retracted=include_retracted))
    engine.close()


@app.command("history")
def history(ctx: typer.Context, memory_id: str) -> None:
    state = _state(ctx)
    engine = state.engine()
    _emit(ctx, engine.history(memory_id, namespace=state.namespace))
    engine.close()


@app.command("correct")
def correct(ctx: typer.Context, memory_id: str, text: str, reason: str = "user correction") -> None:
    state = _state(ctx)
    engine = state.engine()
    _emit(ctx, engine.correct(delta_id=memory_id, text=text, reason=reason, namespace=state.namespace))
    engine.close()


@app.command("forget")
def forget(ctx: typer.Context, memory_id: str, reason: str = typer.Option(..., "--reason")) -> None:
    state = _state(ctx)
    engine = state.engine()
    _emit(ctx, engine.forget(delta_id=memory_id, reason=reason, namespace=state.namespace))
    engine.close()


@app.command("search")
def search(ctx: typer.Context, query: str, top_k: int = 12, episode_prefix: str | None = None) -> None:
    state = _state(ctx)
    engine = state.engine()
    _emit(ctx, engine.search(query, top_k=top_k, episode_prefix=episode_prefix, namespace=state.namespace))
    engine.close()


@app.command("query")
def query(ctx: typer.Context, question: str, top_k: int = 12, episode_prefix: str | None = None) -> None:
    state = _state(ctx)
    engine = state.engine()
    _emit(ctx, engine.query(question, top_k=top_k, episode_prefix=episode_prefix, namespace=state.namespace))
    engine.close()


@app.command("export")
def export_memory(ctx: typer.Context, output: Path, active_only: bool = False) -> None:
    state = _state(ctx)
    engine = state.engine()
    result = engine.export_to(output, namespace=state.namespace, include_retracted=not active_only)
    _emit(ctx, {"output": str(result.resolve())})
    engine.close()


@app.command("backup")
def backup(ctx: typer.Context, destination: Path) -> None:
    engine = _state(ctx).engine()
    _emit(ctx, {"output": str(engine.backup(destination).resolve())})
    engine.close()


@app.command("restore")
def restore(ctx: typer.Context, archive: Path, destination: Path) -> None:
    engine = MemoryEngine.restore_backup(archive, destination)
    _emit(ctx, {"status": "restored", "store_path": engine.store_path, "trace": engine.verify_trace()})
    engine.close()


@jobs_app.command("list")
def jobs_list(ctx: typer.Context, status: str | None = None, limit: int = 100) -> None:
    engine = _state(ctx).engine()
    _emit(ctx, {"jobs": engine.list_jobs(status=status, limit=limit)})
    engine.close()


@jobs_app.command("run")
def jobs_run(ctx: typer.Context, limit: int = 100) -> None:
    engine = _state(ctx).engine()
    _emit(ctx, {"jobs": engine.run_jobs(max_jobs=limit)})
    engine.close()


@episodes_app.command("list")
def episodes_list(ctx: typer.Context) -> None:
    state = _state(ctx)
    engine = state.engine()
    _emit(ctx, {"episodes": engine.list_episodes(namespace=state.namespace)})
    engine.close()


@episodes_app.command("compile")
def episodes_compile(ctx: typer.Context, episode: list[str] = typer.Option(None, "--episode")) -> None:
    state = _state(ctx)
    engine = state.engine()
    _emit(ctx, {"episode_recall_entries": engine.build_episode_recall(episode or None, namespace=state.namespace)})
    engine.close()


@graph_app.command("export")
def graph_export(ctx: typer.Context, format: str = "json", output: Path | None = None) -> None:
    engine = _state(ctx).engine()
    value = engine.export_graph(format=format)
    if output:
        output.write_text(json.dumps(value, indent=2) if isinstance(value, dict) else value, encoding="utf-8")
        _emit(ctx, {"output": str(output.resolve())})
    else:
        _emit(ctx, value)
    engine.close()


@snapshot_app.command("create")
def snapshot_create(ctx: typer.Context, label: str | None = None) -> None:
    engine = _state(ctx).engine()
    _emit(ctx, engine.snapshot.create(label))
    engine.close()


@snapshot_app.command("list")
def snapshot_list(ctx: typer.Context) -> None:
    engine = _state(ctx).engine()
    _emit(ctx, {"snapshots": engine.snapshot.list()})
    engine.close()


@snapshot_app.command("diff")
def snapshot_diff(ctx: typer.Context, a: str, b: str) -> None:
    engine = _state(ctx).engine()
    _emit(ctx, engine.snapshot.diff(a, b))
    engine.close()


@snapshot_app.command("rollback")
def snapshot_rollback(ctx: typer.Context, snapshot_id: str) -> None:
    engine = _state(ctx).engine()
    _emit(ctx, engine.snapshot.rollback(snapshot_id))
    engine.close()


@app.command("verify")
def verify(ctx: typer.Context) -> None:
    engine = _state(ctx).engine()
    _emit(ctx, engine.verify_trace())
    engine.close()


@app.command("migrate")
def migrate(ctx: typer.Context) -> None:
    engine = _state(ctx).engine()
    _emit(ctx, {"status": "migrated", "trace": engine.verify_trace(), "version": engine.VERSION})
    engine.close()


@app.command("check")
def check(ctx: typer.Context) -> None:
    """Check installation, provider credentials, store health, and pending work."""
    engine = _state(ctx).engine()
    _emit(ctx, engine.check())
    engine.close()


@app.command("serve")
def serve(ctx: typer.Context, host: str = "127.0.0.1", port: int = 8765, no_auth: bool = False, studio: bool = False) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter("install server dependencies with: pip install 'trisynapse-memory[server]'") from exc
    from trisynapse_memory.api import create_app

    state = _state(ctx)
    key_path = state.path / ".api-key"
    configured = os.getenv("TRISYNAPSE_MEMORY_API_KEY", "").strip()
    if not no_auth and not configured and not key_path.exists():
        state.path.mkdir(parents=True, exist_ok=True)
        key_path.write_text(secrets.token_urlsafe(32) + "\n", encoding="utf-8")
        key_path.chmod(0o600)
    api_key = None if no_auth else configured or (key_path.read_text().strip() if key_path.exists() else None)
    uvicorn.run(create_app(state.path, api_key=api_key, studio=studio), host=host, port=port)


@models_app.command("current")
def models_current(ctx: typer.Context) -> None:
    engine = _state(ctx).engine()
    _emit(ctx, engine.get_model_configuration_status())
    engine.close()


@models_app.command("providers")
def models_providers(ctx: typer.Context) -> None:
    engine = _state(ctx).engine()
    values = engine.list_providers()
    table = Table("Provider", "Roles", "Credential", "Ready")
    for item in values:
        table.add_row(
            item.id,
            ", ".join(role.value for role in item.roles),
            item.credential_env or "none",
            "yes" if item.credential_configured or item.credential_env is None else "no",
        )
    _emit(ctx, {"providers": values}, table=table)
    engine.close()


@models_app.command("list")
def models_list(
    ctx: typer.Context,
    role: ProviderRole = typer.Option(..., "--role"),
    provider: str = typer.Option(..., "--provider"),
    refresh: bool = typer.Option(False, "--refresh"),
    base_url: str | None = typer.Option(None, "--base-url"),
) -> None:
    engine = _state(ctx).engine()
    values = engine.list_models(role, provider, refresh=refresh, base_url=base_url)
    table = Table("Model", "Roles", "Vision", "Context", "Source")
    for item in values:
        table.add_row(
            item.id,
            ", ".join(value.value for value in item.roles),
            "yes" if item.vision else "—" if item.vision is None else "no",
            str(item.context_length or "—"),
            item.source,
        )
    _emit(ctx, {"models": values}, table=table)
    engine.close()


@models_app.command("set")
def models_set(
    ctx: typer.Context,
    role: ProviderRole,
    provider: str,
    model: str | None = typer.Argument(None),
    base_url: str | None = typer.Option(None, "--base-url"),
) -> None:
    state = _state(ctx)
    engine = state.engine()
    configuration = engine.get_model_configuration()
    selection = ProviderSelection(provider=provider, model=model, base_url=base_url)
    confirm_rebuild = False
    if role == ProviderRole.COMPLETION:
        configuration.completion = selection
    else:
        configuration.embedding = selection
        if configuration.embedding != engine.get_model_configuration().embedding:
            if not state.yes:
                if not sys.stdin.isatty():
                    engine.close()
                    raise typer.BadParameter(
                        "embedding changes rebuild the full vector index; use --yes in non-interactive mode"
                    )
                if not typer.confirm(
                    "Changing embeddings rebuilds the complete vector index and may use provider credits. Continue?"
                ):
                    engine.close()
                    raise typer.Abort()
            confirm_rebuild = True
    result = engine.set_model_configuration(
        configuration,
        confirm_embedding_rebuild=confirm_rebuild,
        wait=confirm_rebuild,
    )
    _emit(ctx, result)
    engine.close()


@models_app.command("test")
def models_test(ctx: typer.Context, role: ProviderRole = typer.Option(..., "--role")) -> None:
    """Send one small, potentially billable request to the selected provider."""
    engine = _state(ctx).engine()
    _emit(ctx, engine.test_model_connection(role))
    engine.close()


@bench_app.command("run")
def bench_run(ctx: typer.Context, suite: str, data_root: str | None = None, max_questions: int = 25, mode: str = "retrieval") -> None:
    state = _state(ctx)
    engine = state.engine()
    model_configuration = engine.get_model_configuration()
    engine.close()
    if suite in {"halumem", "memorydoc"}:
        from trisynapse_memory.benchmarks import run_trace_recall_smoke
        value = run_trace_recall_smoke(
            suite, data_root or f"data/{suite}", max_questions=max_questions,
            mode=mode, model_configuration=model_configuration,
        )
    else:
        from trisynapse_memory.benchmarks import run_trace_recall_benchmark
        value = run_trace_recall_benchmark(
            suite, data_root or f"data/{suite}", max_questions=max_questions,
            mode=mode, model_configuration=model_configuration,
        )
    _emit(ctx, value)


@bench_app.command("gate")
def bench_gate(ctx: typer.Context, data_root: str = "data", mode: str = "retrieval") -> None:
    from trisynapse_memory.benchmarks.release import evaluate_release_gate

    value = evaluate_release_gate(data_root, mode=mode)
    _emit(ctx, value)
    if not value["passed"]:
        raise typer.Exit(1)


def _source_argument(value: str) -> SourceInput:
    if value.startswith("https://"):
        kind = "git" if value.endswith(".git") or "github.com/" in value or "gitlab.com/" in value else "url"
        return SourceInput(kind=kind, url=value)
    path = Path(value).expanduser()
    return SourceInput(kind="directory" if path.is_dir() else "file", path=str(path))


def main() -> None:
    app(prog_name="trisynapse-memory")


if __name__ == "__main__":
    main()
