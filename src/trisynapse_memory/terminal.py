"""Interactive Textual terminal for Trisynapse Memory."""

from __future__ import annotations

import json
import shlex
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.json import JSON
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.suggester import Suggester
from textual.timer import Timer
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Input,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from trisynapse_memory.engine import MemoryEngine, SourceInput
from trisynapse_memory.engine.models import (
    IngestionRun,
    MemoryQueryResult,
    MemoryNamespace,
    MemorySearchResult,
    ModelDescriptor,
    ProviderRole,
    ProviderSelection,
    QueryStep,
)


# Pre-rendered from design_docs/icon.png. The wider cell ratio prevents the
# source mark from looking horizontally compressed in common terminal fonts.
WIDE_LOGO = r"""
                            ▄                  ▄██
                  ▄▄▄████████████████▄▄   ▄▄▄█▀▀▀
              ▄▄███████████████████████████▄
           ▄████████▄▄████████████████████████▄
        ▄███████████████████████████████████████▄
      ▄██████████████████████████████████████████
    ▄████████████████████████████████████████████
   ▄██████████████████████▀▀██████████▀▀██████▀▀
  ███▀▀███████████████████▄▄██████▀  ▀▀▄▄
  ████████████████████████████████▄     ▀█▄▄▄
  ▀█████████████████▀███████████████        ▀▀▀▀▀▀▀▀
     ▀▀▀███▀▀▀▀▀    █████████████▀▀
                       ▀▀▀▀▀▀▀
""".strip("\n")

COMPACT_LOGO = r"""
             ▄▄▄▄▄▄▄▄▄      ▄██
        ▄▄█████████████████▄
     ▄▄█████▄████████████████▄
   ▄██████████████████████████
  ▄█████████████▀███████████▀▀
 ██▄█████████████████▀ ▀█▄
 ▀████████████████████    ▀▀▀▀▀▀
     ▀▀      ▀▀▀▀▀▀▀▀
""".strip("\n")

COMMAND_HELP: dict[str, tuple[str, str]] = {
    "/ingest": ("/ingest SOURCE...", "Import files, folders, pages, or public Git repositories"),
    "/sources": ("/sources", "List retained sources"),
    "/search": ("/search QUERY", "Inspect ranked Trace evidence"),
    "/timeline": ("/timeline", "Open the Trace timeline"),
    "/history": ("/history MEMORY_ID", "Show one memory's lifecycle"),
    "/correct": ("/correct MEMORY_ID TEXT", "Append a correction"),
    "/forget": ("/forget MEMORY_ID REASON", "Logically retract memory"),
    "/remove": ("/remove MEMORY_ID... --yes", "Physically redact memory"),
    "/jobs": ("/jobs", "Show durable jobs"),
    "/namespace": ("/namespace [PROJECT]", "Show or change the project namespace"),
    "/check": ("/check", "Check installation, providers, and store consistency"),
    "/model": ("/model [completion|embedding]", "Choose providers and models"),
    "/config": ("/config", "Show active configuration"),
    "/help": ("/help", "Show commands and usage"),
    "/clear": ("/clear", "Clear terminal output"),
    "/exit": ("/exit", "Close the terminal"),
}

COMMANDS_WITH_ARGUMENTS = {
    "/ingest", "/search", "/history", "/correct", "/forget", "/remove", "/namespace", "/model",
}
MEMORY_ID_COMMANDS = {"/history", "/correct", "/forget", "/remove"}
SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def _write_json(log: RichLog, value: object) -> None:
    """Write structured data using a Rich renderable supported by Textual."""

    log.write(JSON.from_data(value))


def _one_line(value: str, limit: int = 72) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _byte_size(value: int) -> str:
    amount = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB"):
        if amount < 1024 or suffix == "GiB":
            return f"{amount:.0f}{suffix}" if suffix == "B" else f"{amount:.1f}{suffix}"
        amount /= 1024
    return f"{value}B"


def _expects_memory_id(value: str, command: str) -> bool:
    remainder = value.split(" ", 1)[1] if " " in value else ""
    return command == "/remove" or " " not in remainder


def _path_suggestions(fragment: str, *, limit: int = 8) -> list[str]:
    """Return safe, display-ready path completions for an unfinished token."""

    if fragment.startswith(("http://", "https://")) or any(character in fragment for character in "\"'"):
        return []
    if "/" in fragment:
        raw_directory, prefix = fragment.rsplit("/", 1)
        lookup = Path(raw_directory or "/" if fragment.startswith("/") else raw_directory or ".").expanduser()
        display_prefix = f"{raw_directory}/" if raw_directory else "/" if fragment.startswith("/") else ""
    else:
        prefix = fragment
        lookup = Path(".")
        display_prefix = ""
    try:
        children = sorted(
            (child for child in lookup.iterdir() if " " not in child.name and child.name.casefold().startswith(prefix.casefold())),
            key=lambda child: (not child.is_dir(), child.name.casefold()),
        )
    except OSError:
        return []
    return [f"{display_prefix}{child.name}{'/' if child.is_dir() else ''}" for child in children[:limit]]


class MemorySuggester(Suggester):
    """Complete commands, ingestion paths, and memory identifiers."""

    def __init__(self, memory_ids: Callable[[], list[str]]) -> None:
        super().__init__(use_cache=False, case_sensitive=False)
        self._memory_ids = memory_ids

    async def get_suggestion(self, value: str) -> str | None:
        if not value.startswith("/"):
            return None
        if " " not in value:
            for command in COMMAND_HELP:
                if command.casefold().startswith(value.casefold()) and command != value:
                    return command + (" " if command in COMMANDS_WITH_ARGUMENTS else "")
            return None
        command = value.split(" ", 1)[0].casefold()
        fragment = value.rsplit(" ", 1)[-1]
        base = value[: len(value) - len(fragment)]
        if command == "/ingest":
            paths = _path_suggestions(fragment, limit=1)
            return base + paths[0] if paths else None
        if command in MEMORY_ID_COMMANDS and _expects_memory_id(value, command):
            for memory_id in self._memory_ids():
                if memory_id.casefold().startswith(fragment.casefold()) and memory_id != fragment:
                    return base + memory_id
        return None


class CompletionInput(Input):
    """Input whose Tab key accepts Textual's current ghost suggestion."""

    BINDINGS = [Binding("tab", "cursor_right", "Accept suggestion", show=False), *Input.BINDINGS]


class ModelSelectorScreen(ModalScreen[bool]):
    """Searchable provider/model picker shared by both model roles."""

    CSS = """
    ModelSelectorScreen { align: center middle; background: rgba(0,0,0,0.55); }
    #model-dialog { width: 76; max-width: 94%; height: auto; max-height: 92%;
                    padding: 1 2; border: round $primary; background: $surface; }
    #model-dialog Select, #model-dialog Input { margin-bottom: 1; }
    #model-actions { height: auto; margin-top: 1; }
    #model-actions Button { margin-right: 1; }
    #model-message { min-height: 3; color: $text-muted; }
    """

    def __init__(self, engine: MemoryEngine, role: ProviderRole = ProviderRole.COMPLETION) -> None:
        super().__init__()
        self.engine = engine
        self.initial_role = role
        self.models: list[ModelDescriptor] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="model-dialog"):
            yield Static("[bold]Choose models[/]\nSelections are saved in this memory store.")
            yield Select(
                [("Completion", ProviderRole.COMPLETION.value), ("Embedding", ProviderRole.EMBEDDING.value)],
                value=self.initial_role.value,
                id="model-role",
            )
            yield Select([], prompt="Choose a provider", id="model-provider")
            yield Input(placeholder="Search models or enter an exact model ID", id="model-search")
            yield Select([], prompt="Choose a model", id="model-choice")
            yield Input(placeholder="Custom /v1 base URL when required", id="model-base-url")
            yield Checkbox(
                "Yes, rebuild the complete vector index if the embedding choice changes",
                id="model-rebuild",
            )
            yield Static("", id="model-message")
            with Horizontal(id="model-actions"):
                yield Button("Apply", variant="primary", id="model-apply")
                yield Button("Refresh models", id="model-refresh")
                yield Button("Test connection", id="model-test")
                yield Button("Cancel", id="model-cancel")

    def on_mount(self) -> None:
        configuration = self.engine.get_model_configuration()
        selected = configuration.completion if self.initial_role == ProviderRole.COMPLETION else configuration.embedding
        self._set_providers(selected.provider)

    def _role(self) -> ProviderRole:
        value = self.query_one("#model-role", Select).value
        return ProviderRole(str(value))

    def _set_providers(self, selected: str | None = None) -> None:
        role = self._role()
        providers = [item for item in self.engine.list_providers() if role in item.roles]
        widget = self.query_one("#model-provider", Select)
        widget.set_options([
            (
                f"{item.display_name}{'' if item.credential_configured or item.credential_env is None else f' · needs {item.credential_env}'}",
                item.id,
            )
            for item in providers
        ])
        configuration = self.engine.get_model_configuration()
        current = configuration.completion if role == ProviderRole.COMPLETION else configuration.embedding
        value = selected if selected in {item.id for item in providers} else current.provider
        widget.value = value
        self.query_one("#model-base-url", Input).value = current.base_url or ""
        self.query_one("#model-rebuild", Checkbox).display = role == ProviderRole.EMBEDDING
        self._load_models()

    def _load_models(self, *, refresh: bool = False) -> None:
        provider = self.query_one("#model-provider", Select).value
        message = self.query_one("#model-message", Static)
        if provider is Select.BLANK:
            return
        if provider == "none":
            self.models = []
            self.query_one("#model-choice", Select).set_options([])
            message.update("Completion is disabled. Grounded extractive answers remain available.")
            return
        role = self._role()
        provider_id = str(provider)
        base_url = self.query_one("#model-base-url", Input).value or None
        message.update("[cyan]Loading model catalog…[/]")
        self._fetch_models(role, provider_id, refresh, base_url)

    @work(thread=True, group="model-catalog", exclusive=True)
    def _fetch_models(
        self,
        role: ProviderRole,
        provider: str,
        refresh: bool,
        base_url: str | None,
    ) -> None:
        try:
            models = self.engine.list_models(
                role, provider, refresh=refresh, base_url=base_url
            )
        except Exception as exc:
            self.app.call_from_thread(self._models_loaded, role, provider, [], str(exc))
            return
        self.app.call_from_thread(self._models_loaded, role, provider, models, None)

    def _models_loaded(
        self,
        role: ProviderRole,
        provider: str,
        models: list[ModelDescriptor],
        error: str | None,
    ) -> None:
        if not self.is_mounted:
            return
        if role != self._role() or provider != self.query_one("#model-provider", Select).value:
            return
        message = self.query_one("#model-message", Static)
        self.models = models
        self._filter_models()
        if error:
            message.update(
                f"[red]{error}[/]\nYou may still enter an exact model ID after configuring credentials."
            )
        else:
            message.update(
                f"{len(self.models)} models · selecting does not send an inference request."
            )

    def _filter_models(self) -> None:
        fragment = self.query_one("#model-search", Input).value.casefold().strip()
        values = [
            item for item in self.models
            if not fragment or fragment in item.id.casefold() or fragment in item.display_name.casefold()
        ]
        choice = self.query_one("#model-choice", Select)
        choice.set_options([
            (
                f"{item.display_name}{' · vision' if item.vision else ''}{f' · {item.context_length:,} ctx' if item.context_length else ''}",
                item.id,
            )
            for item in values[:500]
        ])
        configuration = self.engine.get_model_configuration()
        current = configuration.completion if self._role() == ProviderRole.COMPLETION else configuration.embedding
        if current.provider == self.query_one("#model-provider", Select).value and any(
            item.id == current.model for item in values
        ):
            choice.value = current.model

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "model-role" and self.is_mounted:
            self._set_providers()
        elif event.select.id == "model-provider" and self.is_mounted:
            self._load_models()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "model-search":
            self._filter_models()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "model-cancel":
            self.dismiss(False)
            return
        if event.button.id == "model-refresh":
            self._load_models(refresh=True)
            return
        role = self._role()
        provider_value = self.query_one("#model-provider", Select).value
        if provider_value is Select.BLANK:
            self.query_one("#model-message", Static).update("[red]Choose a provider.[/]")
            return
        provider = str(provider_value)
        model_value = self.query_one("#model-choice", Select).value
        manual = self.query_one("#model-search", Input).value.strip()
        model = None if provider == "none" else (
            str(model_value) if model_value is not Select.BLANK else manual or None
        )
        try:
            selection = ProviderSelection(
                provider=provider,
                model=model,
                base_url=self.query_one("#model-base-url", Input).value or None,
            )
            if event.button.id == "model-test":
                self.query_one("#model-message", Static).update(
                    "[cyan]Testing connection… This may incur a provider charge.[/]"
                )
                self._test_connection(role, selection)
                return
            configuration = self.engine.get_model_configuration()
            changed_embedding = role == ProviderRole.EMBEDDING and selection != configuration.embedding
            if changed_embedding and not self.query_one("#model-rebuild", Checkbox).value:
                self.query_one("#model-message", Static).update(
                    "[yellow]Select Yes to approve the required embedding rebuild.[/]"
                )
                return
            if role == ProviderRole.COMPLETION:
                configuration.completion = selection
            else:
                configuration.embedding = selection
            self.query_one("#model-message", Static).update(
                "[cyan]Rebuilding embedding index…[/]"
                if changed_embedding
                else "[cyan]Saving model configuration…[/]"
            )
            self._save_model_configuration(configuration, changed_embedding)
        except Exception as exc:
            self.query_one("#model-message", Static).update(f"[red]{exc}[/]")

    @work(thread=True, group="model-test", exclusive=True)
    def _test_connection(self, role: ProviderRole, selection: ProviderSelection) -> None:
        try:
            result = self.engine.test_model_connection(role, selection)
        except Exception as exc:
            self.app.call_from_thread(self._model_action_failed, str(exc))
            return
        self.app.call_from_thread(self._connection_tested, result.ok, result.message)

    def _connection_tested(self, ok: bool, message: str) -> None:
        if self.is_mounted:
            self.query_one("#model-message", Static).update(
                f"{'[green]' if ok else '[red]'}{message}[/] "
                "This test sends a potentially billable request."
            )

    @work(thread=True, group="model-save", exclusive=True)
    def _save_model_configuration(self, configuration: Any, changed_embedding: bool) -> None:
        try:
            result = self.engine.set_model_configuration(
                configuration,
                confirm_embedding_rebuild=changed_embedding,
                wait=changed_embedding,
            )
        except Exception as exc:
            self.app.call_from_thread(self._model_action_failed, str(exc))
            return
        self.app.call_from_thread(
            self._model_configuration_saved,
            result.status,
            result.message,
        )

    def _model_configuration_saved(self, status: str, message: str | None) -> None:
        if not self.is_mounted:
            return
        if status == "rebuild_failed":
            self.query_one("#model-message", Static).update(f"[red]{message}[/]")
            return
        self.dismiss(True)

    def _model_action_failed(self, error: str) -> None:
        if self.is_mounted:
            self.query_one("#model-message", Static).update(f"[red]{error}[/]")


def logo_for_width(width: int) -> str:
    """Return the generated source-faithful logo for the terminal width."""

    return WIDE_LOGO if width >= 72 else COMPACT_LOGO


class MemoryTerminal(App[None]):
    """Grounded query prompt plus operational slash commands."""

    TITLE = "Trisynapse Memory"
    SUB_TITLE = "Store traces. Recall meaning."
    CSS = """
    Screen { layout: vertical; background: $surface; }
    #header { height: 16; padding: 0 1; }
    #brand { width: 30%; min-width: 34; height: 1fr; padding: 1; color: $text; content-align: center middle; }
    #inspector { width: 1fr; height: 1fr; }
    #status { height: 3; padding: 0 2; background: $boost; }
    #activity { width: 1fr; height: 1fr; margin: 0 2; padding: 1 0; }
    .view-log { height: 1fr; border: round $primary; padding: 1; }
    #config-view { padding: 2; }
    #operation-status { height: 1; margin: 0 2; color: $text-muted; }
    #recommendations { height: auto; min-height: 1; margin: 0 2; color: $text-muted; }
    #prompt { margin: 0 2 1 2; }
    .status-cell { width: 1fr; }
    """
    BINDINGS = [("ctrl+c", "interrupt", "Clear/exit"), ("ctrl+l", "clear", "Clear"), ("ctrl+d", "quit", "Exit")]

    def __init__(self, path: Path, namespace: MemoryNamespace) -> None:
        super().__init__()
        self.path = path
        self.namespace = namespace
        self.engine = MemoryEngine.from_env(path, namespace=namespace)
        self._operation_label: str | None = None
        self._operation_started_at = 0.0
        self._spinner_index = 0
        self._operation_status: Static | None = None
        self._operation_timer: Timer | None = None
        self._panel_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="header"):
            yield Static("", id="brand")
            with TabbedContent(initial="sources-tab", id="inspector"):
                with TabPane("Sources", id="sources-tab"):
                    yield RichLog(id="sources-log", classes="view-log", wrap=True, markup=True)
                with TabPane("Trace", id="trace-tab"):
                    yield RichLog(id="trace-log", classes="view-log", wrap=True, markup=True)
                with TabPane("Jobs", id="jobs-tab"):
                    yield RichLog(id="jobs-log", classes="view-log", wrap=True, markup=True)
                with TabPane("Config", id="config-tab"):
                    yield Static(id="config-view")
        with Horizontal(id="status"):
            yield Static(id="store", classes="status-cell")
            yield Static(id="provider", classes="status-cell")
            yield Static(id="storage-status", classes="status-cell")
            yield Static(id="pending", classes="status-cell")
        yield RichLog(id="activity", wrap=True, markup=True)
        yield Static("", id="operation-status")
        yield Static("Type [bold]/[/] to see commands · [bold]Tab[/] or [bold]→[/] accepts a suggestion", id="recommendations")
        yield CompletionInput(
            placeholder="Ask memory, or type /help …",
            suggester=MemorySuggester(self._memory_ids),
            id="prompt",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#brand", Static).update(
            f"{logo_for_width(max(1, int(self.size.width * 0.30)))}\n"
            f"Trisynapse Memory {self.engine.VERSION}"
        )
        self._refresh_status()
        self.query_one("#activity", RichLog).write(
            "Ready. Plain text runs a grounded query. Use [bold]/ingest[/], [bold]/sources[/], or [bold]/help[/]."
        )
        self.query_one("#prompt", Input).focus()
        self._update_config_view()
        self._operation_status = self.query_one("#operation-status", Static)
        self._operation_timer = self.set_interval(0.1, self._animate_operation)
        self._refresh_panels()
        self._panel_timer = self.set_interval(2.5, self._refresh_panels)

    def on_resize(self) -> None:
        brand = self.query_one("#brand", Static)
        brand.update(
            f"{logo_for_width(max(1, int(self.size.width * 0.30)))}\n"
            f"Trisynapse Memory {self.engine.VERSION}"
        )

    def on_unmount(self) -> None:
        if self._operation_timer is not None:
            self._operation_timer.stop()
        if self._panel_timer is not None:
            self._panel_timer.stop()
        self._operation_status = None
        self.engine.close()

    def action_clear(self) -> None:
        for log in self.query(RichLog):
            log.clear()

    def action_interrupt(self) -> None:
        prompt = self.query_one("#prompt", Input)
        if prompt.value:
            prompt.value = ""
        elif self._operation_label:
            self.query_one("#activity", RichLog).write(
                "[yellow]The active operation is finishing safely; it cannot be interrupted mid-write.[/]"
            )
        else:
            self.exit()

    def on_input_changed(self, event: Input.Changed) -> None:
        self.query_one("#recommendations", Static).update(self._recommendations(event.value))

    def on_tabbed_content_tab_activated(
        self, event: TabbedContent.TabActivated
    ) -> None:
        """Render the selected inspector pane without waiting for a worker.

        Textual may activate a hidden pane before the mount-time panel worker has
        returned, especially on Windows. Reading the small, bounded inspector
        page here makes tab selection deterministic while the periodic worker
        continues to refresh panes that are not being interacted with.
        """

        if event.tabbed_content.id != "inspector":
            return
        self._refresh_inspector_pane(event.pane.id)

    def _refresh_inspector_pane(self, pane_id: str | None) -> None:
        try:
            if pane_id == "sources-tab":
                self._render_sources(
                    self.engine.list_sources(
                        namespace=self.namespace,
                        include_removed=True,
                        limit=20,
                    )
                )
            elif pane_id == "trace-tab":
                trace = self.engine.list(
                    namespace=self.namespace,
                    limit=20,
                    include_retracted=True,
                )
                self._render_trace(trace.items)
            elif pane_id == "jobs-tab":
                self._render_jobs(self.engine.list_jobs(limit=20))
            elif pane_id == "config-tab":
                self._update_config_view()
        except Exception:
            # The periodic refresh will retry. Tab selection itself should stay
            # usable if another operation temporarily holds the local store.
            return

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value:
            return
        log = self.query_one("#activity", RichLog)
        if self._operation_label:
            log.write("[yellow]Wait for the active operation to finish before submitting another request.[/]")
            return
        event.input.value = ""
        log.write(Text.assemble(("> ", "bold cyan"), value))
        try:
            if value.startswith("/"):
                self._slash(value)
            else:
                self._start_operation("Filtering private query data")
                self._run_query(value)
        except Exception as exc:
            log.write(Text(str(exc), style="red"))
        self._refresh_status()

    def _start_operation(self, label: str) -> None:
        self._operation_label = label
        self._operation_started_at = time.monotonic()
        self._spinner_index = 0
        self._animate_operation()

    def _animate_operation(self) -> None:
        status = self._operation_status
        if status is None or not status.is_mounted:
            return
        if not self._operation_label:
            status.update("")
            return
        frame = SPINNER_FRAMES[self._spinner_index % len(SPINNER_FRAMES)]
        self._spinner_index += 1
        elapsed = time.monotonic() - self._operation_started_at
        status.update(Text(f"{frame}  {self._operation_label} · {elapsed:.1f}s", style="cyan"))

    def _finish_operation(self) -> None:
        self._operation_label = None
        self.query_one("#operation-status", Static).update("")
        self.query_one("#prompt", Input).focus()
        self._refresh_status()
        self._refresh_panels()

    def _update_operation(self, label: str) -> None:
        self._operation_label = label

    @work(thread=True, group="panel-refresh", exclusive=True)
    def _refresh_panels(self) -> None:
        try:
            sources = self.engine.list_sources(
                namespace=self.namespace,
                include_removed=True,
                limit=20,
            )
            trace = self.engine.list(
                namespace=self.namespace,
                limit=20,
                include_retracted=True,
            )
            jobs = self.engine.list_jobs(limit=20)
        except Exception:
            return
        self.call_from_thread(self._panels_loaded, sources, trace, jobs)

    def _panels_loaded(self, sources: list[Any], trace: Any, jobs: list[Any]) -> None:
        self._render_sources(sources)
        self._render_trace(trace.items)
        self._render_jobs(jobs)
        self._update_config_view()

    def _render_sources(self, sources: list[Any]) -> None:
        log = self.query_one("#sources-log", RichLog)
        log.clear()
        if not sources:
            log.write("[dim]No sources yet · use /ingest PATH or URL[/]")
            return
        for source in sources:
            chunks = f"{source.chunk_count} chunk{'s' if source.chunk_count != 1 else ''}"
            log.write(
                Text.assemble(
                    (_one_line(source.title, 38), "bold"),
                    (f" · {source.kind} · {source.status} · {chunks} · {_byte_size(source.byte_size)}", "dim"),
                )
            )
            log.write(Text.assemble(("  source ", "dim"), (source.id, "cyan")))

    def _render_trace(self, items: list[Any]) -> None:
        log = self.query_one("#trace-log", RichLog)
        log.clear()
        if not items:
            log.write("[dim]No Trace records in this namespace.[/]")
            return
        for item in items:
            log.write(
                Text.assemble(
                    (f"#{item.seq} {item.kind}", "bold"),
                    (f" · {_one_line(item.text, 52)}", "dim"),
                )
            )
            identity = Text.assemble(("  delta ", "dim"), (item.id, "cyan"))
            if item.episode_id:
                identity.append(" · episode ", style="dim")
                identity.append(item.episode_id, style="cyan")
            log.write(identity)

    def _render_jobs(self, jobs: list[Any]) -> None:
        log = self.query_one("#jobs-log", RichLog)
        log.clear()
        if not jobs:
            log.write("[dim]No durable jobs.[/]")
            return
        for job in jobs:
            log.write(
                Text.assemble(
                    (job.status, "bold"),
                    (
                        f" · {job.kind} · attempt {job.attempts}/{job.max_attempts}"
                        f" · {job.updated_at.strftime('%Y-%m-%d %H:%M:%S')}",
                        "dim",
                    ),
                )
            )
            log.write(Text.assemble(("  job ", "dim"), (job.id, "cyan")))

    def _operation_failed(self, error: str) -> None:
        log = self.query_one("#activity", RichLog)
        log.write(Text(error, style="red"))
        log.write("")
        self._finish_operation()

    def _query_step(self, step: QueryStep) -> None:
        duration = f" · {step.duration_ms:.0f}ms" if step.duration_ms is not None else ""
        self.query_one("#activity", RichLog).write(
            Text.assemble(("  ✓ ", "green"), (step.label, "dim"), (duration, "dim"))
        )
        self._operation_label = {
            "input": "Reading query",
            "classification": "Running hybrid retrieval",
            "routes": "Evaluating retrieval confidence",
            "confidence": "Refining or grounding evidence",
            "refinement": "Checking refined evidence",
            "deep_recall": "Grounding Deep Recall results",
            "grounding": "Generating grounded answer",
            "answer": "Recording citation access",
            "audit": "Finishing response",
        }.get(step.phase, "Continuing retrieval")

    @work(thread=True, group="terminal-query", exclusive=True)
    def _run_query(self, question: str) -> None:
        try:
            result = self.engine.query(
                question,
                namespace=self.namespace,
                on_step=lambda step: self.call_from_thread(self._query_step, step),
            )
        except Exception as exc:
            self.call_from_thread(self._operation_failed, str(exc))
            return
        self.call_from_thread(self._query_finished, result)

    def _query_finished(self, result: MemoryQueryResult) -> None:
        log = self.query_one("#activity", RichLog)
        log.write(Text(result.answer))
        for citation in result.citations:
            location = f" · {citation.locator}" if citation.locator else ""
            log.write(Text(f"[{citation.delta_id}]{location}", style="dim"))
        log.write("")
        self._finish_operation()

    @work(thread=True, group="terminal-search", exclusive=True)
    def _run_search(self, query: str) -> None:
        try:
            result = self.engine.search(
                query,
                namespace=self.namespace,
                on_step=lambda step: self.call_from_thread(self._query_step, step),
            )
        except Exception as exc:
            self.call_from_thread(self._operation_failed, str(exc))
            return
        self.call_from_thread(self._search_finished, result)

    def _search_finished(self, result: MemorySearchResult) -> None:
        log = self.query_one("#activity", RichLog)
        if not result.hits:
            log.write("No grounded Trace evidence found.")
        for rank, hit in enumerate(result.hits, 1):
            log.write(Text(f"{rank}. {hit.text} · {hit.score:.3f}", style="dim"))
        log.write("")
        self._finish_operation()

    @work(thread=True, group="terminal-ingestion", exclusive=True)
    def _run_ingestion(self, sources: list[SourceInput]) -> None:
        try:
            result = self.engine.ingest_many(
                sources,
                namespace=self.namespace,
                on_progress=lambda label: self.call_from_thread(
                    self._update_operation, label
                ),
            )
        except Exception as exc:
            self.call_from_thread(self._operation_failed, str(exc))
            return
        self.call_from_thread(self._ingestion_finished, result)

    def _ingestion_finished(self, result: IngestionRun) -> None:
        self.query_one(TabbedContent).active = "sources-tab"
        succeeded = sum(item.status in {"success", "skipped"} for item in result.results)
        failed = sum(item.status == "failed" for item in result.results)
        log = self.query_one("#activity", RichLog)
        log.write(
            Text(
                f"Ingestion finished · {succeeded} succeeded · {failed} failed · run {result.id}",
                style="green" if not failed else "yellow",
            )
        )
        for item in result.results:
            if item.status == "failed":
                log.write(
                    Text(
                        f"  failed source #{item.index + 1}: {item.error or 'unknown error'}",
                        style="red",
                    )
                )
        log.write("")
        self._finish_operation()

    @work(thread=True, group="terminal-command", exclusive=True)
    def _run_command(
        self,
        operation: Callable[[], Any],
        on_success: Callable[[Any], None],
    ) -> None:
        try:
            result = operation()
        except Exception as exc:
            self.call_from_thread(self._operation_failed, str(exc))
            return
        self.call_from_thread(on_success, result)

    def _sources_finished(self, sources: list[Any]) -> None:
        self.query_one(TabbedContent).active = "sources-tab"
        self._render_sources(sources)
        self.query_one("#activity", RichLog).write(
            f"[green]Sources loaded[/] · {len(sources)} retained source(s)."
        )
        self._finish_operation()

    def _timeline_finished(self, value: Any) -> None:
        self.query_one(TabbedContent).active = "trace-tab"
        self._render_trace(value.items)
        self.query_one("#activity", RichLog).write("[green]Trace timeline loaded.[/]")
        self._finish_operation()

    def _jobs_finished(self, jobs: list[Any]) -> None:
        self.query_one(TabbedContent).active = "jobs-tab"
        self._render_jobs(jobs)
        self.query_one("#activity", RichLog).write(
            f"[green]Jobs loaded[/] · {len(jobs)} job(s)."
        )
        self._finish_operation()

    def _history_finished(self, value: Any) -> None:
        self.query_one(TabbedContent).active = "trace-tab"
        self._render_trace(value.events)
        self.query_one("#activity", RichLog).write(
            "[green]Memory history loaded[/] in Trace."
        )
        self._finish_operation()

    def _memory_operation_finished(self, value: Any) -> None:
        _write_json(self.query_one("#activity", RichLog), value.model_dump(mode="json"))
        self.query_one("#activity", RichLog).write("")
        self._finish_operation()

    def _check_finished(self, value: dict[str, Any]) -> None:
        self.query_one(TabbedContent).active = "config-tab"
        self.query_one("#config-view", Static).update(
            "[bold]System check[/]\n\n" + json.dumps(value, indent=2, default=str)
        )
        self.query_one("#activity", RichLog).write("[green]System check completed.[/]")
        self._finish_operation()

    def _recommendations(self, value: str) -> str:
        if not value:
            return "Type [bold]/[/] to see commands · [bold]Tab[/] or [bold]→[/] accepts a suggestion"
        if not value.startswith("/"):
            return "[dim]Press Enter for a grounded query with citations · type / for operations[/]"
        command = value.split(" ", 1)[0].casefold()
        if " " not in value and command not in COMMAND_HELP:
            matches = [
                f"[bold]{name}[/] [dim]{description}[/]"
                for name, (_, description) in COMMAND_HELP.items()
                if name.startswith(command)
            ]
            return "   ".join(matches[:4]) or "[red]No matching command[/] · use /help"
        if command not in COMMAND_HELP:
            return "[red]Unknown command[/] · use /help"
        usage, description = COMMAND_HELP[command]
        context: list[str] = []
        fragment = value.rsplit(" ", 1)[-1] if " " in value else ""
        if command == "/ingest" and " " in value:
            context = _path_suggestions(fragment, limit=3)
        elif command in MEMORY_ID_COMMANDS and _expects_memory_id(value, command):
            context = [memory_id for memory_id in self._memory_ids() if memory_id.startswith(fragment)][:3]
        suffix = f" · [cyan]{'   '.join(context)}[/]" if context else ""
        return f"[bold]{usage}[/] [dim]— {description}[/]{suffix}"

    def _memory_ids(self) -> list[str]:
        page = self.engine.list(namespace=self.namespace, limit=500, include_retracted=True)
        return [item.id for item in reversed(page.items) if not item.text.startswith("[REMOVED]")]

    def _slash(self, value: str) -> None:
        parts = shlex.split(value)
        command = parts[0].lower()
        args = parts[1:]
        activity = self.query_one("#activity", RichLog)
        log = activity
        if command in {"/exit", "/quit"}:
            self.exit()
        elif command == "/clear":
            log.clear()
        elif command == "/help":
            log.write(
                "[bold]Query[/] plain text\n"
                "/ingest SOURCE...  /sources  /search QUERY  /timeline  /history ID\n"
                "/correct ID TEXT  /forget ID REASON  /remove ID... --yes  /jobs\n"
                "/namespace [PROJECT]  /model [completion|embedding]  /check  /config  /clear  /exit"
            )
        elif command == "/ingest":
            if not args:
                raise ValueError("usage: /ingest SOURCE...")
            self.query_one(TabbedContent).active = "sources-tab"
            sources = [_source_from_argument(item) for item in args]
            self._start_operation(
                f"Reading and preprocessing {len(sources)} source{'s' if len(sources) != 1 else ''}"
            )
            self._run_ingestion(sources)
        elif command == "/sources":
            self._start_operation("Loading retained sources")
            self._run_command(
                lambda: self.engine.list_sources(namespace=self.namespace),
                self._sources_finished,
            )
        elif command == "/search":
            if not args:
                raise ValueError("usage: /search QUERY")
            self._start_operation("Filtering private search data")
            self._run_search(" ".join(args))
        elif command in {"/timeline", "/jobs"}:
            if command == "/timeline":
                self._start_operation("Loading Trace timeline")
                self._run_command(
                    lambda: self.engine.list(namespace=self.namespace),
                    self._timeline_finished,
                )
            else:
                self._start_operation("Loading durable jobs")
                self._run_command(self.engine.list_jobs, self._jobs_finished)
        elif command == "/history":
            if not args:
                raise ValueError("usage: /history ID")
            self._start_operation("Loading memory history")
            self._run_command(
                lambda: self.engine.history(args[0], namespace=self.namespace),
                self._history_finished,
            )
        elif command == "/correct":
            if len(args) < 2:
                raise ValueError("usage: /correct ID TEXT")
            self._start_operation("Writing memory correction")
            self._run_command(
                lambda: self.engine.correct(
                    delta_id=args[0],
                    text=" ".join(args[1:]),
                    namespace=self.namespace,
                ),
                self._memory_operation_finished,
            )
        elif command == "/forget":
            if len(args) < 2:
                raise ValueError("usage: /forget ID REASON")
            self._start_operation("Retracting memory")
            self._run_command(
                lambda: self.engine.forget(
                    delta_id=args[0],
                    reason=" ".join(args[1:]),
                    namespace=self.namespace,
                ),
                self._memory_operation_finished,
            )
        elif command == "/remove":
            if "--yes" not in args:
                raise ValueError("physical removal cannot be undone; use /remove ID... --yes")
            delta_ids = [item for item in args if item != "--yes"]
            if not delta_ids:
                raise ValueError("usage: /remove ID... --yes")
            self._start_operation(f"Removing {len(delta_ids)} memory record(s)")
            self._run_command(
                lambda: self.engine.remove(
                    delta_ids=delta_ids,
                    reason="interactive removal",
                    namespace=self.namespace,
                ),
                self._memory_operation_finished,
            )
        elif command == "/namespace":
            if args:
                self.namespace = MemoryNamespace(project_id=args[0])
            log.write(str(self.namespace.model_dump(exclude_none=True)))
            self._update_config_view()
            self._refresh_panels()
        elif command == "/model":
            role = ProviderRole(args[0]) if args else ProviderRole.COMPLETION
            self.push_screen(ModelSelectorScreen(self.engine, role), self._model_updated)
        elif command == "/check":
            self._start_operation("Checking installation and memory store")
            self._run_command(self.engine.check, self._check_finished)
        elif command == "/config":
            self.query_one(TabbedContent).active = "config-tab"
            self._update_config_view()
            activity.write("[green]Configuration opened.[/]")
        else:
            raise ValueError(f"unknown command: {command}; use /help")

    def _refresh_status(self) -> None:
        storage_ready = self.engine.store.is_ready()
        configuration = self.engine.get_model_configuration_status()
        provider = configuration.configuration.completion.provider
        pending_jobs = len(self.engine.list_jobs(status="pending"))
        pending_runs = sum(run.status in {"pending", "running"} for run in self.engine.list_ingestion_runs(namespace=self.namespace))
        self.query_one("#store", Static).update(f"Store\n{self.path}")
        self.query_one("#provider", Static).update(f"Provider\n{provider}")
        self.query_one("#storage-status", Static).update(
            f"Store\n{'ready' if storage_ready else 'UNAVAILABLE'}"
        )
        self.query_one("#pending", Static).update(f"Pending\n{pending_jobs} jobs · {pending_runs} runs")

    def _update_config_view(self) -> None:
        status = self.engine.get_model_configuration_status()
        configuration = status.configuration
        retrieval = self.engine.get_retrieval_configuration()
        self.query_one("#config-view", Static).update(
            "[bold]Active configuration[/]\n\n"
            f"Store: {self.path}\n"
            f"Namespace: {self.namespace.model_dump(exclude_none=True)}\n"
            f"Completion: {configuration.completion.provider} / {configuration.completion.model or 'none'}\n"
            f"Embedding: {configuration.embedding.provider} / {configuration.embedding.model}\n"
            f"Revision: {configuration.revision}\n"
            f"State: {status.status}\n\n"
            f"Retrieval: {retrieval.retrieval_profile} · {', '.join(retrieval.enabled_routes)}\n"
            f"Context: {retrieval.max_context_items} items · {retrieval.max_context_tokens} tokens "
            f"· {retrieval.per_source_context_tokens} per source\n"
            f"Route overrides: {retrieval.route_weights or 'none'}\n\n"
            "Use /model to change providers or models.\n"
            "Retained originals are permission-restricted but not encrypted."
        )

    def _model_updated(self, changed: bool | None) -> None:
        if changed:
            self._refresh_status()
            self._update_config_view()
            self.query_one("#activity", RichLog).write("[green]Model configuration updated.[/]")


def _source_from_argument(value: str) -> SourceInput:
    if value.startswith("https://"):
        kind = "git" if value.endswith(".git") or "github.com/" in value else "url"
        return SourceInput(kind=kind, url=value)
    path = Path(value).expanduser()
    return SourceInput(kind="directory" if path.is_dir() else "file", path=str(path))


def run_terminal(path: Path, namespace: MemoryNamespace) -> None:
    MemoryTerminal(path, namespace).run()
