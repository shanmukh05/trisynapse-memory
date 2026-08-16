"""Interactive Textual terminal for Trisynapse Memory."""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.suggester import Suggester
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
    MemoryNamespace,
    ModelDescriptor,
    ProviderRole,
    ProviderSelection,
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
    "/check": ("/check", "Check installation, providers, and Trace integrity"),
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
        try:
            self.models = self.engine.list_models(
                self._role(), str(provider), refresh=refresh,
                base_url=self.query_one("#model-base-url", Input).value or None,
            )
            self._filter_models()
            message.update(f"{len(self.models)} models · selecting does not send an inference request.")
        except Exception as exc:
            self.models = []
            self.query_one("#model-choice", Select).set_options([])
            message.update(f"[red]{exc}[/]\nYou may still enter an exact model ID after configuring credentials.")

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
                result = self.engine.test_model_connection(role, selection)
                self.query_one("#model-message", Static).update(
                    f"{'[green]' if result.ok else '[red]'}{result.message}[/] "
                    "This test sends a potentially billable request."
                )
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
            result = self.engine.set_model_configuration(
                configuration,
                confirm_embedding_rebuild=changed_embedding,
                wait=changed_embedding,
            )
            if result.status == "rebuild_failed":
                self.query_one("#model-message", Static).update(f"[red]{result.message}[/]")
                return
            self.dismiss(True)
        except Exception as exc:
            self.query_one("#model-message", Static).update(f"[red]{exc}[/]")


def logo_for_width(width: int) -> str:
    """Return the generated source-faithful logo for the terminal width."""

    return WIDE_LOGO if width >= 72 else COMPACT_LOGO


class MemoryTerminal(App[None]):
    """Grounded query prompt plus operational slash commands."""

    TITLE = "Trisynapse Memory"
    SUB_TITLE = "Store traces. Recall meaning."
    CSS = """
    Screen { layout: vertical; background: $surface; }
    #brand { height: auto; padding: 1 2; color: $text; content-align: center middle; }
    #status { height: 3; padding: 0 2; background: $boost; }
    TabbedContent { height: 1fr; margin: 1 2; }
    .view-log { height: 1fr; border: round $primary; padding: 1; }
    #config-view { padding: 2; }
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

    def compose(self) -> ComposeResult:
        yield Static("", id="brand")
        with Horizontal(id="status"):
            yield Static(id="store", classes="status-cell")
            yield Static(id="provider", classes="status-cell")
            yield Static(id="integrity", classes="status-cell")
            yield Static(id="pending", classes="status-cell")
        with TabbedContent(initial="memory-tab"):
            with TabPane("Memory", id="memory-tab"):
                yield RichLog(id="activity", classes="view-log", wrap=True, markup=True)
            with TabPane("Sources / Import", id="sources-tab"):
                yield RichLog(id="sources-log", classes="view-log", wrap=True, markup=True)
            with TabPane("Trace", id="trace-tab"):
                yield RichLog(id="trace-log", classes="view-log", wrap=True, markup=True)
            with TabPane("Jobs / Runs", id="jobs-tab"):
                yield RichLog(id="jobs-log", classes="view-log", wrap=True, markup=True)
            with TabPane("Configuration", id="config-tab"):
                yield Static(id="config-view")
        yield Static("Type [bold]/[/] to see commands · [bold]Tab[/] or [bold]→[/] accepts a suggestion", id="recommendations")
        yield CompletionInput(
            placeholder="Ask memory, or type /help …",
            suggester=MemorySuggester(self._memory_ids),
            id="prompt",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#brand", Static).update(
            f"{logo_for_width(self.size.width)}\nTrisynapse Memory {self.engine.VERSION}  ·  Store traces. Recall meaning."
        )
        self._refresh_status()
        self.query_one("#activity", RichLog).write(
            "Ready. Plain text runs a grounded query. Use [bold]/ingest[/], [bold]/sources[/], or [bold]/help[/]."
        )
        self.query_one("#prompt", Input).focus()
        self._update_config_view()

    def on_resize(self) -> None:
        brand = self.query_one("#brand", Static)
        brand.update(f"{logo_for_width(self.size.width)}\nTrisynapse Memory {self.engine.VERSION}")

    def on_unmount(self) -> None:
        self.engine.close()

    def action_clear(self) -> None:
        for log in self.query(RichLog):
            log.clear()

    def action_interrupt(self) -> None:
        prompt = self.query_one("#prompt", Input)
        if prompt.value:
            prompt.value = ""
        else:
            self.exit()

    def on_input_changed(self, event: Input.Changed) -> None:
        self.query_one("#recommendations", Static).update(self._recommendations(event.value))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        event.input.value = ""
        if not value:
            return
        log = self.query_one("#activity", RichLog)
        self.query_one(TabbedContent).active = "memory-tab"
        log.write(f"[bold cyan]>[/] {value}")
        try:
            if value.startswith("/"):
                self._slash(value)
            else:
                result = self.engine.query(value, namespace=self.namespace)
                log.write(result.answer)
                for citation in result.citations:
                    location = f" · {citation.locator}" if citation.locator else ""
                    log.write(f"[dim][{citation.delta_id}]{location}[/]")
        except Exception as exc:
            log.write(f"[red]{exc}[/]")
        self._refresh_status()

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
        log = self.query_one("#activity", RichLog)
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
            log = self.query_one("#sources-log", RichLog)
            sources = [_source_from_argument(item) for item in args]
            log.write_json(json.dumps(self.engine.ingest_many(sources, namespace=self.namespace).model_dump(mode="json")))
        elif command == "/sources":
            self.query_one(TabbedContent).active = "sources-tab"
            log = self.query_one("#sources-log", RichLog)
            log.write_json(json.dumps({"sources": [item.model_dump(mode="json") for item in self.engine.list_sources(namespace=self.namespace)]}))
        elif command == "/search":
            if not args:
                raise ValueError("usage: /search QUERY")
            log.write_json(json.dumps(self.engine.search(" ".join(args), namespace=self.namespace).model_dump(mode="json")))
        elif command in {"/timeline", "/jobs"}:
            tab_id = "trace-tab" if command == "/timeline" else "jobs-tab"
            self.query_one(TabbedContent).active = tab_id
            log = self.query_one("#trace-log" if command == "/timeline" else "#jobs-log", RichLog)
            value = self.engine.list(namespace=self.namespace).model_dump(mode="json") if command == "/timeline" else {"jobs": [item.model_dump(mode="json") for item in self.engine.list_jobs()]}
            log.write_json(json.dumps(value))
        elif command == "/history":
            self.query_one(TabbedContent).active = "trace-tab"
            log = self.query_one("#trace-log", RichLog)
            log.write_json(json.dumps(self.engine.history(args[0], namespace=self.namespace).model_dump(mode="json")))
        elif command == "/correct":
            if len(args) < 2:
                raise ValueError("usage: /correct ID TEXT")
            log.write_json(json.dumps(self.engine.correct(delta_id=args[0], text=" ".join(args[1:]), namespace=self.namespace).model_dump(mode="json")))
        elif command == "/forget":
            if len(args) < 2:
                raise ValueError("usage: /forget ID REASON")
            log.write_json(json.dumps(self.engine.forget(delta_id=args[0], reason=" ".join(args[1:]), namespace=self.namespace).model_dump(mode="json")))
        elif command == "/remove":
            if "--yes" not in args:
                raise ValueError("physical removal cannot be undone; use /remove ID... --yes")
            delta_ids = [item for item in args if item != "--yes"]
            if not delta_ids:
                raise ValueError("usage: /remove ID... --yes")
            log.write_json(json.dumps(self.engine.remove(delta_ids=delta_ids, reason="interactive removal", namespace=self.namespace).model_dump(mode="json")))
        elif command == "/namespace":
            if args:
                self.namespace = MemoryNamespace(project_id=args[0])
            log.write(str(self.namespace.model_dump(exclude_none=True)))
            self._update_config_view()
        elif command == "/model":
            role = ProviderRole(args[0]) if args else ProviderRole.COMPLETION
            self.push_screen(ModelSelectorScreen(self.engine, role), self._model_updated)
        elif command == "/check":
            self.query_one(TabbedContent).active = "config-tab"
            self.query_one("#config-view", Static).update(
                "[bold]System check[/]\n\n" + json.dumps(self.engine.check(), indent=2, default=str)
            )
        elif command == "/config":
            self.query_one(TabbedContent).active = "config-tab"
            self._update_config_view()
        else:
            raise ValueError(f"unknown command: {command}; use /help")

    def _refresh_status(self) -> None:
        verification = self.engine.verify_trace()
        configuration = self.engine.get_model_configuration_status()
        provider = configuration.configuration.completion.provider
        pending_jobs = len(self.engine.list_jobs(status="pending"))
        pending_runs = sum(run.status in {"pending", "running"} for run in self.engine.list_ingestion_runs(namespace=self.namespace))
        self.query_one("#store", Static).update(f"Store\n{self.path}")
        self.query_one("#provider", Static).update(f"Provider\n{provider}")
        self.query_one("#integrity", Static).update(f"Trace\n{'valid' if verification.valid else 'BROKEN'}")
        self.query_one("#pending", Static).update(f"Pending\n{pending_jobs} jobs · {pending_runs} runs")

    def _update_config_view(self) -> None:
        status = self.engine.get_model_configuration_status()
        configuration = status.configuration
        self.query_one("#config-view", Static).update(
            "[bold]Active configuration[/]\n\n"
            f"Store: {self.path}\n"
            f"Namespace: {self.namespace.model_dump(exclude_none=True)}\n"
            f"Completion: {configuration.completion.provider} / {configuration.completion.model or 'none'}\n"
            f"Embedding: {configuration.embedding.provider} / {configuration.embedding.model}\n"
            f"Revision: {configuration.revision}\n"
            f"State: {status.status}\n\n"
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
