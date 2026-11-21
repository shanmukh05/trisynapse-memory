"""Trisynapse Native adapter for DolphinBench's external runner."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from trisynapse_memory import __version__
from trisynapse_memory.benchmarks.dolphinbench.batch import (
    BatchProjector,
    CostLedger,
    ProjectionQueue,
    projection_deltas,
    response_output_text,
    token_usage,
)
from trisynapse_memory.benchmarks.dolphinbench.download import MANIFEST_SHA256, RELEASE_COMMIT
from trisynapse_memory.engine import MemoryEngine
from trisynapse_memory.engine.models import MemoryNamespace
from trisynapse_memory.engine.providers.registry import ProviderSettings
from trisynapse_memory.engine.retrieval.engine import RetrieverConfig


SYSTEM_PROMPT = """You are the personal agent for {persona} in a simulated environment.
Use memory_search when prior preferences, commitments, dates, relationships, or history may
matter. Use the supplied application tools to perform requested actions. Treat tool results as
authoritative, do not fabricate successful actions, and give the user a concise final response.
The current simulated time is {narrative_time}."""


@dataclass
class InteractionRecordValue:
    settings: dict[str, Any]
    messages: list[dict[str, Any]]
    duration_ms: float | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    app_calls: list[dict[str, Any]] | None = None


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return str(value)


PERSONA_ORDER = ("morgan", "alex", "riley")
PROGRESS_BAR_WIDTH = 20

# The pinned mock server calls datetime.utcnow() and FastMCP logs every request
# at INFO. Editing that release changes DolphinBench's code fingerprint, so the
# child is launched with warnings ignored and its root log level raised.
_QUIET_MCP_LAUNCH = """
import logging
import runpy
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
_basic_config = logging.basicConfig

def basicConfig(*args, **kwargs):
    kwargs["level"] = logging.WARNING
    return _basic_config(*args, **kwargs)

logging.basicConfig = basicConfig
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
"""


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def quiet_mcp_launch_args(server_args: list[str]) -> list[str]:
    """Launch the mock server without deprecation or per-request info logs."""

    return ["-W", "ignore::DeprecationWarning", "-c", _QUIET_MCP_LAUNCH, *server_args]


def find_release_manifest(work_dir: Path) -> Path:
    """Return the release manifest that sits beside this run directory."""

    for parent in work_dir.parents:
        candidate = parent / "release" / "manifest.json"
        if candidate.is_file():
            return candidate
    raise ValueError("DolphinBench release manifest was not found beside the run directory")


def load_persona_totals(manifest_path: Path) -> dict[str, dict[str, int]]:
    """Read pinned session and test counts for morgan, alex, and riley."""

    sources = json.loads(manifest_path.read_text(encoding="utf-8"))["sources"]
    return {
        persona: {
            "sessions": int(sources[persona]["sessions"]),
            "tests": int(sources[persona]["tests"]),
        }
        for persona in PERSONA_ORDER
    }


def overall_position(
    phase: str,
    persona: str,
    item_id: str,
    totals: dict[str, dict[str, int]],
) -> tuple[int, int, int, int]:
    """Return persona index, persona total, overall index, and overall total."""

    key = "sessions" if phase == "ingestion" else "tests"
    offset = 0
    for name in PERSONA_ORDER:
        if name == persona:
            break
        offset += totals[name][key]
    persona_index = int(item_id)
    persona_total = totals[persona][key]
    overall_total = sum(totals[name][key] for name in PERSONA_ORDER)
    return persona_index, persona_total, offset + persona_index, overall_total


def completed_durations_ms(work_dir: Path, phase: str) -> list[float]:
    """Collect wall-clock durations already stored for this phase."""

    root = work_dir / phase
    if not root.is_dir():
        return []
    durations: list[float] = []
    for path in sorted(root.glob("*/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            durations.append(float(payload["row"]["duration_ms"]))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return durations


def format_duration(seconds: float) -> str:
    """Format a non-negative duration for a terminal progress line."""

    whole = max(0, int(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_dolphinbench_progress(
    *,
    phase: str,
    persona: str,
    persona_index: int,
    persona_total: int,
    overall_index: int,
    overall_total: int,
    elapsed_s: float,
    eta_s: float | None,
    item_s: float | None,
) -> str:
    """Build one ingest or evaluate progress line, including an ETA when known."""

    label = "ingest" if phase == "ingestion" else "evaluate"
    filled = 0 if overall_total == 0 else min(
        PROGRESS_BAR_WIDTH, round(PROGRESS_BAR_WIDTH * overall_index / overall_total)
    )
    bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    eta = "unknown" if eta_s is None else format_duration(eta_s)
    status = "running" if item_s is None else f"done {format_duration(item_s)}"
    return (
        f"{label} [{bar}] {overall_index}/{overall_total} "
        f"{persona} {persona_index}/{persona_total} {status} "
        f"elapsed {format_duration(elapsed_s)} eta {eta}"
    )


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _work_path(root: Path, value: Any, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a relative path inside the run output")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{label} escapes the run output")
    return path


class TrisynapseDolphinBenchHarness:
    """GPT-6 Luna tool harness with Trisynapse memory and Batch projection."""

    app_python = sys.executable

    def __init__(self, options: dict[str, Any], work_dir: str | Path) -> None:
        options = _require_mapping(options, "options")
        model = _require_mapping(options.get("model"), "model")
        memory = _require_mapping(options.get("memory"), "memory")
        batch = _require_mapping(options.get("batch"), "batch")
        release = _require_mapping(options.get("release"), "release")
        paths = _require_mapping(options.get("paths"), "paths")
        retrieval = _require_mapping(memory.get("retrieval"), "memory.retrieval")
        self.options = options
        self.work_dir = Path(work_dir).resolve()
        self.model = str(model.get("id", "gpt-6-luna"))
        self.reasoning_effort = str(model.get("reasoning_effort", "high"))
        self.service_tier = str(model.get("service_tier", "default"))
        self.max_output_tokens = int(model.get("max_output_tokens", 4096))
        self.max_turns = int(model.get("max_turns", 60))
        self.live_max_retries = int(model.get("max_retries", 2))
        self.live_retry_base_seconds = float(model.get("retry_base_seconds", 1))
        self.api_key_env = str(model.get("api_key_env", "OPENAI_API_KEY"))
        self.base_url = model.get("base_url")
        self.local_files_only = bool(memory.get("local_files_only", False))
        self.retrieval = RetrieverConfig(
            top_k=int(retrieval.get("top_k", 12)),
            max_context_items=int(retrieval.get("max_context_items", 24)),
            max_context_tokens=int(retrieval.get("max_context_tokens", 6000)),
        )
        self.batch_options = {
            "reasoning_effort": str(batch.get("reasoning_effort", self.reasoning_effort)),
            "max_output_tokens": int(batch.get("max_output_tokens", 4096)),
            "completion_window": str(batch.get("completion_window", "24h")),
            "poll_interval_seconds": float(batch.get("poll_interval_seconds", 30)),
            "max_retries": int(batch.get("max_retries", 2)),
        }
        self.store_root = _work_path(
            self.work_dir, paths.get("persona_stores", "trisynapse/stores"), "paths.persona_stores"
        )
        self.batch_root = _work_path(
            self.work_dir, paths.get("batch_state", "trisynapse/batch"), "paths.batch_state"
        )
        ledger_path = _work_path(
            self.work_dir,
            paths.get("cost_ledger", "trisynapse/cost-ledger.jsonl"),
            "paths.cost_ledger",
        )
        if self.model != "gpt-6-luna":
            raise ValueError("DolphinBench integration is pinned to gpt-6-luna")
        if model.get("endpoint", "/v1/responses") != "/v1/responses":
            raise ValueError("DolphinBench interactive endpoint must be /v1/responses")
        if self.reasoning_effort != "high":
            raise ValueError("DolphinBench integration is pinned to high reasoning")
        if self.service_tier != "default":
            raise ValueError("DolphinBench interactive calls are pinned to standard processing")
        if self.batch_options["completion_window"] != "24h":
            raise ValueError("OpenAI Batch completion_window must be 24h")
        if batch.get("endpoint", "/v1/responses") != "/v1/responses":
            raise ValueError("DolphinBench Batch endpoint must be /v1/responses")
        if batch.get("one_job_per_persona", True) is not True:
            raise ValueError("DolphinBench requires one projection Batch job per persona")
        if release and (
            release.get("commit") != RELEASE_COMMIT
            or release.get("manifest_sha256") != MANIFEST_SHA256
        ):
            raise ValueError("DolphinBench release pin differs from the downloader")
        if self.max_output_tokens < 1 or self.max_turns < 1:
            raise ValueError("model token and turn limits must be positive")
        if self.live_max_retries < 0 or self.live_retry_base_seconds < 0:
            raise ValueError("model retry settings cannot be negative")
        if self.batch_options["max_output_tokens"] < 1:
            raise ValueError("batch max_output_tokens must be positive")
        if self.batch_options["poll_interval_seconds"] < 0:
            raise ValueError("batch poll_interval_seconds cannot be negative")
        if self.batch_options["max_retries"] < 0:
            raise ValueError("batch max_retries cannot be negative")
        self._client: Any | None = None
        self._engines: dict[str, MemoryEngine] = {}
        self._queues: dict[str, ProjectionQueue] = {}
        self._persona_totals: dict[str, dict[str, int]] | None = None
        self._durations_ms: dict[str, list[float]] | None = None
        self.ledger = CostLedger(ledger_path)

    def identity(self) -> dict[str, Any]:
        """Return public, credential-free harness identity without model calls."""

        package = Path(__file__).resolve().parent
        return {
            "run_name": "Trisynapse Memory · GPT-6 Luna High · v0.2.0",
            "harness": {"id": "trisynapse-native", "name": "Trisynapse Native"},
            "model": {
                "provider": "OpenAI",
                "id": self.model,
                "display_name": "GPT-6 Luna (high)",
                "endpoint": "/v1/responses",
                "reasoning_effort": self.reasoning_effort,
                "service_tier": self.service_tier,
                "max_turns": self.max_turns,
                "max_retries": self.live_max_retries,
            },
            "memory": {
                "id": "trisynapse-memory",
                "name": "Trisynapse Memory",
                "version": __version__,
                "embedding": "sentence-transformers/all-MiniLM-L6-v2",
                "retrieval": {
                    "top_k": self.retrieval.top_k,
                    "max_context_items": self.retrieval.max_context_items,
                    "max_context_tokens": self.retrieval.max_context_tokens,
                },
            },
            "batch": {
                "endpoint": "/v1/responses",
                "completion_window": self.batch_options["completion_window"],
                "one_job_per_persona": True,
                "max_retries": self.batch_options["max_retries"],
            },
            "grader": {
                "provider": "OpenAI",
                "model": "gpt-5.6-sol",
                "reasoning_effort": "medium",
                "endpoint": "/v1/chat/completions",
            },
            "paths": {
                "persona_stores": self.store_root.relative_to(self.work_dir).as_posix(),
                "batch_state": self.batch_root.relative_to(self.work_dir).as_posix(),
            },
            "dependencies": {
                "openai": _version("openai"),
                "mcp": _version("mcp"),
                "trisynapse-memory": __version__,
            },
            "code": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(package.glob("*.py"))
            },
        }

    def run_interaction(self, request: Any) -> InteractionRecordValue:
        self._emit_progress(request, finished_ms=None)
        started = time.monotonic()
        result = asyncio.run(self._run_interaction(request))
        self._emit_progress(request, finished_ms=(time.monotonic() - started) * 1000)
        return result

    def _ensure_totals(self) -> dict[str, dict[str, int]]:
        if self._persona_totals is None:
            self._persona_totals = load_persona_totals(find_release_manifest(self.work_dir))
        return self._persona_totals

    def _ensure_durations(self, phase: str) -> list[float]:
        if self._durations_ms is None:
            self._durations_ms = {
                "ingestion": completed_durations_ms(self.work_dir, "ingestion"),
                "tests": completed_durations_ms(self.work_dir, "tests"),
            }
        return self._durations_ms[phase]

    def _emit_progress(self, request: Any, *, finished_ms: float | None) -> None:
        totals = self._ensure_totals()
        persona_index, persona_total, overall_index, overall_total = overall_position(
            request.phase, request.persona, request.interaction_id, totals
        )
        samples = self._ensure_durations(request.phase)
        if finished_ms is None:
            completed = len(samples)
            spent_ms = sum(samples)
            remaining = overall_total - overall_index + 1
            item_s = None
        else:
            completed = len(samples) + 1
            spent_ms = sum(samples) + finished_ms
            remaining = overall_total - overall_index
            item_s = finished_ms / 1000
            samples.append(finished_ms)
        eta_s = None if completed == 0 else (spent_ms / completed / 1000) * remaining
        print(
            format_dolphinbench_progress(
                phase=request.phase,
                persona=request.persona,
                persona_index=persona_index,
                persona_total=persona_total,
                overall_index=overall_index,
                overall_total=overall_total,
                elapsed_s=spent_ms / 1000,
                eta_s=eta_s,
                item_s=item_s,
            ),
            file=sys.stderr,
            flush=True,
        )

    async def _run_interaction(self, request: Any) -> InteractionRecordValue:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        parameters = StdioServerParameters(
            command=request.apps["command"],
            args=quiet_mcp_launch_args(list(request.apps["args"])),
            env=request.apps["env"],
        )
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as apps:
                await apps.initialize()
                tools = (await apps.list_tools()).tools
                return await self.run_agent(request, tools, apps.call_tool)

    async def run_agent(
        self,
        request: Any,
        tools: list[Any],
        call_app: Callable[[str, dict[str, Any]], Awaitable[Any]],
    ) -> InteractionRecordValue:
        """Run one fresh Responses conversation and route its tool calls."""

        system_prompt = SYSTEM_PROMPT.format(
            persona=request.persona, narrative_time=request.narrative_time
        )
        openai_tools = [self._openai_tool(tool) for tool in tools]
        openai_tools.append(self._memory_tool())
        settings = {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
            "parallel_tool_calls": False,
            "tools": openai_tools,
        }
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.dated_message},
        ]
        response_input: Any = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.dated_message},
        ]
        previous_response_id: str | None = None
        final_text = ""
        attempts: list[dict[str, Any]] = []
        for turn in range(self.max_turns):
            response, elapsed_ms = self._create_response(
                turn,
                attempts,
                {
                    "model": self.model,
                    "reasoning": {"effort": self.reasoning_effort},
                    "input": response_input,
                    "tools": openai_tools,
                    "parallel_tool_calls": False,
                    "max_output_tokens": self.max_output_tokens,
                    "service_tier": self.service_tier,
                    **(
                        {"previous_response_id": previous_response_id}
                        if previous_response_id
                        else {}
                    ),
                },
            )
            usage = token_usage(response)
            response_id = str(_field(response, "id"))
            self.ledger.record(
                f"response:{request.phase}:{request.persona}:{request.interaction_id}:{response_id}",
                phase=request.phase,
                usage=usage,
                batch=False,
                model=self.model,
            )
            calls = self._function_calls(response)
            text = response_output_text(response)
            assistant: dict[str, Any] = {
                "role": "assistant",
                "usage": usage.as_submission_usage(),
                "duration_ms": elapsed_ms,
            }
            if text:
                assistant["content"] = text
            if calls:
                assistant["tool_calls"] = calls
            if not text and not calls:
                raise RuntimeError("GPT-6 Luna returned neither text nor tool calls")
            messages.append(assistant)
            if not calls:
                final_text = text
                break
            outputs = []
            for call in calls:
                tool_started = time.monotonic()
                if call["name"] == "memory_search":
                    result = self._memory_search(request.persona, call["arguments"])
                else:
                    result = _jsonable(await call_app(call["name"], call["arguments"]))
                tool_elapsed_ms = (time.monotonic() - tool_started) * 1000
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": result,
                        "duration_ms": tool_elapsed_ms,
                    }
                )
                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["id"],
                        "output": json.dumps(result, ensure_ascii=False),
                    }
                )
            previous_response_id = response_id
            response_input = outputs
        else:
            raise RuntimeError(f"agent exceeded max_turns={self.max_turns}")
        if request.phase == "ingestion":
            self._capture_history(request, final_text, messages)
        return InteractionRecordValue(settings=settings, messages=messages, attempts=attempts)

    def freeze(self, persona: str) -> dict[str, Any]:
        """Complete the persona's Batch projection and return a content checkpoint."""

        print(
            f"ingest {persona} history complete; submitting memory projection batch "
            "(this step can take hours)",
            file=sys.stderr,
            flush=True,
        )
        queue = self._projection_queue(persona)
        projector = BatchProjector(
            self._client_instance(),
            self.batch_root,
            self.ledger,
            model=self.model,
            **self.batch_options,
        )
        engine = self._engine(persona)

        def apply(item: dict[str, Any], payload: dict[str, Any]) -> None:
            namespace = self._namespace(persona)
            engine.append_deltas(
                {**delta, "namespace": namespace} for delta in projection_deltas(item, payload)
            )

        projector.run(persona, queue, apply)
        group_ids = [item["group_id"] for item in queue.items().values()]
        if group_ids:
            engine.build_group_summary(sorted(set(group_ids)), namespace=self._namespace(persona))
        checkpoint = self._checkpoint(persona, engine)
        self._close_engine(persona)
        return checkpoint

    def verify_checkpoint(self, persona: str, checkpoint: dict[str, Any]) -> None:
        actual = self._checkpoint(persona, self._engine(persona))
        if actual != checkpoint:
            raise ValueError(f"DolphinBench checkpoint changed for {persona}")

    def total_cost_usd(self, phase: str) -> float:
        if phase not in {"ingestion", "tests"}:
            raise ValueError("phase must be ingestion or tests")
        return self.ledger.total(phase)

    def _client_instance(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing {self.api_key_env}")
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 0}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def _create_response(
        self,
        turn: int,
        interaction_attempts: list[dict[str, Any]],
        request: dict[str, Any],
    ) -> tuple[Any, float]:
        """Run one Responses call with explicit, exportable transport retries."""

        attempts: list[dict[str, Any]] = []
        total_ms = 0.0
        for number in range(1, self.live_max_retries + 2):
            started = time.monotonic()
            try:
                response = self._client_instance().responses.create(**request)
            except Exception as exc:
                elapsed_ms = (time.monotonic() - started) * 1000
                total_ms += elapsed_ms
                retryable = self._retryable_response_error(exc)
                attempts.append(
                    {
                        "turn": turn + 1,
                        "attempt": number,
                        "ok": False,
                        "latency_seconds": elapsed_ms / 1000,
                        "retryable": retryable,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                    }
                )
                if not retryable or number > self.live_max_retries:
                    interaction_attempts.extend(attempts)
                    raise
                time.sleep(min(self.live_retry_base_seconds * (2 ** (number - 1)), 8.0))
                continue
            elapsed_ms = (time.monotonic() - started) * 1000
            total_ms += elapsed_ms
            if attempts:
                attempts.append(
                    {
                        "turn": turn + 1,
                        "attempt": number,
                        "ok": True,
                        "latency_seconds": elapsed_ms / 1000,
                        "response_id": str(_field(response, "id")),
                    }
                )
                interaction_attempts.extend(attempts)
            return response, total_ms
        raise AssertionError("unreachable response retry loop")

    @staticmethod
    def _retryable_response_error(error: Exception) -> bool:
        status = getattr(error, "status_code", None)
        if status in {408, 409, 429, 500, 502, 503, 504}:
            return True
        name = type(error).__name__.lower()
        return isinstance(error, (TimeoutError, ConnectionError)) or any(
            marker in name for marker in ("timeout", "connection", "ratelimit", "internalserver")
        )

    def _namespace(self, persona: str) -> MemoryNamespace:
        return MemoryNamespace(
            user_id=persona,
            agent_id="trisynapse-native",
            project_id="dolphinbench",
        )

    def _engine(self, persona: str) -> MemoryEngine:
        if persona not in self._engines:
            path = self.store_root / persona
            engine = MemoryEngine.open(
                path,
                namespace=self._namespace(persona),
                completion_provider=ProviderSettings(provider="none"),
                local_files_only=self.local_files_only,
                auto_process=False,
            )
            configuration = engine.get_retrieval_configuration()
            requested = {
                "default_top_k": self.retrieval.top_k,
                "max_context_items": self.retrieval.max_context_items,
                "max_context_tokens": self.retrieval.max_context_tokens,
            }
            if any(getattr(configuration, name) != value for name, value in requested.items()):
                if engine.store.list_records(namespace=self._namespace(persona)):
                    engine.close()
                    raise ValueError(f"retrieval configuration changed for {persona}")
                engine.set_retrieval_configuration(
                    {**configuration.model_dump(mode="python"), **requested},
                    expected_revision=configuration.revision,
                )
            self._engines[persona] = engine
        return self._engines[persona]

    def _close_engine(self, persona: str) -> None:
        engine = self._engines.pop(persona, None)
        if engine is not None:
            engine.close()

    def _memory_search(self, persona: str, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return {"hits": [], "error": "query is required"}
        top_k = min(max(int(arguments.get("top_k", self.retrieval.top_k)), 1), 24)
        result = self._engine(persona).search(
            query,
            top_k=top_k,
            namespace=self._namespace(persona),
            persist=False,
        )
        return {
            "hits": [
                {
                    "id": hit.item_id,
                    "kind": hit.kind,
                    "text": hit.text,
                    "score": hit.score,
                    "observed_at": hit.observed_at.isoformat() if hit.observed_at else None,
                    "source_record_ids": hit.source_record_ids,
                }
                for hit in result.hits
            ]
        }

    def _capture_history(
        self,
        request: Any,
        assistant_text: str,
        transcript: list[dict[str, Any]] | None = None,
    ) -> None:
        engine = self._engine(request.persona)
        namespace = self._namespace(request.persona)
        group_id = f"dolphinbench:{request.persona}:{request.interaction_id}"
        scope = {"benchmark": "dolphinbench", "persona": request.persona}
        user = engine.ingest_raw_source(
            request.dated_message,
            group_id=group_id,
            observed_at=request.narrative_time,
            external_key=f"{group_id}:user",
            source_ref={"type": "dolphinbench", "id": request.interaction_id},
            locator={"kind": "interaction", "role": "user"},
            retrieval_fields={"speaker": "user", "message": request.message},
            modality="conversation",
            source_type="conversation",
            scope=scope,
            namespace=namespace,
            schedule=False,
        )
        assistant = engine.ingest_raw_source(
            "assistant: " + assistant_text,
            group_id=group_id,
            observed_at=request.narrative_time,
            external_key=f"{group_id}:assistant",
            source_ref={"type": "dolphinbench", "id": request.interaction_id},
            locator={"kind": "interaction", "role": "assistant"},
            retrieval_fields={"speaker": "assistant", "message": assistant_text},
            modality="conversation",
            source_type="conversation",
            scope=scope,
            namespace=namespace,
            schedule=False,
        )
        projection_messages = (
            [
                {
                    key: _jsonable(message[key])
                    for key in ("role", "content", "tool_calls", "tool_call_id")
                    if key in message
                }
                for message in transcript
                if message.get("role") in {"user", "assistant", "tool"}
            ]
            if transcript is not None
            else [
                {"role": "user", "content": request.dated_message},
                {"role": "assistant", "content": assistant_text},
            ]
        )
        self._projection_queue(request.persona).enqueue(
            {
                "custom_id": f"{request.persona}:{request.interaction_id}",
                "persona": request.persona,
                "interaction_id": request.interaction_id,
                "group_id": group_id,
                "narrative_time": request.narrative_time,
                "messages": projection_messages,
                "raw_record_ids": [user.id, assistant.id],
            }
        )

    def _projection_queue(self, persona: str) -> ProjectionQueue:
        if persona not in self._queues:
            self._queues[persona] = ProjectionQueue(
                self.batch_root / persona / "projections.jsonl"
            )
        return self._queues[persona]

    def _checkpoint(self, persona: str, engine: MemoryEngine) -> dict[str, Any]:
        validation = engine.validate_store(check_source_blobs=False)
        if not validation.ok:
            raise RuntimeError("invalid Trisynapse store: " + "; ".join(validation.issues))
        export = engine.export(namespace=self._namespace(persona))
        summaries = [
            value.model_dump(mode="json")
            for value in engine.store.group_summary_views(namespace=self._namespace(persona))
        ]
        summaries.sort(key=lambda value: value["group_id"])
        canonical = json.dumps(
            {"export": export, "group_summaries": summaries},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        deltas = export["deltas"]
        return {
            "schema_version": 1,
            "persona": persona,
            "sha256": hashlib.sha256(canonical).hexdigest(),
            "record_count": len(deltas),
            "group_summary_count": len(summaries),
            "max_seq": max((int(item["seq"]) for item in deltas), default=0),
        }

    @staticmethod
    def _memory_tool() -> dict[str, Any]:
        return {
            "type": "function",
            "name": "memory_search",
            "description": "Search the person's read-only long-term memory for relevant history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 24},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": False,
        }

    @staticmethod
    def _openai_tool(tool: Any) -> dict[str, Any]:
        schema = _field(tool, "inputSchema", _field(tool, "input_schema", {})) or {}
        return {
            "type": "function",
            "name": str(_field(tool, "name")),
            "description": str(_field(tool, "description", "") or ""),
            "parameters": _jsonable(schema),
            "strict": False,
        }

    @staticmethod
    def _function_calls(response: Any) -> list[dict[str, Any]]:
        calls = []
        for item in _field(response, "output", []) or []:
            if _field(item, "type") != "function_call":
                continue
            raw = _field(item, "arguments", "{}") or "{}"
            arguments = json.loads(raw) if isinstance(raw, str) else _jsonable(raw)
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be an object")
            calls.append(
                {
                    "id": str(_field(item, "call_id", _field(item, "id"))),
                    "name": str(_field(item, "name")),
                    "arguments": arguments,
                }
            )
        return calls
