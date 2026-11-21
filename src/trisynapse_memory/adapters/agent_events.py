"""Generic coding-agent lifecycle events for automatic capture and context injection.

Owns the event schema and the capture path that turns a hook payload into a
raw source (or a compiled profile). Durable text is chosen by event type;
``MemoryEngine`` still owns persistence. Session-start and pre-compact events
are read-only context lookups and must not write the log.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from trisynapse_memory.engine import MemoryEngine
from trisynapse_memory.engine.models import MemoryNamespace


class AgentEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal[
        "session_start", "user_prompt", "pre_tool", "post_tool", "post_tool_failure",
        "pre_compact", "subagent_start", "subagent_stop", "stop", "session_end",
    ]
    session_id: str
    agent_id: str
    project_id: str = "default"
    user_id: str | None = None
    tool_name: str | None = None
    content: str | None = None
    input: dict[str, Any] | None = None
    output: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def capture_agent_event(engine: MemoryEngine, event: AgentEvent | dict[str, Any]) -> dict[str, Any]:
    value = event if isinstance(event, AgentEvent) else AgentEvent.model_validate(event)
    namespace = MemoryNamespace(
        user_id=value.user_id,
        agent_id=value.agent_id,
        project_id=value.project_id,
        session_id=value.session_id,
    )
    group_id = f"agent:{value.agent_id}:{value.session_id}"
    if value.type in {"session_start", "pre_compact"}:
        query = value.content or "current project decisions, constraints, preferences, failures, and next steps"
        profile = engine.compile_profile(namespace=namespace, query=query)
        return {"captured": False, "context": profile, "namespace": namespace.model_dump(mode="json")}

    text = _event_text(value)
    if not text:
        return {"captured": False, "reason": "event contains no durable content"}
    delta = engine.ingest_raw_source(
        text,
        group_id=group_id,
        source_ref={"type": "agent_event", "agent_id": value.agent_id, "session_id": value.session_id},
        locator={"event_type": value.type, "tool_name": value.tool_name},
        scope={"event_type": value.type, **value.metadata},
        namespace=namespace,
        external_key=value.metadata.get("event_id"),
    )
    response: dict[str, Any] = {"captured": True, "record_id": delta.id}
    if value.type in {"stop", "session_end"}:
        engine.build_group_summary([group_id], namespace=namespace)
        response["context"] = engine.compile_profile(namespace=namespace)
    return response


def _event_text(event: AgentEvent) -> str:
    formatter = _EVENT_TEXT.get(event.type)
    if formatter is None:
        return event.content or ""
    return formatter(event)


def _user_prompt_text(event: AgentEvent) -> str:
    return f"User prompt: {event.content or ''}".strip()


def _pre_tool_text(event: AgentEvent) -> str:
    return f"Tool planned: {event.tool_name or 'unknown'} input={_bounded(event.input)}"


def _post_tool_text(event: AgentEvent) -> str:
    return _tool_result_text(event, "completed")


def _post_tool_failure_text(event: AgentEvent) -> str:
    return _tool_result_text(event, "failed")


def _tool_result_text(event: AgentEvent, status: str) -> str:
    return f"Tool {event.tool_name or 'unknown'} {status}. Output: {(event.output or '')[:8000]}".strip()


def _subagent_text(event: AgentEvent) -> str:
    return f"{event.type.replace('_', ' ')}: {event.content or _bounded(event.metadata)}"


def _session_end_text(event: AgentEvent) -> str:
    return f"Session ended: {event.content or 'No explicit summary supplied.'}"


_EVENT_TEXT = {
    "user_prompt": _user_prompt_text,
    "pre_tool": _pre_tool_text,
    "post_tool": _post_tool_text,
    "post_tool_failure": _post_tool_failure_text,
    "subagent_start": _subagent_text,
    "subagent_stop": _subagent_text,
    "stop": _session_end_text,
    "session_end": _session_end_text,
}


def _bounded(value: Any) -> str:
    text = str(value or "")
    return text[:8000]
