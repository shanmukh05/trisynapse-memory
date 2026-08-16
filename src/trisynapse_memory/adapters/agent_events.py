"""Generic coding-agent lifecycle events for automatic capture and context injection."""

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
    episode_id = f"agent:{value.agent_id}:{value.session_id}"
    if value.type in {"session_start", "pre_compact"}:
        query = value.content or "current project decisions, constraints, preferences, failures, and next steps"
        profile = engine.compile_profile(namespace=namespace, query=query)
        return {"captured": False, "context": profile, "namespace": namespace.model_dump(mode="json")}

    text = _event_text(value)
    if not text:
        return {"captured": False, "reason": "event contains no durable content"}
    delta = engine.ingest_observation(
        text,
        episode_id=episode_id,
        source_ref={"type": "agent_event", "agent_id": value.agent_id, "session_id": value.session_id},
        locator={"event_type": value.type, "tool_name": value.tool_name},
        scope={"event_type": value.type, **value.metadata},
        namespace=namespace,
        external_key=value.metadata.get("event_id"),
    )
    response: dict[str, Any] = {"captured": True, "delta_id": delta.id}
    if value.type in {"stop", "session_end"}:
        engine.build_episode_recall([episode_id], namespace=namespace)
        response["context"] = engine.compile_profile(namespace=namespace)
    return response


def _event_text(event: AgentEvent) -> str:
    if event.type == "user_prompt":
        return f"User prompt: {event.content or ''}".strip()
    if event.type == "pre_tool":
        return f"Tool planned: {event.tool_name or 'unknown'} input={_bounded(event.input)}"
    if event.type in {"post_tool", "post_tool_failure"}:
        status = "failed" if event.type == "post_tool_failure" else "completed"
        return f"Tool {event.tool_name or 'unknown'} {status}. Output: {(event.output or '')[:8000]}".strip()
    if event.type in {"subagent_start", "subagent_stop"}:
        return f"{event.type.replace('_', ' ')}: {event.content or _bounded(event.metadata)}"
    if event.type in {"stop", "session_end"}:
        return f"Session ended: {event.content or 'No explicit summary supplied.'}"
    return event.content or ""


def _bounded(value: Any) -> str:
    text = str(value or "")
    return text[:8000]

