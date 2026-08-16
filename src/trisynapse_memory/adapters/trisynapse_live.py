"""Feature-gated mapping from Trisynapse vault events to trace deltas."""

from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Any

from trisynapse_memory.engine import MemoryEngine


def open_vault_engine(vault_path: str | Path, *, enabled: bool | None = None, **kwargs: Any) -> MemoryEngine:
    return MemoryEngine.open_vault(vault_path, enabled=enabled, **kwargs)


def item_to_deltas(engine: MemoryEngine, item: dict[str, Any], text: str) -> list[Any]:
    _require_enabled()
    item_id = str(item["id"])
    topic_ids = [str(value) for value in item.get("topic_ids", [])]
    return engine.ingest_document(
        text,
        document_id=item_id,
        title=item.get("title"),
        scope={"topic_ids": topic_ids, "item_id": item_id},
        observed_at=item.get("observed_at"),
    )


def chat_message_to_delta(engine: MemoryEngine, message: dict[str, Any]) -> Any:
    _require_enabled()
    session_id = str(message["session_id"])
    return engine.ingest_observation(
        f"{message.get('role', 'user')}: {message.get('content', '')}",
        episode_id=f"chat:{session_id}",
        source_ref={"type": "chat", "id": session_id},
        locator={"kind": "message_index", "message_id": message.get("id")},
        scope={"topic_ids": message.get("topic_ids", []), "session_surfaces": ["main"]},
        observed_at=message.get("created_at"),
        external_key=f"chat_msg:{message.get('id')}",
    )


def wiki_compile_to_deltas(engine: MemoryEngine, *, slug: str, sections: list[dict[str, str]], topic_ids: list[str]) -> list[Any]:
    _require_enabled()
    result = []
    for index, section in enumerate(sections):
        result.append(
            engine.ingest_observation(
                section["text"], episode_id=f"wiki:{slug}",
                source_ref={"type": "wiki", "id": slug},
                locator={"kind": "markdown_heading", "heading": section.get("heading"), "index": index},
                scope={"topic_ids": topic_ids, "article_slug": slug},
                external_key=f"wiki:{slug}:section:{index}:{hashlib.sha256(section['text'].encode()).hexdigest()[:16]}",
            )
        )
    return result


def instruction_rule_to_delta(engine: MemoryEngine, rule: dict[str, Any]) -> Any:
    _require_enabled()
    return engine.store.append(
        kind="annotation", text=str(rule["text"]),
        episode_id=f"core:instruction:{rule['id']}",
        scope={"topic_ids": rule.get("topic_ids", [])},
        payload={"annotation_type": "instruction_rule", "enabled": rule.get("enabled", True)},
        external_key=f"instruction:{rule['id']}:{rule.get('updated_at', '')}",
        confidence=1.0,
    )


def _require_enabled() -> None:
    if os.getenv("TRISYNAPSE_MEMORY_V2_ENABLED", "").lower() not in {"1", "true", "yes"}:
        raise RuntimeError("Trisynapse live adapter is disabled; set TRISYNAPSE_MEMORY_V2_ENABLED=true")
