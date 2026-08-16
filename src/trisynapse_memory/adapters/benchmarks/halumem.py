"""HaluMem dataset adapter."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from trisynapse_memory.adapters.benchmarks.base import (
    BenchmarkAdapter,
    BenchmarkCase,
    BenchmarkQuestion,
    PreparedCase,
)
from trisynapse_memory.engine import MemoryEngine
from trisynapse_memory.engine.models import MemoryNamespace


class HaluMemAdapter(BenchmarkAdapter):
    name = "halumem"
    default_filename = "HaluMem-Medium.jsonl"

    def load_cases(self, path: Path) -> Iterable[BenchmarkCase]:
        with path.open(encoding="utf-8") as handle:
            for line_index, raw in enumerate(handle):
                if not raw.strip():
                    continue
                user = json.loads(raw)
                user_id = str(user.get("uuid") or line_index)
                questions: list[BenchmarkQuestion] = []
                for session_index, session in enumerate(user.get("sessions") or []):
                    for question_index, item in enumerate(session.get("questions") or []):
                        evidence_text = " ".join(
                            str(value.get("memory_content") if isinstance(value, dict) else value)
                            for value in item.get("evidence") or []
                        )
                        questions.append(BenchmarkQuestion(
                            id=f"{user_id}:s{session_index}:q{question_index}",
                            question=str(item.get("question") or ""),
                            gold=str(item.get("answer") or ""),
                            evidence=item.get("evidence") or [],
                            evidence_text=evidence_text,
                        ))
                yield BenchmarkCase(id=user_id, payload=user, questions=tuple(questions))

    def ingest_case(self, engine: MemoryEngine, case: BenchmarkCase) -> PreparedCase:
        namespace = MemoryNamespace(project_id=f"benchmark:halumem:{case.id}")
        episode_ids: list[str] = []
        for session_index, session in enumerate(case.payload.get("sessions") or []):
            episode_id = f"halu:{case.id}:s{session_index:04d}"
            episode_ids.append(episode_id)
            messages = [
                {
                    "id": f"{session_index}:{turn_index}",
                    "role": turn.get("role", "speaker"),
                    "content": turn.get("content", ""),
                    "timestamp": _normalize_timestamp(turn.get("timestamp")),
                }
                for turn_index, turn in enumerate(session.get("dialogue") or [])
            ]
            if messages:
                engine.ingest_messages(messages, episode_id=episode_id, namespace=namespace)
        return PreparedCase(tuple(episode_ids), namespace, episode_prefix=f"halu:{case.id}:")


def _normalize_timestamp(value: object) -> str | None:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%b %d, %Y, %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise ValueError(f"unsupported HaluMem timestamp: {text}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat()
