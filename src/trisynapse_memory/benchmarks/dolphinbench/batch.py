"""Resumable OpenAI Batch projection and durable benchmark cost accounting."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


PROJECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "aliases"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string"},
        "aliases": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["facts", "summary", "aliases"],
    "additionalProperties": False,
}

PROJECTION_SYSTEM_PROMPT = """Extract durable personal memory from one conversation episode.
Keep explicit preferences, commitments, relationships, biographical facts, dates, changes, and
instructions that could matter later. Do not invent facts. Return concise standalone facts, a
brief episode summary, and alternate search phrasings as the required JSON object."""


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


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    def as_submission_usage(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


def token_usage(value: Any) -> TokenUsage:
    """Normalize Responses usage while avoiding cache/reasoning double counts."""

    usage = _field(value, "usage", value) or {}
    total_input = int(_field(usage, "input_tokens", 0) or 0)
    output = int(_field(usage, "output_tokens", 0) or 0)
    input_details = _field(usage, "input_tokens_details", {}) or {}
    output_details = _field(usage, "output_tokens_details", {}) or {}
    cached = int(
        _field(input_details, "cached_tokens", _field(usage, "cache_read_tokens", 0)) or 0
    )
    written = int(_field(usage, "cache_write_tokens", 0) or 0)
    reasoning = int(_field(output_details, "reasoning_tokens", 0) or 0)
    fresh = total_input - cached - written
    values = (fresh, output, cached, written, reasoning)
    if any(item < 0 for item in values) or reasoning > output:
        raise ValueError("invalid OpenAI token usage")
    return TokenUsage(fresh, output, cached, written, reasoning)


@dataclass(frozen=True)
class Pricing:
    standard_input_per_million: float = 0.10
    standard_cached_input_per_million: float = 0.01
    standard_cache_write_per_million: float = 0.125
    standard_output_per_million: float = 0.50
    batch_discount: float = 0.50

    def cost(self, usage: TokenUsage, *, batch: bool) -> float:
        multiplier = self.batch_discount if batch else 1.0
        value = (
            usage.input_tokens * self.standard_input_per_million
            + usage.cache_read_tokens * self.standard_cached_input_per_million
            + usage.cache_write_tokens * self.standard_cache_write_per_million
            + usage.output_tokens * self.standard_output_per_million
        ) / 1_000_000
        return value * multiplier


class CostLedger:
    """Idempotent append-only ledger used by ``total_cost_usd`` after restarts."""

    def __init__(self, path: str | Path, pricing: Pricing | None = None) -> None:
        self.path = Path(path)
        self.pricing = pricing or Pricing()
        self._entries: dict[str, dict[str, Any]] | None = None

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._entries is not None:
            return self._entries
        entries: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            for position, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    event_id = str(row.pop("event_id"))
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(f"invalid cost ledger line {position}") from exc
                existing = entries.get(event_id)
                if existing is not None and existing != row:
                    raise ValueError(f"cost event changed: {event_id}")
                entries[event_id] = row
        self._entries = entries
        return entries

    def record(
        self,
        event_id: str,
        *,
        phase: str,
        usage: TokenUsage,
        batch: bool,
        model: str,
    ) -> None:
        entry = {
            "phase": phase,
            "model": model,
            "batch": batch,
            "usage": usage.as_submission_usage(),
            "cost_usd": self.pricing.cost(usage, batch=batch),
        }
        entries = self._load()
        existing = entries.get(event_id)
        if existing is not None and existing != entry:
            raise ValueError(f"cost event changed: {event_id}")
        if existing is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"event_id": event_id, **entry}, sort_keys=True, ensure_ascii=False)
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        entries[event_id] = entry

    def total(self, phase: str) -> float:
        return float(
            sum(
                entry["cost_usd"]
                for entry in self._load().values()
                if entry["phase"] == phase
            )
        )


class ProjectionQueue:
    """Durable, idempotent projection requests for one persona."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._items: dict[str, dict[str, Any]] | None = None

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._items is not None:
            return self._items
        items: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            for position, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid projection queue line {position}: {exc.msg}") from exc
                custom_id = str(item["custom_id"])
                existing = items.get(custom_id)
                if existing is not None and existing != item:
                    raise ValueError(f"projection request changed: {custom_id}")
                items[custom_id] = item
        self._items = items
        return items

    def enqueue(self, item: dict[str, Any]) -> None:
        custom_id = str(item["custom_id"])
        items = self._load()
        existing = items.get(custom_id)
        if existing is not None and existing != item:
            raise ValueError(f"projection request changed: {custom_id}")
        if existing is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        items[custom_id] = item

    def items(self) -> dict[str, dict[str, Any]]:
        return dict(self._load())


def response_output_text(response: Any) -> str:
    direct = _field(response, "output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    values: list[str] = []
    for item in _field(response, "output", []) or []:
        if _field(item, "type") != "message":
            continue
        for content in _field(item, "content", []) or []:
            if _field(content, "type") in {"output_text", "text"}:
                text = _field(content, "text")
                if isinstance(text, str):
                    values.append(text)
    return "\n".join(values)


def validate_projection(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"facts", "summary", "aliases"}:
        raise ValueError("invalid memory projection object")
    if not isinstance(value["facts"], list) or not isinstance(value["aliases"], list):
        raise ValueError("invalid memory projection arrays")
    if not isinstance(value["summary"], str):
        raise ValueError("invalid memory projection summary")
    facts = []
    for fact in value["facts"]:
        if not isinstance(fact, dict) or set(fact) != {"text", "aliases"}:
            raise ValueError("invalid projected fact")
        if not isinstance(fact["text"], str) or not isinstance(fact["aliases"], list):
            raise ValueError("invalid projected fact fields")
        facts.append(
            {
                "text": fact["text"].strip(),
                "aliases": [str(item).strip() for item in fact["aliases"] if str(item).strip()],
            }
        )
    return {
        "facts": [fact for fact in facts if fact["text"]],
        "summary": value["summary"].strip(),
        "aliases": [str(item).strip() for item in value["aliases"] if str(item).strip()],
    }


def _file_bytes(value: Any) -> bytes:
    content = _field(value, "content", value)
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        return content.encode()
    if hasattr(value, "read"):
        read = value.read()
        return read if isinstance(read, bytes) else str(read).encode()
    raise TypeError("OpenAI file response did not contain bytes")


class BatchProjector:
    """Submit, resume, and apply one projection batch for a persona."""

    TERMINAL = {"completed", "failed", "expired", "cancelled"}

    def __init__(
        self,
        client: Any,
        directory: str | Path,
        ledger: CostLedger,
        *,
        model: str = "gpt-6-luna",
        reasoning_effort: str = "high",
        max_output_tokens: int = 1200,
        completion_window: str = "24h",
        poll_interval_seconds: float = 30,
        max_retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.directory = Path(directory)
        self.ledger = ledger
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.completion_window = completion_window
        self.poll_interval_seconds = poll_interval_seconds
        self.max_retries = max_retries
        self.sleep = sleep

    def run(
        self,
        persona: str,
        queue: ProjectionQueue,
        apply: Callable[[dict[str, Any], dict[str, Any]], None],
    ) -> None:
        items = queue.items()
        state_path = self.directory / persona / "state.json"
        state = _load_json(state_path, {"version": 1, "jobs": [], "applied": []})
        applied = set(state["applied"])
        missing = [custom_id for custom_id in sorted(items) if custom_id not in applied]
        if not missing:
            return
        while missing:
            job = self._resumable_job(state, missing)
            if job is None:
                if len(state["jobs"]) >= self.max_retries + 1:
                    break
                attempt = len(state["jobs"])
                job = self._submit(persona, attempt, missing, items)
                state["jobs"].append(job)
                _atomic_json(state_path, state)
            batch = self._wait(job["batch_id"])
            job["status"] = str(_field(batch, "status"))
            job["output_file_id"] = _field(batch, "output_file_id")
            job["error_file_id"] = _field(batch, "error_file_id")
            _atomic_json(state_path, state)
            completed = self._results(batch, job["batch_id"])
            for custom_id, payload in completed.items():
                if custom_id not in items or custom_id in applied:
                    continue
                apply(items[custom_id], payload)
                applied.add(custom_id)
            state["applied"] = sorted(applied)
            job["processed"] = True
            _atomic_json(state_path, state)
            missing = [custom_id for custom_id in missing if custom_id not in applied]
            if not missing:
                return
        raise RuntimeError(
            f"DolphinBench projection batch did not complete {len(missing)} request(s): "
            + ", ".join(missing[:10])
        )

    def _resumable_job(self, state: dict[str, Any], missing: list[str]) -> dict[str, Any] | None:
        for job in reversed(state["jobs"]):
            covers_missing = set(missing).issubset(set(job.get("custom_ids") or []))
            if covers_missing and not job.get("processed", False):
                return job
        return None

    def _submit(
        self,
        persona: str,
        attempt: int,
        custom_ids: list[str],
        items: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        directory = self.directory / persona
        directory.mkdir(parents=True, exist_ok=True)
        input_path = directory / f"projection-{attempt:02d}.jsonl"
        with input_path.open("w", encoding="utf-8") as stream:
            for custom_id in custom_ids:
                stream.write(json.dumps(self._request(custom_id, items[custom_id]), ensure_ascii=False))
                stream.write("\n")
        with input_path.open("rb") as stream:
            uploaded = self.client.files.create(file=stream, purpose="batch")
        batch = self.client.batches.create(
            input_file_id=_field(uploaded, "id"),
            endpoint="/v1/responses",
            completion_window=self.completion_window,
            metadata={"benchmark": "dolphinbench", "persona": persona, "attempt": str(attempt)},
        )
        return {
            "batch_id": _field(batch, "id"),
            "input_file_id": _field(uploaded, "id"),
            "input_path": str(input_path),
            "custom_ids": list(custom_ids),
            "status": str(_field(batch, "status", "validating")),
            "processed": False,
        }

    def _request(self, custom_id: str, item: dict[str, Any]) -> dict[str, Any]:
        episode = {
            "persona": item["persona"],
            "narrative_time": item["narrative_time"],
            "messages": item["messages"],
        }
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": self.model,
                "reasoning": {"effort": self.reasoning_effort},
                "input": [
                    {"role": "system", "content": PROJECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(episode, ensure_ascii=False)},
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "memory_projection",
                        "strict": True,
                        "schema": PROJECTION_SCHEMA,
                    }
                },
                "max_output_tokens": self.max_output_tokens,
            },
        }

    def _wait(self, batch_id: str) -> Any:
        while True:
            batch = self.client.batches.retrieve(batch_id)
            if str(_field(batch, "status")) in self.TERMINAL:
                return batch
            self.sleep(self.poll_interval_seconds)

    def _results(self, batch: Any, batch_id: str) -> dict[str, dict[str, Any]]:
        output_file_id = _field(batch, "output_file_id")
        if not output_file_id:
            return {}
        content = _file_bytes(self.client.files.content(output_file_id)).decode("utf-8")
        results: dict[str, dict[str, Any]] = {}
        for line in content.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            response = row.get("response") or {}
            if response.get("status_code") != 200:
                continue
            body = response.get("body") or {}
            custom_id = str(row["custom_id"])
            usage = token_usage(body)
            request_id = str(row.get("id") or response.get("request_id") or custom_id)
            self.ledger.record(
                f"batch:{batch_id}:{request_id}",
                phase="ingestion",
                usage=usage,
                batch=True,
                model=self.model,
            )
            try:
                payload = validate_projection(json.loads(response_output_text(body)))
            except (json.JSONDecodeError, ValueError):
                continue
            results[custom_id] = payload
        return results


def projection_deltas(item: dict[str, Any], payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Create idempotent public ``append_deltas`` payloads for one projection."""

    group_id = item["group_id"]
    actor = {"type": "ingest_pipeline", "id": "dolphinbench-batch", "model": "gpt-6-luna"}
    base = {
        "group_id": group_id,
        "evidence_refs": item["raw_record_ids"],
        "actor": actor,
        "observed_at": item["narrative_time"],
        "confidence": 0.8,
        "scope": {"benchmark": "dolphinbench", "persona": item["persona"]},
    }
    for index, fact in enumerate(payload["facts"]):
        aliases = fact["aliases"]
        text = fact["text"]
        if aliases:
            text += "\nAlternate phrasings: " + "; ".join(aliases)
        yield {
            **base,
            "kind": "fact",
            "text": text,
            "payload": {"aliases": aliases, "projection": "openai_batch"},
            "external_key": f"dolphinbench:projection:{item['custom_id']}:fact:{index}",
        }
    summary = payload["summary"]
    if payload["aliases"]:
        summary += "\nSearch aliases: " + "; ".join(payload["aliases"])
    if summary.strip():
        yield {
            **base,
            "kind": "note",
            "text": summary,
            "payload": {"aliases": payload["aliases"], "projection": "openai_batch"},
            "external_key": f"dolphinbench:projection:{item['custom_id']}:summary",
        }
