"""BEAM 100K/10M dataset adapters.

Owns Hugging Face parquet/JSONL conversation flattening and probing-question
normalization. The shared runner still ingests through ``MemoryEngine``. Native
BEAM batches are flattened in plan order, then chunked into groups of 8
messages so fact extraction stays bounded. Turn ids stay aligned with
``source_chat_ids``: unique ids are kept, restarted 10M plan ids are reindexed
in concat order.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

from trisynapse_memory.adapters.benchmarks.base import (
    BenchmarkAdapter,
    BenchmarkCase,
    BenchmarkQuestion,
    PreparedCase,
)
from trisynapse_memory.engine import MemoryEngine
from trisynapse_memory.engine.models import MemoryNamespace, MemoryQueryResult, SearchHit

_SESSION_MESSAGES = 8
_ABILITY_ORDER = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)


class BeamAdapter(BenchmarkAdapter):
    file_glob: ClassVar[str] = "*.parquet"

    def resolve_dataset(self, data_root: str | Path) -> Path:
        path = Path(data_root)
        if path.is_file():
            return path
        pinned = path / self.default_filename
        if pinned.is_file() or (pinned.is_dir() and _dataset_files(pinned)):
            return pinned
        data_dir = path / "data"
        matches = _dataset_files(data_dir, self.file_glob) if data_dir.is_dir() else []
        if len(matches) == 1:
            return matches[0]
        if matches:
            return data_dir
        raise FileNotFoundError(pinned if self.default_filename else path)

    def load_cases(self, path: Path) -> Iterable[BenchmarkCase]:
        for row in _iter_rows(path, self.file_glob):
            yield _beam_case(row)

    def ingest_case(self, engine: MemoryEngine, case: BenchmarkCase) -> PreparedCase:
        namespace = MemoryNamespace(project_id=f"benchmark:{self.name}:{case.id}")
        prefix = f"beam:{case.id}:"
        turns = list(case.payload.get("turns") or [])
        group_ids: list[str] = []
        for session_index, session in enumerate(_chunks(turns, _SESSION_MESSAGES)):
            if not session:
                continue
            group_id = f"{prefix}s{session_index:04d}"
            group_ids.append(group_id)
            engine.ingest_messages(
                [
                    {
                        "id": str(turn.get("id")),
                        "role": turn.get("role") or "speaker",
                        "content": turn.get("content") or "",
                    }
                    for turn in session
                    if str(turn.get("content") or "").strip()
                ],
                group_id=group_id,
                namespace=namespace,
            )
        return PreparedCase(tuple(group_ids), namespace, group_prefix=prefix)

    def result_metadata(
        self,
        question: BenchmarkQuestion,
        result: MemoryQueryResult,
        *,
        retrieval_hits: list[SearchHit] | None = None,
    ) -> dict[str, Any]:
        evidence = {str(value) for value in question.evidence or ()}
        retrieved = _message_ids(retrieval_hits or [])
        cited = _message_ids(result.citations)
        retrieved_evidence = evidence & retrieved
        cited_evidence = evidence & cited
        hit = bool(retrieved_evidence)
        return {
            **question.metadata,
            "gold_evidence_ids": sorted(evidence),
            "retrieved_ids": sorted(retrieved),
            "cited_ids": sorted(cited),
            "evidence_hit": hit,
            "evidence_hit_at_k": hit,
            "evidence_recall_at_k": len(retrieved_evidence) / len(evidence) if evidence else None,
            "all_evidence_retrieved": bool(evidence) and evidence <= retrieved,
            "citation_evidence_recall": len(cited_evidence) / len(evidence) if evidence else None,
            "citation_precision": len(cited_evidence) / len(cited) if cited else None,
            "accuracy": (1.0 if hit else 0.0) if evidence else None,
        }


class Beam100KAdapter(BeamAdapter):
    name = "beam100k"
    default_filename = "data/100K-00000-of-00001.parquet"
    file_glob = "100K-*.parquet"


class Beam10MAdapter(BeamAdapter):
    name = "beam10m"
    default_filename = "data"
    file_glob = "10M-*.parquet"


def _beam_case(row: dict[str, Any]) -> BenchmarkCase:
    conversation_id = str(row.get("conversation_id") or row.get("id") or "conversation")
    seed = row.get("conversation_seed") if isinstance(row.get("conversation_seed"), dict) else {}
    turns = _ordered_turns(row)
    return BenchmarkCase(
        id=conversation_id,
        payload={"conversation_id": conversation_id, "seed": seed, "turns": turns},
        questions=_probing_questions(conversation_id, row.get("probing_questions")),
    )


def _ordered_turns(row: dict[str, Any]) -> list[dict[str, Any]]:
    plans = row.get("plans") or []
    if isinstance(plans, list) and plans:
        turns: list[dict[str, Any]] = []
        for plan in sorted(plans, key=_plan_sort_key):
            if isinstance(plan, dict):
                turns.extend(_turns_from_chat(plan.get("chat")))
        return _reindex(turns)
    return _reindex(_turns_from_chat(row.get("chat")))


def _plan_sort_key(plan: dict[str, Any]) -> tuple[int, int | str]:
    ident = str(plan.get("plan_id") or "")
    try:
        return (0, int(ident))
    except ValueError:
        return (1, ident)


def _turns_from_chat(chat: Any) -> list[dict[str, Any]]:
    if not chat:
        return []
    if _is_session_list(chat):
        return [turn for session in chat for turn in session if isinstance(turn, dict)]
    if isinstance(chat, list) and chat and isinstance(chat[0], dict) and any(
        str(key).startswith("plan-") for key in chat[0]
    ):
        merged: dict[str, Any] = {}
        for block in chat:
            if isinstance(block, dict):
                merged.update(
                    (key, value)
                    for key, value in block.items()
                    if str(key).startswith("plan-") and value is not None
                )
        turns: list[dict[str, Any]] = []
        for key in _plan_keys(merged):
            turns.extend(_turns_from_batches(merged.get(key)))
        return turns
    turns: list[dict[str, Any]] = []
    _walk_turns(chat, turns)
    return turns


def _is_session_list(chat: Any) -> bool:
    if not isinstance(chat, list) or not chat or not isinstance(chat[0], list):
        return False
    first = chat[0][0] if chat[0] else None
    return isinstance(first, dict) and "content" in first


def _plan_keys(block: dict[str, Any]) -> list[str]:
    keys = [key for key in block if str(key).startswith("plan-")]
    def sort_key(key: str) -> tuple[int, str]:
        suffix = str(key).removeprefix("plan-")
        try:
            return (0, f"{int(suffix):04d}")
        except ValueError:
            return (1, suffix)
    return sorted(keys, key=sort_key)


def _turns_from_batches(batches: Any) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    if not isinstance(batches, list):
        return turns
    for batch in batches:
        if isinstance(batch, dict):
            turns.extend(_turns_from_chat(batch.get("turns")))
        elif isinstance(batch, list):
            turns.extend(item for item in batch if isinstance(item, dict) and "content" in item)
    return turns


def _walk_turns(node: Any, turns: list[dict[str, Any]]) -> None:
    if isinstance(node, dict):
        if "content" in node and "role" in node:
            turns.append(node)
            return
        for value in node.values():
            _walk_turns(value, turns)
        return
    if isinstance(node, list):
        for value in node:
            _walk_turns(value, turns)


def _reindex(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids = [str(turn.get("id")) for turn in turns if turn.get("id") is not None]
    if len(ids) == len(turns) and len(set(ids)) == len(ids):
        return turns
    result: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        item = dict(turn)
        item["id"] = index
        result.append(item)
    return result


def _probing_questions(conversation_id: str, raw: Any) -> tuple[BenchmarkQuestion, ...]:
    payload = _parse_probing(raw)
    questions: list[BenchmarkQuestion] = []
    abilities = list(payload)
    ordered = [name for name in _ABILITY_ORDER if name in payload] + [
        name for name in abilities if name not in _ABILITY_ORDER
    ]
    for ability in ordered:
        items = payload.get(ability) or []
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            evidence = tuple(_source_ids(item.get("source_chat_ids")))
            questions.append(BenchmarkQuestion(
                id=f"{conversation_id}:{ability}:{index}",
                question=str(item.get("question") or ""),
                gold=_gold(item),
                evidence=evidence,
                evidence_text=" ".join(_rubric_text(item)),
                metadata={
                    "category": ability,
                    "question_type": ability,
                    "difficulty": item.get("difficulty"),
                },
            ))
    return tuple(questions)


def _parse_probing(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    text = str(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = ast.literal_eval(text)
    return payload if isinstance(payload, dict) else {}


def _gold(item: dict[str, Any]) -> str:
    for key in ("ideal_response", "ideal_answer", "ideal_summary", "answer"):
        value = item.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, list):
            return " ".join(str(part) for part in value if str(part).strip())
        return str(value)
    return ""


def _rubric_text(item: dict[str, Any]) -> list[str]:
    rubric = item.get("rubric") or []
    if isinstance(rubric, str):
        return [rubric] if rubric.strip() else []
    return [str(part).strip() for part in rubric if str(part).strip()]


def _source_ids(value: Any) -> list[str]:
    ids: list[str] = []
    if value is None:
        return ids
    if isinstance(value, dict):
        for item in value.values():
            ids.extend(_source_ids(item))
        return ids
    if isinstance(value, (list, tuple, set)):
        for item in value:
            ids.extend(_source_ids(item))
        return ids
    text = str(value).strip()
    if text:
        ids.append(text)
    return ids


def _chunks(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for index in range(0, len(items), size):
        yield items[index:index + size]


def _message_ids(items: Iterable[Any]) -> set[str]:
    found: set[str] = set()
    for item in items:
        locator = getattr(item, "locator", None)
        if isinstance(locator, dict) and locator.get("message_id") is not None:
            found.add(str(locator["message_id"]))
    return found


def _iter_rows(path: Path, file_glob: str) -> Iterable[dict[str, Any]]:
    for file_path in _dataset_files(path, file_glob) or ([path] if path.is_file() else []):
        suffix = file_path.suffix.lower()
        if suffix == ".jsonl":
            yield from _iter_jsonl(file_path)
        elif suffix == ".parquet":
            yield from _iter_parquet(file_path)
        elif suffix == ".json":
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            rows = payload if isinstance(payload, list) else [payload]
            for row in rows:
                if isinstance(row, dict):
                    yield row


def _dataset_files(path: Path, file_glob: str = "*") -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    files = sorted(path.glob(file_glob))
    if files:
        return [item for item in files if item.is_file()]
    return sorted(
        item for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in {".parquet", ".jsonl", ".json"}
    )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def _iter_parquet(path: Path) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "reading BEAM parquet requires pyarrow (already pulled in by LanceDB). "
            "Reinstall extras or convert the split to JSONL."
        ) from exc
    reader = pq.ParquetFile(path)
    names = list(reader.schema_arrow.names)
    for batch in reader.iter_batches(batch_size=1):
        yield {name: batch.column(name)[0].as_py() for name in names}
