"""MemoryDocDataSet adapter for benchmark_harness.

Combines multi-session conversations with long documents. Hybrid questions require
conversation-to-document routing before reading document evidence.

Official release: https://arxiv.org/abs/2606.04442
When the public JSON drop is unavailable, use data/memorydoc/fixtures/smoke_micro_world.json.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable

from benchmark_harness import BenchmarkQuestion, Ledger, form_session


DEFAULT_FIXTURE = "fixtures/smoke_micro_world.json"
SPLIT_FILES = {
    "smoke": DEFAULT_FIXTURE,
    "test": "memorydoc_test.json",
    "val": "memorydoc_val.json",
    "train": "memorydoc_train.json",
    "all": "memorydoc_v1.json",
}


def smoke_fixture_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "memorydoc" / DEFAULT_FIXTURE


def _find_installed_release(data_dir: Path) -> Path | None:
    """Return any official MemoryDoc JSON found under data_dir."""
    for name in ("memorydoc_v1.json", "memorydoc_train.json", "memorydoc_test.json", "memorydoc_val.json"):
        path = data_dir / name
        if path.exists():
            return path
    matches = sorted(data_dir.glob("memorydoc*.json"))
    return matches[0] if matches else None


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or "").lower())
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_document(text: str, *, chunk_chars: int = 3500) -> list[str]:
    """Split long documents into passage-sized chunks for ledger observations."""
    text = str(text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        if end < len(text):
            para = text.rfind("\n\n", start, end)
            if para > start + chunk_chars // 2:
                end = para
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start = end if end > start else start + chunk_chars
    return chunks


def _resolve_path(data_dir: Path, split: str, *, allow_smoke_fallback: bool = True) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    name = SPLIT_FILES.get(split, split)
    path = data_dir / name
    if path.exists():
        return path

    fixture = smoke_fixture_path()
    if split == "smoke" and fixture.exists():
        return fixture

    installed = _find_installed_release(data_dir)
    if installed is not None:
        print(
            f"MemoryDocDataSet: requested split '{split}' ({name}) not found; "
            f"using installed file {installed.name}."
        )
        return installed

    if allow_smoke_fallback and fixture.exists():
        print(
            f"WARNING: MemoryDocDataSet split '{split}' not found at {path}.\n"
            f"  Falling back to smoke fixture: {fixture}\n"
            "  The official release is not bundled — set MEMORYDOC_SPLIT='smoke' explicitly, "
            "or place memorydoc_v1.json (or train/test/val splits) under data/memorydoc/."
        )
        return fixture

    raise FileNotFoundError(
        f"MemoryDocDataSet file not found: {path}\n"
        "Place the official JSON release under data/memorydoc/ "
        f"(expected names: {', '.join(SPLIT_FILES.values())}), "
        f"or run with MEMORYDOC_SPLIT='smoke' to use {fixture}."
    )


def download_memorydoc(data_dir: Path, split: str = "smoke", *, allow_smoke_fallback: bool = True) -> Path:
    """Resolve dataset path. Falls back to smoke fixture when official splits are missing."""
    return _resolve_path(data_dir, split, allow_smoke_fallback=allow_smoke_fallback)


def load_micro_worlds(path: Path, max_worlds: int) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        worlds = payload
    elif isinstance(payload, dict):
        worlds = payload.get("micro_worlds") or payload.get("worlds") or [payload]
    else:
        raise ValueError(f"Unexpected MemoryDoc JSON top-level type: {type(payload)}")
    return worlds[:max_worlds]


def _utterance_text(item: dict) -> str:
    speaker = item.get("speaker") or item.get("persona") or item.get("role") or "speaker"
    text = item.get("text") or item.get("content") or ""
    return f"{speaker}: {text}".strip()


def form_micro_world(
    world: dict,
    ledger: Ledger,
    *,
    gemini_text: Callable[..., str],
    parse_json: Callable[[str], Any],
    max_doc_chars: int | None = None,
    doc_chunk_chars: int = 3500,
    max_sessions: int | None = None,
    skip_document_extraction: bool = False,
) -> dict[str, dict[str, str]]:
    """Ingest one micro-world. Returns maps for evidence lookup."""
    world_id = str(world.get("micro_world_id") or world.get("world_id") or world.get("id"))
    prefix = f"mwd:{world_id}"
    doc_chunk_map: dict[str, str] = {}
    chat_turn_map: dict[str, str] = {}

    print("Forming MemoryDoc world", world_id, "...")

    for doc in world.get("documents") or []:
        doc_id = str(doc.get("document_id") or doc.get("id"))
        title = str(doc.get("title") or doc_id)
        text = str(doc.get("text") or doc.get("content") or "")
        if max_doc_chars is not None:
            text = text[:max_doc_chars]
        chunks = chunk_document(text, chunk_chars=doc_chunk_chars)
        if not chunks:
            continue
        episode_id = f"{prefix}:doc:{doc_id}"
        turns = [
            (f"[Document: {title} | chunk {i + 1}/{len(chunks)}]\n{chunk}", f"{doc_id}:chunk:{i}")
            for i, chunk in enumerate(chunks)
        ]
        obs_map: dict[str, str] = {}
        form_session(
            ledger,
            episode_id=episode_id,
            session_ts=doc.get("date") or doc.get("timestamp"),
            turns=turns,
            gemini_text=gemini_text if not skip_document_extraction else lambda *a, **k: '{"facts": []}',
            parse_json=parse_json,
            obs_id_map=obs_map,
        )
        for key, obs_id in obs_map.items():
            doc_chunk_map[key] = obs_id

    sessions = list(world.get("conversations") or world.get("sessions") or [])
    if max_sessions is not None:
        sessions = sessions[:max_sessions]
    for sess in sessions:
        session_id = str(sess.get("session_id") or sess.get("event_id") or sess.get("id"))
        episode_id = f"{prefix}:chat:{session_id}"
        utterances = sess.get("utterances") or sess.get("messages") or []
        turns = [
            (_utterance_text(u), str(u.get("utterance_id") or u.get("id") or f"{session_id}:{i}"))
            for i, u in enumerate(utterances)
        ]
        if not turns:
            continue
        obs_map = {}
        form_session(
            ledger,
            episode_id=episode_id,
            session_ts=sess.get("timestamp") or sess.get("event_timestamp"),
            turns=turns,
            gemini_text=gemini_text,
            parse_json=parse_json,
            obs_id_map=obs_map,
        )
        chat_turn_map.update(obs_map)

    return {"doc_chunks": doc_chunk_map, "chat_turns": chat_turn_map, "world_id": world_id}


def form_memorydoc(
    worlds: list[dict],
    ledger: Ledger,
    *,
    gemini_text: Callable[..., str],
    parse_json: Callable[[str], Any],
    max_doc_chars: int | None = None,
    doc_chunk_chars: int = 3500,
    max_sessions_per_world: int | None = None,
    skip_document_extraction: bool = False,
) -> dict[str, dict[str, dict[str, str]]]:
    maps: dict[str, dict[str, dict[str, str]]] = {}
    for world in worlds:
        world_maps = form_micro_world(
            world,
            ledger,
            gemini_text=gemini_text,
            parse_json=parse_json,
            max_doc_chars=max_doc_chars,
            doc_chunk_chars=doc_chunk_chars,
            max_sessions=max_sessions_per_world,
            skip_document_extraction=skip_document_extraction,
        )
        maps[str(world_maps["world_id"])] = world_maps
    return maps


def build_questions(
    worlds: list[dict],
    *,
    max_questions: int,
) -> list[BenchmarkQuestion]:
    questions: list[BenchmarkQuestion] = []
    count = 0
    for world in worlds:
        world_id = str(world.get("micro_world_id") or world.get("world_id") or world.get("id"))
        prefix = f"mwd:{world_id}:"
        qa_items = world.get("qa_pairs") or world.get("questions") or []
        for qi, item in enumerate(qa_items):
            if count >= max_questions:
                return questions
            count += 1
            evidence_refs = item.get("evidence_references") or item.get("evidence") or []
            evidence_ids = []
            for ref in evidence_refs:
                if isinstance(ref, dict):
                    evidence_ids.append(
                        f"{ref.get('source_type')}:{ref.get('source_id')}:{ref.get('passage_span', '')}"
                    )
                else:
                    evidence_ids.append(str(ref))
            source_tag = item.get("source_tag") or item.get("source")
            if not source_tag and item.get("requires_doc_navigation"):
                source_tag = "Hybrid"
            questions.append(
                BenchmarkQuestion(
                    question_id=f"{world_id}:q{qi}",
                    question=str(item.get("question") or ""),
                    gold=str(item.get("gold_answer") or item.get("answer") or ""),
                    category=item.get("category") or item.get("question_category"),
                    ability=str(source_tag or ""),
                    evidence=evidence_ids,
                    episode_prefix=prefix,
                    extra={
                        "micro_world_id": world_id,
                        "source_tag": source_tag,
                        "requires_doc_navigation": bool(item.get("requires_doc_navigation")),
                        "evidence_references": evidence_refs,
                        "difficulty": item.get("difficulty"),
                    },
                )
            )
    return questions


def evidence_in_hits(
    qa: BenchmarkQuestion,
    hits: list,
    world_maps: dict[str, dict[str, dict[str, str]]] | None = None,
) -> bool:
    """Check conversation clue and/or document chunk presence in retrieved hits."""
    world_id = str(qa.extra.get("micro_world_id") or "")
    maps = (world_maps or {}).get(world_id, {})
    hit_ids = {h.item_id for h in hits}
    hit_text = _normalize(" ".join(h.text for h in hits))

    # Structured evidence references from the benchmark.
    for ref in qa.extra.get("evidence_references") or []:
        if not isinstance(ref, dict):
            continue
        source_type = str(ref.get("source_type") or "").lower()
        source_id = str(ref.get("source_id") or "")
        span = str(ref.get("passage_span") or ref.get("excerpt") or "")
        if source_type == "document":
            for key, obs_id in (maps.get("doc_chunks") or {}).items():
                if source_id and source_id in key and obs_id in hit_ids:
                    return True
            if span and _normalize(span)[:80] in hit_text:
                return True
        elif source_type in {"conversation", "chat", "session"}:
            for key, obs_id in (maps.get("chat_turns") or {}).items():
                if source_id and source_id in key and obs_id in hit_ids:
                    return True
            if span and _normalize(span)[:80] in hit_text:
                return True

    # Fallback: gold answer substring in context (weak signal).
    gold = _normalize(qa.gold)
    if len(gold) >= 8 and gold in hit_text:
        return True
    return False


def summarize_by_source_tag(results: list[dict[str, Any]]) -> dict[str, float]:
    by_tag: dict[str, list[bool]] = {}
    for row in results:
        tag = str(row.get("source_tag") or row.get("ability") or "unknown")
        by_tag.setdefault(tag, []).append(bool(row.get("correct")))
    return {k: sum(v) / len(v) for k, v in sorted(by_tag.items()) if v}
