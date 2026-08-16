"""LoCoMo benchmark adapter for benchmark_harness."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Callable

from benchmark_harness import BenchmarkQuestion, Ledger, form_session


LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"


def download_locomo(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "locomo10.json"
    if not path.exists():
        print("Downloading LoCoMo...")
        urllib.request.urlretrieve(LOCOMO_URL, path)
    return path


def iter_sessions(conversation: dict) -> list[tuple[str, str | None, list[dict]]]:
    keys = sorted(k for k in conversation if k.startswith("session_") and not k.endswith("_date_time"))
    return [(key, conversation.get(f"{key}_date_time"), conversation[key]) for key in keys]


def turn_text(turn: dict) -> str:
    text = f"{turn.get('speaker', 'speaker')}: {turn.get('text', '')}".strip()
    if turn.get("blip_caption"):
        text += f"\nImage: {turn['blip_caption']}"
    return text


def load_records(path: Path, max_conversations: int) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return records[:max_conversations]


def form_locomo(
    records: list[dict],
    ledger: Ledger,
    *,
    gemini_text: Callable[..., str],
    parse_json: Callable[[str], Any],
) -> dict[str, dict[str, str]]:
    """Returns sample_id -> {dia_id -> observation_id}."""
    dia_maps: dict[str, dict[str, str]] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        dia_to_obs: dict[str, str] = {}
        print("Forming", sample_id, "...")
        for session_key, session_ts, turns in iter_sessions(record["conversation"]):
            episode_id = f"chat:{sample_id}:{session_key}"
            turn_pairs = []
            for turn in turns:
                text = turn_text(turn)
                external = str(turn["dia_id"]) if turn.get("dia_id") else None
                turn_pairs.append((text, external))
            form_session(
                ledger,
                episode_id=episode_id,
                session_ts=session_ts,
                turns=turn_pairs,
                gemini_text=gemini_text,
                parse_json=parse_json,
                obs_id_map=dia_to_obs,
            )
        dia_maps[sample_id] = dia_to_obs
    return dia_maps


def build_questions(
    records: list[dict],
    *,
    max_questions: int,
) -> tuple[list[BenchmarkQuestion], dict[str, dict[str, str]]]:
    questions: list[BenchmarkQuestion] = []
    dia_maps: dict[str, dict[str, str]] = {}
    count = 0
    for record in records:
        sample_id = str(record["sample_id"])
        for qi, qa in enumerate(record.get("qa", [])):
            if count >= max_questions:
                return questions, dia_maps
            count += 1
            questions.append(
                BenchmarkQuestion(
                    question_id=f"{sample_id}:{qi}",
                    question=str(qa["question"]),
                    gold=str(qa.get("answer") or ""),
                    category=qa.get("category"),
                    evidence=[str(e) for e in (qa.get("evidence") or [])],
                    episode_prefix=None,
                    extra={"sample_id": sample_id, "q_index": qi},
                )
            )
    return questions, dia_maps


def evidence_in_hits(
    qa: BenchmarkQuestion,
    hits: list,
    dia_maps: dict[str, dict[str, str]],
) -> bool:
    sample_id = qa.extra.get("sample_id")
    if not sample_id:
        return False
    dia_map = dia_maps.get(sample_id, {})
    hit_ids = {h.item_id for h in hits}
    for dia in qa.evidence:
        obs_id = dia_map.get(dia)
        if obs_id and obs_id in hit_ids:
            return True
    return False
