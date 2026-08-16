"""HaluMem benchmark adapter for benchmark_harness.

HaluMem evaluates memory extraction, update, and QA. This notebook adapter focuses on
end-to-end QA over formed ledger memory, plus optional extraction recall vs reference
memory points.

Data: https://huggingface.co/datasets/IAAR-Shanghai/HaluMem
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable

from benchmark_harness import BenchmarkQuestion, Ledger, form_session


HF_REPO = "IAAR-Shanghai/HaluMem"
SPLIT_FILES = {
    "medium": "HaluMem-Medium.jsonl",
    "long": "HaluMem-Long.jsonl",
}


def download_halumem(data_dir: Path, split: str = "medium") -> Path:
    """Download HaluMem JSONL from Hugging Face if missing."""
    data_dir.mkdir(parents=True, exist_ok=True)
    name = SPLIT_FILES.get(split, SPLIT_FILES["medium"])
    path = data_dir / name
    if path.exists():
        return path
    try:
        from huggingface_hub import hf_hub_download

        import shutil

        cached = hf_hub_download(HF_REPO, name, repo_type="dataset")
        shutil.copy2(cached, path)
    except Exception as exc:
        raise FileNotFoundError(
            f"HaluMem data not found at {path}. "
            f"Install huggingface_hub and retry, or place {name} manually.\n"
            f"Original error: {exc}"
        ) from exc
    return path


def resolve_data_path(data_dir: Path, split: str = "medium") -> Path:
    return download_halumem(data_dir, split)


def load_users(path: Path, max_users: int) -> list[dict]:
    users: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            users.append(json.loads(line))
            if len(users) >= max_users:
                break
    return users


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def dialogue_turn_text(turn: dict) -> str:
    role = str(turn.get("role") or "speaker")
    content = str(turn.get("content") or "").strip()
    ts = turn.get("timestamp")
    prefix = f"({ts}) " if ts else ""
    return f"{prefix}{role}: {content}".strip()


def form_halumem(
    users: list[dict],
    ledger: Ledger,
    *,
    gemini_text: Callable[..., str],
    parse_json: Callable[[str], Any],
    max_sessions_per_user: int | None = None,
) -> dict[str, dict[str, str]]:
    """Ingest HaluMem users. Returns user_uuid -> {memory_index -> observation_id}."""
    memory_maps: dict[str, dict[str, str]] = {}
    for user in users:
        uuid = str(user["uuid"])
        print("Forming HaluMem user", uuid[:8], "...")
        mem_map: dict[str, str] = {}
        memory_maps[uuid] = mem_map
        sessions = list(user.get("sessions") or [])
        if max_sessions_per_user is not None:
            sessions = sessions[:max_sessions_per_user]
        for si, session in enumerate(sessions):
            episode_id = f"halu:{uuid}:s{si:04d}"
            turns = [(dialogue_turn_text(t), None) for t in (session.get("dialogue") or [])]
            if not turns:
                continue
            session_ts = session.get("start_time") or session.get("end_time")
            form_session(
                ledger,
                episode_id=episode_id,
                session_ts=str(session_ts) if session_ts else None,
                turns=turns,
                gemini_text=gemini_text,
                parse_json=parse_json,
            )
            # Map reference memory point indices to the first observation in the episode.
            obs_ids = [d.id for d in ledger.observations() if d.episode_id == episode_id]
            anchor = obs_ids[0] if obs_ids else None
            if anchor:
                for mp in session.get("memory_points") or []:
                    idx = str(mp.get("index"))
                    if idx:
                        mem_map[f"{si}:{idx}"] = anchor
    return memory_maps


def build_questions(
    users: list[dict],
    *,
    max_questions: int,
    max_sessions_per_user: int | None = None,
) -> list[BenchmarkQuestion]:
    questions: list[BenchmarkQuestion] = []
    count = 0
    for user in users:
        uuid = str(user["uuid"])
        sessions = list(user.get("sessions") or [])
        if max_sessions_per_user is not None:
            sessions = sessions[:max_sessions_per_user]
        for si, session in enumerate(sessions):
            for qi, item in enumerate(session.get("questions") or []):
                if count >= max_questions:
                    return questions
                count += 1
                evidence = [str(e.get("memory_content") or e) for e in (item.get("evidence") or [])]
                questions.append(
                    BenchmarkQuestion(
                        question_id=f"{uuid}:s{si:04d}:q{qi}",
                        question=str(item.get("question") or ""),
                        gold=str(item.get("answer") or ""),
                        category=item.get("question_type"),
                        ability=str(item.get("difficulty") or ""),
                        evidence=evidence,
                        episode_prefix=f"halu:{uuid}:",
                        extra={
                            "user_uuid": uuid,
                            "session_index": si,
                            "question_type": item.get("question_type"),
                            "difficulty": item.get("difficulty"),
                        },
                    )
                )
    return questions


def evidence_in_hits(qa: BenchmarkQuestion, hits: list) -> bool:
    """True if retrieved context overlaps gold memory evidence text."""
    if not qa.evidence:
        return False
    hit_text = _normalize(" ".join(h.text for h in hits))
    for ev in qa.evidence:
        norm = _normalize(str(ev))
        if len(norm) >= 12 and norm in hit_text:
            return True
        # Also accept substantial token overlap for paraphrased extractions.
        tokens = [t for t in re.split(r"[^a-z0-9]+", norm) if len(t) > 3]
        if tokens and sum(1 for t in tokens if t in hit_text) / len(tokens) >= 0.6:
            return True
    return False


def score_extraction_recall(
    users: list[dict],
    ledger: Ledger,
    *,
    max_sessions_per_user: int | None = None,
) -> dict[str, Any]:
    """Compare formed extractions to HaluMem reference memory points (session-level recall)."""
    extraction_text = _normalize(" ".join(d.text for d in ledger.extractions()))
    observation_text = _normalize(" ".join(d.text for d in ledger.observations()))
    combined = f"{observation_text} {extraction_text}".strip()

    total_points = 0
    recalled_points = 0
    per_type: dict[str, list[bool]] = {}

    for user in users:
        sessions = list(user.get("sessions") or [])
        if max_sessions_per_user is not None:
            sessions = sessions[:max_sessions_per_user]
        for session in sessions:
            for mp in session.get("memory_points") or []:
                content = str(mp.get("memory_content") or "").strip()
                if not content:
                    continue
                total_points += 1
                norm = _normalize(content)
                hit = norm in combined
                if not hit:
                    tokens = [t for t in re.split(r"[^a-z0-9]+", norm) if len(t) > 3]
                    hit = bool(tokens) and sum(1 for t in tokens if t in combined) / len(tokens) >= 0.5
                recalled_points += int(hit)
                mtype = str(mp.get("memory_type") or "unknown")
                per_type.setdefault(mtype, []).append(hit)

    return {
        "memory_points_total": total_points,
        "memory_points_recalled": recalled_points,
        "memory_point_recall": (recalled_points / total_points) if total_points else 0.0,
        "memory_point_recall_by_type": {
            k: sum(v) / len(v) for k, v in sorted(per_type.items()) if v
        },
    }
