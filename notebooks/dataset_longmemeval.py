"""LongMemEval (cleaned) benchmark adapter for benchmark_harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from benchmark_harness import BenchmarkQuestion, Ledger, form_session


SPLIT_FILES = {
    "small": "longmemeval_s_cleaned.json",
    "medium": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}


def download_longmemeval(data_dir: Path) -> Path:
    """Download cleaned LongMemEval from Hugging Face if missing."""
    data_dir.mkdir(parents=True, exist_ok=True)
    marker = data_dir / ".downloaded"
    if marker.exists():
        return data_dir
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="xiaowu0162/longmemeval-cleaned",
            repo_type="dataset",
            local_dir=str(data_dir),
            allow_patterns=list(SPLIT_FILES.values()),
        )
        marker.write_text("ok\n", encoding="utf-8")
    except Exception as exc:
        raise FileNotFoundError(
            f"LongMemEval data not found in {data_dir}. "
            f"Run: python -m trisynapse_memory.cli download-longmemeval --data-root {data_dir}\n"
            f"Original error: {exc}"
        ) from exc
    return data_dir


def resolve_data_path(data_dir: Path, split: str = "small") -> Path:
    download_longmemeval(data_dir)
    name = SPLIT_FILES.get(split, SPLIT_FILES["small"])
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing LongMemEval file: {path}")
    return path


def load_records(path: Path, max_questions: int) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return records[:max_questions]


def message_text(message: dict | str) -> tuple[str, str]:
    if isinstance(message, str):
        return "speaker", message
    role = str(message.get("role", "speaker"))
    content = str(message.get("content", ""))
    return role, f"{role}: {content}".strip()


def form_longmemeval(
    records: list[dict],
    ledger: Ledger,
    *,
    gemini_text: Callable[..., str],
    parse_json: Callable[[str], Any],
    max_sessions_per_question: int | None = None,
) -> None:
    for record in records:
        qid = str(record["question_id"])
        print("Forming", qid, "...")
        session_ids = [str(s) for s in record.get("haystack_session_ids", [])]
        sessions = list(record.get("haystack_sessions", []))
        dates = list(record.get("haystack_dates", []))
        if max_sessions_per_question is not None:
            session_ids = session_ids[:max_sessions_per_question]
            sessions = sessions[:max_sessions_per_question]
            dates = dates[:max_sessions_per_question]

        for session_id, messages, date in zip(session_ids, sessions, dates):
            episode_id = f"lme:{qid}:{session_id}"
            turns = [message_text(m)[1] for m in messages]
            turn_pairs = [(t, None) for t in turns]
            form_session(
                ledger,
                episode_id=episode_id,
                session_ts=str(date) if date else None,
                turns=turn_pairs,
                gemini_text=gemini_text,
                parse_json=parse_json,
            )


def build_questions(records: list[dict]) -> list[BenchmarkQuestion]:
    questions: list[BenchmarkQuestion] = []
    for record in records:
        qid = str(record["question_id"])
        questions.append(
            BenchmarkQuestion(
                question_id=qid,
                question=str(record["question"]),
                gold=str(record.get("answer") or ""),
                category=record.get("question_type"),
                ability=str(record.get("question_type") or ""),
                evidence=[str(s) for s in record.get("answer_session_ids", [])],
                episode_prefix=f"lme:{qid}:",
                extra={
                    "question_date": record.get("question_date"),
                    "answer_session_ids": record.get("answer_session_ids", []),
                },
            )
        )
    return questions


def evidence_in_hits(qa: BenchmarkQuestion, hits: list) -> bool:
    """True if any hit comes from a gold answer session."""
    qid = qa.question_id
    session_ids = qa.extra.get("answer_session_ids") or qa.evidence
    gold_episodes = {f"lme:{qid}:{sid}" for sid in session_ids}
    for h in hits:
        ep = str(h.meta.get("episode_id") or "")
        if ep in gold_episodes:
            return True
    return False
