"""Shared benchmark pipeline for memory engine notebooks.

All dataset-specific logic lives in dataset_locomo.py / dataset_longmemeval.py / dataset_halumem.py / dataset_memorydoc.py.
Retrieval lives in memory_retrieval.py. Update those files — notebooks stay thin.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from memory_retrieval import (
    MODULE_VERSION,
    AtlasEntry,
    MemoryRetriever,
    RetrieverConfig,
    TraceEdge,
    answer_policy_for_query,
    build_memory_index,
)

# --- Ledger (Option B) ---


@dataclass
class MemoryDelta:
    id: str
    seq: int
    kind: str
    episode_id: str
    text: str
    observed_at: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    subject: str | None = None
    relation: str | None = None
    obj: str | None = None
    temporal_anchor: str | None = None
    confidence: float = 0.7
    strength_hot: float = 1.0
    strength_cold: float = 1.0
    access_count: int = 0
    edges: list[TraceEdge] = field(default_factory=list)


@dataclass
class CompiledClaim:
    claim_key: str
    text: str
    status: str
    source_delta_ids: list[str]
    temporal_anchor: str | None
    confidence: float


class Ledger:
    def __init__(self) -> None:
        self.deltas: list[MemoryDelta] = []
        self._seq = 0

    def append(self, **kwargs: Any) -> MemoryDelta:
        self._seq += 1
        delta = MemoryDelta(id=f"d{self._seq:05d}", seq=self._seq, **kwargs)
        self.deltas.append(delta)
        return delta

    def observations(self) -> list[MemoryDelta]:
        return [d for d in self.deltas if d.kind == "observation"]

    def extractions(self) -> list[MemoryDelta]:
        return [d for d in self.deltas if d.kind == "extraction"]


# --- Formation / compilation / Atlas ---

EXTRACTION_SYSTEM = """You extract factual statements from a chat session for a memory ledger.
Return JSON: {"facts": [{"subject": str, "relation": str, "object": str, "text": str, "temporal_expression": str|null, "confidence": float}]}
Rules:
- One fact per distinct claim. Use session timestamp to resolve relative dates when possible.
- Do NOT deduplicate or resolve conflicts — append everything you see.
- text should be a clean standalone sentence.
"""

ATLAS_SYSTEM = """You consolidate one conversation episode into a durable Atlas summary.
Return JSON: {
  "concept_or_topic": str,
  "summary": str,
  "alt_phrasings": [str, str, str]
}
Rules:
- summary: 2-4 sentences covering key facts, entities, dates, and outcomes for this episode.
- alt_phrasings: 2-3 short questions or phrases someone might use to find this episode later.
- Resolve relative dates using the episode timestamp when possible.
- Do not invent facts not present in the source material.
"""


def resolve_temporal(expression: str | None, session_ts: str | None) -> str | None:
    if not expression:
        return session_ts
    expr = expression.lower().strip()
    if session_ts and expr in {"today", "now", "this week"}:
        return session_ts
    return expression


def compile_claims(extractions: list[MemoryDelta]) -> list[CompiledClaim]:
    buckets: dict[str, list[MemoryDelta]] = defaultdict(list)
    for ext in extractions:
        key = f"{(ext.subject or '').lower()}|{(ext.relation or '').lower()}"
        if not ext.text:
            continue
        buckets[key].append(ext)

    claims: list[CompiledClaim] = []
    for key, group in buckets.items():
        group = sorted(group, key=lambda d: d.temporal_anchor or d.observed_at or "")
        latest = group[-1]
        objects = {g.obj for g in group if g.obj}
        status = "CONTESTED" if len(objects) > 1 else "ACTIVE"
        claims.append(
            CompiledClaim(
                claim_key=key,
                text=latest.text,
                status=status,
                source_delta_ids=[g.id for g in group],
                temporal_anchor=latest.temporal_anchor,
                confidence=min(1.0, 0.5 + 0.1 * len(group)),
            )
        )
    return claims


def build_episode_atlas(ledger: Ledger, gemini_text: Callable[..., str], parse_json: Callable[[str], Any]) -> list[AtlasEntry]:
    by_episode: dict[str, list[MemoryDelta]] = defaultdict(list)
    for d in ledger.deltas:
        by_episode[d.episode_id].append(d)

    atlas: list[AtlasEntry] = []
    for n, (episode_id, deltas) in enumerate(sorted(by_episode.items()), start=1):
        ordered = sorted(deltas, key=lambda x: x.seq)
        observed_at = next((d.observed_at for d in ordered if d.observed_at), None)
        lines = [f"[{d.kind.upper()}] {d.text}" for d in ordered]
        user = f"Episode: {episode_id}\nTimestamp: {observed_at or 'unknown'}\n\n" + "\n".join(lines)
        raw = gemini_text(ATLAS_SYSTEM, user, json_mode=True)
        try:
            payload = parse_json(raw)
        except Exception as exc:
            print("  atlas parse failed:", episode_id, exc)
            payload = {
                "concept_or_topic": episode_id,
                "summary": " ".join(d.text for d in ordered if d.kind == "observation")[:500],
                "alt_phrasings": [],
            }
        atlas.append(
            AtlasEntry(
                id=f"atlas_{n:04d}",
                concept_or_topic=str(payload.get("concept_or_topic") or episode_id),
                summary=str(payload.get("summary") or "").strip(),
                alt_phrasings=[str(p).strip() for p in (payload.get("alt_phrasings") or []) if str(p).strip()][:3],
                source_trace_ids=[d.id for d in ordered],
                observed_at=observed_at,
                stale=False,
                build_version=1,
            )
        )
    return atlas


def form_session(
    ledger: Ledger,
    *,
    episode_id: str,
    session_ts: str | None,
    turns: list[tuple[str, str | None]],
    gemini_text: Callable[..., str],
    parse_json: Callable[[str], Any],
    obs_id_map: dict[str, str] | None = None,
) -> None:
    """Ingest one episode. turns = [(passage_text, optional_external_id), ...]."""
    obs_ids: list[str] = []
    passage_lines: list[str] = []
    for text, external_id in turns:
        obs = ledger.append(
            kind="observation",
            episode_id=episode_id,
            text=text,
            observed_at=session_ts,
        )
        obs_ids.append(obs.id)
        passage_lines.append(text)
        if obs_id_map is not None and external_id:
            obs_id_map[external_id] = obs.id

    user = f"Session timestamp: {session_ts}\nEpisode: {episode_id}\n\n" + "\n".join(passage_lines)
    raw = gemini_text(EXTRACTION_SYSTEM, user, json_mode=True)
    try:
        facts = parse_json(raw).get("facts", [])
    except Exception as exc:
        print("  extraction parse failed:", episode_id, exc)
        facts = []

    for fact in facts:
        ledger.append(
            kind="extraction",
            episode_id=episode_id,
            text=str(fact.get("text") or "").strip(),
            observed_at=session_ts,
            evidence_refs=obs_ids[:3],
            subject=str(fact.get("subject") or "").strip() or None,
            relation=str(fact.get("relation") or "").strip() or None,
            obj=str(fact.get("object") or "").strip() or None,
            temporal_anchor=resolve_temporal(fact.get("temporal_expression"), session_ts),
            confidence=float(fact.get("confidence") or 0.7),
        )


@dataclass
class MemoryPipeline:
    """Built ledger + index + retriever."""

    ledger: Ledger
    compiled: list[CompiledClaim]
    atlas_entries: list[AtlasEntry]
    memory_index: Any
    retriever: MemoryRetriever


def build_pipeline(
    ledger: Ledger,
    *,
    embed_model: Any,
    sbert_model: str,
    retriever_config: RetrieverConfig | None = None,
) -> MemoryPipeline:
    compiled = compile_claims(ledger.extractions())
    memory_index = build_memory_index(
        deltas=ledger.deltas,
        claims=compiled,
        atlas_entries=[],  # atlas added separately
        embed_model=embed_model,
        embed_model_name=sbert_model,
    )
    return MemoryPipeline(
        ledger=ledger,
        compiled=compiled,
        atlas_entries=[],
        memory_index=memory_index,
        retriever=MemoryRetriever(
            memory_index,
            embed_model,
            extractions=ledger.extractions(),
            compile_fn=compile_claims,
            config=retriever_config or RetrieverConfig(),
        ),
    )


def finalize_pipeline(
    pipeline: MemoryPipeline,
    atlas_entries: list[AtlasEntry],
    *,
    embed_model: Any,
    sbert_model: str,
    retriever_config: RetrieverConfig | None = None,
) -> MemoryPipeline:
    memory_index = build_memory_index(
        deltas=pipeline.ledger.deltas,
        claims=pipeline.compiled,
        atlas_entries=atlas_entries,
        embed_model=embed_model,
        embed_model_name=sbert_model,
    )
    pipeline.atlas_entries = atlas_entries
    pipeline.memory_index = memory_index
    pipeline.retriever = MemoryRetriever(
        memory_index,
        embed_model,
        extractions=pipeline.ledger.extractions(),
        compile_fn=compile_claims,
        config=retriever_config or RetrieverConfig(),
    )
    return pipeline


# --- Answer / judge ---

ANSWER_SYSTEM_BASE = """You answer questions using ONLY the provided memory context.
Return JSON: {"answer": str, "abstain": bool}

Context contains grounded trace records only ([observation] and [extraction] are primary evidence;
[compiled] items are derived summaries — prefer observations/extractions when they disagree).

Rules:
- Give a SHORT direct answer (phrase or short sentence) unless the question asks for a list.
- Only abstain when there is truly zero relevant context.
"""

JUDGE_SYSTEM = """You grade whether a predicted answer is correct for a benchmark question.
Return JSON: {"correct": bool, "reason": str}
Rules:
- Mark correct=true if the prediction conveys the same factual answer as the gold reference, even if wording differs.
- Accept paraphrases, minor formatting differences, and equivalent date/time expressions.
- Mark correct=false if the prediction is wrong, missing the key fact, contradicts gold, or abstains when gold expects an answer.
- Be strict on factual content, lenient on phrasing.
"""


def answer_question(
    question: str,
    hits: list,
    gemini_text: Callable[..., str],
    parse_json: Callable[[str], Any],
    *,
    retrieval_meta: dict | None = None,
) -> dict[str, Any]:
    if not hits:
        return {"answer": "", "abstain": True, "note": "no retrieval hits"}

    policy = answer_policy_for_query(question)
    context_lines = []
    for h in hits:
        date = h.meta.get("observed_at") or h.meta.get("temporal_anchor") or "unknown date"
        contested = " [CONTESTED]" if h.meta.get("status") == "CONTESTED" else ""
        context_lines.append(f"[{h.kind} id={h.item_id}{contested}] ({date}) {h.text}")

    context = "\n".join(context_lines)
    meta_block = ""
    if retrieval_meta:
        meta_block = (
            f"\nRetrieval stage: {retrieval_meta.get('stage')} | "
            f"query_kind: {retrieval_meta.get('query_kind')} | "
            f"drilled_trace: {retrieval_meta.get('drilled_trace_count')}"
        )

    system = ANSWER_SYSTEM_BASE + "\nQuery-specific policy:\n" + policy["instruction"]
    user = f"Question: {question}{meta_block}\n\nContext:\n{context}"
    raw = gemini_text(system, user, json_mode=True)
    try:
        payload = parse_json(raw)
        payload["query_kind"] = policy["query_kind"]
        return payload
    except Exception:
        return {"answer": raw, "abstain": False, "query_kind": policy["query_kind"]}


def judge_answer(
    question: str,
    pred: str,
    gold: str,
    gemini_text: Callable[..., str],
    parse_json: Callable[[str], Any],
) -> dict[str, Any]:
    user = f"Question: {question}\nGold answer: {gold}\nPredicted answer: {pred}\n"
    raw = gemini_text(JUDGE_SYSTEM, user, json_mode=True)
    try:
        payload = parse_json(raw)
        return {"correct": bool(payload.get("correct")), "reason": str(payload.get("reason") or "")}
    except Exception as exc:
        return {"correct": False, "reason": f"judge parse failed: {exc}"}


@dataclass
class BenchmarkQuestion:
    question_id: str
    question: str
    gold: str
    category: Any = None
    ability: str | None = None
    evidence: list[str] = field(default_factory=list)
    episode_prefix: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def run_benchmark(
    pipeline: MemoryPipeline,
    questions: list[BenchmarkQuestion],
    *,
    gemini_text: Callable[..., str],
    parse_json: Callable[[str], Any],
    evidence_fn: Callable[[BenchmarkQuestion, list], bool],
    verbose: bool = True,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for qi, qa in enumerate(questions):
        retrieval = pipeline.retriever.retrieve(qa.question, episode_prefix=qa.episode_prefix)
        hits = retrieval.hits
        diag = retrieval.diagnostics
        ev_ok = evidence_fn(qa, hits)

        pred_payload = answer_question(
            qa.question,
            hits,
            gemini_text,
            parse_json,
            retrieval_meta={
                "stage": retrieval.stage,
                "query_kind": diag.get("query_kind"),
                "drilled_trace_count": diag.get("drilled_trace_count"),
            },
        )
        pred = str(pred_payload.get("answer") or "").strip()
        verdict = judge_answer(qa.question, pred, qa.gold, gemini_text, parse_json)

        row = {
            "question_id": qa.question_id,
            "q_index": qi,
            "category": qa.category,
            "ability": qa.ability,
            "question": qa.question,
            "gold": qa.gold,
            "pred": pred,
            "abstain": bool(pred_payload.get("abstain")),
            "correct": bool(verdict["correct"]),
            "judge_reason": verdict["reason"],
            "query_kind": diag.get("query_kind"),
            "retrieval_stage": retrieval.stage,
            "retrieval_confident": retrieval.confident,
            "retrieval_escalated": retrieval.escalated,
            "refined_query": retrieval.refined_query,
            "routing_seeds": diag.get("routing_seeds"),
            "drilled_trace_count": diag.get("drilled_trace_count"),
            "grounded_observations": diag.get("grounded_observation_count"),
            "grounded_extractions": diag.get("grounded_extraction_count"),
            "top_score": diag.get("top_score"),
            "context_items": len(hits),
            "evidence_in_context": ev_ok,
            **qa.extra,
        }
        results.append(row)
        if verbose:
            mark = "✓" if row["correct"] else "✗"
            print(f"{mark} [{qa.category}|{diag.get('query_kind')}] {qa.question[:65]}...")
            print(f"   gold: {qa.gold[:80]} | pred: {pred[:80] or '(empty)'}")
            print(
                f"   retrieval: stage={retrieval.stage} drilled={diag.get('drilled_trace_count')} ev={ev_ok}"
            )
    return results


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    if not n:
        return {"questions": 0}

    by_cat: dict[str, list[bool]] = defaultdict(list)
    by_qkind: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_cat[str(r.get("category"))].append(r["correct"])
        by_qkind[str(r.get("query_kind") or "unknown")].append(r["correct"])

    return {
        "questions": n,
        "gemini_judge_accuracy": sum(r["correct"] for r in results) / n,
        "gold_evidence_in_context_rate": sum(r["evidence_in_context"] for r in results) / n,
        "empty_prediction_rate": sum(1 for r in results if not r.get("pred")) / n,
        "abstention_rate": sum(r.get("abstain") for r in results) / n,
        "query_lens_escalation_rate": sum(r.get("retrieval_escalated") for r in results) / n,
        "avg_drilled_trace_count": sum(r.get("drilled_trace_count") or 0 for r in results) / n,
        "category_accuracy": {k: sum(v) / len(v) for k, v in sorted(by_cat.items())},
        "query_kind_accuracy": {k: sum(v) / len(v) for k, v in sorted(by_qkind.items())},
    }


def save_run(
    path: Path,
    *,
    benchmark: str,
    architecture: str,
    config: dict[str, Any],
    pipeline: MemoryPipeline,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "architecture": architecture,
        "benchmark": benchmark,
        "config": config,
        "summary": {
            **summary,
            "ledger_deltas": len(pipeline.ledger.deltas),
            "ledger_extractions": len(pipeline.ledger.extractions()),
            "compiled_claims": len(pipeline.compiled),
            "atlas_entries": len(pipeline.atlas_entries),
            "index_items": len(pipeline.memory_index.items),
        },
        "results": results,
    }
    out = path / f"{benchmark}_{payload['run_id']}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def parse_json_loose(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return json.loads(text)
