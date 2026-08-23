"""SQLite persistence for ordered Trace evidence and disposable Recall data."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from trisynapse_memory.engine.models import (
    Actor,
    ModelConfiguration,
    ModelDescriptor,
    EpisodeRecallView,
    MemoryDelta,
    MemoryJob,
    MemoryNamespace,
    IngestionRun,
    RemoveResult,
    RecallSnapshot,
    QueryRun,
    QueryRunPage,
    RetrievalConfiguration,
    RetrievalTrace,
    SourceRecord,
    SnapshotDiff,
    StoreValidation,
)
from trisynapse_memory.engine.retrieval.contracts import RetrievalDocument
from trisynapse_memory.engine.utils import bm25_term_score
from trisynapse_memory.engine.retrieval.tokenization import LEXICAL_TOKENIZER_VERSION, lexical_tokens

def _json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _parse_datetime(value: str | datetime | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _new_id(prefix: str = "d") -> str:
    millis = int(time.time() * 1000)
    return f"{prefix}_{millis:013x}{secrets.token_hex(6)}"


def _field_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        return " ".join(_field_text(item) for item in value if _field_text(item))
    if isinstance(value, dict):
        return " ".join(f"{key} {_field_text(item)}" for key, item in value.items())
    return str(value)


def _normal_index_entity(value: str | None) -> str:
    if not value:
        return ""
    normalized = " ".join(value.casefold().split())
    if normalized in {"i", "you", "he", "she", "they", "we", "it", "user", "speaker"}:
        return ""
    return normalized


class SQLiteTraceStore:
    """Durable ordered Trace with transactional sequence assignment."""

    def __init__(self, root: str | Path) -> None:
        root_path = Path(root).expanduser()
        if root_path.suffix in {".sqlite", ".sqlite3", ".db"}:
            self.root = root_path.parent
            self.db_path = root_path
        else:
            self.root = root_path
            self.db_path = root_path / "trace.sqlite3"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.root.chmod(0o700)
        except OSError:
            pass
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA secure_delete=ON")
        self._migrate()
        try:
            self.db_path.chmod(0o600)
        except OSError:
            pass

    def close(self) -> None:
        self._connection.close()

    def _migrate(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS deltas (
                id TEXT PRIMARY KEY,
                seq INTEGER NOT NULL UNIQUE,
                written_at TEXT NOT NULL,
                observed_at TEXT,
                kind TEXT NOT NULL CHECK(kind IN ('observation','extraction','annotation','access','retraction')),
                actor_json TEXT NOT NULL,
                namespace_json TEXT NOT NULL DEFAULT '{"project_id":"default"}',
                episode_id TEXT,
                evidence_refs_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                privacy_scope_json TEXT NOT NULL,
                scope_json TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                text TEXT NOT NULL,
                subject TEXT,
                relation TEXT,
                object TEXT,
                temporal_anchor TEXT,
                source_ref_json TEXT,
                locator_json TEXT,
                external_key TEXT UNIQUE
            );
            CREATE INDEX IF NOT EXISTS idx_deltas_episode ON deltas(episode_id, seq);
            CREATE INDEX IF NOT EXISTS idx_deltas_kind ON deltas(kind, seq);
            CREATE INDEX IF NOT EXISTS idx_deltas_observed ON deltas(observed_at);

            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 3,
                error TEXT,
                dedup_key TEXT UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at);

            CREATE TABLE IF NOT EXISTS removal_audit (
                remove_id TEXT PRIMARY KEY,
                target_ids_json TEXT NOT NULL,
                requested_by TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS recall_cache (
                cache_key TEXT PRIMARY KEY,
                view_type TEXT NOT NULL,
                scope_ref TEXT NOT NULL,
                evidence_hash TEXT NOT NULL,
                recall_config_id TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                stale INTEGER NOT NULL DEFAULT 0,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_recall_scope ON recall_cache(view_type, scope_ref);

            CREATE TABLE IF NOT EXISTS embedding_cache (
                text_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(text_hash, model)
            );

            CREATE TABLE IF NOT EXISTS retrieval_traces (
                query_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS query_runs (
                id TEXT PRIMARY KEY,
                namespace_key TEXT NOT NULL,
                query_text TEXT NOT NULL,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                stage TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_query_runs_namespace
                ON query_runs(namespace_key, created_at DESC, id DESC);
            CREATE INDEX IF NOT EXISTS idx_query_runs_status
                ON query_runs(namespace_key, status, created_at DESC);

            CREATE TABLE IF NOT EXISTS snapshots (
                id TEXT PRIMARY KEY,
                label TEXT,
                seq_cutoff INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS sources (
                id TEXT PRIMARY KEY,
                source_key TEXT NOT NULL,
                namespace_key TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sources_key ON sources(namespace_key, source_key, created_at);
            CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(namespace_key, status, created_at);

            CREATE TABLE IF NOT EXISTS ingestion_runs (
                id TEXT PRIMARY KEY,
                namespace_key TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ingestion_runs ON ingestion_runs(namespace_key, created_at);

            CREATE TABLE IF NOT EXISTS model_configuration (
                id INTEGER PRIMARY KEY CHECK(id=1),
                current_json TEXT NOT NULL,
                revision INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                pending_json TEXT,
                pending_job_id TEXT,
                last_error TEXT
            );

            CREATE TABLE IF NOT EXISTS retrieval_configuration (
                id INTEGER PRIMARY KEY CHECK(id=1),
                current_json TEXT NOT NULL,
                revision INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS provider_model_cache (
                cache_key TEXT PRIMARY KEY,
                fetched_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS retrieval_documents (
                delta_id TEXT PRIMARY KEY,
                namespace_key TEXT NOT NULL,
                seq INTEGER NOT NULL,
                kind TEXT NOT NULL,
                episode_id TEXT,
                modality TEXT NOT NULL,
                source_type TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                text TEXT NOT NULL,
                index_text TEXT NOT NULL,
                fields_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                observed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_retrieval_documents_namespace
                ON retrieval_documents(namespace_key, active, seq);
            CREATE INDEX IF NOT EXISTS idx_retrieval_documents_modality
                ON retrieval_documents(namespace_key, modality, active);
            CREATE TABLE IF NOT EXISTS retrieval_terms (
                delta_id TEXT NOT NULL,
                namespace_key TEXT NOT NULL,
                term TEXT NOT NULL,
                term_frequency INTEGER NOT NULL,
                PRIMARY KEY(delta_id, term)
            );
            CREATE INDEX IF NOT EXISTS idx_retrieval_terms_lookup
                ON retrieval_terms(namespace_key, term, delta_id);
            CREATE TABLE IF NOT EXISTS retrieval_graph_edges (
                namespace_key TEXT NOT NULL,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_kind TEXT NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY(source_id, target_id, edge_kind)
            );
            CREATE INDEX IF NOT EXISTS idx_retrieval_edges_source
                ON retrieval_graph_edges(namespace_key, source_id);
            CREATE TABLE IF NOT EXISTS retrieval_index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        tables = {row["name"] for row in self._connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        self._migrate_delta_schema()
        self._migrate_removal_audit_schema()
        self._migrate_snapshot_schema()
        if "purge_audit" in tables:
            self._connection.execute(
                """INSERT OR IGNORE INTO removal_audit
                   SELECT purge_id,target_ids_json,requested_by,reason,created_at
                   FROM purge_audit"""
            )
            self._connection.execute("DROP TABLE purge_audit")
        retrieval_columns = {
            row["name"] for row in self._connection.execute("PRAGMA table_info(retrieval_documents)").fetchall()
        }
        if "text" not in retrieval_columns:
            self._connection.execute(
                "ALTER TABLE retrieval_documents ADD COLUMN text TEXT NOT NULL DEFAULT ''"
            )
            self._connection.execute(
                """UPDATE retrieval_documents
                   SET text=COALESCE((SELECT d.text FROM deltas d WHERE d.id=retrieval_documents.delta_id),'')"""
            )
        tokenizer_row = self._connection.execute(
            "SELECT value FROM retrieval_index_metadata WHERE key='lexical_tokenizer_version'"
        ).fetchone()
        if tokenizer_row is None or tokenizer_row["value"] != LEXICAL_TOKENIZER_VERSION:
            self._connection.execute("DELETE FROM retrieval_terms")
            self._connection.execute("DELETE FROM retrieval_graph_edges")
            self._connection.execute("DELETE FROM retrieval_documents")
            self._connection.execute(
                """INSERT OR REPLACE INTO retrieval_index_metadata(key,value)
                   VALUES('lexical_tokenizer_version',?)""",
                (LEXICAL_TOKENIZER_VERSION,),
            )
        for row in self._connection.execute(
            "SELECT id,payload_json FROM ingestion_runs WHERE status='running'"
        ).fetchall():
            payload = json.loads(row["payload_json"])
            payload["status"] = "pending"
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._connection.execute(
                "UPDATE ingestion_runs SET status='pending',updated_at=?,payload_json=? WHERE id=?",
                (payload["updated_at"], _json(payload), row["id"]),
            )
        now = datetime.now(timezone.utc).isoformat()
        for row in self._connection.execute(
            "SELECT id,payload_json FROM query_runs WHERE status IN ('pending','running')"
        ).fetchall():
            payload = json.loads(row["payload_json"])
            payload["status"] = "interrupted"
            payload["error"] = "query execution was interrupted by a server restart"
            payload["updated_at"] = now
            self._connection.execute(
                "UPDATE query_runs SET status='interrupted',updated_at=?,payload_json=? WHERE id=?",
                (now, _json(payload), row["id"]),
            )
        existing_runs = {
            row["id"] for row in self._connection.execute("SELECT id FROM query_runs").fetchall()
        }
        retrieval_configuration = self.get_retrieval_configuration()
        for row in self._connection.execute(
            "SELECT query_id,created_at,payload_json FROM retrieval_traces"
        ).fetchall():
            if row["query_id"] in existing_runs:
                continue
            trace = RetrievalTrace.model_validate_json(row["payload_json"])
            run = QueryRun(
                id=trace.query_id,
                mode="search",
                status="completed",
                namespace=trace.namespace,
                query=trace.query,
                retrieval_trace=trace,
                retrieval_configuration=retrieval_configuration,
                partial=True,
                created_at=trace.created_at,
                updated_at=trace.created_at,
                completed_at=trace.created_at,
            )
            self._put_query_run_uncommitted(run)
        # The retrieval index is disposable and can be backfilled from Trace.
        for row in self._connection.execute(
            """SELECT d.* FROM deltas d
               LEFT JOIN retrieval_documents r ON r.delta_id=d.id
               WHERE r.delta_id IS NULL AND d.kind IN ('observation','extraction')"""
        ).fetchall():
            self._index_delta_uncommitted(self._connection, self._from_row(row))
        for row in self._connection.execute(
            "SELECT * FROM deltas WHERE kind='retraction' ORDER BY seq"
        ).fetchall():
            self._index_delta_uncommitted(self._connection, self._from_row(row))
        self._connection.commit()

    def _migrate_delta_schema(self) -> None:
        """Remove legacy chain columns while preserving ordered evidence."""

        columns = {
            row["name"] for row in self._connection.execute("PRAGMA table_info(deltas)").fetchall()
        }
        if "namespace_json" not in columns:
            self._connection.execute(
                "ALTER TABLE deltas ADD COLUMN namespace_json TEXT NOT NULL DEFAULT '{\"project_id\":\"default\"}'"
            )
            columns.add("namespace_json")
        # These column names exist only in pre-migration stores.
        legacy_chain_columns = {"prev_hash", "hash"}
        if not legacy_chain_columns.issubset(columns):
            return
        self._connection.executescript(
            """
            DROP TABLE IF EXISTS deltas_without_chain;
            CREATE TABLE deltas_without_chain (
                id TEXT PRIMARY KEY,
                seq INTEGER NOT NULL UNIQUE,
                written_at TEXT NOT NULL,
                observed_at TEXT,
                kind TEXT NOT NULL CHECK(kind IN ('observation','extraction','annotation','access','retraction')),
                actor_json TEXT NOT NULL,
                namespace_json TEXT NOT NULL DEFAULT '{"project_id":"default"}',
                episode_id TEXT,
                evidence_refs_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                privacy_scope_json TEXT NOT NULL,
                scope_json TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                text TEXT NOT NULL,
                subject TEXT,
                relation TEXT,
                object TEXT,
                temporal_anchor TEXT,
                source_ref_json TEXT,
                locator_json TEXT,
                external_key TEXT UNIQUE
            );
            INSERT INTO deltas_without_chain(
                id,seq,written_at,observed_at,kind,actor_json,namespace_json,episode_id,
                evidence_refs_json,confidence,privacy_scope_json,scope_json,payload_json,text,
                subject,relation,object,temporal_anchor,source_ref_json,locator_json,external_key
            ) SELECT
                id,seq,written_at,observed_at,kind,actor_json,namespace_json,episode_id,
                evidence_refs_json,confidence,privacy_scope_json,scope_json,payload_json,text,
                subject,relation,object,temporal_anchor,source_ref_json,locator_json,external_key
            FROM deltas ORDER BY seq;
            DROP TABLE deltas;
            ALTER TABLE deltas_without_chain RENAME TO deltas;
            CREATE INDEX idx_deltas_episode ON deltas(episode_id, seq);
            CREATE INDEX idx_deltas_kind ON deltas(kind, seq);
            CREATE INDEX idx_deltas_observed ON deltas(observed_at);
            """
        )

    def _migrate_removal_audit_schema(self) -> None:
        columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(removal_audit)").fetchall()
        }
        legacy_root_columns = {"old_root_hash", "new_root_hash"}
        if not legacy_root_columns.issubset(columns):
            return
        self._connection.executescript(
            """
            DROP TABLE IF EXISTS removal_audit_without_roots;
            CREATE TABLE removal_audit_without_roots (
                remove_id TEXT PRIMARY KEY,
                target_ids_json TEXT NOT NULL,
                requested_by TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            INSERT INTO removal_audit_without_roots(
                remove_id,target_ids_json,requested_by,reason,created_at
            ) SELECT remove_id,target_ids_json,requested_by,reason,created_at
              FROM removal_audit;
            DROP TABLE removal_audit;
            ALTER TABLE removal_audit_without_roots RENAME TO removal_audit;
            """
        )

    def _migrate_snapshot_schema(self) -> None:
        columns = {
            row["name"] for row in self._connection.execute("PRAGMA table_info(snapshots)").fetchall()
        }
        legacy_evidence_column = "evidence_hash"
        if legacy_evidence_column not in columns:
            return
        self._connection.executescript(
            """
            DROP TABLE IF EXISTS snapshots_without_roots;
            CREATE TABLE snapshots_without_roots (
                id TEXT PRIMARY KEY,
                label TEXT,
                seq_cutoff INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO snapshots_without_roots(id,label,seq_cutoff,created_at,active)
                SELECT id,label,seq_cutoff,created_at,active FROM snapshots;
            DROP TABLE snapshots;
            ALTER TABLE snapshots_without_roots RENAME TO snapshots;
            """
        )

    def get_retrieval_configuration(self) -> RetrievalConfiguration:
        row = self._connection.execute(
            "SELECT current_json FROM retrieval_configuration WHERE id=1"
        ).fetchone()
        if row:
            return RetrievalConfiguration.model_validate_json(row["current_json"])
        value = RetrievalConfiguration()
        self._connection.execute(
            "INSERT INTO retrieval_configuration(id,current_json,revision,updated_at) VALUES(1,?,?,?)",
            (value.model_dump_json(), value.revision, value.updated_at.isoformat()),
        )
        self._connection.commit()
        return value

    def save_retrieval_configuration(
        self,
        configuration: RetrievalConfiguration,
        *,
        expected_revision: int,
    ) -> RetrievalConfiguration:
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            row = cursor.execute(
                "SELECT revision FROM retrieval_configuration WHERE id=1"
            ).fetchone()
            current_revision = int(row["revision"]) if row else 0
            if expected_revision != current_revision:
                self._connection.rollback()
                raise ValueError(
                    f"retrieval configuration revision conflict: expected {expected_revision}, current {current_revision}"
                )
            value = configuration.model_copy(deep=True)
            value.revision = current_revision + 1
            value.updated_at = datetime.now(timezone.utc)
            cursor.execute(
                """INSERT INTO retrieval_configuration(id,current_json,revision,updated_at)
                   VALUES(1,?,?,?) ON CONFLICT(id) DO UPDATE SET
                   current_json=excluded.current_json,revision=excluded.revision,updated_at=excluded.updated_at""",
                (value.model_dump_json(), value.revision, value.updated_at.isoformat()),
            )
            self._connection.commit()
            return value

    def get_model_configuration(self) -> ModelConfiguration:
        row = self._connection.execute(
            "SELECT current_json FROM model_configuration WHERE id=1"
        ).fetchone()
        if row:
            return ModelConfiguration.model_validate_json(row["current_json"])
        configuration = ModelConfiguration()
        self._connection.execute(
            "INSERT INTO model_configuration(id,current_json,revision,updated_at) VALUES(1,?,?,?)",
            (configuration.model_dump_json(), configuration.revision, configuration.updated_at.isoformat()),
        )
        self._connection.commit()
        return configuration

    def get_model_configuration_state(
        self,
    ) -> tuple[ModelConfiguration, ModelConfiguration | None, str | None, str | None]:
        current = self.get_model_configuration()
        row = self._connection.execute(
            "SELECT pending_json,pending_job_id,last_error FROM model_configuration WHERE id=1"
        ).fetchone()
        pending = ModelConfiguration.model_validate_json(row["pending_json"]) if row and row["pending_json"] else None
        return current, pending, row["pending_job_id"] if row else None, row["last_error"] if row else None

    def save_model_configuration(
        self,
        configuration: ModelConfiguration,
        *,
        expected_revision: int | None = None,
    ) -> ModelConfiguration:
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            row = cursor.execute(
                "SELECT current_json,revision,pending_json FROM model_configuration WHERE id=1"
            ).fetchone()
            if row is None:
                current = ModelConfiguration()
                current_revision = current.revision
            else:
                current_revision = int(row["revision"])
            if expected_revision is not None and expected_revision != current_revision:
                self._connection.rollback()
                raise ValueError(
                    f"model configuration revision conflict: expected {expected_revision}, current {current_revision}"
                )
            value = configuration.model_copy(deep=True)
            value.revision = current_revision + 1
            value.updated_at = datetime.now(timezone.utc)
            pending_json = row["pending_json"] if row else None
            if pending_json:
                pending = ModelConfiguration.model_validate_json(pending_json)
                pending.completion = value.completion
                pending.revision = value.revision
                pending.updated_at = value.updated_at
                pending_json = pending.model_dump_json()
            cursor.execute(
                """INSERT INTO model_configuration(id,current_json,revision,updated_at,pending_json,pending_job_id,last_error)
                   VALUES(1,?,?,?,?,NULL,NULL)
                   ON CONFLICT(id) DO UPDATE SET current_json=excluded.current_json,
                       revision=excluded.revision,updated_at=excluded.updated_at,
                       pending_json=excluded.pending_json,last_error=NULL""",
                (
                    value.model_dump_json(), value.revision,
                    value.updated_at.isoformat(), pending_json,
                ),
            )
            self._connection.commit()
            return value

    def stage_embedding_configuration(
        self,
        pending: ModelConfiguration,
        job_id: str,
    ) -> None:
        with self._lock:
            row = self._connection.execute(
                "SELECT pending_job_id FROM model_configuration WHERE id=1"
            ).fetchone()
            if row and row["pending_job_id"]:
                raise ValueError("an embedding rebuild is already pending")
            self._connection.execute(
                """UPDATE model_configuration
                   SET pending_json=?,pending_job_id=?,last_error=NULL WHERE id=1""",
                (pending.model_dump_json(), job_id),
            )
            self._connection.commit()

    def activate_pending_embedding(self, job_id: str) -> ModelConfiguration:
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            row = cursor.execute(
                "SELECT current_json,pending_json,pending_job_id,revision FROM model_configuration WHERE id=1"
            ).fetchone()
            if row is None or row["pending_job_id"] != job_id or not row["pending_json"]:
                self._connection.rollback()
                raise ValueError("embedding rebuild is no longer the active pending change")
            current = ModelConfiguration.model_validate_json(row["current_json"])
            pending = ModelConfiguration.model_validate_json(row["pending_json"])
            current.embedding = pending.embedding
            current.revision = int(row["revision"]) + 1
            current.updated_at = datetime.now(timezone.utc)
            cursor.execute(
                """UPDATE model_configuration SET current_json=?,revision=?,updated_at=?,
                       pending_json=NULL,pending_job_id=NULL,last_error=NULL WHERE id=1""",
                (current.model_dump_json(), current.revision, current.updated_at.isoformat()),
            )
            self._connection.commit()
            return current

    def fail_pending_embedding(self, job_id: str, error: str) -> None:
        self._connection.execute(
            """UPDATE model_configuration SET pending_json=NULL,pending_job_id=NULL,last_error=?
               WHERE id=1 AND pending_job_id=?""",
            (error, job_id),
        )
        self._connection.commit()

    def get_cached_models(self, cache_key: str) -> tuple[datetime, list[ModelDescriptor]] | None:
        row = self._connection.execute(
            "SELECT fetched_at,payload_json FROM provider_model_cache WHERE cache_key=?", (cache_key,)
        ).fetchone()
        if row is None:
            return None
        return _parse_datetime(row["fetched_at"]), [
            ModelDescriptor.model_validate(item) for item in json.loads(row["payload_json"])
        ]

    def put_cached_models(self, cache_key: str, models: list[ModelDescriptor]) -> None:
        now = datetime.now(timezone.utc)
        self._connection.execute(
            "INSERT OR REPLACE INTO provider_model_cache VALUES(?,?,?)",
            (
                cache_key,
                now.isoformat(),
                _json([item.model_dump(mode="json") for item in models]),
            ),
        )
        self._connection.commit()

    def find_cached_model(self, provider: str, model_id: str) -> ModelDescriptor | None:
        for row in self._connection.execute(
            "SELECT payload_json FROM provider_model_cache"
        ).fetchall():
            for item in json.loads(row["payload_json"]):
                model = ModelDescriptor.model_validate(item)
                if model.provider == provider and model.id == model_id:
                    return model
        return None

    def has_searchable_content(self) -> bool:
        row = self._connection.execute(
            """SELECT 1 FROM deltas
               WHERE kind IN ('observation','extraction') AND text!='' AND text!='[REMOVED]'
               LIMIT 1"""
        ).fetchone()
        return row is not None

    def append(
        self,
        *,
        kind: str,
        text: str = "",
        observed_at: datetime | str | None = None,
        actor: Actor | dict[str, Any] | None = None,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
        episode_id: str | None = None,
        evidence_refs: Iterable[str] = (),
        confidence: float = 0.7,
        scope: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        subject: str | None = None,
        relation: str | None = None,
        object: str | None = None,
        temporal_anchor: str | None = None,
        source_ref: dict[str, Any] | str | None = None,
        locator: dict[str, Any] | str | None = None,
        external_key: str | None = None,
    ) -> MemoryDelta:
        if external_key:
            existing = self.get_by_external_key(external_key)
            if existing is not None:
                return existing
        actor_model = actor if isinstance(actor, Actor) else Actor.model_validate(actor or {})
        namespace_model = namespace if isinstance(namespace, MemoryNamespace) else MemoryNamespace.model_validate(namespace or {})
        observed = _parse_datetime(observed_at)
        written = datetime.now(timezone.utc)
        evidence = list(dict.fromkeys(evidence_refs))
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                previous = cursor.execute("SELECT seq FROM deltas ORDER BY seq DESC LIMIT 1").fetchone()
                seq = int(previous["seq"]) + 1 if previous else 1
                delta = MemoryDelta(
                    id=_new_id(),
                    seq=seq,
                    written_at=written,
                    observed_at=observed,
                    kind=kind,
                    actor=actor_model,
                    namespace=namespace_model,
                    episode_id=episode_id,
                    evidence_refs=evidence,
                    confidence=confidence,
                    privacy_scope={},
                    scope=scope or {},
                    payload=payload or {},
                    text=text,
                    subject=subject,
                    relation=relation,
                    object=object,
                    temporal_anchor=temporal_anchor,
                    source_ref=source_ref,
                    locator=locator,
                    external_key=external_key,
                )
                cursor.execute(
                    """INSERT INTO deltas(
                           id,seq,written_at,observed_at,kind,actor_json,namespace_json,
                           episode_id,evidence_refs_json,confidence,privacy_scope_json,scope_json,payload_json,
                           text,subject,relation,object,temporal_anchor,source_ref_json,locator_json,external_key
                       ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    self._to_row(delta),
                )
                self._index_delta_uncommitted(cursor, delta)
                if kind != "access":
                    cursor.execute("UPDATE recall_cache SET stale=1 WHERE stale=0")
                self._connection.commit()
                return delta
            except sqlite3.IntegrityError:
                self._connection.rollback()
                if external_key:
                    existing = self.get_by_external_key(external_key)
                    if existing is not None:
                        return existing
                raise
            except Exception:
                self._connection.rollback()
                raise

    @staticmethod
    def _to_row(delta: MemoryDelta) -> tuple[Any, ...]:
        return (
            delta.id,
            delta.seq,
            delta.written_at.isoformat(),
            delta.observed_at.isoformat() if delta.observed_at else None,
            delta.kind,
            _json(delta.actor),
            _json(delta.namespace),
            delta.episode_id,
            _json(delta.evidence_refs),
            delta.confidence,
            _json(delta.privacy_scope),
            _json(delta.scope),
            _json(delta.payload),
            delta.text,
            delta.subject,
            delta.relation,
            delta.object,
            delta.temporal_anchor,
            _json(delta.source_ref) if delta.source_ref is not None else None,
            _json(delta.locator) if delta.locator is not None else None,
            delta.external_key,
        )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> MemoryDelta:
        return MemoryDelta(
            id=row["id"],
            seq=row["seq"],
            written_at=_parse_datetime(row["written_at"]),
            observed_at=_parse_datetime(row["observed_at"]),
            kind=row["kind"],
            actor=json.loads(row["actor_json"]),
            namespace=json.loads(row["namespace_json"]) if "namespace_json" in row.keys() else {},
            episode_id=row["episode_id"],
            evidence_refs=json.loads(row["evidence_refs_json"]),
            confidence=row["confidence"],
            privacy_scope=json.loads(row["privacy_scope_json"]),
            scope=json.loads(row["scope_json"]),
            payload=json.loads(row["payload_json"]),
            text=row["text"],
            subject=row["subject"],
            relation=row["relation"],
            object=row["object"],
            temporal_anchor=row["temporal_anchor"],
            source_ref=json.loads(row["source_ref_json"]) if row["source_ref_json"] else None,
            locator=json.loads(row["locator_json"]) if row["locator_json"] else None,
            external_key=row["external_key"],
        )

    def get(self, delta_id: str) -> MemoryDelta | None:
        row = self._connection.execute("SELECT * FROM deltas WHERE id=?", (delta_id,)).fetchone()
        return self._from_row(row) if row else None

    def get_by_external_key(self, external_key: str) -> MemoryDelta | None:
        row = self._connection.execute("SELECT * FROM deltas WHERE external_key=?", (external_key,)).fetchone()
        return self._from_row(row) if row else None

    def list_deltas(
        self,
        *,
        kinds: Iterable[str] | None = None,
        episode_prefix: str | None = None,
        scope: dict[str, Any] | None = None,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
        seq_cutoff: int | None = None,
        include_retracted: bool = False,
        after_seq: int | None = None,
        limit: int | None = None,
    ) -> list[MemoryDelta]:
        clauses: list[str] = []
        params: list[Any] = []
        if kinds:
            kind_list = list(kinds)
            clauses.append(f"kind IN ({','.join('?' for _ in kind_list)})")
            params.extend(kind_list)
        if episode_prefix:
            clauses.append("episode_id LIKE ?")
            params.append(f"{episode_prefix}%")
        if seq_cutoff is not None:
            clauses.append("seq <= ?")
            params.append(seq_cutoff)
        if after_seq is not None:
            clauses.append("seq > ?")
            params.append(after_seq)
        sql = "SELECT * FROM deltas"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY seq"
        deltas = [self._from_row(row) for row in self._connection.execute(sql, params).fetchall()]
        if not include_retracted:
            targets = self.retracted_ids()
            deltas = [delta for delta in deltas if delta.id not in targets]
        if scope:
            deltas = [delta for delta in deltas if _scope_matches(delta.scope, scope)]
        if namespace is not None:
            requested_namespace = namespace if isinstance(namespace, MemoryNamespace) else MemoryNamespace.model_validate(namespace)
            requested = requested_namespace.model_dump(exclude_none=True)
            deltas = [
                delta for delta in deltas
                if all(getattr(delta.namespace, key) == value for key, value in requested.items())
            ]
        return deltas[:limit] if limit is not None else deltas

    def retracted_ids(self) -> set[str]:
        rows = self._connection.execute("SELECT payload_json FROM deltas WHERE kind='retraction'").fetchall()
        targets: set[str] = set()
        for row in rows:
            payload = json.loads(row["payload_json"])
            targets.update(payload.get("target_delta_ids") or [])
        return targets

    def max_seq(self) -> int:
        row = self._connection.execute("SELECT COALESCE(MAX(seq), 0) AS value FROM deltas").fetchone()
        return int(row["value"])

    def delta_count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) AS value FROM deltas").fetchone()
        return int(row["value"])

    def is_ready(self) -> bool:
        try:
            self._connection.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    def validate(self, *, check_source_blobs: bool = True) -> StoreValidation:
        """Check database structure, ordered records, references, and source bytes."""

        issues: list[str] = []
        malformed_delta_ids: list[str] = []
        broken_evidence_refs: list[str] = []
        missing_source_ids: list[str] = []
        corrupted_source_ids: list[str] = []
        source_blobs_checked = 0
        database_ok = False
        rows: list[sqlite3.Row] = []
        try:
            quick_check = [str(row[0]) for row in self._connection.execute("PRAGMA quick_check").fetchall()]
            database_ok = quick_check == ["ok"]
            if not database_ok:
                issues.extend(f"SQLite: {message}" for message in quick_check)
            rows = self._connection.execute("SELECT * FROM deltas ORDER BY seq").fetchall()
        except sqlite3.Error as exc:
            issues.append(f"SQLite validation failed: {exc}")

        parsed: list[MemoryDelta] = []
        sequence_contiguous = True
        for expected_seq, row in enumerate(rows, 1):
            delta_id = str(row["id"]) if "id" in row.keys() else f"row:{expected_seq}"
            try:
                if int(row["seq"]) != expected_seq:
                    sequence_contiguous = False
                parsed.append(self._from_row(row))
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                malformed_delta_ids.append(delta_id)
                issues.append(f"Malformed delta {delta_id}: {exc}")
        if not sequence_contiguous:
            issues.append("Trace sequence numbers are not contiguous from 1")

        delta_ids = {delta.id for delta in parsed}
        for delta in parsed:
            for evidence_id in delta.evidence_refs:
                if evidence_id not in delta_ids:
                    broken_evidence_refs.append(f"{delta.id}->{evidence_id}")
        if broken_evidence_refs:
            issues.append(f"{len(broken_evidence_refs)} evidence references point to missing deltas")

        if check_source_blobs and database_ok:
            source_rows = self._connection.execute(
                "SELECT id,payload_json FROM sources WHERE status!='removed' ORDER BY created_at"
            ).fetchall()
            source_root = (self.root / "sources").resolve()
            for row in source_rows:
                source_id = str(row["id"])
                try:
                    source = SourceRecord.model_validate_json(row["payload_json"])
                    path = (self.root / source.blob_path).resolve()
                    if source_root not in path.parents or not path.is_file():
                        missing_source_ids.append(source_id)
                        continue
                    source_blobs_checked += 1
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(block)
                    if digest.hexdigest() != source.content_hash:
                        corrupted_source_ids.append(source_id)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    corrupted_source_ids.append(source_id)
                    issues.append(f"Invalid source {source_id}: {exc}")
            if missing_source_ids:
                issues.append(f"{len(missing_source_ids)} retained source blobs are missing")
            if corrupted_source_ids:
                issues.append(f"{len(corrupted_source_ids)} retained source blobs failed SHA-256 validation")

        return StoreValidation(
            ok=database_ok and not issues,
            database_ok=database_ok,
            delta_count=len(rows),
            sequence_contiguous=sequence_contiguous,
            source_blobs_checked=source_blobs_checked,
            missing_source_ids=missing_source_ids,
            corrupted_source_ids=corrupted_source_ids,
            malformed_delta_ids=malformed_delta_ids,
            broken_evidence_refs=broken_evidence_refs,
            issues=issues,
        )

    def history(self, delta_id: str) -> list[MemoryDelta]:
        events = []
        for delta in self.list_deltas(include_retracted=True):
            targets = set(delta.evidence_refs)
            targets.update(delta.payload.get("target_delta_ids") or [])
            if delta.id == delta_id or delta_id in targets:
                events.append(delta)
        return events

    def hard_remove(self, delta_ids: list[str], *, requested_by: str, reason: str) -> RemoveResult:
        """Redact selected payloads and record the lifecycle operation."""

        targets = list(dict.fromkeys(delta_ids))
        if not targets:
            raise ValueError("remove requires at least one delta id")
        remove_id = _new_id("remove")
        created = datetime.now(timezone.utc)
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                placeholders = ",".join("?" for _ in targets)
                existing = {
                    str(row["id"])
                    for row in cursor.execute(
                        f"SELECT id FROM deltas WHERE id IN ({placeholders})", targets
                    ).fetchall()
                }
                missing = [item for item in targets if item not in existing]
                if missing:
                    raise KeyError(f"unknown deltas: {', '.join(missing)}")
                cursor.execute(
                    f"""UPDATE deltas SET
                        text='[REMOVED]', observed_at=NULL, actor_json=?, episode_id=NULL,
                        evidence_refs_json=?, scope_json=?, external_key=NULL,
                        subject=NULL, relation=NULL, object=NULL, temporal_anchor=NULL,
                        source_ref_json=NULL, locator_json=NULL,
                        payload_json=?, privacy_scope_json=?
                        WHERE id IN ({placeholders})""",
                    [
                        _json({"type": "external_api", "id": "remove"}), _json([]), _json({}),
                        _json({"removed": True, "remove_id": remove_id}), _json({}), *targets,
                    ],
                )
                cursor.execute(
                    "INSERT INTO removal_audit VALUES(?,?,?,?,?)",
                    (remove_id, _json(targets), requested_by, reason, created.isoformat()),
                )
                cursor.execute("DELETE FROM embedding_cache")
                cursor.execute("DELETE FROM recall_cache")
                cursor.execute(
                    f"DELETE FROM retrieval_terms WHERE delta_id IN ({placeholders})",
                    targets,
                )
                cursor.execute(
                    f"DELETE FROM retrieval_documents WHERE delta_id IN ({placeholders})",
                    targets,
                )
                cursor.execute(
                    f"DELETE FROM retrieval_graph_edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
                    [*targets, *targets],
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
            self._connection.execute("VACUUM")
            self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return RemoveResult(
            remove_id=remove_id,
            removed_delta_ids=targets,
            requested_by=requested_by,
            created_at=created,
        )

    def put_source(self, source: SourceRecord) -> None:
        key = _json(source.namespace.model_dump(mode="json", exclude_none=True))
        self._connection.execute(
            """INSERT OR REPLACE INTO sources(id,source_key,namespace_key,content_hash,status,created_at,payload_json)
               VALUES(?,?,?,?,?,?,?)""",
            (source.id, source.source_key, key, source.content_hash, source.status, source.created_at.isoformat(), source.model_dump_json()),
        )
        self._connection.commit()

    def get_source(self, source_id: str) -> SourceRecord | None:
        row = self._connection.execute("SELECT payload_json FROM sources WHERE id=?", (source_id,)).fetchone()
        return SourceRecord.model_validate_json(row["payload_json"]) if row else None

    def latest_source(self, source_key: str, namespace: MemoryNamespace) -> SourceRecord | None:
        key = _json(namespace.model_dump(mode="json", exclude_none=True))
        row = self._connection.execute(
            "SELECT payload_json FROM sources WHERE source_key=? AND namespace_key=? ORDER BY created_at DESC LIMIT 1",
            (source_key, key),
        ).fetchone()
        return SourceRecord.model_validate_json(row["payload_json"]) if row else None

    def list_sources(self, namespace: MemoryNamespace, *, include_removed: bool = False) -> list[SourceRecord]:
        key = _json(namespace.model_dump(mode="json", exclude_none=True))
        sql = "SELECT payload_json FROM sources WHERE namespace_key=?"
        params: list[Any] = [key]
        if not include_removed:
            sql += " AND status='active'"
        sql += " ORDER BY created_at DESC"
        return [SourceRecord.model_validate_json(row["payload_json"]) for row in self._connection.execute(sql, params).fetchall()]

    def blob_is_referenced(self, blob_path: str, *, excluding_source_id: str | None = None) -> bool:
        rows = self._connection.execute("SELECT id,payload_json FROM sources WHERE status!='removed'").fetchall()
        for row in rows:
            if excluding_source_id and row["id"] == excluding_source_id:
                continue
            if SourceRecord.model_validate_json(row["payload_json"]).blob_path == blob_path:
                return True
        return False

    def put_ingestion_run(self, run: IngestionRun) -> None:
        key = _json(run.namespace.model_dump(mode="json", exclude_none=True))
        self._connection.execute(
            """INSERT OR REPLACE INTO ingestion_runs(id,namespace_key,status,created_at,updated_at,payload_json)
               VALUES(?,?,?,?,?,?)""",
            (run.id, key, run.status, run.created_at.isoformat(), run.updated_at.isoformat(), run.model_dump_json()),
        )
        self._connection.commit()

    def get_ingestion_run(self, run_id: str) -> IngestionRun | None:
        row = self._connection.execute("SELECT payload_json FROM ingestion_runs WHERE id=?", (run_id,)).fetchone()
        return IngestionRun.model_validate_json(row["payload_json"]) if row else None

    def list_ingestion_runs(self, namespace: MemoryNamespace, *, limit: int = 100) -> list[IngestionRun]:
        key = _json(namespace.model_dump(mode="json", exclude_none=True))
        rows = self._connection.execute(
            "SELECT payload_json FROM ingestion_runs WHERE namespace_key=? ORDER BY created_at DESC LIMIT ?", (key, limit)
        ).fetchall()
        return [IngestionRun.model_validate_json(row["payload_json"]) for row in rows]

    def enqueue_job(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        dedup_key: str | None = None,
        max_attempts: int = 3,
    ) -> MemoryJob:
        if dedup_key:
            row = self._connection.execute(
                "SELECT * FROM jobs WHERE dedup_key=?", (dedup_key,)
            ).fetchone()
            if row:
                return self._job_from_row(row)
        now = datetime.now(timezone.utc)
        job = MemoryJob(id=_new_id("job"), kind=kind, payload=payload, max_attempts=max_attempts, created_at=now, updated_at=now)
        self._connection.execute(
            "INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?)",
            (job.id, job.kind, job.status, _json(job.payload), 0, max_attempts, None, dedup_key, now.isoformat(), now.isoformat()),
        )
        self._connection.commit()
        return job

    def claim_job(self) -> MemoryJob | None:
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            stale_before = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
            cursor.execute(
                """UPDATE jobs
                   SET status=CASE WHEN attempts < max_attempts THEN 'pending' ELSE 'failed' END,
                       error='worker lease expired; job recovered after interruption'
                   WHERE status='running' AND updated_at < ?""",
                (stale_before,),
            )
            row = cursor.execute(
                "SELECT * FROM jobs WHERE status='pending' AND attempts < max_attempts ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                self._connection.commit()
                return None
            now = datetime.now(timezone.utc).isoformat()
            cursor.execute(
                "UPDATE jobs SET status='running', attempts=attempts+1, updated_at=? WHERE id=?", (now, row["id"])
            )
            self._connection.commit()
        return self.get_job(str(row["id"]))

    def finish_job(self, job_id: str, *, error: str | None = None) -> MemoryJob:
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(f"unknown job: {job_id}")
        retry = error is not None and job.attempts < job.max_attempts
        status = "pending" if retry else ("failed" if error else "completed")
        self._connection.execute(
            "UPDATE jobs SET status=?, error=?, updated_at=? WHERE id=?",
            (status, error, datetime.now(timezone.utc).isoformat(), job_id),
        )
        self._connection.commit()
        value = self.get_job(job_id)
        assert value is not None
        return value

    def get_job(self, job_id: str) -> MemoryJob | None:
        row = self._connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._job_from_row(row) if row else None

    def list_jobs(self, *, status: str | None = None, limit: int = 100) -> list[MemoryJob]:
        if status:
            rows = self._connection.execute(
                "SELECT * FROM jobs WHERE status=? ORDER BY created_at DESC LIMIT ?", (status, limit)
            ).fetchall()
        else:
            rows = self._connection.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._job_from_row(row) for row in rows]

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> MemoryJob:
        return MemoryJob(
            id=row["id"], kind=row["kind"], status=row["status"], payload=json.loads(row["payload_json"]),
            attempts=row["attempts"], max_attempts=row["max_attempts"], error=row["error"],
            created_at=_parse_datetime(row["created_at"]), updated_at=_parse_datetime(row["updated_at"]),
        )

    def put_episode_recall(self, view: EpisodeRecallView, recall_config_id: str = "episode-recall-v1") -> None:
        evidence_hash = hashlib.sha256(
            _json(view.source_trace_ids).encode("utf-8")
        ).hexdigest()
        self._connection.execute(
            """INSERT INTO recall_cache(cache_key,view_type,scope_ref,evidence_hash,recall_config_id,generated_at,stale,payload_json)
               VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(cache_key) DO UPDATE SET generated_at=excluded.generated_at,stale=0,payload_json=excluded.payload_json""",
            (view.cache_key, "episode_recall", view.episode_id, evidence_hash, recall_config_id, view.generated_at.isoformat(), 0, view.model_dump_json()),
        )
        self._connection.commit()

    def episode_recall_views(
        self,
        *,
        include_stale: bool = True,
        namespace: MemoryNamespace | dict[str, Any] | None = None,
    ) -> list[EpisodeRecallView]:
        sql = "SELECT payload_json,stale FROM recall_cache WHERE view_type='episode_recall'"
        if not include_stale:
            sql += " AND stale=0"
        views: list[EpisodeRecallView] = []
        for row in self._connection.execute(sql).fetchall():
            view = EpisodeRecallView.model_validate_json(row["payload_json"])
            view.stale = bool(row["stale"])
            views.append(view)
        if namespace is not None:
            requested_namespace = namespace if isinstance(namespace, MemoryNamespace) else MemoryNamespace.model_validate(namespace)
            requested = requested_namespace.model_dump(exclude_none=True)
            views = [view for view in views if all(getattr(view.namespace, key) == value for key, value in requested.items())]
        return views

    def _index_delta_uncommitted(self, cursor: Any, delta: MemoryDelta) -> None:
        """Incrementally maintain disposable lexical/document/graph indexes."""

        if delta.kind == "retraction":
            targets = list(dict.fromkeys(delta.payload.get("target_delta_ids") or []))
            if targets:
                placeholders = ",".join("?" for _ in targets)
                cursor.execute(
                    f"UPDATE retrieval_documents SET active=0 WHERE delta_id IN ({placeholders})",
                    targets,
                )
                cursor.execute(
                    f"DELETE FROM retrieval_terms WHERE delta_id IN ({placeholders})",
                    targets,
                )
                cursor.execute(
                    f"DELETE FROM retrieval_graph_edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
                    [*targets, *targets],
                )
            return
        if delta.kind not in {"observation", "extraction"} or not delta.text or delta.text == "[REMOVED]":
            return

        namespace_key = _json(delta.namespace.model_dump(mode="json", exclude_none=True))
        raw_fields = delta.payload.get("retrieval_fields") or {}
        fields = {
            str(name): _field_text(value)
            for name, value in raw_fields.items()
            if _field_text(value)
        } if isinstance(raw_fields, dict) else {}
        modality = str(delta.payload.get("modality") or "text")
        source_type = str(
            delta.payload.get("source_type")
            or (delta.scope.get("source_kind") if isinstance(delta.scope, dict) else None)
            or modality
        )
        document = RetrievalDocument(
            id=delta.id,
            text=delta.text,
            modality=modality,
            source_type=source_type,
            fields=fields,
            metadata={
                "source_ref": delta.source_ref,
                "locator": delta.locator,
                "scope": delta.scope,
            },
        )
        tokens = lexical_tokens(document.index_text)
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        cursor.execute(
            """INSERT OR REPLACE INTO retrieval_documents(
                   delta_id,namespace_key,seq,kind,episode_id,modality,source_type,text_hash,
                   text,index_text,fields_json,metadata_json,token_count,active,observed_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,1,?)""",
            (
                delta.id, namespace_key, delta.seq, delta.kind, delta.episode_id,
                modality, source_type, hashlib.sha256(document.index_text.encode()).hexdigest(),
                delta.text, document.index_text, _json(fields), _json(document.metadata), len(tokens),
                delta.observed_at.isoformat() if delta.observed_at else None,
            ),
        )
        cursor.execute("DELETE FROM retrieval_terms WHERE delta_id=?", (delta.id,))
        cursor.executemany(
            "INSERT INTO retrieval_terms(delta_id,namespace_key,term,term_frequency) VALUES(?,?,?,?)",
            [(delta.id, namespace_key, term, frequency) for term, frequency in counts.items()],
        )

        if delta.episode_id:
            previous = cursor.execute(
                """SELECT delta_id FROM retrieval_documents
                   WHERE namespace_key=? AND episode_id=? AND active=1 AND seq<?
                   ORDER BY seq DESC LIMIT 1""",
                (namespace_key, delta.episode_id, delta.seq),
            ).fetchone()
            if previous:
                self._put_retrieval_edge_uncommitted(
                    cursor, namespace_key, str(previous["delta_id"]), delta.id, "followed_by", 1.0
                )
                self._put_retrieval_edge_uncommitted(
                    cursor, namespace_key, delta.id, str(previous["delta_id"]), "followed_by", 0.8
                )
        entities = {_normal_index_entity(delta.subject), _normal_index_entity(delta.object)} - {""}
        for entity in entities:
            rows = cursor.execute(
                """SELECT id FROM deltas
                   WHERE id!=? AND kind IN ('observation','extraction')
                     AND namespace_json=?
                     AND (lower(subject)=? OR lower(object)=?)
                   ORDER BY seq DESC LIMIT 6""",
                (delta.id, _json(delta.namespace), entity, entity),
            ).fetchall()
            for row in rows:
                other = str(row["id"])
                self._put_retrieval_edge_uncommitted(
                    cursor, namespace_key, delta.id, other, "about_same_entity", 0.92
                )
                self._put_retrieval_edge_uncommitted(
                    cursor, namespace_key, other, delta.id, "about_same_entity", 0.92
                )

    @staticmethod
    def _put_retrieval_edge_uncommitted(
        cursor: Any,
        namespace_key: str,
        source_id: str,
        target_id: str,
        edge_kind: str,
        weight: float,
    ) -> None:
        if source_id == target_id:
            return
        cursor.execute(
            """INSERT OR REPLACE INTO retrieval_graph_edges(
                   namespace_key,source_id,target_id,edge_kind,weight
               ) VALUES(?,?,?,?,?)""",
            (namespace_key, source_id, target_id, edge_kind, weight),
        )

    def put_retrieval_edges(
        self,
        namespace: MemoryNamespace | dict[str, Any],
        edges: Iterable[tuple[str, str, str, float]],
    ) -> None:
        namespace_model = namespace if isinstance(namespace, MemoryNamespace) else MemoryNamespace.model_validate(namespace)
        namespace_key = _json(namespace_model.model_dump(mode="json", exclude_none=True))
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                for source_id, target_id, edge_kind, weight in edges:
                    self._put_retrieval_edge_uncommitted(
                        cursor, namespace_key, source_id, target_id, edge_kind, weight
                    )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def retrieval_graph(
        self,
        namespace: MemoryNamespace | dict[str, Any],
        *,
        item_ids: Iterable[str] | None = None,
    ) -> dict[str, list[tuple[str, float, str]]]:
        namespace_model = namespace if isinstance(namespace, MemoryNamespace) else MemoryNamespace.model_validate(namespace)
        namespace_key = _json(namespace_model.model_dump(mode="json", exclude_none=True))
        params: list[Any] = [namespace_key]
        sql = """SELECT e.source_id,e.target_id,e.edge_kind,e.weight
                 FROM retrieval_graph_edges e
                 JOIN retrieval_documents s ON s.delta_id=e.source_id AND s.active=1
                 JOIN retrieval_documents t ON t.delta_id=e.target_id AND t.active=1
                 WHERE e.namespace_key=?"""
        selected = list(item_ids or [])
        if selected:
            placeholders = ",".join("?" for _ in selected)
            sql += f" AND e.source_id IN ({placeholders}) AND e.target_id IN ({placeholders})"
            params.extend(selected)
            params.extend(selected)
        adjacency: dict[str, list[tuple[str, float, str]]] = {}
        for row in self._connection.execute(sql, params).fetchall():
            adjacency.setdefault(str(row["source_id"]), []).append(
                (str(row["target_id"]), float(row["weight"]), str(row["edge_kind"]))
            )
        return adjacency

    def retrieval_index_statistics(
        self,
        namespace: MemoryNamespace | dict[str, Any],
    ) -> tuple[Counter[str], float, int]:
        namespace_model = namespace if isinstance(namespace, MemoryNamespace) else MemoryNamespace.model_validate(namespace)
        namespace_key = _json(namespace_model.model_dump(mode="json", exclude_none=True))
        row = self._connection.execute(
            """SELECT COUNT(*) AS n,COALESCE(AVG(token_count),0) AS avgdl
               FROM retrieval_documents WHERE namespace_key=? AND active=1""",
            (namespace_key,),
        ).fetchone()
        terms = self._connection.execute(
            """SELECT t.term,COUNT(*) AS df FROM retrieval_terms t
               JOIN retrieval_documents d ON d.delta_id=t.delta_id AND d.active=1
               WHERE t.namespace_key=? GROUP BY t.term""",
            (namespace_key,),
        ).fetchall()
        return Counter({str(item["term"]): int(item["df"]) for item in terms}), float(row["avgdl"]), int(row["n"])

    def lexical_candidates(
        self,
        query: str,
        namespace: MemoryNamespace | dict[str, Any],
        *,
        limit: int = 500,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[tuple[str, float]]:
        """Read BM25 candidates from the incremental postings index."""

        terms = list(dict.fromkeys(lexical_tokens(query)))
        if not terms:
            return []
        namespace_model = namespace if isinstance(namespace, MemoryNamespace) else MemoryNamespace.model_validate(namespace)
        namespace_key = _json(namespace_model.model_dump(mode="json", exclude_none=True))
        df, avgdl, document_count = self.retrieval_index_statistics(namespace_model)
        placeholders = ",".join("?" for _ in terms)
        rows = self._connection.execute(
            f"""SELECT t.delta_id,t.term,t.term_frequency,d.token_count
                 FROM retrieval_terms t
                 JOIN retrieval_documents d ON d.delta_id=t.delta_id AND d.active=1
                 WHERE t.namespace_key=? AND t.term IN ({placeholders})""",
            [namespace_key, *terms],
        ).fetchall()
        scores: dict[str, float] = {}
        for row in rows:
            term = str(row["term"])
            frequency = int(row["term_frequency"])
            length = int(row["token_count"])
            score = bm25_term_score(
                term_frequency=frequency,
                document_length=length,
                average_document_length=avgdl,
                document_count=document_count,
                document_frequency=df.get(term, 0),
                k1=k1,
                b=b,
            )
            identifier = str(row["delta_id"])
            scores[identifier] = scores.get(identifier, 0.0) + score
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]

    def retrieval_documents(
        self,
        namespace: MemoryNamespace | dict[str, Any],
    ) -> list[RetrievalDocument]:
        namespace_model = namespace if isinstance(namespace, MemoryNamespace) else MemoryNamespace.model_validate(namespace)
        namespace_key = _json(namespace_model.model_dump(mode="json", exclude_none=True))
        rows = self._connection.execute(
            """SELECT delta_id,text,index_text,modality,source_type,fields_json,metadata_json
               FROM retrieval_documents WHERE namespace_key=? AND active=1 ORDER BY seq""",
            (namespace_key,),
        ).fetchall()
        return [
            RetrievalDocument(
                id=str(row["delta_id"]),
                text=str(row["text"]),
                modality=str(row["modality"]),
                source_type=str(row["source_type"]),
                fields=json.loads(row["fields_json"]),
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]

    def get_embeddings(self, text_hashes: list[str], model: str) -> dict[str, list[float]]:
        if not text_hashes:
            return {}
        placeholders = ",".join("?" for _ in text_hashes)
        rows = self._connection.execute(
            f"SELECT text_hash,vector_json FROM embedding_cache WHERE model=? AND text_hash IN ({placeholders})",
            [model, *text_hashes],
        ).fetchall()
        return {row["text_hash"]: json.loads(row["vector_json"]) for row in rows}

    def put_embeddings(self, values: dict[str, list[float]], model: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._connection.executemany(
            "INSERT OR REPLACE INTO embedding_cache VALUES(?,?,?,?)",
            [(key, model, _json(vector), now) for key, vector in values.items()],
        )
        self._connection.commit()

    def list_embeddings(self, model: str) -> dict[str, list[float]]:
        rows = self._connection.execute(
            "SELECT text_hash,vector_json FROM embedding_cache WHERE model=?",
            (model,),
        ).fetchall()
        return {str(row["text_hash"]): json.loads(row["vector_json"]) for row in rows}

    def clear_embeddings(self) -> None:
        self._connection.execute("DELETE FROM embedding_cache")
        self._connection.commit()

    def write_retrieval_trace(self, trace: RetrievalTrace) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO retrieval_traces VALUES(?,?,?)",
            (trace.query_id, trace.created_at.isoformat(), trace.model_dump_json()),
        )
        self._connection.commit()

    def _put_query_run_uncommitted(self, run: QueryRun) -> None:
        namespace_key = _json(run.namespace.model_dump(mode="json", exclude_none=True))
        stage = run.retrieval_trace.stage if run.retrieval_trace else None
        self._connection.execute(
            """INSERT OR REPLACE INTO query_runs(
                   id,namespace_key,query_text,mode,status,stage,created_at,updated_at,payload_json
               ) VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                run.id,
                namespace_key,
                run.query,
                run.mode,
                run.status,
                stage,
                run.created_at.isoformat(),
                run.updated_at.isoformat(),
                run.model_dump_json(),
            ),
        )

    def put_query_run(self, run: QueryRun) -> QueryRun:
        with self._lock:
            self._put_query_run_uncommitted(run)
            self._connection.commit()
        return run

    def get_query_run(self, query_id: str) -> QueryRun | None:
        row = self._connection.execute(
            "SELECT payload_json FROM query_runs WHERE id=?", (query_id,)
        ).fetchone()
        return QueryRun.model_validate_json(row["payload_json"]) if row else None

    def list_query_runs(
        self,
        namespace: MemoryNamespace,
        *,
        limit: int = 50,
        cursor: str | None = None,
        search: str | None = None,
        mode: str | None = None,
        status: str | None = None,
        stage: str | None = None,
        before: datetime | None = None,
    ) -> QueryRunPage:
        key = _json(namespace.model_dump(mode="json", exclude_none=True))
        clauses = ["namespace_key=?"]
        params: list[Any] = [key]
        if cursor:
            cursor_row = self._connection.execute(
                "SELECT created_at,id FROM query_runs WHERE id=?", (cursor,)
            ).fetchone()
            if cursor_row:
                clauses.append("(created_at < ? OR (created_at = ? AND id < ?))")
                params.extend([cursor_row["created_at"], cursor_row["created_at"], cursor_row["id"]])
        if search:
            clauses.append("LOWER(query_text) LIKE ?")
            params.append(f"%{search.lower()}%")
        if mode:
            clauses.append("mode=?")
            params.append(mode)
        if status:
            clauses.append("status=?")
            params.append(status)
        if stage:
            clauses.append("stage=?")
            params.append(stage)
        if before:
            clauses.append("created_at < ?")
            params.append(before.isoformat())
        params.append(limit + 1)
        rows = self._connection.execute(
            f"SELECT id,payload_json FROM query_runs WHERE {' AND '.join(clauses)} "
            "ORDER BY created_at DESC,id DESC LIMIT ?",
            params,
        ).fetchall()
        values = [QueryRun.model_validate_json(row["payload_json"]) for row in rows[:limit]]
        return QueryRunPage(
            runs=values,
            next_cursor=rows[limit - 1]["id"] if len(rows) > limit else None,
        )

    def remove_query_runs(
        self,
        namespace: MemoryNamespace,
        *,
        query_ids: list[str] | None = None,
        before: datetime | None = None,
        all_in_namespace: bool = False,
    ) -> list[str]:
        key = _json(namespace.model_dump(mode="json", exclude_none=True))
        clauses = ["namespace_key=?"]
        params: list[Any] = [key]
        if query_ids:
            placeholders = ",".join("?" for _ in query_ids)
            clauses.append(f"id IN ({placeholders})")
            params.extend(query_ids)
        elif before:
            clauses.append("created_at < ?")
            params.append(before.isoformat())
        elif not all_in_namespace:
            return []
        rows = self._connection.execute(
            f"SELECT id FROM query_runs WHERE {' AND '.join(clauses)}", params
        ).fetchall()
        removed = [str(row["id"]) for row in rows]
        if removed:
            placeholders = ",".join("?" for _ in removed)
            self._connection.execute(f"DELETE FROM query_runs WHERE id IN ({placeholders})", removed)
            self._connection.execute(f"DELETE FROM retrieval_traces WHERE query_id IN ({placeholders})", removed)
            self._connection.commit()
        return removed

    def get_retrieval_trace(self, query_id: str) -> RetrievalTrace | None:
        run = self.get_query_run(query_id)
        if run and run.retrieval_trace:
            return run.retrieval_trace
        row = self._connection.execute("SELECT payload_json FROM retrieval_traces WHERE query_id=?", (query_id,)).fetchone()
        return RetrievalTrace.model_validate_json(row["payload_json"]) if row else None

    def create_snapshot(self, label: str | None = None) -> RecallSnapshot:
        cutoff = self.max_seq()
        snapshot = RecallSnapshot(id=_new_id("snap"), label=label, seq_cutoff=cutoff)
        self._connection.execute(
            "INSERT INTO snapshots VALUES(?,?,?,?,0)",
            (snapshot.id, snapshot.label, snapshot.seq_cutoff, snapshot.created_at.isoformat()),
        )
        self._connection.commit()
        return snapshot

    def list_snapshots(self) -> list[RecallSnapshot]:
        return [
            RecallSnapshot(
                id=row["id"], label=row["label"], seq_cutoff=row["seq_cutoff"],
                created_at=_parse_datetime(row["created_at"]), active=bool(row["active"]),
            )
            for row in self._connection.execute("SELECT * FROM snapshots ORDER BY created_at").fetchall()
        ]

    def snapshot_diff(self, a: str, b: str) -> SnapshotDiff:
        left, right = self._snapshot(a), self._snapshot(b)
        left_ids = {d.id for d in self.list_deltas(seq_cutoff=left.seq_cutoff, include_retracted=True)}
        right_ids = {d.id for d in self.list_deltas(seq_cutoff=right.seq_cutoff, include_retracted=True)}
        return SnapshotDiff(
            from_snapshot=a, to_snapshot=b, added_delta_ids=sorted(right_ids - left_ids),
            removed_delta_ids=sorted(left_ids - right_ids), from_seq=left.seq_cutoff, to_seq=right.seq_cutoff,
        )

    def activate_snapshot(self, snapshot_id: str) -> RecallSnapshot:
        snapshot = self._snapshot(snapshot_id)
        self._connection.execute("UPDATE snapshots SET active=0")
        self._connection.execute("UPDATE snapshots SET active=1 WHERE id=?", (snapshot_id,))
        self._connection.commit()
        snapshot.active = True
        return snapshot

    def active_seq_cutoff(self) -> int | None:
        row = self._connection.execute("SELECT seq_cutoff FROM snapshots WHERE active=1 LIMIT 1").fetchone()
        return int(row["seq_cutoff"]) if row else None

    def _snapshot(self, snapshot_id: str) -> RecallSnapshot:
        row = self._connection.execute("SELECT * FROM snapshots WHERE id=?", (snapshot_id,)).fetchone()
        if not row:
            raise KeyError(f"unknown snapshot: {snapshot_id}")
        return RecallSnapshot(
            id=row["id"], label=row["label"], seq_cutoff=row["seq_cutoff"],
            created_at=_parse_datetime(row["created_at"]), active=bool(row["active"]),
        )


def _scope_matches(candidate: dict[str, Any], requested: dict[str, Any]) -> bool:
    for key, expected in requested.items():
        actual = candidate.get(key)
        if isinstance(expected, list):
            actual_values = actual if isinstance(actual, list) else [actual]
            if not set(expected) & set(actual_values):
                return False
        elif actual != expected:
            return False
    return True
