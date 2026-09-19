import sqlite3
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

GENESIS_HASH = "0" * 64

class AuditLogger:
    def __init__(self, db_path: str = "artifacts/audit/audit_log.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    retrieved_chunk_ids TEXT NOT NULL,
                    regime_tags TEXT NOT NULL,
                    retrieval_index_hash TEXT NOT NULL,
                    adapter_hash TEXT NOT NULL,
                    confidence_scores TEXT NOT NULL,
                    verification_verdict TEXT NOT NULL,
                    final_answer TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    record_hash TEXT NOT NULL
                )
            """)
            conn.commit()

    def _get_last_hash(self) -> str:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT record_hash FROM audit_log ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                return row["record_hash"]
        return GENESIS_HASH

    @staticmethod
    def compute_record_hash(
        query_id: str,
        timestamp: str,
        query_text: str,
        retrieved_chunk_ids: List[str],
        regime_tags: List[str],
        retrieval_index_hash: str,
        adapter_hash: str,
        confidence_scores: List[float],
        verification_verdict: str,
        final_answer: str,
        previous_hash: str
    ) -> str:
        payload = {
            "query_id": query_id,
            "timestamp": timestamp,
            "query_text": query_text,
            "retrieved_chunk_ids": sorted(retrieved_chunk_ids),
            "regime_tags": sorted(regime_tags),
            "retrieval_index_hash": retrieval_index_hash,
            "adapter_hash": adapter_hash,
            "confidence_scores": [round(float(s), 4) for s in confidence_scores],
            "verification_verdict": verification_verdict,
            "final_answer": final_answer,
            "previous_hash": previous_hash
        }
        canonical_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def log_query(
        self,
        query_id: str,
        query_text: str,
        retrieved_chunk_ids: List[str],
        regime_tags: List[str],
        retrieval_index_hash: str,
        adapter_hash: str,
        confidence_scores: List[float],
        verification_verdict: str,
        final_answer: str,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        previous_hash = self._get_last_hash()
        record_hash = self.compute_record_hash(
            query_id=query_id,
            timestamp=timestamp,
            query_text=query_text,
            retrieved_chunk_ids=retrieved_chunk_ids,
            regime_tags=regime_tags,
            retrieval_index_hash=retrieval_index_hash,
            adapter_hash=adapter_hash,
            confidence_scores=confidence_scores,
            verification_verdict=verification_verdict,
            final_answer=final_answer,
            previous_hash=previous_hash
        )

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_log (
                    query_id, timestamp, query_text, retrieved_chunk_ids,
                    regime_tags, retrieval_index_hash, adapter_hash,
                    confidence_scores, verification_verdict, final_answer,
                    previous_hash, record_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                query_id,
                timestamp,
                query_text,
                json.dumps(retrieved_chunk_ids),
                json.dumps(regime_tags),
                retrieval_index_hash,
                adapter_hash,
                json.dumps(confidence_scores),
                verification_verdict,
                final_answer,
                previous_hash,
                record_hash
            ))
            conn.commit()

        return {
            "query_id": query_id,
            "timestamp": timestamp,
            "record_hash": record_hash,
            "previous_hash": previous_hash
        }

    def get_record(self, query_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM audit_log WHERE query_id = ?", (query_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d["retrieved_chunk_ids"] = json.loads(d["retrieved_chunk_ids"])
                d["regime_tags"] = json.loads(d["regime_tags"])
                d["confidence_scores"] = json.loads(d["confidence_scores"])
                return d
        return None

    def list_records(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["retrieved_chunk_ids"] = json.loads(d["retrieved_chunk_ids"])
                d["regime_tags"] = json.loads(d["regime_tags"])
                d["confidence_scores"] = json.loads(d["confidence_scores"])
                results.append(d)
            return results

    def verify_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Traverses the hash chain from genesis and verifies that no past records have been modified.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM audit_log ORDER BY id ASC")
            rows = cursor.fetchall()
            if not rows:
                return True, None

            expected_prev = GENESIS_HASH
            for row in rows:
                d = dict(row)
                chunk_ids = json.loads(d["retrieved_chunk_ids"])
                regime_tags = json.loads(d["regime_tags"])
                conf_scores = json.loads(d["confidence_scores"])

                if d["previous_hash"] != expected_prev:
                    return False, f"Broken chain at record {d['query_id']}: expected prev {expected_prev}, got {d['previous_hash']}"

                recomputed = self.compute_record_hash(
                    query_id=d["query_id"],
                    timestamp=d["timestamp"],
                    query_text=d["query_text"],
                    retrieved_chunk_ids=chunk_ids,
                    regime_tags=regime_tags,
                    retrieval_index_hash=d["retrieval_index_hash"],
                    adapter_hash=d["adapter_hash"],
                    confidence_scores=conf_scores,
                    verification_verdict=d["verification_verdict"],
                    final_answer=d["final_answer"],
                    previous_hash=d["previous_hash"]
                )

                if recomputed != d["record_hash"]:
                    return False, f"Tampered record at {d['query_id']}: expected hash {recomputed}, recorded {d['record_hash']}"

                expected_prev = d["record_hash"]

            return True, None
