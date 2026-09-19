import sqlite3
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

class AdapterRegistry:
    def __init__(self, db_path: str = "artifacts/audit/registry.db"):
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
                CREATE TABLE IF NOT EXISTS adapter_registry (
                    adapter_id TEXT PRIMARY KEY,
                    regime_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    base_model TEXT NOT NULL,
                    training_data_hash TEXT NOT NULL,
                    hyperparameters TEXT NOT NULL,
                    validation_metrics TEXT NOT NULL,
                    merge_lineage TEXT NOT NULL,
                    is_merged INTEGER NOT NULL DEFAULT 0,
                    adapter_path TEXT NOT NULL,
                    adapter_hash TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active'
                )
            """)
            conn.commit()

    def register_adapter(
        self,
        adapter_id: str,
        regime_id: str,
        base_model: str,
        training_data_hash: str,
        hyperparameters: Dict[str, Any],
        validation_metrics: Dict[str, Any],
        merge_lineage: List[str],
        is_merged: bool,
        adapter_path: str,
        adapter_hash: Optional[str] = None
    ) -> str:
        if adapter_hash is None:
            raw = f"{adapter_id}_{regime_id}_{training_data_hash}_{json.dumps(hyperparameters, sort_keys=True)}"
            adapter_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()

        created_at = datetime.utcnow().isoformat() + "Z"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE adapter_registry SET status = 'archived' WHERE status = 'active'")
            cursor.execute("""
                INSERT OR REPLACE INTO adapter_registry (
                    adapter_id, regime_id, created_at, base_model,
                    training_data_hash, hyperparameters, validation_metrics,
                    merge_lineage, is_merged, adapter_path, adapter_hash, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                adapter_id,
                regime_id,
                created_at,
                base_model,
                training_data_hash,
                json.dumps(hyperparameters),
                json.dumps(validation_metrics),
                json.dumps(merge_lineage),
                1 if is_merged else 0,
                str(adapter_path),
                adapter_hash,
                "active"
            ))
            conn.commit()
        return adapter_hash

    def get_adapter(self, adapter_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM adapter_registry WHERE adapter_id = ?", (adapter_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d["hyperparameters"] = json.loads(d["hyperparameters"])
                d["validation_metrics"] = json.loads(d["validation_metrics"])
                d["merge_lineage"] = json.loads(d["merge_lineage"])
                d["is_merged"] = bool(d["is_merged"])
                return d
        return None

    def get_latest_adapter(self, target_regime: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if target_regime:
                cursor.execute("SELECT * FROM adapter_registry WHERE regime_id = ? AND status = 'active' ORDER BY rowid DESC LIMIT 1", (target_regime,))
                row = cursor.fetchone()
                if not row:
                    cursor.execute("SELECT * FROM adapter_registry WHERE regime_id = ? ORDER BY rowid DESC LIMIT 1", (target_regime,))
                    row = cursor.fetchone()
            else:
                cursor.execute("SELECT * FROM adapter_registry WHERE status = 'active' ORDER BY rowid DESC LIMIT 1")
                row = cursor.fetchone()
                if not row:
                    cursor.execute("SELECT * FROM adapter_registry ORDER BY rowid DESC LIMIT 1")
                    row = cursor.fetchone()
            if row:
                d = dict(row)
                d["hyperparameters"] = json.loads(d["hyperparameters"])
                d["validation_metrics"] = json.loads(d["validation_metrics"])
                d["merge_lineage"] = json.loads(d["merge_lineage"])
                d["is_merged"] = bool(d["is_merged"])
                return d
        return None

    def list_all_adapters(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM adapter_registry ORDER BY rowid ASC")
            rows = cursor.fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["hyperparameters"] = json.loads(d["hyperparameters"])
                d["validation_metrics"] = json.loads(d["validation_metrics"])
                d["merge_lineage"] = json.loads(d["merge_lineage"])
                d["is_merged"] = bool(d["is_merged"])
                results.append(d)
            return results
