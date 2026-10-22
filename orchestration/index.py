"""SQLite-backed run index for the SAH-RAG orchestrator (v2)."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    baseline TEXT NOT NULL,
    dataset TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    metrics_json TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_status ON runs(status);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunIndex:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def get_status(self, run_id: str) -> str | None:
        cur = self._conn.execute("SELECT status FROM runs WHERE run_id=?", (run_id,))
        row = cur.fetchone()
        return row[0] if row else None

    def is_done(self, cell) -> bool:
        return self.get_status(cell.run_id) == "done"

    def mark_started(self, cell) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, baseline, dataset, mode, status, started_at) "
            "VALUES (?, ?, ?, ?, 'running', ?)",
            (cell.run_id, cell.baseline, cell.dataset, cell.mode, _now()),
        )
        self._conn.commit()

    def mark_done(self, cell, metrics_json: str) -> None:
        self._conn.execute(
            "UPDATE runs SET status='done', finished_at=?, metrics_json=? WHERE run_id=?",
            (_now(), metrics_json, cell.run_id),
        )
        self._conn.commit()

    def mark_failed(self, cell, error: str) -> None:
        self._conn.execute(
            "UPDATE runs SET status='failed', finished_at=?, error=? WHERE run_id=?",
            (_now(), error, cell.run_id),
        )
        self._conn.commit()

    def list_runs(self) -> Iterable[tuple]:
        cur = self._conn.execute(
            "SELECT run_id, baseline, dataset, mode, status FROM runs ORDER BY started_at"
        )
        return cur.fetchall()