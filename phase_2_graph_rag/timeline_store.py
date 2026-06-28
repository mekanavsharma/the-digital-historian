"""SQLite + lightweight pandas-compatible timeline store."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


_TIMELINE_COLUMNS = [
    "chunk_id",
    "historian",
    "source_label",
    "source_name",
    "relation",
    "target_label",
    "target_name",
    "year",
    "volume",
    "chapter",
    "page",
    "text",
]


class TimelineStore:
    def __init__(self, sqlite_path: str):
        self.sqlite_path = sqlite_path
        Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path, timeout=60)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS timeline_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT,
                    historian TEXT,
                    source_label TEXT,
                    source_name TEXT,
                    relation TEXT,
                    target_label TEXT,
                    target_name TEXT,
                    year INTEGER,
                    volume TEXT,
                    chapter TEXT,
                    page INTEGER,
                    text TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timeline_year ON timeline_facts(year)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timeline_source_name ON timeline_facts(source_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timeline_target_name ON timeline_facts(target_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timeline_relation ON timeline_facts(relation)")
            conn.commit()

    def reset(self) -> None:
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM timeline_facts")
                conn.commit()
        except sqlite3.OperationalError as e:
            print(f"[TimelineStore] reset skipped: {e}")

    def _normalize_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        clean: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            clean.append({col: row.get(col) for col in _TIMELINE_COLUMNS})
        return clean

    def ingest_rows(self, rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0

        rows = self._normalize_rows(rows)

        values = [tuple(r.get(col) for col in _TIMELINE_COLUMNS) for r in rows]
        placeholders = ",".join(["?"] * len(_TIMELINE_COLUMNS))
        sql = f"INSERT INTO timeline_facts ({','.join(_TIMELINE_COLUMNS)}) VALUES ({placeholders})"

        with self._connect() as conn:
            conn.executemany(sql, values)
            conn.commit()

        return len(values)

    def query(self, sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(sql, conn, params=params or ())

    def search_year_range(self, start_year: int, end_year: int, limit: int = 20) -> pd.DataFrame:
        sql = """
            SELECT * FROM timeline_facts
            WHERE year IS NOT NULL AND year BETWEEN ? AND ?
            ORDER BY year ASC, id ASC
            LIMIT ?
        """
        return self.query(sql, (start_year, end_year, limit))

    def search_keyword(self, keyword: str, limit: int = 20) -> pd.DataFrame:
        sql = """
            SELECT * FROM timeline_facts
            WHERE lower(source_name) LIKE ?
               OR lower(target_name) LIKE ?
               OR lower(relation) LIKE ?
               OR lower(text) LIKE ?
            ORDER BY COALESCE(year, 999999) ASC, id ASC
            LIMIT ?
        """
        kw = f"%{keyword.lower()}%"
        return self.query(sql, (kw, kw, kw, kw, limit))
