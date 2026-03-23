"""
SQLite database backend for trade storage.

Replaces CSV-based storage with SQLite for faster queries, better
concurrency, and indexed lookups. Auto-migrates existing trades_ml.csv
on first run.

Usage
-----
    from ml.database import get_db, insert_trade, query_trades
    db = get_db()
    insert_trade(db, record)
    df = query_trades(db, symbol="BTCUSD", limit=100)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "logs" / "trades.db"
CSV_PATH = Path(__file__).resolve().parent.parent / "logs" / "trades_ml.csv"

_local = threading.local()


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id        TEXT UNIQUE,
    timestamp       TEXT,
    symbol          TEXT,
    direction       TEXT,
    entry_price     REAL,
    exit_price      REAL,
    sl              REAL,
    tp              REAL,
    quantity         REAL,
    pnl             REAL,
    pnl_pct         REAL,
    rr_achieved     REAL,
    result          INTEGER,
    strategy        TEXT,
    duration_bars   INTEGER,
    model_confidence REAL,
    model_prediction REAL,
    features_json   TEXT
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_trades_result ON trades(result);",
    "CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);",
]


# ─────────────────────────────────────────────────────────────────────────────
# Connection management
# ─────────────────────────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    """
    Get a thread-local SQLite connection.
    Creates the database and schema on first call.
    """
    if not hasattr(_local, "conn") or _local.conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.execute("PRAGMA journal_mode=WAL;")
        _local.conn.execute("PRAGMA synchronous=NORMAL;")
        _init_schema(_local.conn)
    return _local.conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create tables and indexes if they don't exist."""
    conn.execute(CREATE_TABLE)
    for idx_sql in CREATE_INDEXES:
        conn.execute(idx_sql)
    conn.commit()
    logger.debug("SQLite schema initialized at %s", DB_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Write operations
# ─────────────────────────────────────────────────────────────────────────────

COLUMNS = [
    "trade_id", "timestamp", "symbol", "direction", "entry_price",
    "exit_price", "sl", "tp", "quantity", "pnl", "pnl_pct",
    "rr_achieved", "result", "strategy", "duration_bars",
    "model_confidence", "model_prediction", "features_json",
]


def insert_trade(conn: sqlite3.Connection, record: dict) -> None:
    """Insert a single trade record into the database."""
    row = {col: record.get(col) for col in COLUMNS}

    # Serialise features vector if needed
    if "features" in record and row.get("features_json") is None:
        feat = record["features"]
        if hasattr(feat, "tolist"):
            feat = feat.tolist()
        row["features_json"] = json.dumps([round(float(v), 6) for v in feat])

    placeholders = ", ".join(["?"] * len(COLUMNS))
    col_names = ", ".join(COLUMNS)

    try:
        conn.execute(
            f"INSERT OR IGNORE INTO trades ({col_names}) VALUES ({placeholders})",
            [row.get(c) for c in COLUMNS],
        )
        conn.commit()
    except sqlite3.Error as e:
        logger.warning("SQLite insert failed: %s", e)


def insert_trades_batch(conn: sqlite3.Connection, records: list[dict]) -> int:
    """Insert multiple trades in a single transaction. Returns count inserted."""
    count = 0
    try:
        for record in records:
            row = {col: record.get(col) for col in COLUMNS}
            if "features" in record and row.get("features_json") is None:
                feat = record["features"]
                if hasattr(feat, "tolist"):
                    feat = feat.tolist()
                row["features_json"] = json.dumps([round(float(v), 6) for v in feat])

            placeholders = ", ".join(["?"] * len(COLUMNS))
            col_names = ", ".join(COLUMNS)
            conn.execute(
                f"INSERT OR IGNORE INTO trades ({col_names}) VALUES ({placeholders})",
                [row.get(c) for c in COLUMNS],
            )
            count += 1
        conn.commit()
    except sqlite3.Error as e:
        logger.warning("SQLite batch insert failed: %s", e)
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Read operations
# ─────────────────────────────────────────────────────────────────────────────

def query_trades(
    conn: sqlite3.Connection,
    symbol: Optional[str] = None,
    result: Optional[int] = None,
    strategy: Optional[str] = None,
    limit: Optional[int] = None,
    min_rows: int = 0,
) -> Optional[pd.DataFrame]:
    """
    Query trades with optional filters. Returns a DataFrame or None.
    """
    query = "SELECT * FROM trades WHERE 1=1"
    params = []

    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    if result is not None:
        query += " AND result = ?"
        params.append(result)
    if strategy:
        query += " AND strategy = ?"
        params.append(strategy)

    query += " ORDER BY timestamp ASC"

    if limit:
        query += " LIMIT ?"
        params.append(limit)

    try:
        df = pd.read_sql_query(query, conn, params=params)
        if len(df) < min_rows:
            return None
        return df
    except Exception as e:
        logger.warning("SQLite query failed: %s", e)
        return None


def count_trades(conn: sqlite3.Connection, completed_only: bool = True) -> int:
    """Count trades in the database."""
    query = "SELECT COUNT(*) FROM trades"
    if completed_only:
        query += " WHERE result IN (0, 1)"
    try:
        cursor = conn.execute(query)
        return cursor.fetchone()[0]
    except sqlite3.Error:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# CSV migration
# ─────────────────────────────────────────────────────────────────────────────

def migrate_csv_to_sqlite(conn: Optional[sqlite3.Connection] = None) -> int:
    """
    Import existing trades_ml.csv into SQLite.
    Skips rows that already exist (by trade_id).
    Returns the number of rows imported.
    """
    if not CSV_PATH.exists():
        logger.info("No CSV to migrate")
        return 0

    if conn is None:
        conn = get_db()

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        logger.warning("Failed to read CSV for migration: %s", e)
        return 0

    count = 0
    for _, row in df.iterrows():
        record = row.to_dict()
        # Ensure result is int
        if "result" in record:
            try:
                record["result"] = int(record["result"])
            except (ValueError, TypeError):
                continue

        insert_trade(conn, record)
        count += 1

    logger.info("Migrated %d trades from CSV to SQLite", count)
    return count


def ensure_db_ready() -> sqlite3.Connection:
    """
    Get database connection and auto-migrate CSV if DB is empty.
    """
    conn = get_db()
    if count_trades(conn, completed_only=False) == 0 and CSV_PATH.exists():
        logger.info("Empty database — migrating from CSV...")
        migrate_csv_to_sqlite(conn)
    return conn
