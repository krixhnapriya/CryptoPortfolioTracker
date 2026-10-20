"""SQLite database management for cryptocurrency portfolio holdings."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from src.config import DATABASE_PATH


def get_connection(db_path: Path | str = DATABASE_PATH) -> sqlite3.Connection:
    """Create and return a SQLite database connection with row factory enabled."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str = DATABASE_PATH) -> None:
    """Initialize the database schema (holdings and historical_prices) if not existing."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS holdings (
                coin_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                name TEXT NOT NULL,
                quantity REAL NOT NULL CHECK(quantity > 0),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_prices (
                coin_id TEXT NOT NULL,
                date TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                price_inr REAL NOT NULL CHECK(price_inr >= 0),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (coin_id, date)
            )
            """
        )
        conn.commit()


def get_holdings(db_path: Path | str = DATABASE_PATH) -> list[dict[str, Any]]:
    """Retrieve all cryptocurrency holdings from the database."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT coin_id, symbol, name, quantity, updated_at FROM holdings ORDER BY coin_id ASC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def add_or_update_holding(
    coin_id: str,
    symbol: str,
    name: str,
    quantity: float,
    db_path: Path | str = DATABASE_PATH,
) -> None:
    """
    Insert a new holding or update quantity if the coin already exists.

    Raises:
        ValueError: If coin_id is empty or quantity <= 0.
    """
    coin_id = coin_id.strip().lower()
    symbol = symbol.strip().upper()
    name = name.strip()

    if not coin_id:
        raise ValueError("Cryptocurrency ID cannot be empty.")
    if quantity <= 0:
        raise ValueError("Quantity must be greater than zero.")

    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO holdings (coin_id, symbol, name, quantity, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(coin_id) DO UPDATE SET
                symbol = excluded.symbol,
                name = excluded.name,
                quantity = excluded.quantity,
                updated_at = CURRENT_TIMESTAMP
            """,
            (coin_id, symbol, name, float(quantity)),
        )
        conn.commit()


def delete_holding(coin_id: str, db_path: Path | str = DATABASE_PATH) -> bool:
    """
    Delete a holding by coin_id. Returns True if a record was removed, False otherwise.
    """
    coin_id = coin_id.strip().lower()
    if not coin_id:
        return False

    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM holdings WHERE coin_id = ?", (coin_id,))
        conn.commit()
        return cursor.rowcount > 0


def clear_all_holdings(db_path: Path | str = DATABASE_PATH) -> None:
    """Delete all records from the holdings table."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM holdings")
        conn.commit()


# ============================================================================
# Historical Price Storage Functions (Module 2)
# ============================================================================


def save_historical_prices(
    coin_id: str,
    records: list[dict[str, Any]],
    db_path: Path | str = DATABASE_PATH,
) -> int:
    """
    Save historical price records into SQLite.

    Avoids duplicates using primary key (coin_id, date) with UPSERT.

    Args:
        coin_id: CoinGecko coin identifier.
        records: List of dicts, each with keys: 'date' (str YYYY-MM-DD),
                 'timestamp' (int), 'price_inr' (float).
        db_path: Database path.

    Returns:
        Number of records saved.
    """
    coin_id = coin_id.strip().lower()
    if not coin_id or not records:
        return 0

    init_db(db_path)
    inserted_count = 0

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        for rec in records:
            date_str = str(rec["date"]).strip()
            ts = int(rec["timestamp"])
            price = float(rec["price_inr"])

            if price < 0:
                continue

            cursor.execute(
                """
                INSERT INTO historical_prices (coin_id, date, timestamp, price_inr)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(coin_id, date) DO UPDATE SET
                    timestamp = excluded.timestamp,
                    price_inr = excluded.price_inr
                """,
                (coin_id, date_str, ts, price),
            )
            inserted_count += 1
        conn.commit()

    return inserted_count


def get_historical_prices(
    coin_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    db_path: Path | str = DATABASE_PATH,
) -> list[dict[str, Any]]:
    """
    Retrieve stored historical price records for a cryptocurrency in ascending chronological order.

    Args:
        coin_id: CoinGecko coin ID.
        start_date: Optional filter 'YYYY-MM-DD' (inclusive).
        end_date: Optional filter 'YYYY-MM-DD' (inclusive).
        db_path: Database path.

    Returns:
        List of dicts: [{'date': str, 'timestamp': int, 'price_inr': float}, ...]
    """
    coin_id = coin_id.strip().lower()
    if not coin_id:
        return []

    init_db(db_path)
    query = "SELECT date, timestamp, price_inr FROM historical_prices WHERE coin_id = ?"
    params: list[Any] = [coin_id]

    if start_date:
        query += " AND date >= ?"
        params.append(start_date.strip())
    if end_date:
        query += " AND date <= ?"
        params.append(end_date.strip())

    query += " ORDER BY date ASC, timestamp ASC"

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_stored_date_range(
    coin_id: str,
    db_path: Path | str = DATABASE_PATH,
) -> tuple[str | None, str | None, int]:
    """
    Get the date range and total record count available in local SQLite for a coin.

    Returns:
        (min_date, max_date, total_records)
    """
    coin_id = coin_id.strip().lower()
    if not coin_id:
        return None, None, 0

    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(*) as cnt
            FROM historical_prices
            WHERE coin_id = ?
            """,
            (coin_id,),
        )
        row = cursor.fetchone()
        if row and row["cnt"] > 0:
            return row["min_date"], row["max_date"], row["cnt"]
        return None, None, 0


def delete_historical_prices(
    coin_id: str,
    db_path: Path | str = DATABASE_PATH,
) -> int:
    """Delete all stored historical prices for a coin. Returns rows deleted."""
    coin_id = coin_id.strip().lower()
    if not coin_id:
        return 0

    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM historical_prices WHERE coin_id = ?", (coin_id,))
        conn.commit()
        return cursor.rowcount

