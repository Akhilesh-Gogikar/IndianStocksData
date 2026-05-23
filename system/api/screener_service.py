from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

from .market_service import MarketDataUnavailable, MarketRecordNotFound, screen_companies, table_exists


def now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clean_owner_id(owner_id: str | None) -> str:
    return (owner_id or "default").strip() or "default"


def require_screener_tables(conn: sqlite3.Connection) -> None:
    for table_name in ("saved_screeners", "screener_evaluations"):
        if not table_exists(conn, table_name):
            raise MarketDataUnavailable(f"Screener table '{table_name}' is not available")


def normalize_filters(filters: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(filters or {})
    limit = int(normalized.get("limit") or 50)
    normalized["limit"] = max(1, min(limit, 500))
    return normalized


def parse_filters(value: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def serialize_filters(filters: dict[str, Any]) -> str:
    return json.dumps(normalize_filters(filters), sort_keys=True, separators=(",", ":"))


def public_screener(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    data = dict(row)
    data["filters"] = parse_filters(data.pop("filters_json"))
    return data


def create_screener(
    conn: sqlite3.Connection,
    owner_id: str | None,
    name: str,
    description: str | None,
    filters: dict[str, Any],
) -> dict[str, Any]:
    require_screener_tables(conn)
    now = now_utc()
    owner = clean_owner_id(owner_id)
    conn.execute(
        """
        INSERT INTO saved_screeners (owner_id, name, description, filters_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(owner_id, name) DO UPDATE SET
            description = excluded.description,
            filters_json = excluded.filters_json,
            updated_at = excluded.updated_at
        """,
        (owner, name.strip(), description, serialize_filters(filters), now, now),
    )
    row = conn.execute(
        "SELECT * FROM saved_screeners WHERE owner_id = ? AND name = ?",
        (owner, name.strip()),
    ).fetchone()
    conn.commit()
    return public_screener(row)


def list_screeners(conn: sqlite3.Connection, owner_id: str | None) -> dict[str, Any]:
    require_screener_tables(conn)
    rows = conn.execute(
        """
        SELECT s.*,
               e.evaluated_at AS last_evaluated_at,
               e.result_count AS last_result_count,
               e.top_tickers_json AS last_top_tickers_json
        FROM saved_screeners s
        LEFT JOIN (
            SELECT se.*
            FROM screener_evaluations se
            JOIN (
                SELECT screener_id, MAX(evaluated_at) AS evaluated_at
                FROM screener_evaluations
                GROUP BY screener_id
            ) latest
              ON latest.screener_id = se.screener_id
             AND latest.evaluated_at = se.evaluated_at
        ) e ON e.screener_id = s.screener_id
        WHERE s.owner_id = ?
        ORDER BY s.updated_at DESC, s.screener_id DESC
        """,
        (clean_owner_id(owner_id),),
    ).fetchall()
    data = []
    for row in rows:
        item = public_screener(row)
        item["last_top_tickers"] = json.loads(item.pop("last_top_tickers_json") or "[]")
        data.append(item)
    return {"data": data, "metadata": {"result_count": len(data)}}


def screener_row(conn: sqlite3.Connection, screener_id: int, owner_id: str | None) -> dict[str, Any]:
    require_screener_tables(conn)
    row = conn.execute(
        """
        SELECT *
        FROM saved_screeners
        WHERE screener_id = ? AND owner_id = ?
        """,
        (screener_id, clean_owner_id(owner_id)),
    ).fetchone()
    if not row:
        raise MarketRecordNotFound(f"No screener found for id {screener_id}")
    return public_screener(row)


def get_screener(conn: sqlite3.Connection, screener_id: int, owner_id: str | None) -> dict[str, Any]:
    screener = screener_row(conn, screener_id, owner_id)
    rows = conn.execute(
        """
        SELECT *
        FROM screener_evaluations
        WHERE screener_id = ?
        ORDER BY evaluated_at DESC, evaluation_id DESC
        LIMIT 10
        """,
        (screener_id,),
    ).fetchall()
    history = []
    for row in rows:
        item = dict(row)
        item["top_tickers"] = json.loads(item.pop("top_tickers_json"))
        item["metadata"] = json.loads(item.pop("metadata_json"))
        history.append(item)
    return {"data": {"screener": screener, "history": history}, "metadata": {"history_count": len(history)}}


def evaluate_screener(
    conn: sqlite3.Connection,
    screener_id: int,
    owner_id: str | None,
    persist: bool = True,
) -> dict[str, Any]:
    screener = screener_row(conn, screener_id, owner_id)
    result = screen_companies(conn, screener["filters"])
    top_tickers = [item["company"]["ticker"] for item in result["data"][:25]]
    evaluated_at = now_utc()
    if persist:
        conn.execute(
            """
            INSERT INTO screener_evaluations (
                screener_id, evaluated_at, result_count, top_tickers_json, metadata_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                screener_id,
                evaluated_at,
                result["metadata"]["result_count"],
                json.dumps(top_tickers, separators=(",", ":")),
                json.dumps(result["metadata"], sort_keys=True, separators=(",", ":")),
            ),
        )
        conn.execute("UPDATE saved_screeners SET updated_at = ? WHERE screener_id = ?", (evaluated_at, screener_id))
        conn.commit()
    result["metadata"]["screener_id"] = screener_id
    result["metadata"]["screener_name"] = screener["name"]
    result["metadata"]["evaluated_at"] = evaluated_at
    result["metadata"]["top_tickers"] = top_tickers
    return result
