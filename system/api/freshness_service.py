from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any

from .market_service import normalize_ticker, table_exists


CANONICAL_TABLES = (
    "companies",
    "quote_snapshots",
    "financial_ratios",
    "company_events",
    "company_peers",
)


def now_utc() -> datetime:
    return datetime.now(UTC)


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(f"{value}T00:00:00+00:00")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def age_days(value: str | None) -> float | None:
    parsed = parse_date(value)
    if parsed is None:
        return None
    return round((now_utc() - parsed).total_seconds() / 86400, 3)


def status_for_age(days: float | None, warn_days: int, stale_days: int) -> str:
    if days is None:
        return "unknown"
    if days > stale_days:
        return "stale"
    if days > warn_days:
        return "warning"
    return "fresh"


def latest_run(conn: sqlite3.Connection) -> dict[str, Any] | None:
    if not table_exists(conn, "ingestion_runs"):
        return None
    row = conn.execute("SELECT * FROM ingestion_runs ORDER BY run_id DESC LIMIT 1").fetchone()
    return dict(row) if row else None


def quality_summary(conn: sqlite3.Connection, run_id: int | None) -> dict[str, Any]:
    if run_id is None or not table_exists(conn, "data_quality_audits"):
        return {"status": "unknown", "checks": []}
    rows = conn.execute(
        """
        SELECT check_name, status, details, checked_at
        FROM data_quality_audits
        WHERE run_id = ?
        ORDER BY audit_id DESC
        """,
        (run_id,),
    ).fetchall()
    checks = [dict(row) for row in rows]
    statuses = {str(row["status"]).lower() for row in checks}
    if "fail" in statuses or "failed" in statuses:
        status = "fail"
    elif "warning" in statuses or "warn" in statuses:
        status = "warning"
    elif "pass" in statuses or "passed" in statuses:
        status = "pass"
    else:
        status = "unknown"
    return {"status": status, "checks": checks}


def table_summary(conn: sqlite3.Connection, table_name: str) -> dict[str, Any]:
    if not table_exists(conn, table_name):
        return {
            "table": table_name,
            "available": False,
            "record_count": 0,
            "latest_as_of": None,
            "latest_processed_at": None,
            "latest_run_id": None,
        }
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS record_count,
               MAX(as_of) AS latest_as_of,
               MAX(processed_at) AS latest_processed_at,
               MAX(local_ingestion_run_id) AS latest_run_id
        FROM {table_name}
        """
    ).fetchone()
    return {
        "table": table_name,
        "available": True,
        "record_count": row["record_count"],
        "latest_as_of": row["latest_as_of"],
        "latest_processed_at": row["latest_processed_at"],
        "latest_run_id": row["latest_run_id"],
    }


def overall_status(table_summaries: list[dict[str, Any]], quality_status: str, warn_days: int, stale_days: int) -> str:
    if quality_status == "fail":
        return "quality_fail"
    ages = [age_days(item.get("latest_as_of")) for item in table_summaries if item.get("available") and item.get("record_count")]
    freshness = {status_for_age(days, warn_days, stale_days) for days in ages}
    if "stale" in freshness:
        return "stale"
    if "warning" in freshness or quality_status == "warning":
        return "warning"
    if ages:
        return "fresh"
    return "empty"


def build_freshness_report(conn: sqlite3.Connection, warn_days: int = 2, stale_days: int = 5) -> dict[str, Any]:
    run = latest_run(conn)
    run_id = int(run["run_id"]) if run else None
    quality = quality_summary(conn, run_id)
    tables = [table_summary(conn, table_name) for table_name in CANONICAL_TABLES]
    for item in tables:
        item["age_days"] = age_days(item.get("latest_as_of"))
        item["freshness_status"] = status_for_age(item["age_days"], warn_days, stale_days)
    return {
        "data": {
            "overall_status": overall_status(tables, quality["status"], warn_days, stale_days),
            "latest_run": run,
            "quality": quality,
            "tables": tables,
        },
        "metadata": {
            "warn_days": warn_days,
            "stale_days": stale_days,
            "generated_at": now_utc().replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        },
    }


def latest_ticker_rows(conn: sqlite3.Connection, ticker: str) -> list[dict[str, Any]]:
    symbol = normalize_ticker(ticker)
    specs = [
        ("company", "companies"),
        ("quote", "quote_snapshots"),
        ("ratio", "financial_ratios"),
        ("event", "company_events"),
        ("peer", "company_peers"),
    ]
    rows: list[dict[str, Any]] = []
    for kind, table_name in specs:
        if not table_exists(conn, table_name):
            rows.append({"kind": kind, "table": table_name, "available": False})
            continue
        row = conn.execute(
            f"""
            SELECT COUNT(*) AS record_count,
                   MAX(as_of) AS latest_as_of,
                   MAX(processed_at) AS latest_processed_at,
                   MAX(local_ingestion_run_id) AS latest_run_id,
                   MAX(quality_status) AS quality_status,
                   MAX(data_rights_status) AS data_rights_status
            FROM {table_name}
            WHERE UPPER(ticker) = ?
            """,
            (symbol,),
        ).fetchone()
        rows.append(
            {
                "kind": kind,
                "table": table_name,
                "available": True,
                "record_count": row["record_count"],
                "latest_as_of": row["latest_as_of"],
                "latest_processed_at": row["latest_processed_at"],
                "latest_run_id": row["latest_run_id"],
                "quality_status": row["quality_status"],
                "data_rights_status": row["data_rights_status"],
            }
        )
    return rows


def build_ticker_freshness(conn: sqlite3.Connection, ticker: str, warn_days: int = 2, stale_days: int = 5) -> dict[str, Any]:
    rows = latest_ticker_rows(conn, ticker)
    for item in rows:
        item["age_days"] = age_days(item.get("latest_as_of"))
        item["freshness_status"] = status_for_age(item["age_days"], warn_days, stale_days)
    available = [item for item in rows if item.get("available") and item.get("record_count")]
    freshness = {item["freshness_status"] for item in available}
    if "stale" in freshness:
        status = "stale"
    elif "warning" in freshness:
        status = "warning"
    elif available:
        status = "fresh"
    else:
        status = "missing"
    return {
        "data": {
            "ticker": normalize_ticker(ticker),
            "overall_status": status,
            "sources": rows,
        },
        "metadata": {
            "warn_days": warn_days,
            "stale_days": stale_days,
            "generated_at": now_utc().replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        },
    }
