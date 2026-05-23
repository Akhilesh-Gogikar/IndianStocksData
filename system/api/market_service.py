from __future__ import annotations

import json
import sqlite3
from typing import Any


class MarketDataUnavailable(RuntimeError):
    pass


class MarketRecordNotFound(RuntimeError):
    pass


LINEAGE_COLUMNS = {
    "local_ingestion_run_id",
    "as_of",
    "processed_at",
    "quality_status",
    "data_rights_status",
    "raw_document_id",
}


def normalize_ticker(ticker: str) -> str:
    normalized = (ticker or "").strip().upper()
    if not normalized:
        raise MarketRecordNotFound("Ticker is required")
    return normalized


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def require_table(conn: sqlite3.Connection, table_name: str) -> None:
    if not table_exists(conn, table_name):
        raise MarketDataUnavailable(f"Canonical table '{table_name}' is not available")


def row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def parse_json(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def run_quality_status(conn: sqlite3.Connection, run_id: int | None) -> str:
    if run_id is None or not table_exists(conn, "data_quality_audits"):
        return "unknown"
    rows = conn.execute(
        """
        SELECT LOWER(status) AS status
        FROM data_quality_audits
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchall()
    statuses = {row["status"] for row in rows if row["status"]}
    if "fail" in statuses or "failed" in statuses:
        return "fail"
    if "warn" in statuses or "warning" in statuses:
        return "warning"
    if "pass" in statuses or "passed" in statuses:
        return "pass"
    return "unknown"


def metadata_from_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "as_of": None,
            "processed_at": None,
            "local_ingestion_run_id": None,
            "quality_status": "unknown",
            "data_rights_status": "unknown",
        }
    newest = max(rows, key=lambda item: (item.get("as_of") or "", item.get("processed_at") or ""))
    run_id = newest.get("local_ingestion_run_id")
    rights = {row.get("data_rights_status") or "unknown" for row in rows}
    quality = {row.get("quality_status") or run_quality_status(conn, row.get("local_ingestion_run_id")) for row in rows}
    if "fail" in quality:
        quality_status = "fail"
    elif "warning" in quality:
        quality_status = "warning"
    elif len(quality) == 1:
        quality_status = next(iter(quality))
    else:
        quality_status = "mixed"
    return {
        "as_of": newest.get("as_of"),
        "processed_at": newest.get("processed_at"),
        "local_ingestion_run_id": run_id,
        "quality_status": quality_status,
        "data_rights_status": rights.pop() if len(rights) == 1 else "mixed",
    }


def public_record(row: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in row.items() if key not in LINEAGE_COLUMNS}
    parsed_extra = parse_json(payload.pop("extra_json", None))
    payload.pop("raw_json", None)
    if parsed_extra is not None:
        payload["extra"] = parsed_extra
    return payload


def envelope(conn: sqlite3.Connection, data: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "data": data,
        "metadata": metadata_from_rows(conn, rows),
    }


def get_company(conn: sqlite3.Connection, ticker: str) -> dict[str, Any]:
    require_table(conn, "companies")
    symbol = normalize_ticker(ticker)
    row = row_dict(
        conn.execute(
            """
            SELECT *
            FROM companies
            WHERE UPPER(ticker) = ?
            """,
            (symbol,),
        ).fetchone()
    )
    if row is None:
        raise MarketRecordNotFound(f"No company profile found for {symbol}")
    return envelope(conn, public_record(row), [row])


def get_quote(conn: sqlite3.Connection, ticker: str) -> dict[str, Any]:
    require_table(conn, "quote_snapshots")
    symbol = normalize_ticker(ticker)
    row = row_dict(
        conn.execute(
            """
            SELECT *
            FROM quote_snapshots
            WHERE UPPER(ticker) = ?
            ORDER BY as_of DESC, processed_at DESC, quote_id DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()
    )
    if row is None:
        raise MarketRecordNotFound(f"No quote snapshot found for {symbol}")
    return envelope(conn, public_record(row), [row])


def get_ratios(conn: sqlite3.Connection, ticker: str, period: str | None = None) -> dict[str, Any]:
    require_table(conn, "financial_ratios")
    symbol = normalize_ticker(ticker)
    params: list[Any] = [symbol]
    clause = "UPPER(ticker) = ?"
    if period:
        clause += " AND period = ?"
        params.append(period)
    rows = [
        dict(row)
        for row in conn.execute(
            f"""
            SELECT *
            FROM financial_ratios
            WHERE {clause}
            ORDER BY period_end DESC, ratio_name
            """,
            params,
        ).fetchall()
    ]
    if not rows:
        raise MarketRecordNotFound(f"No financial ratios found for {symbol}")
    data = {
        "ticker": symbol,
        "ratios": [public_record(row) for row in rows],
    }
    return envelope(conn, data, rows)


def get_events(conn: sqlite3.Connection, ticker: str, limit: int) -> dict[str, Any]:
    require_table(conn, "company_events")
    symbol = normalize_ticker(ticker)
    rows = [
        dict(row)
        for row in conn.execute(
            """
            SELECT *
            FROM company_events
            WHERE UPPER(ticker) = ?
            ORDER BY event_date DESC, event_id DESC
            LIMIT ?
            """,
            (symbol, limit),
        ).fetchall()
    ]
    if not rows:
        raise MarketRecordNotFound(f"No events found for {symbol}")
    return envelope(conn, {"ticker": symbol, "events": [public_record(row) for row in rows]}, rows)


def get_peers(conn: sqlite3.Connection, ticker: str) -> dict[str, Any]:
    require_table(conn, "company_peers")
    symbol = normalize_ticker(ticker)
    join_company = table_exists(conn, "companies")
    select_peer = "c.name AS peer_name, c.sector AS peer_sector, c.industry AS peer_industry" if join_company else "NULL AS peer_name, NULL AS peer_sector, NULL AS peer_industry"
    join_clause = "LEFT JOIN companies c ON UPPER(c.ticker) = UPPER(p.peer_ticker)" if join_company else ""
    rows = [
        dict(row)
        for row in conn.execute(
            f"""
            SELECT p.*, {select_peer}
            FROM company_peers p
            {join_clause}
            WHERE UPPER(p.ticker) = ?
            ORDER BY p.score DESC, p.peer_ticker
            """,
            (symbol,),
        ).fetchall()
    ]
    if not rows:
        raise MarketRecordNotFound(f"No peers found for {symbol}")
    return envelope(conn, {"ticker": symbol, "peers": [public_record(row) for row in rows]}, rows)


def latest_quotes_by_ticker(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    if not table_exists(conn, "quote_snapshots"):
        return {}
    rows = conn.execute(
        """
        SELECT *
        FROM quote_snapshots
        ORDER BY ticker, as_of DESC, processed_at DESC, quote_id DESC
        """
    ).fetchall()
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        data = dict(row)
        latest.setdefault(normalize_ticker(data["ticker"]), data)
    return latest


def latest_ratios_by_ticker(conn: sqlite3.Connection) -> dict[str, dict[str, float | None]]:
    if not table_exists(conn, "financial_ratios"):
        return {}
    rows = conn.execute(
        """
        SELECT *
        FROM financial_ratios
        ORDER BY ticker, ratio_name, period_end DESC, as_of DESC, processed_at DESC, ratio_id DESC
        """
    ).fetchall()
    ratios: dict[str, dict[str, float | None]] = {}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        ticker = normalize_ticker(row["ticker"])
        name = row["ratio_name"]
        key = (ticker, name)
        if key in seen:
            continue
        seen.add(key)
        ratios.setdefault(ticker, {})[name] = row["ratio_value"]
    return ratios


def in_range(value: Any, minimum: float | None, maximum: float | None) -> bool:
    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    if minimum is not None and numeric < minimum:
        return False
    if maximum is not None and numeric > maximum:
        return False
    return True


def screen_companies(conn: sqlite3.Connection, filters: dict[str, Any]) -> dict[str, Any]:
    require_table(conn, "companies")
    limit = max(1, min(int(filters.get("limit") or 50), 500))
    tickers = {normalize_ticker(ticker) for ticker in filters.get("tickers") or []}
    rows = [dict(row) for row in conn.execute("SELECT * FROM companies ORDER BY market_cap DESC NULLS LAST, ticker").fetchall()]
    quotes = latest_quotes_by_ticker(conn)
    ratios = latest_ratios_by_ticker(conn)
    results: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []

    for row in rows:
        ticker = normalize_ticker(row["ticker"])
        quote = quotes.get(ticker)
        ratio_values = ratios.get(ticker, {})
        if tickers and ticker not in tickers:
            continue
        if filters.get("sector") and (row.get("sector") or "").lower() != str(filters["sector"]).lower():
            continue
        if filters.get("industry") and (row.get("industry") or "").lower() != str(filters["industry"]).lower():
            continue
        if filters.get("min_market_cap") is not None and not in_range(row.get("market_cap"), filters.get("min_market_cap"), None):
            continue
        if filters.get("max_market_cap") is not None and not in_range(row.get("market_cap"), None, filters.get("max_market_cap")):
            continue
        if filters.get("min_price") is not None and not in_range((quote or {}).get("price"), filters.get("min_price"), None):
            continue
        if filters.get("max_price") is not None and not in_range((quote or {}).get("price"), None, filters.get("max_price")):
            continue
        ratio_filters = filters.get("ratio_filters") or {}
        if any(
            not in_range(ratio_values.get(name), bounds.get("min"), bounds.get("max"))
            for name, bounds in ratio_filters.items()
        ):
            continue

        lineage_rows.append(row)
        if quote:
            lineage_rows.append(quote)
        results.append(
            {
                "company": public_record(row),
                "quote": public_record(quote) if quote else None,
                "ratios": ratio_values,
            }
        )
        if len(results) >= limit:
            break

    response = envelope(conn, results, lineage_rows)
    response["metadata"]["result_count"] = len(results)
    response["metadata"]["filters"] = filters
    return response
