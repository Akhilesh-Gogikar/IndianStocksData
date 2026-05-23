from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from .market_service import (
    MarketDataUnavailable,
    MarketRecordNotFound,
    metadata_from_rows,
    normalize_ticker,
    public_record,
    table_exists,
)
from .watchlist_service import latest_company, latest_quote, latest_ratios


def now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def require_portfolio_tables(conn: sqlite3.Connection) -> None:
    for table_name in ("portfolios", "portfolio_holdings"):
        if not table_exists(conn, table_name):
            raise MarketDataUnavailable(f"Portfolio table '{table_name}' is not available")


def clean_owner_id(owner_id: str | None) -> str:
    return (owner_id or "default").strip() or "default"


def create_portfolio(
    conn: sqlite3.Connection,
    owner_id: str | None,
    name: str,
    description: str | None,
    base_currency: str = "INR",
) -> dict[str, Any]:
    require_portfolio_tables(conn)
    now = now_utc()
    owner = clean_owner_id(owner_id)
    conn.execute(
        """
        INSERT INTO portfolios (owner_id, name, description, base_currency, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(owner_id, name) DO UPDATE SET
            description = excluded.description,
            base_currency = excluded.base_currency,
            updated_at = excluded.updated_at
        """,
        (owner, name.strip(), description, base_currency.upper(), now, now),
    )
    row = conn.execute(
        "SELECT * FROM portfolios WHERE owner_id = ? AND name = ?",
        (owner, name.strip()),
    ).fetchone()
    conn.commit()
    return dict(row)


def list_portfolios(conn: sqlite3.Connection, owner_id: str | None) -> dict[str, Any]:
    require_portfolio_tables(conn)
    rows = conn.execute(
        """
        SELECT p.*, COUNT(h.holding_id) AS holding_count
        FROM portfolios p
        LEFT JOIN portfolio_holdings h ON h.portfolio_id = p.portfolio_id
        WHERE p.owner_id = ?
        GROUP BY p.portfolio_id
        ORDER BY p.updated_at DESC, p.portfolio_id DESC
        """,
        (clean_owner_id(owner_id),),
    ).fetchall()
    return {"data": [dict(row) for row in rows], "metadata": {"result_count": len(rows)}}


def portfolio_row(conn: sqlite3.Connection, portfolio_id: int, owner_id: str | None) -> dict[str, Any]:
    require_portfolio_tables(conn)
    row = conn.execute(
        """
        SELECT *
        FROM portfolios
        WHERE portfolio_id = ? AND owner_id = ?
        """,
        (portfolio_id, clean_owner_id(owner_id)),
    ).fetchone()
    if not row:
        raise MarketRecordNotFound(f"No portfolio found for id {portfolio_id}")
    return dict(row)


def portfolio_holdings(conn: sqlite3.Connection, portfolio_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT *
        FROM portfolio_holdings
        WHERE portfolio_id = ?
        ORDER BY added_at, ticker
        """,
        (portfolio_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def add_holding(
    conn: sqlite3.Connection,
    portfolio_id: int,
    owner_id: str | None,
    ticker: str,
    quantity: float,
    average_cost: float | None,
    notes: str | None,
) -> dict[str, Any]:
    portfolio_row(conn, portfolio_id, owner_id)
    symbol = normalize_ticker(ticker)
    now = now_utc()
    conn.execute(
        """
        INSERT INTO portfolio_holdings (portfolio_id, ticker, quantity, average_cost, notes, added_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(portfolio_id, ticker) DO UPDATE SET
            quantity = excluded.quantity,
            average_cost = excluded.average_cost,
            notes = excluded.notes,
            updated_at = excluded.updated_at
        """,
        (portfolio_id, symbol, quantity, average_cost, notes, now, now),
    )
    conn.execute("UPDATE portfolios SET updated_at = ? WHERE portfolio_id = ?", (now, portfolio_id))
    conn.commit()
    return get_portfolio(conn, portfolio_id, owner_id)


def remove_holding(conn: sqlite3.Connection, portfolio_id: int, owner_id: str | None, ticker: str) -> dict[str, Any]:
    portfolio_row(conn, portfolio_id, owner_id)
    conn.execute(
        "DELETE FROM portfolio_holdings WHERE portfolio_id = ? AND ticker = ?",
        (portfolio_id, normalize_ticker(ticker)),
    )
    conn.execute("UPDATE portfolios SET updated_at = ? WHERE portfolio_id = ?", (now_utc(), portfolio_id))
    conn.commit()
    return get_portfolio(conn, portfolio_id, owner_id)


def holding_payload(conn: sqlite3.Connection, holding: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ticker = holding["ticker"]
    company = latest_company(conn, ticker)
    quote = latest_quote(conn, ticker)
    ratios = latest_ratios(conn, ticker)
    quantity = float(holding["quantity"])
    average_cost = holding.get("average_cost")
    price = quote.get("price") if quote else None
    market_value = quantity * float(price) if price is not None else None
    cost_basis = quantity * float(average_cost) if average_cost is not None else None
    unrealized_pl = market_value - cost_basis if market_value is not None and cost_basis is not None else None
    unrealized_pl_pct = unrealized_pl / cost_basis if unrealized_pl is not None and cost_basis else None
    lineage_rows = [row for row in (company, quote) if row]
    return (
        {
            "ticker": ticker,
            "quantity": quantity,
            "average_cost": average_cost,
            "cost_basis": cost_basis,
            "market_value": market_value,
            "unrealized_pl": unrealized_pl,
            "unrealized_pl_pct": unrealized_pl_pct,
            "notes": holding.get("notes"),
            "company": public_record(company) if company else None,
            "quote": public_record(quote) if quote else None,
            "ratios": ratios,
        },
        lineage_rows,
    )


def xray_from_holdings(conn: sqlite3.Connection, enriched: list[dict[str, Any]]) -> dict[str, Any]:
    total_value = sum(item["market_value"] or 0 for item in enriched)
    total_cost = sum(item["cost_basis"] or 0 for item in enriched)
    sector_values: dict[str, float] = defaultdict(float)
    missing_quotes = []
    for item in enriched:
        value = item["market_value"]
        if value is None:
            missing_quotes.append(item["ticker"])
            continue
        company = item.get("company") or {}
        sector = company.get("sector") or "Unknown"
        sector_values[sector] += value

    concentration = []
    for item in sorted(enriched, key=lambda row: row["market_value"] or 0, reverse=True):
        value = item["market_value"]
        if value is None:
            continue
        concentration.append(
            {
                "ticker": item["ticker"],
                "market_value": value,
                "portfolio_weight": value / total_value if total_value else None,
            }
        )

    return {
        "total_market_value": total_value,
        "total_cost_basis": total_cost,
        "total_unrealized_pl": total_value - total_cost if total_cost else None,
        "total_unrealized_pl_pct": (total_value - total_cost) / total_cost if total_cost else None,
        "positions": len(enriched),
        "positions_with_quotes": sum(1 for item in enriched if item["market_value"] is not None),
        "missing_quotes": missing_quotes,
        "sector_exposure": [
            {
                "sector": sector,
                "market_value": value,
                "portfolio_weight": value / total_value if total_value else None,
            }
            for sector, value in sorted(sector_values.items(), key=lambda pair: pair[1], reverse=True)
        ],
        "top_concentration": concentration[:10],
    }


def get_portfolio(conn: sqlite3.Connection, portfolio_id: int, owner_id: str | None) -> dict[str, Any]:
    portfolio = portfolio_row(conn, portfolio_id, owner_id)
    enriched = []
    lineage_rows = []
    for holding in portfolio_holdings(conn, portfolio_id):
        payload, rows = holding_payload(conn, holding)
        enriched.append(payload)
        lineage_rows.extend(rows)
    return {
        "data": {
            "portfolio": portfolio,
            "holdings": enriched,
            "xray": xray_from_holdings(conn, enriched),
        },
        "metadata": {
            **metadata_from_rows(conn, lineage_rows),
            "holding_count": len(enriched),
        },
    }


def get_portfolio_xray(conn: sqlite3.Connection, portfolio_id: int, owner_id: str | None) -> dict[str, Any]:
    payload = get_portfolio(conn, portfolio_id, owner_id)
    return {
        "data": payload["data"]["xray"],
        "metadata": payload["metadata"],
    }
