"""Owner-level AI morning brief across portfolios, watchlists, and screeners."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Callable

from system.ai.customer_digest import build_portfolio_digest, build_screener_digest, build_watchlist_digest
from system.api.freshness_service import build_freshness_report
from system.api.market_service import MarketDataUnavailable, MarketRecordNotFound
from system.api.portfolio_service import list_portfolios
from system.api.screener_service import list_screeners
from system.api.watchlist_service import list_watchlists


def build_owner_morning_brief(
    conn: sqlite3.Connection,
    owner_id: str | None,
    index_dir: Path,
    focus: str | None = None,
    max_items: int = 2,
    evidence_limit: int = 1,
    persist_screeners: bool = False,
) -> dict[str, Any]:
    owner = (owner_id or "default").strip() or "default"
    freshness = _safe_payload("freshness", lambda: build_freshness_report(conn))
    portfolio_digests = _portfolio_digests(conn, owner, index_dir, focus, max_items, evidence_limit)
    watchlist_digests = _watchlist_digests(conn, owner, index_dir, focus, max_items, evidence_limit)
    screener_digests = _screener_digests(conn, owner, index_dir, focus, max_items, evidence_limit, persist_screeners)
    gaps = _collect_gaps(freshness, portfolio_digests, watchlist_digests, screener_digests)
    priorities = _priorities(freshness, portfolio_digests, watchlist_digests, screener_digests, gaps)
    return {
        "kind": "owner_morning_brief",
        "owner_id": owner,
        "priorities": priorities,
        "freshness": freshness,
        "portfolio_digests": portfolio_digests,
        "watchlist_digests": watchlist_digests,
        "screener_digests": screener_digests,
        "data_gaps": gaps,
        "brief_markdown": _markdown(owner, priorities, portfolio_digests, watchlist_digests, screener_digests, gaps),
    }


def _portfolio_digests(
    conn: sqlite3.Connection,
    owner_id: str,
    index_dir: Path,
    focus: str | None,
    max_items: int,
    evidence_limit: int,
) -> list[dict[str, Any]]:
    listing = _safe_payload("portfolios", lambda: list_portfolios(conn, owner_id))
    return [
        _safe_payload(
            f"portfolio:{item['portfolio_id']}",
            lambda item=item: build_portfolio_digest(
                conn,
                int(item["portfolio_id"]),
                owner_id,
                index_dir,
                focus=focus,
                max_positions=2,
                evidence_limit=evidence_limit,
            ),
        )
        for item in listing.get("data", [])[:max(1, max_items)]
    ]


def _watchlist_digests(
    conn: sqlite3.Connection,
    owner_id: str,
    index_dir: Path,
    focus: str | None,
    max_items: int,
    evidence_limit: int,
) -> list[dict[str, Any]]:
    listing = _safe_payload("watchlists", lambda: list_watchlists(conn, owner_id))
    return [
        _safe_payload(
            f"watchlist:{item['watchlist_id']}",
            lambda item=item: build_watchlist_digest(
                conn,
                int(item["watchlist_id"]),
                owner_id,
                index_dir,
                focus=focus,
                max_items=3,
                evidence_limit=evidence_limit,
            ),
        )
        for item in listing.get("data", [])[:max(1, max_items)]
    ]


def _screener_digests(
    conn: sqlite3.Connection,
    owner_id: str,
    index_dir: Path,
    focus: str | None,
    max_items: int,
    evidence_limit: int,
    persist_screeners: bool,
) -> list[dict[str, Any]]:
    listing = _safe_payload("screeners", lambda: list_screeners(conn, owner_id))
    return [
        _safe_payload(
            f"screener:{item['screener_id']}",
            lambda item=item: build_screener_digest(
                conn,
                int(item["screener_id"]),
                owner_id,
                index_dir,
                focus=focus,
                max_results=3,
                evidence_limit=evidence_limit,
                persist=persist_screeners,
            ),
        )
        for item in listing.get("data", [])[:max(1, max_items)]
    ]


def _safe_payload(label: str, fetcher: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return fetcher()
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError, sqlite3.OperationalError) as exc:
        return {"kind": label, "error": str(exc), "data_gaps": [str(exc)]}


def _collect_gaps(
    freshness: dict[str, Any],
    portfolios: list[dict[str, Any]],
    watchlists: list[dict[str, Any]],
    screeners: list[dict[str, Any]],
) -> list[str]:
    gaps: list[str] = []
    if freshness.get("error"):
        gaps.append(f"freshness: {freshness['error']}")
    for group_name, items in (("portfolio", portfolios), ("watchlist", watchlists), ("screener", screeners)):
        for item in items:
            if item.get("error"):
                gaps.append(f"{group_name}: {item['error']}")
            gaps.extend(f"{group_name}: {gap}" for gap in item.get("data_gaps", []))
    return gaps


def _priorities(
    freshness: dict[str, Any],
    portfolios: list[dict[str, Any]],
    watchlists: list[dict[str, Any]],
    screeners: list[dict[str, Any]],
    gaps: list[str],
) -> list[dict[str, Any]]:
    priorities: list[dict[str, Any]] = []
    status = freshness.get("data", {}).get("overall_status")
    if status and status not in {"fresh", "empty"}:
        priorities.append({"priority": 1, "type": "freshness", "message": f"Review data freshness: {status}."})

    for digest in watchlists:
        triggered = digest.get("alerts", {}).get("metadata", {}).get("triggered_count", 0)
        if triggered:
            name = digest.get("watchlist", {}).get("name", "watchlist")
            priorities.append({"priority": 2, "type": "watchlist_alerts", "message": f"{name} has {triggered} triggered alert checks."})

    for digest in portfolios:
        concentration = (digest.get("xray", {}).get("top_concentration") or [{}])[0]
        weight = concentration.get("portfolio_weight")
        if weight is not None and weight >= 0.35:
            priorities.append({"priority": 3, "type": "portfolio_risk", "message": f"Review {concentration['ticker']} concentration at {weight:.1%}."})

    for digest in screeners:
        result_count = digest.get("evaluation", {}).get("metadata", {}).get("result_count", 0)
        if result_count:
            name = digest.get("screener", {}).get("name", "screener")
            priorities.append({"priority": 4, "type": "opportunity", "message": f"{name} has {result_count} current matches to review."})

    if gaps:
        priorities.append({"priority": 5, "type": "data_gaps", "message": f"Resolve {len(gaps)} data gaps before publishing customer-facing guidance."})
    if not priorities:
        priorities.append({"priority": 1, "type": "status", "message": "No urgent customer action detected from available local data."})
    return sorted(priorities, key=lambda item: item["priority"])


def _markdown(
    owner_id: str,
    priorities: list[dict[str, Any]],
    portfolios: list[dict[str, Any]],
    watchlists: list[dict[str, Any]],
    screeners: list[dict[str, Any]],
    gaps: list[str],
) -> str:
    lines = [f"# AI Morning Brief: {owner_id}", "", "## Priorities"]
    lines.extend(f"- P{item['priority']} {item['type']}: {item['message']}" for item in priorities)
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"- Portfolios: {len(portfolios)}")
    lines.append(f"- Watchlists: {len(watchlists)}")
    lines.append(f"- Screeners: {len(screeners)}")
    lines.append("")
    lines.append("## Data Gaps")
    lines.extend(f"- {gap}" for gap in gaps) if gaps else lines.append("- None detected.")
    return "\n".join(lines)
