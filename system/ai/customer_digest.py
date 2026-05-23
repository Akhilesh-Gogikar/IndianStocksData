"""Customer-level AI digests for portfolios and watchlists."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from system.ai.research_brief import build_research_brief
from system.ai.vector_index import VectorIndexError
from system.api.market_service import MarketDataUnavailable, MarketRecordNotFound
from system.api.portfolio_service import get_portfolio
from system.api.screener_service import evaluate_screener, get_screener
from system.api.watchlist_service import evaluate_alerts, get_watchlist


def build_portfolio_digest(
    conn: sqlite3.Connection,
    portfolio_id: int,
    owner_id: str | None,
    index_dir: Path,
    focus: str | None = None,
    max_positions: int = 5,
    evidence_limit: int = 3,
) -> dict[str, Any]:
    portfolio_payload = get_portfolio(conn, portfolio_id, owner_id)
    holdings = portfolio_payload["data"]["holdings"]
    selected = _top_holdings(holdings, max_positions)
    ticker_digests = [
        _ticker_digest(conn, item["ticker"], index_dir, focus, evidence_limit, item)
        for item in selected
    ]
    xray = portfolio_payload["data"]["xray"]
    actions = _portfolio_actions(xray, ticker_digests)
    return {
        "kind": "portfolio_digest",
        "portfolio": portfolio_payload["data"]["portfolio"],
        "xray": xray,
        "ticker_digests": ticker_digests,
        "customer_next_actions": actions,
        "data_gaps": _collect_gaps(ticker_digests, xray.get("missing_quotes", [])),
        "digest_markdown": _portfolio_markdown(portfolio_payload["data"]["portfolio"], xray, ticker_digests, actions),
        "metadata": portfolio_payload["metadata"],
    }


def build_watchlist_digest(
    conn: sqlite3.Connection,
    watchlist_id: int,
    owner_id: str | None,
    index_dir: Path,
    focus: str | None = None,
    max_items: int = 10,
    evidence_limit: int = 3,
) -> dict[str, Any]:
    watchlist_payload = get_watchlist(conn, watchlist_id, owner_id)
    items = watchlist_payload["data"]["items"][:max_items]
    ticker_digests = [
        _ticker_digest(conn, item["ticker"], index_dir, focus, evidence_limit, item)
        for item in items
    ]
    alerts = _safe_alerts(conn, watchlist_id, owner_id)
    actions = _watchlist_actions(ticker_digests, alerts)
    return {
        "kind": "watchlist_digest",
        "watchlist": watchlist_payload["data"]["watchlist"],
        "ticker_digests": ticker_digests,
        "alerts": alerts,
        "customer_next_actions": actions,
        "data_gaps": _collect_gaps(ticker_digests, []),
        "digest_markdown": _watchlist_markdown(watchlist_payload["data"]["watchlist"], ticker_digests, alerts, actions),
        "metadata": watchlist_payload["metadata"],
    }


def build_screener_digest(
    conn: sqlite3.Connection,
    screener_id: int,
    owner_id: str | None,
    index_dir: Path,
    focus: str | None = None,
    max_results: int = 10,
    evidence_limit: int = 3,
    persist: bool = True,
) -> dict[str, Any]:
    screener_payload = get_screener(conn, screener_id, owner_id)
    evaluation = evaluate_screener(conn, screener_id, owner_id, persist=persist)
    results = evaluation["data"][:max_results]
    ticker_digests = [
        _ticker_digest(conn, item["company"]["ticker"], index_dir, focus, evidence_limit, item)
        for item in results
    ]
    actions = _screener_actions(evaluation, ticker_digests)
    return {
        "kind": "screener_digest",
        "screener": screener_payload["data"]["screener"],
        "evaluation": {
            "metadata": evaluation["metadata"],
            "top_results": results,
        },
        "ticker_digests": ticker_digests,
        "customer_next_actions": actions,
        "data_gaps": _collect_gaps(ticker_digests, []),
        "digest_markdown": _screener_markdown(screener_payload["data"]["screener"], evaluation, ticker_digests, actions),
        "metadata": evaluation["metadata"],
    }


def _top_holdings(holdings: list[dict[str, Any]], max_positions: int) -> list[dict[str, Any]]:
    return sorted(
        holdings,
        key=lambda item: item.get("market_value") or 0,
        reverse=True,
    )[: max(1, max_positions)]


def _ticker_digest(
    conn: sqlite3.Connection,
    ticker: str,
    index_dir: Path,
    focus: str | None,
    evidence_limit: int,
    customer_context: dict[str, Any],
) -> dict[str, Any]:
    try:
        brief = build_research_brief(
            conn,
            ticker,
            index_dir,
            focus=focus,
            evidence_limit=evidence_limit,
        )
        return {
            "ticker": ticker,
            "summary": brief["summary"],
            "answer": brief["answer"],
            "evidence": brief["evidence"],
            "data_gaps": brief["data_gaps"],
            "customer_context": customer_context,
        }
    except VectorIndexError as exc:
        return {
            "ticker": ticker,
            "summary": f"{ticker} needs local evidence before a grounded AI digest can be produced.",
            "answer": {"stance": "needs_more_data", "rationale": str(exc), "confidence": "low"},
            "evidence": [],
            "data_gaps": [str(exc)],
            "customer_context": customer_context,
        }


def _safe_alerts(conn: sqlite3.Connection, watchlist_id: int, owner_id: str | None) -> dict[str, Any]:
    try:
        return evaluate_alerts(conn, watchlist_id, owner_id)
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError, sqlite3.OperationalError) as exc:
        return {"data": [], "metadata": {"triggered_count": 0, "error": str(exc)}}


def _portfolio_actions(xray: dict[str, Any], ticker_digests: list[dict[str, Any]]) -> list[str]:
    actions = []
    missing = xray.get("missing_quotes") or []
    if missing:
        actions.append(f"Refresh quote data for missing holdings: {', '.join(missing)}.")
    if xray.get("top_concentration"):
        top = xray["top_concentration"][0]
        weight = top.get("portfolio_weight")
        if weight is not None and weight >= 0.35:
            actions.append(f"Review concentration risk in {top['ticker']} at {weight:.1%} of known market value.")
    research_ready = [item["ticker"] for item in ticker_digests if item["answer"].get("stance") == "research_ready"]
    if research_ready:
        actions.append(f"Start customer review with cited evidence for: {', '.join(research_ready[:3])}.")
    actions.append("Treat uncited or stale evidence as a data gap, not as a neutral signal.")
    return actions


def _watchlist_actions(ticker_digests: list[dict[str, Any]], alerts: dict[str, Any]) -> list[str]:
    actions = []
    triggered = [item for item in alerts.get("data", []) if item.get("triggered")]
    if triggered:
        tickers = sorted({item["ticker"] for item in triggered})
        actions.append(f"Prioritize triggered alert review for: {', '.join(tickers)}.")
    ready = [item["ticker"] for item in ticker_digests if item["answer"].get("stance") == "research_ready"]
    if ready:
        actions.append(f"Use cited briefs to update watchlist notes for: {', '.join(ready[:5])}.")
    actions.append("Add alert rules for watchlist names that imply monitoring intent but have no triggered checks.")
    return actions


def _screener_actions(evaluation: dict[str, Any], ticker_digests: list[dict[str, Any]]) -> list[str]:
    result_count = evaluation.get("metadata", {}).get("result_count", 0)
    actions = []
    if result_count == 0:
        actions.append("Relax or revise the screener filters before using it in a customer workflow.")
    else:
        top = evaluation.get("metadata", {}).get("top_tickers", [])[:5]
        actions.append(f"Review top screener matches with cited evidence: {', '.join(top)}.")
    ready = [item["ticker"] for item in ticker_digests if item["answer"].get("stance") == "research_ready"]
    if ready:
        actions.append(f"Promote research-ready matches into watchlists or newsletter blocks: {', '.join(ready[:5])}.")
    actions.append("Treat the saved filter as a strategy definition; publish only with freshness and quality metadata.")
    return actions


def _collect_gaps(ticker_digests: list[dict[str, Any]], missing_quotes: list[str]) -> list[str]:
    gaps = [f"Missing quote for {ticker}" for ticker in missing_quotes]
    for item in ticker_digests:
        gaps.extend(f"{item['ticker']}: {gap}" for gap in item.get("data_gaps", []))
    return gaps


def _portfolio_markdown(
    portfolio: dict[str, Any],
    xray: dict[str, Any],
    ticker_digests: list[dict[str, Any]],
    actions: list[str],
) -> str:
    lines = [f"# Portfolio AI Digest: {portfolio['name']}", ""]
    lines.append(f"Known market value: {xray.get('total_market_value')}")
    lines.append("")
    lines.append("## Top Positions")
    for item in ticker_digests:
        lines.append(f"- {item['ticker']}: {item['summary']}")
    lines.append("")
    lines.append("## Next Actions")
    lines.extend(f"- {action}" for action in actions)
    return "\n".join(lines)


def _watchlist_markdown(
    watchlist: dict[str, Any],
    ticker_digests: list[dict[str, Any]],
    alerts: dict[str, Any],
    actions: list[str],
) -> str:
    lines = [f"# Watchlist AI Digest: {watchlist['name']}", ""]
    lines.append(f"Triggered alerts: {alerts.get('metadata', {}).get('triggered_count', 0)}")
    lines.append("")
    lines.append("## Names")
    for item in ticker_digests:
        lines.append(f"- {item['ticker']}: {item['summary']}")
    lines.append("")
    lines.append("## Next Actions")
    lines.extend(f"- {action}" for action in actions)
    return "\n".join(lines)


def _screener_markdown(
    screener: dict[str, Any],
    evaluation: dict[str, Any],
    ticker_digests: list[dict[str, Any]],
    actions: list[str],
) -> str:
    metadata = evaluation.get("metadata", {})
    lines = [f"# Screener AI Digest: {screener['name']}", ""]
    lines.append(f"Matches: {metadata.get('result_count', 0)}")
    lines.append("")
    lines.append("## Top Matches")
    for item in ticker_digests:
        lines.append(f"- {item['ticker']}: {item['summary']}")
    lines.append("")
    lines.append("## Next Actions")
    lines.extend(f"- {action}" for action in actions)
    return "\n".join(lines)
