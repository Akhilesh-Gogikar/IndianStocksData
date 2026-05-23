"""Grounded AI research briefs built from local market evidence."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Callable

from system.ai.vector_index import VectorIndexError, build_vector_index, search_vectors
from system.api.market_service import (
    MarketDataUnavailable,
    MarketRecordNotFound,
    get_company,
    get_events,
    get_peers,
    get_quote,
    get_ratios,
    normalize_ticker,
)


DEFAULT_FOCUS = "investment thesis, catalysts, risks, valuation, and recent business performance"


def build_research_brief(
    conn: sqlite3.Connection,
    ticker: str,
    index_dir: Path,
    focus: str | None = None,
    run_id: int | None = None,
    source_name: str | None = None,
    evidence_limit: int = 8,
    auto_build: bool = True,
    build_limit: int = 1000,
) -> dict[str, Any]:
    symbol = normalize_ticker(ticker)
    brief_focus = (focus or DEFAULT_FOCUS).strip()
    query = f"{symbol} {brief_focus}"

    evidence_payload, vector_note = _search_or_build(
        conn,
        query,
        index_dir,
        run_id=run_id,
        source_name=source_name,
        evidence_limit=evidence_limit,
        auto_build=auto_build,
        build_limit=build_limit,
    )
    evidence = evidence_payload.get("results", [])
    market_context = _market_context(conn, symbol)
    gaps = _data_gaps(evidence, market_context, vector_note)

    return {
        "ticker": symbol,
        "focus": brief_focus,
        "run_id": evidence_payload.get("run_id"),
        "summary": _summary(symbol, evidence, market_context),
        "answer": _answer(evidence, market_context),
        "evidence": [_evidence_item(item) for item in evidence],
        "market_context": market_context,
        "customer_next_actions": _next_actions(symbol, gaps),
        "data_gaps": gaps,
        "brief_markdown": _brief_markdown(symbol, evidence, market_context, gaps),
        "retrieval": {
            "backend": evidence_payload.get("backend"),
            "count": evidence_payload.get("count", 0),
            "notes": vector_note or evidence_payload.get("notes"),
            "query": query,
        },
    }


def _search_or_build(
    conn: sqlite3.Connection,
    query: str,
    index_dir: Path,
    run_id: int | None,
    source_name: str | None,
    evidence_limit: int,
    auto_build: bool,
    build_limit: int,
) -> tuple[dict[str, Any], str | None]:
    try:
        payload = search_vectors(conn, query, index_dir, run_id=run_id, source_name=source_name, k=evidence_limit)
        return payload, None
    except VectorIndexError as exc:
        if not auto_build or "No vector index" not in str(exc):
            raise

    state = build_vector_index(conn, index_dir, run_id=run_id, source_name=source_name, limit=build_limit)
    payload = search_vectors(
        conn,
        query,
        index_dir,
        run_id=int(state["run_id"]),
        source_name=source_name,
        k=evidence_limit,
    )
    return payload, "Vector index was built automatically before search."


def _market_context(conn: sqlite3.Connection, ticker: str) -> dict[str, Any]:
    return {
        "company": _safe_market_call(lambda: get_company(conn, ticker)),
        "quote": _safe_market_call(lambda: get_quote(conn, ticker)),
        "ratios": _safe_market_call(lambda: get_ratios(conn, ticker)),
        "events": _safe_market_call(lambda: get_events(conn, ticker, limit=5)),
        "peers": _safe_market_call(lambda: get_peers(conn, ticker)),
    }


def _safe_market_call(fetcher: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return {"available": True, "payload": fetcher()}
    except (MarketDataUnavailable, MarketRecordNotFound, sqlite3.OperationalError) as exc:
        return {"available": False, "error": str(exc)}


def _summary(ticker: str, evidence: list[dict[str, Any]], market_context: dict[str, Any]) -> str:
    company = market_context["company"]
    name = company["payload"].get("data", {}).get("name") if company["available"] else None
    display = name or ticker
    if evidence:
        return f"{display} has {len(evidence)} local evidence matches for the requested research focus."
    return f"{display} does not yet have enough indexed local evidence for a grounded AI brief."


def _answer(evidence: list[dict[str, Any]], market_context: dict[str, Any]) -> dict[str, Any]:
    top_sources = [item["source_name"] for item in evidence[:3]]
    quote = market_context["quote"]
    rationale = "Local semantic evidence is available." if evidence else "Build or refresh the local repository first."
    if quote["available"]:
        rationale += " Latest processed quote metadata is available."
    return {
        "stance": "research_ready" if evidence else "needs_more_data",
        "rationale": rationale,
        "top_sources": top_sources,
        "confidence": "medium" if evidence else "low",
    }


def _evidence_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": item["document_id"],
        "score": item["score"],
        "source_name": item["source_name"],
        "file_path": item["file_path"],
        "content_sha256": item["content_sha256"],
        "excerpt": item["content_preview"][:700],
    }


def _data_gaps(
    evidence: list[dict[str, Any]],
    market_context: dict[str, Any],
    vector_note: str | None,
) -> list[str]:
    gaps: list[str] = []
    if not evidence:
        gaps.append("No local semantic evidence matched the research query.")
    for key, value in market_context.items():
        if not value["available"]:
            gaps.append(f"{key} unavailable: {value['error']}")
    if vector_note:
        gaps.append(vector_note)
    return gaps


def _next_actions(ticker: str, gaps: list[str]) -> list[str]:
    actions = [
        f"Review the cited evidence before publishing any {ticker} recommendation.",
        "Refresh local data if quote or fundamentals metadata is stale.",
        "Escalate material gaps to ingestion instead of treating missing data as neutral.",
    ]
    if gaps:
        actions.insert(0, "Resolve the listed data gaps before using this brief for customer decisions.")
    return actions


def _brief_markdown(
    ticker: str,
    evidence: list[dict[str, Any]],
    market_context: dict[str, Any],
    gaps: list[str],
) -> str:
    lines = [f"# {ticker} Grounded Research Brief", ""]
    lines.append(_summary(ticker, evidence, market_context))
    lines.append("")
    lines.append("## Evidence")
    if evidence:
        for item in evidence[:5]:
            excerpt = item["content_preview"][:220].replace("\n", " ")
            lines.append(f"- [{item['document_id']}] {item['source_name']} score={item['score']:.4f}: {excerpt}")
    else:
        lines.append("- No indexed evidence found.")
    lines.append("")
    lines.append("## Data Gaps")
    if gaps:
        lines.extend(f"- {gap}" for gap in gaps)
    else:
        lines.append("- None detected.")
    return "\n".join(lines)
