"""Natural-language stock screener copilot."""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from system.api.market_service import screen_companies
from system.api.screener_service import create_screener


SECTOR_ALIASES = {
    "energy": "Energy",
    "oil": "Energy",
    "refinery": "Energy",
    "technology": "Technology",
    "tech": "Technology",
    "it": "Technology",
    "bank": "Financials",
    "banking": "Financials",
    "financial": "Financials",
    "finance": "Financials",
    "pharma": "Healthcare",
    "healthcare": "Healthcare",
    "auto": "Automobile",
    "automobile": "Automobile",
    "consumer": "Consumer",
    "fmcg": "Consumer",
    "telecom": "Telecom",
}

RATIO_ALIASES = {
    "pe": "pe",
    "p/e": "pe",
    "price earnings": "pe",
    "pb": "pb",
    "p/b": "pb",
    "roe": "roe",
    "roce": "roce",
    "debt equity": "debt_to_equity",
}


def run_screener_copilot(
    conn: sqlite3.Connection,
    prompt: str,
    owner_id: str | None = None,
    save: bool = False,
    name: str | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    parsed = interpret_prompt(prompt, limit)
    result = screen_companies(conn, parsed["filters"])
    matches = [_match_explanation(row, parsed["filters"]) for row in result["data"]]
    saved = None
    if save:
        saved = create_screener(
            conn,
            owner_id,
            name or parsed["suggested_name"],
            prompt,
            parsed["filters"],
        )
    return {
        "kind": "screener_copilot",
        "prompt": prompt,
        "filters": parsed["filters"],
        "interpretation": parsed["interpretation"],
        "result": result,
        "match_explanations": matches,
        "saved_screener": saved,
        "customer_next_actions": _next_actions(result, save),
    }


def interpret_prompt(prompt: str, limit: int = 25) -> dict[str, Any]:
    text = (prompt or "").strip()
    normalized = text.lower()
    filters: dict[str, Any] = {"limit": max(1, min(limit, 100))}
    interpretation: list[str] = []

    explicit_limit = _first_number_after(normalized, r"\b(top|first|limit)\b")
    if explicit_limit is not None:
        filters["limit"] = max(1, min(int(explicit_limit), 100))
        interpretation.append(f"Limited results to {filters['limit']}.")

    for token, sector in SECTOR_ALIASES.items():
        if re.search(rf"\b{re.escape(token)}\b", normalized):
            filters["sector"] = sector
            interpretation.append(f"Mapped '{token}' to sector={sector}.")
            break

    ratio_filters: dict[str, dict[str, float]] = {}
    for alias, ratio_name in RATIO_ALIASES.items():
        bounds = _metric_bounds(normalized, alias)
        if bounds:
            ratio_filters[ratio_name] = bounds
            interpretation.append(f"Mapped '{alias}' to ratio filter {ratio_name}={bounds}.")
    if ratio_filters:
        filters["ratio_filters"] = ratio_filters

    price_bounds = _metric_bounds(normalized, "price")
    if price_bounds.get("min") is not None:
        filters["min_price"] = price_bounds["min"]
        interpretation.append(f"Set min_price={price_bounds['min']}.")
    if price_bounds.get("max") is not None:
        filters["max_price"] = price_bounds["max"]
        interpretation.append(f"Set max_price={price_bounds['max']}.")

    market_cap_bounds = _market_cap_bounds(normalized)
    filters.update(market_cap_bounds)
    for key, value in market_cap_bounds.items():
        interpretation.append(f"Set {key}={value}.")

    if "value" in normalized or "cheap" in normalized:
        filters.setdefault("ratio_filters", {}).setdefault("pe", {}).setdefault("max", 30.0)
        interpretation.append("Mapped value/cheap language to PE <= 30 when no PE maximum was specified.")

    if len(filters) == 1:
        interpretation.append("No specific filters were detected; running a broad canonical screen.")

    return {
        "filters": filters,
        "interpretation": interpretation,
        "suggested_name": _suggested_name(text),
    }


def _metric_bounds(text: str, metric: str) -> dict[str, float]:
    escaped = re.escape(metric)
    bounds: dict[str, float] = {}
    max_match = re.search(rf"{escaped}[^\d]{{0,24}}(under|below|less than|lte|<=|max)[^\d]*(\d+(?:\.\d+)?)", text)
    min_match = re.search(rf"{escaped}[^\d]{{0,24}}(over|above|greater than|gte|>=|min)[^\d]*(\d+(?:\.\d+)?)", text)
    reverse_max = re.search(rf"(under|below|less than|lte|<=|max)[^\d]*(\d+(?:\.\d+)?)[^\w]{{0,12}}{escaped}", text)
    reverse_min = re.search(rf"(over|above|greater than|gte|>=|min)[^\d]*(\d+(?:\.\d+)?)[^\w]{{0,12}}{escaped}", text)
    if max_match:
        bounds["max"] = float(max_match.group(2))
    elif reverse_max:
        bounds["max"] = float(reverse_max.group(2))
    if min_match:
        bounds["min"] = float(min_match.group(2))
    elif reverse_min:
        bounds["min"] = float(reverse_min.group(2))
    return bounds


def _market_cap_bounds(text: str) -> dict[str, float]:
    bounds: dict[str, float] = {}
    if "large cap" in text or "large-cap" in text:
        bounds["min_market_cap"] = 500000.0
    if "mid cap" in text or "mid-cap" in text:
        bounds["min_market_cap"] = 50000.0
        bounds["max_market_cap"] = 500000.0
    if "small cap" in text or "small-cap" in text:
        bounds["max_market_cap"] = 50000.0
    explicit = _metric_bounds(text, "market cap")
    if explicit.get("min") is not None:
        bounds["min_market_cap"] = explicit["min"]
    if explicit.get("max") is not None:
        bounds["max_market_cap"] = explicit["max"]
    return bounds


def _first_number_after(text: str, marker_pattern: str) -> float | None:
    match = re.search(rf"{marker_pattern}[^\d]*(\d+(?:\.\d+)?)", text)
    return float(match.group(2)) if match else None


def _match_explanation(row: dict[str, Any], filters: dict[str, Any]) -> dict[str, Any]:
    company = row.get("company") or {}
    quote = row.get("quote") or {}
    ratios = row.get("ratios") or {}
    reasons = []
    if filters.get("sector"):
        reasons.append(f"sector={company.get('sector')}")
    if filters.get("min_market_cap") is not None or filters.get("max_market_cap") is not None:
        reasons.append(f"market_cap={company.get('market_cap')}")
    if filters.get("min_price") is not None or filters.get("max_price") is not None:
        reasons.append(f"price={quote.get('price')}")
    for ratio_name, bounds in (filters.get("ratio_filters") or {}).items():
        reasons.append(f"{ratio_name}={ratios.get(ratio_name)} within {bounds}")
    return {
        "ticker": company.get("ticker"),
        "name": company.get("name"),
        "why_matched": reasons or ["Matched broad canonical screen."],
    }


def _suggested_name(prompt: str) -> str:
    words = re.sub(r"[^A-Za-z0-9 ]+", " ", prompt).split()
    short = " ".join(words[:8]).strip()
    return short or "AI generated screener"


def _next_actions(result: dict[str, Any], saved: bool) -> list[str]:
    actions = []
    count = result.get("metadata", {}).get("result_count", 0)
    if count == 0:
        actions.append("Relax one filter or refresh canonical market data before showing this to customers.")
    else:
        actions.append("Review the explained matches and convert strong candidates into a watchlist.")
    if not saved:
        actions.append("Save this screen if the customer wants repeat monitoring.")
    actions.append("Use research briefs for shortlisted names before making any recommendation.")
    return actions
