from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..dependencies import get_connection
from ..market_service import (
    MarketDataUnavailable,
    MarketRecordNotFound,
    get_company,
    get_events,
    get_peers,
    get_quote,
    get_ratios,
    screen_companies,
)

router = APIRouter(tags=["market"])


class NumericRange(BaseModel):
    min: float | None = None
    max: float | None = None


class ScreenRequest(BaseModel):
    tickers: list[str] | None = None
    sector: str | None = None
    industry: str | None = None
    min_market_cap: float | None = None
    max_market_cap: float | None = None
    min_price: float | None = None
    max_price: float | None = None
    ratio_filters: dict[str, NumericRange] = Field(default_factory=dict)
    limit: int = Field(default=50, ge=1, le=500)


def request_payload(model: ScreenRequest) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def raise_api_error(exc: Exception) -> None:
    if isinstance(exc, MarketDataUnavailable):
        raise HTTPException(
            status_code=503,
            detail={"code": "canonical_data_unavailable", "message": str(exc)},
        ) from exc
    if isinstance(exc, MarketRecordNotFound):
        raise HTTPException(
            status_code=404,
            detail={"code": "market_record_not_found", "message": str(exc)},
        ) from exc
    raise exc


@router.get("/companies/{ticker}")
def company_profile(
    ticker: str,
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return get_company(conn, ticker)
    except (MarketDataUnavailable, MarketRecordNotFound) as exc:
        raise_api_error(exc)


@router.get("/quotes/{ticker}")
def quote_snapshot(
    ticker: str,
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return get_quote(conn, ticker)
    except (MarketDataUnavailable, MarketRecordNotFound) as exc:
        raise_api_error(exc)


@router.get("/ratios/{ticker}")
def financial_ratios(
    ticker: str,
    period: str | None = Query(default=None),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return get_ratios(conn, ticker, period)
    except (MarketDataUnavailable, MarketRecordNotFound) as exc:
        raise_api_error(exc)


@router.get("/events/{ticker}")
def company_events(
    ticker: str,
    limit: int = Query(default=25, ge=1, le=250),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return get_events(conn, ticker, limit)
    except (MarketDataUnavailable, MarketRecordNotFound) as exc:
        raise_api_error(exc)


@router.get("/peers/{ticker}")
def company_peers(
    ticker: str,
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return get_peers(conn, ticker)
    except (MarketDataUnavailable, MarketRecordNotFound) as exc:
        raise_api_error(exc)


@router.post("/screen")
def stock_screen(
    request: ScreenRequest,
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return screen_companies(conn, request_payload(request))
    except (MarketDataUnavailable, MarketRecordNotFound) as exc:
        raise_api_error(exc)
