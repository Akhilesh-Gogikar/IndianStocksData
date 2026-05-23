from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..dependencies import get_connection
from ..market_service import MarketDataUnavailable, MarketRecordNotFound
from ..portfolio_service import (
    add_holding,
    create_portfolio,
    get_portfolio,
    get_portfolio_xray,
    list_portfolios,
    remove_holding,
)

router = APIRouter(prefix="/portfolios", tags=["portfolios"])


class PortfolioCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: str | None = None
    base_currency: str = "INR"
    owner_id: str = "default"


class PortfolioHoldingCreate(BaseModel):
    ticker: str = Field(..., min_length=1)
    quantity: float = Field(..., gt=0)
    average_cost: float | None = Field(default=None, ge=0)
    notes: str | None = None
    owner_id: str = "default"


def api_error(exc: Exception) -> None:
    if isinstance(exc, MarketDataUnavailable):
        raise HTTPException(status_code=503, detail={"code": "portfolios_unavailable", "message": str(exc)}) from exc
    if isinstance(exc, MarketRecordNotFound):
        raise HTTPException(status_code=404, detail={"code": "portfolio_not_found", "message": str(exc)}) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail={"code": "invalid_portfolio_request", "message": str(exc)}) from exc
    raise exc


@router.get("")
def portfolios_index(
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_portfolios(conn, owner_id)
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.post("")
def portfolios_create(
    request: PortfolioCreate,
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return {"data": create_portfolio(conn, request.owner_id, request.name, request.description, request.base_currency)}
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.get("/{portfolio_id}")
def portfolios_show(
    portfolio_id: int,
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return get_portfolio(conn, portfolio_id, owner_id)
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.get("/{portfolio_id}/xray")
def portfolios_xray(
    portfolio_id: int,
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return get_portfolio_xray(conn, portfolio_id, owner_id)
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.post("/{portfolio_id}/holdings")
def portfolios_add_holding(
    portfolio_id: int,
    request: PortfolioHoldingCreate,
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return add_holding(
            conn,
            portfolio_id,
            request.owner_id,
            request.ticker,
            request.quantity,
            request.average_cost,
            request.notes,
        )
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.delete("/{portfolio_id}/holdings/{ticker}")
def portfolios_remove_holding(
    portfolio_id: int,
    ticker: str,
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return remove_holding(conn, portfolio_id, owner_id, ticker)
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)
