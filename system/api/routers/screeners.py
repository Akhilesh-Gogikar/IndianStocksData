from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..dependencies import get_connection
from ..market_service import MarketDataUnavailable, MarketRecordNotFound
from ..screener_service import create_screener, evaluate_screener, get_screener, list_screeners

router = APIRouter(prefix="/screeners", tags=["screeners"])


class ScreenerCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    owner_id: str = "default"


def api_error(exc: Exception) -> None:
    if isinstance(exc, MarketDataUnavailable):
        raise HTTPException(status_code=503, detail={"code": "screeners_unavailable", "message": str(exc)}) from exc
    if isinstance(exc, MarketRecordNotFound):
        raise HTTPException(status_code=404, detail={"code": "screener_not_found", "message": str(exc)}) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail={"code": "invalid_screener_request", "message": str(exc)}) from exc
    raise exc


@router.get("")
def screeners_index(
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_screeners(conn, owner_id)
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.post("")
def screeners_create(
    request: ScreenerCreate,
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return {"data": create_screener(conn, request.owner_id, request.name, request.description, request.filters)}
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.get("/{screener_id}")
def screeners_show(
    screener_id: int,
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return get_screener(conn, screener_id, owner_id)
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.post("/{screener_id}/evaluate")
def screeners_evaluate(
    screener_id: int,
    owner_id: str = Query(default="default"),
    persist: bool = Query(default=True),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return evaluate_screener(conn, screener_id, owner_id, persist=persist)
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)
