from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_connection
from ..freshness_service import build_freshness_report, build_ticker_freshness

router = APIRouter(prefix="/freshness", tags=["freshness"])


@router.get("")
def freshness_index(
    warn_days: int = Query(default=2, ge=0, le=30),
    stale_days: int = Query(default=5, ge=1, le=90),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    return build_freshness_report(conn, warn_days=warn_days, stale_days=stale_days)


@router.get("/{ticker}")
def freshness_ticker(
    ticker: str,
    warn_days: int = Query(default=2, ge=0, le=30),
    stale_days: int = Query(default=5, ge=1, le=90),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    return build_ticker_freshness(conn, ticker, warn_days=warn_days, stale_days=stale_days)
