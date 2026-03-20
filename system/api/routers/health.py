from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, Request

from ..dependencies import get_connection

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request, conn: sqlite3.Connection = Depends(get_connection)) -> dict[str, object]:
    conn.execute("SELECT 1")
    return {
        "status": "ok",
        "profile": request.app.state.profile["name"],
        "active_apis": request.app.state.profile["router_names"],
    }
