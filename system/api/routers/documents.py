from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from ..dependencies import get_connection
from ..query_service import fetch_documents, latest_completed_run

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("/current")
def documents_current(
    source_name: str | None = Query(default=None),
    file_type: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    run = latest_completed_run(conn)
    if not run:
        raise HTTPException(status_code=404, detail="No completed runs found")
    return documents_historical(int(run["run_id"]), source_name, file_type, limit, conn)


@router.get("/historical/{run_id}")
def documents_historical(
    run_id: int,
    source_name: str | None = Query(default=None),
    file_type: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    rows = fetch_documents(conn, run_id, source_name, file_type, limit)
    return {
        "run_id": run_id,
        "count": len(rows),
        "documents": rows,
    }
