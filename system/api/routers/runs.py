from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_connection
from ..query_service import fetch_quality_checks, latest_completed_run

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("/latest")
def runs_latest(conn: sqlite3.Connection = Depends(get_connection)) -> dict[str, Any]:
    run = latest_completed_run(conn)
    if not run:
        raise HTTPException(status_code=404, detail="No completed runs found")

    return {
        "run": dict(run),
        "quality_checks": fetch_quality_checks(conn, int(run["run_id"])),
    }
