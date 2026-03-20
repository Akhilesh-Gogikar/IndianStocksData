from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from system.analyst_report import build_prompt_payload

from ..dependencies import get_connection
from ..query_service import (
    fetch_documents,
    fetch_quality_checks,
    fetch_risk_signals,
    latest_completed_run,
)

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("/context/{ticker}")
def agent_context(
    ticker: str,
    limit: int = Query(default=20, ge=1, le=200),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    run = latest_completed_run(conn)
    if not run:
        raise HTTPException(status_code=404, detail="No completed runs found")

    context = {
        "ticker": ticker,
        "run_id": run["run_id"],
        "run_date": run["run_date"],
        "documents": [
            {
                "source": row["source_name"],
                "type": row["file_type"],
                "content": row["content_preview"][:3000],
            }
            for row in fetch_documents(conn, int(run["run_id"]), limit=limit)
        ],
        "risk_signals": fetch_risk_signals(conn, int(run["run_id"]), limit),
        "quality_checks": fetch_quality_checks(conn, int(run["run_id"])),
    }
    return build_prompt_payload(ticker, context)


@router.get("/workflow/{ticker}")
def agent_workflow(
    ticker: str,
    request: Request,
) -> dict[str, Any]:
    base_url = str(request.base_url).rstrip("/")
    return {
        "ticker": ticker,
        "profile": request.app.state.profile["name"],
        "recommended_sequence": [
            {
                "step": 1,
                "tool": "runs.latest",
                "endpoint": f"{base_url}/runs/latest",
                "purpose": "Check freshness and recent audit results before reasoning.",
            },
            {
                "step": 2,
                "tool": "documents.current",
                "endpoint": f"{base_url}/documents/current?limit=25",
                "purpose": "Fetch the most recent market artifacts for retrieval grounding.",
            },
            {
                "step": 3,
                "tool": "agents.context",
                "endpoint": f"{base_url}/agents/context/{ticker}",
                "purpose": "Build a model-ready payload for analysis and planning.",
            },
        ],
        "operating_guidance": [
            "Prefer the most recent completed run.",
            "Surface quality check failures to downstream agents.",
            "Treat missing coverage as a data gap rather than a neutral signal.",
        ],
    }

