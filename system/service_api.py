"""Database-as-a-service API for Indian market intelligence data.

Provides current + historical access patterns that are friendly to AI agents.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from analyst_report import build_prompt_payload


app = FastAPI(title="Indian Stocks Data Service", version="1.0.0")
DB_PATH = Path("./system/market_intel.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def latest_completed_run(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT run_id, run_date, started_at, finished_at, notes
        FROM ingestion_runs
        WHERE status = 'completed'
        ORDER BY run_id DESC
        LIMIT 1
        """
    ).fetchone()


@app.get("/health")
def health() -> dict[str, str]:
    with closing(get_connection()) as conn:
        conn.execute("SELECT 1")
    return {"status": "ok"}


@app.get("/runs/latest")
def runs_latest() -> dict[str, Any]:
    with closing(get_connection()) as conn:
        run = latest_completed_run(conn)
        if not run:
            raise HTTPException(status_code=404, detail="No completed runs found")

        quality = conn.execute(
            """
            SELECT check_name, status, details, checked_at
            FROM data_quality_audits
            WHERE run_id = ?
            ORDER BY audit_id DESC
            """,
            (run["run_id"],),
        ).fetchall()

        return {
            "run": dict(run),
            "quality_checks": [dict(row) for row in quality],
        }


@app.get("/documents/current")
def documents_current(
    source_name: str | None = Query(default=None),
    file_type: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    with closing(get_connection()) as conn:
        run = latest_completed_run(conn)
        if not run:
            raise HTTPException(status_code=404, detail="No completed runs found")
        return documents_historical(run["run_id"], source_name, file_type, limit)


@app.get("/documents/historical/{run_id}")
def documents_historical(
    run_id: int,
    source_name: str | None = Query(default=None),
    file_type: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    with closing(get_connection()) as conn:
        clauses = ["run_id = ?"]
        params: list[Any] = [run_id]

        if source_name:
            clauses.append("source_name = ?")
            params.append(source_name)
        if file_type:
            clauses.append("file_type = ?")
            params.append(file_type)

        params.append(limit)
        query = f"""
            SELECT document_id, run_id, source_name, file_path, file_type,
                   content_sha256, record_count, source_timestamp, ingested_at, content
            FROM raw_documents
            WHERE {' AND '.join(clauses)}
            ORDER BY document_id DESC
            LIMIT ?
        """
        rows = conn.execute(query, params).fetchall()

        return {
            "run_id": run_id,
            "count": len(rows),
            "documents": [
                {
                    "document_id": row["document_id"],
                    "source_name": row["source_name"],
                    "file_path": row["file_path"],
                    "file_type": row["file_type"],
                    "content_sha256": row["content_sha256"],
                    "record_count": row["record_count"],
                    "source_timestamp": row["source_timestamp"],
                    "ingested_at": row["ingested_at"],
                    "content_preview": row["content"][:1200],
                }
                for row in rows
            ],
        }


@app.get("/agents/context/{ticker}")
def agent_context(ticker: str, limit: int = Query(default=20, ge=1, le=200)) -> dict[str, Any]:
    """LLM/agent optimized context payload including quality audit status."""
    with closing(get_connection()) as conn:
        run = latest_completed_run(conn)
        if not run:
            raise HTTPException(status_code=404, detail="No completed runs found")

        documents = conn.execute(
            """
            SELECT source_name, file_type, content
            FROM raw_documents
            WHERE run_id = ?
            ORDER BY document_id DESC
            LIMIT ?
            """,
            (run["run_id"], limit),
        ).fetchall()

        risk_signals = conn.execute(
            """
            SELECT entity_name, region, risk_type, signal_strength, rationale
            FROM entity_risk_signals
            WHERE run_id = ?
            ORDER BY signal_strength DESC
            LIMIT ?
            """,
            (run["run_id"], limit),
        ).fetchall()

        quality = conn.execute(
            """
            SELECT check_name, status, details
            FROM data_quality_audits
            WHERE run_id = ?
            ORDER BY audit_id DESC
            """,
            (run["run_id"],),
        ).fetchall()

        context = {
            "ticker": ticker,
            "run_id": run["run_id"],
            "run_date": run["run_date"],
            "documents": [
                {"source": d["source_name"], "type": d["file_type"], "content": d["content"][:3000]}
                for d in documents
            ],
            "risk_signals": [
                {
                    "entity": r["entity_name"],
                    "region": r["region"],
                    "risk_type": r["risk_type"],
                    "strength": r["signal_strength"],
                    "rationale": r["rationale"],
                }
                for r in risk_signals
            ],
            "quality_checks": [dict(row) for row in quality],
        }
        return build_prompt_payload(ticker, context)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DBaaS API for market intelligence")
    parser.add_argument("--db-path", default="./system/market_intel.db")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global DB_PATH
    DB_PATH = Path(args.db_path).resolve()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
