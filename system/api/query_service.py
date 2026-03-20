from __future__ import annotations

import sqlite3
from typing import Any


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


def fetch_quality_checks(conn: sqlite3.Connection, run_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT check_name, status, details, checked_at
        FROM data_quality_audits
        WHERE run_id = ?
        ORDER BY audit_id DESC
        """,
        (run_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def fetch_documents(
    conn: sqlite3.Connection,
    run_id: int,
    source_name: str | None = None,
    file_type: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
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
    return [
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
    ]


def fetch_risk_signals(conn: sqlite3.Connection, run_id: int, limit: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT entity_name, region, risk_type, signal_strength, rationale
        FROM entity_risk_signals
        WHERE run_id = ?
        ORDER BY signal_strength DESC
        LIMIT ?
        """,
        (run_id, limit),
    ).fetchall()
    return [
        {
            "entity": row["entity_name"],
            "region": row["region"],
            "risk_type": row["risk_type"],
            "strength": row["signal_strength"],
            "rationale": row["rationale"],
        }
        for row in rows
    ]
