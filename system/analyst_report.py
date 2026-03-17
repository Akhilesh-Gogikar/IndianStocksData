"""Query helper to generate LLM-ready analyst report prompts.

This module does not call any external LLM SDK directly. Instead, it builds a
structured prompt payload that can be passed to your preferred model runtime.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from textwrap import dedent


SYSTEM_PROMPT = dedent(
    """
    You are a senior econometric equity analyst.
    Use global and regional news, entity relationships, and market context to:
    1) assess operational and macro risk,
    2) estimate near-term and medium-term growth,
    3) perform fundamental analysis,
    4) produce an investment recommendation with timeline and confidence.

    Output sections:
    - Executive Summary
    - Macro & Geopolitical Risk
    - Regional Business Risk
    - Fundamental Analysis
    - Relationship/Dependency Mapping
    - Valuation Perspective
    - Recommendation (Buy/Hold/Sell) and time horizon
    - Key Triggers and Monitoring Plan
    """
).strip()


def fetch_context(conn: sqlite3.Connection, ticker: str, limit: int = 20) -> dict:
    run = conn.execute(
        "SELECT run_id, started_at FROM ingestion_runs WHERE status='completed' ORDER BY run_id DESC LIMIT 1"
    ).fetchone()
    if not run:
        raise RuntimeError("No completed ingestion runs found.")

    run_id = run[0]
    documents = conn.execute(
        """
        SELECT source_name, file_type, content
        FROM raw_documents
        WHERE run_id = ?
        ORDER BY document_id DESC
        LIMIT ?
        """,
        (run_id, limit),
    ).fetchall()

    risk_signals = conn.execute(
        """
        SELECT entity_name, region, risk_type, signal_strength, rationale
        FROM entity_risk_signals
        WHERE run_id = ?
        ORDER BY signal_strength DESC
        LIMIT ?
        """,
        (run_id, limit),
    ).fetchall()

    return {
        "ticker": ticker,
        "run_id": run_id,
        "documents": [
            {"source": d[0], "type": d[1], "content": d[2][:3000]} for d in documents
        ],
        "risk_signals": [
            {
                "entity": r[0],
                "region": r[1],
                "risk_type": r[2],
                "strength": r[3],
                "rationale": r[4],
            }
            for r in risk_signals
        ],
    }


def build_prompt_payload(ticker: str, context: dict) -> dict:
    user_prompt = {
        "task": "Generate a detailed investment analyst report for the requested ticker.",
        "constraints": [
            "Ground conclusions in supplied data only.",
            "Call out data gaps explicitly.",
            "Include base, bull, and bear case timelines.",
            "Provide confidence score (0-100).",
        ],
        "target_ticker": ticker,
        "context": context,
    }
    return {"system": SYSTEM_PROMPT, "user": json.dumps(user_prompt, ensure_ascii=False)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LLM prompt payload for analyst report")
    parser.add_argument("--db-path", default="./system/market_intel.db")
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    try:
        context = fetch_context(conn, args.ticker)
        payload = build_prompt_payload(args.ticker, context)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
