#!/usr/bin/env python3
"""Print a compact status summary for the local TickerTape database."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict


def table_count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="Show local TickerTape database status.")
    parser.add_argument("--db", type=Path, default=Path("local_repository/tickertape.sqlite"))
    args = parser.parse_args()

    if not args.db.exists():
        print(json.dumps({"exists": False, "db": str(args.db)}, indent=2))
        return 1

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    try:
        latest = conn.execute("SELECT * FROM sync_runs ORDER BY id DESC LIMIT 1").fetchone()
        counts: Dict[str, Any] = {}
        for table in (
            "sync_runs",
            "companies",
            "latest_stock_data",
            "stock_snapshots",
            "financial_sections",
            "event_sections",
            "sync_results",
        ):
            counts[table] = table_count(conn, table)
        result_counts = [
            dict(row)
            for row in conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM sync_results
                WHERE run_id = (SELECT MAX(id) FROM sync_runs)
                GROUP BY status
                ORDER BY status
                """
            ).fetchall()
        ]
        print(
            json.dumps(
                {
                    "exists": True,
                    "db": str(args.db),
                    "db_bytes": args.db.stat().st_size,
                    "latest_run": dict(latest) if latest else None,
                    "latest_run_result_counts": result_counts,
                    "table_counts": counts,
                },
                indent=2,
            )
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
