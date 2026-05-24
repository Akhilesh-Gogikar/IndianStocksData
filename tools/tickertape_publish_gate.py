#!/usr/bin/env python3
"""Gate TickerTape server publishes on local sync quality."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REQUIRED_TABLES = (
    "sync_runs",
    "companies",
    "latest_stock_data",
    "stock_snapshots",
    "financial_sections",
    "event_sections",
    "sync_results",
)
PUBLISHABLE_RUN_STATUSES = {"completed", "completed_with_failures"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_today() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


@dataclass(frozen=True)
class GateOptions:
    db: Path
    company_list: Path
    snapshot_date: str
    min_success_rate: float = 0.98
    include_all_types: bool = False
    allow_stale_date: bool = False
    failure_limit: int = 500


def load_expected_companies(path: Path, include_all_types: bool = False) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected company list array in {path}")
    subdirectories: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        subdirectory = item.get("subdirectory")
        if not subdirectory:
            continue
        if not include_all_types and item.get("type") != "stocks":
            continue
        subdirectories.append(str(subdirectory))
    return subdirectories


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def table_count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def latest_run(conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM sync_runs ORDER BY id DESC LIMIT 1").fetchone()
    return dict(row) if row else None


def chunks(values: list[str], size: int = 800) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def count_expected_rows(
    conn: sqlite3.Connection,
    table: str,
    expected: list[str],
    *,
    where_sql: str = "",
    params: tuple[Any, ...] = (),
) -> int:
    if not expected:
        return 0
    total = 0
    for batch in chunks(expected):
        placeholders = ",".join("?" for _ in batch)
        sql = f"SELECT COUNT(DISTINCT subdirectory) FROM {table} WHERE subdirectory IN ({placeholders})"
        if where_sql:
            sql += f" AND {where_sql}"
        total += int(conn.execute(sql, (*batch, *params)).fetchone()[0])
    return total


def result_counts(conn: sqlite3.Connection, run_id: int) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT status, COUNT(*) AS count
        FROM sync_results
        WHERE run_id = ?
        GROUP BY status
        """,
        (run_id,),
    ).fetchall()
    return {str(row["status"]): int(row["count"]) for row in rows}


def failed_results(conn: sqlite3.Connection, run_id: int, limit: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT subdirectory, status, fetched_at, http_status, error, raw_json_path
        FROM sync_results
        WHERE run_id = ? AND status = 'failed'
        ORDER BY subdirectory
        LIMIT ?
        """,
        (run_id, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, details: dict[str, Any] | str) -> None:
    checks.append({"name": name, "passed": bool(passed), "details": details})


def evaluate_gate(options: GateOptions) -> dict[str, Any]:
    generated_at = utc_now()
    checks: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "gate_passed": False,
        "generated_at": generated_at,
        "db": str(options.db),
        "company_list": str(options.company_list),
        "snapshot_date": options.snapshot_date,
        "min_success_rate": options.min_success_rate,
        "checks": checks,
        "metrics": {},
        "latest_run": None,
        "failure_report": {"failures": [], "failure_count": 0, "failure_limit": options.failure_limit},
    }

    add_check(checks, "database_exists", options.db.exists(), {"path": str(options.db)})
    add_check(checks, "company_list_exists", options.company_list.exists(), {"path": str(options.company_list)})
    if not options.db.exists() or not options.company_list.exists():
        return manifest

    expected = load_expected_companies(options.company_list, options.include_all_types)
    expected_count = len(expected)
    manifest["metrics"]["expected_company_count"] = expected_count
    add_check(checks, "expected_company_universe_nonempty", expected_count > 0, {"expected_company_count": expected_count})
    if expected_count == 0:
        return manifest

    conn = sqlite3.connect(str(options.db))
    conn.row_factory = sqlite3.Row
    try:
        missing_tables = [table for table in REQUIRED_TABLES if not table_exists(conn, table)]
        add_check(checks, "required_tables_present", not missing_tables, {"missing_tables": missing_tables})
        if missing_tables:
            return manifest

        counts = {table: table_count(conn, table) for table in REQUIRED_TABLES}
        manifest["metrics"]["table_counts"] = counts
        run = latest_run(conn)
        manifest["latest_run"] = run
        add_check(checks, "latest_run_present", run is not None, {})
        if run is None:
            return manifest

        run_id = int(run["id"])
        run_status = str(run.get("status") or "").lower()
        run_snapshot_date = str(run.get("snapshot_date") or "")
        finished_at = run.get("finished_at")
        add_check(checks, "latest_run_finished", bool(finished_at), {"finished_at": finished_at})
        add_check(
            checks,
            "latest_run_status_publishable",
            run_status in PUBLISHABLE_RUN_STATUSES,
            {"status": run_status, "publishable_statuses": sorted(PUBLISHABLE_RUN_STATUSES)},
        )
        add_check(
            checks,
            "snapshot_date_current",
            options.allow_stale_date or run_snapshot_date == options.snapshot_date,
            {"run_snapshot_date": run_snapshot_date, "required_snapshot_date": options.snapshot_date},
        )

        counts_by_status = result_counts(conn, run_id)
        succeeded = counts_by_status.get("succeeded", 0)
        skipped = counts_by_status.get("skipped", 0)
        failed = counts_by_status.get("failed", 0)
        covered_by_run = succeeded + skipped
        attempted = sum(counts_by_status.values())
        companies_present = count_expected_rows(conn, "companies", expected)
        latest_data_count = count_expected_rows(
            conn,
            "latest_stock_data",
            expected,
            where_sql="snapshot_date = ?",
            params=(run_snapshot_date,),
        )
        snapshot_ok_count = count_expected_rows(
            conn,
            "stock_snapshots",
            expected,
            where_sql="snapshot_date = ? AND ok = 1",
            params=(run_snapshot_date,),
        )
        coverage_count = min(covered_by_run, latest_data_count, snapshot_ok_count)
        coverage_rate = coverage_count / expected_count
        max_failures = int(expected_count * (1 - options.min_success_rate))
        manifest["metrics"].update(
            {
                "result_counts": counts_by_status,
                "attempted_result_count": attempted,
                "covered_by_run": covered_by_run,
                "companies_present": companies_present,
                "latest_data_count": latest_data_count,
                "snapshot_ok_count": snapshot_ok_count,
                "coverage_count": coverage_count,
                "coverage_rate": coverage_rate,
                "failure_count": failed,
                "max_failures_at_threshold": max_failures,
            }
        )
        add_check(
            checks,
            "run_attempted_expected_universe",
            attempted >= expected_count,
            {"attempted": attempted, "expected_company_count": expected_count},
        )
        add_check(
            checks,
            "run_total_matches_expected_universe",
            int(run.get("total_companies") or 0) == expected_count,
            {"run_total_companies": int(run.get("total_companies") or 0), "expected_company_count": expected_count},
        )
        add_check(
            checks,
            "company_universe_present",
            companies_present == expected_count,
            {"companies_present": companies_present, "expected_company_count": expected_count},
        )
        add_check(
            checks,
            "coverage_rate_passes_threshold",
            coverage_rate >= options.min_success_rate,
            {"coverage_rate": coverage_rate, "min_success_rate": options.min_success_rate},
        )
        add_check(
            checks,
            "failure_count_within_threshold",
            failed <= max_failures,
            {"failure_count": failed, "max_failures": max_failures},
        )
        add_check(
            checks,
            "canonical_source_sections_present",
            counts["financial_sections"] > 0 and counts["event_sections"] > 0,
            {"financial_sections": counts["financial_sections"], "event_sections": counts["event_sections"]},
        )

        failures = failed_results(conn, run_id, options.failure_limit)
        manifest["failure_report"] = {
            "run_id": run_id,
            "snapshot_date": run_snapshot_date,
            "failure_count": failed,
            "failure_limit": options.failure_limit,
            "failures": failures,
        }
        manifest["gate_passed"] = all(bool(check["passed"]) for check in checks)
        return manifest
    finally:
        conn.close()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate TickerTape server publish on local sync quality.")
    parser.add_argument("--db", type=Path, default=Path("local_repository/tickertape.sqlite"))
    parser.add_argument("--company-list", type=Path, default=Path("full-company-list.json"))
    parser.add_argument("--snapshot-date", default=utc_today())
    parser.add_argument("--min-success-rate", type=float, default=0.98)
    parser.add_argument("--include-all-types", action="store_true")
    parser.add_argument("--allow-stale-date", action="store_true")
    parser.add_argument("--failure-limit", type=int, default=500)
    parser.add_argument("--manifest", type=Path, default=Path("local_repository/logs/tickertape_publish_manifest_latest.json"))
    parser.add_argument("--failure-report", type=Path, default=Path("local_repository/logs/tickertape_publish_failures_latest.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        manifest = evaluate_gate(
            GateOptions(
                db=args.db,
                company_list=args.company_list,
                snapshot_date=args.snapshot_date,
                min_success_rate=args.min_success_rate,
                include_all_types=args.include_all_types,
                allow_stale_date=args.allow_stale_date,
                failure_limit=args.failure_limit,
            )
        )
        write_json(args.manifest, manifest)
        write_json(args.failure_report, dict(manifest.get("failure_report") or {}))
        print(json.dumps(manifest, sort_keys=True))
        return 0 if manifest["gate_passed"] else 2
    except Exception as exc:  # noqa: BLE001 - CLI should emit machine-readable failure.
        error_manifest = {
            "gate_passed": False,
            "generated_at": utc_now(),
            "error": str(exc),
            "db": str(args.db),
            "company_list": str(args.company_list),
        }
        write_json(args.manifest, error_manifest)
        write_json(args.failure_report, {"error": str(exc), "failures": []})
        print(json.dumps(error_manifest, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
