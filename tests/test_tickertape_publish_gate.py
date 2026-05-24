from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from tools.tickertape_publish_gate import GateOptions, evaluate_gate


class TickertapePublishGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.db = self.root / "tickertape.sqlite"
        self.company_list = self.root / "companies.json"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_company_list(self, count: int) -> list[str]:
        subdirectories = [f"company-{index:03d}-TEST" for index in range(count)]
        payload = [{"subdirectory": item, "type": "stocks"} for item in subdirectories]
        self.company_list.write_text(json.dumps(payload), encoding="utf-8")
        return subdirectories

    def _build_db(self, subdirectories: list[str], ok_count: int, status: str = "completed_with_failures") -> None:
        conn = sqlite3.connect(self.db)
        conn.executescript(
            """
            CREATE TABLE sync_runs (
                id INTEGER PRIMARY KEY,
                started_at TEXT,
                finished_at TEXT,
                snapshot_date TEXT,
                company_list_path TEXT,
                db_path TEXT,
                raw_dir TEXT,
                total_companies INTEGER,
                attempted INTEGER,
                succeeded INTEGER,
                failed INTEGER,
                skipped INTEGER,
                status TEXT,
                options_json TEXT,
                error TEXT
            );
            CREATE TABLE companies (subdirectory TEXT PRIMARY KEY);
            CREATE TABLE latest_stock_data (subdirectory TEXT PRIMARY KEY, snapshot_date TEXT);
            CREATE TABLE stock_snapshots (subdirectory TEXT, snapshot_date TEXT, ok INTEGER);
            CREATE TABLE financial_sections (subdirectory TEXT, snapshot_date TEXT);
            CREATE TABLE event_sections (subdirectory TEXT, snapshot_date TEXT);
            CREATE TABLE sync_results (
                run_id INTEGER,
                subdirectory TEXT,
                status TEXT,
                fetched_at TEXT,
                http_status INTEGER,
                error TEXT,
                raw_json_path TEXT
            );
            """
        )
        failed = len(subdirectories) - ok_count
        conn.execute(
            "INSERT INTO sync_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                1,
                "2026-05-23T00:00:00Z",
                "2026-05-23T01:00:00Z",
                "2026-05-23",
                str(self.company_list),
                str(self.db),
                "raw",
                len(subdirectories),
                len(subdirectories),
                ok_count,
                failed,
                0,
                status,
                "{}",
                None,
            ),
        )
        for index, subdirectory in enumerate(subdirectories):
            is_ok = index < ok_count
            conn.execute("INSERT INTO companies VALUES (?)", (subdirectory,))
            conn.execute(
                "INSERT INTO stock_snapshots VALUES (?, ?, ?)",
                (subdirectory, "2026-05-23", 1 if is_ok else 0),
            )
            conn.execute(
                "INSERT INTO sync_results VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    1,
                    subdirectory,
                    "succeeded" if is_ok else "failed",
                    "2026-05-23T01:00:00Z",
                    200 if is_ok else 500,
                    None if is_ok else "source failed",
                    f"raw/{subdirectory}.json.gz" if is_ok else None,
                ),
            )
            if is_ok:
                conn.execute("INSERT INTO latest_stock_data VALUES (?, ?)", (subdirectory, "2026-05-23"))
                conn.execute("INSERT INTO financial_sections VALUES (?, ?)", (subdirectory, "2026-05-23"))
                conn.execute("INSERT INTO event_sections VALUES (?, ?)", (subdirectory, "2026-05-23"))
        conn.commit()
        conn.close()

    def test_gate_passes_when_coverage_meets_threshold(self) -> None:
        subdirectories = self._write_company_list(100)
        self._build_db(subdirectories, ok_count=98)

        manifest = evaluate_gate(
            GateOptions(db=self.db, company_list=self.company_list, snapshot_date="2026-05-23", min_success_rate=0.98)
        )

        self.assertTrue(manifest["gate_passed"])
        self.assertEqual(manifest["metrics"]["coverage_count"], 98)
        self.assertEqual(manifest["failure_report"]["failure_count"], 2)

    def test_gate_blocks_when_coverage_is_too_low(self) -> None:
        subdirectories = self._write_company_list(100)
        self._build_db(subdirectories, ok_count=90)

        manifest = evaluate_gate(
            GateOptions(db=self.db, company_list=self.company_list, snapshot_date="2026-05-23", min_success_rate=0.98)
        )

        self.assertFalse(manifest["gate_passed"])
        self.assertEqual(manifest["metrics"]["coverage_count"], 90)
        failed_checks = {check["name"] for check in manifest["checks"] if not check["passed"]}
        self.assertIn("coverage_rate_passes_threshold", failed_checks)

    def test_gate_blocks_running_latest_run(self) -> None:
        subdirectories = self._write_company_list(10)
        self._build_db(subdirectories, ok_count=10, status="running")
        conn = sqlite3.connect(self.db)
        conn.execute("UPDATE sync_runs SET finished_at = NULL WHERE id = 1")
        conn.commit()
        conn.close()

        manifest = evaluate_gate(
            GateOptions(db=self.db, company_list=self.company_list, snapshot_date="2026-05-23", min_success_rate=0.98)
        )

        self.assertFalse(manifest["gate_passed"])
        failed_checks = {check["name"] for check in manifest["checks"] if not check["passed"]}
        self.assertIn("latest_run_finished", failed_checks)
        self.assertIn("latest_run_status_publishable", failed_checks)


if __name__ == "__main__":
    unittest.main()
