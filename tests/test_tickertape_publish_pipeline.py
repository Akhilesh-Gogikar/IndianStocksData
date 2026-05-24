from __future__ import annotations

import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TickertapePublishPipelineSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.db = self.root / "tickertape.sqlite"
        self.company_list = self.root / "companies.json"
        self.logs = self.root / "logs"
        self.target_db = self.root / "market.db"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _build_blocked_db(self) -> None:
        companies = [{"subdirectory": f"company-{index:03d}-TEST", "type": "stocks"} for index in range(10)]
        self.company_list.write_text(json.dumps(companies), encoding="utf-8")
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
                10,
                10,
                8,
                2,
                0,
                "completed_with_failures",
                "{}",
                None,
            ),
        )
        for index, company in enumerate(companies):
            subdirectory = company["subdirectory"]
            is_ok = index < 8
            conn.execute("INSERT INTO companies VALUES (?)", (subdirectory,))
            conn.execute("INSERT INTO stock_snapshots VALUES (?, ?, ?)", (subdirectory, "2026-05-23", 1 if is_ok else 0))
            conn.execute(
                "INSERT INTO sync_results VALUES (?, ?, ?, ?, ?, ?, ?)",
                (1, subdirectory, "succeeded" if is_ok else "failed", "2026-05-23T01:00:00Z", None, None, None),
            )
            if is_ok:
                conn.execute("INSERT INTO latest_stock_data VALUES (?, ?)", (subdirectory, "2026-05-23"))
                conn.execute("INSERT INTO financial_sections VALUES (?, ?)", (subdirectory, "2026-05-23"))
                conn.execute("INSERT INTO event_sections VALUES (?, ?)", (subdirectory, "2026-05-23"))
        conn.commit()
        conn.close()

    def test_dry_run_prints_orchestration_without_running_sync(self) -> None:
        result = subprocess.run(
            [
                "bash",
                "scripts/run_tickertape_publish_pipeline.sh",
                "--mode",
                "daily",
                "--dry-run",
                "--logs-dir",
                str(self.logs),
                "--db",
                str(self.db),
                "--company-list",
                str(self.company_list),
                "--target-db",
                str(self.target_db),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("sync_pass=primary", result.stdout)
        self.assertIn("tools/tickertape_publish_gate.py", result.stdout)
        self.assertIn("load_tickertape_to_server.sh", result.stdout)

    def test_publish_only_blocks_before_canonicalize_when_gate_fails(self) -> None:
        self._build_blocked_db()

        result = subprocess.run(
            [
                "bash",
                "scripts/run_tickertape_publish_pipeline.sh",
                "--mode",
                "publish-only",
                "--skip-upload",
                "--logs-dir",
                str(self.logs),
                "--db",
                str(self.db),
                "--company-list",
                str(self.company_list),
                "--target-db",
                str(self.target_db),
                "--snapshot-date",
                "2026-05-23",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("publish_gate_blocked", result.stdout)
        self.assertFalse(self.target_db.exists())
        self.assertTrue((self.logs / "tickertape_publish_manifest_latest.json").exists())


if __name__ == "__main__":
    unittest.main()
