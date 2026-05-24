from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from system.api.factory import create_app
from system.canonical_tickertape import import_tickertape_canonical


class CanonicalTickertapeImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.source_db = Path(self.tempdir.name) / "tickertape.sqlite"
        self.target_db = Path(self.tempdir.name) / "market.db"
        self._build_source_db()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _build_source_db(self) -> None:
        conn = sqlite3.connect(self.source_db)
        conn.executescript(
            """
            CREATE TABLE sync_runs (
                id INTEGER PRIMARY KEY,
                snapshot_date TEXT,
                status TEXT,
                finished_at TEXT
            );
            CREATE TABLE companies (
                subdirectory TEXT PRIMARY KEY,
                name TEXT,
                sid TEXT,
                isin TEXT,
                slug TEXT
            );
            CREATE TABLE latest_stock_data (
                subdirectory TEXT PRIMARY KEY,
                name TEXT,
                type TEXT,
                snapshot_date TEXT,
                security_info_json TEXT,
                security_quote_json TEXT,
                scorecard_json TEXT,
                security_summary_json TEXT,
                updated_at TEXT
            );
            CREATE TABLE event_sections (
                subdirectory TEXT,
                snapshot_date TEXT,
                section_key TEXT,
                section_json TEXT,
                updated_at TEXT,
                PRIMARY KEY (subdirectory, snapshot_date, section_key)
            );
            """
        )
        conn.execute("INSERT INTO sync_runs VALUES (7, '2026-05-22', 'completed', '2026-05-22T00:00:00Z')")
        conn.execute(
            "INSERT INTO companies VALUES (?, ?, ?, ?, ?)",
            ("reliance-industries-RELI", "Reliance Industries", "RELI", "INE002A01018", "/stocks/reliance-industries-RELI"),
        )
        conn.execute(
            "INSERT INTO latest_stock_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "reliance-industries-RELI",
                "Reliance Industries Ltd",
                "stocks",
                "2026-05-22",
                json.dumps(
                    {
                        "sid": "RELI",
                        "isin": "INE002A01018",
                        "type": "stock",
                        "tradable": True,
                        "info": {
                            "ticker": "RELIANCE",
                            "name": "Reliance Industries Ltd",
                            "exchange": "NSE",
                            "description": "Integrated energy and consumer company.",
                        },
                        "gic": {"sector": "Energy", "industry": "Oil Gas & Consumable Fuels"},
                        "ratios": {"pe": 25.5, "marketCap": 1900000, "marketCapLabel": "Largecap"},
                    }
                ),
                json.dumps({"price": 2800.5, "o": 2790, "h": 2815, "l": 2775, "vol": 12345, "exchange": "NSE"}),
                json.dumps([{"name": "Performance", "tag": "High", "colour": "green"}]),
                json.dumps({"aboutAndPeers": [{"ticker": "ONGC", "name": "Oil and Natural Gas Corp", "sector": "Energy"}]}),
                "2026-05-22T00:00:00Z",
            ),
        )
        conn.execute(
            "INSERT INTO event_sections VALUES (?, ?, ?, ?, ?)",
            (
                "reliance-industries-RELI",
                "2026-05-22",
                "events-announcements",
                json.dumps({"past": [{"subject": "Board meeting", "broadcastTime": "2026-05-01T10:00:00Z"}]}),
                "2026-05-22T00:00:00Z",
            ),
        )
        conn.commit()
        conn.close()

    def test_import_populates_canonical_api_tables(self) -> None:
        counts = import_tickertape_canonical(
            source_db=self.source_db,
            target_db=self.target_db,
            schema_path=Path("system/schema.sql"),
        )
        self.assertEqual(counts["companies"], 1)
        self.assertEqual(counts["quote_snapshots"], 1)
        self.assertEqual(counts["financial_ratios"], 2)
        self.assertEqual(counts["company_events"], 1)
        self.assertEqual(counts["company_peers"], 1)

        client = TestClient(create_app(self.target_db, profile_name="market-data"))
        company = client.get("/companies/RELIANCE").json()
        self.assertEqual(company["data"]["sector"], "Energy")
        self.assertEqual(company["metadata"]["data_rights_status"], "source-derived-review-required")

        screen = client.post("/screen", json={"sector": "Energy", "ratio_filters": {"pe": {"max": 30}}}).json()
        self.assertEqual(screen["metadata"]["result_count"], 1)
        self.assertEqual(screen["data"][0]["company"]["ticker"], "RELIANCE")

    def test_import_marks_partial_source_warning_without_failed_ticker_rows(self) -> None:
        conn = sqlite3.connect(self.source_db)
        conn.execute("UPDATE sync_runs SET status = 'completed_with_failures' WHERE id = 7")
        conn.execute(
            "INSERT INTO companies VALUES (?, ?, ?, ?, ?)",
            ("missing-bank-MISS", "Missing Bank", "MISS", "INE000A01000", "/stocks/missing-bank-MISS"),
        )
        conn.commit()
        conn.close()

        counts = import_tickertape_canonical(
            source_db=self.source_db,
            target_db=self.target_db,
            schema_path=Path("system/schema.sql"),
        )

        self.assertEqual(counts["quality_status"], "warning")
        self.assertEqual(counts["companies"], 1)

        target = sqlite3.connect(self.target_db)
        target.row_factory = sqlite3.Row
        try:
            missing = target.execute("SELECT * FROM companies WHERE ticker = 'MISS'").fetchone()
            audit = target.execute(
                "SELECT status, details FROM data_quality_audits WHERE check_name = 'canonical_tickertape_import'"
            ).fetchone()
        finally:
            target.close()

        self.assertIsNone(missing)
        self.assertEqual(audit["status"], "warning")


if __name__ == "__main__":
    unittest.main()
