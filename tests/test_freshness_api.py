from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from system.api.factory import create_app


class FreshnessApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "market.db"
        conn = sqlite3.connect(self.db_path)
        conn.executescript(Path("system/schema.sql").read_text())
        conn.execute(
            """
            INSERT INTO ingestion_runs (run_id, run_date, status, started_at, finished_at, notes)
            VALUES (1, '2026-05-22', 'completed', '2026-05-22T00:00:00Z', '2026-05-22T00:05:00Z', 'test')
            """
        )
        conn.execute(
            """
            INSERT INTO data_quality_audits (run_id, check_name, status, details, checked_at)
            VALUES (1, 'canonical_import', 'pass', 'ok', '2026-05-22T00:05:00Z')
            """
        )
        conn.execute(
            """
            INSERT INTO companies (
                ticker, name, exchange, sector, industry, market_cap,
                local_ingestion_run_id, as_of, processed_at, quality_status, data_rights_status
            )
            VALUES ('RELIANCE', 'Reliance Industries', 'NSE', 'Energy', 'Integrated Oil', 1000000,
                    1, '2026-05-22', '2026-05-22T00:05:00Z', 'pass', 'derived-ok')
            """
        )
        conn.execute(
            """
            INSERT INTO quote_snapshots (
                ticker, price, currency, volume, local_ingestion_run_id,
                as_of, processed_at, quality_status, data_rights_status
            )
            VALUES ('RELIANCE', 2800.0, 'INR', 100000, 1, '2026-05-22', '2026-05-22T00:05:00Z', 'pass', 'derived-ok')
            """
        )
        conn.commit()
        conn.close()
        self.client = TestClient(create_app(self.db_path, profile_name="market-data"))

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_freshness_index_reports_quality_and_table_coverage(self) -> None:
        response = self.client.get("/freshness?warn_days=30&stale_days=90")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["data"]["overall_status"], "fresh")
        self.assertEqual(body["data"]["quality"]["status"], "pass")
        tables = {item["table"]: item for item in body["data"]["tables"]}
        self.assertEqual(tables["companies"]["record_count"], 1)
        self.assertEqual(tables["quote_snapshots"]["record_count"], 1)

    def test_ticker_freshness_reports_missing_and_present_sources(self) -> None:
        response = self.client.get("/freshness/RELIANCE?warn_days=30&stale_days=90")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["data"]["ticker"], "RELIANCE")
        self.assertEqual(body["data"]["overall_status"], "fresh")
        sources = {item["kind"]: item for item in body["data"]["sources"]}
        self.assertEqual(sources["company"]["record_count"], 1)
        self.assertEqual(sources["quote"]["record_count"], 1)

        missing = self.client.get("/freshness/UNKNOWN")
        self.assertEqual(missing.status_code, 200)
        self.assertEqual(missing.json()["data"]["overall_status"], "missing")

    def test_capabilities_include_freshness(self) -> None:
        response = self.client.get("/capabilities")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["freshness_ready"])
        self.assertIn("GET /freshness/{ticker}", body["freshness_routes"])


if __name__ == "__main__":
    unittest.main()
