from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from system.api.factory import create_app


class MarketApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "market.db"
        schema = Path("system/schema.sql").read_text()
        conn = sqlite3.connect(self.db_path)
        conn.executescript(schema)
        conn.execute(
            """
            INSERT INTO ingestion_runs (run_id, run_date, status, started_at, finished_at, notes)
            VALUES (1, '2026-05-22', 'completed', '2026-05-22T00:00:00Z', '2026-05-22T00:05:00Z', 'test run')
            """
        )
        conn.execute(
            """
            INSERT INTO data_quality_audits (run_id, check_name, status, details, checked_at)
            VALUES (1, 'source_coverage', 'pass', 'ok', '2026-05-22T00:05:00Z')
            """
        )
        conn.execute(
            """
            INSERT INTO companies (
                ticker, name, exchange, sector, industry, market_cap,
                local_ingestion_run_id, as_of, processed_at, quality_status, data_rights_status
            )
            VALUES
                ('RELIANCE', 'Reliance Industries', 'NSE', 'Energy', 'Integrated Oil', 1000000,
                 1, '2026-05-22', '2026-05-22T00:05:00Z', 'pass', 'derived-ok'),
                ('TCS', 'Tata Consultancy Services', 'NSE', 'Technology', 'IT Services', 900000,
                 1, '2026-05-22', '2026-05-22T00:05:00Z', 'pass', 'derived-ok')
            """
        )
        conn.execute(
            """
            INSERT INTO quote_snapshots (
                ticker, price, currency, volume, local_ingestion_run_id,
                as_of, processed_at, quality_status, data_rights_status
            )
            VALUES ('RELIANCE', 2800.50, 'INR', 123456, 1, '2026-05-22', '2026-05-22T00:05:00Z', 'pass', 'derived-ok')
            """
        )
        conn.execute(
            """
            INSERT INTO financial_ratios (
                ticker, ratio_name, ratio_value, period, period_end,
                local_ingestion_run_id, as_of, processed_at, quality_status, data_rights_status
            )
            VALUES ('RELIANCE', 'pe_ratio', 24.5, 'ttm', '2026-03-31', 1, '2026-05-22', '2026-05-22T00:05:00Z', 'pass', 'derived-ok')
            """
        )
        conn.execute(
            """
            INSERT INTO company_events (
                ticker, event_type, event_date, title, local_ingestion_run_id,
                as_of, processed_at, quality_status, data_rights_status
            )
            VALUES ('RELIANCE', 'earnings', '2026-05-01', 'Quarterly results', 1, '2026-05-22', '2026-05-22T00:05:00Z', 'pass', 'derived-ok')
            """
        )
        conn.execute(
            """
            INSERT INTO company_peers (
                ticker, peer_ticker, relationship, score, local_ingestion_run_id,
                as_of, processed_at, quality_status, data_rights_status
            )
            VALUES ('RELIANCE', 'TCS', 'index_peer', 0.4, 1, '2026-05-22', '2026-05-22T00:05:00Z', 'pass', 'derived-ok')
            """
        )
        conn.commit()
        conn.close()
        self.client = TestClient(create_app(self.db_path, profile_name="market-data"))

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_market_profile_exposes_canonical_routes(self) -> None:
        response = self.client.get("/capabilities")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["market_data_ready"])
        self.assertIn("GET /companies/{ticker}", body["canonical_market_routes"])

    def test_company_quote_ratios_events_and_peers(self) -> None:
        company = self.client.get("/companies/reliance").json()
        self.assertEqual(company["data"]["ticker"], "RELIANCE")
        self.assertEqual(company["metadata"]["quality_status"], "pass")

        quote = self.client.get("/quotes/RELIANCE").json()
        self.assertEqual(quote["data"]["price"], 2800.50)
        self.assertEqual(quote["metadata"]["data_rights_status"], "derived-ok")

        ratios = self.client.get("/ratios/RELIANCE?period=ttm").json()
        self.assertEqual(ratios["data"]["ratios"][0]["ratio_name"], "pe_ratio")

        events = self.client.get("/events/RELIANCE").json()
        self.assertEqual(events["data"]["events"][0]["title"], "Quarterly results")

        peers = self.client.get("/peers/RELIANCE").json()
        self.assertEqual(peers["data"]["peers"][0]["peer_ticker"], "TCS")
        self.assertEqual(peers["data"]["peers"][0]["peer_name"], "Tata Consultancy Services")

    def test_screen_combines_company_quote_and_ratio_filters(self) -> None:
        response = self.client.post(
            "/screen",
            json={
                "sector": "Energy",
                "min_price": 1000,
                "ratio_filters": {"pe_ratio": {"max": 30}},
                "limit": 10,
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["metadata"]["result_count"], 1)
        self.assertEqual(body["data"][0]["company"]["ticker"], "RELIANCE")


if __name__ == "__main__":
    unittest.main()
