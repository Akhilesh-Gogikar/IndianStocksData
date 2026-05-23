from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from system.api.factory import create_app


class ScreenersApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.db_path = Path(self.tempdir.name) / "market.db"
        conn = sqlite3.connect(self.db_path)
        conn.executescript(Path("system/schema.sql").read_text())
        conn.execute(
            """
            INSERT INTO ingestion_runs (run_id, run_date, status, started_at, finished_at, notes)
            VALUES (1, '2026-05-22', 'completed', '2026-05-22T00:00:00Z', '2026-05-22T00:05:00Z', 'test')
            """
        )
        conn.executemany(
            """
            INSERT INTO companies (
                ticker, name, exchange, sector, industry, market_cap,
                local_ingestion_run_id, as_of, processed_at, quality_status, data_rights_status
            )
            VALUES (?, ?, 'NSE', ?, ?, ?, 1, '2026-05-22', '2026-05-22T00:05:00Z', 'pass', 'derived-ok')
            """,
            [
                ("RELIANCE", "Reliance Industries", "Energy", "Integrated Oil", 1000000),
                ("TCS", "Tata Consultancy Services", "Technology", "IT Services", 900000),
            ],
        )
        conn.executemany(
            """
            INSERT INTO quote_snapshots (
                ticker, price, currency, volume, local_ingestion_run_id,
                as_of, processed_at, quality_status, data_rights_status
            )
            VALUES (?, ?, 'INR', ?, 1, '2026-05-22', '2026-05-22T00:05:00Z', 'pass', 'derived-ok')
            """,
            [
                ("RELIANCE", 2800.0, 100000),
                ("TCS", 4000.0, 50000),
            ],
        )
        conn.executemany(
            """
            INSERT INTO financial_ratios (
                ticker, ratio_name, ratio_value, period, period_end,
                local_ingestion_run_id, as_of, processed_at, quality_status, data_rights_status
            )
            VALUES (?, 'pe', ?, 'latest', '2026-03-31', 1, '2026-05-22', '2026-05-22T00:05:00Z', 'pass', 'derived-ok')
            """,
            [("RELIANCE", 24.5), ("TCS", 35.0)],
        )
        conn.commit()
        conn.close()
        self.client = TestClient(create_app(self.db_path, profile_name="market-data"))

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_saved_screener_evaluates_and_persists_history(self) -> None:
        created = self.client.post(
            "/screeners",
            json={
                "owner_id": "customer-a",
                "name": "Energy value",
                "filters": {"sector": "Energy", "ratio_filters": {"pe": {"max": 30}}, "limit": 10},
            },
        )
        self.assertEqual(created.status_code, 200)
        screener_id = created.json()["data"]["screener_id"]

        evaluated = self.client.post(f"/screeners/{screener_id}/evaluate?owner_id=customer-a")
        self.assertEqual(evaluated.status_code, 200)
        body = evaluated.json()
        self.assertEqual(body["metadata"]["result_count"], 1)
        self.assertEqual(body["metadata"]["top_tickers"], ["RELIANCE"])
        self.assertEqual(body["data"][0]["company"]["ticker"], "RELIANCE")

        shown = self.client.get(f"/screeners/{screener_id}?owner_id=customer-a")
        self.assertEqual(shown.status_code, 200)
        self.assertEqual(shown.json()["metadata"]["history_count"], 1)
        self.assertEqual(shown.json()["data"]["history"][0]["top_tickers"], ["RELIANCE"])

    def test_capabilities_include_screeners(self) -> None:
        response = self.client.get("/capabilities")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["screeners_ready"])
        self.assertIn("POST /screeners/{screener_id}/evaluate", body["screener_routes"])

    def test_agent_screener_copilot_interprets_plain_english(self) -> None:
        client = TestClient(create_app(self.db_path, profile_name="agent-runtime"))

        response = client.post(
            "/agents/screener-copilot"
            "?prompt=energy%20companies%20with%20pe%20under%2030"
            "&owner_id=customer-a&save=true&name=Energy%20value"
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "screener_copilot")
        self.assertEqual(body["filters"]["sector"], "Energy")
        self.assertEqual(body["filters"]["ratio_filters"]["pe"]["max"], 30.0)
        self.assertEqual(body["result"]["metadata"]["result_count"], 1)
        self.assertEqual(body["result"]["data"][0]["company"]["ticker"], "RELIANCE")
        self.assertEqual(body["match_explanations"][0]["ticker"], "RELIANCE")
        self.assertIsNotNone(body["saved_screener"])

    def test_agent_screener_digest_returns_customer_actions(self) -> None:
        client = TestClient(
            create_app(
                self.db_path,
                profile_name="agent-runtime",
                vector_index_dir=self.root / "vector_indexes",
            )
        )
        created = client.post(
            "/screeners",
            json={
                "owner_id": "customer-a",
                "name": "Energy value",
                "filters": {"sector": "Energy", "ratio_filters": {"pe": {"max": 30}}, "limit": 10},
            },
        )
        screener_id = created.json()["data"]["screener_id"]

        response = client.post(
            f"/agents/screener-digest/{screener_id}?owner_id=customer-a&focus=customer%20newsletter&evidence_limit=1"
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "screener_digest")
        self.assertEqual(body["ticker_digests"][0]["ticker"], "RELIANCE")
        self.assertEqual(body["metadata"]["top_tickers"], ["RELIANCE"])
        self.assertTrue(body["customer_next_actions"])


if __name__ == "__main__":
    unittest.main()
