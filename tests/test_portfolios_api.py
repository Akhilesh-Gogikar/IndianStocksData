from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from system.api.factory import create_app


class PortfoliosApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.db_path = self.root / "market.db"
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
        conn.execute(
            """
            INSERT INTO financial_ratios (
                ticker, ratio_name, ratio_value, period, period_end,
                local_ingestion_run_id, as_of, processed_at, quality_status, data_rights_status
            )
            VALUES ('RELIANCE', 'pe', 24.5, 'latest', '2026-03-31', 1, '2026-05-22', '2026-05-22T00:05:00Z', 'pass', 'derived-ok')
            """
        )
        conn.executemany(
            """
            INSERT INTO raw_documents (
                document_id, run_id, source_name, file_path, file_type, content,
                content_sha256, record_count, source_timestamp, ingested_at
            )
            VALUES (?, 1, 'tickertape', ?, 'json', ?, ?, 1, '2026-05-22T00:00:00Z', '2026-05-22T00:05:00Z')
            """,
            [
                (
                    1,
                    "/tmp/reliance.json",
                    "Reliance telecom refinery margins improved with stronger retail execution.",
                    "hash-reliance",
                ),
                (
                    2,
                    "/tmp/tcs.json",
                    "TCS demand commentary focused on deal wins, margins, and technology services growth.",
                    "hash-tcs",
                ),
            ],
        )
        conn.commit()
        conn.close()
        self.client = TestClient(create_app(self.db_path, profile_name="market-data"))

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_portfolio_flow_returns_xray(self) -> None:
        created = self.client.post(
            "/portfolios",
            json={"owner_id": "customer-a", "name": "Family portfolio", "base_currency": "INR"},
        )
        self.assertEqual(created.status_code, 200)
        portfolio_id = created.json()["data"]["portfolio_id"]

        reliance = self.client.post(
            f"/portfolios/{portfolio_id}/holdings",
            json={"owner_id": "customer-a", "ticker": "reliance", "quantity": 10, "average_cost": 2500},
        )
        self.assertEqual(reliance.status_code, 200)

        tcs = self.client.post(
            f"/portfolios/{portfolio_id}/holdings",
            json={"owner_id": "customer-a", "ticker": "TCS", "quantity": 2, "average_cost": 3500},
        )
        self.assertEqual(tcs.status_code, 200)

        body = tcs.json()
        self.assertEqual(body["metadata"]["holding_count"], 2)
        self.assertEqual(body["data"]["xray"]["total_market_value"], 36000.0)
        self.assertEqual(body["data"]["xray"]["total_cost_basis"], 32000.0)
        self.assertAlmostEqual(body["data"]["xray"]["total_unrealized_pl_pct"], 0.125)
        self.assertEqual(body["data"]["xray"]["top_concentration"][0]["ticker"], "RELIANCE")

        xray = self.client.get(f"/portfolios/{portfolio_id}/xray?owner_id=customer-a")
        self.assertEqual(xray.status_code, 200)
        self.assertEqual(xray.json()["data"]["positions_with_quotes"], 2)

    def test_capabilities_include_portfolios(self) -> None:
        response = self.client.get("/capabilities")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["portfolios_ready"])
        self.assertIn("GET /portfolios/{portfolio_id}/xray", body["portfolio_routes"])

    def test_agent_portfolio_digest_returns_customer_actions(self) -> None:
        client = TestClient(
            create_app(
                self.db_path,
                profile_name="agent-runtime",
                vector_index_dir=self.root / "vector_indexes",
            )
        )
        created = client.post(
            "/portfolios",
            json={"owner_id": "customer-a", "name": "Family portfolio", "base_currency": "INR"},
        )
        portfolio_id = created.json()["data"]["portfolio_id"]
        client.post(
            f"/portfolios/{portfolio_id}/holdings",
            json={"owner_id": "customer-a", "ticker": "RELIANCE", "quantity": 10, "average_cost": 2500},
        )

        response = client.post(
            f"/agents/portfolio-digest/{portfolio_id}?owner_id=customer-a&focus=telecom%20refinery&evidence_limit=1"
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "portfolio_digest")
        self.assertEqual(body["ticker_digests"][0]["ticker"], "RELIANCE")
        self.assertTrue(body["customer_next_actions"])


if __name__ == "__main__":
    unittest.main()
