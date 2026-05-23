from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from system.api.factory import create_app


class MorningBriefApiTests(unittest.TestCase):
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
            [("RELIANCE", 2800.0, 100000), ("TCS", 4000.0, 50000)],
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
        conn.executemany(
            """
            INSERT INTO raw_documents (
                document_id, run_id, source_name, file_path, file_type, content,
                content_sha256, record_count, source_timestamp, ingested_at
            )
            VALUES (?, 1, 'tickertape', ?, 'json', ?, ?, 1, '2026-05-22T00:00:00Z', '2026-05-22T00:05:00Z')
            """,
            [
                (1, "/tmp/reliance.json", "Reliance telecom refinery margins improved.", "hash-reliance"),
                (2, "/tmp/tcs.json", "TCS technology services demand and margin commentary.", "hash-tcs"),
            ],
        )
        conn.commit()
        conn.close()
        self.client = TestClient(
            create_app(
                self.db_path,
                profile_name="agent-runtime",
                vector_index_dir=self.root / "vector_indexes",
            )
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_morning_brief_prioritizes_owner_actions(self) -> None:
        portfolio = self.client.post(
            "/portfolios",
            json={"owner_id": "customer-a", "name": "Family portfolio", "base_currency": "INR"},
        )
        portfolio_id = portfolio.json()["data"]["portfolio_id"]
        self.client.post(
            f"/portfolios/{portfolio_id}/holdings",
            json={"owner_id": "customer-a", "ticker": "RELIANCE", "quantity": 10, "average_cost": 2500},
        )
        watchlist = self.client.post(
            "/watchlists",
            json={"owner_id": "customer-a", "name": "Core watchlist"},
        )
        watchlist_id = watchlist.json()["data"]["watchlist_id"]
        self.client.post(
            f"/watchlists/{watchlist_id}/items",
            json={"owner_id": "customer-a", "ticker": "RELIANCE"},
        )
        self.client.post(
            "/screeners",
            json={
                "owner_id": "customer-a",
                "name": "Energy value",
                "filters": {"sector": "Energy", "ratio_filters": {"pe": {"max": 30}}, "limit": 10},
            },
        )

        response = self.client.post("/agents/morning-brief?owner_id=customer-a&focus=telecom&evidence_limit=1")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "owner_morning_brief")
        self.assertEqual(body["owner_id"], "customer-a")
        self.assertTrue(body["priorities"])
        self.assertEqual(len(body["portfolio_digests"]), 1)
        self.assertEqual(len(body["watchlist_digests"]), 1)
        self.assertEqual(len(body["screener_digests"]), 1)
        self.assertIn("brief_markdown", body)


if __name__ == "__main__":
    unittest.main()
