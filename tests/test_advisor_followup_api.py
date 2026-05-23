from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from system.api.factory import create_app


class AdvisorFollowupApiTests(unittest.TestCase):
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
        conn.execute(
            """
            INSERT INTO raw_documents (
                document_id, run_id, source_name, file_path, file_type, content,
                content_sha256, record_count, source_timestamp, ingested_at
            )
            VALUES (1, 1, 'tickertape', '/tmp/reliance.json', 'json',
                    'Reliance telecom refinery margins improved.', 'hash-reliance', 1,
                    '2026-05-22T00:00:00Z', '2026-05-22T00:05:00Z')
            """
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

    def test_advisor_followup_returns_customer_ready_pack(self) -> None:
        portfolio = self.client.post(
            "/portfolios",
            json={"owner_id": "customer-a", "name": "Family portfolio", "base_currency": "INR"},
        )
        portfolio_id = portfolio.json()["data"]["portfolio_id"]
        self.client.post(
            f"/portfolios/{portfolio_id}/holdings",
            json={"owner_id": "customer-a", "ticker": "RELIANCE", "quantity": 10, "average_cost": 2500},
        )

        response = self.client.post("/agents/advisor-followup?owner_id=customer-a&focus=telecom&evidence_limit=1")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "advisor_followup_pack")
        self.assertEqual(body["owner_id"], "customer-a")
        self.assertIn("subject", body["customer_email"])
        self.assertTrue(body["meeting_agenda"])
        self.assertTrue(body["advisor_checklist"])
        self.assertTrue(body["compliance_guardrails"]["do_not_say"])
        self.assertIn("followup_markdown", body)


if __name__ == "__main__":
    unittest.main()
