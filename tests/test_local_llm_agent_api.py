from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from system.api.factory import create_app


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class LocalLlmAgentApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "market.db"
        self.client = TestClient(
            create_app(
                self.db_path,
                profile_name="agent-runtime",
                vector_index_dir=Path(self.tempdir.name) / "vector_indexes",
            )
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def seed_market_data(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.executescript(Path("system/schema.sql").read_text())
        conn.execute(
            """
            INSERT INTO ingestion_runs (run_id, run_date, status, started_at, finished_at, notes)
            VALUES (1, '2026-05-23', 'completed', '2026-05-23T00:00:00Z', '2026-05-23T00:05:00Z', 'test')
            """
        )
        conn.execute(
            """
            INSERT INTO companies (
                ticker, name, exchange, sector, industry, market_cap,
                local_ingestion_run_id, as_of, processed_at, quality_status, data_rights_status
            )
            VALUES ('RELIANCE', 'Reliance Industries', 'NSE', 'Energy', 'Integrated Oil', 1000000,
                    1, '2026-05-23', '2026-05-23T00:05:00Z', 'pass', 'derived-ok')
            """
        )
        conn.execute(
            """
            INSERT INTO quote_snapshots (
                ticker, price, currency, volume, local_ingestion_run_id,
                as_of, processed_at, quality_status, data_rights_status
            )
            VALUES ('RELIANCE', 2800.0, 'INR', 100000, 1,
                    '2026-05-23', '2026-05-23T00:05:00Z', 'pass', 'derived-ok')
            """
        )
        conn.execute(
            """
            INSERT INTO financial_ratios (
                ticker, ratio_name, ratio_value, period, period_end,
                local_ingestion_run_id, as_of, processed_at, quality_status, data_rights_status
            )
            VALUES ('RELIANCE', 'PE', 24.5, 'TTM', '2026-03-31',
                    1, '2026-05-23', '2026-05-23T00:05:00Z', 'pass', 'derived-ok')
            """
        )
        conn.execute(
            """
            INSERT INTO company_events (
                ticker, event_type, event_date, title, description, source_url,
                local_ingestion_run_id, as_of, processed_at, quality_status, data_rights_status
            )
            VALUES ('RELIANCE', 'result', '2026-05-20', 'Quarterly result snapshot',
                    'Processed result event', 'https://example.test/reliance',
                    1, '2026-05-23', '2026-05-23T00:05:00Z', 'pass', 'derived-ok')
            """
        )
        conn.commit()
        conn.close()

    def test_local_llm_chat_success_uses_llama_cpp_openai_contract(self) -> None:
        def fake_urlopen(request, timeout):
            body = json.loads(request.data.decode("utf-8"))
            self.assertEqual(request.full_url, "http://127.0.0.1:8080/v1/chat/completions")
            self.assertEqual(body["model"], "local-llama")
            self.assertIn("processed", body["messages"][0]["content"])
            return _FakeResponse({"choices": [{"message": {"content": "Local product answer"}}]})

        with patch.dict(os.environ, {}, clear=False), patch(
            "system.api.routers.agents.urllib.request.urlopen", fake_urlopen
        ):
            response = self.client.post(
                "/agents/local-llm-chat",
                json={"message": "Explain the screener page", "product_id": "stock-screener"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["provider"], "local-llama")
        self.assertEqual(payload["answer"], "Local product answer")

    def test_local_llm_chat_uses_market_data_when_llama_unavailable(self) -> None:
        self.seed_market_data()

        with patch(
            "system.api.routers.agents.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            response = self.client.post(
                "/agents/local-llm-chat",
                json={"message": "tell me about reliance?"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["provider"], "local-market-data")
        self.assertEqual(payload["model"], "sqlite-market-rag")
        self.assertEqual(payload["ticker"], "RELIANCE")
        self.assertIn("Reliance Industries", payload["answer"])
        self.assertGreaterEqual(payload["evidence_count"], 3)
        self.assertIn("companies", {source["source_table"] for source in payload["sources"]})

    def test_local_llm_chat_uses_tickertape_mirror_when_llama_unavailable(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.executescript(
            """
            CREATE TABLE latest_stock_data (
                subdirectory TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT,
                fetched_at TEXT,
                snapshot_date TEXT,
                url TEXT,
                http_status INTEGER,
                final_url TEXT,
                raw_json_path TEXT,
                raw_json_sha256 TEXT,
                page_props_keys_json TEXT,
                security_info_json TEXT,
                security_quote_json TEXT,
                scorecard_json TEXT,
                security_summary_json TEXT,
                labels_json TEXT,
                commentary_json TEXT,
                updated_at TEXT
            );
            CREATE TABLE financial_sections (
                subdirectory TEXT,
                snapshot_date TEXT,
                section_key TEXT,
                section_json TEXT,
                updated_at TEXT
            );
            CREATE TABLE event_sections (
                subdirectory TEXT,
                snapshot_date TEXT,
                section_key TEXT,
                section_json TEXT,
                updated_at TEXT
            );
            """
        )
        rows = [
            ("reliance-chemotex-industries-RELC", "Reliance Chemotex Industries Ltd", "RELC", 10.0),
            ("reliance-industries-RELI", "Reliance Industries Ltd", "RELIANCE", 18329823.7),
        ]
        for subdirectory, name, ticker, market_cap in rows:
            conn.execute(
                """
                INSERT INTO latest_stock_data (
                    subdirectory, name, type, snapshot_date, url, security_info_json,
                    security_quote_json, scorecard_json, updated_at
                )
                VALUES (?, ?, 'stocks', '2026-05-23', 'https://example.test', ?, ?, ?, '2026-05-23T00:00:00Z')
                """,
                (
                    subdirectory,
                    name,
                    json.dumps(
                        {
                            "gic": {"sector": "Energy", "industry": "Oil, Gas & Consumable Fuels"},
                            "info": {
                                "ticker": ticker,
                                "description": f"{name} description",
                                "ratios": {"marketCap": market_cap, "ttmPe": 22.7},
                            },
                        }
                    ),
                    json.dumps({"c": 1349.6, "dyChange": 0.36, "exchange": "NSE"}),
                    json.dumps([{"description": "Scorecard note"}]),
                ),
            )
        conn.execute(
            "INSERT INTO financial_sections VALUES ('reliance-industries-RELI', '2026-05-23', 'income', '[]', '2026-05-23T00:00:00Z')"
        )
        conn.execute(
            "INSERT INTO event_sections VALUES ('reliance-industries-RELI', '2026-05-23', 'events', '[]', '2026-05-23T00:00:00Z')"
        )
        conn.commit()
        conn.close()

        with patch(
            "system.api.routers.agents.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            response = self.client.post(
                "/agents/local-llm-chat",
                json={"message": "tell me about reliance?"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["provider"], "local-market-data")
        self.assertEqual(payload["ticker"], "RELIANCE")
        self.assertIn("Reliance Industries Ltd", payload["answer"])
        self.assertIn("latest_stock_data", payload["source_tables"])

    def test_local_llm_chat_backtesting_has_deterministic_fallback_when_llama_unavailable(self) -> None:
        self.seed_market_data()

        with patch(
            "system.api.routers.agents.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            response = self.client.post(
                "/agents/local-llm-chat",
                json={"message": "run a dry run for the current universe", "product_id": "backtesting"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["provider"], "local-market-data")
        self.assertEqual(payload["product_id"], "backtesting")
        self.assertIn("backtesting", payload)
        self.assertGreaterEqual(payload["backtesting"]["universe_size"], 1)
        self.assertGreaterEqual(payload["backtesting"]["match_count"], 1)
        self.assertTrue(payload["sources"])

    def test_local_llm_chat_malformed_llm_response_falls_back_for_backtesting(self) -> None:
        self.seed_market_data()

        with patch(
            "system.api.routers.agents.urllib.request.urlopen",
            return_value=_FakeResponse({"choices": []}),
        ):
            response = self.client.post(
                "/agents/local-llm-chat",
                json={"message": "backtest this setup", "product_id": "backtesting"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["provider"], "local-market-data")
        self.assertEqual(payload["product_id"], "backtesting")
        self.assertEqual(payload.get("llm_fallback_reason"), "local_llm_malformed_response")

    def test_local_llm_chat_lead_gen_marketplace_routes_with_deterministic_fallback(self) -> None:
        with patch(
            "system.api.routers.agents.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            response = self.client.post(
                "/agents/local-llm-chat",
                json={
                    "message": "Need advisory support for portfolio review this month with budget approval pending.",
                    "product_id": "lead-gen-marketplace",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["provider"], "cerebral-router")
        self.assertEqual(payload["model"], "lead-gen-intent-router-v1")
        self.assertEqual(payload["product_id"], "lead-gen-marketplace")
        self.assertIn("Lead Category:", payload["answer"])
        self.assertIn("Routing Recommendation:", payload["answer"])
        self.assertIn("Disclosure & Consent Checks:", payload["answer"])
        self.assertIn("product_registry", payload["source_tables"])
        self.assertIn("compliance_disclosure", payload["source_tables"])

    def test_local_llm_chat_lead_gen_marketplace_recovers_from_malformed_llama_response(self) -> None:
        with patch(
            "system.api.routers.agents.urllib.request.urlopen",
            return_value=_FakeResponse({"choices": []}),
        ):
            response = self.client.post(
                "/agents/local-llm-chat",
                json={
                    "message": "Route this implementation request to the right partner lane with consent checks.",
                    "product_id": "lead-gen-marketplace",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["provider"], "cerebral-router")
        self.assertEqual(payload["llm_fallback_reason"], "local_llm_malformed_response")
        self.assertIn("Fallback Reason:", payload["answer"])

    def test_local_llm_chat_timeout_returns_gateway_timeout(self) -> None:
        def fake_urlopen(_request, timeout=None):
            raise urllib.error.URLError(TimeoutError("timed out"))

        with patch("system.api.routers.agents.urllib.request.urlopen", fake_urlopen):
            response = self.client.post("/agents/local-llm-chat", json={"message": "Hello"})

        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json()["detail"], "local_llm_timeout")

    def test_local_llm_chat_malformed_response_returns_bad_gateway(self) -> None:
        with patch(
            "system.api.routers.agents.urllib.request.urlopen",
            return_value=_FakeResponse({"choices": []}),
        ):
            response = self.client.post("/agents/local-llm-chat", json={"message": "Hello"})

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"], "local_llm_malformed_response")

    def test_local_llm_status_reports_unavailable_without_failing_discovery(self) -> None:
        with patch(
            "system.api.routers.agents.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            status = self.client.get("/agents/local-llm-status")

        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["status"], "unavailable")

        capabilities = self.client.get("/capabilities").json()
        self.assertEqual(capabilities["local_llm_fallback"]["chat_endpoint"], "POST /agents/local-llm-chat")
        self.assertEqual(
            capabilities["unified_agentic_runtime"]["endpoint"],
            "POST /agents/unified-agentic-runtime",
        )

        manifest = self.client.get("/.well-known/agent-manifest.json").json()
        paths = {tool["path"] for tool in manifest["tools"]}
        self.assertIn("/agents/local-llm-status", paths)
        self.assertIn("/agents/local-llm-chat", paths)
        self.assertIn("/agents/unified-agentic-runtime", paths)

    def test_unified_agentic_runtime_local_llm_with_read_only_sql_and_documents(self) -> None:
        self.seed_market_data()

        def fake_urlopen(request, timeout):
            body = json.loads(request.data.decode("utf-8"))
            self.assertEqual(request.full_url, "http://127.0.0.1:8080/v1/chat/completions")
            self.assertEqual(body["model"], "local-llama")
            return _FakeResponse({"choices": [{"message": {"content": "Unified deep research summary"}}]})

        with patch("system.api.routers.agents.urllib.request.urlopen", fake_urlopen):
            response = self.client.post(
                "/agents/unified-agentic-runtime",
                json={
                    "product_id": "stock-screener",
                    "objective": "Generate a product-level intelligence packet",
                    "provider_preference": "local-llama",
                    "retrieval_mode": "sql",
                    "sql_queries": ["SELECT ticker, name FROM companies LIMIT 1"],
                    "include_deep_research": True,
                    "include_documents": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["kind"], "unified_agentic_runtime")
        self.assertEqual(payload["deep_research"]["provider"], "local-llama")
        self.assertEqual(payload["deep_research"]["answer"], "Unified deep research summary")
        self.assertEqual(payload["data_access"]["sql"]["count"], 1)
        self.assertEqual(payload["data_access"]["sql"]["queries"][0]["row_count"], 1)
        document_names = {item["name"] for item in payload["generated_documents"]}
        self.assertIn("agentic-execution-brief.md", document_names)
        self.assertIn("agentic-evidence-log.md", document_names)

    def test_unified_agentic_runtime_rejects_non_read_only_sql(self) -> None:
        response = self.client.post(
            "/agents/unified-agentic-runtime",
            json={
                "product_id": "stock-screener",
                "objective": "Generate research",
                "retrieval_mode": "sql",
                "sql_queries": ["DELETE FROM companies"],
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("read_only_sql_required", response.json()["detail"])

    def test_unified_agentic_runtime_accepts_firebase_preference_with_fallback(self) -> None:
        self.seed_market_data()

        def fake_urlopen(request, timeout):
            return _FakeResponse({"choices": [{"message": {"content": "Local fallback after firebase attempt"}}]})

        with patch.dict(os.environ, {"FIREBASE_GEMINI_API_KEY": "", "GEMINI_API_KEY": ""}, clear=False), patch(
            "system.api.routers.agents.urllib.request.urlopen",
            fake_urlopen,
        ):
            response = self.client.post(
                "/agents/unified-agentic-runtime",
                json={
                    "product_id": "research-workbench",
                    "objective": "Produce deep research from local evidence",
                    "provider_preference": "firebase-free-tier",
                    "retrieval_mode": "sql",
                    "sql_queries": ["SELECT ticker FROM companies LIMIT 1"],
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["deep_research"]["provider"], "local-llama")
        self.assertIn("firebase-free-tier", payload["deep_research"]["provider_chain_attempted"])
        self.assertIn("local-llama", payload["deep_research"]["provider_chain_attempted"])
        self.assertEqual(payload["deep_research"]["provider_errors"][0]["provider"], "firebase-free-tier")


if __name__ == "__main__":
    unittest.main()
