from __future__ import annotations

import json
import os
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

        manifest = self.client.get("/.well-known/agent-manifest.json").json()
        paths = {tool["path"] for tool in manifest["tools"]}
        self.assertIn("/agents/local-llm-status", paths)
        self.assertIn("/agents/local-llm-chat", paths)


if __name__ == "__main__":
    unittest.main()
