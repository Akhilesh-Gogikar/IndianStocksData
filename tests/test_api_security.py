from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from system.api.factory import create_app


class ApiSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "market.db"
        conn = sqlite3.connect(self.db_path)
        conn.executescript(Path("system/schema.sql").read_text())
        conn.close()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_api_key_is_optional_by_default(self) -> None:
        with patch.dict(os.environ, {"INDIAN_STOCKS_API_KEYS": ""}, clear=False):
            client = TestClient(create_app(self.db_path, profile_name="market-data"))
            self.assertEqual(client.get("/capabilities").status_code, 200)

    def test_api_key_protects_non_public_routes_when_configured(self) -> None:
        with patch.dict(os.environ, {"INDIAN_STOCKS_API_KEYS": "secret"}, clear=False):
            client = TestClient(create_app(self.db_path, profile_name="market-data"))
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/capabilities").status_code, 401)
            self.assertEqual(client.get("/capabilities", headers={"x-api-key": "secret"}).status_code, 200)


if __name__ == "__main__":
    unittest.main()
