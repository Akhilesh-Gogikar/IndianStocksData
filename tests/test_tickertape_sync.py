from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import tickertape_sync  # noqa: E402


class FakeResponse:
    def __init__(self, subdirectory: str) -> None:
        page_props = {
            "securityInfo": {"name": subdirectory},
            "securityQuote": {},
            "scorecard": {},
            "securitySummary": {},
            "labels": [],
            "commentary": {},
        }
        payload = {"props": {"pageProps": page_props}}
        self.status_code = 200
        self.url = f"https://www.tickertape.in/stocks/{subdirectory}"
        self.headers = {}
        self.text = (
            '<script id="__NEXT_DATA__" type="application/json">'
            f"{json.dumps(payload)}"
            "</script>"
        )


class TickertapeSyncConcurrencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.company_list = self.root / "companies.json"
        self.db = self.root / "tickertape.sqlite"
        self.raw_dir = self.root / "raw"
        companies = [
            {"subdirectory": f"company-{index:03d}", "type": "stocks", "name": f"Company {index}"}
            for index in range(6)
        ]
        self.company_list.write_text(json.dumps(companies), encoding="utf-8")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_sync_uses_bounded_workers_and_stores_successes(self) -> None:
        active = 0
        max_active = 0
        active_lock = threading.Lock()

        def fake_fetch_company(entry, timeout, retries, pacer):
            nonlocal active, max_active
            with active_lock:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.01)
            finally:
                with active_lock:
                    active -= 1
            subdirectory = entry["subdirectory"]
            return {
                "entry": entry,
                "url": f"https://www.tickertape.in/stocks/{subdirectory}",
                "fetched_at": tickertape_sync.utc_now(),
                "response": FakeResponse(subdirectory),
                "fetch_error": None,
            }

        args = argparse.Namespace(
            company_list=self.company_list,
            db=self.db,
            raw_dir=self.raw_dir,
            snapshot_date="2026-06-02",
            limit=0,
            offset=0,
            timeout=20,
            retries=0,
            workers=3,
            sleep_min=0.0,
            sleep_max=0.0,
            progress_every=10,
            max_failures=0,
            include_all_types=False,
            force=False,
            resume=False,
        )

        with mock.patch.object(tickertape_sync, "fetch_company", side_effect=fake_fetch_company):
            status = tickertape_sync.sync(args)

        self.assertEqual(status, 0)
        self.assertGreaterEqual(max_active, 2)
        self.assertLessEqual(max_active, args.workers)
        conn = sqlite3.connect(self.db)
        try:
            counts = dict(conn.execute("SELECT status, COUNT(*) FROM sync_results GROUP BY status").fetchall())
        finally:
            conn.close()
        self.assertEqual(counts, {"succeeded": 6})


if __name__ == "__main__":
    unittest.main()
