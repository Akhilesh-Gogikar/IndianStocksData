from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from system.webhook_worker import deliver_pending


class _WebhookHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []
    status_code = 200

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers),
                "raw_body": body,
                "body": json.loads(body.decode("utf-8")),
            }
        )
        self.send_response(self.status_code)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format: str, *args: object) -> None:
        return


class WebhookWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "market.db"
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(Path("system/schema.sql").read_text())

    def tearDown(self) -> None:
        self.conn.close()
        self.tempdir.cleanup()

    def start_server(self, status_code: int = 200) -> tuple[HTTPServer, str]:
        handler = type("TestWebhookHandler", (_WebhookHandler,), {"requests": [], "status_code": status_code})
        server = HTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return server, f"http://127.0.0.1:{server.server_port}/webhook"

    def insert_outbox_event(
        self,
        destination_url: str | None = None,
        signing_secret: str | None = None,
        next_attempt_at: str | None = None,
    ) -> int:
        subscription_id = None
        if signing_secret:
            subscription = self.conn.execute(
                """
                INSERT INTO webhook_subscriptions (
                    owner_id, event_type, endpoint_url, signing_secret, enabled, created_at, updated_at
                )
                VALUES ('customer-a', 'watchlist.alert_triggered', ?, ?, 1,
                        '2026-05-22T00:00:00Z', '2026-05-22T00:00:00Z')
                """,
                (destination_url or "https://hooks.example.com/customer-a/alerts", signing_secret),
            )
            subscription_id = int(subscription.lastrowid)
        cursor = self.conn.execute(
            """
            INSERT INTO webhook_outbox (
                owner_id, subscription_id, destination_url, event_type, aggregate_type, aggregate_id,
                payload_json, status, created_at, next_attempt_at
            )
            VALUES ('customer-a', ?, ?, 'watchlist.alert_triggered', 'watchlist', 7,
                    ?, 'pending', '2026-05-22T00:00:00Z', ?)
            """,
            (
                subscription_id,
                destination_url,
                json.dumps({"ticker": "RELIANCE", "value": 2800.5}, sort_keys=True),
                next_attempt_at,
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def row(self, outbox_id: int) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM webhook_outbox WHERE outbox_id = ?", (outbox_id,)).fetchone()
        assert row is not None
        return row

    def delivery_attempts(self, outbox_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT *
            FROM webhook_delivery_attempts
            WHERE outbox_id = ?
            ORDER BY attempt_id
            """,
            (outbox_id,),
        ).fetchall()

    def test_deliver_pending_posts_event_and_marks_delivered(self) -> None:
        outbox_id = self.insert_outbox_event()
        server, endpoint = self.start_server()

        summary = deliver_pending(self.conn, endpoint_url=endpoint, limit=10, timeout_seconds=2)

        self.assertEqual(summary["selected"], 1)
        self.assertEqual(summary["delivered"], 1)
        row = self.row(outbox_id)
        self.assertEqual(row["status"], "delivered")
        self.assertEqual(row["attempts"], 1)
        self.assertIsNone(row["last_error"])
        self.assertIsNotNone(row["delivered_at"])
        request = server.RequestHandlerClass.requests[0]
        self.assertEqual(request["headers"]["X-Cerebral-Event-Type"], "watchlist.alert_triggered")
        self.assertEqual(request["body"]["outbox_id"], outbox_id)
        self.assertEqual(request["body"]["owner_id"], "customer-a")
        self.assertEqual(request["body"]["payload"]["ticker"], "RELIANCE")
        attempts = self.delivery_attempts(outbox_id)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["status"], "delivered")
        self.assertEqual(attempts[0]["delivered"], 1)
        self.assertEqual(attempts[0]["http_status"], 200)
        self.assertEqual(attempts[0]["endpoint_url"], endpoint)

    def test_deliver_pending_uses_stored_destination_without_fallback(self) -> None:
        server, endpoint = self.start_server()
        outbox_id = self.insert_outbox_event(destination_url=endpoint)

        summary = deliver_pending(self.conn, limit=10, timeout_seconds=2)

        self.assertEqual(summary["selected"], 1)
        self.assertEqual(summary["delivered"], 1)
        self.assertEqual(self.row(outbox_id)["status"], "delivered")
        request = server.RequestHandlerClass.requests[0]
        self.assertEqual(request["body"]["outbox_id"], outbox_id)

    def test_signed_subscription_adds_verifiable_hmac_header(self) -> None:
        secret = "customer-a-signing-secret"
        server, endpoint = self.start_server()
        outbox_id = self.insert_outbox_event(destination_url=endpoint, signing_secret=secret)

        summary = deliver_pending(self.conn, limit=10, timeout_seconds=2)

        self.assertEqual(summary["delivered"], 1)
        request = server.RequestHandlerClass.requests[0]
        signature = request["headers"]["X-Cerebral-Signature"]
        timestamp, digest = signature.split(",")
        timestamp_value = timestamp.removeprefix("t=")
        expected = hmac.new(
            secret.encode("utf-8"),
            f"{timestamp_value}.".encode("utf-8") + request["raw_body"],
            hashlib.sha256,
        ).hexdigest()
        self.assertTrue(hmac.compare_digest(digest.removeprefix("v1="), expected))
        self.assertEqual(request["body"]["outbox_id"], outbox_id)

    def test_delivery_failure_retries_then_marks_failed(self) -> None:
        outbox_id = self.insert_outbox_event()
        _, endpoint = self.start_server(status_code=500)

        first = deliver_pending(
            self.conn,
            endpoint_url=endpoint,
            limit=10,
            timeout_seconds=2,
            max_attempts=2,
            retry_base_seconds=0,
        )
        second = deliver_pending(
            self.conn,
            endpoint_url=endpoint,
            limit=10,
            timeout_seconds=2,
            max_attempts=2,
            retry_base_seconds=0,
        )

        self.assertEqual(first["retryable"], 1)
        self.assertEqual(second["failed"], 1)
        row = self.row(outbox_id)
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["attempts"], 2)
        self.assertIsNone(row["next_attempt_at"])
        self.assertIn("HTTP 500", row["last_error"])
        attempts = self.delivery_attempts(outbox_id)
        self.assertEqual([attempt["status"] for attempt in attempts], ["retryable", "failed"])
        self.assertEqual([attempt["http_status"] for attempt in attempts], [500, 500])
        self.assertTrue(all("HTTP 500" in attempt["error"] for attempt in attempts))

    def test_missing_endpoint_records_skipped_attempt(self) -> None:
        outbox_id = self.insert_outbox_event()

        summary = deliver_pending(self.conn, limit=10, timeout_seconds=2, retry_base_seconds=60)

        self.assertEqual(summary["selected"], 1)
        self.assertEqual(summary["skipped"], 1)
        row = self.row(outbox_id)
        self.assertEqual(row["status"], "pending")
        self.assertEqual(row["attempts"], 1)
        self.assertIsNotNone(row["next_attempt_at"])
        attempts = self.delivery_attempts(outbox_id)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["status"], "skipped")
        self.assertEqual(attempts[0]["delivered"], 0)
        self.assertIsNone(attempts[0]["endpoint_url"])

    def test_future_next_attempt_is_not_selected(self) -> None:
        future = "2999-01-01T00:00:00Z"
        outbox_id = self.insert_outbox_event(destination_url="https://hooks.example.com/customer-a/alerts", next_attempt_at=future)

        summary = deliver_pending(self.conn, limit=10, timeout_seconds=2)

        self.assertEqual(summary["selected"], 0)
        self.assertEqual(self.row(outbox_id)["status"], "pending")
        self.assertEqual(self.delivery_attempts(outbox_id), [])

    def test_retryable_failure_sets_next_attempt_at(self) -> None:
        outbox_id = self.insert_outbox_event()
        _, endpoint = self.start_server(status_code=500)

        summary = deliver_pending(
            self.conn,
            endpoint_url=endpoint,
            limit=10,
            timeout_seconds=2,
            max_attempts=3,
            retry_base_seconds=60,
        )

        self.assertEqual(summary["retryable"], 1)
        row = self.row(outbox_id)
        self.assertEqual(row["status"], "pending")
        self.assertEqual(row["attempts"], 1)
        self.assertIsNotNone(row["next_attempt_at"])
        self.assertGreater(row["next_attempt_at"], row["created_at"])


if __name__ == "__main__":
    unittest.main()
