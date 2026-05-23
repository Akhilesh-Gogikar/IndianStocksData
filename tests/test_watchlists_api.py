from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from system.api.factory import create_app


class WatchlistsApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "market.db"
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
            VALUES ('RELIANCE', 2800.5, 'INR', 123456, 1, '2026-05-22', '2026-05-22T00:05:00Z', 'pass', 'derived-ok')
            """
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
        conn.commit()
        conn.close()
        self.client = TestClient(create_app(self.db_path, profile_name="market-data"))

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_watchlist_flow_enriches_items_and_evaluates_alerts(self) -> None:
        created = self.client.post(
            "/watchlists",
            json={"owner_id": "customer-a", "name": "Core holdings"},
        )
        self.assertEqual(created.status_code, 200)
        watchlist_id = created.json()["data"]["watchlist_id"]

        add_item = self.client.post(
            f"/watchlists/{watchlist_id}/items",
            json={"owner_id": "customer-a", "ticker": "reliance", "notes": "Track quarterly results"},
        )
        self.assertEqual(add_item.status_code, 200)
        item = add_item.json()["data"]["items"][0]
        self.assertEqual(item["ticker"], "RELIANCE")
        self.assertEqual(item["quote"]["price"], 2800.5)
        self.assertEqual(item["ratios"]["pe"], 24.5)

        rule = self.client.post(
            f"/watchlists/{watchlist_id}/alerts",
            json={
                "owner_id": "customer-a",
                "ticker": "RELIANCE",
                "metric": "price",
                "operator": "gte",
                "threshold": 2500,
                "cooldown_minutes": 120,
            },
        )
        self.assertEqual(rule.status_code, 200)
        rule_id = rule.json()["data"]["rule_id"]
        self.assertEqual(rule.json()["data"]["cooldown_minutes"], 120)

        updated_rule = self.client.patch(
            f"/watchlists/{watchlist_id}/alerts/{rule_id}",
            json={"owner_id": "customer-a", "threshold": 2600, "cooldown_minutes": 180},
        )
        self.assertEqual(updated_rule.status_code, 200)
        self.assertEqual(updated_rule.json()["data"]["threshold"], 2600)
        self.assertEqual(updated_rule.json()["data"]["cooldown_minutes"], 180)

        active_rules = self.client.get(f"/watchlists/{watchlist_id}/alerts?owner_id=customer-a")
        self.assertEqual(active_rules.status_code, 200)
        active_rules_body = active_rules.json()
        self.assertEqual(active_rules_body["metadata"]["result_count"], 1)
        self.assertFalse(active_rules_body["metadata"]["include_disabled"])
        self.assertEqual(active_rules_body["data"][0]["rule_id"], rule_id)
        self.assertEqual(active_rules_body["data"][0]["threshold"], 2600)
        self.assertEqual(active_rules_body["data"][0]["cooldown_minutes"], 180)
        self.assertFalse(active_rules_body["metadata"]["include_review_counts"])
        self.assertNotIn("review_summary", active_rules_body["data"][0])
        rules_with_empty_counts = self.client.get(
            f"/watchlists/{watchlist_id}/alerts?owner_id=customer-a&include_review_counts=true"
        )
        self.assertEqual(rules_with_empty_counts.status_code, 200)
        self.assertTrue(rules_with_empty_counts.json()["metadata"]["include_review_counts"])
        self.assertEqual(
            rules_with_empty_counts.json()["data"][0]["review_summary"]["counts"],
            {"open": 0, "reviewed": 0, "dismissed": 0},
        )
        empty_attention_rules = self.client.get(f"/watchlists/{watchlist_id}/alerts?owner_id=customer-a&needs_attention=true")
        self.assertEqual(empty_attention_rules.status_code, 200)
        self.assertTrue(empty_attention_rules.json()["metadata"]["include_review_counts"])
        self.assertTrue(empty_attention_rules.json()["metadata"]["needs_attention"])
        self.assertEqual(empty_attention_rules.json()["metadata"]["result_count"], 0)

        subscription = self.client.post(
            "/watchlists/webhooks/subscriptions",
            json={
                "owner_id": "customer-a",
                "endpoint_url": "https://hooks.example.com/customer-a/alerts",
                "signing_secret": "customer-a-signing-secret",
            },
        )
        self.assertEqual(subscription.status_code, 200)
        subscription_body = subscription.json()["data"]
        self.assertTrue(subscription_body["enabled"])
        self.assertTrue(subscription_body["secret_set"])
        self.assertNotIn("signing_secret", subscription_body)
        self.assertEqual(subscription_body["event_type"], "watchlist.alert_triggered")

        subscriptions = self.client.get("/watchlists/webhooks/subscriptions?owner_id=customer-a")
        self.assertEqual(subscriptions.status_code, 200)
        subscriptions_body = subscriptions.json()
        self.assertEqual(subscriptions_body["metadata"]["result_count"], 1)
        self.assertTrue(subscriptions_body["data"][0]["secret_set"])
        self.assertNotIn("signing_secret", subscriptions_body["data"][0])

        evaluated = self.client.get(
            f"/watchlists/{watchlist_id}/alerts/evaluate?owner_id=customer-a&record_events=true"
        )
        self.assertEqual(evaluated.status_code, 200)
        body = evaluated.json()
        self.assertEqual(body["metadata"]["triggered_count"], 1)
        self.assertEqual(body["metadata"]["recorded_event_count"], 1)
        self.assertEqual(body["metadata"]["outbox_event_count"], 1)
        self.assertEqual(body["metadata"]["suppressed_event_count"], 0)
        self.assertEqual(body["metadata"]["evaluated_count"], 1)
        self.assertEqual(body["metadata"]["evaluatable_metric_count"], 1)
        self.assertEqual(body["metadata"]["available_metric_count"], 1)
        self.assertEqual(body["metadata"]["missing_metric_count"], 0)
        self.assertEqual(body["metadata"]["stale_metric_count"], 0)
        self.assertEqual(body["metadata"]["warning_metric_count"], 0)
        self.assertEqual(body["data"][0]["threshold"], 2600)
        self.assertEqual(body["data"][0]["cooldown_minutes"], 180)
        self.assertTrue(body["data"][0]["available"])
        self.assertTrue(body["data"][0]["evaluatable"])
        self.assertEqual(body["data"][0]["data_status"], "available")
        self.assertEqual(body["data"][0]["freshness_status"], "fresh")
        self.assertIsNone(body["data"][0]["skip_reason"])
        self.assertTrue(body["data"][0]["triggered"])

        readiness = self.client.get(
            f"/watchlists/{watchlist_id}/alerts/readiness?owner_id=customer-a&include_available=true"
        )
        self.assertEqual(readiness.status_code, 200)
        readiness_body = readiness.json()
        self.assertEqual(readiness_body["metadata"]["owner_id"], "customer-a")
        self.assertTrue(readiness_body["metadata"]["include_available"])
        self.assertEqual(readiness_body["metadata"]["result_count"], 0)
        self.assertEqual(readiness_body["data"]["status"], "ready")
        self.assertEqual(readiness_body["data"]["blocked_count"], 0)
        self.assertEqual(readiness_body["data"]["evaluatable_metric_count"], 1)
        self.assertEqual(readiness_body["data"]["available_metric_count"], 1)
        self.assertEqual(readiness_body["data"]["missing_metric_count"], 0)
        self.assertEqual(readiness_body["data"]["stale_metric_count"], 0)
        self.assertEqual(readiness_body["data"]["missing"], [])
        self.assertEqual(readiness_body["data"]["stale"], [])
        self.assertEqual(readiness_body["data"]["available"][0]["ticker"], "RELIANCE")

        duplicate = self.client.get(f"/watchlists/{watchlist_id}/alerts/evaluate?owner_id=customer-a&record_events=true")
        self.assertEqual(duplicate.status_code, 200)
        self.assertEqual(duplicate.json()["metadata"]["recorded_event_count"], 0)
        self.assertEqual(duplicate.json()["metadata"]["suppressed_event_count"], 1)

        events = self.client.get(f"/watchlists/{watchlist_id}/alerts/events?owner_id=customer-a")
        self.assertEqual(events.status_code, 200)
        event_body = events.json()
        self.assertEqual(event_body["metadata"]["result_count"], 1)
        event_id = event_body["data"][0]["event_id"]
        self.assertEqual(event_body["data"][0]["ticker"], "RELIANCE")
        self.assertEqual(event_body["data"][0]["review"]["status"], "open")
        self.assertEqual(event_body["data"][0]["payload"]["value"], 2800.5)
        self.assertEqual(event_body["data"][0]["payload"]["threshold"], 2600)
        self.assertEqual(event_body["data"][0]["payload"]["cooldown_minutes"], 180)

        open_events = self.client.get(f"/watchlists/{watchlist_id}/alerts/events?owner_id=customer-a&review_status=open")
        self.assertEqual(open_events.status_code, 200)
        self.assertEqual(open_events.json()["metadata"]["review_status"], "open")
        self.assertEqual(open_events.json()["metadata"]["result_count"], 1)
        attention_rules = self.client.get(f"/watchlists/{watchlist_id}/alerts?owner_id=customer-a&needs_attention=true")
        self.assertEqual(attention_rules.status_code, 200)
        self.assertEqual(attention_rules.json()["metadata"]["result_count"], 1)
        self.assertEqual(attention_rules.json()["data"][0]["rule_id"], rule_id)
        self.assertEqual(
            attention_rules.json()["data"][0]["review_summary"]["counts"],
            {"open": 1, "reviewed": 0, "dismissed": 0},
        )

        summary = self.client.get(f"/watchlists/{watchlist_id}/alerts/events/summary?owner_id=customer-a")
        self.assertEqual(summary.status_code, 200)
        summary_body = summary.json()
        self.assertEqual(summary_body["data"]["total_events"], 1)
        self.assertEqual(summary_body["data"]["counts"], {"open": 1, "reviewed": 0, "dismissed": 0})
        self.assertEqual(summary_body["data"]["by_status"][0]["status"], "open")
        self.assertEqual(summary_body["data"]["by_status"][0]["event_count"], 1)
        rule_summary = self.client.get(f"/watchlists/{watchlist_id}/alerts/{rule_id}/events/summary?owner_id=customer-a")
        self.assertEqual(rule_summary.status_code, 200)
        self.assertEqual(rule_summary.json()["metadata"]["rule_id"], rule_id)
        self.assertEqual(rule_summary.json()["data"]["counts"], {"open": 1, "reviewed": 0, "dismissed": 0})

        reviewed_event = self.client.patch(
            f"/watchlists/{watchlist_id}/alerts/events/{event_id}",
            json={
                "owner_id": "customer-a",
                "status": "reviewed",
                "reviewed_by": "advisor-1",
                "notes": "Customer contacted.",
            },
        )
        self.assertEqual(reviewed_event.status_code, 200)
        reviewed_event_body = reviewed_event.json()["data"]
        self.assertEqual(reviewed_event_body["event_id"], event_id)
        self.assertEqual(reviewed_event_body["review"]["status"], "reviewed")
        self.assertEqual(reviewed_event_body["review"]["reviewed_by"], "advisor-1")
        self.assertEqual(reviewed_event_body["review"]["notes"], "Customer contacted.")
        review_history = self.client.get(f"/watchlists/{watchlist_id}/alerts/events/{event_id}/reviews?owner_id=customer-a")
        self.assertEqual(review_history.status_code, 200)
        self.assertEqual(review_history.json()["metadata"]["result_count"], 1)
        self.assertEqual(review_history.json()["data"][0]["status"], "reviewed")
        self.assertEqual(review_history.json()["data"][0]["source"], "single")
        self.assertEqual(review_history.json()["data"][0]["batch_size"], 1)
        audit_search = self.client.get(f"/watchlists/{watchlist_id}/alerts/reviews?owner_id=customer-a&status=reviewed")
        self.assertEqual(audit_search.status_code, 200)
        self.assertEqual(audit_search.json()["metadata"]["result_count"], 1)
        self.assertEqual(audit_search.json()["data"][0]["event_id"], event_id)
        self.assertEqual(audit_search.json()["data"][0]["source"], "single")

        reviewed_events = self.client.get(f"/watchlists/{watchlist_id}/alerts/events?owner_id=customer-a&review_status=reviewed")
        self.assertEqual(reviewed_events.status_code, 200)
        self.assertEqual(reviewed_events.json()["metadata"]["review_status"], "reviewed")
        self.assertEqual(reviewed_events.json()["metadata"]["result_count"], 1)
        self.assertEqual(reviewed_events.json()["data"][0]["review"]["status"], "reviewed")
        open_events_after_review = self.client.get(f"/watchlists/{watchlist_id}/alerts/events?owner_id=customer-a&review_status=open")
        self.assertEqual(open_events_after_review.status_code, 200)
        self.assertEqual(open_events_after_review.json()["metadata"]["result_count"], 0)
        attention_after_review = self.client.get(f"/watchlists/{watchlist_id}/alerts?owner_id=customer-a&needs_attention=true")
        self.assertEqual(attention_after_review.status_code, 200)
        self.assertEqual(attention_after_review.json()["metadata"]["result_count"], 0)
        reviewed_summary = self.client.get(f"/watchlists/{watchlist_id}/alerts/events/summary?owner_id=customer-a")
        self.assertEqual(reviewed_summary.status_code, 200)
        self.assertEqual(reviewed_summary.json()["data"]["counts"], {"open": 0, "reviewed": 1, "dismissed": 0})
        self.assertEqual(reviewed_summary.json()["data"]["by_status"][1]["status"], "reviewed")
        self.assertEqual(reviewed_summary.json()["data"]["by_status"][1]["event_count"], 1)
        reviewed_rule_summary = self.client.get(
            f"/watchlists/{watchlist_id}/alerts/{rule_id}/events/summary?owner_id=customer-a"
        )
        self.assertEqual(reviewed_rule_summary.status_code, 200)
        self.assertEqual(reviewed_rule_summary.json()["data"]["total_events"], 1)
        self.assertEqual(reviewed_rule_summary.json()["data"]["counts"], {"open": 0, "reviewed": 1, "dismissed": 0})
        rules_with_review_counts = self.client.get(
            f"/watchlists/{watchlist_id}/alerts?owner_id=customer-a&include_review_counts=true"
        )
        self.assertEqual(rules_with_review_counts.status_code, 200)
        self.assertEqual(
            rules_with_review_counts.json()["data"][0]["review_summary"]["counts"],
            {"open": 0, "reviewed": 1, "dismissed": 0},
        )

        rule_events = self.client.get(
            f"/watchlists/{watchlist_id}/alerts/{rule_id}/events?owner_id=customer-a&review_status=reviewed"
        )
        self.assertEqual(rule_events.status_code, 200)
        rule_event_body = rule_events.json()
        self.assertEqual(rule_event_body["metadata"]["result_count"], 1)
        self.assertEqual(rule_event_body["metadata"]["rule_id"], rule_id)
        self.assertEqual(rule_event_body["metadata"]["review_status"], "reviewed")
        self.assertEqual(rule_event_body["data"][0]["event_id"], event_id)
        self.assertEqual(rule_event_body["data"][0]["review"]["status"], "reviewed")
        self.assertEqual(rule_event_body["data"][0]["payload"]["threshold"], 2600)
        dismissed_rule_events = self.client.get(
            f"/watchlists/{watchlist_id}/alerts/{rule_id}/events?owner_id=customer-a&review_status=dismissed"
        )
        self.assertEqual(dismissed_rule_events.status_code, 200)
        self.assertEqual(dismissed_rule_events.json()["metadata"]["result_count"], 0)
        wrong_owner_rule_summary = self.client.get(
            f"/watchlists/{watchlist_id}/alerts/{rule_id}/events/summary?owner_id=customer-b"
        )
        self.assertEqual(wrong_owner_rule_summary.status_code, 404)
        wrong_owner_review = self.client.patch(
            f"/watchlists/{watchlist_id}/alerts/events/{event_id}",
            json={"owner_id": "customer-b", "status": "dismissed"},
        )
        self.assertEqual(wrong_owner_review.status_code, 404)
        wrong_owner_rule_events = self.client.get(f"/watchlists/{watchlist_id}/alerts/{rule_id}/events?owner_id=customer-b")
        self.assertEqual(wrong_owner_rule_events.status_code, 404)

        disabled_rule = self.client.delete(f"/watchlists/{watchlist_id}/alerts/{rule_id}?owner_id=customer-a")
        self.assertEqual(disabled_rule.status_code, 200)
        self.assertEqual(disabled_rule.json()["data"]["enabled"], 0)
        active_after_disable = self.client.get(f"/watchlists/{watchlist_id}/alerts?owner_id=customer-a")
        self.assertEqual(active_after_disable.status_code, 200)
        self.assertEqual(active_after_disable.json()["metadata"]["result_count"], 0)
        all_rules = self.client.get(f"/watchlists/{watchlist_id}/alerts?owner_id=customer-a&include_disabled=true")
        self.assertEqual(all_rules.status_code, 200)
        all_rules_body = all_rules.json()
        self.assertEqual(all_rules_body["metadata"]["result_count"], 1)
        self.assertTrue(all_rules_body["metadata"]["include_disabled"])
        self.assertEqual(all_rules_body["data"][0]["rule_id"], rule_id)
        self.assertEqual(all_rules_body["data"][0]["enabled"], 0)
        disabled_eval = self.client.get(f"/watchlists/{watchlist_id}/alerts/evaluate?owner_id=customer-a")
        self.assertEqual(disabled_eval.status_code, 200)
        self.assertEqual(disabled_eval.json()["metadata"]["rule_count"], 0)

        outbox = self.client.get("/watchlists/webhooks/outbox?owner_id=customer-a")
        self.assertEqual(outbox.status_code, 200)
        outbox_body = outbox.json()
        self.assertEqual(outbox_body["metadata"]["result_count"], 1)
        self.assertEqual(outbox_body["metadata"]["owner_id"], "customer-a")
        self.assertEqual(outbox_body["data"][0]["owner_id"], "customer-a")
        self.assertEqual(outbox_body["data"][0]["subscription_id"], subscription_body["subscription_id"])
        self.assertEqual(outbox_body["data"][0]["destination_url"], "https://hooks.example.com/customer-a/alerts")
        self.assertEqual(outbox_body["data"][0]["event_type"], "watchlist.alert_triggered")
        self.assertEqual(outbox_body["data"][0]["payload"]["ticker"], "RELIANCE")
        outbox_id = outbox_body["data"][0]["outbox_id"]

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            UPDATE webhook_outbox
            SET status = 'failed',
                attempts = 5,
                next_attempt_at = NULL,
                last_error = 'HTTP 500'
            WHERE outbox_id = ?
            """,
            (outbox_id,),
        )
        conn.execute(
            """
            INSERT INTO webhook_delivery_attempts (
                outbox_id, owner_id, subscription_id, endpoint_url, event_type,
                attempted_at, duration_ms, delivered, status, http_status, error
            )
            VALUES (?, 'customer-a', ?, 'https://hooks.example.com/customer-a/alerts',
                    'watchlist.alert_triggered', '2026-05-22T00:10:00Z', 42, 0,
                    'failed', 500, 'HTTP 500')
            """,
            (outbox_id, subscription_body["subscription_id"]),
        )
        conn.commit()
        conn.close()

        replay = self.client.post(
            f"/watchlists/webhooks/outbox/{outbox_id}/replay",
            json={"owner_id": "customer-a", "reason": "Endpoint fixed"},
        )
        self.assertEqual(replay.status_code, 200)
        replay_body = replay.json()
        self.assertTrue(replay_body["metadata"]["replayed"])
        self.assertEqual(replay_body["data"]["status"], "pending")
        self.assertEqual(replay_body["data"]["attempts"], 0)
        self.assertIsNotNone(replay_body["data"]["next_attempt_at"])
        self.assertIsNone(replay_body["data"]["last_error"])

        deliveries = self.client.get(f"/watchlists/webhooks/deliveries?owner_id=customer-a&outbox_id={outbox_id}")
        self.assertEqual(deliveries.status_code, 200)
        delivery_body = deliveries.json()
        self.assertEqual(delivery_body["metadata"]["result_count"], 2)
        self.assertEqual(delivery_body["data"][0]["status"], "requeued")
        self.assertEqual(delivery_body["data"][0]["error"], "Endpoint fixed")
        self.assertEqual(delivery_body["data"][1]["status"], "failed")
        self.assertEqual(delivery_body["data"][1]["duration_ms"], 42)

        status = self.client.get("/watchlists/webhooks/status?owner_id=customer-a")
        self.assertEqual(status.status_code, 200)
        status_body = status.json()
        self.assertEqual(status_body["metadata"]["owner_id"], "customer-a")
        self.assertEqual(status_body["data"]["subscriptions"]["total"], 1)
        self.assertEqual(status_body["data"]["subscriptions"]["enabled"], 1)
        self.assertEqual(status_body["data"]["subscriptions"]["signed"], 1)
        self.assertEqual(status_body["data"]["outbox"]["status_counts"]["pending"], 1)
        self.assertEqual(status_body["data"]["outbox"]["due_pending_count"], 1)
        self.assertEqual(status_body["data"]["outbox"]["scheduled_pending_count"], 0)
        self.assertEqual(status_body["data"]["deliveries"]["last_attempt"]["status"], "requeued")
        self.assertEqual(status_body["data"]["deliveries"]["recent_problem_attempts"][0]["status"], "failed")

    def test_alert_evaluation_surfaces_missing_metric_data(self) -> None:
        created = self.client.post(
            "/watchlists",
            json={"owner_id": "customer-a", "name": "Sparse data names"},
        )
        self.assertEqual(created.status_code, 200)
        watchlist_id = created.json()["data"]["watchlist_id"]

        add_item = self.client.post(
            f"/watchlists/{watchlist_id}/items",
            json={"owner_id": "customer-a", "ticker": "MISSING", "notes": None},
        )
        self.assertEqual(add_item.status_code, 200)

        rule = self.client.post(
            f"/watchlists/{watchlist_id}/alerts",
            json={
                "owner_id": "customer-a",
                "ticker": "MISSING",
                "metric": "price",
                "operator": "gte",
                "threshold": 1,
            },
        )
        self.assertEqual(rule.status_code, 200)

        evaluated = self.client.get(
            f"/watchlists/{watchlist_id}/alerts/evaluate?owner_id=customer-a&record_events=true"
        )
        self.assertEqual(evaluated.status_code, 200)
        body = evaluated.json()
        self.assertEqual(body["metadata"]["rule_count"], 1)
        self.assertEqual(body["metadata"]["evaluated_count"], 1)
        self.assertEqual(body["metadata"]["evaluatable_metric_count"], 0)
        self.assertEqual(body["metadata"]["available_metric_count"], 0)
        self.assertEqual(body["metadata"]["missing_metric_count"], 1)
        self.assertEqual(body["metadata"]["stale_metric_count"], 0)
        self.assertEqual(body["metadata"]["triggered_count"], 0)
        self.assertEqual(body["metadata"]["recorded_event_count"], 0)
        self.assertEqual(body["metadata"]["outbox_event_count"], 0)
        self.assertEqual(body["metadata"]["suppressed_event_count"], 0)
        self.assertEqual(body["data"][0]["ticker"], "MISSING")
        self.assertIsNone(body["data"][0]["value"])
        self.assertFalse(body["data"][0]["available"])
        self.assertFalse(body["data"][0]["evaluatable"])
        self.assertEqual(body["data"][0]["data_status"], "missing_metric")
        self.assertEqual(body["data"][0]["skip_reason"], "missing_metric")
        self.assertFalse(body["data"][0]["triggered"])

        readiness = self.client.get(f"/watchlists/{watchlist_id}/alerts/readiness?owner_id=customer-a")
        self.assertEqual(readiness.status_code, 200)
        readiness_body = readiness.json()
        self.assertEqual(readiness_body["metadata"]["owner_id"], "customer-a")
        self.assertFalse(readiness_body["metadata"]["include_available"])
        self.assertEqual(readiness_body["metadata"]["result_count"], 1)
        self.assertEqual(readiness_body["data"]["status"], "needs_data")
        self.assertEqual(readiness_body["data"]["blocked_count"], 1)
        self.assertEqual(readiness_body["data"]["missing_metric_count"], 1)
        self.assertEqual(readiness_body["data"]["stale_metric_count"], 0)
        self.assertEqual(readiness_body["data"]["missing"][0]["ticker"], "MISSING")
        self.assertEqual(readiness_body["data"]["stale"], [])
        self.assertEqual(readiness_body["data"]["missing_by_metric"], [{"metric": "price", "count": 1}])
        self.assertNotIn("available", readiness_body["data"])

        outbox = self.client.get("/watchlists/webhooks/outbox?owner_id=customer-a")
        self.assertEqual(outbox.status_code, 200)
        self.assertEqual(outbox.json()["metadata"]["result_count"], 0)

    def test_alert_evaluation_blocks_stale_metric_data(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO quote_snapshots (
                ticker, price, currency, volume, local_ingestion_run_id,
                as_of, processed_at, quality_status, data_rights_status
            )
            VALUES ('STALE', 42.0, 'INR', 1000, 1, '2020-01-01', '2020-01-01T00:05:00Z', 'pass', 'derived-ok')
            """
        )
        conn.commit()
        conn.close()

        created = self.client.post(
            "/watchlists",
            json={"owner_id": "customer-a", "name": "Stale data names"},
        )
        self.assertEqual(created.status_code, 200)
        watchlist_id = created.json()["data"]["watchlist_id"]

        add_item = self.client.post(
            f"/watchlists/{watchlist_id}/items",
            json={"owner_id": "customer-a", "ticker": "STALE", "notes": None},
        )
        self.assertEqual(add_item.status_code, 200)

        rule = self.client.post(
            f"/watchlists/{watchlist_id}/alerts",
            json={
                "owner_id": "customer-a",
                "ticker": "STALE",
                "metric": "price",
                "operator": "gte",
                "threshold": 1,
            },
        )
        self.assertEqual(rule.status_code, 200)

        evaluated = self.client.get(
            f"/watchlists/{watchlist_id}/alerts/evaluate?owner_id=customer-a&record_events=true&warn_days=1&stale_days=2"
        )
        self.assertEqual(evaluated.status_code, 200)
        body = evaluated.json()
        self.assertEqual(body["metadata"]["evaluatable_metric_count"], 0)
        self.assertEqual(body["metadata"]["available_metric_count"], 1)
        self.assertEqual(body["metadata"]["missing_metric_count"], 0)
        self.assertEqual(body["metadata"]["stale_metric_count"], 1)
        self.assertEqual(body["metadata"]["triggered_count"], 0)
        self.assertEqual(body["metadata"]["recorded_event_count"], 0)
        self.assertEqual(body["data"][0]["ticker"], "STALE")
        self.assertEqual(body["data"][0]["value"], 42.0)
        self.assertTrue(body["data"][0]["available"])
        self.assertFalse(body["data"][0]["evaluatable"])
        self.assertEqual(body["data"][0]["data_status"], "stale_metric")
        self.assertEqual(body["data"][0]["skip_reason"], "stale_metric")
        self.assertEqual(body["data"][0]["freshness_status"], "stale")
        self.assertFalse(body["data"][0]["triggered"])

        readiness = self.client.get(
            f"/watchlists/{watchlist_id}/alerts/readiness?owner_id=customer-a&warn_days=1&stale_days=2"
        )
        self.assertEqual(readiness.status_code, 200)
        readiness_body = readiness.json()
        self.assertEqual(readiness_body["metadata"]["result_count"], 1)
        self.assertEqual(readiness_body["metadata"]["warn_days"], 1)
        self.assertEqual(readiness_body["metadata"]["stale_days"], 2)
        self.assertEqual(readiness_body["data"]["status"], "needs_data")
        self.assertEqual(readiness_body["data"]["blocked_count"], 1)
        self.assertEqual(readiness_body["data"]["stale_metric_count"], 1)
        self.assertEqual(readiness_body["data"]["missing_metric_count"], 0)
        self.assertEqual(readiness_body["data"]["stale"][0]["ticker"], "STALE")
        self.assertEqual(readiness_body["data"]["stale_by_metric"], [{"metric": "price", "count": 1}])

        outbox = self.client.get("/watchlists/webhooks/outbox?owner_id=customer-a")
        self.assertEqual(outbox.status_code, 200)
        self.assertEqual(outbox.json()["metadata"]["result_count"], 0)

    def test_alert_evaluation_blocks_quality_and_rights_failures(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO quote_snapshots (
                ticker, price, currency, volume, local_ingestion_run_id,
                as_of, processed_at, quality_status, data_rights_status
            )
            VALUES ('QUALITYFAIL', 99.0, 'INR', 1000, 1, '2026-05-22', '2026-05-22T00:05:00Z', 'fail', 'derived-ok')
            """
        )
        conn.execute(
            """
            INSERT INTO quote_snapshots (
                ticker, price, currency, volume, local_ingestion_run_id,
                as_of, processed_at, quality_status, data_rights_status
            )
            VALUES ('RIGHTSBLOCK', 88.0, 'INR', 1000, 1, '2026-05-22', '2026-05-22T00:05:00Z', 'pass', 'blocked')
            """
        )
        conn.commit()
        conn.close()

        created = self.client.post(
            "/watchlists",
            json={"owner_id": "customer-a", "name": "Unsafe data names"},
        )
        self.assertEqual(created.status_code, 200)
        watchlist_id = created.json()["data"]["watchlist_id"]
        for ticker in ("QUALITYFAIL", "RIGHTSBLOCK"):
            add_item = self.client.post(
                f"/watchlists/{watchlist_id}/items",
                json={"owner_id": "customer-a", "ticker": ticker, "notes": None},
            )
            self.assertEqual(add_item.status_code, 200)
            rule = self.client.post(
                f"/watchlists/{watchlist_id}/alerts",
                json={
                    "owner_id": "customer-a",
                    "ticker": ticker,
                    "metric": "price",
                    "operator": "gte",
                    "threshold": 1,
                },
            )
            self.assertEqual(rule.status_code, 200)

        evaluated = self.client.get(f"/watchlists/{watchlist_id}/alerts/evaluate?owner_id=customer-a&record_events=true")
        self.assertEqual(evaluated.status_code, 200)
        body = evaluated.json()
        statuses = {item["ticker"]: item for item in body["data"]}
        self.assertEqual(body["metadata"]["evaluatable_metric_count"], 0)
        self.assertEqual(body["metadata"]["available_metric_count"], 2)
        self.assertEqual(body["metadata"]["quality_blocked_metric_count"], 1)
        self.assertEqual(body["metadata"]["data_rights_blocked_metric_count"], 1)
        self.assertEqual(body["metadata"]["triggered_count"], 0)
        self.assertEqual(body["metadata"]["recorded_event_count"], 0)
        self.assertFalse(statuses["QUALITYFAIL"]["evaluatable"])
        self.assertEqual(statuses["QUALITYFAIL"]["data_status"], "quality_failed")
        self.assertEqual(statuses["QUALITYFAIL"]["skip_reason"], "quality_failed")
        self.assertEqual(statuses["QUALITYFAIL"]["quality_status"], "fail")
        self.assertFalse(statuses["RIGHTSBLOCK"]["evaluatable"])
        self.assertEqual(statuses["RIGHTSBLOCK"]["data_status"], "data_rights_blocked")
        self.assertEqual(statuses["RIGHTSBLOCK"]["skip_reason"], "data_rights_blocked")
        self.assertEqual(statuses["RIGHTSBLOCK"]["data_rights_status"], "blocked")

        readiness = self.client.get(f"/watchlists/{watchlist_id}/alerts/readiness?owner_id=customer-a")
        self.assertEqual(readiness.status_code, 200)
        readiness_body = readiness.json()
        self.assertEqual(readiness_body["metadata"]["result_count"], 2)
        self.assertEqual(readiness_body["data"]["status"], "needs_data")
        self.assertEqual(readiness_body["data"]["blocked_count"], 2)
        self.assertEqual(readiness_body["data"]["quality_blocked_metric_count"], 1)
        self.assertEqual(readiness_body["data"]["data_rights_blocked_metric_count"], 1)
        self.assertEqual(readiness_body["data"]["quality_blocked"][0]["ticker"], "QUALITYFAIL")
        self.assertEqual(readiness_body["data"]["data_rights_blocked"][0]["ticker"], "RIGHTSBLOCK")

        actions = self.client.get("/watchlists/alerts/readiness/actions?owner_id=customer-a")
        self.assertEqual(actions.status_code, 200)
        actions_body = actions.json()
        self.assertEqual(actions_body["metadata"]["result_count"], 2)
        self.assertEqual(actions_body["metadata"]["blocked_count"], 2)
        self.assertEqual(
            actions_body["metadata"]["by_action_type"],
            [
                {"action_type": "review_data_rights", "count": 1},
                {"action_type": "review_quality_failure", "count": 1},
            ],
        )
        self.assertEqual([item["action_type"] for item in actions_body["data"]], ["review_data_rights", "review_quality_failure"])
        self.assertEqual(actions_body["data"][0]["ticker"], "RIGHTSBLOCK")
        self.assertEqual(actions_body["data"][0]["priority"], 1)
        self.assertEqual(actions_body["data"][0]["readiness_url"], f"/watchlists/{watchlist_id}/alerts/readiness")
        self.assertEqual(actions_body["data"][1]["ticker"], "QUALITYFAIL")
        limited_actions = self.client.get("/watchlists/alerts/readiness/actions?owner_id=customer-a&limit=1")
        self.assertEqual(limited_actions.status_code, 200)
        self.assertEqual(limited_actions.json()["metadata"]["result_count"], 1)
        self.assertEqual(limited_actions.json()["metadata"]["blocked_count"], 2)

        saved_queue = self.client.post(
            "/watchlists/alerts/readiness/action-queue?owner_id=customer-a&title=Alert%20readiness%20fixes"
        )
        self.assertEqual(saved_queue.status_code, 200)
        saved_body = saved_queue.json()
        self.assertEqual(saved_body["metadata"]["source_action_count"], 2)
        self.assertEqual(saved_body["metadata"]["task_count"], 2)
        self.assertTrue(saved_body["metadata"]["created"])
        self.assertFalse(saved_body["metadata"]["replaced_existing"])
        self.assertFalse(saved_body["metadata"]["closed"])
        self.assertEqual(saved_body["data"]["kind"], "saved_advisor_action_queue")
        self.assertEqual(saved_body["data"]["title"], "Alert readiness fixes")
        self.assertEqual(saved_body["data"]["focus"], "alert_readiness")
        self.assertEqual(saved_body["data"]["task_count"], 2)
        self.assertEqual(saved_body["data"]["open_task_count"], 2)
        self.assertEqual(saved_body["data"]["blocked_task_count"], 0)
        self.assertEqual(saved_body["data"]["tasks"][0]["evidence"]["action_type"], "review_data_rights")
        self.assertEqual(saved_body["data"]["source_followup"]["kind"], "alert_readiness_actions")

        replaced_queue = self.client.post(
            "/watchlists/alerts/readiness/action-queue?owner_id=customer-a&title=Updated%20alert%20readiness%20fixes"
        )
        self.assertEqual(replaced_queue.status_code, 200)
        replaced_body = replaced_queue.json()
        self.assertFalse(replaced_body["metadata"]["created"])
        self.assertTrue(replaced_body["metadata"]["replaced_existing"])
        self.assertFalse(replaced_body["metadata"]["closed"])
        self.assertEqual(replaced_body["metadata"]["saved_queue_id"], saved_body["metadata"]["saved_queue_id"])
        self.assertEqual(replaced_body["data"]["queue_id"], saved_body["data"]["queue_id"])
        self.assertEqual(replaced_body["data"]["title"], "Updated alert readiness fixes")
        self.assertEqual(replaced_body["data"]["task_count"], 2)

        conn = sqlite3.connect(self.db_path)
        persisted = conn.execute(
            "SELECT queue_id, owner_id, title, focus FROM advisor_action_queues WHERE queue_id = ?",
            (saved_body["metadata"]["saved_queue_id"],),
        ).fetchone()
        queue_count = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM advisor_action_queues
            WHERE owner_id = 'customer-a' AND focus = 'alert_readiness' AND status IN ('open', 'blocked')
            """
        ).fetchone()[0]
        conn.close()
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted[1], "customer-a")
        self.assertEqual(persisted[2], "Updated alert readiness fixes")
        self.assertEqual(persisted[3], "alert_readiness")
        self.assertEqual(queue_count, 1)

        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE quote_snapshots SET quality_status = 'pass' WHERE ticker = 'QUALITYFAIL'")
        conn.execute("UPDATE quote_snapshots SET data_rights_status = 'derived-ok' WHERE ticker = 'RIGHTSBLOCK'")
        conn.commit()
        conn.close()

        closed_queue = self.client.post("/watchlists/alerts/readiness/action-queue?owner_id=customer-a")
        self.assertEqual(closed_queue.status_code, 200)
        closed_body = closed_queue.json()
        self.assertFalse(closed_body["metadata"]["created"])
        self.assertTrue(closed_body["metadata"]["replaced_existing"])
        self.assertTrue(closed_body["metadata"]["closed"])
        self.assertEqual(closed_body["metadata"]["source_action_count"], 0)
        self.assertEqual(closed_body["metadata"]["task_count"], 0)
        self.assertEqual(closed_body["metadata"]["saved_queue_id"], saved_body["metadata"]["saved_queue_id"])
        self.assertEqual(closed_body["data"]["queue_id"], saved_body["data"]["queue_id"])
        self.assertEqual(closed_body["data"]["status"], "completed")
        self.assertEqual(closed_body["data"]["task_count"], 0)
        self.assertEqual(closed_body["data"]["tasks"], [])

        conn = sqlite3.connect(self.db_path)
        active_queue_count = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM advisor_action_queues
            WHERE owner_id = 'customer-a' AND focus = 'alert_readiness' AND status IN ('open', 'blocked')
            """
        ).fetchone()[0]
        completed_queue = conn.execute(
            "SELECT status, task_count, open_task_count FROM advisor_action_queues WHERE queue_id = ?",
            (saved_body["metadata"]["saved_queue_id"],),
        ).fetchone()
        remaining_tasks = conn.execute(
            "SELECT COUNT(*) AS count FROM advisor_action_queue_tasks WHERE queue_id = ?",
            (saved_body["metadata"]["saved_queue_id"],),
        ).fetchone()[0]
        conn.close()
        self.assertEqual(active_queue_count, 0)
        self.assertEqual(completed_queue, ("completed", 0, 0))
        self.assertEqual(remaining_tasks, 0)

        outbox = self.client.get("/watchlists/webhooks/outbox?owner_id=customer-a")
        self.assertEqual(outbox.status_code, 200)
        self.assertEqual(outbox.json()["metadata"]["result_count"], 0)

    def test_owner_alert_readiness_aggregates_watchlists(self) -> None:
        ready_watchlist = self.client.post(
            "/watchlists",
            json={"owner_id": "customer-a", "name": "Ready alerts"},
        )
        self.assertEqual(ready_watchlist.status_code, 200)
        ready_watchlist_id = ready_watchlist.json()["data"]["watchlist_id"]
        missing_watchlist = self.client.post(
            "/watchlists",
            json={"owner_id": "customer-a", "name": "Needs data alerts"},
        )
        self.assertEqual(missing_watchlist.status_code, 200)
        missing_watchlist_id = missing_watchlist.json()["data"]["watchlist_id"]

        ready_item = self.client.post(
            f"/watchlists/{ready_watchlist_id}/items",
            json={"owner_id": "customer-a", "ticker": "RELIANCE", "notes": None},
        )
        self.assertEqual(ready_item.status_code, 200)
        missing_item = self.client.post(
            f"/watchlists/{missing_watchlist_id}/items",
            json={"owner_id": "customer-a", "ticker": "MISSING", "notes": None},
        )
        self.assertEqual(missing_item.status_code, 200)

        ready_rule = self.client.post(
            f"/watchlists/{ready_watchlist_id}/alerts",
            json={
                "owner_id": "customer-a",
                "ticker": "RELIANCE",
                "metric": "price",
                "operator": "gte",
                "threshold": 2500,
            },
        )
        self.assertEqual(ready_rule.status_code, 200)
        missing_rule = self.client.post(
            f"/watchlists/{missing_watchlist_id}/alerts",
            json={
                "owner_id": "customer-a",
                "ticker": "MISSING",
                "metric": "price",
                "operator": "gte",
                "threshold": 1,
            },
        )
        self.assertEqual(missing_rule.status_code, 200)

        rollup = self.client.get("/watchlists/alerts/readiness?owner_id=customer-a")
        self.assertEqual(rollup.status_code, 200)
        body = rollup.json()
        self.assertEqual(body["metadata"]["owner_id"], "customer-a")
        self.assertEqual(body["metadata"]["watchlist_count"], 2)
        self.assertEqual(body["metadata"]["ready_watchlist_count"], 1)
        self.assertEqual(body["metadata"]["needs_data_watchlist_count"], 1)
        self.assertEqual(body["metadata"]["result_count"], 2)
        self.assertEqual(body["metadata"]["rule_count"], 2)
        self.assertEqual(body["metadata"]["evaluated_count"], 2)
        self.assertEqual(body["metadata"]["blocked_count"], 1)
        self.assertEqual(body["metadata"]["missing_metric_count"], 1)
        readiness_by_name = {item["watchlist"]["name"]: item["readiness"] for item in body["data"]}
        self.assertEqual(readiness_by_name["Ready alerts"]["status"], "ready")
        self.assertEqual(readiness_by_name["Needs data alerts"]["status"], "needs_data")

        needs_data = self.client.get("/watchlists/alerts/readiness?owner_id=customer-a&status=needs_data")
        self.assertEqual(needs_data.status_code, 200)
        needs_data_body = needs_data.json()
        self.assertEqual(needs_data_body["metadata"]["status"], "needs_data")
        self.assertEqual(needs_data_body["metadata"]["watchlist_count"], 2)
        self.assertEqual(needs_data_body["metadata"]["result_count"], 1)
        self.assertEqual(needs_data_body["data"][0]["watchlist"]["name"], "Needs data alerts")

        invalid_status = self.client.get("/watchlists/alerts/readiness?owner_id=customer-a&status=blocked")
        self.assertEqual(invalid_status.status_code, 400)

    def test_webhook_subscription_test_queues_outbox_event(self) -> None:
        subscription = self.client.post(
            "/watchlists/webhooks/subscriptions",
            json={
                "owner_id": "customer-a",
                "endpoint_url": "https://hooks.example.com/customer-a/test",
                "signing_secret": "customer-a-signing-secret",
            },
        )
        self.assertEqual(subscription.status_code, 200)
        subscription_id = subscription.json()["data"]["subscription_id"]

        test_event = self.client.post(
            f"/watchlists/webhooks/subscriptions/{subscription_id}/test",
            json={"owner_id": "customer-a", "message": "Validate setup"},
        )
        self.assertEqual(test_event.status_code, 200)
        body = test_event.json()
        self.assertTrue(body["metadata"]["queued"])
        self.assertEqual(body["metadata"]["subscription_id"], subscription_id)
        self.assertEqual(body["data"]["event_type"], "webhook.subscription_test")
        self.assertEqual(body["data"]["aggregate_type"], "webhook_subscription")
        self.assertEqual(body["data"]["aggregate_id"], subscription_id)
        self.assertEqual(body["data"]["destination_url"], "https://hooks.example.com/customer-a/test")
        self.assertEqual(body["data"]["payload"]["message"], "Validate setup")
        self.assertEqual(body["data"]["payload"]["subscription_id"], subscription_id)

        outbox = self.client.get("/watchlists/webhooks/outbox?owner_id=customer-a")
        self.assertEqual(outbox.status_code, 200)
        self.assertEqual(outbox.json()["metadata"]["result_count"], 1)

    def test_webhook_subscription_lifecycle_updates_and_disables(self) -> None:
        subscription = self.client.post(
            "/watchlists/webhooks/subscriptions",
            json={
                "owner_id": "customer-a",
                "endpoint_url": "https://hooks.example.com/customer-a/old",
                "signing_secret": "customer-a-signing-secret",
            },
        )
        self.assertEqual(subscription.status_code, 200)
        subscription_id = subscription.json()["data"]["subscription_id"]

        updated = self.client.patch(
            f"/watchlists/webhooks/subscriptions/{subscription_id}",
            json={
                "owner_id": "customer-a",
                "endpoint_url": "https://hooks.example.com/customer-a/new",
                "signing_secret": "customer-a-rotated-secret",
                "enabled": True,
            },
        )
        self.assertEqual(updated.status_code, 200)
        updated_body = updated.json()["data"]
        self.assertEqual(updated_body["endpoint_url"], "https://hooks.example.com/customer-a/new")
        self.assertTrue(updated_body["enabled"])
        self.assertTrue(updated_body["secret_set"])
        self.assertNotIn("signing_secret", updated_body)

        disabled = self.client.delete(f"/watchlists/webhooks/subscriptions/{subscription_id}?owner_id=customer-a")
        self.assertEqual(disabled.status_code, 200)
        disabled_body = disabled.json()["data"]
        self.assertFalse(disabled_body["enabled"])
        self.assertEqual(disabled_body["endpoint_url"], "https://hooks.example.com/customer-a/new")

        active = self.client.get("/watchlists/webhooks/subscriptions?owner_id=customer-a")
        self.assertEqual(active.status_code, 200)
        self.assertEqual(active.json()["metadata"]["result_count"], 0)

        all_subscriptions = self.client.get("/watchlists/webhooks/subscriptions?owner_id=customer-a&include_disabled=true")
        self.assertEqual(all_subscriptions.status_code, 200)
        self.assertEqual(all_subscriptions.json()["metadata"]["result_count"], 1)

        test_disabled = self.client.post(
            f"/watchlists/webhooks/subscriptions/{subscription_id}/test",
            json={"owner_id": "customer-a"},
        )
        self.assertEqual(test_disabled.status_code, 400)

    def test_bulk_review_alert_events_updates_multiple_events(self) -> None:
        created = self.client.post(
            "/watchlists",
            json={"owner_id": "customer-a", "name": "Bulk review"},
        )
        self.assertEqual(created.status_code, 200)
        watchlist_id = created.json()["data"]["watchlist_id"]
        add_item = self.client.post(
            f"/watchlists/{watchlist_id}/items",
            json={"owner_id": "customer-a", "ticker": "RELIANCE"},
        )
        self.assertEqual(add_item.status_code, 200)
        for payload in (
            {"owner_id": "customer-a", "ticker": "RELIANCE", "metric": "price", "operator": "gte", "threshold": 2500},
            {"owner_id": "customer-a", "ticker": "RELIANCE", "metric": "pe", "operator": "gte", "threshold": 20},
        ):
            response = self.client.post(f"/watchlists/{watchlist_id}/alerts", json=payload)
            self.assertEqual(response.status_code, 200)
        rules = self.client.get(f"/watchlists/{watchlist_id}/alerts?owner_id=customer-a")
        self.assertEqual(rules.status_code, 200)
        rule_ids = [item["rule_id"] for item in rules.json()["data"]]

        evaluated = self.client.get(f"/watchlists/{watchlist_id}/alerts/evaluate?owner_id=customer-a&record_events=true")
        self.assertEqual(evaluated.status_code, 200)
        self.assertEqual(evaluated.json()["metadata"]["recorded_event_count"], 2)
        events = self.client.get(f"/watchlists/{watchlist_id}/alerts/events?owner_id=customer-a")
        self.assertEqual(events.status_code, 200)
        event_ids = [item["event_id"] for item in events.json()["data"]]
        self.assertEqual(len(event_ids), 2)

        bulk_review = self.client.patch(
            f"/watchlists/{watchlist_id}/alerts/events/bulk",
            json={
                "owner_id": "customer-a",
                "event_ids": event_ids,
                "status": "dismissed",
                "reviewed_by": "advisor-1",
                "notes": "Handled in bulk.",
            },
        )
        self.assertEqual(bulk_review.status_code, 200)
        bulk_body = bulk_review.json()
        self.assertEqual(bulk_body["metadata"]["updated_count"], 2)
        self.assertEqual({item["review"]["status"] for item in bulk_body["data"]}, {"dismissed"})
        bulk_history = self.client.get(f"/watchlists/{watchlist_id}/alerts/events/{event_ids[0]}/reviews?owner_id=customer-a")
        self.assertEqual(bulk_history.status_code, 200)
        self.assertEqual(bulk_history.json()["metadata"]["result_count"], 1)
        self.assertEqual(bulk_history.json()["data"][0]["status"], "dismissed")
        self.assertEqual(bulk_history.json()["data"][0]["source"], "bulk")
        self.assertEqual(bulk_history.json()["data"][0]["batch_size"], 2)
        summary = self.client.get(f"/watchlists/{watchlist_id}/alerts/events/summary?owner_id=customer-a")
        self.assertEqual(summary.status_code, 200)
        self.assertEqual(summary.json()["data"]["counts"], {"open": 0, "reviewed": 0, "dismissed": 2})
        attention_rules = self.client.get(f"/watchlists/{watchlist_id}/alerts?owner_id=customer-a&needs_attention=true")
        self.assertEqual(attention_rules.status_code, 200)
        self.assertEqual(attention_rules.json()["metadata"]["result_count"], 0)

        wrong_owner = self.client.patch(
            f"/watchlists/{watchlist_id}/alerts/events/bulk",
            json={"owner_id": "customer-b", "event_ids": event_ids, "status": "reviewed"},
        )
        self.assertEqual(wrong_owner.status_code, 404)

        reopen_first_rule = self.client.patch(
            f"/watchlists/{watchlist_id}/alerts/events/bulk",
            json={
                "owner_id": "customer-a",
                "current_status": "dismissed",
                "rule_id": rule_ids[0],
                "status": "open",
                "reviewed_by": "advisor-1",
                "notes": "Reopened first rule.",
            },
        )
        self.assertEqual(reopen_first_rule.status_code, 200)
        self.assertEqual(reopen_first_rule.json()["metadata"]["updated_count"], 1)
        self.assertEqual(reopen_first_rule.json()["metadata"]["current_status"], "dismissed")
        self.assertEqual(reopen_first_rule.json()["metadata"]["rule_id"], rule_ids[0])
        self.assertEqual(reopen_first_rule.json()["data"][0]["review"]["status"], "open")
        summary_after_reopen = self.client.get(f"/watchlists/{watchlist_id}/alerts/events/summary?owner_id=customer-a")
        self.assertEqual(summary_after_reopen.status_code, 200)
        self.assertEqual(summary_after_reopen.json()["data"]["counts"], {"open": 1, "reviewed": 0, "dismissed": 1})

        close_open_for_rule = self.client.patch(
            f"/watchlists/{watchlist_id}/alerts/events/bulk",
            json={
                "owner_id": "customer-a",
                "current_status": "open",
                "rule_id": rule_ids[0],
                "status": "reviewed",
                "reviewed_by": "advisor-1",
            },
        )
        self.assertEqual(close_open_for_rule.status_code, 200)
        self.assertEqual(close_open_for_rule.json()["metadata"]["updated_count"], 1)
        final_summary = self.client.get(f"/watchlists/{watchlist_id}/alerts/events/summary?owner_id=customer-a")
        self.assertEqual(final_summary.status_code, 200)
        self.assertEqual(final_summary.json()["data"]["counts"], {"open": 0, "reviewed": 1, "dismissed": 1})
        final_history = self.client.get(
            f"/watchlists/{watchlist_id}/alerts/events/{reopen_first_rule.json()['data'][0]['event_id']}/reviews?owner_id=customer-a"
        )
        self.assertEqual(final_history.status_code, 200)
        self.assertEqual(final_history.json()["metadata"]["result_count"], 3)
        self.assertEqual([item["status"] for item in final_history.json()["data"]], ["reviewed", "open", "dismissed"])
        all_audits = self.client.get(f"/watchlists/{watchlist_id}/alerts/reviews?owner_id=customer-a")
        self.assertEqual(all_audits.status_code, 200)
        self.assertEqual(all_audits.json()["metadata"]["result_count"], 4)
        bulk_audits = self.client.get(f"/watchlists/{watchlist_id}/alerts/reviews?owner_id=customer-a&source=bulk")
        self.assertEqual(bulk_audits.status_code, 200)
        self.assertEqual(bulk_audits.json()["metadata"]["result_count"], 4)
        reviewed_rule_audits = self.client.get(
            f"/watchlists/{watchlist_id}/alerts/reviews?owner_id=customer-a&rule_id={rule_ids[0]}&status=reviewed"
        )
        self.assertEqual(reviewed_rule_audits.status_code, 200)
        self.assertEqual(reviewed_rule_audits.json()["metadata"]["rule_id"], rule_ids[0])
        self.assertEqual(reviewed_rule_audits.json()["metadata"]["status"], "reviewed")
        self.assertEqual(reviewed_rule_audits.json()["metadata"]["result_count"], 1)

    def test_capabilities_include_watchlists(self) -> None:
        response = self.client.get("/capabilities")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["watchlists_ready"])
        self.assertIn("POST /watchlists", body["watchlist_routes"])
        self.assertIn("GET /watchlists/alerts/readiness", body["watchlist_routes"])
        self.assertIn("GET /watchlists/alerts/readiness/actions", body["watchlist_routes"])
        self.assertIn("POST /watchlists/alerts/readiness/action-queue", body["watchlist_routes"])
        self.assertIn("POST /watchlists/{watchlist_id}/alerts", body["watchlist_routes"])
        self.assertIn("GET /watchlists/{watchlist_id}/alerts", body["watchlist_routes"])
        self.assertIn("GET /watchlists/{watchlist_id}/alerts/readiness", body["watchlist_routes"])
        self.assertIn("GET /watchlists/{watchlist_id}/alerts/events", body["watchlist_routes"])
        self.assertIn("GET /watchlists/{watchlist_id}/alerts/events/summary", body["watchlist_routes"])
        self.assertIn("PATCH /watchlists/{watchlist_id}/alerts/events/bulk", body["watchlist_routes"])
        self.assertIn("PATCH /watchlists/{watchlist_id}/alerts/events/{event_id}", body["watchlist_routes"])
        self.assertIn("GET /watchlists/{watchlist_id}/alerts/events/{event_id}/reviews", body["watchlist_routes"])
        self.assertIn("GET /watchlists/{watchlist_id}/alerts/reviews", body["watchlist_routes"])
        self.assertIn("GET /watchlists/{watchlist_id}/alerts/{rule_id}/events", body["watchlist_routes"])
        self.assertIn("GET /watchlists/{watchlist_id}/alerts/{rule_id}/events/summary", body["watchlist_routes"])
        self.assertIn("PATCH /watchlists/{watchlist_id}/alerts/{rule_id}", body["watchlist_routes"])
        self.assertIn("DELETE /watchlists/{watchlist_id}/alerts/{rule_id}", body["watchlist_routes"])
        self.assertIn("GET /watchlists/webhooks/subscriptions", body["watchlist_routes"])
        self.assertIn("POST /watchlists/webhooks/subscriptions", body["watchlist_routes"])
        self.assertIn("PATCH /watchlists/webhooks/subscriptions/{subscription_id}", body["watchlist_routes"])
        self.assertIn("DELETE /watchlists/webhooks/subscriptions/{subscription_id}", body["watchlist_routes"])
        self.assertIn("POST /watchlists/webhooks/subscriptions/{subscription_id}/test", body["watchlist_routes"])
        self.assertIn("GET /watchlists/webhooks/status", body["watchlist_routes"])
        self.assertIn("GET /watchlists/webhooks/outbox", body["watchlist_routes"])
        self.assertIn("POST /watchlists/webhooks/outbox/{outbox_id}/replay", body["watchlist_routes"])
        self.assertIn("GET /watchlists/webhooks/deliveries", body["watchlist_routes"])

    def test_agent_manifest_exposes_full_alert_rule_lifecycle(self) -> None:
        response = self.client.get("/.well-known/agent-manifest.json")
        self.assertEqual(response.status_code, 200)
        tools = {tool["name"]: tool for tool in response.json()["tools"]}
        self.assertEqual(tools["watchlists.add_alert_rule"]["method"], "POST")
        self.assertEqual(tools["watchlists.add_alert_rule"]["path"], "/watchlists/{watchlist_id}/alerts")
        self.assertEqual(tools["watchlists.list_alert_rules"]["method"], "GET")
        self.assertEqual(tools["watchlists.update_alert_rule"]["method"], "PATCH")
        self.assertEqual(tools["watchlists.disable_alert_rule"]["method"], "DELETE")
        self.assertEqual(tools["watchlists.owner_alert_evaluation_readiness"]["method"], "GET")
        self.assertEqual(
            tools["watchlists.owner_alert_evaluation_readiness"]["path"],
            "/watchlists/alerts/readiness",
        )
        self.assertEqual(tools["watchlists.owner_alert_readiness_actions"]["method"], "GET")
        self.assertEqual(
            tools["watchlists.owner_alert_readiness_actions"]["path"],
            "/watchlists/alerts/readiness/actions",
        )
        self.assertEqual(tools["watchlists.save_owner_alert_readiness_action_queue"]["method"], "POST")
        self.assertEqual(
            tools["watchlists.save_owner_alert_readiness_action_queue"]["path"],
            "/watchlists/alerts/readiness/action-queue",
        )
        self.assertEqual(tools["watchlists.alert_evaluation_readiness"]["method"], "GET")
        self.assertEqual(
            tools["watchlists.alert_evaluation_readiness"]["path"],
            "/watchlists/{watchlist_id}/alerts/readiness",
        )
        self.assertEqual(tools["watchlists.alert_event_review_summary"]["method"], "GET")
        self.assertEqual(
            tools["watchlists.alert_event_review_summary"]["path"],
            "/watchlists/{watchlist_id}/alerts/events/summary",
        )
        self.assertEqual(tools["watchlists.review_alert_event"]["method"], "PATCH")
        self.assertEqual(tools["watchlists.review_alert_event"]["path"], "/watchlists/{watchlist_id}/alerts/events/{event_id}")
        self.assertEqual(tools["watchlists.bulk_review_alert_events"]["method"], "PATCH")
        self.assertEqual(tools["watchlists.bulk_review_alert_events"]["path"], "/watchlists/{watchlist_id}/alerts/events/bulk")
        self.assertEqual(tools["watchlists.alert_event_review_history"]["method"], "GET")
        self.assertEqual(
            tools["watchlists.alert_event_review_history"]["path"],
            "/watchlists/{watchlist_id}/alerts/events/{event_id}/reviews",
        )
        self.assertEqual(tools["watchlists.alert_review_audit_history"]["method"], "GET")
        self.assertEqual(tools["watchlists.alert_review_audit_history"]["path"], "/watchlists/{watchlist_id}/alerts/reviews")
        self.assertEqual(tools["watchlists.alert_rule_events"]["method"], "GET")
        self.assertEqual(tools["watchlists.alert_rule_events"]["path"], "/watchlists/{watchlist_id}/alerts/{rule_id}/events")
        self.assertEqual(tools["watchlists.alert_rule_event_review_summary"]["method"], "GET")
        self.assertEqual(
            tools["watchlists.alert_rule_event_review_summary"]["path"],
            "/watchlists/{watchlist_id}/alerts/{rule_id}/events/summary",
        )


if __name__ == "__main__":
    unittest.main()
