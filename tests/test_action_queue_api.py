from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from system.api.factory import create_app


class ActionQueueApiTests(unittest.TestCase):
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

    def create_customer_portfolio(self) -> None:
        portfolio = self.client.post(
            "/portfolios",
            json={"owner_id": "customer-a", "name": "Family portfolio", "base_currency": "INR"},
        )
        portfolio_id = portfolio.json()["data"]["portfolio_id"]
        self.client.post(
            f"/portfolios/{portfolio_id}/holdings",
            json={"owner_id": "customer-a", "ticker": "RELIANCE", "quantity": 10, "average_cost": 2500},
        )

    def create_stale_prepared_notification(self, idempotency_key: str, recipient: str) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """
            INSERT INTO advisor_action_queue_escalation_notifications (
                owner_id, as_of, channel, recipient, status, idempotency_key,
                filter_json, item_count, payload_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "customer-a",
                "2026-05-28",
                "email",
                recipient,
                "prepared",
                idempotency_key,
                json.dumps({"severity": ["critical"]}),
                1,
                json.dumps({"summary": {"item_count": 1}, "items": []}),
                "2026-05-20T09:00:00Z",
                "2026-05-20T09:00:00Z",
            ),
        )
        notification_id = int(cursor.lastrowid)
        conn.commit()
        conn.close()
        return notification_id

    def test_action_queue_returns_trackable_advisor_tasks(self) -> None:
        self.create_customer_portfolio()

        response = self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "advisor_action_queue")
        self.assertEqual(body["owner_id"], "customer-a")
        self.assertGreater(body["task_count"], 0)
        self.assertTrue(body["tasks"])
        self.assertIn("queue_markdown", body)
        self.assertEqual(body["source_followup"]["kind"], "advisor_followup_pack")

        first_task = body["tasks"][0]
        self.assertIn("task_id", first_task)
        self.assertIn("completion_criteria", first_task)
        self.assertIn(first_task["status"], {"open", "blocked"})

    def test_delivery_incidents_can_be_bulk_reviewed(self) -> None:
        first_notification_id = self.create_stale_prepared_notification("bulk-incident-a", "manager-a")
        second_notification_id = self.create_stale_prepared_notification("bulk-incident-b", "manager-b")
        incident_refs = [
            {"notification_id": first_notification_id, "incident_type": "stale_prepared"},
            {"notification_id": second_notification_id, "incident_type": "stale_prepared"},
        ]

        invalid_bulk_review = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incident-reviews"
            "?owner_id=customer-a",
            json={"incident_status": "snoozed", "incidents": incident_refs},
        )
        self.assertEqual(invalid_bulk_review.status_code, 400)

        wrong_owner_bulk_review = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incident-reviews"
            "?owner_id=customer-b",
            json={
                "incident_status": "acknowledged",
                "reviewer": "ops-a",
                "incidents": incident_refs,
            },
        )
        self.assertEqual(wrong_owner_bulk_review.status_code, 404)

        bulk_review = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incident-reviews"
            "?owner_id=customer-a",
            json={
                "incident_status": "snoozed",
                "reviewer": "ops-a",
                "notes": "Pause both manager notifications for provider maintenance.",
                "follow_up_at": "2099-01-01T00:00:00Z",
                "incidents": incident_refs,
            },
        )
        self.assertEqual(bulk_review.status_code, 200)
        bulk_review_body = bulk_review.json()
        self.assertEqual(
            bulk_review_body["kind"],
            "advisor_action_queue_task_escalation_notification_delivery_incident_bulk_review",
        )
        self.assertEqual(bulk_review_body["metadata"]["requested_count"], 2)
        self.assertEqual(bulk_review_body["metadata"]["reviewed_count"], 2)
        self.assertEqual(bulk_review_body["metadata"]["incident_status"], "snoozed")
        self.assertEqual(bulk_review_body["metadata"]["reviewer"], "ops-a")
        self.assertEqual(bulk_review_body["metadata"]["follow_up_at"], "2099-01-01T00:00:00Z")
        self.assertEqual(
            {review["notification_id"] for review in bulk_review_body["data"]["reviews"]},
            {first_notification_id, second_notification_id},
        )
        self.assertEqual(
            {review["incident_status"] for review in bulk_review_body["data"]["reviews"]},
            {"snoozed"},
        )

        default_incidents = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1&limit=5"
        )
        self.assertEqual(default_incidents.status_code, 200)
        self.assertEqual(default_incidents.json()["metadata"]["result_count"], 0)
        self.assertEqual(default_incidents.json()["metadata"]["suppressed_count"], 2)

        visible_incidents = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1&limit=5&include_suppressed=true"
        )
        self.assertEqual(visible_incidents.status_code, 200)
        visible_incidents_body = visible_incidents.json()
        self.assertEqual(visible_incidents_body["metadata"]["result_count"], 2)
        self.assertEqual(
            {incident["suppression_reason"] for incident in visible_incidents_body["data"]},
            {"snoozed_until_follow_up"},
        )

        incident_summary = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incident-summary"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1&max_incidents=5"
        )
        self.assertEqual(incident_summary.status_code, 200)
        incident_summary_body = incident_summary.json()
        self.assertEqual(incident_summary_body["data"]["summary"]["total_count"], 2)
        self.assertEqual(incident_summary_body["data"]["summary"]["actionable_count"], 0)
        self.assertEqual(incident_summary_body["data"]["summary"]["snoozed_count"], 2)
        self.assertEqual(incident_summary_body["metadata"]["suppressed_count"], 2)

        reviews = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incident-reviews"
            "?owner_id=customer-a&incident_type=stale_prepared&incident_status=snoozed&limit=5"
        )
        self.assertEqual(reviews.status_code, 200)
        self.assertEqual(reviews.json()["metadata"]["result_count"], 2)

    def test_delivery_incident_workload_summarizes_ownership_and_followups(self) -> None:
        first_notification_id = self.create_stale_prepared_notification("workload-incident-a", "manager-a")
        second_notification_id = self.create_stale_prepared_notification("workload-incident-b", "manager-b")
        unassigned_notification_id = self.create_stale_prepared_notification("workload-incident-c", "manager-c")
        resolved_notification_id = self.create_stale_prepared_notification("workload-incident-d", "manager-d")
        assigned_refs = [
            {"notification_id": first_notification_id, "incident_type": "stale_prepared"},
            {"notification_id": second_notification_id, "incident_type": "stale_prepared"},
        ]

        assigned_review = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incident-reviews"
            "?owner_id=customer-a",
            json={
                "incident_status": "assigned",
                "reviewer": "ops-a",
                "assigned_to": "worker-a",
                "notes": "Assign provider follow-up to worker-a.",
                "follow_up_at": "2020-01-01T00:00:00Z",
                "incidents": assigned_refs,
            },
        )
        self.assertEqual(assigned_review.status_code, 200)

        resolved_review = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents/"
            f"{resolved_notification_id}/review?owner_id=customer-a",
            json={
                "incident_type": "stale_prepared",
                "incident_status": "resolved",
                "reviewer": "ops-a",
                "notes": "No operator action required.",
            },
        )
        self.assertEqual(resolved_review.status_code, 200)

        workload = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incident-workload"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1&follow_up_within_hours=24&max_incidents=10"
        )
        self.assertEqual(workload.status_code, 200)
        workload_body = workload.json()
        self.assertEqual(
            workload_body["kind"],
            "advisor_action_queue_task_escalation_notification_delivery_incident_workload",
        )
        self.assertEqual(workload_body["data"]["summary"]["total_count"], 4)
        self.assertEqual(workload_body["data"]["summary"]["unresolved_count"], 3)
        self.assertEqual(workload_body["data"]["summary"]["actionable_count"], 3)
        self.assertEqual(workload_body["data"]["summary"]["resolved_count"], 1)
        self.assertEqual(workload_body["data"]["summary"]["assigned_actionable_count"], 2)
        self.assertEqual(workload_body["data"]["summary"]["unassigned_actionable_count"], 1)
        self.assertEqual(workload_body["data"]["summary"]["follow_up_overdue_count"], 2)
        self.assertEqual(workload_body["data"]["summary"]["follow_up_missing_count"], 1)
        self.assertEqual(workload_body["metadata"]["suppressed_count"], 1)

        by_assignee = {row["assigned_to"]: row for row in workload_body["data"]["by_assignee"]}
        self.assertEqual(by_assignee["worker-a"]["unresolved_count"], 2)
        self.assertEqual(by_assignee["worker-a"]["actionable_count"], 2)
        self.assertEqual(by_assignee["worker-a"]["follow_up_overdue_count"], 2)
        self.assertEqual(by_assignee["worker-a"]["medium_count"], 2)
        self.assertEqual(by_assignee["unassigned"]["unresolved_count"], 1)
        self.assertEqual(by_assignee["unassigned"]["follow_up_missing_count"], 1)

        by_follow_up = {
            row["follow_up_status"]: row for row in workload_body["data"]["by_follow_up_status"]
        }
        self.assertEqual(by_follow_up["overdue"]["unresolved_count"], 2)
        self.assertEqual(by_follow_up["missing"]["unresolved_count"], 1)

        assigned_overdue_incidents = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1&assigned_to=worker-a"
            "&follow_up_status=overdue&follow_up_within_hours=24&limit=10"
        )
        self.assertEqual(assigned_overdue_incidents.status_code, 200)
        assigned_overdue_body = assigned_overdue_incidents.json()
        self.assertEqual(assigned_overdue_body["metadata"]["result_count"], 2)
        self.assertEqual(assigned_overdue_body["metadata"]["filtered_count"], 2)
        self.assertEqual(assigned_overdue_body["metadata"]["assigned_to"], "worker-a")
        self.assertEqual(assigned_overdue_body["metadata"]["follow_up_status"], "overdue")
        self.assertEqual(
            {incident["notification_id"] for incident in assigned_overdue_body["data"]},
            {first_notification_id, second_notification_id},
        )
        self.assertEqual(
            {incident["latest_review"]["assigned_to"] for incident in assigned_overdue_body["data"]},
            {"worker-a"},
        )
        self.assertEqual({incident["follow_up_status"] for incident in assigned_overdue_body["data"]}, {"overdue"})

        unassigned_missing_incidents = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1&assigned_to=unassigned"
            "&follow_up_status=missing&limit=10"
        )
        self.assertEqual(unassigned_missing_incidents.status_code, 200)
        unassigned_missing_body = unassigned_missing_incidents.json()
        self.assertEqual(unassigned_missing_body["metadata"]["result_count"], 1)
        self.assertEqual(unassigned_missing_body["data"][0]["notification_id"], unassigned_notification_id)
        self.assertIsNone(unassigned_missing_body["data"][0]["latest_review"])
        self.assertEqual(unassigned_missing_body["data"][0]["follow_up_status"], "missing")

        invalid_follow_up_status = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents"
            "?owner_id=customer-a&follow_up_status=not-real"
        )
        self.assertEqual(invalid_follow_up_status.status_code, 400)

    def test_action_queue_can_be_saved_listed_and_updated(self) -> None:
        self.create_customer_portfolio()

        generated = self.client.post(
            "/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true"
        ).json()
        queue_id = generated["saved_queue_id"]
        task_id = generated["tasks"][0]["task_id"]

        listed = self.client.get("/agents/action-queues?owner_id=customer-a")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["data"][0]["queue_id"], queue_id)

        fetched = self.client.get(f"/agents/action-queues/{queue_id}?owner_id=customer-a")
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.json()["kind"], "saved_advisor_action_queue")
        self.assertEqual(fetched.json()["task_count"], generated["task_count"])

        updated = self.client.patch(
            f"/agents/action-queues/{queue_id}/tasks/{task_id}?owner_id=customer-a",
            json={
                "status": "completed",
                "notes": "Reviewed with advisor.",
                "assigned_to": "advisor-a",
                "due_at": "2026-05-23",
                "updated_by": "advisor-a",
                "update_source": "task-detail",
            },
        )

        self.assertEqual(updated.status_code, 200)
        body = updated.json()
        self.assertEqual(body["tasks"][0]["status"], "completed")
        self.assertEqual(body["tasks"][0]["notes"], "Reviewed with advisor.")
        self.assertEqual(body["tasks"][0]["assigned_to"], "advisor-a")
        self.assertEqual(body["tasks"][0]["due_at"], "2026-05-23")
        self.assertEqual(body["completed_task_count"], 1)

        single_activity = self.client.get(
            f"/agents/action-queues/tasks/activity?owner_id=customer-a&queue_id={queue_id}&task_id={task_id}"
        )
        self.assertEqual(single_activity.status_code, 200)
        single_activity_body = single_activity.json()
        self.assertEqual(single_activity_body["kind"], "advisor_action_queue_task_activity")
        self.assertEqual(single_activity_body["metadata"]["result_count"], 1)
        self.assertEqual(single_activity_body["data"][0]["previous_status"], generated["tasks"][0]["status"])
        self.assertEqual(single_activity_body["data"][0]["new_status"], "completed")
        self.assertIsNone(single_activity_body["data"][0]["previous_notes"])
        self.assertEqual(single_activity_body["data"][0]["new_notes"], "Reviewed with advisor.")
        self.assertIsNone(single_activity_body["data"][0]["previous_assigned_to"])
        self.assertEqual(single_activity_body["data"][0]["new_assigned_to"], "advisor-a")
        self.assertIsNone(single_activity_body["data"][0]["previous_due_at"])
        self.assertEqual(single_activity_body["data"][0]["new_due_at"], "2026-05-23")
        self.assertEqual(single_activity_body["data"][0]["updated_by"], "advisor-a")
        self.assertEqual(single_activity_body["data"][0]["update_source"], "task-detail")

        completed_tasks = self.client.get("/agents/action-queues/tasks?owner_id=customer-a&status=completed")
        self.assertEqual(completed_tasks.status_code, 200)
        completed_body = completed_tasks.json()
        self.assertEqual(completed_body["kind"], "advisor_action_queue_tasks")
        self.assertEqual(completed_body["metadata"]["result_count"], 1)
        self.assertEqual(completed_body["data"][0]["queue_id"], queue_id)
        self.assertEqual(completed_body["data"][0]["task_id"], task_id)
        self.assertEqual(completed_body["data"][0]["status"], "completed")
        self.assertEqual(completed_body["data"][0]["notes"], "Reviewed with advisor.")
        self.assertEqual(completed_body["data"][0]["assigned_to"], "advisor-a")
        self.assertEqual(completed_body["data"][0]["due_at"], "2026-05-23")
        self.assertEqual(
            completed_body["data"][0]["task_url"],
            f"/agents/action-queues/{queue_id}/tasks/{task_id}?owner_id=customer-a",
        )
        self.assertNotIn("evidence", completed_body["data"][0])

        assigned_tasks = self.client.get(
            "/agents/action-queues/tasks?owner_id=customer-a&status=completed&assigned_to=advisor-a&due_before=2026-05-24"
        )
        self.assertEqual(assigned_tasks.status_code, 200)
        self.assertEqual(assigned_tasks.json()["metadata"]["result_count"], 1)

        active_tasks = self.client.get("/agents/action-queues/tasks?owner_id=customer-a&status=active&focus=telecom")
        self.assertEqual(active_tasks.status_code, 200)
        active_body = active_tasks.json()
        self.assertEqual(active_body["metadata"]["result_count"], body["open_task_count"] + body["blocked_task_count"])
        self.assertTrue(all(item["status"] in {"open", "blocked"} for item in active_body["data"]))
        self.assertTrue(all(item["focus"] == "telecom" for item in active_body["data"]))

        invalid_tasks = self.client.get("/agents/action-queues/tasks?owner_id=customer-a&status=unknown")
        self.assertEqual(invalid_tasks.status_code, 400)

        summary = self.client.get("/agents/action-queues/summary?owner_id=customer-a")
        self.assertEqual(summary.status_code, 200)
        summary_body = summary.json()
        attention_count = body["open_task_count"] + body["blocked_task_count"]
        self.assertEqual(summary_body["kind"], "advisor_action_queue_summary")
        self.assertEqual(summary_body["owner_id"], "customer-a")
        self.assertEqual(summary_body["totals"]["queue_count"], 1)
        self.assertEqual(summary_body["totals"]["task_count"], body["task_count"])
        self.assertEqual(summary_body["totals"]["open_task_count"], body["open_task_count"])
        self.assertEqual(summary_body["totals"]["blocked_task_count"], body["blocked_task_count"])
        self.assertEqual(summary_body["totals"]["completed_task_count"], body["completed_task_count"])
        self.assertEqual(summary_body["totals"]["attention_task_count"], attention_count)
        self.assertEqual(summary_body["totals"]["active_queue_count"], 1 if body["status"] in {"open", "blocked"} else 0)
        self.assertEqual(summary_body["totals"]["completed_queue_count"], 1 if body["status"] == "completed" else 0)
        self.assertEqual(summary_body["by_status"][0]["status"], body["status"])
        self.assertEqual(summary_body["by_status"][0]["queue_count"], 1)
        self.assertEqual(sum(item["attention_task_count"] for item in summary_body["task_urgency"]), attention_count)
        if attention_count:
            self.assertEqual(summary_body["by_focus"][0]["focus"], "telecom")
            self.assertEqual(summary_body["by_focus"][0]["attention_task_count"], attention_count)
        else:
            self.assertEqual(summary_body["by_focus"], [])
        self.assertEqual(summary_body["recent_queues"][0]["queue_id"], queue_id)
        self.assertNotIn("tasks", summary_body["recent_queues"][0])

        second_generated = self.client.post(
            "/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true"
        ).json()
        second_queue_id = second_generated["saved_queue_id"]
        second_task_id = second_generated["tasks"][0]["task_id"]
        bulk_update = self.client.patch(
            "/agents/action-queues/tasks?owner_id=customer-a",
            json={
                "status": "deferred",
                "notes": "Deferred from dashboard triage.",
                "assigned_to": "advisor-b",
                "due_at": "2026-05-24",
                "updated_by": "advisor-a",
                "update_source": "dashboard-bulk",
                "tasks": [
                    {"queue_id": queue_id, "task_id": task_id},
                    {"queue_id": second_queue_id, "task_id": second_task_id},
                ],
            },
        )
        self.assertEqual(bulk_update.status_code, 200)
        bulk_body = bulk_update.json()
        self.assertEqual(bulk_body["kind"], "advisor_action_queue_task_bulk_update")
        self.assertEqual(bulk_body["metadata"]["requested_count"], 2)
        self.assertEqual(bulk_body["metadata"]["updated_count"], 2)
        self.assertEqual(bulk_body["metadata"]["updated_queue_count"], 2)
        self.assertEqual({item["status"] for item in bulk_body["data"]["updated_tasks"]}, {"deferred"})
        self.assertEqual({item["notes"] for item in bulk_body["data"]["updated_tasks"]}, {"Deferred from dashboard triage."})
        self.assertEqual({item["assigned_to"] for item in bulk_body["data"]["updated_tasks"]}, {"advisor-b"})
        self.assertEqual({item["due_at"] for item in bulk_body["data"]["updated_tasks"]}, {"2026-05-24"})
        self.assertEqual(bulk_body["metadata"]["updated_by"], "advisor-a")
        self.assertEqual(bulk_body["metadata"]["update_source"], "dashboard-bulk")
        self.assertEqual(bulk_body["metadata"]["assigned_to"], "advisor-b")
        self.assertEqual(bulk_body["metadata"]["due_at"], "2026-05-24")
        self.assertEqual(
            {item["queue_id"] for item in bulk_body["data"]["updated_queues"]},
            {queue_id, second_queue_id},
        )

        workload = self.client.get("/agents/action-queues/tasks/workload?owner_id=customer-a&as_of=2026-05-25")
        self.assertEqual(workload.status_code, 200)
        workload_body = workload.json()
        advisor_b_workload = {item["assignee"]: item for item in workload_body["by_assignee"]}["advisor-b"]
        self.assertEqual(workload_body["kind"], "advisor_action_queue_task_workload")
        self.assertEqual(workload_body["owner_id"], "customer-a")
        self.assertEqual(workload_body["as_of"], "2026-05-25")
        self.assertEqual(workload_body["totals"]["assigned_task_count"], 2)
        self.assertEqual(workload_body["totals"]["overdue_task_count"], 2)
        self.assertEqual(workload_body["due_buckets"][0], {"bucket": "overdue", "task_count": 2})
        self.assertEqual(advisor_b_workload["task_count"], 2)
        self.assertEqual(advisor_b_workload["deferred_task_count"], 2)
        self.assertEqual(advisor_b_workload["overdue_task_count"], 2)
        self.assertEqual(advisor_b_workload["next_due_at"], "2026-05-24")

        escalations = self.client.get(
            "/agents/action-queues/tasks/escalations?owner_id=customer-a&as_of=2026-05-25&limit=10"
        )
        self.assertEqual(escalations.status_code, 200)
        escalation_body = escalations.json()
        advisor_b_escalations = [item for item in escalation_body["data"] if item["assigned_to"] == "advisor-b"]
        reason_counts = {item["reason"]: item["count"] for item in escalation_body["metadata"]["by_reason"]}
        self.assertEqual(escalation_body["kind"], "advisor_action_queue_task_escalations")
        self.assertEqual(escalation_body["owner_id"], "customer-a")
        self.assertEqual(escalation_body["as_of"], "2026-05-25")
        self.assertEqual(len(advisor_b_escalations), 2)
        self.assertEqual({item["severity"] for item in advisor_b_escalations}, {"critical"})
        self.assertTrue(all("overdue" in item["escalation_reasons"] for item in advisor_b_escalations))
        self.assertTrue(all(item["days_overdue"] == 1 for item in advisor_b_escalations))
        self.assertGreaterEqual(reason_counts["overdue"], 2)

        invalid_escalations = self.client.get(
            "/agents/action-queues/tasks/escalations?owner_id=customer-a&as_of=bad-date"
        )
        self.assertEqual(invalid_escalations.status_code, 400)

        other_portfolio = self.client.post(
            "/portfolios",
            json={"owner_id": "customer-b", "name": "Second family portfolio", "base_currency": "INR"},
        )
        other_portfolio_id = other_portfolio.json()["data"]["portfolio_id"]
        self.client.post(
            f"/portfolios/{other_portfolio_id}/holdings",
            json={"owner_id": "customer-b", "ticker": "RELIANCE", "quantity": 5, "average_cost": 2400},
        )
        other_generated = self.client.post(
            "/agents/action-queue?owner_id=customer-b&focus=telecom&evidence_limit=1&save=true"
        ).json()
        other_task_id = other_generated["tasks"][0]["task_id"]
        other_update = self.client.patch(
            f"/agents/action-queues/{other_generated['saved_queue_id']}/tasks/{other_task_id}?owner_id=customer-b",
            json={
                "status": "blocked",
                "assigned_to": "advisor-c",
                "due_at": "2026-05-25",
                "updated_by": "advisor-c",
                "update_source": "manager-summary-fixture",
            },
        )
        self.assertEqual(other_update.status_code, 200)

        escalation_summary = self.client.get(
            "/agents/action-queues/tasks/escalations/summary?as_of=2026-05-25&limit=10"
        )
        self.assertEqual(escalation_summary.status_code, 200)
        escalation_summary_body = escalation_summary.json()
        owner_summaries = {item["owner_id"]: item for item in escalation_summary_body["owners"]}
        self.assertEqual(escalation_summary_body["kind"], "advisor_action_queue_task_escalation_summary")
        self.assertEqual(escalation_summary_body["metadata"]["scope"], "book")
        self.assertGreaterEqual(escalation_summary_body["totals"]["owner_count"], 2)
        self.assertGreaterEqual(escalation_summary_body["totals"]["escalated_task_count"], 3)
        self.assertGreaterEqual(owner_summaries["customer-a"]["overdue_task_count"], 2)
        self.assertGreaterEqual(owner_summaries["customer-a"]["critical_task_count"], 2)
        self.assertGreaterEqual(owner_summaries["customer-b"]["blocked_task_count"], 1)
        self.assertEqual(owner_summaries["customer-b"]["top_tasks"][0]["severity"], "critical")

        scoped_escalation_summary = self.client.get(
            "/agents/action-queues/tasks/escalations/summary?owner_id=customer-b&as_of=2026-05-25"
        )
        self.assertEqual(scoped_escalation_summary.status_code, 200)
        self.assertEqual(scoped_escalation_summary.json()["metadata"]["scope"], "owner")
        self.assertEqual([item["owner_id"] for item in scoped_escalation_summary.json()["owners"]], ["customer-b"])

        invalid_escalation_summary = self.client.get(
            "/agents/action-queues/tasks/escalations/summary?as_of=bad-date"
        )
        self.assertEqual(invalid_escalation_summary.status_code, 400)

        escalation_review = self.client.post(
            f"/agents/action-queues/{queue_id}/tasks/{task_id}/escalation-review?owner_id=customer-a",
            json={
                "review_status": "snoozed",
                "reviewer": "manager-a",
                "notes": "Advisor follow-up already scheduled.",
                "snoozed_until": "2026-05-27",
            },
        )
        self.assertEqual(escalation_review.status_code, 200)
        escalation_review_body = escalation_review.json()
        self.assertEqual(escalation_review_body["kind"], "advisor_action_queue_task_escalation_review")
        self.assertEqual(escalation_review_body["data"]["review"]["review_status"], "snoozed")
        self.assertEqual(escalation_review_body["data"]["review"]["reviewer"], "manager-a")
        self.assertEqual(escalation_review_body["data"]["review"]["snoozed_until"], "2026-05-27")
        self.assertEqual(escalation_review_body["data"]["task"]["task_id"], task_id)

        reviewed_escalations = self.client.get(
            "/agents/action-queues/tasks/escalations?owner_id=customer-a&as_of=2026-05-25&limit=10"
        )
        self.assertEqual(reviewed_escalations.status_code, 200)
        reviewed_task = [
            item for item in reviewed_escalations.json()["data"] if item["queue_id"] == queue_id and item["task_id"] == task_id
        ][0]
        self.assertEqual(reviewed_task["review_status"], "snoozed")
        self.assertEqual(reviewed_task["reviewer"], "manager-a")
        self.assertEqual(reviewed_task["snoozed_until"], "2026-05-27")
        self.assertEqual(reviewed_task["latest_review"]["notes"], "Advisor follow-up already scheduled.")

        reviewed_summary = self.client.get(
            "/agents/action-queues/tasks/escalations/summary?owner_id=customer-a&as_of=2026-05-25"
        )
        self.assertEqual(reviewed_summary.status_code, 200)
        reviewed_owner = reviewed_summary.json()["owners"][0]
        self.assertEqual(reviewed_summary.json()["totals"]["snoozed_task_count"], 1)
        self.assertEqual(reviewed_owner["snoozed_task_count"], 1)
        self.assertGreaterEqual(reviewed_owner["unreviewed_task_count"], 1)
        self.assertIn("snoozed", {item["review_status"] for item in reviewed_owner["top_tasks"]})

        escalation_reviews = self.client.get(
            f"/agents/action-queues/tasks/escalations/reviews?owner_id=customer-a&queue_id={queue_id}&task_id={task_id}"
        )
        self.assertEqual(escalation_reviews.status_code, 200)
        self.assertEqual(escalation_reviews.json()["metadata"]["result_count"], 1)
        self.assertEqual(escalation_reviews.json()["data"][0]["review_status"], "snoozed")
        self.assertEqual(escalation_reviews.json()["data"][0]["current_status"], "deferred")

        invalid_escalation_review = self.client.post(
            f"/agents/action-queues/{queue_id}/tasks/{task_id}/escalation-review?owner_id=customer-a",
            json={"review_status": "unknown"},
        )
        self.assertEqual(invalid_escalation_review.status_code, 400)

        actionable_inbox = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox?owner_id=customer-a&as_of=2026-05-25&limit=10"
        )
        self.assertEqual(actionable_inbox.status_code, 200)
        actionable_body = actionable_inbox.json()
        self.assertEqual(actionable_body["kind"], "advisor_action_queue_task_escalation_inbox")
        self.assertEqual(actionable_body["metadata"]["scope"], "owner")
        self.assertEqual(actionable_body["totals"]["excluded_snoozed_task_count"], 1)
        actionable_keys = {(item["queue_id"], item["task_id"]) for item in actionable_body["data"]}
        self.assertNotIn((queue_id, task_id), actionable_keys)
        self.assertIn((second_queue_id, second_task_id), actionable_keys)
        self.assertTrue(all(item["inbox_status"] != "snoozed" for item in actionable_body["data"]))

        resolved_review = self.client.post(
            f"/agents/action-queues/{second_queue_id}/tasks/{second_task_id}/escalation-review?owner_id=customer-a",
            json={"review_status": "resolved", "reviewer": "manager-a", "notes": "Duplicate follow-up closed."},
        )
        self.assertEqual(resolved_review.status_code, 200)

        future_inbox = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox?owner_id=customer-a&as_of=2026-05-28&limit=10"
        )
        self.assertEqual(future_inbox.status_code, 200)
        future_body = future_inbox.json()
        future_items = {(item["queue_id"], item["task_id"]): item for item in future_body["data"]}
        self.assertIn((queue_id, task_id), future_items)
        self.assertNotIn((second_queue_id, second_task_id), future_items)
        self.assertEqual(future_items[(queue_id, task_id)]["inbox_status"], "snooze_expired")
        self.assertEqual(future_body["totals"]["snooze_expired_task_count"], 1)
        self.assertEqual(future_body["totals"]["excluded_resolved_task_count"], 1)
        self.assertIn("snooze_expired", {item["inbox_status"] for item in future_body["owners"][0]["top_tasks"]})

        filtered_inbox = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox"
            "?owner_id=customer-a&as_of=2026-05-28&severity=critical"
            "&inbox_status=snooze_expired&assigned_to=advisor-b&focus=telecom"
        )
        self.assertEqual(filtered_inbox.status_code, 200)
        filtered_body = filtered_inbox.json()
        self.assertEqual(filtered_body["metadata"]["severity"], ["critical"])
        self.assertEqual(filtered_body["metadata"]["inbox_status"], ["snooze_expired"])
        self.assertEqual(filtered_body["metadata"]["assigned_to"], "advisor-b")
        self.assertEqual(filtered_body["metadata"]["focus"], "telecom")
        self.assertEqual(filtered_body["metadata"]["result_count"], 1)
        self.assertEqual(filtered_body["data"][0]["queue_id"], queue_id)
        self.assertEqual(filtered_body["data"][0]["task_id"], task_id)
        self.assertEqual(filtered_body["data"][0]["inbox_status"], "snooze_expired")

        notification = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification"
            "?owner_id=customer-a&as_of=2026-05-28&severity=critical"
            "&inbox_status=snooze_expired&assigned_to=advisor-b&focus=telecom"
        )
        self.assertEqual(notification.status_code, 200)
        notification_body = notification.json()
        self.assertEqual(notification_body["kind"], "advisor_action_queue_task_escalation_notification")
        self.assertEqual(notification_body["summary"]["item_count"], 1)
        self.assertEqual(notification_body["summary"]["critical_task_count"], 1)
        self.assertEqual(notification_body["metadata"]["source_actionable_task_count"], 1)
        self.assertEqual(notification_body["items"][0]["queue_id"], queue_id)
        self.assertEqual(notification_body["items"][0]["task_id"], task_id)
        self.assertEqual(notification_body["items"][0]["inbox_status"], "snooze_expired")
        self.assertIn("notification_markdown", notification_body)
        self.assertNotIn("rationale", notification_body["items"][0])
        self.assertNotIn("completion_criteria", notification_body["items"][0])
        self.assertNotIn("notes", notification_body["items"][0])
        self.assertNotIn("latest_review", notification_body["items"][0])

        notification_log = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification"
            "?owner_id=customer-a&as_of=2026-05-28&severity=critical"
            "&inbox_status=snooze_expired&assigned_to=advisor-b&focus=telecom",
            json={"channel": "email", "recipient": "manager-a", "status": "sent"},
        )
        self.assertEqual(notification_log.status_code, 200)
        notification_log_body = notification_log.json()
        notification_id = notification_log_body["data"]["notification"]["notification_id"]
        self.assertEqual(notification_log_body["kind"], "advisor_action_queue_task_escalation_notification_log")
        self.assertTrue(notification_log_body["metadata"]["created"])
        self.assertEqual(notification_log_body["data"]["notification"]["status"], "sent")
        self.assertEqual(notification_log_body["data"]["notification"]["channel"], "email")
        self.assertEqual(notification_log_body["data"]["notification"]["recipient"], "manager-a")
        self.assertEqual(notification_log_body["data"]["payload"]["summary"]["item_count"], 1)

        duplicate_notification_log = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification"
            "?owner_id=customer-a&as_of=2026-05-28&severity=critical"
            "&inbox_status=snooze_expired&assigned_to=advisor-b&focus=telecom",
            json={"channel": "email", "recipient": "manager-a", "status": "sent"},
        )
        self.assertEqual(duplicate_notification_log.status_code, 200)
        self.assertFalse(duplicate_notification_log.json()["metadata"]["created"])
        self.assertEqual(duplicate_notification_log.json()["metadata"]["notification_id"], notification_id)

        notification_logs = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/logs"
            "?owner_id=customer-a&channel=email&status=sent"
        )
        self.assertEqual(notification_logs.status_code, 200)
        self.assertEqual(notification_logs.json()["metadata"]["result_count"], 1)
        self.assertEqual(notification_logs.json()["data"][0]["notification_id"], notification_id)
        self.assertEqual(notification_logs.json()["data"][0]["payload_summary"]["item_count"], 1)
        self.assertEqual(notification_logs.json()["data"][0]["filter"]["severity"], ["critical"])

        notification_update = self.client.patch(
            f"/agents/action-queues/tasks/escalations/inbox/notification/logs/{notification_id}"
            "?owner_id=customer-a",
            json={
                "status": "failed",
                "delivery_notes": "SMTP provider rejected the message.",
                "delivered_at": "2026-05-28T10:30:00Z",
            },
        )
        self.assertEqual(notification_update.status_code, 200)
        notification_update_body = notification_update.json()
        self.assertEqual(
            notification_update_body["kind"],
            "advisor_action_queue_task_escalation_notification_update",
        )
        self.assertEqual(notification_update_body["data"]["notification"]["status"], "failed")
        self.assertEqual(
            notification_update_body["data"]["notification"]["delivery_notes"],
            "SMTP provider rejected the message.",
        )
        self.assertEqual(
            notification_update_body["data"]["notification"]["delivered_at"],
            "2026-05-28T10:30:00Z",
        )
        self.assertIsNotNone(notification_update_body["data"]["notification"]["updated_at"])

        failed_notification_logs = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/logs"
            "?owner_id=customer-a&channel=email&status=failed"
        )
        self.assertEqual(failed_notification_logs.status_code, 200)
        self.assertEqual(failed_notification_logs.json()["metadata"]["result_count"], 1)
        self.assertEqual(failed_notification_logs.json()["data"][0]["notification_id"], notification_id)
        self.assertEqual(
            failed_notification_logs.json()["data"][0]["delivery_notes"],
            "SMTP provider rejected the message.",
        )

        prepared_notification_log = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification"
            "?owner_id=customer-a&as_of=2026-05-28&severity=critical"
            "&inbox_status=snooze_expired&assigned_to=advisor-b&focus=telecom",
            json={
                "channel": "email",
                "recipient": "manager-a",
                "status": "prepared",
                "idempotency_key": "manual-prepared-escalation-notification",
            },
        )
        self.assertEqual(prepared_notification_log.status_code, 200)
        prepared_notification_id = prepared_notification_log.json()["data"]["notification"]["notification_id"]
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            UPDATE advisor_action_queue_escalation_notifications
            SET created_at = ?, updated_at = ?
            WHERE notification_id = ?
            """,
            ("2026-05-20T09:00:00Z", "2026-05-20T09:00:00Z", prepared_notification_id),
        )
        conn.commit()
        conn.close()

        notification_summary = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/summary"
            "?owner_id=customer-a&channel=email&recent_limit=3"
        )
        self.assertEqual(notification_summary.status_code, 200)
        notification_summary_body = notification_summary.json()
        self.assertEqual(
            notification_summary_body["kind"],
            "advisor_action_queue_task_escalation_notification_delivery_summary",
        )
        self.assertEqual(notification_summary_body["data"]["summary"]["total_count"], 2)
        self.assertEqual(notification_summary_body["data"]["summary"]["failed_count"], 1)
        self.assertEqual(notification_summary_body["data"]["summary"]["prepared_count"], 1)
        self.assertEqual(notification_summary_body["data"]["summary"]["undelivered_count"], 2)
        self.assertEqual(notification_summary_body["data"]["summary"]["stale_prepared_count"], 1)
        self.assertEqual(notification_summary_body["data"]["status_counts"]["sent"], 0)
        self.assertEqual(notification_summary_body["data"]["status_counts"]["failed"], 1)
        self.assertEqual(notification_summary_body["data"]["status_counts"]["prepared"], 1)
        self.assertEqual(len(notification_summary_body["data"]["recent_failures"]), 1)
        self.assertEqual(
            notification_summary_body["data"]["recent_failures"][0]["notification_id"],
            notification_id,
        )
        self.assertEqual(notification_summary_body["metadata"]["recent_failure_count"], 1)

        notification_delivery_queue = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-queue"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1&limit=5"
        )
        self.assertEqual(notification_delivery_queue.status_code, 200)
        delivery_queue_body = notification_delivery_queue.json()
        self.assertEqual(
            delivery_queue_body["kind"],
            "advisor_action_queue_task_escalation_notification_delivery_queue",
        )
        self.assertEqual(delivery_queue_body["metadata"]["result_count"], 2)
        self.assertEqual(delivery_queue_body["data"][0]["notification_id"], notification_id)
        self.assertEqual(delivery_queue_body["data"][0]["delivery_action"], "retry_failed")
        self.assertEqual(delivery_queue_body["data"][0]["priority"], "high")
        self.assertEqual(delivery_queue_body["data"][0]["payload"]["summary"]["item_count"], 1)
        self.assertEqual(delivery_queue_body["data"][1]["notification_id"], prepared_notification_id)
        self.assertEqual(delivery_queue_body["data"][1]["delivery_action"], "send_prepared")
        self.assertEqual(delivery_queue_body["data"][1]["priority"], "medium")

        notification_delivery_claim = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-claim"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1",
            json={"claimed_by": "worker-a", "lease_seconds": 120},
        )
        self.assertEqual(notification_delivery_claim.status_code, 200)
        delivery_claim_body = notification_delivery_claim.json()
        self.assertEqual(
            delivery_claim_body["kind"],
            "advisor_action_queue_task_escalation_notification_delivery_claim",
        )
        self.assertTrue(delivery_claim_body["metadata"]["claimed"])
        self.assertEqual(delivery_claim_body["metadata"]["notification_id"], notification_id)
        self.assertEqual(delivery_claim_body["metadata"]["claimed_by"], "worker-a")
        self.assertIsNotNone(delivery_claim_body["metadata"]["claim_token"])
        delivery_claim_token = delivery_claim_body["metadata"]["claim_token"]
        self.assertEqual(delivery_claim_body["data"]["claim"]["notification_id"], notification_id)
        self.assertEqual(delivery_claim_body["data"]["claim"]["delivery_claimed_by"], "worker-a")
        self.assertEqual(delivery_claim_body["data"]["claim"]["delivery_attempt_count"], 1)
        self.assertIsNotNone(delivery_claim_body["data"]["claim"]["delivery_claim_token"])
        self.assertEqual(delivery_claim_body["data"]["claim"]["payload"]["summary"]["item_count"], 1)

        invalid_delivery_claim_renewal = self.client.post(
            f"/agents/action-queues/tasks/escalations/inbox/notification/delivery-claim/{notification_id}/renew"
            "?owner_id=customer-a",
            json={"claim_token": "wrong-token", "lease_seconds": 240},
        )
        self.assertEqual(invalid_delivery_claim_renewal.status_code, 400)

        invalid_delivery_claim_renewal_lease = self.client.post(
            f"/agents/action-queues/tasks/escalations/inbox/notification/delivery-claim/{notification_id}/renew"
            "?owner_id=customer-a",
            json={"claim_token": delivery_claim_token, "lease_seconds": 0},
        )
        self.assertEqual(invalid_delivery_claim_renewal_lease.status_code, 400)

        wrong_owner_delivery_claim_renewal = self.client.post(
            f"/agents/action-queues/tasks/escalations/inbox/notification/delivery-claim/{notification_id}/renew"
            "?owner_id=customer-b",
            json={"claim_token": delivery_claim_token, "lease_seconds": 240},
        )
        self.assertEqual(wrong_owner_delivery_claim_renewal.status_code, 404)

        delivery_claim_renewal = self.client.post(
            f"/agents/action-queues/tasks/escalations/inbox/notification/delivery-claim/{notification_id}/renew"
            "?owner_id=customer-a",
            json={"claim_token": delivery_claim_token, "lease_seconds": 240},
        )
        self.assertEqual(delivery_claim_renewal.status_code, 200)
        delivery_claim_renewal_body = delivery_claim_renewal.json()
        self.assertEqual(
            delivery_claim_renewal_body["kind"],
            "advisor_action_queue_task_escalation_notification_delivery_claim_renewal",
        )
        self.assertTrue(delivery_claim_renewal_body["metadata"]["renewed"])
        self.assertEqual(delivery_claim_renewal_body["metadata"]["notification_id"], notification_id)
        self.assertEqual(delivery_claim_renewal_body["metadata"]["claim_token"], delivery_claim_token)
        self.assertEqual(delivery_claim_renewal_body["metadata"]["lease_seconds"], 240)
        self.assertNotEqual(
            delivery_claim_renewal_body["metadata"]["claim_expires_at"],
            delivery_claim_body["metadata"]["claim_expires_at"],
        )
        self.assertEqual(delivery_claim_renewal_body["data"]["claim"]["notification_id"], notification_id)
        self.assertEqual(delivery_claim_renewal_body["data"]["claim"]["delivery_claim_token"], delivery_claim_token)
        self.assertEqual(
            delivery_claim_renewal_body["data"]["claim"]["delivery_claimed_until"],
            delivery_claim_renewal_body["metadata"]["claim_expires_at"],
        )

        delivery_claims = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-claims"
            "?owner_id=customer-a&channel=email&claimed_by=worker-a"
            "&lease_state=active&expiring_within_seconds=60&limit=5"
        )
        self.assertEqual(delivery_claims.status_code, 200)
        delivery_claims_body = delivery_claims.json()
        self.assertEqual(
            delivery_claims_body["kind"],
            "advisor_action_queue_task_escalation_notification_delivery_claims",
        )
        self.assertEqual(delivery_claims_body["metadata"]["result_count"], 1)
        self.assertEqual(delivery_claims_body["metadata"]["lease_state"], "active")
        self.assertEqual(delivery_claims_body["data"][0]["notification_id"], notification_id)
        self.assertEqual(delivery_claims_body["data"][0]["delivery_claimed_by"], "worker-a")
        self.assertEqual(delivery_claims_body["data"][0]["lease_state"], "active")
        self.assertEqual(delivery_claims_body["data"][0]["delivery_action"], "monitor_claim")
        self.assertEqual(delivery_claims_body["data"][0]["priority"], "medium")
        self.assertGreater(delivery_claims_body["data"][0]["lease_seconds_remaining"], 0)
        self.assertEqual(delivery_claims_body["data"][0]["lease_seconds_overdue"], 0)
        self.assertEqual(delivery_claims_body["data"][0]["payload"]["summary"]["item_count"], 1)

        empty_delivery_claims = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-claims"
            "?owner_id=customer-a&claimed_by=worker-b&limit=5"
        )
        self.assertEqual(empty_delivery_claims.status_code, 200)
        self.assertEqual(empty_delivery_claims.json()["metadata"]["result_count"], 0)

        invalid_delivery_claims = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-claims"
            "?owner_id=customer-a&lease_state=missing"
        )
        self.assertEqual(invalid_delivery_claims.status_code, 400)

        invalid_delivery_completion = self.client.post(
            f"/agents/action-queues/tasks/escalations/inbox/notification/delivery-claim/{notification_id}/complete"
            "?owner_id=customer-a",
            json={"claim_token": "wrong-token", "status": "sent"},
        )
        self.assertEqual(invalid_delivery_completion.status_code, 400)

        post_claim_delivery_queue = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-queue"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1&limit=5"
        )
        self.assertEqual(post_claim_delivery_queue.status_code, 200)
        post_claim_delivery_queue_body = post_claim_delivery_queue.json()
        self.assertEqual(post_claim_delivery_queue_body["metadata"]["result_count"], 1)
        self.assertEqual(
            post_claim_delivery_queue_body["data"][0]["notification_id"],
            prepared_notification_id,
        )

        delivery_completion = self.client.post(
            f"/agents/action-queues/tasks/escalations/inbox/notification/delivery-claim/{notification_id}/complete"
            "?owner_id=customer-a",
            json={
                "claim_token": delivery_claim_token,
                "status": "sent",
                "delivery_notes": "Delivered by worker-a.",
                "delivered_at": "2026-05-28T10:35:00Z",
            },
        )
        self.assertEqual(delivery_completion.status_code, 200)
        delivery_completion_body = delivery_completion.json()
        self.assertEqual(
            delivery_completion_body["kind"],
            "advisor_action_queue_task_escalation_notification_delivery_completion",
        )
        self.assertEqual(delivery_completion_body["data"]["notification"]["status"], "sent")
        self.assertEqual(
            delivery_completion_body["data"]["notification"]["delivery_notes"],
            "Delivered by worker-a.",
        )
        self.assertEqual(
            delivery_completion_body["data"]["notification"]["delivered_at"],
            "2026-05-28T10:35:00Z",
        )
        self.assertIsNone(delivery_completion_body["data"]["notification"]["delivery_claimed_by"])
        self.assertIsNone(delivery_completion_body["data"]["notification"]["delivery_claim_token"])
        self.assertEqual(delivery_completion_body["data"]["notification"]["delivery_attempt_count"], 1)

        delivery_attempts = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-attempts"
            f"?owner_id=customer-a&notification_id={notification_id}&status=sent"
        )
        self.assertEqual(delivery_attempts.status_code, 200)
        delivery_attempts_body = delivery_attempts.json()
        self.assertEqual(
            delivery_attempts_body["kind"],
            "advisor_action_queue_task_escalation_notification_delivery_attempts",
        )
        self.assertEqual(delivery_attempts_body["metadata"]["result_count"], 1)
        self.assertEqual(delivery_attempts_body["data"][0]["notification_id"], notification_id)
        self.assertEqual(delivery_attempts_body["data"][0]["status"], "sent")
        self.assertEqual(delivery_attempts_body["data"][0]["claim_token"], delivery_claim_token)
        self.assertEqual(delivery_attempts_body["data"][0]["claimed_by"], "worker-a")
        self.assertEqual(delivery_attempts_body["data"][0]["attempt_number"], 1)
        self.assertEqual(delivery_attempts_body["data"][0]["delivery_notes"], "Delivered by worker-a.")
        self.assertEqual(delivery_attempts_body["data"][0]["delivered_at"], "2026-05-28T10:35:00Z")
        self.assertIsNotNone(delivery_attempts_body["data"][0]["completed_at"])

        invalid_delivery_attempts = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-attempts?status=prepared"
        )
        self.assertEqual(invalid_delivery_attempts.status_code, 400)

        release_notification_log = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification"
            "?owner_id=customer-a&as_of=2026-05-28&severity=critical"
            "&inbox_status=snooze_expired&assigned_to=advisor-b&focus=telecom",
            json={
                "channel": "email",
                "recipient": "manager-a",
                "status": "failed",
                "idempotency_key": "manual-release-escalation-notification",
            },
        )
        self.assertEqual(release_notification_log.status_code, 200)
        release_notification_id = release_notification_log.json()["data"]["notification"]["notification_id"]
        release_delivery_claim = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-claim"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1",
            json={"claimed_by": "worker-release", "lease_seconds": 120},
        )
        self.assertEqual(release_delivery_claim.status_code, 200)
        release_delivery_claim_body = release_delivery_claim.json()
        self.assertTrue(release_delivery_claim_body["metadata"]["claimed"])
        self.assertEqual(release_delivery_claim_body["metadata"]["notification_id"], release_notification_id)
        release_claim_token = release_delivery_claim_body["metadata"]["claim_token"]

        invalid_delivery_claim_release = self.client.post(
            f"/agents/action-queues/tasks/escalations/inbox/notification/delivery-claim/{release_notification_id}/release"
            "?owner_id=customer-a",
            json={"claim_token": "wrong-token", "release_notes": "Do not release."},
        )
        self.assertEqual(invalid_delivery_claim_release.status_code, 400)

        delivery_claim_release = self.client.post(
            f"/agents/action-queues/tasks/escalations/inbox/notification/delivery-claim/{release_notification_id}/release"
            "?owner_id=customer-a",
            json={
                "claim_token": release_claim_token,
                "release_notes": "Worker shutting down before provider send.",
                "released_by": "ops-a",
            },
        )
        self.assertEqual(delivery_claim_release.status_code, 200)
        delivery_claim_release_body = delivery_claim_release.json()
        self.assertEqual(
            delivery_claim_release_body["kind"],
            "advisor_action_queue_task_escalation_notification_delivery_claim_release",
        )
        self.assertTrue(delivery_claim_release_body["metadata"]["released"])
        self.assertGreater(delivery_claim_release_body["metadata"]["release_id"], 0)
        self.assertEqual(delivery_claim_release_body["metadata"]["released_by"], "ops-a")
        self.assertEqual(delivery_claim_release_body["metadata"]["claim_token"], release_claim_token)
        self.assertEqual(delivery_claim_release_body["data"]["notification"]["notification_id"], release_notification_id)
        self.assertIsNone(delivery_claim_release_body["data"]["notification"]["delivery_claimed_by"])
        self.assertIsNone(delivery_claim_release_body["data"]["notification"]["delivery_claim_token"])
        self.assertEqual(
            delivery_claim_release_body["data"]["notification"]["delivery_notes"],
            "Worker shutting down before provider send.",
        )
        self.assertEqual(delivery_claim_release_body["data"]["notification"]["delivery_attempt_count"], 1)
        self.assertEqual(
            delivery_claim_release_body["data"]["release"]["release_id"],
            delivery_claim_release_body["metadata"]["release_id"],
        )
        self.assertEqual(delivery_claim_release_body["data"]["release"]["notification_id"], release_notification_id)
        self.assertEqual(delivery_claim_release_body["data"]["release"]["claim_token"], release_claim_token)
        self.assertEqual(delivery_claim_release_body["data"]["release"]["claimed_by"], "worker-release")
        self.assertEqual(delivery_claim_release_body["data"]["release"]["released_by"], "ops-a")
        self.assertEqual(
            delivery_claim_release_body["data"]["release"]["release_notes"],
            "Worker shutting down before provider send.",
        )
        self.assertEqual(delivery_claim_release_body["data"]["release"]["previous_delivery_attempt_count"], 1)

        delivery_claim_releases = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-claim-releases"
            f"?owner_id=customer-a&notification_id={release_notification_id}&released_by=ops-a&limit=5"
        )
        self.assertEqual(delivery_claim_releases.status_code, 200)
        delivery_claim_releases_body = delivery_claim_releases.json()
        self.assertEqual(
            delivery_claim_releases_body["kind"],
            "advisor_action_queue_task_escalation_notification_delivery_claim_releases",
        )
        self.assertEqual(delivery_claim_releases_body["metadata"]["result_count"], 1)
        self.assertEqual(
            delivery_claim_releases_body["data"][0]["release_id"],
            delivery_claim_release_body["metadata"]["release_id"],
        )
        self.assertEqual(delivery_claim_releases_body["data"][0]["released_by"], "ops-a")

        released_delivery_attempts = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-attempts"
            f"?owner_id=customer-a&notification_id={release_notification_id}"
        )
        self.assertEqual(released_delivery_attempts.status_code, 200)
        self.assertEqual(released_delivery_attempts.json()["metadata"]["result_count"], 0)

        released_delivery_claims = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-claims"
            "?owner_id=customer-a&claimed_by=worker-release&limit=5"
        )
        self.assertEqual(released_delivery_claims.status_code, 200)
        self.assertEqual(released_delivery_claims.json()["metadata"]["result_count"], 0)

        cleanup_release_claim = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-claim"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1",
            json={"claimed_by": "worker-release-cleanup", "lease_seconds": 120},
        )
        self.assertEqual(cleanup_release_claim.status_code, 200)
        cleanup_release_claim_body = cleanup_release_claim.json()
        self.assertEqual(cleanup_release_claim_body["metadata"]["notification_id"], release_notification_id)
        cleanup_release_completion = self.client.post(
            f"/agents/action-queues/tasks/escalations/inbox/notification/delivery-claim/{release_notification_id}/complete"
            "?owner_id=customer-a",
            json={
                "claim_token": cleanup_release_claim_body["metadata"]["claim_token"],
                "status": "skipped",
                "delivery_notes": "Release path verified without delivery.",
            },
        )
        self.assertEqual(cleanup_release_completion.status_code, 200)

        delivery_control_summary = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-control-summary"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1&expiring_within_seconds=60"
        )
        self.assertEqual(delivery_control_summary.status_code, 200)
        delivery_control_summary_body = delivery_control_summary.json()
        self.assertEqual(
            delivery_control_summary_body["kind"],
            "advisor_action_queue_task_escalation_notification_delivery_control_summary",
        )
        self.assertEqual(delivery_control_summary_body["data"]["summary"]["queued_count"], 1)
        self.assertEqual(delivery_control_summary_body["data"]["summary"]["active_claim_count"], 0)
        self.assertEqual(delivery_control_summary_body["data"]["summary"]["expiring_claim_count"], 0)
        self.assertEqual(delivery_control_summary_body["data"]["summary"]["expired_claim_count"], 0)
        self.assertEqual(delivery_control_summary_body["data"]["summary"]["retry_wait_count"], 0)
        self.assertEqual(delivery_control_summary_body["data"]["summary"]["deadletter_count"], 0)
        self.assertEqual(delivery_control_summary_body["data"]["summary"]["delivery_attempt_count"], 2)
        self.assertEqual(delivery_control_summary_body["data"]["summary"]["failed_attempt_count"], 0)
        self.assertEqual(delivery_control_summary_body["data"]["summary"]["claim_release_count"], 1)
        self.assertEqual(delivery_control_summary_body["data"]["summary"]["deadletter_remediation_count"], 0)
        self.assertEqual(delivery_control_summary_body["data"]["summary"]["incident_count"], 0)
        self.assertEqual(delivery_control_summary_body["data"]["worker_claim_counts"], [])
        self.assertEqual(delivery_control_summary_body["metadata"]["channel"], "email")

        delivery_incidents = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1&expiring_within_seconds=60&limit=5"
        )
        self.assertEqual(delivery_incidents.status_code, 200)
        delivery_incidents_body = delivery_incidents.json()
        self.assertEqual(
            delivery_incidents_body["kind"],
            "advisor_action_queue_task_escalation_notification_delivery_incidents",
        )
        self.assertEqual(delivery_incidents_body["metadata"]["result_count"], 1)
        self.assertEqual(delivery_incidents_body["metadata"]["latest_review_count"], 0)
        self.assertFalse(delivery_incidents_body["metadata"]["include_suppressed"])
        self.assertEqual(delivery_incidents_body["metadata"]["suppressed_count"], 0)
        self.assertEqual(delivery_incidents_body["data"][0]["notification_id"], prepared_notification_id)
        self.assertEqual(delivery_incidents_body["data"][0]["incident_type"], "stale_prepared")
        self.assertEqual(delivery_incidents_body["data"][0]["incident_reason"], "prepared_delivery_stale")
        self.assertEqual(delivery_incidents_body["data"][0]["delivery_action"], "send_prepared")
        self.assertEqual(delivery_incidents_body["data"][0]["priority"], "medium")
        self.assertFalse(delivery_incidents_body["data"][0]["is_suppressed"])
        self.assertIsNone(delivery_incidents_body["data"][0]["suppression_reason"])
        self.assertIsNone(delivery_incidents_body["data"][0]["latest_review"])
        self.assertEqual(delivery_incidents_body["data"][0]["payload"]["summary"]["item_count"], 1)

        delivery_incident_summary = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incident-summary"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1&expiring_within_seconds=60&max_incidents=5"
        )
        self.assertEqual(delivery_incident_summary.status_code, 200)
        delivery_incident_summary_body = delivery_incident_summary.json()
        self.assertEqual(
            delivery_incident_summary_body["kind"],
            "advisor_action_queue_task_escalation_notification_delivery_incident_summary",
        )
        self.assertEqual(delivery_incident_summary_body["data"]["summary"]["total_count"], 1)
        self.assertEqual(delivery_incident_summary_body["data"]["summary"]["actionable_count"], 1)
        self.assertEqual(delivery_incident_summary_body["data"]["summary"]["resolved_count"], 0)
        self.assertEqual(delivery_incident_summary_body["data"]["summary"]["snoozed_count"], 0)
        self.assertEqual(delivery_incident_summary_body["data"]["summary"]["unreviewed_count"], 1)
        self.assertEqual(
            delivery_incident_summary_body["data"]["by_incident_type"][0]["incident_type"],
            "stale_prepared",
        )
        self.assertEqual(
            delivery_incident_summary_body["data"]["by_incident_type"][0]["actionable_count"],
            1,
        )
        self.assertEqual(delivery_incident_summary_body["data"]["by_priority"][0]["priority"], "medium")
        self.assertEqual(
            delivery_incident_summary_body["data"]["by_latest_review_status"][0]["incident_status"],
            "unreviewed",
        )
        self.assertEqual(delivery_incident_summary_body["metadata"]["scanned_incident_count"], 1)
        self.assertEqual(delivery_incident_summary_body["metadata"]["suppressed_count"], 0)

        invalid_incident_review = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents/"
            f"{prepared_notification_id}/review?owner_id=customer-a",
            json={"incident_type": "stale_prepared", "incident_status": "assigned"},
        )
        self.assertEqual(invalid_incident_review.status_code, 400)

        wrong_owner_incident_review = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents/"
            f"{prepared_notification_id}/review?owner_id=customer-b",
            json={
                "incident_type": "stale_prepared",
                "incident_status": "acknowledged",
                "reviewer": "ops-a",
            },
        )
        self.assertEqual(wrong_owner_incident_review.status_code, 404)

        delivery_incident_review = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents/"
            f"{prepared_notification_id}/review?owner_id=customer-a",
            json={
                "incident_type": "stale_prepared",
                "incident_status": "assigned",
                "reviewer": "ops-a",
                "assigned_to": "worker-b",
                "notes": "Assign stale prepared email to worker-b.",
                "follow_up_at": "2026-05-28T11:00:00Z",
            },
        )
        self.assertEqual(delivery_incident_review.status_code, 200)
        delivery_incident_review_body = delivery_incident_review.json()
        self.assertEqual(
            delivery_incident_review_body["kind"],
            "advisor_action_queue_task_escalation_notification_delivery_incident_review",
        )
        self.assertEqual(
            delivery_incident_review_body["data"]["review"]["notification_id"],
            prepared_notification_id,
        )
        self.assertEqual(delivery_incident_review_body["data"]["review"]["incident_type"], "stale_prepared")
        self.assertEqual(delivery_incident_review_body["data"]["review"]["incident_status"], "assigned")
        self.assertEqual(delivery_incident_review_body["data"]["review"]["reviewer"], "ops-a")
        self.assertEqual(delivery_incident_review_body["data"]["review"]["assigned_to"], "worker-b")
        self.assertEqual(
            delivery_incident_review_body["data"]["review"]["notes"],
            "Assign stale prepared email to worker-b.",
        )
        self.assertEqual(
            delivery_incident_review_body["data"]["review"]["follow_up_at"],
            "2026-05-28T11:00:00Z",
        )
        self.assertEqual(
            delivery_incident_review_body["data"]["notification"]["notification_id"],
            prepared_notification_id,
        )

        delivery_incident_reviews = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incident-reviews"
            f"?owner_id=customer-a&notification_id={prepared_notification_id}"
            "&incident_type=stale_prepared&incident_status=assigned&limit=5"
        )
        self.assertEqual(delivery_incident_reviews.status_code, 200)
        delivery_incident_reviews_body = delivery_incident_reviews.json()
        self.assertEqual(
            delivery_incident_reviews_body["kind"],
            "advisor_action_queue_task_escalation_notification_delivery_incident_reviews",
        )
        self.assertEqual(delivery_incident_reviews_body["metadata"]["result_count"], 1)
        self.assertEqual(
            delivery_incident_reviews_body["data"][0]["incident_review_id"],
            delivery_incident_review_body["data"]["review"]["incident_review_id"],
        )
        self.assertEqual(delivery_incident_reviews_body["data"][0]["assigned_to"], "worker-b")

        reviewed_delivery_incidents = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1&expiring_within_seconds=60&limit=5"
        )
        self.assertEqual(reviewed_delivery_incidents.status_code, 200)
        reviewed_delivery_incidents_body = reviewed_delivery_incidents.json()
        self.assertEqual(reviewed_delivery_incidents_body["metadata"]["result_count"], 1)
        self.assertEqual(reviewed_delivery_incidents_body["metadata"]["latest_review_count"], 1)
        self.assertEqual(reviewed_delivery_incidents_body["metadata"]["suppressed_count"], 0)
        self.assertFalse(reviewed_delivery_incidents_body["data"][0]["is_suppressed"])
        self.assertIsNone(reviewed_delivery_incidents_body["data"][0]["suppression_reason"])
        self.assertEqual(
            reviewed_delivery_incidents_body["data"][0]["latest_review"]["incident_review_id"],
            delivery_incident_review_body["data"]["review"]["incident_review_id"],
        )
        self.assertEqual(
            reviewed_delivery_incidents_body["data"][0]["latest_review"]["incident_status"],
            "assigned",
        )
        self.assertEqual(reviewed_delivery_incidents_body["data"][0]["latest_review"]["assigned_to"], "worker-b")
        self.assertEqual(reviewed_delivery_incidents_body["data"][0]["latest_review"]["reviewer"], "ops-a")

        snoozed_incident_review = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents/"
            f"{prepared_notification_id}/review?owner_id=customer-a",
            json={
                "incident_type": "stale_prepared",
                "incident_status": "snoozed",
                "reviewer": "ops-a",
                "notes": "Wait for provider window.",
                "follow_up_at": "2099-01-01T00:00:00Z",
            },
        )
        self.assertEqual(snoozed_incident_review.status_code, 200)
        default_snoozed_delivery_incidents = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1&expiring_within_seconds=60&limit=5"
        )
        self.assertEqual(default_snoozed_delivery_incidents.status_code, 200)
        self.assertEqual(default_snoozed_delivery_incidents.json()["metadata"]["result_count"], 0)
        self.assertEqual(default_snoozed_delivery_incidents.json()["metadata"]["suppressed_count"], 1)

        visible_snoozed_delivery_incidents = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1&expiring_within_seconds=60&limit=5"
            "&include_suppressed=true"
        )
        self.assertEqual(visible_snoozed_delivery_incidents.status_code, 200)
        visible_snoozed_delivery_incidents_body = visible_snoozed_delivery_incidents.json()
        self.assertTrue(visible_snoozed_delivery_incidents_body["metadata"]["include_suppressed"])
        self.assertEqual(visible_snoozed_delivery_incidents_body["metadata"]["result_count"], 1)
        self.assertEqual(visible_snoozed_delivery_incidents_body["metadata"]["suppressed_count"], 1)
        self.assertTrue(visible_snoozed_delivery_incidents_body["data"][0]["is_suppressed"])
        self.assertEqual(
            visible_snoozed_delivery_incidents_body["data"][0]["suppression_reason"],
            "snoozed_until_follow_up",
        )
        self.assertEqual(
            visible_snoozed_delivery_incidents_body["data"][0]["latest_review"]["incident_status"],
            "snoozed",
        )

        snoozed_delivery_incident_summary = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incident-summary"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1&expiring_within_seconds=60&max_incidents=5"
        )
        self.assertEqual(snoozed_delivery_incident_summary.status_code, 200)
        snoozed_delivery_incident_summary_body = snoozed_delivery_incident_summary.json()
        self.assertEqual(snoozed_delivery_incident_summary_body["data"]["summary"]["total_count"], 1)
        self.assertEqual(snoozed_delivery_incident_summary_body["data"]["summary"]["actionable_count"], 0)
        self.assertEqual(snoozed_delivery_incident_summary_body["data"]["summary"]["snoozed_count"], 1)
        self.assertEqual(snoozed_delivery_incident_summary_body["data"]["summary"]["resolved_count"], 0)
        self.assertEqual(
            snoozed_delivery_incident_summary_body["data"]["by_latest_review_status"][0]["incident_status"],
            "snoozed",
        )
        self.assertEqual(
            snoozed_delivery_incident_summary_body["data"]["by_latest_review_status"][0]["snoozed_count"],
            1,
        )

        resolved_incident_review = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents/"
            f"{prepared_notification_id}/review?owner_id=customer-a",
            json={
                "incident_type": "stale_prepared",
                "incident_status": "resolved",
                "reviewer": "ops-a",
                "notes": "Resolved before delivery worker pickup.",
            },
        )
        self.assertEqual(resolved_incident_review.status_code, 200)
        default_resolved_delivery_incidents = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1&expiring_within_seconds=60&limit=5"
        )
        self.assertEqual(default_resolved_delivery_incidents.status_code, 200)
        self.assertEqual(default_resolved_delivery_incidents.json()["metadata"]["result_count"], 0)
        self.assertEqual(default_resolved_delivery_incidents.json()["metadata"]["suppressed_count"], 1)

        visible_resolved_delivery_incidents = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1&expiring_within_seconds=60&limit=5"
            "&include_suppressed=true"
        )
        self.assertEqual(visible_resolved_delivery_incidents.status_code, 200)
        visible_resolved_delivery_incidents_body = visible_resolved_delivery_incidents.json()
        self.assertEqual(visible_resolved_delivery_incidents_body["metadata"]["result_count"], 1)
        self.assertEqual(visible_resolved_delivery_incidents_body["metadata"]["suppressed_count"], 1)
        self.assertTrue(visible_resolved_delivery_incidents_body["data"][0]["is_suppressed"])
        self.assertEqual(
            visible_resolved_delivery_incidents_body["data"][0]["suppression_reason"],
            "resolved",
        )
        self.assertEqual(
            visible_resolved_delivery_incidents_body["data"][0]["latest_review"]["incident_status"],
            "resolved",
        )

        resolved_delivery_incident_summary = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incident-summary"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1&expiring_within_seconds=60&max_incidents=5"
        )
        self.assertEqual(resolved_delivery_incident_summary.status_code, 200)
        resolved_delivery_incident_summary_body = resolved_delivery_incident_summary.json()
        self.assertEqual(resolved_delivery_incident_summary_body["data"]["summary"]["total_count"], 1)
        self.assertEqual(resolved_delivery_incident_summary_body["data"]["summary"]["actionable_count"], 0)
        self.assertEqual(resolved_delivery_incident_summary_body["data"]["summary"]["snoozed_count"], 0)
        self.assertEqual(resolved_delivery_incident_summary_body["data"]["summary"]["resolved_count"], 1)
        self.assertEqual(
            resolved_delivery_incident_summary_body["data"]["by_incident_type"][0]["resolved_count"],
            1,
        )
        self.assertEqual(
            resolved_delivery_incident_summary_body["data"]["by_latest_review_status"][0]["incident_status"],
            "resolved",
        )

        prepared_delivery_claim = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-claim"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1",
            json={"claimed_by": "worker-b", "lease_seconds": 120},
        )
        self.assertEqual(prepared_delivery_claim.status_code, 200)
        prepared_delivery_claim_body = prepared_delivery_claim.json()
        self.assertTrue(prepared_delivery_claim_body["metadata"]["claimed"])
        self.assertEqual(
            prepared_delivery_claim_body["metadata"]["notification_id"],
            prepared_notification_id,
        )
        prepared_delivery_claim_token = prepared_delivery_claim_body["metadata"]["claim_token"]
        prepared_failure_completion = self.client.post(
            f"/agents/action-queues/tasks/escalations/inbox/notification/delivery-claim/{prepared_notification_id}/complete"
            "?owner_id=customer-a",
            json={
                "claim_token": prepared_delivery_claim_token,
                "status": "failed",
                "delivery_notes": "Email provider rate limit.",
                "retry_after": "2099-01-01T00:00:00Z",
            },
        )
        self.assertEqual(prepared_failure_completion.status_code, 200)
        prepared_failure_completion_body = prepared_failure_completion.json()
        self.assertEqual(prepared_failure_completion_body["data"]["notification"]["status"], "failed")
        self.assertEqual(
            prepared_failure_completion_body["data"]["notification"]["delivery_retry_after"],
            "2099-01-01T00:00:00Z",
        )
        self.assertEqual(
            prepared_failure_completion_body["metadata"]["retry_after"],
            "2099-01-01T00:00:00Z",
        )

        failed_delivery_attempts = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-attempts"
            f"?owner_id=customer-a&notification_id={prepared_notification_id}&status=failed"
        )
        self.assertEqual(failed_delivery_attempts.status_code, 200)
        self.assertEqual(failed_delivery_attempts.json()["metadata"]["result_count"], 1)
        self.assertEqual(
            failed_delivery_attempts.json()["data"][0]["retry_after"],
            "2099-01-01T00:00:00Z",
        )

        backoff_delivery_queue = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-queue"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1&limit=5"
        )
        self.assertEqual(backoff_delivery_queue.status_code, 200)
        self.assertEqual(backoff_delivery_queue.json()["metadata"]["result_count"], 0)

        backoff_delivery_claim = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-claim"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1",
            json={"claimed_by": "worker-c", "lease_seconds": 120},
        )
        self.assertEqual(backoff_delivery_claim.status_code, 200)
        self.assertFalse(backoff_delivery_claim.json()["metadata"]["claimed"])

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            UPDATE advisor_action_queue_escalation_notifications
            SET delivery_retry_after = ?
            WHERE notification_id = ?
            """,
            ("2026-05-20T00:00:00Z", prepared_notification_id),
        )
        conn.commit()
        conn.close()

        exhaustion_delivery_claim = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-claim"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1",
            json={"claimed_by": "worker-c", "lease_seconds": 120},
        )
        self.assertEqual(exhaustion_delivery_claim.status_code, 200)
        exhaustion_delivery_claim_body = exhaustion_delivery_claim.json()
        self.assertTrue(exhaustion_delivery_claim_body["metadata"]["claimed"])
        self.assertEqual(
            exhaustion_delivery_claim_body["metadata"]["notification_id"],
            prepared_notification_id,
        )
        self.assertEqual(exhaustion_delivery_claim_body["data"]["claim"]["delivery_attempt_count"], 2)
        exhaustion_delivery_claim_token = exhaustion_delivery_claim_body["metadata"]["claim_token"]

        exhaustion_completion = self.client.post(
            f"/agents/action-queues/tasks/escalations/inbox/notification/delivery-claim/{prepared_notification_id}/complete"
            "?owner_id=customer-a",
            json={
                "claim_token": exhaustion_delivery_claim_token,
                "status": "failed",
                "delivery_notes": "Permanent provider failure.",
                "max_attempts": 2,
            },
        )
        self.assertEqual(exhaustion_completion.status_code, 200)
        exhaustion_completion_body = exhaustion_completion.json()
        self.assertEqual(exhaustion_completion_body["data"]["notification"]["status"], "failed")
        self.assertIsNone(exhaustion_completion_body["data"]["notification"]["delivery_retry_after"])
        self.assertIsNotNone(exhaustion_completion_body["data"]["notification"]["delivery_exhausted_at"])
        self.assertEqual(
            exhaustion_completion_body["data"]["notification"]["delivery_exhausted_reason"],
            "max_attempts_reached",
        )
        self.assertTrue(exhaustion_completion_body["metadata"]["exhausted"])
        self.assertEqual(exhaustion_completion_body["metadata"]["max_attempts"], 2)

        exhausted_delivery_attempts = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-attempts"
            f"?owner_id=customer-a&notification_id={prepared_notification_id}&status=failed"
        )
        self.assertEqual(exhausted_delivery_attempts.status_code, 200)
        self.assertEqual(exhausted_delivery_attempts.json()["metadata"]["result_count"], 2)
        self.assertEqual(exhausted_delivery_attempts.json()["data"][0]["attempt_number"], 2)
        self.assertIsNotNone(exhausted_delivery_attempts.json()["data"][0]["exhausted_at"])
        self.assertEqual(
            exhausted_delivery_attempts.json()["data"][0]["exhausted_reason"],
            "max_attempts_reached",
        )

        exhausted_delivery_summary = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/summary"
            "?owner_id=customer-a&channel=email"
        )
        self.assertEqual(exhausted_delivery_summary.status_code, 200)
        self.assertEqual(exhausted_delivery_summary.json()["data"]["summary"]["exhausted_count"], 1)
        self.assertEqual(exhausted_delivery_summary.json()["data"]["summary"]["retry_wait_count"], 0)

        exhausted_delivery_queue = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-queue"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1&limit=5"
        )
        self.assertEqual(exhausted_delivery_queue.status_code, 200)
        self.assertEqual(exhausted_delivery_queue.json()["metadata"]["result_count"], 0)

        exhausted_delivery_claim = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-claim"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1",
            json={"claimed_by": "worker-d", "lease_seconds": 120},
        )
        self.assertEqual(exhausted_delivery_claim.status_code, 200)
        self.assertFalse(exhausted_delivery_claim.json()["metadata"]["claimed"])

        deadletter_queue = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/deadletters"
            "?owner_id=customer-a&channel=email&limit=5"
        )
        self.assertEqual(deadletter_queue.status_code, 200)
        deadletter_queue_body = deadletter_queue.json()
        self.assertEqual(
            deadletter_queue_body["kind"],
            "advisor_action_queue_task_escalation_notification_deadletters",
        )
        self.assertEqual(deadletter_queue_body["metadata"]["result_count"], 1)
        self.assertEqual(deadletter_queue_body["data"][0]["notification_id"], prepared_notification_id)
        self.assertEqual(deadletter_queue_body["data"][0]["deadletter_reason"], "max_attempts_reached")
        self.assertEqual(deadletter_queue_body["data"][0]["delivery_action"], "review_deadletter")
        self.assertEqual(deadletter_queue_body["data"][0]["priority"], "critical")
        self.assertEqual(deadletter_queue_body["data"][0]["delivery_attempt_count"], 2)
        self.assertEqual(deadletter_queue_body["data"][0]["payload"]["summary"]["item_count"], 1)

        deadletter_incidents = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1&limit=5"
        )
        self.assertEqual(deadletter_incidents.status_code, 200)
        deadletter_incidents_body = deadletter_incidents.json()
        self.assertEqual(deadletter_incidents_body["metadata"]["result_count"], 1)
        self.assertEqual(deadletter_incidents_body["data"][0]["notification_id"], prepared_notification_id)
        self.assertEqual(deadletter_incidents_body["data"][0]["incident_type"], "deadletter")
        self.assertEqual(deadletter_incidents_body["data"][0]["deadletter_reason"], "max_attempts_reached")
        self.assertEqual(deadletter_incidents_body["data"][0]["delivery_action"], "review_deadletter")
        self.assertEqual(deadletter_incidents_body["data"][0]["priority"], "critical")
        self.assertFalse(deadletter_incidents_body["data"][0]["is_suppressed"])
        self.assertIsNone(deadletter_incidents_body["data"][0]["suppression_reason"])

        deadletter_incident_review = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents/"
            f"{prepared_notification_id}/review?owner_id=customer-a",
            json={
                "incident_type": "deadletter",
                "incident_status": "assigned",
                "reviewer": "ops-a",
                "assigned_to": "worker-deadletter",
                "notes": "Review exhausted delivery before requeue.",
            },
        )
        self.assertEqual(deadletter_incident_review.status_code, 200)

        wrong_owner_incident_detail = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents/"
            f"{prepared_notification_id}?owner_id=customer-b&incident_type=deadletter"
        )
        self.assertEqual(wrong_owner_incident_detail.status_code, 404)

        invalid_incident_detail = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents/"
            f"{prepared_notification_id}?owner_id=customer-a&incident_type=not-real"
        )
        self.assertEqual(invalid_incident_detail.status_code, 400)

        deadletter_incident_detail = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents/"
            f"{prepared_notification_id}?owner_id=customer-a&incident_type=deadletter&audit_limit=5"
        )
        self.assertEqual(deadletter_incident_detail.status_code, 200)
        deadletter_incident_detail_body = deadletter_incident_detail.json()
        self.assertEqual(
            deadletter_incident_detail_body["kind"],
            "advisor_action_queue_task_escalation_notification_delivery_incident_detail",
        )
        self.assertEqual(deadletter_incident_detail_body["metadata"]["notification_id"], prepared_notification_id)
        self.assertEqual(deadletter_incident_detail_body["metadata"]["incident_type"], "deadletter")
        self.assertEqual(deadletter_incident_detail_body["metadata"]["review_count"], 1)
        self.assertEqual(deadletter_incident_detail_body["metadata"]["delivery_attempt_count"], 2)
        self.assertEqual(deadletter_incident_detail_body["metadata"]["deadletter_remediation_count"], 0)
        self.assertEqual(deadletter_incident_detail_body["metadata"]["claim_release_count"], 0)
        self.assertEqual(
            deadletter_incident_detail_body["data"]["incident"]["latest_review"]["assigned_to"],
            "worker-deadletter",
        )
        self.assertEqual(
            deadletter_incident_detail_body["data"]["review_history"][0]["incident_status"],
            "assigned",
        )
        self.assertEqual(
            {attempt["status"] for attempt in deadletter_incident_detail_body["data"]["delivery_attempts"]},
            {"failed"},
        )
        self.assertEqual(deadletter_incident_detail_body["data"]["deadletter_remediations"], [])
        self.assertEqual(deadletter_incident_detail_body["data"]["claim_releases"], [])
        deadletter_timeline = deadletter_incident_detail_body["data"]["timeline"]
        self.assertEqual(
            deadletter_incident_detail_body["metadata"]["timeline_event_count"],
            len(deadletter_timeline),
        )
        self.assertEqual(
            [event["timeline_index"] for event in deadletter_timeline],
            list(range(1, len(deadletter_timeline) + 1)),
        )
        deadletter_event_types = [event["event_type"] for event in deadletter_timeline]
        self.assertIn("notification_created", deadletter_event_types)
        self.assertIn("delivery_attempt_completed", deadletter_event_types)
        self.assertIn("deadlettered", deadletter_event_types)
        self.assertIn("incident_reviewed", deadletter_event_types)
        self.assertTrue(
            any(
                event["event_type"] == "delivery_attempt_completed"
                and event["attempt_number"] == 2
                and event["status"] == "failed"
                for event in deadletter_timeline
            )
        )
        self.assertTrue(
            any(
                event["event_type"] == "deadlettered"
                and event["deadletter_reason"] == "max_attempts_reached"
                for event in deadletter_timeline
            )
        )
        self.assertTrue(
            any(
                event["event_type"] == "incident_reviewed"
                and event["incident_status"] == "assigned"
                and event["assigned_to"] == "worker-deadletter"
                for event in deadletter_timeline
            )
        )
        deadletter_next_actions = deadletter_incident_detail_body["data"]["next_actions"]
        self.assertEqual(
            deadletter_incident_detail_body["metadata"]["next_action_count"],
            len(deadletter_next_actions),
        )
        deadletter_next_action_ids = {action["action_id"] for action in deadletter_next_actions}
        self.assertIn("requeue_deadletter", deadletter_next_action_ids)
        self.assertIn("resolve_incident", deadletter_next_action_ids)
        self.assertNotIn("assign_incident", deadletter_next_action_ids)
        requeue_next_action = next(
            action for action in deadletter_next_actions if action["action_id"] == "requeue_deadletter"
        )
        self.assertEqual(requeue_next_action["priority"], "critical")
        self.assertEqual(requeue_next_action["method"], "POST")
        self.assertEqual(
            requeue_next_action["path"],
            "/agents/action-queues/tasks/escalations/inbox/notification/"
            f"deadletters/{prepared_notification_id}/requeue",
        )
        self.assertEqual(requeue_next_action["query_params"]["owner_id"], "customer-a")
        self.assertEqual(requeue_next_action["request_body_template"]["requeued_by"], "<operator_id>")

        unavailable_incident_action = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents/"
            f"{prepared_notification_id}/actions?owner_id=customer-a",
            json={
                "action_id": "assign_incident",
                "incident_type": "deadletter",
                "assigned_to": "worker-deadletter",
                "reviewer": "ops-a",
            },
        )
        self.assertEqual(unavailable_incident_action.status_code, 400)

        non_exhausted_requeue = self.client.post(
            f"/agents/action-queues/tasks/escalations/inbox/notification/deadletters/{notification_id}/requeue"
            "?owner_id=customer-a",
            json={"delivery_notes": "Should not requeue delivered notifications."},
        )
        self.assertEqual(non_exhausted_requeue.status_code, 400)

        deadletter_requeue = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents/"
            f"{prepared_notification_id}/actions?owner_id=customer-a",
            json={
                "action_id": "requeue_deadletter",
                "incident_type": "deadletter",
                "delivery_notes": "Provider configuration fixed.",
                "requeued_by": "ops-a",
            },
        )
        self.assertEqual(deadletter_requeue.status_code, 200)
        deadletter_requeue_action_body = deadletter_requeue.json()
        self.assertEqual(
            deadletter_requeue_action_body["kind"],
            "advisor_action_queue_task_escalation_notification_delivery_incident_action_execution",
        )
        self.assertEqual(deadletter_requeue_action_body["metadata"]["action_id"], "requeue_deadletter")
        self.assertEqual(
            deadletter_requeue_action_body["metadata"]["result_kind"],
            "advisor_action_queue_task_escalation_notification_deadletter_requeue",
        )
        self.assertFalse(deadletter_requeue_action_body["metadata"]["post_action_incident_active"])
        self.assertTrue(deadletter_requeue_action_body["metadata"]["post_action_incident_cleared"])
        self.assertEqual(deadletter_requeue_action_body["metadata"]["post_action_next_action_count"], 0)
        self.assertEqual(
            deadletter_requeue_action_body["data"]["action"]["path"],
            "/agents/action-queues/tasks/escalations/inbox/notification/"
            f"deadletters/{prepared_notification_id}/requeue",
        )
        post_action_state = deadletter_requeue_action_body["data"]["post_action_state"]
        self.assertFalse(post_action_state["incident_active"])
        self.assertTrue(post_action_state["incident_cleared"])
        self.assertIsNone(post_action_state["incident_detail"])
        self.assertEqual(post_action_state["next_actions"], [])
        self.assertIsNone(post_action_state["notification"]["delivery_exhausted_at"])
        deadletter_requeue_body = deadletter_requeue_action_body["data"]["result"]
        self.assertEqual(
            deadletter_requeue_body["kind"],
            "advisor_action_queue_task_escalation_notification_deadletter_requeue",
        )
        self.assertTrue(deadletter_requeue_body["metadata"]["requeued"])
        self.assertGreater(deadletter_requeue_body["metadata"]["remediation_id"], 0)
        self.assertIsNone(deadletter_requeue_body["metadata"]["retry_after"])
        self.assertEqual(deadletter_requeue_body["metadata"]["requeued_by"], "ops-a")
        self.assertEqual(deadletter_requeue_body["data"]["notification"]["status"], "failed")
        self.assertIsNone(deadletter_requeue_body["data"]["notification"]["delivery_retry_after"])
        self.assertIsNone(deadletter_requeue_body["data"]["notification"]["delivery_exhausted_at"])
        self.assertIsNone(deadletter_requeue_body["data"]["notification"]["delivery_exhausted_reason"])
        self.assertIsNone(deadletter_requeue_body["data"]["notification"]["delivery_claim_token"])
        self.assertEqual(
            deadletter_requeue_body["data"]["notification"]["delivery_notes"],
            "Provider configuration fixed.",
        )
        self.assertEqual(deadletter_requeue_body["data"]["notification"]["delivery_attempt_count"], 2)
        self.assertEqual(
            deadletter_requeue_body["data"]["remediation"]["notification_id"],
            prepared_notification_id,
        )
        self.assertEqual(deadletter_requeue_body["data"]["remediation"]["remediation_type"], "requeue")
        self.assertEqual(deadletter_requeue_body["data"]["remediation"]["requeued_by"], "ops-a")
        self.assertEqual(
            deadletter_requeue_body["data"]["remediation"]["remediation_notes"],
            "Provider configuration fixed.",
        )
        self.assertEqual(
            deadletter_requeue_body["data"]["remediation"]["previous_delivery_exhausted_reason"],
            "max_attempts_reached",
        )
        self.assertEqual(deadletter_requeue_body["data"]["remediation"]["previous_delivery_attempt_count"], 2)

        deadletter_remediations = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/deadletters/remediations"
            f"?owner_id=customer-a&notification_id={prepared_notification_id}&limit=5"
        )
        self.assertEqual(deadletter_remediations.status_code, 200)
        deadletter_remediations_body = deadletter_remediations.json()
        self.assertEqual(
            deadletter_remediations_body["kind"],
            "advisor_action_queue_task_escalation_notification_deadletter_remediations",
        )
        self.assertEqual(deadletter_remediations_body["metadata"]["result_count"], 1)
        self.assertEqual(
            deadletter_remediations_body["data"][0]["remediation_id"],
            deadletter_requeue_body["metadata"]["remediation_id"],
        )
        self.assertEqual(deadletter_remediations_body["data"][0]["requeued_by"], "ops-a")

        cleared_deadletter_queue = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/deadletters"
            "?owner_id=customer-a&channel=email&limit=5"
        )
        self.assertEqual(cleared_deadletter_queue.status_code, 200)
        self.assertEqual(cleared_deadletter_queue.json()["metadata"]["result_count"], 0)

        requeued_delivery_queue = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-queue"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1&limit=5"
        )
        self.assertEqual(requeued_delivery_queue.status_code, 200)
        self.assertEqual(requeued_delivery_queue.json()["metadata"]["result_count"], 1)
        self.assertEqual(requeued_delivery_queue.json()["data"][0]["notification_id"], prepared_notification_id)
        self.assertEqual(requeued_delivery_queue.json()["data"][0]["delivery_action"], "retry_failed")

        repeated_delivery_completion = self.client.post(
            f"/agents/action-queues/tasks/escalations/inbox/notification/delivery-claim/{notification_id}/complete",
            json={"claim_token": delivery_claim_token, "status": "sent"},
        )
        self.assertEqual(repeated_delivery_completion.status_code, 400)

        invalid_delivery_claim = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-claim"
            "?owner_id=customer-a&channel=email&stale_after_minutes=1",
            json={"claimed_by": "worker-a", "lease_seconds": 0},
        )
        self.assertEqual(invalid_delivery_claim.status_code, 400)

        invalid_notification_owner = self.client.patch(
            f"/agents/action-queues/tasks/escalations/inbox/notification/logs/{notification_id}"
            "?owner_id=customer-b",
            json={"status": "sent"},
        )
        self.assertEqual(invalid_notification_owner.status_code, 404)

        empty_notification_update = self.client.patch(
            f"/agents/action-queues/tasks/escalations/inbox/notification/logs/{notification_id}",
            json={},
        )
        self.assertEqual(empty_notification_update.status_code, 400)

        invalid_notification_update = self.client.patch(
            f"/agents/action-queues/tasks/escalations/inbox/notification/logs/{notification_id}",
            json={"status": "queued"},
        )
        self.assertEqual(invalid_notification_update.status_code, 400)

        invalid_notification_log = self.client.post(
            "/agents/action-queues/tasks/escalations/inbox/notification?as_of=2026-05-28",
            json={"status": "queued"},
        )
        self.assertEqual(invalid_notification_log.status_code, 400)

        invalid_notification = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox/notification?as_of=2026-05-28&inbox_status=stale"
        )
        self.assertEqual(invalid_notification.status_code, 400)

        invalid_inbox_filter = self.client.get(
            "/agents/action-queues/tasks/escalations/inbox?as_of=2026-05-28&severity=urgent"
        )
        self.assertEqual(invalid_inbox_filter.status_code, 400)

        invalid_inbox = self.client.get("/agents/action-queues/tasks/escalations/inbox?as_of=bad-date")
        self.assertEqual(invalid_inbox.status_code, 400)

        bulk_escalation_review = self.client.post(
            "/agents/action-queues/tasks/escalations/reviews?owner_id=customer-a",
            json={
                "review_status": "acknowledged",
                "reviewer": "manager-b",
                "notes": "Reviewed from manager inbox.",
                "tasks": [
                    {"queue_id": queue_id, "task_id": task_id},
                    {"queue_id": second_queue_id, "task_id": second_task_id},
                ],
            },
        )
        self.assertEqual(bulk_escalation_review.status_code, 200)
        bulk_review_body = bulk_escalation_review.json()
        self.assertEqual(bulk_review_body["kind"], "advisor_action_queue_task_escalation_bulk_review")
        self.assertEqual(bulk_review_body["metadata"]["requested_count"], 2)
        self.assertEqual(bulk_review_body["metadata"]["reviewed_count"], 2)
        self.assertEqual(bulk_review_body["metadata"]["review_status"], "acknowledged")
        self.assertEqual(len(bulk_review_body["data"]["reviews"]), 2)
        self.assertEqual(
            {(item["queue_id"], item["task_id"]) for item in bulk_review_body["data"]["tasks"]},
            {(queue_id, task_id), (second_queue_id, second_task_id)},
        )

        bulk_review_history = self.client.get(
            "/agents/action-queues/tasks/escalations/reviews?owner_id=customer-a&limit=10"
        )
        self.assertEqual(bulk_review_history.status_code, 200)
        self.assertGreaterEqual(bulk_review_history.json()["metadata"]["result_count"], 4)
        self.assertGreaterEqual(
            sum(1 for item in bulk_review_history.json()["data"] if item["review_status"] == "acknowledged"),
            2,
        )

        missing_bulk_escalation_review = self.client.post(
            "/agents/action-queues/tasks/escalations/reviews?owner_id=customer-a",
            json={"review_status": "acknowledged", "tasks": [{"queue_id": 999, "task_id": "missing"}]},
        )
        self.assertEqual(missing_bulk_escalation_review.status_code, 404)

        invalid_bulk_snooze = self.client.post(
            "/agents/action-queues/tasks/escalations/reviews?owner_id=customer-a",
            json={
                "review_status": "snoozed",
                "tasks": [{"queue_id": queue_id, "task_id": task_id}],
            },
        )
        self.assertEqual(invalid_bulk_snooze.status_code, 400)

        invalid_workload = self.client.get("/agents/action-queues/tasks/workload?owner_id=customer-a&as_of=bad-date")
        self.assertEqual(invalid_workload.status_code, 400)

        activity = self.client.get("/agents/action-queues/tasks/activity?owner_id=customer-a&limit=10")
        self.assertEqual(activity.status_code, 200)
        activity_body = activity.json()
        self.assertEqual(activity_body["metadata"]["result_count"], 3)
        self.assertEqual([item["update_source"] for item in activity_body["data"][:2]], ["dashboard-bulk", "dashboard-bulk"])
        self.assertEqual(activity_body["data"][0]["new_status"], "deferred")
        self.assertEqual(activity_body["data"][0]["new_assigned_to"], "advisor-b")
        self.assertEqual(activity_body["data"][0]["new_due_at"], "2026-05-24")
        self.assertEqual(activity_body["data"][0]["updated_by"], "advisor-a")
        self.assertIn(activity_body["data"][0]["previous_status"], {"completed", second_generated["tasks"][0]["status"]})

        deferred_tasks = self.client.get("/agents/action-queues/tasks?owner_id=customer-a&status=deferred")
        self.assertEqual(deferred_tasks.status_code, 200)
        self.assertGreaterEqual(deferred_tasks.json()["metadata"]["result_count"], 2)

        missing_bulk_update = self.client.patch(
            "/agents/action-queues/tasks?owner_id=customer-a",
            json={"status": "completed", "tasks": [{"queue_id": 999, "task_id": "missing"}]},
        )
        self.assertEqual(missing_bulk_update.status_code, 404)

        invalid_due_update = self.client.patch(
            f"/agents/action-queues/{queue_id}/tasks/{task_id}?owner_id=customer-a",
            json={"due_at": "not-a-date"},
        )
        self.assertEqual(invalid_due_update.status_code, 400)

    def test_advisor_workbench_prioritizes_saved_queue_tasks(self) -> None:
        self.create_customer_portfolio()
        saved = self.client.post(
            "/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true"
        ).json()

        response = self.client.get("/agents/advisor-workbench?owner_id=customer-a&limit=3")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "advisor_workbench")
        self.assertEqual(body["owner_id"], "customer-a")
        self.assertEqual(body["summary"]["queue_count"], 1)
        self.assertTrue(body["next_actions"])
        self.assertEqual(body["top_recommendation"]["type"], "next_action")
        self.assertEqual(body["top_recommendation"]["queue_id"], saved["saved_queue_id"])
        self.assertLessEqual(len(body["next_actions"]), 3)
        self.assertIn("workbench_markdown", body)

    def test_advisor_outreach_draft_uses_saved_task(self) -> None:
        self.create_customer_portfolio()
        saved = self.client.post(
            "/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true"
        ).json()
        task_id = saved["tasks"][0]["task_id"]

        response = self.client.post(
            f"/agents/advisor-outreach-draft?owner_id=customer-a&queue_id={saved['saved_queue_id']}&task_id={task_id}"
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "advisor_outreach_draft")
        self.assertEqual(body["selection"], "explicit_task")
        self.assertEqual(body["queue_id"], saved["saved_queue_id"])
        self.assertEqual(body["task_id"], task_id)
        self.assertIn("subject", body["customer_email"])
        self.assertTrue(body["meeting_agenda"])
        self.assertTrue(body["compliance_guardrails"]["review_checklist"])
        self.assertTrue(body["approval_required"])
        self.assertIn("draft_markdown", body)

    def test_advisor_outreach_draft_can_use_workbench_top_task(self) -> None:
        self.create_customer_portfolio()
        saved = self.client.post(
            "/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true"
        ).json()

        response = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["selection"], "workbench_top_recommendation")
        self.assertEqual(body["queue_id"], saved["saved_queue_id"])
        self.assertTrue(body["customer_email"]["body"])

    def test_advisor_outreach_draft_can_be_saved_and_reviewed(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")

        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]

        listed = self.client.get("/agents/advisor-outreach-drafts?owner_id=customer-a")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["data"][0]["draft_id"], draft_id)
        self.assertEqual(listed.json()["data"][0]["status"], "draft")

        fetched = self.client.get(f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a")
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.json()["kind"], "saved_advisor_outreach_draft")
        self.assertEqual(fetched.json()["status"], "draft")

        reviewed = self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for review call.", "reviewer": "advisor-a"},
        )

        self.assertEqual(reviewed.status_code, 200)
        body = reviewed.json()
        self.assertEqual(body["status"], "approved")
        self.assertEqual(body["review_notes"], "Approved for review call.")
        self.assertEqual(body["reviewer"], "advisor-a")
        self.assertFalse(body["approval_required"])

    def test_advisor_outreach_compliance_review_flags_risky_copy(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE advisor_outreach_drafts SET body = body || ? WHERE draft_id = ?",
            ("\nThis is guaranteed and risk-free.", draft_id),
        )
        conn.commit()
        conn.close()

        response = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/compliance-review?owner_id=customer-a&save=true"
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "advisor_outreach_compliance_review")
        self.assertIn("saved_review_id", body)
        self.assertEqual(body["risk_level"], "critical")
        self.assertFalse(body["can_approve"])
        self.assertEqual(body["approval_recommendation"], "revise_before_approval")
        self.assertGreaterEqual(body["issue_count"], 1)
        self.assertIn("review_markdown", body)
        reviews = self.client.get(
            f"/agents/advisor-outreach-drafts/{draft_id}/compliance-reviews?owner_id=customer-a"
        )
        self.assertEqual(reviews.status_code, 200)
        self.assertEqual(reviews.json()["data"][0]["review_id"], body["saved_review_id"])

    def test_advisor_outreach_approval_is_compliance_gated(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE advisor_outreach_drafts SET body = body || ? WHERE draft_id = ?",
            ("\nThis is guaranteed and risk-free.", draft_id),
        )
        conn.commit()
        conn.close()

        blocked = self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approve despite risky text.", "reviewer": "advisor-a"},
        )

        self.assertEqual(blocked.status_code, 400)
        self.assertIn("Compliance review blocks approval", blocked.json()["detail"])
        fetched = self.client.get(f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a")
        self.assertEqual(fetched.json()["status"], "draft")
        reviews = self.client.get(
            f"/agents/advisor-outreach-drafts/{draft_id}/compliance-reviews?owner_id=customer-a"
        )
        self.assertEqual(reviews.json()["metadata"]["result_count"], 1)
        self.assertFalse(reviews.json()["data"][0]["can_approve"])

    def test_advisor_outreach_delivery_packet_requires_approval(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]

        blocked = self.client.post(f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a")

        self.assertEqual(blocked.status_code, 400)
        self.assertIn("requires an approved draft", blocked.json()["detail"])

    def test_advisor_outreach_delivery_packet_returns_approved_copy(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        approved = self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        self.assertEqual(approved.status_code, 200)

        response = self.client.post(f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "advisor_outreach_delivery_packet")
        self.assertEqual(body["delivery_status"], "ready")
        self.assertEqual(body["approval_evidence"]["reviewer"], "advisor-a")
        self.assertTrue(body["compliance_review"]["can_approve"])
        self.assertIn("packet_markdown", body)

    def test_advisor_outreach_delivery_packet_can_be_saved_and_marked_delivered(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        approved = self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        self.assertEqual(approved.status_code, 200)

        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )

        self.assertEqual(packet.status_code, 200)
        delivery_id = packet.json()["saved_delivery_id"]
        listed = self.client.get("/agents/advisor-outreach-deliveries?owner_id=customer-a")
        self.assertEqual(listed.json()["data"][0]["delivery_id"], delivery_id)
        fetched = self.client.get(f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a")
        self.assertEqual(fetched.json()["kind"], "saved_advisor_outreach_delivery_record")
        self.assertEqual(fetched.json()["status"], "ready")

        delivered = self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )

        self.assertEqual(delivered.status_code, 200)
        body = delivered.json()
        self.assertEqual(body["status"], "delivered")
        self.assertEqual(body["delivered_by"], "advisor-a")
        self.assertEqual(body["delivery_notes"], "Shared during review call.")
        self.assertIsNotNone(body["delivered_at"])

    def test_advisor_outreach_delivery_dashboard_summarizes_records(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]

        dashboard = self.client.get("/agents/advisor-outreach-delivery-dashboard?owner_id=customer-a")

        self.assertEqual(dashboard.status_code, 200)
        body = dashboard.json()
        self.assertEqual(body["kind"], "advisor_outreach_delivery_dashboard")
        self.assertEqual(body["summary"]["delivery_count"], 1)
        self.assertEqual(body["summary"]["ready_count"], 1)
        self.assertEqual(body["summary"]["delivered_count"], 0)
        self.assertEqual(body["top_recommendation"]["type"], "deliver_ready_packet")
        self.assertEqual(body["ready_deliveries"][0]["delivery_id"], delivery_id)
        self.assertIn("Outreach Delivery Dashboard", body["dashboard_markdown"])

        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        refreshed = self.client.get("/agents/advisor-outreach-delivery-dashboard?owner_id=customer-a").json()
        self.assertEqual(refreshed["summary"]["ready_count"], 0)
        self.assertEqual(refreshed["summary"]["delivered_count"], 1)
        self.assertEqual(refreshed["recent_deliveries"][0]["delivery_id"], delivery_id)

    def test_advisor_outreach_delivery_outcome_closes_delivered_loop(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        before = self.client.get("/agents/advisor-outreach-delivery-dashboard?owner_id=customer-a").json()
        self.assertEqual(before["summary"]["delivered_without_outcome_count"], 1)
        self.assertEqual(before["top_recommendation"]["type"], "record_delivery_outcome")

        outcome = self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        self.assertEqual(outcome.status_code, 200)
        body = outcome.json()
        outcome_id = body["outcome_id"]
        self.assertEqual(body["kind"], "saved_advisor_outreach_delivery_outcome")
        self.assertEqual(body["outcome_type"], "meeting_scheduled")
        self.assertEqual(body["customer_signal"], "positive")
        self.assertEqual(body["next_action"]["action_type"], "prepare_meeting")
        self.assertEqual(body["source_delivery"]["delivery_id"], delivery_id)
        self.assertIn("Outreach Delivery Outcome", body["outcome_markdown"])

        listed = self.client.get(
            f"/agents/advisor-outreach-outcomes?owner_id=customer-a&delivery_id={delivery_id}"
        ).json()
        self.assertEqual(listed["data"][0]["outcome_id"], outcome_id)
        fetched = self.client.get(f"/agents/advisor-outreach-outcomes/{outcome_id}?owner_id=customer-a")
        self.assertEqual(fetched.json()["next_action"]["follow_up_due_at"], "2026-05-29T15:00:00Z")
        after = self.client.get("/agents/advisor-outreach-delivery-dashboard?owner_id=customer-a").json()
        self.assertEqual(after["summary"]["delivered_without_outcome_count"], 0)

    def test_customer_intent_dashboard_ranks_saved_outcomes(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        dashboard = self.client.get("/agents/customer-intent-dashboard?owner_id=customer-a")

        self.assertEqual(dashboard.status_code, 200)
        body = dashboard.json()
        self.assertEqual(body["kind"], "customer_intent_dashboard")
        self.assertEqual(body["summary"]["owner_count"], 1)
        self.assertEqual(body["summary"]["meeting_ready_count"], 1)
        self.assertEqual(body["top_recommendation"]["type"], "prepare_meeting")
        self.assertEqual(body["owner_intents"][0]["owner_id"], "customer-a")
        self.assertEqual(body["owner_intents"][0]["segment"], "meeting_ready")
        self.assertEqual(body["owner_intents"][0]["next_action_type"], "prepare_meeting")
        self.assertEqual(body["recent_outcomes"][0]["delivery_id"], delivery_id)
        self.assertIn("Customer Intent Dashboard", body["dashboard_markdown"])

    def test_customer_intent_action_plan_turns_intent_into_worklist(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        plan = self.client.get("/agents/customer-intent-action-plan?owner_id=customer-a")

        self.assertEqual(plan.status_code, 200)
        body = plan.json()
        self.assertEqual(body["kind"], "customer_intent_action_plan")
        self.assertEqual(body["summary"]["action_count"], 1)
        self.assertEqual(body["top_action"]["owner_id"], "customer-a")
        self.assertEqual(body["top_action"]["priority"], "high")
        self.assertEqual(body["top_action"]["action_type"], "prepare_meeting")
        self.assertEqual(body["top_action"]["evidence"]["latest_delivery_id"], delivery_id)
        self.assertEqual(body["action_items"][0]["segment"], "meeting_ready")
        self.assertTrue(body["action_items"][0]["supporting_routes"])
        self.assertIn("Customer Intent Action Plan", body["plan_markdown"])

    def test_customer_intent_followup_packet_builds_execution_scaffold(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        outcome = self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        ).json()

        packet = self.client.get("/agents/customer-intent-followup-packet?owner_id=customer-a")

        self.assertEqual(packet.status_code, 200)
        body = packet.json()
        self.assertEqual(body["kind"], "customer_intent_followup_packet")
        self.assertEqual(body["packet_type"], "meeting_prep")
        self.assertEqual(body["action_item"]["action_type"], "prepare_meeting")
        self.assertEqual(body["source_outcome"]["outcome_id"], outcome["outcome_id"])
        self.assertTrue(body["customer_copy"]["send_allowed"])
        self.assertTrue(body["customer_copy"]["requires_review"])
        self.assertTrue(body["compliance_review_required"])
        self.assertIn("Customer Intent Follow-up Packet", body["packet_markdown"])

    def test_customer_intent_followup_review_preflights_packet(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        review = self.client.get("/agents/customer-intent-followup-review?owner_id=customer-a")

        self.assertEqual(review.status_code, 200)
        body = review.json()
        self.assertEqual(body["kind"], "customer_intent_followup_review")
        self.assertEqual(body["packet_type"], "meeting_prep")
        self.assertEqual(body["risk_level"], "low")
        self.assertTrue(body["can_prepare_draft"])
        self.assertEqual(body["recommendation"], "ready_for_reviewed_outreach_draft")
        self.assertEqual(body["issue_count"], 0)
        self.assertIsNotNone(body["source_packet"]["source_outcome_id"])
        self.assertIn("Customer Intent Follow-up Review", body["review_markdown"])

    def test_customer_intent_followup_draft_saves_into_review_workflow(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        handoff = self.client.post("/agents/customer-intent-followup-draft?owner_id=customer-a&save=true")

        self.assertEqual(handoff.status_code, 200)
        body = handoff.json()
        saved_draft_id = body["saved_draft_id"]
        self.assertEqual(body["kind"], "customer_intent_followup_draft")
        self.assertEqual(body["selection"], "customer_intent_followup")
        self.assertEqual(body["source_packet"]["action_type"], "prepare_meeting")
        self.assertIn("Customer Intent Follow-up Draft", body["draft_markdown"])
        saved = self.client.get(f"/agents/advisor-outreach-drafts/{saved_draft_id}?owner_id=customer-a")
        self.assertEqual(saved.json()["kind"], "saved_advisor_outreach_draft")
        review = self.client.post(
            f"/agents/advisor-outreach-drafts/{saved_draft_id}/compliance-review?owner_id=customer-a"
        )
        self.assertEqual(review.status_code, 200)
        self.assertTrue(review.json()["can_approve"])

    def test_customer_engagement_timeline_consolidates_workflow_history(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        timeline = self.client.get("/agents/customer-engagement-timeline?owner_id=customer-a")

        self.assertEqual(timeline.status_code, 200)
        body = timeline.json()
        event_types = {event["event_type"] for event in body["events"]}
        self.assertEqual(body["kind"], "customer_engagement_timeline")
        self.assertEqual(body["owner_id"], "customer-a")
        self.assertGreaterEqual(body["summary"]["event_count"], 4)
        self.assertIn("outreach_draft", event_types)
        self.assertIn("compliance_review", event_types)
        self.assertIn("delivery_delivered", event_types)
        self.assertIn("customer_outcome", event_types)
        self.assertEqual(body["top_action"]["owner_id"], "customer-a")
        self.assertIn("Customer Engagement Timeline", body["timeline_markdown"])

    def test_customer_engagement_brief_summarizes_timeline_context(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        brief = self.client.get("/agents/customer-engagement-brief?owner_id=customer-a")

        self.assertEqual(brief.status_code, 200)
        body = brief.json()
        self.assertEqual(body["kind"], "customer_engagement_brief")
        self.assertEqual(body["owner_id"], "customer-a")
        self.assertEqual(body["summary"]["current_segment"], "meeting_ready")
        self.assertEqual(body["current_intent"]["action_type"], "prepare_meeting")
        self.assertIn("recommended_action", body["next_best_action"])
        self.assertTrue(body["talking_points"])
        self.assertTrue(body["avoid"])
        self.assertTrue(body["evidence_references"])
        self.assertIn("Customer Engagement Brief", body["brief_markdown"])

    def test_customer_engagement_cadence_review_allows_reviewed_followup(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        review = self.client.get("/agents/customer-engagement-cadence-review?owner_id=customer-a")

        self.assertEqual(review.status_code, 200)
        body = review.json()
        self.assertEqual(body["kind"], "customer_engagement_cadence_review")
        self.assertTrue(body["contact_allowed"])
        self.assertEqual(body["contact_status"], "ready")
        self.assertEqual(body["issue_count"], 0)
        self.assertEqual(body["current_intent"]["action_type"], "prepare_meeting")
        self.assertEqual(body["next_route"]["path"], "/agents/customer-intent-followup-review?owner_id=customer-a")
        self.assertIn("Customer Engagement Cadence Review", body["review_markdown"])

    def test_customer_engagement_cadence_dashboard_ranks_ready_customers(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        dashboard = self.client.get("/agents/customer-engagement-cadence-dashboard")

        self.assertEqual(dashboard.status_code, 200)
        body = dashboard.json()
        self.assertEqual(body["kind"], "customer_engagement_cadence_dashboard")
        self.assertEqual(body["summary"]["customer_count"], 1)
        self.assertEqual(body["summary"]["ready_count"], 1)
        self.assertEqual(body["top_recommendation"]["owner_id"], "customer-a")
        self.assertEqual(body["customers"][0]["contact_status"], "ready")
        self.assertEqual(body["customers"][0]["action_type"], "prepare_meeting")
        self.assertIn("Customer Engagement Cadence Dashboard", body["dashboard_markdown"])

    def test_customer_engagement_action_queue_turns_cadence_into_tasks(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        queue = self.client.get("/agents/customer-engagement-action-queue")

        self.assertEqual(queue.status_code, 200)
        body = queue.json()
        self.assertEqual(body["kind"], "customer_engagement_action_queue")
        self.assertEqual(body["summary"]["task_count"], 1)
        self.assertEqual(body["summary"]["ready_count"], 1)
        self.assertEqual(body["top_task"]["owner_id"], "customer-a")
        self.assertEqual(body["top_task"]["status"], "ready")
        self.assertEqual(body["top_task"]["action_type"], "prepare_meeting")
        self.assertEqual(body["top_task"]["next_route"]["path"], "/agents/customer-intent-followup-review?owner_id=customer-a")
        self.assertIn("Customer Engagement Action Queue", body["queue_markdown"])

    def test_customer_engagement_task_brief_prepares_queued_task(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        brief = self.client.get("/agents/customer-engagement-task-brief?owner_id=customer-a")

        self.assertEqual(brief.status_code, 200)
        body = brief.json()
        self.assertEqual(body["kind"], "customer_engagement_task_brief")
        self.assertEqual(body["owner_id"], "customer-a")
        self.assertEqual(body["task"]["status"], "ready")
        self.assertEqual(body["task"]["action_type"], "prepare_meeting")
        self.assertEqual(body["execution_plan"]["recommended_route"]["path"], "/agents/customer-intent-followup-review?owner_id=customer-a")
        self.assertGreater(len(body["conversation_guide"]["talking_points"]), 0)
        self.assertIn("Record the customer response", body["execution_plan"]["steps"][-1])
        self.assertIn("Customer Engagement Task Brief", body["brief_markdown"])

    def test_ai_recommendation_effectiveness_dashboard_scores_saved_outcomes(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        dashboard = self.client.get("/agents/ai-recommendation-effectiveness-dashboard?owner_id=customer-a")

        self.assertEqual(dashboard.status_code, 200)
        body = dashboard.json()
        self.assertEqual(body["kind"], "ai_recommendation_effectiveness_dashboard")
        self.assertEqual(body["summary"]["delivery_count"], 1)
        self.assertEqual(body["summary"]["outcome_count"], 1)
        self.assertEqual(body["summary"]["positive_count"], 1)
        self.assertEqual(body["summary"]["meeting_scheduled_count"], 1)
        self.assertEqual(body["summary"]["response_capture_rate"], 1.0)
        self.assertGreater(body["top_recommendation"]["effectiveness_score"], 0)
        self.assertEqual(body["recent_successes"][0]["outcome_type"], "meeting_scheduled")
        self.assertIn("Scale the highest-scoring task pattern", body["learning_recommendations"][0])
        self.assertIn("AI Recommendation Effectiveness Dashboard", body["dashboard_markdown"])

    def test_ai_improvement_backlog_prioritizes_measured_outcome_learning(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        backlog = self.client.get("/agents/ai-improvement-backlog?owner_id=customer-a")

        self.assertEqual(backlog.status_code, 200)
        body = backlog.json()
        self.assertEqual(body["kind"], "ai_improvement_backlog")
        self.assertEqual(body["summary"]["outcome_count"], 1)
        self.assertEqual(body["summary"]["positive_outcome_rate"], 1.0)
        self.assertEqual(body["single_next_improvement"]["improvement_id"], "scale_top_pattern")
        self.assertEqual(body["single_next_improvement"]["priority"], "high")
        self.assertEqual(body["single_next_improvement"]["next_route"]["path"], "/agents/customer-engagement-action-queue?owner_id=customer-a")
        self.assertIn("success_metric", body["single_next_improvement"])
        self.assertIn("AI Improvement Backlog", body["backlog_markdown"])

    def test_ai_improvement_experiment_plan_turns_backlog_item_into_test(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        plan = self.client.get("/agents/ai-improvement-experiment-plan?owner_id=customer-a")

        self.assertEqual(plan.status_code, 200)
        body = plan.json()
        self.assertEqual(body["kind"], "ai_improvement_experiment_plan")
        self.assertEqual(body["improvement"]["improvement_id"], "scale_top_pattern")
        self.assertIn("Scale the best-performing recommendation pattern", body["hypothesis"])
        self.assertEqual(body["baseline"]["current_metrics"]["positive_outcome_rate"], 1.0)
        self.assertEqual(body["treatment"]["workflow_route"]["path"], "/agents/customer-engagement-action-queue?owner_id=customer-a")
        self.assertGreaterEqual(body["sample_criteria"]["minimum_sample"], 1)
        self.assertEqual(body["measurement_route"]["path"], "/agents/ai-recommendation-effectiveness-dashboard?owner_id=customer-a")
        self.assertIn("Stop if compliance review blocks", body["stop_conditions"][1])
        self.assertIn("AI Improvement Experiment Plan", body["experiment_markdown"])

    def test_ai_improvement_experiment_launch_packet_prepares_safe_launch(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        launch = self.client.get("/agents/ai-improvement-experiment-launch-packet?owner_id=customer-a")

        self.assertEqual(launch.status_code, 200)
        body = launch.json()
        self.assertEqual(body["kind"], "ai_improvement_experiment_launch_packet")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertTrue(body["readiness"]["can_launch"])
        self.assertEqual(body["cohort_assignment"]["scope"], "single_owner")
        self.assertEqual(body["treatment"]["workflow_route"]["path"], "/agents/customer-engagement-action-queue?owner_id=customer-a")
        self.assertEqual(body["measurement_route"]["path"], "/agents/ai-recommendation-effectiveness-dashboard?owner_id=customer-a")
        self.assertTrue(any(item["field"] == "outcome_type" for item in body["data_capture_requirements"]))
        self.assertIn("rollback_action", body["rollback_plan"])
        self.assertIn("AI Improvement Experiment Launch Packet", body["launch_markdown"])

    def test_ai_improvement_experiment_readout_recommends_next_decision(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        readout = self.client.get("/agents/ai-improvement-experiment-readout?owner_id=customer-a")

        self.assertEqual(readout.status_code, 200)
        body = readout.json()
        self.assertEqual(body["kind"], "ai_improvement_experiment_readout")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["decision"]["status"], "continue_collecting")
        self.assertEqual(body["sample_status"]["current_sample"], 1)
        self.assertFalse(body["sample_status"]["target_met"])
        self.assertEqual(body["metric_snapshot"]["positive_outcome_rate"], 1.0)
        self.assertTrue(any(result["condition"] == "negative_customer_signal" for result in body["stop_condition_results"]))
        self.assertEqual(body["recommended_next_route"]["path"], "/agents/customer-engagement-action-queue?owner_id=customer-a")
        self.assertIn("AI Improvement Experiment Readout", body["readout_markdown"])

    def test_ai_improvement_rollout_readiness_gates_release_from_readout(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        readiness = self.client.get("/agents/ai-improvement-rollout-readiness?owner_id=customer-a")

        self.assertEqual(readiness.status_code, 200)
        body = readiness.json()
        self.assertEqual(body["kind"], "ai_improvement_rollout_readiness")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["release_gate"]["status"], "needs_more_evidence")
        self.assertFalse(body["release_gate"]["can_rollout"])
        self.assertEqual(body["rollout_phases"][0]["phase"], "continue_pilot")
        self.assertEqual(body["recommended_next_route"]["path"], "/agents/customer-engagement-action-queue?owner_id=customer-a")
        self.assertTrue(any(metric["name"] == "positive_outcome_rate" for metric in body["monitoring_plan"]["metrics"]))
        self.assertIn("AI Improvement Rollout Readiness", body["readiness_markdown"])

    def test_ai_improvement_rollout_monitor_reports_alerts_and_next_action(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        monitor = self.client.get("/agents/ai-improvement-rollout-monitor?owner_id=customer-a")

        self.assertEqual(monitor.status_code, 200)
        body = monitor.json()
        self.assertEqual(body["kind"], "ai_improvement_rollout_monitor")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["status"], "pilot_monitoring")
        self.assertEqual(body["risk_level"], "medium")
        self.assertEqual(body["immediate_action"]["action"], "continue_pilot")
        self.assertTrue(any(alert["code"] == "needs_more_evidence" for alert in body["alerts"]))
        self.assertTrue(any(metric["name"] == "sample_target_met" and metric["status"] == "collecting" for metric in body["tracked_metrics"]))
        self.assertEqual(body["next_check"]["route"]["path"], "/agents/customer-engagement-action-queue?owner_id=customer-a")
        self.assertIn("AI Improvement Rollout Monitor", body["monitor_markdown"])

    def test_ai_improvement_release_packet_packages_monitor_for_humans(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        release = self.client.get("/agents/ai-improvement-release-packet?owner_id=customer-a")

        self.assertEqual(release.status_code, 200)
        body = release.json()
        self.assertEqual(body["kind"], "ai_improvement_release_packet")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["release_status"], "pilot_update")
        self.assertEqual(body["eligibility"]["gate_status"], "needs_more_evidence")
        self.assertFalse(body["eligibility"]["eligible"])
        self.assertTrue(any(item["topic"] == "What to do now" for item in body["advisor_enablement"]))
        self.assertTrue(any("pilot monitoring" in point for point in body["support_talking_points"]))
        self.assertTrue(any(risk["risk"] == "needs_more_evidence" for risk in body["known_risks"]))
        self.assertIn("AI Improvement Release Packet", body["release_markdown"])

    def test_ai_improvement_adoption_playbook_turns_release_into_enablement(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        playbook = self.client.get("/agents/ai-improvement-adoption-playbook?owner_id=customer-a")

        self.assertEqual(playbook.status_code, 200)
        body = playbook.json()
        self.assertEqual(body["kind"], "ai_improvement_adoption_playbook")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["adoption_status"]["status"], "pilot_enablement")
        self.assertEqual(body["next_action"]["action"], "train_pilot_advisors")
        self.assertTrue(any(item["role"] == "advisor" for item in body["role_tasks"]))
        self.assertTrue(any(item["check"] == "record_outcomes" for item in body["training_checklist"]))
        self.assertTrue(any(blocker["risk"] == "not_eligible_for_broad_adoption" for blocker in body["adoption_blockers"]))
        self.assertIn("AI Improvement Adoption Playbook", body["playbook_markdown"])

    def test_ai_improvement_adoption_monitor_tracks_training_and_blockers(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        monitor = self.client.get("/agents/ai-improvement-adoption-monitor?owner_id=customer-a")

        self.assertEqual(monitor.status_code, 200)
        body = monitor.json()
        self.assertEqual(body["kind"], "ai_improvement_adoption_monitor")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["status"], "pilot_training")
        self.assertEqual(body["risk_level"], "medium")
        self.assertEqual(body["training_status"]["status"], "pending")
        self.assertTrue(any(blocker["risk"] == "not_eligible_for_broad_adoption" for blocker in body["blockers"]))
        self.assertEqual(body["customer_language_status"]["status"], "review_required")
        self.assertEqual(body["immediate_action"]["action"], "train_pilot_advisors")
        self.assertIn("AI Improvement Adoption Monitor", body["monitor_markdown"])

    def test_ai_improvement_adoption_impact_ledger_proves_customer_value(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        ledger = self.client.get("/agents/ai-improvement-adoption-impact-ledger?owner_id=customer-a")

        self.assertEqual(ledger.status_code, 200)
        body = ledger.json()
        self.assertEqual(body["kind"], "ai_improvement_adoption_impact_ledger")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["value_status"], "proving_value")
        self.assertEqual(body["customer_impact"]["positive_count"], 1)
        self.assertEqual(body["customer_impact"]["meeting_scheduled_count"], 1)
        self.assertEqual(body["advisor_usage"]["delivery_count"], 1)
        self.assertGreater(body["advisor_usage"]["response_capture_rate"], 0)
        self.assertEqual(body["scale_decision"]["action"], "keep_pilot")
        self.assertTrue(any(account["risk"] == "not_eligible_for_broad_adoption" for account in body["blocked_accounts"]))
        self.assertEqual(body["next_action"]["action"], "train_pilot_advisors")
        self.assertIn("AI Improvement Adoption Impact Ledger", body["ledger_markdown"])

    def test_ai_improvement_scale_decision_packet_keeps_value_in_pilot(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get("/agents/ai-improvement-scale-decision-packet?owner_id=customer-a")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_scale_decision_packet")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["decision"]["status"], "continue_pilot")
        self.assertEqual(body["customer_value_evidence"]["evidence_strength"], "directional")
        self.assertEqual(body["customer_value_evidence"]["positive_count"], 1)
        self.assertEqual(body["rollout_scope"]["scope"], "pilot_only")
        self.assertEqual(body["next_action"]["action"], "train_pilot_advisors")
        self.assertTrue(any(item["blocker"] == "not_eligible_for_broad_adoption" for item in body["blocker_resolution_plan"]))
        self.assertTrue(any(item["role"] == "manager" for item in body["advisor_change_plan"]))
        self.assertIn("AI Improvement Scale Decision Packet", body["packet_markdown"])

    def test_ai_improvement_scale_execution_plan_turns_decision_into_tasks(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get("/agents/ai-improvement-scale-execution-plan?owner_id=customer-a")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_scale_execution_plan")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["execution_status"], "pilot_execution")
        self.assertEqual(body["decision"]["status"], "continue_pilot")
        self.assertEqual(body["rollout_scope"]["scope"], "pilot_only")
        self.assertEqual(body["next_action"]["action"], "resolve_scale_blockers")
        self.assertTrue(any(task["owner"] == "manager" for task in body["execution_tasks"]))
        self.assertTrue(any(guardrail["guardrail"] == "no_broad_enablement" for guardrail in body["guardrails"]))
        self.assertTrue(any(check["check"] == "positive_outcomes_present" and check["status"] == "met" for check in body["customer_proof_checks"]))
        self.assertTrue(any(criteria["criterion"] == "all_blockers_resolved" for criteria in body["acceptance_criteria"]))
        self.assertEqual(body["escalation_path"]["owner"], "manager")
        self.assertIn("AI Improvement Scale Execution Plan", body["plan_markdown"])

    def test_ai_improvement_scale_execution_monitor_tracks_blocked_pilot(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get("/agents/ai-improvement-scale-execution-monitor?owner_id=customer-a")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_scale_execution_monitor")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["status"], "blocked")
        self.assertEqual(body["risk_level"], "medium")
        self.assertEqual(body["task_progress"]["pending_count"], 3)
        self.assertEqual(body["customer_proof_status"]["status"], "met")
        self.assertEqual(body["acceptance_status"]["status"], "blocked")
        self.assertTrue(any(blocker["blocker"] == "all_blockers_resolved" for blocker in body["blockers"]))
        self.assertEqual(body["immediate_action"]["owner"], "manager")
        self.assertEqual(body["immediate_action"]["action"], "resolve_scale_blockers")
        self.assertEqual(body["escalation_path"]["owner"], "manager")
        self.assertIn("AI Improvement Scale Execution Monitor", body["monitor_markdown"])

    def test_ai_improvement_scale_learning_report_feeds_blockers_back_to_roadmap(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get("/agents/ai-improvement-scale-learning-report?owner_id=customer-a")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_scale_learning_report")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["learning_status"], "blocked_but_value_visible")
        self.assertTrue(any(item["learning"] == "customer_proof_status" for item in body["validated_learnings"]))
        self.assertTrue(any("all_blockers_resolved" in item["question"] for item in body["open_questions"]))
        self.assertTrue(any(item["action"] == "update_acceptance_gaps" for item in body["feedback_actions"]))
        self.assertEqual(body["next_improvement_candidate"]["candidate"], "clear_scale_blocker")
        self.assertEqual(body["roadmap_update"]["status"], "feed_back_to_backlog")
        self.assertEqual(body["source_monitor_status"], "blocked")
        self.assertIn("AI Improvement Scale Learning Report", body["report_markdown"])

    def test_ai_improvement_roadmap_refresh_turns_learning_into_backlog_item(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get("/agents/ai-improvement-roadmap-refresh?owner_id=customer-a")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_roadmap_refresh")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["roadmap_status"], "backlog_ready")
        self.assertEqual(body["roadmap_item"]["candidate"], "clear_scale_blocker")
        self.assertEqual(body["roadmap_item"]["priority"], "medium")
        self.assertEqual(body["evidence_package"]["learning_status"], "blocked_but_value_visible")
        self.assertTrue(any(item["owner"] == "manager" for item in body["owner_action_plan"]))
        self.assertTrue(any(gate["gate"] == "customer_proof_preserved" and gate["status"] == "met" for gate in body["acceptance_gates"]))
        self.assertTrue(any(metric["metric"] == "open_question_count" for metric in body["measurement_plan"]))
        self.assertEqual(body["sequencing"]["sequence"], "current_cycle")
        self.assertEqual(body["next_action"]["action"], "resolve_scale_blockers")
        self.assertIn("AI Improvement Roadmap Refresh", body["roadmap_markdown"])

    def test_ai_improvement_backlog_handoff_packages_roadmap_item_for_work(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get("/agents/ai-improvement-backlog-handoff?owner_id=customer-a")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_backlog_handoff")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["handoff_status"], "ready_for_backlog")
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertEqual(body["work_item"]["owner"], "manager")
        self.assertEqual(body["implementation_story"]["i_want"], "to implement clear scale blocker")
        self.assertIn("acceptance gate closure", body["implementation_scope"]["in_scope"])
        self.assertTrue(any(item["dependency"] == "open_questions_resolved" for item in body["dependencies"]))
        self.assertEqual(body["launch_readiness"]["status"], "blocked")
        self.assertEqual(body["next_action"]["action"], "resolve_scale_blockers")
        self.assertEqual(body["source_roadmap_status"], "backlog_ready")
        self.assertEqual(body["source_next_improvement_candidate"]["candidate"], "clear_scale_blocker")
        self.assertIn("AI Improvement Backlog Handoff", body["handoff_markdown"])

    def test_ai_improvement_implementation_kickoff_packet_prepares_blocker_work(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get("/agents/ai-improvement-implementation-kickoff-packet?owner_id=customer-a")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_implementation_kickoff_packet")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["kickoff_status"], "ready_for_blocker_work")
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertEqual(body["work_item"]["owner"], "manager")
        self.assertIn("ai_improvement_decision_chain", body["engineering_scope"]["components"])
        self.assertTrue(any(gate["gate"] == "full_unittest_suite" for gate in body["qa_gates"]))
        self.assertTrue(any(contract["contract"] == "processed_payloads_only" for contract in body["data_contracts"]))
        self.assertTrue(any(guardrail["guardrail"] == "no_broad_rollout_until_ready" for guardrail in body["customer_value_guardrails"]))
        self.assertTrue(any(item["item"] == "open_questions_resolved" for item in body["launch_checklist"]))
        self.assertEqual(body["immediate_action"]["action"], "resolve_scale_blockers")
        self.assertEqual(body["source_launch_readiness"]["status"], "blocked")
        self.assertIn("AI Improvement Implementation Kickoff Packet", body["kickoff_markdown"])

    def test_ai_improvement_implementation_readiness_monitor_tracks_blockers(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get("/agents/ai-improvement-implementation-readiness-monitor?owner_id=customer-a")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_implementation_readiness_monitor")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["status"], "blocked")
        self.assertEqual(body["risk_level"], "high")
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertEqual(body["qa_status"]["status"], "blocked")
        self.assertEqual(body["data_contract_status"]["status"], "active")
        self.assertEqual(body["customer_guardrail_status"]["status"], "blocked")
        self.assertEqual(body["launch_checklist_status"]["status"], "blocked")
        self.assertTrue(any(blocker["blocker"] == "customer_guardrails_blocked" for blocker in body["blockers"]))
        self.assertEqual(body["immediate_action"]["action"], "resolve_scale_blockers")
        self.assertEqual(body["source_kickoff_status"], "ready_for_blocker_work")
        self.assertIn("AI Improvement Implementation Readiness Monitor", body["monitor_markdown"])

    def test_ai_improvement_implementation_blocker_resolution_plan_assigns_unblock_work(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get("/agents/ai-improvement-implementation-blocker-resolution-plan?owner_id=customer-a")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_implementation_blocker_resolution_plan")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["resolution_status"], "blocked_resolution_required")
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertTrue(any(task["blocker"] == "customer_guardrails_blocked" for task in body["resolution_tasks"]))
        self.assertEqual(body["immediate_unblock_action"]["owner"], "manager")
        self.assertEqual(body["immediate_unblock_action"]["action"], "clear_customer_value_guardrails")
        self.assertTrue(any(proof["proof"] == "customer_guardrail_evidence" for proof in body["proof_required"]))
        self.assertTrue(any(criteria["criterion"] == "all_blockers_closed" for criteria in body["exit_criteria"]))
        self.assertTrue(any(item["command"] == "python3 -m unittest discover -s tests" for item in body["qa_rerun_plan"]))
        self.assertEqual(body["customer_guardrail_clearance"]["status"], "blocked")
        self.assertEqual(body["source_readiness_status"], "blocked")
        self.assertEqual(body["source_risk_level"], "high")
        self.assertIn("AI Improvement Implementation Blocker Resolution Plan", body["resolution_markdown"])

    def test_ai_improvement_implementation_unblock_verification_report_keeps_blocked_work_from_launch(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get(
            "/agents/ai-improvement-implementation-unblock-verification-report?owner_id=customer-a"
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_implementation_unblock_verification_report")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["verification_status"], "blocked")
        self.assertFalse(body["ready_to_proceed"])
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertEqual(body["task_status"]["status"], "pending")
        self.assertEqual(body["proof_status"]["status"], "blocked")
        self.assertEqual(body["exit_criteria_status"]["status"], "blocked")
        self.assertEqual(body["qa_rerun_status"]["status"], "pending")
        self.assertEqual(body["customer_guardrail_status"]["status"], "blocked")
        self.assertEqual(body["next_verification_action"]["action"], "clear_customer_value_guardrails")
        self.assertEqual(body["source_resolution_status"], "blocked_resolution_required")
        self.assertEqual(body["source_risk_level"], "high")
        self.assertIn("AI Improvement Implementation Unblock Verification Report", body["verification_markdown"])

    def test_ai_improvement_implementation_qa_review_packet_holds_blocked_work(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get("/agents/ai-improvement-implementation-qa-review-packet?owner_id=customer-a")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_implementation_qa_review_packet")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["qa_decision"]["status"], "hold_qa")
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertIn("full unittest regression", body["qa_scope"]["include"])
        self.assertTrue(any(gap["gap"] == "customer_guardrail_evidence" for gap in body["evidence_gaps"]))
        self.assertTrue(any(gate["gate"] == "python3 -m unittest discover -s tests" for gate in body["test_gates"]))
        self.assertTrue(body["customer_guardrails"]["qa_must_hold"])
        self.assertTrue(any(signoff["signoff"] == "customer_guardrail_owner" for signoff in body["signoff_requirements"]))
        self.assertEqual(body["next_qa_action"]["action"], "clear_customer_value_guardrails")
        self.assertEqual(body["source_verification_status"], "blocked")
        self.assertFalse(body["source_ready_to_proceed"])
        self.assertIn("AI Improvement Implementation QA Review Packet", body["qa_markdown"])

    def test_ai_improvement_implementation_qa_signoff_report_holds_launch(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get("/agents/ai-improvement-implementation-qa-signoff-report?owner_id=customer-a")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_implementation_qa_signoff_report")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["signoff_status"], "blocked")
        self.assertEqual(body["signoff_decision"]["action"], "hold_launch")
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertTrue(any(item["signoff"] == "qa_owner" for item in body["required_signoffs"]))
        self.assertTrue(any(item["gap"] == "customer_guardrail_evidence" for item in body["signoff_gaps"]))
        self.assertTrue(any(item["blocker"] == "customer_guardrail_hold" for item in body["launch_blockers"]))
        self.assertEqual(body["evidence_summary"]["customer_guardrail_status"], "blocked")
        self.assertEqual(body["next_signoff_action"]["action"], "clear_customer_value_guardrails")
        self.assertEqual(body["source_qa_decision"]["status"], "hold_qa")
        self.assertFalse(body["source_ready_to_proceed"])
        self.assertEqual(body["source_risk_level"], "high")
        self.assertIn("AI Improvement Implementation QA Signoff Report", body["signoff_markdown"])

    def test_ai_improvement_launch_review_packet_holds_blocked_signoff(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get("/agents/ai-improvement-launch-review-packet?owner_id=customer-a")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_launch_review_packet")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["launch_decision"]["status"], "hold_launch")
        self.assertEqual(body["launch_scope"]["scope"], "no_launch")
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertTrue(any(item["guardrail"] == "no_launch_with_customer_guardrail_hold" for item in body["customer_guardrails"]))
        self.assertTrue(any(item["monitor"] == "launch_blocker_count" for item in body["monitoring_requirements"]))
        self.assertTrue(any(item["trigger"] == "customer_guardrail_hold" for item in body["rollback_triggers"]))
        self.assertTrue(any(item["blocker"] == "customer_guardrail_hold" for item in body["launch_blockers"]))
        self.assertEqual(body["next_launch_action"]["action"], "clear_customer_value_guardrails")
        self.assertEqual(body["source_signoff_status"], "blocked")
        self.assertEqual(body["source_signoff_decision"]["action"], "hold_launch")
        self.assertEqual(body["source_risk_level"], "high")
        self.assertIn("AI Improvement Launch Review Packet", body["launch_markdown"])

    def test_ai_improvement_launch_execution_plan_turns_hold_into_owned_tasks(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get("/agents/ai-improvement-launch-execution-plan?owner_id=customer-a")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_launch_execution_plan")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["execution_status"], "held")
        self.assertEqual(body["launch_decision"]["status"], "hold_launch")
        self.assertEqual(body["launch_scope"]["scope"], "no_launch")
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertTrue(any(task["action"] == "hold_launch" for task in body["execution_tasks"]))
        self.assertTrue(any(item["monitor"] == "launch_blocker_count" and item["status"] == "needs_attention" for item in body["monitoring_setup"]))
        self.assertTrue(any(item["status"] == "hold" for item in body["rollback_setup"]))
        self.assertTrue(any(item["criterion"] == "launch_decision_ready" for item in body["exit_criteria"]))
        self.assertEqual(body["immediate_action"]["action"], "hold_launch")
        self.assertEqual(body["source_risk_level"], "high")
        self.assertIn("AI Improvement Launch Execution Plan", body["execution_markdown"])

    def test_ai_improvement_launch_execution_monitor_tracks_held_launch(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get("/agents/ai-improvement-launch-execution-monitor?owner_id=customer-a")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_launch_execution_monitor")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["status"], "held")
        self.assertEqual(body["risk_level"], "high")
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertEqual(body["task_status"]["status"], "pending")
        self.assertEqual(body["monitoring_status"]["status"], "needs_attention")
        self.assertEqual(body["rollback_status"]["status"], "hold")
        self.assertEqual(body["exit_criteria_status"]["status"], "blocked")
        self.assertTrue(any(blocker["blocker"] == "rollback_not_armed" for blocker in body["blockers"]))
        self.assertEqual(body["immediate_action"]["action"], "hold_launch")
        self.assertEqual(body["source_execution_status"], "held")
        self.assertEqual(body["source_launch_decision"]["status"], "hold_launch")
        self.assertIn("AI Improvement Launch Execution Monitor", body["monitor_markdown"])

    def test_ai_improvement_launch_outcome_monitor_blocks_prelaunch_outcomes(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get("/agents/ai-improvement-launch-outcome-monitor?owner_id=customer-a")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_launch_outcome_monitor")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["status"], "blocked_pre_launch")
        self.assertEqual(body["risk_level"], "high")
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertEqual(body["launch_health"]["status"], "blocked")
        self.assertEqual(body["value_status"]["status"], "not_measured")
        self.assertEqual(body["customer_signal_status"]["status"], "not_started")
        self.assertEqual(body["rollback_status"]["status"], "not_armed")
        self.assertTrue(any(blocker["blocker"] == "launch_not_executed" for blocker in body["blockers"]))
        self.assertEqual(body["immediate_action"]["action"], "hold_launch")
        self.assertEqual(body["source_execution_monitor_status"], "held")
        self.assertEqual(body["source_execution_status"], "held")
        self.assertEqual(body["source_launch_decision"]["status"], "hold_launch")
        self.assertIn("AI Improvement Launch Outcome Monitor", body["outcome_markdown"])

    def test_ai_improvement_launch_value_proof_packet_blocks_unearned_claims(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get("/agents/ai-improvement-launch-value-proof-packet?owner_id=customer-a")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_launch_value_proof_packet")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["proof_status"], "blocked_pre_launch")
        self.assertEqual(body["risk_level"], "high")
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertEqual(body["customer_value_claim"]["status"], "not_claimable")
        self.assertEqual(body["customer_message"]["status"], "internal_only")
        self.assertTrue(any(item["proof"] == "no_customer_value_claim_before_launch" for item in body["proof_points"]))
        self.assertTrue(any(item["gap"] == "launch_not_executed" for item in body["evidence_gaps"]))
        self.assertEqual(body["advisor_next_action"]["action"], "hold_launch")
        self.assertEqual(body["source_outcome_status"], "blocked_pre_launch")
        self.assertEqual(body["source_outcome_risk_level"], "high")
        self.assertEqual(body["source_launch_decision"]["status"], "hold_launch")
        self.assertIn("AI Improvement Launch Value Proof Packet", body["proof_markdown"])

    def test_ai_improvement_launch_customer_communication_packet_is_internal_when_claims_blocked(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get("/agents/ai-improvement-launch-customer-communication-packet?owner_id=customer-a")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_launch_customer_communication_packet")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["communication_status"], "internal_hold_only")
        self.assertEqual(body["risk_level"], "high")
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertEqual(body["audience"]["visibility"], "internal")
        self.assertEqual(body["message"]["cta"], "Hold customer-facing value claims until launch blockers clear.")
        self.assertTrue(any(gate["gate"] == "proof_status" and gate["status"] == "blocked" for gate in body["review_gates"]))
        self.assertTrue(any(item["claim"] == "ai_improvement_created_customer_value" for item in body["blocked_claims"]))
        self.assertEqual(body["advisor_next_action"]["action"], "hold_launch")
        self.assertEqual(body["source_proof_status"], "blocked_pre_launch")
        self.assertEqual(body["source_customer_claim_status"], "not_claimable")
        self.assertEqual(body["source_launch_decision"]["status"], "hold_launch")
        self.assertIn("AI Improvement Launch Customer Communication Packet", body["communication_markdown"])

    def test_ai_improvement_launch_customer_communication_review_packet_holds_blocked_send(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get(
            "/agents/ai-improvement-launch-customer-communication-review-packet?owner_id=customer-a"
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_launch_customer_communication_review_packet")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["review_status"], "hold_send")
        self.assertEqual(body["risk_level"], "high")
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertEqual(body["send_decision"]["decision"], "do_not_send")
        self.assertTrue(any(item["owner"] == "manager" and item["status"] == "blocked" for item in body["required_approvals"]))
        self.assertTrue(any(item["blocker"] == "review_gate:proof_status" for item in body["send_blockers"]))
        self.assertTrue(any(item["action"] == "hold_customer_send" for item in body["escalation_path"]))
        self.assertEqual(body["approved_message"]["status"], "withheld")
        self.assertEqual(body["advisor_next_action"]["action"], "hold_launch")
        self.assertEqual(body["source_communication_status"], "internal_hold_only")
        self.assertEqual(body["source_customer_claim_status"], "not_claimable")
        self.assertEqual(body["source_launch_decision"]["status"], "hold_launch")
        self.assertIn("AI Improvement Launch Customer Communication Review Packet", body["review_markdown"])

    def test_ai_improvement_launch_customer_communication_delivery_packet_withholds_blocked_send(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get(
            "/agents/ai-improvement-launch-customer-communication-delivery-packet?owner_id=customer-a"
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_launch_customer_communication_delivery_packet")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["delivery_status"], "withheld")
        self.assertEqual(body["risk_level"], "high")
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertEqual(body["channel_plan"]["visibility"], "internal")
        self.assertEqual(body["channel_plan"]["status"], "withheld")
        self.assertEqual(body["delivery_payload"]["status"], "withheld")
        self.assertFalse(body["delivery_payload"]["customer_facing"])
        self.assertTrue(any(item["gate"] == "send_decision" and item["status"] == "blocked" for item in body["delivery_checklist"]))
        self.assertTrue(any(item["event"] == "source_send_decision" and item["status"] == "do_not_send" for item in body["audit_trail"]))
        self.assertEqual(body["follow_up_plan"]["action"], "recheck_after_launch_gates_clear")
        self.assertEqual(body["advisor_next_action"]["action"], "hold_customer_send")
        self.assertEqual(body["source_review_status"], "hold_send")
        self.assertEqual(body["source_send_decision"]["decision"], "do_not_send")
        self.assertEqual(body["source_customer_claim_status"], "not_claimable")
        self.assertEqual(body["source_launch_decision"]["status"], "hold_launch")
        self.assertIn("AI Improvement Launch Customer Communication Delivery Packet", body["delivery_markdown"])

    def test_ai_improvement_launch_customer_communication_delivery_monitor_tracks_withheld_delivery(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get(
            "/agents/ai-improvement-launch-customer-communication-delivery-monitor?owner_id=customer-a"
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_launch_customer_communication_delivery_monitor")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["status"], "blocked")
        self.assertEqual(body["risk_level"], "high")
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertEqual(body["delivery_progress"]["status"], "withheld")
        self.assertFalse(body["delivery_progress"]["customer_facing"])
        self.assertEqual(body["checklist_status"]["status"], "blocked")
        self.assertEqual(body["checklist_status"]["blocked_count"], 3)
        self.assertEqual(body["audit_status"]["status"], "recorded")
        self.assertEqual(body["follow_up_status"]["status"], "waiting")
        self.assertTrue(any(item["blocker"] == "delivery_withheld" for item in body["blockers"]))
        self.assertEqual(body["immediate_action"]["action"], "hold_customer_send")
        self.assertEqual(body["source_delivery_status"], "withheld")
        self.assertEqual(body["source_send_decision"]["decision"], "do_not_send")
        self.assertEqual(body["source_customer_claim_status"], "not_claimable")
        self.assertEqual(body["source_launch_decision"]["status"], "hold_launch")
        self.assertIn("AI Improvement Launch Customer Communication Delivery Monitor", body["monitor_markdown"])

    def test_ai_improvement_launch_customer_communication_delivery_unblock_plan_turns_blockers_into_tasks(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get(
            "/agents/ai-improvement-launch-customer-communication-delivery-unblock-plan?owner_id=customer-a"
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_launch_customer_communication_delivery_unblock_plan")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["plan_status"], "blocked_delivery")
        self.assertEqual(body["risk_level"], "high")
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertTrue(any(task["action"] == "clear_send_hold" for task in body["unblock_tasks"]))
        self.assertTrue(any(task["action"] == "collect_customer_value_proof" for task in body["unblock_tasks"]))
        self.assertTrue(any(gate["gate"] == "send_decision" and gate["status"] == "blocked" for gate in body["proof_gates"]))
        self.assertTrue(any(item["criterion"] == "delivery_not_withheld" and item["status"] == "blocked" for item in body["exit_criteria"]))
        self.assertEqual(body["recheck_plan"]["action"], "recheck_delivery_unblock_after_launch_gates_clear")
        self.assertEqual(body["immediate_action"]["action"], "clear_send_hold")
        self.assertEqual(body["source_monitor_status"], "blocked")
        self.assertEqual(body["source_delivery_status"], "withheld")
        self.assertEqual(body["source_send_decision"]["decision"], "do_not_send")
        self.assertEqual(body["source_customer_claim_status"], "not_claimable")
        self.assertEqual(body["source_launch_decision"]["status"], "hold_launch")
        self.assertIn("AI Improvement Launch Customer Communication Delivery Unblock Plan", body["plan_markdown"])

    def test_ai_improvement_launch_customer_communication_delivery_unblock_verification_report_fails_blocked_plan(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get(
            "/agents/ai-improvement-launch-customer-communication-delivery-unblock-verification-report?owner_id=customer-a"
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_launch_customer_communication_delivery_unblock_verification_report")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["verification_status"], "failed")
        self.assertEqual(body["risk_level"], "high")
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertTrue(any(item["check"] == "proof_gate:send_decision" for item in body["failed_checks"]))
        self.assertTrue(any(item["check"] == "exit_criterion:delivery_not_withheld" for item in body["failed_checks"]))
        self.assertTrue(any(item["check"] == "unblock_task:clear_send_hold" for item in body["failed_checks"]))
        self.assertEqual(body["required_follow_up"][0]["action"], "clear_send_hold")
        self.assertEqual(body["next_action"]["action"], "clear_send_hold")
        self.assertEqual(body["source_plan_status"], "blocked_delivery")
        self.assertEqual(body["source_delivery_status"], "withheld")
        self.assertEqual(body["source_send_decision"]["decision"], "do_not_send")
        self.assertEqual(body["source_customer_claim_status"], "not_claimable")
        self.assertEqual(body["source_launch_decision"]["status"], "hold_launch")
        self.assertIn(
            "AI Improvement Launch Customer Communication Delivery Unblock Verification Report",
            body["verification_markdown"],
        )

    def test_ai_improvement_launch_customer_communication_delivery_send_authorization_packet_holds_failed_verification(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get(
            "/agents/ai-improvement-launch-customer-communication-delivery-send-authorization-packet?owner_id=customer-a"
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_launch_customer_communication_delivery_send_authorization_packet")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["authorization_status"], "hold_send")
        self.assertEqual(body["risk_level"], "high")
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertEqual(body["authorization_decision"]["decision"], "do_not_send")
        self.assertTrue(any(item["requirement"] == "unblock_verification_passed" and item["status"] == "blocked" for item in body["send_requirements"]))
        self.assertTrue(any(item["reason"] == "proof_gate:send_decision" for item in body["blocked_reasons"]))
        self.assertEqual(body["authorized_payload"]["status"], "withheld")
        self.assertFalse(body["authorized_payload"]["customer_facing"])
        self.assertEqual(body["next_action"]["action"], "clear_send_hold")
        self.assertEqual(body["source_verification_status"], "failed")
        self.assertEqual(body["source_delivery_status"], "withheld")
        self.assertEqual(body["source_send_decision"]["decision"], "do_not_send")
        self.assertEqual(body["source_customer_claim_status"], "not_claimable")
        self.assertEqual(body["source_launch_decision"]["status"], "hold_launch")
        self.assertIn(
            "AI Improvement Launch Customer Communication Delivery Send Authorization Packet",
            body["authorization_markdown"],
        )

    def test_ai_improvement_launch_customer_communication_delivery_send_authorization_monitor_tracks_hold(self) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get(
            "/agents/ai-improvement-launch-customer-communication-delivery-send-authorization-monitor?owner_id=customer-a"
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_launch_customer_communication_delivery_send_authorization_monitor")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["status"], "held")
        self.assertEqual(body["risk_level"], "high")
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertEqual(body["authorization_progress"]["status"], "hold_send")
        self.assertEqual(body["authorization_progress"]["decision"], "do_not_send")
        self.assertEqual(body["requirements_status"]["status"], "blocked")
        self.assertEqual(body["requirements_status"]["blocked_count"], 3)
        self.assertEqual(body["blocked_reason_status"]["status"], "blocked")
        self.assertEqual(body["payload_status"]["status"], "withheld")
        self.assertFalse(body["payload_status"]["customer_facing"])
        self.assertTrue(any(item["blocker"] == "send_authorization_held" for item in body["blockers"]))
        self.assertEqual(body["immediate_action"]["action"], "clear_send_hold")
        self.assertEqual(body["source_authorization_status"], "hold_send")
        self.assertEqual(body["source_authorization_decision"]["decision"], "do_not_send")
        self.assertEqual(body["source_verification_status"], "failed")
        self.assertEqual(body["source_delivery_status"], "withheld")
        self.assertEqual(body["source_customer_claim_status"], "not_claimable")
        self.assertEqual(body["source_launch_decision"]["status"], "hold_launch")
        self.assertIn(
            "AI Improvement Launch Customer Communication Delivery Send Authorization Monitor",
            body["monitor_markdown"],
        )

    def test_ai_improvement_launch_customer_communication_delivery_send_authorization_unblock_plan_turns_hold_into_tasks(
        self,
    ) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get(
            "/agents/ai-improvement-launch-customer-communication-delivery-send-authorization-unblock-plan?owner_id=customer-a"
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            body["kind"],
            "ai_improvement_launch_customer_communication_delivery_send_authorization_unblock_plan",
        )
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["plan_status"], "blocked_authorization")
        self.assertEqual(body["risk_level"], "high")
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertTrue(any(task["action"] == "clear_send_authorization_hold" for task in body["unblock_tasks"]))
        self.assertTrue(any(task["action"] == "resolve_send_blocked_reasons" for task in body["unblock_tasks"]))
        self.assertTrue(
            any(
                gate["gate"] == "authorization_decision" and gate["status"] == "blocked"
                for gate in body["authorization_gates"]
            )
        )
        self.assertTrue(
            any(
                item["criterion"] == "authorization_not_held" and item["status"] == "blocked"
                for item in body["exit_criteria"]
            )
        )
        self.assertEqual(body["recheck_plan"]["action"], "recheck_send_authorization_after_requirements_clear")
        self.assertEqual(body["immediate_action"]["action"], "clear_send_authorization_hold")
        self.assertEqual(body["source_monitor_status"], "held")
        self.assertEqual(body["source_authorization_status"], "hold_send")
        self.assertEqual(body["source_authorization_decision"]["decision"], "do_not_send")
        self.assertEqual(body["source_verification_status"], "failed")
        self.assertEqual(body["source_delivery_status"], "withheld")
        self.assertEqual(body["source_customer_claim_status"], "not_claimable")
        self.assertEqual(body["source_launch_decision"]["status"], "hold_launch")
        self.assertIn(
            "AI Improvement Launch Customer Communication Delivery Send Authorization Unblock Plan",
            body["plan_markdown"],
        )

    def test_ai_improvement_launch_customer_communication_delivery_send_authorization_unblock_verification_report_checks_tasks(
        self,
    ) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get(
            "/agents/ai-improvement-launch-customer-communication-delivery-send-authorization-unblock-verification-report?owner_id=customer-a"
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            body["kind"],
            "ai_improvement_launch_customer_communication_delivery_send_authorization_unblock_verification_report",
        )
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["verification_status"], "failed")
        self.assertEqual(body["risk_level"], "high")
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertTrue(
            any(item["check"] == "authorization_gate:authorization_decision" for item in body["failed_checks"])
        )
        self.assertTrue(
            any(item["check"] == "exit_criterion:authorization_not_held" for item in body["failed_checks"])
        )
        self.assertTrue(
            any(item["check"] == "unblock_task:clear_send_authorization_hold" for item in body["failed_checks"])
        )
        self.assertEqual(body["required_follow_up"][0]["action"], "clear_send_authorization_hold")
        self.assertEqual(body["next_action"]["action"], "clear_send_authorization_hold")
        self.assertEqual(body["source_plan_status"], "blocked_authorization")
        self.assertEqual(body["source_monitor_status"], "held")
        self.assertEqual(body["source_authorization_status"], "hold_send")
        self.assertEqual(body["source_authorization_decision"]["decision"], "do_not_send")
        self.assertEqual(body["source_verification_status"], "failed")
        self.assertEqual(body["source_delivery_status"], "withheld")
        self.assertEqual(body["source_customer_claim_status"], "not_claimable")
        self.assertEqual(body["source_launch_decision"]["status"], "hold_launch")
        self.assertIn(
            "AI Improvement Launch Customer Communication Delivery Send Authorization Unblock Verification Report",
            body["verification_markdown"],
        )

    def test_ai_improvement_launch_customer_communication_delivery_send_readiness_packet_blocks_failed_authorization(
        self,
    ) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get(
            "/agents/ai-improvement-launch-customer-communication-delivery-send-readiness-packet?owner_id=customer-a"
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["kind"], "ai_improvement_launch_customer_communication_delivery_send_readiness_packet")
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["readiness_status"], "not_ready")
        self.assertEqual(body["risk_level"], "high")
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertEqual(body["send_gate"]["status"], "blocked")
        self.assertEqual(body["send_gate"]["decision"], "do_not_send")
        self.assertEqual(body["customer_claim"]["status"], "withheld")
        self.assertFalse(body["customer_claim"]["customer_facing"])
        self.assertEqual(body["advisor_review"]["status"], "required")
        self.assertEqual(body["advisor_review"]["action"], "clear_send_authorization_hold")
        self.assertTrue(
            any(item["blocker"] == "send_authorization_unblock_verification_failed" for item in body["blockers"])
        )
        self.assertTrue(any(item["blocker"] == "send_not_authorized" for item in body["blockers"]))
        self.assertTrue(any(item["blocker"] == "customer_claim_not_supported" for item in body["blockers"]))
        self.assertEqual(body["immediate_action"]["action"], "clear_send_authorization_hold")
        self.assertEqual(body["source_verification_status"], "failed")
        self.assertEqual(body["source_plan_status"], "blocked_authorization")
        self.assertEqual(body["source_monitor_status"], "held")
        self.assertEqual(body["source_authorization_status"], "hold_send")
        self.assertEqual(body["source_authorization_decision"]["decision"], "do_not_send")
        self.assertEqual(body["source_delivery_status"], "withheld")
        self.assertEqual(body["source_customer_claim_status"], "not_claimable")
        self.assertEqual(body["source_launch_decision"]["status"], "hold_launch")
        self.assertIn(
            "AI Improvement Launch Customer Communication Delivery Send Readiness Packet",
            body["readiness_markdown"],
        )

    def test_ai_improvement_launch_customer_communication_delivery_send_readiness_review_packet_holds_blocked_send(
        self,
    ) -> None:
        self.create_customer_portfolio()
        self.client.post("/agents/action-queue?owner_id=customer-a&focus=telecom&evidence_limit=1&save=true")
        generated = self.client.post("/agents/advisor-outreach-draft?owner_id=customer-a&save=true").json()
        draft_id = generated["saved_draft_id"]
        self.client.patch(
            f"/agents/advisor-outreach-drafts/{draft_id}?owner_id=customer-a",
            json={"status": "approved", "review_notes": "Approved for delivery.", "reviewer": "advisor-a"},
        )
        packet = self.client.post(
            f"/agents/advisor-outreach-drafts/{draft_id}/delivery-packet?owner_id=customer-a&save=true"
        )
        delivery_id = packet.json()["saved_delivery_id"]
        self.client.patch(
            f"/agents/advisor-outreach-deliveries/{delivery_id}?owner_id=customer-a",
            json={"status": "delivered", "delivery_notes": "Shared during review call.", "delivered_by": "advisor-a"},
        )
        self.client.post(
            f"/agents/advisor-outreach-deliveries/{delivery_id}/outcome?owner_id=customer-a",
            json={
                "response_text": "Customer is interested and asked to schedule a meeting next week.",
                "follow_up_due_at": "2026-05-29T15:00:00Z",
                "recorded_by": "advisor-a",
            },
        )

        response = self.client.get(
            "/agents/ai-improvement-launch-customer-communication-delivery-send-readiness-review-packet?owner_id=customer-a"
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            body["kind"],
            "ai_improvement_launch_customer_communication_delivery_send_readiness_review_packet",
        )
        self.assertEqual(body["experiment_id"], "experiment:scale_top_pattern")
        self.assertEqual(body["review_status"], "hold_send")
        self.assertEqual(body["risk_level"], "high")
        self.assertEqual(body["work_item"]["work_item_id"], "backlog:clear_scale_blocker")
        self.assertEqual(body["send_decision"]["decision"], "do_not_send")
        self.assertTrue(
            any(item["approval"] == "advisor_final_review" and item["status"] == "blocked" for item in body["required_approvals"])
        )
        self.assertTrue(
            any(item["approval"] == "manager_send_release" and item["status"] == "blocked" for item in body["required_approvals"])
        )
        self.assertTrue(any(item["blocker"] == "send_not_authorized" for item in body["send_blockers"]))
        self.assertTrue(any(item["blocker"] == "customer_claim_not_supported" for item in body["send_blockers"]))
        self.assertEqual(body["approved_payload"]["status"], "withheld")
        self.assertFalse(body["approved_payload"]["customer_facing"])
        self.assertEqual(body["advisor_next_action"]["action"], "clear_send_authorization_hold")
        self.assertEqual(body["source_readiness_status"], "not_ready")
        self.assertEqual(body["source_send_gate"]["decision"], "do_not_send")
        self.assertEqual(body["source_customer_claim"]["status"], "withheld")
        self.assertEqual(body["source_advisor_review"]["status"], "required")
        self.assertEqual(body["source_verification_status"], "failed")
        self.assertEqual(body["source_authorization_status"], "hold_send")
        self.assertEqual(body["source_authorization_decision"]["decision"], "do_not_send")
        self.assertEqual(body["source_delivery_status"], "withheld")
        self.assertEqual(body["source_customer_claim_status"], "not_claimable")
        self.assertEqual(body["source_launch_decision"]["status"], "hold_launch")
        self.assertIn(
            "AI Improvement Launch Customer Communication Delivery Send Readiness Review Packet",
            body["review_markdown"],
        )


if __name__ == "__main__":
    unittest.main()
