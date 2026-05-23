"""Advisor action queue generated from customer follow-up packs."""

from __future__ import annotations

import json
import hashlib
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from system.ai.advisor_followup import build_advisor_followup
from system.api.market_service import MarketDataUnavailable, MarketRecordNotFound, table_exists


VALID_TASK_STATUSES = {"open", "blocked", "completed", "deferred"}
ACTIVE_TASK_STATUSES = ("open", "blocked")
VALID_ESCALATION_REVIEW_STATUSES = {"acknowledged", "snoozed", "resolved", "needs_followup"}
VALID_ESCALATION_NOTIFICATION_STATUSES = {"prepared", "sent", "skipped", "failed"}
VALID_DELIVERY_INCIDENT_TYPES = {
    "deadletter",
    "expired_claim",
    "expiring_claim",
    "retry_ready",
    "stale_prepared",
}
VALID_DELIVERY_INCIDENT_REVIEW_STATUSES = {
    "acknowledged",
    "assigned",
    "resolved",
    "snoozed",
    "needs_followup",
}
VALID_DELIVERY_INCIDENT_FOLLOW_UP_STATUSES = {"overdue", "due_soon", "future", "missing"}
VALID_DELIVERY_INCIDENT_ACTION_IDS = {
    "assign_incident",
    "claim_delivery",
    "release_delivery_claim",
    "requeue_deadletter",
    "resolve_incident",
    "review_follow_up",
}


def build_action_queue(
    conn: sqlite3.Connection,
    owner_id: str | None,
    index_dir: Path,
    focus: str | None = None,
    max_items: int = 2,
    evidence_limit: int = 1,
    persist_screeners: bool = False,
    save: bool = False,
    title: str | None = None,
) -> dict[str, Any]:
    followup = build_advisor_followup(
        conn,
        owner_id,
        index_dir,
        focus=focus,
        max_items=max_items,
        evidence_limit=evidence_limit,
        persist_screeners=persist_screeners,
    )
    tasks = _tasks_from_followup(followup)
    queue = {
        "kind": "advisor_action_queue",
        "owner_id": followup["owner_id"],
        "focus": focus,
        "tasks": tasks,
        "task_count": len(tasks),
        "blocked_count": sum(1 for task in tasks if task["status"] == "blocked"),
        "source_followup": followup,
        "queue_markdown": _markdown(followup["owner_id"], tasks),
    }
    if save:
        saved = save_action_queue(conn, queue, focus=focus, title=title)
        queue["saved_queue_id"] = saved["queue_id"]
        queue["saved_status"] = saved["status"]
    return queue


def save_action_queue(
    conn: sqlite3.Connection,
    queue: dict[str, Any],
    focus: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    owner_id = clean_owner_id(queue.get("owner_id"))
    now = now_utc()
    tasks = queue.get("tasks", [])
    counts = _counts(tasks)
    cursor = conn.execute(
        """
        INSERT INTO advisor_action_queues (
            owner_id, title, focus, status, task_count, open_task_count,
            blocked_task_count, completed_task_count, source_followup_json,
            queue_markdown, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            owner_id,
            clean_title(title, owner_id),
            focus,
            _queue_status(counts),
            counts["task_count"],
            counts["open_task_count"],
            counts["blocked_task_count"],
            counts["completed_task_count"],
            json.dumps(queue.get("source_followup", {}), sort_keys=True),
            queue.get("queue_markdown", _markdown(owner_id, tasks)),
            now,
            now,
        ),
    )
    queue_id = int(cursor.lastrowid)
    for task in tasks:
        conn.execute(
            """
            INSERT INTO advisor_action_queue_tasks (
                queue_id, task_id, title, urgency, status, rationale,
                completion_criteria, evidence_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                queue_id,
                task["task_id"],
                task["title"],
                task["urgency"],
                task["status"],
                task["rationale"],
                task["completion_criteria"],
                json.dumps(task.get("evidence", {}), sort_keys=True),
                now,
                now,
            ),
        )
    conn.commit()
    return get_action_queue(conn, queue_id, owner_id)


def list_action_queues(
    conn: sqlite3.Connection,
    owner_id: str | None,
    status: str | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    owner = clean_owner_id(owner_id)
    params: list[Any] = [owner]
    status_clause = ""
    if status:
        status_clause = "AND status = ?"
        params.append(status)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT *
        FROM advisor_action_queues
        WHERE owner_id = ?
        {status_clause}
        ORDER BY updated_at DESC, queue_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return {
        "data": [_queue_summary(dict(row)) for row in rows],
        "metadata": {"owner_id": owner, "status": status, "result_count": len(rows)},
    }


def summarize_action_queues(conn: sqlite3.Connection, owner_id: str | None) -> dict[str, Any]:
    require_action_queue_tables(conn)
    owner = clean_owner_id(owner_id)
    totals = conn.execute(
        """
        SELECT
            COUNT(*) AS queue_count,
            COALESCE(SUM(CASE WHEN status IN ('open', 'blocked') THEN 1 ELSE 0 END), 0) AS active_queue_count,
            COALESCE(SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END), 0) AS completed_queue_count,
            COALESCE(SUM(task_count), 0) AS task_count,
            COALESCE(SUM(open_task_count), 0) AS open_task_count,
            COALESCE(SUM(blocked_task_count), 0) AS blocked_task_count,
            COALESCE(SUM(completed_task_count), 0) AS completed_task_count,
            MAX(updated_at) AS last_updated_at
        FROM advisor_action_queues
        WHERE owner_id = ?
        """,
        (owner,),
    ).fetchone()
    status_rows = conn.execute(
        """
        SELECT
            status,
            COUNT(*) AS queue_count,
            COALESCE(SUM(task_count), 0) AS task_count,
            COALESCE(SUM(open_task_count), 0) AS open_task_count,
            COALESCE(SUM(blocked_task_count), 0) AS blocked_task_count,
            COALESCE(SUM(completed_task_count), 0) AS completed_task_count,
            MAX(updated_at) AS last_updated_at
        FROM advisor_action_queues
        WHERE owner_id = ?
        GROUP BY status
        ORDER BY
            CASE status WHEN 'open' THEN 0 WHEN 'blocked' THEN 1 WHEN 'completed' THEN 2 ELSE 3 END,
            status
        """,
        (owner,),
    ).fetchall()
    focus_rows = conn.execute(
        """
        SELECT
            COALESCE(NULLIF(focus, ''), 'general') AS focus_name,
            COUNT(*) AS queue_count,
            COALESCE(SUM(task_count), 0) AS task_count,
            COALESCE(SUM(open_task_count), 0) AS open_task_count,
            COALESCE(SUM(blocked_task_count), 0) AS blocked_task_count,
            COALESCE(SUM(completed_task_count), 0) AS completed_task_count,
            MAX(updated_at) AS last_updated_at
        FROM advisor_action_queues
        WHERE owner_id = ? AND status IN ('open', 'blocked')
        GROUP BY COALESCE(NULLIF(focus, ''), 'general')
        ORDER BY
            COALESCE(SUM(open_task_count), 0) + COALESCE(SUM(blocked_task_count), 0) DESC,
            MAX(updated_at) DESC
        """,
        (owner,),
    ).fetchall()
    urgency_rows = conn.execute(
        """
        SELECT
            tasks.urgency,
            COALESCE(SUM(CASE WHEN tasks.status = 'open' THEN 1 ELSE 0 END), 0) AS open_task_count,
            COALESCE(SUM(CASE WHEN tasks.status = 'blocked' THEN 1 ELSE 0 END), 0) AS blocked_task_count
        FROM advisor_action_queue_tasks AS tasks
        JOIN advisor_action_queues AS queues ON queues.queue_id = tasks.queue_id
        WHERE queues.owner_id = ?
          AND queues.status IN ('open', 'blocked')
          AND tasks.status IN ('open', 'blocked')
        GROUP BY tasks.urgency
        ORDER BY
            CASE tasks.urgency WHEN 'high' THEN 0 WHEN 'medium' THEN 1 WHEN 'low' THEN 2 ELSE 3 END,
            tasks.urgency
        """,
        (owner,),
    ).fetchall()
    recent_rows = conn.execute(
        """
        SELECT *
        FROM advisor_action_queues
        WHERE owner_id = ?
        ORDER BY updated_at DESC, queue_id DESC
        LIMIT 5
        """,
        (owner,),
    ).fetchall()
    total_counts = _summary_counts(dict(totals))
    return {
        "kind": "advisor_action_queue_summary",
        "owner_id": owner,
        "totals": {
            **total_counts,
            "attention_task_count": total_counts["open_task_count"] + total_counts["blocked_task_count"],
        },
        "by_status": [_summary_counts(dict(row), include_status=True) for row in status_rows],
        "by_focus": [_focus_summary(dict(row)) for row in focus_rows],
        "task_urgency": [_urgency_summary(dict(row)) for row in urgency_rows],
        "recent_queues": [_queue_summary(dict(row)) for row in recent_rows],
        "metadata": {
            "owner_id": owner,
            "generated_at": now_utc(),
            "active_statuses": ["open", "blocked"],
        },
    }


def list_action_queue_tasks(
    conn: sqlite3.Connection,
    owner_id: str | None,
    status: str | None = "active",
    focus: str | None = None,
    urgency: str | None = None,
    assigned_to: str | None = None,
    due_before: str | None = None,
    due_after: str | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    owner = clean_owner_id(owner_id)
    statuses = _task_status_filter(status)
    where = ["queues.owner_id = ?"]
    params: list[Any] = [owner]
    if statuses:
        where.append(f"tasks.status IN ({', '.join('?' for _ in statuses)})")
        params.extend(statuses)
    cleaned_focus = (focus or "").strip()
    if cleaned_focus:
        if cleaned_focus == "general":
            where.append("(queues.focus IS NULL OR queues.focus = '')")
        else:
            where.append("queues.focus = ?")
            params.append(cleaned_focus)
    cleaned_urgency = (urgency or "").strip().lower()
    if cleaned_urgency:
        where.append("tasks.urgency = ?")
        params.append(cleaned_urgency)
    cleaned_assigned_to = _clean_assigned_to(assigned_to) if assigned_to is not None else None
    if assigned_to is not None:
        if cleaned_assigned_to is None:
            where.append("(tasks.assigned_to IS NULL OR tasks.assigned_to = '')")
        else:
            where.append("tasks.assigned_to = ?")
            params.append(cleaned_assigned_to)
    cleaned_due_before = _clean_due_at(due_before) if due_before is not None else None
    if cleaned_due_before is not None:
        where.append("tasks.due_at IS NOT NULL AND tasks.due_at <= ?")
        params.append(cleaned_due_before)
    cleaned_due_after = _clean_due_at(due_after) if due_after is not None else None
    if cleaned_due_after is not None:
        where.append("tasks.due_at IS NOT NULL AND tasks.due_at >= ?")
        params.append(cleaned_due_after)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT
            queues.queue_id,
            queues.owner_id,
            queues.title AS queue_title,
            queues.focus,
            queues.status AS queue_status,
            queues.updated_at AS queue_updated_at,
            tasks.saved_task_id,
            tasks.task_id,
            tasks.title AS task_title,
            tasks.urgency,
            tasks.status AS task_status,
            tasks.rationale,
            tasks.completion_criteria,
            tasks.notes,
            tasks.assigned_to,
            tasks.due_at,
            tasks.created_at AS task_created_at,
            tasks.updated_at AS task_updated_at,
            tasks.completed_at
        FROM advisor_action_queue_tasks AS tasks
        JOIN advisor_action_queues AS queues ON queues.queue_id = tasks.queue_id
        WHERE {" AND ".join(where)}
        ORDER BY
            CASE WHEN tasks.due_at IS NULL OR tasks.due_at = '' THEN 1 ELSE 0 END,
            tasks.due_at ASC,
            CASE tasks.status WHEN 'open' THEN 0 WHEN 'blocked' THEN 1 WHEN 'deferred' THEN 2 WHEN 'completed' THEN 3 ELSE 4 END,
            CASE tasks.urgency WHEN 'high' THEN 0 WHEN 'medium' THEN 1 WHEN 'low' THEN 2 ELSE 3 END,
            tasks.updated_at DESC,
            tasks.saved_task_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return {
        "kind": "advisor_action_queue_tasks",
        "owner_id": owner,
        "data": [_queue_task_summary(dict(row)) for row in rows],
        "metadata": {
            "owner_id": owner,
            "status": status or "all",
            "focus": cleaned_focus or None,
            "urgency": cleaned_urgency or None,
            "assigned_to": cleaned_assigned_to,
            "due_before": cleaned_due_before,
            "due_after": cleaned_due_after,
            "limit": limit,
            "result_count": len(rows),
        },
    }


def summarize_action_queue_task_workload(
    conn: sqlite3.Connection,
    owner_id: str | None,
    as_of: str | None = None,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    owner = clean_owner_id(owner_id)
    as_of_date = _clean_as_of_date(as_of)
    next_week_date = (datetime.fromisoformat(f"{as_of_date}T00:00:00") + timedelta(days=7)).date().isoformat()
    assignee_rows = conn.execute(
        """
        SELECT
            COALESCE(NULLIF(tasks.assigned_to, ''), 'unassigned') AS assignee,
            COUNT(*) AS task_count,
            COALESCE(SUM(CASE WHEN tasks.status = 'open' THEN 1 ELSE 0 END), 0) AS open_task_count,
            COALESCE(SUM(CASE WHEN tasks.status = 'blocked' THEN 1 ELSE 0 END), 0) AS blocked_task_count,
            COALESCE(SUM(CASE WHEN tasks.status = 'deferred' THEN 1 ELSE 0 END), 0) AS deferred_task_count,
            COALESCE(SUM(CASE WHEN tasks.urgency = 'high' THEN 1 ELSE 0 END), 0) AS high_urgency_task_count,
            COALESCE(SUM(CASE WHEN tasks.due_at IS NULL OR tasks.due_at = '' THEN 1 ELSE 0 END), 0) AS unscheduled_task_count,
            COALESCE(SUM(CASE WHEN tasks.due_at IS NOT NULL AND tasks.due_at != '' AND substr(tasks.due_at, 1, 10) < ? THEN 1 ELSE 0 END), 0) AS overdue_task_count,
            COALESCE(SUM(CASE WHEN tasks.due_at IS NOT NULL AND tasks.due_at != '' AND substr(tasks.due_at, 1, 10) = ? THEN 1 ELSE 0 END), 0) AS due_today_task_count,
            COALESCE(SUM(CASE WHEN tasks.due_at IS NOT NULL AND tasks.due_at != '' AND substr(tasks.due_at, 1, 10) > ? AND substr(tasks.due_at, 1, 10) <= ? THEN 1 ELSE 0 END), 0) AS due_next_7_days_task_count,
            MIN(CASE WHEN tasks.due_at IS NOT NULL AND tasks.due_at != '' THEN tasks.due_at ELSE NULL END) AS next_due_at
        FROM advisor_action_queue_tasks AS tasks
        JOIN advisor_action_queues AS queues ON queues.queue_id = tasks.queue_id
        WHERE queues.owner_id = ? AND tasks.status IN ('open', 'blocked', 'deferred')
        GROUP BY COALESCE(NULLIF(tasks.assigned_to, ''), 'unassigned')
        ORDER BY
            overdue_task_count DESC,
            due_today_task_count DESC,
            blocked_task_count DESC,
            task_count DESC,
            assignee
        """,
        (as_of_date, as_of_date, as_of_date, next_week_date, owner),
    ).fetchall()
    totals = _workload_totals([dict(row) for row in assignee_rows])
    return {
        "kind": "advisor_action_queue_task_workload",
        "owner_id": owner,
        "as_of": as_of_date,
        "due_window_end": next_week_date,
        "totals": totals,
        "by_assignee": [_workload_assignee_summary(dict(row)) for row in assignee_rows],
        "due_buckets": [
            {"bucket": "overdue", "task_count": totals["overdue_task_count"]},
            {"bucket": "due_today", "task_count": totals["due_today_task_count"]},
            {"bucket": "due_next_7_days", "task_count": totals["due_next_7_days_task_count"]},
            {"bucket": "unscheduled", "task_count": totals["unscheduled_task_count"]},
        ],
        "metadata": {
            "owner_id": owner,
            "active_statuses": ["open", "blocked", "deferred"],
            "generated_at": now_utc(),
        },
    }


def list_action_queue_task_escalations(
    conn: sqlite3.Connection,
    owner_id: str | None,
    as_of: str | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_review_table(conn)
    owner = clean_owner_id(owner_id)
    as_of_date = _clean_as_of_date(as_of)
    rows = conn.execute(
        """
        SELECT
            queues.queue_id,
            queues.owner_id,
            queues.title AS queue_title,
            queues.focus,
            queues.status AS queue_status,
            queues.updated_at AS queue_updated_at,
            tasks.saved_task_id,
            tasks.task_id,
            tasks.title AS task_title,
            tasks.urgency,
            tasks.status AS task_status,
            tasks.rationale,
            tasks.completion_criteria,
            tasks.notes,
            tasks.assigned_to,
            tasks.due_at,
            tasks.created_at AS task_created_at,
            tasks.updated_at AS task_updated_at,
            tasks.completed_at,
            reviews.review_id AS escalation_review_id,
            reviews.review_status AS escalation_review_status,
            reviews.reviewer AS escalation_review_reviewer,
            reviews.notes AS escalation_review_notes,
            reviews.snoozed_until AS escalation_review_snoozed_until,
            reviews.created_at AS escalation_review_created_at
        FROM advisor_action_queue_tasks AS tasks
        JOIN advisor_action_queues AS queues ON queues.queue_id = tasks.queue_id
        LEFT JOIN advisor_action_queue_escalation_reviews AS reviews
            ON reviews.review_id = (
                SELECT MAX(latest_reviews.review_id)
                FROM advisor_action_queue_escalation_reviews AS latest_reviews
                WHERE latest_reviews.owner_id = queues.owner_id
                    AND latest_reviews.queue_id = tasks.queue_id
                    AND latest_reviews.task_id = tasks.task_id
            )
        WHERE queues.owner_id = ?
            AND tasks.status IN ('open', 'blocked', 'deferred')
            AND (
                tasks.status = 'blocked'
                OR tasks.urgency = 'high'
                OR tasks.assigned_to IS NULL
                OR tasks.assigned_to = ''
                OR (
                    tasks.due_at IS NOT NULL
                    AND tasks.due_at != ''
                    AND substr(tasks.due_at, 1, 10) <= ?
                )
            )
        ORDER BY
            CASE
                WHEN tasks.due_at IS NOT NULL AND tasks.due_at != '' AND substr(tasks.due_at, 1, 10) < ? THEN 0
                WHEN tasks.status = 'blocked' THEN 1
                WHEN tasks.due_at IS NOT NULL AND tasks.due_at != '' AND substr(tasks.due_at, 1, 10) = ? THEN 2
                WHEN tasks.urgency = 'high' THEN 3
                WHEN tasks.assigned_to IS NULL OR tasks.assigned_to = '' THEN 4
                ELSE 5
            END,
            CASE WHEN tasks.due_at IS NULL OR tasks.due_at = '' THEN 1 ELSE 0 END,
            tasks.due_at ASC,
            CASE tasks.urgency WHEN 'high' THEN 0 WHEN 'medium' THEN 1 WHEN 'low' THEN 2 ELSE 3 END,
            tasks.updated_at DESC,
            tasks.saved_task_id DESC
        LIMIT ?
        """,
        (owner, as_of_date, as_of_date, as_of_date, limit),
    ).fetchall()
    escalations = [_task_escalation_summary(dict(row), as_of_date) for row in rows]
    return {
        "kind": "advisor_action_queue_task_escalations",
        "owner_id": owner,
        "as_of": as_of_date,
        "data": escalations,
        "metadata": {
            "owner_id": owner,
            "as_of": as_of_date,
            "active_statuses": ["open", "blocked", "deferred"],
            "limit": limit,
            "result_count": len(escalations),
            "by_severity": _escalation_counts(escalations, "severity"),
            "by_reason": _escalation_reason_counts(escalations),
            "generated_at": now_utc(),
        },
    }


def summarize_action_queue_task_escalations(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    as_of: str | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_review_table(conn)
    cleaned_owner = (owner_id or "").strip() or None
    as_of_date = _clean_as_of_date(as_of)
    where = [
        "tasks.status IN ('open', 'blocked', 'deferred')",
        """
        (
            tasks.status = 'blocked'
            OR tasks.urgency = 'high'
            OR tasks.assigned_to IS NULL
            OR tasks.assigned_to = ''
            OR (
                tasks.due_at IS NOT NULL
                AND tasks.due_at != ''
                AND substr(tasks.due_at, 1, 10) <= ?
            )
        )
        """,
    ]
    params: list[Any] = [as_of_date]
    if cleaned_owner:
        where.insert(0, "queues.owner_id = ?")
        params.insert(0, cleaned_owner)
    rows = conn.execute(
        f"""
        SELECT
            queues.queue_id,
            queues.owner_id,
            queues.title AS queue_title,
            queues.focus,
            queues.status AS queue_status,
            queues.updated_at AS queue_updated_at,
            tasks.saved_task_id,
            tasks.task_id,
            tasks.title AS task_title,
            tasks.urgency,
            tasks.status AS task_status,
            tasks.rationale,
            tasks.completion_criteria,
            tasks.notes,
            tasks.assigned_to,
            tasks.due_at,
            tasks.created_at AS task_created_at,
            tasks.updated_at AS task_updated_at,
            tasks.completed_at,
            reviews.review_id AS escalation_review_id,
            reviews.review_status AS escalation_review_status,
            reviews.reviewer AS escalation_review_reviewer,
            reviews.notes AS escalation_review_notes,
            reviews.snoozed_until AS escalation_review_snoozed_until,
            reviews.created_at AS escalation_review_created_at
        FROM advisor_action_queue_tasks AS tasks
        JOIN advisor_action_queues AS queues ON queues.queue_id = tasks.queue_id
        LEFT JOIN advisor_action_queue_escalation_reviews AS reviews
            ON reviews.review_id = (
                SELECT MAX(latest_reviews.review_id)
                FROM advisor_action_queue_escalation_reviews AS latest_reviews
                WHERE latest_reviews.owner_id = queues.owner_id
                    AND latest_reviews.queue_id = tasks.queue_id
                    AND latest_reviews.task_id = tasks.task_id
            )
        WHERE {" AND ".join(where)}
        ORDER BY
            CASE
                WHEN tasks.due_at IS NOT NULL AND tasks.due_at != '' AND substr(tasks.due_at, 1, 10) < ? THEN 0
                WHEN tasks.status = 'blocked' THEN 1
                WHEN tasks.due_at IS NOT NULL AND tasks.due_at != '' AND substr(tasks.due_at, 1, 10) = ? THEN 2
                WHEN tasks.urgency = 'high' THEN 3
                WHEN tasks.assigned_to IS NULL OR tasks.assigned_to = '' THEN 4
                ELSE 5
            END,
            CASE WHEN tasks.due_at IS NULL OR tasks.due_at = '' THEN 1 ELSE 0 END,
            tasks.due_at ASC,
            CASE tasks.urgency WHEN 'high' THEN 0 WHEN 'medium' THEN 1 WHEN 'low' THEN 2 ELSE 3 END,
            tasks.updated_at DESC,
            tasks.saved_task_id DESC
        """,
        (*params, as_of_date, as_of_date),
    ).fetchall()
    escalations = [_task_escalation_summary(dict(row), as_of_date) for row in rows]
    owner_summaries = _owner_escalation_summaries(escalations)
    return {
        "kind": "advisor_action_queue_task_escalation_summary",
        "owner_id": cleaned_owner,
        "as_of": as_of_date,
        "totals": _escalation_summary_totals(escalations, owner_summaries),
        "owners": owner_summaries[:limit],
        "metadata": {
            "scope": "owner" if cleaned_owner else "book",
            "owner_id": cleaned_owner,
            "as_of": as_of_date,
            "owner_limit": limit,
            "owner_count": len(owner_summaries),
            "active_statuses": ["open", "blocked", "deferred"],
            "by_severity": _escalation_counts(escalations, "severity"),
            "by_reason": _escalation_reason_counts(escalations),
            "generated_at": now_utc(),
        },
    }


def list_action_queue_task_escalation_inbox(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    as_of: str | None = None,
    severity: str | None = None,
    inbox_status: str | None = None,
    assigned_to: str | None = None,
    focus: str | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_review_table(conn)
    cleaned_owner = (owner_id or "").strip() or None
    as_of_date = _clean_as_of_date(as_of)
    cleaned_severities = _clean_escalation_filter_values(severity, {"critical", "high", "medium", "low"}, "severity")
    cleaned_inbox_statuses = _clean_escalation_filter_values(
        inbox_status,
        {"new", "active_review", "needs_followup", "snooze_expired"},
        "inbox_status",
    )
    cleaned_assigned_to = _clean_assigned_to(assigned_to) if assigned_to is not None else None
    cleaned_focus = (focus or "").strip()
    where = [
        "tasks.status IN ('open', 'blocked', 'deferred')",
        """
        (
            tasks.status = 'blocked'
            OR tasks.urgency = 'high'
            OR tasks.assigned_to IS NULL
            OR tasks.assigned_to = ''
            OR (
                tasks.due_at IS NOT NULL
                AND tasks.due_at != ''
                AND substr(tasks.due_at, 1, 10) <= ?
            )
        )
        """,
    ]
    params: list[Any] = [as_of_date]
    if cleaned_owner:
        where.insert(0, "queues.owner_id = ?")
        params.insert(0, cleaned_owner)
    rows = conn.execute(
        f"""
        SELECT
            queues.queue_id,
            queues.owner_id,
            queues.title AS queue_title,
            queues.focus,
            queues.status AS queue_status,
            queues.updated_at AS queue_updated_at,
            tasks.saved_task_id,
            tasks.task_id,
            tasks.title AS task_title,
            tasks.urgency,
            tasks.status AS task_status,
            tasks.rationale,
            tasks.completion_criteria,
            tasks.notes,
            tasks.assigned_to,
            tasks.due_at,
            tasks.created_at AS task_created_at,
            tasks.updated_at AS task_updated_at,
            tasks.completed_at,
            reviews.review_id AS escalation_review_id,
            reviews.review_status AS escalation_review_status,
            reviews.reviewer AS escalation_review_reviewer,
            reviews.notes AS escalation_review_notes,
            reviews.snoozed_until AS escalation_review_snoozed_until,
            reviews.created_at AS escalation_review_created_at
        FROM advisor_action_queue_tasks AS tasks
        JOIN advisor_action_queues AS queues ON queues.queue_id = tasks.queue_id
        LEFT JOIN advisor_action_queue_escalation_reviews AS reviews
            ON reviews.review_id = (
                SELECT MAX(latest_reviews.review_id)
                FROM advisor_action_queue_escalation_reviews AS latest_reviews
                WHERE latest_reviews.owner_id = queues.owner_id
                    AND latest_reviews.queue_id = tasks.queue_id
                    AND latest_reviews.task_id = tasks.task_id
            )
        WHERE {" AND ".join(where)}
        ORDER BY
            CASE
                WHEN tasks.due_at IS NOT NULL AND tasks.due_at != '' AND substr(tasks.due_at, 1, 10) < ? THEN 0
                WHEN tasks.status = 'blocked' THEN 1
                WHEN tasks.due_at IS NOT NULL AND tasks.due_at != '' AND substr(tasks.due_at, 1, 10) = ? THEN 2
                WHEN tasks.urgency = 'high' THEN 3
                WHEN tasks.assigned_to IS NULL OR tasks.assigned_to = '' THEN 4
                ELSE 5
            END,
            CASE WHEN tasks.due_at IS NULL OR tasks.due_at = '' THEN 1 ELSE 0 END,
            tasks.due_at ASC,
            CASE tasks.urgency WHEN 'high' THEN 0 WHEN 'medium' THEN 1 WHEN 'low' THEN 2 ELSE 3 END,
            tasks.updated_at DESC,
            tasks.saved_task_id DESC
        """,
        (*params, as_of_date, as_of_date),
    ).fetchall()
    escalations = [_task_escalation_summary(dict(row), as_of_date) for row in rows]
    actionable: list[dict[str, Any]] = []
    excluded_snoozed_count = 0
    excluded_resolved_count = 0
    for item in escalations:
        inbox_status = _escalation_inbox_status(item, as_of_date)
        item["inbox_status"] = inbox_status
        if inbox_status == "resolved":
            excluded_resolved_count += 1
            continue
        if inbox_status == "snoozed":
            excluded_snoozed_count += 1
            continue
        actionable.append(item)
    filtered_actionable = [
        item
        for item in actionable
        if _matches_escalation_inbox_filters(
            item,
            severities=cleaned_severities,
            inbox_statuses=cleaned_inbox_statuses,
            assigned_to=cleaned_assigned_to,
            assigned_to_filter_present=assigned_to is not None,
            focus=cleaned_focus,
        )
    ]
    owner_summaries = _owner_escalation_summaries(filtered_actionable)
    return {
        "kind": "advisor_action_queue_task_escalation_inbox",
        "owner_id": cleaned_owner,
        "as_of": as_of_date,
        "data": filtered_actionable[:limit],
        "owners": owner_summaries[:limit],
        "totals": {
            **_escalation_inbox_totals(filtered_actionable),
            "excluded_snoozed_task_count": excluded_snoozed_count,
            "excluded_resolved_task_count": excluded_resolved_count,
            "owner_count": len(owner_summaries),
        },
        "metadata": {
            "scope": "owner" if cleaned_owner else "book",
            "owner_id": cleaned_owner,
            "as_of": as_of_date,
            "severity": cleaned_severities,
            "inbox_status": cleaned_inbox_statuses,
            "assigned_to": cleaned_assigned_to,
            "focus": cleaned_focus or None,
            "limit": limit,
            "result_count": min(len(filtered_actionable), limit),
            "actionable_task_count": len(filtered_actionable),
            "unfiltered_actionable_task_count": len(actionable),
            "excluded_snoozed_task_count": excluded_snoozed_count,
            "excluded_resolved_task_count": excluded_resolved_count,
            "generated_at": now_utc(),
        },
    }


def build_action_queue_task_escalation_inbox_notification(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    as_of: str | None = None,
    severity: str | None = None,
    inbox_status: str | None = None,
    assigned_to: str | None = None,
    focus: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    inbox = list_action_queue_task_escalation_inbox(
        conn,
        owner_id=owner_id,
        as_of=as_of,
        severity=severity,
        inbox_status=inbox_status,
        assigned_to=assigned_to,
        focus=focus,
        limit=limit,
    )
    items = [_escalation_notification_item(item) for item in inbox["data"]]
    summary = _escalation_notification_summary(inbox, items)
    return {
        "kind": "advisor_action_queue_task_escalation_notification",
        "owner_id": inbox["owner_id"],
        "as_of": inbox["as_of"],
        "summary": summary,
        "items": items,
        "notification_markdown": _escalation_notification_markdown(summary, items),
        "metadata": {
            "scope": inbox["metadata"]["scope"],
            "owner_id": inbox["metadata"]["owner_id"],
            "as_of": inbox["metadata"]["as_of"],
            "severity": inbox["metadata"]["severity"],
            "inbox_status": inbox["metadata"]["inbox_status"],
            "assigned_to": inbox["metadata"]["assigned_to"],
            "focus": inbox["metadata"]["focus"],
            "limit": limit,
            "item_count": len(items),
            "source_actionable_task_count": inbox["metadata"]["actionable_task_count"],
            "source_unfiltered_actionable_task_count": inbox["metadata"]["unfiltered_actionable_task_count"],
            "excluded_snoozed_task_count": inbox["metadata"]["excluded_snoozed_task_count"],
            "excluded_resolved_task_count": inbox["metadata"]["excluded_resolved_task_count"],
            "generated_at": now_utc(),
        },
    }


def save_action_queue_task_escalation_inbox_notification(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    as_of: str | None = None,
    severity: str | None = None,
    inbox_status: str | None = None,
    assigned_to: str | None = None,
    focus: str | None = None,
    limit: int = 10,
    channel: str | None = None,
    recipient: str | None = None,
    status: str | None = "prepared",
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_notification_table(conn)
    payload = build_action_queue_task_escalation_inbox_notification(
        conn,
        owner_id=owner_id,
        as_of=as_of,
        severity=severity,
        inbox_status=inbox_status,
        assigned_to=assigned_to,
        focus=focus,
        limit=limit,
    )
    cleaned_channel = (channel or "manager_inbox").strip() or "manager_inbox"
    cleaned_recipient = (recipient or "").strip() or None
    cleaned_status = _clean_escalation_notification_status(status)
    key = (idempotency_key or "").strip() or _escalation_notification_idempotency_key(
        payload, cleaned_channel, cleaned_recipient
    )
    existing = _escalation_notification_by_key(conn, key)
    if existing:
        return {
            "kind": "advisor_action_queue_task_escalation_notification_log",
            "owner_id": existing["owner_id"],
            "data": {
                "notification": existing,
                "payload": json.loads(existing["payload_json"]),
            },
            "metadata": {
                "idempotency_key": key,
                "created": False,
                "notification_id": existing["notification_id"],
            },
        }

    now = now_utc()
    cursor = conn.execute(
        """
        INSERT INTO advisor_action_queue_escalation_notifications (
            owner_id, as_of, channel, recipient, status, idempotency_key,
            filter_json, item_count, payload_json, delivery_notes, delivered_at, delivery_retry_after,
            created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload["owner_id"],
            payload["as_of"],
            cleaned_channel,
            cleaned_recipient,
            cleaned_status,
            key,
            json.dumps(_escalation_notification_filter_metadata(payload), sort_keys=True),
            payload["summary"]["item_count"],
            json.dumps(payload, sort_keys=True),
            None,
            now if cleaned_status == "sent" else None,
            None,
            now,
            now,
        ),
    )
    conn.commit()
    notification = _escalation_notification_by_id(conn, int(cursor.lastrowid))
    return {
        "kind": "advisor_action_queue_task_escalation_notification_log",
        "owner_id": notification["owner_id"],
        "data": {
            "notification": notification,
            "payload": payload,
        },
        "metadata": {
            "idempotency_key": key,
            "created": True,
            "notification_id": notification["notification_id"],
        },
    }


def update_action_queue_task_escalation_notification(
    conn: sqlite3.Connection,
    notification_id: int,
    status: str | None = None,
    delivery_notes: str | None = None,
    delivered_at: str | None = None,
    owner_id: str | None = None,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_notification_table(conn)
    current = _escalation_notification_by_id(conn, notification_id)
    cleaned_owner = (owner_id or "").strip() or None
    if cleaned_owner and current["owner_id"] != cleaned_owner:
        raise MarketRecordNotFound(f"No advisor action queue escalation notification found for id {notification_id}")
    status_changed = status is not None
    cleaned_status = _clean_escalation_notification_status(status) if status_changed else current["status"]
    delivered_changed = delivered_at is not None
    cleaned_delivered_at = _clean_iso_datetime(delivered_at, "delivered_at") if delivered_changed else current.get("delivered_at")
    if status_changed and cleaned_status == "sent" and not cleaned_delivered_at:
        cleaned_delivered_at = now_utc()
        delivered_changed = True
    if not status_changed and delivery_notes is None and not delivered_changed:
        raise ValueError("At least one notification update field is required")
    now = now_utc()
    conn.execute(
        """
        UPDATE advisor_action_queue_escalation_notifications
        SET status = ?,
            delivery_notes = COALESCE(?, delivery_notes),
            delivered_at = CASE WHEN ? THEN ? ELSE delivered_at END,
            updated_at = ?
        WHERE notification_id = ?
        """,
        (
            cleaned_status,
            delivery_notes,
            delivered_changed,
            cleaned_delivered_at,
            now,
            notification_id,
        ),
    )
    conn.commit()
    notification = _escalation_notification_by_id(conn, notification_id)
    return {
        "kind": "advisor_action_queue_task_escalation_notification_update",
        "owner_id": notification["owner_id"],
        "data": {"notification": notification},
        "metadata": {
            "notification_id": notification_id,
            "status": cleaned_status,
            "delivery_notes_applied": delivery_notes is not None,
            "delivered_at": cleaned_delivered_at,
        },
    }


def list_action_queue_task_escalation_notifications(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    channel: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_notification_table(conn)
    _ensure_action_queue_escalation_notification_incident_review_table(conn)
    cleaned_owner = (owner_id or "").strip() or None
    cleaned_channel = (channel or "").strip() or None
    cleaned_status = _clean_escalation_notification_status(status) if status is not None else None
    where: list[str] = []
    params: list[Any] = []
    if cleaned_owner:
        where.append("owner_id = ?")
        params.append(cleaned_owner)
    if cleaned_channel:
        where.append("channel = ?")
        params.append(cleaned_channel)
    if cleaned_status:
        where.append("status = ?")
        params.append(cleaned_status)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT *
        FROM advisor_action_queue_escalation_notifications
        {where_sql}
        ORDER BY created_at DESC, notification_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return {
        "kind": "advisor_action_queue_task_escalation_notification_logs",
        "owner_id": cleaned_owner,
        "data": [_escalation_notification_record_from_row(dict(row)) for row in rows],
        "metadata": {
            "owner_id": cleaned_owner,
            "channel": cleaned_channel,
            "status": cleaned_status,
            "limit": limit,
            "result_count": len(rows),
        },
    }


def summarize_action_queue_task_escalation_notification_delivery(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    channel: str | None = None,
    stale_after_minutes: int = 60,
    recent_limit: int = 5,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_notification_table(conn)
    cleaned_owner = (owner_id or "").strip() or None
    cleaned_channel = (channel or "").strip() or None
    if stale_after_minutes < 1:
        raise ValueError("stale_after_minutes must be at least 1")
    if recent_limit < 0:
        raise ValueError("recent_limit must be at least 0")

    where: list[str] = []
    params: list[Any] = []
    if cleaned_owner:
        where.append("owner_id = ?")
        params.append(cleaned_owner)
    if cleaned_channel:
        where.append("channel = ?")
        params.append(cleaned_channel)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    status_rows = conn.execute(
        f"""
        SELECT
            status,
            COUNT(*) AS notification_count,
            MAX(COALESCE(updated_at, created_at)) AS last_updated_at
        FROM advisor_action_queue_escalation_notifications
        {where_sql}
        GROUP BY status
        ORDER BY status
        """,
        params,
    ).fetchall()
    channel_rows = conn.execute(
        f"""
        SELECT
            channel,
            status,
            COUNT(*) AS notification_count,
            MAX(COALESCE(updated_at, created_at)) AS last_updated_at
        FROM advisor_action_queue_escalation_notifications
        {where_sql}
        GROUP BY channel, status
        ORDER BY channel, status
        """,
        params,
    ).fetchall()
    cutoff = (
        datetime.now(UTC).replace(microsecond=0) - timedelta(minutes=stale_after_minutes)
    ).isoformat().replace("+00:00", "Z")
    now = now_utc()
    stale_where = where + ["status = ?", "created_at <= ?"]
    stale_params = params + ["prepared", cutoff]
    stale_row = conn.execute(
        f"""
        SELECT COUNT(*) AS notification_count
        FROM advisor_action_queue_escalation_notifications
        WHERE {' AND '.join(stale_where)}
        """,
        stale_params,
    ).fetchone()
    retry_wait_where = where + ["status = ?", "delivery_retry_after > ?"]
    retry_wait_params = params + ["failed", now]
    retry_wait_row = conn.execute(
        f"""
        SELECT COUNT(*) AS notification_count
        FROM advisor_action_queue_escalation_notifications
        WHERE {' AND '.join(retry_wait_where)}
        """,
        retry_wait_params,
    ).fetchone()
    exhausted_where = where + ["status = ?", "delivery_exhausted_at IS NOT NULL"]
    exhausted_params = params + ["failed"]
    exhausted_row = conn.execute(
        f"""
        SELECT COUNT(*) AS notification_count
        FROM advisor_action_queue_escalation_notifications
        WHERE {' AND '.join(exhausted_where)}
        """,
        exhausted_params,
    ).fetchone()
    recent_where = where + ["status = ?"]
    recent_params = params + ["failed", recent_limit]
    recent_rows = conn.execute(
        f"""
        SELECT *
        FROM advisor_action_queue_escalation_notifications
        WHERE {' AND '.join(recent_where)}
        ORDER BY COALESCE(updated_at, created_at) DESC, notification_id DESC
        LIMIT ?
        """,
        recent_params,
    ).fetchall()

    status_counts = {status: 0 for status in sorted(VALID_ESCALATION_NOTIFICATION_STATUSES)}
    last_updated_at = None
    for row in status_rows:
        status_counts[row["status"]] = row["notification_count"]
        if row["last_updated_at"] and (
            last_updated_at is None or row["last_updated_at"] > last_updated_at
        ):
            last_updated_at = row["last_updated_at"]
    total_count = sum(status_counts.values())
    undelivered_count = status_counts["prepared"] + status_counts["failed"]
    return {
        "kind": "advisor_action_queue_task_escalation_notification_delivery_summary",
        "owner_id": cleaned_owner,
        "data": {
            "summary": {
                "total_count": total_count,
                "sent_count": status_counts["sent"],
                "prepared_count": status_counts["prepared"],
                "failed_count": status_counts["failed"],
                "skipped_count": status_counts["skipped"],
                "undelivered_count": undelivered_count,
                "stale_prepared_count": stale_row["notification_count"] if stale_row else 0,
                "retry_wait_count": retry_wait_row["notification_count"] if retry_wait_row else 0,
                "exhausted_count": exhausted_row["notification_count"] if exhausted_row else 0,
                "last_updated_at": last_updated_at,
            },
            "status_counts": status_counts,
            "channel_status_counts": [
                {
                    "channel": row["channel"],
                    "status": row["status"],
                    "notification_count": row["notification_count"],
                    "last_updated_at": row["last_updated_at"],
                }
                for row in channel_rows
            ],
            "recent_failures": [
                _escalation_notification_record_from_row(dict(row)) for row in recent_rows
            ],
        },
        "metadata": {
            "owner_id": cleaned_owner,
            "channel": cleaned_channel,
            "stale_after_minutes": stale_after_minutes,
            "stale_cutoff": cutoff,
            "recent_failure_count": len(recent_rows),
        },
    }


def summarize_action_queue_task_escalation_notification_delivery_control(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    channel: str | None = None,
    stale_after_minutes: int = 60,
    expiring_within_seconds: int = 300,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_notification_table(conn)
    _ensure_action_queue_escalation_notification_attempt_table(conn)
    _ensure_action_queue_escalation_notification_remediation_table(conn)
    _ensure_action_queue_escalation_notification_claim_release_table(conn)
    cleaned_owner = (owner_id or "").strip() or None
    cleaned_channel = (channel or "").strip() or None
    if stale_after_minutes < 1:
        raise ValueError("stale_after_minutes must be at least 1")
    if expiring_within_seconds < 1:
        raise ValueError("expiring_within_seconds must be at least 1")

    base_where: list[str] = []
    base_params: list[Any] = []
    if cleaned_owner:
        base_where.append("owner_id = ?")
        base_params.append(cleaned_owner)
    if cleaned_channel:
        base_where.append("channel = ?")
        base_params.append(cleaned_channel)

    now_dt = datetime.now(UTC).replace(microsecond=0)
    now = now_dt.isoformat().replace("+00:00", "Z")
    stale_cutoff = (now_dt - timedelta(minutes=stale_after_minutes)).isoformat().replace(
        "+00:00", "Z"
    )
    expiring_cutoff = (now_dt + timedelta(seconds=expiring_within_seconds)).isoformat().replace(
        "+00:00", "Z"
    )

    def count_from(table_name: str, extra_where: list[str], extra_params: list[Any]) -> int:
        where = base_where + extra_where
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        row = conn.execute(
            f"SELECT COUNT(*) AS item_count FROM {table_name} {where_sql}",
            base_params + extra_params,
        ).fetchone()
        return int(row["item_count"] if row else 0)

    queued_count = count_from(
        "advisor_action_queue_escalation_notifications",
        [
            "((status = 'failed' AND delivery_exhausted_at IS NULL "
            "AND (delivery_retry_after IS NULL OR delivery_retry_after <= ?)) "
            "OR (status = 'prepared' AND created_at <= ?))",
            "(delivery_claimed_until IS NULL OR delivery_claimed_until <= ?)",
        ],
        [now, stale_cutoff, now],
    )
    active_claim_count = count_from(
        "advisor_action_queue_escalation_notifications",
        [
            "delivery_claim_token IS NOT NULL",
            "delivery_claimed_until > ?",
        ],
        [expiring_cutoff],
    )
    expiring_claim_count = count_from(
        "advisor_action_queue_escalation_notifications",
        [
            "delivery_claim_token IS NOT NULL",
            "delivery_claimed_until > ?",
            "delivery_claimed_until <= ?",
        ],
        [now, expiring_cutoff],
    )
    expired_claim_count = count_from(
        "advisor_action_queue_escalation_notifications",
        [
            "delivery_claim_token IS NOT NULL",
            "delivery_claimed_until <= ?",
        ],
        [now],
    )
    retry_wait_count = count_from(
        "advisor_action_queue_escalation_notifications",
        ["status = ?", "delivery_retry_after > ?"],
        ["failed", now],
    )
    deadletter_count = count_from(
        "advisor_action_queue_escalation_notifications",
        ["status = ?", "delivery_exhausted_at IS NOT NULL"],
        ["failed"],
    )
    delivery_attempt_count = count_from(
        "advisor_action_queue_escalation_notification_attempts",
        [],
        [],
    )
    failed_attempt_count = count_from(
        "advisor_action_queue_escalation_notification_attempts",
        ["status = ?"],
        ["failed"],
    )
    claim_release_count = count_from(
        "advisor_action_queue_escalation_notification_claim_releases",
        [],
        [],
    )
    deadletter_remediation_count = count_from(
        "advisor_action_queue_escalation_notification_remediations",
        [],
        [],
    )

    worker_where = base_where + [
        "delivery_claim_token IS NOT NULL",
        "delivery_claimed_by IS NOT NULL",
        "delivery_claimed_until IS NOT NULL",
    ]
    worker_rows = conn.execute(
        f"""
        SELECT
            delivery_claimed_by AS claimed_by,
            COUNT(*) AS claim_count,
            SUM(CASE WHEN delivery_claimed_until <= ? THEN 1 ELSE 0 END) AS expired_count,
            SUM(CASE WHEN delivery_claimed_until > ? AND delivery_claimed_until <= ? THEN 1 ELSE 0 END)
                AS expiring_count,
            MIN(delivery_claimed_until) AS next_claim_expires_at
        FROM advisor_action_queue_escalation_notifications
        WHERE {' AND '.join(worker_where)}
        GROUP BY delivery_claimed_by
        ORDER BY expired_count DESC, expiring_count DESC, next_claim_expires_at ASC
        """,
        [now, now, expiring_cutoff, *base_params],
    ).fetchall()

    return {
        "kind": "advisor_action_queue_task_escalation_notification_delivery_control_summary",
        "owner_id": cleaned_owner,
        "data": {
            "summary": {
                "queued_count": queued_count,
                "active_claim_count": active_claim_count,
                "expiring_claim_count": expiring_claim_count,
                "expired_claim_count": expired_claim_count,
                "retry_wait_count": retry_wait_count,
                "deadletter_count": deadletter_count,
                "delivery_attempt_count": delivery_attempt_count,
                "failed_attempt_count": failed_attempt_count,
                "claim_release_count": claim_release_count,
                "deadletter_remediation_count": deadletter_remediation_count,
                "incident_count": expired_claim_count + deadletter_count,
            },
            "worker_claim_counts": [
                {
                    "claimed_by": row["claimed_by"],
                    "claim_count": row["claim_count"],
                    "expired_count": row["expired_count"] or 0,
                    "expiring_count": row["expiring_count"] or 0,
                    "next_claim_expires_at": row["next_claim_expires_at"],
                }
                for row in worker_rows
            ],
        },
        "metadata": {
            "owner_id": cleaned_owner,
            "channel": cleaned_channel,
            "stale_after_minutes": stale_after_minutes,
            "stale_cutoff": stale_cutoff,
            "expiring_within_seconds": expiring_within_seconds,
            "lease_expiring_cutoff": expiring_cutoff,
        },
    }


def list_action_queue_task_escalation_notification_delivery_incidents(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    channel: str | None = None,
    stale_after_minutes: int = 60,
    expiring_within_seconds: int = 300,
    limit: int = 25,
    include_suppressed: bool = False,
    assigned_to: str | None = None,
    follow_up_status: str | None = None,
    follow_up_within_hours: int = 24,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_notification_table(conn)
    cleaned_owner = (owner_id or "").strip() or None
    cleaned_channel = (channel or "").strip() or None
    if stale_after_minutes < 1:
        raise ValueError("stale_after_minutes must be at least 1")
    if expiring_within_seconds < 1:
        raise ValueError("expiring_within_seconds must be at least 1")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if follow_up_within_hours < 1:
        raise ValueError("follow_up_within_hours must be at least 1")
    cleaned_assigned_to = (assigned_to or "").strip() or None
    cleaned_follow_up_status = _clean_delivery_incident_follow_up_status(follow_up_status)

    now_dt = datetime.now(UTC).replace(microsecond=0)
    due_soon_cutoff_dt = now_dt + timedelta(hours=follow_up_within_hours)
    now = now_dt.isoformat().replace("+00:00", "Z")
    due_soon_cutoff = due_soon_cutoff_dt.isoformat().replace("+00:00", "Z")
    stale_cutoff = (now_dt - timedelta(minutes=stale_after_minutes)).isoformat().replace(
        "+00:00", "Z"
    )
    expiring_cutoff = (now_dt + timedelta(seconds=expiring_within_seconds)).isoformat().replace(
        "+00:00", "Z"
    )
    base_where: list[str] = []
    base_params: list[Any] = []
    if cleaned_owner:
        base_where.append("owner_id = ?")
        base_params.append(cleaned_owner)
    if cleaned_channel:
        base_where.append("channel = ?")
        base_params.append(cleaned_channel)

    incident_specs = [
        {
            "incident_type": "deadletter",
            "priority": "critical",
            "priority_rank": 0,
            "delivery_action": "review_deadletter",
            "incident_reason": "delivery_exhausted",
            "where": ["status = 'failed'", "delivery_exhausted_at IS NOT NULL"],
            "params": [],
        },
        {
            "incident_type": "expired_claim",
            "priority": "critical",
            "priority_rank": 1,
            "delivery_action": "reclaim_or_release",
            "incident_reason": "delivery_claim_expired",
            "where": ["delivery_claim_token IS NOT NULL", "delivery_claimed_until <= ?"],
            "params": [now],
        },
        {
            "incident_type": "expiring_claim",
            "priority": "high",
            "priority_rank": 2,
            "delivery_action": "renew_claim",
            "incident_reason": "delivery_claim_expiring",
            "where": [
                "delivery_claim_token IS NOT NULL",
                "delivery_claimed_until > ?",
                "delivery_claimed_until <= ?",
            ],
            "params": [now, expiring_cutoff],
        },
        {
            "incident_type": "retry_ready",
            "priority": "high",
            "priority_rank": 3,
            "delivery_action": "retry_failed",
            "incident_reason": "failed_delivery_ready",
            "where": [
                "status = 'failed'",
                "delivery_exhausted_at IS NULL",
                "(delivery_retry_after IS NULL OR delivery_retry_after <= ?)",
                "delivery_claim_token IS NULL",
            ],
            "params": [now],
        },
        {
            "incident_type": "stale_prepared",
            "priority": "medium",
            "priority_rank": 4,
            "delivery_action": "send_prepared",
            "incident_reason": "prepared_delivery_stale",
            "where": [
                "status = 'prepared'",
                "created_at <= ?",
                "delivery_claim_token IS NULL",
            ],
            "params": [stale_cutoff],
        },
    ]
    incidents: list[dict[str, Any]] = []
    for spec in incident_specs:
        where = base_where + spec["where"]
        rows = conn.execute(
            f"""
            SELECT *
            FROM advisor_action_queue_escalation_notifications
            WHERE {' AND '.join(where)}
            ORDER BY COALESCE(
                delivery_exhausted_at,
                delivery_claimed_until,
                delivery_retry_after,
                updated_at,
                created_at
            ) ASC,
            notification_id ASC
            LIMIT ?
            """,
            [*base_params, *spec["params"], limit],
        ).fetchall()
        for row in rows:
            item = _escalation_notification_delivery_incident_item(
                dict(row),
                incident_type=spec["incident_type"],
                incident_reason=spec["incident_reason"],
                delivery_action=spec["delivery_action"],
                priority=spec["priority"],
                now_dt=now_dt,
                expiring_within_seconds=expiring_within_seconds,
            )
            item["_priority_rank"] = spec["priority_rank"]
            incidents.append(item)

    incidents.sort(key=lambda item: (item["_priority_rank"], _delivery_incident_sort_at(item)))
    candidate_data = []
    for item in incidents:
        item.pop("_priority_rank", None)
        candidate_data.append(item)
    latest_reviews = _latest_escalation_notification_incident_reviews(conn, candidate_data)
    visible_data = []
    suppressed_count = 0
    filtered_count = 0
    for item in candidate_data:
        key = (item["notification_id"], item["incident_type"])
        item["latest_review"] = latest_reviews.get(key)
        is_suppressed, suppression_reason = _delivery_incident_suppression(item, now_dt)
        item["is_suppressed"] = is_suppressed
        item["suppression_reason"] = suppression_reason
        item["follow_up_status"] = _delivery_incident_follow_up_status(
            item.get("latest_review") or {},
            now_dt,
            due_soon_cutoff_dt,
        )
        if not _delivery_incident_matches_filters(item, cleaned_assigned_to, cleaned_follow_up_status):
            continue
        filtered_count += 1
        if is_suppressed:
            suppressed_count += 1
            if not include_suppressed:
                continue
        visible_data.append(item)
    data = visible_data[:limit]
    return {
        "kind": "advisor_action_queue_task_escalation_notification_delivery_incidents",
        "owner_id": cleaned_owner,
        "data": data,
        "metadata": {
            "owner_id": cleaned_owner,
            "channel": cleaned_channel,
            "stale_after_minutes": stale_after_minutes,
            "stale_cutoff": stale_cutoff,
            "expiring_within_seconds": expiring_within_seconds,
            "lease_expiring_cutoff": expiring_cutoff,
            "limit": limit,
            "include_suppressed": include_suppressed,
            "assigned_to": cleaned_assigned_to,
            "follow_up_status": cleaned_follow_up_status,
            "follow_up_within_hours": follow_up_within_hours,
            "follow_up_due_soon_cutoff": due_soon_cutoff,
            "filtered_count": filtered_count,
            "result_count": len(data),
            "latest_review_count": len(latest_reviews),
            "suppressed_count": suppressed_count,
        },
    }


def summarize_action_queue_task_escalation_notification_delivery_incidents(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    channel: str | None = None,
    stale_after_minutes: int = 60,
    expiring_within_seconds: int = 300,
    max_incidents: int = 10000,
) -> dict[str, Any]:
    if max_incidents < 1:
        raise ValueError("max_incidents must be at least 1")

    incident_feed = list_action_queue_task_escalation_notification_delivery_incidents(
        conn,
        owner_id=owner_id,
        channel=channel,
        stale_after_minutes=stale_after_minutes,
        expiring_within_seconds=expiring_within_seconds,
        limit=max_incidents,
        include_suppressed=True,
    )
    incidents = incident_feed["data"]
    metadata = incident_feed["metadata"]
    summary = {
        "total_count": 0,
        "actionable_count": 0,
        "resolved_count": 0,
        "snoozed_count": 0,
        "unreviewed_count": 0,
        "critical_count": 0,
        "high_count": 0,
        "medium_count": 0,
        "actionable_critical_count": 0,
        "actionable_high_count": 0,
        "actionable_medium_count": 0,
    }
    by_incident_type: dict[str, dict[str, Any]] = {}
    by_priority: dict[str, dict[str, Any]] = {}
    by_latest_review_status: dict[str, dict[str, Any]] = {}

    for incident in incidents:
        incident_type = incident.get("incident_type") or "unknown"
        priority = incident.get("priority") or "unknown"
        latest_review = incident.get("latest_review") or {}
        latest_review_status = latest_review.get("incident_status") or "unreviewed"
        bucket = _delivery_incident_summary_bucket(incident)

        summary["total_count"] += 1
        summary[f"{bucket}_count"] += 1
        if latest_review_status == "unreviewed":
            summary["unreviewed_count"] += 1
        if priority in {"critical", "high", "medium"}:
            summary[f"{priority}_count"] += 1
            if bucket == "actionable":
                summary[f"actionable_{priority}_count"] += 1

        _increment_delivery_incident_count_row(by_incident_type, "incident_type", incident_type, bucket)
        _increment_delivery_incident_count_row(by_priority, "priority", priority, bucket)
        _increment_delivery_incident_count_row(
            by_latest_review_status,
            "incident_status",
            latest_review_status,
            bucket,
        )

    incident_type_order = ["deadletter", "expired_claim", "expiring_claim", "retry_ready", "stale_prepared"]
    priority_order = ["critical", "high", "medium", "low", "unknown"]
    status_order = ["unreviewed", "acknowledged", "assigned", "needs_followup", "snoozed", "resolved"]
    return {
        "kind": "advisor_action_queue_task_escalation_notification_delivery_incident_summary",
        "owner_id": incident_feed["owner_id"],
        "data": {
            "summary": summary,
            "by_incident_type": _ordered_delivery_incident_count_rows(
                by_incident_type,
                "incident_type",
                incident_type_order,
            ),
            "by_priority": _ordered_delivery_incident_count_rows(by_priority, "priority", priority_order),
            "by_latest_review_status": _ordered_delivery_incident_count_rows(
                by_latest_review_status,
                "incident_status",
                status_order,
            ),
        },
        "metadata": {
            "owner_id": metadata["owner_id"],
            "channel": metadata["channel"],
            "stale_after_minutes": metadata["stale_after_minutes"],
            "stale_cutoff": metadata["stale_cutoff"],
            "expiring_within_seconds": metadata["expiring_within_seconds"],
            "lease_expiring_cutoff": metadata["lease_expiring_cutoff"],
            "max_incidents": max_incidents,
            "scanned_incident_count": len(incidents),
            "summary_limited": len(incidents) >= max_incidents,
            "latest_review_count": metadata["latest_review_count"],
            "suppressed_count": metadata["suppressed_count"],
        },
    }


def summarize_action_queue_task_escalation_notification_delivery_incident_workload(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    channel: str | None = None,
    stale_after_minutes: int = 60,
    expiring_within_seconds: int = 300,
    follow_up_within_hours: int = 24,
    max_incidents: int = 10000,
) -> dict[str, Any]:
    if follow_up_within_hours < 1:
        raise ValueError("follow_up_within_hours must be at least 1")
    if max_incidents < 1:
        raise ValueError("max_incidents must be at least 1")

    incident_feed = list_action_queue_task_escalation_notification_delivery_incidents(
        conn,
        owner_id=owner_id,
        channel=channel,
        stale_after_minutes=stale_after_minutes,
        expiring_within_seconds=expiring_within_seconds,
        limit=max_incidents,
        include_suppressed=True,
    )
    incidents = incident_feed["data"]
    metadata = incident_feed["metadata"]
    now_dt = datetime.now(UTC).replace(microsecond=0)
    due_soon_cutoff_dt = now_dt + timedelta(hours=follow_up_within_hours)
    now = now_dt.isoformat().replace("+00:00", "Z")
    due_soon_cutoff = due_soon_cutoff_dt.isoformat().replace("+00:00", "Z")
    summary = {
        "total_count": 0,
        "unresolved_count": 0,
        "actionable_count": 0,
        "assigned_actionable_count": 0,
        "unassigned_actionable_count": 0,
        "snoozed_count": 0,
        "resolved_count": 0,
        "follow_up_overdue_count": 0,
        "follow_up_due_soon_count": 0,
        "follow_up_future_count": 0,
        "follow_up_missing_count": 0,
    }
    by_assignee: dict[str, dict[str, Any]] = {}
    by_follow_up_status: dict[str, dict[str, Any]] = {}

    for incident in incidents:
        latest_review = incident.get("latest_review") or {}
        bucket = _delivery_incident_summary_bucket(incident)
        follow_up_status = _delivery_incident_follow_up_status(
            latest_review,
            now_dt,
            due_soon_cutoff_dt,
        )
        summary["total_count"] += 1
        summary[f"{bucket}_count"] += 1
        if bucket == "resolved":
            continue

        summary["unresolved_count"] += 1
        summary[f"follow_up_{follow_up_status}_count"] += 1
        assignee = latest_review.get("assigned_to") or "unassigned"
        priority = incident.get("priority") or "unknown"
        row = by_assignee.setdefault(
            assignee,
            {
                "assigned_to": assignee,
                "unresolved_count": 0,
                "actionable_count": 0,
                "snoozed_count": 0,
                "critical_count": 0,
                "high_count": 0,
                "medium_count": 0,
                "follow_up_overdue_count": 0,
                "follow_up_due_soon_count": 0,
                "follow_up_future_count": 0,
                "follow_up_missing_count": 0,
            },
        )
        row["unresolved_count"] += 1
        row[f"{bucket}_count"] += 1
        row[f"follow_up_{follow_up_status}_count"] += 1
        if priority in {"critical", "high", "medium"}:
            row[f"{priority}_count"] += 1
        if bucket == "actionable":
            if assignee == "unassigned":
                summary["unassigned_actionable_count"] += 1
            else:
                summary["assigned_actionable_count"] += 1

        follow_up_row = by_follow_up_status.setdefault(
            follow_up_status,
            {
                "follow_up_status": follow_up_status,
                "unresolved_count": 0,
                "actionable_count": 0,
                "snoozed_count": 0,
            },
        )
        follow_up_row["unresolved_count"] += 1
        follow_up_row[f"{bucket}_count"] += 1

    follow_up_order = ["overdue", "due_soon", "missing", "future"]
    return {
        "kind": "advisor_action_queue_task_escalation_notification_delivery_incident_workload",
        "owner_id": incident_feed["owner_id"],
        "data": {
            "summary": summary,
            "by_assignee": _ordered_delivery_incident_workload_rows(by_assignee),
            "by_follow_up_status": _ordered_delivery_incident_count_rows(
                by_follow_up_status,
                "follow_up_status",
                follow_up_order,
            ),
        },
        "metadata": {
            "owner_id": metadata["owner_id"],
            "channel": metadata["channel"],
            "stale_after_minutes": metadata["stale_after_minutes"],
            "stale_cutoff": metadata["stale_cutoff"],
            "expiring_within_seconds": metadata["expiring_within_seconds"],
            "lease_expiring_cutoff": metadata["lease_expiring_cutoff"],
            "follow_up_within_hours": follow_up_within_hours,
            "now": now,
            "follow_up_due_soon_cutoff": due_soon_cutoff,
            "max_incidents": max_incidents,
            "scanned_incident_count": len(incidents),
            "summary_limited": len(incidents) >= max_incidents,
            "latest_review_count": metadata["latest_review_count"],
            "suppressed_count": metadata["suppressed_count"],
        },
    }


def review_action_queue_task_escalation_notification_delivery_incident(
    conn: sqlite3.Connection,
    notification_id: int,
    incident_type: str | None,
    incident_status: str | None,
    owner_id: str | None = None,
    reviewer: str | None = None,
    assigned_to: str | None = None,
    notes: str | None = None,
    follow_up_at: str | None = None,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_notification_table(conn)
    _ensure_action_queue_escalation_notification_incident_review_table(conn)
    cleaned_owner = (owner_id or "").strip() or None
    cleaned_incident_type = _clean_delivery_incident_type(incident_type)
    cleaned_incident_status = _clean_delivery_incident_review_status(incident_status)
    cleaned_reviewer = (reviewer or "").strip() or None
    cleaned_assigned_to = (assigned_to or "").strip() or None
    cleaned_follow_up_at = _clean_iso_datetime(follow_up_at, "follow_up_at") if follow_up_at is not None else None
    if cleaned_incident_status == "assigned" and not cleaned_assigned_to:
        raise ValueError("assigned_to is required when incident_status is assigned")
    if cleaned_incident_status == "snoozed" and not cleaned_follow_up_at:
        raise ValueError("follow_up_at is required when incident_status is snoozed")

    current = _escalation_notification_by_id(conn, notification_id)
    if cleaned_owner and current["owner_id"] != cleaned_owner:
        raise MarketRecordNotFound(
            f"No advisor action queue escalation notification found for id {notification_id}"
        )
    now = now_utc()
    cursor = conn.execute(
        """
        INSERT INTO advisor_action_queue_escalation_notification_incident_reviews (
            notification_id, owner_id, channel, incident_type, incident_status,
            reviewer, assigned_to, notes, follow_up_at, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            notification_id,
            current["owner_id"],
            current["channel"],
            cleaned_incident_type,
            cleaned_incident_status,
            cleaned_reviewer,
            cleaned_assigned_to,
            notes,
            cleaned_follow_up_at,
            now,
        ),
    )
    conn.commit()
    review = _escalation_notification_incident_review_by_id(conn, int(cursor.lastrowid))
    return {
        "kind": "advisor_action_queue_task_escalation_notification_delivery_incident_review",
        "owner_id": current["owner_id"],
        "data": {
            "review": review,
            "notification": current,
        },
        "metadata": {
            "notification_id": notification_id,
            "incident_type": cleaned_incident_type,
            "incident_status": cleaned_incident_status,
            "reviewer": cleaned_reviewer,
            "assigned_to": cleaned_assigned_to,
        },
    }


def bulk_review_action_queue_task_escalation_notification_delivery_incidents(
    conn: sqlite3.Connection,
    incident_refs: list[dict[str, Any]],
    incident_status: str | None,
    owner_id: str | None = None,
    reviewer: str | None = None,
    assigned_to: str | None = None,
    notes: str | None = None,
    follow_up_at: str | None = None,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_notification_table(conn)
    _ensure_action_queue_escalation_notification_incident_review_table(conn)
    refs = _clean_delivery_incident_refs(incident_refs)
    if not refs:
        raise ValueError("At least one incident reference is required")
    cleaned_owner = (owner_id or "").strip() or None
    cleaned_incident_status = _clean_delivery_incident_review_status(incident_status)
    cleaned_reviewer = (reviewer or "").strip() or None
    cleaned_assigned_to = (assigned_to or "").strip() or None
    cleaned_follow_up_at = _clean_iso_datetime(follow_up_at, "follow_up_at") if follow_up_at is not None else None
    if cleaned_incident_status == "assigned" and not cleaned_assigned_to:
        raise ValueError("assigned_to is required when incident_status is assigned")
    if cleaned_incident_status == "snoozed" and not cleaned_follow_up_at:
        raise ValueError("follow_up_at is required when incident_status is snoozed")

    current_notifications: list[dict[str, Any]] = []
    for ref in refs:
        current = _escalation_notification_by_id(conn, ref["notification_id"])
        if cleaned_owner and current["owner_id"] != cleaned_owner:
            raise MarketRecordNotFound(
                f"No advisor action queue escalation notification found for id {ref['notification_id']}"
            )
        current_notifications.append(current)

    now = now_utc()
    created_review_ids: list[int] = []
    for ref, current in zip(refs, current_notifications, strict=True):
        cursor = conn.execute(
            """
            INSERT INTO advisor_action_queue_escalation_notification_incident_reviews (
                notification_id, owner_id, channel, incident_type, incident_status,
                reviewer, assigned_to, notes, follow_up_at, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ref["notification_id"],
                current["owner_id"],
                current["channel"],
                ref["incident_type"],
                cleaned_incident_status,
                cleaned_reviewer,
                cleaned_assigned_to,
                notes,
                cleaned_follow_up_at,
                now,
            ),
        )
        created_review_ids.append(int(cursor.lastrowid))
    conn.commit()

    return {
        "kind": "advisor_action_queue_task_escalation_notification_delivery_incident_bulk_review",
        "owner_id": cleaned_owner,
        "data": {
            "reviews": [
                _escalation_notification_incident_review_by_id(conn, review_id)
                for review_id in created_review_ids
            ],
            "notifications": current_notifications,
        },
        "metadata": {
            "owner_id": cleaned_owner,
            "requested_count": len(refs),
            "reviewed_count": len(created_review_ids),
            "incident_status": cleaned_incident_status,
            "reviewer": cleaned_reviewer,
            "assigned_to": cleaned_assigned_to,
            "follow_up_at": cleaned_follow_up_at,
        },
    }


def get_action_queue_task_escalation_notification_delivery_incident(
    conn: sqlite3.Connection,
    notification_id: int,
    incident_type: str | None,
    owner_id: str | None = None,
    stale_after_minutes: int = 60,
    expiring_within_seconds: int = 300,
    follow_up_within_hours: int = 24,
    audit_limit: int = 25,
) -> dict[str, Any]:
    if notification_id < 1:
        raise ValueError("notification_id must be at least 1")
    if audit_limit < 1:
        raise ValueError("audit_limit must be at least 1")
    cleaned_incident_type = _clean_delivery_incident_type(incident_type)
    current = _escalation_notification_by_id(conn, notification_id)
    cleaned_owner = (owner_id or "").strip() or None
    if cleaned_owner and current["owner_id"] != cleaned_owner:
        raise MarketRecordNotFound(
            f"No advisor action queue escalation notification found for id {notification_id}"
        )

    incident_feed = list_action_queue_task_escalation_notification_delivery_incidents(
        conn,
        owner_id=current["owner_id"],
        channel=current["channel"],
        stale_after_minutes=stale_after_minutes,
        expiring_within_seconds=expiring_within_seconds,
        limit=50000,
        include_suppressed=True,
        follow_up_within_hours=follow_up_within_hours,
    )
    incident = next(
        (
            item
            for item in incident_feed["data"]
            if item["notification_id"] == notification_id
            and item["incident_type"] == cleaned_incident_type
        ),
        None,
    )
    if incident is None:
        raise MarketRecordNotFound(
            f"No active delivery incident {cleaned_incident_type} found for notification {notification_id}"
        )

    reviews = list_action_queue_task_escalation_notification_delivery_incident_reviews(
        conn,
        owner_id=current["owner_id"],
        notification_id=notification_id,
        incident_type=cleaned_incident_type,
        limit=audit_limit,
    )
    attempts = list_action_queue_task_escalation_notification_delivery_attempts(
        conn,
        owner_id=current["owner_id"],
        notification_id=notification_id,
        limit=audit_limit,
    )
    remediations = list_action_queue_task_escalation_notification_deadletter_remediations(
        conn,
        owner_id=current["owner_id"],
        notification_id=notification_id,
        limit=audit_limit,
    )
    claim_releases = list_action_queue_task_escalation_notification_delivery_claim_releases(
        conn,
        owner_id=current["owner_id"],
        notification_id=notification_id,
        limit=audit_limit,
    )
    timeline = _delivery_incident_timeline(
        current,
        incident,
        reviews["data"],
        attempts["data"],
        remediations["data"],
        claim_releases["data"],
    )
    next_actions = _delivery_incident_next_actions(current, incident)
    return {
        "kind": "advisor_action_queue_task_escalation_notification_delivery_incident_detail",
        "owner_id": current["owner_id"],
        "data": {
            "incident": incident,
            "notification": current,
            "review_history": reviews["data"],
            "delivery_attempts": attempts["data"],
            "deadletter_remediations": remediations["data"],
            "claim_releases": claim_releases["data"],
            "timeline": timeline,
            "next_actions": next_actions,
        },
        "metadata": {
            "owner_id": current["owner_id"],
            "notification_id": notification_id,
            "incident_type": cleaned_incident_type,
            "audit_limit": audit_limit,
            "review_count": reviews["metadata"]["result_count"],
            "delivery_attempt_count": attempts["metadata"]["result_count"],
            "deadletter_remediation_count": remediations["metadata"]["result_count"],
            "claim_release_count": claim_releases["metadata"]["result_count"],
            "timeline_event_count": len(timeline),
            "next_action_count": len(next_actions),
        },
    }


def execute_action_queue_task_escalation_notification_delivery_incident_action(
    conn: sqlite3.Connection,
    notification_id: int,
    action_id: str | None,
    incident_type: str | None,
    owner_id: str | None = None,
    reviewer: str | None = None,
    assigned_to: str | None = None,
    notes: str | None = None,
    follow_up_at: str | None = None,
    delivery_notes: str | None = None,
    requeued_by: str | None = None,
    retry_after: str | None = None,
    claim_token: str | None = None,
    release_notes: str | None = None,
    released_by: str | None = None,
    claimed_by: str | None = None,
    lease_seconds: int = 300,
    stale_after_minutes: int = 60,
    expiring_within_seconds: int = 300,
    follow_up_within_hours: int = 24,
) -> dict[str, Any]:
    cleaned_action_id = _clean_delivery_incident_action_id(action_id)
    detail = get_action_queue_task_escalation_notification_delivery_incident(
        conn,
        notification_id=notification_id,
        incident_type=incident_type,
        owner_id=owner_id,
        stale_after_minutes=stale_after_minutes,
        expiring_within_seconds=expiring_within_seconds,
        follow_up_within_hours=follow_up_within_hours,
    )
    action = next(
        (
            item
            for item in detail["data"]["next_actions"]
            if item["action_id"] == cleaned_action_id
        ),
        None,
    )
    if action is None:
        raise ValueError(f"action_id {cleaned_action_id} is not currently available")

    if cleaned_action_id == "assign_incident":
        result = review_action_queue_task_escalation_notification_delivery_incident(
            conn,
            notification_id=notification_id,
            owner_id=owner_id,
            incident_type=incident_type,
            incident_status="assigned",
            reviewer=reviewer,
            assigned_to=assigned_to,
            notes=notes,
        )
    elif cleaned_action_id == "review_follow_up":
        result = review_action_queue_task_escalation_notification_delivery_incident(
            conn,
            notification_id=notification_id,
            owner_id=owner_id,
            incident_type=incident_type,
            incident_status="assigned",
            reviewer=reviewer,
            assigned_to=assigned_to,
            notes=notes,
            follow_up_at=follow_up_at,
        )
    elif cleaned_action_id == "resolve_incident":
        result = review_action_queue_task_escalation_notification_delivery_incident(
            conn,
            notification_id=notification_id,
            owner_id=owner_id,
            incident_type=incident_type,
            incident_status="resolved",
            reviewer=reviewer,
            notes=notes,
        )
    elif cleaned_action_id == "requeue_deadletter":
        result = requeue_action_queue_task_escalation_notification_deadletter(
            conn,
            notification_id=notification_id,
            owner_id=owner_id,
            retry_after=retry_after,
            delivery_notes=delivery_notes,
            requeued_by=requeued_by,
        )
    elif cleaned_action_id == "release_delivery_claim":
        result = release_action_queue_task_escalation_notification_delivery_claim(
            conn,
            notification_id=notification_id,
            owner_id=owner_id,
            claim_token=claim_token or detail["data"]["notification"].get("delivery_claim_token"),
            release_notes=release_notes,
            released_by=released_by,
        )
    elif cleaned_action_id == "claim_delivery":
        result = _claim_action_queue_task_escalation_notification_delivery_by_id(
            conn,
            notification_id=notification_id,
            owner_id=owner_id,
            claimed_by=claimed_by,
            stale_after_minutes=stale_after_minutes,
            lease_seconds=lease_seconds,
        )
    else:
        raise ValueError(f"Unsupported delivery incident action_id {cleaned_action_id}")

    post_action_state = _delivery_incident_post_action_state(
        conn,
        notification_id=notification_id,
        incident_type=incident_type,
        owner_id=owner_id,
        stale_after_minutes=stale_after_minutes,
        expiring_within_seconds=expiring_within_seconds,
        follow_up_within_hours=follow_up_within_hours,
    )
    return {
        "kind": "advisor_action_queue_task_escalation_notification_delivery_incident_action_execution",
        "owner_id": detail["owner_id"],
        "data": {
            "action": action,
            "result": result,
            "post_action_state": post_action_state,
        },
        "metadata": {
            "owner_id": detail["owner_id"],
            "notification_id": notification_id,
            "incident_type": detail["metadata"]["incident_type"],
            "action_id": cleaned_action_id,
            "result_kind": result["kind"],
            "post_action_incident_active": post_action_state["incident_active"],
            "post_action_incident_cleared": post_action_state["incident_cleared"],
            "post_action_next_action_count": post_action_state["next_action_count"],
        },
    }


def _delivery_incident_post_action_state(
    conn: sqlite3.Connection,
    notification_id: int,
    incident_type: str | None,
    owner_id: str | None = None,
    stale_after_minutes: int = 60,
    expiring_within_seconds: int = 300,
    follow_up_within_hours: int = 24,
) -> dict[str, Any]:
    try:
        detail = get_action_queue_task_escalation_notification_delivery_incident(
            conn,
            notification_id=notification_id,
            incident_type=incident_type,
            owner_id=owner_id,
            stale_after_minutes=stale_after_minutes,
            expiring_within_seconds=expiring_within_seconds,
            follow_up_within_hours=follow_up_within_hours,
        )
    except MarketRecordNotFound:
        current = _escalation_notification_by_id(conn, notification_id)
        cleaned_owner = (owner_id or "").strip() or None
        if cleaned_owner and current["owner_id"] != cleaned_owner:
            raise MarketRecordNotFound(
                f"No advisor action queue escalation notification found for id {notification_id}"
            )
        return {
            "incident_active": False,
            "incident_cleared": True,
            "incident_detail": None,
            "notification": current,
            "next_actions": [],
            "next_action_count": 0,
        }
    return {
        "incident_active": True,
        "incident_cleared": False,
        "incident_detail": detail,
        "notification": detail["data"]["notification"],
        "next_actions": detail["data"]["next_actions"],
        "next_action_count": detail["metadata"]["next_action_count"],
    }


def list_action_queue_task_escalation_notification_delivery_incident_reviews(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    notification_id: int | None = None,
    incident_type: str | None = None,
    incident_status: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_notification_table(conn)
    _ensure_action_queue_escalation_notification_incident_review_table(conn)
    cleaned_owner = (owner_id or "").strip() or None
    cleaned_incident_type = _clean_delivery_incident_type(incident_type) if incident_type is not None else None
    cleaned_incident_status = (
        _clean_delivery_incident_review_status(incident_status) if incident_status is not None else None
    )
    if notification_id is not None and notification_id < 1:
        raise ValueError("notification_id must be at least 1")
    if limit < 1:
        raise ValueError("limit must be at least 1")

    where: list[str] = []
    params: list[Any] = []
    if cleaned_owner:
        where.append("owner_id = ?")
        params.append(cleaned_owner)
    if notification_id is not None:
        where.append("notification_id = ?")
        params.append(notification_id)
    if cleaned_incident_type:
        where.append("incident_type = ?")
        params.append(cleaned_incident_type)
    if cleaned_incident_status:
        where.append("incident_status = ?")
        params.append(cleaned_incident_status)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT *
        FROM advisor_action_queue_escalation_notification_incident_reviews
        {where_sql}
        ORDER BY created_at DESC, incident_review_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return {
        "kind": "advisor_action_queue_task_escalation_notification_delivery_incident_reviews",
        "owner_id": cleaned_owner,
        "data": [
            _escalation_notification_incident_review_record_from_row(dict(row)) for row in rows
        ],
        "metadata": {
            "owner_id": cleaned_owner,
            "notification_id": notification_id,
            "incident_type": cleaned_incident_type,
            "incident_status": cleaned_incident_status,
            "limit": limit,
            "result_count": len(rows),
        },
    }


def list_action_queue_task_escalation_notification_delivery_queue(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    channel: str | None = None,
    stale_after_minutes: int = 60,
    limit: int = 25,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_notification_table(conn)
    cleaned_owner = (owner_id or "").strip() or None
    cleaned_channel = (channel or "").strip() or None
    if stale_after_minutes < 1:
        raise ValueError("stale_after_minutes must be at least 1")
    if limit < 1:
        raise ValueError("limit must be at least 1")

    where: list[str] = []
    params: list[Any] = []
    if cleaned_owner:
        where.append("owner_id = ?")
        params.append(cleaned_owner)
    if cleaned_channel:
        where.append("channel = ?")
        params.append(cleaned_channel)
    now = now_utc()
    cutoff = (
        datetime.now(UTC).replace(microsecond=0) - timedelta(minutes=stale_after_minutes)
    ).isoformat().replace("+00:00", "Z")
    where.append(
        "((status = 'failed' AND delivery_exhausted_at IS NULL "
        "AND (delivery_retry_after IS NULL OR delivery_retry_after <= ?)) "
        "OR (status = 'prepared' AND created_at <= ?))"
    )
    where.append("(delivery_claimed_until IS NULL OR delivery_claimed_until <= ?)")
    params.extend([now, cutoff, now, limit])
    rows = conn.execute(
        f"""
        SELECT *
        FROM advisor_action_queue_escalation_notifications
        WHERE {' AND '.join(where)}
        ORDER BY
            CASE status
                WHEN 'failed' THEN 0
                WHEN 'prepared' THEN 1
                ELSE 2
            END,
            COALESCE(updated_at, created_at) ASC,
            notification_id ASC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return {
        "kind": "advisor_action_queue_task_escalation_notification_delivery_queue",
        "owner_id": cleaned_owner,
        "data": [_escalation_notification_delivery_queue_item(dict(row)) for row in rows],
        "metadata": {
            "owner_id": cleaned_owner,
            "channel": cleaned_channel,
            "stale_after_minutes": stale_after_minutes,
            "stale_cutoff": cutoff,
            "limit": limit,
            "result_count": len(rows),
        },
    }


def list_action_queue_task_escalation_notification_delivery_claims(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    channel: str | None = None,
    claimed_by: str | None = None,
    lease_state: str | None = None,
    expiring_within_seconds: int = 300,
    limit: int = 25,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_notification_table(conn)
    cleaned_owner = (owner_id or "").strip() or None
    cleaned_channel = (channel or "").strip() or None
    cleaned_claimed_by = (claimed_by or "").strip() or None
    cleaned_lease_state = (lease_state or "").strip().lower() or None
    valid_lease_states = {"active", "expiring_soon", "expired"}
    if cleaned_lease_state and cleaned_lease_state not in valid_lease_states:
        raise ValueError("lease_state must be one of: active, expiring_soon, expired")
    if expiring_within_seconds < 1:
        raise ValueError("expiring_within_seconds must be at least 1")
    if limit < 1:
        raise ValueError("limit must be at least 1")

    now_dt = datetime.now(UTC).replace(microsecond=0)
    now = now_dt.isoformat().replace("+00:00", "Z")
    expiring_cutoff = (now_dt + timedelta(seconds=expiring_within_seconds)).isoformat().replace(
        "+00:00", "Z"
    )
    where = [
        "delivery_claim_token IS NOT NULL",
        "delivery_claimed_by IS NOT NULL",
        "delivery_claimed_until IS NOT NULL",
    ]
    params: list[Any] = []
    if cleaned_owner:
        where.append("owner_id = ?")
        params.append(cleaned_owner)
    if cleaned_channel:
        where.append("channel = ?")
        params.append(cleaned_channel)
    if cleaned_claimed_by:
        where.append("delivery_claimed_by = ?")
        params.append(cleaned_claimed_by)
    if cleaned_lease_state == "expired":
        where.append("delivery_claimed_until <= ?")
        params.append(now)
    elif cleaned_lease_state == "expiring_soon":
        where.append("delivery_claimed_until > ? AND delivery_claimed_until <= ?")
        params.extend([now, expiring_cutoff])
    elif cleaned_lease_state == "active":
        where.append("delivery_claimed_until > ?")
        params.append(expiring_cutoff)

    rows = conn.execute(
        f"""
        SELECT *
        FROM advisor_action_queue_escalation_notifications
        WHERE {' AND '.join(where)}
        ORDER BY
            CASE
                WHEN delivery_claimed_until <= ? THEN 0
                WHEN delivery_claimed_until <= ? THEN 1
                ELSE 2
            END,
            delivery_claimed_until ASC,
            notification_id ASC
        LIMIT ?
        """,
        [*params, now, expiring_cutoff, limit],
    ).fetchall()
    return {
        "kind": "advisor_action_queue_task_escalation_notification_delivery_claims",
        "owner_id": cleaned_owner,
        "data": [
            _escalation_notification_delivery_claim_item(dict(row), now_dt, expiring_within_seconds)
            for row in rows
        ],
        "metadata": {
            "owner_id": cleaned_owner,
            "channel": cleaned_channel,
            "claimed_by": cleaned_claimed_by,
            "lease_state": cleaned_lease_state,
            "expiring_within_seconds": expiring_within_seconds,
            "lease_expiring_cutoff": expiring_cutoff,
            "limit": limit,
            "result_count": len(rows),
        },
    }


def list_action_queue_task_escalation_notification_deadletters(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    channel: str | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_notification_table(conn)
    cleaned_owner = (owner_id or "").strip() or None
    cleaned_channel = (channel or "").strip() or None
    if limit < 1:
        raise ValueError("limit must be at least 1")

    where = ["status = ?", "delivery_exhausted_at IS NOT NULL"]
    params: list[Any] = ["failed"]
    if cleaned_owner:
        where.append("owner_id = ?")
        params.append(cleaned_owner)
    if cleaned_channel:
        where.append("channel = ?")
        params.append(cleaned_channel)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT *
        FROM advisor_action_queue_escalation_notifications
        WHERE {' AND '.join(where)}
        ORDER BY delivery_exhausted_at DESC, notification_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return {
        "kind": "advisor_action_queue_task_escalation_notification_deadletters",
        "owner_id": cleaned_owner,
        "data": [_escalation_notification_deadletter_item(dict(row)) for row in rows],
        "metadata": {
            "owner_id": cleaned_owner,
            "channel": cleaned_channel,
            "limit": limit,
            "result_count": len(rows),
        },
    }


def requeue_action_queue_task_escalation_notification_deadletter(
    conn: sqlite3.Connection,
    notification_id: int,
    owner_id: str | None = None,
    retry_after: str | None = None,
    delivery_notes: str | None = None,
    requeued_by: str | None = None,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_notification_table(conn)
    _ensure_action_queue_escalation_notification_remediation_table(conn)
    cleaned_owner = (owner_id or "").strip() or None
    cleaned_requeued_by = (requeued_by or "").strip() or None
    cleaned_retry_after = _clean_iso_datetime(retry_after, "retry_after") if retry_after is not None else None
    now = now_utc()

    conn.commit()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT *
            FROM advisor_action_queue_escalation_notifications
            WHERE notification_id = ?
            """,
            (notification_id,),
        ).fetchone()
        if not row:
            raise MarketRecordNotFound(
                f"No advisor action queue escalation notification found for id {notification_id}"
            )
        current = dict(row)
        if cleaned_owner and current["owner_id"] != cleaned_owner:
            raise MarketRecordNotFound(
                f"No advisor action queue escalation notification found for id {notification_id}"
            )
        if not current.get("delivery_exhausted_at"):
            raise ValueError("Only exhausted escalation notifications can be requeued")

        cursor = conn.execute(
            """
            INSERT INTO advisor_action_queue_escalation_notification_remediations (
                notification_id, owner_id, channel, recipient, remediation_type,
                remediation_notes, requeued_by, retry_after,
                previous_delivery_exhausted_at, previous_delivery_exhausted_reason,
                previous_delivery_attempt_count, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                notification_id,
                current["owner_id"],
                current["channel"],
                current["recipient"],
                "requeue",
                delivery_notes,
                cleaned_requeued_by,
                cleaned_retry_after,
                current.get("delivery_exhausted_at"),
                current.get("delivery_exhausted_reason"),
                current.get("delivery_attempt_count") or 0,
                now,
            ),
        )
        conn.execute(
            """
            UPDATE advisor_action_queue_escalation_notifications
            SET status = 'failed',
                delivery_notes = COALESCE(?, delivery_notes),
                delivered_at = NULL,
                delivery_retry_after = ?,
                delivery_exhausted_at = NULL,
                delivery_exhausted_reason = NULL,
                delivery_claimed_by = NULL,
                delivery_claimed_at = NULL,
                delivery_claimed_until = NULL,
                delivery_claim_token = NULL,
                updated_at = ?
            WHERE notification_id = ?
            """,
            (
                delivery_notes,
                cleaned_retry_after,
                now,
                notification_id,
            ),
        )
        requeued_row = conn.execute(
            """
            SELECT *
            FROM advisor_action_queue_escalation_notifications
            WHERE notification_id = ?
            """,
            (notification_id,),
        ).fetchone()
        remediation_row = conn.execute(
            """
            SELECT *
            FROM advisor_action_queue_escalation_notification_remediations
            WHERE remediation_id = ?
            """,
            (int(cursor.lastrowid),),
        ).fetchone()
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    notification = _escalation_notification_record_from_row(dict(requeued_row))
    remediation = _escalation_notification_remediation_record_from_row(dict(remediation_row))
    return {
        "kind": "advisor_action_queue_task_escalation_notification_deadletter_requeue",
        "owner_id": notification["owner_id"],
        "data": {"notification": notification, "remediation": remediation},
        "metadata": {
            "notification_id": notification_id,
            "remediation_id": remediation["remediation_id"],
            "requeued": True,
            "retry_after": cleaned_retry_after,
            "delivery_notes_applied": delivery_notes is not None,
            "requeued_by": cleaned_requeued_by,
        },
    }


def list_action_queue_task_escalation_notification_deadletter_remediations(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    notification_id: int | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_notification_table(conn)
    _ensure_action_queue_escalation_notification_remediation_table(conn)
    cleaned_owner = (owner_id or "").strip() or None
    if notification_id is not None and notification_id < 1:
        raise ValueError("notification_id must be at least 1")
    if limit < 1:
        raise ValueError("limit must be at least 1")

    where: list[str] = []
    params: list[Any] = []
    if cleaned_owner:
        where.append("owner_id = ?")
        params.append(cleaned_owner)
    if notification_id is not None:
        where.append("notification_id = ?")
        params.append(notification_id)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT *
        FROM advisor_action_queue_escalation_notification_remediations
        {where_sql}
        ORDER BY created_at DESC, remediation_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return {
        "kind": "advisor_action_queue_task_escalation_notification_deadletter_remediations",
        "owner_id": cleaned_owner,
        "data": [_escalation_notification_remediation_record_from_row(dict(row)) for row in rows],
        "metadata": {
            "owner_id": cleaned_owner,
            "notification_id": notification_id,
            "limit": limit,
            "result_count": len(rows),
        },
    }


def claim_action_queue_task_escalation_notification_delivery(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    channel: str | None = None,
    claimed_by: str | None = None,
    stale_after_minutes: int = 60,
    lease_seconds: int = 300,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_notification_table(conn)
    cleaned_owner = (owner_id or "").strip() or None
    cleaned_channel = (channel or "").strip() or None
    cleaned_claimed_by = (claimed_by or "delivery-worker").strip() or "delivery-worker"
    if stale_after_minutes < 1:
        raise ValueError("stale_after_minutes must be at least 1")
    if lease_seconds < 1:
        raise ValueError("lease_seconds must be at least 1")

    now_dt = datetime.now(UTC).replace(microsecond=0)
    now = now_dt.isoformat().replace("+00:00", "Z")
    claim_until = (now_dt + timedelta(seconds=lease_seconds)).isoformat().replace("+00:00", "Z")
    cutoff = (now_dt - timedelta(minutes=stale_after_minutes)).isoformat().replace("+00:00", "Z")
    where: list[str] = []
    params: list[Any] = []
    if cleaned_owner:
        where.append("owner_id = ?")
        params.append(cleaned_owner)
    if cleaned_channel:
        where.append("channel = ?")
        params.append(cleaned_channel)
    where.append(
        "((status = 'failed' AND delivery_exhausted_at IS NULL "
        "AND (delivery_retry_after IS NULL OR delivery_retry_after <= ?)) "
        "OR (status = 'prepared' AND created_at <= ?))"
    )
    where.append("(delivery_claimed_until IS NULL OR delivery_claimed_until <= ?)")
    params.extend([now, cutoff, now])
    select_sql = f"""
        SELECT *
        FROM advisor_action_queue_escalation_notifications
        WHERE {' AND '.join(where)}
        ORDER BY
            CASE status
                WHEN 'failed' THEN 0
                WHEN 'prepared' THEN 1
                ELSE 2
            END,
            COALESCE(updated_at, created_at) ASC,
            notification_id ASC
        LIMIT 1
    """

    conn.commit()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(select_sql, params).fetchone()
        if not row:
            conn.commit()
            return {
                "kind": "advisor_action_queue_task_escalation_notification_delivery_claim",
                "owner_id": cleaned_owner,
                "data": {"claim": None},
                "metadata": {
                    "claimed": False,
                    "owner_id": cleaned_owner,
                    "channel": cleaned_channel,
                    "stale_after_minutes": stale_after_minutes,
                    "stale_cutoff": cutoff,
                    "lease_seconds": lease_seconds,
                },
            }
        row_dict = dict(row)
        attempt_count = int(row_dict.get("delivery_attempt_count") or 0) + 1
        claim_token = _escalation_notification_claim_token(
            row_dict["notification_id"], cleaned_claimed_by, now, attempt_count
        )
        conn.execute(
            """
            UPDATE advisor_action_queue_escalation_notifications
            SET delivery_claimed_by = ?,
                delivery_claimed_at = ?,
                delivery_claimed_until = ?,
                delivery_claim_token = ?,
                delivery_attempt_count = ?,
                updated_at = ?
            WHERE notification_id = ?
            """,
            (
                cleaned_claimed_by,
                now,
                claim_until,
                claim_token,
                attempt_count,
                now,
                row_dict["notification_id"],
            ),
        )
        claimed_row = conn.execute(
            """
            SELECT *
            FROM advisor_action_queue_escalation_notifications
            WHERE notification_id = ?
            """,
            (row_dict["notification_id"],),
        ).fetchone()
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return {
        "kind": "advisor_action_queue_task_escalation_notification_delivery_claim",
        "owner_id": cleaned_owner,
        "data": {"claim": _escalation_notification_delivery_queue_item(dict(claimed_row))},
        "metadata": {
            "claimed": True,
            "owner_id": cleaned_owner,
            "channel": cleaned_channel,
            "notification_id": row_dict["notification_id"],
            "claim_token": claim_token,
            "claimed_by": cleaned_claimed_by,
            "lease_seconds": lease_seconds,
            "claim_expires_at": claim_until,
            "stale_after_minutes": stale_after_minutes,
            "stale_cutoff": cutoff,
        },
    }


def _claim_action_queue_task_escalation_notification_delivery_by_id(
    conn: sqlite3.Connection,
    notification_id: int,
    owner_id: str | None = None,
    claimed_by: str | None = None,
    stale_after_minutes: int = 60,
    lease_seconds: int = 300,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_notification_table(conn)
    cleaned_owner = (owner_id or "").strip() or None
    cleaned_claimed_by = (claimed_by or "delivery-worker").strip() or "delivery-worker"
    if notification_id < 1:
        raise ValueError("notification_id must be at least 1")
    if stale_after_minutes < 1:
        raise ValueError("stale_after_minutes must be at least 1")
    if lease_seconds < 1:
        raise ValueError("lease_seconds must be at least 1")

    now_dt = datetime.now(UTC).replace(microsecond=0)
    now = now_dt.isoformat().replace("+00:00", "Z")
    claim_until = (now_dt + timedelta(seconds=lease_seconds)).isoformat().replace("+00:00", "Z")
    cutoff = (now_dt - timedelta(minutes=stale_after_minutes)).isoformat().replace("+00:00", "Z")

    conn.commit()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT *
            FROM advisor_action_queue_escalation_notifications
            WHERE notification_id = ?
            """,
            (notification_id,),
        ).fetchone()
        if not row:
            raise MarketRecordNotFound(
                f"No advisor action queue escalation notification found for id {notification_id}"
            )
        row_dict = dict(row)
        if cleaned_owner and row_dict["owner_id"] != cleaned_owner:
            raise MarketRecordNotFound(
                f"No advisor action queue escalation notification found for id {notification_id}"
            )
        retry_ready = (
            row_dict["status"] == "failed"
            and row_dict.get("delivery_exhausted_at") is None
            and (
                row_dict.get("delivery_retry_after") is None
                or row_dict.get("delivery_retry_after") <= now
            )
        )
        stale_prepared = row_dict["status"] == "prepared" and row_dict["created_at"] <= cutoff
        claim_expired = (
            row_dict.get("delivery_claimed_until") is None
            or row_dict.get("delivery_claimed_until") <= now
        )
        if not (retry_ready or stale_prepared):
            raise ValueError("Notification is not retry-ready or stale prepared")
        if not claim_expired:
            raise ValueError("Notification delivery claim is still active")

        attempt_count = int(row_dict.get("delivery_attempt_count") or 0) + 1
        claim_token = _escalation_notification_claim_token(
            notification_id, cleaned_claimed_by, now, attempt_count
        )
        conn.execute(
            """
            UPDATE advisor_action_queue_escalation_notifications
            SET delivery_claimed_by = ?,
                delivery_claimed_at = ?,
                delivery_claimed_until = ?,
                delivery_claim_token = ?,
                delivery_attempt_count = ?,
                updated_at = ?
            WHERE notification_id = ?
            """,
            (
                cleaned_claimed_by,
                now,
                claim_until,
                claim_token,
                attempt_count,
                now,
                notification_id,
            ),
        )
        claimed_row = conn.execute(
            """
            SELECT *
            FROM advisor_action_queue_escalation_notifications
            WHERE notification_id = ?
            """,
            (notification_id,),
        ).fetchone()
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return {
        "kind": "advisor_action_queue_task_escalation_notification_delivery_claim",
        "owner_id": cleaned_owner or row_dict["owner_id"],
        "data": {"claim": _escalation_notification_delivery_queue_item(dict(claimed_row))},
        "metadata": {
            "claimed": True,
            "owner_id": cleaned_owner or row_dict["owner_id"],
            "channel": row_dict["channel"],
            "notification_id": notification_id,
            "claim_token": claim_token,
            "claimed_by": cleaned_claimed_by,
            "lease_seconds": lease_seconds,
            "claim_expires_at": claim_until,
            "stale_after_minutes": stale_after_minutes,
            "stale_cutoff": cutoff,
        },
    }


def renew_action_queue_task_escalation_notification_delivery_claim(
    conn: sqlite3.Connection,
    notification_id: int,
    claim_token: str | None,
    lease_seconds: int = 300,
    owner_id: str | None = None,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_notification_table(conn)
    cleaned_owner = (owner_id or "").strip() or None
    cleaned_claim_token = (claim_token or "").strip()
    if not cleaned_claim_token:
        raise ValueError("claim_token is required")
    if lease_seconds < 1:
        raise ValueError("lease_seconds must be at least 1")

    now_dt = datetime.now(UTC).replace(microsecond=0)
    now = now_dt.isoformat().replace("+00:00", "Z")
    claim_until = (now_dt + timedelta(seconds=lease_seconds)).isoformat().replace("+00:00", "Z")
    conn.commit()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT *
            FROM advisor_action_queue_escalation_notifications
            WHERE notification_id = ?
            """,
            (notification_id,),
        ).fetchone()
        if not row:
            raise MarketRecordNotFound(
                f"No advisor action queue escalation notification found for id {notification_id}"
            )
        current = dict(row)
        if cleaned_owner and current["owner_id"] != cleaned_owner:
            raise MarketRecordNotFound(
                f"No advisor action queue escalation notification found for id {notification_id}"
            )
        if current.get("delivery_claim_token") != cleaned_claim_token:
            raise ValueError("delivery claim token is invalid or expired")
        if current.get("delivery_claimed_until") and current["delivery_claimed_until"] <= now:
            raise ValueError("delivery claim token is invalid or expired")

        conn.execute(
            """
            UPDATE advisor_action_queue_escalation_notifications
            SET delivery_claimed_until = ?,
                updated_at = ?
            WHERE notification_id = ?
            """,
            (claim_until, now, notification_id),
        )
        renewed_row = conn.execute(
            """
            SELECT *
            FROM advisor_action_queue_escalation_notifications
            WHERE notification_id = ?
            """,
            (notification_id,),
        ).fetchone()
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return {
        "kind": "advisor_action_queue_task_escalation_notification_delivery_claim_renewal",
        "owner_id": cleaned_owner,
        "data": {"claim": _escalation_notification_delivery_queue_item(dict(renewed_row))},
        "metadata": {
            "renewed": True,
            "owner_id": cleaned_owner,
            "notification_id": notification_id,
            "claim_token": cleaned_claim_token,
            "lease_seconds": lease_seconds,
            "claim_expires_at": claim_until,
        },
    }


def release_action_queue_task_escalation_notification_delivery_claim(
    conn: sqlite3.Connection,
    notification_id: int,
    claim_token: str | None,
    owner_id: str | None = None,
    release_notes: str | None = None,
    released_by: str | None = None,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_notification_table(conn)
    _ensure_action_queue_escalation_notification_claim_release_table(conn)
    cleaned_owner = (owner_id or "").strip() or None
    cleaned_claim_token = (claim_token or "").strip()
    cleaned_released_by = (released_by or "").strip() or None
    if not cleaned_claim_token:
        raise ValueError("claim_token is required")

    now = now_utc()
    conn.commit()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT *
            FROM advisor_action_queue_escalation_notifications
            WHERE notification_id = ?
            """,
            (notification_id,),
        ).fetchone()
        if not row:
            raise MarketRecordNotFound(
                f"No advisor action queue escalation notification found for id {notification_id}"
            )
        current = dict(row)
        if cleaned_owner and current["owner_id"] != cleaned_owner:
            raise MarketRecordNotFound(
                f"No advisor action queue escalation notification found for id {notification_id}"
            )
        if current.get("delivery_claim_token") != cleaned_claim_token:
            raise ValueError("delivery claim token is invalid or expired")
        previous_claimed_until = current.get("delivery_claimed_until")

        cursor = conn.execute(
            """
            INSERT INTO advisor_action_queue_escalation_notification_claim_releases (
                notification_id, owner_id, channel, recipient, status, claim_token,
                claimed_by, claimed_at, claimed_until, released_by, release_notes,
                previous_delivery_attempt_count, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                notification_id,
                current["owner_id"],
                current["channel"],
                current["recipient"],
                current["status"],
                cleaned_claim_token,
                current.get("delivery_claimed_by"),
                current.get("delivery_claimed_at"),
                previous_claimed_until,
                cleaned_released_by,
                release_notes,
                current.get("delivery_attempt_count") or 0,
                now,
            ),
        )
        conn.execute(
            """
            UPDATE advisor_action_queue_escalation_notifications
            SET delivery_notes = COALESCE(?, delivery_notes),
                delivery_claimed_by = NULL,
                delivery_claimed_at = NULL,
                delivery_claimed_until = NULL,
                delivery_claim_token = NULL,
                updated_at = ?
            WHERE notification_id = ?
            """,
            (release_notes, now, notification_id),
        )
        released_row = conn.execute(
            """
            SELECT *
            FROM advisor_action_queue_escalation_notifications
            WHERE notification_id = ?
            """,
            (notification_id,),
        ).fetchone()
        release_row = conn.execute(
            """
            SELECT *
            FROM advisor_action_queue_escalation_notification_claim_releases
            WHERE release_id = ?
            """,
            (int(cursor.lastrowid),),
        ).fetchone()
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    notification = _escalation_notification_record_from_row(dict(released_row))
    release = _escalation_notification_claim_release_record_from_row(dict(release_row))
    return {
        "kind": "advisor_action_queue_task_escalation_notification_delivery_claim_release",
        "owner_id": notification["owner_id"],
        "data": {"notification": notification, "release": release},
        "metadata": {
            "released": True,
            "owner_id": cleaned_owner,
            "notification_id": notification_id,
            "release_id": release["release_id"],
            "claim_token": cleaned_claim_token,
            "released_by": cleaned_released_by,
            "previous_claimed_until": previous_claimed_until,
            "release_notes_applied": release_notes is not None,
        },
    }


def list_action_queue_task_escalation_notification_delivery_claim_releases(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    notification_id: int | None = None,
    released_by: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_notification_table(conn)
    _ensure_action_queue_escalation_notification_claim_release_table(conn)
    cleaned_owner = (owner_id or "").strip() or None
    cleaned_released_by = (released_by or "").strip() or None
    if notification_id is not None and notification_id < 1:
        raise ValueError("notification_id must be at least 1")
    if limit < 1:
        raise ValueError("limit must be at least 1")

    where: list[str] = []
    params: list[Any] = []
    if cleaned_owner:
        where.append("owner_id = ?")
        params.append(cleaned_owner)
    if notification_id is not None:
        where.append("notification_id = ?")
        params.append(notification_id)
    if cleaned_released_by:
        where.append("released_by = ?")
        params.append(cleaned_released_by)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT *
        FROM advisor_action_queue_escalation_notification_claim_releases
        {where_sql}
        ORDER BY created_at DESC, release_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return {
        "kind": "advisor_action_queue_task_escalation_notification_delivery_claim_releases",
        "owner_id": cleaned_owner,
        "data": [_escalation_notification_claim_release_record_from_row(dict(row)) for row in rows],
        "metadata": {
            "owner_id": cleaned_owner,
            "notification_id": notification_id,
            "released_by": cleaned_released_by,
            "limit": limit,
            "result_count": len(rows),
        },
    }


def complete_action_queue_task_escalation_notification_delivery_claim(
    conn: sqlite3.Connection,
    notification_id: int,
    claim_token: str | None,
    status: str | None,
    delivery_notes: str | None = None,
    delivered_at: str | None = None,
    retry_after: str | None = None,
    max_attempts: int = 3,
    owner_id: str | None = None,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_notification_table(conn)
    _ensure_action_queue_escalation_notification_attempt_table(conn)
    cleaned_owner = (owner_id or "").strip() or None
    cleaned_claim_token = (claim_token or "").strip()
    if not cleaned_claim_token:
        raise ValueError("claim_token is required")
    cleaned_status = _clean_escalation_notification_status(status)
    if cleaned_status == "prepared":
        raise ValueError("delivery claim completion status must be sent, failed, or skipped")
    cleaned_delivered_at = _clean_iso_datetime(delivered_at, "delivered_at") if delivered_at is not None else None
    cleaned_retry_after = _clean_iso_datetime(retry_after, "retry_after") if retry_after is not None else None
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    now_dt = datetime.now(UTC).replace(microsecond=0)
    now = now_dt.isoformat().replace("+00:00", "Z")
    conn.commit()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT *
            FROM advisor_action_queue_escalation_notifications
            WHERE notification_id = ?
            """,
            (notification_id,),
        ).fetchone()
        if not row:
            raise MarketRecordNotFound(
                f"No advisor action queue escalation notification found for id {notification_id}"
            )
        current = dict(row)
        if cleaned_owner and current["owner_id"] != cleaned_owner:
            raise MarketRecordNotFound(
                f"No advisor action queue escalation notification found for id {notification_id}"
            )
        if current.get("delivery_claim_token") != cleaned_claim_token:
            raise ValueError("delivery claim token is invalid or expired")
        if current.get("delivery_claimed_until") and current["delivery_claimed_until"] <= now:
            raise ValueError("delivery claim token is invalid or expired")
        if cleaned_status == "sent" and not cleaned_delivered_at:
            cleaned_delivered_at = now

        attempt_number = int(current.get("delivery_attempt_count") or 0)
        exhausted_at = None
        exhausted_reason = None
        if cleaned_status == "failed":
            if attempt_number >= max_attempts:
                exhausted_at = now
                exhausted_reason = "max_attempts_reached"
                cleaned_retry_after = None
            elif not cleaned_retry_after:
                delay_seconds = min(300 * (2 ** max(attempt_number - 1, 0)), 3600)
                cleaned_retry_after = (now_dt + timedelta(seconds=delay_seconds)).isoformat().replace(
                    "+00:00", "Z"
                )
        conn.execute(
            """
            UPDATE advisor_action_queue_escalation_notifications
            SET status = ?,
                delivery_notes = COALESCE(?, delivery_notes),
                delivered_at = ?,
                delivery_retry_after = ?,
                delivery_exhausted_at = ?,
                delivery_exhausted_reason = ?,
                delivery_claimed_by = NULL,
                delivery_claimed_at = NULL,
                delivery_claimed_until = NULL,
                delivery_claim_token = NULL,
                updated_at = ?
            WHERE notification_id = ?
            """,
            (
                cleaned_status,
                delivery_notes,
                cleaned_delivered_at,
                cleaned_retry_after if cleaned_status == "failed" else None,
                exhausted_at,
                exhausted_reason,
                now,
                notification_id,
            ),
        )
        conn.execute(
            """
            INSERT INTO advisor_action_queue_escalation_notification_attempts (
                notification_id, owner_id, channel, recipient, status, claim_token,
                claimed_by, attempt_number, delivery_notes, delivered_at, retry_after,
                exhausted_at, exhausted_reason, claimed_at, completed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                notification_id,
                current["owner_id"],
                current["channel"],
                current["recipient"],
                cleaned_status,
                cleaned_claim_token,
                current.get("delivery_claimed_by"),
                attempt_number,
                delivery_notes,
                cleaned_delivered_at,
                cleaned_retry_after if cleaned_status == "failed" else None,
                exhausted_at,
                exhausted_reason,
                current.get("delivery_claimed_at"),
                now,
            ),
        )
        completed_row = conn.execute(
            """
            SELECT *
            FROM advisor_action_queue_escalation_notifications
            WHERE notification_id = ?
            """,
            (notification_id,),
        ).fetchone()
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    notification = _escalation_notification_record_from_row(dict(completed_row))
    return {
        "kind": "advisor_action_queue_task_escalation_notification_delivery_completion",
        "owner_id": notification["owner_id"],
        "data": {"notification": notification},
        "metadata": {
            "notification_id": notification_id,
            "status": cleaned_status,
            "claim_token": cleaned_claim_token,
            "delivery_notes_applied": delivery_notes is not None,
            "delivered_at": cleaned_delivered_at,
            "retry_after": cleaned_retry_after if cleaned_status == "failed" else None,
            "exhausted": exhausted_at is not None,
            "exhausted_at": exhausted_at,
            "exhausted_reason": exhausted_reason,
            "max_attempts": max_attempts,
        },
    }


def list_action_queue_task_escalation_notification_delivery_attempts(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    notification_id: int | None = None,
    status: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_notification_table(conn)
    _ensure_action_queue_escalation_notification_attempt_table(conn)
    cleaned_owner = (owner_id or "").strip() or None
    cleaned_status = _clean_escalation_notification_status(status) if status is not None else None
    if cleaned_status == "prepared":
        raise ValueError("delivery attempt status must be sent, failed, or skipped")
    if notification_id is not None and notification_id < 1:
        raise ValueError("notification_id must be at least 1")
    if limit < 1:
        raise ValueError("limit must be at least 1")

    where: list[str] = []
    params: list[Any] = []
    if cleaned_owner:
        where.append("owner_id = ?")
        params.append(cleaned_owner)
    if notification_id is not None:
        where.append("notification_id = ?")
        params.append(notification_id)
    if cleaned_status:
        where.append("status = ?")
        params.append(cleaned_status)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT *
        FROM advisor_action_queue_escalation_notification_attempts
        {where_sql}
        ORDER BY completed_at DESC, attempt_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return {
        "kind": "advisor_action_queue_task_escalation_notification_delivery_attempts",
        "owner_id": cleaned_owner,
        "data": [_escalation_notification_delivery_attempt_record_from_row(dict(row)) for row in rows],
        "metadata": {
            "owner_id": cleaned_owner,
            "notification_id": notification_id,
            "status": cleaned_status,
            "limit": limit,
            "result_count": len(rows),
        },
    }


def get_action_queue(conn: sqlite3.Connection, queue_id: int, owner_id: str | None) -> dict[str, Any]:
    require_action_queue_tables(conn)
    owner = clean_owner_id(owner_id)
    row = conn.execute(
        """
        SELECT *
        FROM advisor_action_queues
        WHERE queue_id = ? AND owner_id = ?
        """,
        (queue_id, owner),
    ).fetchone()
    if not row:
        raise MarketRecordNotFound(f"No advisor action queue found for id {queue_id}")
    return _queue_from_row(conn, dict(row))


def update_action_queue_task(
    conn: sqlite3.Connection,
    queue_id: int,
    owner_id: str | None,
    task_id: str,
    status: str | None = None,
    notes: str | None = None,
    assigned_to: str | None = None,
    due_at: str | None = None,
    updated_by: str | None = None,
    update_source: str = "single",
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    queue = get_action_queue(conn, queue_id, owner_id)
    previous_task = _queue_task_by_id(conn, queue["owner_id"], queue_id, task_id)
    status_changed = status is not None
    normalized = status.strip().lower() if status_changed else previous_task["status"]
    if normalized not in VALID_TASK_STATUSES:
        raise ValueError(f"Task status must be one of: {', '.join(sorted(VALID_TASK_STATUSES))}")
    assigned_changed = assigned_to is not None
    cleaned_assigned_to = _clean_assigned_to(assigned_to) if assigned_changed else previous_task["assigned_to"]
    due_changed = due_at is not None
    cleaned_due_at = _clean_due_at(due_at) if due_changed else previous_task["due_at"]
    if not status_changed and notes is None and not assigned_changed and not due_changed:
        raise ValueError("At least one task update field is required")
    now = now_utc()
    cursor = conn.execute(
        """
        UPDATE advisor_action_queue_tasks
        SET status = ?, notes = COALESCE(?, notes), updated_at = ?,
            assigned_to = CASE WHEN ? THEN ? ELSE assigned_to END,
            due_at = CASE WHEN ? THEN ? ELSE due_at END,
            completed_at = CASE
                WHEN ? THEN CASE WHEN ? = 'completed' THEN ? ELSE NULL END
                ELSE completed_at
            END
        WHERE queue_id = ? AND task_id = ?
        """,
        (
            normalized,
            notes,
            now,
            assigned_changed,
            cleaned_assigned_to,
            due_changed,
            cleaned_due_at,
            status_changed,
            normalized,
            now,
            queue_id,
            task_id,
        ),
    )
    if cursor.rowcount == 0:
        raise MarketRecordNotFound(f"No advisor action queue task found for id {task_id}")
    _record_task_update(
        conn,
        owner_id=queue["owner_id"],
        queue_id=queue_id,
        task_id=task_id,
        previous_status=previous_task["status"],
        new_status=normalized,
        previous_notes=previous_task["notes"],
        new_notes=notes if notes is not None else previous_task["notes"],
        previous_assigned_to=previous_task["assigned_to"],
        new_assigned_to=cleaned_assigned_to,
        previous_due_at=previous_task["due_at"],
        new_due_at=cleaned_due_at,
        updated_by=updated_by,
        update_source=update_source,
        created_at=now,
    )
    _refresh_queue_rollup(conn, queue_id, queue["owner_id"])
    conn.commit()
    return get_action_queue(conn, queue_id, queue["owner_id"])


def bulk_update_action_queue_tasks(
    conn: sqlite3.Connection,
    owner_id: str | None,
    task_refs: list[dict[str, Any]],
    status: str | None = None,
    notes: str | None = None,
    assigned_to: str | None = None,
    due_at: str | None = None,
    updated_by: str | None = None,
    update_source: str = "bulk",
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    owner = clean_owner_id(owner_id)
    status_changed = status is not None
    normalized = status.strip().lower() if status_changed else None
    if normalized is not None and normalized not in VALID_TASK_STATUSES:
        raise ValueError(f"Task status must be one of: {', '.join(sorted(VALID_TASK_STATUSES))}")
    assigned_changed = assigned_to is not None
    cleaned_assigned_to = _clean_assigned_to(assigned_to) if assigned_changed else None
    due_changed = due_at is not None
    cleaned_due_at = _clean_due_at(due_at) if due_changed else None
    if not status_changed and notes is None and not assigned_changed and not due_changed:
        raise ValueError("At least one task update field is required")
    refs = _clean_task_refs(task_refs)
    if not refs:
        raise ValueError("At least one task reference is required")

    missing = [ref for ref in refs if not _action_queue_task_exists(conn, owner, ref["queue_id"], ref["task_id"])]
    if missing:
        missing_refs = ", ".join(f"{ref['queue_id']}:{ref['task_id']}" for ref in missing)
        raise MarketRecordNotFound(f"No advisor action queue task found for: {missing_refs}")

    previous_tasks = {
        (ref["queue_id"], ref["task_id"]): _queue_task_by_id(conn, owner, ref["queue_id"], ref["task_id"])
        for ref in refs
    }
    now = now_utc()
    for ref in refs:
        previous_task = previous_tasks[(ref["queue_id"], ref["task_id"])]
        next_status = normalized or previous_task["status"]
        next_assigned_to = cleaned_assigned_to if assigned_changed else previous_task["assigned_to"]
        next_due_at = cleaned_due_at if due_changed else previous_task["due_at"]
        conn.execute(
            """
            UPDATE advisor_action_queue_tasks
            SET status = ?, notes = COALESCE(?, notes), updated_at = ?,
                assigned_to = CASE WHEN ? THEN ? ELSE assigned_to END,
                due_at = CASE WHEN ? THEN ? ELSE due_at END,
                completed_at = CASE
                    WHEN ? THEN CASE WHEN ? = 'completed' THEN ? ELSE NULL END
                    ELSE completed_at
                END
            WHERE queue_id = ? AND task_id = ?
            """,
            (
                next_status,
                notes,
                now,
                assigned_changed,
                next_assigned_to,
                due_changed,
                next_due_at,
                status_changed,
                next_status,
                now,
                ref["queue_id"],
                ref["task_id"],
            ),
        )
        _record_task_update(
            conn,
            owner_id=owner,
            queue_id=ref["queue_id"],
            task_id=ref["task_id"],
            previous_status=previous_task["status"],
            new_status=next_status,
            previous_notes=previous_task["notes"],
            new_notes=notes if notes is not None else previous_task["notes"],
            previous_assigned_to=previous_task["assigned_to"],
            new_assigned_to=next_assigned_to,
            previous_due_at=previous_task["due_at"],
            new_due_at=next_due_at,
            updated_by=updated_by,
            update_source=update_source,
            created_at=now,
        )
    touched_queue_ids = sorted({ref["queue_id"] for ref in refs})
    for queue_id in touched_queue_ids:
        _refresh_queue_rollup(conn, queue_id, owner)
    conn.commit()

    updated_tasks = [_queue_task_by_id(conn, owner, ref["queue_id"], ref["task_id"]) for ref in refs]
    updated_queues = [_queue_summary_by_id(conn, owner, queue_id) for queue_id in touched_queue_ids]
    return {
        "kind": "advisor_action_queue_task_bulk_update",
        "owner_id": owner,
        "data": {
            "updated_tasks": updated_tasks,
            "updated_queues": updated_queues,
        },
        "metadata": {
            "owner_id": owner,
            "requested_count": len(refs),
            "updated_count": len(updated_tasks),
            "updated_queue_count": len(updated_queues),
            "status": normalized,
            "notes_applied": notes is not None,
            "assigned_to": cleaned_assigned_to,
            "due_at": cleaned_due_at,
            "updated_by": updated_by,
            "update_source": update_source,
        },
    }


def list_action_queue_task_updates(
    conn: sqlite3.Connection,
    owner_id: str | None,
    queue_id: int | None = None,
    task_id: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_activity_table(conn)
    owner = clean_owner_id(owner_id)
    where = ["updates.owner_id = ?"]
    params: list[Any] = [owner]
    if queue_id is not None:
        where.append("updates.queue_id = ?")
        params.append(queue_id)
    cleaned_task_id = (task_id or "").strip()
    if cleaned_task_id:
        where.append("updates.task_id = ?")
        params.append(cleaned_task_id)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT
            updates.*,
            queues.title AS queue_title,
            queues.focus,
            tasks.title AS task_title,
            tasks.status AS current_status,
            tasks.assigned_to AS current_assigned_to,
            tasks.due_at AS current_due_at
        FROM advisor_action_queue_task_updates AS updates
        JOIN advisor_action_queues AS queues ON queues.queue_id = updates.queue_id
        LEFT JOIN advisor_action_queue_tasks AS tasks
            ON tasks.queue_id = updates.queue_id AND tasks.task_id = updates.task_id
        WHERE {" AND ".join(where)}
        ORDER BY updates.created_at DESC, updates.update_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return {
        "kind": "advisor_action_queue_task_activity",
        "owner_id": owner,
        "data": [_task_update_from_row(dict(row)) for row in rows],
        "metadata": {
            "owner_id": owner,
            "queue_id": queue_id,
            "task_id": cleaned_task_id or None,
            "limit": limit,
            "result_count": len(rows),
        },
    }


def review_action_queue_task_escalation(
    conn: sqlite3.Connection,
    owner_id: str | None,
    queue_id: int,
    task_id: str,
    review_status: str,
    reviewer: str | None = None,
    notes: str | None = None,
    snoozed_until: str | None = None,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_review_table(conn)
    owner = clean_owner_id(owner_id)
    cleaned_task_id = (task_id or "").strip()
    if not cleaned_task_id:
        raise ValueError("task_id is required")
    task = _queue_task_by_id(conn, owner, queue_id, cleaned_task_id)
    cleaned_status = _clean_escalation_review_status(review_status)
    cleaned_snoozed_until = _clean_due_at(snoozed_until) if snoozed_until is not None else None
    if cleaned_status == "snoozed" and not cleaned_snoozed_until:
        raise ValueError("snoozed_until is required when review_status is snoozed")
    now = now_utc()
    cursor = conn.execute(
        """
        INSERT INTO advisor_action_queue_escalation_reviews (
            owner_id, queue_id, task_id, review_status, reviewer,
            notes, snoozed_until, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            owner,
            queue_id,
            cleaned_task_id,
            cleaned_status,
            (reviewer or "").strip() or None,
            notes,
            cleaned_snoozed_until,
            now,
        ),
    )
    conn.commit()
    review = _escalation_review_by_id(conn, int(cursor.lastrowid))
    return {
        "kind": "advisor_action_queue_task_escalation_review",
        "owner_id": owner,
        "data": {
            "review": review,
            "task": task,
        },
        "metadata": {
            "owner_id": owner,
            "queue_id": queue_id,
            "task_id": cleaned_task_id,
            "review_status": cleaned_status,
            "reviewer": (reviewer or "").strip() or None,
        },
    }


def bulk_review_action_queue_task_escalations(
    conn: sqlite3.Connection,
    owner_id: str | None,
    task_refs: list[dict[str, Any]],
    review_status: str,
    reviewer: str | None = None,
    notes: str | None = None,
    snoozed_until: str | None = None,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_review_table(conn)
    owner = clean_owner_id(owner_id)
    refs = _clean_task_refs(task_refs)
    if not refs:
        raise ValueError("At least one task reference is required")
    cleaned_status = _clean_escalation_review_status(review_status)
    cleaned_snoozed_until = _clean_due_at(snoozed_until) if snoozed_until is not None else None
    if cleaned_status == "snoozed" and not cleaned_snoozed_until:
        raise ValueError("snoozed_until is required when review_status is snoozed")
    missing = [ref for ref in refs if not _action_queue_task_exists(conn, owner, ref["queue_id"], ref["task_id"])]
    if missing:
        missing_refs = ", ".join(f"{ref['queue_id']}:{ref['task_id']}" for ref in missing)
        raise MarketRecordNotFound(f"No advisor action queue task found for: {missing_refs}")

    now = now_utc()
    created_review_ids: list[int] = []
    for ref in refs:
        cursor = conn.execute(
            """
            INSERT INTO advisor_action_queue_escalation_reviews (
                owner_id, queue_id, task_id, review_status, reviewer,
                notes, snoozed_until, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                owner,
                ref["queue_id"],
                ref["task_id"],
                cleaned_status,
                (reviewer or "").strip() or None,
                notes,
                cleaned_snoozed_until,
                now,
            ),
        )
        created_review_ids.append(int(cursor.lastrowid))
    conn.commit()

    reviews = [_escalation_review_by_id(conn, review_id) for review_id in created_review_ids]
    reviewed_tasks = [_queue_task_by_id(conn, owner, ref["queue_id"], ref["task_id"]) for ref in refs]
    return {
        "kind": "advisor_action_queue_task_escalation_bulk_review",
        "owner_id": owner,
        "data": {
            "reviews": reviews,
            "tasks": reviewed_tasks,
        },
        "metadata": {
            "owner_id": owner,
            "requested_count": len(refs),
            "reviewed_count": len(reviews),
            "review_status": cleaned_status,
            "reviewer": (reviewer or "").strip() or None,
            "snoozed_until": cleaned_snoozed_until,
        },
    }


def list_action_queue_task_escalation_reviews(
    conn: sqlite3.Connection,
    owner_id: str | None,
    queue_id: int | None = None,
    task_id: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    _ensure_action_queue_escalation_review_table(conn)
    owner = clean_owner_id(owner_id)
    where = ["reviews.owner_id = ?"]
    params: list[Any] = [owner]
    if queue_id is not None:
        where.append("reviews.queue_id = ?")
        params.append(queue_id)
    cleaned_task_id = (task_id or "").strip()
    if cleaned_task_id:
        where.append("reviews.task_id = ?")
        params.append(cleaned_task_id)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT
            reviews.*,
            queues.title AS queue_title,
            queues.focus,
            tasks.title AS task_title,
            tasks.status AS current_status,
            tasks.assigned_to AS current_assigned_to,
            tasks.due_at AS current_due_at
        FROM advisor_action_queue_escalation_reviews AS reviews
        JOIN advisor_action_queues AS queues ON queues.queue_id = reviews.queue_id
        LEFT JOIN advisor_action_queue_tasks AS tasks
            ON tasks.queue_id = reviews.queue_id AND tasks.task_id = reviews.task_id
        WHERE {" AND ".join(where)}
        ORDER BY reviews.created_at DESC, reviews.review_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return {
        "kind": "advisor_action_queue_task_escalation_reviews",
        "owner_id": owner,
        "data": [_escalation_review_from_row(dict(row)) for row in rows],
        "metadata": {
            "owner_id": owner,
            "queue_id": queue_id,
            "task_id": cleaned_task_id or None,
            "limit": limit,
            "result_count": len(rows),
        },
    }


def clean_owner_id(owner_id: str | None) -> str:
    return (owner_id or "default").strip() or "default"


def clean_title(title: str | None, owner_id: str) -> str:
    cleaned = (title or "").strip()
    return cleaned or f"Advisor action queue for {owner_id}"


def now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def require_action_queue_tables(conn: sqlite3.Connection) -> None:
    for table_name in ("advisor_action_queues", "advisor_action_queue_tasks"):
        if not table_exists(conn, table_name):
            raise MarketDataUnavailable(f"Advisor action queue table '{table_name}' is not available")
    _ensure_action_queue_task_columns(conn)


def _ensure_action_queue_task_columns(conn: sqlite3.Connection) -> None:
    _ensure_columns(
        conn,
        "advisor_action_queue_tasks",
        {
            "assigned_to": "assigned_to TEXT",
            "due_at": "due_at TEXT",
        },
    )


def _ensure_action_queue_activity_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS advisor_action_queue_task_updates (
            update_id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id TEXT NOT NULL DEFAULT 'default',
            queue_id INTEGER NOT NULL,
            task_id TEXT NOT NULL,
            previous_status TEXT,
            new_status TEXT NOT NULL,
            previous_notes TEXT,
            new_notes TEXT,
            previous_assigned_to TEXT,
            new_assigned_to TEXT,
            previous_due_at TEXT,
            new_due_at TEXT,
            updated_by TEXT,
            update_source TEXT NOT NULL DEFAULT 'api',
            created_at TEXT NOT NULL,
            FOREIGN KEY(queue_id) REFERENCES advisor_action_queues(queue_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_task_updates_owner
        ON advisor_action_queue_task_updates(owner_id, created_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_task_updates_task
        ON advisor_action_queue_task_updates(queue_id, task_id, created_at DESC)
        """
    )
    _ensure_columns(
        conn,
        "advisor_action_queue_task_updates",
        {
            "previous_assigned_to": "previous_assigned_to TEXT",
            "new_assigned_to": "new_assigned_to TEXT",
            "previous_due_at": "previous_due_at TEXT",
            "new_due_at": "new_due_at TEXT",
        },
    )


def _ensure_action_queue_escalation_review_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS advisor_action_queue_escalation_reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id TEXT NOT NULL DEFAULT 'default',
            queue_id INTEGER NOT NULL,
            task_id TEXT NOT NULL,
            review_status TEXT NOT NULL,
            reviewer TEXT,
            notes TEXT,
            snoozed_until TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(queue_id) REFERENCES advisor_action_queues(queue_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_reviews_owner
        ON advisor_action_queue_escalation_reviews(owner_id, created_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_reviews_task
        ON advisor_action_queue_escalation_reviews(queue_id, task_id, created_at DESC)
        """
    )


def _ensure_action_queue_escalation_notification_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS advisor_action_queue_escalation_notifications (
            notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id TEXT,
            as_of TEXT NOT NULL,
            channel TEXT NOT NULL,
            recipient TEXT,
            status TEXT NOT NULL,
            idempotency_key TEXT NOT NULL UNIQUE,
            filter_json TEXT NOT NULL,
            item_count INTEGER NOT NULL DEFAULT 0,
            payload_json TEXT NOT NULL,
            delivery_notes TEXT,
            delivered_at TEXT,
            delivery_retry_after TEXT,
            delivery_exhausted_at TEXT,
            delivery_exhausted_reason TEXT,
            delivery_claimed_by TEXT,
            delivery_claimed_at TEXT,
            delivery_claimed_until TEXT,
            delivery_claim_token TEXT,
            delivery_attempt_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    _ensure_columns(
        conn,
        "advisor_action_queue_escalation_notifications",
        {
            "delivery_notes": "delivery_notes TEXT",
            "delivered_at": "delivered_at TEXT",
            "delivery_retry_after": "delivery_retry_after TEXT",
            "delivery_exhausted_at": "delivery_exhausted_at TEXT",
            "delivery_exhausted_reason": "delivery_exhausted_reason TEXT",
            "delivery_claimed_by": "delivery_claimed_by TEXT",
            "delivery_claimed_at": "delivery_claimed_at TEXT",
            "delivery_claimed_until": "delivery_claimed_until TEXT",
            "delivery_claim_token": "delivery_claim_token TEXT",
            "delivery_attempt_count": "delivery_attempt_count INTEGER NOT NULL DEFAULT 0",
            "updated_at": "updated_at TEXT",
        },
    )
    conn.execute(
        """
        UPDATE advisor_action_queue_escalation_notifications
        SET updated_at = created_at
        WHERE updated_at IS NULL
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_notifications_owner
        ON advisor_action_queue_escalation_notifications(owner_id, created_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_notifications_channel
        ON advisor_action_queue_escalation_notifications(channel, status, created_at DESC)
        """
    )


def _ensure_action_queue_escalation_notification_attempt_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS advisor_action_queue_escalation_notification_attempts (
            attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
            notification_id INTEGER NOT NULL,
            owner_id TEXT,
            channel TEXT NOT NULL,
            recipient TEXT,
            status TEXT NOT NULL,
            claim_token TEXT NOT NULL,
            claimed_by TEXT,
            attempt_number INTEGER NOT NULL,
            delivery_notes TEXT,
            delivered_at TEXT,
            retry_after TEXT,
            exhausted_at TEXT,
            exhausted_reason TEXT,
            claimed_at TEXT,
            completed_at TEXT NOT NULL,
            FOREIGN KEY(notification_id) REFERENCES advisor_action_queue_escalation_notifications(notification_id)
                ON DELETE CASCADE
        )
        """
    )
    _ensure_columns(
        conn,
        "advisor_action_queue_escalation_notification_attempts",
        {
            "retry_after": "retry_after TEXT",
            "exhausted_at": "exhausted_at TEXT",
            "exhausted_reason": "exhausted_reason TEXT",
        },
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_notification_attempts_owner
        ON advisor_action_queue_escalation_notification_attempts(owner_id, status, completed_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_notification_attempts_notification
        ON advisor_action_queue_escalation_notification_attempts(notification_id, completed_at DESC)
        """
    )


def _ensure_action_queue_escalation_notification_remediation_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS advisor_action_queue_escalation_notification_remediations (
            remediation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            notification_id INTEGER NOT NULL,
            owner_id TEXT,
            channel TEXT NOT NULL,
            recipient TEXT,
            remediation_type TEXT NOT NULL,
            remediation_notes TEXT,
            requeued_by TEXT,
            retry_after TEXT,
            previous_delivery_exhausted_at TEXT,
            previous_delivery_exhausted_reason TEXT,
            previous_delivery_attempt_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY(notification_id) REFERENCES advisor_action_queue_escalation_notifications(notification_id)
                ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_notification_remediations_owner
        ON advisor_action_queue_escalation_notification_remediations(owner_id, created_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_notification_remediations_notification
        ON advisor_action_queue_escalation_notification_remediations(notification_id, created_at DESC)
        """
    )


def _ensure_action_queue_escalation_notification_claim_release_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS advisor_action_queue_escalation_notification_claim_releases (
            release_id INTEGER PRIMARY KEY AUTOINCREMENT,
            notification_id INTEGER NOT NULL,
            owner_id TEXT,
            channel TEXT NOT NULL,
            recipient TEXT,
            status TEXT NOT NULL,
            claim_token TEXT NOT NULL,
            claimed_by TEXT,
            claimed_at TEXT,
            claimed_until TEXT,
            released_by TEXT,
            release_notes TEXT,
            previous_delivery_attempt_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY(notification_id) REFERENCES advisor_action_queue_escalation_notifications(notification_id)
                ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_notification_claim_releases_owner
        ON advisor_action_queue_escalation_notification_claim_releases(owner_id, created_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_notification_claim_releases_notification
        ON advisor_action_queue_escalation_notification_claim_releases(notification_id, created_at DESC)
        """
    )


def _ensure_action_queue_escalation_notification_incident_review_table(
    conn: sqlite3.Connection,
) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS advisor_action_queue_escalation_notification_incident_reviews (
            incident_review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            notification_id INTEGER NOT NULL,
            owner_id TEXT,
            channel TEXT NOT NULL,
            incident_type TEXT NOT NULL,
            incident_status TEXT NOT NULL,
            reviewer TEXT,
            assigned_to TEXT,
            notes TEXT,
            follow_up_at TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(notification_id) REFERENCES advisor_action_queue_escalation_notifications(notification_id)
                ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_notification_incident_reviews_owner
        ON advisor_action_queue_escalation_notification_incident_reviews(owner_id, created_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_notification_incident_reviews_notification
        ON advisor_action_queue_escalation_notification_incident_reviews(notification_id, created_at DESC)
        """
    )


def _ensure_columns(conn: sqlite3.Connection, table_name: str, columns: dict[str, str]) -> None:
    existing = {
        row["name"] if isinstance(row, sqlite3.Row) else row[1]
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    for column_name, ddl in columns.items():
        if column_name not in existing:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {ddl}")


def _tasks_from_followup(followup: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    source_brief = followup.get("source_brief", {})
    gaps = source_brief.get("data_gaps", [])

    if gaps:
        tasks.append(
            _task(
                "resolve-data-gaps",
                "Resolve or disclose data gaps",
                "high",
                "Data gaps are present before customer-facing guidance can be trusted.",
                "Data gaps are either fixed in ingestion or explicitly disclosed in the customer note.",
                blocked=False,
                evidence={"data_gaps": gaps[:10]},
            )
        )

    for item in followup.get("advisor_checklist", []):
        task_id = _slug(item)
        blocked = "Resolve or disclose" not in item and bool(gaps)
        tasks.append(
            _task(
                task_id,
                item,
                "medium" if not blocked else "blocked",
                "Advisor checklist item generated from the customer follow-up pack.",
                "Advisor confirms the checklist item is complete.",
                blocked=blocked,
                evidence={"checklist_item": item},
            )
        )

    for point in followup.get("meeting_agenda", []):
        for agenda_item in point.get("items", [])[:3]:
            tasks.append(
                _task(
                    _slug(f"{point['section']} {agenda_item}"),
                    f"Prepare agenda item: {agenda_item}",
                    "medium",
                    f"Supports meeting section: {point['section']}.",
                    "Agenda item is reviewed and ready for the customer conversation.",
                    blocked=bool(gaps) and point["section"] != "Data quality check",
                    evidence={"meeting_section": point["section"], "agenda_item": agenda_item},
                )
            )

    tasks.append(
        _task(
            "review-customer-email",
            "Review customer email draft",
            "medium" if not gaps else "blocked",
            "Customer copy should be reviewed before sending.",
            "Email is approved, edited, or intentionally withheld.",
            blocked=bool(gaps),
            evidence=followup.get("customer_email", {}),
        )
    )
    return _dedupe(tasks)


def _task(
    task_id: str,
    title: str,
    urgency: str,
    rationale: str,
    completion_criteria: str,
    blocked: bool,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "title": title,
        "urgency": "blocked" if blocked else urgency,
        "status": "blocked" if blocked else "open",
        "rationale": rationale,
        "completion_criteria": completion_criteria,
        "evidence": evidence,
    }


def _dedupe(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for task in tasks:
        original = task["task_id"]
        task_id = original
        suffix = 2
        while task_id in seen:
            task_id = f"{original}-{suffix}"
            suffix += 1
        task["task_id"] = task_id
        seen.add(task_id)
        deduped.append(task)
    return deduped


def _queue_tasks(conn: sqlite3.Connection, queue_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT *
        FROM advisor_action_queue_tasks
        WHERE queue_id = ?
        ORDER BY saved_task_id
        """,
        (queue_id,),
    ).fetchall()
    return [_task_from_row(dict(row)) for row in rows]


def _task_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "saved_task_id": row["saved_task_id"],
        "task_id": row["task_id"],
        "title": row["title"],
        "urgency": row["urgency"],
        "status": row["status"],
        "rationale": row["rationale"],
        "completion_criteria": row["completion_criteria"],
        "evidence": json.loads(row["evidence_json"]),
        "notes": row["notes"],
        "assigned_to": row["assigned_to"],
        "due_at": row["due_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "completed_at": row["completed_at"],
    }


def _queue_from_row(conn: sqlite3.Connection, row: dict[str, Any]) -> dict[str, Any]:
    tasks = _queue_tasks(conn, row["queue_id"])
    return {
        "kind": "saved_advisor_action_queue",
        "queue_id": row["queue_id"],
        "owner_id": row["owner_id"],
        "title": row["title"],
        "focus": row["focus"],
        "status": row["status"],
        "tasks": tasks,
        "task_count": row["task_count"],
        "open_task_count": row["open_task_count"],
        "blocked_task_count": row["blocked_task_count"],
        "completed_task_count": row["completed_task_count"],
        "source_followup": json.loads(row["source_followup_json"]),
        "queue_markdown": row["queue_markdown"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _queue_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "queue_id": row["queue_id"],
        "owner_id": row["owner_id"],
        "title": row["title"],
        "focus": row["focus"],
        "status": row["status"],
        "task_count": row["task_count"],
        "open_task_count": row["open_task_count"],
        "blocked_task_count": row["blocked_task_count"],
        "completed_task_count": row["completed_task_count"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _summary_counts(row: dict[str, Any], include_status: bool = False) -> dict[str, Any]:
    summary = {
        "queue_count": int(row.get("queue_count") or 0),
        "active_queue_count": int(row.get("active_queue_count") or 0),
        "completed_queue_count": int(row.get("completed_queue_count") or 0),
        "task_count": int(row.get("task_count") or 0),
        "open_task_count": int(row.get("open_task_count") or 0),
        "blocked_task_count": int(row.get("blocked_task_count") or 0),
        "completed_task_count": int(row.get("completed_task_count") or 0),
        "last_updated_at": row.get("last_updated_at"),
    }
    if include_status:
        summary = {
            "status": row["status"],
            **{key: value for key, value in summary.items() if key not in {"active_queue_count", "completed_queue_count"}},
        }
    return summary


def _focus_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "focus": row["focus_name"],
        "queue_count": int(row.get("queue_count") or 0),
        "task_count": int(row.get("task_count") or 0),
        "open_task_count": int(row.get("open_task_count") or 0),
        "blocked_task_count": int(row.get("blocked_task_count") or 0),
        "completed_task_count": int(row.get("completed_task_count") or 0),
        "attention_task_count": int(row.get("open_task_count") or 0) + int(row.get("blocked_task_count") or 0),
        "last_updated_at": row.get("last_updated_at"),
    }


def _urgency_summary(row: dict[str, Any]) -> dict[str, Any]:
    open_count = int(row.get("open_task_count") or 0)
    blocked_count = int(row.get("blocked_task_count") or 0)
    return {
        "urgency": row["urgency"],
        "open_task_count": open_count,
        "blocked_task_count": blocked_count,
        "attention_task_count": open_count + blocked_count,
    }


def _task_status_filter(status: str | None) -> list[str]:
    cleaned = (status or "all").strip().lower()
    if cleaned in {"", "all"}:
        return []
    if cleaned == "active":
        return list(ACTIVE_TASK_STATUSES)
    statuses = [part.strip().lower() for part in cleaned.split(",") if part.strip()]
    invalid = [item for item in statuses if item not in VALID_TASK_STATUSES]
    if invalid:
        raise ValueError(f"Task status must be active, all, or one of: {', '.join(sorted(VALID_TASK_STATUSES))}")
    return statuses


def _clean_assigned_to(assigned_to: str | None) -> str | None:
    cleaned = (assigned_to or "").strip()
    if not cleaned or cleaned.lower() == "unassigned":
        return None
    return cleaned


def _clean_due_at(due_at: str | None) -> str | None:
    cleaned = (due_at or "").strip()
    if not cleaned:
        return None
    try:
        if len(cleaned) == 10:
            datetime.fromisoformat(f"{cleaned}T00:00:00")
        else:
            datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("due_at must be an ISO-8601 date or datetime") from exc
    return cleaned


def _clean_iso_datetime(value: str | None, field_name: str) -> str | None:
    cleaned = (value or "").strip()
    if not cleaned:
        return None
    if len(cleaned) == 10:
        raise ValueError(f"{field_name} must be an ISO-8601 datetime")
    try:
        datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 datetime") from exc
    return cleaned


def _parse_iso_datetime(value: str | None) -> datetime | None:
    cleaned = (value or "").strip()
    if not cleaned:
        return None
    parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _clean_as_of_date(as_of: str | None) -> str:
    cleaned = (as_of or "").strip()
    if not cleaned:
        return datetime.now(UTC).date().isoformat()
    try:
        if len(cleaned) == 10:
            return datetime.fromisoformat(f"{cleaned}T00:00:00").date().isoformat()
        return datetime.fromisoformat(cleaned.replace("Z", "+00:00")).date().isoformat()
    except ValueError as exc:
        raise ValueError("as_of must be an ISO-8601 date or datetime") from exc


def _clean_escalation_review_status(review_status: str | None) -> str:
    cleaned = (review_status or "").strip().lower()
    if cleaned not in VALID_ESCALATION_REVIEW_STATUSES:
        raise ValueError(
            f"review_status must be one of: {', '.join(sorted(VALID_ESCALATION_REVIEW_STATUSES))}"
        )
    return cleaned


def _clean_escalation_notification_status(status: str | None) -> str:
    cleaned = (status or "prepared").strip().lower()
    if cleaned not in VALID_ESCALATION_NOTIFICATION_STATUSES:
        raise ValueError(
            f"notification status must be one of: {', '.join(sorted(VALID_ESCALATION_NOTIFICATION_STATUSES))}"
        )
    return cleaned


def _clean_delivery_incident_type(incident_type: str | None) -> str:
    cleaned = (incident_type or "").strip().lower()
    if cleaned not in VALID_DELIVERY_INCIDENT_TYPES:
        raise ValueError(
            f"incident_type must be one of: {', '.join(sorted(VALID_DELIVERY_INCIDENT_TYPES))}"
        )
    return cleaned


def _clean_delivery_incident_action_id(action_id: str | None) -> str:
    cleaned = (action_id or "").strip().lower()
    if cleaned not in VALID_DELIVERY_INCIDENT_ACTION_IDS:
        raise ValueError(
            f"action_id must be one of: {', '.join(sorted(VALID_DELIVERY_INCIDENT_ACTION_IDS))}"
        )
    return cleaned


def _clean_delivery_incident_review_status(incident_status: str | None) -> str:
    cleaned = (incident_status or "").strip().lower()
    if cleaned not in VALID_DELIVERY_INCIDENT_REVIEW_STATUSES:
        raise ValueError(
            f"incident_status must be one of: {', '.join(sorted(VALID_DELIVERY_INCIDENT_REVIEW_STATUSES))}"
        )
    return cleaned


def _clean_delivery_incident_follow_up_status(follow_up_status: str | None) -> str | None:
    cleaned = (follow_up_status or "").strip().lower()
    if not cleaned:
        return None
    if cleaned not in VALID_DELIVERY_INCIDENT_FOLLOW_UP_STATUSES:
        raise ValueError(
            f"follow_up_status must be one of: {', '.join(sorted(VALID_DELIVERY_INCIDENT_FOLLOW_UP_STATUSES))}"
        )
    return cleaned


def _clean_task_refs(task_refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for ref in task_refs:
        queue_id = int(ref.get("queue_id") or 0)
        task_id = str(ref.get("task_id") or "").strip()
        if queue_id <= 0 or not task_id:
            raise ValueError("Each task reference requires queue_id and task_id")
        key = (queue_id, task_id)
        if key in seen:
            continue
        seen.add(key)
        refs.append({"queue_id": queue_id, "task_id": task_id})
    return refs


def _clean_delivery_incident_refs(incident_refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for ref in incident_refs:
        try:
            notification_id = int(ref.get("notification_id") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("Each incident reference requires notification_id and incident_type") from exc
        incident_type = _clean_delivery_incident_type(ref.get("incident_type"))
        if notification_id <= 0:
            raise ValueError("Each incident reference requires notification_id and incident_type")
        key = (notification_id, incident_type)
        if key in seen:
            continue
        seen.add(key)
        refs.append({"notification_id": notification_id, "incident_type": incident_type})
    return refs


def _action_queue_task_exists(conn: sqlite3.Connection, owner_id: str, queue_id: int, task_id: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM advisor_action_queue_tasks AS tasks
        JOIN advisor_action_queues AS queues ON queues.queue_id = tasks.queue_id
        WHERE queues.owner_id = ? AND queues.queue_id = ? AND tasks.task_id = ?
        """,
        (owner_id, queue_id, task_id),
    ).fetchone()
    return row is not None


def _queue_task_by_id(conn: sqlite3.Connection, owner_id: str, queue_id: int, task_id: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT
            queues.queue_id,
            queues.owner_id,
            queues.title AS queue_title,
            queues.focus,
            queues.status AS queue_status,
            queues.updated_at AS queue_updated_at,
            tasks.saved_task_id,
            tasks.task_id,
            tasks.title AS task_title,
            tasks.urgency,
            tasks.status AS task_status,
            tasks.rationale,
            tasks.completion_criteria,
            tasks.notes,
            tasks.assigned_to,
            tasks.due_at,
            tasks.created_at AS task_created_at,
            tasks.updated_at AS task_updated_at,
            tasks.completed_at
        FROM advisor_action_queue_tasks AS tasks
        JOIN advisor_action_queues AS queues ON queues.queue_id = tasks.queue_id
        WHERE queues.owner_id = ? AND queues.queue_id = ? AND tasks.task_id = ?
        """,
        (owner_id, queue_id, task_id),
    ).fetchone()
    if not row:
        raise MarketRecordNotFound(f"No advisor action queue task found for id {task_id}")
    return _queue_task_summary(dict(row))


def _queue_summary_by_id(conn: sqlite3.Connection, owner_id: str, queue_id: int) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT *
        FROM advisor_action_queues
        WHERE owner_id = ? AND queue_id = ?
        """,
        (owner_id, queue_id),
    ).fetchone()
    if not row:
        raise MarketRecordNotFound(f"No advisor action queue found for id {queue_id}")
    return _queue_summary(dict(row))


def _workload_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    fields = [
        "task_count",
        "open_task_count",
        "blocked_task_count",
        "deferred_task_count",
        "high_urgency_task_count",
        "unscheduled_task_count",
        "overdue_task_count",
        "due_today_task_count",
        "due_next_7_days_task_count",
    ]
    totals = {field: sum(int(row.get(field) or 0) for row in rows) for field in fields}
    totals["assigned_task_count"] = sum(int(row.get("task_count") or 0) for row in rows if row.get("assignee") != "unassigned")
    totals["unassigned_task_count"] = sum(int(row.get("task_count") or 0) for row in rows if row.get("assignee") == "unassigned")
    return totals


def _workload_assignee_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "assignee": row["assignee"],
        "task_count": int(row.get("task_count") or 0),
        "open_task_count": int(row.get("open_task_count") or 0),
        "blocked_task_count": int(row.get("blocked_task_count") or 0),
        "deferred_task_count": int(row.get("deferred_task_count") or 0),
        "high_urgency_task_count": int(row.get("high_urgency_task_count") or 0),
        "unscheduled_task_count": int(row.get("unscheduled_task_count") or 0),
        "overdue_task_count": int(row.get("overdue_task_count") or 0),
        "due_today_task_count": int(row.get("due_today_task_count") or 0),
        "due_next_7_days_task_count": int(row.get("due_next_7_days_task_count") or 0),
        "next_due_at": row.get("next_due_at"),
    }


def _task_escalation_summary(row: dict[str, Any], as_of_date: str) -> dict[str, Any]:
    task = _queue_task_summary(row)
    due_date = _due_date_part(task.get("due_at"))
    days_overdue = _days_overdue(due_date, as_of_date)
    reasons = _task_escalation_reasons(task, due_date, as_of_date)
    latest_review = _latest_escalation_review_from_row(row)
    review_status = latest_review["review_status"] if latest_review else "unreviewed"
    task.update(
        {
            "escalation_reasons": reasons,
            "severity": _escalation_severity(reasons),
            "days_overdue": days_overdue,
            "recommended_action": _escalation_recommended_action(reasons),
            "review_status": review_status,
            "reviewed_at": latest_review["created_at"] if latest_review else None,
            "reviewer": latest_review["reviewer"] if latest_review else None,
            "snoozed_until": latest_review["snoozed_until"] if latest_review else None,
            "latest_review": latest_review,
        }
    )
    return task


def _due_date_part(due_at: str | None) -> str | None:
    cleaned = (due_at or "").strip()
    if not cleaned:
        return None
    return cleaned[:10]


def _days_overdue(due_date: str | None, as_of_date: str) -> int:
    if not due_date or due_date >= as_of_date:
        return 0
    due = datetime.fromisoformat(f"{due_date}T00:00:00").date()
    as_of = datetime.fromisoformat(f"{as_of_date}T00:00:00").date()
    return max((as_of - due).days, 0)


def _task_escalation_reasons(task: dict[str, Any], due_date: str | None, as_of_date: str) -> list[str]:
    reasons: list[str] = []
    if due_date and due_date < as_of_date:
        reasons.append("overdue")
    if task.get("status") == "blocked":
        reasons.append("blocked")
    if due_date and due_date == as_of_date:
        reasons.append("due_today")
    if task.get("urgency") == "high":
        reasons.append("high_urgency")
    if not (task.get("assigned_to") or "").strip():
        reasons.append("unassigned")
    return reasons


def _escalation_severity(reasons: list[str]) -> str:
    if "overdue" in reasons or "blocked" in reasons:
        return "critical"
    if "due_today" in reasons or "high_urgency" in reasons:
        return "high"
    if "unassigned" in reasons:
        return "medium"
    return "low"


def _escalation_recommended_action(reasons: list[str]) -> str:
    if "overdue" in reasons:
        return "Review immediately, reset owner and due date, or close the task."
    if "blocked" in reasons:
        return "Remove the blocker or assign a manager to unblock execution."
    if "due_today" in reasons:
        return "Confirm completion plan before end of day."
    if "high_urgency" in reasons:
        return "Verify the customer-facing action is owned and scheduled."
    if "unassigned" in reasons:
        return "Assign an advisor and set an accountable next step."
    return "Review task context and decide the next accountable action."


def _escalation_inbox_status(item: dict[str, Any], as_of_date: str) -> str:
    review_status = item.get("review_status") or "unreviewed"
    if review_status == "resolved":
        return "resolved"
    if review_status == "snoozed":
        snoozed_until = _due_date_part(item.get("snoozed_until"))
        if snoozed_until and snoozed_until > as_of_date:
            return "snoozed"
        return "snooze_expired"
    if review_status == "needs_followup":
        return "needs_followup"
    if review_status == "acknowledged":
        return "active_review"
    return "new"


def _escalation_inbox_totals(escalations: list[dict[str, Any]]) -> dict[str, int]:
    statuses = ["new", "active_review", "needs_followup", "snooze_expired"]
    return {
        "actionable_task_count": len(escalations),
        **{
            f"{status}_task_count": sum(1 for item in escalations if item.get("inbox_status") == status)
            for status in statuses
        },
        "critical_task_count": sum(1 for item in escalations if item["severity"] == "critical"),
        "high_task_count": sum(1 for item in escalations if item["severity"] == "high"),
        "overdue_task_count": sum(1 for item in escalations if "overdue" in item["escalation_reasons"]),
        "blocked_task_count": sum(1 for item in escalations if "blocked" in item["escalation_reasons"]),
    }


def _escalation_notification_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "owner_id": item["owner_id"],
        "queue_id": item["queue_id"],
        "task_id": item["task_id"],
        "title": item["title"],
        "focus": item["focus"],
        "severity": item["severity"],
        "inbox_status": item["inbox_status"],
        "review_status": item["review_status"],
        "assigned_to": item["assigned_to"],
        "due_at": item["due_at"],
        "days_overdue": item["days_overdue"],
        "escalation_reasons": item["escalation_reasons"],
        "recommended_action": item["recommended_action"],
        "task_url": item["task_url"],
    }


def _escalation_notification_summary(inbox: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    item_count = len(items)
    critical_count = sum(1 for item in items if item["severity"] == "critical")
    high_count = sum(1 for item in items if item["severity"] == "high")
    expired_snooze_count = sum(1 for item in items if item["inbox_status"] == "snooze_expired")
    if item_count:
        headline = f"{item_count} manager escalation{'s' if item_count != 1 else ''} need attention"
    else:
        headline = "No manager escalations need attention"
    return {
        "headline": headline,
        "as_of": inbox["as_of"],
        "item_count": item_count,
        "critical_task_count": critical_count,
        "high_task_count": high_count,
        "snooze_expired_task_count": expired_snooze_count,
        "excluded_snoozed_task_count": inbox["totals"]["excluded_snoozed_task_count"],
        "excluded_resolved_task_count": inbox["totals"]["excluded_resolved_task_count"],
    }


def _escalation_notification_markdown(summary: dict[str, Any], items: list[dict[str, Any]]) -> str:
    lines = [f"### {summary['headline']} as of {summary['as_of']}"]
    if not items:
        lines.append("")
        lines.append("No immediate manager action is required.")
        return "\n".join(lines)
    lines.append(
        f"Critical: {summary['critical_task_count']} | High: {summary['high_task_count']} | "
        f"Expired snoozes: {summary['snooze_expired_task_count']}"
    )
    for item in items:
        assignee = item["assigned_to"] or "unassigned"
        due_at = item["due_at"] or "unscheduled"
        reasons = ", ".join(item["escalation_reasons"])
        lines.append(
            f"- [{item['severity']}] {item['owner_id']} / {assignee}: {item['title']} "
            f"(due {due_at}; {item['inbox_status']}; {reasons})"
        )
    return "\n".join(lines)


def _escalation_notification_filter_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata", {})
    return {
        "scope": metadata.get("scope"),
        "owner_id": metadata.get("owner_id"),
        "as_of": metadata.get("as_of"),
        "severity": metadata.get("severity"),
        "inbox_status": metadata.get("inbox_status"),
        "assigned_to": metadata.get("assigned_to"),
        "focus": metadata.get("focus"),
        "limit": metadata.get("limit"),
    }


def _escalation_notification_idempotency_key(
    payload: dict[str, Any], channel: str, recipient: str | None
) -> str:
    material = {
        "channel": channel,
        "recipient": recipient,
        "filters": _escalation_notification_filter_metadata(payload),
        "items": [
            {
                "owner_id": item["owner_id"],
                "queue_id": item["queue_id"],
                "task_id": item["task_id"],
                "severity": item["severity"],
                "inbox_status": item["inbox_status"],
            }
            for item in payload.get("items", [])
        ],
    }
    digest = hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()
    return f"escalation-notification:{digest}"


def _escalation_notification_claim_token(
    notification_id: int, claimed_by: str, claimed_at: str, attempt_count: int
) -> str:
    material = {
        "notification_id": notification_id,
        "claimed_by": claimed_by,
        "claimed_at": claimed_at,
        "attempt_count": attempt_count,
    }
    digest = hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()
    return f"escalation-notification-claim:{digest}"


def _escalation_notification_by_key(conn: sqlite3.Connection, idempotency_key: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT *
        FROM advisor_action_queue_escalation_notifications
        WHERE idempotency_key = ?
        """,
        (idempotency_key,),
    ).fetchone()
    if not row:
        return None
    return _escalation_notification_record_from_row(dict(row), include_payload_json=True)


def _escalation_notification_by_id(conn: sqlite3.Connection, notification_id: int) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT *
        FROM advisor_action_queue_escalation_notifications
        WHERE notification_id = ?
        """,
        (notification_id,),
    ).fetchone()
    if not row:
        raise MarketRecordNotFound(f"No advisor action queue escalation notification found for id {notification_id}")
    return _escalation_notification_record_from_row(dict(row), include_payload_json=True)


def _escalation_notification_incident_review_by_id(
    conn: sqlite3.Connection, incident_review_id: int
) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT *
        FROM advisor_action_queue_escalation_notification_incident_reviews
        WHERE incident_review_id = ?
        """,
        (incident_review_id,),
    ).fetchone()
    if not row:
        raise MarketRecordNotFound(
            f"No advisor action queue escalation notification incident review found for id {incident_review_id}"
        )
    return _escalation_notification_incident_review_record_from_row(dict(row))


def _escalation_notification_record_from_row(
    row: dict[str, Any], include_payload_json: bool = False
) -> dict[str, Any]:
    payload = json.loads(row["payload_json"])
    record = {
        "notification_id": row["notification_id"],
        "owner_id": row["owner_id"],
        "as_of": row["as_of"],
        "channel": row["channel"],
        "recipient": row["recipient"],
        "status": row["status"],
        "idempotency_key": row["idempotency_key"],
        "filter": json.loads(row["filter_json"]),
        "item_count": row["item_count"],
        "payload_summary": payload.get("summary", {}),
        "delivery_notes": row.get("delivery_notes"),
        "delivered_at": row.get("delivered_at"),
        "delivery_retry_after": row.get("delivery_retry_after"),
        "delivery_exhausted_at": row.get("delivery_exhausted_at"),
        "delivery_exhausted_reason": row.get("delivery_exhausted_reason"),
        "delivery_claimed_by": row.get("delivery_claimed_by"),
        "delivery_claimed_at": row.get("delivery_claimed_at"),
        "delivery_claimed_until": row.get("delivery_claimed_until"),
        "delivery_claim_token": row.get("delivery_claim_token"),
        "delivery_attempt_count": row.get("delivery_attempt_count") or 0,
        "created_at": row["created_at"],
        "updated_at": row.get("updated_at"),
    }
    if include_payload_json:
        record["payload_json"] = row["payload_json"]
    return record


def _escalation_notification_delivery_queue_item(row: dict[str, Any]) -> dict[str, Any]:
    record = _escalation_notification_record_from_row(row, include_payload_json=True)
    payload_json = record.pop("payload_json")
    record["payload"] = json.loads(payload_json)
    if record["status"] == "failed":
        record["delivery_action"] = "retry_failed"
        record["delivery_reason"] = "failed_delivery"
        record["priority"] = "high"
    else:
        record["delivery_action"] = "send_prepared"
        record["delivery_reason"] = "stale_prepared"
        record["priority"] = "medium"
    return record


def _escalation_notification_delivery_claim_item(
    row: dict[str, Any], now_dt: datetime, expiring_within_seconds: int
) -> dict[str, Any]:
    record = _escalation_notification_record_from_row(row, include_payload_json=True)
    payload_json = record.pop("payload_json")
    record["payload"] = json.loads(payload_json)
    claim_until = _parse_iso_datetime(row.get("delivery_claimed_until"))
    seconds_until_expiry = int((claim_until - now_dt).total_seconds()) if claim_until else 0
    if seconds_until_expiry <= 0:
        lease_state = "expired"
        priority = "critical"
        delivery_action = "reclaim_available"
    elif seconds_until_expiry <= expiring_within_seconds:
        lease_state = "expiring_soon"
        priority = "high"
        delivery_action = "renew_claim"
    else:
        lease_state = "active"
        priority = "medium"
        delivery_action = "monitor_claim"
    record["lease_state"] = lease_state
    record["lease_seconds_remaining"] = max(seconds_until_expiry, 0)
    record["lease_seconds_overdue"] = max(-seconds_until_expiry, 0)
    record["delivery_action"] = delivery_action
    record["priority"] = priority
    return record


def _escalation_notification_delivery_incident_item(
    row: dict[str, Any],
    incident_type: str,
    incident_reason: str,
    delivery_action: str,
    priority: str,
    now_dt: datetime,
    expiring_within_seconds: int,
) -> dict[str, Any]:
    record = _escalation_notification_record_from_row(row, include_payload_json=True)
    payload_json = record.pop("payload_json")
    record["payload"] = json.loads(payload_json)
    record["incident_type"] = incident_type
    record["incident_reason"] = incident_reason
    record["delivery_action"] = delivery_action
    record["priority"] = priority
    if row.get("delivery_exhausted_at"):
        record["deadletter_reason"] = row.get("delivery_exhausted_reason") or "delivery_exhausted"
        record["deadlettered_at"] = row.get("delivery_exhausted_at")
    if row.get("delivery_claimed_until"):
        claim_until = _parse_iso_datetime(row.get("delivery_claimed_until"))
        seconds_until_expiry = int((claim_until - now_dt).total_seconds()) if claim_until else 0
        if seconds_until_expiry <= 0:
            lease_state = "expired"
        elif seconds_until_expiry <= expiring_within_seconds:
            lease_state = "expiring_soon"
        else:
            lease_state = "active"
        record["lease_state"] = lease_state
        record["lease_seconds_remaining"] = max(seconds_until_expiry, 0)
        record["lease_seconds_overdue"] = max(-seconds_until_expiry, 0)
    return record


def _delivery_incident_sort_at(item: dict[str, Any]) -> str:
    return (
        item.get("deadlettered_at")
        or item.get("delivery_claimed_until")
        or item.get("delivery_retry_after")
        or item.get("updated_at")
        or item.get("created_at")
        or ""
    )


def _delivery_incident_timeline(
    notification: dict[str, Any],
    incident: dict[str, Any],
    reviews: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    remediations: list[dict[str, Any]],
    claim_releases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    notification_id = notification["notification_id"]
    events: list[dict[str, Any]] = []

    def add_event(
        event_type: str,
        event_at: str | None,
        source: str,
        summary: str,
        sort_priority: int,
        **fields: Any,
    ) -> None:
        event = {
            "event_type": event_type,
            "event_at": event_at,
            "source": source,
            "summary": summary,
            "notification_id": notification_id,
            "_sort_priority": sort_priority,
        }
        event.update({key: value for key, value in fields.items() if value is not None})
        events.append(event)

    add_event(
        "notification_created",
        notification.get("created_at"),
        "notification",
        "Notification record created",
        10,
        status=notification.get("status"),
        channel=notification.get("channel"),
        recipient=notification.get("recipient"),
    )
    if notification.get("updated_at") and notification.get("updated_at") != notification.get("created_at"):
        add_event(
            "notification_updated",
            notification.get("updated_at"),
            "notification",
            "Notification record last updated",
            20,
            status=notification.get("status"),
            delivery_attempt_count=notification.get("delivery_attempt_count"),
        )
    if incident.get("deadlettered_at"):
        add_event(
            "deadlettered",
            incident.get("deadlettered_at"),
            "incident",
            "Delivery exhausted and entered the deadletter queue",
            40,
            incident_type=incident.get("incident_type"),
            incident_reason=incident.get("incident_reason"),
            deadletter_reason=incident.get("deadletter_reason"),
            delivery_attempt_count=incident.get("delivery_attempt_count"),
        )
    else:
        add_event(
            "incident_detected",
            _delivery_incident_sort_at(incident),
            "incident",
            f"Delivery incident detected: {incident.get('incident_type')}",
            40,
            incident_type=incident.get("incident_type"),
            incident_reason=incident.get("incident_reason"),
            delivery_action=incident.get("delivery_action"),
        )

    for attempt in attempts:
        attempt_number = attempt.get("attempt_number")
        add_event(
            "delivery_attempt_completed",
            attempt.get("completed_at") or attempt.get("delivered_at") or attempt.get("claimed_at"),
            "delivery_attempt",
            f"Delivery attempt {attempt_number or '?'} {attempt.get('status')}",
            30,
            attempt_id=attempt.get("attempt_id"),
            attempt_number=attempt_number,
            status=attempt.get("status"),
            claimed_by=attempt.get("claimed_by"),
            exhausted_reason=attempt.get("exhausted_reason"),
            retry_after=attempt.get("retry_after"),
        )
    for review in reviews:
        add_event(
            "incident_reviewed",
            review.get("created_at"),
            "incident_review",
            f"Incident reviewed: {review.get('incident_status')}",
            50,
            incident_review_id=review.get("incident_review_id"),
            incident_type=review.get("incident_type"),
            incident_status=review.get("incident_status"),
            reviewer=review.get("reviewer"),
            assigned_to=review.get("assigned_to"),
            follow_up_at=review.get("follow_up_at"),
        )
    for remediation in remediations:
        add_event(
            "deadletter_remediated",
            remediation.get("created_at"),
            "deadletter_remediation",
            f"Deadletter remediation: {remediation.get('remediation_type')}",
            60,
            remediation_id=remediation.get("remediation_id"),
            remediation_type=remediation.get("remediation_type"),
            requeued_by=remediation.get("requeued_by"),
            retry_after=remediation.get("retry_after"),
            previous_delivery_exhausted_reason=remediation.get("previous_delivery_exhausted_reason"),
        )
    for release in claim_releases:
        add_event(
            "delivery_claim_released",
            release.get("created_at"),
            "claim_release",
            "Delivery claim released",
            35,
            release_id=release.get("release_id"),
            status=release.get("status"),
            claimed_by=release.get("claimed_by"),
            released_by=release.get("released_by"),
            claimed_until=release.get("claimed_until"),
        )

    events.sort(key=_delivery_incident_timeline_sort_key)
    for index, event in enumerate(events, start=1):
        event.pop("_sort_priority", None)
        event["timeline_index"] = index
    return events


def _delivery_incident_timeline_sort_key(event: dict[str, Any]) -> tuple[datetime, int, str, str]:
    event_at = event.get("event_at")
    parsed: datetime | None
    try:
        parsed = _parse_iso_datetime(event_at)
    except ValueError:
        parsed = None
    source_id = (
        event.get("attempt_id")
        or event.get("release_id")
        or event.get("incident_review_id")
        or event.get("remediation_id")
        or 0
    )
    return (
        parsed or datetime.max.replace(tzinfo=UTC),
        int(event.get("_sort_priority") or 99),
        str(event.get("event_type") or ""),
        str(source_id),
    )


def _delivery_incident_next_actions(
    notification: dict[str, Any], incident: dict[str, Any]
) -> list[dict[str, Any]]:
    notification_id = notification["notification_id"]
    owner_id = notification["owner_id"]
    channel = notification["channel"]
    incident_type = incident.get("incident_type")
    latest_review = incident.get("latest_review") or {}
    latest_status = latest_review.get("incident_status")
    review_path = (
        "/agents/action-queues/tasks/escalations/inbox/notification/"
        f"delivery-incidents/{notification_id}/review"
    )
    actions: list[dict[str, Any]] = []

    def add_action(
        action_id: str,
        label: str,
        priority: str,
        reason: str,
        method: str,
        path: str,
        request_body_template: dict[str, Any],
        query_params: dict[str, Any] | None = None,
    ) -> None:
        actions.append(
            {
                "action_id": action_id,
                "label": label,
                "priority": priority,
                "reason": reason,
                "method": method,
                "path": path,
                "query_params": query_params or {"owner_id": owner_id},
                "request_body_template": request_body_template,
            }
        )

    if latest_status == "resolved":
        return actions

    if latest_status not in {"assigned", "snoozed"}:
        add_action(
            "assign_incident",
            "Assign incident",
            "high",
            "No active operator assignment is recorded for this incident.",
            "POST",
            review_path,
            {
                "incident_type": incident_type,
                "incident_status": "assigned",
                "reviewer": "<operator_id>",
                "assigned_to": "<operator_id>",
                "notes": "<triage notes>",
            },
        )

    if latest_status == "needs_followup" or incident.get("follow_up_status") in {"overdue", "due_soon"}:
        add_action(
            "review_follow_up",
            "Review follow-up",
            "high" if incident.get("follow_up_status") == "overdue" else "medium",
            "The latest incident review has a follow-up that is due or close to due.",
            "POST",
            review_path,
            {
                "incident_type": incident_type,
                "incident_status": "assigned",
                "reviewer": "<operator_id>",
                "assigned_to": latest_review.get("assigned_to") or "<operator_id>",
                "notes": "<follow-up decision>",
            },
        )

    if incident_type == "deadletter":
        add_action(
            "requeue_deadletter",
            "Requeue deadletter",
            "critical",
            "Delivery exhausted and needs operator remediation before it can be retried.",
            "POST",
            (
                "/agents/action-queues/tasks/escalations/inbox/notification/"
                f"deadletters/{notification_id}/requeue"
            ),
            {
                "delivery_notes": "<remediation notes>",
                "requeued_by": "<operator_id>",
                "retry_after": None,
            },
        )
    elif incident_type in {"expired_claim", "expiring_claim"}:
        add_action(
            "release_delivery_claim",
            "Release delivery claim",
            "critical" if incident_type == "expired_claim" else "high",
            "The delivery lease is expired or close to expiry and may block another worker.",
            "POST",
            (
                "/agents/action-queues/tasks/escalations/inbox/notification/"
                f"delivery-claim/{notification_id}/release"
            ),
            {
                "claim_token": notification.get("delivery_claim_token") or "<claim_token>",
                "release_notes": "<release reason>",
                "released_by": "<operator_id>",
            },
        )
    elif incident_type in {"retry_ready", "stale_prepared"}:
        add_action(
            "claim_delivery",
            "Claim delivery work",
            "high",
            "This owner and channel have retry-ready or stale prepared delivery work.",
            "POST",
            "/agents/action-queues/tasks/escalations/inbox/notification/delivery-claim",
            {"claimed_by": "<worker_id>", "lease_seconds": 300},
            query_params={"owner_id": owner_id, "channel": channel},
        )

    if latest_status != "snoozed":
        add_action(
            "resolve_incident",
            "Resolve incident",
            "low",
            "Use after remediation or verification confirms the incident no longer needs operator action.",
            "POST",
            review_path,
            {
                "incident_type": incident_type,
                "incident_status": "resolved",
                "reviewer": "<operator_id>",
                "notes": "<resolution notes>",
            },
        )

    return actions


def _delivery_incident_suppression(item: dict[str, Any], now_dt: datetime) -> tuple[bool, str | None]:
    latest_review = item.get("latest_review") or {}
    incident_status = latest_review.get("incident_status")
    if incident_status == "resolved":
        return True, "resolved"
    if incident_status == "snoozed":
        follow_up_at = _parse_iso_datetime(latest_review.get("follow_up_at"))
        if follow_up_at and follow_up_at > now_dt:
            return True, "snoozed_until_follow_up"
    return False, None


def _delivery_incident_summary_bucket(item: dict[str, Any]) -> str:
    suppression_reason = item.get("suppression_reason")
    if suppression_reason == "resolved":
        return "resolved"
    if suppression_reason == "snoozed_until_follow_up":
        return "snoozed"
    return "actionable"


def _delivery_incident_follow_up_status(
    latest_review: dict[str, Any], now_dt: datetime, due_soon_cutoff_dt: datetime
) -> str:
    follow_up_at = _parse_iso_datetime(latest_review.get("follow_up_at"))
    if follow_up_at is None:
        return "missing"
    if follow_up_at <= now_dt:
        return "overdue"
    if follow_up_at <= due_soon_cutoff_dt:
        return "due_soon"
    return "future"


def _delivery_incident_matches_filters(
    item: dict[str, Any], assigned_to: str | None, follow_up_status: str | None
) -> bool:
    if assigned_to:
        latest_review = item.get("latest_review") or {}
        item_assignee = latest_review.get("assigned_to") or "unassigned"
        if item_assignee != assigned_to:
            return False
    if follow_up_status and item.get("follow_up_status") != follow_up_status:
        return False
    return True


def _increment_delivery_incident_count_row(
    rows: dict[str, dict[str, Any]], label_name: str, label_value: str, bucket: str
) -> None:
    row = rows.setdefault(
        label_value,
        {
            label_name: label_value,
            "total_count": 0,
            "actionable_count": 0,
            "resolved_count": 0,
            "snoozed_count": 0,
        },
    )
    row["total_count"] += 1
    row[f"{bucket}_count"] += 1


def _ordered_delivery_incident_count_rows(
    rows: dict[str, dict[str, Any]], label_name: str, preferred_order: list[str]
) -> list[dict[str, Any]]:
    order = {value: index for index, value in enumerate(preferred_order)}
    return sorted(
        rows.values(),
        key=lambda row: (
            order.get(row[label_name], len(order)),
            row[label_name],
        ),
    )


def _ordered_delivery_incident_workload_rows(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows.values(),
        key=lambda row: (
            row["assigned_to"] == "unassigned",
            -row["follow_up_overdue_count"],
            -row["actionable_count"],
            -row["critical_count"],
            row["assigned_to"],
        ),
    )


def _latest_escalation_notification_incident_reviews(
    conn: sqlite3.Connection, incidents: list[dict[str, Any]]
) -> dict[tuple[int, str], dict[str, Any]]:
    reviews: dict[tuple[int, str], dict[str, Any]] = {}
    seen: set[tuple[int, str]] = set()
    for incident in incidents:
        notification_id = int(incident.get("notification_id") or 0)
        incident_type = str(incident.get("incident_type") or "").strip()
        key = (notification_id, incident_type)
        if notification_id <= 0 or not incident_type or key in seen:
            continue
        seen.add(key)
        row = conn.execute(
            """
            SELECT *
            FROM advisor_action_queue_escalation_notification_incident_reviews
            WHERE notification_id = ? AND incident_type = ?
            ORDER BY created_at DESC, incident_review_id DESC
            LIMIT 1
            """,
            (notification_id, incident_type),
        ).fetchone()
        if row:
            reviews[key] = _escalation_notification_incident_review_record_from_row(dict(row))
    return reviews


def _escalation_notification_deadletter_item(row: dict[str, Any]) -> dict[str, Any]:
    record = _escalation_notification_record_from_row(row, include_payload_json=True)
    payload_json = record.pop("payload_json")
    record["payload"] = json.loads(payload_json)
    record["deadletter_reason"] = row.get("delivery_exhausted_reason") or "delivery_exhausted"
    record["deadlettered_at"] = row.get("delivery_exhausted_at")
    record["delivery_action"] = "review_deadletter"
    record["priority"] = "critical"
    return record


def _escalation_notification_delivery_attempt_record_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt_id": row["attempt_id"],
        "notification_id": row["notification_id"],
        "owner_id": row["owner_id"],
        "channel": row["channel"],
        "recipient": row["recipient"],
        "status": row["status"],
        "claim_token": row["claim_token"],
        "claimed_by": row["claimed_by"],
        "attempt_number": row["attempt_number"],
        "delivery_notes": row["delivery_notes"],
        "delivered_at": row["delivered_at"],
        "retry_after": row.get("retry_after"),
        "exhausted_at": row.get("exhausted_at"),
        "exhausted_reason": row.get("exhausted_reason"),
        "claimed_at": row["claimed_at"],
        "completed_at": row["completed_at"],
    }


def _escalation_notification_remediation_record_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "remediation_id": row["remediation_id"],
        "notification_id": row["notification_id"],
        "owner_id": row["owner_id"],
        "channel": row["channel"],
        "recipient": row["recipient"],
        "remediation_type": row["remediation_type"],
        "remediation_notes": row.get("remediation_notes"),
        "requeued_by": row.get("requeued_by"),
        "retry_after": row.get("retry_after"),
        "previous_delivery_exhausted_at": row.get("previous_delivery_exhausted_at"),
        "previous_delivery_exhausted_reason": row.get("previous_delivery_exhausted_reason"),
        "previous_delivery_attempt_count": row.get("previous_delivery_attempt_count") or 0,
        "created_at": row["created_at"],
    }


def _escalation_notification_claim_release_record_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "release_id": row["release_id"],
        "notification_id": row["notification_id"],
        "owner_id": row["owner_id"],
        "channel": row["channel"],
        "recipient": row["recipient"],
        "status": row["status"],
        "claim_token": row["claim_token"],
        "claimed_by": row.get("claimed_by"),
        "claimed_at": row.get("claimed_at"),
        "claimed_until": row.get("claimed_until"),
        "released_by": row.get("released_by"),
        "release_notes": row.get("release_notes"),
        "previous_delivery_attempt_count": row.get("previous_delivery_attempt_count") or 0,
        "created_at": row["created_at"],
    }


def _escalation_notification_incident_review_record_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "incident_review_id": row["incident_review_id"],
        "notification_id": row["notification_id"],
        "owner_id": row["owner_id"],
        "channel": row["channel"],
        "incident_type": row["incident_type"],
        "incident_status": row["incident_status"],
        "reviewer": row.get("reviewer"),
        "assigned_to": row.get("assigned_to"),
        "notes": row.get("notes"),
        "follow_up_at": row.get("follow_up_at"),
        "created_at": row["created_at"],
    }


def _clean_escalation_filter_values(value: str | None, allowed: set[str], field_name: str) -> list[str]:
    cleaned = [part.strip().lower() for part in (value or "").split(",") if part.strip()]
    invalid = [part for part in cleaned if part not in allowed]
    if invalid:
        raise ValueError(f"{field_name} must be one of: {', '.join(sorted(allowed))}")
    return cleaned


def _matches_escalation_inbox_filters(
    item: dict[str, Any],
    severities: list[str],
    inbox_statuses: list[str],
    assigned_to: str | None,
    assigned_to_filter_present: bool,
    focus: str,
) -> bool:
    if severities and item.get("severity") not in severities:
        return False
    if inbox_statuses and item.get("inbox_status") not in inbox_statuses:
        return False
    if assigned_to_filter_present:
        current_assignee = _clean_assigned_to(item.get("assigned_to"))
        if current_assignee != assigned_to:
            return False
    if focus:
        current_focus = item.get("focus") or ""
        if focus == "general":
            return current_focus == ""
        if current_focus != focus:
            return False
    return True


def _escalation_counts(escalations: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for item in escalations:
        value = str(item.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return [{"severity": value, "count": counts[value]} for value in sorted(counts, key=_severity_sort_key)]


def _escalation_reason_counts(escalations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for item in escalations:
        for reason in item.get("escalation_reasons", []):
            counts[reason] = counts.get(reason, 0) + 1
    return [{"reason": reason, "count": counts[reason]} for reason in sorted(counts)]


def _severity_sort_key(severity: str) -> tuple[int, str]:
    return {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(severity, 4), severity


def _owner_escalation_summaries(escalations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    owners: dict[str, dict[str, Any]] = {}
    for item in escalations:
        owner_id = item["owner_id"]
        owner = owners.setdefault(
            owner_id,
            {
                "owner_id": owner_id,
                "escalated_task_count": 0,
                "critical_task_count": 0,
                "high_task_count": 0,
                "medium_task_count": 0,
                "low_task_count": 0,
                "overdue_task_count": 0,
                "blocked_task_count": 0,
                "due_today_task_count": 0,
                "high_urgency_task_count": 0,
                "unassigned_task_count": 0,
                "unreviewed_task_count": 0,
                "acknowledged_task_count": 0,
                "snoozed_task_count": 0,
                "resolved_review_task_count": 0,
                "needs_followup_task_count": 0,
                "new_inbox_task_count": 0,
                "active_review_task_count": 0,
                "snooze_expired_task_count": 0,
                "oldest_due_at": None,
                "latest_task_updated_at": None,
                "recommended_action": None,
                "top_tasks": [],
            },
        )
        owner["escalated_task_count"] += 1
        owner[f"{item['severity']}_task_count"] += 1
        for reason in item["escalation_reasons"]:
            field = f"{reason}_task_count"
            owner[field] = int(owner.get(field) or 0) + 1
        owner[_review_status_count_field(item.get("review_status"))] += 1
        if item.get("inbox_status"):
            owner[_inbox_status_count_field(item.get("inbox_status"))] += 1
        if item.get("due_at") and (owner["oldest_due_at"] is None or item["due_at"] < owner["oldest_due_at"]):
            owner["oldest_due_at"] = item["due_at"]
        if item.get("updated_at") and (
            owner["latest_task_updated_at"] is None or item["updated_at"] > owner["latest_task_updated_at"]
        ):
            owner["latest_task_updated_at"] = item["updated_at"]
        if owner["recommended_action"] is None:
            owner["recommended_action"] = item["recommended_action"]
        if len(owner["top_tasks"]) < 3:
            owner["top_tasks"].append(_escalation_task_pointer(item))
    return sorted(
        owners.values(),
        key=lambda item: (
            -int(item["critical_task_count"]),
            -int(item["high_task_count"]),
            -int(item["escalated_task_count"]),
            item["oldest_due_at"] is None,
            item["oldest_due_at"] or "",
            item["owner_id"],
        ),
    )


def _escalation_summary_totals(
    escalations: list[dict[str, Any]], owner_summaries: list[dict[str, Any]]
) -> dict[str, int]:
    return {
        "owner_count": len(owner_summaries),
        "escalated_task_count": len(escalations),
        "critical_task_count": sum(1 for item in escalations if item["severity"] == "critical"),
        "high_task_count": sum(1 for item in escalations if item["severity"] == "high"),
        "medium_task_count": sum(1 for item in escalations if item["severity"] == "medium"),
        "low_task_count": sum(1 for item in escalations if item["severity"] == "low"),
        "overdue_task_count": sum(1 for item in escalations if "overdue" in item["escalation_reasons"]),
        "blocked_task_count": sum(1 for item in escalations if "blocked" in item["escalation_reasons"]),
        "due_today_task_count": sum(1 for item in escalations if "due_today" in item["escalation_reasons"]),
        "high_urgency_task_count": sum(1 for item in escalations if "high_urgency" in item["escalation_reasons"]),
        "unassigned_task_count": sum(1 for item in escalations if "unassigned" in item["escalation_reasons"]),
        "unreviewed_task_count": sum(1 for item in escalations if item.get("review_status") == "unreviewed"),
        "acknowledged_task_count": sum(1 for item in escalations if item.get("review_status") == "acknowledged"),
        "snoozed_task_count": sum(1 for item in escalations if item.get("review_status") == "snoozed"),
        "resolved_review_task_count": sum(1 for item in escalations if item.get("review_status") == "resolved"),
        "needs_followup_task_count": sum(1 for item in escalations if item.get("review_status") == "needs_followup"),
    }


def _escalation_task_pointer(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "owner_id": item["owner_id"],
        "queue_id": item["queue_id"],
        "task_id": item["task_id"],
        "title": item["title"],
        "status": item["status"],
        "urgency": item["urgency"],
        "assigned_to": item["assigned_to"],
        "due_at": item["due_at"],
        "severity": item["severity"],
        "escalation_reasons": item["escalation_reasons"],
        "days_overdue": item["days_overdue"],
        "review_status": item["review_status"],
        "inbox_status": item.get("inbox_status"),
        "reviewed_at": item["reviewed_at"],
        "reviewer": item["reviewer"],
        "snoozed_until": item["snoozed_until"],
        "task_url": item["task_url"],
    }


def _review_status_count_field(review_status: str | None) -> str:
    return {
        "acknowledged": "acknowledged_task_count",
        "snoozed": "snoozed_task_count",
        "resolved": "resolved_review_task_count",
        "needs_followup": "needs_followup_task_count",
    }.get(review_status or "unreviewed", "unreviewed_task_count")


def _inbox_status_count_field(inbox_status: str | None) -> str:
    return {
        "active_review": "active_review_task_count",
        "needs_followup": "needs_followup_task_count",
        "snooze_expired": "snooze_expired_task_count",
    }.get(inbox_status or "new", "new_inbox_task_count")


def _latest_escalation_review_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    review_id = row.get("escalation_review_id")
    if review_id is None:
        return None
    return {
        "review_id": review_id,
        "owner_id": row["owner_id"],
        "queue_id": row["queue_id"],
        "task_id": row["task_id"],
        "review_status": row.get("escalation_review_status"),
        "reviewer": row.get("escalation_review_reviewer"),
        "notes": row.get("escalation_review_notes"),
        "snoozed_until": row.get("escalation_review_snoozed_until"),
        "created_at": row.get("escalation_review_created_at"),
    }


def _record_task_update(
    conn: sqlite3.Connection,
    owner_id: str,
    queue_id: int,
    task_id: str,
    previous_status: str | None,
    new_status: str,
    previous_notes: str | None,
    new_notes: str | None,
    previous_assigned_to: str | None,
    new_assigned_to: str | None,
    previous_due_at: str | None,
    new_due_at: str | None,
    updated_by: str | None,
    update_source: str,
    created_at: str,
) -> None:
    _ensure_action_queue_activity_table(conn)
    conn.execute(
        """
        INSERT INTO advisor_action_queue_task_updates (
            owner_id, queue_id, task_id, previous_status, new_status,
            previous_notes, new_notes, previous_assigned_to, new_assigned_to,
            previous_due_at, new_due_at, updated_by, update_source, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            owner_id,
            queue_id,
            task_id,
            previous_status,
            new_status,
            previous_notes,
            new_notes,
            previous_assigned_to,
            new_assigned_to,
            previous_due_at,
            new_due_at,
            (updated_by or "").strip() or None,
            (update_source or "api").strip() or "api",
            created_at,
        ),
    )


def _task_update_from_row(row: dict[str, Any]) -> dict[str, Any]:
    queue_id = row["queue_id"]
    task_id = row["task_id"]
    owner_id = row["owner_id"]
    return {
        "update_id": row["update_id"],
        "owner_id": owner_id,
        "queue_id": queue_id,
        "task_id": task_id,
        "queue_title": row["queue_title"],
        "focus": row["focus"],
        "task_title": row["task_title"],
        "current_status": row["current_status"],
        "current_assigned_to": row["current_assigned_to"],
        "current_due_at": row["current_due_at"],
        "previous_status": row["previous_status"],
        "new_status": row["new_status"],
        "previous_notes": row["previous_notes"],
        "new_notes": row["new_notes"],
        "previous_assigned_to": row["previous_assigned_to"],
        "new_assigned_to": row["new_assigned_to"],
        "previous_due_at": row["previous_due_at"],
        "new_due_at": row["new_due_at"],
        "updated_by": row["updated_by"],
        "update_source": row["update_source"],
        "created_at": row["created_at"],
        "task_url": f"/agents/action-queues/{queue_id}/tasks/{task_id}?owner_id={owner_id}",
    }


def _escalation_review_by_id(conn: sqlite3.Connection, review_id: int) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT
            reviews.*,
            queues.title AS queue_title,
            queues.focus,
            tasks.title AS task_title,
            tasks.status AS current_status,
            tasks.assigned_to AS current_assigned_to,
            tasks.due_at AS current_due_at
        FROM advisor_action_queue_escalation_reviews AS reviews
        JOIN advisor_action_queues AS queues ON queues.queue_id = reviews.queue_id
        LEFT JOIN advisor_action_queue_tasks AS tasks
            ON tasks.queue_id = reviews.queue_id AND tasks.task_id = reviews.task_id
        WHERE reviews.review_id = ?
        """,
        (review_id,),
    ).fetchone()
    if not row:
        raise MarketRecordNotFound(f"No advisor action queue escalation review found for id {review_id}")
    return _escalation_review_from_row(dict(row))


def _escalation_review_from_row(row: dict[str, Any]) -> dict[str, Any]:
    queue_id = row["queue_id"]
    task_id = row["task_id"]
    owner_id = row["owner_id"]
    return {
        "review_id": row["review_id"],
        "owner_id": owner_id,
        "queue_id": queue_id,
        "task_id": task_id,
        "queue_title": row["queue_title"],
        "focus": row["focus"],
        "task_title": row["task_title"],
        "current_status": row["current_status"],
        "current_assigned_to": row["current_assigned_to"],
        "current_due_at": row["current_due_at"],
        "review_status": row["review_status"],
        "reviewer": row["reviewer"],
        "notes": row["notes"],
        "snoozed_until": row["snoozed_until"],
        "created_at": row["created_at"],
        "task_url": f"/agents/action-queues/{queue_id}/tasks/{task_id}?owner_id={owner_id}",
    }


def _queue_task_summary(row: dict[str, Any]) -> dict[str, Any]:
    queue_id = row["queue_id"]
    task_id = row["task_id"]
    owner_id = row["owner_id"]
    return {
        "queue_id": queue_id,
        "owner_id": owner_id,
        "queue_title": row["queue_title"],
        "focus": row["focus"],
        "queue_status": row["queue_status"],
        "queue_updated_at": row["queue_updated_at"],
        "saved_task_id": row["saved_task_id"],
        "task_id": task_id,
        "title": row["task_title"],
        "urgency": row["urgency"],
        "status": row["task_status"],
        "rationale": row["rationale"],
        "completion_criteria": row["completion_criteria"],
        "notes": row["notes"],
        "assigned_to": row["assigned_to"],
        "due_at": row["due_at"],
        "created_at": row["task_created_at"],
        "updated_at": row["task_updated_at"],
        "completed_at": row["completed_at"],
        "task_url": f"/agents/action-queues/{queue_id}/tasks/{task_id}?owner_id={owner_id}",
    }


def _refresh_queue_rollup(conn: sqlite3.Connection, queue_id: int, owner_id: str) -> None:
    tasks = _queue_tasks(conn, queue_id)
    counts = _counts(tasks)
    now = now_utc()
    conn.execute(
        """
        UPDATE advisor_action_queues
        SET status = ?, task_count = ?, open_task_count = ?, blocked_task_count = ?,
            completed_task_count = ?, queue_markdown = ?, updated_at = ?
        WHERE queue_id = ? AND owner_id = ?
        """,
        (
            _queue_status(counts),
            counts["task_count"],
            counts["open_task_count"],
            counts["blocked_task_count"],
            counts["completed_task_count"],
            _markdown(owner_id, tasks),
            now,
            queue_id,
            owner_id,
        ),
    )


def _counts(tasks: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "task_count": len(tasks),
        "open_task_count": sum(1 for task in tasks if task["status"] == "open"),
        "blocked_task_count": sum(1 for task in tasks if task["status"] == "blocked"),
        "completed_task_count": sum(1 for task in tasks if task["status"] == "completed"),
    }


def _queue_status(counts: dict[str, int]) -> str:
    if counts["task_count"] and counts["completed_task_count"] == counts["task_count"]:
        return "completed"
    if counts["open_task_count"] == 0 and counts["blocked_task_count"] > 0:
        return "blocked"
    return "open"


def _slug(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value)
    parts = [part for part in cleaned.split("-") if part]
    return "-".join(parts[:8]) or "task"


def _markdown(owner_id: str, tasks: list[dict[str, Any]]) -> str:
    lines = [f"# Advisor Action Queue: {owner_id}", ""]
    for task in tasks:
        lines.append(f"- [{task['status']}] {task['title']} ({task['urgency']})")
    return "\n".join(lines)
