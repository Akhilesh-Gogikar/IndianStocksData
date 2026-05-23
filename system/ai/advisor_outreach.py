"""Customer outreach drafts for saved advisor action queue tasks."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

from system.ai.action_queue import clean_owner_id, get_action_queue
from system.ai.advisor_workbench import build_advisor_workbench
from system.api.market_service import MarketDataUnavailable, MarketRecordNotFound, table_exists


VALID_DRAFT_STATUSES = {"draft", "reviewed", "approved", "rejected"}
VALID_DELIVERY_STATUSES = {"ready", "delivered", "void"}
VALID_OUTCOME_TYPES = {
    "interested",
    "meeting_scheduled",
    "needs_more_information",
    "not_interested",
    "no_response",
    "resolved",
    "other",
}
DELIVERY_STALE_AFTER_DAYS = 7
RISKY_PHRASES = {
    "guaranteed": "Avoid language that implies guaranteed outcomes.",
    "guarantee": "Avoid language that implies guaranteed outcomes.",
    "risk-free": "Avoid language that implies no risk.",
    "no risk": "Avoid language that implies no risk.",
    "sure thing": "Avoid language that implies certainty.",
    "will outperform": "Avoid language that predicts certain outperformance.",
    "price target": "Avoid unsupported price-target language.",
    "buy now": "Avoid direct trading instructions in outreach drafts.",
    "sell now": "Avoid direct trading instructions in outreach drafts.",
}


def build_advisor_outreach_draft(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    queue_id: int | None = None,
    task_id: str | None = None,
    include_blocked: bool = True,
    save: bool = False,
) -> dict[str, Any]:
    owner = clean_owner_id(owner_id)
    selection = "explicit_task" if queue_id else "workbench_top_recommendation"
    if queue_id is None:
        recommendation = _top_recommendation(conn, owner, include_blocked)
        queue_id = recommendation["queue_id"]
        task_id = recommendation["task_id"]

    queue = get_action_queue(conn, queue_id, owner)
    task = _select_task(queue, task_id)
    followup = queue.get("source_followup", {})
    email = _customer_email(owner, queue, task, followup)
    agenda = _meeting_agenda(task, followup)
    guardrails = _guardrails(task, followup)
    draft = {
        "kind": "advisor_outreach_draft",
        "owner_id": owner,
        "queue_id": queue["queue_id"],
        "task_id": task["task_id"],
        "selection": selection,
        "source_task": task,
        "source_queue": {
            "queue_id": queue["queue_id"],
            "title": queue["title"],
            "focus": queue["focus"],
            "status": queue["status"],
        },
        "customer_email": email,
        "meeting_agenda": agenda,
        "compliance_guardrails": guardrails,
        "approval_required": True,
        "draft_markdown": _markdown(owner, task, email, agenda, guardrails),
    }
    if save:
        saved = save_outreach_draft(conn, draft)
        draft["saved_draft_id"] = saved["draft_id"]
        draft["saved_status"] = saved["status"]
    return draft


def save_outreach_draft(conn: sqlite3.Connection, draft: dict[str, Any]) -> dict[str, Any]:
    require_outreach_draft_table(conn)
    owner = clean_owner_id(draft.get("owner_id"))
    now = now_utc()
    cursor = conn.execute(
        """
        INSERT INTO advisor_outreach_drafts (
            owner_id, queue_id, task_id, status, selection, subject, body,
            meeting_agenda_json, compliance_guardrails_json, source_task_json,
            source_queue_json, draft_markdown, created_at, updated_at
        )
        VALUES (?, ?, ?, 'draft', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            owner,
            draft["queue_id"],
            draft["task_id"],
            draft["selection"],
            draft["customer_email"]["subject"],
            draft["customer_email"]["body"],
            json.dumps(draft["meeting_agenda"], sort_keys=True),
            json.dumps(draft["compliance_guardrails"], sort_keys=True),
            json.dumps(draft["source_task"], sort_keys=True),
            json.dumps(draft["source_queue"], sort_keys=True),
            draft["draft_markdown"],
            now,
            now,
        ),
    )
    conn.commit()
    return get_outreach_draft(conn, int(cursor.lastrowid), owner)


def list_outreach_drafts(
    conn: sqlite3.Connection,
    owner_id: str | None,
    status: str | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    require_outreach_draft_table(conn)
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
        FROM advisor_outreach_drafts
        WHERE owner_id = ?
        {status_clause}
        ORDER BY updated_at DESC, draft_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return {
        "data": [_draft_summary(dict(row)) for row in rows],
        "metadata": {"owner_id": owner, "status": status, "result_count": len(rows)},
    }


def get_outreach_draft(conn: sqlite3.Connection, draft_id: int, owner_id: str | None) -> dict[str, Any]:
    require_outreach_draft_table(conn)
    owner = clean_owner_id(owner_id)
    row = conn.execute(
        """
        SELECT *
        FROM advisor_outreach_drafts
        WHERE draft_id = ? AND owner_id = ?
        """,
        (draft_id, owner),
    ).fetchone()
    if not row:
        raise MarketRecordNotFound(f"No advisor outreach draft found for id {draft_id}")
    return _draft_from_row(dict(row))


def update_outreach_draft_review(
    conn: sqlite3.Connection,
    draft_id: int,
    owner_id: str | None,
    status: str,
    review_notes: str | None = None,
    reviewer: str | None = None,
    override_compliance: bool = False,
) -> dict[str, Any]:
    require_outreach_draft_table(conn)
    owner = clean_owner_id(owner_id)
    normalized = status.strip().lower()
    if normalized not in VALID_DRAFT_STATUSES:
        raise ValueError(f"Draft status must be one of: {', '.join(sorted(VALID_DRAFT_STATUSES))}")
    if normalized == "approved" and not override_compliance:
        compliance = build_outreach_compliance_review(conn, draft_id, owner, save=True)
        if not compliance["can_approve"]:
            raise ValueError(
                "Compliance review blocks approval: "
                f"risk_level={compliance['risk_level']}; issue_count={compliance['issue_count']}"
            )
    now = now_utc()
    cursor = conn.execute(
        """
        UPDATE advisor_outreach_drafts
        SET status = ?, review_notes = COALESCE(?, review_notes),
            reviewer = COALESCE(?, reviewer), reviewed_at = ?, updated_at = ?
        WHERE draft_id = ? AND owner_id = ?
        """,
        (normalized, review_notes, reviewer, now, now, draft_id, owner),
    )
    if cursor.rowcount == 0:
        raise MarketRecordNotFound(f"No advisor outreach draft found for id {draft_id}")
    conn.commit()
    return get_outreach_draft(conn, draft_id, owner)


def build_outreach_compliance_review(
    conn: sqlite3.Connection,
    draft_id: int,
    owner_id: str | None,
    save: bool = False,
) -> dict[str, Any]:
    draft = get_outreach_draft(conn, draft_id, owner_id)
    email = draft["customer_email"]
    body = f"{email.get('subject', '')}\n{email.get('body', '')}"
    lowered = body.lower()
    issues: list[dict[str, Any]] = []
    passed_checks: list[str] = []

    if not email.get("subject") or not email.get("body"):
        issues.append(_issue("critical", "missing_email_copy", "Draft must include both subject and body."))
    else:
        passed_checks.append("Email subject and body are present.")

    for phrase, message in RISKY_PHRASES.items():
        if phrase in lowered:
            issues.append(_issue("critical", "risky_phrase", message, phrase=phrase))

    guardrails = draft.get("compliance_guardrails", {})
    if guardrails.get("requires_disclosure"):
        if _has_disclosure_language(lowered):
            passed_checks.append("Required disclosure language is present.")
        else:
            issues.append(
                _issue(
                    "high",
                    "missing_disclosure",
                    "Draft requires disclosure language before approval.",
                )
            )
    else:
        passed_checks.append("No required disclosure was flagged by the source draft.")

    source_task = draft.get("source_task", {})
    if source_task.get("status") == "blocked":
        issues.append(
            _issue(
                "high",
                "blocked_source_task",
                "The source task is blocked and should not be approved as customer-ready.",
            )
        )
    else:
        passed_checks.append("Source task is not blocked.")

    checklist = guardrails.get("review_checklist", [])
    if checklist:
        passed_checks.append("Advisor review checklist is attached.")
    else:
        issues.append(_issue("medium", "missing_review_checklist", "Draft should include an advisor review checklist."))

    risk_level = _risk_level(issues)
    can_approve = risk_level in {"low", "medium"}
    recommendation = "ready_for_advisor_review" if can_approve else "revise_before_approval"
    review = {
        "kind": "advisor_outreach_compliance_review",
        "draft_id": draft["draft_id"],
        "owner_id": draft["owner_id"],
        "queue_id": draft["queue_id"],
        "task_id": draft["task_id"],
        "draft_status": draft["status"],
        "risk_level": risk_level,
        "can_approve": can_approve,
        "approval_recommendation": recommendation,
        "issue_count": len(issues),
        "issues": issues,
        "passed_checks": passed_checks,
        "source_draft": {
            "subject": email.get("subject"),
            "status": draft["status"],
            "reviewer": draft.get("reviewer"),
            "reviewed_at": draft.get("reviewed_at"),
        },
        "review_markdown": _compliance_markdown(draft, risk_level, recommendation, issues, passed_checks),
    }
    if save:
        saved = save_outreach_compliance_review(conn, review)
        review["saved_review_id"] = saved["review_id"]
    return review


def save_outreach_compliance_review(conn: sqlite3.Connection, review: dict[str, Any]) -> dict[str, Any]:
    require_outreach_compliance_review_table(conn)
    now = now_utc()
    cursor = conn.execute(
        """
        INSERT INTO advisor_outreach_compliance_reviews (
            draft_id, owner_id, queue_id, task_id, draft_status, risk_level,
            can_approve, approval_recommendation, issue_count, issues_json,
            passed_checks_json, source_draft_json, review_markdown, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            review["draft_id"],
            review["owner_id"],
            review["queue_id"],
            review["task_id"],
            review["draft_status"],
            review["risk_level"],
            1 if review["can_approve"] else 0,
            review["approval_recommendation"],
            review["issue_count"],
            json.dumps(review["issues"], sort_keys=True),
            json.dumps(review["passed_checks"], sort_keys=True),
            json.dumps(review["source_draft"], sort_keys=True),
            review["review_markdown"],
            now,
        ),
    )
    conn.commit()
    return get_outreach_compliance_review(conn, int(cursor.lastrowid), review["owner_id"])


def list_outreach_compliance_reviews(
    conn: sqlite3.Connection,
    draft_id: int,
    owner_id: str | None,
    limit: int = 25,
) -> dict[str, Any]:
    require_outreach_compliance_review_table(conn)
    owner = clean_owner_id(owner_id)
    rows = conn.execute(
        """
        SELECT *
        FROM advisor_outreach_compliance_reviews
        WHERE draft_id = ? AND owner_id = ?
        ORDER BY created_at DESC, review_id DESC
        LIMIT ?
        """,
        (draft_id, owner, limit),
    ).fetchall()
    return {
        "data": [_compliance_review_summary(dict(row)) for row in rows],
        "metadata": {"owner_id": owner, "draft_id": draft_id, "result_count": len(rows)},
    }


def get_outreach_compliance_review(
    conn: sqlite3.Connection,
    review_id: int,
    owner_id: str | None,
) -> dict[str, Any]:
    require_outreach_compliance_review_table(conn)
    owner = clean_owner_id(owner_id)
    row = conn.execute(
        """
        SELECT *
        FROM advisor_outreach_compliance_reviews
        WHERE review_id = ? AND owner_id = ?
        """,
        (review_id, owner),
    ).fetchone()
    if not row:
        raise MarketRecordNotFound(f"No advisor outreach compliance review found for id {review_id}")
    return _compliance_review_from_row(dict(row))


def build_outreach_delivery_packet(
    conn: sqlite3.Connection,
    draft_id: int,
    owner_id: str | None,
    save: bool = False,
) -> dict[str, Any]:
    draft = get_outreach_draft(conn, draft_id, owner_id)
    if draft["status"] != "approved":
        raise ValueError("Outreach delivery packet requires an approved draft")
    compliance = build_outreach_compliance_review(conn, draft_id, draft["owner_id"], save=True)
    if not compliance["can_approve"]:
        raise ValueError(
            "Outreach delivery packet blocked by compliance review: "
            f"risk_level={compliance['risk_level']}; issue_count={compliance['issue_count']}"
        )
    packet = {
        "kind": "advisor_outreach_delivery_packet",
        "draft_id": draft["draft_id"],
        "owner_id": draft["owner_id"],
        "queue_id": draft["queue_id"],
        "task_id": draft["task_id"],
        "delivery_status": "ready",
        "customer_email": draft["customer_email"],
        "meeting_agenda": draft["meeting_agenda"],
        "compliance_review": {
            "review_id": compliance.get("saved_review_id"),
            "risk_level": compliance["risk_level"],
            "can_approve": compliance["can_approve"],
            "issue_count": compliance["issue_count"],
            "approval_recommendation": compliance["approval_recommendation"],
        },
        "approval_evidence": {
            "status": draft["status"],
            "reviewer": draft["reviewer"],
            "review_notes": draft["review_notes"],
            "reviewed_at": draft["reviewed_at"],
        },
        "source_task": draft["source_task"],
    }
    packet["packet_markdown"] = _delivery_packet_markdown(packet)
    if save:
        saved = save_outreach_delivery_record(conn, packet)
        packet["saved_delivery_id"] = saved["delivery_id"]
        packet["saved_status"] = saved["status"]
    return packet


def save_outreach_delivery_record(conn: sqlite3.Connection, packet: dict[str, Any]) -> dict[str, Any]:
    require_outreach_delivery_record_table(conn)
    now = now_utc()
    cursor = conn.execute(
        """
        INSERT INTO advisor_outreach_delivery_records (
            draft_id, owner_id, queue_id, task_id, status, customer_email_json,
            meeting_agenda_json, compliance_review_json, approval_evidence_json,
            source_task_json, packet_markdown, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, 'ready', ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            packet["draft_id"],
            packet["owner_id"],
            packet["queue_id"],
            packet["task_id"],
            json.dumps(packet["customer_email"], sort_keys=True),
            json.dumps(packet["meeting_agenda"], sort_keys=True),
            json.dumps(packet["compliance_review"], sort_keys=True),
            json.dumps(packet["approval_evidence"], sort_keys=True),
            json.dumps(packet["source_task"], sort_keys=True),
            packet["packet_markdown"],
            now,
            now,
        ),
    )
    conn.commit()
    return get_outreach_delivery_record(conn, int(cursor.lastrowid), packet["owner_id"])


def list_outreach_delivery_records(
    conn: sqlite3.Connection,
    owner_id: str | None,
    status: str | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    require_outreach_delivery_record_table(conn)
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
        FROM advisor_outreach_delivery_records
        WHERE owner_id = ?
        {status_clause}
        ORDER BY updated_at DESC, delivery_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return {
        "data": [_delivery_record_summary(dict(row)) for row in rows],
        "metadata": {"owner_id": owner, "status": status, "result_count": len(rows)},
    }


def get_outreach_delivery_record(conn: sqlite3.Connection, delivery_id: int, owner_id: str | None) -> dict[str, Any]:
    require_outreach_delivery_record_table(conn)
    owner = clean_owner_id(owner_id)
    row = conn.execute(
        """
        SELECT *
        FROM advisor_outreach_delivery_records
        WHERE delivery_id = ? AND owner_id = ?
        """,
        (delivery_id, owner),
    ).fetchone()
    if not row:
        raise MarketRecordNotFound(f"No advisor outreach delivery record found for id {delivery_id}")
    return _delivery_record_from_row(dict(row))


def update_outreach_delivery_record(
    conn: sqlite3.Connection,
    delivery_id: int,
    owner_id: str | None,
    status: str,
    delivery_notes: str | None = None,
    delivered_by: str | None = None,
) -> dict[str, Any]:
    require_outreach_delivery_record_table(conn)
    owner = clean_owner_id(owner_id)
    normalized = status.strip().lower()
    if normalized not in VALID_DELIVERY_STATUSES:
        raise ValueError(f"Delivery status must be one of: {', '.join(sorted(VALID_DELIVERY_STATUSES))}")
    now = now_utc()
    delivered_at = now if normalized == "delivered" else None
    cursor = conn.execute(
        """
        UPDATE advisor_outreach_delivery_records
        SET status = ?, delivery_notes = COALESCE(?, delivery_notes),
            delivered_by = COALESCE(?, delivered_by), delivered_at = ?, updated_at = ?
        WHERE delivery_id = ? AND owner_id = ?
        """,
        (normalized, delivery_notes, delivered_by, delivered_at, now, delivery_id, owner),
    )
    if cursor.rowcount == 0:
        raise MarketRecordNotFound(f"No advisor outreach delivery record found for id {delivery_id}")
    conn.commit()
    return get_outreach_delivery_record(conn, delivery_id, owner)


def build_outreach_delivery_dashboard(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    require_outreach_delivery_record_table(conn)
    owner = clean_owner_id(owner_id)
    bounded_limit = max(1, min(limit, 50))
    stale_before = (
        (datetime.now(UTC) - timedelta(days=DELIVERY_STALE_AFTER_DAYS))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    counts = dict(
        conn.execute(
            """
            SELECT
                COUNT(*) AS delivery_count,
                SUM(CASE WHEN status = 'ready' THEN 1 ELSE 0 END) AS ready_count,
                SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) AS delivered_count,
                SUM(CASE WHEN status = 'void' THEN 1 ELSE 0 END) AS void_count,
                SUM(CASE WHEN status = 'ready' AND updated_at <= ? THEN 1 ELSE 0 END) AS stale_ready_count
            FROM advisor_outreach_delivery_records
            WHERE owner_id = ?
            """,
            (stale_before, owner),
        ).fetchone()
    )
    approved_without_delivery_count = _approved_drafts_without_delivery_count(conn, owner)
    summary = {
        "owner_id": owner,
        "delivery_count": int(counts["delivery_count"] or 0),
        "ready_count": int(counts["ready_count"] or 0),
        "delivered_count": int(counts["delivered_count"] or 0),
        "void_count": int(counts["void_count"] or 0),
        "stale_ready_count": int(counts["stale_ready_count"] or 0),
        "approved_without_delivery_count": approved_without_delivery_count,
        "stale_after_days": DELIVERY_STALE_AFTER_DAYS,
    }
    ready_deliveries = _delivery_dashboard_records(conn, owner, "ready", bounded_limit)
    stale_ready_deliveries = _stale_delivery_dashboard_records(conn, owner, stale_before, bounded_limit)
    recent_deliveries = _delivery_dashboard_records(conn, owner, "delivered", bounded_limit)
    voided_deliveries = _delivery_dashboard_records(conn, owner, "void", bounded_limit)
    approved_without_delivery = _approved_drafts_without_delivery(conn, owner, bounded_limit)
    delivered_without_outcome = _delivered_without_outcome_records(conn, owner, bounded_limit)
    summary["delivered_without_outcome_count"] = _delivered_without_outcome_count(conn, owner)
    top_recommendation = _delivery_dashboard_recommendation(
        delivered_without_outcome,
        stale_ready_deliveries,
        ready_deliveries,
        approved_without_delivery,
    )
    dashboard = {
        "kind": "advisor_outreach_delivery_dashboard",
        "owner_id": owner,
        "generated_at": now_utc(),
        "summary": summary,
        "top_recommendation": top_recommendation,
        "ready_deliveries": ready_deliveries,
        "stale_ready_deliveries": stale_ready_deliveries,
        "recent_deliveries": recent_deliveries,
        "voided_deliveries": voided_deliveries,
        "approved_without_delivery": approved_without_delivery,
        "delivered_without_outcome": delivered_without_outcome,
    }
    dashboard["dashboard_markdown"] = _delivery_dashboard_markdown(dashboard)
    return dashboard


def save_outreach_delivery_outcome(
    conn: sqlite3.Connection,
    delivery_id: int,
    owner_id: str | None,
    outcome_type: str | None = None,
    response_text: str | None = None,
    follow_up_due_at: str | None = None,
    recorded_by: str | None = None,
) -> dict[str, Any]:
    require_outreach_delivery_outcome_table(conn)
    owner = clean_owner_id(owner_id)
    delivery = get_outreach_delivery_record(conn, delivery_id, owner)
    if delivery["status"] != "delivered":
        raise ValueError("Delivery outcome can only be recorded after the outreach delivery is marked delivered")
    normalized = _normalize_outcome_type(outcome_type, response_text)
    customer_signal = _outcome_customer_signal(normalized)
    next_action = _outcome_next_action(normalized, delivery, follow_up_due_at)
    source_delivery = _outcome_source_delivery(delivery)
    markdown = _outcome_markdown(delivery, normalized, customer_signal, next_action, response_text)
    now = now_utc()
    cursor = conn.execute(
        """
        INSERT INTO advisor_outreach_delivery_outcomes (
            delivery_id, draft_id, owner_id, queue_id, task_id, outcome_type,
            customer_signal, response_text, next_action_json, follow_up_due_at,
            recorded_by, source_delivery_json, outcome_markdown, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            delivery["delivery_id"],
            delivery["draft_id"],
            owner,
            delivery["queue_id"],
            delivery["task_id"],
            normalized,
            customer_signal,
            response_text.strip() if response_text else None,
            json.dumps(next_action, sort_keys=True),
            follow_up_due_at,
            recorded_by,
            json.dumps(source_delivery, sort_keys=True),
            markdown,
            now,
        ),
    )
    conn.commit()
    return get_outreach_delivery_outcome(conn, int(cursor.lastrowid), owner)


def list_outreach_delivery_outcomes(
    conn: sqlite3.Connection,
    owner_id: str | None,
    delivery_id: int | None = None,
    outcome_type: str | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    require_outreach_delivery_outcome_table(conn)
    owner = clean_owner_id(owner_id)
    bounded_limit = max(1, min(limit, 100))
    params: list[Any] = [owner]
    filters = ["owner_id = ?"]
    if delivery_id is not None:
        filters.append("delivery_id = ?")
        params.append(delivery_id)
    if outcome_type:
        normalized = outcome_type.strip().lower()
        if normalized not in VALID_OUTCOME_TYPES:
            raise ValueError(f"Outcome type must be one of: {', '.join(sorted(VALID_OUTCOME_TYPES))}")
        filters.append("outcome_type = ?")
        params.append(normalized)
    params.append(bounded_limit)
    rows = conn.execute(
        f"""
        SELECT *
        FROM advisor_outreach_delivery_outcomes
        WHERE {' AND '.join(filters)}
        ORDER BY created_at DESC, outcome_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return {
        "data": [_outcome_summary(dict(row)) for row in rows],
        "metadata": {
            "owner_id": owner,
            "delivery_id": delivery_id,
            "outcome_type": outcome_type,
            "result_count": len(rows),
        },
    }


def get_outreach_delivery_outcome(
    conn: sqlite3.Connection,
    outcome_id: int,
    owner_id: str | None,
) -> dict[str, Any]:
    require_outreach_delivery_outcome_table(conn)
    owner = clean_owner_id(owner_id)
    row = conn.execute(
        """
        SELECT *
        FROM advisor_outreach_delivery_outcomes
        WHERE outcome_id = ? AND owner_id = ?
        """,
        (outcome_id, owner),
    ).fetchone()
    if not row:
        raise MarketRecordNotFound(f"No advisor outreach delivery outcome found for id {outcome_id}")
    return _outcome_from_row(dict(row))


def build_customer_intent_dashboard(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    require_outreach_delivery_outcome_table(conn)
    owner = clean_owner_id(owner_id) if owner_id else None
    bounded_limit = max(1, min(limit, 50))
    owner_intents = _customer_intent_owner_rows(conn, owner, bounded_limit)
    pending_by_owner = _pending_outcome_counts_by_owner(conn, owner, bounded_limit)
    by_owner = {row["owner_id"]: row for row in owner_intents}
    for pending in pending_by_owner:
        existing = by_owner.get(pending["owner_id"])
        if existing:
            existing["pending_outcome_count"] = pending["pending_outcome_count"]
            existing["intent_score"] += pending["pending_outcome_count"]
            if pending["pending_outcome_count"] > 0:
                existing["segment"] = "needs_outcome_capture"
                existing["recommended_action"] = "Record missing delivery outcomes before ranking customer follow-up."
                existing["next_action_type"] = "record_outcome"
        else:
            by_owner[pending["owner_id"]] = _customer_intent_pending_only_row(pending)
    ranked = sorted(by_owner.values(), key=_customer_intent_sort_key, reverse=True)[:bounded_limit]
    summary = _customer_intent_summary(ranked)
    top_recommendation = _customer_intent_recommendation(ranked)
    dashboard = {
        "kind": "customer_intent_dashboard",
        "owner_id": owner,
        "generated_at": now_utc(),
        "summary": summary,
        "top_recommendation": top_recommendation,
        "owner_intents": ranked,
        "recent_outcomes": _customer_intent_recent_outcomes(conn, owner, bounded_limit),
    }
    dashboard["dashboard_markdown"] = _customer_intent_markdown(dashboard)
    return dashboard


def build_customer_intent_action_plan(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    dashboard = build_customer_intent_dashboard(conn, owner_id, limit=limit)
    action_items = [
        _customer_intent_action_item(conn, row, rank)
        for rank, row in enumerate(dashboard["owner_intents"], start=1)
    ]
    plan = {
        "kind": "customer_intent_action_plan",
        "owner_id": dashboard["owner_id"],
        "generated_at": now_utc(),
        "summary": _customer_intent_action_plan_summary(action_items),
        "top_action": action_items[0] if action_items else _empty_customer_intent_action(),
        "action_items": action_items,
        "source_dashboard_summary": dashboard["summary"],
    }
    plan["plan_markdown"] = _customer_intent_action_plan_markdown(plan)
    return plan


def build_customer_intent_followup_packet(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    plan = build_customer_intent_action_plan(conn, owner_id, limit=limit)
    action = plan["top_action"]
    source_outcome = _source_outcome_for_action(conn, action)
    packet = {
        "kind": "customer_intent_followup_packet",
        "owner_id": action.get("owner_id"),
        "generated_at": now_utc(),
        "packet_type": _customer_intent_packet_type(action),
        "action_item": action,
        "advisor_instructions": _customer_intent_advisor_instructions(action),
        "customer_copy": _customer_intent_customer_copy(action, source_outcome),
        "source_outcome": source_outcome,
        "compliance_review_required": _customer_intent_customer_copy_allowed(action),
        "compliance_guardrails": action.get("compliance_guardrails", []),
        "source_plan_summary": plan["summary"],
    }
    packet["packet_markdown"] = _customer_intent_followup_packet_markdown(packet)
    return packet


def build_customer_intent_followup_review(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    packet = build_customer_intent_followup_packet(conn, owner_id, limit=limit)
    issues, passed_checks = _customer_intent_followup_review_checks(packet)
    risk_level = _risk_level(issues)
    can_prepare_draft = packet["customer_copy"]["send_allowed"] and risk_level in {"low", "medium"}
    if not packet["customer_copy"]["send_allowed"]:
        recommendation = "internal_action_only"
    elif can_prepare_draft:
        recommendation = "ready_for_reviewed_outreach_draft"
    else:
        recommendation = "revise_packet_before_draft"
    review = {
        "kind": "customer_intent_followup_review",
        "owner_id": packet["owner_id"],
        "generated_at": now_utc(),
        "packet_type": packet["packet_type"],
        "risk_level": risk_level,
        "can_prepare_draft": can_prepare_draft,
        "recommendation": recommendation,
        "issue_count": len(issues),
        "issues": issues,
        "passed_checks": passed_checks,
        "source_packet": _customer_intent_review_source_packet(packet),
    }
    review["review_markdown"] = _customer_intent_followup_review_markdown(review)
    return review


def build_customer_intent_followup_draft(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    limit: int = 10,
    save: bool = False,
) -> dict[str, Any]:
    packet = build_customer_intent_followup_packet(conn, owner_id, limit=limit)
    issues, passed_checks = _customer_intent_followup_review_checks(packet)
    risk_level = _risk_level(issues)
    can_prepare_draft = packet["customer_copy"]["send_allowed"] and risk_level in {"low", "medium"}
    if not can_prepare_draft:
        raise ValueError(
            "Customer intent follow-up packet is not ready for draft handoff: "
            f"risk_level={risk_level}; issue_count={len(issues)}"
        )
    draft = _customer_intent_followup_draft(packet, risk_level, passed_checks)
    if save:
        saved = save_outreach_draft(conn, draft)
        draft["saved_draft_id"] = saved["draft_id"]
        draft["saved_status"] = saved["status"]
    return draft


def build_customer_engagement_timeline(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    require_outreach_delivery_outcome_table(conn)
    owner = clean_owner_id(owner_id)
    bounded_limit = max(1, min(limit, 100))
    events = (
        _timeline_draft_events(conn, owner, bounded_limit)
        + _timeline_compliance_review_events(conn, owner, bounded_limit)
        + _timeline_delivery_events(conn, owner, bounded_limit)
        + _timeline_outcome_events(conn, owner, bounded_limit)
    )
    events.sort(key=lambda event: (event["occurred_at"] or "", event["event_id"]), reverse=True)
    events = events[:bounded_limit]
    action_plan = build_customer_intent_action_plan(conn, owner, limit=1)
    timeline = {
        "kind": "customer_engagement_timeline",
        "owner_id": owner,
        "generated_at": now_utc(),
        "summary": _timeline_summary(events),
        "top_action": action_plan["top_action"],
        "events": events,
    }
    timeline["timeline_markdown"] = _customer_engagement_timeline_markdown(timeline)
    return timeline


def build_customer_engagement_brief(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    timeline = build_customer_engagement_timeline(conn, owner_id, limit=limit)
    top_action = timeline["top_action"]
    highlights = timeline["events"][: min(5, len(timeline["events"]))]
    brief = {
        "kind": "customer_engagement_brief",
        "owner_id": timeline["owner_id"],
        "generated_at": now_utc(),
        "summary": {
            "event_count": timeline["summary"]["event_count"],
            "latest_event_type": timeline["summary"]["latest_event_type"],
            "latest_event_at": timeline["summary"]["latest_event_at"],
            "current_segment": top_action.get("segment"),
            "priority": top_action.get("priority"),
        },
        "current_intent": _engagement_brief_current_intent(top_action),
        "next_best_action": _engagement_brief_next_action(top_action),
        "talking_points": _engagement_brief_talking_points(top_action, highlights),
        "avoid": _engagement_brief_avoid(top_action),
        "evidence_references": _engagement_brief_evidence_references(highlights, top_action),
        "source_timeline_summary": timeline["summary"],
    }
    brief["brief_markdown"] = _customer_engagement_brief_markdown(brief)
    return brief


def build_customer_engagement_cadence_review(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    brief = build_customer_engagement_brief(conn, owner_id, limit=limit)
    issues, passed_checks = _engagement_cadence_checks(brief)
    risk_level = _risk_level(issues)
    contact_allowed = risk_level not in {"critical", "high"}
    status = _engagement_cadence_status(risk_level, contact_allowed)
    review = {
        "kind": "customer_engagement_cadence_review",
        "owner_id": brief["owner_id"],
        "generated_at": now_utc(),
        "contact_allowed": contact_allowed,
        "contact_status": status,
        "recommendation": _engagement_cadence_recommendation(brief, status),
        "issue_count": len(issues),
        "issues": issues,
        "passed_checks": passed_checks,
        "next_route": _engagement_cadence_next_route(brief, contact_allowed),
        "current_intent": brief["current_intent"],
        "source_brief_summary": brief["summary"],
    }
    review["review_markdown"] = _customer_engagement_cadence_review_markdown(review)
    return review


def build_customer_engagement_cadence_dashboard(
    conn: sqlite3.Connection,
    limit: int = 25,
) -> dict[str, Any]:
    require_outreach_delivery_outcome_table(conn)
    bounded_limit = max(1, min(limit, 100))
    owners = _engagement_owner_ids(conn, bounded_limit)
    reviews = [build_customer_engagement_cadence_review(conn, owner_id, limit=10) for owner_id in owners]
    rows = [_engagement_cadence_dashboard_row(review) for review in reviews]
    rows.sort(key=_engagement_cadence_dashboard_sort_key, reverse=True)
    rows = rows[:bounded_limit]
    dashboard = {
        "kind": "customer_engagement_cadence_dashboard",
        "generated_at": now_utc(),
        "summary": _engagement_cadence_dashboard_summary(rows),
        "top_recommendation": _engagement_cadence_dashboard_recommendation(rows),
        "customers": rows,
    }
    dashboard["dashboard_markdown"] = _customer_engagement_cadence_dashboard_markdown(dashboard)
    return dashboard


def build_customer_engagement_action_queue(
    conn: sqlite3.Connection,
    limit: int = 25,
) -> dict[str, Any]:
    dashboard = build_customer_engagement_cadence_dashboard(conn, limit=limit)
    tasks = [
        _engagement_action_queue_task(row, rank)
        for rank, row in enumerate(dashboard["customers"], start=1)
    ]
    queue = {
        "kind": "customer_engagement_action_queue",
        "generated_at": now_utc(),
        "summary": _engagement_action_queue_summary(tasks),
        "top_task": tasks[0] if tasks else _empty_engagement_action_queue_task(),
        "tasks": tasks,
        "source_dashboard_summary": dashboard["summary"],
    }
    queue["queue_markdown"] = _customer_engagement_action_queue_markdown(queue)
    return queue


def build_customer_engagement_task_brief(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    selected_owner_id = owner_id
    selected_task: dict[str, Any] | None = None
    if selected_owner_id is None:
        queue = build_customer_engagement_action_queue(conn, limit=25)
        selected_task = queue.get("top_task") or {}
        selected_owner_id = selected_task.get("owner_id")
    if not selected_owner_id:
        task_brief = _empty_customer_engagement_task_brief()
        task_brief["brief_markdown"] = _customer_engagement_task_brief_markdown(task_brief)
        return task_brief

    engagement_brief = build_customer_engagement_brief(conn, selected_owner_id, limit=limit)
    cadence_review = build_customer_engagement_cadence_review(conn, selected_owner_id, limit=limit)
    if selected_task is None:
        selected_task = _engagement_action_queue_task(_engagement_cadence_dashboard_row(cadence_review), rank=1)

    task_brief = {
        "kind": "customer_engagement_task_brief",
        "owner_id": selected_owner_id,
        "generated_at": now_utc(),
        "task": selected_task,
        "customer_context": _engagement_task_brief_context(engagement_brief, cadence_review),
        "execution_plan": _engagement_task_brief_execution_plan(selected_task),
        "conversation_guide": _engagement_task_brief_conversation_guide(engagement_brief, selected_task),
        "compliance_guardrails": _engagement_task_brief_guardrails(selected_task, cadence_review),
        "completion_measurement": _engagement_task_brief_completion(selected_task),
        "source_review_summary": {
            "contact_status": cadence_review["contact_status"],
            "contact_allowed": cadence_review["contact_allowed"],
            "issue_count": cadence_review["issue_count"],
            "passed_check_count": len(cadence_review["passed_checks"]),
        },
    }
    task_brief["brief_markdown"] = _customer_engagement_task_brief_markdown(task_brief)
    return task_brief


def build_ai_recommendation_effectiveness_dashboard(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    require_outreach_delivery_outcome_table(conn)
    owner = clean_owner_id(owner_id) if owner_id else None
    bounded_limit = max(1, min(limit, 100))
    rows = _effectiveness_delivery_rows(conn, owner, max(100, bounded_limit * 10))
    by_task = _effectiveness_by_task(rows)[:bounded_limit]
    dashboard = {
        "kind": "ai_recommendation_effectiveness_dashboard",
        "owner_id": owner,
        "generated_at": now_utc(),
        "summary": _effectiveness_summary(rows),
        "top_recommendation": by_task[0] if by_task else _empty_effectiveness_recommendation(),
        "recommendation_effectiveness": by_task,
        "recent_successes": _effectiveness_recent_successes(rows, bounded_limit),
        "learning_recommendations": _effectiveness_learning_recommendations(rows, by_task),
    }
    dashboard["dashboard_markdown"] = _ai_recommendation_effectiveness_markdown(dashboard)
    return dashboard


def build_ai_improvement_backlog(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    bounded_limit = max(1, min(limit, 50))
    effectiveness = build_ai_recommendation_effectiveness_dashboard(conn, owner_id, limit=max(bounded_limit, 25))
    items = _ai_improvement_backlog_items(effectiveness)[:bounded_limit]
    backlog = {
        "kind": "ai_improvement_backlog",
        "owner_id": effectiveness["owner_id"],
        "generated_at": now_utc(),
        "summary": _ai_improvement_backlog_summary(items, effectiveness["summary"]),
        "single_next_improvement": items[0] if items else _empty_ai_improvement_item(),
        "improvements": items,
        "source_effectiveness_summary": effectiveness["summary"],
    }
    backlog["backlog_markdown"] = _ai_improvement_backlog_markdown(backlog)
    return backlog


def build_ai_improvement_experiment_plan(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    backlog = build_ai_improvement_backlog(conn, owner_id, limit=limit)
    improvement = _select_ai_improvement(backlog["improvements"], improvement_id)
    plan = {
        "kind": "ai_improvement_experiment_plan",
        "owner_id": backlog["owner_id"],
        "generated_at": now_utc(),
        "improvement": improvement,
        "hypothesis": _ai_experiment_hypothesis(improvement),
        "baseline": _ai_experiment_baseline(improvement, backlog["source_effectiveness_summary"]),
        "treatment": _ai_experiment_treatment(improvement),
        "sample_criteria": _ai_experiment_sample_criteria(improvement),
        "success_metrics": _ai_experiment_success_metrics(improvement, backlog["source_effectiveness_summary"]),
        "stop_conditions": _ai_experiment_stop_conditions(improvement),
        "measurement_route": {"method": "GET", "path": "/agents/ai-recommendation-effectiveness-dashboard"},
        "source_backlog_summary": backlog["summary"],
    }
    if backlog["owner_id"]:
        plan["measurement_route"]["path"] += f"?owner_id={backlog['owner_id']}"
    plan["experiment_markdown"] = _ai_improvement_experiment_markdown(plan)
    return plan


def build_ai_improvement_experiment_launch_packet(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    plan = build_ai_improvement_experiment_plan(conn, owner_id, improvement_id=improvement_id, limit=limit)
    packet = {
        "kind": "ai_improvement_experiment_launch_packet",
        "owner_id": plan["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": f"experiment:{plan['improvement']['improvement_id']}",
        "readiness": _ai_experiment_launch_readiness(plan),
        "cohort_assignment": _ai_experiment_launch_cohort_assignment(plan),
        "control": _ai_experiment_launch_control(plan),
        "treatment": plan["treatment"],
        "launch_checklist": _ai_experiment_launch_checklist(plan),
        "data_capture_requirements": _ai_experiment_data_capture_requirements(plan),
        "rollback_plan": _ai_experiment_rollback_plan(plan),
        "measurement_route": plan["measurement_route"],
        "source_experiment_summary": {
            "improvement_id": plan["improvement"]["improvement_id"],
            "priority": plan["improvement"]["priority"],
            "hypothesis": plan["hypothesis"],
        },
    }
    packet["launch_markdown"] = _ai_improvement_experiment_launch_markdown(packet)
    return packet


def build_ai_improvement_experiment_readout(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    launch = build_ai_improvement_experiment_launch_packet(conn, owner_id, improvement_id=improvement_id, limit=limit)
    effectiveness = build_ai_recommendation_effectiveness_dashboard(conn, owner_id, limit=max(limit, 25))
    metric_snapshot = _ai_experiment_readout_metrics(effectiveness)
    sample_status = _ai_experiment_readout_sample_status(launch, metric_snapshot)
    stop_results = _ai_experiment_readout_stop_results(launch, metric_snapshot)
    decision = _ai_experiment_readout_decision(launch, sample_status, stop_results, metric_snapshot)
    readout = {
        "kind": "ai_improvement_experiment_readout",
        "owner_id": launch["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": launch["experiment_id"],
        "decision": decision,
        "metric_snapshot": metric_snapshot,
        "sample_status": sample_status,
        "stop_condition_results": stop_results,
        "recommended_next_route": _ai_experiment_readout_next_route(launch, decision),
        "source_launch_readiness": launch["readiness"],
        "source_experiment_summary": launch["source_experiment_summary"],
    }
    readout["readout_markdown"] = _ai_improvement_experiment_readout_markdown(readout)
    return readout


def build_ai_improvement_rollout_readiness(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    readout = build_ai_improvement_experiment_readout(conn, owner_id, improvement_id=improvement_id, limit=limit)
    release_gate = _ai_rollout_release_gate(readout)
    readiness = {
        "kind": "ai_improvement_rollout_readiness",
        "owner_id": readout["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": readout["experiment_id"],
        "release_gate": release_gate,
        "customer_impact": _ai_rollout_customer_impact(readout),
        "rollout_phases": _ai_rollout_phases(readout, release_gate),
        "monitoring_plan": _ai_rollout_monitoring_plan(readout),
        "rollback_triggers": _ai_rollout_rollback_triggers(readout),
        "approval_checklist": _ai_rollout_approval_checklist(readout, release_gate),
        "recommended_next_route": _ai_rollout_next_route(readout, release_gate),
        "source_readout_decision": readout["decision"],
        "source_metric_snapshot": readout["metric_snapshot"],
    }
    readiness["readiness_markdown"] = _ai_improvement_rollout_readiness_markdown(readiness)
    return readiness


def build_ai_improvement_rollout_monitor(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    readiness = build_ai_improvement_rollout_readiness(conn, owner_id, improvement_id=improvement_id, limit=limit)
    alerts = _ai_rollout_monitor_alerts(readiness)
    monitor = {
        "kind": "ai_improvement_rollout_monitor",
        "owner_id": readiness["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": readiness["experiment_id"],
        "status": _ai_rollout_monitor_status(readiness, alerts),
        "risk_level": _ai_rollout_monitor_risk_level(readiness, alerts),
        "alerts": alerts,
        "tracked_metrics": _ai_rollout_monitor_metrics(readiness),
        "next_check": _ai_rollout_monitor_next_check(readiness),
        "immediate_action": _ai_rollout_monitor_immediate_action(readiness, alerts),
        "source_release_gate": readiness["release_gate"],
        "source_rollout_phase": readiness["rollout_phases"][0] if readiness["rollout_phases"] else None,
    }
    monitor["monitor_markdown"] = _ai_improvement_rollout_monitor_markdown(monitor)
    return monitor


def build_ai_improvement_release_packet(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    monitor = build_ai_improvement_rollout_monitor(conn, owner_id, improvement_id=improvement_id, limit=limit)
    packet = {
        "kind": "ai_improvement_release_packet",
        "owner_id": monitor["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": monitor["experiment_id"],
        "release_status": _ai_release_packet_status(monitor),
        "customer_value_summary": _ai_release_customer_value_summary(monitor),
        "eligibility": _ai_release_eligibility(monitor),
        "advisor_enablement": _ai_release_advisor_enablement(monitor),
        "support_talking_points": _ai_release_support_talking_points(monitor),
        "known_risks": _ai_release_known_risks(monitor),
        "rollback_guidance": _ai_release_rollback_guidance(monitor),
        "source_monitor_status": monitor["status"],
        "source_alert_count": len(monitor["alerts"]),
    }
    packet["release_markdown"] = _ai_improvement_release_packet_markdown(packet)
    return packet


def build_ai_improvement_adoption_playbook(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    release = build_ai_improvement_release_packet(conn, owner_id, improvement_id=improvement_id, limit=limit)
    playbook = {
        "kind": "ai_improvement_adoption_playbook",
        "owner_id": release["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": release["experiment_id"],
        "adoption_status": _ai_adoption_status(release),
        "role_tasks": _ai_adoption_role_tasks(release),
        "training_checklist": _ai_adoption_training_checklist(release),
        "customer_language": _ai_adoption_customer_language(release),
        "adoption_blockers": _ai_adoption_blockers(release),
        "success_signals": _ai_adoption_success_signals(release),
        "next_action": _ai_adoption_next_action(release),
        "source_release_status": release["release_status"],
        "source_eligibility": release["eligibility"],
    }
    playbook["playbook_markdown"] = _ai_improvement_adoption_playbook_markdown(playbook)
    return playbook


def build_ai_improvement_adoption_monitor(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    playbook = build_ai_improvement_adoption_playbook(conn, owner_id, improvement_id=improvement_id, limit=limit)
    blockers = playbook["adoption_blockers"]
    training = playbook["training_checklist"]
    monitor = {
        "kind": "ai_improvement_adoption_monitor",
        "owner_id": playbook["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": playbook["experiment_id"],
        "status": _ai_adoption_monitor_status(playbook, blockers),
        "risk_level": _ai_adoption_monitor_risk_level(blockers),
        "training_status": _ai_adoption_training_status(training),
        "blockers": blockers,
        "success_signals": playbook["success_signals"],
        "customer_language_status": _ai_adoption_customer_language_status(playbook),
        "immediate_action": playbook["next_action"],
        "source_adoption_status": playbook["adoption_status"],
    }
    monitor["monitor_markdown"] = _ai_improvement_adoption_monitor_markdown(monitor)
    return monitor


def build_ai_improvement_adoption_impact_ledger(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    monitor = build_ai_improvement_adoption_monitor(conn, owner_id, improvement_id=improvement_id, limit=limit)
    readout = build_ai_improvement_experiment_readout(conn, owner_id, improvement_id=improvement_id, limit=limit)
    metrics = readout["metric_snapshot"]
    ledger = {
        "kind": "ai_improvement_adoption_impact_ledger",
        "owner_id": monitor["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": monitor["experiment_id"],
        "value_status": _ai_adoption_impact_value_status(monitor, metrics),
        "customer_impact": _ai_adoption_impact_customer_impact(metrics),
        "advisor_usage": _ai_adoption_impact_advisor_usage(metrics),
        "scale_decision": _ai_adoption_impact_scale_decision(monitor, readout),
        "blocked_accounts": _ai_adoption_impact_blocked_accounts(monitor),
        "proof_points": _ai_adoption_impact_proof_points(metrics, monitor),
        "next_action": monitor["immediate_action"],
        "source_monitor_status": monitor["status"],
        "source_risk_level": monitor["risk_level"],
        "source_readout_decision": readout["decision"],
    }
    ledger["ledger_markdown"] = _ai_improvement_adoption_impact_ledger_markdown(ledger)
    return ledger


def build_ai_improvement_scale_decision_packet(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    ledger = build_ai_improvement_adoption_impact_ledger(conn, owner_id, improvement_id=improvement_id, limit=limit)
    decision = _ai_scale_packet_decision(ledger)
    packet = {
        "kind": "ai_improvement_scale_decision_packet",
        "owner_id": ledger["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": ledger["experiment_id"],
        "decision": decision,
        "executive_summary": _ai_scale_packet_executive_summary(ledger, decision),
        "customer_value_evidence": _ai_scale_packet_customer_value_evidence(ledger),
        "advisor_change_plan": _ai_scale_packet_advisor_change_plan(ledger, decision),
        "blocker_resolution_plan": _ai_scale_packet_blocker_resolution_plan(ledger),
        "rollout_scope": _ai_scale_packet_rollout_scope(ledger),
        "next_action": _ai_scale_packet_next_action(ledger, decision),
        "source_value_status": ledger["value_status"],
        "source_scale_decision": ledger["scale_decision"],
        "source_monitor_status": ledger["source_monitor_status"],
    }
    packet["packet_markdown"] = _ai_improvement_scale_decision_packet_markdown(packet)
    return packet


def build_ai_improvement_scale_execution_plan(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    packet = build_ai_improvement_scale_decision_packet(conn, owner_id, improvement_id=improvement_id, limit=limit)
    tasks = _ai_scale_execution_tasks(packet)
    plan = {
        "kind": "ai_improvement_scale_execution_plan",
        "owner_id": packet["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": packet["experiment_id"],
        "execution_status": _ai_scale_execution_status(packet),
        "decision": packet["decision"],
        "rollout_scope": packet["rollout_scope"],
        "execution_tasks": tasks,
        "guardrails": _ai_scale_execution_guardrails(packet),
        "customer_proof_checks": _ai_scale_execution_customer_proof_checks(packet),
        "acceptance_criteria": _ai_scale_execution_acceptance_criteria(packet),
        "escalation_path": _ai_scale_execution_escalation_path(packet),
        "next_action": _ai_scale_execution_next_action(tasks),
        "source_value_status": packet["source_value_status"],
        "source_monitor_status": packet["source_monitor_status"],
    }
    plan["plan_markdown"] = _ai_improvement_scale_execution_plan_markdown(plan)
    return plan


def build_ai_improvement_scale_execution_monitor(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    plan = build_ai_improvement_scale_execution_plan(conn, owner_id, improvement_id=improvement_id, limit=limit)
    task_progress = _ai_scale_execution_task_progress(plan["execution_tasks"])
    guardrail_status = _ai_scale_execution_guardrail_status(plan["guardrails"])
    proof_status = _ai_scale_execution_proof_status(plan["customer_proof_checks"])
    acceptance_status = _ai_scale_execution_acceptance_status(plan["acceptance_criteria"])
    blockers = _ai_scale_execution_monitor_blockers(plan, proof_status, acceptance_status)
    monitor = {
        "kind": "ai_improvement_scale_execution_monitor",
        "owner_id": plan["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": plan["experiment_id"],
        "status": _ai_scale_execution_monitor_status(plan, task_progress, proof_status, acceptance_status),
        "risk_level": _ai_scale_execution_monitor_risk_level(proof_status, acceptance_status, blockers),
        "task_progress": task_progress,
        "guardrail_status": guardrail_status,
        "customer_proof_status": proof_status,
        "acceptance_status": acceptance_status,
        "blockers": blockers,
        "immediate_action": plan["next_action"],
        "escalation_path": plan["escalation_path"],
        "source_execution_status": plan["execution_status"],
        "source_decision": plan["decision"],
    }
    monitor["monitor_markdown"] = _ai_improvement_scale_execution_monitor_markdown(monitor)
    return monitor


def build_ai_improvement_scale_learning_report(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    monitor = build_ai_improvement_scale_execution_monitor(conn, owner_id, improvement_id=improvement_id, limit=limit)
    report = {
        "kind": "ai_improvement_scale_learning_report",
        "owner_id": monitor["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": monitor["experiment_id"],
        "learning_status": _ai_scale_learning_status(monitor),
        "validated_learnings": _ai_scale_validated_learnings(monitor),
        "open_questions": _ai_scale_open_questions(monitor),
        "feedback_actions": _ai_scale_feedback_actions(monitor),
        "next_improvement_candidate": _ai_scale_next_improvement_candidate(monitor),
        "roadmap_update": _ai_scale_learning_roadmap_update(monitor),
        "source_monitor_status": monitor["status"],
        "source_risk_level": monitor["risk_level"],
        "source_decision": monitor["source_decision"],
    }
    report["report_markdown"] = _ai_improvement_scale_learning_report_markdown(report)
    return report


def build_ai_improvement_roadmap_refresh(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    report = build_ai_improvement_scale_learning_report(conn, owner_id, improvement_id=improvement_id, limit=limit)
    roadmap_item = _ai_roadmap_refresh_item(report)
    refresh = {
        "kind": "ai_improvement_roadmap_refresh",
        "owner_id": report["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": report["experiment_id"],
        "roadmap_status": _ai_roadmap_refresh_status(report),
        "roadmap_item": roadmap_item,
        "owner_action_plan": _ai_roadmap_owner_action_plan(report),
        "evidence_package": _ai_roadmap_evidence_package(report),
        "acceptance_gates": _ai_roadmap_acceptance_gates(report),
        "measurement_plan": _ai_roadmap_measurement_plan(report),
        "sequencing": _ai_roadmap_sequencing(report),
        "next_action": _ai_roadmap_next_action(report),
        "source_learning_status": report["learning_status"],
        "source_roadmap_update": report["roadmap_update"],
        "source_next_improvement_candidate": report["next_improvement_candidate"],
    }
    refresh["roadmap_markdown"] = _ai_improvement_roadmap_refresh_markdown(refresh)
    return refresh


def build_ai_improvement_backlog_handoff(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    refresh = build_ai_improvement_roadmap_refresh(conn, owner_id, improvement_id=improvement_id, limit=limit)
    handoff = {
        "kind": "ai_improvement_backlog_handoff",
        "owner_id": refresh["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": refresh["experiment_id"],
        "handoff_status": _ai_backlog_handoff_status(refresh),
        "work_item": _ai_backlog_handoff_work_item(refresh),
        "implementation_story": _ai_backlog_handoff_story(refresh),
        "implementation_scope": _ai_backlog_handoff_scope(refresh),
        "owner_actions": refresh["owner_action_plan"],
        "dependencies": _ai_backlog_handoff_dependencies(refresh),
        "acceptance_gates": refresh["acceptance_gates"],
        "measurement_plan": refresh["measurement_plan"],
        "launch_readiness": _ai_backlog_handoff_launch_readiness(refresh),
        "next_action": refresh["next_action"],
        "source_roadmap_status": refresh["roadmap_status"],
        "source_next_improvement_candidate": refresh["source_next_improvement_candidate"],
    }
    handoff["handoff_markdown"] = _ai_improvement_backlog_handoff_markdown(handoff)
    return handoff


def build_ai_improvement_implementation_kickoff_packet(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    handoff = build_ai_improvement_backlog_handoff(conn, owner_id, improvement_id=improvement_id, limit=limit)
    packet = {
        "kind": "ai_improvement_implementation_kickoff_packet",
        "owner_id": handoff["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": handoff["experiment_id"],
        "kickoff_status": _ai_implementation_kickoff_status(handoff),
        "work_item": handoff["work_item"],
        "implementation_story": handoff["implementation_story"],
        "engineering_scope": _ai_implementation_kickoff_engineering_scope(handoff),
        "qa_gates": _ai_implementation_kickoff_qa_gates(handoff),
        "data_contracts": _ai_implementation_kickoff_data_contracts(handoff),
        "customer_value_guardrails": _ai_implementation_kickoff_customer_value_guardrails(handoff),
        "launch_checklist": _ai_implementation_kickoff_launch_checklist(handoff),
        "immediate_action": handoff["next_action"],
        "source_handoff_status": handoff["handoff_status"],
        "source_launch_readiness": handoff["launch_readiness"],
    }
    packet["kickoff_markdown"] = _ai_improvement_implementation_kickoff_packet_markdown(packet)
    return packet


def build_ai_improvement_implementation_readiness_monitor(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    packet = build_ai_improvement_implementation_kickoff_packet(
        conn, owner_id, improvement_id=improvement_id, limit=limit
    )
    qa_status = _ai_implementation_readiness_qa_status(packet)
    data_contract_status = _ai_implementation_readiness_data_contract_status(packet)
    guardrail_status = _ai_implementation_readiness_guardrail_status(packet)
    checklist_status = _ai_implementation_readiness_checklist_status(packet)
    blockers = _ai_implementation_readiness_blockers(qa_status, data_contract_status, guardrail_status, checklist_status)
    monitor = {
        "kind": "ai_improvement_implementation_readiness_monitor",
        "owner_id": packet["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": packet["experiment_id"],
        "status": _ai_implementation_readiness_status(packet, blockers),
        "risk_level": _ai_implementation_readiness_risk_level(blockers),
        "work_item": packet["work_item"],
        "qa_status": qa_status,
        "data_contract_status": data_contract_status,
        "customer_guardrail_status": guardrail_status,
        "launch_checklist_status": checklist_status,
        "blockers": blockers,
        "immediate_action": packet["immediate_action"],
        "source_kickoff_status": packet["kickoff_status"],
        "source_launch_readiness": packet["source_launch_readiness"],
    }
    monitor["monitor_markdown"] = _ai_improvement_implementation_readiness_monitor_markdown(monitor)
    return monitor


def build_ai_improvement_implementation_blocker_resolution_plan(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    monitor = build_ai_improvement_implementation_readiness_monitor(
        conn, owner_id, improvement_id=improvement_id, limit=limit
    )
    resolution_tasks = _ai_implementation_blocker_resolution_tasks(monitor)
    plan = {
        "kind": "ai_improvement_implementation_blocker_resolution_plan",
        "owner_id": monitor["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": monitor["experiment_id"],
        "resolution_status": _ai_implementation_blocker_resolution_status(monitor),
        "work_item": monitor["work_item"],
        "resolution_tasks": resolution_tasks,
        "proof_required": _ai_implementation_blocker_proof_required(monitor),
        "exit_criteria": _ai_implementation_blocker_exit_criteria(monitor),
        "qa_rerun_plan": _ai_implementation_blocker_qa_rerun_plan(monitor),
        "customer_guardrail_clearance": _ai_implementation_blocker_customer_guardrail_clearance(monitor),
        "immediate_unblock_action": _ai_implementation_blocker_immediate_action(resolution_tasks, monitor),
        "source_readiness_status": monitor["status"],
        "source_risk_level": monitor["risk_level"],
    }
    plan["resolution_markdown"] = _ai_improvement_implementation_blocker_resolution_plan_markdown(plan)
    return plan


def build_ai_improvement_implementation_unblock_verification_report(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    plan = build_ai_improvement_implementation_blocker_resolution_plan(
        conn, owner_id, improvement_id=improvement_id, limit=limit
    )
    task_status = _ai_implementation_unblock_task_status(plan)
    proof_status = _ai_implementation_unblock_proof_status(plan)
    exit_status = _ai_implementation_unblock_exit_status(plan)
    qa_rerun_status = _ai_implementation_unblock_qa_rerun_status(plan)
    ready_to_proceed = _ai_implementation_unblock_ready(task_status, proof_status, exit_status, qa_rerun_status)
    report = {
        "kind": "ai_improvement_implementation_unblock_verification_report",
        "owner_id": plan["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": plan["experiment_id"],
        "verification_status": _ai_implementation_unblock_verification_status(plan, ready_to_proceed),
        "ready_to_proceed": ready_to_proceed,
        "work_item": plan["work_item"],
        "task_status": task_status,
        "proof_status": proof_status,
        "exit_criteria_status": exit_status,
        "qa_rerun_status": qa_rerun_status,
        "customer_guardrail_status": plan["customer_guardrail_clearance"],
        "next_verification_action": _ai_implementation_unblock_next_action(plan, ready_to_proceed),
        "source_resolution_status": plan["resolution_status"],
        "source_risk_level": plan["source_risk_level"],
    }
    report["verification_markdown"] = _ai_improvement_implementation_unblock_verification_report_markdown(report)
    return report


def build_ai_improvement_implementation_qa_review_packet(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    verification = build_ai_improvement_implementation_unblock_verification_report(
        conn, owner_id, improvement_id=improvement_id, limit=limit
    )
    qa_decision = _ai_implementation_qa_review_decision(verification)
    packet = {
        "kind": "ai_improvement_implementation_qa_review_packet",
        "owner_id": verification["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": verification["experiment_id"],
        "qa_decision": qa_decision,
        "work_item": verification["work_item"],
        "qa_scope": _ai_implementation_qa_review_scope(verification),
        "evidence_gaps": _ai_implementation_qa_evidence_gaps(verification),
        "test_gates": _ai_implementation_qa_test_gates(verification),
        "customer_guardrails": _ai_implementation_qa_customer_guardrails(verification),
        "signoff_requirements": _ai_implementation_qa_signoff_requirements(verification),
        "next_qa_action": _ai_implementation_qa_next_action(verification, qa_decision),
        "source_verification_status": verification["verification_status"],
        "source_ready_to_proceed": verification["ready_to_proceed"],
        "source_risk_level": verification["source_risk_level"],
    }
    packet["qa_markdown"] = _ai_improvement_implementation_qa_review_packet_markdown(packet)
    return packet


def build_ai_improvement_implementation_qa_signoff_report(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    qa_packet = build_ai_improvement_implementation_qa_review_packet(
        conn, owner_id, improvement_id=improvement_id, limit=limit
    )
    signoff_decision = _ai_implementation_qa_signoff_decision(qa_packet)
    report = {
        "kind": "ai_improvement_implementation_qa_signoff_report",
        "owner_id": qa_packet["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": qa_packet["experiment_id"],
        "signoff_status": _ai_implementation_qa_signoff_status(qa_packet),
        "signoff_decision": signoff_decision,
        "work_item": qa_packet["work_item"],
        "required_signoffs": qa_packet["signoff_requirements"],
        "signoff_gaps": _ai_implementation_qa_signoff_gaps(qa_packet),
        "launch_blockers": _ai_implementation_qa_launch_blockers(qa_packet),
        "evidence_summary": _ai_implementation_qa_signoff_evidence_summary(qa_packet),
        "next_signoff_action": _ai_implementation_qa_signoff_next_action(qa_packet, signoff_decision),
        "source_qa_decision": qa_packet["qa_decision"],
        "source_ready_to_proceed": qa_packet["source_ready_to_proceed"],
        "source_risk_level": qa_packet["source_risk_level"],
    }
    report["signoff_markdown"] = _ai_improvement_implementation_qa_signoff_report_markdown(report)
    return report


def build_ai_improvement_launch_review_packet(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    signoff = build_ai_improvement_implementation_qa_signoff_report(
        conn, owner_id, improvement_id=improvement_id, limit=limit
    )
    launch_decision = _ai_launch_review_decision(signoff)
    packet = {
        "kind": "ai_improvement_launch_review_packet",
        "owner_id": signoff["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": signoff["experiment_id"],
        "launch_decision": launch_decision,
        "work_item": signoff["work_item"],
        "launch_scope": _ai_launch_review_scope(signoff, launch_decision),
        "customer_guardrails": _ai_launch_review_customer_guardrails(signoff),
        "monitoring_requirements": _ai_launch_review_monitoring_requirements(signoff),
        "rollback_triggers": _ai_launch_review_rollback_triggers(signoff),
        "evidence_gaps": signoff["signoff_gaps"],
        "launch_blockers": signoff["launch_blockers"],
        "next_launch_action": _ai_launch_review_next_action(signoff, launch_decision),
        "source_signoff_status": signoff["signoff_status"],
        "source_signoff_decision": signoff["signoff_decision"],
        "source_risk_level": signoff["source_risk_level"],
    }
    packet["launch_markdown"] = _ai_improvement_launch_review_packet_markdown(packet)
    return packet


def build_ai_improvement_launch_execution_plan(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    launch = build_ai_improvement_launch_review_packet(conn, owner_id, improvement_id=improvement_id, limit=limit)
    tasks = _ai_launch_execution_tasks(launch)
    plan = {
        "kind": "ai_improvement_launch_execution_plan",
        "owner_id": launch["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": launch["experiment_id"],
        "execution_status": _ai_launch_execution_status(launch),
        "launch_decision": launch["launch_decision"],
        "work_item": launch["work_item"],
        "launch_scope": launch["launch_scope"],
        "execution_tasks": tasks,
        "monitoring_setup": _ai_launch_execution_monitoring_setup(launch),
        "rollback_setup": _ai_launch_execution_rollback_setup(launch),
        "customer_guardrails": launch["customer_guardrails"],
        "exit_criteria": _ai_launch_execution_exit_criteria(launch),
        "immediate_action": _ai_launch_execution_immediate_action(tasks, launch),
        "source_launch_blockers": launch["launch_blockers"],
        "source_risk_level": launch["source_risk_level"],
    }
    plan["execution_markdown"] = _ai_improvement_launch_execution_plan_markdown(plan)
    return plan


def build_ai_improvement_launch_execution_monitor(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    plan = build_ai_improvement_launch_execution_plan(conn, owner_id, improvement_id=improvement_id, limit=limit)
    task_status = _ai_launch_execution_task_status(plan)
    monitoring_status = _ai_launch_execution_monitoring_status(plan)
    rollback_status = _ai_launch_execution_rollback_status(plan)
    exit_criteria_status = _ai_launch_execution_exit_status(plan)
    blockers = _ai_launch_execution_monitor_blockers(
        task_status,
        monitoring_status,
        rollback_status,
        exit_criteria_status,
    )
    monitor = {
        "kind": "ai_improvement_launch_execution_monitor",
        "owner_id": plan["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": plan["experiment_id"],
        "status": _ai_launch_execution_monitor_status(plan, blockers),
        "risk_level": _ai_launch_execution_monitor_risk_level(blockers),
        "work_item": plan["work_item"],
        "task_status": task_status,
        "monitoring_status": monitoring_status,
        "rollback_status": rollback_status,
        "exit_criteria_status": exit_criteria_status,
        "blockers": blockers,
        "immediate_action": plan["immediate_action"],
        "source_execution_status": plan["execution_status"],
        "source_launch_decision": plan["launch_decision"],
    }
    monitor["monitor_markdown"] = _ai_improvement_launch_execution_monitor_markdown(monitor)
    return monitor


def build_ai_improvement_launch_outcome_monitor(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    execution = build_ai_improvement_launch_execution_monitor(conn, owner_id, improvement_id=improvement_id, limit=limit)
    launch_health = _ai_launch_outcome_health(execution)
    value_status = _ai_launch_outcome_value_status(execution)
    customer_signal_status = _ai_launch_outcome_customer_signal_status(execution)
    rollback_status = _ai_launch_outcome_rollback_status(execution)
    blockers = _ai_launch_outcome_blockers(execution, launch_health, value_status, rollback_status)
    monitor = {
        "kind": "ai_improvement_launch_outcome_monitor",
        "owner_id": execution["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": execution["experiment_id"],
        "status": _ai_launch_outcome_monitor_status(blockers),
        "risk_level": _ai_launch_outcome_risk_level(blockers),
        "work_item": execution["work_item"],
        "launch_health": launch_health,
        "value_status": value_status,
        "customer_signal_status": customer_signal_status,
        "rollback_status": rollback_status,
        "blockers": blockers,
        "immediate_action": _ai_launch_outcome_next_action(execution, blockers),
        "source_execution_monitor_status": execution["status"],
        "source_execution_status": execution["source_execution_status"],
        "source_launch_decision": execution["source_launch_decision"],
    }
    monitor["outcome_markdown"] = _ai_improvement_launch_outcome_monitor_markdown(monitor)
    return monitor


def build_ai_improvement_launch_value_proof_packet(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    outcome = build_ai_improvement_launch_outcome_monitor(conn, owner_id, improvement_id=improvement_id, limit=limit)
    packet = {
        "kind": "ai_improvement_launch_value_proof_packet",
        "owner_id": outcome["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": outcome["experiment_id"],
        "proof_status": _ai_launch_value_proof_status(outcome),
        "risk_level": outcome["risk_level"],
        "work_item": outcome["work_item"],
        "customer_value_claim": _ai_launch_value_customer_claim(outcome),
        "proof_points": _ai_launch_value_proof_points(outcome),
        "evidence_gaps": _ai_launch_value_evidence_gaps(outcome),
        "customer_message": _ai_launch_value_customer_message(outcome),
        "advisor_next_action": outcome["immediate_action"],
        "source_outcome_status": outcome["status"],
        "source_outcome_risk_level": outcome["risk_level"],
        "source_launch_decision": outcome["source_launch_decision"],
    }
    packet["proof_markdown"] = _ai_improvement_launch_value_proof_packet_markdown(packet)
    return packet


def build_ai_improvement_launch_customer_communication_packet(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    proof = build_ai_improvement_launch_value_proof_packet(conn, owner_id, improvement_id=improvement_id, limit=limit)
    packet = {
        "kind": "ai_improvement_launch_customer_communication_packet",
        "owner_id": proof["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": proof["experiment_id"],
        "communication_status": _ai_launch_customer_communication_status(proof),
        "risk_level": proof["risk_level"],
        "work_item": proof["work_item"],
        "audience": _ai_launch_customer_communication_audience(proof),
        "message": _ai_launch_customer_communication_message(proof),
        "review_gates": _ai_launch_customer_communication_review_gates(proof),
        "blocked_claims": _ai_launch_customer_communication_blocked_claims(proof),
        "advisor_next_action": proof["advisor_next_action"],
        "source_proof_status": proof["proof_status"],
        "source_customer_claim_status": proof["customer_value_claim"]["status"],
        "source_launch_decision": proof["source_launch_decision"],
    }
    packet["communication_markdown"] = _ai_improvement_launch_customer_communication_packet_markdown(packet)
    return packet


def build_ai_improvement_launch_customer_communication_review_packet(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    communication = build_ai_improvement_launch_customer_communication_packet(
        conn,
        owner_id,
        improvement_id=improvement_id,
        limit=limit,
    )
    send_decision = _ai_launch_customer_communication_send_decision(communication)
    packet = {
        "kind": "ai_improvement_launch_customer_communication_review_packet",
        "owner_id": communication["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": communication["experiment_id"],
        "review_status": send_decision["status"],
        "risk_level": communication["risk_level"],
        "work_item": communication["work_item"],
        "send_decision": send_decision,
        "required_approvals": _ai_launch_customer_communication_required_approvals(communication),
        "send_blockers": _ai_launch_customer_communication_send_blockers(communication),
        "escalation_path": _ai_launch_customer_communication_escalation_path(communication),
        "approved_message": _ai_launch_customer_communication_approved_message(communication, send_decision),
        "advisor_next_action": communication["advisor_next_action"],
        "source_communication_status": communication["communication_status"],
        "source_customer_claim_status": communication["source_customer_claim_status"],
        "source_launch_decision": communication["source_launch_decision"],
    }
    packet["review_markdown"] = _ai_improvement_launch_customer_communication_review_packet_markdown(packet)
    return packet


def build_ai_improvement_launch_customer_communication_delivery_packet(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    review = build_ai_improvement_launch_customer_communication_review_packet(
        conn,
        owner_id,
        improvement_id=improvement_id,
        limit=limit,
    )
    packet = {
        "kind": "ai_improvement_launch_customer_communication_delivery_packet",
        "owner_id": review["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": review["experiment_id"],
        "delivery_status": _ai_launch_customer_communication_delivery_status(review),
        "risk_level": review["risk_level"],
        "work_item": review["work_item"],
        "channel_plan": _ai_launch_customer_communication_delivery_channel_plan(review),
        "delivery_payload": _ai_launch_customer_communication_delivery_payload(review),
        "delivery_checklist": _ai_launch_customer_communication_delivery_checklist(review),
        "audit_trail": _ai_launch_customer_communication_delivery_audit_trail(review),
        "follow_up_plan": _ai_launch_customer_communication_delivery_follow_up_plan(review),
        "advisor_next_action": _ai_launch_customer_communication_delivery_next_action(review),
        "source_review_status": review["review_status"],
        "source_send_decision": review["send_decision"],
        "source_customer_claim_status": review["source_customer_claim_status"],
        "source_launch_decision": review["source_launch_decision"],
    }
    packet["delivery_markdown"] = _ai_improvement_launch_customer_communication_delivery_packet_markdown(packet)
    return packet


def build_ai_improvement_launch_customer_communication_delivery_monitor(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    delivery = build_ai_improvement_launch_customer_communication_delivery_packet(
        conn,
        owner_id,
        improvement_id=improvement_id,
        limit=limit,
    )
    delivery_progress = _ai_launch_customer_communication_delivery_progress(delivery)
    checklist_status = _ai_launch_customer_communication_delivery_checklist_status(delivery)
    audit_status = _ai_launch_customer_communication_delivery_audit_status(delivery)
    follow_up_status = _ai_launch_customer_communication_delivery_follow_up_status(delivery)
    blockers = _ai_launch_customer_communication_delivery_monitor_blockers(
        delivery,
        delivery_progress,
        checklist_status,
        follow_up_status,
    )
    monitor = {
        "kind": "ai_improvement_launch_customer_communication_delivery_monitor",
        "owner_id": delivery["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": delivery["experiment_id"],
        "status": _ai_launch_customer_communication_delivery_monitor_status(delivery, blockers),
        "risk_level": _ai_launch_customer_communication_delivery_monitor_risk_level(delivery, blockers),
        "work_item": delivery["work_item"],
        "delivery_progress": delivery_progress,
        "checklist_status": checklist_status,
        "audit_status": audit_status,
        "follow_up_status": follow_up_status,
        "blockers": blockers,
        "immediate_action": delivery["advisor_next_action"],
        "source_delivery_status": delivery["delivery_status"],
        "source_send_decision": delivery["source_send_decision"],
        "source_customer_claim_status": delivery["source_customer_claim_status"],
        "source_launch_decision": delivery["source_launch_decision"],
    }
    monitor["monitor_markdown"] = _ai_improvement_launch_customer_communication_delivery_monitor_markdown(monitor)
    return monitor


def build_ai_improvement_launch_customer_communication_delivery_unblock_plan(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    monitor = build_ai_improvement_launch_customer_communication_delivery_monitor(
        conn,
        owner_id,
        improvement_id=improvement_id,
        limit=limit,
    )
    unblock_tasks = _ai_launch_customer_communication_delivery_unblock_tasks(monitor)
    plan = {
        "kind": "ai_improvement_launch_customer_communication_delivery_unblock_plan",
        "owner_id": monitor["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": monitor["experiment_id"],
        "plan_status": _ai_launch_customer_communication_delivery_unblock_status(monitor),
        "risk_level": monitor["risk_level"],
        "work_item": monitor["work_item"],
        "unblock_tasks": unblock_tasks,
        "proof_gates": _ai_launch_customer_communication_delivery_unblock_proof_gates(monitor),
        "exit_criteria": _ai_launch_customer_communication_delivery_unblock_exit_criteria(monitor),
        "recheck_plan": _ai_launch_customer_communication_delivery_unblock_recheck_plan(monitor),
        "immediate_action": unblock_tasks[0] if unblock_tasks else monitor["immediate_action"],
        "source_monitor_status": monitor["status"],
        "source_delivery_status": monitor["source_delivery_status"],
        "source_send_decision": monitor["source_send_decision"],
        "source_customer_claim_status": monitor["source_customer_claim_status"],
        "source_launch_decision": monitor["source_launch_decision"],
    }
    plan["plan_markdown"] = _ai_improvement_launch_customer_communication_delivery_unblock_plan_markdown(plan)
    return plan


def build_ai_improvement_launch_customer_communication_delivery_unblock_verification_report(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    plan = build_ai_improvement_launch_customer_communication_delivery_unblock_plan(
        conn,
        owner_id,
        improvement_id=improvement_id,
        limit=limit,
    )
    verification_results = _ai_launch_customer_communication_delivery_unblock_verification_results(plan)
    report = {
        "kind": "ai_improvement_launch_customer_communication_delivery_unblock_verification_report",
        "owner_id": plan["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": plan["experiment_id"],
        "verification_status": _ai_launch_customer_communication_delivery_unblock_verification_status(
            verification_results
        ),
        "risk_level": plan["risk_level"],
        "work_item": plan["work_item"],
        "verification_results": verification_results,
        "failed_checks": _ai_launch_customer_communication_delivery_unblock_failed_checks(verification_results),
        "required_follow_up": _ai_launch_customer_communication_delivery_unblock_required_follow_up(
            plan,
            verification_results,
        ),
        "next_action": _ai_launch_customer_communication_delivery_unblock_verification_next_action(
            plan,
            verification_results,
        ),
        "source_plan_status": plan["plan_status"],
        "source_delivery_status": plan["source_delivery_status"],
        "source_send_decision": plan["source_send_decision"],
        "source_customer_claim_status": plan["source_customer_claim_status"],
        "source_launch_decision": plan["source_launch_decision"],
    }
    report["verification_markdown"] = (
        _ai_improvement_launch_customer_communication_delivery_unblock_verification_report_markdown(report)
    )
    return report


def build_ai_improvement_launch_customer_communication_delivery_send_authorization_packet(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    verification = build_ai_improvement_launch_customer_communication_delivery_unblock_verification_report(
        conn,
        owner_id,
        improvement_id=improvement_id,
        limit=limit,
    )
    authorization_decision = _ai_launch_customer_communication_delivery_send_authorization_decision(verification)
    packet = {
        "kind": "ai_improvement_launch_customer_communication_delivery_send_authorization_packet",
        "owner_id": verification["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": verification["experiment_id"],
        "authorization_status": authorization_decision["status"],
        "risk_level": verification["risk_level"],
        "work_item": verification["work_item"],
        "authorization_decision": authorization_decision,
        "send_requirements": _ai_launch_customer_communication_delivery_send_requirements(verification),
        "blocked_reasons": _ai_launch_customer_communication_delivery_send_blocked_reasons(verification),
        "authorized_payload": _ai_launch_customer_communication_delivery_authorized_payload(verification),
        "next_action": _ai_launch_customer_communication_delivery_send_authorization_next_action(
            verification,
            authorization_decision,
        ),
        "source_verification_status": verification["verification_status"],
        "source_delivery_status": verification["source_delivery_status"],
        "source_send_decision": verification["source_send_decision"],
        "source_customer_claim_status": verification["source_customer_claim_status"],
        "source_launch_decision": verification["source_launch_decision"],
    }
    packet["authorization_markdown"] = (
        _ai_improvement_launch_customer_communication_delivery_send_authorization_packet_markdown(packet)
    )
    return packet


def build_ai_improvement_launch_customer_communication_delivery_send_authorization_monitor(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    authorization = build_ai_improvement_launch_customer_communication_delivery_send_authorization_packet(
        conn,
        owner_id,
        improvement_id=improvement_id,
        limit=limit,
    )
    requirements_status = _ai_launch_customer_communication_delivery_send_requirement_status(authorization)
    blocked_reason_status = _ai_launch_customer_communication_delivery_send_blocked_reason_status(authorization)
    payload_status = _ai_launch_customer_communication_delivery_send_payload_status(authorization)
    blockers = _ai_launch_customer_communication_delivery_send_authorization_monitor_blockers(
        authorization,
        requirements_status,
        blocked_reason_status,
        payload_status,
    )
    monitor = {
        "kind": "ai_improvement_launch_customer_communication_delivery_send_authorization_monitor",
        "owner_id": authorization["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": authorization["experiment_id"],
        "status": _ai_launch_customer_communication_delivery_send_authorization_monitor_status(authorization, blockers),
        "risk_level": _ai_launch_customer_communication_delivery_send_authorization_monitor_risk_level(
            authorization,
            blockers,
        ),
        "work_item": authorization["work_item"],
        "authorization_progress": _ai_launch_customer_communication_delivery_send_authorization_progress(authorization),
        "requirements_status": requirements_status,
        "blocked_reason_status": blocked_reason_status,
        "payload_status": payload_status,
        "blockers": blockers,
        "immediate_action": authorization["next_action"],
        "source_authorization_status": authorization["authorization_status"],
        "source_authorization_decision": authorization["authorization_decision"],
        "source_verification_status": authorization["source_verification_status"],
        "source_delivery_status": authorization["source_delivery_status"],
        "source_customer_claim_status": authorization["source_customer_claim_status"],
        "source_launch_decision": authorization["source_launch_decision"],
    }
    monitor["monitor_markdown"] = (
        _ai_improvement_launch_customer_communication_delivery_send_authorization_monitor_markdown(monitor)
    )
    return monitor


def build_ai_improvement_launch_customer_communication_delivery_send_authorization_unblock_plan(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    monitor = build_ai_improvement_launch_customer_communication_delivery_send_authorization_monitor(
        conn,
        owner_id,
        improvement_id=improvement_id,
        limit=limit,
    )
    unblock_tasks = _ai_launch_customer_communication_delivery_send_authorization_unblock_tasks(monitor)
    plan = {
        "kind": "ai_improvement_launch_customer_communication_delivery_send_authorization_unblock_plan",
        "owner_id": monitor["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": monitor["experiment_id"],
        "plan_status": _ai_launch_customer_communication_delivery_send_authorization_unblock_status(monitor),
        "risk_level": monitor["risk_level"],
        "work_item": monitor["work_item"],
        "unblock_tasks": unblock_tasks,
        "authorization_gates": _ai_launch_customer_communication_delivery_send_authorization_unblock_gates(monitor),
        "exit_criteria": _ai_launch_customer_communication_delivery_send_authorization_unblock_exit_criteria(monitor),
        "recheck_plan": _ai_launch_customer_communication_delivery_send_authorization_unblock_recheck_plan(monitor),
        "immediate_action": unblock_tasks[0] if unblock_tasks else monitor["immediate_action"],
        "source_monitor_status": monitor["status"],
        "source_authorization_status": monitor["source_authorization_status"],
        "source_authorization_decision": monitor["source_authorization_decision"],
        "source_verification_status": monitor["source_verification_status"],
        "source_delivery_status": monitor["source_delivery_status"],
        "source_customer_claim_status": monitor["source_customer_claim_status"],
        "source_launch_decision": monitor["source_launch_decision"],
    }
    plan["plan_markdown"] = (
        _ai_improvement_launch_customer_communication_delivery_send_authorization_unblock_plan_markdown(plan)
    )
    return plan


def build_ai_improvement_launch_customer_communication_delivery_send_authorization_unblock_verification_report(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    plan = build_ai_improvement_launch_customer_communication_delivery_send_authorization_unblock_plan(
        conn,
        owner_id,
        improvement_id=improvement_id,
        limit=limit,
    )
    verification_results = _ai_launch_customer_communication_delivery_send_authorization_unblock_verification_results(
        plan
    )
    report = {
        "kind": "ai_improvement_launch_customer_communication_delivery_send_authorization_unblock_verification_report",
        "owner_id": plan["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": plan["experiment_id"],
        "verification_status": _ai_launch_customer_communication_delivery_send_authorization_unblock_verification_status(
            verification_results
        ),
        "risk_level": plan["risk_level"],
        "work_item": plan["work_item"],
        "verification_results": verification_results,
        "failed_checks": _ai_launch_customer_communication_delivery_send_authorization_unblock_failed_checks(
            verification_results
        ),
        "required_follow_up": _ai_launch_customer_communication_delivery_send_authorization_unblock_required_follow_up(
            plan,
            verification_results,
        ),
        "next_action": _ai_launch_customer_communication_delivery_send_authorization_unblock_verification_next_action(
            plan,
            verification_results,
        ),
        "source_plan_status": plan["plan_status"],
        "source_monitor_status": plan["source_monitor_status"],
        "source_authorization_status": plan["source_authorization_status"],
        "source_authorization_decision": plan["source_authorization_decision"],
        "source_verification_status": plan["source_verification_status"],
        "source_delivery_status": plan["source_delivery_status"],
        "source_customer_claim_status": plan["source_customer_claim_status"],
        "source_launch_decision": plan["source_launch_decision"],
    }
    report["verification_markdown"] = (
        _ai_improvement_launch_customer_communication_delivery_send_authorization_unblock_verification_report_markdown(
            report
        )
    )
    return report


def build_ai_improvement_launch_customer_communication_delivery_send_readiness_packet(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    verification = (
        build_ai_improvement_launch_customer_communication_delivery_send_authorization_unblock_verification_report(
            conn,
            owner_id,
            improvement_id=improvement_id,
            limit=limit,
        )
    )
    blockers = _ai_launch_customer_communication_delivery_send_readiness_blockers(verification)
    packet = {
        "kind": "ai_improvement_launch_customer_communication_delivery_send_readiness_packet",
        "owner_id": verification["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": verification["experiment_id"],
        "readiness_status": _ai_launch_customer_communication_delivery_send_readiness_status(verification, blockers),
        "risk_level": _ai_launch_customer_communication_delivery_send_readiness_risk_level(verification, blockers),
        "work_item": verification["work_item"],
        "send_gate": _ai_launch_customer_communication_delivery_send_readiness_gate(verification, blockers),
        "customer_claim": _ai_launch_customer_communication_delivery_send_readiness_customer_claim(verification),
        "advisor_review": _ai_launch_customer_communication_delivery_send_readiness_advisor_review(
            verification,
            blockers,
        ),
        "blockers": blockers,
        "immediate_action": verification["next_action"],
        "source_verification_status": verification["verification_status"],
        "source_plan_status": verification["source_plan_status"],
        "source_monitor_status": verification["source_monitor_status"],
        "source_authorization_status": verification["source_authorization_status"],
        "source_authorization_decision": verification["source_authorization_decision"],
        "source_delivery_status": verification["source_delivery_status"],
        "source_customer_claim_status": verification["source_customer_claim_status"],
        "source_launch_decision": verification["source_launch_decision"],
    }
    packet["readiness_markdown"] = _ai_improvement_launch_customer_communication_delivery_send_readiness_packet_markdown(
        packet
    )
    return packet


def build_ai_improvement_launch_customer_communication_delivery_send_readiness_review_packet(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    improvement_id: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    readiness = build_ai_improvement_launch_customer_communication_delivery_send_readiness_packet(
        conn,
        owner_id,
        improvement_id=improvement_id,
        limit=limit,
    )
    send_decision = _ai_launch_customer_communication_delivery_send_readiness_review_decision(readiness)
    packet = {
        "kind": "ai_improvement_launch_customer_communication_delivery_send_readiness_review_packet",
        "owner_id": readiness["owner_id"],
        "generated_at": now_utc(),
        "experiment_id": readiness["experiment_id"],
        "review_status": send_decision["status"],
        "risk_level": readiness["risk_level"],
        "work_item": readiness["work_item"],
        "send_decision": send_decision,
        "required_approvals": _ai_launch_customer_communication_delivery_send_readiness_review_required_approvals(
            readiness
        ),
        "send_blockers": _ai_launch_customer_communication_delivery_send_readiness_review_blockers(readiness),
        "approved_payload": _ai_launch_customer_communication_delivery_send_readiness_review_payload(
            readiness,
            send_decision,
        ),
        "advisor_next_action": readiness["immediate_action"],
        "source_readiness_status": readiness["readiness_status"],
        "source_send_gate": readiness["send_gate"],
        "source_customer_claim": readiness["customer_claim"],
        "source_advisor_review": readiness["advisor_review"],
        "source_verification_status": readiness["source_verification_status"],
        "source_plan_status": readiness["source_plan_status"],
        "source_monitor_status": readiness["source_monitor_status"],
        "source_authorization_status": readiness["source_authorization_status"],
        "source_authorization_decision": readiness["source_authorization_decision"],
        "source_delivery_status": readiness["source_delivery_status"],
        "source_customer_claim_status": readiness["source_customer_claim_status"],
        "source_launch_decision": readiness["source_launch_decision"],
    }
    packet["review_markdown"] = (
        _ai_improvement_launch_customer_communication_delivery_send_readiness_review_packet_markdown(packet)
    )
    return packet


def require_outreach_draft_table(conn: sqlite3.Connection) -> None:
    if not table_exists(conn, "advisor_outreach_drafts"):
        raise MarketDataUnavailable("Advisor outreach draft table 'advisor_outreach_drafts' is not available")


def require_outreach_compliance_review_table(conn: sqlite3.Connection) -> None:
    require_outreach_draft_table(conn)
    if not table_exists(conn, "advisor_outreach_compliance_reviews"):
        raise MarketDataUnavailable(
            "Advisor outreach compliance review table 'advisor_outreach_compliance_reviews' is not available"
        )


def require_outreach_delivery_record_table(conn: sqlite3.Connection) -> None:
    require_outreach_compliance_review_table(conn)
    if not table_exists(conn, "advisor_outreach_delivery_records"):
        raise MarketDataUnavailable(
            "Advisor outreach delivery record table 'advisor_outreach_delivery_records' is not available"
        )


def require_outreach_delivery_outcome_table(conn: sqlite3.Connection) -> None:
    require_outreach_delivery_record_table(conn)
    if not table_exists(conn, "advisor_outreach_delivery_outcomes"):
        raise MarketDataUnavailable(
            "Advisor outreach delivery outcome table 'advisor_outreach_delivery_outcomes' is not available"
        )


def now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _top_recommendation(conn: sqlite3.Connection, owner_id: str, include_blocked: bool) -> dict[str, Any]:
    workbench = build_advisor_workbench(conn, owner_id=owner_id, limit=1, include_blocked=include_blocked)
    recommendation = workbench.get("top_recommendation")
    if not recommendation:
        raise MarketRecordNotFound("No open or blocked advisor action queue tasks found")
    return recommendation


def _draft_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "saved_advisor_outreach_draft",
        "draft_id": row["draft_id"],
        "owner_id": row["owner_id"],
        "queue_id": row["queue_id"],
        "task_id": row["task_id"],
        "status": row["status"],
        "selection": row["selection"],
        "source_task": json.loads(row["source_task_json"]),
        "source_queue": json.loads(row["source_queue_json"]),
        "customer_email": {"subject": row["subject"], "body": row["body"]},
        "meeting_agenda": json.loads(row["meeting_agenda_json"]),
        "compliance_guardrails": json.loads(row["compliance_guardrails_json"]),
        "approval_required": row["status"] != "approved",
        "draft_markdown": row["draft_markdown"],
        "review_notes": row["review_notes"],
        "reviewer": row["reviewer"],
        "reviewed_at": row["reviewed_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _draft_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "draft_id": row["draft_id"],
        "owner_id": row["owner_id"],
        "queue_id": row["queue_id"],
        "task_id": row["task_id"],
        "status": row["status"],
        "selection": row["selection"],
        "subject": row["subject"],
        "reviewer": row["reviewer"],
        "reviewed_at": row["reviewed_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _compliance_review_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "saved_advisor_outreach_compliance_review",
        "review_id": row["review_id"],
        "draft_id": row["draft_id"],
        "owner_id": row["owner_id"],
        "queue_id": row["queue_id"],
        "task_id": row["task_id"],
        "draft_status": row["draft_status"],
        "risk_level": row["risk_level"],
        "can_approve": bool(row["can_approve"]),
        "approval_recommendation": row["approval_recommendation"],
        "issue_count": row["issue_count"],
        "issues": json.loads(row["issues_json"]),
        "passed_checks": json.loads(row["passed_checks_json"]),
        "source_draft": json.loads(row["source_draft_json"]),
        "review_markdown": row["review_markdown"],
        "created_at": row["created_at"],
    }


def _compliance_review_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "review_id": row["review_id"],
        "draft_id": row["draft_id"],
        "owner_id": row["owner_id"],
        "queue_id": row["queue_id"],
        "task_id": row["task_id"],
        "draft_status": row["draft_status"],
        "risk_level": row["risk_level"],
        "can_approve": bool(row["can_approve"]),
        "approval_recommendation": row["approval_recommendation"],
        "issue_count": row["issue_count"],
        "created_at": row["created_at"],
    }


def _delivery_record_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "saved_advisor_outreach_delivery_record",
        "delivery_id": row["delivery_id"],
        "draft_id": row["draft_id"],
        "owner_id": row["owner_id"],
        "queue_id": row["queue_id"],
        "task_id": row["task_id"],
        "status": row["status"],
        "customer_email": json.loads(row["customer_email_json"]),
        "meeting_agenda": json.loads(row["meeting_agenda_json"]),
        "compliance_review": json.loads(row["compliance_review_json"]),
        "approval_evidence": json.loads(row["approval_evidence_json"]),
        "source_task": json.loads(row["source_task_json"]),
        "packet_markdown": row["packet_markdown"],
        "delivery_notes": row["delivery_notes"],
        "delivered_by": row["delivered_by"],
        "delivered_at": row["delivered_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _delivery_record_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "delivery_id": row["delivery_id"],
        "draft_id": row["draft_id"],
        "owner_id": row["owner_id"],
        "queue_id": row["queue_id"],
        "task_id": row["task_id"],
        "status": row["status"],
        "delivered_by": row["delivered_by"],
        "delivered_at": row["delivered_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _outcome_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "saved_advisor_outreach_delivery_outcome",
        "outcome_id": row["outcome_id"],
        "delivery_id": row["delivery_id"],
        "draft_id": row["draft_id"],
        "owner_id": row["owner_id"],
        "queue_id": row["queue_id"],
        "task_id": row["task_id"],
        "outcome_type": row["outcome_type"],
        "customer_signal": row["customer_signal"],
        "response_text": row["response_text"],
        "next_action": json.loads(row["next_action_json"]),
        "follow_up_due_at": row["follow_up_due_at"],
        "recorded_by": row["recorded_by"],
        "source_delivery": json.loads(row["source_delivery_json"]),
        "outcome_markdown": row["outcome_markdown"],
        "created_at": row["created_at"],
    }


def _outcome_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "outcome_id": row["outcome_id"],
        "delivery_id": row["delivery_id"],
        "draft_id": row["draft_id"],
        "owner_id": row["owner_id"],
        "queue_id": row["queue_id"],
        "task_id": row["task_id"],
        "outcome_type": row["outcome_type"],
        "customer_signal": row["customer_signal"],
        "next_action": json.loads(row["next_action_json"]),
        "follow_up_due_at": row["follow_up_due_at"],
        "recorded_by": row["recorded_by"],
        "created_at": row["created_at"],
    }


def _customer_intent_owner_rows(
    conn: sqlite3.Connection,
    owner_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    filters, params = _customer_intent_filters(owner_id)
    rows = conn.execute(
        f"""
        SELECT
            owner_id,
            COUNT(*) AS outcome_count,
            SUM(CASE WHEN customer_signal = 'positive' THEN 1 ELSE 0 END) AS positive_count,
            SUM(CASE WHEN customer_signal = 'negative' THEN 1 ELSE 0 END) AS negative_count,
            SUM(CASE WHEN customer_signal = 'neutral' THEN 1 ELSE 0 END) AS neutral_count,
            SUM(CASE WHEN outcome_type = 'meeting_scheduled' THEN 1 ELSE 0 END) AS meeting_scheduled_count,
            SUM(CASE WHEN outcome_type = 'needs_more_information' THEN 1 ELSE 0 END) AS needs_information_count,
            SUM(CASE WHEN outcome_type = 'no_response' THEN 1 ELSE 0 END) AS no_response_count,
            SUM(CASE WHEN outcome_type = 'not_interested' THEN 1 ELSE 0 END) AS not_interested_count,
            MAX(created_at) AS last_outcome_at
        FROM advisor_outreach_delivery_outcomes
        WHERE {' AND '.join(filters)}
        GROUP BY owner_id
        ORDER BY last_outcome_at DESC, owner_id ASC
        LIMIT ?
        """,
        [*params, limit],
    ).fetchall()
    intents: list[dict[str, Any]] = []
    for row in rows:
        stats = dict(row)
        latest = _latest_outcome_for_owner(conn, stats["owner_id"])
        intents.append(_customer_intent_row(stats, latest, pending_outcome_count=0))
    return intents


def _customer_intent_filters(owner_id: str | None) -> tuple[list[str], list[Any]]:
    if owner_id:
        return ["owner_id = ?"], [owner_id]
    return ["1 = 1"], []


def _latest_outcome_for_owner(conn: sqlite3.Connection, owner_id: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT *
        FROM advisor_outreach_delivery_outcomes
        WHERE owner_id = ?
        ORDER BY created_at DESC, outcome_id DESC
        LIMIT 1
        """,
        (owner_id,),
    ).fetchone()
    return _outcome_summary(dict(row)) if row else {}


def _customer_intent_row(
    stats: dict[str, Any],
    latest: dict[str, Any],
    pending_outcome_count: int,
) -> dict[str, Any]:
    latest_action = latest.get("next_action") or {}
    score = _customer_intent_score(stats, pending_outcome_count)
    segment = _customer_intent_segment(latest.get("outcome_type"), pending_outcome_count)
    return {
        "owner_id": stats["owner_id"],
        "segment": segment,
        "intent_score": score,
        "outcome_count": int(stats["outcome_count"] or 0),
        "positive_count": int(stats["positive_count"] or 0),
        "negative_count": int(stats["negative_count"] or 0),
        "neutral_count": int(stats["neutral_count"] or 0),
        "meeting_scheduled_count": int(stats["meeting_scheduled_count"] or 0),
        "needs_information_count": int(stats["needs_information_count"] or 0),
        "no_response_count": int(stats["no_response_count"] or 0),
        "not_interested_count": int(stats["not_interested_count"] or 0),
        "pending_outcome_count": pending_outcome_count,
        "latest_outcome_id": latest.get("outcome_id"),
        "latest_delivery_id": latest.get("delivery_id"),
        "latest_outcome_type": latest.get("outcome_type"),
        "latest_customer_signal": latest.get("customer_signal"),
        "last_activity_at": stats["last_outcome_at"],
        "next_action_type": latest_action.get("action_type"),
        "recommended_action": latest_action.get("action", "Review this customer's latest outreach outcome."),
        "follow_up_due_at": latest.get("follow_up_due_at"),
    }


def _customer_intent_score(stats: dict[str, Any], pending_outcome_count: int) -> int:
    return (
        int(stats["meeting_scheduled_count"] or 0) * 4
        + int(stats["positive_count"] or 0) * 3
        + int(stats["needs_information_count"] or 0) * 2
        + pending_outcome_count
        - int(stats["no_response_count"] or 0)
        - int(stats["negative_count"] or 0) * 3
    )


def _customer_intent_segment(outcome_type: str | None, pending_outcome_count: int) -> str:
    if pending_outcome_count > 0:
        return "needs_outcome_capture"
    if outcome_type == "meeting_scheduled":
        return "meeting_ready"
    if outcome_type == "needs_more_information":
        return "needs_answer"
    if outcome_type == "interested":
        return "engaged"
    if outcome_type == "not_interested":
        return "paused"
    if outcome_type == "no_response":
        return "dormant"
    if outcome_type == "resolved":
        return "resolved"
    return "monitor"


def _pending_outcome_counts_by_owner(
    conn: sqlite3.Connection,
    owner_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    filters = ["r.status = 'delivered'", "o.outcome_id IS NULL"]
    params: list[Any] = []
    if owner_id:
        filters.append("r.owner_id = ?")
        params.append(owner_id)
    rows = conn.execute(
        f"""
        SELECT
            r.owner_id,
            COUNT(*) AS pending_outcome_count,
            MAX(COALESCE(r.delivered_at, r.updated_at)) AS last_delivery_at
        FROM advisor_outreach_delivery_records r
        LEFT JOIN advisor_outreach_delivery_outcomes o
            ON o.delivery_id = r.delivery_id AND o.owner_id = r.owner_id
        WHERE {' AND '.join(filters)}
        GROUP BY r.owner_id
        ORDER BY last_delivery_at DESC, r.owner_id ASC
        LIMIT ?
        """,
        [*params, limit],
    ).fetchall()
    return [
        {
            "owner_id": row["owner_id"],
            "pending_outcome_count": int(row["pending_outcome_count"] or 0),
            "last_delivery_at": row["last_delivery_at"],
        }
        for row in rows
    ]


def _customer_intent_pending_only_row(pending: dict[str, Any]) -> dict[str, Any]:
    return {
        "owner_id": pending["owner_id"],
        "segment": "needs_outcome_capture",
        "intent_score": pending["pending_outcome_count"],
        "outcome_count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "neutral_count": 0,
        "meeting_scheduled_count": 0,
        "needs_information_count": 0,
        "no_response_count": 0,
        "not_interested_count": 0,
        "pending_outcome_count": pending["pending_outcome_count"],
        "latest_outcome_id": None,
        "latest_delivery_id": None,
        "latest_outcome_type": None,
        "latest_customer_signal": None,
        "last_activity_at": pending["last_delivery_at"],
        "next_action_type": "record_outcome",
        "recommended_action": "Record missing delivery outcomes before ranking customer follow-up.",
        "follow_up_due_at": None,
    }


def _customer_intent_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    segment_priority = {
        "needs_outcome_capture": 6,
        "needs_answer": 5,
        "meeting_ready": 4,
        "engaged": 3,
        "dormant": 2,
        "monitor": 1,
        "resolved": 0,
        "paused": -1,
    }
    return (
        segment_priority.get(row["segment"], 0),
        int(row["intent_score"]),
        row["last_activity_at"] or "",
    )


def _customer_intent_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "owner_count": len(rows),
        "needs_outcome_capture_count": sum(1 for row in rows if row["segment"] == "needs_outcome_capture"),
        "needs_answer_count": sum(1 for row in rows if row["segment"] == "needs_answer"),
        "meeting_ready_count": sum(1 for row in rows if row["segment"] == "meeting_ready"),
        "engaged_count": sum(1 for row in rows if row["segment"] == "engaged"),
        "dormant_count": sum(1 for row in rows if row["segment"] == "dormant"),
        "paused_count": sum(1 for row in rows if row["segment"] == "paused"),
        "pending_outcome_count": sum(row["pending_outcome_count"] for row in rows),
        "positive_outcome_count": sum(row["positive_count"] for row in rows),
        "negative_outcome_count": sum(row["negative_count"] for row in rows),
    }


def _customer_intent_recommendation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "type": "no_customer_intent",
            "action": "Record customer outreach outcomes to build customer intent ranking.",
        }
    top = rows[0]
    return {
        "type": top["next_action_type"] or top["segment"],
        "owner_id": top["owner_id"],
        "segment": top["segment"],
        "intent_score": top["intent_score"],
        "action": top["recommended_action"],
    }


def _customer_intent_recent_outcomes(
    conn: sqlite3.Connection,
    owner_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    filters, params = _customer_intent_filters(owner_id)
    rows = conn.execute(
        f"""
        SELECT *
        FROM advisor_outreach_delivery_outcomes
        WHERE {' AND '.join(filters)}
        ORDER BY created_at DESC, outcome_id DESC
        LIMIT ?
        """,
        [*params, limit],
    ).fetchall()
    return [_outcome_summary(dict(row)) for row in rows]


def _customer_intent_markdown(dashboard: dict[str, Any]) -> str:
    summary = dashboard["summary"]
    lines = [
        "# Customer Intent Dashboard",
        f"- Owners ranked: {summary['owner_count']}",
        f"- Needs outcome capture: {summary['needs_outcome_capture_count']}",
        f"- Needs answer: {summary['needs_answer_count']}",
        f"- Meeting ready: {summary['meeting_ready_count']}",
        f"- Engaged: {summary['engaged_count']}",
        f"- Dormant: {summary['dormant_count']}",
        "",
        "## Next action",
        f"- {dashboard['top_recommendation']['action']}",
        "",
        "## Ranked customers",
    ]
    for row in dashboard["owner_intents"]:
        lines.append(
            f"- {row['owner_id']}: {row['segment']} "
            f"(score {row['intent_score']}) - {row['recommended_action']}"
        )
    if not dashboard["owner_intents"]:
        lines.append("- None")
    return "\n".join(lines)


def _customer_intent_action_item(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    pending_delivery = _latest_pending_delivery_for_owner(conn, row["owner_id"])
    return {
        "rank": rank,
        "owner_id": row["owner_id"],
        "segment": row["segment"],
        "priority": _customer_intent_action_priority(row),
        "action_type": row["next_action_type"] or row["segment"],
        "recommended_action": row["recommended_action"],
        "rationale": _customer_intent_action_rationale(row),
        "evidence": {
            "intent_score": row["intent_score"],
            "latest_outcome_id": row["latest_outcome_id"],
            "latest_delivery_id": row["latest_delivery_id"] or pending_delivery.get("delivery_id"),
            "latest_outcome_type": row["latest_outcome_type"],
            "latest_customer_signal": row["latest_customer_signal"],
            "pending_outcome_count": row["pending_outcome_count"],
            "last_activity_at": row["last_activity_at"],
            "follow_up_due_at": row["follow_up_due_at"],
        },
        "supporting_routes": _customer_intent_supporting_routes(row, pending_delivery),
        "compliance_guardrails": _customer_intent_compliance_guardrails(row),
    }


def _latest_pending_delivery_for_owner(conn: sqlite3.Connection, owner_id: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT r.*
        FROM advisor_outreach_delivery_records r
        LEFT JOIN advisor_outreach_delivery_outcomes o
            ON o.delivery_id = r.delivery_id AND o.owner_id = r.owner_id
        WHERE r.owner_id = ? AND r.status = 'delivered' AND o.outcome_id IS NULL
        ORDER BY COALESCE(r.delivered_at, r.updated_at) DESC, r.delivery_id DESC
        LIMIT 1
        """,
        (owner_id,),
    ).fetchone()
    return _delivery_record_summary(dict(row)) if row else {}


def _customer_intent_action_priority(row: dict[str, Any]) -> str:
    if row["segment"] in {"needs_outcome_capture", "needs_answer", "meeting_ready"}:
        return "high"
    if row["segment"] == "engaged":
        return "medium"
    if row["segment"] in {"paused", "resolved"}:
        return "hold"
    return "low"


def _customer_intent_action_rationale(row: dict[str, Any]) -> str:
    if row["segment"] == "needs_outcome_capture":
        return "A delivered outreach packet is missing customer-response capture, so downstream intent ranking is incomplete."
    if row["segment"] == "meeting_ready":
        return "The latest recorded outcome scheduled a meeting, making preparation the next highest-value action."
    if row["segment"] == "needs_answer":
        return "The customer asked for more information, so a cited answer should come before another sales touch."
    if row["segment"] == "engaged":
        return "The latest customer signal is positive and should be followed while context is fresh."
    if row["segment"] == "dormant":
        return "The latest outcome indicates no response; a low-friction nudge or dormant mark prevents wasted advisor time."
    if row["segment"] == "paused":
        return "The customer declined or expressed negative intent, so active outreach should pause."
    if row["segment"] == "resolved":
        return "The outreach loop is complete and should remain available as future intent evidence."
    return "The customer has recorded outreach activity that should be monitored for the next compliant step."


def _customer_intent_supporting_routes(
    row: dict[str, Any],
    pending_delivery: dict[str, Any],
) -> list[dict[str, str]]:
    owner = row["owner_id"]
    routes: list[dict[str, str]] = []
    if row["segment"] == "needs_outcome_capture" and pending_delivery:
        routes.append(
            {
                "method": "POST",
                "path": f"/agents/advisor-outreach-deliveries/{pending_delivery['delivery_id']}/outcome?owner_id={owner}",
                "purpose": "Record the missing customer response outcome.",
            }
        )
    if row["latest_outcome_id"]:
        routes.append(
            {
                "method": "GET",
                "path": f"/agents/advisor-outreach-outcomes/{row['latest_outcome_id']}?owner_id={owner}",
                "purpose": "Inspect the saved outcome, next action, and source delivery evidence.",
            }
        )
    routes.append(
        {
            "method": "GET",
            "path": f"/agents/customer-intent-dashboard?owner_id={owner}",
            "purpose": "Refresh the owner-level intent signal before acting.",
        }
    )
    return routes


def _customer_intent_compliance_guardrails(row: dict[str, Any]) -> list[str]:
    guardrails = [
        "Use saved outcome and delivery evidence; do not invent customer intent.",
        "Route any new customer-facing copy through outreach draft review and compliance review.",
    ]
    if row["segment"] == "paused":
        guardrails.append("Do not restart outreach unless the customer re-engages or a reviewer approves a new basis.")
    if row["segment"] in {"needs_answer", "meeting_ready"}:
        guardrails.append("Use local cited evidence for factual claims and preserve required disclosures.")
    return guardrails


def _customer_intent_action_plan_summary(action_items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "action_count": len(action_items),
        "high_priority_count": sum(1 for item in action_items if item["priority"] == "high"),
        "medium_priority_count": sum(1 for item in action_items if item["priority"] == "medium"),
        "low_priority_count": sum(1 for item in action_items if item["priority"] == "low"),
        "hold_count": sum(1 for item in action_items if item["priority"] == "hold"),
    }


def _empty_customer_intent_action() -> dict[str, Any]:
    return {
        "rank": None,
        "owner_id": None,
        "segment": "none",
        "priority": "low",
        "action_type": "record_outcomes",
        "recommended_action": "Record customer outreach outcomes to build an actionable intent plan.",
        "rationale": "No saved customer intent evidence is available yet.",
        "evidence": {},
        "supporting_routes": [{"method": "GET", "path": "/agents/customer-intent-dashboard", "purpose": "Check current intent coverage."}],
        "compliance_guardrails": ["Capture only derived workflow outcomes and avoid exporting raw customer notes."],
    }


def _customer_intent_action_plan_markdown(plan: dict[str, Any]) -> str:
    summary = plan["summary"]
    lines = [
        "# Customer Intent Action Plan",
        f"- Actions: {summary['action_count']}",
        f"- High priority: {summary['high_priority_count']}",
        f"- Medium priority: {summary['medium_priority_count']}",
        f"- Low priority: {summary['low_priority_count']}",
        f"- Hold: {summary['hold_count']}",
        "",
        "## Top action",
        f"- {plan['top_action']['recommended_action']}",
        "",
        "## Worklist",
    ]
    for item in plan["action_items"]:
        lines.append(
            f"- {item['rank']}. {item['owner_id']} [{item['priority']}]: "
            f"{item['recommended_action']}"
        )
    if not plan["action_items"]:
        lines.append("- None")
    return "\n".join(lines)


def _source_outcome_for_action(conn: sqlite3.Connection, action: dict[str, Any]) -> dict[str, Any] | None:
    outcome_id = action.get("evidence", {}).get("latest_outcome_id")
    owner_id = action.get("owner_id")
    if not outcome_id or not owner_id:
        return None
    try:
        return get_outreach_delivery_outcome(conn, int(outcome_id), owner_id)
    except MarketRecordNotFound:
        return None


def _customer_intent_packet_type(action: dict[str, Any]) -> str:
    action_type = action.get("action_type")
    if action_type == "prepare_meeting":
        return "meeting_prep"
    if _customer_intent_customer_copy_allowed(action):
        return "customer_followup"
    return "internal_task"


def _customer_intent_customer_copy_allowed(action: dict[str, Any]) -> bool:
    return action.get("action_type") in {"prepare_meeting", "send_follow_up", "answer_question", "schedule_nudge"}


def _customer_intent_advisor_instructions(action: dict[str, Any]) -> list[str]:
    action_type = action.get("action_type")
    if action_type == "record_outcome":
        return [
            "Open the latest delivered outreach record for this owner.",
            "Record the customer response outcome before generating more customer-facing copy.",
        ]
    if action_type == "prepare_meeting":
        return [
            "Confirm the meeting time and agenda with the customer.",
            "Refresh local portfolio, watchlist, and market context before the meeting.",
            "Route any new customer-facing language through compliance review before sending.",
        ]
    if action_type == "answer_question":
        return [
            "Answer the customer's question using local cited evidence.",
            "Avoid unsupported recommendations or price-target language.",
            "Save the response as a reviewed outreach draft before delivery.",
        ]
    if action_type == "pause_outreach":
        return [
            "Do not send new outreach for this topic.",
            "Preserve the decline reason for future suitability and cadence checks.",
        ]
    if action_type == "close_loop":
        return [
            "Mark the workflow complete in the advisor notes.",
            "Retain the outcome as customer-intent evidence for future ranking.",
        ]
    return [
        "Review the source outcome and action-plan evidence.",
        "Use the recommended action only if it is still consistent with customer context and compliance policy.",
    ]


def _customer_intent_customer_copy(action: dict[str, Any], source_outcome: dict[str, Any] | None) -> dict[str, Any]:
    if not _customer_intent_customer_copy_allowed(action):
        return {
            "send_allowed": False,
            "subject": None,
            "body": None,
            "reason": "This action is internal-only or requires missing outcome capture first.",
        }
    owner_id = action.get("owner_id") or "customer"
    action_type = action.get("action_type")
    if action_type == "prepare_meeting":
        subject = "Confirming your portfolio review discussion"
        body = (
            "Hi, thanks for confirming time to discuss your portfolio. "
            "I will prepare the agenda around your latest question and the current local market context. "
            "Before we meet, please share any priorities you want covered."
        )
    elif action_type == "answer_question":
        subject = "Following up with the information you requested"
        body = (
            "Hi, I am pulling together a concise answer using our latest local market evidence. "
            "I will keep the response focused on your question and call out any data limitations or disclosures."
        )
    elif action_type == "schedule_nudge":
        subject = "Quick follow-up on the portfolio note"
        body = (
            "Hi, I wanted to check whether the earlier portfolio note is still useful to review. "
            "If this is not a priority right now, I can pause follow-up."
        )
    else:
        subject = "Next steps from our portfolio discussion"
        body = (
            "Hi, thanks for your response. I can help with the next step and will keep the guidance tied to current, reviewed evidence."
        )
    return {
        "send_allowed": True,
        "owner_id": owner_id,
        "subject": subject,
        "body": body,
        "source_outcome_id": source_outcome.get("outcome_id") if source_outcome else None,
        "requires_review": True,
    }


def _customer_intent_followup_packet_markdown(packet: dict[str, Any]) -> str:
    copy = packet["customer_copy"]
    lines = [
        "# Customer Intent Follow-up Packet",
        f"- Owner: {packet['owner_id']}",
        f"- Packet type: {packet['packet_type']}",
        f"- Compliance review required: {packet['compliance_review_required']}",
        "",
        "## Advisor instructions",
    ]
    lines.extend([f"- {instruction}" for instruction in packet["advisor_instructions"]])
    lines.extend(["", "## Customer copy"])
    if copy["send_allowed"]:
        lines.extend([f"- Subject: {copy['subject']}", "", copy["body"]])
    else:
        lines.append(f"- Not generated: {copy['reason']}")
    return "\n".join(lines)


def _customer_intent_followup_review_checks(packet: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[dict[str, Any]] = []
    passed_checks: list[str] = []
    copy = packet["customer_copy"]
    action = packet["action_item"]

    if copy["send_allowed"]:
        subject = copy.get("subject") or ""
        body = copy.get("body") or ""
        combined = f"{subject}\n{body}".lower()
        if subject and body:
            passed_checks.append("Customer copy includes subject and body.")
        else:
            issues.append(_issue("critical", "missing_customer_copy", "Customer copy must include subject and body."))
        for phrase, message in RISKY_PHRASES.items():
            if phrase in combined:
                issues.append(_issue("critical", "risky_phrase", message, phrase=phrase))
        if copy.get("requires_review") and packet.get("compliance_review_required"):
            passed_checks.append("Customer copy is marked for compliance review before delivery.")
        else:
            issues.append(_issue("high", "missing_review_gate", "Customer copy must require review before delivery."))
        if _has_disclosure_language(combined):
            passed_checks.append("Customer copy includes review/disclosure framing.")
        else:
            issues.append(_issue("medium", "missing_review_framing", "Customer copy should make review/disclosure framing explicit."))
        if packet.get("source_outcome"):
            passed_checks.append("Source outcome evidence is attached.")
        else:
            issues.append(_issue("high", "missing_source_outcome", "Customer copy should be tied to a saved source outcome."))
    else:
        passed_checks.append("Packet is internal-only and does not generate customer-facing copy.")

    if action.get("evidence", {}).get("intent_score") is not None:
        passed_checks.append("Intent score evidence is attached.")
    else:
        issues.append(_issue("medium", "missing_intent_score", "Packet should include intent score evidence."))
    if packet.get("compliance_guardrails"):
        passed_checks.append("Compliance guardrails are attached.")
    else:
        issues.append(_issue("medium", "missing_guardrails", "Packet should include compliance guardrails."))
    return issues, passed_checks


def _customer_intent_review_source_packet(packet: dict[str, Any]) -> dict[str, Any]:
    copy = packet["customer_copy"]
    action = packet["action_item"]
    return {
        "owner_id": packet["owner_id"],
        "packet_type": packet["packet_type"],
        "action_type": action.get("action_type"),
        "priority": action.get("priority"),
        "source_outcome_id": copy.get("source_outcome_id"),
        "send_allowed": copy.get("send_allowed"),
        "requires_review": copy.get("requires_review"),
    }


def _customer_intent_followup_review_markdown(review: dict[str, Any]) -> str:
    lines = [
        "# Customer Intent Follow-up Review",
        f"- Owner: {review['owner_id']}",
        f"- Packet type: {review['packet_type']}",
        f"- Risk level: {review['risk_level']}",
        f"- Recommendation: {review['recommendation']}",
        "",
        "## Issues",
    ]
    if review["issues"]:
        lines.extend(f"- [{issue['severity']}] {issue['message']}" for issue in review["issues"])
    else:
        lines.append("- No issues found.")
    lines.extend(["", "## Passed Checks"])
    lines.extend(f"- {check}" for check in review["passed_checks"])
    return "\n".join(lines)


def _customer_intent_followup_draft(
    packet: dict[str, Any],
    risk_level: str,
    passed_checks: list[str],
) -> dict[str, Any]:
    source_outcome = packet.get("source_outcome") or {}
    action = packet["action_item"]
    copy = packet["customer_copy"]
    queue_id = source_outcome.get("queue_id")
    task_id = source_outcome.get("task_id")
    if queue_id is None or task_id is None:
        raise ValueError("Customer intent follow-up draft requires a source outcome with queue and task references")
    guardrails = _customer_intent_followup_draft_guardrails(packet, risk_level, passed_checks)
    source_task = _customer_intent_followup_source_task(packet, source_outcome)
    source_queue = _customer_intent_followup_source_queue(packet, source_outcome)
    agenda = _customer_intent_followup_meeting_agenda(packet)
    draft = {
        "kind": "customer_intent_followup_draft",
        "owner_id": packet["owner_id"],
        "queue_id": queue_id,
        "task_id": task_id,
        "selection": "customer_intent_followup",
        "source_task": source_task,
        "source_queue": source_queue,
        "customer_email": {"subject": copy["subject"], "body": copy["body"]},
        "meeting_agenda": agenda,
        "compliance_guardrails": guardrails,
        "approval_required": True,
        "source_packet": {
            "packet_type": packet["packet_type"],
            "source_outcome_id": copy.get("source_outcome_id"),
            "action_type": action.get("action_type"),
            "priority": action.get("priority"),
        },
    }
    draft["draft_markdown"] = _customer_intent_followup_draft_markdown(draft)
    return draft


def _customer_intent_followup_source_task(
    packet: dict[str, Any],
    source_outcome: dict[str, Any],
) -> dict[str, Any]:
    action = packet["action_item"]
    evidence = action.get("evidence", {})
    return {
        "task_id": source_outcome["task_id"],
        "title": "Customer intent follow-up",
        "urgency": action.get("priority", "medium"),
        "status": "open",
        "rationale": action.get("rationale", "Customer intent indicates a follow-up action is available."),
        "completion_criteria": action.get("recommended_action", "Complete the reviewed follow-up action."),
        "evidence": {
            "intent_score": evidence.get("intent_score"),
            "latest_outcome_id": evidence.get("latest_outcome_id"),
            "latest_delivery_id": evidence.get("latest_delivery_id"),
            "latest_outcome_type": evidence.get("latest_outcome_type"),
            "latest_customer_signal": evidence.get("latest_customer_signal"),
        },
    }


def _customer_intent_followup_source_queue(
    packet: dict[str, Any],
    source_outcome: dict[str, Any],
) -> dict[str, Any]:
    return {
        "queue_id": source_outcome["queue_id"],
        "title": "Customer intent follow-up queue",
        "focus": "customer_intent",
        "source": "customer_intent_followup_packet",
        "packet_type": packet["packet_type"],
    }


def _customer_intent_followup_meeting_agenda(packet: dict[str, Any]) -> list[dict[str, Any]]:
    action = packet["action_item"]
    return [
        {
            "section": "Customer intent signal",
            "items": [
                f"Segment: {action.get('segment')}",
                f"Priority: {action.get('priority')}",
                action.get("rationale", "Review customer intent evidence."),
            ],
        },
        {
            "section": "Advisor next step",
            "items": [action.get("recommended_action", "Review and complete the follow-up action.")],
        },
    ]


def _customer_intent_followup_draft_guardrails(
    packet: dict[str, Any],
    risk_level: str,
    passed_checks: list[str],
) -> dict[str, Any]:
    return {
        "do_say": [
            "Use the saved source outcome as the basis for follow-up.",
            "Confirm this is for advisor review before customer delivery.",
            "Call out that claims should be tied to current local evidence.",
        ],
        "do_not_say": [
            "Do not present the draft as final advice before approval.",
            "Do not add recommendations that are not supported by saved evidence.",
            "Do not remove required review or disclosure framing.",
        ],
        "requires_disclosure": True,
        "review_checklist": [
            "Confirm the source outcome still reflects customer intent.",
            "Confirm the customer copy matches the reviewed packet.",
            "Confirm compliance review remains low or medium risk before approval.",
            f"Packet preflight risk level was {risk_level}.",
            f"Preflight checks passed: {len(passed_checks)}.",
        ],
        "source_packet_guardrails": packet.get("compliance_guardrails", []),
    }


def _customer_intent_followup_draft_markdown(draft: dict[str, Any]) -> str:
    lines = [
        f"# Customer Intent Follow-up Draft: {draft['owner_id']}",
        "",
        f"Subject: {draft['customer_email']['subject']}",
        "",
        draft["customer_email"]["body"],
        "",
        "## Meeting Agenda",
    ]
    for section in draft["meeting_agenda"]:
        lines.append(f"- {section['section']}: {'; '.join(section['items'])}")
    lines.extend(["", "## Review Checklist"])
    lines.extend(f"- {item}" for item in draft["compliance_guardrails"]["review_checklist"])
    return "\n".join(lines)


def _timeline_draft_events(conn: sqlite3.Connection, owner_id: str, limit: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT draft_id, queue_id, task_id, status, selection, subject, created_at, updated_at
        FROM advisor_outreach_drafts
        WHERE owner_id = ?
        ORDER BY updated_at DESC, draft_id DESC
        LIMIT ?
        """,
        (owner_id, limit),
    ).fetchall()
    return [
        {
            "event_id": f"draft:{row['draft_id']}",
            "event_type": "outreach_draft",
            "occurred_at": row["created_at"],
            "title": f"Outreach draft created: {row['subject']}",
            "summary": f"Draft {row['draft_id']} is {row['status']} from {row['selection']}.",
            "references": {"draft_id": row["draft_id"], "queue_id": row["queue_id"], "task_id": row["task_id"]},
        }
        for row in rows
    ]


def _timeline_compliance_review_events(conn: sqlite3.Connection, owner_id: str, limit: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT review_id, draft_id, queue_id, task_id, risk_level, can_approve,
               approval_recommendation, issue_count, created_at
        FROM advisor_outreach_compliance_reviews
        WHERE owner_id = ?
        ORDER BY created_at DESC, review_id DESC
        LIMIT ?
        """,
        (owner_id, limit),
    ).fetchall()
    return [
        {
            "event_id": f"review:{row['review_id']}",
            "event_type": "compliance_review",
            "occurred_at": row["created_at"],
            "title": f"Compliance review {row['risk_level']}",
            "summary": f"{row['approval_recommendation']} with {row['issue_count']} issue(s).",
            "references": {
                "review_id": row["review_id"],
                "draft_id": row["draft_id"],
                "queue_id": row["queue_id"],
                "task_id": row["task_id"],
                "can_approve": bool(row["can_approve"]),
            },
        }
        for row in rows
    ]


def _timeline_delivery_events(conn: sqlite3.Connection, owner_id: str, limit: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT delivery_id, draft_id, queue_id, task_id, status, delivered_by,
               delivered_at, created_at, updated_at
        FROM advisor_outreach_delivery_records
        WHERE owner_id = ?
        ORDER BY updated_at DESC, delivery_id DESC
        LIMIT ?
        """,
        (owner_id, limit),
    ).fetchall()
    events: list[dict[str, Any]] = []
    for row in rows:
        status = row["status"]
        occurred_at = row["delivered_at"] or row["updated_at"] or row["created_at"]
        events.append(
            {
                "event_id": f"delivery:{row['delivery_id']}",
                "event_type": f"delivery_{status}",
                "occurred_at": occurred_at,
                "title": f"Delivery {status}",
                "summary": f"Delivery record {row['delivery_id']} is {status}.",
                "references": {
                    "delivery_id": row["delivery_id"],
                    "draft_id": row["draft_id"],
                    "queue_id": row["queue_id"],
                    "task_id": row["task_id"],
                    "delivered_by": row["delivered_by"],
                },
            }
        )
    return events


def _timeline_outcome_events(conn: sqlite3.Connection, owner_id: str, limit: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT outcome_id, delivery_id, draft_id, queue_id, task_id, outcome_type,
               customer_signal, next_action_json, follow_up_due_at, recorded_by, created_at
        FROM advisor_outreach_delivery_outcomes
        WHERE owner_id = ?
        ORDER BY created_at DESC, outcome_id DESC
        LIMIT ?
        """,
        (owner_id, limit),
    ).fetchall()
    events: list[dict[str, Any]] = []
    for row in rows:
        next_action = json.loads(row["next_action_json"])
        events.append(
            {
                "event_id": f"outcome:{row['outcome_id']}",
                "event_type": "customer_outcome",
                "occurred_at": row["created_at"],
                "title": f"Customer outcome: {row['outcome_type']}",
                "summary": next_action.get("action", "Customer outcome was recorded."),
                "references": {
                    "outcome_id": row["outcome_id"],
                    "delivery_id": row["delivery_id"],
                    "draft_id": row["draft_id"],
                    "queue_id": row["queue_id"],
                    "task_id": row["task_id"],
                    "customer_signal": row["customer_signal"],
                    "follow_up_due_at": row["follow_up_due_at"],
                    "recorded_by": row["recorded_by"],
                },
            }
        )
    return events


def _timeline_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "event_count": len(events),
        "draft_event_count": sum(1 for event in events if event["event_type"] == "outreach_draft"),
        "review_event_count": sum(1 for event in events if event["event_type"] == "compliance_review"),
        "delivery_event_count": sum(1 for event in events if event["event_type"].startswith("delivery_")),
        "outcome_event_count": sum(1 for event in events if event["event_type"] == "customer_outcome"),
        "latest_event_type": events[0]["event_type"] if events else None,
        "latest_event_at": events[0]["occurred_at"] if events else None,
    }


def _customer_engagement_timeline_markdown(timeline: dict[str, Any]) -> str:
    summary = timeline["summary"]
    lines = [
        "# Customer Engagement Timeline",
        f"- Owner: {timeline['owner_id']}",
        f"- Events: {summary['event_count']}",
        f"- Latest event: {summary['latest_event_type'] or 'none'}",
        "",
        "## Current top action",
        f"- {timeline['top_action']['recommended_action']}",
        "",
        "## Events",
    ]
    for event in timeline["events"]:
        lines.append(f"- {event['occurred_at']}: {event['title']} - {event['summary']}")
    if not timeline["events"]:
        lines.append("- None")
    return "\n".join(lines)


def _engagement_brief_current_intent(top_action: dict[str, Any]) -> dict[str, Any]:
    evidence = top_action.get("evidence", {})
    return {
        "owner_id": top_action.get("owner_id"),
        "segment": top_action.get("segment"),
        "priority": top_action.get("priority"),
        "action_type": top_action.get("action_type"),
        "intent_score": evidence.get("intent_score"),
        "latest_outcome_type": evidence.get("latest_outcome_type"),
        "latest_customer_signal": evidence.get("latest_customer_signal"),
        "follow_up_due_at": evidence.get("follow_up_due_at"),
    }


def _engagement_brief_next_action(top_action: dict[str, Any]) -> dict[str, Any]:
    routes = top_action.get("supporting_routes") or []
    return {
        "action_type": top_action.get("action_type"),
        "recommended_action": top_action.get("recommended_action"),
        "rationale": top_action.get("rationale"),
        "primary_route": routes[0] if routes else None,
    }


def _engagement_brief_talking_points(
    top_action: dict[str, Any],
    events: list[dict[str, Any]],
) -> list[str]:
    points = [
        f"Lead with the current customer intent segment: {top_action.get('segment') or 'unknown'}.",
        top_action.get("recommended_action") or "Review the customer engagement history before acting.",
    ]
    outcome_events = [event for event in events if event["event_type"] == "customer_outcome"]
    if outcome_events:
        points.append(f"Reference the latest customer outcome: {outcome_events[0]['summary']}")
    delivery_events = [event for event in events if event["event_type"].startswith("delivery_")]
    if delivery_events:
        points.append(f"Confirm the latest delivery status: {delivery_events[0]['title']}.")
    return points


def _engagement_brief_avoid(top_action: dict[str, Any]) -> list[str]:
    avoid = [
        "Do not invent customer intent beyond saved outcomes and delivery records.",
        "Do not send customer-facing copy without the review and approval workflow.",
    ]
    if top_action.get("segment") == "paused":
        avoid.append("Do not restart outreach after a negative signal without an approved new basis.")
    if top_action.get("action_type") == "record_outcome":
        avoid.append("Do not draft new outreach until the missing outcome is recorded.")
    return avoid


def _engagement_brief_evidence_references(
    events: list[dict[str, Any]],
    top_action: dict[str, Any],
) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for event in events:
        references.append(
            {
                "event_id": event["event_id"],
                "event_type": event["event_type"],
                "occurred_at": event["occurred_at"],
                "references": event["references"],
            }
        )
    if top_action.get("evidence"):
        references.append({"event_id": "top_action", "event_type": "intent_action", "references": top_action["evidence"]})
    return references


def _customer_engagement_brief_markdown(brief: dict[str, Any]) -> str:
    summary = brief["summary"]
    lines = [
        "# Customer Engagement Brief",
        f"- Owner: {brief['owner_id']}",
        f"- Latest event: {summary['latest_event_type'] or 'none'}",
        f"- Current segment: {summary['current_segment'] or 'none'}",
        f"- Priority: {summary['priority'] or 'none'}",
        "",
        "## Next best action",
        f"- {brief['next_best_action']['recommended_action']}",
        "",
        "## Talking points",
    ]
    lines.extend(f"- {point}" for point in brief["talking_points"])
    lines.extend(["", "## Avoid"])
    lines.extend(f"- {item}" for item in brief["avoid"])
    return "\n".join(lines)


def _engagement_cadence_checks(brief: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[dict[str, Any]] = []
    passed_checks: list[str] = []
    intent = brief["current_intent"]
    action = brief["next_best_action"]
    action_type = action.get("action_type")
    segment = intent.get("segment")
    signal = intent.get("latest_customer_signal")

    if action_type == "record_outcome":
        issues.append(
            _issue(
                "critical",
                "missing_outcome_capture",
                "A delivered outreach record is missing customer-response capture; record the outcome before new contact.",
            )
        )
    else:
        passed_checks.append("No missing-outcome capture block is attached to the top action.")

    if segment == "paused" or signal == "negative":
        issues.append(
            _issue(
                "critical",
                "customer_paused_or_negative",
                "Latest customer signal indicates outreach should pause unless a reviewer approves a new basis.",
            )
        )
    else:
        passed_checks.append("No paused or negative customer signal is blocking contact.")

    if action_type in {"prepare_meeting", "answer_question", "send_follow_up", "schedule_nudge"}:
        passed_checks.append("Top action is eligible for reviewed customer contact.")
    elif action_type in {"pause_outreach", "close_loop"}:
        issues.append(_issue("high", "internal_only_action", "Top action is internal-only and should not produce customer contact."))
    else:
        issues.append(_issue("medium", "review_action_type", "Top action should be reviewed before customer contact."))

    if action.get("primary_route"):
        passed_checks.append("A primary next route is available.")
    else:
        issues.append(_issue("medium", "missing_next_route", "Cadence review should include a primary route for the advisor."))

    if brief.get("evidence_references"):
        passed_checks.append("Timeline evidence references are attached.")
    else:
        issues.append(_issue("medium", "missing_evidence_references", "Cadence review should include timeline evidence references."))

    return issues, passed_checks


def _engagement_cadence_status(risk_level: str, contact_allowed: bool) -> str:
    if not contact_allowed:
        return "blocked"
    if risk_level == "medium":
        return "caution"
    return "ready"


def _engagement_cadence_recommendation(brief: dict[str, Any], status: str) -> str:
    action = brief["next_best_action"]
    if status == "blocked":
        return "Resolve cadence blockers before customer contact."
    if status == "caution":
        return "Proceed only through reviewed outreach workflow with attached evidence and guardrails."
    return action.get("recommended_action") or "Proceed through the reviewed outreach workflow."


def _engagement_cadence_next_route(brief: dict[str, Any], contact_allowed: bool) -> dict[str, Any] | None:
    action = brief["next_best_action"]
    owner_id = brief["owner_id"]
    if not contact_allowed:
        return action.get("primary_route")
    return {
        "method": "GET",
        "path": f"/agents/customer-intent-followup-review?owner_id={owner_id}",
        "purpose": "Preflight the generated follow-up packet before drafting or delivery.",
    }


def _customer_engagement_cadence_review_markdown(review: dict[str, Any]) -> str:
    lines = [
        "# Customer Engagement Cadence Review",
        f"- Owner: {review['owner_id']}",
        f"- Contact allowed: {review['contact_allowed']}",
        f"- Status: {review['contact_status']}",
        f"- Recommendation: {review['recommendation']}",
        "",
        "## Issues",
    ]
    if review["issues"]:
        lines.extend(f"- [{issue['severity']}] {issue['message']}" for issue in review["issues"])
    else:
        lines.append("- No issues found.")
    lines.extend(["", "## Passed Checks"])
    lines.extend(f"- {check}" for check in review["passed_checks"])
    return "\n".join(lines)


def _engagement_owner_ids(conn: sqlite3.Connection, limit: int) -> list[str]:
    rows = conn.execute(
        """
        SELECT owner_id, MAX(last_activity_at) AS last_activity_at
        FROM (
            SELECT owner_id, updated_at AS last_activity_at FROM advisor_outreach_drafts
            UNION ALL
            SELECT owner_id, created_at AS last_activity_at FROM advisor_outreach_compliance_reviews
            UNION ALL
            SELECT owner_id, updated_at AS last_activity_at FROM advisor_outreach_delivery_records
            UNION ALL
            SELECT owner_id, created_at AS last_activity_at FROM advisor_outreach_delivery_outcomes
        )
        GROUP BY owner_id
        ORDER BY last_activity_at DESC, owner_id ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [row["owner_id"] for row in rows]


def _engagement_cadence_dashboard_row(review: dict[str, Any]) -> dict[str, Any]:
    intent = review["current_intent"]
    route = review.get("next_route") or {}
    return {
        "owner_id": review["owner_id"],
        "contact_status": review["contact_status"],
        "contact_allowed": review["contact_allowed"],
        "issue_count": review["issue_count"],
        "segment": intent.get("segment"),
        "action_type": intent.get("action_type"),
        "priority": intent.get("priority"),
        "intent_score": intent.get("intent_score"),
        "recommendation": review["recommendation"],
        "next_route": route,
        "latest_event_at": review["source_brief_summary"].get("latest_event_at"),
    }


def _engagement_cadence_dashboard_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    status_priority = {"ready": 3, "caution": 2, "blocked": 1}
    priority_score = {"high": 3, "medium": 2, "low": 1, "hold": 0, None: 0}
    return (
        status_priority.get(row["contact_status"], 0),
        priority_score.get(row["priority"], 0),
        row["latest_event_at"] or "",
    )


def _engagement_cadence_dashboard_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "customer_count": len(rows),
        "ready_count": sum(1 for row in rows if row["contact_status"] == "ready"),
        "caution_count": sum(1 for row in rows if row["contact_status"] == "caution"),
        "blocked_count": sum(1 for row in rows if row["contact_status"] == "blocked"),
        "contact_allowed_count": sum(1 for row in rows if row["contact_allowed"]),
        "total_issue_count": sum(row["issue_count"] for row in rows),
    }


def _engagement_cadence_dashboard_recommendation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "type": "no_engagement_history",
            "action": "Capture outreach outcomes before building a cadence dashboard.",
        }
    top = rows[0]
    return {
        "type": top["contact_status"],
        "owner_id": top["owner_id"],
        "action_type": top["action_type"],
        "action": top["recommendation"],
        "next_route": top["next_route"],
    }


def _customer_engagement_cadence_dashboard_markdown(dashboard: dict[str, Any]) -> str:
    summary = dashboard["summary"]
    lines = [
        "# Customer Engagement Cadence Dashboard",
        f"- Customers: {summary['customer_count']}",
        f"- Ready: {summary['ready_count']}",
        f"- Caution: {summary['caution_count']}",
        f"- Blocked: {summary['blocked_count']}",
        "",
        "## Top recommendation",
        f"- {dashboard['top_recommendation']['action']}",
        "",
        "## Customers",
    ]
    for row in dashboard["customers"]:
        lines.append(f"- {row['owner_id']} [{row['contact_status']}]: {row['recommendation']}")
    if not dashboard["customers"]:
        lines.append("- None")
    return "\n".join(lines)


def _engagement_action_queue_task(row: dict[str, Any], rank: int) -> dict[str, Any]:
    contact_status = row["contact_status"]
    action_type = row.get("action_type") or "review"
    owner_id = row["owner_id"]
    priority = _engagement_action_queue_priority(row)
    return {
        "rank": rank,
        "task_id": f"engagement:{owner_id}:{action_type}",
        "owner_id": owner_id,
        "status": "ready" if contact_status == "ready" else "blocked" if contact_status == "blocked" else "review",
        "priority": priority,
        "contact_status": contact_status,
        "action_type": action_type,
        "title": _engagement_action_queue_title(row),
        "rationale": _engagement_action_queue_rationale(row),
        "completion_criteria": _engagement_action_queue_completion(row),
        "next_route": row.get("next_route"),
        "evidence": {
            "segment": row.get("segment"),
            "intent_score": row.get("intent_score"),
            "issue_count": row.get("issue_count"),
            "latest_event_at": row.get("latest_event_at"),
        },
        "guardrails": _engagement_action_queue_guardrails(row),
    }


def _engagement_action_queue_priority(row: dict[str, Any]) -> str:
    if row["contact_status"] == "ready" and row.get("priority") == "high":
        return "high"
    if row["contact_status"] == "blocked":
        return "blocked"
    if row["contact_status"] == "caution":
        return "medium"
    return row.get("priority") or "low"


def _engagement_action_queue_title(row: dict[str, Any]) -> str:
    if row["contact_status"] == "blocked":
        return f"Resolve cadence blockers for {row['owner_id']}"
    if row["contact_status"] == "caution":
        return f"Review cadence caution for {row['owner_id']}"
    return f"Run {row.get('action_type') or 'next action'} for {row['owner_id']}"


def _engagement_action_queue_rationale(row: dict[str, Any]) -> str:
    if row["contact_status"] == "blocked":
        return "Cadence review found blockers that must be cleared before customer contact."
    if row["contact_status"] == "caution":
        return "Customer contact may be possible, but the cadence review requires advisor judgment first."
    return row.get("recommendation") or "Cadence review marked this customer ready for the next reviewed action."


def _engagement_action_queue_completion(row: dict[str, Any]) -> str:
    route = row.get("next_route") or {}
    if row["contact_status"] == "blocked":
        return "Blocker is resolved or the customer remains paused with no new outreach."
    if route.get("path"):
        return f"Run {route['method']} {route['path']} and record the resulting workflow state."
    return "Review the customer engagement brief and update the workflow state."


def _engagement_action_queue_guardrails(row: dict[str, Any]) -> list[str]:
    guardrails = [
        "Use saved cadence-review evidence; do not invent customer intent.",
        "Use the reviewed outreach workflow before customer-facing delivery.",
    ]
    if row["contact_status"] == "blocked":
        guardrails.append("Do not contact the customer until blockers are cleared.")
    if row["contact_status"] == "caution":
        guardrails.append("Confirm the caution reason before drafting or sending follow-up.")
    return guardrails


def _engagement_action_queue_summary(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_count": len(tasks),
        "ready_count": sum(1 for task in tasks if task["status"] == "ready"),
        "review_count": sum(1 for task in tasks if task["status"] == "review"),
        "blocked_count": sum(1 for task in tasks if task["status"] == "blocked"),
        "high_priority_count": sum(1 for task in tasks if task["priority"] == "high"),
    }


def _empty_engagement_action_queue_task() -> dict[str, Any]:
    return {
        "rank": None,
        "task_id": None,
        "owner_id": None,
        "status": "empty",
        "priority": "low",
        "contact_status": "none",
        "action_type": "capture_engagement",
        "title": "Capture customer engagement outcomes",
        "rationale": "No customer cadence evidence is available yet.",
        "completion_criteria": "Record outreach delivery outcomes to populate the action queue.",
        "next_route": {"method": "GET", "path": "/agents/customer-engagement-cadence-dashboard", "purpose": "Check cadence coverage."},
        "evidence": {},
        "guardrails": ["Do not create customer-facing follow-up without saved engagement evidence."],
    }


def _empty_customer_engagement_task_brief() -> dict[str, Any]:
    return {
        "kind": "customer_engagement_task_brief",
        "owner_id": None,
        "generated_at": now_utc(),
        "task": _empty_engagement_action_queue_task(),
        "customer_context": {
            "segment": None,
            "priority": "low",
            "intent_score": 0,
            "latest_customer_signal": None,
            "latest_event_at": None,
            "contact_status": "empty",
            "contact_allowed": False,
        },
        "execution_plan": {
            "objective": "Capture customer engagement outcomes",
            "recommended_route": {"method": "GET", "path": "/agents/customer-engagement-action-queue"},
            "steps": ["Record delivery outcomes before generating a customer-specific task brief."],
            "stop_condition": "No customer-specific outreach should run without saved engagement evidence.",
        },
        "conversation_guide": {"opening": None, "talking_points": [], "proof_points": [], "avoid": []},
        "compliance_guardrails": ["Do not infer customer intent without saved engagement evidence."],
        "completion_measurement": {
            "success_state": "At least one delivered outreach outcome is recorded.",
            "record_route": {"method": "POST", "path": "/agents/advisor-outreach-deliveries/{delivery_id}/outcome"},
        },
        "source_review_summary": {
            "contact_status": "empty",
            "contact_allowed": False,
            "issue_count": 0,
            "passed_check_count": 0,
        },
    }


def _engagement_task_brief_context(
    engagement_brief: dict[str, Any],
    cadence_review: dict[str, Any],
) -> dict[str, Any]:
    intent = engagement_brief["current_intent"]
    return {
        "segment": intent.get("segment"),
        "priority": intent.get("priority"),
        "intent_score": intent.get("intent_score"),
        "latest_customer_signal": intent.get("latest_customer_signal"),
        "latest_outcome_type": intent.get("latest_outcome_type"),
        "follow_up_due_at": intent.get("follow_up_due_at"),
        "latest_event_at": engagement_brief["summary"].get("latest_event_at"),
        "contact_status": cadence_review["contact_status"],
        "contact_allowed": cadence_review["contact_allowed"],
    }


def _engagement_task_brief_execution_plan(task: dict[str, Any]) -> dict[str, Any]:
    if task["status"] == "blocked":
        steps = [
            "Review the cadence blocker before any customer contact.",
            "Resolve the blocker or leave the customer paused.",
            "Regenerate the cadence review after the workflow state changes.",
        ]
        stop_condition = "Customer contact remains blocked until the cadence review passes."
    elif task["status"] == "review":
        steps = [
            "Open the reviewed workflow route and inspect the caution reason.",
            "Confirm the evidence supports customer contact.",
            "Escalate or proceed through the saved review workflow.",
        ]
        stop_condition = "Do not send outreach unless the reviewed workflow remains passing."
    else:
        steps = [
            "Open the reviewed workflow route before drafting customer-facing copy.",
            "Use the talking points and proof points attached to this brief.",
            "Record the customer response or workflow outcome after the action is completed.",
        ]
        stop_condition = "Stop if the route no longer returns a passing review."
    return {
        "objective": task["title"],
        "recommended_route": task.get("next_route"),
        "steps": steps,
        "stop_condition": stop_condition,
    }


def _engagement_task_brief_conversation_guide(
    engagement_brief: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    owner_id = task.get("owner_id") or engagement_brief.get("owner_id")
    segment = engagement_brief["current_intent"].get("segment") or "unknown"
    return {
        "opening": f"Start with {owner_id}'s latest saved intent segment: {segment}.",
        "talking_points": engagement_brief.get("talking_points", [])[:5],
        "proof_points": engagement_brief.get("evidence_references", [])[:5],
        "avoid": engagement_brief.get("avoid", []),
    }


def _engagement_task_brief_guardrails(
    task: dict[str, Any],
    cadence_review: dict[str, Any],
) -> list[str]:
    guardrails = list(task.get("guardrails") or [])
    if cadence_review["issues"]:
        guardrails.append("Resolve or explicitly review every cadence issue before customer-facing action.")
    guardrails.append("Record the resulting delivery outcome so the next action is grounded in saved evidence.")
    return guardrails


def _engagement_task_brief_completion(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "success_state": task.get("completion_criteria") or "Workflow state is updated after advisor action.",
        "record_route": {"method": "POST", "path": "/agents/advisor-outreach-deliveries/{delivery_id}/outcome"},
        "quality_check": "Next cadence review should show an updated latest event or resolved blocker.",
    }


def _customer_engagement_action_queue_markdown(queue: dict[str, Any]) -> str:
    summary = queue["summary"]
    lines = [
        "# Customer Engagement Action Queue",
        f"- Tasks: {summary['task_count']}",
        f"- Ready: {summary['ready_count']}",
        f"- Review: {summary['review_count']}",
        f"- Blocked: {summary['blocked_count']}",
        "",
        "## Top task",
        f"- {queue['top_task']['title']}",
        "",
        "## Tasks",
    ]
    for task in queue["tasks"]:
        lines.append(f"- {task['rank']}. {task['owner_id']} [{task['status']}]: {task['title']}")
    if not queue["tasks"]:
        lines.append("- None")
    return "\n".join(lines)


def _customer_engagement_task_brief_markdown(task_brief: dict[str, Any]) -> str:
    task = task_brief["task"]
    plan = task_brief["execution_plan"]
    lines = [
        "# Customer Engagement Task Brief",
        f"- Owner: {task_brief['owner_id'] or 'none'}",
        f"- Task: {task['title']}",
        f"- Status: {task['status']}",
        "",
        "## Execution Plan",
    ]
    lines.extend(f"- {step}" for step in plan["steps"])
    lines.extend(["", "## Talking Points"])
    talking_points = task_brief["conversation_guide"].get("talking_points") or []
    if talking_points:
        lines.extend(f"- {point}" for point in talking_points)
    else:
        lines.append("- None")
    lines.extend(["", "## Guardrails"])
    lines.extend(f"- {guardrail}" for guardrail in task_brief["compliance_guardrails"])
    return "\n".join(lines)


def _effectiveness_delivery_rows(
    conn: sqlite3.Connection,
    owner_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    params: list[Any] = []
    owner_clause = ""
    if owner_id:
        owner_clause = "WHERE d.owner_id = ?"
        params.append(owner_id)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT
            d.delivery_id,
            d.draft_id,
            d.owner_id,
            d.queue_id,
            d.task_id,
            d.status,
            d.source_task_json,
            d.delivered_at,
            d.created_at AS delivery_created_at,
            d.updated_at AS delivery_updated_at,
            o.outcome_id,
            o.outcome_type,
            o.customer_signal,
            o.next_action_json,
            o.follow_up_due_at,
            o.created_at AS outcome_created_at
        FROM advisor_outreach_delivery_records d
        LEFT JOIN advisor_outreach_delivery_outcomes o
            ON o.delivery_id = d.delivery_id
        {owner_clause}
        ORDER BY COALESCE(o.created_at, d.updated_at) DESC, d.delivery_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def _effectiveness_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    delivery_statuses = _effectiveness_delivery_statuses(rows)
    delivered_ids = {delivery_id for delivery_id, status in delivery_statuses.items() if status == "delivered"}
    captured_ids = {row["delivery_id"] for row in rows if row.get("outcome_id") is not None}
    outcome_rows = [row for row in rows if row.get("outcome_id") is not None]
    positive_count = sum(1 for row in outcome_rows if row["customer_signal"] == "positive")
    return {
        "delivery_count": len(delivery_statuses),
        "delivered_count": len(delivered_ids),
        "outcome_count": len(outcome_rows),
        "captured_delivery_count": len(captured_ids),
        "pending_outcome_count": len(delivered_ids - captured_ids),
        "positive_count": positive_count,
        "meeting_scheduled_count": sum(1 for row in outcome_rows if row["outcome_type"] == "meeting_scheduled"),
        "negative_count": sum(1 for row in outcome_rows if row["customer_signal"] == "negative"),
        "no_response_count": sum(1 for row in outcome_rows if row["customer_signal"] == "no_response"),
        "response_capture_rate": _safe_rate(len(captured_ids & delivered_ids), len(delivered_ids)),
        "positive_outcome_rate": _safe_rate(positive_count, len(outcome_rows)),
    }


def _effectiveness_by_task(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = row["task_id"]
        group = groups.setdefault(
            task_id,
            {
                "task_id": task_id,
                "title": _effectiveness_task_title(row),
                "owner_ids": set(),
                "delivery_ids": {},
                "captured_delivery_ids": set(),
                "outcome_count": 0,
                "positive_count": 0,
                "meeting_scheduled_count": 0,
                "negative_count": 0,
                "no_response_count": 0,
                "latest_outcome_at": None,
            },
        )
        group["owner_ids"].add(row["owner_id"])
        group["delivery_ids"][row["delivery_id"]] = row["status"]
        if row.get("outcome_id") is None:
            continue
        group["captured_delivery_ids"].add(row["delivery_id"])
        group["outcome_count"] += 1
        group["positive_count"] += 1 if row["customer_signal"] == "positive" else 0
        group["meeting_scheduled_count"] += 1 if row["outcome_type"] == "meeting_scheduled" else 0
        group["negative_count"] += 1 if row["customer_signal"] == "negative" else 0
        group["no_response_count"] += 1 if row["customer_signal"] == "no_response" else 0
        group["latest_outcome_at"] = max(group["latest_outcome_at"] or "", row["outcome_created_at"] or "")
    results = [_effectiveness_group_result(group) for group in groups.values()]
    results.sort(key=lambda item: (item["effectiveness_score"], item["outcome_count"], item["latest_outcome_at"] or ""), reverse=True)
    return results


def _effectiveness_group_result(group: dict[str, Any]) -> dict[str, Any]:
    delivered_ids = {delivery_id for delivery_id, status in group["delivery_ids"].items() if status == "delivered"}
    outcome_count = group["outcome_count"]
    score = (
        group["meeting_scheduled_count"] * 4
        + (group["positive_count"] - group["meeting_scheduled_count"]) * 3
        - group["negative_count"] * 3
        - group["no_response_count"]
    )
    return {
        "task_id": group["task_id"],
        "title": group["title"],
        "owner_count": len(group["owner_ids"]),
        "delivery_count": len(group["delivery_ids"]),
        "delivered_count": len(delivered_ids),
        "outcome_count": outcome_count,
        "pending_outcome_count": len(delivered_ids - group["captured_delivery_ids"]),
        "positive_count": group["positive_count"],
        "meeting_scheduled_count": group["meeting_scheduled_count"],
        "negative_count": group["negative_count"],
        "no_response_count": group["no_response_count"],
        "positive_outcome_rate": _safe_rate(group["positive_count"], outcome_count),
        "meeting_scheduled_rate": _safe_rate(group["meeting_scheduled_count"], outcome_count),
        "effectiveness_score": round(score / max(outcome_count, 1), 3),
        "latest_outcome_at": group["latest_outcome_at"] or None,
    }


def _effectiveness_recent_successes(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    successes: list[dict[str, Any]] = []
    for row in rows:
        if row.get("outcome_id") is None or row["customer_signal"] != "positive":
            continue
        successes.append(
            {
                "outcome_id": row["outcome_id"],
                "owner_id": row["owner_id"],
                "task_id": row["task_id"],
                "outcome_type": row["outcome_type"],
                "next_action": _safe_json(row.get("next_action_json"), {}),
                "created_at": row["outcome_created_at"],
            }
        )
    return successes[:limit]


def _effectiveness_learning_recommendations(
    rows: list[dict[str, Any]],
    by_task: list[dict[str, Any]],
) -> list[str]:
    summary = _effectiveness_summary(rows)
    recommendations: list[str] = []
    if summary["pending_outcome_count"]:
        recommendations.append("Capture missing delivery outcomes before trusting recommendation performance metrics.")
    if by_task:
        recommendations.append(f"Scale the highest-scoring task pattern: {by_task[0]['title']}.")
    if summary["no_response_count"] > summary["positive_count"]:
        recommendations.append("Revise outreach copy and cadence for recommendations producing more no-response outcomes than positive signals.")
    if not recommendations:
        recommendations.append("Generate and deliver reviewed outreach, then record outcomes to start the AI feedback loop.")
    return recommendations


def _effectiveness_delivery_statuses(rows: list[dict[str, Any]]) -> dict[int, str]:
    statuses: dict[int, str] = {}
    for row in rows:
        statuses[row["delivery_id"]] = row["status"]
    return statuses


def _effectiveness_task_title(row: dict[str, Any]) -> str:
    source_task = _safe_json(row.get("source_task_json"), {})
    return source_task.get("title") or row["task_id"]


def _empty_effectiveness_recommendation() -> dict[str, Any]:
    return {
        "task_id": None,
        "title": "Record outreach delivery outcomes",
        "owner_count": 0,
        "delivery_count": 0,
        "delivered_count": 0,
        "outcome_count": 0,
        "pending_outcome_count": 0,
        "positive_count": 0,
        "meeting_scheduled_count": 0,
        "negative_count": 0,
        "no_response_count": 0,
        "positive_outcome_rate": 0,
        "meeting_scheduled_rate": 0,
        "effectiveness_score": 0,
        "latest_outcome_at": None,
    }


def _ai_recommendation_effectiveness_markdown(dashboard: dict[str, Any]) -> str:
    summary = dashboard["summary"]
    lines = [
        "# AI Recommendation Effectiveness Dashboard",
        f"- Deliveries: {summary['delivery_count']}",
        f"- Outcomes: {summary['outcome_count']}",
        f"- Positive outcome rate: {summary['positive_outcome_rate']}",
        f"- Response capture rate: {summary['response_capture_rate']}",
        "",
        "## Top recommendation pattern",
        f"- {dashboard['top_recommendation']['title']}",
        "",
        "## Learning recommendations",
    ]
    lines.extend(f"- {item}" for item in dashboard["learning_recommendations"])
    return "\n".join(lines)


def _ai_improvement_backlog_items(effectiveness: dict[str, Any]) -> list[dict[str, Any]]:
    summary = effectiveness["summary"]
    owner_id = effectiveness.get("owner_id")
    owner_query = f"?owner_id={owner_id}" if owner_id else ""
    items: list[dict[str, Any]] = []
    if summary["pending_outcome_count"] > 0 or summary["response_capture_rate"] < 1:
        items.append(
            _ai_improvement_item(
                "capture_outcomes",
                "high",
                "Close the outcome-capture gap",
                "Recommendation quality cannot be trusted while delivered outreach is missing customer outcomes.",
                "Raise response_capture_rate to 1.0 for delivered outreach.",
                {"method": "GET", "path": f"/agents/advisor-outreach-delivery-dashboard{owner_query}"},
                {"pending_outcome_count": summary["pending_outcome_count"], "response_capture_rate": summary["response_capture_rate"]},
                ["Do not tune prompts from uncaptured delivery results."],
            )
        )
    top = effectiveness["top_recommendation"]
    if top.get("task_id") and top["outcome_count"] > 0 and top["effectiveness_score"] > 0:
        items.append(
            _ai_improvement_item(
                "scale_top_pattern",
                "high" if top["meeting_scheduled_count"] else "medium",
                "Scale the best-performing recommendation pattern",
                f"The measured top pattern is producing positive customer outcomes: {top['title']}.",
                "Keep positive_outcome_rate above 0.5 while increasing delivered_count.",
                {"method": "GET", "path": f"/agents/customer-engagement-action-queue{owner_query}"},
                {
                    "task_id": top["task_id"],
                    "positive_outcome_rate": top["positive_outcome_rate"],
                    "meeting_scheduled_rate": top["meeting_scheduled_rate"],
                    "effectiveness_score": top["effectiveness_score"],
                },
                ["Scale only through reviewed outreach and recorded outcomes."],
            )
        )
    if summary["no_response_count"] > summary["positive_count"]:
        items.append(
            _ai_improvement_item(
                "revise_low_response_copy",
                "medium",
                "Revise copy and cadence for low-response recommendations",
                "No-response outcomes exceed positive signals, so customer-facing copy or cadence likely needs adjustment.",
                "Reduce no_response_count below positive_count on the next measured batch.",
                {"method": "GET", "path": f"/agents/customer-engagement-task-brief{owner_query}"},
                {"no_response_count": summary["no_response_count"], "positive_count": summary["positive_count"]},
                ["Do not increase outreach frequency without cadence review passing."],
            )
        )
    if summary["outcome_count"] == 0:
        items.append(
            _ai_improvement_item(
                "start_feedback_loop",
                "high",
                "Start the AI feedback loop",
                "No saved customer outcomes exist yet, so AI improvements have no measured target.",
                "Record at least one delivered outreach outcome.",
                {"method": "GET", "path": f"/agents/customer-engagement-action-queue{owner_query}"},
                {"outcome_count": 0, "delivery_count": summary["delivery_count"]},
                ["Treat every recommendation as provisional until outcome evidence exists."],
            )
        )
    if not items:
        items.append(
            _ai_improvement_item(
                "monitor_effectiveness",
                "low",
                "Monitor recommendation effectiveness",
                "Current measured recommendation performance has no urgent gap.",
                "Maintain positive_outcome_rate and response_capture_rate on the next measured batch.",
                {"method": "GET", "path": f"/agents/ai-recommendation-effectiveness-dashboard{owner_query}"},
                summary,
                ["Keep reviewing outcome quality before changing ranking logic."],
            )
        )
    return _rank_ai_improvement_items(items)


def _ai_improvement_item(
    improvement_id: str,
    priority: str,
    title: str,
    rationale: str,
    success_metric: str,
    next_route: dict[str, Any],
    evidence: dict[str, Any],
    guardrails: list[str],
) -> dict[str, Any]:
    return {
        "improvement_id": improvement_id,
        "priority": priority,
        "title": title,
        "rationale": rationale,
        "success_metric": success_metric,
        "next_route": next_route,
        "evidence": evidence,
        "guardrails": [
            "Use saved local outcomes; do not infer customer reaction from generated copy alone.",
            *guardrails,
        ],
    }


def _rank_ai_improvement_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority_score = {"high": 3, "medium": 2, "low": 1}
    ranked = sorted(items, key=lambda item: priority_score.get(item["priority"], 0), reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked


def _ai_improvement_backlog_summary(
    items: list[dict[str, Any]],
    effectiveness_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "improvement_count": len(items),
        "high_priority_count": sum(1 for item in items if item["priority"] == "high"),
        "medium_priority_count": sum(1 for item in items if item["priority"] == "medium"),
        "low_priority_count": sum(1 for item in items if item["priority"] == "low"),
        "outcome_count": effectiveness_summary["outcome_count"],
        "positive_outcome_rate": effectiveness_summary["positive_outcome_rate"],
        "response_capture_rate": effectiveness_summary["response_capture_rate"],
    }


def _empty_ai_improvement_item() -> dict[str, Any]:
    return {
        "rank": None,
        "improvement_id": None,
        "priority": "low",
        "title": "No AI improvement available",
        "rationale": "No effectiveness evidence is available.",
        "success_metric": "Record customer outcomes.",
        "next_route": {"method": "GET", "path": "/agents/ai-recommendation-effectiveness-dashboard"},
        "evidence": {},
        "guardrails": ["Do not change recommendation logic without saved outcome evidence."],
    }


def _ai_improvement_backlog_markdown(backlog: dict[str, Any]) -> str:
    summary = backlog["summary"]
    lines = [
        "# AI Improvement Backlog",
        f"- Improvements: {summary['improvement_count']}",
        f"- High priority: {summary['high_priority_count']}",
        f"- Positive outcome rate: {summary['positive_outcome_rate']}",
        f"- Response capture rate: {summary['response_capture_rate']}",
        "",
        "## Single next improvement",
        f"- {backlog['single_next_improvement']['title']}",
        "",
        "## Backlog",
    ]
    lines.extend(
        f"- {item['rank']}. [{item['priority']}] {item['title']}"
        for item in backlog["improvements"]
    )
    return "\n".join(lines)


def _select_ai_improvement(
    improvements: list[dict[str, Any]],
    improvement_id: str | None,
) -> dict[str, Any]:
    if improvement_id:
        for improvement in improvements:
            if improvement["improvement_id"] == improvement_id:
                return improvement
        raise MarketRecordNotFound(f"No AI improvement backlog item found for id {improvement_id}")
    if improvements:
        return improvements[0]
    return _empty_ai_improvement_item()


def _ai_experiment_hypothesis(improvement: dict[str, Any]) -> str:
    return f"If Cerebral Insights implements '{improvement['title']}', then {improvement['success_metric']}"


def _ai_experiment_baseline(
    improvement: dict[str, Any],
    effectiveness_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "current_state": improvement["rationale"],
        "current_metrics": {
            "outcome_count": effectiveness_summary["outcome_count"],
            "positive_outcome_rate": effectiveness_summary["positive_outcome_rate"],
            "response_capture_rate": effectiveness_summary["response_capture_rate"],
        },
        "evidence": improvement["evidence"],
    }


def _ai_experiment_treatment(improvement: dict[str, Any]) -> dict[str, Any]:
    route = improvement.get("next_route") or {}
    return {
        "change": improvement["title"],
        "workflow_route": route,
        "implementation_scope": _ai_experiment_scope(improvement["improvement_id"]),
        "rollback": "Return to the current reviewed workflow if guardrails fail or metrics regress.",
    }


def _ai_experiment_scope(improvement_id: str | None) -> str:
    if improvement_id == "capture_outcomes":
        return "Outcome capture prompts, dashboard workflow, and advisor completion nudges."
    if improvement_id == "scale_top_pattern":
        return "Ranking, task brief wording, and surfaced next routes for the measured high-performing pattern."
    if improvement_id == "revise_low_response_copy":
        return "Customer-facing draft copy, cadence wording, and pre-send review checks for low-response patterns."
    if improvement_id == "start_feedback_loop":
        return "First-run delivery and outcome recording workflow."
    return "Monitoring and measurement workflow only."


def _ai_experiment_sample_criteria(improvement: dict[str, Any]) -> dict[str, Any]:
    return {
        "include": [
            "Only saved advisor outreach deliveries with reviewed approval evidence.",
            "Only customer outcomes recorded in local SQLite.",
            "Only tasks matching the selected improvement evidence when a task_id is present.",
        ],
        "exclude": [
            "Unsent drafts, unapproved copy, or generated recommendations without delivery records.",
            "Any customer marked blocked or paused by cadence review.",
        ],
        "minimum_sample": 5 if improvement["improvement_id"] != "start_feedback_loop" else 1,
    }


def _ai_experiment_success_metrics(
    improvement: dict[str, Any],
    effectiveness_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    metrics = [
        {
            "name": "primary_success_metric",
            "target": improvement["success_metric"],
            "current_value": improvement["evidence"],
        },
        {
            "name": "positive_outcome_rate",
            "target": "Improve or preserve positive outcome rate without increasing negative outcomes.",
            "current_value": effectiveness_summary["positive_outcome_rate"],
        },
        {
            "name": "response_capture_rate",
            "target": "Maintain response capture rate at or above the baseline.",
            "current_value": effectiveness_summary["response_capture_rate"],
        },
    ]
    return metrics


def _ai_experiment_stop_conditions(improvement: dict[str, Any]) -> list[str]:
    return [
        "Stop if cadence review blocks customer contact.",
        "Stop if compliance review blocks the generated outreach.",
        "Stop if negative customer signals increase on the measured batch.",
        *improvement.get("guardrails", []),
    ]


def _ai_improvement_experiment_markdown(plan: dict[str, Any]) -> str:
    improvement = plan["improvement"]
    lines = [
        "# AI Improvement Experiment Plan",
        f"- Improvement: {improvement['title']}",
        f"- Priority: {improvement['priority']}",
        f"- Hypothesis: {plan['hypothesis']}",
        "",
        "## Success Metrics",
    ]
    lines.extend(f"- {metric['name']}: {metric['target']}" for metric in plan["success_metrics"])
    lines.extend(["", "## Stop Conditions"])
    lines.extend(f"- {condition}" for condition in plan["stop_conditions"])
    return "\n".join(lines)


def _ai_experiment_launch_readiness(plan: dict[str, Any]) -> dict[str, Any]:
    checks = [
        _launch_check("hypothesis_defined", bool(plan.get("hypothesis")), "Experiment hypothesis is defined."),
        _launch_check("measurement_route_defined", bool(plan.get("measurement_route", {}).get("path")), "Measurement route is available."),
        _launch_check("treatment_route_defined", bool(plan.get("treatment", {}).get("workflow_route", {}).get("path")), "Treatment workflow route is available."),
        _launch_check("stop_conditions_defined", bool(plan.get("stop_conditions")), "Stop conditions are present."),
    ]
    current_outcomes = plan["baseline"]["current_metrics"]["outcome_count"]
    minimum_sample = plan["sample_criteria"]["minimum_sample"]
    if current_outcomes < minimum_sample:
        checks.append(
            {
                "check_id": "sample_size_target",
                "status": "warning",
                "message": f"Current outcomes ({current_outcomes}) are below target sample ({minimum_sample}); launch as a pilot only.",
            }
        )
    else:
        checks.append(_launch_check("sample_size_target", True, "Current outcome evidence meets the target sample."))
    blockers = [check for check in checks if check["status"] == "blocked"]
    return {
        "can_launch": not blockers,
        "status": "ready" if not blockers else "blocked",
        "checks": checks,
        "blockers": blockers,
    }


def _launch_check(check_id: str, passed: bool, message: str) -> dict[str, Any]:
    return {"check_id": check_id, "status": "passed" if passed else "blocked", "message": message}


def _ai_experiment_launch_cohort_assignment(plan: dict[str, Any]) -> dict[str, Any]:
    owner_id = plan.get("owner_id")
    return {
        "assignment_policy": "deterministic_local_holdout",
        "scope": "single_owner" if owner_id else "book_of_business",
        "owner_id": owner_id,
        "control": "Continue current reviewed outreach workflow for comparable eligible tasks.",
        "treatment": plan["treatment"]["change"],
        "minimum_sample": plan["sample_criteria"]["minimum_sample"],
        "eligibility": plan["sample_criteria"]["include"],
        "exclusions": plan["sample_criteria"]["exclude"],
    }


def _ai_experiment_launch_control(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": "current_reviewed_workflow",
        "description": "Use the current approved cadence, compliance review, delivery, and outcome capture path.",
        "measurement": plan["baseline"]["current_metrics"],
    }


def _ai_experiment_launch_checklist(plan: dict[str, Any]) -> list[dict[str, Any]]:
    route = plan["treatment"]["workflow_route"]
    return [
        {"step": 1, "task": "Review the experiment hypothesis and baseline metrics.", "route": None},
        {"step": 2, "task": "Confirm cadence and compliance review pass before any customer-facing action.", "route": route},
        {"step": 3, "task": "Apply the treatment only to eligible saved workflow items.", "route": route},
        {"step": 4, "task": "Record delivery outcomes immediately after customer response or follow-up expiry.", "route": {"method": "POST", "path": "/agents/advisor-outreach-deliveries/{delivery_id}/outcome"}},
        {"step": 5, "task": "Re-measure recommendation effectiveness after the pilot batch.", "route": plan["measurement_route"]},
    ]


def _ai_experiment_data_capture_requirements(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"field": "delivery_id", "required": True, "source": "advisor_outreach_delivery_records"},
        {"field": "draft_id", "required": True, "source": "advisor_outreach_drafts"},
        {"field": "task_id", "required": True, "source": "advisor_action_queue_tasks"},
        {"field": "outcome_type", "required": True, "source": "advisor_outreach_delivery_outcomes"},
        {"field": "customer_signal", "required": True, "source": "advisor_outreach_delivery_outcomes"},
        {"field": "follow_up_due_at", "required": False, "source": "advisor_outreach_delivery_outcomes"},
        {"field": "success_metrics", "required": True, "source": plan["measurement_route"]["path"]},
    ]


def _ai_experiment_rollback_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "rollback_route": plan["measurement_route"],
        "rollback_action": plan["treatment"]["rollback"],
        "trigger_conditions": plan["stop_conditions"],
        "post_rollback_check": "Regenerate the AI recommendation effectiveness dashboard and confirm no further treatment tasks are active.",
    }


def _ai_improvement_experiment_launch_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Experiment Launch Packet",
        f"- Experiment: {packet['experiment_id']}",
        f"- Launch status: {packet['readiness']['status']}",
        f"- Can launch: {packet['readiness']['can_launch']}",
        "",
        "## Checklist",
    ]
    lines.extend(f"- {item['step']}. {item['task']}" for item in packet["launch_checklist"])
    lines.extend(["", "## Data Capture"])
    lines.extend(f"- {item['field']} ({'required' if item['required'] else 'optional'})" for item in packet["data_capture_requirements"])
    return "\n".join(lines)


def _ai_experiment_readout_metrics(effectiveness: dict[str, Any]) -> dict[str, Any]:
    summary = effectiveness["summary"]
    top = effectiveness["top_recommendation"]
    return {
        "delivery_count": summary["delivery_count"],
        "outcome_count": summary["outcome_count"],
        "pending_outcome_count": summary["pending_outcome_count"],
        "positive_count": summary["positive_count"],
        "meeting_scheduled_count": summary["meeting_scheduled_count"],
        "negative_count": summary["negative_count"],
        "no_response_count": summary["no_response_count"],
        "positive_outcome_rate": summary["positive_outcome_rate"],
        "response_capture_rate": summary["response_capture_rate"],
        "top_effectiveness_score": top.get("effectiveness_score", 0),
        "top_task_id": top.get("task_id"),
    }


def _ai_experiment_readout_sample_status(
    launch: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    minimum_sample = launch["cohort_assignment"]["minimum_sample"]
    current_sample = metrics["outcome_count"]
    return {
        "minimum_sample": minimum_sample,
        "current_sample": current_sample,
        "remaining_sample": max(0, minimum_sample - current_sample),
        "target_met": current_sample >= minimum_sample,
    }


def _ai_experiment_readout_stop_results(
    launch: dict[str, Any],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline = launch["control"]["measurement"]
    results = [
        {
            "condition": "launch_readiness",
            "status": "clear" if launch["readiness"]["can_launch"] else "triggered",
            "evidence": {"readiness_status": launch["readiness"]["status"]},
        },
        {
            "condition": "negative_customer_signal",
            "status": "triggered" if metrics["negative_count"] > metrics["positive_count"] else "clear",
            "evidence": {"negative_count": metrics["negative_count"], "positive_count": metrics["positive_count"]},
        },
        {
            "condition": "response_capture_regression",
            "status": "triggered" if metrics["response_capture_rate"] < baseline["response_capture_rate"] else "clear",
            "evidence": {
                "current_response_capture_rate": metrics["response_capture_rate"],
                "baseline_response_capture_rate": baseline["response_capture_rate"],
            },
        },
        {
            "condition": "cadence_or_compliance_block",
            "status": "monitor",
            "evidence": {"route": launch["treatment"]["workflow_route"]},
        },
    ]
    return results


def _ai_experiment_readout_decision(
    launch: dict[str, Any],
    sample_status: dict[str, Any],
    stop_results: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    triggered = [result for result in stop_results if result["status"] == "triggered"]
    baseline = launch["control"]["measurement"]
    if triggered:
        return {
            "status": "rollback",
            "rationale": "One or more stop conditions were triggered.",
            "triggered_conditions": [result["condition"] for result in triggered],
        }
    if not sample_status["target_met"]:
        return {
            "status": "continue_collecting",
            "rationale": "Outcome sample is below the launch packet target; continue the pilot before deciding.",
            "triggered_conditions": [],
        }
    if (
        metrics["positive_outcome_rate"] >= baseline["positive_outcome_rate"]
        and metrics["response_capture_rate"] >= baseline["response_capture_rate"]
        and metrics["negative_count"] == 0
    ):
        return {
            "status": "ship",
            "rationale": "Measured outcomes meet or exceed baseline without negative signal regression.",
            "triggered_conditions": [],
        }
    return {
        "status": "continue",
        "rationale": "Sample target is met, but the treatment has not clearly beaten the baseline.",
        "triggered_conditions": [],
    }


def _ai_experiment_readout_next_route(
    launch: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    if decision["status"] == "rollback":
        return launch["rollback_plan"]["rollback_route"]
    if decision["status"] == "continue_collecting":
        return launch["treatment"]["workflow_route"]
    return launch["measurement_route"]


def _ai_improvement_experiment_readout_markdown(readout: dict[str, Any]) -> str:
    metrics = readout["metric_snapshot"]
    lines = [
        "# AI Improvement Experiment Readout",
        f"- Experiment: {readout['experiment_id']}",
        f"- Decision: {readout['decision']['status']}",
        f"- Rationale: {readout['decision']['rationale']}",
        f"- Outcomes: {metrics['outcome_count']}",
        f"- Positive outcome rate: {metrics['positive_outcome_rate']}",
        "",
        "## Stop Conditions",
    ]
    lines.extend(f"- {result['condition']}: {result['status']}" for result in readout["stop_condition_results"])
    return "\n".join(lines)


def _ai_rollout_release_gate(readout: dict[str, Any]) -> dict[str, Any]:
    decision = readout["decision"]["status"]
    if decision == "ship":
        return {
            "status": "ready_to_rollout",
            "can_rollout": True,
            "rationale": "Experiment readout met launch criteria and is eligible for controlled rollout.",
        }
    if decision == "rollback":
        return {
            "status": "blocked",
            "can_rollout": False,
            "rationale": "Experiment readout triggered rollback conditions.",
        }
    if decision == "continue_collecting":
        return {
            "status": "needs_more_evidence",
            "can_rollout": False,
            "rationale": "Outcome sample is below the launch target; keep the pilot running before rollout.",
        }
    return {
        "status": "needs_iteration",
        "can_rollout": False,
        "rationale": "Experiment evidence is not strong enough for rollout yet.",
    }


def _ai_rollout_customer_impact(readout: dict[str, Any]) -> dict[str, Any]:
    metrics = readout["metric_snapshot"]
    return {
        "expected_value": "Improve advisor recommendations using measured customer outcomes instead of unverified generated intent.",
        "current_positive_outcome_rate": metrics["positive_outcome_rate"],
        "current_response_capture_rate": metrics["response_capture_rate"],
        "meeting_scheduled_count": metrics["meeting_scheduled_count"],
        "risk_note": "Customer-facing rollout remains gated by cadence and compliance review.",
    }


def _ai_rollout_phases(
    readout: dict[str, Any],
    release_gate: dict[str, Any],
) -> list[dict[str, Any]]:
    if release_gate["status"] == "ready_to_rollout":
        return [
            {"phase": "limited_rollout", "status": "ready", "criteria": "Apply to eligible reviewed outreach tasks first."},
            {"phase": "monitor", "status": "pending", "criteria": "Watch outcome and stop-condition metrics after rollout."},
            {"phase": "expand", "status": "pending", "criteria": "Expand only if positive outcome rate and capture rate hold."},
        ]
    if release_gate["status"] == "needs_more_evidence":
        return [
            {"phase": "continue_pilot", "status": "active", "criteria": "Collect the remaining outcome sample before rollout."},
            {"phase": "re_readout", "status": "pending", "criteria": "Regenerate the experiment readout after more outcomes are captured."},
        ]
    if release_gate["status"] == "blocked":
        return [
            {"phase": "rollback", "status": "required", "criteria": "Stop treatment use and return to the current reviewed workflow."},
            {"phase": "diagnose", "status": "pending", "criteria": "Review triggered stop conditions before any relaunch."},
        ]
    return [
        {"phase": "iterate", "status": "active", "criteria": "Revise the treatment and run another measured pilot."},
        {"phase": "re_readout", "status": "pending", "criteria": "Regenerate readout after treatment changes and new outcomes."},
    ]


def _ai_rollout_monitoring_plan(readout: dict[str, Any]) -> dict[str, Any]:
    return {
        "measurement_route": readout["recommended_next_route"],
        "metrics": [
            {"name": "positive_outcome_rate", "current_value": readout["metric_snapshot"]["positive_outcome_rate"]},
            {"name": "response_capture_rate", "current_value": readout["metric_snapshot"]["response_capture_rate"]},
            {"name": "negative_count", "current_value": readout["metric_snapshot"]["negative_count"]},
            {"name": "sample_target_met", "current_value": readout["sample_status"]["target_met"]},
        ],
        "cadence": "Regenerate after every pilot batch or any negative customer signal.",
    }


def _ai_rollout_rollback_triggers(readout: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "trigger": result["condition"],
            "status": result["status"],
            "evidence": result["evidence"],
        }
        for result in readout["stop_condition_results"]
        if result["status"] in {"triggered", "monitor"}
    ]


def _ai_rollout_approval_checklist(
    readout: dict[str, Any],
    release_gate: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {"check": "release_gate", "passed": release_gate["can_rollout"], "detail": release_gate["rationale"]},
        {"check": "sample_target_met", "passed": readout["sample_status"]["target_met"], "detail": f"{readout['sample_status']['current_sample']} outcomes captured."},
        {"check": "no_stop_conditions_triggered", "passed": not readout["decision"]["triggered_conditions"], "detail": ", ".join(readout["decision"]["triggered_conditions"]) or "No triggered stop conditions."},
        {"check": "measurement_route_available", "passed": bool(readout["recommended_next_route"].get("path")), "detail": readout["recommended_next_route"].get("path")},
    ]


def _ai_rollout_next_route(
    readout: dict[str, Any],
    release_gate: dict[str, Any],
) -> dict[str, Any]:
    if release_gate["status"] == "ready_to_rollout":
        return {"method": "GET", "path": "/agents/ai-improvement-experiment-readout", "purpose": "Monitor rollout after release."}
    return readout["recommended_next_route"]


def _ai_improvement_rollout_readiness_markdown(readiness: dict[str, Any]) -> str:
    gate = readiness["release_gate"]
    lines = [
        "# AI Improvement Rollout Readiness",
        f"- Experiment: {readiness['experiment_id']}",
        f"- Gate: {gate['status']}",
        f"- Can rollout: {gate['can_rollout']}",
        f"- Rationale: {gate['rationale']}",
        "",
        "## Rollout Phases",
    ]
    lines.extend(f"- {phase['phase']}: {phase['status']}" for phase in readiness["rollout_phases"])
    lines.extend(["", "## Approval Checklist"])
    lines.extend(f"- {item['check']}: {item['passed']}" for item in readiness["approval_checklist"])
    return "\n".join(lines)


def _ai_rollout_monitor_alerts(readiness: dict[str, Any]) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    gate = readiness["release_gate"]
    if gate["status"] == "blocked":
        alerts.append({"severity": "critical", "code": "rollout_blocked", "message": gate["rationale"]})
    if gate["status"] == "needs_more_evidence":
        alerts.append({"severity": "medium", "code": "needs_more_evidence", "message": gate["rationale"]})
    for trigger in readiness["rollback_triggers"]:
        if trigger["status"] == "triggered":
            alerts.append({"severity": "critical", "code": trigger["trigger"], "message": "Rollback trigger is active.", "evidence": trigger["evidence"]})
        elif trigger["status"] == "monitor":
            alerts.append({"severity": "low", "code": trigger["trigger"], "message": "Monitor this rollback trigger during rollout.", "evidence": trigger["evidence"]})
    for check in readiness["approval_checklist"]:
        if not check["passed"]:
            alerts.append({"severity": "medium", "code": check["check"], "message": check["detail"]})
    return alerts


def _ai_rollout_monitor_status(
    readiness: dict[str, Any],
    alerts: list[dict[str, Any]],
) -> str:
    if any(alert["severity"] == "critical" for alert in alerts):
        return "rollback_required"
    if readiness["release_gate"]["can_rollout"]:
        return "monitoring_ready"
    if readiness["release_gate"]["status"] == "needs_more_evidence":
        return "pilot_monitoring"
    return "attention_required"


def _ai_rollout_monitor_risk_level(
    readiness: dict[str, Any],
    alerts: list[dict[str, Any]],
) -> str:
    if any(alert["severity"] == "critical" for alert in alerts):
        return "high"
    if readiness["release_gate"]["status"] in {"needs_more_evidence", "needs_iteration"}:
        return "medium"
    if alerts:
        return "low"
    return "low"


def _ai_rollout_monitor_metrics(readiness: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = readiness["monitoring_plan"]["metrics"]
    return [
        {
            **metric,
            "status": _ai_rollout_metric_status(metric),
        }
        for metric in metrics
    ]


def _ai_rollout_metric_status(metric: dict[str, Any]) -> str:
    name = metric["name"]
    value = metric["current_value"]
    if name == "negative_count" and value:
        return "alert"
    if name == "sample_target_met" and not value:
        return "collecting"
    if name in {"positive_outcome_rate", "response_capture_rate"} and value == 0:
        return "needs_data"
    return "ok"


def _ai_rollout_monitor_next_check(readiness: dict[str, Any]) -> dict[str, Any]:
    phase = readiness["rollout_phases"][0] if readiness["rollout_phases"] else {}
    return {
        "cadence": readiness["monitoring_plan"]["cadence"],
        "phase": phase.get("phase"),
        "route": readiness["monitoring_plan"]["measurement_route"],
    }


def _ai_rollout_monitor_immediate_action(
    readiness: dict[str, Any],
    alerts: list[dict[str, Any]],
) -> dict[str, Any]:
    if any(alert["severity"] == "critical" for alert in alerts):
        return {
            "action": "rollback",
            "route": readiness["recommended_next_route"],
            "rationale": "Critical rollout alert is active.",
        }
    if readiness["release_gate"]["status"] == "needs_more_evidence":
        return {
            "action": "continue_pilot",
            "route": readiness["recommended_next_route"],
            "rationale": "Collect more outcomes before rollout.",
        }
    if readiness["release_gate"]["can_rollout"]:
        return {
            "action": "monitor_rollout",
            "route": readiness["recommended_next_route"],
            "rationale": "Release gate is open; monitor after rollout.",
        }
    return {
        "action": "iterate_before_rollout",
        "route": readiness["recommended_next_route"],
        "rationale": readiness["release_gate"]["rationale"],
    }


def _ai_improvement_rollout_monitor_markdown(monitor: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Rollout Monitor",
        f"- Experiment: {monitor['experiment_id']}",
        f"- Status: {monitor['status']}",
        f"- Risk level: {monitor['risk_level']}",
        f"- Immediate action: {monitor['immediate_action']['action']}",
        "",
        "## Alerts",
    ]
    if monitor["alerts"]:
        lines.extend(f"- [{alert['severity']}] {alert['code']}: {alert['message']}" for alert in monitor["alerts"])
    else:
        lines.append("- None")
    lines.extend(["", "## Metrics"])
    lines.extend(f"- {metric['name']}: {metric['current_value']} ({metric['status']})" for metric in monitor["tracked_metrics"])
    return "\n".join(lines)


def _ai_release_packet_status(monitor: dict[str, Any]) -> str:
    if monitor["status"] == "rollback_required":
        return "rollback_notice"
    if monitor["source_release_gate"].get("can_rollout"):
        return "ready_to_announce"
    if monitor["status"] == "pilot_monitoring":
        return "pilot_update"
    return "hold"


def _ai_release_customer_value_summary(monitor: dict[str, Any]) -> dict[str, Any]:
    return {
        "headline": "Cerebral Insights is improving AI recommendations using measured customer outcomes.",
        "current_status": monitor["status"],
        "risk_level": monitor["risk_level"],
        "primary_value": "Advisors get recommendations that are tied to observed customer responses, not just generated intent.",
        "evidence": {metric["name"]: metric["current_value"] for metric in monitor["tracked_metrics"]},
    }


def _ai_release_eligibility(monitor: dict[str, Any]) -> dict[str, Any]:
    phase = monitor.get("source_rollout_phase") or {}
    return {
        "owner_id": monitor["owner_id"],
        "phase": phase.get("phase"),
        "phase_status": phase.get("status"),
        "criteria": phase.get("criteria"),
        "eligible": monitor["source_release_gate"].get("can_rollout", False),
        "gate_status": monitor["source_release_gate"].get("status"),
    }


def _ai_release_advisor_enablement(monitor: dict[str, Any]) -> list[dict[str, Any]]:
    action = monitor["immediate_action"]
    return [
        {
            "topic": "What changed",
            "message": "AI improvements are being evaluated against saved outreach outcomes and rollout guardrails.",
        },
        {
            "topic": "What to do now",
            "message": action["rationale"],
            "route": action["route"],
        },
        {
            "topic": "What to record",
            "message": "Keep recording delivery outcomes so the improvement can be measured and safely promoted or rolled back.",
        },
    ]


def _ai_release_support_talking_points(monitor: dict[str, Any]) -> list[str]:
    points = [
        "The improvement is measured against local customer outcome evidence.",
        "Customer-facing recommendations remain gated by cadence and compliance checks.",
    ]
    if monitor["status"] == "pilot_monitoring":
        points.append("The improvement is still in pilot monitoring and needs more captured outcomes before broad rollout.")
    if monitor["status"] == "rollback_required":
        points.append("The improvement has triggered a rollback path and should not be expanded.")
    return points


def _ai_release_known_risks(monitor: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "risk": alert["code"],
            "severity": alert["severity"],
            "mitigation": alert["message"],
        }
        for alert in monitor["alerts"]
    ]


def _ai_release_rollback_guidance(monitor: dict[str, Any]) -> dict[str, Any]:
    if monitor["status"] == "rollback_required":
        action = "Rollback immediately using the recommended route."
    elif monitor["risk_level"] == "medium":
        action = "Do not expand rollout until the monitor clears the evidence gap."
    else:
        action = "Continue monitoring after release and rollback if a critical alert appears."
    return {
        "action": action,
        "route": monitor["immediate_action"]["route"],
        "trigger_summary": [alert["code"] for alert in monitor["alerts"] if alert["severity"] in {"critical", "medium"}],
    }


def _ai_improvement_release_packet_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Release Packet",
        f"- Experiment: {packet['experiment_id']}",
        f"- Release status: {packet['release_status']}",
        f"- Monitor status: {packet['source_monitor_status']}",
        "",
        "## Customer Value",
        f"- {packet['customer_value_summary']['primary_value']}",
        "",
        "## Advisor Enablement",
    ]
    lines.extend(f"- {item['topic']}: {item['message']}" for item in packet["advisor_enablement"])
    lines.extend(["", "## Known Risks"])
    if packet["known_risks"]:
        lines.extend(f"- [{risk['severity']}] {risk['risk']}: {risk['mitigation']}" for risk in packet["known_risks"])
    else:
        lines.append("- None")
    return "\n".join(lines)


def _ai_adoption_status(release: dict[str, Any]) -> dict[str, Any]:
    release_status = release["release_status"]
    if release_status == "ready_to_announce":
        return {"status": "ready_to_enable", "rationale": "Release gate is open and advisors can be enabled."}
    if release_status == "pilot_update":
        return {"status": "pilot_enablement", "rationale": "Enable only pilot users while more evidence is collected."}
    if release_status == "rollback_notice":
        return {"status": "pause_enablement", "rationale": "Rollback guidance is active; do not train broader users."}
    return {"status": "hold", "rationale": "Release packet is not ready for advisor adoption."}


def _ai_adoption_role_tasks(release: dict[str, Any]) -> list[dict[str, Any]]:
    next_route = _ai_release_next_route(release)
    return [
        {
            "role": "advisor",
            "task": "Use the eligible AI improvement only inside the reviewed workflow and record every customer outcome.",
            "route": next_route,
        },
        {
            "role": "manager",
            "task": "Review adoption blockers and confirm advisors do not expand beyond the current eligibility gate.",
            "route": release["rollback_guidance"]["route"],
        },
        {
            "role": "support",
            "task": "Use the support talking points and escalate any rollback trigger or customer confusion.",
            "route": release["rollback_guidance"]["route"],
        },
    ]


def _ai_adoption_training_checklist(release: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"check": "understand_customer_value", "required": True, "detail": release["customer_value_summary"]["primary_value"]},
        {"check": "confirm_eligibility", "required": True, "detail": release["eligibility"]["gate_status"]},
        {"check": "review_known_risks", "required": True, "detail": f"{len(release['known_risks'])} risks listed."},
        {"check": "record_outcomes", "required": True, "detail": "Every delivered outreach needs outcome capture."},
        {"check": "know_rollback_path", "required": True, "detail": release["rollback_guidance"]["action"]},
    ]


def _ai_adoption_customer_language(release: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": release["customer_value_summary"]["headline"],
        "talking_points": release["support_talking_points"],
        "avoid": [
            "Do not claim the AI improvement is broadly released when the gate is not open.",
            "Do not describe generated recommendations as final advice without review.",
        ],
    }


def _ai_adoption_blockers(release: dict[str, Any]) -> list[dict[str, Any]]:
    blockers = list(release["known_risks"])
    if not release["eligibility"]["eligible"]:
        blockers.append(
            {
                "risk": "not_eligible_for_broad_adoption",
                "severity": "medium",
                "mitigation": release["eligibility"].get("criteria") or release["rollback_guidance"]["action"],
            }
        )
    return blockers


def _ai_adoption_success_signals(release: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = release["customer_value_summary"].get("evidence", {})
    return [
        {"signal": "positive_outcome_rate", "current_value": evidence.get("positive_outcome_rate")},
        {"signal": "response_capture_rate", "current_value": evidence.get("response_capture_rate")},
        {"signal": "sample_target_met", "current_value": evidence.get("sample_target_met")},
    ]


def _ai_adoption_next_action(release: dict[str, Any]) -> dict[str, Any]:
    if release["release_status"] == "rollback_notice":
        return {"action": "pause_adoption", "route": release["rollback_guidance"]["route"], "rationale": release["rollback_guidance"]["action"]}
    if release["release_status"] == "pilot_update":
        return {"action": "train_pilot_advisors", "route": _ai_release_next_route(release), "rationale": "Keep enablement scoped to the pilot while collecting outcomes."}
    if release["release_status"] == "ready_to_announce":
        return {"action": "enable_advisors", "route": _ai_release_next_route(release), "rationale": "Release gate is open for advisor enablement."}
    return {"action": "hold_enablement", "route": release["rollback_guidance"]["route"], "rationale": "Wait for rollout monitor to clear before adoption."}


def _ai_release_next_route(release: dict[str, Any]) -> dict[str, Any]:
    for item in release["advisor_enablement"]:
        route = item.get("route")
        if route:
            return route
    return release["rollback_guidance"]["route"]


def _ai_improvement_adoption_playbook_markdown(playbook: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Adoption Playbook",
        f"- Experiment: {playbook['experiment_id']}",
        f"- Adoption status: {playbook['adoption_status']['status']}",
        f"- Next action: {playbook['next_action']['action']}",
        "",
        "## Role Tasks",
    ]
    lines.extend(f"- {item['role']}: {item['task']}" for item in playbook["role_tasks"])
    lines.extend(["", "## Training Checklist"])
    lines.extend(f"- {item['check']}: {item['detail']}" for item in playbook["training_checklist"])
    return "\n".join(lines)


def _ai_adoption_monitor_status(playbook: dict[str, Any], blockers: list[dict[str, Any]]) -> str:
    if playbook["next_action"]["action"] == "pause_adoption":
        return "paused"
    if any(blocker.get("severity") == "critical" for blocker in blockers):
        return "blocked"
    adoption_status = playbook["adoption_status"]["status"]
    if adoption_status == "pilot_enablement":
        return "pilot_training"
    if adoption_status == "ready_to_enable":
        return "ready"
    return "hold"


def _ai_adoption_monitor_risk_level(blockers: list[dict[str, Any]]) -> str:
    severities = {blocker.get("severity") for blocker in blockers}
    if "critical" in severities or "high" in severities:
        return "high"
    if "medium" in severities:
        return "medium"
    return "low"


def _ai_adoption_training_status(training: list[dict[str, Any]]) -> dict[str, Any]:
    required_checks = [item for item in training if item.get("required")]
    complete_checks = [item for item in required_checks if item.get("completed")]
    return {
        "status": "ready" if required_checks and len(complete_checks) == len(required_checks) else "pending",
        "required_count": len(required_checks),
        "complete_count": len(complete_checks),
        "checklist": training,
    }


def _ai_adoption_customer_language_status(playbook: dict[str, Any]) -> dict[str, Any]:
    customer_language = playbook["customer_language"]
    return {
        "status": "ready" if playbook["source_eligibility"]["eligible"] else "review_required",
        "talking_point_count": len(customer_language.get("talking_points", [])),
        "avoid_count": len(customer_language.get("avoid", [])),
        "safe_to_use": playbook["source_eligibility"]["eligible"],
    }


def _ai_improvement_adoption_monitor_markdown(monitor: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Adoption Monitor",
        f"- Experiment: {monitor['experiment_id']}",
        f"- Status: {monitor['status']}",
        f"- Risk level: {monitor['risk_level']}",
        f"- Training: {monitor['training_status']['status']}",
        f"- Customer language: {monitor['customer_language_status']['status']}",
        f"- Immediate action: {monitor['immediate_action']['action']}",
        "",
        "## Blockers",
    ]
    lines.extend(f"- {item['risk']}: {item.get('mitigation')}" for item in monitor["blockers"])
    lines.extend(["", "## Success Signals"])
    lines.extend(f"- {item['signal']}: {item.get('current_value')}" for item in monitor["success_signals"])
    return "\n".join(lines)


def _ai_adoption_impact_value_status(monitor: dict[str, Any], metrics: dict[str, Any]) -> str:
    if monitor["status"] in {"blocked", "paused"}:
        return "blocked"
    if metrics["outcome_count"] <= 0:
        return "unmeasured"
    if metrics["positive_count"] > 0 or metrics["meeting_scheduled_count"] > 0:
        return "proving_value"
    return "needs_more_signal"


def _ai_adoption_impact_customer_impact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "outcome_count": metrics["outcome_count"],
        "positive_count": metrics["positive_count"],
        "meeting_scheduled_count": metrics["meeting_scheduled_count"],
        "negative_count": metrics["negative_count"],
        "no_response_count": metrics["no_response_count"],
        "positive_outcome_rate": metrics["positive_outcome_rate"],
    }


def _ai_adoption_impact_advisor_usage(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "delivery_count": metrics["delivery_count"],
        "pending_outcome_count": metrics["pending_outcome_count"],
        "response_capture_rate": metrics["response_capture_rate"],
        "top_task_id": metrics["top_task_id"],
    }


def _ai_adoption_impact_scale_decision(monitor: dict[str, Any], readout: dict[str, Any]) -> dict[str, Any]:
    if monitor["status"] in {"blocked", "paused"}:
        return {"action": "do_not_scale", "rationale": "Adoption monitor is blocked or paused."}
    if monitor["risk_level"] == "high":
        return {"action": "keep_blocked", "rationale": "High adoption risk needs resolution before expansion."}
    if readout["decision"]["status"] == "ship" and monitor["status"] == "ready":
        return {"action": "scale_adoption", "rationale": "Readout and adoption monitor both support expansion."}
    return {"action": "keep_pilot", "rationale": "Keep adoption scoped until more outcome evidence and training completion are available."}


def _ai_adoption_impact_blocked_accounts(monitor: dict[str, Any]) -> list[dict[str, Any]]:
    if monitor["status"] == "ready" and monitor["risk_level"] == "low":
        return []
    return [
        {
            "owner_id": monitor["owner_id"],
            "risk": blocker["risk"],
            "severity": blocker.get("severity", "unknown"),
            "mitigation": blocker.get("mitigation"),
        }
        for blocker in monitor["blockers"]
    ]


def _ai_adoption_impact_proof_points(metrics: dict[str, Any], monitor: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"proof": "captured_outcomes", "value": metrics["outcome_count"]},
        {"proof": "positive_customer_signals", "value": metrics["positive_count"]},
        {"proof": "meetings_scheduled", "value": metrics["meeting_scheduled_count"]},
        {"proof": "response_capture_rate", "value": metrics["response_capture_rate"]},
        {"proof": "adoption_risk_level", "value": monitor["risk_level"]},
    ]


def _ai_improvement_adoption_impact_ledger_markdown(ledger: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Adoption Impact Ledger",
        f"- Experiment: {ledger['experiment_id']}",
        f"- Value status: {ledger['value_status']}",
        f"- Scale action: {ledger['scale_decision']['action']}",
        f"- Next action: {ledger['next_action']['action']}",
        "",
        "## Customer Impact",
        f"- Positive outcomes: {ledger['customer_impact']['positive_count']}",
        f"- Meetings scheduled: {ledger['customer_impact']['meeting_scheduled_count']}",
        "",
        "## Advisor Usage",
        f"- Deliveries: {ledger['advisor_usage']['delivery_count']}",
        f"- Response capture rate: {ledger['advisor_usage']['response_capture_rate']}",
    ]
    return "\n".join(lines)


def _ai_scale_packet_decision(ledger: dict[str, Any]) -> dict[str, Any]:
    scale_action = ledger["scale_decision"]["action"]
    if scale_action == "scale_adoption":
        return {"status": "scale", "rationale": "Measured value and adoption readiness support expansion."}
    if scale_action == "keep_pilot" and ledger["value_status"] == "proving_value":
        return {"status": "continue_pilot", "rationale": "Customer value is visible, but rollout blockers remain."}
    if scale_action in {"do_not_scale", "keep_blocked"}:
        return {"status": "hold", "rationale": ledger["scale_decision"]["rationale"]}
    return {"status": "collect_more_evidence", "rationale": "Value signal is not yet strong enough for a scale decision."}


def _ai_scale_packet_executive_summary(ledger: dict[str, Any], decision: dict[str, Any]) -> str:
    impact = ledger["customer_impact"]
    return (
        f"{decision['status']} for {ledger['experiment_id']}: "
        f"{impact['positive_count']} positive outcomes and {impact['meeting_scheduled_count']} meetings are recorded; "
        f"current risk is {ledger['source_risk_level']}."
    )


def _ai_scale_packet_customer_value_evidence(ledger: dict[str, Any]) -> dict[str, Any]:
    impact = ledger["customer_impact"]
    return {
        "evidence_strength": "directional" if ledger["value_status"] == "proving_value" else ledger["value_status"],
        "positive_outcome_rate": impact["positive_outcome_rate"],
        "positive_count": impact["positive_count"],
        "meeting_scheduled_count": impact["meeting_scheduled_count"],
        "negative_count": impact["negative_count"],
        "proof_points": ledger["proof_points"],
    }


def _ai_scale_packet_advisor_change_plan(ledger: dict[str, Any], decision: dict[str, Any]) -> list[dict[str, Any]]:
    if decision["status"] == "scale":
        return [
            {"role": "advisor", "change": "Use the improved AI workflow for eligible reviewed outreach tasks."},
            {"role": "manager", "change": "Monitor adoption proof points and exceptions daily during expansion."},
        ]
    if decision["status"] == "continue_pilot":
        return [
            {"role": "advisor", "change": "Keep using the improvement only with pilot advisors and capture every outcome."},
            {"role": "manager", "change": "Resolve rollout blockers before broad enablement."},
            {"role": "support", "change": "Use reviewed customer language and escalate confusion or risk signals."},
        ]
    return [
        {"role": "advisor", "change": "Pause expansion and continue the current reviewed workflow."},
        {"role": "manager", "change": "Review blockers and decide whether to revise, relaunch, or stop the improvement."},
    ]


def _ai_scale_packet_blocker_resolution_plan(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "blocker": account["risk"],
            "severity": account["severity"],
            "resolution": account.get("mitigation") or "Resolve before expanding adoption.",
        }
        for account in ledger["blocked_accounts"]
    ]


def _ai_scale_packet_rollout_scope(ledger: dict[str, Any]) -> dict[str, Any]:
    action = ledger["scale_decision"]["action"]
    if action == "scale_adoption":
        return {"scope": "eligible_accounts", "guardrail": "Expand only where adoption monitor remains ready."}
    if action == "keep_pilot":
        return {"scope": "pilot_only", "guardrail": "Do not broaden until blockers and training are cleared."}
    return {"scope": "paused", "guardrail": "No expansion until the scale decision changes."}


def _ai_scale_packet_next_action(ledger: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    if decision["status"] == "continue_pilot":
        return ledger["next_action"]
    if decision["status"] == "scale":
        return {"action": "expand_eligible_adoption", "rationale": decision["rationale"]}
    return {"action": "review_scale_blockers", "rationale": decision["rationale"]}


def _ai_improvement_scale_decision_packet_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Scale Decision Packet",
        f"- Experiment: {packet['experiment_id']}",
        f"- Decision: {packet['decision']['status']}",
        f"- Rollout scope: {packet['rollout_scope']['scope']}",
        f"- Next action: {packet['next_action']['action']}",
        "",
        "## Customer Value Evidence",
        f"- Evidence strength: {packet['customer_value_evidence']['evidence_strength']}",
        f"- Positive outcomes: {packet['customer_value_evidence']['positive_count']}",
        f"- Meetings scheduled: {packet['customer_value_evidence']['meeting_scheduled_count']}",
        "",
        "## Advisor Change Plan",
    ]
    lines.extend(f"- {item['role']}: {item['change']}" for item in packet["advisor_change_plan"])
    return "\n".join(lines)


def _ai_scale_execution_status(packet: dict[str, Any]) -> str:
    decision = packet["decision"]["status"]
    if decision == "scale":
        return "ready_to_execute"
    if decision == "continue_pilot":
        return "pilot_execution"
    if decision == "collect_more_evidence":
        return "evidence_collection"
    return "blocked"


def _ai_scale_execution_tasks(packet: dict[str, Any]) -> list[dict[str, Any]]:
    decision = packet["decision"]["status"]
    if decision == "scale":
        return [
            {"owner": "manager", "action": "open_eligible_rollout", "status": "pending", "detail": packet["rollout_scope"]["guardrail"]},
            {"owner": "advisor", "action": "use_scaled_workflow", "status": "pending", "detail": "Use the improved workflow only for eligible reviewed outreach tasks."},
            {"owner": "support", "action": "monitor_customer_confusion", "status": "pending", "detail": "Escalate customer confusion or rollback triggers immediately."},
        ]
    if decision == "continue_pilot":
        return [
            {"owner": "manager", "action": "resolve_scale_blockers", "status": "pending", "detail": "Clear blocker resolution plan before broad enablement."},
            {"owner": "advisor", "action": "capture_pilot_outcomes", "status": "pending", "detail": "Capture every pilot outreach outcome before re-reading the ledger."},
            {"owner": "support", "action": "review_customer_language", "status": "pending", "detail": "Keep customer-facing language inside the reviewed support guidance."},
        ]
    return [
        {"owner": "manager", "action": "review_scale_blockers", "status": "pending", "detail": packet["decision"]["rationale"]},
        {"owner": "advisor", "action": "pause_expansion", "status": "pending", "detail": "Continue the current reviewed workflow until the decision changes."},
    ]


def _ai_scale_execution_guardrails(packet: dict[str, Any]) -> list[dict[str, Any]]:
    guardrails = [{"guardrail": "rollout_scope", "status": "active", "detail": packet["rollout_scope"]["guardrail"]}]
    if packet["rollout_scope"]["scope"] != "eligible_accounts":
        guardrails.append({"guardrail": "no_broad_enablement", "status": "active", "detail": "Do not broaden adoption until the scale packet changes."})
    if packet["blocker_resolution_plan"]:
        guardrails.append({"guardrail": "blockers_first", "status": "active", "detail": "Resolve listed blockers before expansion."})
    return guardrails


def _ai_scale_execution_customer_proof_checks(packet: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = packet["customer_value_evidence"]
    return [
        {"check": "positive_outcomes_present", "status": "met" if evidence["positive_count"] > 0 else "missing", "value": evidence["positive_count"]},
        {"check": "meetings_scheduled_present", "status": "met" if evidence["meeting_scheduled_count"] > 0 else "missing", "value": evidence["meeting_scheduled_count"]},
        {"check": "no_negative_regression", "status": "met" if evidence["negative_count"] == 0 else "review", "value": evidence["negative_count"]},
    ]


def _ai_scale_execution_acceptance_criteria(packet: dict[str, Any]) -> list[dict[str, Any]]:
    blockers = packet["blocker_resolution_plan"]
    return [
        {"criterion": "decision_scope_followed", "required": True, "target": packet["rollout_scope"]["scope"]},
        {"criterion": "all_blockers_resolved", "required": True, "target": len(blockers) == 0},
        {"criterion": "customer_proof_preserved", "required": True, "target": packet["customer_value_evidence"]["evidence_strength"]},
        {"criterion": "advisor_change_plan_acknowledged", "required": True, "target": len(packet["advisor_change_plan"])},
    ]


def _ai_scale_execution_escalation_path(packet: dict[str, Any]) -> dict[str, Any]:
    if packet["blocker_resolution_plan"]:
        return {
            "owner": "manager",
            "trigger": "Any blocker remains unresolved before expansion.",
            "action": "Keep rollout scoped and regenerate the scale decision packet after blocker review.",
        }
    return {
        "owner": "support",
        "trigger": "Customer confusion, negative outcome, or response-capture drop.",
        "action": "Escalate to manager and pause expansion until reviewed.",
    }


def _ai_scale_execution_next_action(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    task = tasks[0]
    return {"owner": task["owner"], "action": task["action"], "detail": task["detail"]}


def _ai_improvement_scale_execution_plan_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Scale Execution Plan",
        f"- Experiment: {plan['experiment_id']}",
        f"- Execution status: {plan['execution_status']}",
        f"- Rollout scope: {plan['rollout_scope']['scope']}",
        f"- Next action: {plan['next_action']['action']}",
        "",
        "## Execution Tasks",
    ]
    lines.extend(f"- {item['owner']}: {item['action']} - {item['detail']}" for item in plan["execution_tasks"])
    lines.extend(["", "## Guardrails"])
    lines.extend(f"- {item['guardrail']}: {item['detail']}" for item in plan["guardrails"])
    return "\n".join(lines)


def _ai_scale_execution_task_progress(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    complete_count = sum(1 for task in tasks if task.get("status") == "complete")
    pending_count = sum(1 for task in tasks if task.get("status") != "complete")
    return {
        "status": "complete" if tasks and complete_count == len(tasks) else "pending",
        "complete_count": complete_count,
        "pending_count": pending_count,
        "total_count": len(tasks),
        "tasks": tasks,
    }


def _ai_scale_execution_guardrail_status(guardrails: list[dict[str, Any]]) -> dict[str, Any]:
    active_count = sum(1 for guardrail in guardrails if guardrail.get("status") == "active")
    return {
        "status": "active" if active_count else "clear",
        "active_count": active_count,
        "guardrails": guardrails,
    }


def _ai_scale_execution_proof_status(checks: list[dict[str, Any]]) -> dict[str, Any]:
    unmet = [check for check in checks if check.get("status") != "met"]
    return {
        "status": "met" if not unmet else "needs_review",
        "met_count": len(checks) - len(unmet),
        "unmet_count": len(unmet),
        "checks": checks,
    }


def _ai_scale_execution_acceptance_status(criteria: list[dict[str, Any]]) -> dict[str, Any]:
    assessed = []
    missing_required_count = 0
    for criterion in criteria:
        met = bool(criterion.get("target"))
        status = "met" if met else "blocked"
        if criterion.get("required") and not met:
            missing_required_count += 1
        assessed.append({**criterion, "status": status})
    return {
        "status": "met" if missing_required_count == 0 else "blocked",
        "missing_required_count": missing_required_count,
        "criteria": assessed,
    }


def _ai_scale_execution_monitor_blockers(
    plan: dict[str, Any],
    proof_status: dict[str, Any],
    acceptance_status: dict[str, Any],
) -> list[dict[str, Any]]:
    blockers = [
        {
            "blocker": item["criterion"],
            "severity": "medium",
            "resolution": "Complete this acceptance criterion before expansion.",
        }
        for item in acceptance_status["criteria"]
        if item["status"] != "met"
    ]
    if proof_status["status"] != "met":
        blockers.append(
            {
                "blocker": "customer_proof_incomplete",
                "severity": "high",
                "resolution": "Capture clean customer proof before changing rollout scope.",
            }
        )
    if plan["source_monitor_status"] in {"blocked", "paused"}:
        blockers.append(
            {
                "blocker": "source_monitor_not_clear",
                "severity": "high",
                "resolution": "Resolve the upstream adoption monitor state before execution continues.",
            }
        )
    return blockers


def _ai_scale_execution_monitor_status(
    plan: dict[str, Any],
    task_progress: dict[str, Any],
    proof_status: dict[str, Any],
    acceptance_status: dict[str, Any],
) -> str:
    if plan["execution_status"] == "blocked" or acceptance_status["status"] == "blocked":
        return "blocked"
    if proof_status["status"] != "met":
        return "needs_customer_proof"
    if plan["rollout_scope"]["scope"] == "eligible_accounts" and task_progress["status"] == "complete":
        return "ready_to_expand"
    return "in_progress"


def _ai_scale_execution_monitor_risk_level(
    proof_status: dict[str, Any],
    acceptance_status: dict[str, Any],
    blockers: list[dict[str, Any]],
) -> str:
    if proof_status["status"] != "met" or any(blocker["severity"] == "high" for blocker in blockers):
        return "high"
    if acceptance_status["status"] == "blocked" or blockers:
        return "medium"
    return "low"


def _ai_improvement_scale_execution_monitor_markdown(monitor: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Scale Execution Monitor",
        f"- Experiment: {monitor['experiment_id']}",
        f"- Status: {monitor['status']}",
        f"- Risk level: {monitor['risk_level']}",
        f"- Pending tasks: {monitor['task_progress']['pending_count']}",
        f"- Acceptance: {monitor['acceptance_status']['status']}",
        f"- Immediate action: {monitor['immediate_action']['action']}",
        "",
        "## Blockers",
    ]
    lines.extend(f"- {item['blocker']}: {item['resolution']}" for item in monitor["blockers"])
    return "\n".join(lines)


def _ai_scale_learning_status(monitor: dict[str, Any]) -> str:
    if monitor["status"] == "ready_to_expand":
        return "validated_for_expansion"
    if monitor["status"] == "blocked" and monitor["customer_proof_status"]["status"] == "met":
        return "blocked_but_value_visible"
    if monitor["status"] == "needs_customer_proof":
        return "evidence_gap"
    return "active_learning"


def _ai_scale_validated_learnings(monitor: dict[str, Any]) -> list[dict[str, Any]]:
    learnings = [
        {
            "learning": "customer_proof_status",
            "status": monitor["customer_proof_status"]["status"],
            "evidence": monitor["customer_proof_status"]["checks"],
        },
        {
            "learning": "execution_risk_level",
            "status": monitor["risk_level"],
            "evidence": monitor["blockers"],
        },
    ]
    if monitor["guardrail_status"]["active_count"]:
        learnings.append(
            {
                "learning": "guardrails_are_active",
                "status": "active",
                "evidence": monitor["guardrail_status"]["guardrails"],
            }
        )
    return learnings


def _ai_scale_open_questions(monitor: dict[str, Any]) -> list[dict[str, Any]]:
    questions = [
        {
            "question": f"How should we resolve {blocker['blocker']}?",
            "owner": monitor["immediate_action"]["owner"],
            "evidence_needed": blocker["resolution"],
        }
        for blocker in monitor["blockers"]
    ]
    if monitor["task_progress"]["pending_count"]:
        questions.append(
            {
                "question": "Which execution tasks are still pending?",
                "owner": monitor["immediate_action"]["owner"],
                "evidence_needed": "Complete or update pending execution tasks before the next scale decision.",
            }
        )
    return questions


def _ai_scale_feedback_actions(monitor: dict[str, Any]) -> list[dict[str, Any]]:
    actions = [
        {
            "owner": monitor["immediate_action"]["owner"],
            "action": monitor["immediate_action"]["action"],
            "detail": monitor["immediate_action"]["detail"],
        }
    ]
    if monitor["customer_proof_status"]["status"] != "met":
        actions.append(
            {
                "owner": "advisor",
                "action": "capture_more_customer_proof",
                "detail": "Record more customer outcomes before revisiting scale.",
            }
        )
    if monitor["acceptance_status"]["status"] == "blocked":
        actions.append(
            {
                "owner": "manager",
                "action": "update_acceptance_gaps",
                "detail": "Turn blocked acceptance criteria into the next AI improvement backlog input.",
            }
        )
    return actions


def _ai_scale_next_improvement_candidate(monitor: dict[str, Any]) -> dict[str, Any]:
    if monitor["blockers"]:
        blocker = monitor["blockers"][0]
        return {
            "candidate": "clear_scale_blocker",
            "priority": "high" if blocker["severity"] == "high" else "medium",
            "rationale": blocker["resolution"],
        }
    if monitor["status"] == "ready_to_expand":
        return {
            "candidate": "expand_eligible_adoption",
            "priority": "medium",
            "rationale": "Execution monitor is ready for broader eligible rollout.",
        }
    return {
        "candidate": "collect_execution_evidence",
        "priority": "medium",
        "rationale": "More task and customer-proof evidence is needed before the next scale decision.",
    }


def _ai_scale_learning_roadmap_update(monitor: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "feed_back_to_backlog" if monitor["blockers"] else "ready_for_next_scale_review",
        "reason": monitor["status"],
        "risk_level": monitor["risk_level"],
    }


def _ai_improvement_scale_learning_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Scale Learning Report",
        f"- Experiment: {report['experiment_id']}",
        f"- Learning status: {report['learning_status']}",
        f"- Next candidate: {report['next_improvement_candidate']['candidate']}",
        f"- Roadmap update: {report['roadmap_update']['status']}",
        "",
        "## Feedback Actions",
    ]
    lines.extend(f"- {item['owner']}: {item['action']} - {item['detail']}" for item in report["feedback_actions"])
    lines.extend(["", "## Open Questions"])
    lines.extend(f"- {item['question']}" for item in report["open_questions"])
    return "\n".join(lines)


def _ai_roadmap_refresh_status(report: dict[str, Any]) -> str:
    if report["roadmap_update"]["status"] == "feed_back_to_backlog":
        return "backlog_ready"
    if report["learning_status"] == "validated_for_expansion":
        return "ready_for_scale_review"
    return "needs_more_learning"


def _ai_roadmap_refresh_item(report: dict[str, Any]) -> dict[str, Any]:
    candidate = report["next_improvement_candidate"]
    return {
        "item_id": f"roadmap:{candidate['candidate']}",
        "candidate": candidate["candidate"],
        "priority": candidate["priority"],
        "rationale": candidate["rationale"],
        "customer_value": _ai_roadmap_customer_value(report),
        "source_experiment_id": report["experiment_id"],
    }


def _ai_roadmap_customer_value(report: dict[str, Any]) -> str:
    if report["learning_status"] == "blocked_but_value_visible":
        return "Preserve visible customer value while removing the blocker that prevents safe scale."
    if report["learning_status"] == "validated_for_expansion":
        return "Expand a validated AI improvement to more eligible customer workflows."
    return "Collect enough evidence to identify the next customer-value AI improvement."


def _ai_roadmap_owner_action_plan(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "owner": item["owner"],
            "action": item["action"],
            "detail": item["detail"],
            "status": "pending",
        }
        for item in report["feedback_actions"]
    ]


def _ai_roadmap_evidence_package(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "learning_status": report["learning_status"],
        "risk_level": report["source_risk_level"],
        "validated_learning_count": len(report["validated_learnings"]),
        "open_question_count": len(report["open_questions"]),
        "validated_learnings": report["validated_learnings"],
        "open_questions": report["open_questions"],
    }


def _ai_roadmap_acceptance_gates(report: dict[str, Any]) -> list[dict[str, Any]]:
    proof_learning = next((item for item in report["validated_learnings"] if item["learning"] == "customer_proof_status"), {})
    return [
        {
            "gate": "customer_proof_preserved",
            "status": "met" if proof_learning.get("status") == "met" else "pending",
            "required": True,
        },
        {
            "gate": "open_questions_resolved",
            "status": "met" if not report["open_questions"] else "pending",
            "required": True,
        },
        {
            "gate": "feedback_actions_assigned",
            "status": "met" if report["feedback_actions"] else "pending",
            "required": True,
        },
    ]


def _ai_roadmap_measurement_plan(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"metric": "validated_learning_count", "current_value": len(report["validated_learnings"]), "target": "increase_or_hold"},
        {"metric": "open_question_count", "current_value": len(report["open_questions"]), "target": 0},
        {"metric": "feedback_action_count", "current_value": len(report["feedback_actions"]), "target": "all_assigned"},
        {"metric": "source_risk_level", "current_value": report["source_risk_level"], "target": "low"},
    ]


def _ai_roadmap_sequencing(report: dict[str, Any]) -> dict[str, Any]:
    candidate = report["next_improvement_candidate"]
    if candidate["priority"] == "high":
        return {"sequence": "next", "rationale": "High-priority learning should be addressed before the next scale cycle."}
    return {"sequence": "current_cycle", "rationale": "Address this candidate inside the current AI improvement cycle."}


def _ai_roadmap_next_action(report: dict[str, Any]) -> dict[str, Any]:
    if report["feedback_actions"]:
        return report["feedback_actions"][0]
    candidate = report["next_improvement_candidate"]
    return {"owner": "manager", "action": candidate["candidate"], "detail": candidate["rationale"]}


def _ai_improvement_roadmap_refresh_markdown(refresh: dict[str, Any]) -> str:
    item = refresh["roadmap_item"]
    lines = [
        "# AI Improvement Roadmap Refresh",
        f"- Experiment: {refresh['experiment_id']}",
        f"- Roadmap status: {refresh['roadmap_status']}",
        f"- Candidate: {item['candidate']}",
        f"- Priority: {item['priority']}",
        f"- Next action: {refresh['next_action']['action']}",
        "",
        "## Acceptance Gates",
    ]
    lines.extend(f"- {item['gate']}: {item['status']}" for item in refresh["acceptance_gates"])
    lines.extend(["", "## Measurement Plan"])
    lines.extend(f"- {item['metric']}: {item['current_value']} -> {item['target']}" for item in refresh["measurement_plan"])
    return "\n".join(lines)


def _ai_backlog_handoff_status(refresh: dict[str, Any]) -> str:
    if refresh["roadmap_status"] == "backlog_ready":
        return "ready_for_backlog"
    if refresh["roadmap_status"] == "ready_for_scale_review":
        return "ready_for_review"
    return "needs_more_learning"


def _ai_backlog_handoff_work_item(refresh: dict[str, Any]) -> dict[str, Any]:
    item = refresh["roadmap_item"]
    return {
        "work_item_id": item["item_id"].replace("roadmap:", "backlog:"),
        "title": item["candidate"].replace("_", " ").title(),
        "priority": item["priority"],
        "owner": refresh["next_action"].get("owner", "manager"),
        "source_roadmap_item": item["item_id"],
    }


def _ai_backlog_handoff_story(refresh: dict[str, Any]) -> dict[str, Any]:
    item = refresh["roadmap_item"]
    return {
        "as_a": "Cerebral Insights operator",
        "i_want": f"to implement {item['candidate'].replace('_', ' ')}",
        "so_that": item["customer_value"],
        "rationale": item["rationale"],
    }


def _ai_backlog_handoff_scope(refresh: dict[str, Any]) -> dict[str, Any]:
    return {
        "in_scope": [
            refresh["roadmap_item"]["candidate"],
            "owner action follow-through",
            "acceptance gate closure",
            "measurement plan verification",
        ],
        "out_of_scope": [
            "broad AI rollout before acceptance gates pass",
            "new customer-facing claims without proof checks",
        ],
    }


def _ai_backlog_handoff_dependencies(refresh: dict[str, Any]) -> list[dict[str, Any]]:
    dependencies = [
        {
            "dependency": gate["gate"],
            "status": gate["status"],
            "required": gate["required"],
        }
        for gate in refresh["acceptance_gates"]
        if gate["status"] != "met"
    ]
    if not dependencies:
        dependencies.append({"dependency": "scale_review", "status": "ready", "required": True})
    return dependencies


def _ai_backlog_handoff_launch_readiness(refresh: dict[str, Any]) -> dict[str, Any]:
    blocked = [gate for gate in refresh["acceptance_gates"] if gate["required"] and gate["status"] != "met"]
    return {
        "status": "ready" if not blocked else "blocked",
        "blocked_gate_count": len(blocked),
        "blocked_gates": [gate["gate"] for gate in blocked],
    }


def _ai_improvement_backlog_handoff_markdown(handoff: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Backlog Handoff",
        f"- Experiment: {handoff['experiment_id']}",
        f"- Handoff status: {handoff['handoff_status']}",
        f"- Work item: {handoff['work_item']['title']}",
        f"- Priority: {handoff['work_item']['priority']}",
        f"- Launch readiness: {handoff['launch_readiness']['status']}",
        f"- Next action: {handoff['next_action']['action']}",
        "",
        "## Acceptance Gates",
    ]
    lines.extend(f"- {item['gate']}: {item['status']}" for item in handoff["acceptance_gates"])
    lines.extend(["", "## Dependencies"])
    lines.extend(f"- {item['dependency']}: {item['status']}" for item in handoff["dependencies"])
    return "\n".join(lines)


def _ai_implementation_kickoff_status(handoff: dict[str, Any]) -> str:
    if handoff["handoff_status"] != "ready_for_backlog":
        return "needs_product_review"
    if handoff["launch_readiness"]["status"] == "blocked":
        return "ready_for_blocker_work"
    return "ready_to_kickoff"


def _ai_implementation_kickoff_engineering_scope(handoff: dict[str, Any]) -> dict[str, Any]:
    return {
        "components": [
            "ai_improvement_decision_chain",
            "agent_api_route",
            "discovery_and_capability_docs",
            "regression_tests",
        ],
        "in_scope": handoff["implementation_scope"]["in_scope"],
        "out_of_scope": handoff["implementation_scope"]["out_of_scope"],
    }


def _ai_implementation_kickoff_qa_gates(handoff: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"gate": "compile_system_and_tests", "required": True, "status": "pending"},
        {"gate": "focused_action_queue_api_suite", "required": True, "status": "pending"},
        {"gate": "full_unittest_suite", "required": True, "status": "pending"},
        {
            "gate": "launch_readiness_unblocked",
            "required": True,
            "status": handoff["launch_readiness"]["status"],
        },
    ]


def _ai_implementation_kickoff_data_contracts(handoff: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "contract": "processed_payloads_only",
            "status": "active",
            "detail": "Use processed outcome, roadmap, and measurement payloads; keep raw provider data out of implementation handoff.",
        },
        {
            "contract": "source_experiment_trace",
            "status": "active",
            "detail": handoff["experiment_id"],
        },
        {
            "contract": "acceptance_gate_trace",
            "status": "active",
            "detail": f"{len(handoff['acceptance_gates'])} gates linked from roadmap refresh.",
        },
    ]


def _ai_implementation_kickoff_customer_value_guardrails(handoff: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "guardrail": "preserve_customer_value",
            "status": "active",
            "detail": handoff["implementation_story"]["so_that"],
        },
        {
            "guardrail": "no_broad_rollout_until_ready",
            "status": handoff["launch_readiness"]["status"],
            "detail": "Do not broaden AI adoption until launch readiness is unblocked.",
        },
        {
            "guardrail": "measure_before_claiming_value",
            "status": "active",
            "detail": "Use the measurement plan before describing the improvement as scaled.",
        },
    ]


def _ai_implementation_kickoff_launch_checklist(handoff: dict[str, Any]) -> list[dict[str, Any]]:
    checklist = [
        {
            "item": gate["gate"],
            "status": gate["status"],
            "required": gate["required"],
        }
        for gate in handoff["acceptance_gates"]
    ]
    checklist.extend(
        {
            "item": dependency["dependency"],
            "status": dependency["status"],
            "required": dependency["required"],
        }
        for dependency in handoff["dependencies"]
    )
    return checklist


def _ai_improvement_implementation_kickoff_packet_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Implementation Kickoff Packet",
        f"- Experiment: {packet['experiment_id']}",
        f"- Kickoff status: {packet['kickoff_status']}",
        f"- Work item: {packet['work_item']['title']}",
        f"- Owner: {packet['work_item']['owner']}",
        f"- Immediate action: {packet['immediate_action']['action']}",
        "",
        "## QA Gates",
    ]
    lines.extend(f"- {item['gate']}: {item['status']}" for item in packet["qa_gates"])
    lines.extend(["", "## Launch Checklist"])
    lines.extend(f"- {item['item']}: {item['status']}" for item in packet["launch_checklist"])
    return "\n".join(lines)


def _ai_implementation_readiness_qa_status(packet: dict[str, Any]) -> dict[str, Any]:
    blocked = [gate for gate in packet["qa_gates"] if gate.get("required") and gate.get("status") not in {"ready", "met"}]
    return {
        "status": "ready" if not blocked else "blocked",
        "blocked_count": len(blocked),
        "gates": packet["qa_gates"],
    }


def _ai_implementation_readiness_data_contract_status(packet: dict[str, Any]) -> dict[str, Any]:
    inactive = [contract for contract in packet["data_contracts"] if contract.get("status") != "active"]
    return {
        "status": "active" if not inactive else "needs_review",
        "inactive_count": len(inactive),
        "contracts": packet["data_contracts"],
    }


def _ai_implementation_readiness_guardrail_status(packet: dict[str, Any]) -> dict[str, Any]:
    blocked = [guardrail for guardrail in packet["customer_value_guardrails"] if guardrail.get("status") == "blocked"]
    return {
        "status": "clear" if not blocked else "blocked",
        "blocked_count": len(blocked),
        "guardrails": packet["customer_value_guardrails"],
    }


def _ai_implementation_readiness_checklist_status(packet: dict[str, Any]) -> dict[str, Any]:
    pending = [item for item in packet["launch_checklist"] if item.get("required") and item.get("status") not in {"met", "ready"}]
    return {
        "status": "ready" if not pending else "blocked",
        "pending_count": len(pending),
        "items": packet["launch_checklist"],
    }


def _ai_implementation_readiness_blockers(
    qa_status: dict[str, Any],
    data_contract_status: dict[str, Any],
    guardrail_status: dict[str, Any],
    checklist_status: dict[str, Any],
) -> list[dict[str, Any]]:
    blockers = []
    if qa_status["status"] != "ready":
        blockers.append({"blocker": "qa_gates_not_ready", "severity": "medium", "resolution": "Complete required QA gates."})
    if data_contract_status["status"] != "active":
        blockers.append({"blocker": "data_contracts_not_active", "severity": "high", "resolution": "Resolve inactive data contracts before implementation proceeds."})
    if guardrail_status["status"] != "clear":
        blockers.append({"blocker": "customer_guardrails_blocked", "severity": "high", "resolution": "Clear customer-value guardrails before launch."})
    if checklist_status["status"] != "ready":
        blockers.append({"blocker": "launch_checklist_blocked", "severity": "medium", "resolution": "Complete required launch checklist items."})
    return blockers


def _ai_implementation_readiness_status(packet: dict[str, Any], blockers: list[dict[str, Any]]) -> str:
    if not blockers and packet["kickoff_status"] == "ready_to_kickoff":
        return "launch_ready"
    if packet["kickoff_status"] == "ready_for_blocker_work":
        return "blocked"
    if blockers:
        return "needs_work"
    return "ready_for_qa"


def _ai_implementation_readiness_risk_level(blockers: list[dict[str, Any]]) -> str:
    if any(blocker["severity"] == "high" for blocker in blockers):
        return "high"
    if blockers:
        return "medium"
    return "low"


def _ai_improvement_implementation_readiness_monitor_markdown(monitor: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Implementation Readiness Monitor",
        f"- Experiment: {monitor['experiment_id']}",
        f"- Status: {monitor['status']}",
        f"- Risk level: {monitor['risk_level']}",
        f"- Work item: {monitor['work_item']['title']}",
        f"- Immediate action: {monitor['immediate_action']['action']}",
        "",
        "## Blockers",
    ]
    lines.extend(f"- {item['blocker']}: {item['resolution']}" for item in monitor["blockers"])
    return "\n".join(lines)


def _ai_implementation_blocker_resolution_status(monitor: dict[str, Any]) -> str:
    if not monitor["blockers"]:
        return "ready_to_verify"
    if monitor["risk_level"] == "high":
        return "blocked_resolution_required"
    return "resolution_in_progress"


def _ai_implementation_blocker_resolution_tasks(monitor: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "owner": _ai_implementation_blocker_owner(blocker),
            "action": _ai_implementation_blocker_action(blocker),
            "status": "pending",
            "blocker": blocker["blocker"],
            "severity": blocker["severity"],
            "resolution": blocker["resolution"],
        }
        for blocker in monitor["blockers"]
    ]


def _ai_implementation_blocker_owner(blocker: dict[str, Any]) -> str:
    if blocker["blocker"] == "qa_gates_not_ready":
        return "qa"
    if blocker["blocker"] == "data_contracts_not_active":
        return "engineering"
    return "manager"


def _ai_implementation_blocker_action(blocker: dict[str, Any]) -> str:
    actions = {
        "qa_gates_not_ready": "complete_required_qa_gates",
        "data_contracts_not_active": "activate_data_contracts",
        "customer_guardrails_blocked": "clear_customer_value_guardrails",
        "launch_checklist_blocked": "complete_launch_checklist",
    }
    return actions.get(blocker["blocker"], "resolve_implementation_blocker")


def _ai_implementation_blocker_proof_required(monitor: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"proof": "qa_gate_evidence", "status": monitor["qa_status"]["status"], "required": True},
        {"proof": "data_contract_evidence", "status": monitor["data_contract_status"]["status"], "required": True},
        {"proof": "customer_guardrail_evidence", "status": monitor["customer_guardrail_status"]["status"], "required": True},
        {"proof": "launch_checklist_evidence", "status": monitor["launch_checklist_status"]["status"], "required": True},
    ]


def _ai_implementation_blocker_exit_criteria(monitor: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"criterion": "readiness_status_unblocked", "target": "ready_for_qa_or_launch_ready", "required": True},
        {"criterion": "risk_level_reduced", "target": "low", "current_value": monitor["risk_level"], "required": True},
        {"criterion": "all_blockers_closed", "target": 0, "current_value": len(monitor["blockers"]), "required": True},
    ]


def _ai_implementation_blocker_qa_rerun_plan(monitor: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"command": "python3 -m compileall -q system tests", "reason": "Verify implementation modules still compile."},
        {"command": "python3 -m unittest discover -s tests -p 'test_action_queue_api.py'", "reason": "Rerun focused AI improvement route coverage."},
        {"command": "python3 -m unittest discover -s tests", "reason": "Confirm no broader API regression after blocker fixes."},
    ]


def _ai_implementation_blocker_customer_guardrail_clearance(monitor: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": monitor["customer_guardrail_status"]["status"],
        "blocked_count": monitor["customer_guardrail_status"]["blocked_count"],
        "required_action": "Clear customer-value guardrails before any launch or broad rollout.",
    }


def _ai_implementation_blocker_immediate_action(
    resolution_tasks: list[dict[str, Any]],
    monitor: dict[str, Any],
) -> dict[str, Any]:
    high = next((task for task in resolution_tasks if task["severity"] == "high"), None)
    task = high or (resolution_tasks[0] if resolution_tasks else None)
    if not task:
        return monitor["immediate_action"]
    return {"owner": task["owner"], "action": task["action"], "detail": task["resolution"]}


def _ai_improvement_implementation_blocker_resolution_plan_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Implementation Blocker Resolution Plan",
        f"- Experiment: {plan['experiment_id']}",
        f"- Resolution status: {plan['resolution_status']}",
        f"- Work item: {plan['work_item']['title']}",
        f"- Immediate action: {plan['immediate_unblock_action']['action']}",
        "",
        "## Resolution Tasks",
    ]
    lines.extend(f"- {item['owner']}: {item['action']} - {item['resolution']}" for item in plan["resolution_tasks"])
    lines.extend(["", "## Exit Criteria"])
    lines.extend(f"- {item['criterion']}: {item['target']}" for item in plan["exit_criteria"])
    return "\n".join(lines)


def _ai_implementation_unblock_task_status(plan: dict[str, Any]) -> dict[str, Any]:
    complete_count = sum(1 for task in plan["resolution_tasks"] if task.get("status") == "complete")
    pending_count = len(plan["resolution_tasks"]) - complete_count
    return {
        "status": "complete" if plan["resolution_tasks"] and pending_count == 0 else "pending",
        "complete_count": complete_count,
        "pending_count": pending_count,
        "tasks": plan["resolution_tasks"],
    }


def _ai_implementation_unblock_proof_status(plan: dict[str, Any]) -> dict[str, Any]:
    passing = {"active", "clear", "met", "ready", "complete"}
    missing = [item for item in plan["proof_required"] if item.get("status") not in passing]
    return {
        "status": "verified" if not missing else "blocked",
        "verified_count": len(plan["proof_required"]) - len(missing),
        "missing_count": len(missing),
        "proof": plan["proof_required"],
    }


def _ai_implementation_unblock_exit_status(plan: dict[str, Any]) -> dict[str, Any]:
    assessed = []
    failed_count = 0
    for item in plan["exit_criteria"]:
        met = _ai_implementation_unblock_exit_criterion_met(item, plan)
        if item.get("required") and not met:
            failed_count += 1
        assessed.append({**item, "status": "met" if met else "blocked"})
    return {
        "status": "met" if failed_count == 0 else "blocked",
        "failed_count": failed_count,
        "criteria": assessed,
    }


def _ai_implementation_unblock_exit_criterion_met(item: dict[str, Any], plan: dict[str, Any]) -> bool:
    target = item.get("target")
    current = item.get("current_value")
    if item["criterion"] == "readiness_status_unblocked":
        return plan["source_readiness_status"] in {"ready_for_qa", "launch_ready"}
    if item["criterion"] == "risk_level_reduced":
        return current == target
    if item["criterion"] == "all_blockers_closed":
        return current == target
    return current == target


def _ai_implementation_unblock_qa_rerun_status(plan: dict[str, Any]) -> dict[str, Any]:
    completed_count = sum(1 for item in plan["qa_rerun_plan"] if item.get("status") == "passed")
    pending_count = len(plan["qa_rerun_plan"]) - completed_count
    return {
        "status": "passed" if plan["qa_rerun_plan"] and pending_count == 0 else "pending",
        "completed_count": completed_count,
        "pending_count": pending_count,
        "commands": plan["qa_rerun_plan"],
    }


def _ai_implementation_unblock_ready(
    task_status: dict[str, Any],
    proof_status: dict[str, Any],
    exit_status: dict[str, Any],
    qa_rerun_status: dict[str, Any],
) -> bool:
    return (
        task_status["status"] == "complete"
        and proof_status["status"] == "verified"
        and exit_status["status"] == "met"
        and qa_rerun_status["status"] == "passed"
    )


def _ai_implementation_unblock_verification_status(plan: dict[str, Any], ready_to_proceed: bool) -> str:
    if ready_to_proceed:
        return "verified"
    if plan["source_risk_level"] == "high":
        return "blocked"
    return "needs_evidence"


def _ai_implementation_unblock_next_action(plan: dict[str, Any], ready_to_proceed: bool) -> dict[str, Any]:
    if ready_to_proceed:
        return {"owner": "manager", "action": "move_to_qa_or_launch_review", "detail": "All unblock proof has been verified."}
    return plan["immediate_unblock_action"]


def _ai_improvement_implementation_unblock_verification_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Implementation Unblock Verification Report",
        f"- Experiment: {report['experiment_id']}",
        f"- Verification status: {report['verification_status']}",
        f"- Ready to proceed: {report['ready_to_proceed']}",
        f"- Work item: {report['work_item']['title']}",
        f"- Next action: {report['next_verification_action']['action']}",
        "",
        "## Exit Criteria",
    ]
    lines.extend(f"- {item['criterion']}: {item['status']}" for item in report["exit_criteria_status"]["criteria"])
    lines.extend(["", "## Proof Required"])
    lines.extend(f"- {item['proof']}: {item['status']}" for item in report["proof_status"]["proof"])
    return "\n".join(lines)


def _ai_implementation_qa_review_decision(verification: dict[str, Any]) -> dict[str, Any]:
    if verification["ready_to_proceed"]:
        return {"status": "ready_for_qa", "rationale": "Unblock verification passed all required checks."}
    if verification["verification_status"] == "blocked":
        return {"status": "hold_qa", "rationale": "Implementation unblock verification is still blocked."}
    return {"status": "collect_more_evidence", "rationale": "Verification needs more evidence before QA starts."}


def _ai_implementation_qa_review_scope(verification: dict[str, Any]) -> dict[str, Any]:
    return {
        "include": [
            "focused action queue API route coverage",
            "full unittest regression",
            "customer guardrail verification",
            "unblock evidence review",
        ],
        "exclude": [
            "launch approval while verification is blocked",
            "new customer-facing AI claims without guardrail signoff",
        ],
        "work_item_id": verification["work_item"]["work_item_id"],
    }


def _ai_implementation_qa_evidence_gaps(verification: dict[str, Any]) -> list[dict[str, Any]]:
    gaps = []
    for proof in verification["proof_status"]["proof"]:
        if proof["status"] not in {"verified", "active", "clear", "met", "ready", "complete"}:
            gaps.append({"gap": proof["proof"], "status": proof["status"], "required": proof["required"]})
    for criterion in verification["exit_criteria_status"]["criteria"]:
        if criterion["status"] != "met":
            gaps.append({"gap": criterion["criterion"], "status": criterion["status"], "required": criterion["required"]})
    if verification["qa_rerun_status"]["status"] != "passed":
        gaps.append({"gap": "qa_reruns_not_passed", "status": verification["qa_rerun_status"]["status"], "required": True})
    return gaps


def _ai_implementation_qa_test_gates(verification: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "gate": command["command"],
            "status": "pending" if verification["qa_rerun_status"]["status"] != "passed" else "ready",
            "reason": command["reason"],
        }
        for command in verification["qa_rerun_status"]["commands"]
    ]


def _ai_implementation_qa_customer_guardrails(verification: dict[str, Any]) -> dict[str, Any]:
    guardrail = verification["customer_guardrail_status"]
    return {
        "status": guardrail["status"],
        "blocked_count": guardrail["blocked_count"],
        "required_action": guardrail["required_action"],
        "qa_must_hold": guardrail["status"] != "clear",
    }


def _ai_implementation_qa_signoff_requirements(verification: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"signoff": "qa_owner", "status": "pending", "required": True},
        {"signoff": "manager", "status": "pending", "required": True},
        {
            "signoff": "customer_guardrail_owner",
            "status": "pending" if verification["customer_guardrail_status"]["status"] != "clear" else "ready",
            "required": True,
        },
    ]


def _ai_implementation_qa_next_action(verification: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    if decision["status"] == "ready_for_qa":
        return {"owner": "qa", "action": "start_qa_review", "detail": decision["rationale"]}
    return verification["next_verification_action"]


def _ai_improvement_implementation_qa_review_packet_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Implementation QA Review Packet",
        f"- Experiment: {packet['experiment_id']}",
        f"- QA decision: {packet['qa_decision']['status']}",
        f"- Work item: {packet['work_item']['title']}",
        f"- Next action: {packet['next_qa_action']['action']}",
        "",
        "## Evidence Gaps",
    ]
    lines.extend(f"- {item['gap']}: {item['status']}" for item in packet["evidence_gaps"])
    lines.extend(["", "## Test Gates"])
    lines.extend(f"- {item['gate']}: {item['status']}" for item in packet["test_gates"])
    return "\n".join(lines)


def _ai_implementation_qa_signoff_status(packet: dict[str, Any]) -> str:
    if packet["qa_decision"]["status"] == "hold_qa":
        return "blocked"
    if packet["evidence_gaps"] or packet["customer_guardrails"]["qa_must_hold"]:
        return "pending_evidence"
    if any(item.get("required") and item.get("status") != "approved" for item in packet["signoff_requirements"]):
        return "pending_signoff"
    return "approved"


def _ai_implementation_qa_signoff_decision(packet: dict[str, Any]) -> dict[str, Any]:
    status = _ai_implementation_qa_signoff_status(packet)
    if status == "approved":
        return {"action": "approve_for_launch_review", "rationale": "QA signoff requirements are complete."}
    if status == "blocked":
        return {"action": "hold_launch", "rationale": "QA review is held by unresolved verification or guardrail blockers."}
    return {"action": "collect_signoff_evidence", "rationale": "Required QA evidence or signoffs are incomplete."}


def _ai_implementation_qa_signoff_gaps(packet: dict[str, Any]) -> list[dict[str, Any]]:
    gaps = [
        {
            "gap": item["signoff"],
            "status": item["status"],
            "required": item["required"],
        }
        for item in packet["signoff_requirements"]
        if item.get("required") and item.get("status") != "approved"
    ]
    gaps.extend(packet["evidence_gaps"])
    return gaps


def _ai_implementation_qa_launch_blockers(packet: dict[str, Any]) -> list[dict[str, Any]]:
    blockers = []
    if packet["qa_decision"]["status"] != "ready_for_qa":
        blockers.append(
            {
                "blocker": "qa_not_ready",
                "severity": "high" if packet["source_risk_level"] == "high" else "medium",
                "resolution": packet["qa_decision"]["rationale"],
            }
        )
    if packet["customer_guardrails"]["qa_must_hold"]:
        blockers.append(
            {
                "blocker": "customer_guardrail_hold",
                "severity": "high",
                "resolution": packet["customer_guardrails"]["required_action"],
            }
        )
    if packet["evidence_gaps"]:
        blockers.append(
            {
                "blocker": "qa_evidence_gaps",
                "severity": "medium",
                "resolution": "Close all QA evidence gaps before launch signoff.",
            }
        )
    return blockers


def _ai_implementation_qa_signoff_evidence_summary(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_gap_count": len(packet["evidence_gaps"]),
        "test_gate_count": len(packet["test_gates"]),
        "pending_test_gate_count": sum(1 for item in packet["test_gates"] if item.get("status") != "ready"),
        "required_signoff_count": sum(1 for item in packet["signoff_requirements"] if item.get("required")),
        "customer_guardrail_status": packet["customer_guardrails"]["status"],
    }


def _ai_implementation_qa_signoff_next_action(packet: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    if decision["action"] == "approve_for_launch_review":
        return {"owner": "manager", "action": "start_launch_review", "detail": decision["rationale"]}
    return packet["next_qa_action"]


def _ai_improvement_implementation_qa_signoff_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Implementation QA Signoff Report",
        f"- Experiment: {report['experiment_id']}",
        f"- Signoff status: {report['signoff_status']}",
        f"- Decision: {report['signoff_decision']['action']}",
        f"- Work item: {report['work_item']['title']}",
        f"- Next action: {report['next_signoff_action']['action']}",
        "",
        "## Launch Blockers",
    ]
    lines.extend(f"- {item['blocker']}: {item['resolution']}" for item in report["launch_blockers"])
    lines.extend(["", "## Signoff Gaps"])
    lines.extend(f"- {item['gap']}: {item['status']}" for item in report["signoff_gaps"])
    return "\n".join(lines)


def _ai_launch_review_decision(signoff: dict[str, Any]) -> dict[str, Any]:
    if signoff["signoff_decision"]["action"] == "approve_for_launch_review" and not signoff["launch_blockers"]:
        return {"status": "ready_to_launch", "rationale": "QA signoff is approved and no launch blockers remain."}
    if signoff["signoff_decision"]["action"] == "hold_launch":
        return {"status": "hold_launch", "rationale": signoff["signoff_decision"]["rationale"]}
    return {"status": "needs_launch_evidence", "rationale": "Launch review needs additional evidence or signoff completion."}


def _ai_launch_review_scope(signoff: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    if decision["status"] == "ready_to_launch":
        return {"scope": "eligible_accounts", "guardrail": "Launch only inside approved eligibility and monitoring gates."}
    return {"scope": "no_launch", "guardrail": "Do not launch until QA signoff and guardrails are clear."}


def _ai_launch_review_customer_guardrails(signoff: dict[str, Any]) -> list[dict[str, Any]]:
    guardrails = [
        {
            "guardrail": "no_launch_with_customer_guardrail_hold",
            "status": "blocked" if any(item["blocker"] == "customer_guardrail_hold" for item in signoff["launch_blockers"]) else "clear",
            "detail": "Customer guardrails must be clear before launch.",
        },
        {
            "guardrail": "no_new_claims_without_evidence",
            "status": "active",
            "detail": "Customer-facing claims require cleared evidence gaps.",
        },
    ]
    return guardrails


def _ai_launch_review_monitoring_requirements(signoff: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"monitor": "qa_signoff_status", "current_value": signoff["signoff_status"], "target": "approved"},
        {"monitor": "launch_blocker_count", "current_value": len(signoff["launch_blockers"]), "target": 0},
        {"monitor": "signoff_gap_count", "current_value": len(signoff["signoff_gaps"]), "target": 0},
        {"monitor": "risk_level", "current_value": signoff["source_risk_level"], "target": "low"},
    ]


def _ai_launch_review_rollback_triggers(signoff: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"trigger": "new_launch_blocker", "action": "pause_launch_review"},
        {"trigger": "customer_guardrail_hold", "action": "hold_launch"},
        {"trigger": "qa_signoff_regression", "action": "return_to_qa_review"},
        {"trigger": "risk_level_high", "action": "return_to_blocker_resolution" if signoff["source_risk_level"] == "high" else "monitor"},
    ]


def _ai_launch_review_next_action(signoff: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    if decision["status"] == "ready_to_launch":
        return {"owner": "manager", "action": "start_controlled_launch", "detail": decision["rationale"]}
    return signoff["next_signoff_action"]


def _ai_improvement_launch_review_packet_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Launch Review Packet",
        f"- Experiment: {packet['experiment_id']}",
        f"- Launch decision: {packet['launch_decision']['status']}",
        f"- Scope: {packet['launch_scope']['scope']}",
        f"- Work item: {packet['work_item']['title']}",
        f"- Next action: {packet['next_launch_action']['action']}",
        "",
        "## Launch Blockers",
    ]
    lines.extend(f"- {item['blocker']}: {item['resolution']}" for item in packet["launch_blockers"])
    lines.extend(["", "## Monitoring Requirements"])
    lines.extend(f"- {item['monitor']}: {item['current_value']} -> {item['target']}" for item in packet["monitoring_requirements"])
    return "\n".join(lines)


def _ai_launch_execution_status(launch: dict[str, Any]) -> str:
    if launch["launch_decision"]["status"] == "ready_to_launch":
        return "ready_to_execute"
    if launch["launch_decision"]["status"] == "hold_launch":
        return "held"
    return "needs_launch_evidence"


def _ai_launch_execution_tasks(launch: dict[str, Any]) -> list[dict[str, Any]]:
    if launch["launch_decision"]["status"] == "ready_to_launch":
        return [
            {"owner": "manager", "action": "open_controlled_launch", "status": "pending", "detail": launch["launch_scope"]["guardrail"]},
            {"owner": "qa", "action": "watch_launch_gates", "status": "pending", "detail": "Monitor QA and launch guardrails during rollout."},
            {"owner": "support", "action": "watch_customer_signals", "status": "pending", "detail": "Escalate customer confusion or rollback triggers."},
        ]
    return [
        {"owner": "manager", "action": "hold_launch", "status": "pending", "detail": launch["launch_decision"]["rationale"]},
        {"owner": "manager", "action": "clear_launch_blockers", "status": "pending", "detail": "Resolve launch blockers before any launch execution."},
        {"owner": "qa", "action": "preserve_no_launch_gate", "status": "pending", "detail": launch["launch_scope"]["guardrail"]},
    ]


def _ai_launch_execution_monitoring_setup(launch: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "monitor": item["monitor"],
            "current_value": item["current_value"],
            "target": item["target"],
            "status": "ready" if item["current_value"] == item["target"] else "needs_attention",
        }
        for item in launch["monitoring_requirements"]
    ]


def _ai_launch_execution_rollback_setup(launch: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "trigger": item["trigger"],
            "action": item["action"],
            "status": "armed" if launch["launch_decision"]["status"] == "ready_to_launch" else "hold",
        }
        for item in launch["rollback_triggers"]
    ]


def _ai_launch_execution_exit_criteria(launch: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"criterion": "launch_decision_ready", "target": "ready_to_launch", "current_value": launch["launch_decision"]["status"], "required": True},
        {"criterion": "launch_blockers_cleared", "target": 0, "current_value": len(launch["launch_blockers"]), "required": True},
        {"criterion": "evidence_gaps_cleared", "target": 0, "current_value": len(launch["evidence_gaps"]), "required": True},
        {"criterion": "risk_level_low", "target": "low", "current_value": launch["source_risk_level"], "required": True},
    ]


def _ai_launch_execution_immediate_action(tasks: list[dict[str, Any]], launch: dict[str, Any]) -> dict[str, Any]:
    task = tasks[0] if tasks else None
    if task:
        return {"owner": task["owner"], "action": task["action"], "detail": task["detail"]}
    return launch["next_launch_action"]


def _ai_improvement_launch_execution_plan_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Launch Execution Plan",
        f"- Experiment: {plan['experiment_id']}",
        f"- Execution status: {plan['execution_status']}",
        f"- Launch decision: {plan['launch_decision']['status']}",
        f"- Scope: {plan['launch_scope']['scope']}",
        f"- Immediate action: {plan['immediate_action']['action']}",
        "",
        "## Execution Tasks",
    ]
    lines.extend(f"- {item['owner']}: {item['action']} - {item['detail']}" for item in plan["execution_tasks"])
    lines.extend(["", "## Exit Criteria"])
    lines.extend(f"- {item['criterion']}: {item['current_value']} -> {item['target']}" for item in plan["exit_criteria"])
    return "\n".join(lines)


def _ai_launch_execution_task_status(plan: dict[str, Any]) -> dict[str, Any]:
    complete_statuses = {"complete", "completed", "done"}
    tasks = plan["execution_tasks"]
    completed = [item for item in tasks if item["status"] in complete_statuses]
    pending = [item for item in tasks if item["status"] not in complete_statuses]
    return {
        "status": "complete" if tasks and not pending else "pending",
        "total_count": len(tasks),
        "completed_count": len(completed),
        "pending_count": len(pending),
        "pending_actions": [item["action"] for item in pending],
    }


def _ai_launch_execution_monitoring_status(plan: dict[str, Any]) -> dict[str, Any]:
    checks = plan["monitoring_setup"]
    ready = [item for item in checks if item["status"] == "ready"]
    needs_attention = [item for item in checks if item["status"] != "ready"]
    return {
        "status": "ready" if checks and not needs_attention else "needs_attention",
        "ready_count": len(ready),
        "needs_attention_count": len(needs_attention),
        "monitors_needing_attention": [item["monitor"] for item in needs_attention],
    }


def _ai_launch_execution_rollback_status(plan: dict[str, Any]) -> dict[str, Any]:
    triggers = plan["rollback_setup"]
    armed = [item for item in triggers if item["status"] == "armed"]
    held = [item for item in triggers if item["status"] != "armed"]
    return {
        "status": "armed" if triggers and not held else "hold",
        "armed_count": len(armed),
        "hold_count": len(held),
        "held_triggers": [item["trigger"] for item in held],
    }


def _ai_launch_execution_exit_status(plan: dict[str, Any]) -> dict[str, Any]:
    criteria = []
    for item in plan["exit_criteria"]:
        met = item["current_value"] == item["target"]
        criteria.append({**item, "status": "met" if met else "blocked"})
    blocked = [item for item in criteria if item["required"] and item["status"] != "met"]
    return {
        "status": "met" if not blocked else "blocked",
        "met_count": len(criteria) - len(blocked),
        "blocked_count": len(blocked),
        "criteria": criteria,
        "blocked_criteria": [item["criterion"] for item in blocked],
    }


def _ai_launch_execution_monitor_blockers(
    task_status: dict[str, Any],
    monitoring_status: dict[str, Any],
    rollback_status: dict[str, Any],
    exit_criteria_status: dict[str, Any],
) -> list[dict[str, Any]]:
    blockers = []
    if task_status["pending_count"]:
        blockers.append(
            {
                "blocker": "launch_tasks_pending",
                "severity": "medium",
                "count": task_status["pending_count"],
                "resolution": "Complete or explicitly defer pending launch execution tasks.",
            }
        )
    if monitoring_status["needs_attention_count"]:
        blockers.append(
            {
                "blocker": "monitoring_attention_required",
                "severity": "medium",
                "count": monitoring_status["needs_attention_count"],
                "resolution": "Bring launch monitors to their target values before execution.",
            }
        )
    if rollback_status["hold_count"]:
        blockers.append(
            {
                "blocker": "rollback_not_armed",
                "severity": "high",
                "count": rollback_status["hold_count"],
                "resolution": "Arm rollback triggers before any controlled launch.",
            }
        )
    if exit_criteria_status["blocked_count"]:
        blockers.append(
            {
                "blocker": "launch_exit_criteria_blocked",
                "severity": "high",
                "count": exit_criteria_status["blocked_count"],
                "resolution": "Clear required launch exit criteria before moving forward.",
            }
        )
    return blockers


def _ai_launch_execution_monitor_status(plan: dict[str, Any], blockers: list[dict[str, Any]]) -> str:
    if plan["execution_status"] == "held" or any(item["severity"] == "high" for item in blockers):
        return "held"
    if not blockers and plan["execution_status"] == "ready_to_execute":
        return "ready"
    return "in_progress"


def _ai_launch_execution_monitor_risk_level(blockers: list[dict[str, Any]]) -> str:
    if any(item["severity"] == "high" for item in blockers):
        return "high"
    if blockers:
        return "medium"
    return "low"


def _ai_improvement_launch_execution_monitor_markdown(monitor: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Launch Execution Monitor",
        f"- Experiment: {monitor['experiment_id']}",
        f"- Monitor status: {monitor['status']}",
        f"- Risk level: {monitor['risk_level']}",
        f"- Immediate action: {monitor['immediate_action']['action']}",
        "",
        "## Readiness",
        f"- Tasks: {monitor['task_status']['status']} ({monitor['task_status']['pending_count']} pending)",
        f"- Monitoring: {monitor['monitoring_status']['status']} ({monitor['monitoring_status']['needs_attention_count']} attention)",
        f"- Rollback: {monitor['rollback_status']['status']} ({monitor['rollback_status']['hold_count']} held)",
        f"- Exit criteria: {monitor['exit_criteria_status']['status']} ({monitor['exit_criteria_status']['blocked_count']} blocked)",
        "",
        "## Blockers",
    ]
    if monitor["blockers"]:
        lines.extend(f"- {item['blocker']} ({item['severity']}): {item['resolution']}" for item in monitor["blockers"])
    else:
        lines.append("- None")
    return "\n".join(lines)


def _ai_launch_outcome_health(execution: dict[str, Any]) -> dict[str, Any]:
    high_blockers = [item for item in execution["blockers"] if item["severity"] == "high"]
    if execution["status"] == "held":
        status = "blocked"
    elif execution["status"] == "ready":
        status = "ready_to_measure"
    else:
        status = "monitoring"
    return {
        "status": status,
        "execution_status": execution["source_execution_status"],
        "execution_monitor_status": execution["status"],
        "blocker_count": len(execution["blockers"]),
        "high_blocker_count": len(high_blockers),
    }


def _ai_launch_outcome_value_status(execution: dict[str, Any]) -> dict[str, Any]:
    if execution["status"] == "held":
        return {
            "status": "not_measured",
            "readiness": "blocked_pre_launch",
            "metric": "customer_value_realization",
            "current_value": "not_started",
            "target": "positive_customer_signal_after_controlled_launch",
        }
    return {
        "status": "pending_measurement",
        "readiness": "ready_for_controlled_launch",
        "metric": "customer_value_realization",
        "current_value": "pending",
        "target": "positive_customer_signal_after_controlled_launch",
    }


def _ai_launch_outcome_customer_signal_status(execution: dict[str, Any]) -> dict[str, Any]:
    if execution["status"] == "held":
        return {
            "status": "not_started",
            "signals": ["customer_confusion", "advisor_follow_up_quality", "support_escalation"],
            "detail": "Customer outcome signals start after launch blockers clear.",
        }
    return {
        "status": "watching",
        "signals": ["customer_confusion", "advisor_follow_up_quality", "support_escalation"],
        "detail": "Watch customer signals during the controlled launch window.",
    }


def _ai_launch_outcome_rollback_status(execution: dict[str, Any]) -> dict[str, Any]:
    if execution["rollback_status"]["status"] == "armed":
        return {
            "status": "armed",
            "held_trigger_count": 0,
            "detail": "Rollback triggers are armed for launch outcome monitoring.",
        }
    return {
        "status": "not_armed",
        "held_trigger_count": execution["rollback_status"]["hold_count"],
        "detail": "Rollback triggers must be armed before outcome monitoring can start.",
    }


def _ai_launch_outcome_blockers(
    execution: dict[str, Any],
    launch_health: dict[str, Any],
    value_status: dict[str, Any],
    rollback_status: dict[str, Any],
) -> list[dict[str, Any]]:
    blockers = []
    if launch_health["status"] == "blocked":
        blockers.append(
            {
                "blocker": "launch_not_executed",
                "severity": "high",
                "resolution": "Keep outcome monitoring blocked until launch execution leaves held status.",
            }
        )
    if rollback_status["status"] != "armed":
        blockers.append(
            {
                "blocker": "rollback_not_armed",
                "severity": "high",
                "resolution": "Arm rollback triggers before measuring live customer outcomes.",
            }
        )
    if value_status["status"] == "not_measured":
        blockers.append(
            {
                "blocker": "customer_value_not_measurable",
                "severity": "medium",
                "resolution": "Start value measurement after the controlled launch is active.",
            }
        )
    return blockers


def _ai_launch_outcome_monitor_status(blockers: list[dict[str, Any]]) -> str:
    if any(item["blocker"] == "launch_not_executed" for item in blockers):
        return "blocked_pre_launch"
    if any(item["severity"] == "high" for item in blockers):
        return "rollback_review"
    if blockers:
        return "watching"
    return "measuring"


def _ai_launch_outcome_risk_level(blockers: list[dict[str, Any]]) -> str:
    if any(item["severity"] == "high" for item in blockers):
        return "high"
    if blockers:
        return "medium"
    return "low"


def _ai_launch_outcome_next_action(execution: dict[str, Any], blockers: list[dict[str, Any]]) -> dict[str, Any]:
    if blockers:
        return {
            "owner": execution["immediate_action"]["owner"],
            "action": execution["immediate_action"]["action"],
            "detail": blockers[0]["resolution"],
        }
    return {
        "owner": "manager",
        "action": "start_outcome_measurement",
        "detail": "Measure customer value, support signals, and rollback triggers during the launch window.",
    }


def _ai_improvement_launch_outcome_monitor_markdown(monitor: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Launch Outcome Monitor",
        f"- Experiment: {monitor['experiment_id']}",
        f"- Outcome status: {monitor['status']}",
        f"- Risk level: {monitor['risk_level']}",
        f"- Immediate action: {monitor['immediate_action']['action']}",
        "",
        "## Outcome Readiness",
        f"- Launch health: {monitor['launch_health']['status']}",
        f"- Customer value: {monitor['value_status']['status']}",
        f"- Customer signals: {monitor['customer_signal_status']['status']}",
        f"- Rollback: {monitor['rollback_status']['status']}",
        "",
        "## Blockers",
    ]
    if monitor["blockers"]:
        lines.extend(f"- {item['blocker']} ({item['severity']}): {item['resolution']}" for item in monitor["blockers"])
    else:
        lines.append("- None")
    return "\n".join(lines)


def _ai_launch_value_proof_status(outcome: dict[str, Any]) -> str:
    if outcome["status"] == "blocked_pre_launch":
        return "blocked_pre_launch"
    if outcome["risk_level"] == "high":
        return "needs_risk_review"
    if outcome["value_status"]["status"] == "pending_measurement":
        return "ready_to_collect_proof"
    return "proof_ready"


def _ai_launch_value_customer_claim(outcome: dict[str, Any]) -> dict[str, Any]:
    if outcome["value_status"]["status"] == "not_measured":
        return {
            "status": "not_claimable",
            "claim": "Customer value cannot be claimed until the controlled launch is active.",
            "evidence": outcome["value_status"]["current_value"],
        }
    return {
        "status": "pending_evidence",
        "claim": "Customer value proof is pending controlled-launch outcome signals.",
        "evidence": outcome["value_status"]["current_value"],
    }


def _ai_launch_value_proof_points(outcome: dict[str, Any]) -> list[dict[str, Any]]:
    if outcome["status"] == "blocked_pre_launch":
        return [
            {
                "proof": "no_customer_value_claim_before_launch",
                "status": "active_guardrail",
                "detail": "The launch is held, so the packet should preserve customer trust instead of claiming value.",
            },
            {
                "proof": "rollback_readiness_required",
                "status": outcome["rollback_status"]["status"],
                "detail": outcome["rollback_status"]["detail"],
            },
        ]
    return [
        {
            "proof": "controlled_launch_active",
            "status": outcome["launch_health"]["status"],
            "detail": "Outcome monitoring can collect customer-value evidence.",
        },
        {
            "proof": "customer_signal_watch",
            "status": outcome["customer_signal_status"]["status"],
            "detail": outcome["customer_signal_status"]["detail"],
        },
    ]


def _ai_launch_value_evidence_gaps(outcome: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "gap": item["blocker"],
            "severity": item["severity"],
            "resolution": item["resolution"],
        }
        for item in outcome["blockers"]
    ]


def _ai_launch_value_customer_message(outcome: dict[str, Any]) -> dict[str, str]:
    if outcome["status"] == "blocked_pre_launch":
        return {
            "status": "internal_only",
            "headline": "Launch held; value proof pending.",
            "detail": "Do not present customer value as achieved until launch blockers clear and outcome signals exist.",
        }
    return {
        "status": "customer_ready_after_review",
        "headline": "Controlled launch value proof is being measured.",
        "detail": "Share only measured customer signals and keep rollback criteria visible.",
    }


def _ai_improvement_launch_value_proof_packet_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Launch Value Proof Packet",
        f"- Experiment: {packet['experiment_id']}",
        f"- Proof status: {packet['proof_status']}",
        f"- Risk level: {packet['risk_level']}",
        f"- Customer claim: {packet['customer_value_claim']['status']}",
        f"- Advisor action: {packet['advisor_next_action']['action']}",
        "",
        "## Proof Points",
    ]
    lines.extend(f"- {item['proof']}: {item['status']} - {item['detail']}" for item in packet["proof_points"])
    lines.extend(["", "## Evidence Gaps"])
    if packet["evidence_gaps"]:
        lines.extend(f"- {item['gap']} ({item['severity']}): {item['resolution']}" for item in packet["evidence_gaps"])
    else:
        lines.append("- None")
    return "\n".join(lines)


def _ai_launch_customer_communication_status(proof: dict[str, Any]) -> str:
    if proof["customer_message"]["status"] == "internal_only":
        return "internal_hold_only"
    if proof["proof_status"] in {"ready_to_collect_proof", "proof_ready"}:
        return "ready_for_review"
    return "needs_risk_review"


def _ai_launch_customer_communication_audience(proof: dict[str, Any]) -> dict[str, str]:
    if proof["customer_message"]["status"] == "internal_only":
        return {
            "primary": "advisor",
            "visibility": "internal",
            "reason": "Customer value proof is blocked before launch.",
        }
    return {
        "primary": "customer",
        "visibility": "review_required",
        "reason": "Customer-facing copy requires proof and rollback review.",
    }


def _ai_launch_customer_communication_message(proof: dict[str, Any]) -> dict[str, str]:
    if proof["customer_message"]["status"] == "internal_only":
        return {
            "headline": proof["customer_message"]["headline"],
            "body": proof["customer_message"]["detail"],
            "cta": "Hold customer-facing value claims until launch blockers clear.",
        }
    return {
        "headline": proof["customer_message"]["headline"],
        "body": proof["customer_message"]["detail"],
        "cta": "Share measured customer-value evidence after advisor review.",
    }


def _ai_launch_customer_communication_review_gates(proof: dict[str, Any]) -> list[dict[str, str]]:
    gates = [
        {
            "gate": "proof_status",
            "status": "blocked" if proof["proof_status"] == "blocked_pre_launch" else "ready",
            "requirement": "Customer-facing value claims require non-blocked proof status.",
        },
        {
            "gate": "customer_claim",
            "status": "blocked" if proof["customer_value_claim"]["status"] == "not_claimable" else "ready",
            "requirement": "Customer claim must be evidence-backed before external use.",
        },
    ]
    if proof["risk_level"] == "high":
        gates.append(
            {
                "gate": "risk_review",
                "status": "blocked",
                "requirement": "High-risk launch communication requires manager review.",
            }
        )
    return gates


def _ai_launch_customer_communication_blocked_claims(proof: dict[str, Any]) -> list[dict[str, str]]:
    if proof["customer_value_claim"]["status"] != "not_claimable":
        return []
    return [
        {
            "claim": "ai_improvement_created_customer_value",
            "reason": proof["customer_value_claim"]["claim"],
        },
        {
            "claim": "launch_success",
            "reason": "The source launch decision is not ready to launch.",
        },
    ]


def _ai_improvement_launch_customer_communication_packet_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Launch Customer Communication Packet",
        f"- Experiment: {packet['experiment_id']}",
        f"- Communication status: {packet['communication_status']}",
        f"- Audience: {packet['audience']['primary']} ({packet['audience']['visibility']})",
        f"- Advisor action: {packet['advisor_next_action']['action']}",
        "",
        "## Message",
        f"- {packet['message']['headline']}",
        f"- {packet['message']['body']}",
        f"- CTA: {packet['message']['cta']}",
        "",
        "## Review Gates",
    ]
    lines.extend(f"- {item['gate']}: {item['status']} - {item['requirement']}" for item in packet["review_gates"])
    lines.extend(["", "## Blocked Claims"])
    if packet["blocked_claims"]:
        lines.extend(f"- {item['claim']}: {item['reason']}" for item in packet["blocked_claims"])
    else:
        lines.append("- None")
    return "\n".join(lines)


def _ai_launch_customer_communication_send_decision(communication: dict[str, Any]) -> dict[str, str]:
    if communication["communication_status"] == "internal_hold_only":
        return {
            "status": "hold_send",
            "decision": "do_not_send",
            "rationale": "Customer-facing communication is blocked until proof status and customer claims are review-ready.",
        }
    if any(gate["status"] == "blocked" for gate in communication["review_gates"]):
        return {
            "status": "needs_review",
            "decision": "review_before_send",
            "rationale": "Review gates must clear before customer-facing communication is sent.",
        }
    return {
        "status": "approved_to_send",
        "decision": "send_after_advisor_review",
        "rationale": "Communication is customer-ready after advisor review.",
    }


def _ai_launch_customer_communication_required_approvals(communication: dict[str, Any]) -> list[dict[str, str]]:
    approvals = [
        {
            "owner": "advisor",
            "approval": "message_accuracy_review",
            "status": "blocked" if communication["communication_status"] == "internal_hold_only" else "required",
        }
    ]
    if communication["risk_level"] == "high":
        approvals.append({"owner": "manager", "approval": "high_risk_send_review", "status": "blocked"})
    if communication["blocked_claims"]:
        approvals.append({"owner": "compliance", "approval": "blocked_claim_review", "status": "blocked"})
    return approvals


def _ai_launch_customer_communication_send_blockers(communication: dict[str, Any]) -> list[dict[str, str]]:
    blockers = []
    blockers.extend(
        {
            "blocker": f"review_gate:{gate['gate']}",
            "severity": "high" if gate["status"] == "blocked" else "medium",
            "resolution": gate["requirement"],
        }
        for gate in communication["review_gates"]
        if gate["status"] == "blocked"
    )
    blockers.extend(
        {
            "blocker": f"blocked_claim:{claim['claim']}",
            "severity": "high",
            "resolution": claim["reason"],
        }
        for claim in communication["blocked_claims"]
    )
    return blockers


def _ai_launch_customer_communication_escalation_path(communication: dict[str, Any]) -> list[dict[str, str]]:
    if communication["communication_status"] == "internal_hold_only":
        return [
            {
                "owner": "manager",
                "action": "hold_customer_send",
                "detail": "Keep the message internal until launch and value proof blockers clear.",
            },
            {
                "owner": "advisor",
                "action": "clear_launch_proof_gaps",
                "detail": communication["message"]["cta"],
            },
        ]
    return [
        {
            "owner": "advisor",
            "action": "review_customer_message",
            "detail": "Confirm the customer message matches measured proof before sending.",
        }
    ]


def _ai_launch_customer_communication_approved_message(
    communication: dict[str, Any],
    send_decision: dict[str, str],
) -> dict[str, str]:
    if send_decision["decision"] == "do_not_send":
        return {
            "status": "withheld",
            "headline": communication["message"]["headline"],
            "body": "Customer-facing copy withheld until proof and launch gates clear.",
        }
    return {
        "status": "ready_after_review",
        "headline": communication["message"]["headline"],
        "body": communication["message"]["body"],
    }


def _ai_improvement_launch_customer_communication_review_packet_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Launch Customer Communication Review Packet",
        f"- Experiment: {packet['experiment_id']}",
        f"- Review status: {packet['review_status']}",
        f"- Send decision: {packet['send_decision']['decision']}",
        f"- Advisor action: {packet['advisor_next_action']['action']}",
        "",
        "## Required Approvals",
    ]
    lines.extend(f"- {item['owner']}: {item['approval']} ({item['status']})" for item in packet["required_approvals"])
    lines.extend(["", "## Send Blockers"])
    if packet["send_blockers"]:
        lines.extend(f"- {item['blocker']} ({item['severity']}): {item['resolution']}" for item in packet["send_blockers"])
    else:
        lines.append("- None")
    return "\n".join(lines)


def _ai_launch_customer_communication_delivery_status(review: dict[str, Any]) -> str:
    if review["send_decision"]["decision"] == "do_not_send":
        return "withheld"
    if review["review_status"] == "approved_to_send":
        return "ready_to_deliver"
    return "pending_review"


def _ai_launch_customer_communication_delivery_channel_plan(review: dict[str, Any]) -> dict[str, str]:
    if review["send_decision"]["decision"] == "do_not_send":
        return {
            "channel": "internal_task",
            "visibility": "internal",
            "status": "withheld",
            "detail": "Do not deliver customer-facing copy until review gates clear.",
        }
    return {
        "channel": "advisor_email",
        "visibility": "customer",
        "status": "ready_after_review",
        "detail": "Deliver the reviewed message with measured proof and rollback criteria visible.",
    }


def _ai_launch_customer_communication_delivery_payload(review: dict[str, Any]) -> dict[str, Any]:
    withheld = review["approved_message"]["status"] == "withheld"
    return {
        "status": "withheld" if withheld else "ready",
        "customer_facing": not withheld,
        "headline": review["approved_message"]["headline"],
        "body": review["approved_message"]["body"],
    }


def _ai_launch_customer_communication_delivery_checklist(review: dict[str, Any]) -> list[dict[str, str]]:
    approvals_blocked = any(item["status"] == "blocked" for item in review["required_approvals"])
    return [
        {
            "gate": "send_decision",
            "status": "blocked" if review["send_decision"]["decision"] == "do_not_send" else "ready",
            "detail": review["send_decision"]["rationale"],
        },
        {
            "gate": "required_approvals",
            "status": "blocked" if approvals_blocked else "ready",
            "detail": "All required approvals must clear before delivery.",
        },
        {
            "gate": "send_blockers",
            "status": "blocked" if review["send_blockers"] else "ready",
            "detail": "No send blockers may remain before delivery.",
        },
    ]


def _ai_launch_customer_communication_delivery_audit_trail(review: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"event": "source_review_status", "status": review["review_status"]},
        {"event": "source_send_decision", "status": review["send_decision"]["decision"]},
        {"event": "approved_message", "status": review["approved_message"]["status"]},
    ]


def _ai_launch_customer_communication_delivery_follow_up_plan(review: dict[str, Any]) -> dict[str, str]:
    if review["send_decision"]["decision"] == "do_not_send":
        return {
            "owner": "manager",
            "action": "recheck_after_launch_gates_clear",
            "status": "waiting",
            "detail": review["escalation_path"][0]["detail"] if review["escalation_path"] else "Recheck delivery after blockers clear.",
        }
    return {
        "owner": "advisor",
        "action": "confirm_delivery_outcome",
        "status": "pending",
        "detail": "Record customer response after the reviewed message is delivered.",
    }


def _ai_launch_customer_communication_delivery_next_action(review: dict[str, Any]) -> dict[str, str]:
    if review["send_decision"]["decision"] == "do_not_send" and review["escalation_path"]:
        return review["escalation_path"][0]
    return review["advisor_next_action"]


def _ai_improvement_launch_customer_communication_delivery_packet_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Launch Customer Communication Delivery Packet",
        f"- Experiment: {packet['experiment_id']}",
        f"- Delivery status: {packet['delivery_status']}",
        f"- Channel: {packet['channel_plan']['channel']} ({packet['channel_plan']['visibility']})",
        f"- Advisor action: {packet['advisor_next_action']['action']}",
        "",
        "## Delivery Checklist",
    ]
    lines.extend(f"- {item['gate']}: {item['status']} - {item['detail']}" for item in packet["delivery_checklist"])
    lines.extend(["", "## Payload"])
    lines.append(f"- {packet['delivery_payload']['status']}: {packet['delivery_payload']['headline']}")
    return "\n".join(lines)


def _ai_launch_customer_communication_delivery_progress(delivery: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": delivery["delivery_status"],
        "channel": delivery["channel_plan"]["channel"],
        "visibility": delivery["channel_plan"]["visibility"],
        "customer_facing": delivery["delivery_payload"]["customer_facing"],
    }


def _ai_launch_customer_communication_delivery_checklist_status(delivery: dict[str, Any]) -> dict[str, Any]:
    blocked = [item for item in delivery["delivery_checklist"] if item["status"] == "blocked"]
    ready = [item for item in delivery["delivery_checklist"] if item["status"] == "ready"]
    return {
        "status": "blocked" if blocked else "ready",
        "blocked_count": len(blocked),
        "ready_count": len(ready),
        "blocked_gates": [item["gate"] for item in blocked],
    }


def _ai_launch_customer_communication_delivery_audit_status(delivery: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "recorded" if delivery["audit_trail"] else "missing",
        "event_count": len(delivery["audit_trail"]),
        "events": [item["event"] for item in delivery["audit_trail"]],
    }


def _ai_launch_customer_communication_delivery_follow_up_status(delivery: dict[str, Any]) -> dict[str, str]:
    return {
        "status": delivery["follow_up_plan"]["status"],
        "owner": delivery["follow_up_plan"]["owner"],
        "action": delivery["follow_up_plan"]["action"],
        "detail": delivery["follow_up_plan"]["detail"],
    }


def _ai_launch_customer_communication_delivery_monitor_blockers(
    delivery: dict[str, Any],
    delivery_progress: dict[str, Any],
    checklist_status: dict[str, Any],
    follow_up_status: dict[str, str],
) -> list[dict[str, Any]]:
    blockers = []
    if delivery_progress["status"] == "withheld":
        blockers.append(
            {
                "blocker": "delivery_withheld",
                "severity": "high",
                "resolution": "Keep customer communication withheld until send decision and proof gates clear.",
            }
        )
    if checklist_status["blocked_count"]:
        blockers.append(
            {
                "blocker": "delivery_checklist_blocked",
                "severity": "high",
                "count": checklist_status["blocked_count"],
                "resolution": "Clear blocked delivery checklist gates before customer-facing delivery.",
            }
        )
    if follow_up_status["status"] == "waiting":
        blockers.append(
            {
                "blocker": "follow_up_waiting_on_launch_gates",
                "severity": "medium",
                "resolution": follow_up_status["detail"],
            }
        )
    if delivery["source_customer_claim_status"] == "not_claimable":
        blockers.append(
            {
                "blocker": "customer_value_claim_not_claimable",
                "severity": "high",
                "resolution": "Do not deliver customer-facing value claims until they are evidence-backed.",
            }
        )
    return blockers


def _ai_launch_customer_communication_delivery_monitor_status(
    delivery: dict[str, Any],
    blockers: list[dict[str, Any]],
) -> str:
    if delivery["delivery_status"] == "withheld":
        return "blocked"
    if any(item["severity"] == "high" for item in blockers):
        return "needs_review"
    if delivery["delivery_status"] == "ready_to_deliver":
        return "ready"
    return "monitoring"


def _ai_launch_customer_communication_delivery_monitor_risk_level(
    delivery: dict[str, Any],
    blockers: list[dict[str, Any]],
) -> str:
    if delivery["risk_level"] == "high" or any(item["severity"] == "high" for item in blockers):
        return "high"
    if blockers:
        return "medium"
    return "low"


def _ai_improvement_launch_customer_communication_delivery_monitor_markdown(monitor: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Launch Customer Communication Delivery Monitor",
        f"- Experiment: {monitor['experiment_id']}",
        f"- Monitor status: {monitor['status']}",
        f"- Risk level: {monitor['risk_level']}",
        f"- Immediate action: {monitor['immediate_action']['action']}",
        "",
        "## Delivery State",
        f"- Progress: {monitor['delivery_progress']['status']} via {monitor['delivery_progress']['channel']}",
        f"- Checklist: {monitor['checklist_status']['status']} ({monitor['checklist_status']['blocked_count']} blocked)",
        f"- Audit: {monitor['audit_status']['status']} ({monitor['audit_status']['event_count']} events)",
        f"- Follow-up: {monitor['follow_up_status']['status']} ({monitor['follow_up_status']['action']})",
        "",
        "## Blockers",
    ]
    if monitor["blockers"]:
        lines.extend(f"- {item['blocker']} ({item['severity']}): {item['resolution']}" for item in monitor["blockers"])
    else:
        lines.append("- None")
    return "\n".join(lines)


def _ai_launch_customer_communication_delivery_unblock_status(monitor: dict[str, Any]) -> str:
    if monitor["status"] == "blocked":
        return "blocked_delivery"
    if monitor["status"] == "ready":
        return "ready_to_send"
    return "needs_review"


def _ai_launch_customer_communication_delivery_unblock_tasks(monitor: dict[str, Any]) -> list[dict[str, Any]]:
    owner_by_blocker = {
        "delivery_withheld": "manager",
        "delivery_checklist_blocked": "advisor",
        "follow_up_waiting_on_launch_gates": "manager",
        "customer_value_claim_not_claimable": "compliance",
    }
    action_by_blocker = {
        "delivery_withheld": "clear_send_hold",
        "delivery_checklist_blocked": "clear_delivery_checklist_gates",
        "follow_up_waiting_on_launch_gates": "recheck_launch_gates",
        "customer_value_claim_not_claimable": "collect_customer_value_proof",
    }
    return [
        {
            "owner": owner_by_blocker.get(item["blocker"], "advisor"),
            "action": action_by_blocker.get(item["blocker"], "clear_delivery_blocker"),
            "status": "pending",
            "blocker": item["blocker"],
            "severity": item["severity"],
            "detail": item["resolution"],
        }
        for item in monitor["blockers"]
    ]


def _ai_launch_customer_communication_delivery_unblock_proof_gates(monitor: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "gate": "send_decision",
            "status": "blocked" if monitor["source_send_decision"]["decision"] == "do_not_send" else "ready",
            "target": "send_after_advisor_review",
            "current_value": monitor["source_send_decision"]["decision"],
        },
        {
            "gate": "delivery_checklist",
            "status": monitor["checklist_status"]["status"],
            "target": "ready",
            "current_value": monitor["checklist_status"]["status"],
            "blocked_gates": monitor["checklist_status"]["blocked_gates"],
        },
        {
            "gate": "customer_value_claim",
            "status": "blocked" if monitor["source_customer_claim_status"] == "not_claimable" else "ready",
            "target": "claimable",
            "current_value": monitor["source_customer_claim_status"],
        },
    ]


def _ai_launch_customer_communication_delivery_unblock_exit_criteria(monitor: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "criterion": "delivery_not_withheld",
            "target": "ready_to_deliver",
            "current_value": monitor["source_delivery_status"],
            "status": "blocked" if monitor["source_delivery_status"] == "withheld" else "met",
        },
        {
            "criterion": "send_decision_cleared",
            "target": "send_after_advisor_review",
            "current_value": monitor["source_send_decision"]["decision"],
            "status": "blocked" if monitor["source_send_decision"]["decision"] == "do_not_send" else "met",
        },
        {
            "criterion": "customer_value_claim_supported",
            "target": "claimable",
            "current_value": monitor["source_customer_claim_status"],
            "status": "blocked" if monitor["source_customer_claim_status"] == "not_claimable" else "met",
        },
    ]


def _ai_launch_customer_communication_delivery_unblock_recheck_plan(monitor: dict[str, Any]) -> dict[str, str]:
    if monitor["status"] == "blocked":
        return {
            "owner": "manager",
            "action": "recheck_delivery_unblock_after_launch_gates_clear",
            "status": "waiting",
            "detail": "Re-run the delivery monitor after proof gates, checklist blockers, and send decision clear.",
        }
    return {
        "owner": "advisor",
        "action": "prepare_customer_delivery",
        "status": "ready",
        "detail": "Delivery gates are clear enough to prepare customer communication.",
    }


def _ai_improvement_launch_customer_communication_delivery_unblock_plan_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# AI Improvement Launch Customer Communication Delivery Unblock Plan",
        f"- Experiment: {plan['experiment_id']}",
        f"- Plan status: {plan['plan_status']}",
        f"- Risk level: {plan['risk_level']}",
        f"- Immediate action: {plan['immediate_action']['action']}",
        "",
        "## Unblock Tasks",
    ]
    if plan["unblock_tasks"]:
        lines.extend(f"- {item['owner']}: {item['action']} - {item['detail']}" for item in plan["unblock_tasks"])
    else:
        lines.append("- None")
    lines.extend(["", "## Exit Criteria"])
    lines.extend(f"- {item['criterion']}: {item['current_value']} -> {item['target']} ({item['status']})" for item in plan["exit_criteria"])
    return "\n".join(lines)


def _ai_launch_customer_communication_delivery_unblock_verification_results(
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    results = []
    results.extend(
        {
            "check": f"proof_gate:{item['gate']}",
            "status": "passed" if item["status"] == "ready" else "failed",
            "current_value": item["current_value"],
            "target": item["target"],
            "detail": f"Proof gate {item['gate']} is {item['status']}.",
        }
        for item in plan["proof_gates"]
    )
    results.extend(
        {
            "check": f"exit_criterion:{item['criterion']}",
            "status": "passed" if item["status"] == "met" else "failed",
            "current_value": item["current_value"],
            "target": item["target"],
            "detail": f"Exit criterion {item['criterion']} is {item['status']}.",
        }
        for item in plan["exit_criteria"]
    )
    results.extend(
        {
            "check": f"unblock_task:{item['action']}",
            "status": "passed" if item["status"] in {"done", "complete", "completed"} else "failed",
            "current_value": item["status"],
            "target": "completed",
            "detail": item["detail"],
        }
        for item in plan["unblock_tasks"]
    )
    return results


def _ai_launch_customer_communication_delivery_unblock_verification_status(
    verification_results: list[dict[str, Any]],
) -> str:
    if any(item["status"] == "failed" for item in verification_results):
        return "failed"
    return "passed"


def _ai_launch_customer_communication_delivery_unblock_failed_checks(
    verification_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [item for item in verification_results if item["status"] == "failed"]


def _ai_launch_customer_communication_delivery_unblock_required_follow_up(
    plan: dict[str, Any],
    verification_results: list[dict[str, Any]],
) -> list[dict[str, str]]:
    failed = _ai_launch_customer_communication_delivery_unblock_failed_checks(verification_results)
    if not failed:
        return [
            {
                "owner": "advisor",
                "action": "prepare_customer_delivery",
                "detail": "All unblock verification checks passed.",
            }
        ]
    return [
        {
            "owner": plan["immediate_action"].get("owner", "advisor"),
            "action": plan["immediate_action"]["action"],
            "detail": failed[0]["detail"],
        }
    ]


def _ai_launch_customer_communication_delivery_unblock_verification_next_action(
    plan: dict[str, Any],
    verification_results: list[dict[str, Any]],
) -> dict[str, str]:
    follow_up = _ai_launch_customer_communication_delivery_unblock_required_follow_up(plan, verification_results)
    return follow_up[0]


def _ai_improvement_launch_customer_communication_delivery_unblock_verification_report_markdown(
    report: dict[str, Any],
) -> str:
    lines = [
        "# AI Improvement Launch Customer Communication Delivery Unblock Verification Report",
        f"- Experiment: {report['experiment_id']}",
        f"- Verification status: {report['verification_status']}",
        f"- Risk level: {report['risk_level']}",
        f"- Next action: {report['next_action']['action']}",
        "",
        "## Failed Checks",
    ]
    if report["failed_checks"]:
        lines.extend(f"- {item['check']}: {item['current_value']} -> {item['target']}" for item in report["failed_checks"])
    else:
        lines.append("- None")
    return "\n".join(lines)


def _ai_launch_customer_communication_delivery_send_authorization_decision(
    verification: dict[str, Any],
) -> dict[str, str]:
    if verification["verification_status"] != "passed":
        return {
            "status": "hold_send",
            "decision": "do_not_send",
            "rationale": "Unblock verification failed, so customer-facing delivery remains unauthorized.",
        }
    if verification["risk_level"] == "high":
        return {
            "status": "needs_manager_authorization",
            "decision": "review_before_send",
            "rationale": "High-risk communication requires manager authorization before send.",
        }
    return {
        "status": "authorized",
        "decision": "send_after_advisor_review",
        "rationale": "Verification passed and send requirements are clear.",
    }


def _ai_launch_customer_communication_delivery_send_requirements(
    verification: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "requirement": "unblock_verification_passed",
            "status": "met" if verification["verification_status"] == "passed" else "blocked",
            "current_value": verification["verification_status"],
        },
        {
            "requirement": "delivery_status_ready",
            "status": "met" if verification["source_delivery_status"] == "ready_to_deliver" else "blocked",
            "current_value": verification["source_delivery_status"],
        },
        {
            "requirement": "customer_claim_supported",
            "status": "met" if verification["source_customer_claim_status"] != "not_claimable" else "blocked",
            "current_value": verification["source_customer_claim_status"],
        },
    ]


def _ai_launch_customer_communication_delivery_send_blocked_reasons(
    verification: dict[str, Any],
) -> list[dict[str, str]]:
    reasons = [
        {
            "reason": item["check"],
            "detail": item["detail"],
        }
        for item in verification["failed_checks"]
    ]
    if verification["source_send_decision"]["decision"] == "do_not_send":
        reasons.append(
            {
                "reason": "source_send_decision",
                "detail": verification["source_send_decision"]["rationale"],
            }
        )
    return reasons


def _ai_launch_customer_communication_delivery_authorized_payload(
    verification: dict[str, Any],
) -> dict[str, Any]:
    authorized = verification["verification_status"] == "passed" and verification["source_delivery_status"] == "ready_to_deliver"
    return {
        "status": "authorized" if authorized else "withheld",
        "customer_facing": authorized,
        "detail": "Customer-facing payload withheld until verification passes." if not authorized else "Payload may be sent after advisor review.",
    }


def _ai_launch_customer_communication_delivery_send_authorization_next_action(
    verification: dict[str, Any],
    authorization_decision: dict[str, str],
) -> dict[str, str]:
    if authorization_decision["decision"] == "do_not_send":
        return verification["next_action"]
    if authorization_decision["decision"] == "review_before_send":
        return {
            "owner": "manager",
            "action": "authorize_high_risk_customer_send",
            "detail": authorization_decision["rationale"],
        }
    return {
        "owner": "advisor",
        "action": "send_customer_communication",
        "detail": authorization_decision["rationale"],
    }


def _ai_improvement_launch_customer_communication_delivery_send_authorization_packet_markdown(
    packet: dict[str, Any],
) -> str:
    lines = [
        "# AI Improvement Launch Customer Communication Delivery Send Authorization Packet",
        f"- Experiment: {packet['experiment_id']}",
        f"- Authorization status: {packet['authorization_status']}",
        f"- Decision: {packet['authorization_decision']['decision']}",
        f"- Next action: {packet['next_action']['action']}",
        "",
        "## Send Requirements",
    ]
    lines.extend(
        f"- {item['requirement']}: {item['status']} ({item['current_value']})"
        for item in packet["send_requirements"]
    )
    lines.extend(["", "## Blocked Reasons"])
    if packet["blocked_reasons"]:
        lines.extend(f"- {item['reason']}: {item['detail']}" for item in packet["blocked_reasons"])
    else:
        lines.append("- None")
    return "\n".join(lines)


def _ai_launch_customer_communication_delivery_send_authorization_progress(
    authorization: dict[str, Any],
) -> dict[str, str]:
    return {
        "status": authorization["authorization_status"],
        "decision": authorization["authorization_decision"]["decision"],
        "rationale": authorization["authorization_decision"]["rationale"],
    }


def _ai_launch_customer_communication_delivery_send_requirement_status(
    authorization: dict[str, Any],
) -> dict[str, Any]:
    blocked = [item for item in authorization["send_requirements"] if item["status"] != "met"]
    met = [item for item in authorization["send_requirements"] if item["status"] == "met"]
    return {
        "status": "blocked" if blocked else "met",
        "blocked_count": len(blocked),
        "met_count": len(met),
        "blocked_requirements": [item["requirement"] for item in blocked],
    }


def _ai_launch_customer_communication_delivery_send_blocked_reason_status(
    authorization: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "blocked" if authorization["blocked_reasons"] else "clear",
        "count": len(authorization["blocked_reasons"]),
        "reasons": [item["reason"] for item in authorization["blocked_reasons"]],
    }


def _ai_launch_customer_communication_delivery_send_payload_status(
    authorization: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": authorization["authorized_payload"]["status"],
        "customer_facing": authorization["authorized_payload"]["customer_facing"],
        "detail": authorization["authorized_payload"]["detail"],
    }


def _ai_launch_customer_communication_delivery_send_authorization_monitor_blockers(
    authorization: dict[str, Any],
    requirements_status: dict[str, Any],
    blocked_reason_status: dict[str, Any],
    payload_status: dict[str, Any],
) -> list[dict[str, Any]]:
    blockers = []
    if authorization["authorization_decision"]["decision"] == "do_not_send":
        blockers.append(
            {
                "blocker": "send_authorization_held",
                "severity": "high",
                "resolution": authorization["authorization_decision"]["rationale"],
            }
        )
    if requirements_status["blocked_count"]:
        blockers.append(
            {
                "blocker": "send_requirements_blocked",
                "severity": "high",
                "count": requirements_status["blocked_count"],
                "resolution": "Clear all send requirements before customer-facing delivery.",
            }
        )
    if blocked_reason_status["count"]:
        blockers.append(
            {
                "blocker": "send_blocked_reasons_present",
                "severity": "high",
                "count": blocked_reason_status["count"],
                "resolution": "Resolve blocked reasons before authorizing send.",
            }
        )
    if not payload_status["customer_facing"]:
        blockers.append(
            {
                "blocker": "payload_not_customer_facing",
                "severity": "medium",
                "resolution": payload_status["detail"],
            }
        )
    return blockers


def _ai_launch_customer_communication_delivery_send_authorization_monitor_status(
    authorization: dict[str, Any],
    blockers: list[dict[str, Any]],
) -> str:
    if authorization["authorization_status"] == "hold_send":
        return "held"
    if any(item["severity"] == "high" for item in blockers):
        return "blocked"
    if authorization["authorization_status"] == "authorized":
        return "ready"
    return "needs_review"


def _ai_launch_customer_communication_delivery_send_authorization_monitor_risk_level(
    authorization: dict[str, Any],
    blockers: list[dict[str, Any]],
) -> str:
    if authorization["risk_level"] == "high" or any(item["severity"] == "high" for item in blockers):
        return "high"
    if blockers:
        return "medium"
    return "low"


def _ai_improvement_launch_customer_communication_delivery_send_authorization_monitor_markdown(
    monitor: dict[str, Any],
) -> str:
    lines = [
        "# AI Improvement Launch Customer Communication Delivery Send Authorization Monitor",
        f"- Experiment: {monitor['experiment_id']}",
        f"- Monitor status: {monitor['status']}",
        f"- Authorization decision: {monitor['authorization_progress']['decision']}",
        f"- Immediate action: {monitor['immediate_action']['action']}",
        "",
        "## Authorization State",
        f"- Requirements: {monitor['requirements_status']['status']} ({monitor['requirements_status']['blocked_count']} blocked)",
        f"- Blocked reasons: {monitor['blocked_reason_status']['count']}",
        f"- Payload: {monitor['payload_status']['status']}",
        "",
        "## Blockers",
    ]
    if monitor["blockers"]:
        lines.extend(f"- {item['blocker']} ({item['severity']}): {item['resolution']}" for item in monitor["blockers"])
    else:
        lines.append("- None")
    return "\n".join(lines)


def _ai_launch_customer_communication_delivery_send_authorization_unblock_status(
    monitor: dict[str, Any],
) -> str:
    if monitor["status"] in {"held", "blocked"}:
        return "blocked_authorization"
    if monitor["status"] == "ready":
        return "ready_to_send"
    return "needs_review"


def _ai_launch_customer_communication_delivery_send_authorization_unblock_tasks(
    monitor: dict[str, Any],
) -> list[dict[str, Any]]:
    owner_by_blocker = {
        "send_authorization_held": "manager",
        "send_requirements_blocked": "advisor",
        "send_blocked_reasons_present": "compliance",
        "payload_not_customer_facing": "advisor",
    }
    action_by_blocker = {
        "send_authorization_held": "clear_send_authorization_hold",
        "send_requirements_blocked": "clear_send_requirements",
        "send_blocked_reasons_present": "resolve_send_blocked_reasons",
        "payload_not_customer_facing": "prepare_customer_facing_payload",
    }
    return [
        {
            "owner": owner_by_blocker.get(item["blocker"], "advisor"),
            "action": action_by_blocker.get(item["blocker"], "clear_send_authorization_blocker"),
            "status": "pending",
            "blocker": item["blocker"],
            "severity": item["severity"],
            "detail": item["resolution"],
        }
        for item in monitor["blockers"]
    ]


def _ai_launch_customer_communication_delivery_send_authorization_unblock_gates(
    monitor: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "gate": "authorization_decision",
            "status": "blocked" if monitor["source_authorization_decision"]["decision"] == "do_not_send" else "ready",
            "target": "send_after_advisor_review",
            "current_value": monitor["source_authorization_decision"]["decision"],
        },
        {
            "gate": "send_requirements",
            "status": monitor["requirements_status"]["status"],
            "target": "met",
            "current_value": monitor["requirements_status"]["status"],
            "blocked_requirements": monitor["requirements_status"]["blocked_requirements"],
        },
        {
            "gate": "payload_customer_facing",
            "status": "ready" if monitor["payload_status"]["customer_facing"] else "blocked",
            "target": True,
            "current_value": monitor["payload_status"]["customer_facing"],
        },
    ]


def _ai_launch_customer_communication_delivery_send_authorization_unblock_exit_criteria(
    monitor: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "criterion": "authorization_not_held",
            "target": "authorized",
            "current_value": monitor["source_authorization_status"],
            "status": "blocked" if monitor["source_authorization_status"] == "hold_send" else "met",
        },
        {
            "criterion": "send_requirements_met",
            "target": "met",
            "current_value": monitor["requirements_status"]["status"],
            "status": "blocked" if monitor["requirements_status"]["status"] != "met" else "met",
        },
        {
            "criterion": "payload_customer_facing",
            "target": True,
            "current_value": monitor["payload_status"]["customer_facing"],
            "status": "blocked" if not monitor["payload_status"]["customer_facing"] else "met",
        },
    ]


def _ai_launch_customer_communication_delivery_send_authorization_unblock_recheck_plan(
    monitor: dict[str, Any],
) -> dict[str, str]:
    if monitor["status"] in {"held", "blocked"}:
        return {
            "owner": "manager",
            "action": "recheck_send_authorization_after_requirements_clear",
            "status": "waiting",
            "detail": "Re-run send authorization after requirements, blocked reasons, and payload exposure clear.",
        }
    return {
        "owner": "advisor",
        "action": "prepare_customer_send",
        "status": "ready",
        "detail": "Authorization gates are clear enough to prepare customer communication send.",
    }


def _ai_improvement_launch_customer_communication_delivery_send_authorization_unblock_plan_markdown(
    plan: dict[str, Any],
) -> str:
    lines = [
        "# AI Improvement Launch Customer Communication Delivery Send Authorization Unblock Plan",
        f"- Experiment: {plan['experiment_id']}",
        f"- Plan status: {plan['plan_status']}",
        f"- Risk level: {plan['risk_level']}",
        f"- Immediate action: {plan['immediate_action']['action']}",
        "",
        "## Unblock Tasks",
    ]
    if plan["unblock_tasks"]:
        lines.extend(f"- {item['owner']}: {item['action']} - {item['detail']}" for item in plan["unblock_tasks"])
    else:
        lines.append("- None")
    lines.extend(["", "## Exit Criteria"])
    lines.extend(
        f"- {item['criterion']}: {item['current_value']} -> {item['target']} ({item['status']})"
        for item in plan["exit_criteria"]
    )
    return "\n".join(lines)


def _ai_launch_customer_communication_delivery_send_authorization_unblock_verification_results(
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    results = []
    results.extend(
        {
            "check": f"authorization_gate:{item['gate']}",
            "status": "passed" if item["status"] in {"ready", "met"} else "failed",
            "current_value": item["current_value"],
            "target": item["target"],
            "detail": f"Authorization gate {item['gate']} is {item['status']}.",
        }
        for item in plan["authorization_gates"]
    )
    results.extend(
        {
            "check": f"exit_criterion:{item['criterion']}",
            "status": "passed" if item["status"] == "met" else "failed",
            "current_value": item["current_value"],
            "target": item["target"],
            "detail": f"Exit criterion {item['criterion']} is {item['status']}.",
        }
        for item in plan["exit_criteria"]
    )
    results.extend(
        {
            "check": f"unblock_task:{item['action']}",
            "status": "passed" if item["status"] in {"done", "complete", "completed"} else "failed",
            "current_value": item["status"],
            "target": "completed",
            "detail": item["detail"],
        }
        for item in plan["unblock_tasks"]
    )
    return results


def _ai_launch_customer_communication_delivery_send_authorization_unblock_verification_status(
    verification_results: list[dict[str, Any]],
) -> str:
    if any(item["status"] == "failed" for item in verification_results):
        return "failed"
    return "passed"


def _ai_launch_customer_communication_delivery_send_authorization_unblock_failed_checks(
    verification_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [item for item in verification_results if item["status"] == "failed"]


def _ai_launch_customer_communication_delivery_send_authorization_unblock_required_follow_up(
    plan: dict[str, Any],
    verification_results: list[dict[str, Any]],
) -> list[dict[str, str]]:
    failed = _ai_launch_customer_communication_delivery_send_authorization_unblock_failed_checks(
        verification_results
    )
    if not failed:
        return [
            {
                "owner": "advisor",
                "action": "prepare_customer_send",
                "detail": "All send authorization unblock verification checks passed.",
            }
        ]
    return [
        {
            "owner": plan["immediate_action"].get("owner", "advisor"),
            "action": plan["immediate_action"]["action"],
            "detail": failed[0]["detail"],
        }
    ]


def _ai_launch_customer_communication_delivery_send_authorization_unblock_verification_next_action(
    plan: dict[str, Any],
    verification_results: list[dict[str, Any]],
) -> dict[str, str]:
    follow_up = _ai_launch_customer_communication_delivery_send_authorization_unblock_required_follow_up(
        plan,
        verification_results,
    )
    return follow_up[0]


def _ai_improvement_launch_customer_communication_delivery_send_authorization_unblock_verification_report_markdown(
    report: dict[str, Any],
) -> str:
    lines = [
        "# AI Improvement Launch Customer Communication Delivery Send Authorization Unblock Verification Report",
        f"- Experiment: {report['experiment_id']}",
        f"- Verification status: {report['verification_status']}",
        f"- Risk level: {report['risk_level']}",
        f"- Next action: {report['next_action']['action']}",
        "",
        "## Failed Checks",
    ]
    if report["failed_checks"]:
        lines.extend(
            f"- {item['check']}: {item['current_value']} -> {item['target']}" for item in report["failed_checks"]
        )
    else:
        lines.append("- None")
    return "\n".join(lines)


def _ai_launch_customer_communication_delivery_send_readiness_blockers(
    verification: dict[str, Any],
) -> list[dict[str, Any]]:
    blockers = []
    if verification["verification_status"] != "passed":
        blockers.append(
            {
                "blocker": "send_authorization_unblock_verification_failed",
                "severity": "high",
                "count": len(verification["failed_checks"]),
                "resolution": "Complete failed authorization unblock checks before preparing a customer send.",
            }
        )
    if verification["source_authorization_status"] != "authorized":
        blockers.append(
            {
                "blocker": "send_not_authorized",
                "severity": "high",
                "resolution": verification["source_authorization_decision"]["rationale"],
            }
        )
    if verification["source_customer_claim_status"] == "not_claimable":
        blockers.append(
            {
                "blocker": "customer_claim_not_supported",
                "severity": "high",
                "resolution": "Withhold customer-facing claims until value proof and send authorization are clear.",
            }
        )
    if verification["source_delivery_status"] == "withheld":
        blockers.append(
            {
                "blocker": "delivery_withheld",
                "severity": "medium",
                "resolution": "Keep delivery withheld until send readiness is confirmed.",
            }
        )
    return blockers


def _ai_launch_customer_communication_delivery_send_readiness_status(
    verification: dict[str, Any],
    blockers: list[dict[str, Any]],
) -> str:
    if any(item["severity"] == "high" for item in blockers):
        return "not_ready"
    if verification["verification_status"] == "passed":
        return "ready_for_advisor_send"
    return "needs_review"


def _ai_launch_customer_communication_delivery_send_readiness_risk_level(
    verification: dict[str, Any],
    blockers: list[dict[str, Any]],
) -> str:
    if verification["risk_level"] == "high" or any(item["severity"] == "high" for item in blockers):
        return "high"
    if blockers:
        return "medium"
    return "low"


def _ai_launch_customer_communication_delivery_send_readiness_gate(
    verification: dict[str, Any],
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    if blockers:
        return {
            "status": "blocked",
            "decision": "do_not_send",
            "blocked_count": len(blockers),
            "detail": "Customer communication send is blocked until authorization unblock verification passes.",
        }
    return {
        "status": "ready",
        "decision": "send_after_advisor_review",
        "blocked_count": 0,
        "detail": "Send authorization is clear enough for final advisor review.",
    }


def _ai_launch_customer_communication_delivery_send_readiness_customer_claim(
    verification: dict[str, Any],
) -> dict[str, Any]:
    if verification["verification_status"] != "passed" or verification["source_customer_claim_status"] == "not_claimable":
        return {
            "status": "withheld",
            "customer_facing": False,
            "claim": "",
            "detail": "Customer-facing value claim is withheld until send readiness clears.",
        }
    return {
        "status": "ready",
        "customer_facing": True,
        "claim": "Cerebral Insights has a reviewed improvement ready for advisor-led customer communication.",
        "detail": "Claim is ready for final advisor review before send.",
    }


def _ai_launch_customer_communication_delivery_send_readiness_advisor_review(
    verification: dict[str, Any],
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    if blockers:
        return {
            "status": "required",
            "owner": verification["next_action"].get("owner", "advisor"),
            "action": verification["next_action"]["action"],
            "detail": "Resolve readiness blockers before any customer-facing send.",
        }
    return {
        "status": "ready",
        "owner": "advisor",
        "action": "review_and_prepare_customer_send",
        "detail": "Review customer copy and send channel before delivery.",
    }


def _ai_improvement_launch_customer_communication_delivery_send_readiness_packet_markdown(
    packet: dict[str, Any],
) -> str:
    lines = [
        "# AI Improvement Launch Customer Communication Delivery Send Readiness Packet",
        f"- Experiment: {packet['experiment_id']}",
        f"- Readiness status: {packet['readiness_status']}",
        f"- Send decision: {packet['send_gate']['decision']}",
        f"- Immediate action: {packet['immediate_action']['action']}",
        "",
        "## Blockers",
    ]
    if packet["blockers"]:
        lines.extend(f"- {item['blocker']} ({item['severity']}): {item['resolution']}" for item in packet["blockers"])
    else:
        lines.append("- None")
    return "\n".join(lines)


def _ai_launch_customer_communication_delivery_send_readiness_review_decision(
    readiness: dict[str, Any],
) -> dict[str, str]:
    if readiness["send_gate"]["decision"] == "do_not_send" or readiness["readiness_status"] == "not_ready":
        return {
            "status": "hold_send",
            "decision": "do_not_send",
            "rationale": readiness["send_gate"]["detail"],
        }
    if readiness["readiness_status"] == "ready_for_advisor_send":
        return {
            "status": "approved_for_review",
            "decision": "send_after_advisor_review",
            "rationale": "Send readiness is clear enough for final advisor review.",
        }
    return {
        "status": "needs_review",
        "decision": "hold_for_review",
        "rationale": "Send readiness needs advisor review before customer-facing delivery.",
    }


def _ai_launch_customer_communication_delivery_send_readiness_review_required_approvals(
    readiness: dict[str, Any],
) -> list[dict[str, str]]:
    status = "blocked" if readiness["send_gate"]["status"] == "blocked" else "required"
    approvals = [
        {
            "approval": "advisor_final_review",
            "owner": "advisor",
            "status": status,
            "detail": readiness["advisor_review"]["detail"],
        }
    ]
    if readiness["risk_level"] == "high":
        approvals.append(
            {
                "approval": "manager_send_release",
                "owner": "manager",
                "status": "blocked" if readiness["send_gate"]["status"] == "blocked" else "required",
                "detail": "Manager release is required for high-risk customer communication send.",
            }
        )
    return approvals


def _ai_launch_customer_communication_delivery_send_readiness_review_blockers(
    readiness: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "blocker": item["blocker"],
            "severity": item["severity"],
            "resolution": item["resolution"],
        }
        for item in readiness["blockers"]
    ]


def _ai_launch_customer_communication_delivery_send_readiness_review_payload(
    readiness: dict[str, Any],
    send_decision: dict[str, str],
) -> dict[str, Any]:
    if send_decision["decision"] != "send_after_advisor_review":
        return {
            "status": "withheld",
            "customer_facing": False,
            "subject": "",
            "body": "",
            "detail": "Customer payload is withheld while send readiness review is holding the send.",
        }
    return {
        "status": "ready_for_review",
        "customer_facing": True,
        "subject": "Cerebral Insights improvement update",
        "body": readiness["customer_claim"]["claim"],
        "detail": "Payload is ready for final advisor review before send.",
    }


def _ai_improvement_launch_customer_communication_delivery_send_readiness_review_packet_markdown(
    packet: dict[str, Any],
) -> str:
    lines = [
        "# AI Improvement Launch Customer Communication Delivery Send Readiness Review Packet",
        f"- Experiment: {packet['experiment_id']}",
        f"- Review status: {packet['review_status']}",
        f"- Send decision: {packet['send_decision']['decision']}",
        f"- Advisor next action: {packet['advisor_next_action']['action']}",
        "",
        "## Send Blockers",
    ]
    if packet["send_blockers"]:
        lines.extend(
            f"- {item['blocker']} ({item['severity']}): {item['resolution']}" for item in packet["send_blockers"]
        )
    else:
        lines.append("- None")
    return "\n".join(lines)


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0
    return round(numerator / denominator, 3)


def _safe_json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def _delivery_dashboard_records(
    conn: sqlite3.Connection,
    owner_id: str,
    status: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT *
        FROM advisor_outreach_delivery_records
        WHERE owner_id = ? AND status = ?
        ORDER BY updated_at DESC, delivery_id DESC
        LIMIT ?
        """,
        (owner_id, status, limit),
    ).fetchall()
    return [_delivery_record_summary(dict(row)) for row in rows]


def _delivered_without_outcome_count(conn: sqlite3.Connection, owner_id: str) -> int:
    if not table_exists(conn, "advisor_outreach_delivery_outcomes"):
        return 0
    row = conn.execute(
        """
        SELECT COUNT(*) AS delivered_without_outcome_count
        FROM advisor_outreach_delivery_records r
        LEFT JOIN advisor_outreach_delivery_outcomes o
            ON o.delivery_id = r.delivery_id AND o.owner_id = r.owner_id
        WHERE r.owner_id = ? AND r.status = 'delivered' AND o.outcome_id IS NULL
        """,
        (owner_id,),
    ).fetchone()
    return int(row["delivered_without_outcome_count"] or 0)


def _delivered_without_outcome_records(
    conn: sqlite3.Connection,
    owner_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    if not table_exists(conn, "advisor_outreach_delivery_outcomes"):
        return []
    rows = conn.execute(
        """
        SELECT r.*
        FROM advisor_outreach_delivery_records r
        LEFT JOIN advisor_outreach_delivery_outcomes o
            ON o.delivery_id = r.delivery_id AND o.owner_id = r.owner_id
        WHERE r.owner_id = ? AND r.status = 'delivered' AND o.outcome_id IS NULL
        ORDER BY r.delivered_at DESC, r.delivery_id DESC
        LIMIT ?
        """,
        (owner_id, limit),
    ).fetchall()
    return [_delivery_record_summary(dict(row)) for row in rows]


def _stale_delivery_dashboard_records(
    conn: sqlite3.Connection,
    owner_id: str,
    stale_before: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT *
        FROM advisor_outreach_delivery_records
        WHERE owner_id = ? AND status = 'ready' AND updated_at <= ?
        ORDER BY updated_at ASC, delivery_id ASC
        LIMIT ?
        """,
        (owner_id, stale_before, limit),
    ).fetchall()
    return [_delivery_record_summary(dict(row)) for row in rows]


def _approved_drafts_without_delivery_count(conn: sqlite3.Connection, owner_id: str) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS approved_without_delivery_count
        FROM advisor_outreach_drafts d
        LEFT JOIN advisor_outreach_delivery_records r
            ON r.draft_id = d.draft_id AND r.owner_id = d.owner_id
        WHERE d.owner_id = ? AND d.status = 'approved' AND r.delivery_id IS NULL
        """,
        (owner_id,),
    ).fetchone()
    return int(row["approved_without_delivery_count"] or 0)


def _approved_drafts_without_delivery(
    conn: sqlite3.Connection,
    owner_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            d.draft_id,
            d.owner_id,
            d.queue_id,
            d.task_id,
            d.status,
            d.selection,
            d.subject,
            d.reviewer,
            d.reviewed_at,
            d.created_at,
            d.updated_at
        FROM advisor_outreach_drafts d
        LEFT JOIN advisor_outreach_delivery_records r
            ON r.draft_id = d.draft_id AND r.owner_id = d.owner_id
        WHERE d.owner_id = ? AND d.status = 'approved' AND r.delivery_id IS NULL
        ORDER BY d.updated_at DESC, d.draft_id DESC
        LIMIT ?
        """,
        (owner_id, limit),
    ).fetchall()
    return [_draft_summary(dict(row)) for row in rows]


def _delivery_dashboard_recommendation(
    delivered_without_outcome: list[dict[str, Any]],
    stale_ready_deliveries: list[dict[str, Any]],
    ready_deliveries: list[dict[str, Any]],
    approved_without_delivery: list[dict[str, Any]],
) -> dict[str, Any]:
    if delivered_without_outcome:
        delivery = delivered_without_outcome[0]
        return {
            "type": "record_delivery_outcome",
            "delivery_id": delivery["delivery_id"],
            "draft_id": delivery["draft_id"],
            "action": f"Record the customer outcome for delivered outreach packet {delivery['delivery_id']}.",
        }
    if stale_ready_deliveries:
        delivery = stale_ready_deliveries[0]
        return {
            "type": "review_stale_ready_packet",
            "delivery_id": delivery["delivery_id"],
            "draft_id": delivery["draft_id"],
            "action": f"Review ready delivery packet {delivery['delivery_id']} because it is older than the stale threshold.",
        }
    if ready_deliveries:
        delivery = ready_deliveries[0]
        return {
            "type": "deliver_ready_packet",
            "delivery_id": delivery["delivery_id"],
            "draft_id": delivery["draft_id"],
            "action": f"Deliver ready outreach packet {delivery['delivery_id']} and mark it delivered after customer handoff.",
        }
    if approved_without_delivery:
        draft = approved_without_delivery[0]
        return {
            "type": "build_delivery_packet",
            "draft_id": draft["draft_id"],
            "action": f"Build and save a delivery packet for approved outreach draft {draft['draft_id']}.",
        }
    return {
        "type": "none",
        "action": "No delivery action is waiting for this owner.",
    }


def _delivery_dashboard_markdown(dashboard: dict[str, Any]) -> str:
    summary = dashboard["summary"]
    recommendation = dashboard["top_recommendation"]
    lines = [
        "# Outreach Delivery Dashboard",
        f"- Owner: {dashboard['owner_id']}",
        f"- Ready: {summary['ready_count']}",
        f"- Delivered: {summary['delivered_count']}",
        f"- Void: {summary['void_count']}",
        f"- Stale ready: {summary['stale_ready_count']} after {summary['stale_after_days']} days",
        f"- Delivered without outcome: {summary['delivered_without_outcome_count']}",
        f"- Approved drafts needing packets: {summary['approved_without_delivery_count']}",
        "",
        "## Next action",
        f"- {recommendation['action']}",
        "",
        "## Ready packets",
    ]
    ready = dashboard["ready_deliveries"]
    if ready:
        lines.extend([f"- Delivery {row['delivery_id']} from draft {row['draft_id']} updated {row['updated_at']}" for row in ready])
    else:
        lines.append("- None")
    return "\n".join(lines)


def _normalize_outcome_type(outcome_type: str | None, response_text: str | None) -> str:
    if outcome_type:
        normalized = outcome_type.strip().lower()
        if normalized not in VALID_OUTCOME_TYPES:
            raise ValueError(f"Outcome type must be one of: {', '.join(sorted(VALID_OUTCOME_TYPES))}")
        return normalized
    return _infer_outcome_type(response_text)


def _infer_outcome_type(response_text: str | None) -> str:
    text = (response_text or "").strip().lower()
    if not text:
        return "no_response"
    if any(marker in text for marker in ("not interested", "no thanks", "declined", "unsubscribe", "stop contacting")):
        return "not_interested"
    if any(marker in text for marker in ("meeting", "call", "scheduled", "booked", "calendar")):
        return "meeting_scheduled"
    if any(marker in text for marker in ("more information", "more info", "details", "clarify", "question", "explain", "?")):
        return "needs_more_information"
    if any(marker in text for marker in ("interested", "proceed", "sounds good", "yes", "approved")):
        return "interested"
    if any(marker in text for marker in ("resolved", "done", "completed", "handled")):
        return "resolved"
    return "other"


def _outcome_customer_signal(outcome_type: str) -> str:
    if outcome_type in {"interested", "meeting_scheduled", "resolved"}:
        return "positive"
    if outcome_type == "not_interested":
        return "negative"
    if outcome_type == "no_response":
        return "no_response"
    return "neutral"


def _outcome_next_action(
    outcome_type: str,
    delivery: dict[str, Any],
    follow_up_due_at: str | None,
) -> dict[str, Any]:
    base = {
        "delivery_id": delivery["delivery_id"],
        "draft_id": delivery["draft_id"],
        "task_id": delivery["task_id"],
        "follow_up_due_at": follow_up_due_at,
    }
    if outcome_type == "meeting_scheduled":
        return {
            **base,
            "action_type": "prepare_meeting",
            "action": "Prepare a meeting agenda using the approved outreach packet and current portfolio context.",
        }
    if outcome_type == "interested":
        return {
            **base,
            "action_type": "send_follow_up",
            "action": "Send a concise follow-up with next steps and any required compliance disclosure.",
        }
    if outcome_type == "needs_more_information":
        return {
            **base,
            "action_type": "answer_question",
            "action": "Answer the customer's open question with cited local evidence before asking for a decision.",
        }
    if outcome_type == "not_interested":
        return {
            **base,
            "action_type": "pause_outreach",
            "action": "Pause active outreach for this topic and preserve the decline reason for future suitability checks.",
        }
    if outcome_type == "no_response":
        return {
            **base,
            "action_type": "schedule_nudge",
            "action": "Schedule a low-friction follow-up nudge or mark the thread dormant if the cadence limit is reached.",
        }
    if outcome_type == "resolved":
        return {
            **base,
            "action_type": "close_loop",
            "action": "Close the outreach loop and retain the outcome for future customer-intent ranking.",
        }
    return {
        **base,
        "action_type": "review_response",
        "action": "Review the response and choose the next compliant follow-up action.",
    }


def _outcome_source_delivery(delivery: dict[str, Any]) -> dict[str, Any]:
    return {
        "delivery_id": delivery["delivery_id"],
        "draft_id": delivery["draft_id"],
        "owner_id": delivery["owner_id"],
        "queue_id": delivery["queue_id"],
        "task_id": delivery["task_id"],
        "status": delivery["status"],
        "delivered_by": delivery["delivered_by"],
        "delivered_at": delivery["delivered_at"],
    }


def _outcome_markdown(
    delivery: dict[str, Any],
    outcome_type: str,
    customer_signal: str,
    next_action: dict[str, Any],
    response_text: str | None,
) -> str:
    lines = [
        "# Outreach Delivery Outcome",
        f"- Delivery: {delivery['delivery_id']}",
        f"- Draft: {delivery['draft_id']}",
        f"- Outcome: {outcome_type}",
        f"- Customer signal: {customer_signal}",
        f"- Next action: {next_action['action']}",
    ]
    if response_text:
        lines.extend(["", "## Advisor note", response_text.strip()])
    return "\n".join(lines)


def _issue(severity: str, code: str, message: str, phrase: str | None = None) -> dict[str, Any]:
    issue = {"severity": severity, "code": code, "message": message}
    if phrase:
        issue["phrase"] = phrase
    return issue


def _has_disclosure_language(text: str) -> bool:
    markers = ("review", "confirm", "disclosure", "not final", "final guidance", "data check", "data gap")
    return any(marker in text for marker in markers)


def _risk_level(issues: list[dict[str, Any]]) -> str:
    severities = {issue["severity"] for issue in issues}
    if "critical" in severities:
        return "critical"
    if "high" in severities:
        return "high"
    if "medium" in severities:
        return "medium"
    return "low"


def _compliance_markdown(
    draft: dict[str, Any],
    risk_level: str,
    recommendation: str,
    issues: list[dict[str, Any]],
    passed_checks: list[str],
) -> str:
    lines = [
        f"# Outreach Compliance Review: draft {draft['draft_id']}",
        "",
        f"- Risk level: {risk_level}",
        f"- Recommendation: {recommendation}",
        "",
        "## Issues",
    ]
    if issues:
        lines.extend(f"- [{issue['severity']}] {issue['message']}" for issue in issues)
    else:
        lines.append("- No issues found.")
    lines.extend(["", "## Passed Checks"])
    lines.extend(f"- {check}" for check in passed_checks)
    return "\n".join(lines)


def _delivery_packet_markdown(packet: dict[str, Any]) -> str:
    lines = [
        f"# Advisor Outreach Delivery Packet: draft {packet['draft_id']}",
        "",
        f"- Owner: {packet['owner_id']}",
        f"- Delivery status: {packet['delivery_status']}",
        f"- Compliance risk: {packet['compliance_review']['risk_level']}",
        f"- Approved by: {packet['approval_evidence'].get('reviewer') or 'unknown'}",
        "",
        f"Subject: {packet['customer_email']['subject']}",
        "",
        packet["customer_email"]["body"],
        "",
        "## Meeting Agenda",
    ]
    for section in packet["meeting_agenda"]:
        lines.append(f"- {section['section']}: {'; '.join(section['items'])}")
    return "\n".join(lines)


def _select_task(queue: dict[str, Any], task_id: str | None) -> dict[str, Any]:
    tasks = queue.get("tasks", [])
    if task_id:
        for task in tasks:
            if task["task_id"] == task_id:
                return task
        raise MarketRecordNotFound(f"No advisor action queue task found for id {task_id}")
    for status in ("open", "blocked", "deferred"):
        for task in tasks:
            if task["status"] == status:
                return task
    raise MarketRecordNotFound(f"No actionable tasks found for queue {queue['queue_id']}")


def _customer_email(owner_id: str, queue: dict[str, Any], task: dict[str, Any], followup: dict[str, Any]) -> dict[str, str]:
    subject = f"Follow-up on your Cerebral Insights review: {task['title']}"
    body_lines = [
        f"Hi {owner_id},",
        "",
        "I reviewed your latest Cerebral Insights queue and the next item worth discussing is:",
        f"- {task['title']}",
        "",
        f"Why it matters: {task['rationale']}",
        f"Suggested completion point: {task['completion_criteria']}",
    ]
    reference_email = followup.get("customer_email", {})
    if reference_email.get("body"):
        body_lines.extend(["", "Context from the latest review:", _excerpt(reference_email["body"])])
    if queue.get("focus"):
        body_lines.extend(["", f"Focus area: {queue['focus']}"])
    body_lines.extend(["", "Please review this before we treat it as final guidance."])
    return {"subject": subject, "body": "\n".join(body_lines)}


def _meeting_agenda(task: dict[str, Any], followup: dict[str, Any]) -> list[dict[str, Any]]:
    agenda = [
        {"section": "Task to review", "items": [task["title"], task["rationale"]]},
        {"section": "Completion decision", "items": [task["completion_criteria"]]},
    ]
    source_agenda = followup.get("meeting_agenda", [])
    if source_agenda:
        agenda.append({"section": "Source context", "items": _flatten_agenda(source_agenda)[:3]})
    return agenda


def _guardrails(task: dict[str, Any], followup: dict[str, Any]) -> dict[str, Any]:
    source = followup.get("compliance_guardrails", {})
    do_say = list(source.get("do_say", []))
    do_not_say = list(source.get("do_not_say", []))
    do_say.append("This is a draft for advisor review before customer delivery.")
    do_not_say.append("Do not send this draft without confirming the task evidence and disclosures.")
    return {
        "do_say": _dedupe(do_say),
        "do_not_say": _dedupe(do_not_say),
        "requires_disclosure": bool(source.get("requires_disclosure")) or task["status"] == "blocked",
        "review_checklist": [
            "Confirm the source evidence is current.",
            "Confirm the language is suitable for the customer relationship.",
            "Confirm blocked tasks are not presented as completed analysis.",
        ],
    }


def _markdown(
    owner_id: str,
    task: dict[str, Any],
    email: dict[str, str],
    agenda: list[dict[str, Any]],
    guardrails: dict[str, Any],
) -> str:
    lines = [
        f"# Advisor Outreach Draft: {owner_id}",
        "",
        f"## Task: {task['title']}",
        "",
        f"Subject: {email['subject']}",
        "",
        email["body"],
        "",
        "## Meeting Agenda",
    ]
    for section in agenda:
        lines.append(f"- {section['section']}: {'; '.join(section['items'])}")
    lines.extend(["", "## Review Checklist"])
    lines.extend(f"- {item}" for item in guardrails["review_checklist"])
    return "\n".join(lines)


def _excerpt(text: str, limit: int = 500) -> str:
    compact = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _flatten_agenda(agenda: list[dict[str, Any]]) -> list[str]:
    items: list[str] = []
    for section in agenda:
        items.extend(str(item) for item in section.get("items", []))
    return items


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
