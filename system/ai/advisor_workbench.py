"""Advisor workbench over saved action queues."""

from __future__ import annotations

import sqlite3
from typing import Any

from system.ai.action_queue import clean_owner_id, require_action_queue_tables


URGENCY_RANK = {"high": 0, "medium": 1, "low": 2, "blocked": 3}
STATUS_RANK = {"open": 0, "blocked": 1, "deferred": 2}


def build_advisor_workbench(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    limit: int = 10,
    include_blocked: bool = True,
) -> dict[str, Any]:
    require_action_queue_tables(conn)
    owner = clean_owner_id(owner_id) if owner_id else None
    rows = _candidate_rows(conn, owner, include_blocked)
    actions = [_action_from_row(dict(row)) for row in rows]
    actions.sort(key=_rank)

    next_actions = [action for action in actions if action["status"] == "open"][:limit]
    blocked_actions = [action for action in actions if action["status"] == "blocked"][:limit]
    summary = _summary(conn, owner)
    recommendation = _recommendation(next_actions, blocked_actions)
    return {
        "kind": "advisor_workbench",
        "owner_id": owner or "all",
        "summary": summary,
        "top_recommendation": recommendation,
        "next_actions": next_actions,
        "blocked_actions": blocked_actions if include_blocked else [],
        "workbench_markdown": _markdown(owner or "all", summary, next_actions, blocked_actions if include_blocked else []),
    }


def _candidate_rows(conn: sqlite3.Connection, owner_id: str | None, include_blocked: bool) -> list[sqlite3.Row]:
    statuses = ["open"]
    if include_blocked:
        statuses.append("blocked")
    placeholders = ", ".join("?" for _ in statuses)
    where = [
        "q.status != 'completed'",
        f"t.status IN ({placeholders})",
    ]
    params: list[Any] = list(statuses)
    if owner_id:
        where.append("q.owner_id = ?")
        params.append(owner_id)
    return conn.execute(
        f"""
        SELECT
            q.queue_id,
            q.owner_id,
            q.title AS queue_title,
            q.focus,
            q.status AS queue_status,
            q.updated_at AS queue_updated_at,
            t.saved_task_id,
            t.task_id,
            t.title AS task_title,
            t.urgency,
            t.status AS task_status,
            t.rationale,
            t.completion_criteria,
            t.notes,
            t.created_at AS task_created_at,
            t.updated_at AS task_updated_at
        FROM advisor_action_queues q
        JOIN advisor_action_queue_tasks t ON t.queue_id = q.queue_id
        WHERE {" AND ".join(where)}
        """,
        params,
    ).fetchall()


def _summary(conn: sqlite3.Connection, owner_id: str | None) -> dict[str, Any]:
    where = ""
    params: list[Any] = []
    if owner_id:
        where = "WHERE owner_id = ?"
        params.append(owner_id)
    queue_row = conn.execute(
        f"""
        SELECT
            COUNT(*) AS queue_count,
            SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) AS open_queue_count,
            SUM(CASE WHEN status = 'blocked' THEN 1 ELSE 0 END) AS blocked_queue_count,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_queue_count
        FROM advisor_action_queues
        {where}
        """,
        params,
    ).fetchone()
    task_row = conn.execute(
        f"""
        SELECT
            COUNT(*) AS task_count,
            SUM(CASE WHEN t.status = 'open' THEN 1 ELSE 0 END) AS open_task_count,
            SUM(CASE WHEN t.status = 'blocked' THEN 1 ELSE 0 END) AS blocked_task_count,
            SUM(CASE WHEN t.status = 'completed' THEN 1 ELSE 0 END) AS completed_task_count
        FROM advisor_action_queue_tasks t
        JOIN advisor_action_queues q ON q.queue_id = t.queue_id
        {where.replace("owner_id", "q.owner_id")}
        """,
        params,
    ).fetchone()
    return {
        "queue_count": _int(queue_row["queue_count"]),
        "open_queue_count": _int(queue_row["open_queue_count"]),
        "blocked_queue_count": _int(queue_row["blocked_queue_count"]),
        "completed_queue_count": _int(queue_row["completed_queue_count"]),
        "task_count": _int(task_row["task_count"]),
        "open_task_count": _int(task_row["open_task_count"]),
        "blocked_task_count": _int(task_row["blocked_task_count"]),
        "completed_task_count": _int(task_row["completed_task_count"]),
    }


def _action_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "queue_id": row["queue_id"],
        "owner_id": row["owner_id"],
        "queue_title": row["queue_title"],
        "focus": row["focus"],
        "queue_status": row["queue_status"],
        "queue_updated_at": row["queue_updated_at"],
        "saved_task_id": row["saved_task_id"],
        "task_id": row["task_id"],
        "title": row["task_title"],
        "urgency": row["urgency"],
        "status": row["task_status"],
        "rationale": row["rationale"],
        "completion_criteria": row["completion_criteria"],
        "notes": row["notes"],
        "task_created_at": row["task_created_at"],
        "task_updated_at": row["task_updated_at"],
    }


def _rank(action: dict[str, Any]) -> tuple[int, int, str, int]:
    return (
        STATUS_RANK.get(action["status"], 9),
        URGENCY_RANK.get(action["urgency"], 9),
        action["task_created_at"],
        action["saved_task_id"],
    )


def _recommendation(next_actions: list[dict[str, Any]], blocked_actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if next_actions:
        action = next_actions[0]
        return {
            "type": "next_action",
            "owner_id": action["owner_id"],
            "queue_id": action["queue_id"],
            "task_id": action["task_id"],
            "title": action["title"],
            "why": action["rationale"],
        }
    if blocked_actions:
        action = blocked_actions[0]
        return {
            "type": "unblock",
            "owner_id": action["owner_id"],
            "queue_id": action["queue_id"],
            "task_id": action["task_id"],
            "title": action["title"],
            "why": "No open tasks are available; resolve the highest-priority blocker first.",
        }
    return None


def _markdown(
    owner_id: str,
    summary: dict[str, Any],
    next_actions: list[dict[str, Any]],
    blocked_actions: list[dict[str, Any]],
) -> str:
    lines = [
        f"# Advisor Workbench: {owner_id}",
        "",
        f"- Open tasks: {summary['open_task_count']}",
        f"- Blocked tasks: {summary['blocked_task_count']}",
        f"- Completed tasks: {summary['completed_task_count']}",
        "",
        "## Next Actions",
    ]
    if next_actions:
        for action in next_actions:
            lines.append(f"- {action['owner_id']} / queue {action['queue_id']}: {action['title']} ({action['urgency']})")
    else:
        lines.append("- No open tasks.")
    if blocked_actions:
        lines.extend(["", "## Blockers"])
        for action in blocked_actions:
            lines.append(f"- {action['owner_id']} / queue {action['queue_id']}: {action['title']}")
    return "\n".join(lines)


def _int(value: Any) -> int:
    return int(value or 0)
