"""Customer-ready follow-up pack generated from the AI morning brief."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from system.ai.morning_brief import build_owner_morning_brief


def build_advisor_followup(
    conn: sqlite3.Connection,
    owner_id: str | None,
    index_dir: Path,
    focus: str | None = None,
    max_items: int = 2,
    evidence_limit: int = 1,
    persist_screeners: bool = False,
) -> dict[str, Any]:
    morning = build_owner_morning_brief(
        conn,
        owner_id,
        index_dir,
        focus=focus,
        max_items=max_items,
        evidence_limit=evidence_limit,
        persist_screeners=persist_screeners,
    )
    customer_name = morning["owner_id"]
    talking_points = _talking_points(morning)
    compliance = _compliance_guardrails(morning)
    advisor_checklist = _advisor_checklist(morning)
    return {
        "kind": "advisor_followup_pack",
        "owner_id": morning["owner_id"],
        "source_brief": morning,
        "customer_email": _customer_email(customer_name, talking_points, morning),
        "meeting_agenda": _meeting_agenda(talking_points, advisor_checklist),
        "advisor_checklist": advisor_checklist,
        "compliance_guardrails": compliance,
        "followup_markdown": _markdown(customer_name, talking_points, advisor_checklist, compliance),
    }


def _talking_points(morning: dict[str, Any]) -> list[dict[str, Any]]:
    points = [
        {
            "rank": idx + 1,
            "topic": item["type"],
            "message": item["message"],
            "supporting_action": _action_for_priority(item),
        }
        for idx, item in enumerate(morning.get("priorities", [])[:5])
    ]
    if not points:
        points.append(
            {
                "rank": 1,
                "topic": "status",
                "message": "No urgent customer action detected from available local data.",
                "supporting_action": "Offer a standard portfolio or watchlist review.",
            }
        )
    return points


def _action_for_priority(priority: dict[str, Any]) -> str:
    kind = priority.get("type")
    if kind == "freshness":
        return "Refresh or qualify stale data before discussing specific instruments."
    if kind == "watchlist_alerts":
        return "Open the triggered alert context and confirm whether it needs customer outreach."
    if kind == "portfolio_risk":
        return "Prepare concentration and diversification context before the customer conversation."
    if kind == "opportunity":
        return "Review screener matches and shortlist names that also have fresh evidence."
    if kind == "data_gaps":
        return "Resolve the listed gaps before making customer-facing claims."
    return "Use the morning brief as context and keep claims evidence-backed."


def _advisor_checklist(morning: dict[str, Any]) -> list[str]:
    checklist = [
        "Confirm freshness and quality status before sending.",
        "Review cited evidence for each ticker mentioned.",
        "Remove any claim that is not supported by local data.",
    ]
    if morning.get("data_gaps"):
        checklist.insert(0, f"Resolve or disclose {len(morning['data_gaps'])} data gaps.")
    if morning.get("portfolio_digests"):
        checklist.append("Check portfolio concentration and missing quote flags.")
    if morning.get("watchlist_digests"):
        checklist.append("Review triggered watchlist alerts and suppress duplicates.")
    if morning.get("screener_digests"):
        checklist.append("Validate screener matches before presenting them as opportunities.")
    return checklist


def _compliance_guardrails(morning: dict[str, Any]) -> dict[str, Any]:
    return {
        "do_say": [
            "This is a data-backed review agenda from local records.",
            "Some items may require freshness or evidence checks before action.",
            "We can review risks, concentration, and watchlist changes together.",
        ],
        "do_not_say": [
            "Do not present the brief as investment advice by itself.",
            "Do not imply guaranteed returns or price targets.",
            "Do not hide missing data, stale data, or failed quality checks.",
        ],
        "requires_disclosure": bool(morning.get("data_gaps")),
    }


def _customer_email(customer_name: str, talking_points: list[dict[str, Any]], morning: dict[str, Any]) -> dict[str, str]:
    subject = "Your market review is ready"
    opening = f"Hi {customer_name},"
    body_lines = [
        opening,
        "",
        "I reviewed your latest Cerebral Insights brief and found a few items worth discussing.",
    ]
    for point in talking_points[:3]:
        body_lines.append(f"- {point['message']}")
    if morning.get("data_gaps"):
        body_lines.append("- A few data checks need to be confirmed before we treat the brief as final.")
    body_lines.extend(
        [
            "",
            "If useful, we can use this as the agenda for a quick review.",
        ]
    )
    return {"subject": subject, "body": "\n".join(body_lines)}


def _meeting_agenda(talking_points: list[dict[str, Any]], checklist: list[str]) -> list[dict[str, Any]]:
    agenda = [
        {"section": "Data quality check", "items": checklist[:2]},
        {"section": "Top customer priorities", "items": [point["message"] for point in talking_points]},
        {"section": "Next actions", "items": [point["supporting_action"] for point in talking_points[:3]]},
    ]
    return agenda


def _markdown(
    customer_name: str,
    talking_points: list[dict[str, Any]],
    checklist: list[str],
    compliance: dict[str, Any],
) -> str:
    lines = [f"# Advisor Follow-up Pack: {customer_name}", "", "## Talking Points"]
    lines.extend(f"- {point['message']}" for point in talking_points)
    lines.append("")
    lines.append("## Advisor Checklist")
    lines.extend(f"- {item}" for item in checklist)
    lines.append("")
    lines.append("## Do Not Say")
    lines.extend(f"- {item}" for item in compliance["do_not_say"])
    return "\n".join(lines)
