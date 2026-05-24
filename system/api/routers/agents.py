from __future__ import annotations

import json
import os
import sqlite3
import urllib.error
import urllib.request
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from system.ai.action_queue import (
    bulk_review_action_queue_task_escalation_notification_delivery_incidents,
    bulk_update_action_queue_tasks,
    bulk_review_action_queue_task_escalations,
    build_action_queue,
    build_action_queue_task_escalation_inbox_notification,
    claim_action_queue_task_escalation_notification_delivery,
    complete_action_queue_task_escalation_notification_delivery_claim,
    execute_action_queue_task_escalation_notification_delivery_incident_action,
    get_action_queue,
    get_action_queue_task_escalation_notification_delivery_incident,
    list_action_queues,
    list_action_queue_task_escalation_notification_deadletters,
    list_action_queue_task_escalation_notification_deadletter_remediations,
    list_action_queue_task_escalation_notification_delivery_claim_releases,
    list_action_queue_task_escalation_notification_delivery_claims,
    list_action_queue_task_escalation_notification_delivery_attempts,
    list_action_queue_task_escalation_notification_delivery_incidents,
    list_action_queue_task_escalation_notification_delivery_queue,
    list_action_queue_task_escalation_notification_delivery_incident_reviews,
    list_action_queue_task_escalation_notifications,
    list_action_queue_task_escalation_inbox,
    list_action_queue_task_escalation_reviews,
    list_action_queue_task_escalations,
    list_action_queue_tasks,
    list_action_queue_task_updates,
    requeue_action_queue_task_escalation_notification_deadletter,
    release_action_queue_task_escalation_notification_delivery_claim,
    renew_action_queue_task_escalation_notification_delivery_claim,
    review_action_queue_task_escalation_notification_delivery_incident,
    review_action_queue_task_escalation,
    summarize_action_queue_task_escalation_notification_delivery_incidents,
    save_action_queue_task_escalation_inbox_notification,
    summarize_action_queues,
    summarize_action_queue_task_escalation_notification_delivery_control,
    summarize_action_queue_task_escalation_notification_delivery,
    summarize_action_queue_task_escalation_notification_delivery_incident_workload,
    summarize_action_queue_task_escalations,
    summarize_action_queue_task_workload,
    update_action_queue_task,
    update_action_queue_task_escalation_notification,
)
from system.ai.advisor_followup import build_advisor_followup
from system.ai.advisor_outreach import (
    build_advisor_outreach_draft,
    build_ai_improvement_backlog,
    build_ai_improvement_experiment_plan,
    build_ai_improvement_experiment_launch_packet,
    build_ai_improvement_experiment_readout,
    build_ai_improvement_adoption_impact_ledger,
    build_ai_improvement_adoption_monitor,
    build_ai_improvement_adoption_playbook,
    build_ai_improvement_release_packet,
    build_ai_improvement_rollout_monitor,
    build_ai_improvement_rollout_readiness,
    build_ai_improvement_scale_decision_packet,
    build_ai_improvement_scale_execution_monitor,
    build_ai_improvement_scale_execution_plan,
    build_ai_improvement_scale_learning_report,
    build_ai_improvement_roadmap_refresh,
    build_ai_improvement_backlog_handoff,
    build_ai_improvement_implementation_kickoff_packet,
    build_ai_improvement_implementation_blocker_resolution_plan,
    build_ai_improvement_implementation_readiness_monitor,
    build_ai_improvement_implementation_unblock_verification_report,
    build_ai_improvement_implementation_qa_review_packet,
    build_ai_improvement_implementation_qa_signoff_report,
    build_ai_improvement_launch_review_packet,
    build_ai_improvement_launch_execution_plan,
    build_ai_improvement_launch_execution_monitor,
    build_ai_improvement_launch_outcome_monitor,
    build_ai_improvement_launch_value_proof_packet,
    build_ai_improvement_launch_customer_communication_packet,
    build_ai_improvement_launch_customer_communication_review_packet,
    build_ai_improvement_launch_customer_communication_delivery_packet,
    build_ai_improvement_launch_customer_communication_delivery_monitor,
    build_ai_improvement_launch_customer_communication_delivery_unblock_plan,
    build_ai_improvement_launch_customer_communication_delivery_unblock_verification_report,
    build_ai_improvement_launch_customer_communication_delivery_send_authorization_packet,
    build_ai_improvement_launch_customer_communication_delivery_send_authorization_monitor,
    build_ai_improvement_launch_customer_communication_delivery_send_authorization_unblock_plan,
    build_ai_improvement_launch_customer_communication_delivery_send_authorization_unblock_verification_report,
    build_ai_improvement_launch_customer_communication_delivery_send_execution_handoff_packet,
    build_ai_improvement_launch_customer_communication_delivery_send_readiness_packet,
    build_ai_improvement_launch_customer_communication_delivery_send_readiness_review_packet,
    build_ai_recommendation_effectiveness_dashboard,
    build_customer_engagement_brief,
    build_customer_engagement_action_queue,
    build_customer_engagement_cadence_dashboard,
    build_customer_engagement_cadence_review,
    build_customer_engagement_task_brief,
    build_customer_engagement_timeline,
    build_customer_intent_action_plan,
    build_customer_intent_dashboard,
    build_customer_intent_followup_draft,
    build_customer_intent_followup_packet,
    build_customer_intent_followup_review,
    build_outreach_compliance_review,
    build_outreach_delivery_dashboard,
    build_outreach_delivery_packet,
    get_outreach_delivery_record,
    get_outreach_delivery_outcome,
    get_outreach_draft,
    list_outreach_compliance_reviews,
    list_outreach_delivery_records,
    list_outreach_delivery_outcomes,
    list_outreach_drafts,
    update_outreach_delivery_record,
    update_outreach_draft_review,
    save_outreach_delivery_outcome,
)
from system.ai.advisor_workbench import build_advisor_workbench
from system.ai.customer_digest import build_portfolio_digest, build_screener_digest, build_watchlist_digest
from system.ai.morning_brief import build_owner_morning_brief
from system.ai.research_brief import build_research_brief
from system.ai.screener_copilot import run_screener_copilot
from system.ai.vector_index import VectorIndexError
from system.analyst_report import build_prompt_payload
from system.api.market_service import MarketDataUnavailable, MarketRecordNotFound

from ..dependencies import get_connection
from ..query_service import (
    fetch_documents,
    fetch_quality_checks,
    fetch_risk_signals,
    latest_completed_run,
)

router = APIRouter(prefix="/agents", tags=["agents"])

DEFAULT_LLAMA_CPP_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_LLAMA_CPP_MODEL = "local-llama"
DEFAULT_LLAMA_CPP_TIMEOUT_SECONDS = 20


class ActionQueueTaskUpdate(BaseModel):
    status: str | None = None
    notes: str | None = None
    assigned_to: str | None = None
    due_at: str | None = None
    updated_by: str | None = None
    update_source: str | None = None


class ActionQueueTaskRef(BaseModel):
    queue_id: int
    task_id: str


class ActionQueueTaskBulkUpdate(BaseModel):
    tasks: list[ActionQueueTaskRef]
    status: str | None = None
    notes: str | None = None
    assigned_to: str | None = None
    due_at: str | None = None
    updated_by: str | None = None
    update_source: str | None = None


class ActionQueueEscalationReview(BaseModel):
    review_status: str
    reviewer: str | None = None
    notes: str | None = None
    snoozed_until: str | None = None


class LocalLlmChatRequest(BaseModel):
    message: str | None = None
    messages: list[dict[str, Any]] | None = None
    product_id: str | None = None
    context: dict[str, Any] | None = None
    max_tokens: int | None = 512
    temperature: float | None = 0.2


class ActionQueueEscalationBulkReview(ActionQueueEscalationReview):
    tasks: list[ActionQueueTaskRef]


class ActionQueueEscalationNotificationLog(BaseModel):
    channel: str | None = None
    recipient: str | None = None
    status: str | None = "prepared"
    idempotency_key: str | None = None


class ActionQueueEscalationNotificationUpdate(BaseModel):
    status: str | None = None
    delivery_notes: str | None = None
    delivered_at: str | None = None


class ActionQueueEscalationNotificationDeliveryClaim(BaseModel):
    claimed_by: str | None = None
    lease_seconds: int | None = 300


class ActionQueueEscalationNotificationDeliveryCompletion(BaseModel):
    claim_token: str | None = None
    status: str | None = None
    delivery_notes: str | None = None
    delivered_at: str | None = None
    retry_after: str | None = None
    max_attempts: int | None = 3


class ActionQueueEscalationNotificationDeliveryClaimRenewal(BaseModel):
    claim_token: str | None = None
    lease_seconds: int | None = 300


class ActionQueueEscalationNotificationDeliveryClaimRelease(BaseModel):
    claim_token: str | None = None
    release_notes: str | None = None
    released_by: str | None = None


class ActionQueueEscalationNotificationDeliveryIncidentReview(BaseModel):
    incident_type: str | None = None
    incident_status: str | None = None
    reviewer: str | None = None
    assigned_to: str | None = None
    notes: str | None = None
    follow_up_at: str | None = None


class ActionQueueEscalationNotificationDeliveryIncidentRef(BaseModel):
    notification_id: int
    incident_type: str


class ActionQueueEscalationNotificationDeliveryIncidentBulkReview(
    ActionQueueEscalationNotificationDeliveryIncidentReview
):
    incidents: list[ActionQueueEscalationNotificationDeliveryIncidentRef]


class ActionQueueEscalationNotificationDeliveryIncidentAction(BaseModel):
    action_id: str
    incident_type: str
    reviewer: str | None = None
    assigned_to: str | None = None
    notes: str | None = None
    follow_up_at: str | None = None
    delivery_notes: str | None = None
    requeued_by: str | None = None
    retry_after: str | None = None
    claim_token: str | None = None
    release_notes: str | None = None
    released_by: str | None = None
    claimed_by: str | None = None
    lease_seconds: int | None = 300


class ActionQueueEscalationNotificationDeadletterRequeue(BaseModel):
    retry_after: str | None = None
    delivery_notes: str | None = None
    requeued_by: str | None = None


class OutreachDraftReviewUpdate(BaseModel):
    status: str
    review_notes: str | None = None
    reviewer: str | None = None
    override_compliance: bool = False


class OutreachDeliveryUpdate(BaseModel):
    status: str
    delivery_notes: str | None = None
    delivered_by: str | None = None


class OutreachDeliveryOutcomeCreate(BaseModel):
    outcome_type: str | None = None
    response_text: str | None = None
    follow_up_due_at: str | None = None
    recorded_by: str | None = None


def _llama_cpp_config() -> dict[str, Any]:
    timeout_raw = os.getenv("LLAMA_CPP_TIMEOUT_SECONDS", str(DEFAULT_LLAMA_CPP_TIMEOUT_SECONDS))
    try:
        timeout = max(1.0, float(timeout_raw))
    except ValueError:
        timeout = float(DEFAULT_LLAMA_CPP_TIMEOUT_SECONDS)
    return {
        "base_url": os.getenv("LLAMA_CPP_BASE_URL", DEFAULT_LLAMA_CPP_BASE_URL).rstrip("/"),
        "model": os.getenv("LLAMA_CPP_MODEL", DEFAULT_LLAMA_CPP_MODEL),
        "timeout_seconds": timeout,
    }


def _request_llama_cpp_json(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    config = _llama_cpp_config()
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"{config['base_url']}/{path.lstrip('/')}",
        data=data,
        method="POST" if payload is not None else "GET",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=config["timeout_seconds"]) as response:
            raw = response.read().decode("utf-8")
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="local_llm_timeout") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, TimeoutError):
            raise HTTPException(status_code=504, detail="local_llm_timeout") from exc
        raise HTTPException(status_code=503, detail="local_llm_unavailable") from exc
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="local_llm_malformed_response") from exc


def _local_llm_messages(payload: LocalLlmChatRequest) -> list[dict[str, str]]:
    if payload.messages:
        return [
            {"role": str(item.get("role") or "user"), "content": str(item.get("content") or "")}
            for item in payload.messages
            if str(item.get("content") or "").strip()
        ]
    message = (payload.message or "").strip()
    if not message:
        return []
    context = payload.context or {}
    product_id = payload.product_id or context.get("product_id") or "general"
    system = (
        "You are Cerebral Insights local llama.cpp fallback. Use only processed, "
        "LLM-friendly Indian equities product payload context. Keep responses factual, "
        "non-advisory, and cite missing evidence as unavailable."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Product: {product_id}\nPrompt: {message}"},
    ]


def _extract_llama_cpp_answer(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise HTTPException(status_code=502, detail="local_llm_malformed_response")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    answer = str(content or "").strip()
    if not answer:
        raise HTTPException(status_code=502, detail="local_llm_empty_response")
    return answer


@router.get("/local-llm-status")
def local_llm_status() -> dict[str, Any]:
    config = _llama_cpp_config()
    try:
        models = _request_llama_cpp_json("/models")
    except HTTPException as exc:
        return {
            "status": "unavailable",
            "provider": "local-llama",
            "base_url": config["base_url"],
            "model": config["model"],
            "detail": exc.detail,
        }
    return {
        "status": "available",
        "provider": "local-llama",
        "base_url": config["base_url"],
        "model": config["model"],
        "models": models.get("data", []),
    }


@router.post("/local-llm-chat")
def local_llm_chat(payload: LocalLlmChatRequest) -> dict[str, Any]:
    messages = _local_llm_messages(payload)
    if not messages:
        raise HTTPException(status_code=422, detail="message_required")
    config = _llama_cpp_config()
    request_payload = {
        "model": config["model"],
        "messages": messages,
        "stream": False,
        "temperature": payload.temperature,
        "max_tokens": payload.max_tokens,
    }
    response = _request_llama_cpp_json("/chat/completions", request_payload)
    return {
        "status": "answer_ready",
        "provider": "local-llama",
        "model": config["model"],
        "answer": _extract_llama_cpp_answer(response),
        "sources": [],
        "usage": response.get("usage", {}),
    }


@router.get("/context/{ticker}")
def agent_context(
    ticker: str,
    limit: int = Query(default=20, ge=1, le=200),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    run = latest_completed_run(conn)
    if not run:
        raise HTTPException(status_code=404, detail="No completed runs found")

    context = {
        "ticker": ticker,
        "run_id": run["run_id"],
        "run_date": run["run_date"],
        "documents": [
            {
                "source": row["source_name"],
                "type": row["file_type"],
                "content": row["content_preview"][:3000],
            }
            for row in fetch_documents(conn, int(run["run_id"]), limit=limit)
        ],
        "risk_signals": fetch_risk_signals(conn, int(run["run_id"]), limit),
        "quality_checks": fetch_quality_checks(conn, int(run["run_id"])),
    }
    return build_prompt_payload(ticker, context)


@router.post("/research-brief/{ticker}")
def agent_research_brief(
    ticker: str,
    request: Request,
    focus: str | None = Query(default=None),
    run_id: int | None = Query(default=None),
    source_name: str | None = Query(default=None),
    evidence_limit: int = Query(default=8, ge=1, le=25),
    auto_build: bool = Query(default=True),
    build_limit: int = Query(default=1000, ge=1, le=100000),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_research_brief(
            conn,
            ticker,
            request.app.state.vector_index_dir,
            focus=focus,
            run_id=run_id,
            source_name=source_name,
            evidence_limit=evidence_limit,
            auto_build=auto_build,
            build_limit=build_limit,
        )
    except VectorIndexError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/portfolio-digest/{portfolio_id}")
def agent_portfolio_digest(
    portfolio_id: int,
    request: Request,
    owner_id: str = Query(default="default"),
    focus: str | None = Query(default=None),
    max_positions: int = Query(default=5, ge=1, le=25),
    evidence_limit: int = Query(default=3, ge=1, le=10),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_portfolio_digest(
            conn,
            portfolio_id,
            owner_id,
            request.app.state.vector_index_dir,
            focus=focus,
            max_positions=max_positions,
            evidence_limit=evidence_limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/watchlist-digest/{watchlist_id}")
def agent_watchlist_digest(
    watchlist_id: int,
    request: Request,
    owner_id: str = Query(default="default"),
    focus: str | None = Query(default=None),
    max_items: int = Query(default=10, ge=1, le=50),
    evidence_limit: int = Query(default=3, ge=1, le=10),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_watchlist_digest(
            conn,
            watchlist_id,
            owner_id,
            request.app.state.vector_index_dir,
            focus=focus,
            max_items=max_items,
            evidence_limit=evidence_limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/screener-digest/{screener_id}")
def agent_screener_digest(
    screener_id: int,
    request: Request,
    owner_id: str = Query(default="default"),
    focus: str | None = Query(default=None),
    max_results: int = Query(default=10, ge=1, le=50),
    evidence_limit: int = Query(default=3, ge=1, le=10),
    persist: bool = Query(default=True),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_screener_digest(
            conn,
            screener_id,
            owner_id,
            request.app.state.vector_index_dir,
            focus=focus,
            max_results=max_results,
            evidence_limit=evidence_limit,
            persist=persist,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/screener-copilot")
def agent_screener_copilot(
    prompt: str = Query(..., min_length=1),
    owner_id: str = Query(default="default"),
    save: bool = Query(default=False),
    name: str | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return run_screener_copilot(
            conn,
            prompt,
            owner_id=owner_id,
            save=save,
            name=name,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/morning-brief")
def agent_morning_brief(
    request: Request,
    owner_id: str = Query(default="default"),
    focus: str | None = Query(default=None),
    max_items: int = Query(default=2, ge=1, le=10),
    evidence_limit: int = Query(default=1, ge=1, le=5),
    persist_screeners: bool = Query(default=False),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    return build_owner_morning_brief(
        conn,
        owner_id,
        request.app.state.vector_index_dir,
        focus=focus,
        max_items=max_items,
        evidence_limit=evidence_limit,
        persist_screeners=persist_screeners,
    )


@router.post("/advisor-followup")
def agent_advisor_followup(
    request: Request,
    owner_id: str = Query(default="default"),
    focus: str | None = Query(default=None),
    max_items: int = Query(default=2, ge=1, le=10),
    evidence_limit: int = Query(default=1, ge=1, le=5),
    persist_screeners: bool = Query(default=False),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    return build_advisor_followup(
        conn,
        owner_id,
        request.app.state.vector_index_dir,
        focus=focus,
        max_items=max_items,
        evidence_limit=evidence_limit,
        persist_screeners=persist_screeners,
    )


@router.get("/advisor-workbench")
def agent_advisor_workbench(
    owner_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    include_blocked: bool = Query(default=True),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_advisor_workbench(conn, owner_id=owner_id, limit=limit, include_blocked=include_blocked)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/advisor-outreach-draft")
def agent_advisor_outreach_draft(
    owner_id: str = Query(default="default"),
    queue_id: int | None = Query(default=None),
    task_id: str | None = Query(default=None),
    include_blocked: bool = Query(default=True),
    save: bool = Query(default=False),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_advisor_outreach_draft(
            conn,
            owner_id=owner_id,
            queue_id=queue_id,
            task_id=task_id,
            include_blocked=include_blocked,
            save=save,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/advisor-outreach-drafts")
def agent_advisor_outreach_drafts(
    owner_id: str = Query(default="default"),
    status: str | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_outreach_drafts(conn, owner_id, status=status, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/advisor-outreach-drafts/{draft_id}")
def agent_saved_advisor_outreach_draft(
    draft_id: int,
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return get_outreach_draft(conn, draft_id, owner_id)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/advisor-outreach-drafts/{draft_id}")
def agent_update_advisor_outreach_draft_review(
    draft_id: int,
    request: OutreachDraftReviewUpdate,
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return update_outreach_draft_review(
            conn,
            draft_id,
            owner_id,
            request.status,
            review_notes=request.review_notes,
            reviewer=request.reviewer,
            override_compliance=request.override_compliance,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/advisor-outreach-drafts/{draft_id}/compliance-review")
def agent_advisor_outreach_compliance_review(
    draft_id: int,
    owner_id: str = Query(default="default"),
    save: bool = Query(default=False),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_outreach_compliance_review(conn, draft_id, owner_id, save=save)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/advisor-outreach-drafts/{draft_id}/compliance-reviews")
def agent_advisor_outreach_compliance_reviews(
    draft_id: int,
    owner_id: str = Query(default="default"),
    limit: int = Query(default=25, ge=1, le=100),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_outreach_compliance_reviews(conn, draft_id, owner_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/advisor-outreach-drafts/{draft_id}/delivery-packet")
def agent_advisor_outreach_delivery_packet(
    draft_id: int,
    owner_id: str = Query(default="default"),
    save: bool = Query(default=False),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_outreach_delivery_packet(conn, draft_id, owner_id, save=save)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/advisor-outreach-delivery-dashboard")
def agent_advisor_outreach_delivery_dashboard(
    owner_id: str = Query(default="default"),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_outreach_delivery_dashboard(conn, owner_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/customer-intent-dashboard")
def agent_customer_intent_dashboard(
    owner_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_customer_intent_dashboard(conn, owner_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/customer-engagement-timeline")
def agent_customer_engagement_timeline(
    owner_id: str = Query(default="default"),
    limit: int = Query(default=25, ge=1, le=100),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_customer_engagement_timeline(conn, owner_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/customer-engagement-brief")
def agent_customer_engagement_brief(
    owner_id: str = Query(default="default"),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_customer_engagement_brief(conn, owner_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/customer-engagement-cadence-review")
def agent_customer_engagement_cadence_review(
    owner_id: str = Query(default="default"),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_customer_engagement_cadence_review(conn, owner_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/customer-engagement-cadence-dashboard")
def agent_customer_engagement_cadence_dashboard(
    limit: int = Query(default=25, ge=1, le=100),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_customer_engagement_cadence_dashboard(conn, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/customer-engagement-action-queue")
def agent_customer_engagement_action_queue(
    limit: int = Query(default=25, ge=1, le=100),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_customer_engagement_action_queue(conn, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/customer-engagement-task-brief")
def agent_customer_engagement_task_brief(
    owner_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_customer_engagement_task_brief(conn, owner_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/ai-recommendation-effectiveness-dashboard")
def agent_ai_recommendation_effectiveness_dashboard(
    owner_id: str | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_recommendation_effectiveness_dashboard(conn, owner_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/ai-improvement-backlog")
def agent_ai_improvement_backlog(
    owner_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_backlog(conn, owner_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/ai-improvement-experiment-plan")
def agent_ai_improvement_experiment_plan(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_experiment_plan(conn, owner_id, improvement_id=improvement_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-experiment-launch-packet")
def agent_ai_improvement_experiment_launch_packet(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_experiment_launch_packet(conn, owner_id, improvement_id=improvement_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-experiment-readout")
def agent_ai_improvement_experiment_readout(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_experiment_readout(conn, owner_id, improvement_id=improvement_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-rollout-readiness")
def agent_ai_improvement_rollout_readiness(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_rollout_readiness(conn, owner_id, improvement_id=improvement_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-rollout-monitor")
def agent_ai_improvement_rollout_monitor(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_rollout_monitor(conn, owner_id, improvement_id=improvement_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-release-packet")
def agent_ai_improvement_release_packet(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_release_packet(conn, owner_id, improvement_id=improvement_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-adoption-playbook")
def agent_ai_improvement_adoption_playbook(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_adoption_playbook(conn, owner_id, improvement_id=improvement_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-adoption-monitor")
def agent_ai_improvement_adoption_monitor(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_adoption_monitor(conn, owner_id, improvement_id=improvement_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-adoption-impact-ledger")
def agent_ai_improvement_adoption_impact_ledger(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_adoption_impact_ledger(conn, owner_id, improvement_id=improvement_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-scale-decision-packet")
def agent_ai_improvement_scale_decision_packet(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_scale_decision_packet(conn, owner_id, improvement_id=improvement_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-scale-execution-plan")
def agent_ai_improvement_scale_execution_plan(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_scale_execution_plan(conn, owner_id, improvement_id=improvement_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-scale-execution-monitor")
def agent_ai_improvement_scale_execution_monitor(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_scale_execution_monitor(conn, owner_id, improvement_id=improvement_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-scale-learning-report")
def agent_ai_improvement_scale_learning_report(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_scale_learning_report(conn, owner_id, improvement_id=improvement_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-roadmap-refresh")
def agent_ai_improvement_roadmap_refresh(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_roadmap_refresh(conn, owner_id, improvement_id=improvement_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-backlog-handoff")
def agent_ai_improvement_backlog_handoff(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_backlog_handoff(conn, owner_id, improvement_id=improvement_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-implementation-kickoff-packet")
def agent_ai_improvement_implementation_kickoff_packet(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_implementation_kickoff_packet(conn, owner_id, improvement_id=improvement_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-implementation-readiness-monitor")
def agent_ai_improvement_implementation_readiness_monitor(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_implementation_readiness_monitor(conn, owner_id, improvement_id=improvement_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-implementation-blocker-resolution-plan")
def agent_ai_improvement_implementation_blocker_resolution_plan(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_implementation_blocker_resolution_plan(conn, owner_id, improvement_id=improvement_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-implementation-unblock-verification-report")
def agent_ai_improvement_implementation_unblock_verification_report(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_implementation_unblock_verification_report(
            conn, owner_id, improvement_id=improvement_id, limit=limit
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-implementation-qa-review-packet")
def agent_ai_improvement_implementation_qa_review_packet(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_implementation_qa_review_packet(
            conn, owner_id, improvement_id=improvement_id, limit=limit
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-implementation-qa-signoff-report")
def agent_ai_improvement_implementation_qa_signoff_report(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_implementation_qa_signoff_report(
            conn, owner_id, improvement_id=improvement_id, limit=limit
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-launch-review-packet")
def agent_ai_improvement_launch_review_packet(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_launch_review_packet(conn, owner_id, improvement_id=improvement_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-launch-execution-plan")
def agent_ai_improvement_launch_execution_plan(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_launch_execution_plan(conn, owner_id, improvement_id=improvement_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-launch-execution-monitor")
def agent_ai_improvement_launch_execution_monitor(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_launch_execution_monitor(conn, owner_id, improvement_id=improvement_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-launch-outcome-monitor")
def agent_ai_improvement_launch_outcome_monitor(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_launch_outcome_monitor(conn, owner_id, improvement_id=improvement_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-launch-value-proof-packet")
def agent_ai_improvement_launch_value_proof_packet(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_launch_value_proof_packet(conn, owner_id, improvement_id=improvement_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-launch-customer-communication-packet")
def agent_ai_improvement_launch_customer_communication_packet(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_launch_customer_communication_packet(
            conn,
            owner_id,
            improvement_id=improvement_id,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-launch-customer-communication-review-packet")
def agent_ai_improvement_launch_customer_communication_review_packet(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_launch_customer_communication_review_packet(
            conn,
            owner_id,
            improvement_id=improvement_id,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-launch-customer-communication-delivery-packet")
def agent_ai_improvement_launch_customer_communication_delivery_packet(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_launch_customer_communication_delivery_packet(
            conn,
            owner_id,
            improvement_id=improvement_id,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-launch-customer-communication-delivery-monitor")
def agent_ai_improvement_launch_customer_communication_delivery_monitor(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_launch_customer_communication_delivery_monitor(
            conn,
            owner_id,
            improvement_id=improvement_id,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-launch-customer-communication-delivery-unblock-plan")
def agent_ai_improvement_launch_customer_communication_delivery_unblock_plan(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_launch_customer_communication_delivery_unblock_plan(
            conn,
            owner_id,
            improvement_id=improvement_id,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-launch-customer-communication-delivery-unblock-verification-report")
def agent_ai_improvement_launch_customer_communication_delivery_unblock_verification_report(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_launch_customer_communication_delivery_unblock_verification_report(
            conn,
            owner_id,
            improvement_id=improvement_id,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-launch-customer-communication-delivery-send-authorization-packet")
def agent_ai_improvement_launch_customer_communication_delivery_send_authorization_packet(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_launch_customer_communication_delivery_send_authorization_packet(
            conn,
            owner_id,
            improvement_id=improvement_id,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-launch-customer-communication-delivery-send-authorization-monitor")
def agent_ai_improvement_launch_customer_communication_delivery_send_authorization_monitor(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_launch_customer_communication_delivery_send_authorization_monitor(
            conn,
            owner_id,
            improvement_id=improvement_id,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-launch-customer-communication-delivery-send-authorization-unblock-plan")
def agent_ai_improvement_launch_customer_communication_delivery_send_authorization_unblock_plan(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_launch_customer_communication_delivery_send_authorization_unblock_plan(
            conn,
            owner_id,
            improvement_id=improvement_id,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-launch-customer-communication-delivery-send-authorization-unblock-verification-report")
def agent_ai_improvement_launch_customer_communication_delivery_send_authorization_unblock_verification_report(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_launch_customer_communication_delivery_send_authorization_unblock_verification_report(
            conn,
            owner_id,
            improvement_id=improvement_id,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-launch-customer-communication-delivery-send-readiness-packet")
def agent_ai_improvement_launch_customer_communication_delivery_send_readiness_packet(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_launch_customer_communication_delivery_send_readiness_packet(
            conn,
            owner_id,
            improvement_id=improvement_id,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-launch-customer-communication-delivery-send-readiness-review-packet")
def agent_ai_improvement_launch_customer_communication_delivery_send_readiness_review_packet(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_launch_customer_communication_delivery_send_readiness_review_packet(
            conn,
            owner_id,
            improvement_id=improvement_id,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ai-improvement-launch-customer-communication-delivery-send-execution-handoff-packet")
def agent_ai_improvement_launch_customer_communication_delivery_send_execution_handoff_packet(
    owner_id: str | None = Query(default=None),
    improvement_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_ai_improvement_launch_customer_communication_delivery_send_execution_handoff_packet(
            conn,
            owner_id,
            improvement_id=improvement_id,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/customer-intent-action-plan")
def agent_customer_intent_action_plan(
    owner_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_customer_intent_action_plan(conn, owner_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/customer-intent-followup-packet")
def agent_customer_intent_followup_packet(
    owner_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_customer_intent_followup_packet(conn, owner_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/customer-intent-followup-review")
def agent_customer_intent_followup_review(
    owner_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_customer_intent_followup_review(conn, owner_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/customer-intent-followup-draft")
def agent_customer_intent_followup_draft(
    owner_id: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    save: bool = Query(default=False),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_customer_intent_followup_draft(conn, owner_id, limit=limit, save=save)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/advisor-outreach-deliveries")
def agent_advisor_outreach_deliveries(
    owner_id: str = Query(default="default"),
    status: str | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_outreach_delivery_records(conn, owner_id, status=status, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/advisor-outreach-deliveries/{delivery_id}")
def agent_saved_advisor_outreach_delivery(
    delivery_id: int,
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return get_outreach_delivery_record(conn, delivery_id, owner_id)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/advisor-outreach-deliveries/{delivery_id}")
def agent_update_advisor_outreach_delivery(
    delivery_id: int,
    request: OutreachDeliveryUpdate,
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return update_outreach_delivery_record(
            conn,
            delivery_id,
            owner_id,
            request.status,
            delivery_notes=request.delivery_notes,
            delivered_by=request.delivered_by,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/advisor-outreach-deliveries/{delivery_id}/outcome")
def agent_create_advisor_outreach_delivery_outcome(
    delivery_id: int,
    request: OutreachDeliveryOutcomeCreate,
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return save_outreach_delivery_outcome(
            conn,
            delivery_id,
            owner_id,
            outcome_type=request.outcome_type,
            response_text=request.response_text,
            follow_up_due_at=request.follow_up_due_at,
            recorded_by=request.recorded_by,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/advisor-outreach-outcomes")
def agent_advisor_outreach_delivery_outcomes(
    owner_id: str = Query(default="default"),
    delivery_id: int | None = Query(default=None),
    outcome_type: str | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_outreach_delivery_outcomes(
            conn,
            owner_id,
            delivery_id=delivery_id,
            outcome_type=outcome_type,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/advisor-outreach-outcomes/{outcome_id}")
def agent_saved_advisor_outreach_delivery_outcome(
    outcome_id: int,
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return get_outreach_delivery_outcome(conn, outcome_id, owner_id)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/action-queue")
def agent_action_queue(
    request: Request,
    owner_id: str = Query(default="default"),
    title: str | None = Query(default=None),
    focus: str | None = Query(default=None),
    max_items: int = Query(default=2, ge=1, le=10),
    evidence_limit: int = Query(default=1, ge=1, le=5),
    persist_screeners: bool = Query(default=False),
    save: bool = Query(default=False),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_action_queue(
            conn,
            owner_id,
            request.app.state.vector_index_dir,
            focus=focus,
            max_items=max_items,
            evidence_limit=evidence_limit,
            persist_screeners=persist_screeners,
            save=save,
            title=title,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/action-queues")
def agent_action_queues(
    owner_id: str = Query(default="default"),
    status: str | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_action_queues(conn, owner_id, status=status, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/action-queues/summary")
def agent_action_queue_summary(
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return summarize_action_queues(conn, owner_id)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/action-queues/tasks")
def agent_action_queue_tasks(
    owner_id: str = Query(default="default"),
    status: str = Query(default="active"),
    focus: str | None = Query(default=None),
    urgency: str | None = Query(default=None),
    assigned_to: str | None = Query(default=None),
    due_before: str | None = Query(default=None),
    due_after: str | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_action_queue_tasks(
            conn,
            owner_id,
            status=status,
            focus=focus,
            urgency=urgency,
            assigned_to=assigned_to,
            due_before=due_before,
            due_after=due_after,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/action-queues/tasks/escalations/summary")
def agent_action_queue_task_escalation_summary(
    owner_id: str | None = Query(default=None),
    as_of: str | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return summarize_action_queue_task_escalations(conn, owner_id=owner_id, as_of=as_of, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/action-queues/tasks/escalations/inbox")
def agent_action_queue_task_escalation_inbox(
    owner_id: str | None = Query(default=None),
    as_of: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    inbox_status: str | None = Query(default=None),
    assigned_to: str | None = Query(default=None),
    focus: str | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_action_queue_task_escalation_inbox(
            conn,
            owner_id=owner_id,
            as_of=as_of,
            severity=severity,
            inbox_status=inbox_status,
            assigned_to=assigned_to,
            focus=focus,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/action-queues/tasks/escalations/inbox/notification")
def agent_action_queue_task_escalation_inbox_notification(
    owner_id: str | None = Query(default=None),
    as_of: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    inbox_status: str | None = Query(default=None),
    assigned_to: str | None = Query(default=None),
    focus: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_action_queue_task_escalation_inbox_notification(
            conn,
            owner_id=owner_id,
            as_of=as_of,
            severity=severity,
            inbox_status=inbox_status,
            assigned_to=assigned_to,
            focus=focus,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/action-queues/tasks/escalations/inbox/notification")
def agent_log_action_queue_task_escalation_inbox_notification(
    request: ActionQueueEscalationNotificationLog,
    owner_id: str | None = Query(default=None),
    as_of: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    inbox_status: str | None = Query(default=None),
    assigned_to: str | None = Query(default=None),
    focus: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return save_action_queue_task_escalation_inbox_notification(
            conn,
            owner_id=owner_id,
            as_of=as_of,
            severity=severity,
            inbox_status=inbox_status,
            assigned_to=assigned_to,
            focus=focus,
            limit=limit,
            channel=request.channel,
            recipient=request.recipient,
            status=request.status,
            idempotency_key=request.idempotency_key,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/action-queues/tasks/escalations/inbox/notification/logs")
def agent_action_queue_task_escalation_notification_logs(
    owner_id: str | None = Query(default=None),
    channel: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_action_queue_task_escalation_notifications(
            conn,
            owner_id=owner_id,
            channel=channel,
            status=status,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/action-queues/tasks/escalations/inbox/notification/summary")
def agent_action_queue_task_escalation_notification_delivery_summary(
    owner_id: str | None = Query(default=None),
    channel: str | None = Query(default=None),
    stale_after_minutes: int = Query(default=60, ge=1, le=10080),
    recent_limit: int = Query(default=5, ge=0, le=25),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return summarize_action_queue_task_escalation_notification_delivery(
            conn,
            owner_id=owner_id,
            channel=channel,
            stale_after_minutes=stale_after_minutes,
            recent_limit=recent_limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/action-queues/tasks/escalations/inbox/notification/delivery-control-summary")
def agent_action_queue_task_escalation_notification_delivery_control_summary(
    owner_id: str | None = Query(default=None),
    channel: str | None = Query(default=None),
    stale_after_minutes: int = Query(default=60, ge=1, le=10080),
    expiring_within_seconds: int = Query(default=300, ge=1, le=86400),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return summarize_action_queue_task_escalation_notification_delivery_control(
            conn,
            owner_id=owner_id,
            channel=channel,
            stale_after_minutes=stale_after_minutes,
            expiring_within_seconds=expiring_within_seconds,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/action-queues/tasks/escalations/inbox/notification/delivery-queue")
def agent_action_queue_task_escalation_notification_delivery_queue(
    owner_id: str | None = Query(default=None),
    channel: str | None = Query(default=None),
    stale_after_minutes: int = Query(default=60, ge=1, le=10080),
    limit: int = Query(default=25, ge=1, le=100),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_action_queue_task_escalation_notification_delivery_queue(
            conn,
            owner_id=owner_id,
            channel=channel,
            stale_after_minutes=stale_after_minutes,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/action-queues/tasks/escalations/inbox/notification/delivery-incident-summary")
def agent_action_queue_task_escalation_notification_delivery_incident_summary(
    owner_id: str | None = Query(default=None),
    channel: str | None = Query(default=None),
    stale_after_minutes: int = Query(default=60, ge=1, le=10080),
    expiring_within_seconds: int = Query(default=300, ge=1, le=86400),
    max_incidents: int = Query(default=10000, ge=1, le=50000),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return summarize_action_queue_task_escalation_notification_delivery_incidents(
            conn,
            owner_id=owner_id,
            channel=channel,
            stale_after_minutes=stale_after_minutes,
            expiring_within_seconds=expiring_within_seconds,
            max_incidents=max_incidents,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/action-queues/tasks/escalations/inbox/notification/delivery-incident-workload")
def agent_action_queue_task_escalation_notification_delivery_incident_workload(
    owner_id: str | None = Query(default=None),
    channel: str | None = Query(default=None),
    stale_after_minutes: int = Query(default=60, ge=1, le=10080),
    expiring_within_seconds: int = Query(default=300, ge=1, le=86400),
    follow_up_within_hours: int = Query(default=24, ge=1, le=8760),
    max_incidents: int = Query(default=10000, ge=1, le=50000),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return summarize_action_queue_task_escalation_notification_delivery_incident_workload(
            conn,
            owner_id=owner_id,
            channel=channel,
            stale_after_minutes=stale_after_minutes,
            expiring_within_seconds=expiring_within_seconds,
            follow_up_within_hours=follow_up_within_hours,
            max_incidents=max_incidents,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/action-queues/tasks/escalations/inbox/notification/delivery-incidents")
def agent_action_queue_task_escalation_notification_delivery_incidents(
    owner_id: str | None = Query(default=None),
    channel: str | None = Query(default=None),
    assigned_to: str | None = Query(default=None),
    follow_up_status: str | None = Query(default=None),
    follow_up_within_hours: int = Query(default=24, ge=1, le=8760),
    stale_after_minutes: int = Query(default=60, ge=1, le=10080),
    expiring_within_seconds: int = Query(default=300, ge=1, le=86400),
    limit: int = Query(default=25, ge=1, le=100),
    include_suppressed: bool = Query(default=False),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_action_queue_task_escalation_notification_delivery_incidents(
            conn,
            owner_id=owner_id,
            channel=channel,
            stale_after_minutes=stale_after_minutes,
            expiring_within_seconds=expiring_within_seconds,
            limit=limit,
            include_suppressed=include_suppressed,
            assigned_to=assigned_to,
            follow_up_status=follow_up_status,
            follow_up_within_hours=follow_up_within_hours,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/action-queues/tasks/escalations/inbox/notification/delivery-incidents/{notification_id}")
def agent_action_queue_task_escalation_notification_delivery_incident_detail(
    notification_id: int,
    incident_type: str | None = Query(default=None),
    owner_id: str | None = Query(default=None),
    stale_after_minutes: int = Query(default=60, ge=1, le=10080),
    expiring_within_seconds: int = Query(default=300, ge=1, le=86400),
    follow_up_within_hours: int = Query(default=24, ge=1, le=8760),
    audit_limit: int = Query(default=25, ge=1, le=200),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return get_action_queue_task_escalation_notification_delivery_incident(
            conn,
            notification_id=notification_id,
            owner_id=owner_id,
            incident_type=incident_type,
            stale_after_minutes=stale_after_minutes,
            expiring_within_seconds=expiring_within_seconds,
            follow_up_within_hours=follow_up_within_hours,
            audit_limit=audit_limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/action-queues/tasks/escalations/inbox/notification/delivery-incident-reviews")
def agent_action_queue_task_escalation_notification_delivery_incident_reviews(
    owner_id: str | None = Query(default=None),
    notification_id: int | None = Query(default=None, ge=1),
    incident_type: str | None = Query(default=None),
    incident_status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_action_queue_task_escalation_notification_delivery_incident_reviews(
            conn,
            owner_id=owner_id,
            notification_id=notification_id,
            incident_type=incident_type,
            incident_status=incident_status,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/action-queues/tasks/escalations/inbox/notification/delivery-incident-reviews")
def agent_bulk_review_action_queue_task_escalation_notification_delivery_incidents(
    request: ActionQueueEscalationNotificationDeliveryIncidentBulkReview,
    owner_id: str | None = Query(default=None),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return bulk_review_action_queue_task_escalation_notification_delivery_incidents(
            conn,
            [
                incident.model_dump() if hasattr(incident, "model_dump") else incident.dict()
                for incident in request.incidents
            ],
            request.incident_status,
            owner_id=owner_id,
            reviewer=request.reviewer,
            assigned_to=request.assigned_to,
            notes=request.notes,
            follow_up_at=request.follow_up_at,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/action-queues/tasks/escalations/inbox/notification/delivery-incidents/{notification_id}/review")
def agent_review_action_queue_task_escalation_notification_delivery_incident(
    notification_id: int,
    request: ActionQueueEscalationNotificationDeliveryIncidentReview,
    owner_id: str | None = Query(default=None),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return review_action_queue_task_escalation_notification_delivery_incident(
            conn,
            notification_id=notification_id,
            owner_id=owner_id,
            incident_type=request.incident_type,
            incident_status=request.incident_status,
            reviewer=request.reviewer,
            assigned_to=request.assigned_to,
            notes=request.notes,
            follow_up_at=request.follow_up_at,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/action-queues/tasks/escalations/inbox/notification/delivery-incidents/{notification_id}/actions")
def agent_execute_action_queue_task_escalation_notification_delivery_incident_action(
    notification_id: int,
    request: ActionQueueEscalationNotificationDeliveryIncidentAction,
    owner_id: str | None = Query(default=None),
    stale_after_minutes: int = Query(default=60, ge=1, le=10080),
    expiring_within_seconds: int = Query(default=300, ge=1, le=86400),
    follow_up_within_hours: int = Query(default=24, ge=1, le=8760),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return execute_action_queue_task_escalation_notification_delivery_incident_action(
            conn,
            notification_id=notification_id,
            owner_id=owner_id,
            action_id=request.action_id,
            incident_type=request.incident_type,
            reviewer=request.reviewer,
            assigned_to=request.assigned_to,
            notes=request.notes,
            follow_up_at=request.follow_up_at,
            delivery_notes=request.delivery_notes,
            requeued_by=request.requeued_by,
            retry_after=request.retry_after,
            claim_token=request.claim_token,
            release_notes=request.release_notes,
            released_by=request.released_by,
            claimed_by=request.claimed_by,
            lease_seconds=request.lease_seconds if request.lease_seconds is not None else 300,
            stale_after_minutes=stale_after_minutes,
            expiring_within_seconds=expiring_within_seconds,
            follow_up_within_hours=follow_up_within_hours,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/action-queues/tasks/escalations/inbox/notification/delivery-claims")
def agent_action_queue_task_escalation_notification_delivery_claims(
    owner_id: str | None = Query(default=None),
    channel: str | None = Query(default=None),
    claimed_by: str | None = Query(default=None),
    lease_state: str | None = Query(default=None),
    expiring_within_seconds: int = Query(default=300, ge=1, le=86400),
    limit: int = Query(default=25, ge=1, le=100),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_action_queue_task_escalation_notification_delivery_claims(
            conn,
            owner_id=owner_id,
            channel=channel,
            claimed_by=claimed_by,
            lease_state=lease_state,
            expiring_within_seconds=expiring_within_seconds,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/action-queues/tasks/escalations/inbox/notification/deadletters")
def agent_action_queue_task_escalation_notification_deadletters(
    owner_id: str | None = Query(default=None),
    channel: str | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_action_queue_task_escalation_notification_deadletters(
            conn,
            owner_id=owner_id,
            channel=channel,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/action-queues/tasks/escalations/inbox/notification/deadletters/remediations")
def agent_action_queue_task_escalation_notification_deadletter_remediations(
    owner_id: str | None = Query(default=None),
    notification_id: int | None = Query(default=None, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_action_queue_task_escalation_notification_deadletter_remediations(
            conn,
            owner_id=owner_id,
            notification_id=notification_id,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/action-queues/tasks/escalations/inbox/notification/deadletters/{notification_id}/requeue")
def agent_requeue_action_queue_task_escalation_notification_deadletter(
    notification_id: int,
    request: ActionQueueEscalationNotificationDeadletterRequeue,
    owner_id: str | None = Query(default=None),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return requeue_action_queue_task_escalation_notification_deadletter(
            conn,
            notification_id=notification_id,
            owner_id=owner_id,
            retry_after=request.retry_after,
            delivery_notes=request.delivery_notes,
            requeued_by=request.requeued_by,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/action-queues/tasks/escalations/inbox/notification/delivery-attempts")
def agent_action_queue_task_escalation_notification_delivery_attempts(
    owner_id: str | None = Query(default=None),
    notification_id: int | None = Query(default=None, ge=1),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_action_queue_task_escalation_notification_delivery_attempts(
            conn,
            owner_id=owner_id,
            notification_id=notification_id,
            status=status,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/action-queues/tasks/escalations/inbox/notification/delivery-claim-releases")
def agent_action_queue_task_escalation_notification_delivery_claim_releases(
    owner_id: str | None = Query(default=None),
    notification_id: int | None = Query(default=None, ge=1),
    released_by: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_action_queue_task_escalation_notification_delivery_claim_releases(
            conn,
            owner_id=owner_id,
            notification_id=notification_id,
            released_by=released_by,
            limit=limit,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/action-queues/tasks/escalations/inbox/notification/delivery-claim")
def agent_claim_action_queue_task_escalation_notification_delivery(
    request: ActionQueueEscalationNotificationDeliveryClaim,
    owner_id: str | None = Query(default=None),
    channel: str | None = Query(default=None),
    stale_after_minutes: int = Query(default=60, ge=1, le=10080),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return claim_action_queue_task_escalation_notification_delivery(
            conn,
            owner_id=owner_id,
            channel=channel,
            claimed_by=request.claimed_by,
            stale_after_minutes=stale_after_minutes,
            lease_seconds=request.lease_seconds if request.lease_seconds is not None else 300,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/action-queues/tasks/escalations/inbox/notification/delivery-claim/{notification_id}/renew")
def agent_renew_action_queue_task_escalation_notification_delivery_claim(
    notification_id: int,
    request: ActionQueueEscalationNotificationDeliveryClaimRenewal,
    owner_id: str | None = Query(default=None),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return renew_action_queue_task_escalation_notification_delivery_claim(
            conn,
            notification_id=notification_id,
            owner_id=owner_id,
            claim_token=request.claim_token,
            lease_seconds=request.lease_seconds if request.lease_seconds is not None else 300,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/action-queues/tasks/escalations/inbox/notification/delivery-claim/{notification_id}/release")
def agent_release_action_queue_task_escalation_notification_delivery_claim(
    notification_id: int,
    request: ActionQueueEscalationNotificationDeliveryClaimRelease,
    owner_id: str | None = Query(default=None),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return release_action_queue_task_escalation_notification_delivery_claim(
            conn,
            notification_id=notification_id,
            owner_id=owner_id,
            claim_token=request.claim_token,
            release_notes=request.release_notes,
            released_by=request.released_by,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/action-queues/tasks/escalations/inbox/notification/delivery-claim/{notification_id}/complete")
def agent_complete_action_queue_task_escalation_notification_delivery_claim(
    notification_id: int,
    request: ActionQueueEscalationNotificationDeliveryCompletion,
    owner_id: str | None = Query(default=None),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return complete_action_queue_task_escalation_notification_delivery_claim(
            conn,
            notification_id=notification_id,
            owner_id=owner_id,
            claim_token=request.claim_token,
            status=request.status,
            delivery_notes=request.delivery_notes,
            delivered_at=request.delivered_at,
            retry_after=request.retry_after,
            max_attempts=request.max_attempts if request.max_attempts is not None else 3,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/action-queues/tasks/escalations/inbox/notification/logs/{notification_id}")
def agent_update_action_queue_task_escalation_notification_log(
    notification_id: int,
    request: ActionQueueEscalationNotificationUpdate,
    owner_id: str | None = Query(default=None),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return update_action_queue_task_escalation_notification(
            conn,
            notification_id=notification_id,
            owner_id=owner_id,
            status=request.status,
            delivery_notes=request.delivery_notes,
            delivered_at=request.delivered_at,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/action-queues/tasks/escalations")
def agent_action_queue_task_escalations(
    owner_id: str = Query(default="default"),
    as_of: str | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_action_queue_task_escalations(conn, owner_id, as_of=as_of, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/action-queues/tasks/workload")
def agent_action_queue_task_workload(
    owner_id: str = Query(default="default"),
    as_of: str | None = Query(default=None),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return summarize_action_queue_task_workload(conn, owner_id, as_of=as_of)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/action-queues/tasks")
def agent_bulk_update_action_queue_tasks(
    request: ActionQueueTaskBulkUpdate,
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return bulk_update_action_queue_tasks(
            conn,
            owner_id,
            [task.model_dump() if hasattr(task, "model_dump") else task.dict() for task in request.tasks],
            request.status,
            notes=request.notes,
            assigned_to=request.assigned_to,
            due_at=request.due_at,
            updated_by=request.updated_by,
            update_source=request.update_source or "bulk",
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/action-queues/tasks/activity")
def agent_action_queue_task_activity(
    owner_id: str = Query(default="default"),
    queue_id: int | None = Query(default=None),
    task_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_action_queue_task_updates(conn, owner_id, queue_id=queue_id, task_id=task_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/action-queues/tasks/escalations/reviews")
def agent_action_queue_task_escalation_reviews(
    owner_id: str = Query(default="default"),
    queue_id: int | None = Query(default=None),
    task_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_action_queue_task_escalation_reviews(conn, owner_id, queue_id=queue_id, task_id=task_id, limit=limit)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/action-queues/tasks/escalations/reviews")
def agent_bulk_review_action_queue_task_escalations(
    request: ActionQueueEscalationBulkReview,
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return bulk_review_action_queue_task_escalations(
            conn,
            owner_id,
            [task.model_dump() if hasattr(task, "model_dump") else task.dict() for task in request.tasks],
            request.review_status,
            reviewer=request.reviewer,
            notes=request.notes,
            snoozed_until=request.snoozed_until,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/action-queues/{queue_id}")
def agent_saved_action_queue(
    queue_id: int,
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return get_action_queue(conn, queue_id, owner_id)
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/action-queues/{queue_id}/tasks/{task_id}")
def agent_update_action_queue_task(
    queue_id: int,
    task_id: str,
    request: ActionQueueTaskUpdate,
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return update_action_queue_task(
            conn,
            queue_id,
            owner_id,
            task_id,
            request.status,
            notes=request.notes,
            assigned_to=request.assigned_to,
            due_at=request.due_at,
            updated_by=request.updated_by,
            update_source=request.update_source or "single",
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/action-queues/{queue_id}/tasks/{task_id}/escalation-review")
def agent_review_action_queue_task_escalation(
    queue_id: int,
    task_id: str,
    request: ActionQueueEscalationReview,
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return review_action_queue_task_escalation(
            conn,
            owner_id,
            queue_id,
            task_id,
            request.review_status,
            reviewer=request.reviewer,
            notes=request.notes,
            snoozed_until=request.snoozed_until,
        )
    except MarketDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MarketRecordNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/workflow/{ticker}")
def agent_workflow(
    ticker: str,
    request: Request,
) -> dict[str, Any]:
    base_url = str(request.base_url).rstrip("/")
    return {
        "ticker": ticker,
        "profile": request.app.state.profile["name"],
        "recommended_sequence": [
            {
                "step": 1,
                "tool": "runs.latest",
                "endpoint": f"{base_url}/runs/latest",
                "purpose": "Check freshness and recent audit results before reasoning.",
            },
            {
                "step": 2,
                "tool": "freshness.ticker",
                "endpoint": f"{base_url}/freshness/{ticker}",
                "purpose": "Check ticker-level freshness and quality before producing customer-facing output.",
            },
            {
                "step": 3,
                "tool": "market.company_profile",
                "endpoint": f"{base_url}/companies/{ticker}",
                "purpose": "Fetch the canonical profile before using raw or semantic evidence.",
            },
            {
                "step": 4,
                "tool": "market.quote_snapshot",
                "endpoint": f"{base_url}/quotes/{ticker}",
                "purpose": "Fetch latest quote freshness and quality metadata.",
            },
            {
                "step": 5,
                "tool": "agents.advisor_workbench",
                "endpoint": f"{base_url}/agents/advisor-workbench?limit=10",
                "purpose": "Prioritize saved customer follow-up tasks before generating new work.",
            },
            {
                "step": 6,
                "tool": "agents.advisor_outreach_draft",
                "endpoint": f"{base_url}/agents/advisor-outreach-draft?owner_id=default&save=true",
                "purpose": "Turn the top saved follow-up task into a saved customer outreach draft for review.",
            },
            {
                "step": 7,
                "tool": "agents.advisor_outreach_compliance_review",
                "endpoint": f"{base_url}/agents/advisor-outreach-drafts/{{draft_id}}/compliance-review?owner_id=default",
                "purpose": "Review a saved customer outreach draft for disclosure, blocked-task, and risky-language issues.",
            },
            {
                "step": 8,
                "tool": "agents.action_queue",
                "endpoint": f"{base_url}/agents/action-queue?owner_id=default&save=true",
                "purpose": "Turn advisor follow-up packs into a saved task queue with blockers and completion criteria.",
            },
            {
                "step": 9,
                "tool": "agents.advisor_followup",
                "endpoint": f"{base_url}/agents/advisor-followup?owner_id=default",
                "purpose": "Convert the owner morning brief into customer-ready follow-up copy and advisor guardrails.",
            },
            {
                "step": 10,
                "tool": "agents.morning_brief",
                "endpoint": f"{base_url}/agents/morning-brief?owner_id=default",
                "purpose": "Prioritize customer actions across portfolios, watchlists, screeners, and freshness.",
            },
            {
                "step": 11,
                "tool": "agents.screener_copilot",
                "endpoint": f"{base_url}/agents/screener-copilot?prompt=energy%20companies%20with%20pe%20under%2030",
                "purpose": "Turn a customer stock-discovery idea into a structured screen with explained matches.",
            },
            {
                "step": 12,
                "tool": "screeners.index",
                "endpoint": f"{base_url}/screeners?owner_id=default",
                "purpose": "Use saved screeners for reusable strategy, newsletter, widget, or alert context.",
            },
            {
                "step": 13,
                "tool": "watchlists.show",
                "endpoint": f"{base_url}/watchlists?owner_id=default",
                "purpose": "Use saved watchlists when building user-specific monitoring or digest context.",
            },
            {
                "step": 14,
                "tool": "portfolios.index",
                "endpoint": f"{base_url}/portfolios?owner_id=default",
                "purpose": "Use saved portfolios when building customer-specific exposure or X-ray context.",
            },
            {
                "step": 15,
                "tool": "vectors.search",
                "endpoint": f"{base_url}/vectors/search?query={ticker}&k=10",
                "purpose": "Retrieve semantically similar local documents before building the final context.",
            },
            {
                "step": 16,
                "tool": "documents.current",
                "endpoint": f"{base_url}/documents/current?limit=25",
                "purpose": "Fetch the most recent market artifacts for retrieval grounding.",
            },
            {
                "step": 17,
                "tool": "agents.screener_digest",
                "endpoint": f"{base_url}/agents/screener-digest/{{screener_id}}",
                "purpose": "Generate a customer-level screener digest with cited evidence and reusable next actions.",
            },
            {
                "step": 18,
                "tool": "agents.research_brief",
                "endpoint": f"{base_url}/agents/research-brief/{ticker}",
                "purpose": "Generate a cited AI research brief from local market and semantic evidence.",
            },
            {
                "step": 19,
                "tool": "agents.portfolio_digest",
                "endpoint": f"{base_url}/agents/portfolio-digest/{{portfolio_id}}",
                "purpose": "Generate a customer-level portfolio digest with evidence and next actions.",
            },
            {
                "step": 20,
                "tool": "agents.watchlist_digest",
                "endpoint": f"{base_url}/agents/watchlist-digest/{{watchlist_id}}",
                "purpose": "Generate a customer-level watchlist digest with evidence and alert context.",
            },
            {
                "step": 21,
                "tool": "agents.context",
                "endpoint": f"{base_url}/agents/context/{ticker}",
                "purpose": "Build a model-ready payload for analysis and planning.",
            },
        ],
        "operating_guidance": [
            "Prefer the most recent completed run.",
            "Surface quality check failures to downstream agents.",
            "Treat missing coverage as a data gap rather than a neutral signal.",
        ],
    }
