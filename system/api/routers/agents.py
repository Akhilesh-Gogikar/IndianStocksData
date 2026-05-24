from __future__ import annotations

import json
import os
import re
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
from system.ai.multi_agent_harness import build_unified_agentic_runtime
from system.ai.research_brief import build_research_brief
from system.ai.screener_copilot import run_screener_copilot
from system.ai.vector_index import VectorIndexError
from system.analyst_report import build_prompt_payload
from system.api.market_service import (
    MarketDataUnavailable,
    MarketRecordNotFound,
    get_company,
    get_events,
    get_quote,
    get_ratios,
    table_exists,
)

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
DEFAULT_FIREBASE_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_FIREBASE_GEMINI_MODEL = "gemini-1.5-flash"
LOCAL_MARKET_DATA_MODEL = "sqlite-market-rag"
LEAD_GEN_PRODUCT_ID = "lead-gen-marketplace"
LEAD_GEN_ROUTER_PROVIDER = "cerebral-router"
LEAD_GEN_ROUTER_MODEL = "lead-gen-intent-router-v1"
MARKET_QUERY_STOPWORDS = {
    "ABOUT",
    "AN",
    "AND",
    "COMPANY",
    "EXPLAIN",
    "FOR",
    "GIVE",
    "IN",
    "INFO",
    "ME",
    "OF",
    "ON",
    "PLEASE",
    "STOCK",
    "TELL",
    "THE",
    "THIS",
}


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


def _normalized_product_id(payload: LocalLlmChatRequest) -> str:
    context = payload.context if isinstance(payload.context, dict) else {}
    raw = payload.product_id or context.get("product_id") or "general"
    product_id = str(raw).strip().lower()
    return product_id or "general"


class UnifiedAgenticRuntimeRequest(BaseModel):
    product_id: str
    objective: str
    owner_id: str | None = "default"
    retrieval_mode: str | None = "hybrid"
    provider_preference: str | None = "auto"
    focus: str | None = None
    sql_queries: list[str] | None = None
    rag_query: str | None = None
    run_id: int | None = None
    source_name: str | None = None
    evidence_limit: int = 8
    max_rows_per_query: int = 25
    include_deep_research: bool = True
    include_documents: bool = True
    auto_build_rag_index: bool = True
    rag_build_limit: int = 1000
    context: dict[str, Any] | None = None
    max_tokens: int | None = 768
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
    product_id = _normalized_product_id(payload)
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


def _firebase_free_tier_config() -> dict[str, Any]:
    timeout_raw = os.getenv("FIREBASE_GEMINI_TIMEOUT_SECONDS", "20")
    try:
        timeout = max(1.0, float(timeout_raw))
    except ValueError:
        timeout = 20.0
    return {
        "api_key": os.getenv("FIREBASE_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY"),
        "base_url": os.getenv("FIREBASE_GEMINI_BASE_URL", DEFAULT_FIREBASE_GEMINI_BASE_URL).rstrip("/"),
        "model": os.getenv("FIREBASE_GEMINI_MODEL", DEFAULT_FIREBASE_GEMINI_MODEL),
        "timeout_seconds": timeout,
    }


def _request_firebase_gemini_json(prompt: str, max_tokens: int | None, temperature: float | None) -> dict[str, Any]:
    config = _firebase_free_tier_config()
    api_key = str(config.get("api_key") or "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="firebase_free_tier_unavailable")
    generation_config: dict[str, Any] = {}
    if max_tokens is not None:
        generation_config["maxOutputTokens"] = max(32, int(max_tokens))
    if temperature is not None:
        generation_config["temperature"] = float(temperature)
    request_payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    request = urllib.request.Request(
        f"{config['base_url']}/models/{config['model']}:generateContent?key={api_key}",
        data=json.dumps(request_payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=config["timeout_seconds"]) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise HTTPException(status_code=503, detail="firebase_free_tier_auth_failed") from exc
        if exc.code == 429:
            raise HTTPException(status_code=503, detail="firebase_free_tier_quota_exhausted") from exc
        raise HTTPException(status_code=503, detail=f"firebase_free_tier_http_{exc.code}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, TimeoutError):
            raise HTTPException(status_code=504, detail="firebase_free_tier_timeout") from exc
        raise HTTPException(status_code=503, detail="firebase_free_tier_unavailable") from exc
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="firebase_free_tier_malformed_response") from exc


def _extract_firebase_answer(response: dict[str, Any]) -> str:
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise HTTPException(status_code=502, detail="firebase_free_tier_malformed_response")
    content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        raise HTTPException(status_code=502, detail="firebase_free_tier_malformed_response")
    text = " ".join(str(part.get("text") or "").strip() for part in parts if isinstance(part, dict)).strip()
    if not text:
        raise HTTPException(status_code=502, detail="firebase_free_tier_empty_response")
    return text


def _provider_chain(provider_preference: str | None) -> list[str]:
    preference = (provider_preference or "auto").strip().lower()
    if preference == "firebase-free-tier":
        return ["firebase-free-tier", "local-llama", "local-market-data"]
    if preference == "local-llama":
        return ["local-llama", "local-market-data"]
    if preference == "local-market-data":
        return ["local-market-data"]
    return ["firebase-free-tier", "local-llama", "local-market-data"]


def _local_llama_deep_research(prompt: str, payload: UnifiedAgenticRuntimeRequest) -> dict[str, Any]:
    config = _llama_cpp_config()
    request_payload = {
        "model": config["model"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the unified Cerebral Insights deep-research agent. "
                    "Use only supplied processed evidence and stay non-advisory."
                ),
            },
            {"role": "user", "content": prompt},
        ],
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
        "usage": response.get("usage", {}),
    }


def _firebase_free_tier_deep_research(prompt: str, payload: UnifiedAgenticRuntimeRequest) -> dict[str, Any]:
    config = _firebase_free_tier_config()
    response = _request_firebase_gemini_json(prompt, payload.max_tokens, payload.temperature)
    return {
        "status": "answer_ready",
        "provider": "firebase-free-tier",
        "model": config["model"],
        "answer": _extract_firebase_answer(response),
        "usage": response.get("usageMetadata", {}),
    }


def _deterministic_deep_research_fallback(
    prompt: str,
    payload: UnifiedAgenticRuntimeRequest,
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    market_answer = _build_market_data_answer(
        LocalLlmChatRequest(
            message=payload.objective,
            product_id=payload.product_id,
            context=payload.context,
        ),
        conn,
    )
    if market_answer is not None:
        return {
            "status": "answer_ready",
            "provider": "local-market-data",
            "model": LOCAL_MARKET_DATA_MODEL,
            "answer": market_answer.get("answer", ""),
            "usage": {},
            "sources": market_answer.get("sources", []),
        }
    condensed_prompt = re.sub(r"\s+", " ", prompt).strip()
    if len(condensed_prompt) > 700:
        condensed_prompt = f"{condensed_prompt[:700]}..."
    return {
        "status": "answer_ready",
        "provider": "local-market-data",
        "model": LOCAL_MARKET_DATA_MODEL,
        "answer": (
            "Deterministic fallback response: use generated SQL/RAG evidence logs to continue the workflow. "
            f"Prompt summary: {condensed_prompt}"
        ),
        "usage": {},
        "sources": [],
    }


def _run_unified_deep_research(
    prompt: str,
    payload: UnifiedAgenticRuntimeRequest,
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    attempted: list[str] = []
    errors: list[dict[str, str]] = []
    for provider in _provider_chain(payload.provider_preference):
        attempted.append(provider)
        try:
            if provider == "firebase-free-tier":
                result = _firebase_free_tier_deep_research(prompt, payload)
            elif provider == "local-llama":
                result = _local_llama_deep_research(prompt, payload)
            else:
                result = _deterministic_deep_research_fallback(prompt, payload, conn)
            result["provider_chain_attempted"] = attempted.copy()
            if errors:
                result["provider_errors"] = errors
            return result
        except HTTPException as exc:
            errors.append({"provider": provider, "detail": str(exc.detail)})
    fallback = _deterministic_deep_research_fallback(prompt, payload, conn)
    fallback["provider_chain_attempted"] = attempted
    if errors:
        fallback["provider_errors"] = errors
    return fallback


def _market_query_tokens(prompt: str | None) -> list[str]:
    tokens = [token.upper() for token in re.findall(r"[A-Za-z0-9]+", prompt or "")]
    return [
        token
        for token in tokens
        if len(token) > 1 and token not in MARKET_QUERY_STOPWORDS
    ]


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    try:
        return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    except sqlite3.Error:
        return set()


def _parse_json_blob(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return None


def _resolve_market_ticker(conn: sqlite3.Connection, prompt: str | None) -> str | None:
    if not table_exists(conn, "companies") or "ticker" not in _table_columns(conn, "companies"):
        return None
    tokens = _market_query_tokens(prompt)
    if not tokens:
        return None
    placeholders = ",".join("?" for _ in tokens)
    try:
        row = conn.execute(
            f"""
            SELECT ticker
            FROM companies
            WHERE UPPER(ticker) IN ({placeholders})
            ORDER BY market_cap IS NULL, market_cap DESC, ticker
            LIMIT 1
            """,
            tokens,
        ).fetchone()
    except sqlite3.Error:
        return None
    if row is not None:
        return str(row["ticker"]).upper()
    clauses = " OR ".join("LOWER(name) LIKE ?" for _ in tokens)
    try:
        row = conn.execute(
            f"""
            SELECT ticker
            FROM companies
            WHERE {clauses}
            ORDER BY market_cap IS NULL, market_cap DESC, ticker
            LIMIT 1
            """,
            [f"%{token.lower()}%" for token in tokens],
        ).fetchone()
    except sqlite3.Error:
        return None
    return str(row["ticker"]).upper() if row is not None else None


def _format_market_value(value: Any, currency: str | None = None) -> str | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1_00_00_00_000:
        formatted = f"{number / 1_00_00_00_000:.2f}k cr"
    elif abs(number) >= 1_00_00_000:
        formatted = f"{number / 1_00_00_000:.2f} cr"
    elif abs(number) >= 1_00_000:
        formatted = f"{number / 1_00_000:.2f} lakh"
    else:
        formatted = f"{number:,.2f}".rstrip("0").rstrip(".")
    return f"{currency} {formatted}" if currency else formatted


def _market_source(
    *,
    title: str,
    source_table: str,
    snippet: str,
    snapshot_date: str | None,
    freshness_status: str = "processed-local",
) -> dict[str, Any]:
    return {
        "title": title,
        "source_table": source_table,
        "snapshot_date": snapshot_date,
        "freshness_status": freshness_status,
        "snippet": snippet,
    }


def _safe_market_call(callable_obj, *args: Any) -> dict[str, Any] | None:
    try:
        return callable_obj(*args)
    except (MarketDataUnavailable, MarketRecordNotFound, sqlite3.Error):
        return None


def _build_canonical_market_data_answer(
    payload: LocalLlmChatRequest,
    conn: sqlite3.Connection,
) -> dict[str, Any] | None:
    ticker = _resolve_market_ticker(conn, payload.message)
    if ticker is None:
        return None
    company_payload = _safe_market_call(get_company, conn, ticker)
    if not company_payload:
        return None
    company = company_payload.get("data") or {}
    quote_payload = _safe_market_call(get_quote, conn, ticker)
    ratios_payload = _safe_market_call(get_ratios, conn, ticker, None)
    events_payload = _safe_market_call(get_events, conn, ticker, 3)

    quote = (quote_payload or {}).get("data") or {}
    ratios = ((ratios_payload or {}).get("data") or {}).get("ratios") or []
    events = ((events_payload or {}).get("data") or {}).get("events") or []
    currency = quote.get("currency") or company.get("currency")

    parts = [
        f"{company.get('name') or ticker} ({ticker})",
        f"{company.get('exchange')} listed" if company.get("exchange") else None,
        f"{company.get('sector')} sector" if company.get("sector") else None,
        f"{company.get('industry')} industry" if company.get("industry") else None,
    ]
    overview = ", ".join(part for part in parts if part)
    answer_sections = [overview + "."]
    if quote.get("price") is not None:
        answer_sections.append(
            f"Latest processed quote is {_format_market_value(quote.get('price'), currency)}"
            f" as of {quote.get('as_of') or 'the latest local snapshot'}."
        )
    if company.get("market_cap") is not None:
        answer_sections.append(
            f"Processed market cap is {_format_market_value(company.get('market_cap'), currency)}."
        )
    if ratios:
        ratio_text = "; ".join(
            f"{item.get('ratio_name')}: {_format_market_value(item.get('ratio_value'))}"
            for item in ratios[:4]
            if item.get("ratio_name") and item.get("ratio_value") is not None
        )
        if ratio_text:
            answer_sections.append(f"Recent ratio context: {ratio_text}.")
    if events:
        event_titles = "; ".join(
            f"{event.get('event_date') or 'undated'} - {event.get('title')}"
            for event in events[:3]
            if event.get("title")
        )
        if event_titles:
            answer_sections.append(f"Recent local event records: {event_titles}.")
    answer_sections.append("This is processed local market-data context, not investment advice.")

    metadata = company_payload.get("metadata") or {}
    sources = [
        _market_source(
            title=f"{ticker} company profile",
            source_table="companies",
            snapshot_date=metadata.get("as_of"),
            snippet=f"{company.get('name') or ticker}: {company.get('sector') or 'sector unavailable'} / {company.get('industry') or 'industry unavailable'}",
        )
    ]
    if quote:
        quote_meta = (quote_payload or {}).get("metadata") or {}
        sources.append(
            _market_source(
                title=f"{ticker} latest quote",
                source_table="quote_snapshots",
                snapshot_date=quote_meta.get("as_of") or quote.get("as_of"),
                snippet=f"Price {_format_market_value(quote.get('price'), currency) or 'unavailable'}; volume {_format_market_value(quote.get('volume')) or 'unavailable'}",
            )
        )
    if ratios:
        ratio_meta = (ratios_payload or {}).get("metadata") or {}
        sources.append(
            _market_source(
                title=f"{ticker} financial ratios",
                source_table="financial_ratios",
                snapshot_date=ratio_meta.get("as_of"),
                snippet=f"{len(ratios)} ratio records available in the local processed store",
            )
        )
    if events:
        event_meta = (events_payload or {}).get("metadata") or {}
        sources.append(
            _market_source(
                title=f"{ticker} company events",
                source_table="company_events",
                snapshot_date=event_meta.get("as_of"),
                snippet=f"{len(events)} recent event records available in the local processed store",
            )
        )
    latest_dates = [source.get("snapshot_date") for source in sources if source.get("snapshot_date")]
    return {
        "status": "answer_ready",
        "provider": "local-market-data",
        "model": LOCAL_MARKET_DATA_MODEL,
        "answer": " ".join(answer_sections),
        "query": payload.message,
        "ticker": ticker,
        "evidence_count": len(sources),
        "latest_source_date": max(latest_dates) if latest_dates else None,
        "sources": sources,
        "source_tables": [source["source_table"] for source in sources],
        "recommended_workflow": {
            "title": "Cited company research",
            "next_step": "Open the attached evidence records before using this in a buyer or advisor workflow.",
        },
    }


def _tickertape_row_score(row: dict[str, Any], tokens: list[str]) -> float:
    info = _parse_json_blob(row.get("security_info_json")) or {}
    nested_info = info.get("info") if isinstance(info, dict) and isinstance(info.get("info"), dict) else {}
    ratios = nested_info.get("ratios") if isinstance(nested_info.get("ratios"), dict) else {}
    market_cap = ratios.get("marketCap") or (info.get("ratios") or {}).get("marketCap") if isinstance(info, dict) else None
    try:
        score = float(market_cap or 0)
    except (TypeError, ValueError):
        score = 0.0
    name = str(row.get("name") or "").upper()
    subdirectory = str(row.get("subdirectory") or "").upper()
    ticker = str(nested_info.get("ticker") or info.get("sid") or subdirectory.rsplit("-", 1)[-1]).upper()
    if ticker in tokens:
        score += 1_000_000_000_000
    if any(token in name for token in tokens):
        score += 1_000_000_000
    return score


def _resolve_tickertape_row(conn: sqlite3.Connection, prompt: str | None) -> dict[str, Any] | None:
    if not table_exists(conn, "latest_stock_data"):
        return None
    tokens = _market_query_tokens(prompt)
    if not tokens:
        return None
    clauses = " OR ".join("(LOWER(name) LIKE ? OR LOWER(subdirectory) LIKE ?)" for _ in tokens)
    params = [item for token in tokens for item in (f"%{token.lower()}%", f"%{token.lower()}%")]
    try:
        rows = [dict(row) for row in conn.execute(
            f"""
            SELECT *
            FROM latest_stock_data
            WHERE {clauses}
            """,
            params,
        ).fetchall()]
    except sqlite3.Error:
        return None
    if not rows:
        return None
    return max(rows, key=lambda row: _tickertape_row_score(row, tokens))


def _plain_number(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        return f"{float(value):,.2f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def _median_numeric(values: list[float]) -> float | None:
    clean = sorted(value for value in values if value is not None)
    if not clean:
        return None
    middle = len(clean) // 2
    if len(clean) % 2 == 1:
        return float(clean[middle])
    return float((clean[middle - 1] + clean[middle]) / 2)


def _build_backtesting_market_data_answer(
    payload: LocalLlmChatRequest,
    conn: sqlite3.Connection,
) -> dict[str, Any] | None:
    product_id = _normalized_product_id(payload)
    if product_id != "backtesting":
        return None
    if not table_exists(conn, "companies"):
        return None

    try:
        rows = [dict(row) for row in conn.execute(
            """
            SELECT c.ticker, c.name, c.sector, c.market_cap,
                   q.price, q.previous_close, q.volume, q.as_of
            FROM companies c
            LEFT JOIN quote_snapshots q
              ON q.quote_id = (
                SELECT qs.quote_id
                FROM quote_snapshots qs
                WHERE qs.ticker = c.ticker
                ORDER BY qs.as_of DESC, qs.processed_at DESC
                LIMIT 1
              )
            ORDER BY c.market_cap IS NULL, c.market_cap DESC, c.ticker
            LIMIT 240
            """
        ).fetchall()]
    except sqlite3.Error:
        return None

    if not rows:
        return None

    candidates = []
    for row in rows:
        price = row.get("price")
        previous_close = row.get("previous_close")
        volume = row.get("volume")
        try:
            turnover = float(price) * float(volume) if price is not None and volume is not None else None
        except (TypeError, ValueError):
            turnover = None
        try:
            change_pct = (
                ((float(price) - float(previous_close)) / float(previous_close) * 100.0)
                if price is not None and previous_close not in (None, 0, 0.0)
                else None
            )
        except (TypeError, ValueError, ZeroDivisionError):
            change_pct = None
        risk_proxy = abs(change_pct) if change_pct is not None else None
        candidates.append(
            {
                "ticker": str(row.get("ticker") or "").upper(),
                "name": row.get("name"),
                "sector": row.get("sector"),
                "as_of": row.get("as_of"),
                "market_cap": row.get("market_cap"),
                "turnover": turnover,
                "change_pct": change_pct,
                "risk_proxy": risk_proxy,
            }
        )

    scored = [item for item in candidates if item.get("ticker")]
    if not scored:
        return None

    turnover_median = _median_numeric([float(item["turnover"]) for item in scored if item.get("turnover") is not None])
    risk_median = _median_numeric([float(item["risk_proxy"]) for item in scored if item.get("risk_proxy") is not None])

    filtered = []
    for item in scored:
        turnover = item.get("turnover")
        risk_proxy = item.get("risk_proxy")
        change_pct = item.get("change_pct")
        meets_turnover = turnover_median is None or (turnover is not None and turnover >= turnover_median)
        meets_risk = risk_median is None or (risk_proxy is not None and risk_proxy <= risk_median)
        meets_momentum = change_pct is None or change_pct > 0
        if meets_turnover and meets_risk and meets_momentum:
            filtered.append(item)

    status = "rule_matched"
    if not filtered:
        status = "rule_degraded_to_turnover_rank"
        filtered = sorted(scored, key=lambda item: item.get("turnover") or 0.0, reverse=True)[:6]
    else:
        filtered.sort(key=lambda item: item.get("turnover") or 0.0, reverse=True)
        filtered = filtered[:6]

    latest_dates = [str(item.get("as_of")) for item in filtered if item.get("as_of")]
    candidate_labels = ", ".join(
        f"{item.get('name') or item['ticker']} ({item['ticker']})"
        for item in filtered[:3]
    )
    answer = (
        f"Backtesting dry run scanned {len(scored)} processed companies using a deterministic rule "
        f"(turnover >= median, risk <= median, positive momentum when available). "
        f"Matched {len(filtered)} candidates. Top names: {candidate_labels or 'none yet'}."
    )
    if status != "rule_matched":
        answer += " Rule fallback used turnover ranking because strict rule conditions had no matches."
    answer += " This is processed local market-data context, not investment advice."

    sources = [
        _market_source(
            title="Backtesting universe",
            source_table="companies",
            snapshot_date=max(latest_dates) if latest_dates else None,
            snippet=f"{len(scored)} processed companies considered for deterministic dry run",
        ),
    ]
    if any(item.get("as_of") for item in scored):
        sources.append(
            _market_source(
                title="Backtesting quote snapshots",
                source_table="quote_snapshots",
                snapshot_date=max(str(item.get("as_of")) for item in scored if item.get("as_of")),
                snippet="Latest per-ticker quote snapshots used for turnover and momentum proxies",
            )
        )

    return {
        "status": "answer_ready",
        "provider": "local-market-data",
        "model": LOCAL_MARKET_DATA_MODEL,
        "answer": answer,
        "query": payload.message,
        "product_id": "backtesting",
        "backtesting": {
            "status": status,
            "rule": "turnover>=median && risk<=median && change_pct>0_when_available",
            "universe_size": len(scored),
            "match_count": len(filtered),
            "turnover_median": turnover_median,
            "risk_median": risk_median,
            "candidates": filtered,
        },
        "evidence_count": len(sources),
        "latest_source_date": max(latest_dates) if latest_dates else None,
        "sources": sources,
        "source_tables": [source["source_table"] for source in sources],
        "recommended_workflow": {
            "title": "Deterministic backtesting dry run",
            "next_step": "Refine factors and re-run against fresh processed snapshots before using the output in any buyer or advisor workflow.",
        },
    }


def _build_tickertape_market_data_answer(
    payload: LocalLlmChatRequest,
    conn: sqlite3.Connection,
) -> dict[str, Any] | None:
    row = _resolve_tickertape_row(conn, payload.message)
    if row is None:
        return None
    info = _parse_json_blob(row.get("security_info_json")) or {}
    quote = _parse_json_blob(row.get("security_quote_json")) or {}
    scorecard = _parse_json_blob(row.get("scorecard_json")) or []
    nested_info = info.get("info") if isinstance(info, dict) and isinstance(info.get("info"), dict) else {}
    gic = info.get("gic") if isinstance(info, dict) and isinstance(info.get("gic"), dict) else {}
    ratios = nested_info.get("ratios") if isinstance(nested_info.get("ratios"), dict) else {}
    ticker = str(nested_info.get("ticker") or info.get("sid") or row.get("subdirectory", "").rsplit("-", 1)[-1]).upper()
    name = str(row.get("name") or nested_info.get("name") or ticker)
    sector = gic.get("sector") or nested_info.get("sector")
    industry = gic.get("industry") or nested_info.get("industry")
    description = nested_info.get("description")
    price = quote.get("c") if isinstance(quote, dict) else None
    exchange = quote.get("exchange") if isinstance(quote, dict) else None
    dy_change = quote.get("dyChange") if isinstance(quote, dict) else None
    market_cap = ratios.get("marketCap")
    pe = ratios.get("ttmPe") or ratios.get("apef")
    summary_items = []
    if isinstance(scorecard, list):
        summary_items = [
            item.get("description")
            for item in scorecard
            if isinstance(item, dict) and item.get("description")
        ][:2]

    answer_sections = [f"{name} ({ticker}) is available in the local Tickertape mirror."]
    if exchange or sector or industry:
        answer_sections.append(
            ", ".join(part for part in [
                f"Exchange: {exchange}" if exchange else None,
                f"sector: {sector}" if sector else None,
                f"industry: {industry}" if industry else None,
            ] if part) + "."
        )
    if description:
        answer_sections.append(str(description).strip())
    if price is not None:
        movement = f", day change {_plain_number(dy_change)}%" if dy_change is not None else ""
        answer_sections.append(f"Latest local quote field is {_plain_number(price)}{movement}.")
    metric_parts = []
    if market_cap is not None:
        metric_parts.append(f"market-cap field {_plain_number(market_cap)}")
    if pe is not None:
        metric_parts.append(f"TTM PE {_plain_number(pe)}")
    if metric_parts:
        answer_sections.append("Processed ratio context: " + "; ".join(metric_parts) + ".")
    if summary_items:
        answer_sections.append("Scorecard notes: " + " ".join(str(item).strip() for item in summary_items))
    answer_sections.append("This is processed local market-data context, not investment advice.")

    sources = [
        _market_source(
            title=f"{ticker} latest stock data",
            source_table="latest_stock_data",
            snapshot_date=row.get("snapshot_date"),
            snippet=f"{name}; raw path {row.get('raw_json_path') or 'unavailable'}",
        )
    ]
    for table_name in ("financial_sections", "event_sections"):
        if not table_exists(conn, table_name):
            continue
        count_row = conn.execute(
            f"SELECT COUNT(*) AS count, MAX(snapshot_date) AS snapshot_date FROM {table_name} WHERE subdirectory = ?",
            (row.get("subdirectory"),),
        ).fetchone()
        if count_row is not None and int(count_row["count"] or 0) > 0:
            sources.append(
                _market_source(
                    title=f"{ticker} {table_name.replace('_', ' ')}",
                    source_table=table_name,
                    snapshot_date=count_row["snapshot_date"],
                    snippet=f"{count_row['count']} processed section records available locally",
                )
            )
    latest_dates = [source.get("snapshot_date") for source in sources if source.get("snapshot_date")]
    return {
        "status": "answer_ready",
        "provider": "local-market-data",
        "model": LOCAL_MARKET_DATA_MODEL,
        "answer": " ".join(answer_sections),
        "query": payload.message,
        "ticker": ticker,
        "evidence_count": len(sources),
        "latest_source_date": max(latest_dates) if latest_dates else None,
        "sources": sources,
        "source_tables": [source["source_table"] for source in sources],
        "recommended_workflow": {
            "title": "Cited company research",
            "next_step": "Open the attached local Tickertape evidence before buyer or advisor use.",
        },
    }


def _build_market_data_answer(
    payload: LocalLlmChatRequest,
    conn: sqlite3.Connection,
) -> dict[str, Any] | None:
    return (
        _build_backtesting_market_data_answer(payload, conn)
        or
        _build_canonical_market_data_answer(payload, conn)
        or _build_tickertape_market_data_answer(payload, conn)
    )


def _payload_prompt(payload: LocalLlmChatRequest) -> str:
    message = (payload.message or "").strip()
    if message:
        return message
    if payload.messages:
        for item in reversed(payload.messages):
            content = str(item.get("content") or "").strip()
            if content:
                return content
    return ""


def _lead_gen_intent_details(prompt: str) -> dict[str, str]:
    lowered = (prompt or "").lower()
    rules = [
        ("advisory", "Advisory and planner request", "Registered advisor desk", ("advisor", "advisory", "ria", "pms", "wealth")),
        ("research", "Research and diligence request", "Research operations desk", ("research", "analysis", "diligence", "peer", "thesis")),
        ("data", "Data/API request", "Data infrastructure desk", ("api", "data", "dataset", "feed", "export", "integration")),
        ("tooling", "Workflow tooling request", "Product operations desk", ("tool", "workflow", "automation", "dashboard", "copilot")),
        ("implementation", "Implementation request", "Implementation partner desk", ("implement", "deploy", "migration", "onboarding", "setup", "build")),
    ]
    for intent_id, label, lane, keywords in rules:
        if any(keyword in lowered for keyword in keywords):
            return {"id": intent_id, "label": label, "lane": lane}
    return {"id": "general", "label": "General market-intelligence request", "lane": "Qualification triage desk"}


def _lead_gen_missing_fields(prompt: str) -> list[str]:
    lowered = (prompt or "").lower()
    checks = [
        ("buyer type and operating context", ("advisor", "ria", "team", "firm", "company", "founder", "operator", "investor")),
        ("scope (advisory, research, data, tooling, implementation)", ("advisory", "research", "data", "api", "tool", "workflow", "implementation", "integration")),
        ("timeline and urgency", ("today", "tomorrow", "week", "month", "quarter", "urgent", "asap", "deadline", "timeline")),
        ("budget range or commercial constraints", ("₹", "inr", "$", "budget", "pricing", "cost", "fee", "subscription")),
        ("explicit consent to share details with matched providers", ("consent", "approve", "permission", "disclosure", "share details", "allow")),
    ]
    missing = []
    for label, keywords in checks:
        if not any(keyword in lowered for keyword in keywords):
            missing.append(label)
    return missing


def _build_lead_gen_marketplace_answer(
    payload: LocalLlmChatRequest,
    reason: str = "",
) -> dict[str, Any] | None:
    if _normalized_product_id(payload) != LEAD_GEN_PRODUCT_ID:
        return None
    prompt = _payload_prompt(payload)
    if not prompt:
        return None
    intent = _lead_gen_intent_details(prompt)
    missing_fields = _lead_gen_missing_fields(prompt)
    missing_text = "\n".join(f"- {item}" for item in missing_fields) if missing_fields else "- None. Intake is complete enough to route."
    fallback_reason = reason or "Live providers were unavailable for this request."
    answer = "\n".join(
        [
            f"Lead Category: {intent['label']}",
            f"Routing Recommendation: {intent['lane']}",
            "Missing Information:",
            missing_text,
            "Disclosure & Consent Checks:",
            "- Confirm non-advisory posture before provider contact.",
            "- Confirm explicit consent before sharing request details.",
            "- Capture data-usage boundaries for shared artifacts.",
            "Handoff Steps:",
            "- Assign a human reviewer for compliance sign-off.",
            "- Attach the intake packet, evidence context, and consent status.",
            "- Route to the recommended lane and track acceptance.",
            f"Fallback Reason: {fallback_reason}",
        ]
    )
    sources = [
        _market_source(
            title="Lead routing policy contract",
            source_table="product_registry",
            snapshot_date=None,
            snippet="Deterministic demand-routing policy for product 13 (Qualified Demand Router).",
        ),
        _market_source(
            title="Disclosure and consent control",
            source_table="compliance_disclosure",
            snapshot_date=None,
            snippet="Routing is blocked until disclosure and consent gates are satisfied.",
        ),
    ]
    return {
        "status": "answer_ready",
        "provider": LEAD_GEN_ROUTER_PROVIDER,
        "model": LEAD_GEN_ROUTER_MODEL,
        "answer": answer,
        "query": prompt,
        "product_id": LEAD_GEN_PRODUCT_ID,
        "intent": intent,
        "missing_fields": missing_fields,
        "evidence_count": len(sources),
        "sources": sources,
        "source_tables": [source["source_table"] for source in sources],
        "recommended_workflow": {
            "title": "Qualified demand routing",
            "next_step": "Collect missing fields, confirm disclosure consent, then route through the recommended lane.",
        },
    }


def _build_local_llm_guaranteed_fallback(
    payload: LocalLlmChatRequest,
    conn: sqlite3.Connection,
    reason: str = "",
) -> dict[str, Any] | None:
    market_answer = _build_market_data_answer(payload, conn)
    if market_answer is not None:
        return market_answer
    return _build_lead_gen_marketplace_answer(payload, reason)


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
def local_llm_chat(
    payload: LocalLlmChatRequest,
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
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
    try:
        response = _request_llama_cpp_json("/chat/completions", request_payload)
    except HTTPException as exc:
        fallback = _build_local_llm_guaranteed_fallback(payload, conn, str(exc.detail))
        if fallback is not None:
            fallback.setdefault("llm_fallback_reason", exc.detail)
            return fallback
        raise exc
    try:
        answer = _extract_llama_cpp_answer(response)
    except HTTPException as exc:
        fallback = _build_local_llm_guaranteed_fallback(payload, conn, str(exc.detail))
        if fallback is not None:
            fallback.setdefault("llm_fallback_reason", exc.detail)
            return fallback
        raise exc
    return {
        "status": "answer_ready",
        "provider": "local-llama",
        "model": config["model"],
        "answer": answer,
        "product_id": _normalized_product_id(payload),
        "sources": [],
        "usage": response.get("usage", {}),
    }


@router.post("/unified-agentic-runtime")
def unified_agentic_runtime(
    payload: UnifiedAgenticRuntimeRequest,
    request: Request,
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_unified_agentic_runtime(
            conn,
            request.app.state.vector_index_dir,
            product_id=payload.product_id,
            objective=payload.objective,
            owner_id=payload.owner_id,
            retrieval_mode=payload.retrieval_mode,
            provider_request=payload.provider_preference,
            focus=payload.focus,
            sql_queries=payload.sql_queries,
            rag_query=payload.rag_query,
            run_id=payload.run_id,
            source_name=payload.source_name,
            evidence_limit=payload.evidence_limit,
            max_rows_per_query=payload.max_rows_per_query,
            include_deep_research=payload.include_deep_research,
            include_documents=payload.include_documents,
            auto_build_rag_index=payload.auto_build_rag_index,
            rag_build_limit=payload.rag_build_limit,
            context=payload.context,
            deep_research_infer=lambda prompt: _run_unified_deep_research(prompt, payload, conn),
        )
    except (VectorIndexError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
