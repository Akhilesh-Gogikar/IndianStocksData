from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..dependencies import get_connection
from ..market_service import MarketDataUnavailable, MarketRecordNotFound
from ..watchlist_service import (
    add_alert_rule,
    add_watchlist_item,
    alert_evaluation_readiness,
    alert_event_history,
    alert_event_review_audit_history,
    alert_event_review_summary,
    alert_review_audit_history,
    alert_rule_event_history,
    alert_rule_event_review_summary,
    create_watchlist,
    create_webhook_subscription,
    disable_webhook_subscription,
    disable_alert_rule,
    enqueue_webhook_subscription_test,
    evaluate_alerts,
    get_watchlist,
    list_alert_rules,
    list_webhook_subscriptions,
    list_watchlists,
    owner_alert_evaluation_readiness,
    owner_alert_readiness_actions,
    remove_watchlist_item,
    replay_webhook_outbox,
    save_owner_alert_readiness_action_queue,
    update_alert_rule,
    update_alert_event_review,
    update_alert_event_reviews,
    update_webhook_subscription,
    webhook_delivery_attempts,
    webhook_outbox,
    webhook_status,
)

router = APIRouter(prefix="/watchlists", tags=["watchlists"])


class WatchlistCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: str | None = None
    owner_id: str = "default"


class WatchlistItemCreate(BaseModel):
    ticker: str = Field(..., min_length=1)
    notes: str | None = None
    owner_id: str = "default"


class AlertRuleCreate(BaseModel):
    metric: str = Field(..., min_length=1)
    operator: str = Field(..., pattern="^(lt|lte|gt|gte)$")
    threshold: float
    cooldown_minutes: int = Field(default=60, ge=0)
    ticker: str | None = None
    enabled: bool = True
    owner_id: str = "default"


class AlertRuleUpdate(BaseModel):
    owner_id: str = "default"
    metric: str | None = Field(default=None, min_length=1)
    operator: str | None = Field(default=None, pattern="^(lt|lte|gt|gte)$")
    threshold: float | None = None
    cooldown_minutes: int | None = Field(default=None, ge=0)
    ticker: str | None = None
    enabled: bool | None = None


class AlertEventReviewUpdate(BaseModel):
    owner_id: str = "default"
    status: str = Field(..., pattern="^(open|reviewed|dismissed)$")
    reviewed_by: str | None = None
    notes: str | None = None


class AlertEventBulkReviewUpdate(BaseModel):
    owner_id: str = "default"
    event_ids: list[int] | None = Field(default=None, min_length=1, max_length=500)
    current_status: str | None = Field(default=None, pattern="^(open|reviewed|dismissed)$")
    rule_id: int | None = Field(default=None, ge=1)
    limit: int = Field(default=500, ge=1, le=500)
    status: str = Field(..., pattern="^(open|reviewed|dismissed)$")
    reviewed_by: str | None = None
    notes: str | None = None


class WebhookSubscriptionCreate(BaseModel):
    endpoint_url: str = Field(..., min_length=1)
    event_type: str = "watchlist.alert_triggered"
    signing_secret: str | None = Field(default=None, min_length=16)
    enabled: bool = True
    owner_id: str = "default"


class WebhookSubscriptionUpdate(BaseModel):
    owner_id: str = "default"
    endpoint_url: str | None = Field(default=None, min_length=1)
    event_type: str | None = None
    signing_secret: str | None = Field(default=None, min_length=16)
    enabled: bool | None = None


class WebhookReplayRequest(BaseModel):
    owner_id: str = "default"
    reset_attempts: bool = True
    reason: str | None = None


class WebhookSubscriptionTestRequest(BaseModel):
    owner_id: str = "default"
    message: str | None = None


def api_error(exc: Exception) -> None:
    if isinstance(exc, MarketDataUnavailable):
        raise HTTPException(status_code=503, detail={"code": "watchlists_unavailable", "message": str(exc)}) from exc
    if isinstance(exc, MarketRecordNotFound):
        raise HTTPException(status_code=404, detail={"code": "watchlist_not_found", "message": str(exc)}) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail={"code": "invalid_watchlist_request", "message": str(exc)}) from exc
    raise exc


@router.get("")
def watchlists_index(
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_watchlists(conn, owner_id)
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.get("/alerts/readiness")
def watchlists_owner_alert_readiness(
    owner_id: str = Query(default="default"),
    include_available: bool = Query(default=False),
    warn_days: int = Query(default=2, ge=0),
    stale_days: int = Query(default=5, ge=0),
    status: str | None = Query(default=None),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return owner_alert_evaluation_readiness(
            conn,
            owner_id,
            include_available=include_available,
            warn_days=warn_days,
            stale_days=stale_days,
            status=status,
        )
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.get("/alerts/readiness/actions")
def watchlists_owner_alert_readiness_actions(
    owner_id: str = Query(default="default"),
    warn_days: int = Query(default=2, ge=0),
    stale_days: int = Query(default=5, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return owner_alert_readiness_actions(
            conn,
            owner_id,
            warn_days=warn_days,
            stale_days=stale_days,
            limit=limit,
        )
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.post("/alerts/readiness/action-queue")
def watchlists_save_owner_alert_readiness_action_queue(
    owner_id: str = Query(default="default"),
    warn_days: int = Query(default=2, ge=0),
    stale_days: int = Query(default=5, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    title: str | None = Query(default=None),
    replace_existing: bool = Query(default=True),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return save_owner_alert_readiness_action_queue(
            conn,
            owner_id,
            warn_days=warn_days,
            stale_days=stale_days,
            limit=limit,
            title=title,
            replace_existing=replace_existing,
        )
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.post("")
def watchlists_create(
    request: WatchlistCreate,
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return {"data": create_watchlist(conn, request.owner_id, request.name, request.description)}
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.get("/{watchlist_id}")
def watchlists_show(
    watchlist_id: int,
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return get_watchlist(conn, watchlist_id, owner_id)
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.post("/{watchlist_id}/items")
def watchlists_add_item(
    watchlist_id: int,
    request: WatchlistItemCreate,
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return add_watchlist_item(conn, watchlist_id, request.owner_id, request.ticker, request.notes)
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.delete("/{watchlist_id}/items/{ticker}")
def watchlists_remove_item(
    watchlist_id: int,
    ticker: str,
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return remove_watchlist_item(conn, watchlist_id, owner_id, ticker)
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.post("/{watchlist_id}/alerts")
def watchlists_add_alert(
    watchlist_id: int,
    request: AlertRuleCreate,
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        rule = add_alert_rule(
            conn,
            watchlist_id,
            request.owner_id,
            request.ticker,
            request.metric,
            request.operator,
            request.threshold,
            request.enabled,
            request.cooldown_minutes,
        )
        return {"data": rule}
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.get("/{watchlist_id}/alerts")
def watchlists_alert_rules(
    watchlist_id: int,
    owner_id: str = Query(default="default"),
    include_disabled: bool = Query(default=False),
    include_review_counts: bool = Query(default=False),
    needs_attention: bool = Query(default=False),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_alert_rules(
            conn,
            watchlist_id,
            owner_id,
            include_disabled=include_disabled,
            include_review_counts=include_review_counts,
            needs_attention=needs_attention,
        )
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.patch("/{watchlist_id}/alerts/{rule_id}")
def watchlists_update_alert(
    watchlist_id: int,
    rule_id: int,
    request: AlertRuleUpdate,
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        rule = update_alert_rule(
            conn,
            watchlist_id,
            request.owner_id,
            rule_id,
            ticker=request.ticker,
            metric=request.metric,
            operator=request.operator,
            threshold=request.threshold,
            cooldown_minutes=request.cooldown_minutes,
            enabled=request.enabled,
        )
        return {"data": rule}
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.delete("/{watchlist_id}/alerts/{rule_id}")
def watchlists_disable_alert(
    watchlist_id: int,
    rule_id: int,
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        rule = disable_alert_rule(conn, watchlist_id, owner_id, rule_id)
        return {"data": rule}
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.get("/{watchlist_id}/alerts/evaluate")
def watchlists_evaluate_alerts(
    watchlist_id: int,
    owner_id: str = Query(default="default"),
    record_events: bool = Query(default=False),
    warn_days: int = Query(default=2, ge=0),
    stale_days: int = Query(default=5, ge=0),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return evaluate_alerts(
            conn,
            watchlist_id,
            owner_id,
            record_events=record_events,
            warn_days=warn_days,
            stale_days=stale_days,
        )
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.get("/{watchlist_id}/alerts/readiness")
def watchlists_alert_readiness(
    watchlist_id: int,
    owner_id: str = Query(default="default"),
    include_available: bool = Query(default=False),
    warn_days: int = Query(default=2, ge=0),
    stale_days: int = Query(default=5, ge=0),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return alert_evaluation_readiness(
            conn,
            watchlist_id,
            owner_id,
            include_available=include_available,
            warn_days=warn_days,
            stale_days=stale_days,
        )
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.get("/{watchlist_id}/alerts/events")
def watchlists_alert_events(
    watchlist_id: int,
    owner_id: str = Query(default="default"),
    review_status: str | None = Query(default=None, pattern="^(open|reviewed|dismissed)$"),
    limit: int = Query(default=50, ge=1, le=500),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return alert_event_history(conn, watchlist_id, owner_id, limit=limit, review_status=review_status)
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.get("/{watchlist_id}/alerts/events/summary")
def watchlists_alert_event_review_summary(
    watchlist_id: int,
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return alert_event_review_summary(conn, watchlist_id, owner_id)
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.patch("/{watchlist_id}/alerts/events")
@router.patch("/{watchlist_id}/alerts/events/bulk")
def watchlists_update_alert_event_reviews(
    watchlist_id: int,
    request: AlertEventBulkReviewUpdate,
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return update_alert_event_reviews(
            conn,
            watchlist_id,
            request.owner_id,
            request.event_ids,
            request.status,
            reviewed_by=request.reviewed_by,
            notes=request.notes,
            current_status=request.current_status,
            rule_id=request.rule_id,
            limit=request.limit,
        )
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.patch("/{watchlist_id}/alerts/events/{event_id}")
def watchlists_update_alert_event_review(
    watchlist_id: int,
    event_id: int,
    request: AlertEventReviewUpdate,
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return update_alert_event_review(
            conn,
            watchlist_id,
            request.owner_id,
            event_id,
            request.status,
            reviewed_by=request.reviewed_by,
            notes=request.notes,
        )
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.get("/{watchlist_id}/alerts/events/{event_id}/reviews")
def watchlists_alert_event_review_audits(
    watchlist_id: int,
    event_id: int,
    owner_id: str = Query(default="default"),
    limit: int = Query(default=50, ge=1, le=500),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return alert_event_review_audit_history(conn, watchlist_id, owner_id, event_id, limit=limit)
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.get("/{watchlist_id}/alerts/reviews")
def watchlists_alert_review_audits(
    watchlist_id: int,
    owner_id: str = Query(default="default"),
    event_id: int | None = Query(default=None, ge=1),
    rule_id: int | None = Query(default=None, ge=1),
    status: str | None = Query(default=None, pattern="^(open|reviewed|dismissed)$"),
    source: str | None = Query(default=None, pattern="^(single|bulk)$"),
    limit: int = Query(default=50, ge=1, le=500),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return alert_review_audit_history(
            conn,
            watchlist_id,
            owner_id,
            event_id=event_id,
            rule_id=rule_id,
            status=status,
            source=source,
            limit=limit,
        )
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.get("/{watchlist_id}/alerts/{rule_id}/events/summary")
def watchlists_alert_rule_event_review_summary(
    watchlist_id: int,
    rule_id: int,
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return alert_rule_event_review_summary(conn, watchlist_id, owner_id, rule_id)
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.get("/{watchlist_id}/alerts/{rule_id}/events")
def watchlists_alert_rule_events(
    watchlist_id: int,
    rule_id: int,
    owner_id: str = Query(default="default"),
    review_status: str | None = Query(default=None, pattern="^(open|reviewed|dismissed)$"),
    limit: int = Query(default=50, ge=1, le=500),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return alert_rule_event_history(conn, watchlist_id, owner_id, rule_id, limit=limit, review_status=review_status)
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.get("/webhooks/subscriptions")
def watchlists_webhook_subscriptions(
    owner_id: str = Query(default="default"),
    include_disabled: bool = Query(default=False),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return list_webhook_subscriptions(conn, owner_id, include_disabled=include_disabled)
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.post("/webhooks/subscriptions")
def watchlists_create_webhook_subscription(
    request: WebhookSubscriptionCreate,
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        subscription = create_webhook_subscription(
            conn,
            request.owner_id,
            request.endpoint_url,
            event_type=request.event_type,
            signing_secret=request.signing_secret,
            enabled=request.enabled,
        )
        return {"data": subscription}
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.patch("/webhooks/subscriptions/{subscription_id}")
def watchlists_update_webhook_subscription(
    subscription_id: int,
    request: WebhookSubscriptionUpdate,
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return update_webhook_subscription(
            conn,
            subscription_id,
            request.owner_id,
            endpoint_url=request.endpoint_url,
            event_type=request.event_type,
            signing_secret=request.signing_secret,
            enabled=request.enabled,
        )
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.delete("/webhooks/subscriptions/{subscription_id}")
def watchlists_disable_webhook_subscription(
    subscription_id: int,
    owner_id: str = Query(default="default"),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return disable_webhook_subscription(conn, subscription_id, owner_id)
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.post("/webhooks/subscriptions/{subscription_id}/test")
def watchlists_test_webhook_subscription(
    subscription_id: int,
    request: WebhookSubscriptionTestRequest,
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return enqueue_webhook_subscription_test(
            conn,
            subscription_id,
            owner_id=request.owner_id,
            message=request.message,
        )
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.get("/webhooks/status")
def watchlists_webhook_status(
    owner_id: str | None = Query(default=None),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return webhook_status(conn, owner_id=owner_id)
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.get("/webhooks/outbox")
def watchlists_webhook_outbox(
    status: str = Query(default="pending"),
    owner_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return webhook_outbox(conn, status=status, limit=limit, owner_id=owner_id)
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.post("/webhooks/outbox/{outbox_id}/replay")
def watchlists_replay_webhook_outbox(
    outbox_id: int,
    request: WebhookReplayRequest,
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return replay_webhook_outbox(
            conn,
            outbox_id,
            owner_id=request.owner_id,
            reset_attempts=request.reset_attempts,
            reason=request.reason,
        )
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)


@router.get("/webhooks/deliveries")
def watchlists_webhook_deliveries(
    owner_id: str | None = Query(default=None),
    outbox_id: int | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return webhook_delivery_attempts(conn, owner_id=owner_id, outbox_id=outbox_id, status=status, limit=limit)
    except (MarketDataUnavailable, MarketRecordNotFound, ValueError) as exc:
        api_error(exc)
