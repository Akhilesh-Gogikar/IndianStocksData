from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

from .freshness_service import age_days as freshness_age_days
from .freshness_service import status_for_age
from .market_service import (
    MarketDataUnavailable,
    MarketRecordNotFound,
    metadata_from_rows,
    normalize_ticker,
    public_record,
    table_exists,
)


VALID_OPERATORS = {"lt", "lte", "gt", "gte"}
VALID_EVENT_REVIEW_STATUSES = {"open", "reviewed", "dismissed"}
BLOCKING_QUALITY_STATUSES = {"fail", "failed"}
BLOCKING_DATA_RIGHTS_STATUSES = {
    "blocked",
    "raw",
    "raw-only",
    "restricted",
    "source-derived-review-required",
    "source-review-required",
}


def now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def require_watchlist_tables(conn: sqlite3.Connection) -> None:
    for table_name in ("watchlists", "watchlist_items", "watchlist_alert_rules"):
        if not table_exists(conn, table_name):
            raise MarketDataUnavailable(f"Watchlist table '{table_name}' is not available")


def require_alert_event_tables(conn: sqlite3.Connection) -> None:
    for table_name in (
        "alert_events",
        "alert_event_reviews",
        "alert_event_review_audits",
        "webhook_subscriptions",
        "webhook_outbox",
        "webhook_delivery_attempts",
    ):
        if not table_exists(conn, table_name):
            raise MarketDataUnavailable(f"Alert event table '{table_name}' is not available")


def clean_owner_id(owner_id: str | None) -> str:
    return (owner_id or "default").strip() or "default"


def clean_event_type(event_type: str | None) -> str:
    return (event_type or "watchlist.alert_triggered").strip() or "watchlist.alert_triggered"


def clean_endpoint_url(endpoint_url: str) -> str:
    endpoint = endpoint_url.strip()
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Webhook endpoint_url must be an absolute http(s) URL")
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Webhook endpoint_url must use https outside localhost")
    return endpoint


def clean_signing_secret(signing_secret: str | None) -> str | None:
    if signing_secret is None:
        return None
    secret = signing_secret.strip()
    if not secret:
        return None
    if len(secret) < 16:
        raise ValueError("Webhook signing_secret must be at least 16 characters")
    return secret


def clean_event_review_status(review_status: str | None) -> str | None:
    if review_status is None:
        return None
    status = review_status.strip().lower()
    if not status:
        return None
    if status not in VALID_EVENT_REVIEW_STATUSES:
        supported = ", ".join(sorted(VALID_EVENT_REVIEW_STATUSES))
        raise ValueError(f"Unsupported alert event review status '{review_status}'. Expected one of: {supported}")
    return status


def clean_cooldown_minutes(cooldown_minutes: int | None) -> int:
    cooldown = 60 if cooldown_minutes is None else int(cooldown_minutes)
    if cooldown < 0:
        raise ValueError("Alert cooldown_minutes must be greater than or equal to 0")
    return cooldown


def public_webhook_subscription(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload["enabled"] = bool(payload["enabled"])
    payload["secret_set"] = bool(payload.pop("signing_secret", None))
    return payload


def watchlist_row(conn: sqlite3.Connection, watchlist_id: int, owner_id: str) -> dict[str, Any]:
    require_watchlist_tables(conn)
    row = conn.execute(
        """
        SELECT *
        FROM watchlists
        WHERE watchlist_id = ? AND owner_id = ?
        """,
        (watchlist_id, clean_owner_id(owner_id)),
    ).fetchone()
    if not row:
        raise MarketRecordNotFound(f"No watchlist found for id {watchlist_id}")
    return dict(row)


def create_webhook_subscription(
    conn: sqlite3.Connection,
    owner_id: str | None,
    endpoint_url: str,
    event_type: str | None = "watchlist.alert_triggered",
    signing_secret: str | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    require_alert_event_tables(conn)
    owner = clean_owner_id(owner_id)
    event = clean_event_type(event_type)
    endpoint = clean_endpoint_url(endpoint_url)
    secret = clean_signing_secret(signing_secret)
    now = now_utc()
    cursor = conn.execute(
        """
        INSERT INTO webhook_subscriptions (
            owner_id, event_type, endpoint_url, signing_secret, enabled, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(owner_id, event_type, endpoint_url) DO UPDATE SET
            signing_secret = COALESCE(excluded.signing_secret, webhook_subscriptions.signing_secret),
            enabled = excluded.enabled,
            updated_at = excluded.updated_at
        """,
        (owner, event, endpoint, secret, 1 if enabled else 0, now, now),
    )
    row = conn.execute(
        """
        SELECT *
        FROM webhook_subscriptions
        WHERE owner_id = ? AND event_type = ? AND endpoint_url = ?
        """,
        (owner, event, endpoint),
    ).fetchone()
    conn.commit()
    payload = public_webhook_subscription(dict(row))
    payload["created"] = cursor.lastrowid != 0
    return payload


def update_webhook_subscription(
    conn: sqlite3.Connection,
    subscription_id: int,
    owner_id: str | None,
    endpoint_url: str | None = None,
    event_type: str | None = None,
    signing_secret: str | None = None,
    enabled: bool | None = None,
) -> dict[str, Any]:
    current = webhook_subscription_row(conn, subscription_id, owner_id)
    endpoint = clean_endpoint_url(endpoint_url) if endpoint_url is not None else current["endpoint_url"]
    event = clean_event_type(event_type) if event_type is not None else current["event_type"]
    secret = clean_signing_secret(signing_secret) if signing_secret is not None else current["signing_secret"]
    enabled_value = int(current["enabled"] if enabled is None else enabled)
    now = now_utc()
    try:
        conn.execute(
            """
            UPDATE webhook_subscriptions
            SET endpoint_url = ?,
                event_type = ?,
                signing_secret = ?,
                enabled = ?,
                updated_at = ?
            WHERE subscription_id = ? AND owner_id = ?
            """,
            (endpoint, event, secret, enabled_value, now, subscription_id, current["owner_id"]),
        )
        conn.commit()
    except sqlite3.IntegrityError as exc:
        raise ValueError("Webhook subscription endpoint already exists for this owner and event type") from exc
    updated = webhook_subscription_row(conn, subscription_id, current["owner_id"])
    return {"data": public_webhook_subscription(updated), "metadata": {"updated": True}}


def disable_webhook_subscription(
    conn: sqlite3.Connection,
    subscription_id: int,
    owner_id: str | None,
) -> dict[str, Any]:
    return update_webhook_subscription(conn, subscription_id, owner_id, enabled=False)


def list_webhook_subscriptions(
    conn: sqlite3.Connection,
    owner_id: str | None,
    include_disabled: bool = False,
) -> dict[str, Any]:
    require_alert_event_tables(conn)
    owner = clean_owner_id(owner_id)
    query = """
        SELECT *
        FROM webhook_subscriptions
        WHERE owner_id = ?
    """
    params: list[Any] = [owner]
    if not include_disabled:
        query += " AND enabled = 1"
    query += " ORDER BY event_type, subscription_id"
    rows = conn.execute(query, params).fetchall()
    data = []
    for row in rows:
        data.append(public_webhook_subscription(dict(row)))
    return {"data": data, "metadata": {"owner_id": owner, "result_count": len(data)}}


def webhook_subscription_row(
    conn: sqlite3.Connection,
    subscription_id: int,
    owner_id: str | None,
) -> dict[str, Any]:
    require_alert_event_tables(conn)
    owner = clean_owner_id(owner_id)
    row = conn.execute(
        """
        SELECT *
        FROM webhook_subscriptions
        WHERE subscription_id = ? AND owner_id = ?
        """,
        (subscription_id, owner),
    ).fetchone()
    if not row:
        raise MarketRecordNotFound(f"No webhook subscription found for id {subscription_id}")
    return dict(row)


def enqueue_webhook_subscription_test(
    conn: sqlite3.Connection,
    subscription_id: int,
    owner_id: str | None,
    message: str | None = None,
) -> dict[str, Any]:
    subscription = webhook_subscription_row(conn, subscription_id, owner_id)
    if not subscription["enabled"]:
        raise ValueError("Disabled webhook subscriptions cannot be tested")
    now = now_utc()
    payload = {
        "owner_id": subscription["owner_id"],
        "subscription_id": subscription_id,
        "event_type": "webhook.subscription_test",
        "message": (message or "Webhook subscription test event").strip(),
        "requested_at": now,
    }
    cursor = conn.execute(
        """
        INSERT INTO webhook_outbox (
            owner_id, subscription_id, destination_url, event_type,
            aggregate_type, aggregate_id, payload_json, status, created_at, next_attempt_at
        )
        VALUES (?, ?, ?, 'webhook.subscription_test',
                'webhook_subscription', ?, ?, 'pending', ?, ?)
        """,
        (
            subscription["owner_id"],
            subscription_id,
            subscription["endpoint_url"],
            subscription_id,
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            now,
            now,
        ),
    )
    conn.commit()
    outbox = webhook_outbox_row(conn, int(cursor.lastrowid), owner_id=subscription["owner_id"])
    return {
        "data": outbox,
        "metadata": {
            "queued": True,
            "subscription_id": subscription_id,
            "owner_id": subscription["owner_id"],
        },
    }


def active_webhook_subscriptions(
    conn: sqlite3.Connection,
    owner_id: str,
    event_type: str,
) -> list[dict[str, Any]]:
    require_alert_event_tables(conn)
    rows = conn.execute(
        """
        SELECT *
        FROM webhook_subscriptions
        WHERE owner_id = ? AND event_type = ? AND enabled = 1
        ORDER BY subscription_id
        """,
        (clean_owner_id(owner_id), clean_event_type(event_type)),
    ).fetchall()
    return [dict(row) for row in rows]


def create_watchlist(conn: sqlite3.Connection, owner_id: str | None, name: str, description: str | None) -> dict[str, Any]:
    require_watchlist_tables(conn)
    now = now_utc()
    owner = clean_owner_id(owner_id)
    cursor = conn.execute(
        """
        INSERT INTO watchlists (owner_id, name, description, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(owner_id, name) DO UPDATE SET
            description = excluded.description,
            updated_at = excluded.updated_at
        """,
        (owner, name.strip(), description, now, now),
    )
    row = conn.execute(
        """
        SELECT *
        FROM watchlists
        WHERE owner_id = ? AND name = ?
        """,
        (owner, name.strip()),
    ).fetchone()
    conn.commit()
    payload = dict(row)
    payload["created"] = cursor.lastrowid != 0
    return payload


def list_watchlists(conn: sqlite3.Connection, owner_id: str | None) -> dict[str, Any]:
    require_watchlist_tables(conn)
    rows = conn.execute(
        """
        SELECT w.*, COUNT(i.item_id) AS item_count
        FROM watchlists w
        LEFT JOIN watchlist_items i ON i.watchlist_id = w.watchlist_id
        WHERE w.owner_id = ?
        GROUP BY w.watchlist_id
        ORDER BY w.updated_at DESC, w.watchlist_id DESC
        """,
        (clean_owner_id(owner_id),),
    ).fetchall()
    return {"data": [dict(row) for row in rows], "metadata": {"result_count": len(rows)}}


def latest_company(conn: sqlite3.Connection, ticker: str) -> dict[str, Any] | None:
    if not table_exists(conn, "companies"):
        return None
    row = conn.execute(
        "SELECT * FROM companies WHERE UPPER(ticker) = ?",
        (normalize_ticker(ticker),),
    ).fetchone()
    return dict(row) if row else None


def latest_quote(conn: sqlite3.Connection, ticker: str) -> dict[str, Any] | None:
    if not table_exists(conn, "quote_snapshots"):
        return None
    row = conn.execute(
        """
        SELECT *
        FROM quote_snapshots
        WHERE UPPER(ticker) = ?
        ORDER BY as_of DESC, processed_at DESC, quote_id DESC
        LIMIT 1
        """,
        (normalize_ticker(ticker),),
    ).fetchone()
    return dict(row) if row else None


def latest_ratios(conn: sqlite3.Connection, ticker: str) -> dict[str, float | None]:
    if not table_exists(conn, "financial_ratios"):
        return {}
    rows = conn.execute(
        """
        SELECT *
        FROM financial_ratios
        WHERE UPPER(ticker) = ?
        ORDER BY ratio_name, period_end DESC, as_of DESC, processed_at DESC, ratio_id DESC
        """,
        (normalize_ticker(ticker),),
    ).fetchall()
    ratios: dict[str, float | None] = {}
    for row in rows:
        ratios.setdefault(row["ratio_name"], row["ratio_value"])
    return ratios


def latest_ratio(conn: sqlite3.Connection, ticker: str, metric: str) -> dict[str, Any] | None:
    if not table_exists(conn, "financial_ratios"):
        return None
    row = conn.execute(
        """
        SELECT *
        FROM financial_ratios
        WHERE UPPER(ticker) = ? AND ratio_name = ?
        ORDER BY period_end DESC, as_of DESC, processed_at DESC, ratio_id DESC
        LIMIT 1
        """,
        (normalize_ticker(ticker), metric),
    ).fetchone()
    return dict(row) if row else None


def watchlist_items(conn: sqlite3.Connection, watchlist_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT *
        FROM watchlist_items
        WHERE watchlist_id = ?
        ORDER BY added_at, ticker
        """,
        (watchlist_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def build_item_payload(conn: sqlite3.Connection, item: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ticker = item["ticker"]
    company = latest_company(conn, ticker)
    quote = latest_quote(conn, ticker)
    ratios = latest_ratios(conn, ticker)
    lineage_rows = [row for row in (company, quote) if row]
    payload = {
        "ticker": ticker,
        "notes": item.get("notes"),
        "added_at": item.get("added_at"),
        "company": public_record(company) if company else None,
        "quote": public_record(quote) if quote else None,
        "ratios": ratios,
    }
    return payload, lineage_rows


def get_watchlist(conn: sqlite3.Connection, watchlist_id: int, owner_id: str | None) -> dict[str, Any]:
    watchlist = watchlist_row(conn, watchlist_id, clean_owner_id(owner_id))
    items = watchlist_items(conn, watchlist_id)
    payload_items = []
    lineage_rows = []
    for item in items:
        payload, rows = build_item_payload(conn, item)
        payload_items.append(payload)
        lineage_rows.extend(rows)
    return {
        "data": {
            "watchlist": watchlist,
            "items": payload_items,
        },
        "metadata": {
            **metadata_from_rows(conn, lineage_rows),
            "item_count": len(payload_items),
        },
    }


def add_watchlist_item(
    conn: sqlite3.Connection,
    watchlist_id: int,
    owner_id: str | None,
    ticker: str,
    notes: str | None,
) -> dict[str, Any]:
    owner = clean_owner_id(owner_id)
    watchlist_row(conn, watchlist_id, owner)
    symbol = normalize_ticker(ticker)
    now = now_utc()
    conn.execute(
        """
        INSERT INTO watchlist_items (watchlist_id, ticker, notes, added_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(watchlist_id, ticker) DO UPDATE SET notes = excluded.notes
        """,
        (watchlist_id, symbol, notes, now),
    )
    conn.execute("UPDATE watchlists SET updated_at = ? WHERE watchlist_id = ?", (now, watchlist_id))
    conn.commit()
    return get_watchlist(conn, watchlist_id, owner_id)


def remove_watchlist_item(conn: sqlite3.Connection, watchlist_id: int, owner_id: str | None, ticker: str) -> dict[str, Any]:
    watchlist_row(conn, watchlist_id, clean_owner_id(owner_id))
    symbol = normalize_ticker(ticker)
    conn.execute(
        "DELETE FROM watchlist_items WHERE watchlist_id = ? AND ticker = ?",
        (watchlist_id, symbol),
    )
    conn.execute("UPDATE watchlists SET updated_at = ? WHERE watchlist_id = ?", (now_utc(), watchlist_id))
    conn.commit()
    return get_watchlist(conn, watchlist_id, owner_id)


def add_alert_rule(
    conn: sqlite3.Connection,
    watchlist_id: int,
    owner_id: str | None,
    ticker: str | None,
    metric: str,
    operator: str,
    threshold: float,
    enabled: bool = True,
    cooldown_minutes: int | None = 60,
) -> dict[str, Any]:
    owner = clean_owner_id(owner_id)
    watchlist_row(conn, watchlist_id, owner)
    if operator not in VALID_OPERATORS:
        raise ValueError(f"Unsupported operator: {operator}")
    cooldown = clean_cooldown_minutes(cooldown_minutes)
    now = now_utc()
    cursor = conn.execute(
        """
        INSERT INTO watchlist_alert_rules (
            watchlist_id, ticker, metric, operator, threshold, cooldown_minutes, enabled, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            watchlist_id,
            normalize_ticker(ticker) if ticker else None,
            metric.strip(),
            operator,
            threshold,
            cooldown,
            1 if enabled else 0,
            now,
            now,
        ),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM watchlist_alert_rules WHERE rule_id = ?",
        (cursor.lastrowid,),
    ).fetchone()
    return dict(row)


def alert_rule_row(conn: sqlite3.Connection, watchlist_id: int, owner_id: str | None, rule_id: int) -> dict[str, Any]:
    owner = clean_owner_id(owner_id)
    watchlist_row(conn, watchlist_id, owner)
    row = conn.execute(
        """
        SELECT *
        FROM watchlist_alert_rules
        WHERE watchlist_id = ? AND rule_id = ?
        """,
        (watchlist_id, rule_id),
    ).fetchone()
    if not row:
        raise MarketRecordNotFound(f"No alert rule found for id {rule_id}")
    return dict(row)


def list_alert_rules(
    conn: sqlite3.Connection,
    watchlist_id: int,
    owner_id: str | None,
    include_disabled: bool = False,
    include_review_counts: bool = False,
    needs_attention: bool = False,
) -> dict[str, Any]:
    owner = clean_owner_id(owner_id)
    include_counts = include_review_counts or needs_attention
    watchlist_row(conn, watchlist_id, owner)
    query = """
        SELECT *
        FROM watchlist_alert_rules
        WHERE watchlist_id = ?
    """
    params: list[Any] = [watchlist_id]
    if not include_disabled:
        query += " AND enabled = 1"
    query += " ORDER BY rule_id"
    rows = conn.execute(query, params).fetchall()
    data = [dict(row) for row in rows]
    if include_counts and data:
        require_alert_event_tables(conn)
        rule_ids = [int(row["rule_id"]) for row in data]
        placeholders = ", ".join("?" for _ in rule_ids)
        summary_rows = conn.execute(
            f"""
            SELECT
                ae.rule_id,
                COALESCE(aer.status, 'open') AS review_status,
                COUNT(*) AS event_count,
                MAX(ae.triggered_at) AS latest_triggered_at
            FROM alert_events ae
            LEFT JOIN alert_event_reviews aer ON aer.event_id = ae.event_id
            WHERE ae.watchlist_id = ? AND ae.rule_id IN ({placeholders})
            GROUP BY ae.rule_id, COALESCE(aer.status, 'open')
            """,
            [watchlist_id, *rule_ids],
        ).fetchall()
        summaries: dict[int, dict[str, Any]] = {}
        for rule_id in rule_ids:
            summaries[rule_id] = {
                "total_events": 0,
                "counts": {"open": 0, "reviewed": 0, "dismissed": 0},
                "latest_triggered_at": None,
                "by_status": [
                    {"status": "open", "event_count": 0, "latest_triggered_at": None},
                    {"status": "reviewed", "event_count": 0, "latest_triggered_at": None},
                    {"status": "dismissed", "event_count": 0, "latest_triggered_at": None},
                ],
            }
        for row in summary_rows:
            rule_id = int(row["rule_id"])
            status = row["review_status"] or "open"
            event_count = int(row["event_count"])
            latest_triggered_at = row["latest_triggered_at"]
            summary = summaries[rule_id]
            summary["counts"][status] = event_count
            summary["total_events"] += event_count
            for item in summary["by_status"]:
                if item["status"] == status:
                    item["event_count"] = event_count
                    item["latest_triggered_at"] = latest_triggered_at
                    break
        for summary in summaries.values():
            summary["latest_triggered_at"] = max(
                (item["latest_triggered_at"] for item in summary["by_status"] if item["latest_triggered_at"]),
                default=None,
            )
        for row in data:
            row["review_summary"] = summaries[int(row["rule_id"])]
    if needs_attention:
        data = [
            row
            for row in data
            if row.get("review_summary", {}).get("counts", {}).get("open", 0) > 0
        ]
    return {
        "data": data,
        "metadata": {
            "watchlist_id": watchlist_id,
            "owner_id": owner,
            "result_count": len(data),
            "include_disabled": include_disabled,
            "include_review_counts": include_counts,
            "needs_attention": needs_attention,
        },
    }


def update_alert_rule(
    conn: sqlite3.Connection,
    watchlist_id: int,
    owner_id: str | None,
    rule_id: int,
    ticker: str | None = None,
    metric: str | None = None,
    operator: str | None = None,
    threshold: float | None = None,
    cooldown_minutes: int | None = None,
    enabled: bool | None = None,
) -> dict[str, Any]:
    current = alert_rule_row(conn, watchlist_id, owner_id, rule_id)
    next_operator = operator if operator is not None else current["operator"]
    if next_operator not in VALID_OPERATORS:
        raise ValueError(f"Unsupported operator: {next_operator}")
    next_ticker = normalize_ticker(ticker) if ticker is not None else current["ticker"]
    next_metric = metric.strip() if metric is not None else current["metric"]
    next_threshold = threshold if threshold is not None else current["threshold"]
    next_cooldown = (
        clean_cooldown_minutes(cooldown_minutes)
        if cooldown_minutes is not None
        else int(current["cooldown_minutes"])
    )
    next_enabled = int(current["enabled"] if enabled is None else enabled)
    conn.execute(
        """
        UPDATE watchlist_alert_rules
        SET ticker = ?,
            metric = ?,
            operator = ?,
            threshold = ?,
            cooldown_minutes = ?,
            enabled = ?,
            updated_at = ?
        WHERE watchlist_id = ? AND rule_id = ?
        """,
        (
            next_ticker,
            next_metric,
            next_operator,
            next_threshold,
            next_cooldown,
            next_enabled,
            now_utc(),
            watchlist_id,
            rule_id,
        ),
    )
    conn.commit()
    return alert_rule_row(conn, watchlist_id, owner_id, rule_id)


def disable_alert_rule(conn: sqlite3.Connection, watchlist_id: int, owner_id: str | None, rule_id: int) -> dict[str, Any]:
    return update_alert_rule(conn, watchlist_id, owner_id, rule_id, enabled=False)


def metric_value(conn: sqlite3.Connection, ticker: str, metric: str) -> float | None:
    if metric == "price":
        quote = latest_quote(conn, ticker)
        return quote.get("price") if quote else None
    if metric == "market_cap":
        company = latest_company(conn, ticker)
        return company.get("market_cap") if company else None
    ratios = latest_ratios(conn, ticker)
    return ratios.get(metric)


def metric_snapshot(
    conn: sqlite3.Connection,
    ticker: str,
    metric: str,
    warn_days: int,
    stale_days: int,
) -> dict[str, Any]:
    source_table = "financial_ratios"
    value_key = "ratio_value"
    if metric == "price":
        row = latest_quote(conn, ticker)
        source_table = "quote_snapshots"
        value_key = "price"
    elif metric == "market_cap":
        row = latest_company(conn, ticker)
        source_table = "companies"
        value_key = "market_cap"
    else:
        row = latest_ratio(conn, ticker, metric)

    value = row.get(value_key) if row else None
    age = freshness_age_days(row.get("as_of") if row else None)
    freshness_status = status_for_age(age, warn_days, stale_days)
    available = value is not None
    quality_status = row.get("quality_status") if row else None
    data_rights_status = row.get("data_rights_status") if row else None
    quality_key = str(quality_status or "unknown").strip().lower()
    rights_key = str(data_rights_status or "unknown").strip().lower()
    stale = available and freshness_status == "stale"
    warning = available and freshness_status == "warning"
    quality_blocked = available and quality_key in BLOCKING_QUALITY_STATUSES
    rights_blocked = available and rights_key in BLOCKING_DATA_RIGHTS_STATUSES
    if not available:
        data_status = "missing_metric"
    elif quality_blocked:
        data_status = "quality_failed"
    elif rights_blocked:
        data_status = "data_rights_blocked"
    elif stale:
        data_status = "stale_metric"
    elif warning:
        data_status = "warning_metric"
    else:
        data_status = "available"
    blocked_statuses = {"missing_metric", "stale_metric", "quality_failed", "data_rights_blocked"}
    return {
        "value": value,
        "available": available,
        "evaluatable": available and not stale and not quality_blocked and not rights_blocked,
        "data_status": data_status,
        "skip_reason": data_status if data_status in blocked_statuses else None,
        "freshness_status": freshness_status,
        "metric_age_days": age,
        "metric_as_of": row.get("as_of") if row else None,
        "metric_processed_at": row.get("processed_at") if row else None,
        "quality_status": quality_status,
        "data_rights_status": data_rights_status,
        "source_table": source_table,
    }


def compare(value: float | None, operator: str, threshold: float) -> bool:
    if value is None:
        return False
    if operator == "lt":
        return value < threshold
    if operator == "lte":
        return value <= threshold
    if operator == "gt":
        return value > threshold
    if operator == "gte":
        return value >= threshold
    return False


def trigger_dedupe_key(watchlist_id: int, rule_id: int, ticker: str, triggered_at: str) -> str:
    return f"watchlist:{watchlist_id}:rule:{rule_id}:ticker:{ticker}:minute:{triggered_at[:16]}"


def alert_in_cooldown(
    conn: sqlite3.Connection,
    watchlist_id: int,
    rule_id: int,
    ticker: str,
    cooldown_minutes: int,
    triggered_at: str,
) -> bool:
    if cooldown_minutes <= 0:
        return False
    cutoff = (
        datetime.fromisoformat(triggered_at.replace("Z", "+00:00")) - timedelta(minutes=cooldown_minutes)
    ).isoformat().replace("+00:00", "Z")
    row = conn.execute(
        """
        SELECT 1
        FROM alert_events
        WHERE watchlist_id = ?
          AND rule_id = ?
          AND ticker = ?
          AND triggered_at >= ?
        LIMIT 1
        """,
        (watchlist_id, rule_id, ticker, cutoff),
    ).fetchone()
    return row is not None


def record_alert_events(
    conn: sqlite3.Connection,
    watchlist_id: int,
    owner_id: str,
    evaluations: list[dict[str, Any]],
) -> dict[str, int]:
    require_alert_event_tables(conn)
    owner = clean_owner_id(owner_id)
    event_type = "watchlist.alert_triggered"
    subscriptions = active_webhook_subscriptions(conn, owner, event_type)
    destinations = subscriptions or [{"subscription_id": None, "endpoint_url": None}]
    created_events = 0
    created_outbox = 0
    suppressed_events = 0
    triggered_at = now_utc()
    for evaluation in evaluations:
        if not evaluation["triggered"]:
            continue
        cooldown_minutes = clean_cooldown_minutes(evaluation.get("cooldown_minutes"))
        if alert_in_cooldown(
            conn,
            watchlist_id,
            evaluation["rule_id"],
            evaluation["ticker"],
            cooldown_minutes,
            triggered_at,
        ):
            suppressed_events += 1
            continue
        payload = {
            "watchlist_id": watchlist_id,
            "owner_id": owner,
            "rule_id": evaluation["rule_id"],
            "ticker": evaluation["ticker"],
            "metric": evaluation["metric"],
            "operator": evaluation["operator"],
            "threshold": evaluation["threshold"],
            "cooldown_minutes": cooldown_minutes,
            "value": evaluation["value"],
            "triggered_at": triggered_at,
        }
        dedupe_key = trigger_dedupe_key(watchlist_id, evaluation["rule_id"], evaluation["ticker"], triggered_at)
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO alert_events (
                watchlist_id, rule_id, ticker, metric, operator, threshold, value,
                triggered_at, dedupe_key, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                watchlist_id,
                evaluation["rule_id"],
                evaluation["ticker"],
                evaluation["metric"],
                evaluation["operator"],
                evaluation["threshold"],
                evaluation["value"],
                triggered_at,
                dedupe_key,
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
            ),
        )
        if cursor.rowcount:
            created_events += 1
            for destination in destinations:
                conn.execute(
                    """
                    INSERT INTO webhook_outbox (
                        owner_id, subscription_id, destination_url, event_type,
                        aggregate_type, aggregate_id, payload_json, status, created_at, next_attempt_at
                    )
                    VALUES (?, ?, ?, ?, 'watchlist', ?, ?, 'pending', ?, ?)
                    """,
                    (
                        owner,
                        destination.get("subscription_id"),
                        destination.get("endpoint_url"),
                        event_type,
                        watchlist_id,
                        json.dumps(payload, sort_keys=True, separators=(",", ":")),
                        triggered_at,
                        triggered_at,
                    ),
                )
                created_outbox += 1
    return {"created_events": created_events, "created_outbox": created_outbox, "suppressed_events": suppressed_events}


def serialize_alert_event_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["payload"] = json.loads(item.pop("payload_json"))
    item["review"] = {
        "status": item.pop("review_status", None) or "open",
        "reviewed_by": item.pop("reviewed_by", None),
        "reviewed_at": item.pop("reviewed_at", None),
        "notes": item.pop("review_notes", None),
    }
    return item


def alert_event_review_summary(
    conn: sqlite3.Connection,
    watchlist_id: int,
    owner_id: str | None,
) -> dict[str, Any]:
    owner = clean_owner_id(owner_id)
    watchlist_row(conn, watchlist_id, owner)
    require_alert_event_tables(conn)
    rows = conn.execute(
        """
        SELECT
            COALESCE(aer.status, 'open') AS review_status,
            COUNT(*) AS event_count,
            MAX(ae.triggered_at) AS latest_triggered_at
        FROM alert_events ae
        LEFT JOIN alert_event_reviews aer ON aer.event_id = ae.event_id
        WHERE ae.watchlist_id = ?
        GROUP BY COALESCE(aer.status, 'open')
        """,
        (watchlist_id,),
    ).fetchall()
    counts = {"open": 0, "reviewed": 0, "dismissed": 0}
    latest_by_status = {"open": None, "reviewed": None, "dismissed": None}
    for row in rows:
        status = row["review_status"] or "open"
        counts[status] = int(row["event_count"])
        latest_by_status[status] = row["latest_triggered_at"]
    latest_triggered_at = max((value for value in latest_by_status.values() if value), default=None)
    by_status = [
        {
            "status": status,
            "event_count": counts[status],
            "latest_triggered_at": latest_by_status[status],
        }
        for status in ("open", "reviewed", "dismissed")
    ]
    return {
        "data": {
            "total_events": sum(counts.values()),
            "counts": counts,
            "latest_triggered_at": latest_triggered_at,
            "by_status": by_status,
        },
        "metadata": {"watchlist_id": watchlist_id, "owner_id": owner},
    }


def alert_rule_event_review_summary(
    conn: sqlite3.Connection,
    watchlist_id: int,
    owner_id: str | None,
    rule_id: int,
) -> dict[str, Any]:
    owner = clean_owner_id(owner_id)
    alert_rule_row(conn, watchlist_id, owner, rule_id)
    require_alert_event_tables(conn)
    rows = conn.execute(
        """
        SELECT
            COALESCE(aer.status, 'open') AS review_status,
            COUNT(*) AS event_count,
            MAX(ae.triggered_at) AS latest_triggered_at
        FROM alert_events ae
        LEFT JOIN alert_event_reviews aer ON aer.event_id = ae.event_id
        WHERE ae.watchlist_id = ? AND ae.rule_id = ?
        GROUP BY COALESCE(aer.status, 'open')
        """,
        (watchlist_id, rule_id),
    ).fetchall()
    counts = {"open": 0, "reviewed": 0, "dismissed": 0}
    latest_by_status = {"open": None, "reviewed": None, "dismissed": None}
    for row in rows:
        status = row["review_status"] or "open"
        counts[status] = int(row["event_count"])
        latest_by_status[status] = row["latest_triggered_at"]
    latest_triggered_at = max((value for value in latest_by_status.values() if value), default=None)
    by_status = [
        {
            "status": status,
            "event_count": counts[status],
            "latest_triggered_at": latest_by_status[status],
        }
        for status in ("open", "reviewed", "dismissed")
    ]
    return {
        "data": {
            "total_events": sum(counts.values()),
            "counts": counts,
            "latest_triggered_at": latest_triggered_at,
            "by_status": by_status,
        },
        "metadata": {"watchlist_id": watchlist_id, "rule_id": rule_id, "owner_id": owner},
    }


def alert_event_history(
    conn: sqlite3.Connection,
    watchlist_id: int,
    owner_id: str | None,
    limit: int = 50,
    review_status: str | None = None,
) -> dict[str, Any]:
    owner = clean_owner_id(owner_id)
    filtered_status = clean_event_review_status(review_status)
    watchlist_row(conn, watchlist_id, owner)
    require_alert_event_tables(conn)
    conditions = ["ae.watchlist_id = ?"]
    params: list[Any] = [watchlist_id]
    if filtered_status is not None:
        conditions.append("COALESCE(aer.status, 'open') = ?")
        params.append(filtered_status)
    params.append(limit)
    where_clause = " AND ".join(conditions)
    rows = conn.execute(
        f"""
        SELECT
            ae.*,
            aer.status AS review_status,
            aer.reviewed_by,
            aer.reviewed_at,
            aer.notes AS review_notes
        FROM alert_events ae
        LEFT JOIN alert_event_reviews aer ON aer.event_id = ae.event_id
        WHERE {where_clause}
        ORDER BY ae.triggered_at DESC, ae.event_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    data = [serialize_alert_event_row(row) for row in rows]
    return {
        "data": data,
        "metadata": {
            "watchlist_id": watchlist_id,
            "owner_id": owner,
            "review_status": filtered_status,
            "result_count": len(data),
        },
    }


def alert_rule_event_history(
    conn: sqlite3.Connection,
    watchlist_id: int,
    owner_id: str | None,
    rule_id: int,
    limit: int = 50,
    review_status: str | None = None,
) -> dict[str, Any]:
    owner = clean_owner_id(owner_id)
    filtered_status = clean_event_review_status(review_status)
    alert_rule_row(conn, watchlist_id, owner, rule_id)
    require_alert_event_tables(conn)
    conditions = ["ae.watchlist_id = ?", "ae.rule_id = ?"]
    params: list[Any] = [watchlist_id, rule_id]
    if filtered_status is not None:
        conditions.append("COALESCE(aer.status, 'open') = ?")
        params.append(filtered_status)
    params.append(limit)
    where_clause = " AND ".join(conditions)
    rows = conn.execute(
        f"""
        SELECT
            ae.*,
            aer.status AS review_status,
            aer.reviewed_by,
            aer.reviewed_at,
            aer.notes AS review_notes
        FROM alert_events ae
        LEFT JOIN alert_event_reviews aer ON aer.event_id = ae.event_id
        WHERE {where_clause}
        ORDER BY ae.triggered_at DESC, ae.event_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    data = [serialize_alert_event_row(row) for row in rows]
    return {
        "data": data,
        "metadata": {
            "watchlist_id": watchlist_id,
            "rule_id": rule_id,
            "owner_id": owner,
            "review_status": filtered_status,
            "result_count": len(data),
        },
    }


def alert_event_row(
    conn: sqlite3.Connection,
    watchlist_id: int,
    owner_id: str | None,
    event_id: int,
) -> dict[str, Any]:
    owner = clean_owner_id(owner_id)
    watchlist_row(conn, watchlist_id, owner)
    require_alert_event_tables(conn)
    row = conn.execute(
        """
        SELECT
            ae.*,
            aer.status AS review_status,
            aer.reviewed_by,
            aer.reviewed_at,
            aer.notes AS review_notes
        FROM alert_events ae
        LEFT JOIN alert_event_reviews aer ON aer.event_id = ae.event_id
        WHERE ae.watchlist_id = ? AND ae.event_id = ?
        """,
        (watchlist_id, event_id),
    ).fetchone()
    if not row:
        raise MarketRecordNotFound(f"No alert event found for id {event_id}")
    return serialize_alert_event_row(row)


def record_alert_event_review_audits(
    conn: sqlite3.Connection,
    watchlist_id: int,
    owner_id: str,
    event_ids: list[int],
    status: str,
    reviewed_by: str | None,
    reviewed_at: str,
    notes: str | None,
    source: str,
) -> None:
    conn.executemany(
        """
        INSERT INTO alert_event_review_audits (
            event_id, watchlist_id, owner_id, status, reviewed_by, reviewed_at, notes, source, batch_size
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                event_id,
                watchlist_id,
                owner_id,
                status,
                reviewed_by,
                reviewed_at,
                notes,
                source,
                len(event_ids),
            )
            for event_id in event_ids
        ],
    )


def alert_event_review_audit_history(
    conn: sqlite3.Connection,
    watchlist_id: int,
    owner_id: str | None,
    event_id: int,
    limit: int = 50,
) -> dict[str, Any]:
    owner = clean_owner_id(owner_id)
    alert_event_row(conn, watchlist_id, owner, event_id)
    rows = conn.execute(
        """
        SELECT *
        FROM alert_event_review_audits
        WHERE watchlist_id = ? AND event_id = ? AND owner_id = ?
        ORDER BY reviewed_at DESC, audit_id DESC
        LIMIT ?
        """,
        (watchlist_id, event_id, owner, limit),
    ).fetchall()
    data = [dict(row) for row in rows]
    return {
        "data": data,
        "metadata": {
            "watchlist_id": watchlist_id,
            "event_id": event_id,
            "owner_id": owner,
            "result_count": len(data),
        },
    }


def alert_review_audit_history(
    conn: sqlite3.Connection,
    watchlist_id: int,
    owner_id: str | None,
    event_id: int | None = None,
    rule_id: int | None = None,
    status: str | None = None,
    source: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    owner = clean_owner_id(owner_id)
    filtered_status = clean_event_review_status(status)
    watchlist_row(conn, watchlist_id, owner)
    require_alert_event_tables(conn)
    if rule_id is not None:
        alert_rule_row(conn, watchlist_id, owner, rule_id)
    if event_id is not None:
        alert_event_row(conn, watchlist_id, owner, event_id)
    clean_source = source.strip().lower() if source else None
    if clean_source and clean_source not in {"single", "bulk"}:
        raise ValueError("Alert review audit source must be 'single' or 'bulk'")

    conditions = ["ara.watchlist_id = ?", "ara.owner_id = ?"]
    params: list[Any] = [watchlist_id, owner]
    if event_id is not None:
        conditions.append("ara.event_id = ?")
        params.append(event_id)
    if rule_id is not None:
        conditions.append("ae.rule_id = ?")
        params.append(rule_id)
    if filtered_status is not None:
        conditions.append("ara.status = ?")
        params.append(filtered_status)
    if clean_source:
        conditions.append("ara.source = ?")
        params.append(clean_source)
    params.append(limit)
    where_clause = " AND ".join(conditions)
    rows = conn.execute(
        f"""
        SELECT
            ara.*,
            ae.rule_id,
            ae.ticker,
            ae.metric,
            ae.triggered_at
        FROM alert_event_review_audits ara
        JOIN alert_events ae ON ae.event_id = ara.event_id
        WHERE {where_clause}
        ORDER BY ara.reviewed_at DESC, ara.audit_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    data = [dict(row) for row in rows]
    return {
        "data": data,
        "metadata": {
            "watchlist_id": watchlist_id,
            "owner_id": owner,
            "event_id": event_id,
            "rule_id": rule_id,
            "status": filtered_status,
            "source": clean_source,
            "result_count": len(data),
        },
    }


def update_alert_event_review(
    conn: sqlite3.Connection,
    watchlist_id: int,
    owner_id: str | None,
    event_id: int,
    status: str,
    reviewed_by: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    owner = clean_owner_id(owner_id)
    next_status = status.strip().lower()
    if next_status not in VALID_EVENT_REVIEW_STATUSES:
        supported = ", ".join(sorted(VALID_EVENT_REVIEW_STATUSES))
        raise ValueError(f"Unsupported alert event review status '{status}'. Expected one of: {supported}")
    alert_event_row(conn, watchlist_id, owner, event_id)
    reviewed_at = now_utc()
    reviewer = reviewed_by.strip() if reviewed_by else None
    review_notes = notes.strip() if notes else None
    conn.execute(
        """
        INSERT INTO alert_event_reviews (event_id, owner_id, status, reviewed_by, reviewed_at, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_id) DO UPDATE SET
            owner_id = excluded.owner_id,
            status = excluded.status,
            reviewed_by = excluded.reviewed_by,
            reviewed_at = excluded.reviewed_at,
            notes = excluded.notes
        """,
        (
            event_id,
            owner,
            next_status,
            reviewer,
            reviewed_at,
            review_notes,
        ),
    )
    record_alert_event_review_audits(
        conn,
        watchlist_id,
        owner,
        [event_id],
        next_status,
        reviewer,
        reviewed_at,
        review_notes,
        "single",
    )
    conn.commit()
    return {"data": alert_event_row(conn, watchlist_id, owner, event_id)}


def update_alert_event_reviews(
    conn: sqlite3.Connection,
    watchlist_id: int,
    owner_id: str | None,
    event_ids: list[int] | None,
    status: str,
    reviewed_by: str | None = None,
    notes: str | None = None,
    current_status: str | None = None,
    rule_id: int | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    owner = clean_owner_id(owner_id)
    next_status = clean_event_review_status(status)
    if next_status is None:
        raise ValueError("Alert event review status is required")
    filtered_status = clean_event_review_status(current_status)
    if event_ids and (filtered_status is not None or rule_id is not None):
        raise ValueError("Use either event_ids or filter fields, not both")
    if not event_ids and filtered_status is None and rule_id is None:
        raise ValueError("Provide event_ids or a current_status/rule_id filter")
    watchlist_row(conn, watchlist_id, owner)
    require_alert_event_tables(conn)
    if rule_id is not None:
        alert_rule_row(conn, watchlist_id, owner, rule_id)

    if event_ids:
        unique_event_ids = sorted({int(event_id) for event_id in event_ids})
    else:
        capped_limit = max(1, min(int(limit), 500))
        conditions = ["ae.watchlist_id = ?"]
        params: list[Any] = [watchlist_id]
        if rule_id is not None:
            conditions.append("ae.rule_id = ?")
            params.append(rule_id)
        if filtered_status is not None:
            conditions.append("COALESCE(aer.status, 'open') = ?")
            params.append(filtered_status)
        params.append(capped_limit)
        where_clause = " AND ".join(conditions)
        rows = conn.execute(
            f"""
            SELECT ae.event_id
            FROM alert_events ae
            LEFT JOIN alert_event_reviews aer ON aer.event_id = ae.event_id
            WHERE {where_clause}
            ORDER BY ae.triggered_at DESC, ae.event_id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        unique_event_ids = [int(row["event_id"]) for row in rows]
    if not unique_event_ids:
        raise ValueError("At least one alert event id is required")

    placeholders = ", ".join("?" for _ in unique_event_ids)
    rows = conn.execute(
        f"""
        SELECT event_id
        FROM alert_events
        WHERE watchlist_id = ? AND event_id IN ({placeholders})
        """,
        [watchlist_id, *unique_event_ids],
    ).fetchall()
    found_event_ids = {int(row["event_id"]) for row in rows}
    missing_event_ids = [event_id for event_id in unique_event_ids if event_id not in found_event_ids]
    if missing_event_ids:
        raise MarketRecordNotFound(f"No alert event found for ids: {missing_event_ids}")

    reviewed_at = now_utc()
    reviewer = reviewed_by.strip() if reviewed_by else None
    review_notes = notes.strip() if notes else None
    review_rows = [
        (
            event_id,
            owner,
            next_status,
            reviewer,
            reviewed_at,
            review_notes,
        )
        for event_id in unique_event_ids
    ]
    conn.executemany(
        """
        INSERT INTO alert_event_reviews (event_id, owner_id, status, reviewed_by, reviewed_at, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_id) DO UPDATE SET
            owner_id = excluded.owner_id,
            status = excluded.status,
            reviewed_by = excluded.reviewed_by,
            reviewed_at = excluded.reviewed_at,
            notes = excluded.notes
        """,
        review_rows,
    )
    record_alert_event_review_audits(
        conn,
        watchlist_id,
        owner,
        unique_event_ids,
        next_status,
        reviewer,
        reviewed_at,
        review_notes,
        "bulk",
    )
    conn.commit()
    data = [alert_event_row(conn, watchlist_id, owner, event_id) for event_id in unique_event_ids]
    return {
        "data": data,
        "metadata": {
            "watchlist_id": watchlist_id,
            "owner_id": owner,
            "status": next_status,
            "requested_count": len(event_ids) if event_ids else len(unique_event_ids),
            "updated_count": len(data),
            "current_status": filtered_status,
            "rule_id": rule_id,
        },
    }


def webhook_outbox(
    conn: sqlite3.Connection,
    status: str = "pending",
    limit: int = 50,
    owner_id: str | None = None,
) -> dict[str, Any]:
    require_alert_event_tables(conn)
    owner = clean_owner_id(owner_id) if owner_id is not None else None
    params: list[Any] = [status]
    owner_filter = ""
    if owner is not None:
        owner_filter = "AND owner_id = ?"
        params.append(owner)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT *
        FROM webhook_outbox
        WHERE status = ?
        {owner_filter}
        ORDER BY created_at, outbox_id
        LIMIT ?
        """,
        params,
    ).fetchall()
    data = []
    for row in rows:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json"))
        data.append(item)
    metadata: dict[str, Any] = {"status": status, "result_count": len(data)}
    if owner is not None:
        metadata["owner_id"] = owner
    return {"data": data, "metadata": metadata}


def webhook_outbox_row(conn: sqlite3.Connection, outbox_id: int, owner_id: str | None = None) -> dict[str, Any]:
    require_alert_event_tables(conn)
    owner = clean_owner_id(owner_id) if owner_id is not None else None
    query = "SELECT * FROM webhook_outbox WHERE outbox_id = ?"
    params: list[Any] = [outbox_id]
    if owner is not None:
        query += " AND owner_id = ?"
        params.append(owner)
    row = conn.execute(query, params).fetchone()
    if not row:
        raise MarketRecordNotFound(f"No webhook outbox row found for id {outbox_id}")
    item = dict(row)
    item["payload"] = json.loads(item.pop("payload_json"))
    return item


def replay_webhook_outbox(
    conn: sqlite3.Connection,
    outbox_id: int,
    owner_id: str | None = None,
    reset_attempts: bool = True,
    reason: str | None = None,
) -> dict[str, Any]:
    row = webhook_outbox_row(conn, outbox_id, owner_id=owner_id)
    if row["status"] == "delivered":
        raise ValueError("Delivered webhook outbox rows cannot be replayed")
    now = now_utc()
    attempts = 0 if reset_attempts else int(row["attempts"])
    replay_reason = (reason or "Manual webhook replay requested").strip()
    conn.execute(
        """
        UPDATE webhook_outbox
        SET status = 'pending',
            attempts = ?,
            next_attempt_at = ?,
            delivered_at = NULL,
            last_error = NULL
        WHERE outbox_id = ?
        """,
        (attempts, now, outbox_id),
    )
    conn.execute(
        """
        INSERT INTO webhook_delivery_attempts (
            outbox_id, owner_id, subscription_id, endpoint_url, event_type,
            attempted_at, duration_ms, delivered, status, http_status, error
        )
        VALUES (?, ?, ?, ?, ?, ?, 0, 0, 'requeued', NULL, ?)
        """,
        (
            outbox_id,
            row["owner_id"],
            row["subscription_id"],
            row["destination_url"],
            row["event_type"],
            now,
            replay_reason[:500],
        ),
    )
    conn.commit()
    updated = webhook_outbox_row(conn, outbox_id, owner_id=owner_id)
    return {"data": updated, "metadata": {"replayed": True, "reset_attempts": reset_attempts}}


def webhook_delivery_attempts(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    outbox_id: int | None = None,
    status: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    require_alert_event_tables(conn)
    owner = clean_owner_id(owner_id) if owner_id is not None else None
    clauses = []
    params: list[Any] = []
    if owner is not None:
        clauses.append("owner_id = ?")
        params.append(owner)
    if outbox_id is not None:
        clauses.append("outbox_id = ?")
        params.append(outbox_id)
    if status is not None:
        clauses.append("status = ?")
        params.append(status.strip())
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT *
        FROM webhook_delivery_attempts
        {where}
        ORDER BY attempted_at DESC, attempt_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    data = [dict(row) for row in rows]
    metadata: dict[str, Any] = {"result_count": len(data)}
    if owner is not None:
        metadata["owner_id"] = owner
    if outbox_id is not None:
        metadata["outbox_id"] = outbox_id
    if status is not None:
        metadata["status"] = status.strip()
    return {"data": data, "metadata": metadata}


def webhook_status(conn: sqlite3.Connection, owner_id: str | None = None) -> dict[str, Any]:
    require_alert_event_tables(conn)
    owner = clean_owner_id(owner_id) if owner_id is not None else None
    now = now_utc()
    owner_clause = "WHERE owner_id = ?" if owner is not None else ""
    owner_params: list[Any] = [owner] if owner is not None else []

    subscription_row = conn.execute(
        f"""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN enabled = 1 THEN 1 ELSE 0 END) AS enabled,
            SUM(CASE WHEN enabled = 0 THEN 1 ELSE 0 END) AS disabled,
            SUM(CASE WHEN signing_secret IS NOT NULL AND signing_secret != '' THEN 1 ELSE 0 END) AS signed
        FROM webhook_subscriptions
        {owner_clause}
        """,
        owner_params,
    ).fetchone()
    subscriptions = {key: int(subscription_row[key] or 0) for key in ("total", "enabled", "disabled", "signed")}

    outbox_rows = conn.execute(
        f"""
        SELECT status, COUNT(*) AS count
        FROM webhook_outbox
        {owner_clause}
        GROUP BY status
        ORDER BY status
        """,
        owner_params,
    ).fetchall()
    status_counts = {row["status"]: int(row["count"]) for row in outbox_rows}

    pending_clause = "WHERE status = 'pending'"
    pending_params: list[Any] = []
    if owner is not None:
        pending_clause += " AND owner_id = ?"
        pending_params.append(owner)
    due_pending = conn.execute(
        f"""
        SELECT COUNT(*) AS count
        FROM webhook_outbox
        {pending_clause}
          AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
        """,
        [*pending_params, now],
    ).fetchone()["count"]
    scheduled_pending = conn.execute(
        f"""
        SELECT COUNT(*) AS count
        FROM webhook_outbox
        {pending_clause}
          AND next_attempt_at > ?
        """,
        [*pending_params, now],
    ).fetchone()["count"]

    attempts_clause = "WHERE owner_id = ?" if owner is not None else ""
    attempts_params: list[Any] = [owner] if owner is not None else []
    last_attempt = conn.execute(
        f"""
        SELECT *
        FROM webhook_delivery_attempts
        {attempts_clause}
        ORDER BY attempted_at DESC, attempt_id DESC
        LIMIT 1
        """,
        attempts_params,
    ).fetchone()

    problem_clause = "WHERE status IN ('failed', 'retryable', 'skipped')"
    problem_params: list[Any] = []
    if owner is not None:
        problem_clause += " AND owner_id = ?"
        problem_params.append(owner)
    problem_rows = conn.execute(
        f"""
        SELECT *
        FROM webhook_delivery_attempts
        {problem_clause}
        ORDER BY attempted_at DESC, attempt_id DESC
        LIMIT 5
        """,
        problem_params,
    ).fetchall()

    metadata: dict[str, Any] = {"checked_at": now}
    if owner is not None:
        metadata["owner_id"] = owner
    return {
        "data": {
            "subscriptions": subscriptions,
            "outbox": {
                "status_counts": status_counts,
                "due_pending_count": int(due_pending or 0),
                "scheduled_pending_count": int(scheduled_pending or 0),
            },
            "deliveries": {
                "last_attempt": dict(last_attempt) if last_attempt else None,
                "recent_problem_attempts": [dict(row) for row in problem_rows],
            },
        },
        "metadata": metadata,
    }


def evaluate_alerts(
    conn: sqlite3.Connection,
    watchlist_id: int,
    owner_id: str | None,
    record_events: bool = False,
    warn_days: int = 2,
    stale_days: int = 5,
) -> dict[str, Any]:
    owner = clean_owner_id(owner_id)
    watchlist_row(conn, watchlist_id, owner)
    tickers = [item["ticker"] for item in watchlist_items(conn, watchlist_id)]
    rules = conn.execute(
        """
        SELECT *
        FROM watchlist_alert_rules
        WHERE watchlist_id = ? AND enabled = 1
        ORDER BY rule_id
        """,
        (watchlist_id,),
    ).fetchall()
    evaluations = []
    for rule_row in rules:
        rule = dict(rule_row)
        applicable = [rule["ticker"]] if rule["ticker"] else tickers
        for ticker in applicable:
            snapshot = metric_snapshot(conn, ticker, rule["metric"], warn_days, stale_days)
            value = snapshot["value"]
            triggered = compare(value, rule["operator"], rule["threshold"]) if snapshot["evaluatable"] else False
            evaluations.append(
                {
                    "rule_id": rule["rule_id"],
                    "ticker": ticker,
                    "metric": rule["metric"],
                    "operator": rule["operator"],
                    "threshold": rule["threshold"],
                    "cooldown_minutes": rule["cooldown_minutes"],
                    "value": value,
                    "available": snapshot["available"],
                    "evaluatable": snapshot["evaluatable"],
                    "data_status": snapshot["data_status"],
                    "skip_reason": snapshot["skip_reason"],
                    "freshness_status": snapshot["freshness_status"],
                    "metric_age_days": snapshot["metric_age_days"],
                    "metric_as_of": snapshot["metric_as_of"],
                    "metric_processed_at": snapshot["metric_processed_at"],
                    "quality_status": snapshot["quality_status"],
                    "data_rights_status": snapshot["data_rights_status"],
                    "source_table": snapshot["source_table"],
                    "triggered": triggered,
                }
            )
    event_result = (
        record_alert_events(conn, watchlist_id, owner, evaluations)
        if record_events
        else {"created_events": 0, "created_outbox": 0, "suppressed_events": 0}
    )
    if record_events:
        conn.commit()
    return {
        "data": evaluations,
        "metadata": {
            "watchlist_id": watchlist_id,
            "rule_count": len(rules),
            "evaluated_count": len(evaluations),
            "evaluatable_metric_count": sum(1 for item in evaluations if item["evaluatable"]),
            "available_metric_count": sum(1 for item in evaluations if item["available"]),
            "missing_metric_count": sum(1 for item in evaluations if not item["available"]),
            "warning_metric_count": sum(1 for item in evaluations if item["data_status"] == "warning_metric"),
            "stale_metric_count": sum(1 for item in evaluations if item["data_status"] == "stale_metric"),
            "quality_blocked_metric_count": sum(1 for item in evaluations if item["data_status"] == "quality_failed"),
            "data_rights_blocked_metric_count": sum(
                1 for item in evaluations if item["data_status"] == "data_rights_blocked"
            ),
            "triggered_count": sum(1 for item in evaluations if item["triggered"]),
            "recorded_event_count": event_result["created_events"],
            "outbox_event_count": event_result["created_outbox"],
            "suppressed_event_count": event_result["suppressed_events"],
            "warn_days": warn_days,
            "stale_days": stale_days,
        },
    }


def alert_evaluation_readiness(
    conn: sqlite3.Connection,
    watchlist_id: int,
    owner_id: str | None,
    include_available: bool = False,
    warn_days: int = 2,
    stale_days: int = 5,
) -> dict[str, Any]:
    owner = clean_owner_id(owner_id)
    evaluated = evaluate_alerts(
        conn,
        watchlist_id,
        owner,
        record_events=False,
        warn_days=warn_days,
        stale_days=stale_days,
    )
    evaluations = evaluated["data"]
    missing = [item for item in evaluations if not item["available"]]
    stale = [item for item in evaluations if item["data_status"] == "stale_metric"]
    quality_blocked = [item for item in evaluations if item["data_status"] == "quality_failed"]
    data_rights_blocked = [item for item in evaluations if item["data_status"] == "data_rights_blocked"]
    blocked = [item for item in evaluations if not item["evaluatable"]]
    available = [item for item in evaluations if item["available"]]
    missing_by_metric: dict[str, int] = {}
    for item in missing:
        metric = item["metric"]
        missing_by_metric[metric] = missing_by_metric.get(metric, 0) + 1
    stale_by_metric: dict[str, int] = {}
    for item in stale:
        metric = item["metric"]
        stale_by_metric[metric] = stale_by_metric.get(metric, 0) + 1

    data: dict[str, Any] = {
        "status": "ready" if not blocked else "needs_data",
        "watchlist_id": watchlist_id,
        "rule_count": evaluated["metadata"]["rule_count"],
        "evaluated_count": evaluated["metadata"]["evaluated_count"],
        "evaluatable_metric_count": evaluated["metadata"]["evaluatable_metric_count"],
        "available_metric_count": len(available),
        "missing_metric_count": len(missing),
        "stale_metric_count": len(stale),
        "quality_blocked_metric_count": len(quality_blocked),
        "data_rights_blocked_metric_count": len(data_rights_blocked),
        "warning_metric_count": evaluated["metadata"]["warning_metric_count"],
        "blocked_count": len(blocked),
        "missing": missing,
        "stale": stale,
        "quality_blocked": quality_blocked,
        "data_rights_blocked": data_rights_blocked,
        "missing_by_metric": [
            {"metric": metric, "count": count}
            for metric, count in sorted(missing_by_metric.items())
        ],
        "stale_by_metric": [
            {"metric": metric, "count": count}
            for metric, count in sorted(stale_by_metric.items())
        ],
    }
    if include_available:
        data["available"] = available
    return {
        "data": data,
        "metadata": {
            "watchlist_id": watchlist_id,
            "owner_id": owner,
            "include_available": include_available,
            "result_count": len(blocked),
            "warn_days": warn_days,
            "stale_days": stale_days,
            "checked_at": now_utc(),
        },
    }


def owner_alert_evaluation_readiness(
    conn: sqlite3.Connection,
    owner_id: str | None,
    include_available: bool = False,
    warn_days: int = 2,
    stale_days: int = 5,
    status: str | None = None,
) -> dict[str, Any]:
    owner = clean_owner_id(owner_id)
    requested_status = status.strip() if status is not None else None
    if requested_status not in {None, "ready", "needs_data"}:
        raise ValueError("Unsupported readiness status filter")

    watchlists = list_watchlists(conn, owner)["data"]
    rows = []
    totals = {
        "rule_count": 0,
        "evaluated_count": 0,
        "evaluatable_metric_count": 0,
        "available_metric_count": 0,
        "missing_metric_count": 0,
        "stale_metric_count": 0,
        "quality_blocked_metric_count": 0,
        "data_rights_blocked_metric_count": 0,
        "warning_metric_count": 0,
        "blocked_count": 0,
    }
    ready_count = 0
    needs_data_count = 0

    for watchlist in watchlists:
        readiness = alert_evaluation_readiness(
            conn,
            watchlist["watchlist_id"],
            owner,
            include_available=include_available,
            warn_days=warn_days,
            stale_days=stale_days,
        )["data"]
        if readiness["status"] == "ready":
            ready_count += 1
        else:
            needs_data_count += 1
        for key in totals:
            totals[key] += int(readiness.get(key) or 0)
        if requested_status is None or readiness["status"] == requested_status:
            rows.append({"watchlist": watchlist, "readiness": readiness})

    return {
        "data": rows,
        "metadata": {
            "owner_id": owner,
            "status": requested_status,
            "include_available": include_available,
            "watchlist_count": len(watchlists),
            "ready_watchlist_count": ready_count,
            "needs_data_watchlist_count": needs_data_count,
            "result_count": len(rows),
            "warn_days": warn_days,
            "stale_days": stale_days,
            "checked_at": now_utc(),
            **totals,
        },
    }


def owner_alert_readiness_actions(
    conn: sqlite3.Connection,
    owner_id: str | None,
    warn_days: int = 2,
    stale_days: int = 5,
    limit: int = 100,
) -> dict[str, Any]:
    owner = clean_owner_id(owner_id)
    if limit < 1:
        raise ValueError("limit must be at least 1")
    readiness = owner_alert_evaluation_readiness(
        conn,
        owner,
        include_available=False,
        warn_days=warn_days,
        stale_days=stale_days,
        status="needs_data",
    )
    action_specs = {
        "data_rights_blocked": {
            "priority": 1,
            "action_type": "review_data_rights",
            "title": "Review data-rights block before enabling alert evaluation",
        },
        "quality_failed": {
            "priority": 2,
            "action_type": "review_quality_failure",
            "title": "Resolve failed metric quality before enabling alert evaluation",
        },
        "stale_metric": {
            "priority": 3,
            "action_type": "refresh_stale_metric",
            "title": "Refresh stale local metric before alert evaluation",
        },
        "missing_metric": {
            "priority": 4,
            "action_type": "ingest_missing_metric",
            "title": "Ingest missing local metric before alert evaluation",
        },
    }
    actions = []
    for row in readiness["data"]:
        watchlist = row["watchlist"]
        blocked = [
            *row["readiness"]["data_rights_blocked"],
            *row["readiness"]["quality_blocked"],
            *row["readiness"]["stale"],
            *row["readiness"]["missing"],
        ]
        for item in blocked:
            spec = action_specs[item["data_status"]]
            action = {
                "action_id": (
                    f"alert-readiness:{watchlist['watchlist_id']}:{item['rule_id']}:"
                    f"{item['ticker']}:{item['metric']}:{item['data_status']}"
                ),
                "priority": spec["priority"],
                "action_type": spec["action_type"],
                "title": spec["title"],
                "reason": item["skip_reason"],
                "owner_id": owner,
                "watchlist_id": watchlist["watchlist_id"],
                "watchlist_name": watchlist["name"],
                "rule_id": item["rule_id"],
                "ticker": item["ticker"],
                "metric": item["metric"],
                "data_status": item["data_status"],
                "freshness_status": item["freshness_status"],
                "metric_as_of": item["metric_as_of"],
                "metric_age_days": item["metric_age_days"],
                "quality_status": item["quality_status"],
                "data_rights_status": item["data_rights_status"],
                "source_table": item["source_table"],
                "readiness_url": f"/watchlists/{watchlist['watchlist_id']}/alerts/readiness",
                "freshness_url": f"/freshness/{item['ticker']}",
            }
            actions.append(action)

    actions.sort(key=lambda item: (item["priority"], item["watchlist_id"], item["rule_id"], item["ticker"], item["metric"]))
    actions = actions[:limit]
    by_type: dict[str, int] = {}
    for action in actions:
        by_type[action["action_type"]] = by_type.get(action["action_type"], 0) + 1
    return {
        "data": actions,
        "metadata": {
            "owner_id": owner,
            "result_count": len(actions),
            "limit": limit,
            "watchlist_count": readiness["metadata"]["watchlist_count"],
            "needs_data_watchlist_count": readiness["metadata"]["needs_data_watchlist_count"],
            "blocked_count": readiness["metadata"]["blocked_count"],
            "by_action_type": [
                {"action_type": action_type, "count": count}
                for action_type, count in sorted(by_type.items())
            ],
            "warn_days": warn_days,
            "stale_days": stale_days,
            "checked_at": now_utc(),
        },
    }


def save_owner_alert_readiness_action_queue(
    conn: sqlite3.Connection,
    owner_id: str | None,
    warn_days: int = 2,
    stale_days: int = 5,
    limit: int = 100,
    title: str | None = None,
    replace_existing: bool = True,
) -> dict[str, Any]:
    from system.ai.action_queue import require_action_queue_tables, save_action_queue

    owner = clean_owner_id(owner_id)
    require_action_queue_tables(conn)
    actions_result = owner_alert_readiness_actions(
        conn,
        owner,
        warn_days=warn_days,
        stale_days=stale_days,
        limit=limit,
    )
    tasks = [_alert_readiness_action_task(action) for action in actions_result["data"]]
    queue = {
        "kind": "alert_readiness_action_queue",
        "owner_id": owner,
        "tasks": tasks,
        "source_followup": {
            "kind": "alert_readiness_actions",
            "owner_id": owner,
            "actions": actions_result["data"],
            "metadata": actions_result["metadata"],
        },
        "queue_markdown": _alert_readiness_queue_markdown(owner, tasks),
    }
    if replace_existing:
        existing = conn.execute(
            """
            SELECT *
            FROM advisor_action_queues
            WHERE owner_id = ? AND focus = 'alert_readiness' AND status IN ('open', 'blocked')
            ORDER BY updated_at DESC, queue_id DESC
            LIMIT 1
            """,
            (owner,),
        ).fetchone()
        if existing:
            closed = not tasks
            saved = _replace_alert_readiness_action_queue(
                conn,
                dict(existing),
                queue,
                title=title or existing["title"],
            )
            return {
                "data": saved,
                "metadata": {
                    "owner_id": owner,
                    "source_action_count": actions_result["metadata"]["result_count"],
                    "saved_queue_id": saved["queue_id"],
                    "task_count": saved["task_count"],
                    "warn_days": warn_days,
                    "stale_days": stale_days,
                    "created": False,
                    "replaced_existing": True,
                    "closed": closed,
                },
            }
    saved = save_action_queue(
        conn,
        queue,
        focus="alert_readiness",
        title=title or f"Alert readiness actions for {owner}",
    )
    return {
        "data": saved,
        "metadata": {
            "owner_id": owner,
            "source_action_count": actions_result["metadata"]["result_count"],
            "saved_queue_id": saved["queue_id"],
            "task_count": saved["task_count"],
            "warn_days": warn_days,
            "stale_days": stale_days,
            "created": True,
            "replaced_existing": False,
            "closed": False,
        },
    }


def _replace_alert_readiness_action_queue(
    conn: sqlite3.Connection,
    existing: dict[str, Any],
    queue: dict[str, Any],
    title: str,
) -> dict[str, Any]:
    from system.ai.action_queue import get_action_queue

    owner = clean_owner_id(existing["owner_id"])
    queue_id = int(existing["queue_id"])
    tasks = queue.get("tasks", [])
    counts = _alert_readiness_task_counts(tasks)
    now = now_utc()
    conn.execute(
        """
        UPDATE advisor_action_queues
        SET title = ?, status = ?, task_count = ?, open_task_count = ?,
            blocked_task_count = ?, completed_task_count = ?, source_followup_json = ?,
            queue_markdown = ?, updated_at = ?
        WHERE queue_id = ? AND owner_id = ?
        """,
        (
            title,
            _alert_readiness_queue_status(counts),
            counts["task_count"],
            counts["open_task_count"],
            counts["blocked_task_count"],
            counts["completed_task_count"],
            json.dumps(queue.get("source_followup", {}), sort_keys=True),
            queue.get("queue_markdown", _alert_readiness_queue_markdown(owner, tasks)),
            now,
            queue_id,
            owner,
        ),
    )
    conn.execute("DELETE FROM advisor_action_queue_tasks WHERE queue_id = ?", (queue_id,))
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
    return get_action_queue(conn, queue_id, owner)


def _alert_readiness_task_counts(tasks: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "task_count": len(tasks),
        "open_task_count": sum(1 for task in tasks if task["status"] == "open"),
        "blocked_task_count": sum(1 for task in tasks if task["status"] == "blocked"),
        "completed_task_count": sum(1 for task in tasks if task["status"] == "completed"),
    }


def _alert_readiness_queue_status(counts: dict[str, int]) -> str:
    if counts["task_count"] == 0:
        return "completed"
    if counts["task_count"] and counts["completed_task_count"] == counts["task_count"]:
        return "completed"
    if counts["open_task_count"] == 0 and counts["blocked_task_count"] > 0:
        return "blocked"
    return "open"


def _alert_readiness_action_task(action: dict[str, Any]) -> dict[str, Any]:
    task_id = (
        action["action_id"]
        .replace(":", "-")
        .replace("_", "-")
        .lower()
    )
    urgency = "high" if action["priority"] <= 2 else "medium"
    criteria_by_type = {
        "review_data_rights": "Data-rights status is approved or the alert rule is intentionally disabled.",
        "review_quality_failure": "The failed metric quality issue is corrected or the alert rule is intentionally disabled.",
        "refresh_stale_metric": "The metric has fresh local data or the alert rule is intentionally disabled.",
        "ingest_missing_metric": "The missing metric is ingested locally or the alert rule is intentionally disabled.",
    }
    return {
        "task_id": task_id,
        "title": f"{action['title']}: {action['ticker']} {action['metric']}",
        "urgency": urgency,
        "status": "open",
        "rationale": (
            f"{action['ticker']} {action['metric']} blocks alert readiness for "
            f"{action['watchlist_name']} because {action['data_status']}."
        ),
        "completion_criteria": criteria_by_type[action["action_type"]],
        "evidence": action,
    }


def _alert_readiness_queue_markdown(owner_id: str, tasks: list[dict[str, Any]]) -> str:
    lines = [f"# Alert Readiness Action Queue: {owner_id}", ""]
    if not tasks:
        lines.append("No alert readiness blockers found.")
    for task in tasks:
        evidence = task["evidence"]
        lines.append(
            f"- [{task['status']}] {task['title']} ({task['urgency']}) "
            f"- watchlist {evidence['watchlist_id']}, rule {evidence['rule_id']}"
        )
    return "\n".join(lines)
