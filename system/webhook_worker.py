from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DEFAULT_DB = Path("./system/market_intel.db")
MAX_ERROR_LENGTH = 500
DEFAULT_RETRY_BASE_SECONDS = 60
DEFAULT_RETRY_MAX_SECONDS = 3600


@dataclass(frozen=True)
class DeliveryResult:
    delivered: bool
    status_code: int | None
    error: str | None = None


def now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def retry_delay_seconds(attempt_number: int, base_seconds: int, max_seconds: int) -> int:
    attempt = max(attempt_number, 1)
    return min(max_seconds, base_seconds * (2 ** (attempt - 1)))


def next_retry_at(attempt_number: int, base_seconds: int, max_seconds: int) -> str:
    delay = retry_delay_seconds(attempt_number, base_seconds, max_seconds)
    return (datetime.now(UTC).replace(microsecond=0) + timedelta(seconds=delay)).isoformat().replace("+00:00", "Z")


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def pending_events(conn: sqlite3.Connection, limit: int) -> list[dict[str, Any]]:
    due_at = now_utc()
    rows = conn.execute(
        """
        SELECT webhook_outbox.*, webhook_subscriptions.signing_secret AS signing_secret
        FROM webhook_outbox
        LEFT JOIN webhook_subscriptions
            ON webhook_outbox.subscription_id = webhook_subscriptions.subscription_id
        WHERE webhook_outbox.status = 'pending'
          AND (webhook_outbox.next_attempt_at IS NULL OR webhook_outbox.next_attempt_at <= ?)
        ORDER BY COALESCE(webhook_outbox.next_attempt_at, webhook_outbox.created_at), webhook_outbox.outbox_id
        LIMIT ?
        """,
        (due_at, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def event_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(row["payload_json"])
    event = {
        "outbox_id": row["outbox_id"],
        "event_type": row["event_type"],
        "aggregate_type": row["aggregate_type"],
        "aggregate_id": row["aggregate_id"],
        "created_at": row["created_at"],
        "payload": payload,
    }
    if row.get("owner_id") is not None:
        event["owner_id"] = row["owner_id"]
    if row.get("subscription_id") is not None:
        event["subscription_id"] = row["subscription_id"]
    return event


def delivery_endpoint(row: dict[str, Any], fallback_endpoint_url: str | None) -> str | None:
    destination_url = (row.get("destination_url") or "").strip()
    fallback = (fallback_endpoint_url or "").strip()
    return destination_url or fallback or None


def signature_headers(signing_secret: str | None, body: bytes, timestamp: int | None = None) -> dict[str, str]:
    secret = (signing_secret or "").strip()
    if not secret:
        return {}
    issued_at = int(time.time()) if timestamp is None else timestamp
    signed_payload = f"{issued_at}.".encode("utf-8") + body
    digest = hmac.new(secret.encode("utf-8"), signed_payload, hashlib.sha256).hexdigest()
    return {"X-Cerebral-Signature": f"t={issued_at},v1={digest}"}


def post_event(row: dict[str, Any], endpoint_url: str, timeout_seconds: float) -> DeliveryResult:
    body = json.dumps(event_payload(row), sort_keys=True, separators=(",", ":")).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "cerebral-insights-webhook-worker/1.0",
        "X-Cerebral-Event-Type": row["event_type"],
        "X-Cerebral-Outbox-Id": str(row["outbox_id"]),
    }
    headers.update(signature_headers(row.get("signing_secret"), body))
    request = Request(
        endpoint_url,
        data=body,
        method="POST",
        headers=headers,
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status_code = response.getcode()
            return DeliveryResult(delivered=200 <= status_code < 300, status_code=status_code)
    except HTTPError as exc:
        return DeliveryResult(delivered=False, status_code=exc.code, error=f"HTTP {exc.code}: {exc.reason}")
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        return DeliveryResult(delivered=False, status_code=None, error=f"URL error: {reason}")
    except OSError as exc:
        return DeliveryResult(delivered=False, status_code=None, error=f"Network error: {exc}")


def mark_delivered(conn: sqlite3.Connection, outbox_id: int) -> None:
    conn.execute(
        """
        UPDATE webhook_outbox
        SET status = 'delivered',
            delivered_at = ?,
            next_attempt_at = NULL,
            attempts = attempts + 1,
            last_error = NULL
        WHERE outbox_id = ?
        """,
        (now_utc(), outbox_id),
    )


def mark_failed(
    conn: sqlite3.Connection,
    outbox_id: int,
    error: str,
    max_attempts: int,
    retry_base_seconds: int,
    retry_max_seconds: int,
) -> str:
    row = conn.execute("SELECT attempts FROM webhook_outbox WHERE outbox_id = ?", (outbox_id,)).fetchone()
    next_attempts = int(row["attempts"]) + 1 if row else 1
    status = "failed" if next_attempts >= max_attempts else "pending"
    next_attempt = None if status == "failed" else next_retry_at(next_attempts, retry_base_seconds, retry_max_seconds)
    conn.execute(
        """
        UPDATE webhook_outbox
        SET status = ?,
            attempts = ?,
            next_attempt_at = ?,
            last_error = ?
        WHERE outbox_id = ?
        """,
        (status, next_attempts, next_attempt, error[:MAX_ERROR_LENGTH], outbox_id),
    )
    return status


def mark_skipped(
    conn: sqlite3.Connection,
    outbox_id: int,
    retry_base_seconds: int,
    retry_max_seconds: int,
) -> None:
    row = conn.execute("SELECT attempts FROM webhook_outbox WHERE outbox_id = ?", (outbox_id,)).fetchone()
    next_attempts = int(row["attempts"]) + 1 if row else 1
    conn.execute(
        """
        UPDATE webhook_outbox
        SET attempts = ?,
            next_attempt_at = ?,
            last_error = 'No webhook destination available'
        WHERE outbox_id = ?
        """,
        (next_attempts, next_retry_at(next_attempts, retry_base_seconds, retry_max_seconds), outbox_id),
    )


def record_delivery_attempt(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    endpoint_url: str | None,
    status: str,
    duration_ms: int,
    result: DeliveryResult | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO webhook_delivery_attempts (
            outbox_id, owner_id, subscription_id, endpoint_url, event_type,
            attempted_at, duration_ms, delivered, status, http_status, error
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["outbox_id"],
            row.get("owner_id") or "default",
            row.get("subscription_id"),
            endpoint_url,
            row["event_type"],
            now_utc(),
            max(duration_ms, 0),
            1 if result and result.delivered else 0,
            status,
            result.status_code if result else None,
            (result.error[:MAX_ERROR_LENGTH] if result and result.error else None),
        ),
    )


def deliver_pending(
    conn: sqlite3.Connection,
    endpoint_url: str | None = None,
    limit: int = 50,
    timeout_seconds: float = 10.0,
    max_attempts: int = 5,
    retry_base_seconds: int = DEFAULT_RETRY_BASE_SECONDS,
    retry_max_seconds: int = DEFAULT_RETRY_MAX_SECONDS,
    dry_run: bool = False,
) -> dict[str, Any]:
    events = pending_events(conn, limit)
    summary: dict[str, Any] = {
        "selected": len(events),
        "delivered": 0,
        "retryable": 0,
        "failed": 0,
        "skipped": 0,
        "dry_run": dry_run,
    }
    for row in events:
        if dry_run:
            continue
        endpoint = delivery_endpoint(row, endpoint_url)
        if not endpoint:
            mark_skipped(conn, int(row["outbox_id"]), retry_base_seconds, retry_max_seconds)
            record_delivery_attempt(conn, row, None, "skipped", 0)
            summary["skipped"] += 1
            continue
        started = time.monotonic()
        result = post_event(row, endpoint, timeout_seconds)
        duration_ms = int((time.monotonic() - started) * 1000)
        if result.delivered:
            mark_delivered(conn, int(row["outbox_id"]))
            record_delivery_attempt(conn, row, endpoint, "delivered", duration_ms, result=result)
            summary["delivered"] += 1
            continue
        error = result.error or f"HTTP {result.status_code}"
        status = mark_failed(
            conn,
            int(row["outbox_id"]),
            error,
            max_attempts=max_attempts,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
        )
        if status == "failed":
            record_delivery_attempt(conn, row, endpoint, "failed", duration_ms, result=result)
            summary["failed"] += 1
        else:
            record_delivery_attempt(conn, row, endpoint, "retryable", duration_ms, result=result)
            summary["retryable"] += 1
    conn.commit()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deliver pending Cerebral Insights webhook outbox events")
    parser.add_argument("--db-path", default=str(DEFAULT_DB), help="Serving API SQLite database")
    parser.add_argument("--endpoint-url", help="Fallback endpoint for outbox rows without a stored destination")
    parser.add_argument("--limit", type=int, default=50, help="Maximum pending events to deliver")
    parser.add_argument("--timeout-seconds", type=float, default=10.0, help="Per-request network timeout")
    parser.add_argument("--max-attempts", type=int, default=5, help="Attempts before marking an event failed")
    parser.add_argument("--retry-base-seconds", type=int, default=DEFAULT_RETRY_BASE_SECONDS, help="Initial retry delay")
    parser.add_argument("--retry-max-seconds", type=int, default=DEFAULT_RETRY_MAX_SECONDS, help="Maximum retry delay")
    parser.add_argument("--dry-run", action="store_true", help="Count pending events without posting")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with connect(Path(args.db_path).resolve()) as conn:
        summary = deliver_pending(
            conn,
            endpoint_url=args.endpoint_url,
            limit=args.limit,
            timeout_seconds=args.timeout_seconds,
            max_attempts=args.max_attempts,
            retry_base_seconds=args.retry_base_seconds,
            retry_max_seconds=args.retry_max_seconds,
            dry_run=args.dry_run,
        )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
