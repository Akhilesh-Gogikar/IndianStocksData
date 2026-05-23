#!/usr/bin/env python3
"""Sync TickerTape stock page data into a local SQLite repository."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import random
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


NEXT_DATA_RE = re.compile(
    r"<script[^>]+id=[\"']__NEXT_DATA__[\"'][^>]*>([\s\S]*?)</script>",
    re.IGNORECASE,
)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def snapshot_date_default() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def load_companies(path: Path, include_all_types: bool) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}, got {type(payload).__name__}")
    companies = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        subdirectory = item.get("subdirectory")
        if not subdirectory:
            continue
        if not include_all_types and item.get("type") != "stocks":
            continue
        companies.append(item)
    return companies


def connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sync_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            snapshot_date TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'tickertape',
            company_list_path TEXT NOT NULL,
            db_path TEXT NOT NULL,
            raw_dir TEXT NOT NULL,
            total_companies INTEGER NOT NULL DEFAULT 0,
            attempted INTEGER NOT NULL DEFAULT 0,
            succeeded INTEGER NOT NULL DEFAULT 0,
            failed INTEGER NOT NULL DEFAULT 0,
            skipped INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'running',
            options_json TEXT NOT NULL,
            error TEXT
        );

        CREATE TABLE IF NOT EXISTS companies (
            subdirectory TEXT PRIMARY KEY,
            name TEXT,
            type TEXT,
            sid TEXT,
            slug_id TEXT,
            isin TEXT,
            slug TEXT,
            tradable INTEGER,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            last_fetched_at TEXT
        );

        CREATE TABLE IF NOT EXISTS latest_stock_data (
            subdirectory TEXT PRIMARY KEY REFERENCES companies(subdirectory),
            name TEXT,
            type TEXT,
            fetched_at TEXT NOT NULL,
            snapshot_date TEXT NOT NULL,
            url TEXT NOT NULL,
            http_status INTEGER,
            final_url TEXT,
            raw_json_path TEXT,
            raw_json_sha256 TEXT,
            page_props_keys_json TEXT,
            security_info_json TEXT,
            security_quote_json TEXT,
            scorecard_json TEXT,
            security_summary_json TEXT,
            labels_json TEXT,
            commentary_json TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS stock_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES sync_runs(id),
            subdirectory TEXT NOT NULL REFERENCES companies(subdirectory),
            snapshot_date TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            url TEXT NOT NULL,
            http_status INTEGER,
            ok INTEGER NOT NULL,
            final_url TEXT,
            html_bytes INTEGER,
            raw_json_path TEXT,
            raw_json_sha256 TEXT,
            page_props_keys_json TEXT,
            error TEXT,
            UNIQUE(subdirectory, snapshot_date)
        );

        CREATE TABLE IF NOT EXISTS financial_sections (
            subdirectory TEXT NOT NULL REFERENCES companies(subdirectory),
            snapshot_date TEXT NOT NULL,
            section_key TEXT NOT NULL,
            section_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (subdirectory, snapshot_date, section_key)
        );

        CREATE TABLE IF NOT EXISTS event_sections (
            subdirectory TEXT NOT NULL REFERENCES companies(subdirectory),
            snapshot_date TEXT NOT NULL,
            section_key TEXT NOT NULL,
            section_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (subdirectory, snapshot_date, section_key)
        );

        CREATE TABLE IF NOT EXISTS sync_results (
            run_id INTEGER NOT NULL REFERENCES sync_runs(id),
            subdirectory TEXT NOT NULL,
            status TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            http_status INTEGER,
            error TEXT,
            raw_json_path TEXT,
            PRIMARY KEY (run_id, subdirectory)
        );

        CREATE INDEX IF NOT EXISTS idx_stock_snapshots_date ON stock_snapshots(snapshot_date);
        CREATE INDEX IF NOT EXISTS idx_stock_snapshots_subdirectory ON stock_snapshots(subdirectory);
        CREATE INDEX IF NOT EXISTS idx_sync_results_status ON sync_results(status);
        """
    )
    conn.commit()


def begin_run(conn: sqlite3.Connection, args: argparse.Namespace, total: int) -> int:
    options = {key: jsonable(value) for key, value in vars(args).items() if key not in {"headers"}}
    cursor = conn.execute(
        """
        INSERT INTO sync_runs (
            started_at, snapshot_date, company_list_path, db_path, raw_dir,
            total_companies, options_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            utc_now(),
            args.snapshot_date,
            str(args.company_list),
            str(args.db),
            str(args.raw_dir),
            total,
            json_text(options),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def finish_run(conn: sqlite3.Connection, run_id: int, status: str, error: Optional[str] = None) -> None:
    counts = conn.execute(
        """
        SELECT
            SUM(CASE WHEN status = 'succeeded' THEN 1 ELSE 0 END) AS succeeded,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
            SUM(CASE WHEN status = 'skipped' THEN 1 ELSE 0 END) AS skipped,
            COUNT(*) AS attempted
        FROM sync_results WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    conn.execute(
        """
        UPDATE sync_runs
        SET finished_at = ?, status = ?, error = ?, attempted = ?, succeeded = ?, failed = ?, skipped = ?
        WHERE id = ?
        """,
        (
            utc_now(),
            status,
            error,
            int(counts["attempted"] or 0),
            int(counts["succeeded"] or 0),
            int(counts["failed"] or 0),
            int(counts["skipped"] or 0),
            run_id,
        ),
    )
    conn.commit()


def already_fetched(conn: sqlite3.Connection, subdirectory: str, snapshot_date: str) -> bool:
    row = conn.execute(
        "SELECT ok FROM stock_snapshots WHERE subdirectory = ? AND snapshot_date = ?",
        (subdirectory, snapshot_date),
    ).fetchone()
    return bool(row and int(row["ok"]) == 1)


def upsert_company(conn: sqlite3.Connection, entry: Dict[str, Any], page_props: Optional[Dict[str, Any]]) -> None:
    now = utc_now()
    security_info = {}
    if page_props:
        security_info = page_props.get("securityInfo") or {}
    conn.execute(
        """
        INSERT INTO companies (
            subdirectory, name, type, sid, slug_id, isin, slug, tradable,
            first_seen_at, last_seen_at, last_fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(subdirectory) DO UPDATE SET
            name = excluded.name,
            type = excluded.type,
            sid = COALESCE(excluded.sid, companies.sid),
            slug_id = COALESCE(excluded.slug_id, companies.slug_id),
            isin = COALESCE(excluded.isin, companies.isin),
            slug = COALESCE(excluded.slug, companies.slug),
            tradable = COALESCE(excluded.tradable, companies.tradable),
            last_seen_at = excluded.last_seen_at,
            last_fetched_at = COALESCE(excluded.last_fetched_at, companies.last_fetched_at)
        """,
        (
            entry.get("subdirectory"),
            entry.get("name"),
            entry.get("type"),
            page_props.get("sid") if page_props else None,
            page_props.get("slugId") if page_props else None,
            security_info.get("isin") if isinstance(security_info, dict) else None,
            security_info.get("slug") if isinstance(security_info, dict) else None,
            int(bool(security_info.get("tradable"))) if isinstance(security_info, dict) and security_info.get("tradable") is not None else None,
            now,
            now,
            now if page_props else None,
        ),
    )


def fetch_html(session: requests.Session, url: str, timeout: int, retries: int) -> Tuple[Optional[requests.Response], Optional[str]]:
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = session.get(url, timeout=timeout, allow_redirects=True)
            return response, None
        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt < retries:
                time.sleep(min(10.0, 1.5 * (attempt + 1)))
    return None, last_error


def extract_page_props(html: str) -> Dict[str, Any]:
    match = NEXT_DATA_RE.search(html)
    if not match:
        raise ValueError("__NEXT_DATA__ script not found")
    data = json.loads(match.group(1))
    page_props = (((data or {}).get("props") or {}).get("pageProps") or {})
    if not isinstance(page_props, dict) or not page_props:
        raise ValueError("pageProps missing or empty")
    return page_props


def write_raw(raw_dir: Path, snapshot_date: str, subdirectory: str, page_props: Dict[str, Any]) -> Tuple[Path, str]:
    target_dir = raw_dir / snapshot_date
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_path = target_dir / f"{subdirectory}.page_props.json.gz"
    encoded = json_text(page_props).encode("utf-8")
    sha256 = hashlib.sha256(encoded).hexdigest()
    with gzip.open(raw_path, "wb") as handle:
        handle.write(encoded)
    return raw_path, sha256


def section_items(page_props: Dict[str, Any], prefixes: Iterable[str]) -> List[Tuple[str, Any]]:
    results = []
    for key, value in page_props.items():
        if any(key.startswith(prefix) for prefix in prefixes):
            results.append((key, value))
    return results


def store_success(
    conn: sqlite3.Connection,
    run_id: int,
    entry: Dict[str, Any],
    snapshot_date: str,
    fetched_at: str,
    url: str,
    response: requests.Response,
    html_bytes: int,
    page_props: Dict[str, Any],
    raw_path: Path,
    raw_sha: str,
) -> None:
    subdirectory = entry["subdirectory"]
    page_keys = list(page_props.keys())
    security_info = page_props.get("securityInfo") or {}
    security_quote = page_props.get("securityQuote") or {}
    scorecard = page_props.get("scorecard") or {}
    security_summary = page_props.get("securitySummary") or {}
    labels = page_props.get("labels") or {}
    commentary = page_props.get("commentary") or {}

    upsert_company(conn, entry, page_props)
    conn.execute(
        """
        INSERT INTO latest_stock_data (
            subdirectory, name, type, fetched_at, snapshot_date, url, http_status,
            final_url, raw_json_path, raw_json_sha256, page_props_keys_json,
            security_info_json, security_quote_json, scorecard_json,
            security_summary_json, labels_json, commentary_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(subdirectory) DO UPDATE SET
            name = excluded.name,
            type = excluded.type,
            fetched_at = excluded.fetched_at,
            snapshot_date = excluded.snapshot_date,
            url = excluded.url,
            http_status = excluded.http_status,
            final_url = excluded.final_url,
            raw_json_path = excluded.raw_json_path,
            raw_json_sha256 = excluded.raw_json_sha256,
            page_props_keys_json = excluded.page_props_keys_json,
            security_info_json = excluded.security_info_json,
            security_quote_json = excluded.security_quote_json,
            scorecard_json = excluded.scorecard_json,
            security_summary_json = excluded.security_summary_json,
            labels_json = excluded.labels_json,
            commentary_json = excluded.commentary_json,
            updated_at = excluded.updated_at
        """,
        (
            subdirectory,
            entry.get("name"),
            entry.get("type"),
            fetched_at,
            snapshot_date,
            url,
            response.status_code,
            response.url,
            str(raw_path),
            raw_sha,
            json_text(page_keys),
            json_text(security_info),
            json_text(security_quote),
            json_text(scorecard),
            json_text(security_summary),
            json_text(labels),
            json_text(commentary),
            utc_now(),
        ),
    )
    conn.execute(
        """
        INSERT INTO stock_snapshots (
            run_id, subdirectory, snapshot_date, fetched_at, url, http_status,
            ok, final_url, html_bytes, raw_json_path, raw_json_sha256,
            page_props_keys_json, error
        ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, NULL)
        ON CONFLICT(subdirectory, snapshot_date) DO UPDATE SET
            run_id = excluded.run_id,
            fetched_at = excluded.fetched_at,
            url = excluded.url,
            http_status = excluded.http_status,
            ok = excluded.ok,
            final_url = excluded.final_url,
            html_bytes = excluded.html_bytes,
            raw_json_path = excluded.raw_json_path,
            raw_json_sha256 = excluded.raw_json_sha256,
            page_props_keys_json = excluded.page_props_keys_json,
            error = NULL
        """,
        (
            run_id,
            subdirectory,
            snapshot_date,
            fetched_at,
            url,
            response.status_code,
            response.url,
            html_bytes,
            str(raw_path),
            raw_sha,
            json_text(page_keys),
        ),
    )
    conn.execute("DELETE FROM financial_sections WHERE subdirectory = ? AND snapshot_date = ?", (subdirectory, snapshot_date))
    conn.execute("DELETE FROM event_sections WHERE subdirectory = ? AND snapshot_date = ?", (subdirectory, snapshot_date))
    for key, value in section_items(page_props, ("income-", "balancesheet-", "cashflow-")):
        conn.execute(
            """
            INSERT OR REPLACE INTO financial_sections
            (subdirectory, snapshot_date, section_key, section_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (subdirectory, snapshot_date, key, json_text(value), utc_now()),
        )
    for key, value in section_items(page_props, ("events-",)):
        conn.execute(
            """
            INSERT OR REPLACE INTO event_sections
            (subdirectory, snapshot_date, section_key, section_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (subdirectory, snapshot_date, key, json_text(value), utc_now()),
        )
    conn.execute(
        """
        INSERT OR REPLACE INTO sync_results
        (run_id, subdirectory, status, fetched_at, http_status, error, raw_json_path)
        VALUES (?, ?, 'succeeded', ?, ?, NULL, ?)
        """,
        (run_id, subdirectory, fetched_at, response.status_code, str(raw_path)),
    )


def store_failure(
    conn: sqlite3.Connection,
    run_id: int,
    entry: Dict[str, Any],
    snapshot_date: str,
    fetched_at: str,
    url: str,
    http_status: Optional[int],
    error: str,
) -> None:
    subdirectory = entry["subdirectory"]
    upsert_company(conn, entry, None)
    conn.execute(
        """
        INSERT INTO stock_snapshots (
            run_id, subdirectory, snapshot_date, fetched_at, url, http_status,
            ok, final_url, html_bytes, raw_json_path, raw_json_sha256,
            page_props_keys_json, error
        ) VALUES (?, ?, ?, ?, ?, ?, 0, NULL, NULL, NULL, NULL, NULL, ?)
        ON CONFLICT(subdirectory, snapshot_date) DO UPDATE SET
            run_id = excluded.run_id,
            fetched_at = excluded.fetched_at,
            url = excluded.url,
            http_status = excluded.http_status,
            ok = 0,
            error = excluded.error
        """,
        (run_id, subdirectory, snapshot_date, fetched_at, url, http_status, error),
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO sync_results
        (run_id, subdirectory, status, fetched_at, http_status, error, raw_json_path)
        VALUES (?, ?, 'failed', ?, ?, ?, NULL)
        """,
        (run_id, subdirectory, fetched_at, http_status, error),
    )


def store_skip(conn: sqlite3.Connection, run_id: int, entry: Dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO sync_results
        (run_id, subdirectory, status, fetched_at, http_status, error, raw_json_path)
        VALUES (?, ?, 'skipped', ?, NULL, NULL, NULL)
        """,
        (run_id, entry["subdirectory"], utc_now()),
    )


def sleep_between(min_seconds: float, max_seconds: float) -> None:
    if max_seconds <= 0:
        return
    low = max(0.0, min_seconds)
    high = max(low, max_seconds)
    time.sleep(random.uniform(low, high))


def sync(args: argparse.Namespace) -> int:
    companies = load_companies(args.company_list, args.include_all_types)
    if args.offset:
        companies = companies[args.offset :]
    if args.limit:
        companies = companies[: args.limit]

    conn = connect_db(args.db)
    init_db(conn)
    run_id = begin_run(conn, args, len(companies))
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    print(
        f"tickertape_sync started run_id={run_id} companies={len(companies)} "
        f"snapshot_date={args.snapshot_date} db={args.db}"
    )

    succeeded = failed = skipped = 0
    started = time.monotonic()
    try:
        for index, entry in enumerate(companies, start=1):
            subdirectory = entry["subdirectory"]
            url = f"https://www.tickertape.in/stocks/{subdirectory}"
            fetched_at = utc_now()

            if args.resume and not args.force and already_fetched(conn, subdirectory, args.snapshot_date):
                store_skip(conn, run_id, entry)
                conn.commit()
                skipped += 1
                continue

            response, fetch_error = fetch_html(session, url, args.timeout, args.retries)
            if response is None:
                store_failure(conn, run_id, entry, args.snapshot_date, fetched_at, url, None, fetch_error or "request failed")
                conn.commit()
                failed += 1
            elif response.status_code != 200:
                store_failure(
                    conn,
                    run_id,
                    entry,
                    args.snapshot_date,
                    fetched_at,
                    url,
                    response.status_code,
                    f"HTTP {response.status_code}",
                )
                conn.commit()
                failed += 1
            else:
                try:
                    html = response.text
                    page_props = extract_page_props(html)
                    raw_path, raw_sha = write_raw(args.raw_dir, args.snapshot_date, subdirectory, page_props)
                    store_success(
                        conn,
                        run_id,
                        entry,
                        args.snapshot_date,
                        fetched_at,
                        url,
                        response,
                        len(html.encode("utf-8")),
                        page_props,
                        raw_path,
                        raw_sha,
                    )
                    conn.commit()
                    succeeded += 1
                except Exception as exc:  # noqa: BLE001 - store data errors in sync log
                    store_failure(conn, run_id, entry, args.snapshot_date, fetched_at, url, response.status_code, str(exc))
                    conn.commit()
                    failed += 1

            if args.max_failures and failed >= args.max_failures:
                raise RuntimeError(f"max failures reached: {failed}")

            if index % args.progress_every == 0 or index == len(companies):
                elapsed = time.monotonic() - started
                print(
                    f"progress index={index}/{len(companies)} succeeded={succeeded} "
                    f"failed={failed} skipped={skipped} elapsed_sec={elapsed:.1f}"
                )

            sleep_between(args.sleep_min, args.sleep_max)

        status = "completed_with_failures" if failed else "completed"
        finish_run(conn, run_id, status)
        print(
            f"tickertape_sync finished run_id={run_id} status={status} "
            f"succeeded={succeeded} failed={failed} skipped={skipped}"
        )
        return 0 if failed == 0 else 2
    except Exception as exc:  # noqa: BLE001 - top-level sync failure is recorded
        finish_run(conn, run_id, "failed", str(exc))
        print(f"tickertape_sync failed run_id={run_id} error={exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync TickerTape stock pages into a local SQLite repository.")
    parser.add_argument("--company-list", type=Path, default=Path("full-company-list.json"))
    parser.add_argument("--db", type=Path, default=Path("local_repository/tickertape.sqlite"))
    parser.add_argument("--raw-dir", type=Path, default=Path("local_repository/raw"))
    parser.add_argument("--snapshot-date", default=snapshot_date_default())
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep-min", type=float, default=0.75)
    parser.add_argument("--sleep-max", type=float, default=2.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-failures", type=int, default=0)
    parser.add_argument("--include-all-types", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return sync(args)


if __name__ == "__main__":
    raise SystemExit(main())
