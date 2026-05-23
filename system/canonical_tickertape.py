from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DEFAULT_SOURCE_DB = Path("./local_repository/tickertape.sqlite")
DEFAULT_TARGET_DB = Path("./system/market_intel.db")
DEFAULT_SCHEMA = Path("./system/schema.sql")
DATA_RIGHTS_STATUS = "source-derived-review-required"


def now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_json(value: Any, default: Any = None) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def as_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def normalize_symbol(value: Any) -> str | None:
    if value in (None, ""):
        return None
    symbol = str(value).strip().upper()
    return symbol or None


def number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        is not None
    )


def initialize_target(conn: sqlite3.Connection, schema_path: Path) -> None:
    conn.executescript(schema_path.read_text(encoding="utf-8"))
    conn.commit()


def latest_source_run(source: sqlite3.Connection) -> dict[str, Any] | None:
    if not table_exists(source, "sync_runs"):
        return None
    row = source.execute(
        """
        SELECT *
        FROM sync_runs
        WHERE finished_at IS NOT NULL
          AND status IN ('completed', 'completed_with_failures')
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()
    return dict(row) if row else None


def source_quality_status(source_run: dict[str, Any] | None) -> str:
    status = str((source_run or {}).get("status") or "").lower()
    if status in {"completed", "success", "ok"}:
        return "pass"
    if status in {"completed_with_failures", "partial", "warning"}:
        return "warning"
    if status:
        return "fail"
    return "unknown"


def ensure_import_run(
    target: sqlite3.Connection,
    source_run: dict[str, Any] | None,
    snapshot_date: str,
    processed_at: str,
) -> int:
    source_run_id = (source_run or {}).get("id")
    notes = f"canonical_tickertape_import source_run_id={source_run_id} snapshot_date={snapshot_date}"
    existing = target.execute(
        "SELECT run_id FROM ingestion_runs WHERE notes = ? ORDER BY run_id DESC LIMIT 1",
        (notes,),
    ).fetchone()
    if existing:
        run_id = int(existing["run_id"])
        target.execute(
            """
            UPDATE ingestion_runs
            SET run_date = ?, status = 'completed', started_at = ?, finished_at = ?
            WHERE run_id = ?
            """,
            (snapshot_date, processed_at, processed_at, run_id),
        )
        return run_id

    cursor = target.execute(
        """
        INSERT INTO ingestion_runs (run_date, status, started_at, finished_at, notes)
        VALUES (?, 'completed', ?, ?, ?)
        """,
        (snapshot_date, processed_at, processed_at, notes),
    )
    return int(cursor.lastrowid)


def clear_import_rows(target: sqlite3.Connection, run_id: int) -> None:
    for table in ("company_peers", "company_events", "financial_ratios", "quote_snapshots"):
        target.execute(f"DELETE FROM {table} WHERE local_ingestion_run_id = ?", (run_id,))
    target.execute(
        "DELETE FROM data_quality_audits WHERE run_id = ? AND check_name = 'canonical_tickertape_import'",
        (run_id,),
    )


def latest_rows(source: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = source.execute(
        """
        SELECT l.*, c.isin AS company_isin, c.sid AS company_sid, c.slug AS company_slug
        FROM latest_stock_data l
        LEFT JOIN companies c ON c.subdirectory = l.subdirectory
        ORDER BY l.snapshot_date DESC, l.updated_at DESC, l.subdirectory
        """
    ).fetchall()
    return [dict(row) for row in rows]


def compact_scorecard(scorecard: Any) -> list[dict[str, Any]]:
    if not isinstance(scorecard, list):
        return []
    compact = []
    for item in scorecard:
        if isinstance(item, dict):
            compact.append(
                {
                    "name": item.get("name"),
                    "tag": item.get("tag"),
                    "colour": item.get("colour"),
                    "rank": item.get("rank"),
                }
            )
    return compact


def ticker_from(row: dict[str, Any], info: dict[str, Any], quote: dict[str, Any]) -> str | None:
    info_block = info.get("info") or {}
    return (
        normalize_symbol(info_block.get("ticker"))
        or normalize_symbol(quote.get("ticker"))
        or normalize_symbol(row.get("company_sid"))
        or normalize_symbol(info.get("sid"))
    )


def company_payload(row: dict[str, Any], info: dict[str, Any], quote: dict[str, Any]) -> dict[str, Any] | None:
    ticker = ticker_from(row, info, quote)
    if not ticker:
        return None
    info_block = info.get("info") or {}
    gic = info.get("gic") or {}
    ratios = info.get("ratios") or {}
    return {
        "ticker": ticker,
        "name": info_block.get("name") or row.get("name") or ticker,
        "exchange": info_block.get("exchange") or quote.get("exchange"),
        "isin": row.get("company_isin") or info.get("isin"),
        "sector": gic.get("sector") or info_block.get("sector"),
        "industry": gic.get("industry") or gic.get("subindustry"),
        "market_cap": number(ratios.get("marketCap") or quote.get("marketCap")),
        "website": info_block.get("website"),
        "description": (info_block.get("description") or "").strip() or None,
        "extra_json": as_json(
            {
                "source": "tickertape",
                "subdirectory": row.get("subdirectory"),
                "type": info.get("type") or row.get("type"),
                "tradable": info.get("tradable"),
                "industry_group": gic.get("industrygroup"),
                "source_slug": info.get("slug") or row.get("company_slug"),
                "scorecard": compact_scorecard(parse_json(row.get("scorecard_json"), [])),
            }
        ),
    }


def insert_company(target: sqlite3.Connection, payload: dict[str, Any], run_id: int, as_of: str, processed_at: str, quality_status: str) -> None:
    target.execute(
        """
        INSERT INTO companies (
            ticker, name, exchange, isin, sector, industry, market_cap, website,
            description, local_ingestion_run_id, as_of, processed_at, quality_status,
            data_rights_status, extra_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            name = excluded.name,
            exchange = excluded.exchange,
            isin = excluded.isin,
            sector = excluded.sector,
            industry = excluded.industry,
            market_cap = excluded.market_cap,
            website = excluded.website,
            description = excluded.description,
            local_ingestion_run_id = excluded.local_ingestion_run_id,
            as_of = excluded.as_of,
            processed_at = excluded.processed_at,
            quality_status = excluded.quality_status,
            data_rights_status = excluded.data_rights_status,
            extra_json = excluded.extra_json
        """,
        (
            payload["ticker"],
            payload["name"],
            payload["exchange"],
            payload["isin"],
            payload["sector"],
            payload["industry"],
            payload["market_cap"],
            payload["website"],
            payload["description"],
            run_id,
            as_of,
            processed_at,
            quality_status,
            DATA_RIGHTS_STATUS,
            payload["extra_json"],
        ),
    )


def insert_quote(target: sqlite3.Connection, ticker: str, quote: dict[str, Any], run_id: int, as_of: str, processed_at: str, quality_status: str) -> None:
    if not quote:
        return
    target.execute(
        """
        INSERT INTO quote_snapshots (
            ticker, price, currency, open_price, high_price, low_price,
            previous_close, volume, raw_json, local_ingestion_run_id, as_of,
            processed_at, quality_status, data_rights_status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ticker,
            number(quote.get("price") or quote.get("c")),
            "INR",
            number(quote.get("o")),
            number(quote.get("h")),
            number(quote.get("l")),
            number(quote.get("prevClose") or quote.get("pc")),
            number(quote.get("vol")),
            as_json(
                {
                    "change": quote.get("change"),
                    "day_change": quote.get("dyChange"),
                    "turnover": quote.get("turnover"),
                    "high_52w": quote.get("high52w"),
                    "low_52w": quote.get("low52w"),
                    "source_timestamp": quote.get("ts"),
                }
            ),
            run_id,
            as_of,
            processed_at,
            quality_status,
            DATA_RIGHTS_STATUS,
        ),
    )


def insert_ratios(target: sqlite3.Connection, ticker: str, ratios: dict[str, Any], run_id: int, as_of: str, processed_at: str, quality_status: str) -> int:
    inserted = 0
    for name, value in sorted((ratios or {}).items()):
        numeric = number(value)
        if numeric is None:
            continue
        target.execute(
            """
            INSERT INTO financial_ratios (
                ticker, ratio_name, ratio_value, period, period_end,
                local_ingestion_run_id, as_of, processed_at, quality_status,
                data_rights_status
            )
            VALUES (?, ?, ?, 'latest', ?, ?, ?, ?, ?, ?)
            """,
            (ticker, name, numeric, as_of, run_id, as_of, processed_at, quality_status, DATA_RIGHTS_STATUS),
        )
        inserted += 1
    return inserted


def event_date(item: dict[str, Any]) -> str | None:
    for key in ("exDate", "broadcastTime", "date", "eventDate", "announcementDate", "createdAt"):
        value = item.get(key)
        if value:
            return str(value)[:10]
    return None


def insert_events(target: sqlite3.Connection, source: sqlite3.Connection, subdirectory: str, ticker: str, run_id: int, as_of: str, processed_at: str, quality_status: str) -> int:
    if not table_exists(source, "event_sections"):
        return 0
    count = 0
    rows = source.execute(
        "SELECT section_key, section_json FROM event_sections WHERE subdirectory = ?",
        (subdirectory,),
    ).fetchall()
    for row in rows:
        section = parse_json(row["section_json"], {})
        for bucket in ("upcoming", "past"):
            events = section.get(bucket) if isinstance(section, dict) else None
            if not isinstance(events, list):
                continue
            for item in events:
                if not isinstance(item, dict):
                    continue
                title = item.get("subject") or item.get("title") or item.get("description") or row["section_key"]
                target.execute(
                    """
                    INSERT INTO company_events (
                        ticker, event_type, event_date, title, description, source_url, raw_json,
                        local_ingestion_run_id, as_of, processed_at, quality_status, data_rights_status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ticker,
                        row["section_key"],
                        event_date(item),
                        str(title)[:240],
                        item.get("description"),
                        item.get("attachement") or item.get("attachment") or item.get("url"),
                        as_json({"bucket": bucket, "source_id": item.get("_id")}),
                        run_id,
                        as_of,
                        processed_at,
                        quality_status,
                        DATA_RIGHTS_STATUS,
                    ),
                )
                count += 1
    return count


def insert_peers(target: sqlite3.Connection, ticker: str, summary: dict[str, Any], run_id: int, as_of: str, processed_at: str, quality_status: str) -> int:
    peers = summary.get("aboutAndPeers") if isinstance(summary, dict) else None
    if not isinstance(peers, list):
        return 0
    count = 0
    for item in peers:
        if not isinstance(item, dict):
            continue
        peer_ticker = normalize_symbol(item.get("ticker") or item.get("sid"))
        if not peer_ticker or peer_ticker == ticker:
            continue
        target.execute(
            """
            INSERT INTO company_peers (
                ticker, peer_ticker, relationship, score, raw_json,
                local_ingestion_run_id, as_of, processed_at, quality_status, data_rights_status
            )
            VALUES (?, ?, 'tickertape_peer', NULL, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                peer_ticker,
                as_json({"name": item.get("name"), "sector": item.get("sector"), "source_slug": item.get("slug")}),
                run_id,
                as_of,
                processed_at,
                quality_status,
                DATA_RIGHTS_STATUS,
            ),
        )
        count += 1
    return count


def import_tickertape_canonical(source_db: Path = DEFAULT_SOURCE_DB, target_db: Path = DEFAULT_TARGET_DB, schema_path: Path = DEFAULT_SCHEMA, limit: int | None = None) -> dict[str, int | str]:
    source = sqlite3.connect(source_db)
    source.row_factory = sqlite3.Row
    target = sqlite3.connect(target_db)
    target.row_factory = sqlite3.Row
    try:
        initialize_target(target, schema_path)
        source_run = latest_source_run(source)
        rows = latest_rows(source)
        if limit is not None:
            rows = rows[:limit]
        snapshot_date = (source_run or {}).get("snapshot_date") or (rows[0]["snapshot_date"] if rows else now_utc()[:10])
        processed_at = now_utc()
        quality_status = source_quality_status(source_run)
        run_id = ensure_import_run(target, source_run, snapshot_date, processed_at)
        clear_import_rows(target, run_id)
        counts: dict[str, int | str] = {
            "run_id": run_id,
            "companies": 0,
            "quote_snapshots": 0,
            "financial_ratios": 0,
            "company_events": 0,
            "company_peers": 0,
        }
        for row in rows:
            info = parse_json(row.get("security_info_json"), {})
            quote = parse_json(row.get("security_quote_json"), {})
            summary = parse_json(row.get("security_summary_json"), {})
            payload = company_payload(row, info, quote)
            if not payload:
                continue
            ticker = payload["ticker"]
            as_of = row.get("snapshot_date") or snapshot_date
            insert_company(target, payload, int(run_id), as_of, processed_at, quality_status)
            counts["companies"] = int(counts["companies"]) + 1
            insert_quote(target, ticker, quote, int(run_id), as_of, processed_at, quality_status)
            if quote:
                counts["quote_snapshots"] = int(counts["quote_snapshots"]) + 1
            counts["financial_ratios"] = int(counts["financial_ratios"]) + insert_ratios(target, ticker, info.get("ratios") or {}, int(run_id), as_of, processed_at, quality_status)
            counts["company_events"] = int(counts["company_events"]) + insert_events(target, source, row["subdirectory"], ticker, int(run_id), as_of, processed_at, quality_status)
            counts["company_peers"] = int(counts["company_peers"]) + insert_peers(target, ticker, summary, int(run_id), as_of, processed_at, quality_status)
        target.execute(
            """
            INSERT INTO data_quality_audits (run_id, check_name, status, details, checked_at)
            VALUES (?, 'canonical_tickertape_import', ?, ?, ?)
            """,
            (run_id, quality_status, json.dumps(counts, sort_keys=True), processed_at),
        )
        target.commit()
        counts["quality_status"] = quality_status
        counts["snapshot_date"] = snapshot_date
        return counts
    finally:
        source.close()
        target.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import local Tickertape data into canonical serving tables")
    parser.add_argument("--source-db", default=str(DEFAULT_SOURCE_DB), help="Local Tickertape SQLite database")
    parser.add_argument("--target-db", default=str(DEFAULT_TARGET_DB), help="Serving API SQLite database")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA), help="Serving API schema file")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    counts = import_tickertape_canonical(
        source_db=Path(args.source_db).resolve(),
        target_db=Path(args.target_db).resolve(),
        schema_path=Path(args.schema).resolve(),
        limit=args.limit,
    )
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
