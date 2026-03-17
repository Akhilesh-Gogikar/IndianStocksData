"""Daily orchestration pipeline for scraping + ingestion.

This module wraps existing scraper scripts in the repository, executes them,
and stores normalized artifacts in a query-friendly SQLite database.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ScraperJob:
    name: str
    script: str
    output_globs: tuple[str, ...]


DEFAULT_JOBS: tuple[ScraperJob, ...] = (
    ScraperJob(
        name="news",
        script="News_data_extractor.py",
        output_globs=("*.csv", "*.json"),
    ),
    ScraperJob(
        name="price_and_news",
        script="News_Price_Downloader.py",
        output_globs=("*.csv", "*.json"),
    ),
    ScraperJob(
        name="tickertape",
        script="TickerTape_Scraper.py",
        output_globs=("*.csv", "*.json"),
    ),
)


def initialize_db(connection: sqlite3.Connection, schema_file: Path) -> None:
    connection.executescript(schema_file.read_text(encoding="utf-8"))
    connection.commit()


def start_run(connection: sqlite3.Connection) -> int:
    now = dt.datetime.utcnow().isoformat()
    cursor = connection.execute(
        """
        INSERT INTO ingestion_runs (run_date, status, started_at)
        VALUES (?, 'running', ?)
        """,
        (dt.date.today().isoformat(), now),
    )
    connection.commit()
    return int(cursor.lastrowid)


def finish_run(connection: sqlite3.Connection, run_id: int, status: str, notes: str = "") -> None:
    connection.execute(
        """
        UPDATE ingestion_runs
        SET status = ?, finished_at = ?, notes = ?
        WHERE run_id = ?
        """,
        (status, dt.datetime.utcnow().isoformat(), notes, run_id),
    )
    connection.commit()


def run_job(job: ScraperJob, repo_root: Path) -> None:
    script_path = repo_root / job.script
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    subprocess.run(
        ["python", str(script_path)],
        check=True,
        cwd=repo_root,
    )


def collect_files(repo_root: Path, patterns: Iterable[str]) -> list[Path]:
    collected: list[Path] = []
    for pattern in patterns:
        collected.extend(Path(match).resolve() for match in glob.glob(str(repo_root / pattern)))
    return sorted(set(collected))


def parse_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        return json.dumps(json.loads(file_path.read_text(encoding="utf-8")), ensure_ascii=False)
    if suffix == ".csv":
        with file_path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return json.dumps(rows, ensure_ascii=False)
    return file_path.read_text(encoding="utf-8", errors="ignore")


def ingest_documents(connection: sqlite3.Connection, run_id: int, source_name: str, files: list[Path]) -> int:
    inserted = 0
    for file_path in files:
        file_type = file_path.suffix.lower().lstrip(".") or "unknown"
        content = parse_file(file_path)
        connection.execute(
            """
            INSERT OR IGNORE INTO raw_documents
                (run_id, source_name, file_path, file_type, content, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                source_name,
                str(file_path),
                file_type,
                content,
                dt.datetime.utcnow().isoformat(),
            ),
        )
        inserted += 1
    connection.commit()
    return inserted


def pipeline(repo_root: Path, db_path: Path, jobs: tuple[ScraperJob, ...] = DEFAULT_JOBS) -> int:
    schema_file = repo_root / "system" / "schema.sql"
    conn = sqlite3.connect(db_path)
    initialize_db(conn, schema_file)

    run_id = start_run(conn)
    try:
        total = 0
        for job in jobs:
            run_job(job, repo_root)
            files = collect_files(repo_root, job.output_globs)
            total += ingest_documents(conn, run_id, job.name, files)

        finish_run(conn, run_id, "completed", notes=f"Ingested {total} files")
        return run_id
    except Exception as exc:  # noqa: BLE001
        finish_run(conn, run_id, "failed", notes=str(exc))
        raise
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run daily scraping + ingestion pipeline")
    parser.add_argument("--repo-root", default=".", help="Repository root where scraper scripts are stored")
    parser.add_argument("--db-path", default="./system/market_intel.db", help="SQLite DB path")
    args = parser.parse_args()

    run_id = pipeline(Path(args.repo_root).resolve(), Path(args.db_path).resolve())
    print(f"Pipeline completed successfully. run_id={run_id}")


if __name__ == "__main__":
    main()
