"""Local vector index support for agent-facing retrieval.

The module keeps raw source data local, stores deterministic document
embeddings in SQLite, and uses TurboVec when available for compressed search.
When TurboVec is not installed, it falls back to exact cosine search over the
stored vectors so the API surface remains usable.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import re
import sqlite3
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_EMBEDDING_DIM = 256
DEFAULT_EMBEDDING_MODEL = "local-hash-v1"
DEFAULT_BIT_WIDTH = 4
TOKEN_RE = re.compile(r"[A-Za-z0-9_.$%-]+")


class VectorIndexError(RuntimeError):
    """Raised when vector index operations cannot be completed."""


@dataclass(frozen=True)
class TurboVecRuntime:
    available: bool
    reason: str
    numpy: Any | None = None
    id_map_index: Any | None = None


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def turbovec_runtime() -> TurboVecRuntime:
    try:
        import numpy as np  # type: ignore[import-not-found]
        from turbovec import IdMapIndex  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return TurboVecRuntime(
            available=False,
            reason=f"TurboVec unavailable: {exc}. Install with `pip install turbovec`.",
        )
    return TurboVecRuntime(available=True, reason="TurboVec available", numpy=np, id_map_index=IdMapIndex)


def initialize_vector_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS document_embeddings (
            embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL UNIQUE,
            run_id INTEGER NOT NULL,
            source_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            content_preview TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            embedding_dim INTEGER NOT NULL,
            embedding_blob BLOB NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(document_id) REFERENCES raw_documents(document_id)
        );

        CREATE INDEX IF NOT EXISTS idx_document_embeddings_run
            ON document_embeddings(run_id, source_name);
        CREATE INDEX IF NOT EXISTS idx_document_embeddings_hash
            ON document_embeddings(content_sha256);

        CREATE TABLE IF NOT EXISTS vector_index_state (
            state_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            source_name TEXT,
            backend TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            embedding_dim INTEGER NOT NULL,
            bit_width INTEGER NOT NULL,
            index_path TEXT,
            item_count INTEGER NOT NULL,
            built_at TEXT NOT NULL,
            notes TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_vector_index_state_latest
            ON vector_index_state(run_id, source_name, built_at);
        """
    )
    conn.commit()


def hash_embedding(text: str, dim: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    if dim < 8:
        raise VectorIndexError("embedding_dim must be at least 8")

    vector = [0.0] * dim
    tokens = TOKEN_RE.findall((text or "").lower()) or ["empty"]
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8, person=b"isdvec01").digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]


def vector_to_blob(values: list[float]) -> bytes:
    return array("f", values).tobytes()


def blob_to_vector(blob: bytes) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    return list(values)


def latest_completed_run_id(conn: sqlite3.Connection) -> int | None:
    try:
        row = conn.execute(
            """
            SELECT run_id
            FROM ingestion_runs
            WHERE status = 'completed'
            ORDER BY run_id DESC
            LIMIT 1
            """
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    return int(row["run_id"] if isinstance(row, sqlite3.Row) else row[0]) if row else None


def _resolve_run_id(conn: sqlite3.Connection, run_id: int | None) -> int:
    resolved = run_id if run_id is not None else latest_completed_run_id(conn)
    if resolved is None:
        raise VectorIndexError("No completed ingestion run found for vector indexing")
    return int(resolved)


def _document_rows(
    conn: sqlite3.Connection,
    run_id: int,
    source_name: str | None,
    limit: int,
) -> list[sqlite3.Row]:
    clauses = ["run_id = ?"]
    params: list[Any] = [run_id]
    if source_name:
        clauses.append("source_name = ?")
        params.append(source_name)
    params.append(limit)
    return conn.execute(
        f"""
        SELECT document_id, run_id, source_name, file_path, content_sha256, content
        FROM raw_documents
        WHERE {' AND '.join(clauses)}
        ORDER BY document_id ASC
        LIMIT ?
        """,
        params,
    ).fetchall()


def _embedding_rows(
    conn: sqlite3.Connection,
    run_id: int,
    source_name: str | None,
    embedding_dim: int,
    embedding_model: str,
    limit: int,
) -> list[sqlite3.Row]:
    clauses = ["run_id = ?", "embedding_dim = ?", "embedding_model = ?"]
    params: list[Any] = [run_id, embedding_dim, embedding_model]
    if source_name:
        clauses.append("source_name = ?")
        params.append(source_name)
    params.append(limit)
    return conn.execute(
        f"""
        SELECT document_id, run_id, source_name, file_path, content_sha256,
               content_preview, embedding_blob
        FROM document_embeddings
        WHERE {' AND '.join(clauses)}
        ORDER BY document_id ASC
        LIMIT ?
        """,
        params,
    ).fetchall()


def ensure_document_embeddings(
    conn: sqlite3.Connection,
    run_id: int,
    source_name: str | None = None,
    limit: int = 1000,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> int:
    initialize_vector_schema(conn)
    rows = _document_rows(conn, run_id, source_name, limit)
    now = utc_now()
    upserted = 0

    for row in rows:
        content = row["content"] or ""
        content_preview = content[:2000]
        embedding = vector_to_blob(hash_embedding(content, embedding_dim))
        conn.execute(
            """
            INSERT OR REPLACE INTO document_embeddings (
                document_id, run_id, source_name, file_path, content_sha256,
                content_preview, embedding_model, embedding_dim, embedding_blob, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["document_id"],
                row["run_id"],
                row["source_name"],
                row["file_path"],
                row["content_sha256"],
                content_preview,
                embedding_model,
                embedding_dim,
                embedding,
                now,
            ),
        )
        upserted += 1

    conn.commit()
    return upserted


def latest_vector_state(
    conn: sqlite3.Connection,
    run_id: int | None = None,
    source_name: str | None = None,
) -> dict[str, Any] | None:
    initialize_vector_schema(conn)
    clauses: list[str] = []
    params: list[Any] = []
    if run_id is not None:
        clauses.append("run_id = ?")
        params.append(run_id)
    if source_name is not None:
        clauses.append("source_name = ?")
        params.append(source_name)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    row = conn.execute(
        f"""
        SELECT state_id, run_id, source_name, backend, embedding_model,
               embedding_dim, bit_width, index_path, item_count, built_at, notes
        FROM vector_index_state
        {where}
        ORDER BY built_at DESC, state_id DESC
        LIMIT 1
        """,
        params,
    ).fetchone()
    return dict(row) if row else None


def vector_status(conn: sqlite3.Connection, index_dir: Path) -> dict[str, Any]:
    runtime = turbovec_runtime()
    state = latest_vector_state(conn)
    return {
        "backend": "turbovec" if runtime.available else "exact",
        "turbovec_available": runtime.available,
        "turbovec_reason": runtime.reason,
        "index_dir": str(index_dir),
        "latest_index": state,
    }


def _write_index_state(
    conn: sqlite3.Connection,
    run_id: int,
    source_name: str | None,
    backend: str,
    embedding_model: str,
    embedding_dim: int,
    bit_width: int,
    index_path: Path | None,
    item_count: int,
    notes: str,
) -> dict[str, Any]:
    built_at = utc_now()
    cursor = conn.execute(
        """
        INSERT INTO vector_index_state (
            run_id, source_name, backend, embedding_model, embedding_dim,
            bit_width, index_path, item_count, built_at, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            source_name,
            backend,
            embedding_model,
            embedding_dim,
            bit_width,
            str(index_path) if index_path else None,
            item_count,
            built_at,
            notes,
        ),
    )
    conn.commit()
    return {
        "state_id": int(cursor.lastrowid),
        "run_id": run_id,
        "source_name": source_name,
        "backend": backend,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "bit_width": bit_width,
        "index_path": str(index_path) if index_path else None,
        "item_count": item_count,
        "built_at": built_at,
        "notes": notes,
    }


def build_vector_index(
    conn: sqlite3.Connection,
    index_dir: Path,
    run_id: int | None = None,
    source_name: str | None = None,
    limit: int = 1000,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    bit_width: int = DEFAULT_BIT_WIDTH,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, Any]:
    if bit_width not in {2, 4}:
        raise VectorIndexError("bit_width must be 2 or 4")
    if limit < 1:
        raise VectorIndexError("limit must be positive")

    resolved_run_id = _resolve_run_id(conn, run_id)
    ensure_document_embeddings(conn, resolved_run_id, source_name, limit, embedding_dim, embedding_model)
    rows = _embedding_rows(conn, resolved_run_id, source_name, embedding_dim, embedding_model, limit)
    if not rows:
        raise VectorIndexError("No raw documents found to index")

    runtime = turbovec_runtime()
    if runtime.available:
        index_dir.mkdir(parents=True, exist_ok=True)
        source_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", source_name or "all").strip("-") or "all"
        index_path = index_dir / f"run_{resolved_run_id}_{source_slug}_{embedding_dim}d_{bit_width}bit.tvim"
        vectors = runtime.numpy.asarray(
            [blob_to_vector(row["embedding_blob"]) for row in rows],
            dtype=runtime.numpy.float32,
        )
        ids = runtime.numpy.asarray([int(row["document_id"]) for row in rows], dtype=runtime.numpy.uint64)
        index = runtime.id_map_index(dim=embedding_dim, bit_width=bit_width)
        index.add_with_ids(vectors, ids)
        index.prepare()
        index.write(str(index_path))
        return _write_index_state(
            conn,
            resolved_run_id,
            source_name,
            "turbovec",
            embedding_model,
            embedding_dim,
            bit_width,
            index_path,
            len(rows),
            "TurboVec IdMapIndex built from local document embeddings.",
        )

    return _write_index_state(
        conn,
        resolved_run_id,
        source_name,
        "exact",
        embedding_model,
        embedding_dim,
        bit_width,
        None,
        len(rows),
        runtime.reason,
    )


def _score_exact(query_vector: list[float], rows: list[sqlite3.Row]) -> list[tuple[float, sqlite3.Row]]:
    scored = []
    for row in rows:
        vector = blob_to_vector(row["embedding_blob"])
        score = sum(a * b for a, b in zip(query_vector, vector))
        scored.append((score, row))
    return sorted(scored, key=lambda item: item[0], reverse=True)


def _rows_for_ids(conn: sqlite3.Connection, ids: list[int]) -> dict[int, sqlite3.Row]:
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"""
        SELECT document_id, run_id, source_name, file_path, content_sha256, content_preview
        FROM document_embeddings
        WHERE document_id IN ({placeholders})
        """,
        ids,
    ).fetchall()
    return {int(row["document_id"]): row for row in rows}


def search_vectors(
    conn: sqlite3.Connection,
    query: str,
    index_dir: Path,
    run_id: int | None = None,
    source_name: str | None = None,
    k: int = 10,
) -> dict[str, Any]:
    if not query.strip():
        raise VectorIndexError("query must not be empty")
    if k < 1:
        raise VectorIndexError("k must be positive")

    initialize_vector_schema(conn)
    state = latest_vector_state(conn, run_id, source_name) or latest_vector_state(conn, run_id)
    if not state:
        raise VectorIndexError("No vector index has been built yet. Call POST /vectors/rebuild first.")

    resolved_run_id = int(run_id or state["run_id"])
    embedding_dim = int(state["embedding_dim"])
    embedding_model = str(state["embedding_model"])
    query_vector = hash_embedding(query, embedding_dim)
    rows = _embedding_rows(conn, resolved_run_id, source_name, embedding_dim, embedding_model, 100000)
    if not rows:
        return {
            "query": query,
            "run_id": resolved_run_id,
            "backend": state["backend"],
            "count": 0,
            "results": [],
            "notes": "No embeddings matched the requested filters.",
        }

    backend = "exact"
    notes = "Exact local fallback search."
    runtime = turbovec_runtime()
    index_path_value = state.get("index_path")
    index_path = Path(index_path_value) if index_path_value else index_dir / ""

    if state["backend"] == "turbovec" and runtime.available and index_path_value and index_path.exists():
        try:
            index = runtime.id_map_index.load(str(index_path))
            queries = runtime.numpy.asarray([query_vector], dtype=runtime.numpy.float32)
            allowlist = None
            if source_name:
                allowlist = runtime.numpy.asarray([int(row["document_id"]) for row in rows], dtype=runtime.numpy.uint64)
            scores, ids = index.search(queries, k=min(k, len(rows)), allowlist=allowlist)
            id_values = [int(value) for value in ids[0].tolist()]
            row_map = _rows_for_ids(conn, id_values)
            results = [
                _format_result(row_map[doc_id], float(score))
                for doc_id, score in zip(id_values, scores[0].tolist())
                if doc_id in row_map
            ]
            backend = "turbovec"
            notes = "TurboVec IdMapIndex search."
            return {
                "query": query,
                "run_id": resolved_run_id,
                "backend": backend,
                "count": len(results),
                "results": results,
                "notes": notes,
            }
        except Exception as exc:  # noqa: BLE001
            notes = f"TurboVec search failed; used exact fallback. Error: {exc}"

    scored = _score_exact(query_vector, rows)[:k]
    return {
        "query": query,
        "run_id": resolved_run_id,
        "backend": backend,
        "count": len(scored),
        "results": [_format_result(row, float(score)) for score, row in scored],
        "notes": notes,
    }


def _format_result(row: sqlite3.Row, score: float) -> dict[str, Any]:
    return {
        "document_id": int(row["document_id"]),
        "score": score,
        "source_name": row["source_name"],
        "file_path": row["file_path"],
        "content_sha256": row["content_sha256"],
        "content_preview": row["content_preview"],
    }
