from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from system.ai.vector_index import (
    DEFAULT_BIT_WIDTH,
    DEFAULT_EMBEDDING_DIM,
    VectorIndexError,
    build_vector_index,
    search_vectors,
    vector_status,
)

from ..dependencies import get_connection


router = APIRouter(prefix="/vectors", tags=["vectors"])


@router.get("/status")
def vectors_status(
    request: Request,
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    return vector_status(conn, request.app.state.vector_index_dir)


@router.post("/rebuild")
def vectors_rebuild(
    request: Request,
    run_id: int | None = Query(default=None),
    source_name: str | None = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=100000),
    embedding_dim: int = Query(default=DEFAULT_EMBEDDING_DIM, ge=8, le=4096),
    bit_width: int = Query(default=DEFAULT_BIT_WIDTH, ge=2, le=4),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return build_vector_index(
            conn,
            request.app.state.vector_index_dir,
            run_id=run_id,
            source_name=source_name,
            limit=limit,
            embedding_dim=embedding_dim,
            bit_width=bit_width,
        )
    except VectorIndexError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except sqlite3.OperationalError as exc:
        detail = f"Database is not initialized for raw documents: {exc}"
        raise HTTPException(status_code=400, detail=detail) from exc


@router.get("/search")
def vectors_search(
    request: Request,
    query: str = Query(..., min_length=1),
    run_id: int | None = Query(default=None),
    source_name: str | None = Query(default=None),
    k: int = Query(default=10, ge=1, le=100),
    conn: sqlite3.Connection = Depends(get_connection),
) -> dict[str, Any]:
    try:
        return search_vectors(
            conn,
            query,
            request.app.state.vector_index_dir,
            run_id=run_id,
            source_name=source_name,
            k=k,
        )
    except VectorIndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
