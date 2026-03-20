from __future__ import annotations

from collections.abc import Generator

from fastapi import Depends, Request

from .database import DatabaseConfig



def get_database_config(request: Request) -> DatabaseConfig:
    return request.app.state.database



def get_connection(
    database: DatabaseConfig = Depends(get_database_config),
) -> Generator:
    conn = database.connect()
    try:
        yield conn
    finally:
        conn.close()
