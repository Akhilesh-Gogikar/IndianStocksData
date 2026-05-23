from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response


PUBLIC_PREFIXES = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/.well-known/agent-manifest.json",
)


@dataclass(frozen=True)
class SecuritySettings:
    api_keys: frozenset[str]
    rate_limit_per_minute: int

    @property
    def api_key_required(self) -> bool:
        return bool(self.api_keys)

    def public_metadata(self) -> dict[str, object]:
        return {
            "api_key_required": self.api_key_required,
            "rate_limit_per_minute": self.rate_limit_per_minute or None,
        }


def settings_from_env() -> SecuritySettings:
    raw_keys = os.getenv("INDIAN_STOCKS_API_KEYS") or os.getenv("API_KEYS") or ""
    keys = frozenset(key.strip() for key in raw_keys.split(",") if key.strip())
    try:
        rate_limit = int(os.getenv("INDIAN_STOCKS_RATE_LIMIT_PER_MINUTE", "0") or "0")
    except ValueError:
        rate_limit = 0
    return SecuritySettings(api_keys=keys, rate_limit_per_minute=max(0, rate_limit))


def is_public_path(path: str) -> bool:
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in PUBLIC_PREFIXES)


def request_api_key(request: Request) -> str | None:
    header_key = request.headers.get("x-api-key")
    if header_key:
        return header_key.strip()
    authorization = request.headers.get("authorization") or ""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token:
        return token.strip()
    return None


class ApiKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, api_keys: frozenset[str]) -> None:
        super().__init__(app)
        self.api_keys = api_keys

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.method == "OPTIONS" or is_public_path(request.url.path):
            return await call_next(request)
        api_key = request_api_key(request)
        if api_key not in self.api_keys:
            return JSONResponse(
                status_code=401,
                content={"detail": {"code": "api_key_required", "message": "A valid API key is required."}},
            )
        request.state.api_key = api_key
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, limit_per_minute: int) -> None:
        super().__init__(app)
        self.limit_per_minute = limit_per_minute
        self.hits: dict[tuple[str, int], int] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.method == "OPTIONS" or is_public_path(request.url.path):
            return await call_next(request)
        bucket = int(time.time() // 60)
        client_key = request_api_key(request) or (request.client.host if request.client else "unknown")
        key = (client_key, bucket)
        self.hits[key] = self.hits.get(key, 0) + 1
        if self.hits[key] > self.limit_per_minute:
            return JSONResponse(
                status_code=429,
                content={"detail": {"code": "rate_limit_exceeded", "message": "Too many requests."}},
                headers={"Retry-After": "60"},
            )
        if len(self.hits) > 10000:
            self.hits = {stored_key: count for stored_key, count in self.hits.items() if stored_key[1] >= bucket - 1}
        return await call_next(request)


def configure_security(app: FastAPI, settings: SecuritySettings | None = None) -> SecuritySettings:
    active = settings or settings_from_env()
    if active.rate_limit_per_minute:
        app.add_middleware(RateLimitMiddleware, limit_per_minute=active.rate_limit_per_minute)
    if active.api_key_required:
        app.add_middleware(ApiKeyMiddleware, api_keys=active.api_keys)
    return active
