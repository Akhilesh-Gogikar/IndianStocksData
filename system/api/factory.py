from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, FastAPI

from .database import DatabaseConfig
from .routers import agents, capabilities, discovery, documents, freshness, health, market, portfolios, runs, screeners, vectors, watchlists
from .security import configure_security


@dataclass(frozen=True)
class ApiProfile:
    name: str
    title: str
    description: str
    routers: tuple[APIRouter, ...]


PROFILE_REGISTRY: dict[str, ApiProfile] = {
    "full": ApiProfile(
        name="full",
        title="Indian Stocks Data Platform API",
        description="Unified market-data, retrieval, and agent endpoints.",
        routers=(
            health.router,
            capabilities.router,
            discovery.router,
            runs.router,
            freshness.router,
            market.router,
            screeners.router,
            watchlists.router,
            portfolios.router,
            documents.router,
            vectors.router,
            agents.router,
        ),
    ),
    "market-data": ApiProfile(
        name="market-data",
        title="Indian Stocks Market Data API",
        description="Lean API focused on canonical market data, run status, and source documents.",
        routers=(
            health.router,
            capabilities.router,
            discovery.router,
            runs.router,
            freshness.router,
            market.router,
            screeners.router,
            watchlists.router,
            portfolios.router,
            documents.router,
        ),
    ),
    "agent-runtime": ApiProfile(
        name="agent-runtime",
        title="Indian Stocks Agent Runtime API",
        description="Agent-facing API with discovery, context, and orchestration guidance.",
        routers=(
            health.router,
            capabilities.router,
            discovery.router,
            runs.router,
            freshness.router,
            market.router,
            screeners.router,
            watchlists.router,
            portfolios.router,
            vectors.router,
            agents.router,
        ),
    ),
}


def list_profiles() -> list[str]:
    return sorted(PROFILE_REGISTRY)



def create_app(
    db_path: Path,
    profile_name: str = "full",
    vector_index_dir: Path | None = None,
) -> FastAPI:
    if profile_name not in PROFILE_REGISTRY:
        supported = ", ".join(list_profiles())
        raise ValueError(f"Unsupported profile '{profile_name}'. Expected one of: {supported}")

    profile = PROFILE_REGISTRY[profile_name]
    app = FastAPI(title=profile.title, version="2.0.0", description=profile.description)
    app.state.database = DatabaseConfig(db_path)
    app.state.vector_index_dir = vector_index_dir or Path("./local_repository/vector_indexes").resolve()
    app.state.profile = {
        "name": profile.name,
        "title": profile.title,
        "description": profile.description,
        "router_names": [router.tags[0] if router.tags else "untagged" for router in profile.routers],
    }
    app.state.security = configure_security(app).public_metadata()

    for router in profile.routers:
        app.include_router(router)

    return app
