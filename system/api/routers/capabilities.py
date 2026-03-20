from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["capabilities"])


@router.get("/capabilities")
def capabilities(request: Request) -> dict[str, object]:
    profile = request.app.state.profile
    return {
        "profile": profile,
        "agent_ready": True,
        "notes": [
            "Profiles are composable so one repository can publish multiple APIs.",
            "Use the agent manifest and OpenAPI document for tool discovery.",
            "Quality checks are first-class signals and should be consumed by agents.",
        ],
    }
