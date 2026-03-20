from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter(tags=["discovery"])


@router.get("/.well-known/agent-manifest.json", include_in_schema=False)
def agent_manifest(request: Request) -> dict[str, Any]:
    base_url = str(request.base_url).rstrip("/")
    profile = request.app.state.profile
    return {
        "name": profile["title"],
        "description": profile["description"],
        "profile": profile["name"],
        "openapi_url": f"{base_url}{request.app.openapi_url}",
        "capabilities_url": f"{base_url}/capabilities",
        "tools": [
            {
                "name": "runs.latest",
                "method": "GET",
                "path": "/runs/latest",
                "description": "Retrieve the latest completed ingestion run and its quality audits.",
            },
            {
                "name": "documents.current",
                "method": "GET",
                "path": "/documents/current",
                "description": "List the newest raw artifacts for retrieval-augmented workflows.",
            },
            {
                "name": "agents.context",
                "method": "GET",
                "path": "/agents/context/{ticker}",
                "description": "Return a prompt payload for an analyst or execution agent.",
            },
            {
                "name": "agents.workflow",
                "method": "GET",
                "path": "/agents/workflow/{ticker}",
                "description": "Return a recommended multi-step workflow for an agent to follow.",
            },
        ],
    }
