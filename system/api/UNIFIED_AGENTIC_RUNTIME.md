# Unified Agentic Runtime (All Products)

## Purpose
One shared multi-agent runtime now exists for all product surfaces so each product can invoke the same orchestrator while still applying product-specific objectives, context, and retrieval mode.

## Entry Point
- API route: `POST /agents/unified-agentic-runtime`
- Runtime profile: available in `agent-runtime` and `full` API profiles.

## What It Does
The runtime executes a fixed multi-agent harness:

1. Planner agent converts objective to an execution plan.
2. Retrieval router runs one of `rag`, `sql`, or `hybrid`.
3. Deep-research agent synthesizes grounded findings.
4. Documenter emits reusable markdown artifacts.

## Data Access Rules
- SQL access is strictly read-only (`SELECT`/`WITH` only).
- Multi-statement SQL and write/mutation keywords are blocked.
- RAG retrieval uses local vector index evidence (`search_vectors`) and can auto-build index if enabled.
- Raw upstream files remain local. Outputs use processed payloads, retrieval excerpts, and metadata.

## Provider Routing
Provider chain for deep research:

1. Firebase free-tier (`provider_preference: firebase-free-tier` or `auto`).
2. Local `llama.cpp` OpenAI-compatible runtime.
3. Deterministic market-data fallback.

If Firebase credentials or quota are unavailable, runtime falls through to local llama automatically.

## Agentic Document Outputs
Each runtime call can produce:
- `agentic-execution-brief.md`
- `agentic-evidence-log.md`

These artifacts are generated in-memory and returned in the API response for downstream product workflows (UI export, workflow handoff, audit trail, or storage).

## Product Integration Pattern
Every product sends:
- `product_id`
- `objective`
- `retrieval_mode`
- optional `sql_queries` and/or `rag_query`
- optional `context`, `focus`, and provider preference

This allows one runtime to serve research, screener, portfolio, watchlist, buyer-room, and API-builder surfaces while preserving consistent guardrails.
