# API-First Development Plan

## Operating Assumption

Every data refresh pulls and stores a raw copy of the source payload locally before any parsing, enrichment, scoring, report generation, widget rendering, or AI-agent response is produced.

The business should therefore be built as a local-first data refinery plus API-first service platform. Raw source data stays on the local ingestion side. The local system massages, normalizes, enriches, compresses, and restructures it into LLM-friendly and product-friendly derived payloads. The server receives only those processed service payloads and uses them to power APIs, widgets, dashboards, and agentic AI products.

In this plan, "make the data our own" means creating differentiated derived data products through cleaning, normalization, feature engineering, entity resolution, summaries, risk flags, explainable scores, lineage, and LLM-ready context packaging. It does not remove the need to respect source terms, licenses, and commercial-use restrictions.

## Business Goal

Build one shared Indian equities data and intelligence platform, then package it into multiple sellable surfaces:

- REST API for developers, websites, fintech products, and internal apps.
- Agent-runtime API for AI systems that need deterministic market-data tools.
- Embeddable widgets for finance websites and communities.
- Dashboard products for research workflows, screeners, data quality, and sector intelligence.
- Content and research-workspace products built on the same canonical data layer.

The API is the business core. The website is distribution. The local raw data store is the audit, replay, and recovery layer. The server-side store is the product-serving layer.

## Non-Negotiable Data Architecture

1. Pull raw data from each provider or source on a local schedule.
2. Store the exact raw response or file locally as an immutable artifact.
3. Attach source, pull timestamp, hash, schema hint, source rights, and ingestion run ID.
4. Parse raw artifacts into local canonical tables only after local raw storage succeeds.
5. Massage and process local canonical data into LLM-friendly service payloads.
6. Ship only processed payloads, indexes, summaries, features, and service-ready documents to the server.
7. Run data quality checks before publishing server-side derived records.
8. Expose freshness, confidence, and processing timestamps in every external response.
9. Never let public products depend directly on scraper outputs or unprocessed raw files.

Current repo anchors:

- `system/schema.sql` already includes `ingestion_runs`, `raw_documents`, `entity_risk_signals`, `data_quality_audits`, and `analyst_reports`.
- `system/daily_pipeline.py` is the right orchestration entrypoint.
- `system/service_api.py` and `system/api/factory.py` already support profile-specific FastAPI services.
- Existing API profiles are `full`, `market-data`, and `agent-runtime`.

## Local-To-Server Data Flow

The intended operating model is:

1. Local puller collects raw market, company, event, news, and reference data.
2. Local raw artifact store keeps exact source copies for audit and replay.
3. Local canonicalizer converts source-specific fields into stable internal entities.
4. Local processor creates value-added derived data:
   - clean company profiles;
   - quote snapshots;
   - financial ratios;
   - event timelines;
   - peer maps;
   - sector summaries;
   - anomaly flags;
   - scoring features;
   - LLM context packs;
   - compact retrieval chunks;
   - cited company and sector briefs.
5. Local publisher validates payloads and pushes only service-ready output to the server.
6. Server stores processed payloads, serves APIs, powers widgets, and responds to agent calls.

Server-side services should not need access to raw scraped data for normal operation. Raw data remains local unless there is a deliberate backup, audit, or licensed redistribution reason to move it.

## Target Product Surfaces

### 1. Market Data API

Primary buyer: developers, websites, fintech tools, dashboards, and internal products.

Initial endpoints:

- `GET /health`
- `GET /capabilities`
- `GET /runs/latest`
- `GET /documents/current`
- `GET /documents/historical/{run_id}`
- `GET /companies/{ticker}`
- `GET /quotes/{ticker}`
- `GET /ratios/{ticker}`
- `GET /events/{ticker}`
- `GET /peers/{ticker}`
- `POST /screen`

Build rule: each endpoint must return `as_of`, `processed_at`, `local_ingestion_run_id`, `quality_status`, and `data_rights_status` where applicable.

### 2. Agent Runtime API

Primary buyer: agentic AI systems and workflow automation.

Current repo routes already include:

- `GET /agents/context/{ticker}`
- `GET /agents/workflow/{ticker}`
- `GET /.well-known/agent-manifest.json`

Next agent tools:

- `get_company_profile(symbol)`
- `get_quote_snapshot(symbol)`
- `get_financials(symbol, period)`
- `get_ratios(symbol)`
- `compare_peers(symbol, metrics)`
- `screen_stocks(filters)`
- `detect_data_anomalies(symbol_or_sector)`
- `generate_company_brief(symbol, mode)`
- `generate_sector_digest(sector, date_range)`

Build rule: return JSON first, bounded prose second. Do not emit buy, sell, hold, target-price, or personalized allocation language unless the business intentionally enters the regulated advisory or research-analyst path.

### 3. Widget API

Primary buyer: finance blogs, investor communities, education sites, and partner websites.

Initial widgets:

- Stock profile card.
- Quote and freshness badge.
- Peer comparison table.
- Sector snapshot.
- Event calendar.
- Screener result block.

Build rule: widgets call public API endpoints. They should not own business logic.

### 4. Research Workspace

Primary buyer: analysts, serious retail investors, small funds, RIAs, RAs, PMS teams, and content teams.

Initial modules:

- Ticker dashboard.
- Peer and sector comparison.
- Data quality panel.
- Event and news timeline.
- AI-generated brief drafts with source citations.
- Saved watchlists and screeners.

Build rule: this can use richer UX and AI workflows, but the data contract must remain the same as the API contract.

## Development Phases

### Phase 0: Data Rights And Source Registry

Goal: know what can legally be pulled locally, transformed, uploaded, and used commercially.

Deliverables:

- `sources` registry with provider name, URL or API, rights status, allowed use, rate limits, and owner.
- Per-artifact data-rights metadata in the raw store.
- Source classification: internal, licensed, permitted-public, local-research-only, upload-allowed, commercial-serving-allowed, blocked-commercial.
- Compliance copy rules for non-advisory product language.

Exit criteria:

- No server payload is published from a source with unknown upload or commercial-serving rights.
- Every API response can trace back to source metadata.

### Phase 1: Immutable Raw Pull Layer

Goal: make the local raw pull reliable and replayable.

Deliverables:

- One ingestion runner for each source.
- Raw artifact storage under a consistent local layout.
- Hash-based duplicate detection.
- `ingestion_runs` status lifecycle: started, completed, failed, partial.
- Run logs that record source, payload path, hash, row count, and error summary.

Exit criteria:

- A failed parse never destroys the raw payload.
- A historical run can be reprocessed from raw artifacts.
- The system can show the latest successful pull per source.

### Phase 2: Local Canonical And LLM-Friendly Processing

Goal: normalize raw source data locally, then transform it into service-ready and LLM-friendly payloads.

Deliverables:

- Canonical company profile, quote, financials, ratios, events, peer, sector, and news schemas.
- Symbol mapping between NSE, BSE, broker, and third-party identifiers.
- Corporate-action and missing-field handling.
- Versioned transforms from raw artifacts to canonical rows.
- Retrieval chunks with stable IDs, short summaries, source lineage, and bounded token sizes.
- Entity-centric context packs for tickers, sectors, peers, events, and risk flags.
- Feature tables and explainable scores that are differentiated from raw provider data.
- Publishable JSON documents for server upload.

Exit criteria:

- Server APIs do not depend on raw provider field names.
- Server payloads keep lineage back to local raw artifact IDs without exposing raw payloads.
- LLM context packs fit bounded prompt and retrieval budgets.

### Phase 3: Local Publish Gate

Goal: upload only processed data that passes minimum trust, freshness, and compliance checks.

Deliverables:

- Freshness checks.
- Missing-field checks.
- Outlier and anomaly checks.
- Source disagreement checks.
- Run-level quality summary in `data_quality_audits`.
- Alerting when a critical source fails or goes stale.
- Upload manifest with payload hashes, schema versions, and record counts.
- Server publish status for accepted, rejected, and stale payloads.

Exit criteria:

- API responses expose quality status.
- Dashboard and agent surfaces can refuse or flag stale processed data.
- Server can rebuild product indexes from the latest accepted processed payloads.

### Phase 4: Server API Productization

Goal: turn processed payloads into a sellable API without requiring raw source data on the server.

Deliverables:

- Versioned API namespace.
- API keys and usage tracking.
- Rate limits by plan.
- OpenAPI documentation.
- Stable pagination, filtering, and error formats.
- Market-data profile hardened for external customers.
- Agent-runtime profile hardened for agent clients.
- Upload endpoint or sync job for local-to-server processed payload publication.

Exit criteria:

- A new developer can sign up, get a key, read docs, and call five useful endpoints.
- Usage can be measured by customer, endpoint, and source cost.

### Phase 5: Website, Widgets, And Dashboard

Goal: build distribution and monetizable UX on top of the API.

Deliverables:

- Marketing website.
- API docs.
- Widget configurator.
- Authenticated dashboard.
- Status and data freshness page.
- Customer onboarding flow.

Exit criteria:

- No frontend bypasses the API.
- Widgets can be embedded on a third-party page with a public token or signed config.

### Phase 6: Agentic AI Layer

Goal: make the platform usable by AI agents without scraping pages.

Deliverables:

- Remote MCP-compatible tool surface or equivalent agent manifest.
- Deterministic tool responses for profile, quote, ratios, events, peers, screening, and data anomalies.
- Agent workflow templates for company brief, sector digest, and portfolio x-ray.
- Strict response limits and source-citation requirements.

Exit criteria:

- An AI client can discover tools, call bounded endpoints, and produce a cited brief without raw scraping.

## Infrastructure Plan

Start simple:

- SQLite for local development and first internal deployment.
- PostgreSQL on the server when customer-facing API usage begins.
- Local filesystem raw artifacts first; optional encrypted backup later.
- Processed payload sync from local machine to server.
- Cron or systemd timer first; Airflow, Dagster, or Prefect only when source count and dependency handling justify it.
- Docker profiles for `market-data`, `agent-runtime`, and `full`.

Production baseline:

- `api.cerebralinsights.com` for REST API.
- `mcp.cerebralinsights.com` for agent runtime.
- `docs.cerebralinsights.com` for documentation.
- `widgets.cerebralinsights.com` for embeddable components.
- `status.cerebralinsights.com` for uptime and data freshness.

## First 30-Day Build Backlog

1. Add a source registry with local-pull, upload, and commercial-serving rights status.
2. Harden `daily_pipeline.py` so every source pull creates a local raw artifact before parsing.
3. Add replay command: reprocess a specific `run_id` from local raw artifacts.
4. Add canonical local tables for company profile, quote snapshot, ratios, events, peers, sectors, and news.
5. Add LLM-friendly context pack generation for ticker, sector, peer, event, and anomaly views.
6. Add processed payload export with manifest, hashes, schema version, record count, and lineage references.
7. Add server-side import for processed payloads.
8. Add route modules for company, quote, ratios, events, peers, and screeners over server-side processed data.
9. Add response metadata: `as_of`, `processed_at`, `local_ingestion_run_id`, and `quality_status`.
10. Add agent manifest entries for the first deterministic tools.
11. Add basic API key middleware before any public launch.
12. Add one embeddable stock profile widget that consumes the public API.

## Commercial Packaging

### Free / Developer Preview

- Limited calls.
- Delayed or sample data.
- Documentation and sandbox endpoints.

### Website / Widget Plan

- Public widgets.
- Data freshness badges.
- Stock cards and peer tables.
- Moderate rate limits.

### API Plan

- Full REST API.
- Higher rate limits.
- Historical run access.
- Webhooks for freshness or data quality changes.

### Agent Runtime Plan

- Agent manifest.
- Bounded deterministic tools.
- Higher-quality context payloads.
- Workflow templates.

### Enterprise / Research Desk

- Dashboard access.
- Bulk exports where data rights permit.
- Custom data quality checks.
- Private deployment or custom source integrations.

## Risk Controls

- Data rights risk: block commercial serving until source rights are known.
- SEBI risk: default to analytics, education, workflow, and infrastructure language.
- Quality risk: expose confidence and freshness instead of hiding uncertainty.
- Vendor risk: keep raw artifacts local and source-specific, but expose canonical API contracts.
- Cost risk: measure source pull cost, API call volume, and cache hit rate.
- Trust risk: retain immutable local raw copies so every derived answer can be audited.

## Definition Of Done

The platform is ready for the first external pilot when:

- Daily local raw pulls are reliable.
- Local raw artifacts are immutable and replayable.
- Canonical local tables are populated from raw artifacts.
- LLM-friendly processed payloads are generated locally.
- Server receives only processed service payloads.
- Data quality gates run before every upload and publication.
- Market-data and agent-runtime API profiles are documented.
- API responses include freshness, quality, and lineage metadata.
- Compliance-sensitive language is blocked by default.
- A website or AI agent can use the API without direct access to scraper code or raw files.
