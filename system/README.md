# Daily Market Intelligence System

This folder adds a production-oriented scaffold on top of your existing scraping scripts so you can:

1. Run scrapers on a daily schedule.
2. Persist all fetched artifacts in a queryable database.
3. Track data-quality and freshness checks per run.
4. Serve current + historical data to AI agents through composable APIs.
5. Generate analyst-report payloads with recommendation timeline requirements.
6. Publish different API surfaces from one repository for product, internal, and agent consumers.

## Components

- `schema.sql`: relational schema for ingestion runs, raw artifacts, quality audits, risk signals, and generated reports.
- `daily_pipeline.py`: orchestration runner that executes existing scraper scripts and ingests CSV/JSON outputs with hashes.
- `service_api.py`: CLI entrypoint for launching one of several FastAPI profiles.
- `api/factory.py`: app factory + profile registry for `full`, `market-data`, and `agent-runtime` APIs.
- `api/routers/`: modular routers that can be recombined into multiple APIs.
- `analyst_report.py`: query utility that builds model-ready prompt payloads for detailed investment analysis.
- `Dockerfile` + `docker-compose.yml`: container deployment that can switch profiles using `API_PROFILE`.
- `Makefile`: shortcuts for ingestion and running profile-specific APIs.

## API profiles

### 1. `full`
A unified deployment for teams that want market data, retrieval, and agent tooling in one API.

### 2. `market-data`
A smaller surface focused on:

- `GET /health`
- `GET /capabilities`
- `GET /runs/latest`
- `GET /documents/current`
- `GET /documents/historical/{run_id}`

### 3. `agent-runtime`
An AI-native surface focused on orchestration and tool discovery:

- `GET /health`
- `GET /capabilities`
- `GET /runs/latest`
- `GET /agents/context/{ticker}`
- `GET /agents/workflow/{ticker}`
- `GET /.well-known/agent-manifest.json`

## Why this is more ready for agentic AI

- **Composable APIs:** you can expose a narrow, safer API for one consumer and a richer agent API for another.
- **Tool discovery:** the agent manifest and OpenAPI document make it easier for agents to auto-discover tools.
- **Workflow guidance:** `/agents/workflow/{ticker}` tells downstream agents which calls to make and in what order.
- **Trust-aware reasoning:** quality checks remain first-class payloads so agents can reason about data reliability.

## Quick start (local)

```bash
make api-profiles
python system/daily_pipeline.py --repo-root . --db-path ./system/market_intel.db
python system/service_api.py --db-path ./system/market_intel.db --profile full --host 0.0.0.0 --port 8000
python system/service_api.py --db-path ./system/market_intel.db --profile market-data --host 0.0.0.0 --port 8001
python system/service_api.py --db-path ./system/market_intel.db --profile agent-runtime --host 0.0.0.0 --port 8002
```

## Quick start (Docker)

```bash
cd system
API_PROFILE=agent-runtime docker compose up --build -d
```

Service URL example: `http://localhost:8000`

## Suggested deployment architecture

- **Ingestion layer:** cron/Airflow runs `daily_pipeline.py` once daily.
- **Storage layer:** SQLite for edge/single-node deployments, move to PostgreSQL for horizontal scale.
- **API layer:** deploy profile-specific services behind separate domains or paths.
- **Validation layer:** monitor `data_quality_audits`; alert when `status=fail`.
- **Reasoning layer:** AI agents consume `/agents/context/{ticker}`, the agent manifest, and workflow plans.
- **Serving layer:** persist generated markdown report + recommendation into `analyst_reports`.

## Recommended next steps

1. Add first-class normalized tables for OHLCV, fundamentals, and entity relationships.
2. Add authentication/rate limiting before exposing public agent endpoints.
3. Move to PostgreSQL + read replicas for multi-agent concurrent querying.
4. Add signed run manifests and retention policies for long-term auditability.
5. Publish an MCP server or function-calling wrapper on top of the `agent-runtime` profile.
