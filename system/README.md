# Daily Market Intelligence System

This folder adds a production-oriented scaffold on top of your existing scraping scripts so you can:

1. Run scrapers on a daily schedule.
2. Persist all fetched artifacts in a queryable database.
3. Track data-quality and freshness checks per run.
4. Serve current + historical data to AI agents through a DBaaS API.
5. Generate analyst-report payloads with recommendation timeline requirements.

## Components

- `schema.sql`: relational schema for ingestion runs, raw artifacts, quality audits, risk signals, and generated reports.
- `daily_pipeline.py`: orchestration runner that executes existing scraper scripts and ingests CSV/JSON outputs with hashes.
- `service_api.py`: FastAPI DBaaS service for latest/historical retrieval and AI-agent context payloads.
- `analyst_report.py`: query utility that builds model-ready prompt payloads for detailed investment analysis.
- `Dockerfile` + `docker-compose.yml`: one-command container deployment for the DBaaS service.

## DBaaS endpoints for AI agents

- `GET /health`: liveness probe.
- `GET /runs/latest`: latest successful run and data-quality audit results.
- `GET /documents/current`: current document set with integrity metadata.
- `GET /documents/historical/{run_id}`: historical snapshot lookup by run id.
- `GET /agents/context/{ticker}`: LLM-ready payload (documents + risk signals + quality checks).

## Quick start (local)

```bash
python system/daily_pipeline.py --repo-root . --db-path ./system/market_intel.db
python system/service_api.py --db-path ./system/market_intel.db --host 0.0.0.0 --port 8000
```

## Quick start (Docker)

```bash
cd system
docker compose up --build -d
```

Service URL: `http://localhost:8000`

## Reliability features added for accurate market data

- **Historical immutability by run:** every ingestion creates a run-scoped snapshot in `raw_documents`.
- **Integrity proof:** each document stores `content_sha256` to verify unchanged payloads.
- **Freshness metadata:** source file timestamp + ingestion timestamp stored per document.
- **Coverage checks:** each configured source is validated and logged in `data_quality_audits`.
- **Agent-facing trust signals:** quality checks are included directly in `/runs/latest` and `/agents/context/{ticker}`.

## Suggested deployment architecture

- **Ingestion layer:** cron/Airflow runs `daily_pipeline.py` once daily.
- **Storage layer:** SQLite for edge/single-node deployments, move to PostgreSQL for horizontal scale.
- **Validation layer:** monitor `data_quality_audits`; alert when `status=fail`.
- **Reasoning layer:** AI agents consume `/agents/context/{ticker}` and include quality flags in decisions.
- **Serving layer:** persist generated markdown report + recommendation into `analyst_reports`.

## Next steps

1. Add dedicated parsers to normalize prices/news/fundamentals into first-class tables.
2. Add exchange-level reconciliation checks (NSE/BSE cross-source mismatch audit).
3. Move to PostgreSQL + read replicas for multi-agent concurrent querying.
4. Add signed run manifests and retention policies for long-term auditability.
