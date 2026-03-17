# Daily Market Intelligence System

This folder adds a production-oriented scaffold on top of your existing scraping scripts so you can:

1. Run scrapers on a daily schedule.
2. Persist all fetched artifacts in a queryable database.
3. Prepare structured context for an LLM econometric analyst.
4. Generate analyst-report payloads with recommendation timeline requirements.

## Components

- `schema.sql`: core relational schema for ingestion runs, raw artifacts, extracted risk signals, and generated reports.
- `daily_pipeline.py`: orchestration runner that executes existing scraper scripts and ingests CSV/JSON outputs.
- `analyst_report.py`: query utility that builds model-ready prompt payloads for detailed investment analysis.

## Suggested deployment architecture

- **Ingestion layer:** cron/Airflow runs `daily_pipeline.py` once daily.
- **Storage layer:** SQLite for local prototyping (or migrate schema to PostgreSQL for scale).
- **Enrichment layer:** additional NLP job extracts entity/region/risk signals into `entity_risk_signals`.
- **Reasoning layer:** your LLM runtime consumes payload from `analyst_report.py`.
- **Serving layer:** save generated markdown report + recommendation into `analyst_reports`.

## Quick start

```bash
python system/daily_pipeline.py --repo-root . --db-path ./system/market_intel.db
python system/analyst_report.py --db-path ./system/market_intel.db --ticker RELIANCE
```

## How this maps to your objective

- Reacts to global/regional news: ingest all article/data outputs daily and enrich with explicit risk signals.
- Assesses business risk and growth: LLM prompt enforces macro, regional, dependency, and scenario analysis.
- Performs fundamental analysis: report template includes valuation and business fundamentals sections.
- Gives investment recommendation and timeline: output format requires Buy/Hold/Sell + horizon + confidence.

## Next steps

1. Add a dedicated parser that maps each scraper output into normalized tables (`prices`, `news_items`, `financials`).
2. Add a knowledge-graph stage for supplier/customer/country dependency relationships.
3. Add guardrails/evaluation: backtest recommendation quality and consistency metrics.
4. Swap SQLite with PostgreSQL + pgvector if you need semantic retrieval at larger scale.
