# Website And Agentic AI Business Plan

## Goal

Build Indian equities products whose primary users are:

1. Websites that embed, license, or consume market intelligence.
2. Agentic AI systems that call tools, APIs, and structured data endpoints.

The right strategy is not to build 13 separate companies. Build one shared Indian equities intelligence platform, then package it into multiple website-facing and agent-facing products.

## Core Platform

### Data Layer

- Company master and identifiers.
- Quote snapshots and historical prices.
- Financial statements, ratios, scorecards, and sector tags.
- Events, announcements, corporate actions, and peer groups.
- Snapshot history, source metadata, freshness checks, and parse logs.

### Product Layer

- REST API with OpenAPI documentation.
- MCP server for agentic AI tools.
- Website widgets and embeddable charts.
- Dashboard app for human users.
- Webhooks for alerts, events, and anomaly feeds.
- Agent-readable documentation through `/llms.txt` and markdown docs.

### Trust Layer

- Licensed or permitted data sources before commercial launch.
- Field-level timestamps and source references.
- Compliance-safe language by default.
- Clear separation between analytics, education, research workflow, and regulated advice.

## Agentic AI Product Surface

AI agents should be able to call narrow, deterministic tools instead of scraping pages.

Initial MCP and API tools:

- `get_company_profile(symbol)`
- `get_quote_snapshot(symbol)`
- `get_financials(symbol, period)`
- `get_ratios(symbol)`
- `get_events(symbol, event_type, date_range)`
- `compare_peers(symbol, metrics)`
- `screen_stocks(filters)`
- `get_sector_snapshot(sector)`
- `detect_data_anomalies(symbol_or_sector)`
- `generate_company_brief(symbol, mode)`
- `generate_sector_digest(sector, date_range)`
- `analyze_portfolio(holdings)`
- `run_factor_backtest(strategy_config)`
- `create_widget_config(symbols, widget_type)`

Agent output rules:

- Return JSON first, prose second.
- Include source timestamps and confidence flags.
- Avoid buy, sell, hold, target price, or personalized allocation language unless the product is operated under the right registration.
- Keep tool responses bounded and paginated so agents do not ingest huge payloads.

## Website Product Surface

Websites should be able to embed or call:

- Stock profile cards.
- Peer comparison tables.
- Sector dashboards.
- Event calendars.
- Financial chart widgets.
- Screener widgets.
- Newsletter data blocks.
- API-backed company pages.
- Data freshness badges.

Website buyer examples:

- Finance blogs.
- Investor communities.
- Education sites.
- Fintech products.
- Broker partner sites.
- RIA, RA, and PMS firm websites.
- Corporate strategy and competitor-intelligence portals.

## Business Model Packaging

| ID | Business | Website Product | Agentic AI Product | First Commercial MVP |
|---|---|---|---|---|
| 01 | Stock screener SaaS | Hosted screener and embeddable screener widget | `screen_stocks` tool | 20 filters over 500 liquid stocks |
| 02 | Watchlist alert app | User dashboard and alert settings page | `monitor_watchlist` workflow | Daily email/webhook alerts |
| 03 | Indian equities API | Developer portal and docs | OpenAPI plus MCP tools | Company, quote, ratios, events, peers endpoints |
| 04 | AI research workspace | Research dashboard | `generate_company_brief` and `compare_peers` | Cited company brief generator |
| 05 | Portfolio X-ray tool | Upload portal and report page | `analyze_portfolio` tool | CSV upload diagnostic report |
| 06 | Newsletter/content engine | Publisher dashboard | `generate_sector_digest` tool | Weekly sector recap generator |
| 07 | Backtesting platform | Strategy builder UI | `run_factor_backtest` tool | Monthly rebalance factor test |
| 08 | B2B RIA/RA/PMS dashboard | Registered-professional workspace | Research workflow tools | Watchlist and report export dashboard |
| 09 | Data quality/anomaly feed | Status dashboard and webhook config | `detect_data_anomalies` tool | Daily anomaly feed |
| 10 | Education product | Course site and interactive examples | Tutor agent tools | 10 lessons using real company data |
| 11 | Competitor/sector intelligence | Sector intelligence portal | `get_sector_snapshot` and peer tools | Three sector dashboards |
| 12 | White-label widgets | Widget builder and embed docs | `create_widget_config` tool | Stock card and peer table embeds |
| 13 | Lead-gen marketplace | Verified professional directory | Matching assistant with strict guardrails | Manual matching directory |

## Build Sequence

### Phase 0: Data And Legal Gate

- Inventory every data field and source.
- Decide what can be used commercially.
- Replace scrape-only dependencies with licensed, permitted, or public-source data where needed.
- Define prohibited wording for non-registered products.

### Phase 1: Canonical Data Platform

- Normalize company, quote, financial, ratio, event, peer, and sector tables.
- Add snapshot history.
- Add source timestamps and parse version.
- Add daily data quality checks.

### Phase 2: API And Agent Foundation

- Build REST API.
- Publish OpenAPI spec.
- Build MCP server over the same service layer.
- Add `/llms.txt`, markdown docs, examples, and tool schemas.
- Add API keys, rate limits, usage logs, and response pagination.

### Phase 3: Website Distribution

- Build embeddable widgets.
- Build developer portal.
- Build demo company pages.
- Build publisher-ready newsletter blocks.
- Ship white-label documentation.

### Phase 4: First Revenue Wedges

Prioritize these first:

1. Indian equities API.
2. White-label widgets.
3. Data quality/anomaly feed.
4. B2B RIA/RA/PMS dashboard.
5. Newsletter/content engine.

These have cleaner website and agentic AI buyers than broad retail stock advice.

### Phase 5: Advanced Products

- Screener SaaS.
- AI research workspace.
- Backtesting platform.
- Portfolio X-ray.
- Competitor and sector intelligence.
- Education product.

### Phase 6: Regulated Extensions

Only after legal setup:

- Paid research reports.
- Model portfolios.
- Personalized portfolio advice.
- Lead-gen involving regulated professionals.

## Codex Automation Operating Model

Codex automations can run the operating backbone:

- Daily data ingestion smoke tests.
- Field freshness checks.
- Parser drift detection.
- API contract tests.
- MCP tool tests.
- Widget screenshot checks.
- Newsletter draft generation.
- Anomaly report generation.
- Documentation updates from OpenAPI schemas.
- Weekly business metrics summary.

Production customer workloads should eventually move off the laptop to a hosted server, but the laptop can run the early development and validation loop.

## Success Metrics

- API: active keys, requests per day, retained developers.
- Widgets: embedded sites, pageviews, load success rate.
- Agents: tool calls, successful task completion, low invalid-response rate.
- Dashboards: weekly active users, saved screens, exported reports.
- Content: publisher retention, open rate, click rate.
- Data quality: freshness SLA, parse error rate, anomaly precision.

## Standards To Support

- OpenAPI for HTTP API discovery by humans and machines.
- MCP for agent tool access.
- `/llms.txt` and markdown documentation for agent-readable website context.
- JSON Schema for input and output validation.
- Webhooks for website and agent workflows.

## Strategic Rule

Do not sell this as 13 disconnected products. Sell one trusted Indian equities intelligence layer through many packaging channels: API, MCP, widgets, dashboards, reports, and education.
