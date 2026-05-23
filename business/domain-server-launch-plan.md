# Domain And Server Launch Plan

## Decision

Use the two domains for different jobs:

- `cerebralinsights.com`: the serious B2B and agentic AI brand.
- `knowledgeninja.com`: the education, content, and acquisition brand.

`cerebralinsights.com` should be the main commercial platform for websites, APIs, widgets, data feeds, dashboards, and AI-agent tools. `knowledgeninja.com` should be the top-of-funnel property for lessons, explainers, examples, newsletters, and lightweight public demos.

## Brand Roles

### Cerebral Insights

Positioning: Indian equities intelligence infrastructure for websites and AI agents.

Primary products:

- Indian equities API.
- MCP server for agentic AI.
- White-label website widgets.
- B2B RIA/RA/PMS dashboard.
- Data quality and anomaly feed.
- Sector and competitor intelligence dashboards.

### Knowledge Ninja

Positioning: learn and automate financial research workflows.

Primary products:

- Education product.
- Newsletter and content engine.
- Public market dashboards.
- Tutorials showing how to use the Cerebral Insights API and agent tools.
- Lead capture for developers, publishers, analysts, and finance professionals.

## Subdomain Layout

### `cerebralinsights.com`

- `www.cerebralinsights.com`: marketing site.
- `app.cerebralinsights.com`: dashboards and authenticated product UI.
- `api.cerebralinsights.com`: REST API.
- `mcp.cerebralinsights.com`: remote MCP endpoint for agentic AI clients.
- `docs.cerebralinsights.com`: API, MCP, widget, and webhook documentation.
- `widgets.cerebralinsights.com`: embeddable JavaScript widgets.
- `status.cerebralinsights.com`: uptime and incident page.

### `knowledgeninja.com`

- `www.knowledgeninja.com`: public education and content site.
- `learn.knowledgeninja.com`: courses and lessons.
- `newsletter.knowledgeninja.com`: market recaps and signup.
- `labs.knowledgeninja.com`: public demos and experiments.

## First Server Purchase

Buy one Contabo `Cloud VPS 30` first.

Current official Contabo VPS page lists this tier with:

- 8 vCPU cores.
- 24 GB RAM.
- 200 GB NVMe or 400 GB SSD.
- 3 snapshots.
- 600 Mbit/s port.
- Unlimited traffic.

Recommended location: India / Mumbai if available in the configurator. This gives the best alignment with Indian equities users and Indian website/API latency. If Mumbai has availability or pricing problems, use Singapore. If the first buyers are mostly US-based, use US East.

Operating system: Ubuntu 24.04 LTS.

Storage choice: NVMe.

Backups: enable Contabo snapshots, but also add independent off-server backups. Provider snapshots are not enough.

## When To Buy A Second Server

Buy a second smaller `Cloud VPS 20` when one of these is true:

- Paying customers depend on uptime.
- The production database is no longer comfortable sharing a machine with scrapers and workers.
- Scraping, parsing, or AI jobs disturb API latency.
- You need staging that matches production.

Second-server split:

- VPS 30: production app, API, MCP, database, Redis.
- VPS 20: staging, worker jobs, demos, monitoring, and internal tools.

Later, split Postgres onto its own machine or managed database only after real usage justifies it.

## Initial Architecture

Run everything through Docker Compose at first.

Services:

- Caddy or Traefik for HTTPS and reverse proxy.
- FastAPI backend.
- Next.js frontend.
- Postgres for normalized equities data.
- Redis for queues, caching, and rate limits.
- Worker service for scrapes, parse jobs, alerts, and scheduled reports.
- MCP server backed by the same service layer as the API.
- Widget static bundle service.
- Uptime Kuma for monitoring.
- Prometheus and Grafana after the first production users.

Do not run paid customer workloads from a laptop. The laptop remains the development and Codex automation control surface.

## Data Product Foundation

Before launching paid products:

- Normalize company, quote, financial, ratio, event, peer, sector, and source metadata tables.
- Store source URL, scrape time, parser version, and field freshness.
- Add daily data quality checks.
- Avoid raw third-party data redistribution unless licensed.
- Add a compliance copy layer that blocks buy, sell, hold, target price, guaranteed return, and personalized allocation language.

## Website And Agentic AI Deliverables

### Website Users

First deliverables:

- Stock profile card widget.
- Peer comparison table widget.
- Sector snapshot page.
- Data freshness badge.
- Publisher-ready newsletter block.

### Agentic AI Users

First deliverables:

- OpenAPI spec.
- `/llms.txt`.
- MCP tools for company profile, quote snapshot, ratios, peers, events, stock screening, anomaly detection, and sector digest generation.
- JSON Schema validation for all tool inputs and outputs.

## DNS And Infrastructure Checklist

1. Put both domains behind Cloudflare DNS.
2. Create DNS records for `www`, `app`, `api`, `mcp`, `docs`, `widgets`, and `status`.
3. Point production records to the Contabo server IP.
4. Enable HTTPS through Caddy or Traefik.
5. Set up transactional email with a dedicated provider before sending alerts or newsletters.
6. Add SPF, DKIM, and DMARC records.
7. Add daily database backup to off-server object storage.
8. Add uptime checks for website, API, MCP, and worker heartbeat.
9. Add API keys and rate limits before inviting external users.
10. Add terms, privacy policy, disclaimer, and data-source notes before public launch.

## Launch Order

### Week 1: Infrastructure

- Buy VPS 30.
- Configure server, firewall, Docker, reverse proxy, backups, and monitoring.
- Point staging subdomains first.

### Week 2: Core API

- Build normalized database schema.
- Load a small clean universe of liquid stocks.
- Ship company, quote, ratios, events, peers, and sector endpoints.

### Week 3: Agent Surface

- Publish OpenAPI docs.
- Add `/llms.txt`.
- Ship the first MCP server.
- Add bounded JSON responses and source timestamps.

### Week 4: Website Surface

- Ship stock card widget.
- Ship peer table widget.
- Ship sector dashboard demo.
- Build docs and examples for publishers.

### Weeks 5-8: First Revenue Wedges

- Sell API beta access.
- Sell widget pilots to publishers.
- Sell anomaly feed to data-heavy users.
- Sell B2B dashboard pilots to registered professionals.
- Use Knowledge Ninja to publish education and drive leads.

## First Offer To Sell

The first offer should be:

> Cerebral Insights gives websites and AI agents clean Indian equities intelligence through APIs, widgets, and agent tools.

Avoid selling:

- Stock tips.
- Personalized recommendations.
- Target prices.
- Model portfolios.

Those can come later only if the legal and SEBI structure is deliberate.

## Sources Checked

- Contabo VPS official options: `https://contabo.com/en/vps/`
- Contabo locations: `https://contabo.com/en/locations/`
