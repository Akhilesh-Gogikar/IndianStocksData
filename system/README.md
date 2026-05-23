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
- `webhook_worker.py`: one-shot delivery worker for pending alert webhook outbox events.
- `api/factory.py`: app factory + profile registry for `full`, `market-data`, and `agent-runtime` APIs.
- `api/routers/`: modular routers that can be recombined into multiple APIs.
- `ai/vector_index.py`: local document embeddings plus optional TurboVec compressed vector search for AI retrieval.
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
- `GET /freshness`
- `GET /freshness/{ticker}`
- `GET /companies/{ticker}`
- `GET /quotes/{ticker}`
- `GET /ratios/{ticker}`
- `GET /events/{ticker}`
- `GET /peers/{ticker}`
- `POST /screen`
- `GET /screeners`
- `POST /screeners`
- `POST /screeners/{screener_id}/evaluate`
- `GET /watchlists`
- `POST /watchlists`
- `POST /watchlists/{watchlist_id}/items`
- `POST /watchlists/{watchlist_id}/alerts`
- `GET /watchlists/alerts/readiness`
- `GET /watchlists/alerts/readiness/actions`
- `POST /watchlists/alerts/readiness/action-queue`
- `GET /watchlists/{watchlist_id}/alerts`
- `PATCH /watchlists/{watchlist_id}/alerts/{rule_id}`
- `DELETE /watchlists/{watchlist_id}/alerts/{rule_id}`
- `GET /watchlists/{watchlist_id}/alerts/evaluate`
- `GET /watchlists/{watchlist_id}/alerts/readiness`
- `GET /watchlists/{watchlist_id}/alerts/events`
- `GET /watchlists/{watchlist_id}/alerts/events/summary`
- `PATCH /watchlists/{watchlist_id}/alerts/events/bulk`
- `PATCH /watchlists/{watchlist_id}/alerts/events/{event_id}`
- `GET /watchlists/{watchlist_id}/alerts/events/{event_id}/reviews`
- `GET /watchlists/{watchlist_id}/alerts/reviews`
- `GET /watchlists/{watchlist_id}/alerts/{rule_id}/events`
- `GET /watchlists/{watchlist_id}/alerts/{rule_id}/events/summary`
- `GET /watchlists/webhooks/subscriptions`
- `POST /watchlists/webhooks/subscriptions`
- `PATCH /watchlists/webhooks/subscriptions/{subscription_id}`
- `DELETE /watchlists/webhooks/subscriptions/{subscription_id}`
- `POST /watchlists/webhooks/subscriptions/{subscription_id}/test`
- `GET /watchlists/webhooks/status`
- `GET /watchlists/webhooks/outbox`
- `POST /watchlists/webhooks/outbox/{outbox_id}/replay`
- `GET /watchlists/webhooks/deliveries`
- `GET /portfolios`
- `POST /portfolios`
- `POST /portfolios/{portfolio_id}/holdings`
- `GET /portfolios/{portfolio_id}/xray`
- `GET /documents/current`
- `GET /documents/historical/{run_id}`

Alert event history endpoints accept `review_status=open|reviewed|dismissed` for advisor queues and dashboards. Alert rule listing accepts `include_review_counts=true` to attach per-rule review summaries and `needs_attention=true` to return only rules with open alert events.
Alert evaluation responses include `available`, `evaluatable`, `data_status`, `skip_reason`, freshness fields, quality/data-rights fields, `available_metric_count`, `missing_metric_count`, `stale_metric_count`, `quality_blocked_metric_count`, and `data_rights_blocked_metric_count` so missing, stale, failed-quality, or rights-blocked local data is visible instead of silently reading as no alert.
Alert readiness summarizes the same checks without recording events and returns `status=ready|needs_data`, blocked evaluations, missing counts, stale counts, quality blocks, and data-rights blocks by metric.
Owner-level alert readiness aggregates those checks across every watchlist for an owner and accepts `status=ready|needs_data` filtering for dashboard queues.
Owner-level alert readiness actions turn blocked readiness rows into prioritized remediation actions for refresh, ingestion, quality, and data-rights workflows.
Owner-level alert readiness action queues persist those remediation actions into the saved advisor action queue tables for assignment, progress tracking, and completion updates; by default repeated saves replace the current open `alert_readiness` queue instead of creating duplicates, and a no-blocker save closes the existing queue as completed.
Bulk alert review accepts either explicit `event_ids` or filters such as `current_status=open` and `rule_id`.

### 3. `agent-runtime`
An AI-native surface focused on orchestration and tool discovery:

- `GET /health`
- `GET /capabilities`
- `GET /runs/latest`
- `GET /freshness/{ticker}`
- `GET /agents/context/{ticker}`
- `GET /agents/workflow/{ticker}`
- `GET /agents/advisor-workbench`
- `POST /agents/advisor-outreach-draft`
- `GET /agents/advisor-outreach-drafts`
- `GET /agents/advisor-outreach-drafts/{draft_id}`
- `PATCH /agents/advisor-outreach-drafts/{draft_id}`
- `POST /agents/advisor-outreach-drafts/{draft_id}/compliance-review`
- `GET /agents/advisor-outreach-drafts/{draft_id}/compliance-reviews`
- `POST /agents/advisor-outreach-drafts/{draft_id}/delivery-packet`
- `GET /agents/advisor-outreach-delivery-dashboard`
- `GET /agents/customer-intent-dashboard`
- `GET /agents/customer-engagement-timeline`
- `GET /agents/customer-engagement-brief`
- `GET /agents/customer-engagement-cadence-review`
- `GET /agents/customer-engagement-cadence-dashboard`
- `GET /agents/customer-engagement-action-queue`
- `GET /agents/customer-engagement-task-brief`
- `GET /agents/ai-recommendation-effectiveness-dashboard`
- `GET /agents/ai-improvement-backlog`
- `GET /agents/ai-improvement-experiment-plan`
- `GET /agents/ai-improvement-experiment-launch-packet`
- `GET /agents/ai-improvement-experiment-readout`
- `GET /agents/ai-improvement-rollout-readiness`
- `GET /agents/ai-improvement-rollout-monitor`
- `GET /agents/ai-improvement-release-packet`
- `GET /agents/ai-improvement-adoption-playbook`
- `GET /agents/ai-improvement-adoption-monitor`
- `GET /agents/ai-improvement-adoption-impact-ledger`
- `GET /agents/ai-improvement-scale-decision-packet`
- `GET /agents/ai-improvement-scale-execution-plan`
- `GET /agents/ai-improvement-scale-execution-monitor`
- `GET /agents/ai-improvement-scale-learning-report`
- `GET /agents/ai-improvement-roadmap-refresh`
- `GET /agents/ai-improvement-backlog-handoff`
- `GET /agents/ai-improvement-implementation-kickoff-packet`
- `GET /agents/ai-improvement-implementation-readiness-monitor`
- `GET /agents/ai-improvement-implementation-blocker-resolution-plan`
- `GET /agents/ai-improvement-implementation-unblock-verification-report`
- `GET /agents/ai-improvement-implementation-qa-review-packet`
- `GET /agents/ai-improvement-implementation-qa-signoff-report`
- `GET /agents/ai-improvement-launch-review-packet`
- `GET /agents/ai-improvement-launch-execution-plan`
- `GET /agents/ai-improvement-launch-execution-monitor`
- `GET /agents/ai-improvement-launch-outcome-monitor`
- `GET /agents/ai-improvement-launch-value-proof-packet`
- `GET /agents/ai-improvement-launch-customer-communication-packet`
- `GET /agents/ai-improvement-launch-customer-communication-review-packet`
- `GET /agents/ai-improvement-launch-customer-communication-delivery-packet`
- `GET /agents/ai-improvement-launch-customer-communication-delivery-monitor`
- `GET /agents/ai-improvement-launch-customer-communication-delivery-unblock-plan`
- `GET /agents/ai-improvement-launch-customer-communication-delivery-unblock-verification-report`
- `GET /agents/ai-improvement-launch-customer-communication-delivery-send-authorization-packet`
- `GET /agents/ai-improvement-launch-customer-communication-delivery-send-authorization-monitor`
- `GET /agents/ai-improvement-launch-customer-communication-delivery-send-authorization-unblock-plan`
- `GET /agents/ai-improvement-launch-customer-communication-delivery-send-authorization-unblock-verification-report`
- `GET /agents/ai-improvement-launch-customer-communication-delivery-send-readiness-packet`
- `GET /agents/ai-improvement-launch-customer-communication-delivery-send-readiness-review-packet`
- `GET /agents/ai-improvement-launch-customer-communication-delivery-send-execution-handoff-packet`
- `GET /agents/customer-intent-action-plan`
- `GET /agents/customer-intent-followup-packet`
- `GET /agents/customer-intent-followup-review`
- `POST /agents/customer-intent-followup-draft`
- `GET /agents/advisor-outreach-deliveries`
- `GET /agents/advisor-outreach-deliveries/{delivery_id}`
- `PATCH /agents/advisor-outreach-deliveries/{delivery_id}`
- `POST /agents/advisor-outreach-deliveries/{delivery_id}/outcome`
- `GET /agents/advisor-outreach-outcomes`
- `GET /agents/advisor-outreach-outcomes/{outcome_id}`
- `POST /agents/action-queue`
- `GET /agents/action-queues/summary`
- `GET /agents/action-queues/tasks`
- `GET /agents/action-queues/tasks/workload`
- `GET /agents/action-queues/tasks/escalations/summary`
- `GET /agents/action-queues/tasks/escalations/inbox`
- `GET /agents/action-queues/tasks/escalations/inbox/notification`
- `POST /agents/action-queues/tasks/escalations/inbox/notification`
- `GET /agents/action-queues/tasks/escalations/inbox/notification/logs`
- `GET /agents/action-queues/tasks/escalations/inbox/notification/summary`
- `GET /agents/action-queues/tasks/escalations/inbox/notification/delivery-control-summary`
- `GET /agents/action-queues/tasks/escalations/inbox/notification/delivery-queue`
- `GET /agents/action-queues/tasks/escalations/inbox/notification/delivery-incident-summary`
- `GET /agents/action-queues/tasks/escalations/inbox/notification/delivery-incident-workload`
- `GET /agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents` (use `assigned_to`, `follow_up_status`, or `include_suppressed=true` for incident drill-down)
- `GET /agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents/{notification_id}` (hydrates audit context, chronological incident timeline, and operator next actions)
- `POST /agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents/{notification_id}/actions` (executes validated next actions and returns post-action incident state)
- `GET /agents/action-queues/tasks/escalations/inbox/notification/delivery-incident-reviews`
- `POST /agents/action-queues/tasks/escalations/inbox/notification/delivery-incident-reviews`
- `POST /agents/action-queues/tasks/escalations/inbox/notification/delivery-incidents/{notification_id}/review`
- `GET /agents/action-queues/tasks/escalations/inbox/notification/delivery-claims`
- `GET /agents/action-queues/tasks/escalations/inbox/notification/deadletters`
- `POST /agents/action-queues/tasks/escalations/inbox/notification/deadletters/{notification_id}/requeue`
- `GET /agents/action-queues/tasks/escalations/inbox/notification/deadletters/remediations`
- `GET /agents/action-queues/tasks/escalations/inbox/notification/delivery-attempts`
- `GET /agents/action-queues/tasks/escalations/inbox/notification/delivery-claim-releases`
- `POST /agents/action-queues/tasks/escalations/inbox/notification/delivery-claim`
- `POST /agents/action-queues/tasks/escalations/inbox/notification/delivery-claim/{notification_id}/renew`
- `POST /agents/action-queues/tasks/escalations/inbox/notification/delivery-claim/{notification_id}/release`
- `POST /agents/action-queues/tasks/escalations/inbox/notification/delivery-claim/{notification_id}/complete`
- `PATCH /agents/action-queues/tasks/escalations/inbox/notification/logs/{notification_id}`
- `GET /agents/action-queues/tasks/escalations`
- `GET /agents/action-queues/tasks/escalations/reviews`
- `POST /agents/action-queues/tasks/escalations/reviews`
- `PATCH /agents/action-queues/tasks`
- `GET /agents/action-queues/tasks/activity`
- `GET /agents/action-queues`
- `GET /agents/action-queues/{queue_id}`
- `PATCH /agents/action-queues/{queue_id}/tasks/{task_id}`
- `POST /agents/action-queues/{queue_id}/tasks/{task_id}/escalation-review`
- `POST /agents/advisor-followup`
- `POST /agents/morning-brief`
- `POST /agents/research-brief/{ticker}`
- `POST /agents/screener-copilot`
- `POST /agents/screener-digest/{screener_id}`
- `POST /agents/portfolio-digest/{portfolio_id}`
- `POST /agents/watchlist-digest/{watchlist_id}`
- `GET /vectors/status`
- `POST /vectors/rebuild`
- `GET /vectors/search?query=...`
- `GET /.well-known/agent-manifest.json`

## Why this is more ready for agentic AI

- **Composable APIs:** you can expose a narrow, safer API for one consumer and a richer agent API for another.
- **Tool discovery:** the agent manifest and OpenAPI document make it easier for agents to auto-discover tools.
- **Workflow guidance:** `/agents/workflow/{ticker}` tells downstream agents which calls to make and in what order.
- **Local semantic retrieval:** `/vectors/*` builds deterministic local embeddings and uses TurboVec compressed search when installed.
- **Grounded research briefs:** `/agents/research-brief/{ticker}` turns local vector evidence into cited AI-ready insight.
- **AI morning briefs:** `/agents/morning-brief` prioritizes what a customer should review today across freshness, portfolios, watchlists, and screeners.
- **Advisor follow-up packs:** `/agents/advisor-followup` turns a morning brief into customer email copy, meeting agenda, and compliance guardrails.
- **Advisor action queues:** `/agents/action-queue` turns follow-up packs into trackable tasks with urgency, blockers, and completion criteria.
- **Durable queue tracking:** `/agents/action-queues/*` preserves task status, notes, assignee, due date, manager escalation reviews, idempotent notification logs, notification delivery status transitions, delivery health summaries, retry-ready notification delivery queues, delivery incident summaries, delivery incident ownership workload, assignee and follow-up drill-down filters, incident detail hydration, single and bulk delivery-incident triage, dead-letter triage queues, atomic delivery claims, claim-token-validated completions, immutable delivery-attempt audit trails, retry backoff scheduling, and max-attempt exhaustion so customer follow-up work can continue across sessions, with `/agents/action-queues/summary` and `/agents/action-queues/tasks` providing dashboard-ready rollups, workload summaries, book-level escalation dashboards, filtered snooze-aware escalation inboxes, compact manager notification payloads, escalation feeds, task feeds, bulk manager review, bulk triage, and task-update audit trails without full queue hydration.
- **Advisor workbench:** `/agents/advisor-workbench` ranks saved queue tasks into the next best customer follow-up actions.
- **Advisor outreach drafts:** `/agents/advisor-outreach-draft` turns the top saved task into reviewable customer email, agenda, and compliance copy.
- **Outreach review workflow:** `/agents/advisor-outreach-drafts/*` preserves approval status, reviewer notes, and an audit path before customer delivery.
- **Outreach compliance reviews:** `/agents/advisor-outreach-drafts/*/compliance-review` flags risky phrases, missing disclosures, and blocked source tasks before approval.
- **Compliance-gated approval:** draft approval is blocked when compliance review fails unless a reviewer supplies an explicit override.
- **Compliance review audit trails:** `/agents/advisor-outreach-drafts/*/compliance-reviews` preserves saved review results for each outreach approval attempt.
- **Approved delivery packets:** `/agents/advisor-outreach-drafts/*/delivery-packet` exposes customer-ready outreach only after approval and a fresh passing compliance review.
- **Outreach delivery records:** `/agents/advisor-outreach-deliveries/*` tracks prepared and delivered customer-ready packets with compliance evidence.
- **Outreach delivery dashboard:** `/agents/advisor-outreach-delivery-dashboard` summarizes ready, stale, delivered, voided, and missing delivery packets.
- **Outreach outcome capture:** `/agents/advisor-outreach-deliveries/*/outcome` records customer responses and recommends the next follow-up action.
- **Customer intent dashboard:** `/agents/customer-intent-dashboard` ranks owners by local outreach outcomes and pending next actions.
- **Customer engagement timeline:** `/agents/customer-engagement-timeline` consolidates outreach, review, delivery, and outcome history for context-aware action.
- **Customer engagement brief:** `/agents/customer-engagement-brief` compresses timeline context into current intent, talking points, avoid-lists, and evidence references.
- **Customer engagement cadence review:** `/agents/customer-engagement-cadence-review` decides whether customer contact is appropriate now and which route should run next.
- **Customer engagement cadence dashboard:** `/agents/customer-engagement-cadence-dashboard` ranks owners by contact readiness across the advisor book.
- **Customer engagement action queue:** `/agents/customer-engagement-action-queue` converts cadence readiness into executable advisor tasks.
- **Customer engagement task brief:** `/agents/customer-engagement-task-brief` turns queued work into execution-ready talk tracks, proof points, guardrails, and completion criteria.
- **AI recommendation effectiveness dashboard:** `/agents/ai-recommendation-effectiveness-dashboard` measures which guided outreach actions convert into customer outcomes and positive signals.
- **AI improvement backlog:** `/agents/ai-improvement-backlog` turns outcome evidence into ranked AI improvement work with success metrics and guardrails.
- **AI improvement experiment plan:** `/agents/ai-improvement-experiment-plan` converts the top AI improvement into a measurable hypothesis, treatment, metrics, and stop conditions.
- **AI improvement experiment launch packet:** `/agents/ai-improvement-experiment-launch-packet` turns an experiment plan into launch readiness, cohort rules, data capture, and rollback criteria.
- **AI improvement experiment readout:** `/agents/ai-improvement-experiment-readout` turns launch evidence into continue, ship, rollback, or collect-more decisions.
- **AI improvement rollout readiness:** `/agents/ai-improvement-rollout-readiness` turns a readout into release gates, rollout phases, monitoring, and rollback triggers.
- **AI improvement rollout monitor:** `/agents/ai-improvement-rollout-monitor` surfaces rollout status, alerts, tracked metrics, rollback risk, and immediate next action.
- **AI improvement release packet:** `/agents/ai-improvement-release-packet` turns rollout monitor state into advisor enablement, support talking points, risks, and rollback guidance.
- **AI improvement adoption playbook:** `/agents/ai-improvement-adoption-playbook` turns a release packet into advisor tasks, training checks, customer language, blockers, and success signals.
- **AI improvement adoption monitor:** `/agents/ai-improvement-adoption-monitor` tracks advisor readiness, training status, language safety, blockers, success signals, and immediate next action.
- **AI improvement adoption impact ledger:** `/agents/ai-improvement-adoption-impact-ledger` proves customer value by tying adoption to outcomes, advisor usage, blocked accounts, proof points, and scale decisions.
- **AI improvement scale decision packet:** `/agents/ai-improvement-scale-decision-packet` converts measured impact into scale, pilot, hold, or evidence-collection decisions with customer proof and advisor-change guidance.
- **AI improvement scale execution plan:** `/agents/ai-improvement-scale-execution-plan` turns scale decisions into accountable rollout tasks, guardrails, proof checks, acceptance criteria, escalation, and next action.
- **AI improvement scale execution monitor:** `/agents/ai-improvement-scale-execution-monitor` tracks execution progress, guardrails, proof checks, acceptance gaps, blockers, risk, and immediate owner action.
- **AI improvement scale learning report:** `/agents/ai-improvement-scale-learning-report` turns execution monitoring into validated learnings, open questions, feedback actions, roadmap updates, and next improvement candidates.
- **AI improvement roadmap refresh:** `/agents/ai-improvement-roadmap-refresh` turns scale learnings into backlog-ready roadmap items with priority, owner actions, evidence, acceptance gates, sequencing, and measurement plans.
- **AI improvement backlog handoff:** `/agents/ai-improvement-backlog-handoff` packages roadmap refreshes into implementation-ready work items with story, scope, dependencies, acceptance gates, measurement, and launch readiness.
- **AI improvement implementation kickoff packet:** `/agents/ai-improvement-implementation-kickoff-packet` turns backlog handoffs into engineering scope, QA gates, data contracts, customer guardrails, launch checklists, and immediate action.
- **AI improvement implementation readiness monitor:** `/agents/ai-improvement-implementation-readiness-monitor` tracks QA gates, data contracts, customer guardrails, launch checklist, blockers, risk, and immediate owner action.
- **AI improvement implementation blocker resolution plan:** `/agents/ai-improvement-implementation-blocker-resolution-plan` converts readiness blockers into owned remediation tasks, proof requirements, exit criteria, QA reruns, guardrail clearance, and unblock action.
- **AI improvement implementation unblock verification report:** `/agents/ai-improvement-implementation-unblock-verification-report` checks remediation tasks, proof, exit criteria, QA reruns, guardrails, and next verification action before QA or launch.
- **AI improvement implementation QA review packet:** `/agents/ai-improvement-implementation-qa-review-packet` packages QA scope, evidence gaps, test gates, customer guardrails, signoff requirements, and next QA action.
- **AI improvement implementation QA signoff report:** `/agents/ai-improvement-implementation-qa-signoff-report` produces final hold or launch-review decisions with required signoffs, evidence gaps, launch blockers, guardrails, and next signoff action.
- **AI improvement launch review packet:** `/agents/ai-improvement-launch-review-packet` turns QA signoff into final launch or hold packets with scope, guardrails, monitoring requirements, rollback triggers, blockers, and next launch action.
- **AI improvement launch execution plan:** `/agents/ai-improvement-launch-execution-plan` converts launch review decisions into owned launch or hold tasks, monitoring setup, rollback setup, guardrails, exit criteria, and immediate action.
- **AI improvement launch execution monitor:** `/agents/ai-improvement-launch-execution-monitor` tracks launch execution progress, monitoring setup, rollback readiness, exit criteria, blockers, risk, and immediate owner action.
- **AI improvement launch outcome monitor:** `/agents/ai-improvement-launch-outcome-monitor` tracks post-launch customer-value readiness, customer signals, rollback readiness, blockers, risk, and next owner action.
- **AI improvement launch value proof packet:** `/agents/ai-improvement-launch-value-proof-packet` packages launch outcome state into customer-value claimability, proof points, evidence gaps, customer-safe language, risk, and advisor next action.
- **AI improvement launch customer communication packet:** `/agents/ai-improvement-launch-customer-communication-packet` turns launch value proof into customer-safe advisor communication, audience visibility, review gates, blocked claims, and next action.
- **AI improvement launch customer communication review packet:** `/agents/ai-improvement-launch-customer-communication-review-packet` decides send or hold with required approvals, send blockers, escalation path, approved copy, and advisor next action.
- **AI improvement launch customer communication delivery packet:** `/agents/ai-improvement-launch-customer-communication-delivery-packet` converts reviewed communication into delivery status, channel plan, payload, checklist, audit trail, follow-up plan, and next action.
- **AI improvement launch customer communication delivery monitor:** `/agents/ai-improvement-launch-customer-communication-delivery-monitor` tracks delivery progress, checklist blockers, audit status, follow-up state, risk, and immediate action.
- **AI improvement launch customer communication delivery unblock plan:** `/agents/ai-improvement-launch-customer-communication-delivery-unblock-plan` turns blocked delivery monitoring into owner tasks, proof gates, exit criteria, recheck plan, and immediate action.
- **AI improvement launch customer communication delivery unblock verification report:** `/agents/ai-improvement-launch-customer-communication-delivery-unblock-verification-report` checks proof gates, exit criteria, unblock tasks, failed checks, required follow-up, risk, and next action.
- **AI improvement launch customer communication delivery send authorization packet:** `/agents/ai-improvement-launch-customer-communication-delivery-send-authorization-packet` decides send or hold after unblock verification with requirements, blocked reasons, payload status, risk, and next action.
- **AI improvement launch customer communication delivery send authorization monitor:** `/agents/ai-improvement-launch-customer-communication-delivery-send-authorization-monitor` tracks send authorization state, blocked requirements, blocked reasons, payload exposure, risk, and immediate action.
- **AI improvement launch customer communication delivery send authorization unblock plan:** `/agents/ai-improvement-launch-customer-communication-delivery-send-authorization-unblock-plan` turns held send authorization into owner tasks, authorization gates, exit criteria, recheck plan, and immediate action.
- **AI improvement launch customer communication delivery send authorization unblock verification report:** `/agents/ai-improvement-launch-customer-communication-delivery-send-authorization-unblock-verification-report` checks authorization gates, exit criteria, unblock tasks, failed checks, required follow-up, risk, and next action.
- **AI improvement launch customer communication delivery send readiness packet:** `/agents/ai-improvement-launch-customer-communication-delivery-send-readiness-packet` packages send gate, customer claim status, blockers, advisor review, risk, and immediate action.
- **AI improvement launch customer communication delivery send readiness review packet:** `/agents/ai-improvement-launch-customer-communication-delivery-send-readiness-review-packet` decides send or hold with approvals, blockers, approved payload, and advisor next action.
- **AI improvement launch customer communication delivery send execution handoff packet:** `/agents/ai-improvement-launch-customer-communication-delivery-send-execution-handoff-packet` converts send review into operator-safe handoff status, execution gate, payload, audit trail, blockers, and immediate action.
- **Customer intent action plan:** `/agents/customer-intent-action-plan` turns ranked intent into an evidence-backed advisor worklist.
- **Customer intent follow-up packets:** `/agents/customer-intent-followup-packet` converts the top intent action into compliant execution scaffolding.
- **Customer intent follow-up review:** `/agents/customer-intent-followup-review` preflights packet copy, evidence, review gates, and guardrails.
- **Customer intent follow-up drafts:** `/agents/customer-intent-followup-draft` hands passing packets into the saved outreach review workflow.
- **Screener copilot:** `/agents/screener-copilot` turns plain-English discovery ideas into structured screens with explained matches.
- **Personalized customer digests:** `/agents/portfolio-digest/*` and `/agents/watchlist-digest/*` turn holdings and watchlists into action-oriented AI payloads.
- **Durable alert delivery:** watchlist alert rules can be edited or disabled, recorded as cooldown-aware events, and drained through a retrying webhook outbox worker.
- **Trust-aware reasoning:** quality checks remain first-class payloads so agents can reason about data reliability.

## Quick start (local)

```bash
make api-profiles
python system/daily_pipeline.py --repo-root . --db-path ./system/market_intel.db
python system/canonical_tickertape.py --source-db ./local_repository/tickertape.sqlite --target-db ./system/market_intel.db
python system/service_api.py --db-path ./system/market_intel.db --profile full --host 0.0.0.0 --port 8000
python system/service_api.py --db-path ./system/market_intel.db --profile market-data --host 0.0.0.0 --port 8001
python system/service_api.py --db-path ./system/market_intel.db --profile agent-runtime --host 0.0.0.0 --port 8002
python system/service_api.py --db-path ./system/market_intel.db --profile agent-runtime --vector-index-dir ./local_repository/vector_indexes
python system/webhook_worker.py --db-path ./system/market_intel.db
python system/webhook_worker.py --db-path ./system/market_intel.db --endpoint-url https://example.com/webhook
```

`make canonicalize` runs the Tickertape import command and populates the serving tables used by `/companies`, `/quotes`, `/ratios`, `/events`, `/peers`, and `/screen`.
`POST /watchlists/webhooks/subscriptions` registers customer-specific destinations for alert notifications. Include `signing_secret` to have the worker send `X-Cerebral-Signature: t=<timestamp>,v1=<hmac-sha256>` over the exact JSON body. `PATCH /watchlists/webhooks/subscriptions/{subscription_id}` updates endpoint, enabled state, event type, or signing secret; `DELETE` disables a subscription while keeping history. `POST /watchlists/webhooks/subscriptions/{subscription_id}/test` queues a signed test event for endpoint verification. `GET /watchlists/webhooks/status` summarizes subscription health, pending delivery pressure, and recent delivery problems. `make deliver-webhooks` drains destination-specific `webhook_outbox` rows and marks them delivered, retryable, or failed. `POST /watchlists/webhooks/outbox/{outbox_id}/replay` requeues a failed or pending row for immediate redelivery after endpoint repair. `GET /watchlists/webhooks/deliveries` exposes the per-attempt audit trail for endpoint, HTTP status, error, duration, and retry state. `WEBHOOK_URL=https://example.com/webhook` remains available as a fallback for outbox rows without a stored destination.

## Public API hardening

API keys are optional in local development and enabled by environment variable:

```bash
INDIAN_STOCKS_API_KEYS=key_one,key_two make api-market-data
INDIAN_STOCKS_API_KEYS=key_one INDIAN_STOCKS_RATE_LIMIT_PER_MINUTE=120 make api-agent-runtime
```

When keys are configured, non-public routes require `x-api-key: <key>` or `Authorization: Bearer <key>`. `/health`, docs, OpenAPI, and the agent manifest remain public for operational discovery.

## Quick start (Docker)

```bash
cd system
API_PROFILE=agent-runtime docker compose up --build -d
```

Service URL example: `http://localhost:8000`

## Suggested deployment architecture

- **Ingestion layer:** cron/Airflow runs `daily_pipeline.py` once daily.
- **Storage layer:** SQLite for edge/single-node deployments, move to PostgreSQL for horizontal scale.
- **API layer:** deploy profile-specific services behind separate domains or paths; keep widgets, agents, and dashboards on the same canonical market contract.
- **Validation layer:** monitor `data_quality_audits`; alert when `status=fail`.
- **Reasoning layer:** AI agents consume `/agents/context/{ticker}`, the agent manifest, and workflow plans.
- **Vector layer:** keep raw data local, build `/vectors/rebuild`, and query `/vectors/search` for semantic grounding.
- **Serving layer:** persist generated markdown report + recommendation into `analyst_reports`.

## Recommended next steps

1. Populate the canonical `companies`, `quote_snapshots`, `financial_ratios`, `company_events`, and `company_peers` tables from local raw artifacts.
2. Add authentication/rate limiting before exposing public agent endpoints.
3. Move to PostgreSQL + read replicas for multi-agent concurrent querying.
4. Add signed run manifests and retention policies for long-term auditability.
5. Publish an MCP server or function-calling wrapper on top of the `agent-runtime` profile.
