from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["capabilities"])


@router.get("/capabilities")
def capabilities(request: Request) -> dict[str, object]:
    profile = request.app.state.profile
    vector_search_ready = "vectors" in profile["router_names"]
    freshness_ready = "freshness" in profile["router_names"]
    market_data_ready = "market" in profile["router_names"]
    screeners_ready = "screeners" in profile["router_names"]
    watchlists_ready = "watchlists" in profile["router_names"]
    portfolios_ready = "portfolios" in profile["router_names"]
    return {
        "profile": profile,
        "security": request.app.state.security,
        "agent_ready": True,
        "freshness_ready": freshness_ready,
        "market_data_ready": market_data_ready,
        "screeners_ready": screeners_ready,
        "watchlists_ready": watchlists_ready,
        "portfolios_ready": portfolios_ready,
        "vector_search_ready": vector_search_ready,
        "canonical_market_routes": [
            "GET /companies/{ticker}",
            "GET /quotes/{ticker}",
            "GET /ratios/{ticker}",
            "GET /events/{ticker}",
            "GET /peers/{ticker}",
            "POST /screen",
        ]
        if market_data_ready
        else [],
        "freshness_routes": [
            "GET /freshness",
            "GET /freshness/{ticker}",
        ]
        if freshness_ready
        else [],
        "screener_routes": [
            "GET /screeners",
            "POST /screeners",
            "GET /screeners/{screener_id}",
            "POST /screeners/{screener_id}/evaluate",
        ]
        if screeners_ready
        else [],
        "watchlist_routes": [
            "GET /watchlists",
            "POST /watchlists",
            "GET /watchlists/{watchlist_id}",
            "POST /watchlists/{watchlist_id}/items",
            "DELETE /watchlists/{watchlist_id}/items/{ticker}",
            "GET /watchlists/alerts/readiness",
            "GET /watchlists/alerts/readiness/actions",
            "POST /watchlists/alerts/readiness/action-queue",
            "POST /watchlists/{watchlist_id}/alerts",
            "GET /watchlists/{watchlist_id}/alerts",
            "PATCH /watchlists/{watchlist_id}/alerts/{rule_id}",
            "DELETE /watchlists/{watchlist_id}/alerts/{rule_id}",
            "GET /watchlists/{watchlist_id}/alerts/evaluate",
            "GET /watchlists/{watchlist_id}/alerts/readiness",
            "GET /watchlists/{watchlist_id}/alerts/events",
            "GET /watchlists/{watchlist_id}/alerts/events/summary",
            "PATCH /watchlists/{watchlist_id}/alerts/events/bulk",
            "PATCH /watchlists/{watchlist_id}/alerts/events/{event_id}",
            "GET /watchlists/{watchlist_id}/alerts/events/{event_id}/reviews",
            "GET /watchlists/{watchlist_id}/alerts/reviews",
            "GET /watchlists/{watchlist_id}/alerts/{rule_id}/events",
            "GET /watchlists/{watchlist_id}/alerts/{rule_id}/events/summary",
            "GET /watchlists/webhooks/subscriptions",
            "POST /watchlists/webhooks/subscriptions",
            "PATCH /watchlists/webhooks/subscriptions/{subscription_id}",
            "DELETE /watchlists/webhooks/subscriptions/{subscription_id}",
            "POST /watchlists/webhooks/subscriptions/{subscription_id}/test",
            "GET /watchlists/webhooks/status",
            "GET /watchlists/webhooks/outbox",
            "POST /watchlists/webhooks/outbox/{outbox_id}/replay",
            "GET /watchlists/webhooks/deliveries",
        ]
        if watchlists_ready
        else [],
        "portfolio_routes": [
            "GET /portfolios",
            "POST /portfolios",
            "GET /portfolios/{portfolio_id}",
            "GET /portfolios/{portfolio_id}/xray",
            "POST /portfolios/{portfolio_id}/holdings",
            "DELETE /portfolios/{portfolio_id}/holdings/{ticker}",
        ]
        if portfolios_ready
        else [],
        "notes": [
            "Profiles are composable so one repository can publish multiple APIs.",
            "Canonical market routes return data plus freshness, quality, lineage, and rights metadata.",
            "Use the agent manifest and OpenAPI document for tool discovery.",
            "Quality checks are first-class signals and should be consumed by agents.",
            "Vector search uses local SQLite embeddings and TurboVec when the optional package is installed.",
            "Research briefs convert local evidence into cited customer-facing AI insight payloads.",
            "Portfolio and watchlist digests personalize those insights into customer-level actions.",
            "The screener copilot turns plain-English discovery ideas into repeatable canonical screens.",
            "Morning briefs prioritize customer actions across freshness, portfolios, watchlists, and screeners.",
            "Advisor follow-up packs turn morning briefs into customer-ready copy and compliance guardrails.",
            "Advisor action queues turn follow-up packs into trackable tasks with blockers and completion criteria.",
            "Saved advisor action queues preserve task status, notes, assignee, due date, manager escalation reviews, idempotent notification logs, notification delivery status transitions, delivery health summaries, delivery-control summaries, prioritized delivery-incident feeds with latest triage state, actionable suppression, assignee filters, and follow-up drill-downs, delivery-incident summaries, delivery-incident ownership workload, incident detail hydration with chronological timelines, operator next actions, validated action execution, and post-action incident state, single and bulk delivery-incident triage reviews, retry-ready notification delivery queues, active delivery lease monitoring, dead-letter triage and requeue remediation, operator remediation audit records, atomic delivery claims, claim-token-validated lease renewals, releases, and completions, immutable delivery-attempt and claim-release audit trails, retry backoff scheduling, and max-attempt exhaustion across customer follow-up sessions, with summary rollups, workload summaries, book-level escalation dashboards, filtered snooze-aware escalation inboxes, compact manager notification payloads, escalation feeds, lightweight task feeds, bulk manager review, bulk task triage, and audit trails for dashboards.",
            "Advisor workbenches rank saved queue tasks into the next best customer follow-up actions.",
            "Advisor outreach drafts turn the top saved task into reviewable customer email, agenda, and compliance copy.",
            "Saved outreach draft reviews preserve approval status, reviewer notes, and an audit path before customer delivery.",
            "Outreach compliance reviews flag risky phrases, missing disclosures, and blocked source tasks before approval.",
            "Outreach approvals are gated by the compliance review unless an explicit human override is supplied.",
            "Compliance review audit trails preserve saved review results for each outreach approval attempt.",
            "Delivery packets expose customer-ready outreach only after approval and a fresh passing compliance review.",
            "Saved outreach delivery records track prepared and delivered customer-ready packets with compliance evidence.",
            "Outreach delivery dashboards summarize ready, stale, delivered, voided, and missing delivery packets.",
            "Outreach outcome capture records customer responses and turns them into the next follow-up action.",
            "Customer intent dashboards rank owners by local outreach outcomes and pending next actions.",
            "Customer engagement timelines consolidate outreach, review, delivery, and outcome history for context-aware action.",
            "Customer engagement briefs compress timeline context into current intent, talking points, avoid-lists, and evidence references.",
            "Customer engagement cadence reviews decide whether contact is appropriate now and which route should run next.",
            "Customer engagement cadence dashboards rank owners by contact readiness across the advisor book.",
            "Customer engagement action queues convert cadence readiness into executable advisor tasks.",
            "Customer engagement task briefs turn queued work into talk tracks, proof points, guardrails, and completion criteria.",
            "AI recommendation effectiveness dashboards measure which guided outreach actions convert into customer outcomes.",
            "AI improvement backlogs turn outcome evidence into ranked model, prompt, and workflow improvements.",
            "AI improvement experiment plans convert backlog items into measurable hypotheses, treatments, metrics, and stop conditions.",
            "AI improvement experiment launch packets turn experiment plans into launch checklists, cohort rules, data capture, and rollback criteria.",
            "AI improvement experiment readouts turn launch evidence into continue, ship, rollback, or collect-more decisions.",
            "AI improvement rollout readiness reports convert readouts into release gates, rollout phases, monitoring, and rollback triggers.",
            "AI improvement rollout monitors surface rollout status, alerts, tracked metrics, rollback risk, and immediate next action.",
            "AI improvement release packets turn rollout monitor state into advisor enablement, support talking points, risks, and rollback guidance.",
            "AI improvement adoption playbooks turn release packets into advisor tasks, training checks, customer language, blockers, and success signals.",
            "AI improvement adoption monitors track advisor readiness, training status, language safety, blockers, success signals, and immediate next action.",
            "AI improvement adoption impact ledgers prove customer value by tying adoption to outcomes, advisor usage, blocked accounts, proof points, and scale decisions.",
            "AI improvement scale decision packets convert measured impact into scale, pilot, hold, or evidence-collection decisions with customer proof and advisor-change guidance.",
            "AI improvement scale execution plans turn scale decisions into accountable tasks, guardrails, proof checks, acceptance criteria, escalation, and next action.",
            "AI improvement scale execution monitors track execution progress, guardrails, proof checks, acceptance gaps, blockers, risk, and immediate owner action.",
            "AI improvement scale learning reports turn execution monitoring into validated learnings, open questions, feedback actions, roadmap updates, and next improvement candidates.",
            "AI improvement roadmap refreshes turn scale learnings into backlog-ready roadmap items with priority, owner actions, evidence, acceptance gates, sequencing, and measurement plans.",
            "AI improvement backlog handoffs package roadmap refreshes into implementation-ready work items with story, scope, dependencies, acceptance gates, measurement, and launch readiness.",
            "AI improvement implementation kickoff packets turn backlog handoffs into engineering scope, QA gates, data contracts, customer guardrails, launch checklists, and immediate action.",
            "AI improvement implementation readiness monitors track QA gates, data contracts, customer guardrails, launch checklist, blockers, risk, and immediate owner action.",
            "AI improvement implementation blocker resolution plans convert readiness blockers into owned remediation tasks, proof requirements, exit criteria, QA reruns, guardrail clearance, and unblock action.",
            "AI improvement implementation unblock verification reports check remediation tasks, proof, exit criteria, QA reruns, guardrails, and next verification action before QA or launch.",
            "AI improvement implementation QA review packets package QA scope, evidence gaps, test gates, customer guardrails, signoff requirements, and next QA action.",
            "AI improvement implementation QA signoff reports produce final hold or launch-review decisions with required signoffs, evidence gaps, launch blockers, guardrails, and next signoff action.",
            "AI improvement launch review packets turn QA signoff into final launch or hold packets with scope, guardrails, monitoring requirements, rollback triggers, blockers, and next launch action.",
            "AI improvement launch execution plans convert launch review decisions into owned launch or hold tasks, monitoring setup, rollback setup, guardrails, exit criteria, and immediate action.",
            "AI improvement launch execution monitors track launch execution progress, monitoring setup, rollback readiness, exit criteria, blockers, risk, and immediate owner action.",
            "AI improvement launch outcome monitors track post-launch customer-value readiness, customer signals, rollback readiness, blockers, risk, and next owner action.",
            "AI improvement launch value proof packets package outcome state into customer-value claimability, proof points, evidence gaps, customer-safe language, risk, and advisor next action.",
            "AI improvement launch customer communication packets turn value proof into customer-safe advisor communication, visibility, review gates, blocked claims, and next action.",
            "AI improvement launch customer communication review packets decide send or hold with required approvals, send blockers, escalation path, approved copy, and advisor next action.",
            "AI improvement launch customer communication delivery packets convert reviewed communication into delivery status, channel plan, payload, checklist, audit trail, follow-up plan, and next action.",
            "AI improvement launch customer communication delivery monitors track delivery progress, checklist blockers, audit status, follow-up state, risk, and immediate action.",
            "AI improvement launch customer communication delivery unblock plans turn blocked delivery monitoring into owner tasks, proof gates, exit criteria, recheck plan, and immediate action.",
            "AI improvement launch customer communication delivery unblock verification reports check proof gates, exit criteria, unblock tasks, failed checks, required follow-up, risk, and next action.",
            "AI improvement launch customer communication delivery send authorization packets decide send or hold after unblock verification with requirements, blocked reasons, payload status, risk, and next action.",
            "AI improvement launch customer communication delivery send authorization monitors track send authorization state, blocked requirements, blocked reasons, payload exposure, risk, and immediate action.",
            "AI improvement launch customer communication delivery send authorization unblock plans turn held send authorization into owner tasks, authorization gates, exit criteria, recheck plan, and immediate action.",
            "AI improvement launch customer communication delivery send authorization unblock verification reports check authorization gates, exit criteria, unblock tasks, failed checks, required follow-up, risk, and next action.",
            "AI improvement launch customer communication delivery send readiness packets package send gate, customer claim status, blockers, advisor review, risk, and immediate action.",
            "AI improvement launch customer communication delivery send readiness review packets decide send or hold with approvals, blockers, approved payload, and advisor next action.",
            "AI improvement launch customer communication delivery send execution handoff packets convert send review into operator-safe handoff status, execution gate, payload, audit trail, blockers, and immediate action.",
            "Customer intent action plans turn ranked intent into evidence-backed advisor worklists.",
            "Customer intent follow-up packets convert top actions into compliant execution scaffolds.",
            "Customer intent follow-up reviews preflight packet copy, evidence, review gates, and guardrails.",
            "Customer intent follow-up drafts hand passing packets into the saved outreach review workflow.",
        ],
    }
