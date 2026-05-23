CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS raw_documents (
    document_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    source_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_type TEXT NOT NULL,
    content TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    record_count INTEGER,
    source_timestamp TEXT,
    ingested_at TEXT NOT NULL,
    UNIQUE(run_id, file_path),
    FOREIGN KEY(run_id) REFERENCES ingestion_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_raw_documents_run_source ON raw_documents(run_id, source_name);
CREATE INDEX IF NOT EXISTS idx_raw_documents_ingested_at ON raw_documents(ingested_at);

CREATE TABLE IF NOT EXISTS entity_risk_signals (
    signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    entity_name TEXT NOT NULL,
    region TEXT,
    risk_type TEXT,
    signal_strength REAL,
    rationale TEXT,
    captured_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES ingestion_runs(run_id)
);

CREATE TABLE IF NOT EXISTS data_quality_audits (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    check_name TEXT NOT NULL,
    status TEXT NOT NULL,
    details TEXT NOT NULL,
    checked_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES ingestion_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_quality_audits_run ON data_quality_audits(run_id, status);

CREATE TABLE IF NOT EXISTS companies (
    ticker TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    exchange TEXT,
    isin TEXT,
    sector TEXT,
    industry TEXT,
    market_cap REAL,
    website TEXT,
    description TEXT,
    raw_document_id INTEGER,
    local_ingestion_run_id INTEGER NOT NULL,
    as_of TEXT NOT NULL,
    processed_at TEXT NOT NULL,
    quality_status TEXT NOT NULL DEFAULT 'unknown',
    data_rights_status TEXT NOT NULL DEFAULT 'unknown',
    extra_json TEXT,
    FOREIGN KEY(local_ingestion_run_id) REFERENCES ingestion_runs(run_id),
    FOREIGN KEY(raw_document_id) REFERENCES raw_documents(document_id)
);

CREATE INDEX IF NOT EXISTS idx_companies_sector ON companies(sector, industry);
CREATE INDEX IF NOT EXISTS idx_companies_market_cap ON companies(market_cap DESC);

CREATE TABLE IF NOT EXISTS quote_snapshots (
    quote_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    price REAL,
    currency TEXT,
    open_price REAL,
    high_price REAL,
    low_price REAL,
    previous_close REAL,
    volume REAL,
    raw_json TEXT,
    raw_document_id INTEGER,
    local_ingestion_run_id INTEGER NOT NULL,
    as_of TEXT NOT NULL,
    processed_at TEXT NOT NULL,
    quality_status TEXT NOT NULL DEFAULT 'unknown',
    data_rights_status TEXT NOT NULL DEFAULT 'unknown',
    FOREIGN KEY(local_ingestion_run_id) REFERENCES ingestion_runs(run_id),
    FOREIGN KEY(raw_document_id) REFERENCES raw_documents(document_id)
);

CREATE INDEX IF NOT EXISTS idx_quote_snapshots_ticker_asof
    ON quote_snapshots(ticker, as_of DESC, processed_at DESC);

CREATE TABLE IF NOT EXISTS financial_ratios (
    ratio_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    ratio_name TEXT NOT NULL,
    ratio_value REAL,
    period TEXT,
    period_end TEXT,
    raw_json TEXT,
    raw_document_id INTEGER,
    local_ingestion_run_id INTEGER NOT NULL,
    as_of TEXT NOT NULL,
    processed_at TEXT NOT NULL,
    quality_status TEXT NOT NULL DEFAULT 'unknown',
    data_rights_status TEXT NOT NULL DEFAULT 'unknown',
    FOREIGN KEY(local_ingestion_run_id) REFERENCES ingestion_runs(run_id),
    FOREIGN KEY(raw_document_id) REFERENCES raw_documents(document_id)
);

CREATE INDEX IF NOT EXISTS idx_financial_ratios_ticker_name
    ON financial_ratios(ticker, ratio_name, period_end DESC);

CREATE TABLE IF NOT EXISTS company_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    event_type TEXT,
    event_date TEXT,
    title TEXT NOT NULL,
    description TEXT,
    source_url TEXT,
    raw_json TEXT,
    raw_document_id INTEGER,
    local_ingestion_run_id INTEGER NOT NULL,
    as_of TEXT NOT NULL,
    processed_at TEXT NOT NULL,
    quality_status TEXT NOT NULL DEFAULT 'unknown',
    data_rights_status TEXT NOT NULL DEFAULT 'unknown',
    FOREIGN KEY(local_ingestion_run_id) REFERENCES ingestion_runs(run_id),
    FOREIGN KEY(raw_document_id) REFERENCES raw_documents(document_id)
);

CREATE INDEX IF NOT EXISTS idx_company_events_ticker_date
    ON company_events(ticker, event_date DESC);

CREATE TABLE IF NOT EXISTS company_peers (
    peer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    peer_ticker TEXT NOT NULL,
    relationship TEXT,
    score REAL,
    raw_json TEXT,
    raw_document_id INTEGER,
    local_ingestion_run_id INTEGER NOT NULL,
    as_of TEXT NOT NULL,
    processed_at TEXT NOT NULL,
    quality_status TEXT NOT NULL DEFAULT 'unknown',
    data_rights_status TEXT NOT NULL DEFAULT 'unknown',
    FOREIGN KEY(local_ingestion_run_id) REFERENCES ingestion_runs(run_id),
    FOREIGN KEY(raw_document_id) REFERENCES raw_documents(document_id)
);

CREATE INDEX IF NOT EXISTS idx_company_peers_ticker_score
    ON company_peers(ticker, score DESC);

CREATE TABLE IF NOT EXISTS watchlists (
    watchlist_id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(owner_id, name)
);

CREATE INDEX IF NOT EXISTS idx_watchlists_owner ON watchlists(owner_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS watchlist_items (
    item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    watchlist_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    notes TEXT,
    added_at TEXT NOT NULL,
    UNIQUE(watchlist_id, ticker),
    FOREIGN KEY(watchlist_id) REFERENCES watchlists(watchlist_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_watchlist_items_watchlist ON watchlist_items(watchlist_id, ticker);

CREATE TABLE IF NOT EXISTS watchlist_alert_rules (
    rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
    watchlist_id INTEGER NOT NULL,
    ticker TEXT,
    metric TEXT NOT NULL,
    operator TEXT NOT NULL CHECK(operator IN ('lt', 'lte', 'gt', 'gte')),
    threshold REAL NOT NULL,
    cooldown_minutes INTEGER NOT NULL DEFAULT 60,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(watchlist_id) REFERENCES watchlists(watchlist_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_watchlist_alert_rules_watchlist
    ON watchlist_alert_rules(watchlist_id, enabled, ticker);

CREATE TABLE IF NOT EXISTS alert_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    watchlist_id INTEGER NOT NULL,
    rule_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    metric TEXT NOT NULL,
    operator TEXT NOT NULL,
    threshold REAL NOT NULL,
    value REAL,
    triggered_at TEXT NOT NULL,
    dedupe_key TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(watchlist_id) REFERENCES watchlists(watchlist_id) ON DELETE CASCADE,
    FOREIGN KEY(rule_id) REFERENCES watchlist_alert_rules(rule_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_alert_events_watchlist
    ON alert_events(watchlist_id, triggered_at DESC);

CREATE INDEX IF NOT EXISTS idx_alert_events_rule
    ON alert_events(rule_id, triggered_at DESC);

CREATE TABLE IF NOT EXISTS alert_event_reviews (
    event_id INTEGER PRIMARY KEY,
    owner_id TEXT NOT NULL DEFAULT 'default',
    status TEXT NOT NULL CHECK(status IN ('open', 'reviewed', 'dismissed')),
    reviewed_by TEXT,
    reviewed_at TEXT NOT NULL,
    notes TEXT,
    FOREIGN KEY(event_id) REFERENCES alert_events(event_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_alert_event_reviews_owner_status
    ON alert_event_reviews(owner_id, status, reviewed_at DESC);

CREATE TABLE IF NOT EXISTS alert_event_review_audits (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL,
    watchlist_id INTEGER NOT NULL,
    owner_id TEXT NOT NULL DEFAULT 'default',
    status TEXT NOT NULL CHECK(status IN ('open', 'reviewed', 'dismissed')),
    reviewed_by TEXT,
    reviewed_at TEXT NOT NULL,
    notes TEXT,
    source TEXT NOT NULL DEFAULT 'single',
    batch_size INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY(event_id) REFERENCES alert_events(event_id) ON DELETE CASCADE,
    FOREIGN KEY(watchlist_id) REFERENCES watchlists(watchlist_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_alert_event_review_audits_event
    ON alert_event_review_audits(event_id, reviewed_at DESC);

CREATE TABLE IF NOT EXISTS webhook_subscriptions (
    subscription_id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id TEXT NOT NULL DEFAULT 'default',
    event_type TEXT NOT NULL,
    endpoint_url TEXT NOT NULL,
    signing_secret TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(owner_id, event_type, endpoint_url)
);

CREATE INDEX IF NOT EXISTS idx_webhook_subscriptions_owner
    ON webhook_subscriptions(owner_id, event_type, enabled);

CREATE TABLE IF NOT EXISTS webhook_outbox (
    outbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id TEXT NOT NULL DEFAULT 'default',
    subscription_id INTEGER,
    destination_url TEXT,
    event_type TEXT NOT NULL,
    aggregate_type TEXT NOT NULL,
    aggregate_id INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    next_attempt_at TEXT,
    delivered_at TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    FOREIGN KEY(subscription_id) REFERENCES webhook_subscriptions(subscription_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_webhook_outbox_status
    ON webhook_outbox(status, next_attempt_at, created_at);

CREATE INDEX IF NOT EXISTS idx_webhook_outbox_owner_status
    ON webhook_outbox(owner_id, status, next_attempt_at, created_at);

CREATE TABLE IF NOT EXISTS webhook_delivery_attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    outbox_id INTEGER NOT NULL,
    owner_id TEXT NOT NULL DEFAULT 'default',
    subscription_id INTEGER,
    endpoint_url TEXT,
    event_type TEXT NOT NULL,
    attempted_at TEXT NOT NULL,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    delivered INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    http_status INTEGER,
    error TEXT,
    FOREIGN KEY(outbox_id) REFERENCES webhook_outbox(outbox_id) ON DELETE CASCADE,
    FOREIGN KEY(subscription_id) REFERENCES webhook_subscriptions(subscription_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_webhook_delivery_attempts_outbox
    ON webhook_delivery_attempts(outbox_id, attempted_at DESC);

CREATE INDEX IF NOT EXISTS idx_webhook_delivery_attempts_owner_status
    ON webhook_delivery_attempts(owner_id, status, attempted_at DESC);

CREATE TABLE IF NOT EXISTS portfolios (
    portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    description TEXT,
    base_currency TEXT NOT NULL DEFAULT 'INR',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(owner_id, name)
);

CREATE INDEX IF NOT EXISTS idx_portfolios_owner ON portfolios(owner_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS portfolio_holdings (
    holding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    quantity REAL NOT NULL,
    average_cost REAL,
    notes TEXT,
    added_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(portfolio_id, ticker),
    FOREIGN KEY(portfolio_id) REFERENCES portfolios(portfolio_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_portfolio_holdings_portfolio
    ON portfolio_holdings(portfolio_id, ticker);

CREATE TABLE IF NOT EXISTS saved_screeners (
    screener_id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    description TEXT,
    filters_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(owner_id, name)
);

CREATE INDEX IF NOT EXISTS idx_saved_screeners_owner
    ON saved_screeners(owner_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS screener_evaluations (
    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    screener_id INTEGER NOT NULL,
    evaluated_at TEXT NOT NULL,
    result_count INTEGER NOT NULL,
    top_tickers_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    FOREIGN KEY(screener_id) REFERENCES saved_screeners(screener_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_screener_evaluations_latest
    ON screener_evaluations(screener_id, evaluated_at DESC);

CREATE TABLE IF NOT EXISTS advisor_action_queues (
    queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id TEXT NOT NULL DEFAULT 'default',
    title TEXT NOT NULL,
    focus TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    task_count INTEGER NOT NULL DEFAULT 0,
    open_task_count INTEGER NOT NULL DEFAULT 0,
    blocked_task_count INTEGER NOT NULL DEFAULT 0,
    completed_task_count INTEGER NOT NULL DEFAULT 0,
    source_followup_json TEXT NOT NULL,
    queue_markdown TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_advisor_action_queues_owner
    ON advisor_action_queues(owner_id, status, updated_at DESC);

CREATE TABLE IF NOT EXISTS advisor_action_queue_tasks (
    saved_task_id INTEGER PRIMARY KEY AUTOINCREMENT,
    queue_id INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    title TEXT NOT NULL,
    urgency TEXT NOT NULL,
    status TEXT NOT NULL,
    rationale TEXT NOT NULL,
    completion_criteria TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    notes TEXT,
    assigned_to TEXT,
    due_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE(queue_id, task_id),
    FOREIGN KEY(queue_id) REFERENCES advisor_action_queues(queue_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_tasks_queue
    ON advisor_action_queue_tasks(queue_id, status, updated_at DESC);

CREATE TABLE IF NOT EXISTS advisor_action_queue_task_updates (
    update_id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id TEXT NOT NULL DEFAULT 'default',
    queue_id INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    previous_status TEXT,
    new_status TEXT NOT NULL,
    previous_notes TEXT,
    new_notes TEXT,
    previous_assigned_to TEXT,
    new_assigned_to TEXT,
    previous_due_at TEXT,
    new_due_at TEXT,
    updated_by TEXT,
    update_source TEXT NOT NULL DEFAULT 'api',
    created_at TEXT NOT NULL,
    FOREIGN KEY(queue_id) REFERENCES advisor_action_queues(queue_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_task_updates_owner
    ON advisor_action_queue_task_updates(owner_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_task_updates_task
    ON advisor_action_queue_task_updates(queue_id, task_id, created_at DESC);

CREATE TABLE IF NOT EXISTS advisor_action_queue_escalation_reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id TEXT NOT NULL DEFAULT 'default',
    queue_id INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    review_status TEXT NOT NULL,
    reviewer TEXT,
    notes TEXT,
    snoozed_until TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(queue_id) REFERENCES advisor_action_queues(queue_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_reviews_owner
    ON advisor_action_queue_escalation_reviews(owner_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_reviews_task
    ON advisor_action_queue_escalation_reviews(queue_id, task_id, created_at DESC);

CREATE TABLE IF NOT EXISTS advisor_action_queue_escalation_notifications (
    notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id TEXT,
    as_of TEXT NOT NULL,
    channel TEXT NOT NULL,
    recipient TEXT,
    status TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    filter_json TEXT NOT NULL,
    item_count INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL,
    delivery_notes TEXT,
    delivered_at TEXT,
    delivery_retry_after TEXT,
    delivery_exhausted_at TEXT,
    delivery_exhausted_reason TEXT,
    delivery_claimed_by TEXT,
    delivery_claimed_at TEXT,
    delivery_claimed_until TEXT,
    delivery_claim_token TEXT,
    delivery_attempt_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_notifications_owner
    ON advisor_action_queue_escalation_notifications(owner_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_notifications_channel
    ON advisor_action_queue_escalation_notifications(channel, status, created_at DESC);

CREATE TABLE IF NOT EXISTS advisor_action_queue_escalation_notification_attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    notification_id INTEGER NOT NULL,
    owner_id TEXT,
    channel TEXT NOT NULL,
    recipient TEXT,
    status TEXT NOT NULL,
    claim_token TEXT NOT NULL,
    claimed_by TEXT,
    attempt_number INTEGER NOT NULL,
    delivery_notes TEXT,
    delivered_at TEXT,
    retry_after TEXT,
    exhausted_at TEXT,
    exhausted_reason TEXT,
    claimed_at TEXT,
    completed_at TEXT NOT NULL,
    FOREIGN KEY(notification_id) REFERENCES advisor_action_queue_escalation_notifications(notification_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_notification_attempts_owner
    ON advisor_action_queue_escalation_notification_attempts(owner_id, status, completed_at DESC);

CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_notification_attempts_notification
    ON advisor_action_queue_escalation_notification_attempts(notification_id, completed_at DESC);

CREATE TABLE IF NOT EXISTS advisor_action_queue_escalation_notification_remediations (
    remediation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    notification_id INTEGER NOT NULL,
    owner_id TEXT,
    channel TEXT NOT NULL,
    recipient TEXT,
    remediation_type TEXT NOT NULL,
    remediation_notes TEXT,
    requeued_by TEXT,
    retry_after TEXT,
    previous_delivery_exhausted_at TEXT,
    previous_delivery_exhausted_reason TEXT,
    previous_delivery_attempt_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(notification_id) REFERENCES advisor_action_queue_escalation_notifications(notification_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_notification_remediations_owner
    ON advisor_action_queue_escalation_notification_remediations(owner_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_notification_remediations_notification
    ON advisor_action_queue_escalation_notification_remediations(notification_id, created_at DESC);

CREATE TABLE IF NOT EXISTS advisor_action_queue_escalation_notification_claim_releases (
    release_id INTEGER PRIMARY KEY AUTOINCREMENT,
    notification_id INTEGER NOT NULL,
    owner_id TEXT,
    channel TEXT NOT NULL,
    recipient TEXT,
    status TEXT NOT NULL,
    claim_token TEXT NOT NULL,
    claimed_by TEXT,
    claimed_at TEXT,
    claimed_until TEXT,
    released_by TEXT,
    release_notes TEXT,
    previous_delivery_attempt_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(notification_id) REFERENCES advisor_action_queue_escalation_notifications(notification_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_notification_claim_releases_owner
    ON advisor_action_queue_escalation_notification_claim_releases(owner_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_notification_claim_releases_notification
    ON advisor_action_queue_escalation_notification_claim_releases(notification_id, created_at DESC);

CREATE TABLE IF NOT EXISTS advisor_action_queue_escalation_notification_incident_reviews (
    incident_review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    notification_id INTEGER NOT NULL,
    owner_id TEXT,
    channel TEXT NOT NULL,
    incident_type TEXT NOT NULL,
    incident_status TEXT NOT NULL,
    reviewer TEXT,
    assigned_to TEXT,
    notes TEXT,
    follow_up_at TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(notification_id) REFERENCES advisor_action_queue_escalation_notifications(notification_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_notification_incident_reviews_owner
    ON advisor_action_queue_escalation_notification_incident_reviews(owner_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_advisor_action_queue_escalation_notification_incident_reviews_notification
    ON advisor_action_queue_escalation_notification_incident_reviews(notification_id, created_at DESC);

CREATE TABLE IF NOT EXISTS advisor_outreach_drafts (
    draft_id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id TEXT NOT NULL DEFAULT 'default',
    queue_id INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    selection TEXT NOT NULL,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    meeting_agenda_json TEXT NOT NULL,
    compliance_guardrails_json TEXT NOT NULL,
    source_task_json TEXT NOT NULL,
    source_queue_json TEXT NOT NULL,
    draft_markdown TEXT NOT NULL,
    review_notes TEXT,
    reviewer TEXT,
    reviewed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(queue_id) REFERENCES advisor_action_queues(queue_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_advisor_outreach_drafts_owner
    ON advisor_outreach_drafts(owner_id, status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_advisor_outreach_drafts_queue
    ON advisor_outreach_drafts(queue_id, task_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS advisor_outreach_compliance_reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    draft_id INTEGER NOT NULL,
    owner_id TEXT NOT NULL DEFAULT 'default',
    queue_id INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    draft_status TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    can_approve INTEGER NOT NULL,
    approval_recommendation TEXT NOT NULL,
    issue_count INTEGER NOT NULL,
    issues_json TEXT NOT NULL,
    passed_checks_json TEXT NOT NULL,
    source_draft_json TEXT NOT NULL,
    review_markdown TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(draft_id) REFERENCES advisor_outreach_drafts(draft_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_advisor_outreach_compliance_reviews_draft
    ON advisor_outreach_compliance_reviews(draft_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_advisor_outreach_compliance_reviews_owner
    ON advisor_outreach_compliance_reviews(owner_id, risk_level, created_at DESC);

CREATE TABLE IF NOT EXISTS advisor_outreach_delivery_records (
    delivery_id INTEGER PRIMARY KEY AUTOINCREMENT,
    draft_id INTEGER NOT NULL,
    owner_id TEXT NOT NULL DEFAULT 'default',
    queue_id INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ready',
    customer_email_json TEXT NOT NULL,
    meeting_agenda_json TEXT NOT NULL,
    compliance_review_json TEXT NOT NULL,
    approval_evidence_json TEXT NOT NULL,
    source_task_json TEXT NOT NULL,
    packet_markdown TEXT NOT NULL,
    delivery_notes TEXT,
    delivered_by TEXT,
    delivered_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(draft_id) REFERENCES advisor_outreach_drafts(draft_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_advisor_outreach_delivery_records_owner
    ON advisor_outreach_delivery_records(owner_id, status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_advisor_outreach_delivery_records_draft
    ON advisor_outreach_delivery_records(draft_id, created_at DESC);

CREATE TABLE IF NOT EXISTS advisor_outreach_delivery_outcomes (
    outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
    delivery_id INTEGER NOT NULL,
    draft_id INTEGER NOT NULL,
    owner_id TEXT NOT NULL DEFAULT 'default',
    queue_id INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    outcome_type TEXT NOT NULL,
    customer_signal TEXT NOT NULL,
    response_text TEXT,
    next_action_json TEXT NOT NULL,
    follow_up_due_at TEXT,
    recorded_by TEXT,
    source_delivery_json TEXT NOT NULL,
    outcome_markdown TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(delivery_id) REFERENCES advisor_outreach_delivery_records(delivery_id) ON DELETE CASCADE,
    FOREIGN KEY(draft_id) REFERENCES advisor_outreach_drafts(draft_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_advisor_outreach_delivery_outcomes_owner
    ON advisor_outreach_delivery_outcomes(owner_id, outcome_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_advisor_outreach_delivery_outcomes_delivery
    ON advisor_outreach_delivery_outcomes(delivery_id, created_at DESC);

CREATE TABLE IF NOT EXISTS document_embeddings (
    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL UNIQUE,
    run_id INTEGER NOT NULL,
    source_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    content_preview TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL,
    embedding_blob BLOB NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(document_id) REFERENCES raw_documents(document_id)
);

CREATE INDEX IF NOT EXISTS idx_document_embeddings_run ON document_embeddings(run_id, source_name);
CREATE INDEX IF NOT EXISTS idx_document_embeddings_hash ON document_embeddings(content_sha256);

CREATE TABLE IF NOT EXISTS vector_index_state (
    state_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    source_name TEXT,
    backend TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL,
    bit_width INTEGER NOT NULL,
    index_path TEXT,
    item_count INTEGER NOT NULL,
    built_at TEXT NOT NULL,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_vector_index_state_latest
    ON vector_index_state(run_id, source_name, built_at);

CREATE TABLE IF NOT EXISTS analyst_reports (
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    report_markdown TEXT NOT NULL,
    recommendation TEXT,
    investment_horizon TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES ingestion_runs(run_id)
);
