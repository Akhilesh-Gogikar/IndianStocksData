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
    ingested_at TEXT NOT NULL,
    UNIQUE(run_id, file_path),
    FOREIGN KEY(run_id) REFERENCES ingestion_runs(run_id)
);

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
