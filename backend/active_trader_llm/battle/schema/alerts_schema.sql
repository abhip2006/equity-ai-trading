-- ============================================================================
-- BATTLE ALERTS DATABASE SCHEMA
-- ============================================================================
-- Stores all alerts generated during battle execution
-- Supports querying, filtering, and historical analysis

-- Alerts table: stores all triggered alerts
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Alert identification
    alert_type TEXT NOT NULL,  -- model_crash, critical_drawdown, etc.
    severity TEXT NOT NULL,    -- INFO, WARNING, CRITICAL, EMERGENCY
    model_id TEXT NOT NULL,    -- Which model triggered this alert

    -- Alert content
    message TEXT NOT NULL,
    metadata TEXT,             -- JSON blob with additional data

    -- Timing
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Status tracking
    acknowledged BOOLEAN DEFAULT 0,
    acknowledged_at TIMESTAMP,
    acknowledged_by TEXT,

    -- Resolution
    resolved BOOLEAN DEFAULT 0,
    resolved_at TIMESTAMP,
    resolution_notes TEXT
);

-- Indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_alerts_model_time
    ON alerts(model_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_alerts_type_time
    ON alerts(alert_type, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_alerts_severity_time
    ON alerts(severity, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_alerts_unresolved
    ON alerts(resolved, timestamp DESC);

-- Alert summary view: quick statistics by model
CREATE VIEW IF NOT EXISTS alert_summary_by_model AS
SELECT
    model_id,
    COUNT(*) as total_alerts,
    SUM(CASE WHEN severity = 'CRITICAL' OR severity = 'EMERGENCY' THEN 1 ELSE 0 END) as critical_alerts,
    SUM(CASE WHEN severity = 'WARNING' THEN 1 ELSE 0 END) as warning_alerts,
    SUM(CASE WHEN severity = 'INFO' THEN 1 ELSE 0 END) as info_alerts,
    MAX(timestamp) as last_alert_time
FROM alerts
GROUP BY model_id;

-- Alert summary by type
CREATE VIEW IF NOT EXISTS alert_summary_by_type AS
SELECT
    alert_type,
    COUNT(*) as total_count,
    COUNT(DISTINCT model_id) as affected_models,
    MIN(timestamp) as first_occurrence,
    MAX(timestamp) as last_occurrence
FROM alerts
GROUP BY alert_type;

-- Recent critical alerts view
CREATE VIEW IF NOT EXISTS recent_critical_alerts AS
SELECT
    id,
    alert_type,
    severity,
    model_id,
    message,
    timestamp
FROM alerts
WHERE severity IN ('CRITICAL', 'EMERGENCY')
    AND timestamp > datetime('now', '-24 hours')
ORDER BY timestamp DESC;
