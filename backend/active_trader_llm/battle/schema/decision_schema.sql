-- Decision Log Database Schema
-- Stores all trading decisions with full reasoning for transparency

-- Decision log table
CREATE TABLE IF NOT EXISTS decision_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    decision TEXT NOT NULL CHECK(decision IN ('buy', 'sell', 'hold', 'close')),
    reasoning TEXT NOT NULL,
    confidence REAL CHECK(confidence >= 0 AND confidence <= 100),

    -- Decision context
    timestamp TIMESTAMP NOT NULL,
    current_price REAL CHECK(current_price > 0),
    position_size REAL CHECK(position_size >= 0),

    -- Trade parameters (populated if buy decision)
    entry_price REAL CHECK(entry_price > 0),
    stop_loss REAL CHECK(stop_loss > 0),
    take_profit REAL CHECK(take_profit > 0),
    direction TEXT CHECK(direction IN ('long', 'short', NULL)),

    -- Execution tracking
    executed BOOLEAN NOT NULL DEFAULT 0,
    execution_price REAL CHECK(execution_price > 0),
    execution_time TIMESTAMP,
    execution_status TEXT CHECK(execution_status IN ('filled', 'rejected', 'pending', 'cancelled', NULL))
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_decision_model_time ON decision_log(model_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_decision_symbol_time ON decision_log(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_decision_model_symbol ON decision_log(model_id, symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_decision_executed ON decision_log(executed, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_decision_timestamp ON decision_log(timestamp DESC);

-- Decision comparison view (for comparing model decisions on same symbol/time)
CREATE VIEW IF NOT EXISTS decision_comparison AS
SELECT
    d1.symbol,
    d1.timestamp,
    d1.current_price,
    d1.model_id as model_1_id,
    d1.decision as model_1_decision,
    d1.confidence as model_1_confidence,
    d2.model_id as model_2_id,
    d2.decision as model_2_decision,
    d2.confidence as model_2_confidence,
    ABS(julianday(d1.timestamp) - julianday(d2.timestamp)) * 24 * 60 as time_diff_minutes
FROM decision_log d1
JOIN decision_log d2 ON d1.symbol = d2.symbol
    AND d1.model_id < d2.model_id
    AND ABS(julianday(d1.timestamp) - julianday(d2.timestamp)) * 24 * 60 <= 5
ORDER BY d1.timestamp DESC;
