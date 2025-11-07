-- Metrics Timeseries Database Schema
-- Stores all performance metrics and timeseries data for trading battle

-- Hourly equity snapshots
CREATE TABLE IF NOT EXISTS hourly_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    equity REAL NOT NULL CHECK(equity > 0),
    cash REAL NOT NULL CHECK(cash >= 0),
    positions_value REAL NOT NULL CHECK(positions_value >= 0),
    positions_count INTEGER NOT NULL CHECK(positions_count >= 0),
    UNIQUE(model_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_hourly_model_time ON hourly_snapshots(model_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_hourly_timestamp ON hourly_snapshots(timestamp DESC);

-- Daily summary metrics (end of day snapshots)
CREATE TABLE IF NOT EXISTS daily_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    date DATE NOT NULL,

    -- Equity metrics
    opening_equity REAL NOT NULL CHECK(opening_equity > 0),
    closing_equity REAL NOT NULL CHECK(closing_equity > 0),
    high_equity REAL NOT NULL CHECK(high_equity > 0),
    low_equity REAL NOT NULL CHECK(low_equity > 0),

    -- Daily P&L
    daily_pnl REAL NOT NULL,
    daily_return_pct REAL NOT NULL,

    -- Trading activity
    trades_opened INTEGER NOT NULL CHECK(trades_opened >= 0),
    trades_closed INTEGER NOT NULL CHECK(trades_closed >= 0),

    -- Risk metrics
    max_drawdown_pct REAL,
    sharpe_ratio REAL,

    UNIQUE(model_id, date)
);

CREATE INDEX IF NOT EXISTS idx_daily_model_date ON daily_summaries(model_id, date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_summaries(date DESC);

-- Trade log (executed trades for metrics calculation)
CREATE TABLE IF NOT EXISTS trade_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('long', 'short')),

    -- Entry
    entry_price REAL NOT NULL CHECK(entry_price > 0),
    entry_time TIMESTAMP NOT NULL,
    shares INTEGER NOT NULL CHECK(shares > 0),

    -- Exit
    exit_price REAL NOT NULL CHECK(exit_price > 0),
    exit_time TIMESTAMP NOT NULL,
    exit_reason TEXT NOT NULL CHECK(exit_reason IN ('stop_loss', 'take_profit', 'manual', 'eod_close')),

    -- P&L
    realized_pnl REAL NOT NULL,
    realized_pnl_pct REAL NOT NULL,

    -- Strategy
    strategy TEXT,

    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trade_model_time ON trade_log(model_id, exit_time DESC);
CREATE INDEX IF NOT EXISTS idx_trade_symbol ON trade_log(symbol, exit_time DESC);
CREATE INDEX IF NOT EXISTS idx_trade_exit_time ON trade_log(exit_time DESC);

-- Decision log (all trading decisions with reasoning)
-- Note: This is also in decision_schema.sql but duplicated here for completeness
CREATE TABLE IF NOT EXISTS decision_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    decision TEXT NOT NULL CHECK(decision IN ('buy', 'sell', 'hold', 'close')),
    reasoning TEXT NOT NULL,
    confidence REAL CHECK(confidence >= 0 AND confidence <= 100),

    -- Context
    timestamp TIMESTAMP NOT NULL,
    current_price REAL CHECK(current_price > 0),
    position_size REAL CHECK(position_size >= 0),

    -- Trade parameters (if buy)
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

CREATE INDEX IF NOT EXISTS idx_decision_model_time ON decision_log(model_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_decision_symbol_time ON decision_log(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_decision_model_symbol ON decision_log(model_id, symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_decision_executed ON decision_log(executed, timestamp DESC);

-- Weekly summaries (aggregated from daily)
CREATE TABLE IF NOT EXISTS weekly_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    week_start DATE NOT NULL,
    week_end DATE NOT NULL,

    -- Week metrics
    starting_equity REAL NOT NULL CHECK(starting_equity > 0),
    ending_equity REAL NOT NULL CHECK(ending_equity > 0),
    weekly_return_pct REAL NOT NULL,

    -- Trading activity
    total_trades INTEGER NOT NULL CHECK(total_trades >= 0),
    winning_trades INTEGER NOT NULL CHECK(winning_trades >= 0),
    losing_trades INTEGER NOT NULL CHECK(losing_trades >= 0),
    win_rate REAL,

    -- Risk metrics
    sharpe_ratio REAL,
    max_drawdown_pct REAL,

    UNIQUE(model_id, week_start)
);

CREATE INDEX IF NOT EXISTS idx_weekly_model_week ON weekly_summaries(model_id, week_start DESC);

-- Monthly summaries (aggregated from daily)
CREATE TABLE IF NOT EXISTS monthly_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL CHECK(month >= 1 AND month <= 12),

    -- Month metrics
    starting_equity REAL NOT NULL CHECK(starting_equity > 0),
    ending_equity REAL NOT NULL CHECK(ending_equity > 0),
    monthly_return_pct REAL NOT NULL,

    -- Trading activity
    total_trades INTEGER NOT NULL CHECK(total_trades >= 0),
    winning_trades INTEGER NOT NULL CHECK(winning_trades >= 0),
    losing_trades INTEGER NOT NULL CHECK(losing_trades >= 0),
    win_rate REAL,
    profit_factor REAL,

    -- Risk metrics
    sharpe_ratio REAL,
    max_drawdown_pct REAL,

    UNIQUE(model_id, year, month)
);

CREATE INDEX IF NOT EXISTS idx_monthly_model_year_month ON monthly_summaries(model_id, year DESC, month DESC);
