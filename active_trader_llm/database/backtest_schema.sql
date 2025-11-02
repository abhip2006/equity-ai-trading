-- Backtest database schema
-- This schema stores all backtesting data including runs, metrics, signals, debates, and LLM interactions

-- Backtest run metadata
CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name TEXT,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    initial_capital REAL NOT NULL CHECK(initial_capital > 0),
    final_capital REAL,
    total_return_pct REAL,
    config_snapshot TEXT,  -- JSON snapshot of config used
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'running' CHECK(status IN ('running', 'completed', 'failed')),
    error_message TEXT
);

-- Daily equity curve tracking
CREATE TABLE IF NOT EXISTS daily_equity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_run_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    equity REAL NOT NULL CHECK(equity >= 0),
    cash REAL NOT NULL CHECK(cash >= 0),
    positions_value REAL NOT NULL DEFAULT 0,
    daily_return_pct REAL,
    cumulative_return_pct REAL,
    num_open_positions INTEGER NOT NULL DEFAULT 0,
    benchmark_value REAL,  -- SPY value for comparison
    FOREIGN KEY (backtest_run_id) REFERENCES backtest_runs(id) ON DELETE CASCADE
);

-- Performance metrics for each backtest run
CREATE TABLE IF NOT EXISTS backtest_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_run_id INTEGER NOT NULL,

    -- Return metrics
    total_return_pct REAL,
    cagr_pct REAL,

    -- Risk metrics
    sharpe_ratio REAL,
    sortino_ratio REAL,
    max_drawdown_pct REAL,
    max_drawdown_duration_days INTEGER,
    calmar_ratio REAL,
    volatility_annualized_pct REAL,
    downside_deviation_pct REAL,

    -- Trade statistics
    total_trades INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades INTEGER NOT NULL DEFAULT 0,
    win_rate_pct REAL,
    profit_factor REAL,
    avg_win_pct REAL,
    avg_loss_pct REAL,
    avg_win_dollars REAL,
    avg_loss_dollars REAL,
    largest_win_dollars REAL,
    largest_loss_dollars REAL,
    avg_hold_time_days REAL,
    max_consecutive_wins INTEGER,
    max_consecutive_losses INTEGER,

    -- Risk-adjusted metrics
    recovery_factor REAL,  -- Net profit / max drawdown
    ulcer_index REAL,

    -- Benchmark comparison
    benchmark_return_pct REAL,
    alpha REAL,
    beta REAL,

    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (backtest_run_id) REFERENCES backtest_runs(id) ON DELETE CASCADE
);

-- Analyst signals for each decision cycle
CREATE TABLE IF NOT EXISTS analyst_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_run_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    analyst_type TEXT NOT NULL CHECK(analyst_type IN ('technical', 'breadth', 'sentiment', 'macro')),
    signal TEXT CHECK(signal IN ('long', 'short', 'neutral', NULL)),
    confidence REAL CHECK(confidence >= 0 AND confidence <= 1),
    reasons TEXT,  -- JSON array of reasons
    timeframe TEXT,  -- e.g., 'intraday', 'swing', 'position'
    full_output TEXT,  -- Complete analyst output (JSON)
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (backtest_run_id) REFERENCES backtest_runs(id) ON DELETE CASCADE
);

-- Researcher debates (bull vs bear)
CREATE TABLE IF NOT EXISTS researcher_debates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_run_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    bull_thesis TEXT,
    bear_thesis TEXT,
    bull_strength REAL CHECK(bull_strength >= 0 AND bull_strength <= 1),
    bear_strength REAL CHECK(bear_strength >= 0 AND bear_strength <= 1),
    full_output TEXT,  -- Complete debate output (JSON)
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (backtest_run_id) REFERENCES backtest_runs(id) ON DELETE CASCADE
);

-- Trader plans (final trade decisions)
CREATE TABLE IF NOT EXISTS trader_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_run_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL CHECK(action IN ('enter', 'exit', 'hold', 'none')),
    direction TEXT CHECK(direction IN ('long', 'short', NULL)),
    entry_price REAL CHECK(entry_price > 0 OR entry_price IS NULL),
    stop_loss REAL CHECK(stop_loss > 0 OR stop_loss IS NULL),
    take_profit REAL CHECK(take_profit > 0 OR take_profit IS NULL),
    position_size_pct REAL CHECK(position_size_pct > 0 OR position_size_pct IS NULL),
    shares INTEGER CHECK(shares > 0 OR shares IS NULL),
    strategy TEXT,
    rationale TEXT,
    risk_reward_ratio REAL,
    validation_passed BOOLEAN NOT NULL DEFAULT 0,
    validation_errors TEXT,  -- JSON array of validation errors if any
    executed BOOLEAN NOT NULL DEFAULT 0,
    execution_price REAL,
    execution_reason TEXT,  -- 'filled', 'rejected', 'skipped', etc.
    full_output TEXT,  -- Complete trade plan output (JSON)
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (backtest_run_id) REFERENCES backtest_runs(id) ON DELETE CASCADE
);

-- LLM interactions (prompts and responses)
CREATE TABLE IF NOT EXISTS llm_interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_run_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    symbol TEXT,
    interaction_type TEXT NOT NULL,  -- 'analyst', 'researcher', 'trader', 'risk'
    agent_name TEXT,  -- specific agent name
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    model TEXT,  -- e.g., 'claude-3-5-sonnet-20241022'
    tokens_input INTEGER,
    tokens_output INTEGER,
    cost_usd REAL,
    latency_ms INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (backtest_run_id) REFERENCES backtest_runs(id) ON DELETE CASCADE
);

-- Indexes for efficient querying

-- backtest_runs indexes
CREATE INDEX IF NOT EXISTS idx_backtest_runs_status ON backtest_runs(status);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_dates ON backtest_runs(start_date, end_date);

-- daily_equity indexes
CREATE INDEX IF NOT EXISTS idx_daily_equity_run ON daily_equity(backtest_run_id);
CREATE INDEX IF NOT EXISTS idx_daily_equity_date ON daily_equity(date);
CREATE INDEX IF NOT EXISTS idx_daily_equity_run_date ON daily_equity(backtest_run_id, date);

-- backtest_metrics indexes
CREATE INDEX IF NOT EXISTS idx_backtest_metrics_run ON backtest_metrics(backtest_run_id);

-- analyst_signals indexes
CREATE INDEX IF NOT EXISTS idx_analyst_signals_run ON analyst_signals(backtest_run_id);
CREATE INDEX IF NOT EXISTS idx_analyst_signals_symbol ON analyst_signals(symbol);
CREATE INDEX IF NOT EXISTS idx_analyst_signals_date ON analyst_signals(date);
CREATE INDEX IF NOT EXISTS idx_analyst_signals_type ON analyst_signals(analyst_type);

-- researcher_debates indexes
CREATE INDEX IF NOT EXISTS idx_researcher_debates_run ON researcher_debates(backtest_run_id);
CREATE INDEX IF NOT EXISTS idx_researcher_debates_symbol ON researcher_debates(symbol);
CREATE INDEX IF NOT EXISTS idx_researcher_debates_date ON researcher_debates(date);

-- trader_plans indexes
CREATE INDEX IF NOT EXISTS idx_trader_plans_run ON trader_plans(backtest_run_id);
CREATE INDEX IF NOT EXISTS idx_trader_plans_symbol ON trader_plans(symbol);
CREATE INDEX IF NOT EXISTS idx_trader_plans_date ON trader_plans(date);
CREATE INDEX IF NOT EXISTS idx_trader_plans_action ON trader_plans(action);
CREATE INDEX IF NOT EXISTS idx_trader_plans_executed ON trader_plans(executed);

-- llm_interactions indexes
CREATE INDEX IF NOT EXISTS idx_llm_interactions_run ON llm_interactions(backtest_run_id);
CREATE INDEX IF NOT EXISTS idx_llm_interactions_type ON llm_interactions(interaction_type);
CREATE INDEX IF NOT EXISTS idx_llm_interactions_date ON llm_interactions(date);
