-- Position tracking database schema
-- This schema tracks both open and closed positions for the trading system

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('long', 'short')),
    entry_price REAL NOT NULL CHECK(entry_price > 0),
    stop_loss REAL NOT NULL CHECK(stop_loss > 0),
    take_profit REAL NOT NULL CHECK(take_profit > 0),
    shares INTEGER NOT NULL CHECK(shares > 0),
    position_size_pct REAL CHECK(position_size_pct > 0 AND position_size_pct <= 100),
    strategy TEXT,
    opened_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    exit_price REAL CHECK(exit_price > 0),
    exit_reason TEXT CHECK(exit_reason IN ('stop_loss', 'take_profit', 'manual', 'eod_close', NULL)),
    realized_pnl REAL,
    status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open', 'closed'))
);

-- Index for querying positions by symbol
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);

-- Index for querying positions by status (open vs closed)
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);

-- Index for querying positions by opened_at timestamp
CREATE INDEX IF NOT EXISTS idx_positions_opened_at ON positions(opened_at);

-- Composite index for common queries (symbol + status)
CREATE INDEX IF NOT EXISTS idx_positions_symbol_status ON positions(symbol, status);

-- Index for querying positions by strategy
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy);
