-- Leaderboard Database Schema
-- Stores rankings and competitive standings

-- Historical leaderboard snapshots
CREATE TABLE IF NOT EXISTS leaderboard_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    timeframe TEXT NOT NULL CHECK(timeframe IN ('daily', 'weekly', 'monthly', 'all_time')),
    snapshot_time TIMESTAMP NOT NULL,
    rank INTEGER NOT NULL CHECK(rank >= 1),

    -- Performance metrics
    total_return_pct REAL NOT NULL,
    total_return_dollars REAL NOT NULL,
    current_equity REAL NOT NULL CHECK(current_equity > 0),

    -- Risk metrics
    sharpe_ratio REAL,
    max_drawdown_pct REAL,

    -- Trading metrics
    total_trades INTEGER NOT NULL CHECK(total_trades >= 0),
    win_rate REAL,
    profit_factor REAL,

    -- Position metrics
    open_positions_count INTEGER NOT NULL CHECK(open_positions_count >= 0),
    total_exposure_pct REAL NOT NULL CHECK(total_exposure_pct >= 0),

    UNIQUE(model_id, timeframe, snapshot_time)
);

CREATE INDEX IF NOT EXISTS idx_leaderboard_timeframe_time ON leaderboard_snapshots(timeframe, snapshot_time DESC);
CREATE INDEX IF NOT EXISTS idx_leaderboard_model_timeframe ON leaderboard_snapshots(model_id, timeframe, snapshot_time DESC);
CREATE INDEX IF NOT EXISTS idx_leaderboard_rank ON leaderboard_snapshots(timeframe, rank, snapshot_time DESC);

-- Current standings (real-time updated)
CREATE TABLE IF NOT EXISTS current_standings (
    model_id TEXT PRIMARY KEY,

    -- Performance metrics
    total_return_pct REAL NOT NULL,
    total_return_dollars REAL NOT NULL,
    current_equity REAL NOT NULL CHECK(current_equity > 0),

    -- Risk metrics
    sharpe_ratio REAL,
    max_drawdown_pct REAL,

    -- Trading metrics
    total_trades INTEGER NOT NULL CHECK(total_trades >= 0),
    win_rate REAL,
    profit_factor REAL,

    -- Position metrics
    open_positions_count INTEGER NOT NULL CHECK(open_positions_count >= 0),
    total_exposure_pct REAL NOT NULL CHECK(total_exposure_pct >= 0),

    -- Metadata
    last_updated TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_standings_return ON current_standings(total_return_pct DESC);
CREATE INDEX IF NOT EXISTS idx_standings_sharpe ON current_standings(sharpe_ratio DESC);

-- Rank history view (for tracking rank changes over time)
CREATE VIEW IF NOT EXISTS rank_history AS
SELECT
    model_id,
    timeframe,
    snapshot_time,
    rank,
    total_return_pct,
    LAG(rank) OVER (PARTITION BY model_id, timeframe ORDER BY snapshot_time) as previous_rank,
    rank - LAG(rank) OVER (PARTITION BY model_id, timeframe ORDER BY snapshot_time) as rank_change
FROM leaderboard_snapshots
ORDER BY snapshot_time DESC;

-- Top performers view (top 10 by timeframe)
CREATE VIEW IF NOT EXISTS top_performers AS
SELECT
    ls.model_id,
    ls.timeframe,
    ls.rank,
    ls.total_return_pct,
    ls.sharpe_ratio,
    ls.max_drawdown_pct,
    ls.snapshot_time
FROM leaderboard_snapshots ls
INNER JOIN (
    SELECT timeframe, MAX(snapshot_time) as latest_time
    FROM leaderboard_snapshots
    GROUP BY timeframe
) latest ON ls.timeframe = latest.timeframe AND ls.snapshot_time = latest.latest_time
WHERE ls.rank <= 10
ORDER BY ls.timeframe, ls.rank;
