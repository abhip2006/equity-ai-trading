# Battle System Database Schema Documentation

This directory contains the SQL schemas for the trading battle system's metrics, leaderboard, and decision logging databases.

## Database Files

The battle system uses SQLite databases stored in `data/battle/`:

- `metrics_timeseries.db` - Metrics and performance timeseries data
- `leaderboard.db` - Rankings and competitive standings (can share same DB as metrics)
- `decisions.db` - Decision log with full reasoning for transparency

## Schema Files

### metrics_schema.sql

Creates tables for storing performance metrics and timeseries data:

#### Tables

**hourly_snapshots**
- Stores equity snapshots on an hourly basis
- Fields: equity, cash, positions_value, positions_count
- Indexed by model_id and timestamp
- Used for equity curves and drawdown calculations

**daily_summaries**
- End-of-day summary metrics
- Fields: opening/closing/high/low equity, daily P&L, trades opened/closed
- Indexed by model_id and date
- Used for daily performance reports

**trade_log**
- All executed trades with entry/exit details
- Fields: entry/exit price/time, shares, realized P&L, exit reason
- Indexed by model_id, symbol, and exit_time
- Used for win rate, profit factor calculations

**decision_log** (also in decision_schema.sql)
- All trading decisions with reasoning
- Links to execution status
- Used for transparency and decision analysis

**weekly_summaries** and **monthly_summaries**
- Aggregated performance metrics by week/month
- Pre-calculated for faster leaderboard queries
- Fields: return %, trades, win rate, risk metrics

#### Indexes

All tables have composite indexes for efficient queries:
- `(model_id, timestamp)` - Fast per-model timeseries queries
- `(timestamp)` - Fast all-models historical queries
- `(symbol, timestamp)` - Fast symbol-specific queries

### decision_schema.sql

Creates the decision log table with transparency focus:

#### Tables

**decision_log**
- Every decision logged with full LLM reasoning
- Fields:
  - Core: model_id, symbol, decision, reasoning, confidence
  - Context: timestamp, current_price, position_size
  - Trade params: entry_price, stop_loss, take_profit, direction
  - Execution: executed, execution_price/time/status

#### Views

**decision_comparison**
- Compares decisions made by different models on the same symbol within 5-minute windows
- Useful for analyzing decision divergence
- Shows confidence levels and time differences

#### Indexes

- `(model_id, timestamp)` - Fast model decision history
- `(symbol, timestamp)` - Fast symbol decision history
- `(model_id, symbol, timestamp)` - Fast model-symbol queries
- `(executed, timestamp)` - Fast execution filtering

### leaderboard_schema.sql

Creates ranking and standings tables:

#### Tables

**leaderboard_snapshots**
- Historical leaderboard positions
- Captures rank at different timeframes (daily/weekly/monthly/all-time)
- Fields: rank, all performance metrics
- Indexed by timeframe and snapshot_time

**current_standings**
- Real-time standings (single row per model)
- Updated after every trade
- Fast queries for current leaderboard
- Indexed by total_return_pct and sharpe_ratio for sorting

#### Views

**rank_history**
- Tracks rank changes over time
- Shows previous rank and rank delta
- Useful for rank progression charts

**top_performers**
- Pre-filtered view of top 10 models by timeframe
- Latest snapshot for each timeframe
- Fast queries for dashboard display

## Database Design Decisions

### Normalization vs. Denormalization

**Normalized:**
- Core data (trades, decisions) stored once
- Avoids data duplication

**Denormalized:**
- Summaries (daily/weekly/monthly) pre-aggregated
- Leaderboard snapshots store full metrics
- Trade-off: storage for query speed

### Indexing Strategy

All tables indexed for common query patterns:
1. Time-series queries (model + timestamp)
2. Cross-model comparisons (symbol + timestamp)
3. Leaderboard sorting (return %, Sharpe, drawdown)

### Data Integrity

Constraints ensure data quality:
- CHECK constraints on valid ranges (prices > 0, percentages 0-100)
- UNIQUE constraints prevent duplicates (model + timestamp)
- Foreign key relationships where applicable

### Scaling Considerations

Current design handles:
- Millions of decisions/trades
- Hourly snapshots for years
- Fast queries (<100ms) with proper indexes

For very large scale (>10M rows):
- Consider partitioning by date
- Archive old snapshots to separate DB
- Use materialized views for complex aggregations

## Usage Examples

### Recording Equity Snapshot

```python
from active_trader_llm.battle import MetricsEngine

engine = MetricsEngine("data/battle/metrics_timeseries.db")
engine.record_equity_snapshot(
    model_id="claude-sonnet-3.5",
    equity=105000.0,
    cash=50000.0,
    positions_value=55000.0,
    positions_count=3
)
```

### Logging Decision

```python
from active_trader_llm.battle import DecisionLogger

logger = DecisionLogger("data/battle/decisions.db")
decision_id = logger.log_decision(
    model_id="gpt-4o",
    symbol="AAPL",
    decision="buy",
    reasoning="Strong momentum breakout above $175 resistance...",
    confidence=85.0,
    entry_price=175.50,
    stop_loss=172.00,
    take_profit=180.00
)
```

### Updating Leaderboard

```python
from active_trader_llm.battle import Leaderboard, MetricsEngine

metrics = MetricsEngine("data/battle/metrics_timeseries.db")
leaderboard = Leaderboard("data/battle/metrics_timeseries.db")

# Get comprehensive metrics
snapshot = metrics.get_all_metrics(
    model_id="claude-sonnet-3.5",
    current_equity=105000,
    initial_capital=100000,
    trades=trade_list
)

# Update standings
leaderboard.update_standings(snapshot)

# Get rankings
rankings = leaderboard.get_rankings('all_time')
for entry in rankings:
    print(f"#{entry.rank} {entry.model_id}: {entry.total_return_pct:.2f}%")
```

### Querying Metrics

```python
# Get equity curve for Sharpe calculation
equity_curve = engine.get_equity_curve("claude-sonnet-3.5", limit=252)

# Get returns series
returns = engine.get_returns_series("claude-sonnet-3.5", limit=252)

# Calculate Sharpe ratio
sharpe = engine.calculate_sharpe_ratio(returns)
```

### Decision Analysis

```python
# Compare decisions across models for same symbol
decisions = logger.get_symbol_decisions("AAPL", since=datetime.now() - timedelta(days=1))

for model_id, model_decisions in decisions.items():
    print(f"\n{model_id}:")
    for decision in model_decisions:
        print(f"  {decision.decision} @ {decision.timestamp}")
        print(f"  Confidence: {decision.confidence}%")
        print(f"  Reasoning: {decision.reasoning[:100]}...")
```

## Maintenance

### Regular Tasks

1. **Snapshot Leaderboards** (hourly/daily/weekly)
   ```python
   leaderboard.snapshot_leaderboard('daily')
   ```

2. **Archive Old Data** (monthly)
   - Export old snapshots to backup DB
   - Delete snapshots older than 1 year from main DB

3. **Vacuum Database** (weekly)
   ```bash
   sqlite3 data/battle/metrics_timeseries.db "VACUUM;"
   ```

4. **Analyze Statistics** (after bulk inserts)
   ```bash
   sqlite3 data/battle/metrics_timeseries.db "ANALYZE;"
   ```

### Backup Strategy

- Daily: Full database backup
- Real-time: Write-ahead logging (WAL mode)
- Disaster recovery: Replicate to S3/cloud storage

### Performance Monitoring

Monitor these metrics:
- Query execution time (aim for <100ms)
- Database file size growth
- Index usage (EXPLAIN QUERY PLAN)
- Write throughput (decisions/trades per second)

## Future Enhancements

Potential improvements:
1. **Partitioning** - Partition large tables by date range
2. **Compression** - Compress old snapshot data
3. **Materialized Views** - Pre-compute complex aggregations
4. **Sharding** - Distribute models across multiple databases
5. **Time-series DB** - Migrate metrics to specialized TSDB (InfluxDB, TimescaleDB)
