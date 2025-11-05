"""
Backtest tracking models and database initialization.

This module provides Pydantic models for backtest management and
database initialization utilities for the backtest tracking system.
"""

import sqlite3
import logging
from datetime import datetime, date
from typing import Optional, Literal, Dict, Any, List
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BacktestRun(BaseModel):
    """Metadata for a backtest run."""
    id: Optional[int] = None
    run_name: Optional[str] = Field(None, description="Optional name for this backtest run")
    start_date: str = Field(..., description="Start date for backtest (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for backtest (YYYY-MM-DD)")
    initial_capital: float = Field(..., gt=0, description="Starting capital")
    final_capital: Optional[float] = Field(None, description="Ending capital")
    total_return_pct: Optional[float] = Field(None, description="Total return percentage")
    config_snapshot: Optional[str] = Field(None, description="JSON snapshot of config")
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: Literal['running', 'completed', 'failed'] = Field(default='running')
    error_message: Optional[str] = None


class DailyEquity(BaseModel):
    """Daily equity curve point."""
    id: Optional[int] = None
    backtest_run_id: int = Field(..., description="Foreign key to backtest run")
    date: str = Field(..., description="Date (YYYY-MM-DD)")
    equity: float = Field(..., ge=0, description="Total portfolio equity")
    cash: float = Field(..., ge=0, description="Cash balance")
    positions_value: float = Field(default=0, ge=0, description="Total value of open positions")
    daily_return_pct: Optional[float] = Field(None, description="Daily return percentage")
    cumulative_return_pct: Optional[float] = Field(None, description="Cumulative return from start")
    num_open_positions: int = Field(default=0, description="Number of open positions")
    benchmark_value: Optional[float] = Field(None, description="Benchmark value for comparison")


class BacktestMetrics(BaseModel):
    """Complete performance metrics for a backtest run."""
    id: Optional[int] = None
    backtest_run_id: int = Field(..., description="Foreign key to backtest run")

    # Return metrics
    total_return_pct: Optional[float] = None
    cagr_pct: Optional[float] = None

    # Risk metrics
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    max_drawdown_duration_days: Optional[int] = None
    calmar_ratio: Optional[float] = None
    volatility_annualized_pct: Optional[float] = None
    downside_deviation_pct: Optional[float] = None

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate_pct: Optional[float] = None
    profit_factor: Optional[float] = None
    avg_win_pct: Optional[float] = None
    avg_loss_pct: Optional[float] = None
    avg_win_dollars: Optional[float] = None
    avg_loss_dollars: Optional[float] = None
    largest_win_dollars: Optional[float] = None
    largest_loss_dollars: Optional[float] = None
    avg_hold_time_days: Optional[float] = None
    max_consecutive_wins: Optional[int] = None
    max_consecutive_losses: Optional[int] = None

    # Risk-adjusted metrics
    recovery_factor: Optional[float] = None
    ulcer_index: Optional[float] = None

    # Benchmark comparison
    benchmark_return_pct: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None

    created_at: datetime = Field(default_factory=datetime.now)


class AnalystSignal(BaseModel):
    """Analyst signal for a specific symbol and date."""
    id: Optional[int] = None
    backtest_run_id: int = Field(..., description="Foreign key to backtest run")
    date: str = Field(..., description="Date (YYYY-MM-DD)")
    symbol: str = Field(..., description="Stock symbol")
    analyst_type: Literal['technical', 'breadth', 'sentiment', 'macro']
    signal: Optional[Literal['long', 'short', 'neutral']] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)
    reasons: Optional[str] = Field(None, description="JSON array of reasons")
    timeframe: Optional[str] = Field(None, description="Trading timeframe")
    full_output: Optional[str] = Field(None, description="Complete analyst output as JSON")
    created_at: datetime = Field(default_factory=datetime.now)


class ResearcherDebate(BaseModel):
    """Bull vs Bear debate for a specific symbol and date."""
    id: Optional[int] = None
    backtest_run_id: int = Field(..., description="Foreign key to backtest run")
    date: str = Field(..., description="Date (YYYY-MM-DD)")
    symbol: str = Field(..., description="Stock symbol")
    bull_thesis: Optional[str] = None
    bear_thesis: Optional[str] = None
    bull_strength: Optional[float] = Field(None, ge=0, le=1)
    bear_strength: Optional[float] = Field(None, ge=0, le=1)
    full_output: Optional[str] = Field(None, description="Complete debate output as JSON")
    created_at: datetime = Field(default_factory=datetime.now)


class TraderPlan(BaseModel):
    """Final trade plan from trader agent."""
    id: Optional[int] = None
    backtest_run_id: int = Field(..., description="Foreign key to backtest run")
    date: str = Field(..., description="Date (YYYY-MM-DD)")
    symbol: str = Field(..., description="Stock symbol")
    action: Literal['enter', 'exit', 'hold', 'none']
    direction: Optional[Literal['long', 'short']] = None
    entry_price: Optional[float] = Field(None, gt=0)
    stop_loss: Optional[float] = Field(None, gt=0)
    take_profit: Optional[float] = Field(None, gt=0)
    position_size_pct: Optional[float] = Field(None, gt=0)
    shares: Optional[int] = Field(None, gt=0)
    strategy: Optional[str] = None
    rationale: Optional[str] = None
    risk_reward_ratio: Optional[float] = None
    validation_passed: bool = False
    validation_errors: Optional[str] = Field(None, description="JSON array of validation errors")
    executed: bool = False
    execution_price: Optional[float] = None
    execution_reason: Optional[str] = None
    full_output: Optional[str] = Field(None, description="Complete trade plan as JSON")
    created_at: datetime = Field(default_factory=datetime.now)


class LLMInteraction(BaseModel):
    """LLM prompt and response for tracking."""
    id: Optional[int] = None
    backtest_run_id: int = Field(..., description="Foreign key to backtest run")
    date: str = Field(..., description="Date (YYYY-MM-DD)")
    symbol: Optional[str] = None
    interaction_type: str = Field(..., description="Type of interaction")
    agent_name: Optional[str] = Field(None, description="Specific agent name")
    prompt: str = Field(..., description="Prompt sent to LLM")
    response: str = Field(..., description="Response from LLM")
    model: Optional[str] = Field(None, description="Model used")
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_ms: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)


def init_backtest_db(db_path: str) -> None:
    """
    Initialize the backtest database by creating tables from schema.

    Args:
        db_path: Path to SQLite database file

    Raises:
        sqlite3.Error: If database initialization fails
    """
    # Ensure parent directory exists
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    # Get path to schema file
    schema_path = Path(__file__).parent / 'backtest_schema.sql'

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    # Read schema
    with open(schema_path, 'r') as f:
        schema_sql = f.read()

    # Initialize database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Execute schema (supports multiple statements)
        cursor.executescript(schema_sql)

        conn.commit()
        logger.info(f"Backtest database initialized successfully at {db_path}")

        # Verify tables were created
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'backtest%' OR name LIKE '%analyst%' OR name LIKE '%researcher%' OR name LIKE '%trader%' OR name LIKE '%llm%' OR name = 'daily_equity'")
        tables = cursor.fetchall()
        logger.info(f"Created backtest tables: {[t[0] for t in tables]}")

    except sqlite3.Error as e:
        logger.error(f"Backtest database initialization failed: {e}")
        raise
    finally:
        conn.close()


class BacktestDatabase:
    """
    Helper class for managing backtest database operations.

    Provides high-level methods for inserting and querying backtest data.
    """

    def __init__(self, db_path: str):
        """
        Initialize backtest database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        init_backtest_db(db_path)

    def create_run(self, run: BacktestRun) -> int:
        """Create a new backtest run and return its ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO backtest_runs
            (run_name, start_date, end_date, initial_capital, config_snapshot, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (run.run_name, run.start_date, run.end_date, run.initial_capital,
              run.config_snapshot, run.status))

        run_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return run_id

    def update_run(self, run_id: int, **kwargs) -> None:
        """Update backtest run fields."""
        if not kwargs:
            return

        # Whitelist of allowed column names to prevent SQL injection
        allowed_columns = {
            'run_name', 'start_date', 'end_date', 'initial_capital',
            'final_capital', 'total_return_pct', 'config_snapshot',
            'completed_at', 'status', 'error_message'
        }

        # Validate all column names against whitelist
        for column_name in kwargs.keys():
            if column_name not in allowed_columns:
                raise ValueError(f"Invalid column name: {column_name}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        set_clause = ", ".join([f"{k} = ?" for k in kwargs.keys()])
        values = list(kwargs.values()) + [run_id]

        cursor.execute(f"UPDATE backtest_runs SET {set_clause} WHERE id = ?", values)
        conn.commit()
        conn.close()

    def insert_daily_equity(self, equity: DailyEquity) -> int:
        """Insert daily equity record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO daily_equity
            (backtest_run_id, date, equity, cash, positions_value, daily_return_pct,
             cumulative_return_pct, num_open_positions, benchmark_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (equity.backtest_run_id, equity.date, equity.equity, equity.cash,
              equity.positions_value, equity.daily_return_pct, equity.cumulative_return_pct,
              equity.num_open_positions, equity.benchmark_value))

        equity_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return equity_id

    def insert_metrics(self, metrics: BacktestMetrics) -> int:
        """Insert backtest metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO backtest_metrics
            (backtest_run_id, total_return_pct, cagr_pct, sharpe_ratio, sortino_ratio,
             max_drawdown_pct, max_drawdown_duration_days, calmar_ratio,
             volatility_annualized_pct, downside_deviation_pct,
             total_trades, winning_trades, losing_trades, win_rate_pct, profit_factor,
             avg_win_pct, avg_loss_pct, avg_win_dollars, avg_loss_dollars,
             largest_win_dollars, largest_loss_dollars, avg_hold_time_days,
             max_consecutive_wins, max_consecutive_losses,
             recovery_factor, ulcer_index, benchmark_return_pct, alpha, beta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.backtest_run_id, metrics.total_return_pct, metrics.cagr_pct,
            metrics.sharpe_ratio, metrics.sortino_ratio, metrics.max_drawdown_pct,
            metrics.max_drawdown_duration_days, metrics.calmar_ratio,
            metrics.volatility_annualized_pct, metrics.downside_deviation_pct,
            metrics.total_trades, metrics.winning_trades, metrics.losing_trades,
            metrics.win_rate_pct, metrics.profit_factor, metrics.avg_win_pct,
            metrics.avg_loss_pct, metrics.avg_win_dollars, metrics.avg_loss_dollars,
            metrics.largest_win_dollars, metrics.largest_loss_dollars,
            metrics.avg_hold_time_days, metrics.max_consecutive_wins,
            metrics.max_consecutive_losses, metrics.recovery_factor, metrics.ulcer_index,
            metrics.benchmark_return_pct, metrics.alpha, metrics.beta
        ))

        metrics_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return metrics_id

    def insert_analyst_signal(self, signal: AnalystSignal) -> int:
        """Insert analyst signal."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO analyst_signals
            (backtest_run_id, date, symbol, analyst_type, signal, confidence,
             reasons, timeframe, full_output)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (signal.backtest_run_id, signal.date, signal.symbol, signal.analyst_type,
              signal.signal, signal.confidence, signal.reasons, signal.timeframe,
              signal.full_output))

        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return signal_id

    def insert_researcher_debate(self, debate: ResearcherDebate) -> int:
        """Insert researcher debate."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO researcher_debates
            (backtest_run_id, date, symbol, bull_thesis, bear_thesis,
             bull_strength, bear_strength, full_output)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (debate.backtest_run_id, debate.date, debate.symbol, debate.bull_thesis,
              debate.bear_thesis, debate.bull_strength, debate.bear_strength,
              debate.full_output))

        debate_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return debate_id

    def insert_trader_plan(self, plan: TraderPlan) -> int:
        """Insert trader plan."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO trader_plans
            (backtest_run_id, date, symbol, action, direction, entry_price,
             stop_loss, take_profit, position_size_pct, shares, strategy,
             rationale, risk_reward_ratio, validation_passed, validation_errors,
             executed, execution_price, execution_reason, full_output)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            plan.backtest_run_id, plan.date, plan.symbol, plan.action, plan.direction,
            plan.entry_price, plan.stop_loss, plan.take_profit, plan.position_size_pct,
            plan.shares, plan.strategy, plan.rationale, plan.risk_reward_ratio,
            plan.validation_passed, plan.validation_errors, plan.executed,
            plan.execution_price, plan.execution_reason, plan.full_output
        ))

        plan_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return plan_id

    def insert_llm_interaction(self, interaction: LLMInteraction) -> int:
        """Insert LLM interaction."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO llm_interactions
            (backtest_run_id, date, symbol, interaction_type, agent_name,
             prompt, response, model, tokens_input, tokens_output,
             cost_usd, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            interaction.backtest_run_id, interaction.date, interaction.symbol,
            interaction.interaction_type, interaction.agent_name, interaction.prompt,
            interaction.response, interaction.model, interaction.tokens_input,
            interaction.tokens_output, interaction.cost_usd, interaction.latency_ms
        ))

        interaction_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return interaction_id


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import tempfile
    import os

    print("=" * 60)
    print("Testing Backtest Database")
    print("=" * 60)

    # Create temp database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test_backtest.db')
        print(f"\nInitializing test database at: {db_path}")

        # Initialize database
        db = BacktestDatabase(db_path)

        # Create a backtest run
        run = BacktestRun(
            run_name="Test Run",
            start_date="2024-01-01",
            end_date="2024-01-05",
            initial_capital=100000.0,
            config_snapshot='{"mode": "backtest"}'
        )

        run_id = db.create_run(run)
        print(f"\nCreated backtest run with ID: {run_id}")

        # Insert daily equity
        equity = DailyEquity(
            backtest_run_id=run_id,
            date="2024-01-01",
            equity=100000.0,
            cash=100000.0,
            positions_value=0.0
        )
        equity_id = db.insert_daily_equity(equity)
        print(f"Inserted daily equity with ID: {equity_id}")

        # Insert analyst signal
        signal = AnalystSignal(
            backtest_run_id=run_id,
            date="2024-01-01",
            symbol="AAPL",
            analyst_type="technical",
            signal="long",
            confidence=0.8
        )
        signal_id = db.insert_analyst_signal(signal)
        print(f"Inserted analyst signal with ID: {signal_id}")

        # Update run to completed
        db.update_run(run_id, status='completed', final_capital=105000.0)
        print(f"Updated run {run_id} to completed status")

        # Verify data
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM backtest_runs WHERE id = ?", (run_id,))
        run_row = cursor.fetchone()
        print(f"\nBacktest run: {run_row[1]} ({run_row[10]})")

        cursor.execute("SELECT COUNT(*) FROM analyst_signals WHERE backtest_run_id = ?", (run_id,))
        signal_count = cursor.fetchone()[0]
        print(f"Analyst signals: {signal_count}")

        conn.close()

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
