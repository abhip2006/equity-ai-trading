"""
Metrics Engine: Calculates comprehensive trading performance metrics.

This module handles all performance metric calculations:
- Total return and P&L
- Sharpe ratio (risk-adjusted returns)
- Maximum drawdown
- Win rate and profit factor
- Daily P&L changes
- Comprehensive performance snapshots
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MetricsSnapshot(BaseModel):
    """Comprehensive performance metrics snapshot"""
    model_id: str
    timestamp: datetime

    # Equity metrics
    current_equity: float = Field(gt=0)
    initial_capital: float = Field(gt=0)
    total_return_pct: float  # Percentage return
    total_return_dollars: float  # Dollar return

    # Risk metrics
    sharpe_ratio: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    max_drawdown_dollars: Optional[float] = None

    # Trading metrics
    total_trades: int = Field(ge=0)
    winning_trades: int = Field(ge=0)
    losing_trades: int = Field(ge=0)
    win_rate: Optional[float] = None  # 0-100 percentage
    profit_factor: Optional[float] = None  # Gross profit / Gross loss

    # Position metrics
    open_positions_count: int = Field(ge=0)
    total_exposure_dollars: float = Field(ge=0)
    total_exposure_pct: float = Field(ge=0)  # 0-100 percentage

    # Daily metrics
    daily_pnl: Optional[float] = None
    daily_return_pct: Optional[float] = None


class EquitySnapshot(BaseModel):
    """Equity value at a point in time"""
    timestamp: datetime
    equity: float = Field(gt=0)
    cash: float = Field(ge=0)
    positions_value: float = Field(ge=0)
    positions_count: int = Field(ge=0)


class TradeRecord(BaseModel):
    """Individual trade record for metrics calculation"""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    shares: int
    realized_pnl: float
    opened_at: datetime
    closed_at: datetime
    exit_reason: str


class MetricsEngine:
    """
    Calculates comprehensive trading performance metrics.

    Handles:
    - Return calculations (total, daily)
    - Risk metrics (Sharpe, drawdown)
    - Trading metrics (win rate, profit factor)
    - Real-time metric updates
    """

    def __init__(self, db_path: str):
        """
        Initialize MetricsEngine with database connection.

        Args:
            db_path: Path to metrics timeseries database
        """
        self.db_path = db_path
        self._ensure_database()
        logger.info(f"MetricsEngine initialized with database: {db_path}")

    def _ensure_database(self):
        """Create database tables if they don't exist"""
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Read schema if available
        schema_path = Path(__file__).parent / 'schema' / 'metrics_schema.sql'

        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            cursor.executescript(schema_sql)
            logger.info("Created metrics database from schema file")
        else:
            # Fallback inline schema (minimal for now, full schema in separate file)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hourly_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    equity REAL NOT NULL CHECK(equity > 0),
                    cash REAL NOT NULL CHECK(cash >= 0),
                    positions_value REAL NOT NULL CHECK(positions_value >= 0),
                    positions_count INTEGER NOT NULL CHECK(positions_count >= 0),
                    UNIQUE(model_id, timestamp)
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_hourly_model_time
                ON hourly_snapshots(model_id, timestamp DESC)
            """)

            logger.info("Created metrics database with inline schema")

        conn.commit()
        conn.close()

    def calculate_total_return(self, equity: float, initial_capital: float) -> float:
        """
        Calculate total return percentage.

        Args:
            equity: Current equity value
            initial_capital: Starting capital

        Returns:
            Total return as percentage (e.g., 15.5 for 15.5%)
        """
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive")

        return_pct = ((equity - initial_capital) / initial_capital) * 100
        return return_pct

    def calculate_sharpe_ratio(
        self,
        returns_series: List[float],
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> Optional[float]:
        """
        Calculate Sharpe ratio (risk-adjusted returns).

        Formula: Sharpe = (mean_return - risk_free_rate) / std_return * sqrt(periods_per_year)

        Args:
            returns_series: List of period returns (e.g., daily returns as decimals)
            risk_free_rate: Annual risk-free rate (default 0.0)
            periods_per_year: Trading periods per year (252 for daily, 52 for weekly)

        Returns:
            Sharpe ratio or None if insufficient data
        """
        if len(returns_series) < 2:
            return None

        returns_array = np.array(returns_series)

        # Calculate mean and std of returns
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)  # Sample std

        if std_return == 0:
            return None  # No volatility

        # Annualized Sharpe ratio
        sharpe = (mean_return - risk_free_rate / periods_per_year) / std_return * np.sqrt(periods_per_year)

        return float(sharpe)

    def calculate_max_drawdown(self, equity_curve: List[float]) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate maximum drawdown from equity curve.

        Args:
            equity_curve: List of equity values over time

        Returns:
            Tuple of (max_drawdown_pct, max_drawdown_dollars) or (None, None) if insufficient data
        """
        if len(equity_curve) < 2:
            return None, None

        equity_array = np.array(equity_curve)

        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_array)

        # Calculate drawdown at each point
        drawdown_dollars = running_max - equity_array

        # Maximum drawdown
        max_dd_dollars = float(np.max(drawdown_dollars))

        # Maximum drawdown percentage
        # Find the peak before the maximum drawdown
        max_dd_idx = np.argmax(drawdown_dollars)
        peak_value = running_max[max_dd_idx]

        if peak_value > 0:
            max_dd_pct = (max_dd_dollars / peak_value) * 100
        else:
            max_dd_pct = 0.0

        return max_dd_pct, max_dd_dollars

    def calculate_win_rate(self, trades: List[TradeRecord]) -> Optional[float]:
        """
        Calculate win rate from closed trades.

        Args:
            trades: List of TradeRecord objects

        Returns:
            Win rate as percentage (0-100) or None if no trades
        """
        if not trades:
            return None

        winning_trades = sum(1 for trade in trades if trade.realized_pnl > 0)
        total_trades = len(trades)

        win_rate = (winning_trades / total_trades) * 100
        return win_rate

    def calculate_profit_factor(self, trades: List[TradeRecord]) -> Optional[float]:
        """
        Calculate profit factor (gross profit / gross loss).

        Args:
            trades: List of TradeRecord objects

        Returns:
            Profit factor or None if no losing trades
        """
        if not trades:
            return None

        gross_profit = sum(trade.realized_pnl for trade in trades if trade.realized_pnl > 0)
        gross_loss = abs(sum(trade.realized_pnl for trade in trades if trade.realized_pnl < 0))

        if gross_loss == 0:
            # All trades are winners or breakeven
            return None if gross_profit == 0 else float('inf')

        profit_factor = gross_profit / gross_loss
        return profit_factor

    def calculate_daily_pnl(self, current_equity: float, previous_equity: float) -> float:
        """
        Calculate daily P&L change.

        Args:
            current_equity: Current equity value
            previous_equity: Previous day equity value

        Returns:
            Daily P&L in dollars
        """
        return current_equity - previous_equity

    def calculate_daily_return_pct(self, current_equity: float, previous_equity: float) -> float:
        """
        Calculate daily return percentage.

        Args:
            current_equity: Current equity value
            previous_equity: Previous day equity value

        Returns:
            Daily return as percentage
        """
        if previous_equity <= 0:
            return 0.0

        return ((current_equity - previous_equity) / previous_equity) * 100

    def get_all_metrics(
        self,
        model_id: str,
        current_equity: float,
        initial_capital: float,
        trades: List[TradeRecord],
        open_positions_count: int = 0,
        total_exposure_dollars: float = 0.0,
        equity_curve: Optional[List[float]] = None,
        returns_series: Optional[List[float]] = None,
        previous_equity: Optional[float] = None
    ) -> MetricsSnapshot:
        """
        Get comprehensive metrics snapshot for a model.

        Args:
            model_id: Model identifier
            current_equity: Current equity value
            initial_capital: Starting capital
            trades: List of closed trades
            open_positions_count: Number of open positions
            total_exposure_dollars: Total dollar exposure
            equity_curve: Historical equity values for drawdown calculation
            returns_series: Historical returns for Sharpe calculation
            previous_equity: Previous equity for daily P&L

        Returns:
            MetricsSnapshot with all calculated metrics
        """
        # Calculate total return
        total_return_pct = self.calculate_total_return(current_equity, initial_capital)
        total_return_dollars = current_equity - initial_capital

        # Calculate risk metrics
        sharpe_ratio = None
        if returns_series and len(returns_series) >= 2:
            sharpe_ratio = self.calculate_sharpe_ratio(returns_series)

        max_drawdown_pct = None
        max_drawdown_dollars = None
        if equity_curve and len(equity_curve) >= 2:
            max_drawdown_pct, max_drawdown_dollars = self.calculate_max_drawdown(equity_curve)

        # Calculate trading metrics
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade.realized_pnl > 0)
        losing_trades = sum(1 for trade in trades if trade.realized_pnl < 0)

        win_rate = self.calculate_win_rate(trades) if trades else None
        profit_factor = self.calculate_profit_factor(trades) if trades else None

        # Calculate exposure percentage
        total_exposure_pct = (total_exposure_dollars / current_equity) * 100 if current_equity > 0 else 0.0

        # Calculate daily metrics
        daily_pnl = None
        daily_return_pct = None
        if previous_equity is not None:
            daily_pnl = self.calculate_daily_pnl(current_equity, previous_equity)
            daily_return_pct = self.calculate_daily_return_pct(current_equity, previous_equity)

        return MetricsSnapshot(
            model_id=model_id,
            timestamp=datetime.now(),
            current_equity=current_equity,
            initial_capital=initial_capital,
            total_return_pct=total_return_pct,
            total_return_dollars=total_return_dollars,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            max_drawdown_dollars=max_drawdown_dollars,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            open_positions_count=open_positions_count,
            total_exposure_dollars=total_exposure_dollars,
            total_exposure_pct=total_exposure_pct,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return_pct
        )

    def record_equity_snapshot(
        self,
        model_id: str,
        equity: float,
        cash: float,
        positions_value: float,
        positions_count: int,
        timestamp: Optional[datetime] = None
    ):
        """
        Record equity snapshot to database.

        Args:
            model_id: Model identifier
            equity: Total equity value
            cash: Cash balance
            positions_value: Total value of positions
            positions_count: Number of positions
            timestamp: Snapshot timestamp (defaults to now)
        """
        snapshot_time = timestamp or datetime.now()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO hourly_snapshots (model_id, timestamp, equity, cash, positions_value, positions_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (model_id, snapshot_time, equity, cash, positions_value, positions_count))

            conn.commit()
            logger.debug(f"Recorded equity snapshot for {model_id}: ${equity:,.2f}")
        except sqlite3.IntegrityError:
            # Duplicate timestamp, update instead
            cursor.execute("""
                UPDATE hourly_snapshots
                SET equity = ?, cash = ?, positions_value = ?, positions_count = ?
                WHERE model_id = ? AND timestamp = ?
            """, (equity, cash, positions_value, positions_count, model_id, snapshot_time))
            conn.commit()
            logger.debug(f"Updated equity snapshot for {model_id}: ${equity:,.2f}")
        finally:
            conn.close()

    def get_equity_curve(
        self,
        model_id: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[float]:
        """
        Get equity curve (historical equity values).

        Args:
            model_id: Model identifier
            since: Only return snapshots after this time
            limit: Maximum number of snapshots to return

        Returns:
            List of equity values ordered by timestamp
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT equity FROM hourly_snapshots
            WHERE model_id = ?
        """
        params = [model_id]

        if since:
            query += " AND timestamp >= ?"
            params.append(since)

        query += " ORDER BY timestamp ASC"

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]

    def get_returns_series(
        self,
        model_id: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[float]:
        """
        Calculate returns series from equity snapshots.

        Args:
            model_id: Model identifier
            since: Only return snapshots after this time
            limit: Maximum number of returns to calculate

        Returns:
            List of period returns (as decimals, e.g., 0.05 for 5%)
        """
        equity_curve = self.get_equity_curve(model_id, since, limit)

        if len(equity_curve) < 2:
            return []

        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i-1] > 0:
                period_return = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                returns.append(period_return)

        return returns


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Initialize metrics engine
    db_path = "/tmp/test_metrics.db"
    import os
    if os.path.exists(db_path):
        os.remove(db_path)

    engine = MetricsEngine(db_path)

    # Example trades
    trades = [
        TradeRecord(
            symbol="AAPL",
            direction="long",
            entry_price=175.00,
            exit_price=178.00,
            shares=100,
            realized_pnl=300.00,
            opened_at=datetime.now() - timedelta(hours=2),
            closed_at=datetime.now() - timedelta(hours=1),
            exit_reason="take_profit"
        ),
        TradeRecord(
            symbol="TSLA",
            direction="short",
            entry_price=250.00,
            exit_price=248.00,
            shares=50,
            realized_pnl=100.00,
            opened_at=datetime.now() - timedelta(hours=3),
            closed_at=datetime.now() - timedelta(hours=2),
            exit_reason="take_profit"
        ),
        TradeRecord(
            symbol="NVDA",
            direction="long",
            entry_price=500.00,
            exit_price=495.00,
            shares=20,
            realized_pnl=-100.00,
            opened_at=datetime.now() - timedelta(hours=1),
            closed_at=datetime.now(),
            exit_reason="stop_loss"
        )
    ]

    # Example equity curve
    equity_curve = [100000, 101000, 100500, 102000, 101500, 103000]

    # Example returns series
    returns_series = [0.01, -0.005, 0.015, -0.005, 0.015]

    # Get comprehensive metrics
    print("\n=== Comprehensive Metrics ===")
    metrics = engine.get_all_metrics(
        model_id="gpt-4o",
        current_equity=103000,
        initial_capital=100000,
        trades=trades,
        open_positions_count=2,
        total_exposure_dollars=50000,
        equity_curve=equity_curve,
        returns_series=returns_series,
        previous_equity=102000
    )

    print(f"Model: {metrics.model_id}")
    print(f"Total Return: {metrics.total_return_pct:.2f}% (${metrics.total_return_dollars:,.2f})")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}" if metrics.sharpe_ratio else "Sharpe Ratio: N/A")
    print(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}% (${metrics.max_drawdown_dollars:,.2f})" if metrics.max_drawdown_pct else "Max Drawdown: N/A")
    print(f"Win Rate: {metrics.win_rate:.2f}%" if metrics.win_rate else "Win Rate: N/A")
    print(f"Profit Factor: {metrics.profit_factor:.2f}" if metrics.profit_factor else "Profit Factor: N/A")
    print(f"Daily P&L: ${metrics.daily_pnl:,.2f}" if metrics.daily_pnl else "Daily P&L: N/A")

    # Record snapshots
    print("\n=== Recording Equity Snapshots ===")
    for i, equity in enumerate(equity_curve):
        engine.record_equity_snapshot(
            model_id="gpt-4o",
            equity=equity,
            cash=equity * 0.5,
            positions_value=equity * 0.5,
            positions_count=2,
            timestamp=datetime.now() - timedelta(hours=len(equity_curve)-i)
        )

    # Retrieve equity curve
    print("\n=== Retrieved Equity Curve ===")
    retrieved_curve = engine.get_equity_curve("gpt-4o")
    print(f"Equity values: {retrieved_curve}")

    print("\n=== Demo Complete ===")
