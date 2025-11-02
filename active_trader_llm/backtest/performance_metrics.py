"""
Performance Metrics Calculator for Backtesting.

Calculates comprehensive trading and portfolio performance metrics
from backtest results.
"""

import logging
import sqlite3
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from active_trader_llm.database.backtest_models import BacktestMetrics

logger = logging.getLogger(__name__)


class PerformanceMetricsCalculator:
    """
    Calculates comprehensive performance metrics from backtest results.

    Computes returns, risk metrics, trade statistics, and benchmark comparisons.
    """

    def __init__(self, db_path: str):
        """
        Initialize metrics calculator.

        Args:
            db_path: Path to database containing backtest results
        """
        self.db_path = db_path

    def calculate_metrics(self, backtest_run_id: int) -> BacktestMetrics:
        """
        Calculate all performance metrics for a backtest run.

        Args:
            backtest_run_id: ID of the backtest run

        Returns:
            BacktestMetrics with all calculated metrics
        """
        logger.info(f"Calculating metrics for backtest run {backtest_run_id}")

        # Load data from database
        equity_curve = self._load_equity_curve(backtest_run_id)
        trades = self._load_trades(backtest_run_id)

        if equity_curve.empty:
            logger.warning("No equity data found")
            return BacktestMetrics(backtest_run_id=backtest_run_id)

        # Calculate return metrics
        return_metrics = self._calculate_return_metrics(equity_curve)

        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(equity_curve)

        # Calculate trade statistics
        trade_stats = self._calculate_trade_statistics(trades)

        # Calculate benchmark comparison
        benchmark_metrics = self._calculate_benchmark_metrics(equity_curve)

        # Combine all metrics
        metrics = BacktestMetrics(
            backtest_run_id=backtest_run_id,
            **return_metrics,
            **risk_metrics,
            **trade_stats,
            **benchmark_metrics
        )

        logger.info("Metrics calculation complete")
        return metrics

    def _load_equity_curve(self, backtest_run_id: int) -> pd.DataFrame:
        """Load daily equity data from database."""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT date, equity, cash, positions_value, daily_return_pct,
                   cumulative_return_pct, num_open_positions, benchmark_value
            FROM daily_equity
            WHERE backtest_run_id = ?
            ORDER BY date ASC
        """

        df = pd.read_sql_query(query, conn, params=(backtest_run_id,))
        conn.close()

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

        return df

    def _load_trades(self, backtest_run_id: int) -> pd.DataFrame:
        """Load closed positions (completed trades) from database."""
        # Get positions database path
        positions_db_path = self.db_path.replace('.db', '_positions.db')

        try:
            conn = sqlite3.connect(positions_db_path)

            query = """
                SELECT symbol, direction, entry_price, exit_price, shares,
                       stop_loss, take_profit, opened_at, closed_at,
                       exit_reason, realized_pnl, strategy
                FROM positions
                WHERE status = 'closed'
                ORDER BY closed_at ASC
            """

            df = pd.read_sql_query(query, conn)
            conn.close()

            if not df.empty:
                df['opened_at'] = pd.to_datetime(df['opened_at'])
                df['closed_at'] = pd.to_datetime(df['closed_at'])
                df['hold_time_days'] = (df['closed_at'] - df['opened_at']).dt.total_seconds() / 86400

            return df

        except Exception as e:
            logger.warning(f"Could not load trades: {e}")
            return pd.DataFrame()

    def _calculate_return_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """Calculate return-based metrics."""
        if equity_curve.empty:
            return {}

        initial_equity = equity_curve['equity'].iloc[0]
        final_equity = equity_curve['equity'].iloc[-1]

        total_return_pct = ((final_equity - initial_equity) / initial_equity) * 100

        # CAGR (Compound Annual Growth Rate)
        num_days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if num_days > 0:
            years = num_days / 365.25
            cagr_pct = (((final_equity / initial_equity) ** (1 / years)) - 1) * 100
        else:
            cagr_pct = None

        return {
            'total_return_pct': round(total_return_pct, 2),
            'cagr_pct': round(cagr_pct, 2) if cagr_pct is not None else None
        }

    def _calculate_risk_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk-based metrics."""
        if equity_curve.empty:
            return {}

        # Returns series
        returns = equity_curve['daily_return_pct'].dropna()

        if len(returns) == 0:
            return {}

        # Sharpe Ratio (assuming 252 trading days, 0% risk-free rate)
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = None

        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            downside_deviation = downside_returns.std()
            sortino_ratio = (returns.mean() / downside_deviation) * np.sqrt(252)
            downside_deviation_pct = downside_deviation
        else:
            sortino_ratio = None
            downside_deviation_pct = None

        # Volatility (annualized)
        volatility_annualized_pct = returns.std() * np.sqrt(252)

        # Maximum Drawdown
        cumulative_returns = (1 + returns / 100).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max * 100

        max_drawdown_pct = drawdown.min()

        # Drawdown duration
        is_drawdown = drawdown < 0
        drawdown_periods = []
        start_idx = None

        for idx, in_dd in enumerate(is_drawdown):
            if in_dd and start_idx is None:
                start_idx = idx
            elif not in_dd and start_idx is not None:
                drawdown_periods.append(idx - start_idx)
                start_idx = None

        if start_idx is not None:
            drawdown_periods.append(len(is_drawdown) - start_idx)

        max_drawdown_duration_days = max(drawdown_periods) if drawdown_periods else None

        # Calmar Ratio (CAGR / Max Drawdown)
        if max_drawdown_pct is not None and max_drawdown_pct < 0:
            cagr = returns.mean() * 252  # Approximate annualized return
            calmar_ratio = abs(cagr / max_drawdown_pct)
        else:
            calmar_ratio = None

        # Ulcer Index (risk measure that considers depth and duration of drawdowns)
        ulcer_index = np.sqrt((drawdown ** 2).mean())

        # Recovery Factor (Net Profit / Max Drawdown)
        initial_equity = equity_curve['equity'].iloc[0]
        final_equity = equity_curve['equity'].iloc[-1]
        net_profit = final_equity - initial_equity

        if max_drawdown_pct is not None and max_drawdown_pct < 0:
            max_dd_dollars = initial_equity * (abs(max_drawdown_pct) / 100)
            recovery_factor = net_profit / max_dd_dollars if max_dd_dollars > 0 else None
        else:
            recovery_factor = None

        return {
            'sharpe_ratio': round(sharpe_ratio, 2) if sharpe_ratio is not None else None,
            'sortino_ratio': round(sortino_ratio, 2) if sortino_ratio is not None else None,
            'max_drawdown_pct': round(max_drawdown_pct, 2) if max_drawdown_pct is not None else None,
            'max_drawdown_duration_days': max_drawdown_duration_days,
            'calmar_ratio': round(calmar_ratio, 2) if calmar_ratio is not None else None,
            'volatility_annualized_pct': round(volatility_annualized_pct, 2),
            'downside_deviation_pct': round(downside_deviation_pct, 2) if downside_deviation_pct is not None else None,
            'recovery_factor': round(recovery_factor, 2) if recovery_factor is not None else None,
            'ulcer_index': round(ulcer_index, 2) if ulcer_index is not None else None
        }

    def _calculate_trade_statistics(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trade-based statistics."""
        if trades.empty:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0
            }

        total_trades = len(trades)
        winning_trades = len(trades[trades['realized_pnl'] > 0])
        losing_trades = len(trades[trades['realized_pnl'] < 0])
        breakeven_trades = len(trades[trades['realized_pnl'] == 0])

        # Win rate
        win_rate_pct = (winning_trades / total_trades * 100) if total_trades > 0 else None

        # Profit factor (total wins / total losses)
        total_wins = trades[trades['realized_pnl'] > 0]['realized_pnl'].sum()
        total_losses = abs(trades[trades['realized_pnl'] < 0]['realized_pnl'].sum())
        profit_factor = (total_wins / total_losses) if total_losses > 0 else None

        # Average win/loss
        wins = trades[trades['realized_pnl'] > 0]['realized_pnl']
        losses = trades[trades['realized_pnl'] < 0]['realized_pnl']

        avg_win_dollars = wins.mean() if len(wins) > 0 else None
        avg_loss_dollars = losses.mean() if len(losses) > 0 else None

        # Calculate % returns for each trade
        trades['return_pct'] = trades.apply(
            lambda row: (row['realized_pnl'] / (row['entry_price'] * row['shares'])) * 100
            if row['entry_price'] * row['shares'] > 0 else 0,
            axis=1
        )

        wins_pct = trades[trades['realized_pnl'] > 0]['return_pct']
        losses_pct = trades[trades['realized_pnl'] < 0]['return_pct']

        avg_win_pct = wins_pct.mean() if len(wins_pct) > 0 else None
        avg_loss_pct = losses_pct.mean() if len(losses_pct) > 0 else None

        # Largest win/loss
        largest_win_dollars = wins.max() if len(wins) > 0 else None
        largest_loss_dollars = losses.min() if len(losses) > 0 else None

        # Average hold time
        avg_hold_time_days = trades['hold_time_days'].mean() if 'hold_time_days' in trades.columns else None

        # Consecutive wins/losses
        is_winner = trades['realized_pnl'] > 0
        max_consecutive_wins = self._max_consecutive(is_winner)
        max_consecutive_losses = self._max_consecutive(~is_winner)

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': round(win_rate_pct, 2) if win_rate_pct is not None else None,
            'profit_factor': round(profit_factor, 2) if profit_factor is not None else None,
            'avg_win_pct': round(avg_win_pct, 2) if avg_win_pct is not None else None,
            'avg_loss_pct': round(avg_loss_pct, 2) if avg_loss_pct is not None else None,
            'avg_win_dollars': round(avg_win_dollars, 2) if avg_win_dollars is not None else None,
            'avg_loss_dollars': round(avg_loss_dollars, 2) if avg_loss_dollars is not None else None,
            'largest_win_dollars': round(largest_win_dollars, 2) if largest_win_dollars is not None else None,
            'largest_loss_dollars': round(largest_loss_dollars, 2) if largest_loss_dollars is not None else None,
            'avg_hold_time_days': round(avg_hold_time_days, 2) if avg_hold_time_days is not None else None,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses
        }

    def _max_consecutive(self, series: pd.Series) -> int:
        """Calculate maximum consecutive True values in a boolean series."""
        if len(series) == 0:
            return 0

        max_count = 0
        current_count = 0

        for value in series:
            if value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def _calculate_benchmark_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """Calculate benchmark comparison metrics."""
        if equity_curve.empty or 'benchmark_value' not in equity_curve.columns:
            return {}

        # Filter out None benchmark values
        valid_benchmark = equity_curve[equity_curve['benchmark_value'].notna()]

        if valid_benchmark.empty:
            return {}

        # Benchmark return
        initial_benchmark = valid_benchmark['benchmark_value'].iloc[0]
        final_benchmark = valid_benchmark['benchmark_value'].iloc[-1]

        benchmark_return_pct = ((final_benchmark - initial_benchmark) / initial_benchmark) * 100

        # Alpha and Beta (simplified)
        # Beta: covariance(strategy, benchmark) / variance(benchmark)
        # Alpha: strategy_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))

        # Calculate returns
        strategy_returns = equity_curve['daily_return_pct'].dropna()

        # Calculate benchmark daily returns
        benchmark_equity = valid_benchmark['benchmark_value']
        benchmark_returns = benchmark_equity.pct_change() * 100
        benchmark_returns = benchmark_returns.dropna()

        # Align the two series
        aligned = pd.DataFrame({
            'strategy': strategy_returns,
            'benchmark': benchmark_returns
        }).dropna()

        if len(aligned) > 1:
            covariance = aligned['strategy'].cov(aligned['benchmark'])
            benchmark_variance = aligned['benchmark'].var()

            if benchmark_variance > 0:
                beta = covariance / benchmark_variance
            else:
                beta = None

            # Alpha (excess return vs benchmark, adjusted for beta)
            if beta is not None:
                risk_free_rate = 0  # Assuming 0% for simplicity
                strategy_mean_return = aligned['strategy'].mean() * 252  # Annualized
                benchmark_mean_return = aligned['benchmark'].mean() * 252  # Annualized
                alpha = strategy_mean_return - (risk_free_rate + beta * (benchmark_mean_return - risk_free_rate))
            else:
                alpha = None
        else:
            beta = None
            alpha = None

        return {
            'benchmark_return_pct': round(benchmark_return_pct, 2) if benchmark_return_pct is not None else None,
            'alpha': round(alpha, 2) if alpha is not None else None,
            'beta': round(beta, 2) if beta is not None else None
        }


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Testing PerformanceMetricsCalculator")
    print("=" * 60)

    # Note: This requires an actual backtest to have been run
    # For testing, you would need to run a backtest first

    print("\nThis module requires actual backtest data to test.")
    print("Run a backtest first using backtest_engine.py, then use this calculator.")
    print("\nExample usage:")
    print("  from active_trader_llm.backtest.performance_metrics import PerformanceMetricsCalculator")
    print("  calc = PerformanceMetricsCalculator('data/backtest.db')")
    print("  metrics = calc.calculate_metrics(run_id=1)")
    print("  print(metrics)")
