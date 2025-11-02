"""
Backtest Report Generator.

Generates comprehensive TEST_RESULT.md markdown reports from backtest results.
"""

import logging
import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

from active_trader_llm.database.backtest_models import BacktestMetrics
from active_trader_llm.backtest.performance_metrics import PerformanceMetricsCalculator

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates markdown reports from backtest results.

    Creates TEST_RESULT.md with comprehensive backtest analysis.
    """

    def __init__(self, db_path: str):
        """
        Initialize report generator.

        Args:
            db_path: Path to backtest database
        """
        self.db_path = db_path
        self.metrics_calculator = PerformanceMetricsCalculator(db_path)

    def generate_report(
        self,
        backtest_run_id: int,
        output_path: str = "TEST_RESULT.md"
    ) -> None:
        """
        Generate complete backtest report.

        Args:
            backtest_run_id: ID of backtest run to report on
            output_path: Where to save the report
        """
        logger.info(f"Generating report for backtest run {backtest_run_id}")

        # Load run metadata
        run_metadata = self._load_run_metadata(backtest_run_id)

        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(backtest_run_id)

        # Save metrics to database
        from active_trader_llm.database.backtest_models import BacktestDatabase
        db = BacktestDatabase(self.db_path)
        db.insert_metrics(metrics)

        # Load equity curve
        equity_curve = self._load_equity_curve(backtest_run_id)

        # Load trades
        trades = self._load_trades(backtest_run_id)

        # Generate report sections
        report_parts = []

        report_parts.append(self._generate_header(run_metadata))
        report_parts.append(self._generate_summary(run_metadata, metrics))
        report_parts.append(self._generate_performance_metrics(metrics))
        report_parts.append(self._generate_trade_statistics(metrics, trades))
        report_parts.append(self._generate_equity_curve_data(equity_curve))
        report_parts.append(self._generate_trade_log(trades))
        report_parts.append(self._generate_database_references(backtest_run_id))
        report_parts.append(self._generate_footer())

        # Write report
        report_content = "\n\n".join(report_parts)

        with open(output_path, 'w') as f:
            f.write(report_content)

        logger.info(f"Report generated: {output_path}")

    def _load_run_metadata(self, backtest_run_id: int) -> Dict[str, Any]:
        """Load backtest run metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT run_name, start_date, end_date, initial_capital, final_capital,
                   total_return_pct, status, created_at, completed_at, error_message
            FROM backtest_runs
            WHERE id = ?
        """, (backtest_run_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            raise ValueError(f"Backtest run {backtest_run_id} not found")

        return {
            'run_name': row[0],
            'start_date': row[1],
            'end_date': row[2],
            'initial_capital': row[3],
            'final_capital': row[4],
            'total_return_pct': row[5],
            'status': row[6],
            'created_at': row[7],
            'completed_at': row[8],
            'error_message': row[9]
        }

    def _load_equity_curve(self, backtest_run_id: int) -> pd.DataFrame:
        """Load equity curve data."""
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

        return df

    def _load_trades(self, backtest_run_id: int) -> pd.DataFrame:
        """Load trade data."""
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

            return df

        except Exception as e:
            logger.warning(f"Could not load trades: {e}")
            return pd.DataFrame()

    def _generate_header(self, metadata: Dict[str, Any]) -> str:
        """Generate report header."""
        status_emoji = "✅" if metadata['status'] == 'completed' else "❌"

        return f"""# Backtest Results: {metadata['run_name']}

{status_emoji} **Status:** {metadata['status'].upper()}

**Test Period:** {metadata['start_date']} to {metadata['end_date']}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---"""

    def _generate_summary(self, metadata: Dict[str, Any], metrics: BacktestMetrics) -> str:
        """Generate executive summary."""
        initial = metadata['initial_capital']
        final = metadata['final_capital'] or initial
        total_return = metadata['total_return_pct'] or 0

        # Determine pass/fail
        if metadata['status'] == 'completed' and metadata['error_message'] is None:
            verdict = "✅ **PASS** - Workflow executed successfully"
        else:
            verdict = f"❌ **FAIL** - {metadata['error_message'] or 'Errors occurred'}"

        return f"""## Executive Summary

{verdict}

### Portfolio Performance
- **Initial Capital:** ${initial:,.2f}
- **Final Capital:** ${final:,.2f}
- **Total Return:** {total_return:.2f}%
- **Total Trades:** {metrics.total_trades}
- **Win Rate:** {metrics.win_rate_pct or 0:.1f}% ({metrics.winning_trades}W / {metrics.losing_trades}L)"""

    def _generate_performance_metrics(self, metrics: BacktestMetrics) -> str:
        """Generate performance metrics section."""
        sections = []

        # Return Metrics
        sections.append("""### Return Metrics
| Metric | Value |
|--------|-------|""")

        if metrics.total_return_pct is not None:
            sections.append(f"| Total Return | {metrics.total_return_pct:.2f}% |")
        if metrics.cagr_pct is not None:
            sections.append(f"| CAGR | {metrics.cagr_pct:.2f}% |")

        # Risk Metrics
        sections.append("""
### Risk Metrics
| Metric | Value |
|--------|-------|""")

        if metrics.sharpe_ratio is not None:
            sections.append(f"| Sharpe Ratio | {metrics.sharpe_ratio:.2f} |")
        if metrics.sortino_ratio is not None:
            sections.append(f"| Sortino Ratio | {metrics.sortino_ratio:.2f} |")
        if metrics.max_drawdown_pct is not None:
            sections.append(f"| Max Drawdown | {metrics.max_drawdown_pct:.2f}% |")
        if metrics.max_drawdown_duration_days is not None:
            sections.append(f"| Max Drawdown Duration | {metrics.max_drawdown_duration_days} days |")
        if metrics.volatility_annualized_pct is not None:
            sections.append(f"| Volatility (Annual) | {metrics.volatility_annualized_pct:.2f}% |")
        if metrics.calmar_ratio is not None:
            sections.append(f"| Calmar Ratio | {metrics.calmar_ratio:.2f} |")

        # Benchmark Comparison
        if metrics.benchmark_return_pct is not None:
            sections.append("""
### Benchmark Comparison (SPY)
| Metric | Value |
|--------|-------|""")
            sections.append(f"| Benchmark Return | {metrics.benchmark_return_pct:.2f}% |")
            if metrics.alpha is not None:
                sections.append(f"| Alpha | {metrics.alpha:.2f}% |")
            if metrics.beta is not None:
                sections.append(f"| Beta | {metrics.beta:.2f} |")

        return "## Performance Metrics\n\n" + "\n".join(sections)

    def _generate_trade_statistics(self, metrics: BacktestMetrics, trades: pd.DataFrame) -> str:
        """Generate trade statistics section."""
        sections = ["""## Trade Statistics

| Metric | Value |
|--------|-------|"""]

        sections.append(f"| Total Trades | {metrics.total_trades} |")
        sections.append(f"| Winning Trades | {metrics.winning_trades} |")
        sections.append(f"| Losing Trades | {metrics.losing_trades} |")

        if metrics.win_rate_pct is not None:
            sections.append(f"| Win Rate | {metrics.win_rate_pct:.2f}% |")
        if metrics.profit_factor is not None:
            sections.append(f"| Profit Factor | {metrics.profit_factor:.2f} |")
        if metrics.avg_win_pct is not None:
            sections.append(f"| Average Win | {metrics.avg_win_pct:.2f}% (${metrics.avg_win_dollars:.2f}) |")
        if metrics.avg_loss_pct is not None:
            sections.append(f"| Average Loss | {metrics.avg_loss_pct:.2f}% (${metrics.avg_loss_dollars:.2f}) |")
        if metrics.largest_win_dollars is not None:
            sections.append(f"| Largest Win | ${metrics.largest_win_dollars:.2f} |")
        if metrics.largest_loss_dollars is not None:
            sections.append(f"| Largest Loss | ${metrics.largest_loss_dollars:.2f} |")
        if metrics.avg_hold_time_days is not None:
            sections.append(f"| Average Hold Time | {metrics.avg_hold_time_days:.1f} days |")
        if metrics.max_consecutive_wins is not None:
            sections.append(f"| Max Consecutive Wins | {metrics.max_consecutive_wins} |")
        if metrics.max_consecutive_losses is not None:
            sections.append(f"| Max Consecutive Losses | {metrics.max_consecutive_losses} |")

        return "\n".join(sections)

    def _generate_equity_curve_data(self, equity_curve: pd.DataFrame) -> str:
        """Generate equity curve section."""
        if equity_curve.empty:
            return "## Equity Curve\n\nNo equity data available."

        sections = ["""## Equity Curve

| Date | Equity | Cash | Positions Value | Daily Return % | Cumulative Return % | Open Positions |
|------|--------|------|-----------------|----------------|---------------------|----------------|"""]

        for _, row in equity_curve.iterrows():
            date = row['date']
            equity = row['equity']
            cash = row['cash']
            positions = row['positions_value']
            daily_ret = row['daily_return_pct'] if pd.notna(row['daily_return_pct']) else 0.0
            cum_ret = row['cumulative_return_pct'] if pd.notna(row['cumulative_return_pct']) else 0.0
            num_pos = int(row['num_open_positions'])

            sections.append(
                f"| {date} | ${equity:,.2f} | ${cash:,.2f} | ${positions:,.2f} | "
                f"{daily_ret:+.2f}% | {cum_ret:+.2f}% | {num_pos} |"
            )

        return "\n".join(sections)

    def _generate_trade_log(self, trades: pd.DataFrame) -> str:
        """Generate trade log section."""
        if trades.empty:
            return "## Trade Log\n\nNo closed trades."

        sections = ["""## Trade Log

| Symbol | Direction | Entry | Exit | Shares | Stop Loss | Take Profit | Exit Reason | P&L | Return % | Strategy |
|--------|-----------|-------|------|--------|-----------|-------------|-------------|-----|----------|----------|"""]

        for _, trade in trades.iterrows():
            symbol = trade['symbol']
            direction = trade['direction']
            entry = trade['entry_price']
            exit_price = trade['exit_price']
            shares = trade['shares']
            stop = trade['stop_loss']
            target = trade['take_profit']
            exit_reason = trade['exit_reason']
            pnl = trade['realized_pnl']
            return_pct = (pnl / (entry * shares)) * 100 if entry * shares > 0 else 0
            strategy = trade.get('strategy', 'N/A')

            sections.append(
                f"| {symbol} | {direction.upper()} | ${entry:.2f} | ${exit_price:.2f} | {shares} | "
                f"${stop:.2f} | ${target:.2f} | {exit_reason} | ${pnl:+.2f} | {return_pct:+.2f}% | {strategy} |"
            )

        return "\n".join(sections)

    def _generate_database_references(self, backtest_run_id: int) -> str:
        """Generate database reference section."""
        return f"""## Database References

All detailed data is stored in the database for further analysis:

- **Backtest Run ID:** `{backtest_run_id}`
- **Database:** `{self.db_path}`
- **Positions Database:** `{self.db_path.replace('.db', '_positions.db')}`

### Available Tables:
- `backtest_runs` - Backtest run metadata
- `backtest_metrics` - Performance metrics
- `daily_equity` - Daily equity curve
- `analyst_signals` - All analyst signals
- `researcher_debates` - Bull/bear debates
- `trader_plans` - Trade plans
- `llm_interactions` - LLM prompts and responses
- `positions` - Position tracking (in positions DB)

### Query Examples:

```sql
-- Get all analyst signals for this run
SELECT * FROM analyst_signals WHERE backtest_run_id = {backtest_run_id};

-- Get all trade plans
SELECT * FROM trader_plans WHERE backtest_run_id = {backtest_run_id};

-- Get daily equity curve
SELECT * FROM daily_equity WHERE backtest_run_id = {backtest_run_id} ORDER BY date;
```"""

    def _generate_footer(self) -> str:
        """Generate report footer."""
        return """---

## Notes

- This backtest uses realistic execution simulation with slippage and commissions
- All LLM reasoning and signals are stored in the database
- Position management follows the same logic as live trading
- Stop losses and take profits are monitored and executed automatically

**Generated by ActiveTrader-LLM Backtest Engine**"""


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Testing ReportGenerator")
    print("=" * 60)

    print("\nThis module requires actual backtest data to test.")
    print("Run a backtest first, then generate a report.")
    print("\nExample usage:")
    print("  from active_trader_llm.backtest.report_generator import ReportGenerator")
    print("  generator = ReportGenerator('data/backtest.db')")
    print("  generator.generate_report(run_id=1, output_path='TEST_RESULT.md')")
