"""
Backtest Engine - Orchestrates historical walk-forward testing.

This module runs the trading system over historical data, simulating
realistic execution and tracking comprehensive metrics.
"""

import logging
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict, Any
import pandas as pd
import json
from pathlib import Path

from active_trader_llm.config.loader import load_config
from active_trader_llm.database.backtest_models import (
    BacktestDatabase, BacktestRun, DailyEquity, AnalystSignal,
    ResearcherDebate, TraderPlan, LLMInteraction
)
from active_trader_llm.execution.position_manager import PositionManager
from active_trader_llm.data_ingestion.price_volume_ingestor import PriceVolumeIngestor
from active_trader_llm.risk.risk_manager import PortfolioState

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Historical backtesting engine for ActiveTrader-LLM.

    Runs the complete trading workflow over historical dates, tracking
    all signals, debates, plans, LLM interactions, and performance metrics.
    """

    def __init__(
        self,
        config_path: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000.0,
        backtest_db_path: str = "data/backtest.db"
    ):
        """
        Initialize backtest engine.

        Args:
            config_path: Path to trading config YAML
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            initial_capital: Starting capital
            backtest_db_path: Path to backtest database
        """
        self.config_path = config_path
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        self.initial_capital = initial_capital
        self.backtest_db_path = backtest_db_path

        # Load config
        self.config = load_config(config_path)

        # Override mode to backtest
        self.config.mode = "backtest"

        # Initialize database
        self.db = BacktestDatabase(backtest_db_path)

        # Position manager (use temporary backtest positions DB)
        backtest_positions_db = backtest_db_path.replace('.db', '_positions.db')
        self.position_manager = PositionManager(backtest_positions_db)

        # Clear any existing positions from previous backtest runs
        self.position_manager.clear_all_positions()

        # Track backtest run
        self.run_id: Optional[int] = None
        self.current_date: Optional[date] = None

        # Equity tracking
        self.cash = initial_capital
        self.equity_curve: List[Dict[str, Any]] = []

        # Data ingestor for benchmark
        self.ingestor = PriceVolumeIngestor(cache_db_path="data/price_cache.db")

        # Benchmark tracking (SPY)
        self.benchmark_initial_value: Optional[float] = None
        self.benchmark_data: Dict[str, float] = {}

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        logger.info(f"BacktestEngine initialized: {start_date} to {end_date}")
        logger.info(f"Initial capital: ${initial_capital:,.2f}")

    def get_current_state(self) -> PortfolioState:
        """
        Provide real-time portfolio state for backtesting.

        This method is called by ActiveTraderLLM during risk checks and
        position sizing to get the current portfolio state, ensuring
        decisions are made with up-to-date cash and equity values.

        Returns:
            PortfolioState with current cash, equity, and positions
        """
        open_positions = self.position_manager.get_open_positions()
        positions_value = sum([p.entry_price * p.shares for p in open_positions])

        return PortfolioState(
            cash=self.cash,
            equity=self.cash + positions_value,
            positions=[],  # Position details populated by position_manager
            daily_pnl=0.0  # P&L calculated by position_manager
        )

    def get_market_days(self) -> List[date]:
        """
        Get list of market days between start and end dates.

        Uses yfinance to determine actual trading days by fetching
        SPY data and checking which dates have trading activity.

        Returns:
            List of market dates
        """
        logger.info("Determining market days...")

        # Fetch SPY data to determine market days
        try:
            spy_data = self.ingestor.fetch_prices(
                universe=['SPY'],
                interval='1d',
                lookback_days=(self.end_date - self.start_date).days + 400,  # Extra buffer for historical data
                use_cache=False  # Force fresh fetch to ensure we get historical data
            )

            if spy_data.empty:
                logger.warning("Could not fetch SPY data for market days, using calendar days")
                return self._get_calendar_business_days()

            # Convert timestamp column to datetime
            spy_data['timestamp'] = pd.to_datetime(spy_data['timestamp'])

            # Filter to our date range
            spy_filtered = spy_data[
                (spy_data['timestamp'].dt.date >= self.start_date) &
                (spy_data['timestamp'].dt.date <= self.end_date)
            ]

            market_dates = spy_filtered['timestamp'].dt.date.unique().tolist()
            market_dates.sort()

            # Store benchmark data
            for _, row in spy_filtered.iterrows():
                date_key = row['timestamp'].date()
                self.benchmark_data[str(date_key)] = row['close']

            # Set initial benchmark value
            if market_dates and str(market_dates[0]) in self.benchmark_data:
                self.benchmark_initial_value = self.benchmark_data[str(market_dates[0])]

            logger.info(f"Found {len(market_dates)} market days")
            return market_dates

        except Exception as e:
            logger.error(f"Error fetching market days: {e}")
            return self._get_calendar_business_days()

    def _get_calendar_business_days(self) -> List[date]:
        """Get business days (Mon-Fri) as fallback."""
        days = []
        current = self.start_date
        while current <= self.end_date:
            # Monday = 0, Sunday = 6
            if current.weekday() < 5:  # Monday to Friday
                days.append(current)
            current += timedelta(days=1)
        return days

    def run(self, run_name: Optional[str] = None) -> int:
        """
        Execute the backtest.

        Args:
            run_name: Optional name for this backtest run

        Returns:
            Backtest run ID
        """
        logger.info("=" * 80)
        logger.info("STARTING BACKTEST")
        logger.info("=" * 80)

        # Create backtest run record
        run = BacktestRun(
            run_name=run_name or f"Backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_date=str(self.start_date),
            end_date=str(self.end_date),
            initial_capital=self.initial_capital,
            config_snapshot=json.dumps(self.config.model_dump(), default=str),
            status='running'
        )

        self.run_id = self.db.create_run(run)
        logger.info(f"Created backtest run ID: {self.run_id}")

        try:
            # Get market days
            market_days = self.get_market_days()

            if not market_days:
                raise ValueError("No market days found in date range")

            logger.info(f"Backtesting {len(market_days)} days: {market_days[0]} to {market_days[-1]}")

            # Run decision cycle for each day
            for day_idx, market_date in enumerate(market_days, 1):
                logger.info("\n" + "=" * 80)
                logger.info(f"DAY {day_idx}/{len(market_days)}: {market_date}")
                logger.info("=" * 80)

                self.current_date = market_date
                self.run_day(market_date)

                # Record daily equity
                self.record_daily_equity(market_date)

            # Mark run as completed
            final_equity = self.equity_curve[-1]['equity'] if self.equity_curve else self.initial_capital
            total_return_pct = ((final_equity - self.initial_capital) / self.initial_capital) * 100

            self.db.update_run(
                self.run_id,
                status='completed',
                completed_at=datetime.now(),
                final_capital=final_equity,
                total_return_pct=total_return_pct
            )

            logger.info("\n" + "=" * 80)
            logger.info("BACKTEST COMPLETED")
            logger.info("=" * 80)
            logger.info(f"Final equity: ${final_equity:,.2f}")
            logger.info(f"Total return: {total_return_pct:.2f}%")

            return self.run_id

        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            self.db.update_run(
                self.run_id,
                status='failed',
                completed_at=datetime.now(),
                error_message=str(e)
            )
            raise

    def run_day(self, market_date: date):
        """
        Run decision cycle for a single day.

        This imports and uses the main ActiveTraderLLM system, but
        with date-specific data fetching and enhanced logging.

        Args:
            market_date: Date to run
        """
        from active_trader_llm.main import ActiveTraderLLM

        # Initialize trading system with our config in backtest mode
        # Pass self as backtest_state_provider for real-time portfolio state queries
        trader = ActiveTraderLLM(self.config_path, mode_override="backtest", backtest_state_provider=self)

        # Validate backtest mode setup
        assert trader.simulated_broker is not None, "Backtest mode must use SimulatedBroker"
        assert trader.broker_executor is None, "Backtest mode should not use AlpacaBrokerExecutor"
        logger.debug("Backtest mode validation passed: using SimulatedBroker")

        # Replace position manager with our backtest one
        trader.position_manager = self.position_manager
        from active_trader_llm.execution.exit_monitor import ExitMonitor
        trader.exit_monitor = ExitMonitor(self.position_manager)
        trader.risk_manager.position_manager = self.position_manager

        # Update simulated broker and portfolio state with current backtest values
        if trader.simulated_broker:
            trader.simulated_broker.update_cash(self.cash, self.cash)
            logger.debug(f"Updated SimulatedBroker: cash=${self.cash:.2f}")

        trader.portfolio_state.cash = self.cash
        trader.portfolio_state.equity = self.cash  # Will be updated with position values

        # Monkey-patch the ingestor to fetch data only up to this date
        original_fetch = trader.ingestor.fetch_prices

        def date_limited_fetch(universe, interval, lookback_days, use_cache=True):
            """Fetch data only up to current backtest date."""
            # Calculate end date for data fetch
            end_date = market_date
            start_date = end_date - timedelta(days=lookback_days)

            logger.debug(f"Fetching data from {start_date} to {end_date}")

            # Fetch data
            df = original_fetch(
                universe=universe,
                interval=interval,
                lookback_days=lookback_days,
                use_cache=use_cache
            )

            if df.empty:
                logger.warning(f"fetch_prices returned empty dataframe for {universe}")
                return df

            # Filter to only data up to current date
            # Check if index looks corrupted (all dates in 1970 epoch)
            if isinstance(df.index, pd.DatetimeIndex) and df.index.min().year == 1970:
                # Index has wrong epoch - likely needs unit specification
                logger.warning(f"[BACKTEST] Detected corrupted timestamp index (year 1970), attempting to fix...")
                # Try to get the original timestamp column if it exists
                if 'timestamp' in df.columns:
                    df.index = pd.to_datetime(df['timestamp'])
                    logger.info(f"[BACKTEST] Restored index from 'timestamp' column")
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            logger.info(f"[BACKTEST] Before filter: {len(df)} rows, date range: {df.index.min()} to {df.index.max()}")
            logger.info(f"[BACKTEST] Market date for filtering: {market_date}")

            df = df[df.index.date <= market_date]

            logger.info(f"[BACKTEST] After filter (<=market_date): {len(df)} rows")
            if df.empty:
                logger.warning(f"[BACKTEST] Filtering to market_date {market_date} resulted in empty dataframe!")

            return df

        # Apply the monkey-patch to filter historical data correctly
        trader.ingestor.fetch_prices = date_limited_fetch
        logger.info(f"[BACKTEST] Historical data filtering enabled - market_date: {market_date}")

        # Wrap LLM calls to log to database
        self._wrap_llm_logging(trader)

        # Track open positions before decision cycle
        open_positions_before = set(p.symbol for p in self.position_manager.get_open_positions())

        # Run the decision cycle
        try:
            trader.run_decision_cycle()

            # After decision cycle, deduct cash for any newly opened positions
            open_positions_after = self.position_manager.get_open_positions()
            newly_opened = [p for p in open_positions_after if p.symbol not in open_positions_before]

            for pos in newly_opened:
                # Deduct cash for the position cost + commission
                position_cost = pos.entry_price * pos.shares
                commission = trader.simulated_broker.commission_per_trade if trader.simulated_broker else 1.0
                total_cost = position_cost + commission

                self.cash -= total_cost
                logger.debug(f"Cash deducted for opening {pos.symbol}: -${total_cost:.2f} (position: ${position_cost:.2f}, commission: ${commission:.2f})")

            # Update cash based on any closed positions
            closed_today = self.position_manager.get_closed_positions(since=datetime.combine(market_date, datetime.min.time()))
            for pos in closed_today:
                # Add back the position value (with realized P&L already included)
                position_close_value = pos.exit_price * pos.shares
                self.cash += position_close_value
                logger.debug(f"Cash updated after closing {pos.symbol}: +${position_close_value:.2f}, realized P&L: ${pos.realized_pnl:.2f}")

            # Synchronize state back to SimulatedBroker and portfolio_state
            # This ensures all components have consistent state for next cycle
            if trader.simulated_broker:
                # Calculate current equity (cash + position values)
                open_positions = self.position_manager.get_open_positions()
                positions_value = sum([p.entry_price * p.shares for p in open_positions])
                current_equity = self.cash + positions_value

                trader.simulated_broker.update_cash(self.cash, current_equity)
                logger.info(f"Synced state to SimulatedBroker: cash=${self.cash:.2f}, equity=${current_equity:.2f}")

            # Update trader's portfolio_state
            trader.portfolio_state.cash = self.cash
            open_positions = self.position_manager.get_open_positions()
            positions_value = sum([p.entry_price * p.shares for p in open_positions])
            trader.portfolio_state.equity = self.cash + positions_value
            logger.info(f"Synced state to portfolio_state: cash=${trader.portfolio_state.cash:.2f}, equity=${trader.portfolio_state.equity:.2f}")

        except Exception as e:
            logger.error(f"Decision cycle failed for {market_date}: {e}")
            # Continue to next day rather than failing entire backtest

    def _wrap_llm_logging(self, trader):
        """
        Wrap analyst/researcher/trader LLM calls to log to database.

        This is a simplified version - in production, you'd want to
        modify the actual agent classes to support logging callbacks.
        """
        # TODO: This is a placeholder for LLM interaction logging
        # In a full implementation, you would:
        # 1. Modify analyst classes to accept a logging callback
        # 2. Capture prompts, responses, and metadata
        # 3. Store in llm_interactions table

        # For now, we'll log signals, debates, and plans in the decision cycle
        pass

    def record_daily_equity(self, market_date: date):
        """
        Record daily equity curve point.

        Args:
            market_date: Date for this equity point
        """
        # Get current prices for all positions
        try:
            # Fetch current prices for all symbols in positions
            open_positions = self.position_manager.get_open_positions()

            current_prices = {}
            if open_positions:
                symbols = [p.symbol for p in open_positions]
                # Use the data ingestor to get prices for the market_date
                try:
                    # Calculate lookback from market_date to ensure we have data for that specific date
                    days_ago = (date.today() - market_date).days
                    lookback = max(days_ago + 5, 10)  # Add buffer to ensure we get the date

                    price_df = self.ingestor.fetch_prices(
                        universe=symbols,
                        interval='1d',
                        lookback_days=lookback,
                        use_cache=True
                    )

                    if not price_df.empty:
                        # Set timestamp column as index for date filtering
                        if 'timestamp' in price_df.columns:
                            price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
                            price_df = price_df.set_index('timestamp')
                        else:
                            price_df.index = pd.to_datetime(price_df.index)

                        logger.debug(f"Price data date range: {price_df.index.min()} to {price_df.index.max()}")
                        logger.debug(f"Looking for market_date: {market_date}")

                        today_prices = price_df[price_df.index.date == market_date]
                        logger.debug(f"Filtered prices shape: {today_prices.shape}, empty: {today_prices.empty}")

                        if not today_prices.empty:
                            # Handle both MultiIndex and flat column formats
                            if isinstance(today_prices.columns, pd.MultiIndex):
                                # MultiIndex format: (symbol, field)
                                logger.debug(f"Available symbols in data: {list(today_prices.columns.get_level_values(0).unique())}")
                                for symbol in symbols:
                                    if symbol in today_prices.columns.get_level_values(0):
                                        current_prices[symbol] = float(today_prices[symbol]['close'].iloc[-1])
                                        logger.debug(f"EOD price for {symbol}: ${current_prices[symbol]:.2f}")
                                    else:
                                        logger.warning(f"Symbol {symbol} not found in price data columns")
                            else:
                                # Flat format with 'symbol' column
                                if 'symbol' in today_prices.columns:
                                    logger.debug(f"Available symbols in data: {today_prices['symbol'].unique().tolist()}")
                                    for symbol in symbols:
                                        symbol_data = today_prices[today_prices['symbol'] == symbol]
                                        if not symbol_data.empty:
                                            current_prices[symbol] = float(symbol_data['close'].iloc[0])
                                            logger.debug(f"EOD price for {symbol}: ${current_prices[symbol]:.2f}")
                                        else:
                                            logger.warning(f"Symbol {symbol} not found in price data")
                                else:
                                    logger.warning("Unexpected price data format - no 'symbol' column or MultiIndex")
                        else:
                            logger.warning(f"No price data found for market_date {market_date}")
                except Exception as e:
                    logger.warning(f"Could not fetch current prices: {e}", exc_info=True)

            # Calculate approximate equity for exposure calculation
            # (will be refined after getting position values)
            approx_equity = self.cash

            # Get portfolio state
            portfolio_state = self.position_manager.get_portfolio_state(current_prices, approx_equity)

            positions_value = sum([
                p['position_value'] for p in portfolio_state.get('open_positions', [])
            ])

            # Calculate actual equity
            equity = self.cash + positions_value

            # Calculate returns
            daily_return_pct = None
            cumulative_return_pct = ((equity - self.initial_capital) / self.initial_capital) * 100

            if self.equity_curve:
                prev_equity = self.equity_curve[-1]['equity']
                if prev_equity > 0:
                    daily_return_pct = ((equity - prev_equity) / prev_equity) * 100

            # Get benchmark value
            benchmark_value = None
            if str(market_date) in self.benchmark_data:
                spy_price = self.benchmark_data[str(market_date)]
                if self.benchmark_initial_value and self.benchmark_initial_value > 0:
                    # Calculate equivalent SPY value if we invested initial capital
                    benchmark_value = self.initial_capital * (spy_price / self.benchmark_initial_value)

            # Store in database
            equity_record = DailyEquity(
                backtest_run_id=self.run_id,
                date=str(market_date),
                equity=equity,
                cash=self.cash,
                positions_value=positions_value,
                daily_return_pct=daily_return_pct,
                cumulative_return_pct=cumulative_return_pct,
                num_open_positions=len(portfolio_state.get('positions', [])),
                benchmark_value=benchmark_value
            )

            self.db.insert_daily_equity(equity_record)

            # Store in memory for metrics calculation
            self.equity_curve.append({
                'date': market_date,
                'equity': equity,
                'cash': self.cash,
                'positions_value': positions_value,
                'daily_return_pct': daily_return_pct,
                'cumulative_return_pct': cumulative_return_pct,
                'num_positions': len(portfolio_state.get('positions', [])),
                'benchmark_value': benchmark_value
            })

            logger.info(
                f"EOD Equity: ${equity:,.2f} (Cash: ${self.cash:,.2f}, "
                f"Positions: ${positions_value:,.2f}, Return: {cumulative_return_pct:.2f}%)"
            )

        except Exception as e:
            logger.error(f"Error recording daily equity: {e}")

    def get_run_id(self) -> Optional[int]:
        """Get the current backtest run ID."""
        return self.run_id


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run a simple 5-day backtest
    from datetime import date, timedelta

    end_date = date.today() - timedelta(days=1)  # Yesterday
    start_date = end_date - timedelta(days=10)  # Get ~5 market days

    engine = BacktestEngine(
        config_path="config.yaml",
        start_date=str(start_date),
        end_date=str(end_date),
        initial_capital=100000.0
    )

    run_id = engine.run(run_name="Test_5Day_Backtest")
    print(f"\nBacktest complete! Run ID: {run_id}")
    print(f"Database: {engine.backtest_db_path}")
