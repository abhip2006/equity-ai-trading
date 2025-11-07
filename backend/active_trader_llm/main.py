#!/usr/bin/env python3
"""
ActiveTrader-LLM Main Entry Point

Orchestrates the entire multi-agent trading pipeline:
1. Data ingestion
2. Feature engineering
3. Analyst signals
4. Researcher debate
5. Trade plan synthesis
6. Risk validation
7. Execution logging
8. Learning/strategy switching

OPTIMIZATION NOTES:
- Account info caching: Reduces redundant API calls in live/paper-live mode
- Cache TTL: 5 seconds (sufficient for intra-cycle operations)
- Typical reduction: ~5-10 API calls per cycle → 0-1 API calls
- Battle mode: Each model instance uses cached data from ModelInstance
- Can be disabled by passing use_cache=False to get_current_portfolio_state()
"""

import logging
import argparse
from pathlib import Path
from datetime import datetime, time
from typing import Dict, List, Optional
import uuid
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from .env file
load_dotenv()

from active_trader_llm.config.loader import load_config
from active_trader_llm.battle.shared_data_feed import DataSnapshot
from active_trader_llm.data_ingestion.price_volume_ingestor import PriceVolumeIngestor
from active_trader_llm.data_ingestion.macro_data_ingestor import MacroDataIngestor
from active_trader_llm.feature_engineering.feature_builder import FeatureBuilder
# from active_trader_llm.analysts.technical_analyst import TechnicalAnalyst  # Removed - no LLM interpretation
# from active_trader_llm.analysts.sentiment_analyst import SentimentAnalyst  # Removed - not implemented
# from active_trader_llm.analysts.macro_analyst import MacroAnalyst  # Removed - using raw macro data
# from active_trader_llm.researchers.bull_bear import ResearcherDebate  # Removed - trader decides from raw data
from active_trader_llm.trader.trader_agent import TraderAgent
from active_trader_llm.risk.risk_manager import RiskManager, PortfolioState
from active_trader_llm.memory.memory_manager import MemoryManager, TradeMemory
# REMOVED: StrategyMonitor - LLM decides dynamically
# from active_trader_llm.learning.strategy_monitor import StrategyMonitor
from active_trader_llm.utils.logging_json import JSONLogger
from active_trader_llm.scanners.scanner_orchestrator import ScannerOrchestrator
from active_trader_llm.execution.position_manager import PositionManager
from active_trader_llm.execution.exit_monitor import ExitMonitor
from active_trader_llm.execution.alpaca_broker import AlpacaBrokerExecutor
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ActiveTraderLLM:
    """
    Main orchestrator for the multi-agent trading system.
    """

    def __init__(self, config_path: str, mode_override: str = None, cached_account_info: dict = None):
        """
        Initialize the trading system.

        Args:
            config_path: Path to YAML configuration file
            mode_override: Optional mode override for different trading modes
            cached_account_info: Optional cached account info from ModelInstance (reduces API calls)
        """
        logger.info(f"Initializing ActiveTrader-LLM from {config_path}")

        # Store cached account info to avoid redundant fetches
        self._cached_account_info = cached_account_info
        self._account_cache_timestamp = datetime.now() if cached_account_info else None

        # Load configuration
        self.config = load_config(config_path)

        # Apply mode override if provided
        if mode_override:
            self.config.mode = mode_override
            logger.info(f"Mode overridden to: {mode_override}")

        logger.info(f"Mode: {self.config.mode}")

        # Initialize components
        self.ingestor = PriceVolumeIngestor(
            cache_db_path="data/price_cache.db",
            mode=self.config.mode  # Pass mode for Alpaca/yfinance selection
        )
        self.macro_ingestor = MacroDataIngestor(
            cache_duration_seconds=self.config.macro_data.cache_duration_seconds
        ) if self.config.enable_macro else None
        self.feature_builder = FeatureBuilder(self.config.model_dump())

        # Initialize analysts
        api_key = self.config.llm.api_key
        model = self.config.llm.model
        provider = self.config.llm.provider
        base_url = getattr(self.config.llm, 'base_url', None)  # Optional for OpenAI-compatible APIs

        # technical_analyst removed - validation done directly, no LLM interpretation needed
        self.sentiment_analyst = None  # Sentiment analyst not implemented
        # macro_analyst removed - using raw macro data instead of LLM interpretation

        # researcher_debate removed - trader_agent makes decisions from raw data only

        # Initialize trader with validation and unified LLM client
        from active_trader_llm.trader.trade_plan_validator import ValidationConfig
        validation_config = ValidationConfig(
            max_position_pct=self.config.trade_validation.max_position_pct,
            min_risk_reward_ratio=self.config.trade_validation.min_risk_reward_ratio,
            max_price_deviation_pct=self.config.trade_validation.max_price_deviation_pct,
            min_stop_distance_pct=self.config.trade_validation.min_stop_distance_pct,
            max_stop_distance_pct=self.config.trade_validation.max_stop_distance_pct
        )
        self.trader_agent = TraderAgent(
            api_key=api_key,
            model=model,
            provider=provider,
            base_url=base_url,
            validation_config=validation_config,
            enable_validation=self.config.trade_validation.enabled
        )

        # Initialize risk manager
        self.risk_manager = RiskManager(self.config.risk_parameters.model_dump())

        # Initialize memory and learning
        self.memory = MemoryManager(self.config.database_path)
        # REMOVED: Strategy monitoring - LLM decides dynamically
        # self.strategy_monitor = StrategyMonitor(
        #     self.memory,
        #     self.config.strategy_switching.model_dump()
        # )

        # Initialize position management
        self.position_manager = PositionManager(self.config.database_path.replace('trading.db', 'positions.db'))
        self.exit_monitor = ExitMonitor(self.position_manager)

        # Pass position_manager to risk_manager
        self.risk_manager.position_manager = self.position_manager

        # Initialize logging
        self.json_logger = JSONLogger(self.config.log_path)

        # Initialize broker executor
        self.broker_executor = None

        if self.config.mode in ["paper-live", "live"]:
            # FORCED: Always use Alpaca broker (no simulated fallback)
            # Battle configs have alpaca credentials in config.alpaca.*
            # Regular configs have them in config.execution.*
            if hasattr(self.config, 'alpaca'):
                # Battle config structure
                api_key_env_name = self.config.alpaca.api_key_env
                secret_key_env_name = self.config.alpaca.secret_key_env
                base_url = self.config.alpaca.base_url
            else:
                # Regular config structure
                api_key_env_name = self.config.execution.alpaca_api_key_env
                secret_key_env_name = self.config.execution.alpaca_secret_key_env
                base_url = self.config.execution.alpaca_base_url

            api_key = os.getenv(api_key_env_name)
            secret_key = os.getenv(secret_key_env_name)

            if not api_key or not secret_key:
                raise ValueError(f"Missing Alpaca credentials: {api_key_env_name} or {secret_key_env_name} not set in environment")

            self.broker_executor = AlpacaBrokerExecutor(
                api_key=api_key,
                secret_key=secret_key,
                paper=self.config.execution.paper_trading,
                base_url=base_url,
                extended_hours=getattr(self.config.execution, 'extended_hours', False)
            )
            logger.info(f"Alpaca broker executor initialized (paper={self.config.execution.paper_trading})")

        # Initialize scanner (if enabled)
        self.scanner = None
        if self.config.scanner.enabled:
            try:
                self.scanner = ScannerOrchestrator(
                    data_fetcher=self.ingestor,
                    alpaca_api_key=os.getenv('ALPACA_API_KEY'),
                    alpaca_secret_key=os.getenv('ALPACA_SECRET_KEY'),
                    alpaca_base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
                    paper=self.config.execution.paper_trading,  # Thread execution mode
                    openai_api_key=os.getenv('OPENAI_API_KEY'),
                    anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
                    llm_provider=provider,
                    llm_model=model,
                    requests_per_minute=200  # Standard plan
                )
                mode = "PAPER" if self.config.execution.paper_trading else "LIVE"
                logger.info(f"Market scanner initialized (two-stage mode enabled, {mode} mode, using {provider}/{model})")
            except Exception as e:
                logger.warning(f"Failed to initialize scanner: {e}")
                logger.warning("Falling back to fixed universe mode")
                self.config.scanner.enabled = False

        # REMOVED: Current strategy tracking - LLM decides dynamically
        # self.current_strategy = self.config.strategies[0].name

        # Initialize portfolio state (mode-dependent)
        if self.config.mode in ["paper-live", "live"] and self.broker_executor:
            # Paper-live/live: use cached account info if available, otherwise fetch from Alpaca
            try:
                # OPTIMIZATION: Use cached account info to avoid initial API call
                if self._cached_account_info:
                    account_info = self._cached_account_info
                    logger.info("Portfolio initialized from CACHED account info (0 API calls)")
                else:
                    account_info = self.broker_executor.get_account_info()
                    logger.debug("Portfolio initialized from Alpaca API fetch (1 API call)")

                self.portfolio_state = PortfolioState(
                    cash=float(account_info['cash']),
                    equity=float(account_info['equity']),
                    positions=[],  # Will be populated from position_manager
                    daily_pnl=0.0
                )
                logger.info(
                    f"Portfolio initialized: Alpaca account with "
                    f"${float(account_info['equity']):,.2f} equity, "
                    f"${float(account_info['cash']):,.2f} cash"
                )
            except Exception as e:
                logger.error(f"Failed to fetch Alpaca account info: {e}")
                # Fallback to default
                self.portfolio_state = PortfolioState(
                    cash=0.0, equity=0.0, positions=[], daily_pnl=0.0
                )
        else:
            # Fallback for other modes
            self.portfolio_state = PortfolioState(
                cash=100000.0, equity=100000.0, positions=[], daily_pnl=0.0
            )
            logger.info("Portfolio initialized with default $100,000")

        # Scanner state
        self.scanned_universe = None

        logger.info("ActiveTrader-LLM initialized successfully")

    def update_account_cache(self, account_info: dict):
        """
        Update cached account info (called by battle orchestrator or broker).

        Args:
            account_info: Dict with cash, equity, and other account fields
        """
        self._cached_account_info = account_info
        self._account_cache_timestamp = datetime.now()
        logger.debug("Account cache updated externally")

    def invalidate_account_cache(self):
        """
        Invalidate the account cache (forces fresh fetch on next query).
        """
        self._cached_account_info = None
        self._account_cache_timestamp = None
        logger.debug("Account cache invalidated")

    def get_current_portfolio_state(self, use_cache: bool = True) -> PortfolioState:
        """
        Get current portfolio state (mode-aware).

        Args:
            use_cache: If True, use cached account info when available (default=True)
                      Set to False to force fresh fetch from broker

        Returns:
            PortfolioState with fresh values:
            - Paper-live/live: Uses cached data or fetches fresh from Alpaca broker

        This ensures position sizing and risk checks always use
        accurate account balances in live trading.
        """
        if self.config.mode in ["paper-live", "live"] and self.broker_executor:
            # OPTIMIZATION: Use cached account info if recent enough (within 5 seconds)
            cache_ttl_seconds = 5
            cache_is_fresh = (
                use_cache and
                self._cached_account_info and
                self._account_cache_timestamp and
                (datetime.now() - self._account_cache_timestamp).total_seconds() < cache_ttl_seconds
            )

            if cache_is_fresh:
                # Use cached data - NO API CALL
                account_info = self._cached_account_info
                logger.debug("Portfolio state source: CACHED account info (0 API calls)")
            else:
                # Fetch fresh from Alpaca
                try:
                    account_info = self.broker_executor.get_account_info()
                    # Update cache
                    self._cached_account_info = account_info
                    self._account_cache_timestamp = datetime.now()
                    logger.debug("Portfolio state source: Alpaca API fetch (1 API call, cache updated)")
                except Exception as e:
                    logger.warning(f"Failed to fetch fresh portfolio state from Alpaca: {e}")
                    if self._cached_account_info:
                        logger.warning("Using stale cached state as fallback")
                        account_info = self._cached_account_info
                    else:
                        logger.warning("Using last known state as fallback")
                        return self.portfolio_state

            # Get positions from current portfolio_state (already synced from position_manager)
            positions_list = self.portfolio_state.positions

            fresh_state = PortfolioState(
                cash=float(account_info['cash']),
                equity=float(account_info['equity']),
                positions=positions_list,
                daily_pnl=0.0  # Could calculate from last_equity
            )

            logger.debug(
                f"Portfolio state: equity=${fresh_state.equity:,.2f}, cash=${fresh_state.cash:,.2f}"
            )
            return fresh_state

        else:
            # Fallback to tracked state
            logger.debug("Portfolio state source: Tracked state (fallback, 0 API calls)")
            return self.portfolio_state

    def _sync_positions_from_alpaca(self):
        """
        Sync positions from Alpaca to local database.

        This ensures local position tracking is always in sync with broker state.
        Runs at the start of each decision cycle in live/paper-live mode.
        """
        if not self.broker_executor:
            logger.warning("No broker executor available for position sync")
            return

        try:
            # Fetch all positions from Alpaca
            alpaca_positions = self.broker_executor.get_positions()

            if not alpaca_positions:
                logger.info("  No positions in Alpaca to sync")
                return

            logger.info(f"  Found {len(alpaca_positions)} positions in Alpaca")

            # Get all open orders to find bracket orders
            try:
                from alpaca.trading.requests import GetOrdersRequest
                from alpaca.trading.enums import QueryOrderStatus

                order_request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
                all_orders = self.broker_executor.client.get_orders(filter=order_request)
                logger.debug(f"  Found {len(all_orders)} open orders")
            except Exception as e:
                logger.warning(f"  Failed to fetch open orders: {e}")
                all_orders = []

            synced_count = 0
            skipped_count = 0

            # Get account equity for position size calculation
            account = self.broker_executor.client.get_account()
            account_equity = float(account.equity)

            for pos in alpaca_positions:
                symbol = pos.symbol
                qty = pos.qty
                entry_price = pos.avg_entry_price

                # Check if position already exists in local database
                if self.position_manager.has_open_position(symbol):
                    logger.debug(f"  {symbol}: Already in local DB - skipping")
                    skipped_count += 1
                    continue

                # Determine direction
                direction = 'long' if qty > 0 else 'short'
                shares = abs(int(qty))

                # Find bracket orders for this symbol
                symbol_orders = [o for o in all_orders if o.symbol == symbol]

                # Look for stop_loss and take_profit orders
                stop_loss_price = None
                take_profit_price = None

                for order in symbol_orders:
                    if hasattr(order, 'stop_price') and order.stop_price:
                        stop_loss_price = float(order.stop_price)

                    if hasattr(order, 'limit_price') and order.limit_price:
                        # Check if this is a take-profit order (not entry order)
                        if hasattr(order, 'order_class') and 'bracket' in str(order.order_class).lower():
                            take_profit_price = float(order.limit_price)

                # Only sync if we have both stop-loss and take-profit
                if stop_loss_price and take_profit_price:
                    # Calculate position size percentage
                    position_value = abs(qty) * entry_price
                    position_size_pct = (position_value / account_equity) * 100.0

                    # Add position to local database
                    self.position_manager.open_position(
                        symbol=symbol,
                        direction=direction,
                        entry_price=entry_price,
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price,
                        shares=shares,
                        strategy='synced_from_alpaca',
                        position_size_pct=position_size_pct,
                        fill_timestamp=datetime.now()
                    )

                    logger.info(f"  {symbol}: SYNCED ({direction}, stop=${stop_loss_price:.2f}, target=${take_profit_price:.2f})")
                    synced_count += 1
                else:
                    logger.debug(f"  {symbol}: Missing bracket orders - skipped")
                    skipped_count += 1

            if synced_count > 0:
                logger.info(f"  Position sync complete: {synced_count} synced, {skipped_count} skipped")
            else:
                logger.debug(f"  No new positions to sync ({skipped_count} already in DB)")

        except Exception as e:
            logger.error(f"Error syncing positions from Alpaca: {e}", exc_info=True)
            raise

    def _apply_programmatic_filters(
        self,
        analyst_outputs: Dict,
        current_prices: Dict,
        min_volume_ratio: float = 2.0,
        min_momentum_5d: float = 2.0,
        max_distance_from_high: float = 10.0,
        min_liquidity: float = 500_000_000,
        min_adr: float = 1.0,
        max_adr: float = 15.0
    ) -> List[str]:
        """
        Apply programmatic filters to reduce API calls before trader agent analysis.

        Filters stocks using basic technical criteria to identify high-quality candidates.
        Only candidates that pass all filters will be sent to the LLM for analysis.

        Args:
            analyst_outputs: Dict of symbol -> analyst data
            current_prices: Dict of symbol -> current price
            min_volume_ratio: Minimum volume ratio (current / average)
            min_momentum_5d: Minimum 5-day price change percentage
            max_distance_from_high: Maximum distance from 52-week high (%)
            min_liquidity: Minimum daily liquidity ($)
            min_adr: Minimum average daily range (%)
            max_adr: Maximum average daily range (%)

        Returns:
            List of symbols that pass all filters
        """
        logger.info(f"\n=== PROGRAMMATIC PRE-FILTERING (Reducing API Calls) ===")
        logger.info(f"Initial candidates: {len(analyst_outputs)}")

        filtered_symbols = []
        filter_stats = {
            'volume': 0,
            'momentum': 0,
            '52w_high': 0,
            'liquidity': 0,
            'adr': 0,
            'passed_all': 0
        }

        for symbol, data in analyst_outputs.items():
            features = data.get('features', {})
            ohlcv = features.get('ohlcv', {})
            daily_indicators = features.get('daily_indicators', {})

            # Get required metrics
            current_price = current_prices.get(symbol, 0.0)
            current_volume = ohlcv.get('volume', 0)
            avg_volume = features.get('avg_volume', 1)

            # Calculate metrics
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0.0

            # Calculate 5-day momentum from price series
            price_series = data.get('price_series', [])
            if len(price_series) >= 6:
                price_5d_ago = price_series[-6]
                momentum_5d = ((current_price - price_5d_ago) / price_5d_ago * 100) if price_5d_ago > 0 else 0.0
            else:
                momentum_5d = 0.0

            # Get 52-week high
            high_52w = daily_indicators.get('high_52w', current_price)
            distance_from_high = ((current_price - high_52w) / high_52w * 100) if high_52w > 0 else -100.0

            # Calculate daily liquidity (price * volume)
            daily_liquidity = current_price * current_volume

            # Calculate ADR (Average Daily Range)
            atr = daily_indicators.get('ATR_14_daily', None)
            # Skip ADR check if ATR data is missing (don't penalize symbols for missing data)
            if atr is not None and atr > 0:
                adr_percent = (atr / current_price * 100) if current_price > 0 else 0.0
            else:
                adr_percent = None  # Signal that ADR data is unavailable

            # Apply filters
            passed = True

            # Filter 1: Volume ratio
            if volume_ratio < min_volume_ratio:
                passed = False
            else:
                filter_stats['volume'] += 1

            # Filter 2: Momentum
            if passed and momentum_5d < min_momentum_5d:
                passed = False
            else:
                if passed:
                    filter_stats['momentum'] += 1

            # Filter 3: Distance from 52-week high
            if passed and distance_from_high < -max_distance_from_high:
                passed = False
            else:
                if passed:
                    filter_stats['52w_high'] += 1

            # Filter 4: Daily liquidity
            if passed and daily_liquidity < min_liquidity:
                passed = False
            else:
                if passed:
                    filter_stats['liquidity'] += 1

            # Filter 5: ADR range (skip if ATR data unavailable)
            if passed and adr_percent is not None:
                if adr_percent < min_adr or adr_percent > max_adr:
                    passed = False
                else:
                    filter_stats['adr'] += 1
            else:
                # ATR data missing, pass this filter automatically
                if passed:
                    filter_stats['adr'] += 1

            if passed:
                filter_stats['passed_all'] += 1
                filtered_symbols.append(symbol)
                adr_str = f"{adr_percent:.2f}%" if adr_percent is not None else "N/A"
                logger.debug(
                    f"{symbol}: PASS - Vol ratio: {volume_ratio:.2f}x, "
                    f"Momentum: {momentum_5d:+.2f}%, "
                    f"From high: {distance_from_high:.2f}%, "
                    f"Liquidity: ${daily_liquidity/1e6:.1f}M, "
                    f"ADR: {adr_str}"
                )

        logger.info(f"\nFilter Results:")
        logger.info(f"  Passed volume (>{min_volume_ratio}x): {filter_stats['volume']}")
        logger.info(f"  Passed momentum (>{min_momentum_5d}%): {filter_stats['momentum']}")
        logger.info(f"  Passed 52w high (<{max_distance_from_high}%): {filter_stats['52w_high']}")
        logger.info(f"  Passed liquidity (>${min_liquidity/1e6:.0f}M): {filter_stats['liquidity']}")
        logger.info(f"  Passed ADR ({min_adr}-{max_adr}%): {filter_stats['adr']}")
        logger.info(f"  PASSED ALL FILTERS: {filter_stats['passed_all']}")

        # Avoid division by zero when no analyst outputs
        if len(analyst_outputs) > 0:
            reduction_pct = 100*(1-len(filtered_symbols)/len(analyst_outputs))
            logger.info(f"\nReduction: {len(analyst_outputs)} → {len(filtered_symbols)} symbols ({reduction_pct:.1f}% filtered)")
            logger.info(f"API call savings: ~{len(analyst_outputs) - len(filtered_symbols)} calls avoided\n")
        else:
            logger.info(f"\nReduction: 0 → 0 symbols (no candidates to filter)\n")

        return filtered_symbols

    def run_decision_cycle(self, pre_fetched_snapshot: Optional[DataSnapshot] = None):
        """
        Execute one complete decision cycle.

        Args:
            pre_fetched_snapshot: Optional pre-fetched market data snapshot from SharedDataFeed.
                                 If provided, skips data fetching and feature engineering (Steps 1-2).
                                 This significantly improves performance in battle mode (80%+ reduction).

        Workflow:
        0. Load positions and check exits
        1. Fetch latest data (SKIPPED if pre_fetched_snapshot provided)
        2. Compute features (SKIPPED if pre_fetched_snapshot provided)
        3. Run analysts
        4. Run researcher debate
        5. Generate trade plans
        6. Risk validation
        7. Log decisions
        8. (Execution would happen here in paper-live mode)
        """
        logger.info("=" * 80)
        logger.info(f"Decision Cycle Start: {datetime.now()}")
        logger.info("=" * 80)

        try:
            # Step 0: Sync positions with Alpaca (for live/paper-live mode)
            if self.config.mode in ["paper-live", "live"] and self.broker_executor:
                logger.info("Step 0: Syncing positions with Alpaca...")
                try:
                    self._sync_positions_from_alpaca()
                except Exception as e:
                    logger.error(f"Position sync failed: {e}", exc_info=True)
                    logger.warning("Continuing with local positions only")

            # Step 0a: Determine active universe
            if self.config.scanner.enabled and self.scanner:
                # Run full scanner
                logger.info("Step 0a: Running two-stage market scanner...")
                try:
                    self.scanned_universe = self.scanner.run_full_scan(
                        force_refresh_universe=False,
                        refresh_hours=self.config.scanner.refresh_universe_hours,
                        batch_size=self.config.scanner.stage2.batch_size,
                        max_batches=self.config.scanner.stage2.max_batches,
                        max_candidates=self.config.scanner.stage2.max_candidates
                    )
                    logger.info(f"Scanner found {len(self.scanned_universe)} candidates")
                    active_universe = self.scanned_universe
                except Exception as e:
                    logger.error(f"Scanner failed: {e}")
                    logger.warning("Falling back to fixed universe")
                    active_universe = self.config.data_sources.universe
            elif self.config.scanner.universe_source == "file" and self.config.scanner.universe_file_path:
                # Load from pre-filtered universe file
                logger.info(f"Step 0a: Loading pre-filtered universe from {self.config.scanner.universe_file_path}...")
                try:
                    file_path = Path(self.config.scanner.universe_file_path)
                    if file_path.exists():
                        with open(file_path, 'r') as f:
                            active_universe = [line.strip() for line in f if line.strip()]
                        logger.info(f"Loaded {len(active_universe)} symbols from pre-filtered universe")
                        self.scanned_universe = active_universe
                    else:
                        logger.error(f"Universe file not found: {file_path}")
                        logger.warning("Falling back to fixed universe from config")
                        active_universe = self.config.data_sources.universe
                except Exception as e:
                    logger.error(f"Failed to load universe file: {e}")
                    logger.warning("Falling back to fixed universe")
                    active_universe = self.config.data_sources.universe
            else:
                # Use fixed universe from config
                active_universe = self.config.data_sources.universe
                logger.info(f"Using fixed universe from config: {len(active_universe)} symbols")

            # Step 1 & 2: Fetch data and compute features (OR use pre-fetched snapshot)
            if pre_fetched_snapshot:
                # Use pre-fetched data from SharedDataFeed (battle mode optimization)
                logger.info("Step 1-2: Using PRE-FETCHED market snapshot (SKIPPED data loading)")
                logger.info(f"  Snapshot timestamp: {pre_fetched_snapshot.timestamp}")
                logger.info(f"  Symbols in snapshot: {len(pre_fetched_snapshot.symbols)}")

                # Reconstruct data structures from snapshot
                price_df = pd.DataFrame(pre_fetched_snapshot.price_data)
                features_dict = pre_fetched_snapshot.features
                market_snapshot = type('MarketSnapshot', (), pre_fetched_snapshot.market_breadth)() if pre_fetched_snapshot.market_breadth else None

                # Convert market_breadth dict to object for dot notation (backward compatibility)
                if pre_fetched_snapshot.market_breadth:
                    from types import SimpleNamespace
                    market_snapshot = SimpleNamespace(**pre_fetched_snapshot.market_breadth)

                logger.info("✓ Pre-fetched data loaded successfully (massive performance gain)")
            else:
                # Original flow: Fetch data independently
                logger.info("Step 1: Fetching market data...")
                price_df = self.ingestor.fetch_prices(
                    universe=active_universe,
                    interval=self.config.data_sources.interval,
                    lookback_days=self.config.data_sources.lookback_days,
                    use_cache=self.config.cost_control.cache_data
                )

                if price_df.empty:
                    logger.error("No price data fetched. Aborting cycle.")
                    return

                # Step 2: Compute features
                logger.info("Step 2: Computing technical features...")
                features_dict = self.feature_builder.build_features(price_df)

                if not features_dict:
                    logger.error("No features computed. Aborting cycle.")
                    return

                market_snapshot = self.feature_builder.build_market_snapshot(price_df, features_dict)
            logger.info(
                f"Market breadth: {market_snapshot.stocks_advancing} advancing / "
                f"{market_snapshot.stocks_declining} declining "
                f"(total: {market_snapshot.total_stocks}) | "
                f"New highs: {market_snapshot.new_highs}, lows: {market_snapshot.new_lows}"
            )

            # Step 0b: Load positions and check exits (after we have current prices)
            logger.info("\nStep 0b: Checking existing positions...")

            # Get current prices for all symbols (including active universe)
            symbol_data = {}
            for symbol in active_universe:
                if symbol in features_dict:
                    # Get current price from features
                    current_price = features_dict[symbol].close
                    symbol_data[symbol] = {'price': current_price}

            # Build current_prices dict
            current_prices = {symbol: data['price'] for symbol, data in symbol_data.items()}

            # Check for exits on existing positions
            if self.exit_monitor:
                closed_positions = self.exit_monitor.check_exits(current_prices)
                if closed_positions:
                    for pos in closed_positions:
                        logger.info(
                            f"Position CLOSED: {pos.symbol} {pos.direction} @ {pos.exit_price:.2f} "
                            f"(reason: {pos.exit_reason}, P&L: ${pos.realized_pnl:.2f})"
                        )

            # Get current portfolio state from position manager (returns dict)
            portfolio_dict = self.position_manager.get_portfolio_state(current_prices, self.portfolio_state.equity)

            # Update positions list in portfolio_state
            self.portfolio_state.positions = portfolio_dict.get('open_positions', [])

            # In paper-live/live mode, refresh cash/equity from Alpaca (using cache)
            if self.config.mode in ["paper-live", "live"] and self.broker_executor:
                try:
                    # OPTIMIZATION: Use get_current_portfolio_state with caching (0-1 API calls instead of 1)
                    cached_state = self.get_current_portfolio_state(use_cache=True)
                    self.portfolio_state.cash = cached_state.cash
                    self.portfolio_state.equity = cached_state.equity
                    logger.debug(f"Refreshed portfolio state (cached): equity=${self.portfolio_state.equity:,.2f}, cash=${self.portfolio_state.cash:,.2f}")
                except Exception as e:
                    logger.warning(f"Could not refresh account info from Alpaca: {e}")

            logger.info(
                f"Portfolio: {len(portfolio_dict.get('open_positions', []))} open positions, "
                f"Equity: ${self.portfolio_state.equity:,.2f}, "
                f"Cash: ${self.portfolio_state.cash:,.2f}, "
                f"Unrealized P&L: ${portfolio_dict.get('total_unrealized_pnl', 0):.2f}"
            )

            # Step 3: Run analysts
            logger.info("Step 3: Running analyst agents...")
            analyst_outputs = {}

            for symbol, features in features_dict.items():
                logger.info(f"\nPreparing data for {symbol}...")

                # Validate critical data (price) - NO LLM call needed
                price = features.close
                atr = features.daily_indicators.get('ATR_14_daily')

                if price is None or price == 0.0:
                    logger.warning(f"{symbol}: Price data missing or zero - skipping symbol")
                    continue

                # ATR is useful but not critical - log warning if missing but don't skip
                if atr is None:
                    logger.debug(f"{symbol}: ATR data missing - proceeding without ATR")

                # Extract price series from price_df for trader_agent (last 20 bars)
                symbol_prices = price_df[price_df['symbol'] == symbol].tail(20)
                price_series = symbol_prices['close'].tolist() if not symbol_prices.empty else [features.close]

                # Handle both Pydantic models and SimpleNamespace for market_snapshot
                if hasattr(market_snapshot, 'model_dump'):
                    breadth_dict = market_snapshot.model_dump()
                else:
                    # For SimpleNamespace (from pre-fetched snapshot), convert to dict
                    breadth_dict = vars(market_snapshot) if hasattr(market_snapshot, '__dict__') else {}

                analyst_outputs[symbol] = {
                    'features': features.model_dump(),  # Raw features for trader_agent
                    'price_series': price_series,  # Historical prices for LLM
                    'breadth': breadth_dict  # Add breadth to each symbol's context
                }

                # Add optional analysts
                if self.sentiment_analyst:
                    sent_signal = self.sentiment_analyst.analyze(symbol)
                    if sent_signal:
                        analyst_outputs[symbol]['sentiment'] = sent_signal.model_dump()
                    else:
                        logger.warning(f"{symbol}: Sentiment analysis returned None (stub mode)")
                        # Don't add sentiment to analyst_outputs - trading will proceed without it

            # Macro data (if enabled) - RAW DATA ONLY, no LLM interpretation
            if self.macro_ingestor:
                logger.info("Fetching macro-economic data...")
                macro_snapshot = self.macro_ingestor.fetch_all()

                if macro_snapshot:
                    # Log key macro metrics
                    vix_str = f"VIX: {macro_snapshot.vix:.1f}" if macro_snapshot.vix else "VIX: N/A"
                    yield_10y_str = f"10Y: {macro_snapshot.treasury_10y:.2f}%" if macro_snapshot.treasury_10y else "10Y: N/A"
                    logger.info(f"Macro data: {vix_str}, {yield_10y_str}")

                    # Add raw macro context to ALL symbols (no LLM interpretation)
                    for symbol in analyst_outputs.keys():
                        analyst_outputs[symbol]['macro'] = macro_snapshot.model_dump()
                else:
                    logger.warning("Macro data fetch failed - trading will proceed without macro context")

            # Step 4: Generate trade plans (Fully Autonomous LLM Decision from Raw Data)
            logger.info("Step 4: Generating trade plans...")
            trade_plans = []

            # Build account state for LLM context
            portfolio_dict = self.position_manager.get_portfolio_state(current_prices, self.portfolio_state.equity)
            account_state = {
                'cash': self.portfolio_state.cash,
                'equity': self.portfolio_state.equity,
                'position_count': portfolio_dict['position_count'],
                'total_exposure': portfolio_dict['total_exposure'],
                'exposure_pct': portfolio_dict['total_exposure_pct']
            }

            # Step 4a: Apply programmatic pre-filtering to reduce API calls
            # Read filter settings from config
            batch_config = self.config.model_dump().get('batch_processing', {})
            filters = batch_config.get('filters', {})

            # Check if batch processing is enabled (for filtering)
            batch_enabled = batch_config.get('enabled', False)

            if batch_enabled:
                # Apply programmatic filters if batch processing is enabled
                filtered_symbols = self._apply_programmatic_filters(
                    analyst_outputs,
                    current_prices,
                    min_volume_ratio=filters.get('min_volume_ratio', 2.0),
                    min_momentum_5d=filters.get('min_momentum_5d', 2.0),
                    max_distance_from_high=filters.get('max_distance_from_high', 10.0),
                    min_liquidity=filters.get('min_liquidity', 500_000_000),
                    min_adr=filters.get('min_adr', 1.0),
                    max_adr=filters.get('max_adr', 15.0)
                )
            else:
                # No filtering - let LLM evaluate all symbols
                logger.info(f"\n=== PROGRAMMATIC FILTERING DISABLED ===")
                logger.info(f"Batch processing disabled - LLM will evaluate all {len(analyst_outputs)} symbols")
                filtered_symbols = list(analyst_outputs.keys())

            # Step 4b: Separate existing positions from new candidates
            symbols_with_positions = []
            new_candidate_symbols = []

            for symbol in analyst_outputs.keys():
                if self.position_manager.has_open_position(symbol):
                    symbols_with_positions.append(symbol)
                elif symbol in filtered_symbols:
                    new_candidate_symbols.append(symbol)

            logger.info(f"\nSymbol breakdown:")
            logger.info(f"  Existing positions to review: {len(symbols_with_positions)}")
            logger.info(f"  New candidates (passed filters): {len(new_candidate_symbols)}")
            logger.info(f"  Total symbols processed: {len(analyst_outputs)}")

            # Step 4c: Review existing positions individually (close/hold decisions)
            logger.info(f"\n=== REVIEWING {len(symbols_with_positions)} EXISTING POSITIONS ===")
            for symbol in symbols_with_positions:

                # Check if we already have a position in this symbol
                existing_position = None

                if self.position_manager.has_open_position(symbol):
                    # Get position details for LLM to decide hold/close
                    open_positions = [p for p in portfolio_dict['open_positions'] if p['symbol'] == symbol]
                    if open_positions:
                        existing_position = open_positions[0]

                # Prepare all open positions for comparative analysis
                all_open_positions_data = []
                all_open_positions = self.position_manager.get_open_positions()
                for open_pos in all_open_positions:
                    pos_current_price = current_prices.get(open_pos.symbol, open_pos.entry_price)
                    pos_unrealized_pnl = (pos_current_price - open_pos.entry_price) * open_pos.shares
                    pos_days_held = (datetime.now() - open_pos.entry_timestamp).days if hasattr(open_pos, 'entry_timestamp') else 0

                    all_open_positions_data.append({
                        'symbol': open_pos.symbol,
                        'direction': open_pos.direction,
                        'entry_price': open_pos.entry_price,
                        'current_price': pos_current_price,
                        'stop_loss': open_pos.stop_loss,
                        'take_profit': open_pos.take_profit,
                        'shares': open_pos.shares,
                        'unrealized_pnl': pos_unrealized_pnl,
                        'days_held': pos_days_held,
                        'original_rationale': getattr(open_pos, 'entry_rationale', 'Not available'),
                        'invalidation_condition': getattr(open_pos, 'invalidation_condition', 'Not specified')
                    })

                # Call trader agent with full context (nof1.ai style - raw data only)
                plan = self.trader_agent.decide(
                    symbol,
                    analyst_outputs[symbol],
                    (None, None),  # researcher_outputs removed - not used in prompt
                    self.config.risk_parameters.model_dump(),
                    memory_context=None,  # TODO: Add recent performance context
                    existing_position=existing_position,
                    account_state=account_state,
                    all_open_positions=all_open_positions_data
                )

                if plan:
                    # Handle different action types
                    if plan.action == 'close':
                        # LLM decided to close existing position
                        logger.info(f"{symbol}: LLM decided to CLOSE position (invalidation triggered)")

                        # Execute the LLM's close decision
                        if self.position_manager.has_open_position(symbol):
                            # Fetch fresh real-time price (NEVER use stale data for closes)
                            logger.debug(f"{symbol}: Fetching fresh price for position close")
                            fresh_price_df = self.ingestor.fetch_prices(
                                universe=[symbol],
                                interval=self.config.data_sources.interval,
                                lookback_days=1,
                                use_cache=False  # Force fresh data
                            )

                            if not fresh_price_df.empty:
                                current_price = fresh_price_df[fresh_price_df['symbol'] == symbol]['close'].iloc[-1]
                                logger.debug(f"{symbol}: Fresh price fetched: ${current_price:.2f}")
                            else:
                                # Fallback to current_prices dict only if fresh fetch fails
                                current_price = current_prices.get(symbol, 0.0)
                                logger.warning(f"{symbol}: Failed to fetch fresh price, using cached: ${current_price:.2f}")

                            closed_pos = self.position_manager.close_position(
                                symbol=symbol,
                                exit_price=current_price,
                                exit_reason=f"LLM decision: {plan.rationale[:100]}"
                            )
                            logger.info(f"Position CLOSED: {symbol} @ {closed_pos.exit_price:.2f}, P&L: ${closed_pos.realized_pnl:.2f}")
                        continue

                    elif plan.action == 'hold':
                        # LLM decided to keep existing position
                        logger.info(f"{symbol}: LLM decided to HOLD existing position (confidence: {plan.confidence:.2f})")
                        continue

                    elif plan.action == 'pass':
                        # LLM decided not to trade
                        logger.info(f"{symbol}: LLM passed on opportunity")
                        continue

                    elif plan.action in ['open_long', 'open_short']:
                        # LLM decided to open new position (from existing position review)
                        trade_plans.append(plan)
                        logger.info(
                            f"Trade plan for {symbol}: {plan.action} @ {plan.entry:.2f} "
                            f"(size: {plan.position_size_pct*100:.1f}%, confidence: {plan.confidence:.2f})"
                        )

            # Step 4d: Batch process new candidates (MAJOR API CALL SAVINGS)
            # MODIFIED: Execute trades immediately after each batch for instant execution
            logger.info(f"\n=== BATCH ANALYZING {len(new_candidate_symbols)} NEW CANDIDATES ===")

            if new_candidate_symbols:
                # Process in batches (size from config)
                batch_size = batch_config.get('batch_size', 10)
                total_batches = (len(new_candidate_symbols) + batch_size - 1) // batch_size

                logger.info(f"Processing {len(new_candidate_symbols)} candidates in {total_batches} batches of {batch_size}")
                logger.info("IMMEDIATE EXECUTION MODE: Trades will be executed after each batch completes")

                total_plans_generated = 0
                total_plans_approved = 0
                total_plans_executed = 0

                for batch_idx in range(0, len(new_candidate_symbols), batch_size):
                    batch_symbols = new_candidate_symbols[batch_idx:batch_idx + batch_size]
                    batch_num = (batch_idx // batch_size) + 1

                    logger.info(f"\n{'='*80}")
                    logger.info(f"BATCH {batch_num}/{total_batches}: Analyzing {len(batch_symbols)} stocks")
                    logger.info(f"{'='*80}")
                    logger.debug(f"  Symbols: {', '.join(batch_symbols)}")

                    # Single API call for entire batch
                    batch_plans = self.trader_agent.decide_batch(
                        symbols=batch_symbols,
                        analyst_outputs=analyst_outputs,
                        risk_params=self.config.risk_parameters.model_dump(),
                        account_state=account_state
                    )

                    # Filter for open actions only
                    batch_trade_plans = []
                    for plan in batch_plans:
                        if plan.action in ['open_long', 'open_short']:
                            batch_trade_plans.append(plan)
                            total_plans_generated += 1
                            logger.info(
                                f"Trade plan from batch: {plan.symbol} {plan.action} @ {plan.entry:.2f} "
                                f"(size: {plan.position_size_pct*100:.1f}%, confidence: {plan.confidence:.2f})"
                            )

                    if not batch_trade_plans:
                        logger.info(f"Batch {batch_num}: No trade plans generated")
                        continue

                    # IMMEDIATE RISK VALIDATION FOR THIS BATCH
                    logger.info(f"\nBatch {batch_num}: Risk validation ({len(batch_trade_plans)} plans)...")
                    batch_approved_plans = []

                    for plan in batch_trade_plans:
                        # OPTIMIZATION: Use cached portfolio state for risk validation
                        # Cache valid for 5 seconds - reduces API calls during batch validation
                        current_portfolio = self.get_current_portfolio_state(use_cache=True)
                        decision = self.risk_manager.evaluate(plan, current_portfolio, current_prices)

                        # Log decision
                        trade_id = str(uuid.uuid4())
                        self.json_logger.log_decision(
                            trade_id=trade_id,
                            timestamp=datetime.now().isoformat(),
                            symbol=plan.symbol,
                            analyst_outputs=analyst_outputs[plan.symbol],
                            researcher_outputs=None,  # Removed - trader decides from raw data only
                            trade_plan=plan.model_dump(),
                            risk_decision=decision.model_dump()
                        )

                        if decision.approved:
                            # Apply modifications if any
                            if decision.modifications:
                                for key, value in decision.modifications.items():
                                    setattr(plan, key, value)

                            batch_approved_plans.append((trade_id, plan))
                            total_plans_approved += 1
                            logger.info(f"APPROVED: {plan.symbol} {plan.direction} (trade_id: {trade_id})")
                        else:
                            logger.warning(f"REJECTED: {plan.symbol} - {decision.reason}")

                    logger.info(f"Batch {batch_num}: Approved {len(batch_approved_plans)}/{len(batch_trade_plans)} plans")

                    # IMMEDIATE EXECUTION FOR THIS BATCH
                    if not batch_approved_plans:
                        logger.info(f"Batch {batch_num}: No plans approved, skipping execution")
                        continue

                    if self.config.mode in ["paper-live", "live"] and self.broker_executor:
                        logger.info(f"Batch {batch_num}: EXECUTING {len(batch_approved_plans)} trades via {self.config.execution.broker}...")

                        for trade_id, plan in batch_approved_plans:
                            try:
                                # IMPORTANT: Refresh portfolio state before EACH trade
                                current_portfolio = self.get_current_portfolio_state(use_cache=True)
                                current_cash = current_portfolio.cash
                                current_equity = current_portfolio.equity

                                # Check if we have enough cash for this position
                                position_value = plan.position_size_pct * current_equity
                                if position_value > current_cash:
                                    logger.warning(
                                        f"Skipping {plan.symbol}: Insufficient cash "
                                        f"(need ${position_value:,.2f}, have ${current_portfolio.cash:,.2f})"
                                    )
                                    continue  # Skip this trade

                                logger.info(
                                    f"Pre-trade check: ${current_cash:,.2f} cash available, "
                                    f"${position_value:,.2f} needed for {plan.symbol}"
                                )

                                # Submit order to broker
                                result = self.broker_executor.submit_trade(
                                    plan=plan,
                                    order_type=self.config.execution.order_type,
                                    time_in_force=self.config.execution.time_in_force
                                )

                                # Log execution result
                                if result.success:
                                    total_plans_executed += 1
                                    logger.info(f"✓ ORDER SUBMITTED: {plan.symbol} - Order ID: {result.order_id}")

                                    # For market orders, filled price may be available immediately
                                    filled_price = result.filled_price if result.filled_price else plan.entry
                                    filled_qty = result.filled_qty if result.filled_qty else 0

                                    self.json_logger.log_execution(
                                        trade_id=trade_id,
                                        timestamp=result.timestamp,
                                        symbol=plan.symbol,
                                        direction=plan.direction,
                                        filled_price=filled_price,
                                        filled_qty=filled_qty,
                                        slippage=filled_price - plan.entry if result.filled_price else 0.0,
                                        execution_method=f"{self.config.execution.broker}_{'paper' if self.config.execution.paper_trading else 'live'}",
                                        broker_order_id=result.order_id,
                                        order_status=result.status
                                    )

                                    # Record position in database (if filled)
                                    if filled_qty > 0:
                                        opened_position = self.position_manager.open_position(
                                            symbol=plan.symbol,
                                            direction=plan.direction,
                                            entry_price=filled_price,
                                            stop_loss=plan.stop_loss,
                                            take_profit=plan.take_profit,
                                            shares=int(filled_qty),
                                            strategy=None,  # Strategy removed - LLM decides dynamically
                                            position_size_pct=plan.position_size_pct * 100,  # Convert to percentage
                                            fill_timestamp=datetime.now()
                                        )

                                        logger.info(
                                            f"Position OPENED: {opened_position.symbol} {opened_position.direction} "
                                            f"@ {opened_position.entry_price:.2f}, {opened_position.shares} shares, "
                                            f"Stop: {opened_position.stop_loss:.2f}, Target: {opened_position.take_profit:.2f}"
                                        )

                                else:
                                    logger.error(f"Order failed for {plan.symbol}: {result.error_message}")
                                    self.json_logger.log_error(
                                        timestamp=result.timestamp,
                                        component="execution",
                                        error_type="OrderSubmissionError",
                                        error_message=result.error_message,
                                        context={"symbol": plan.symbol, "trade_id": trade_id}
                                    )

                            except Exception as e:
                                logger.error(f"Error executing trade for {plan.symbol}: {e}", exc_info=True)
                                self.json_logger.log_error(
                                    timestamp=datetime.now().isoformat(),
                                    component="execution",
                                    error_type=type(e).__name__,
                                    error_message=str(e),
                                    context={"symbol": plan.symbol, "trade_id": trade_id}
                                )

                        logger.info(f"Batch {batch_num}: Executed {total_plans_executed} trades")

                # Summary after all batches complete
                logger.info(f"\n{'='*80}")
                logger.info(f"BATCH PROCESSING COMPLETE")
                logger.info(f"{'='*80}")
                logger.info(f"Total plans generated: {total_plans_generated}")
                logger.info(f"Total plans approved: {total_plans_approved}")
                logger.info(f"Total trades executed: {total_plans_executed}")

            # Step 6: Check for end-of-day position closing
            if hasattr(self.config, 'auto_close_eod') and self.config.auto_close_eod:
                # Check if we're near market close (e.g., 3:55 PM)
                current_time = datetime.now().time()
                eod_close_time = time(15, 55)  # 3:55 PM

                if current_time >= eod_close_time:
                    logger.info("\nClosing all positions for end-of-day")
                    closed = self.exit_monitor.close_all_positions(current_prices, reason='eod_close')
                    for pos in closed:
                        logger.info(f"EOD CLOSE: {pos.symbol} @ {pos.exit_price:.2f}, P&L: ${pos.realized_pnl:.2f}")

            logger.info("Decision cycle completed successfully")

        except Exception as e:
            logger.error(f"Error in decision cycle: {e}", exc_info=True)
            self.json_logger.log_error(
                timestamp=datetime.now().isoformat(),
                component="decision_cycle",
                error_type=type(e).__name__,
                error_message=str(e),
                context={}
            )

    def run_learning_update(self):
        """
        Execute end-of-day learning update.

        - Display portfolio performance summary
        - Log recent trade statistics

        REMOVED: Strategy switching - LLM decides dynamically
        """
        logger.info("\n" + "=" * 80)
        logger.info("LEARNING UPDATE (Performance Review)")
        logger.info("=" * 80)

        try:
            # Display recent performance summary
            recent_trades = self.memory.get_recent_trades(limit=50)

            if recent_trades:
                wins = sum(1 for t in recent_trades if t.pnl and t.pnl > 0)
                total_pnl = sum(t.pnl for t in recent_trades if t.pnl)

                logger.info(f"\nRECENT PERFORMANCE (Last {len(recent_trades)} trades):")
                logger.info(f"  Win Rate: {wins}/{len(recent_trades)} ({100*wins/len(recent_trades):.1f}%)")
                logger.info(f"  Total P&L: ${total_pnl:,.2f}")
                logger.info(f"  Avg per trade: ${total_pnl/len(recent_trades):,.2f}")
            else:
                logger.info("\nNo recent trades to analyze")

        except Exception as e:
            logger.error(f"Error in learning update: {e}", exc_info=True)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="ActiveTrader-LLM: Multi-agent trading system")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--cycles',
        type=int,
        default=1,
        help='Number of decision cycles to run'
    )
    parser.add_argument(
        '--learning',
        action='store_true',
        help='Run learning update after cycles'
    )

    args = parser.parse_args()

    # Initialize system
    system = ActiveTraderLLM(args.config)

    # Run decision cycles
    for i in range(args.cycles):
        logger.info(f"\n{'='*80}")
        logger.info(f"CYCLE {i+1}/{args.cycles}")
        logger.info(f"{'='*80}\n")

        system.run_decision_cycle()

    # Run learning update if requested
    if args.learning:
        system.run_learning_update()

    logger.info("\n" + "="*80)
    logger.info("ActiveTrader-LLM session completed")
    logger.info("="*80)


if __name__ == "__main__":
    main()
