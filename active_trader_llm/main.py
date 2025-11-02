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
"""

import logging
import argparse
from pathlib import Path
from datetime import datetime, time
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from active_trader_llm.config.loader import load_config
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
from active_trader_llm.execution.simulated_broker import SimulatedBroker
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

    def __init__(self, config_path: str, mode_override: str = None):
        """
        Initialize the trading system.

        Args:
            config_path: Path to YAML configuration file
            mode_override: Optional mode override (e.g., "backtest" for backtesting)
        """
        logger.info(f"Initializing ActiveTrader-LLM from {config_path}")

        # Load configuration
        self.config = load_config(config_path)

        # Apply mode override if provided
        if mode_override:
            self.config.mode = mode_override
            logger.info(f"Mode overridden to: {mode_override}")

        logger.info(f"Mode: {self.config.mode}")

        # Initialize components
        self.ingestor = PriceVolumeIngestor(cache_db_path="data/price_cache.db")
        self.macro_ingestor = MacroDataIngestor(
            cache_duration_seconds=self.config.macro_data.cache_duration_seconds
        ) if self.config.enable_macro else None
        self.feature_builder = FeatureBuilder(self.config.model_dump())

        # Initialize analysts
        api_key = self.config.llm.api_key
        model = self.config.llm.model

        # technical_analyst removed - validation done directly, no LLM interpretation needed
        self.sentiment_analyst = None  # Sentiment analyst not implemented
        # macro_analyst removed - using raw macro data instead of LLM interpretation

        # researcher_debate removed - trader_agent makes decisions from raw data only

        # Initialize trader with validation
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
        self.simulated_broker = None

        if self.config.mode == "backtest":
            # Use SimulatedBroker for backtesting
            self.simulated_broker = SimulatedBroker(
                initial_cash=100000.0,  # Will be updated by BacktestEngine
                commission_per_trade=self.config.execution.commission_per_trade,
                slippage_bps=self.config.execution.slippage_bps
            )
            logger.info("SimulatedBroker initialized for backtest mode")
        elif self.config.mode in ["paper-live", "live"]:
            if self.config.execution.broker == "alpaca":
                try:
                    api_key = os.getenv(self.config.execution.alpaca_api_key_env)
                    secret_key = os.getenv(self.config.execution.alpaca_secret_key_env)

                    self.broker_executor = AlpacaBrokerExecutor(
                        api_key=api_key,
                        secret_key=secret_key,
                        paper=self.config.execution.paper_trading,
                        base_url=self.config.execution.alpaca_base_url
                    )
                    logger.info(f"Alpaca broker executor initialized (paper={self.config.execution.paper_trading})")
                except Exception as e:
                    logger.error(f"Failed to initialize Alpaca executor: {e}")
                    logger.warning("Falling back to simulated execution")
                    self.broker_executor = None
            else:
                logger.info("Using simulated execution (no broker)")

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
                    anthropic_api_key=api_key,
                    requests_per_minute=200  # Standard plan
                )
                mode = "PAPER" if self.config.execution.paper_trading else "LIVE"
                logger.info(f"Market scanner initialized (two-stage mode enabled, {mode} mode)")
            except Exception as e:
                logger.warning(f"Failed to initialize scanner: {e}")
                logger.warning("Falling back to fixed universe mode")
                self.config.scanner.enabled = False

        # REMOVED: Current strategy tracking - LLM decides dynamically
        # self.current_strategy = self.config.strategies[0].name

        # Initialize portfolio state (mode-dependent)
        if self.config.mode == "backtest":
            # Backtest: use tracked state (updated by SimulatedBroker)
            self.portfolio_state = PortfolioState(
                cash=100000.0,  # TODO: Make configurable via config
                equity=100000.0,
                positions=[],
                daily_pnl=0.0
            )
            logger.info(f"Portfolio initialized: Backtest mode with $100,000 initial capital")

        elif self.config.mode in ["paper-live", "live"] and self.broker_executor:
            # Paper-live/live: fetch from Alpaca immediately
            try:
                account_info = self.broker_executor.get_account_info()
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

    def get_current_portfolio_state(self) -> PortfolioState:
        """
        Get current portfolio state (mode-aware).

        Returns:
            PortfolioState with fresh values:
            - Backtest: Returns internally tracked state
            - Paper-live/live: Fetches fresh from Alpaca broker

        This ensures position sizing and risk checks always use
        accurate account balances in live trading.
        """
        if self.config.mode == "backtest":
            # Use tracked state for backtest
            logger.debug("Portfolio state source: Backtest tracking")
            return self.portfolio_state

        elif self.config.mode in ["paper-live", "live"] and self.broker_executor:
            # Fetch fresh from Alpaca
            try:
                account_info = self.broker_executor.get_account_info()

                # Get positions from current portfolio_state (already synced from position_manager)
                positions_list = self.portfolio_state.positions

                fresh_state = PortfolioState(
                    cash=float(account_info['cash']),
                    equity=float(account_info['equity']),
                    positions=positions_list,
                    daily_pnl=0.0  # Could calculate from last_equity
                )

                logger.debug(
                    f"Portfolio state source: Alpaca API "
                    f"(equity=${fresh_state.equity:,.2f}, cash=${fresh_state.cash:,.2f})"
                )
                return fresh_state

            except Exception as e:
                logger.warning(f"Failed to fetch fresh portfolio state from Alpaca: {e}")
                logger.warning("Using last known state as fallback")
                return self.portfolio_state

        else:
            # Fallback to tracked state
            logger.debug("Portfolio state source: Tracked state (fallback)")
            return self.portfolio_state

    def run_decision_cycle(self):
        """
        Execute one complete decision cycle.

        Workflow:
        0. Load positions and check exits
        1. Fetch latest data
        2. Compute features
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
            # Step 0a: Run scanner if enabled (determines universe)
            if self.config.scanner.enabled and self.scanner:
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

                    # Use scanned universe for this cycle
                    active_universe = self.scanned_universe
                except Exception as e:
                    logger.error(f"Scanner failed: {e}")
                    logger.warning("Falling back to fixed universe")
                    active_universe = self.config.data_sources.universe
            else:
                # Use fixed universe from config
                active_universe = self.config.data_sources.universe
                logger.info(f"Using fixed universe: {active_universe}")

            # Step 1: Fetch data
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

            # In paper-live/live mode, refresh cash/equity from Alpaca
            if self.config.mode in ["paper-live", "live"] and self.broker_executor:
                try:
                    account_info = self.broker_executor.get_account_info()
                    self.portfolio_state.cash = float(account_info['cash'])
                    self.portfolio_state.equity = float(account_info['equity'])
                    logger.debug(f"Refreshed portfolio state from Alpaca: equity=${self.portfolio_state.equity:,.2f}, cash=${self.portfolio_state.cash:,.2f}")
                except Exception as e:
                    logger.warning(f"Could not refresh account info from Alpaca: {e}")
            # Note: In backtest mode, cash and equity are updated by SimulatedBroker

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

                # Validate critical data (price, ATR) - NO LLM call needed
                price = features.close
                atr = features.daily_indicators.get('ATR_14_daily')

                if price is None or price == 0.0:
                    logger.warning(f"{symbol}: Price data missing or zero - skipping symbol")
                    continue
                if atr is None:
                    logger.warning(f"{symbol}: ATR data missing - skipping symbol")
                    continue

                # Extract price series from price_df for trader_agent (last 20 bars)
                symbol_prices = price_df[price_df['symbol'] == symbol].tail(20)
                price_series = symbol_prices['close'].tolist() if not symbol_prices.empty else [features.close]

                analyst_outputs[symbol] = {
                    'features': features.model_dump(),  # Raw features for trader_agent
                    'price_series': price_series,  # Historical prices for LLM
                    'breadth': market_snapshot.model_dump()  # Add breadth to each symbol's context
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

            for symbol in analyst_outputs.keys():

                # Check if we already have a position in this symbol
                existing_position = None
                if self.position_manager.has_open_position(symbol):
                    # Get position details for LLM to decide hold/close
                    open_positions = [p for p in portfolio_dict['open_positions'] if p['symbol'] == symbol]
                    if open_positions:
                        existing_position = open_positions[0]

                # Call trader agent with full context (nof1.ai style - raw data only)
                plan = self.trader_agent.decide(
                    symbol,
                    analyst_outputs[symbol],
                    (None, None),  # researcher_outputs removed - not used in prompt
                    self.config.risk_parameters.model_dump(),
                    memory_context=None,  # TODO: Add recent performance context
                    existing_position=existing_position,
                    account_state=account_state
                )

                if plan:
                    # Handle different action types
                    if plan.action == 'close':
                        # LLM decided to close existing position
                        logger.info(f"{symbol}: LLM decided to CLOSE position (invalidation triggered)")
                        if self.position_manager.has_open_position(symbol):
                            closed_pos = self.position_manager.close_position(
                                symbol=symbol,
                                exit_price=current_prices.get(symbol, 0.0),
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
                        # LLM decided to open new position
                        trade_plans.append(plan)
                        logger.info(
                            f"Trade plan for {symbol}: {plan.action} @ {plan.entry:.2f} "
                            f"(size: {plan.position_size_pct*100:.1f}%, confidence: {plan.confidence:.2f})"
                        )

            if not trade_plans:
                logger.info("No trade plans generated this cycle.")
                return

            # Step 6: Risk validation
            logger.info("Step 6: Risk validation...")
            approved_plans = []

            for plan in trade_plans:
                # Fetch fresh portfolio state (from Alpaca in live mode)
                current_portfolio = self.get_current_portfolio_state()
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

                    approved_plans.append((trade_id, plan))
                    logger.info(f"APPROVED: {plan.symbol} {plan.direction} (trade_id: {trade_id})")
                else:
                    logger.warning(f"REJECTED: {plan.symbol} - {decision.reason}")

            logger.info(f"\nApproved: {len(approved_plans)}/{len(trade_plans)} trade plans")

            # Step 7: Execution
            if self.config.mode == "backtest" and self.simulated_broker:
                logger.info("Step 7: Simulated execution via SimulatedBroker...")

                for trade_id, plan in approved_plans:
                    # Submit trade to simulated broker
                    result = self.simulated_broker.submit_trade(plan)

                    if result.success:
                        # Log execution
                        self.json_logger.log_execution(
                            trade_id=trade_id,
                            timestamp=result.timestamp,
                            symbol=plan.symbol,
                            direction=plan.direction,
                            filled_price=result.filled_price,
                            filled_qty=result.filled_qty,
                            slippage=result.filled_price - plan.entry,
                            execution_method="simulated"
                        )

                        logger.info(f"Simulated execution: {plan.symbol} @ {result.filled_price:.2f}")

                        # Open position in position manager
                        opened_position = self.position_manager.open_position(
                            symbol=plan.symbol,
                            direction=plan.direction,
                            entry_price=result.filled_price,
                            stop_loss=plan.stop_loss,
                            take_profit=plan.take_profit,
                            shares=int(result.filled_qty),
                            strategy=None,  # Strategy removed - LLM decides dynamically
                            position_size_pct=plan.position_size_pct * 100,  # Convert to percentage
                            fill_timestamp=datetime.now()
                        )

                        logger.info(
                            f"Position OPENED: {opened_position.symbol} {opened_position.direction} "
                            f"@ {opened_position.entry_price:.2f}, {opened_position.shares} shares, "
                            f"Stop: {opened_position.stop_loss:.2f}, Target: {opened_position.take_profit:.2f}"
                        )

                        # Immediately update portfolio state for intra-cycle accuracy
                        # This ensures subsequent trades in the same cycle see updated cash
                        position_cost = result.filled_price * result.filled_qty
                        commission = self.simulated_broker.commission_per_trade if self.simulated_broker else 1.0
                        total_cost = position_cost + commission

                        # Deduct from portfolio_state
                        self.portfolio_state.cash -= total_cost

                        # Update equity (cash + all position values)
                        open_positions = self.position_manager.get_open_positions()
                        positions_value = sum([p.entry_price * p.shares for p in open_positions])
                        self.portfolio_state.equity = self.portfolio_state.cash + positions_value

                        logger.info(
                            f"Portfolio state updated: cash=${self.portfolio_state.cash:.2f}, "
                            f"equity=${self.portfolio_state.equity:.2f} (deducted ${total_cost:.2f})"
                        )

                    else:
                        logger.error(f"Simulated order failed for {plan.symbol}: {result.error_message}")

            elif self.config.mode in ["paper-live", "live"] and self.broker_executor:
                logger.info(f"Step 7: Live execution via {self.config.execution.broker}...")

                for trade_id, plan in approved_plans:
                    try:
                        # Submit order to broker
                        result = self.broker_executor.submit_trade(
                            plan=plan,
                            order_type=self.config.execution.order_type,
                            time_in_force=self.config.execution.time_in_force
                        )

                        # Log execution result
                        if result.success:
                            logger.info(f"Order submitted: {plan.symbol} - Order ID: {result.order_id}")

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

            else:
                logger.warning("No execution performed (broker not configured or mode is not paper-live/live)")

            # Step 8: Check for end-of-day position closing
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
