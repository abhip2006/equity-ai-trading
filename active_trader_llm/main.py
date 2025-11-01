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
from datetime import datetime
import uuid

from config.loader import load_config
from data_ingestion.price_volume_ingestor import PriceVolumeIngestor
from feature_engineering.indicators import FeatureEngineer
from analysts.technical_analyst import TechnicalAnalyst
from analysts.breadth_health_analyst import BreadthHealthAnalyst
from analysts.sentiment_analyst import SentimentAnalyst
from analysts.macro_analyst import MacroAnalyst
from researchers.bull_bear import ResearcherDebate
from trader.trader_agent import TraderAgent
from risk.risk_manager import RiskManager, PortfolioState
from memory.memory_manager import MemoryManager, TradeMemory
from learning.strategy_monitor import StrategyMonitor
from utils.logging_json import JSONLogger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ActiveTraderLLM:
    """
    Main orchestrator for the multi-agent trading system.
    """

    def __init__(self, config_path: str):
        """
        Initialize the trading system.

        Args:
            config_path: Path to YAML configuration file
        """
        logger.info(f"Initializing ActiveTrader-LLM from {config_path}")

        # Load configuration
        self.config = load_config(config_path)
        logger.info(f"Mode: {self.config.mode}")

        # Initialize components
        self.ingestor = PriceVolumeIngestor(cache_db_path="data/price_cache.db")
        self.feature_engineer = FeatureEngineer()

        # Initialize analysts
        api_key = self.config.llm.api_key
        model = self.config.llm.model

        self.technical_analyst = TechnicalAnalyst(api_key=api_key, model=model)
        self.breadth_analyst = BreadthHealthAnalyst(api_key=api_key, model=model)
        self.sentiment_analyst = SentimentAnalyst() if self.config.enable_sentiment else None
        self.macro_analyst = MacroAnalyst() if self.config.enable_macro else None

        # Initialize researchers
        self.researcher_debate = ResearcherDebate(api_key=api_key, model=model)

        # Initialize trader
        self.trader_agent = TraderAgent(api_key=api_key, model=model)

        # Initialize risk manager
        self.risk_manager = RiskManager(self.config.risk_parameters.dict())

        # Initialize memory and learning
        self.memory = MemoryManager(self.config.database_path)
        self.strategy_monitor = StrategyMonitor(
            self.memory,
            self.config.strategy_switching.dict()
        )

        # Initialize logging
        self.json_logger = JSONLogger(self.config.log_path)

        # Current state
        self.current_strategy = self.config.strategies[0].name
        self.portfolio_state = PortfolioState(
            cash=100000.0,  # Initial capital
            equity=100000.0,
            positions=[],
            daily_pnl=0.0
        )

        logger.info("ActiveTrader-LLM initialized successfully")

    def run_decision_cycle(self):
        """
        Execute one complete decision cycle.

        Workflow:
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
        logger.info(f"DECISION CYCLE: {datetime.now().isoformat()}")
        logger.info("=" * 80)

        try:
            # Step 1: Fetch data
            logger.info("Step 1: Fetching market data...")
            price_df = self.ingestor.fetch_prices(
                universe=self.config.data_sources.universe,
                interval=self.config.data_sources.interval,
                lookback_days=self.config.data_sources.lookback_days,
                use_cache=self.config.cost_control.cache_data
            )

            if price_df.empty:
                logger.error("No price data fetched. Aborting cycle.")
                return

            # Step 2: Compute features
            logger.info("Step 2: Computing technical features...")
            features_dict = self.feature_engineer.build_features(price_df)

            if not features_dict:
                logger.error("No features computed. Aborting cycle.")
                return

            market_snapshot = self.feature_engineer.compute_breadth_features(price_df, features_dict)
            logger.info(f"Market regime: {market_snapshot.regime_hint} (breadth: {market_snapshot.breadth_score:.2f})")

            # Step 3: Run analysts
            logger.info("Step 3: Running analyst agents...")
            analyst_outputs = {}

            for symbol, features in features_dict.items():
                logger.info(f"\nAnalyzing {symbol}...")

                # Technical analyst
                tech_signal = self.technical_analyst.analyze(
                    symbol,
                    features.dict(),
                    market_snapshot.dict(),
                    memory_context=self.memory.get_context_summary(symbol)
                )
                analyst_outputs[symbol] = {
                    'technical': tech_signal.dict()
                }

                # Add optional analysts
                if self.sentiment_analyst:
                    sent_signal = self.sentiment_analyst.analyze(symbol)
                    analyst_outputs[symbol]['sentiment'] = sent_signal.dict()

            # Breadth analyst (market-wide)
            breadth_signal = self.breadth_analyst.analyze(market_snapshot.dict())
            logger.info(f"Breadth analysis: {breadth_signal.regime} (confidence: {breadth_signal.confidence:.2f})")

            # Macro analyst (if enabled)
            if self.macro_analyst:
                macro_signal = self.macro_analyst.analyze({'vix': market_snapshot.vix})

            # Step 4: Researcher debate
            logger.info("Step 4: Running researcher debate...")
            debates = {}

            for symbol in features_dict.keys():
                bull, bear = self.researcher_debate.debate(
                    symbol,
                    analyst_outputs[symbol],
                    market_snapshot.dict()
                )

                debates[symbol] = {
                    'bull': bull.dict(),
                    'bear': bear.dict()
                }

                logger.info(f"{symbol} debate: Bull {bull.confidence:.2f} vs Bear {bear.confidence:.2f}")

            # Step 5: Generate trade plans
            logger.info("Step 5: Generating trade plans...")
            trade_plans = []

            for symbol in features_dict.keys():
                bull, bear = debates[symbol]['bull'], debates[symbol]['bear']

                from researchers.bull_bear import BullThesis, BearThesis
                bull_thesis = BullThesis(**bull)
                bear_thesis = BearThesis(**bear)

                plan = self.trader_agent.decide(
                    symbol,
                    analyst_outputs[symbol],
                    (bull_thesis, bear_thesis),
                    [s.dict() for s in self.config.strategies],
                    self.config.risk_parameters.dict()
                )

                if plan:
                    trade_plans.append(plan)
                    logger.info(f"Trade plan generated for {symbol}: {plan.direction} {plan.strategy}")

            if not trade_plans:
                logger.info("No trade plans generated this cycle.")
                return

            # Step 6: Risk validation
            logger.info("Step 6: Risk validation...")
            approved_plans = []

            for plan in trade_plans:
                decision = self.risk_manager.evaluate(plan, self.portfolio_state)

                # Log decision
                trade_id = str(uuid.uuid4())
                self.json_logger.log_decision(
                    trade_id=trade_id,
                    timestamp=datetime.now().isoformat(),
                    symbol=plan.symbol,
                    analyst_outputs=analyst_outputs[plan.symbol],
                    researcher_outputs=debates[plan.symbol],
                    trade_plan=plan.dict(),
                    risk_decision=decision.dict()
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

            # Step 7: Execution (paper trading simulation)
            if self.config.mode == "backtest":
                logger.info("Step 7: Simulated execution (backtest mode)...")
                for trade_id, plan in approved_plans:
                    # Simulate execution with small slippage
                    slippage = 0.02  # $0.02 slippage
                    filled_price = plan.entry + slippage

                    self.json_logger.log_execution(
                        trade_id=trade_id,
                        timestamp=datetime.now().isoformat(),
                        symbol=plan.symbol,
                        direction=plan.direction,
                        filled_price=filled_price,
                        filled_qty=100,  # Fixed for simulation
                        slippage=slippage,
                        execution_method="backtest"
                    )

                    logger.info(f"Simulated execution: {plan.symbol} @ {filled_price:.2f}")

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

        - Update strategy statistics
        - Check for strategy degradation
        - Switch strategies if needed
        """
        logger.info("\n" + "=" * 80)
        logger.info("LEARNING UPDATE")
        logger.info("=" * 80)

        try:
            # Get current regime
            # In production, this would use latest market snapshot
            current_regime = "trending_bull"  # Placeholder

            # Check if current strategy needs switching
            available_strategies = [s.name for s in self.config.strategies if s.enabled]

            recommendation = self.strategy_monitor.monitor_and_switch(
                self.current_strategy,
                available_strategies,
                current_regime
            )

            if recommendation:
                logger.info(f"\nSTRATEGY SWITCH RECOMMENDED:")
                logger.info(f"  From: {recommendation.current_strategy}")
                logger.info(f"  To: {recommendation.recommended_strategy}")
                logger.info(f"  Reason: {recommendation.reason}")
                logger.info(f"  Confidence: {recommendation.confidence:.2f}")

                # Log the switch
                self.json_logger.log_strategy_switch(
                    timestamp=datetime.now().isoformat(),
                    old_strategy=recommendation.current_strategy,
                    new_strategy=recommendation.recommended_strategy,
                    regime=current_regime,
                    reason=recommendation.reason,
                    metrics=recommendation.metrics
                )

                # Apply the switch
                self.current_strategy = recommendation.recommended_strategy
                logger.info(f"Switched to {self.current_strategy}")

            else:
                logger.info(f"Current strategy ({self.current_strategy}) performing adequately")

            # Display performance summary
            stats = self.memory.get_strategy_stats()
            logger.info("\nSTRATEGY PERFORMANCE SUMMARY:")
            for s in stats:
                logger.info(f"{s.strategy_name} ({s.regime}): "
                           f"WR={s.win_rate:.1%}, Net=${s.total_pnl:.2f}, "
                           f"Trades={s.total_trades}")

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
