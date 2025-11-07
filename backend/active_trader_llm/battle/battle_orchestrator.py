"""
Battle Orchestrator: Main coordinator for multi-model competition.

Orchestrates the entire battle by:
- Spawning all model instances
- Managing shared data feed
- Coordinating synchronized cycles
- Monitoring model health
- Providing graceful shutdown
- Handling model failures
"""

import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from active_trader_llm.battle.model_instance import ModelInstance
from active_trader_llm.battle.shared_data_feed import SharedDataFeed
from active_trader_llm.battle.cycle_coordinator import CycleCoordinator
from active_trader_llm.data_ingestion.price_volume_ingestor import PriceVolumeIngestor
from active_trader_llm.feature_engineering.feature_builder import FeatureBuilder

logger = logging.getLogger(__name__)


class BattleOrchestrator:
    """
    Main orchestrator for multi-model trading competition.

    Coordinates 6 model instances running in parallel:
    - Spawns isolated model instances
    - Manages centralized data feed
    - Synchronizes trading cycles
    - Monitors health and performance
    - Handles errors and recovery
    """

    def __init__(
        self,
        model_configs: List[Dict[str, Any]],
        base_config_path: str,
        universe: List[str],
        interval: str = "1d",
        lookback_days: int = 90,
        database_dir: str = "data/battle"
    ):
        """
        Initialize battle orchestrator.

        Args:
            model_configs: List of LLM configurations for each model
                Each dict should contain: model_id, provider, model, api_key, temperature
            base_config_path: Path to base config.yaml template
            universe: List of symbols to trade
            interval: Price data interval (1d, 1h, etc.)
            lookback_days: Historical lookback period
            database_dir: Directory for battle databases
        """
        self.model_configs = model_configs
        self.base_config_path = base_config_path
        self.universe = universe
        self.interval = interval
        self.lookback_days = lookback_days
        self.database_dir = Path(database_dir)

        # Components (initialized later)
        self.data_feed: Optional[SharedDataFeed] = None
        self.model_instances: List[ModelInstance] = []
        self.cycle_coordinator: Optional[CycleCoordinator] = None

        # State tracking
        self.is_running = False
        self.cycle_count = 0
        self.start_time: Optional[datetime] = None

        logger.info(f"BattleOrchestrator created with {len(model_configs)} models")
        logger.info(f"  Universe: {len(universe)} symbols")
        logger.info(f"  Database dir: {database_dir}")

    def initialize_models(self) -> bool:
        """
        Initialize all model instances and shared components.

        Returns:
            True if all models initialized successfully, False otherwise
        """
        try:
            logger.info("\n" + "="*80)
            logger.info("INITIALIZING BATTLE ORCHESTRATOR")
            logger.info("="*80)

            # Create database directory
            self.database_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Initialize data feed
            logger.info("\nStep 1: Initializing shared data feed...")
            data_fetcher = PriceVolumeIngestor(cache_db_path="data/price_cache.db")

            # Create minimal config for FeatureBuilder with technical indicators and market breadth
            feature_config = {
                'technical_indicators': {
                    'quality': {
                        'min_bars_required': 20,
                        'reject_incomplete': False
                    },
                    'daily': {
                        'ema_5': {'enabled': True, 'period': 5},
                        'ema_10': {'enabled': True, 'period': 10},
                        'ema_20': {'enabled': True, 'period': 20},
                        'sma_50': {'enabled': True, 'period': 50},
                        'sma_200': {'enabled': True, 'period': 200},
                        'rsi': {'enabled': True, 'period': 14},
                        'atr': {'enabled': True, 'period': 14}
                    },
                    'weekly': {
                        'ema_10': {'enabled': True, 'period': 10, 'conversion': 50},
                        'sma_21': {'enabled': True, 'period': 21, 'conversion': 105},
                        'sma_30': {'enabled': True, 'period': 30, 'conversion': 150}
                    }
                },
                'market_breadth': {
                    'enabled': True,
                    'new_highs_lows': {
                        'lookback_days': 252,
                        'min_data_bars': 20,
                        'high_threshold_pct': 0.98,
                        'low_threshold_pct': 1.02
                    }
                }
            }
            feature_builder = FeatureBuilder(feature_config)

            self.data_feed = SharedDataFeed(
                data_fetcher=data_fetcher,
                feature_builder=feature_builder,
                universe=self.universe,
                interval=self.interval,
                lookback_days=self.lookback_days,
                cache_enabled=True
            )
            logger.info("Shared data feed initialized")

            # Step 2: Create model instances
            logger.info(f"\nStep 2: Creating {len(self.model_configs)} model instances...")
            for config in self.model_configs:
                model_id = config.get('model_id')
                if not model_id:
                    logger.error("Model config missing 'model_id' field")
                    continue

                logger.info(f"\nCreating instance: {model_id}")
                instance = ModelInstance(
                    model_id=model_id,
                    llm_config=config,
                    base_config_path=self.base_config_path,
                    database_dir=str(self.database_dir)
                )

                # Initialize the instance
                if instance.initialize():
                    self.model_instances.append(instance)
                    self.data_feed.subscribe(model_id)
                    logger.info(f"  ✓ {model_id} initialized successfully")
                else:
                    logger.error(f"  ✗ {model_id} initialization failed")

            if not self.model_instances:
                logger.error("No models initialized successfully")
                return False

            logger.info(f"\n{len(self.model_instances)}/{len(self.model_configs)} models initialized")

            # Step 3: Initialize cycle coordinator
            logger.info("\nStep 3: Initializing cycle coordinator...")
            self.cycle_coordinator = CycleCoordinator(
                models=self.model_instances,
                data_feed=self.data_feed,
                max_workers=len(self.model_instances),
                model_timeout_seconds=300,  # 5 minute timeout per model
                straggler_policy="wait"  # Wait for slow models
            )
            logger.info("Cycle coordinator initialized")

            logger.info("\n" + "="*80)
            logger.info("INITIALIZATION COMPLETE")
            logger.info("="*80 + "\n")

            return True

        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def start_competition(
        self,
        num_cycles: int = 1,
        cycle_interval_seconds: int = 0,
        stop_on_error: bool = False
    ) -> bool:
        """
        Start the model competition.

        Runs multiple synchronized trading cycles across all models.

        Args:
            num_cycles: Number of cycles to run
            cycle_interval_seconds: Delay between cycles (0 = run continuously)
            stop_on_error: If True, stop on first error; if False, continue

        Returns:
            True if competition completed successfully, False otherwise
        """
        if not self.model_instances or not self.cycle_coordinator:
            logger.error("Cannot start competition - not initialized")
            return False

        try:
            self.is_running = True
            self.start_time = datetime.now()

            logger.info("\n" + "="*80)
            logger.info(f"STARTING COMPETITION: {num_cycles} cycles")
            logger.info("="*80)

            for cycle in range(1, num_cycles + 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"CYCLE {cycle}/{num_cycles}")
                logger.info(f"{'='*80}")

                # Check model health before each cycle
                unhealthy_models = [m for m in self.model_instances if not m.is_healthy()]
                if unhealthy_models:
                    logger.warning(f"Unhealthy models detected: {[m.model_id for m in unhealthy_models]}")

                    if stop_on_error:
                        logger.error("Stopping competition due to unhealthy models")
                        return False
                    else:
                        logger.warning("Continuing despite unhealthy models")

                # Trigger synchronized cycle
                cycle_result = self.cycle_coordinator.trigger_cycle()

                # Check cycle results
                if cycle_result.get_failure_count() > 0:
                    logger.warning(
                        f"Cycle {cycle} had {cycle_result.get_failure_count()} failures"
                    )

                    if stop_on_error and cycle_result.get_failure_count() == len(self.model_instances):
                        logger.error("All models failed - stopping competition")
                        return False

                self.cycle_count += 1

                # Wait before next cycle (if interval specified)
                if cycle < num_cycles and cycle_interval_seconds > 0:
                    logger.info(f"Waiting {cycle_interval_seconds}s before next cycle...")
                    time.sleep(cycle_interval_seconds)

            # Competition complete
            duration = (datetime.now() - self.start_time).total_seconds()

            logger.info("\n" + "="*80)
            logger.info("COMPETITION COMPLETE")
            logger.info("="*80)
            logger.info(f"Total cycles: {self.cycle_count}")
            logger.info(f"Duration: {duration:.2f}s")
            logger.info(f"Avg cycle time: {duration/self.cycle_count:.2f}s")

            # Print final model states
            self._print_model_summary()

            return True

        except KeyboardInterrupt:
            logger.warning("\nCompetition interrupted by user")
            return False

        except Exception as e:
            logger.error(f"Error during competition: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

        finally:
            self.is_running = False

    def broadcast_data(self, force_refresh: bool = False):
        """
        Broadcast market data to all models.

        This is called internally by cycle_coordinator.
        Exposed here for manual testing/debugging.

        Args:
            force_refresh: Force fresh data fetch
        """
        if not self.data_feed:
            logger.error("Data feed not initialized")
            return

        snapshot = self.data_feed.fetch_market_data(force_refresh=force_refresh)
        if snapshot:
            logger.info(f"Data broadcast: {len(snapshot.symbols)} symbols at {snapshot.timestamp}")
        else:
            logger.error("Failed to fetch data for broadcast")

    def collect_decisions(self) -> Dict[str, Any]:
        """
        Collect current state from all models.

        Returns:
            Dict mapping model_id to model state
        """
        states = {}

        for model in self.model_instances:
            states[model.model_id] = model.get_current_state()

        return states

    def shutdown(self) -> bool:
        """
        Gracefully shutdown the battle orchestrator.

        Shuts down all model instances and cleans up resources.

        Returns:
            True if shutdown succeeded, False otherwise
        """
        try:
            logger.info("\n" + "="*80)
            logger.info("SHUTTING DOWN BATTLE ORCHESTRATOR")
            logger.info("="*80)

            self.is_running = False

            # Shutdown all model instances
            for model in self.model_instances:
                logger.info(f"Shutting down {model.model_id}...")
                model.shutdown()

            # Clear instances
            self.model_instances.clear()

            # Print final summary
            if self.cycle_coordinator:
                stats = self.cycle_coordinator.get_cycle_stats()
                logger.info(f"\nFinal Statistics:")
                logger.info(f"  Total cycles: {stats['total_cycles']}")
                logger.info(f"  Avg duration: {stats['avg_duration']:.2f}s")
                logger.info(f"  Avg success rate: {stats['avg_success_rate']*100:.1f}%")

            logger.info("\n" + "="*80)
            logger.info("SHUTDOWN COMPLETE")
            logger.info("="*80 + "\n")

            return True

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _print_model_summary(self):
        """Print summary of all model states"""
        logger.info("\n" + "="*80)
        logger.info("MODEL SUMMARY")
        logger.info("="*80)

        for model in self.model_instances:
            state = model.get_current_state()
            logger.info(f"\n{state['model_id']}:")
            logger.info(f"  Status: {state['status']}")
            logger.info(f"  Cycles: {state['cycles_completed']}")
            logger.info(f"  Errors: {state['error_count']}")

            if state.get('portfolio'):
                portfolio = state['portfolio']
                logger.info(f"  Portfolio:")
                logger.info(f"    Equity: ${portfolio['equity']:,.2f}")
                logger.info(f"    Cash: ${portfolio['cash']:,.2f}")
                logger.info(f"    Positions: {portfolio['position_count']}")
                logger.info(f"    P&L: ${portfolio['daily_pnl']:,.2f}")


# Example usage
if __name__ == "__main__":
    import os
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example: Create 6 model configs
    model_configs = [
        {
            'model_id': 'gpt4o',
            'provider': 'openai',
            'model': 'gpt-4o',
            'api_key': os.getenv('OPENAI_API_KEY'),
            'temperature': 0.3
        },
        {
            'model_id': 'gpt4_turbo',
            'provider': 'openai',
            'model': 'gpt-4-turbo',
            'api_key': os.getenv('OPENAI_API_KEY'),
            'temperature': 0.3
        },
        {
            'model_id': 'gpt35_turbo',
            'provider': 'openai',
            'model': 'gpt-3.5-turbo',
            'api_key': os.getenv('OPENAI_API_KEY'),
            'temperature': 0.3
        },
        {
            'model_id': 'claude_sonnet',
            'provider': 'anthropic',
            'model': 'claude-3-5-sonnet-20241022',
            'api_key': os.getenv('ANTHROPIC_API_KEY'),
            'temperature': 0.3
        },
        {
            'model_id': 'claude_opus',
            'provider': 'anthropic',
            'model': 'claude-3-opus-20240229',
            'api_key': os.getenv('ANTHROPIC_API_KEY'),
            'temperature': 0.3
        },
        {
            'model_id': 'claude_haiku',
            'provider': 'anthropic',
            'model': 'claude-3-haiku-20240307',
            'api_key': os.getenv('ANTHROPIC_API_KEY'),
            'temperature': 0.3
        }
    ]

    # Create orchestrator
    orchestrator = BattleOrchestrator(
        model_configs=model_configs,
        base_config_path="config.yaml",
        universe=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        interval="1d",
        lookback_days=90,
        database_dir="data/battle"
    )

    # Initialize
    if orchestrator.initialize_models():
        # Start competition (1 cycle for testing)
        orchestrator.start_competition(num_cycles=1, cycle_interval_seconds=0)

        # Shutdown
        orchestrator.shutdown()
    else:
        print("Failed to initialize models")
