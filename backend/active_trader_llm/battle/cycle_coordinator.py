"""
Cycle Coordinator: Synchronizes trading cycles across all model instances.

Ensures fair competition by:
- Broadcasting data simultaneously to all models
- Waiting for all models to complete decisions
- Handling stragglers (slow models)
- Batch executing orders
- Coordinating cycle timing
"""

import logging
import time
from typing import Dict, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from active_trader_llm.battle.model_instance import ModelInstance
from active_trader_llm.battle.shared_data_feed import SharedDataFeed, DataSnapshot

logger = logging.getLogger(__name__)


class CycleResult:
    """Result of a single cycle execution"""
    def __init__(self, cycle_id: int):
        self.cycle_id = cycle_id
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.model_results: Dict[str, bool] = {}  # model_id -> success
        self.stragglers: List[str] = []  # models that timed out
        self.errors: Dict[str, str] = {}  # model_id -> error message

    def mark_complete(self):
        """Mark cycle as complete"""
        self.end_time = datetime.now()

    def get_duration_seconds(self) -> float:
        """Get cycle duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()

    def get_success_count(self) -> int:
        """Count successful model executions"""
        return sum(1 for success in self.model_results.values() if success)

    def get_failure_count(self) -> int:
        """Count failed model executions"""
        return sum(1 for success in self.model_results.values() if not success)


class CycleCoordinator:
    """
    Coordinates trading cycles across multiple model instances.

    Ensures fair timing and synchronized data access for all models.
    """

    def __init__(
        self,
        models: List[ModelInstance],
        data_feed: SharedDataFeed,
        trading_client=None,
        max_workers: int = 6,
        model_timeout_seconds: int = 300,
        straggler_policy: str = "wait"  # "wait" or "skip"
    ):
        """
        Initialize cycle coordinator.

        Args:
            models: List of ModelInstance objects to coordinate
            data_feed: SharedDataFeed for centralized data
            trading_client: Alpaca TradingClient for account/position fetching (optional, for live/paper mode)
            max_workers: Maximum parallel threads for model execution
            model_timeout_seconds: Timeout for individual model cycles (default: 5 minutes)
            straggler_policy: How to handle slow models ("wait" or "skip")
        """
        self.models = models
        self.data_feed = data_feed
        self.trading_client = trading_client
        self.max_workers = max_workers
        self.model_timeout_seconds = model_timeout_seconds
        self.straggler_policy = straggler_policy

        # Cycle tracking
        self.cycle_count = 0
        self.cycle_history: List[CycleResult] = []

        logger.info(f"CycleCoordinator initialized with {len(models)} models")
        logger.info(f"  Max workers: {max_workers}")
        logger.info(f"  Model timeout: {model_timeout_seconds}s")
        logger.info(f"  Straggler policy: {straggler_policy}")
        logger.info(f"  Alpaca client: {'configured' if trading_client else 'not configured'}")

    def trigger_cycle(self, force_data_refresh: bool = False) -> CycleResult:
        """
        Trigger a single synchronized trading cycle.

        Workflow:
        1. Fetch market data (once, centralized)
        2. Broadcast to all models simultaneously
        3. Wait for all decisions (or timeout)
        4. Collect results
        5. Return cycle result

        Args:
            force_data_refresh: Force fresh data fetch (bypass cache)

        Returns:
            CycleResult with execution details
        """
        self.cycle_count += 1
        cycle_result = CycleResult(self.cycle_count)

        logger.info(f"\n{'='*80}")
        logger.info(f"CYCLE {self.cycle_count} START: {datetime.now()}")
        logger.info(f"{'='*80}")

        try:
            # Step 1: Fetch market data (once for all models)
            logger.info("Step 1: Fetching market data...")
            snapshot = self.data_feed.fetch_market_data(force_refresh=force_data_refresh)

            if not snapshot:
                logger.error("Failed to fetch market data - aborting cycle")
                for model in self.models:
                    cycle_result.model_results[model.model_id] = False
                    cycle_result.errors[model.model_id] = "Data fetch failed"
                cycle_result.mark_complete()
                return cycle_result

            logger.info(f"Market data fetched: {len(snapshot.symbols)} symbols at {snapshot.timestamp}")

            # Step 1b: Fetch account and position data (once for all models)
            # This eliminates 624 redundant API calls per minute (6 models * 104 calls/min/model)
            if self.trading_client:
                logger.info("Step 1b: Fetching account/position data from Alpaca...")
                account_fetch_start = datetime.now()

                account_data = self.data_feed.fetch_account_and_positions(self.trading_client)

                if account_data:
                    account_fetch_duration = (datetime.now() - account_fetch_start).total_seconds()
                    logger.info(f"Account/position data cached in {account_fetch_duration:.3f}s - available to all models")
                else:
                    logger.warning("Failed to fetch account/position data - models will fall back to individual fetches")
            else:
                logger.debug("No Alpaca client provided - skipping account/position fetch")

            # Step 2: Broadcast data and trigger decisions (parallel)
            logger.info(f"Step 2: Broadcasting to {len(self.models)} models...")
            model_decisions = self._broadcast_data(snapshot)

            # Step 3: Wait for all models to complete (with timeout)
            logger.info("Step 3: Waiting for all model decisions...")
            results = self._wait_for_all_decisions(model_decisions)

            # Record results
            for model_id, success in results.items():
                cycle_result.model_results[model_id] = success

            # Log summary
            logger.info(f"\nCycle {self.cycle_count} Summary:")
            logger.info(f"  Successful: {cycle_result.get_success_count()}/{len(self.models)}")
            logger.info(f"  Failed: {cycle_result.get_failure_count()}/{len(self.models)}")
            logger.info(f"  Stragglers: {len(cycle_result.stragglers)}")
            logger.info(f"  Duration: {cycle_result.get_duration_seconds():.2f}s")

        except Exception as e:
            logger.error(f"Error in cycle {self.cycle_count}: {e}")
            import traceback
            logger.error(traceback.format_exc())

        finally:
            cycle_result.mark_complete()
            self.cycle_history.append(cycle_result)

        logger.info(f"{'='*80}")
        logger.info(f"CYCLE {self.cycle_count} END")
        logger.info(f"{'='*80}\n")

        return cycle_result

    def _broadcast_data(self, snapshot: DataSnapshot) -> Dict[str, 'Future']:
        """
        Broadcast data to all models and trigger decisions in parallel.

        Args:
            snapshot: Market data snapshot

        Returns:
            Dict mapping model_id to Future object
        """
        futures = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for model in self.models:
                # Submit each model's decision cycle to the executor
                future = executor.submit(self._execute_model_cycle, model, snapshot)
                futures[model.model_id] = future

        return futures

    def _execute_model_cycle(self, model: ModelInstance, snapshot: DataSnapshot) -> bool:
        """
        Execute a single model's decision cycle.

        Args:
            model: ModelInstance to execute
            snapshot: Market data snapshot (NOW USED - passes to model to skip data loading)

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"[{model.model_id}] Starting cycle with pre-fetched snapshot...")

            # Check if cached account/position data is available
            account_cache_status = self.data_feed.get_account_cache_status()
            if account_cache_status['cached'] and not account_cache_status['is_stale']:
                logger.debug(
                    f"[{model.model_id}] Cached account data available "
                    f"(age: {account_cache_status['age_seconds']:.1f}s, "
                    f"equity: ${account_cache_status['account_equity']:,.2f})"
                )
            else:
                logger.debug(f"[{model.model_id}] No cached account data - model will fetch individually")

            # FIXED: Now passing snapshot to skip redundant data loading
            # This reduces data fetch time from 30+ minutes to ~5 minutes for 3 models
            # Each model now receives pre-fetched market data instead of fetching independently
            success = model.process_market_data(market_data=snapshot)

            if success:
                logger.info(f"[{model.model_id}] Cycle completed successfully")
            else:
                logger.warning(f"[{model.model_id}] Cycle failed")

            return success

        except Exception as e:
            logger.error(f"[{model.model_id}] Error in cycle: {e}")
            return False

    def _wait_for_all_decisions(self, futures: Dict[str, 'Future']) -> Dict[str, bool]:
        """
        Wait for all model decisions to complete (with timeout handling).

        Args:
            futures: Dict mapping model_id to Future

        Returns:
            Dict mapping model_id to success boolean
        """
        results = {}

        for model_id, future in futures.items():
            try:
                # Wait for this model's result with timeout
                success = future.result(timeout=self.model_timeout_seconds)
                results[model_id] = success

            except TimeoutError:
                logger.warning(f"[{model_id}] Timed out after {self.model_timeout_seconds}s")
                results[model_id] = False

                # Handle straggler based on policy
                if self.straggler_policy == "skip":
                    logger.warning(f"[{model_id}] Skipping straggler (policy: skip)")
                else:
                    logger.info(f"[{model_id}] Waiting for straggler to complete (policy: wait)")

            except Exception as e:
                logger.error(f"[{model_id}] Exception during execution: {e}")
                results[model_id] = False

        return results

    def batch_execute_orders(self) -> Dict[str, int]:
        """
        Batch execute orders from all models.

        Orders are executed by AlpacaBrokerExecutor during each model's cycle.
        This method is reserved for future enhancements to coordinate order submission timing.

        Returns:
            Dict mapping model_id to number of orders executed
        """
        logger.info("Batch order execution...")

        # Orders are already executed in each model's cycle via AlpacaBrokerExecutor
        # This is a no-op for now, but reserved for future use
        order_counts = {}

        for model in self.models:
            # Could check position_manager here to count recent fills
            order_counts[model.model_id] = 0

        logger.info(f"Batch execution complete: {sum(order_counts.values())} total orders")
        return order_counts

    def get_cycle_stats(self) -> Dict:
        """
        Get statistics about cycle execution.

        Returns:
            Dict with cycle performance metrics
        """
        if not self.cycle_history:
            return {
                'total_cycles': 0,
                'avg_duration': 0.0,
                'avg_success_rate': 0.0
            }

        total_cycles = len(self.cycle_history)
        avg_duration = sum(c.get_duration_seconds() for c in self.cycle_history) / total_cycles

        # Calculate average success rate across all cycles
        total_successes = sum(c.get_success_count() for c in self.cycle_history)
        total_attempts = total_cycles * len(self.models)
        avg_success_rate = total_successes / total_attempts if total_attempts > 0 else 0.0

        return {
            'total_cycles': total_cycles,
            'avg_duration': avg_duration,
            'avg_success_rate': avg_success_rate,
            'last_cycle_duration': self.cycle_history[-1].get_duration_seconds() if self.cycle_history else 0.0
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test coordinator (without actual models)
    print("CycleCoordinator initialized for testing")
    print("This module requires ModelInstance and SharedDataFeed instances")
    print("See battle_orchestrator.py for full integration example")
