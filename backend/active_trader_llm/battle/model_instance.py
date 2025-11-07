"""
Model Instance: Wrapper for isolated ActiveTraderLLM execution.

Encapsulates a complete trading model instance with:
- Isolated configuration (model, database)
- Independent execution state
- Health monitoring
- Graceful lifecycle management
"""

import logging
import traceback
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from active_trader_llm.main import ActiveTraderLLM
from active_trader_llm.config.loader import load_config

logger = logging.getLogger(__name__)


class ModelInstanceState:
    """State tracking for a model instance"""
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.status = "initialized"  # initialized, running, error, shutdown
        self.cycles_completed = 0
        self.last_cycle_time: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self.error_count = 0


class ModelInstance:
    """
    Wrapper for a single ActiveTraderLLM instance in the battle.

    Each model runs in isolation with:
    - Separate database for position tracking
    - Independent configuration
    - Isolated state management
    - Error recovery
    """

    def __init__(
        self,
        model_id: str,
        llm_config: Dict[str, Any],
        base_config_path: str,
        database_dir: str = "data/battle",
        shared_data_feed=None
    ):
        """
        Initialize model instance.

        Args:
            model_id: Unique identifier for this model (e.g., "gpt4o", "claude_sonnet")
            llm_config: LLM configuration (provider, model, api_key, temperature, etc.)
            base_config_path: Path to base config.yaml template
            database_dir: Directory for model-specific databases
            shared_data_feed: Optional SharedDataFeed instance for caching account/position data
        """
        self.model_id = model_id
        self.llm_config = llm_config
        self.base_config_path = base_config_path
        self.database_dir = Path(database_dir)
        self.shared_data_feed = shared_data_feed

        # State tracking
        self.state = ModelInstanceState(model_id)

        # Model instance (initialized later)
        self.trader_instance: Optional[ActiveTraderLLM] = None

        # Database path (isolated per model)
        self.database_path = str(self.database_dir / f"{model_id}_trading.db")
        self.positions_db_path = str(self.database_dir / f"{model_id}_positions.db")

        # Local cache for account/position data (two-tier caching)
        self._cached_account_info: Optional[Dict] = None
        self._cached_positions: Optional[list] = None
        self._cache_timestamp: Optional[datetime] = None
        self.CACHE_TTL_SECONDS = 5

        logger.info(f"ModelInstance created: {model_id}")
        logger.info(f"  LLM: {llm_config.get('provider')}/{llm_config.get('model')}")
        logger.info(f"  Database: {self.database_path}")
        logger.info(f"  Caching enabled: {shared_data_feed is not None}")

    def initialize(self) -> bool:
        """
        Initialize the ActiveTraderLLM instance.

        Returns:
            True if initialization succeeded, False otherwise
        """
        try:
            logger.info(f"Initializing model instance: {self.model_id}")

            # Create database directory if needed
            self.database_dir.mkdir(parents=True, exist_ok=True)

            # Load base configuration as raw YAML to preserve structure
            import yaml
            with open(self.base_config_path, 'r') as f:
                base_config_dict = yaml.safe_load(f)

            # Set mode to paper-live for battle (required for Alpaca trading)
            if 'battle' in base_config_dict and 'mode' in base_config_dict['battle']:
                base_config_dict['mode'] = base_config_dict['battle']['mode']
                logger.info(f"  Mode: {base_config_dict['mode']}")

            # Ensure 'llm' section exists (create if missing)
            if 'llm' not in base_config_dict:
                base_config_dict['llm'] = {}

            # Override LLM configuration
            base_config_dict['llm']['provider'] = self.llm_config.get('provider', 'openai')
            base_config_dict['llm']['model'] = self.llm_config.get('model', 'gpt-4.1-nano')
            base_config_dict['llm']['api_key'] = self.llm_config.get('api_key')
            base_config_dict['llm']['temperature'] = self.llm_config.get('temperature', 0.3)
            base_config_dict['llm']['max_tokens'] = self.llm_config.get('max_tokens', 2000)

            # Add base_url if provided (for DeepSeek, etc.)
            if self.llm_config.get('base_url'):
                base_config_dict['llm']['base_url'] = self.llm_config['base_url']
            # Remove base_url if it exists but shouldn't (for OpenAI/Anthropic/Google)
            elif 'base_url' in base_config_dict['llm']:
                del base_config_dict['llm']['base_url']

            # Override database paths (isolated per model)
            base_config_dict['database_path'] = self.database_path

            # Override Alpaca credentials if provided (battle mode)
            if 'alpaca' in self.llm_config:
                if 'execution' not in base_config_dict:
                    base_config_dict['execution'] = {}

                alpaca_config = self.llm_config['alpaca']
                base_config_dict['execution']['alpaca_api_key_env'] = alpaca_config.get('api_key_env', 'ALPACA_API_KEY')
                base_config_dict['execution']['alpaca_secret_key_env'] = alpaca_config.get('secret_key_env', 'ALPACA_SECRET_KEY')
                base_config_dict['execution']['alpaca_base_url'] = alpaca_config.get('base_url', 'https://paper-api.alpaca.markets')

                logger.info(f"  Alpaca API key env: {base_config_dict['execution']['alpaca_api_key_env']}")
                logger.info(f"  Alpaca secret key env: {base_config_dict['execution']['alpaca_secret_key_env']}")
                logger.info(f"  Alpaca base URL: {base_config_dict['execution']['alpaca_base_url']}")

            # Ensure technical_indicators section exists with all required keys
            if 'technical_indicators' not in base_config_dict:
                base_config_dict['technical_indicators'] = {}
            if 'quality' not in base_config_dict['technical_indicators']:
                base_config_dict['technical_indicators']['quality'] = {
                    'min_bars_required': 20,
                    'reject_incomplete': False
                }
            if 'daily' not in base_config_dict['technical_indicators']:
                base_config_dict['technical_indicators']['daily'] = {}
            if 'weekly' not in base_config_dict['technical_indicators']:
                base_config_dict['technical_indicators']['weekly'] = {}

            # Ensure weekly indicators have conversion fields (required for daily->weekly transformation)
            weekly = base_config_dict['technical_indicators']['weekly']
            if 'ema_10' in weekly and 'conversion' not in weekly['ema_10']:
                weekly['ema_10']['conversion'] = 50  # 10 weeks * 5 trading days
            if 'sma_21' in weekly and 'conversion' not in weekly['sma_21']:
                weekly['sma_21']['conversion'] = 105  # 21 weeks * 5 trading days
            if 'sma_30' in weekly and 'conversion' not in weekly['sma_30']:
                weekly['sma_30']['conversion'] = 150  # 30 weeks * 5 trading days
            if 'sma_50' in weekly and 'conversion' not in weekly['sma_50']:
                weekly['sma_50']['conversion'] = 250  # 50 weeks * 5 trading days

            # Ensure market_breadth section exists
            if 'market_breadth' not in base_config_dict:
                base_config_dict['market_breadth'] = {
                    'enabled': True,
                    'new_highs_lows': {
                        'lookback_days': 252,
                        'min_data_bars': 20,
                        'high_threshold_pct': 0.98,
                        'low_threshold_pct': 1.02
                    }
                }

            # Ensure batch_processing section exists (required for LLM trading decisions)
            if 'batch_processing' not in base_config_dict:
                base_config_dict['batch_processing'] = {
                    'enabled': False,  # Disabled by default - LLM evaluates all symbols individually
                    'batch_size': 10,
                    'filters': {
                        'min_volume_ratio': 2.0,
                        'min_momentum_5d': 2.0,
                        'max_distance_from_high': 10.0,
                        'min_liquidity': 500_000_000,
                        'min_adr': 1.0,
                        'max_adr': 15.0
                    },
                    'log_savings': True,
                    'log_filter_details': False
                }

            # Create temporary config file for this model
            temp_config_path = self.database_dir / f"{self.model_id}_config.yaml"

            # Save modified config, preserving all nested structures
            with open(temp_config_path, 'w') as f:
                yaml.dump(base_config_dict, f, default_flow_style=False, sort_keys=False)

            # Initialize ActiveTraderLLM with modified config
            # Use the mode from config (paper-live for Alpaca trading)
            self.trader_instance = ActiveTraderLLM(
                config_path=str(temp_config_path)
            )

            self.state.status = "running"
            logger.info(f"Model instance {self.model_id} initialized successfully")
            return True

        except Exception as e:
            self.state.status = "error"
            self.state.last_error = str(e)
            self.state.error_count += 1
            logger.error(f"Failed to initialize model instance {self.model_id}: {e}")
            logger.error(traceback.format_exc())
            return False

    def process_market_data(self, market_data: Optional['DataSnapshot'] = None) -> bool:
        """
        Execute one decision cycle with optional pre-fetched market data.

        Args:
            market_data: Optional pre-fetched DataSnapshot from SharedDataFeed.
                        If provided, passes to trader_instance to skip data loading.
                        This provides 80%+ performance improvement in battle mode.

        Returns:
            True if cycle completed successfully, False otherwise
        """
        if self.state.status != "running":
            logger.warning(f"Model {self.model_id} not in running state (status: {self.state.status})")
            return False

        if not self.trader_instance:
            logger.error(f"Model {self.model_id} not initialized")
            return False

        try:
            start_time = datetime.now()

            if market_data:
                logger.info(f"[{self.model_id}] Starting decision cycle with PRE-FETCHED data...")
                # Pass snapshot to trader to skip data loading
                self.trader_instance.run_decision_cycle(pre_fetched_snapshot=market_data)
            else:
                logger.info(f"[{self.model_id}] Starting decision cycle (independent data fetch)...")
                # Original flow - trader fetches data independently
                self.trader_instance.run_decision_cycle()

            # Update state
            self.state.cycles_completed += 1
            self.state.last_cycle_time = datetime.now()

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"[{self.model_id}] Cycle completed in {duration:.2f}s")

            return True

        except Exception as e:
            self.state.status = "error"
            self.state.last_error = str(e)
            self.state.error_count += 1
            logger.error(f"[{self.model_id}] Error in decision cycle: {e}")
            logger.error(traceback.format_exc())
            return False

    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current state of this model instance.

        Returns:
            Dict with instance state, performance, and health metrics
        """
        state_dict = {
            'model_id': self.model_id,
            'status': self.state.status,
            'cycles_completed': self.state.cycles_completed,
            'last_cycle_time': self.state.last_cycle_time.isoformat() if self.state.last_cycle_time else None,
            'error_count': self.state.error_count,
            'last_error': self.state.last_error,
            'llm_config': {
                'provider': self.llm_config.get('provider'),
                'model': self.llm_config.get('model'),
                'temperature': self.llm_config.get('temperature')
            }
        }

        # Add portfolio state if available
        if self.trader_instance:
            try:
                portfolio = self.trader_instance.get_current_portfolio_state()
                state_dict['portfolio'] = {
                    'cash': portfolio.cash,
                    'equity': portfolio.equity,
                    'position_count': len(portfolio.positions),
                    'daily_pnl': portfolio.daily_pnl
                }
            except Exception as e:
                logger.warning(f"Could not fetch portfolio state for {self.model_id}: {e}")
                state_dict['portfolio'] = None

        return state_dict

    def shutdown(self) -> bool:
        """
        Gracefully shutdown this model instance.

        Returns:
            True if shutdown succeeded, False otherwise
        """
        try:
            logger.info(f"Shutting down model instance: {self.model_id}")

            # Clean up trader instance
            if self.trader_instance:
                # Could add cleanup logic here if needed
                self.trader_instance = None

            self.state.status = "shutdown"
            logger.info(f"Model instance {self.model_id} shutdown complete")
            return True

        except Exception as e:
            logger.error(f"Error shutting down model instance {self.model_id}: {e}")
            logger.error(traceback.format_exc())
            return False

    def get_account_info(self) -> Optional[Dict]:
        """
        Get account information with two-tier caching.

        First checks local cache (instance-level), then SharedDataFeed cache (global-level).
        This reduces redundant API calls while maintaining data freshness.

        Returns:
            Account info dict or None if unavailable
        """
        try:
            # Check local cache first
            if self._is_local_cache_fresh():
                logger.debug(f"[{self.model_id}] Account cache hit (local)")
                return self._cached_account_info

            # Check SharedDataFeed cache
            if self.shared_data_feed:
                cached_account = self.shared_data_feed.get_cached_account()
                if cached_account:
                    logger.debug(f"[{self.model_id}] Account cache hit (shared)")
                    # Update local cache
                    self._cached_account_info = self._convert_account_to_dict(cached_account)
                    self._cache_timestamp = datetime.now()
                    return self._cached_account_info
                else:
                    logger.debug(f"[{self.model_id}] Account cache miss (shared cache stale)")
            else:
                logger.warning(f"[{self.model_id}] No SharedDataFeed available for caching")

            # Cache miss or unavailable
            return None

        except Exception as e:
            logger.error(f"[{self.model_id}] Error getting cached account info: {e}")
            return None

    def get_positions(self) -> Optional[list]:
        """
        Get position data with two-tier caching.

        First checks local cache (instance-level), then SharedDataFeed cache (global-level).
        This reduces redundant API calls while maintaining data freshness.

        Returns:
            List of position objects or None if unavailable
        """
        try:
            # Check local cache first
            if self._is_local_cache_fresh():
                logger.debug(f"[{self.model_id}] Positions cache hit (local)")
                return self._cached_positions

            # Check SharedDataFeed cache
            if self.shared_data_feed:
                cached_positions = self.shared_data_feed.get_cached_positions()
                if cached_positions:
                    logger.debug(f"[{self.model_id}] Positions cache hit (shared, {len(cached_positions)} positions)")
                    # Update local cache
                    self._cached_positions = cached_positions
                    self._cache_timestamp = datetime.now()
                    return self._cached_positions
                else:
                    logger.debug(f"[{self.model_id}] Positions cache miss (shared cache stale)")
            else:
                logger.warning(f"[{self.model_id}] No SharedDataFeed available for caching")

            # Cache miss or unavailable
            return None

        except Exception as e:
            logger.error(f"[{self.model_id}] Error getting cached positions: {e}")
            return None

    def invalidate_cache(self):
        """
        Invalidate local cache to force refresh on next access.

        Useful after trade execution or when fresh data is needed.
        """
        self._cached_account_info = None
        self._cached_positions = None
        self._cache_timestamp = None
        logger.debug(f"[{self.model_id}] Local cache invalidated")

    def _is_local_cache_fresh(self) -> bool:
        """
        Check if local cache is fresh (within TTL).

        Returns:
            True if cache is fresh, False otherwise
        """
        if self._cache_timestamp is None:
            return False

        age_seconds = (datetime.now() - self._cache_timestamp).total_seconds()
        return age_seconds <= self.CACHE_TTL_SECONDS

    def _convert_account_to_dict(self, account) -> Dict:
        """
        Convert Alpaca account object to dict for local caching.

        Args:
            account: Alpaca account object

        Returns:
            Dict with account fields
        """
        try:
            # Handle both dict and object types
            if isinstance(account, dict):
                return account

            # Convert Alpaca account object to dict
            return {
                'account_number': getattr(account, 'account_number', None),
                'cash': float(getattr(account, 'cash', 0)),
                'equity': float(getattr(account, 'equity', 0)),
                'buying_power': float(getattr(account, 'buying_power', 0)),
                'portfolio_value': float(getattr(account, 'portfolio_value', 0)),
                'long_market_value': float(getattr(account, 'long_market_value', 0)),
                'short_market_value': float(getattr(account, 'short_market_value', 0)),
                'initial_margin': float(getattr(account, 'initial_margin', 0)),
                'maintenance_margin': float(getattr(account, 'maintenance_margin', 0)),
                'daytrade_count': getattr(account, 'daytrade_count', 0),
                'pattern_day_trader': getattr(account, 'pattern_day_trader', False),
                'trading_blocked': getattr(account, 'trading_blocked', False),
                'account_blocked': getattr(account, 'account_blocked', False),
            }
        except Exception as e:
            logger.error(f"[{self.model_id}] Error converting account to dict: {e}")
            return {}

    def is_healthy(self) -> bool:
        """
        Check if this model instance is healthy.

        Returns:
            True if healthy, False if experiencing errors
        """
        # Consider unhealthy if:
        # - Status is error
        # - Error count is too high (>5 consecutive errors)
        # - Last cycle was too long ago (>10 minutes) - TODO: implement timeout check

        if self.state.status == "error":
            return False

        if self.state.error_count > 5:
            logger.warning(f"Model {self.model_id} has {self.state.error_count} errors")
            return False

        return True


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test model instance creation
    test_config = {
        'provider': 'openai',
        'model': 'gpt-4.1-nano',
        'api_key': 'test-key',
        'temperature': 0.6
    }

    instance = ModelInstance(
        model_id="test_gpt35",
        llm_config=test_config,
        base_config_path="config.yaml",
        database_dir="data/battle_test"
    )

    print(f"\nModel Instance: {instance.model_id}")
    print(f"Database: {instance.database_path}")
    print(f"Status: {instance.state.status}")
