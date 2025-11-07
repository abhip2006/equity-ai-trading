"""
Shared Data Feed: Centralized market data fetching for all model instances.

Ensures data consistency by:
- Fetching market data once per cycle
- Fetching account/position data once per cycle (eliminates 624 redundant API calls/min)
- Caching for all models to use
- Broadcasting identical snapshots to all models
- Preventing data staleness issues
- Thread-safe caching with configurable TTL
"""

import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from threading import Lock

logger = logging.getLogger(__name__)


class DataSnapshot:
    """
    Immutable market data snapshot for a single cycle.
    """
    def __init__(
        self,
        timestamp: datetime,
        symbols: List[str],
        price_data: Dict,
        features: Dict,
        market_breadth: Optional[Dict] = None,
        macro_data: Optional[Dict] = None
    ):
        self.timestamp = timestamp
        self.symbols = symbols
        self.price_data = price_data
        self.features = features
        self.market_breadth = market_breadth
        self.macro_data = macro_data

    def is_stale(self, max_age_seconds: int = 300) -> bool:
        """Check if this snapshot is too old (default: 5 minutes)"""
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > max_age_seconds


class SharedDataFeed:
    """
    Centralized data feed for battle orchestration.

    Fetches market data once and broadcasts to all model instances.
    Fetches account/position data once per cycle from Alpaca API.
    Ensures all models receive identical data snapshots for fair competition.

    Account/Position Caching:
    - Eliminates 624 redundant API calls per minute (6 models * 104 calls/min/model)
    - 5-second TTL ensures fresh data while preventing rate limits
    - Thread-safe access with lock protection
    - Automatic staleness detection
    """

    def __init__(
        self,
        data_fetcher,
        feature_builder,
        universe: List[str],
        interval: str = "1d",
        lookback_days: int = 90,
        cache_enabled: bool = True
    ):
        """
        Initialize shared data feed.

        Args:
            data_fetcher: PriceVolumeIngestor instance
            feature_builder: FeatureBuilder instance
            universe: List of symbols to fetch
            interval: Price interval (1h, 1d, etc.)
            lookback_days: Historical lookback period
            cache_enabled: Whether to cache data
        """
        self.data_fetcher = data_fetcher
        self.feature_builder = feature_builder
        self.universe = universe
        self.interval = interval
        self.lookback_days = lookback_days
        self.cache_enabled = cache_enabled

        # Cache management
        self._cache_lock = Lock()
        self._cached_snapshot: Optional[DataSnapshot] = None
        self._subscribers: Set[str] = set()

        # Account and position cache
        self._cached_account = None
        self._cached_positions = None
        self._account_cache_timestamp: Optional[datetime] = None
        self._account_cache_ttl: int = 5  # seconds

        logger.info(f"SharedDataFeed initialized with {len(universe)} symbols")

    def subscribe(self, model_id: str):
        """
        Subscribe a model instance to this data feed.

        Args:
            model_id: Unique identifier for the model
        """
        with self._cache_lock:
            self._subscribers.add(model_id)
            logger.info(f"Model {model_id} subscribed to data feed ({len(self._subscribers)} total subscribers)")

    def unsubscribe(self, model_id: str):
        """
        Unsubscribe a model instance from this data feed.

        Args:
            model_id: Unique identifier for the model
        """
        with self._cache_lock:
            self._subscribers.discard(model_id)
            logger.info(f"Model {model_id} unsubscribed from data feed ({len(self._subscribers)} remaining)")

    def fetch_market_data(self, force_refresh: bool = False) -> Optional[DataSnapshot]:
        """
        Fetch fresh market data for all symbols.

        This is the primary method called once per cycle by the coordinator.
        Data is fetched once and cached for all models to use.

        Args:
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            DataSnapshot with all market data, or None if fetch failed
        """
        with self._cache_lock:
            # Check cache first (unless force refresh)
            if not force_refresh and self._cached_snapshot and not self._cached_snapshot.is_stale():
                logger.info("Using cached market data snapshot")
                return self._cached_snapshot

            try:
                fetch_start = datetime.now()
                logger.info(f"Fetching market data for {len(self.universe)} symbols...")

                # Fetch price data
                price_df = self.data_fetcher.fetch_prices(
                    universe=self.universe,
                    interval=self.interval,
                    lookback_days=self.lookback_days,
                    use_cache=self.cache_enabled
                )

                if price_df.empty:
                    logger.error("No price data fetched")
                    return None

                # Compute features
                logger.info("Computing technical features...")
                features_dict = self.feature_builder.build_features(price_df)

                if not features_dict:
                    logger.error("No features computed")
                    return None

                # Build market snapshot (breadth indicators)
                market_breadth = self.feature_builder.build_market_snapshot(price_df, features_dict)

                # Create immutable snapshot
                snapshot = DataSnapshot(
                    timestamp=datetime.now(),
                    symbols=list(features_dict.keys()),
                    price_data=price_df.to_dict('records'),  # Convert DataFrame to dict
                    features=features_dict,
                    market_breadth=market_breadth.model_dump() if market_breadth else None,
                    macro_data=None  # TODO: Add macro data fetching if enabled
                )

                # Cache the snapshot
                self._cached_snapshot = snapshot

                duration = (datetime.now() - fetch_start).total_seconds()
                logger.info(f"Market data fetched in {duration:.2f}s ({len(snapshot.symbols)} symbols)")

                return snapshot

            except Exception as e:
                logger.error(f"Error fetching market data: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None

    def get_cached_data(self) -> Optional[DataSnapshot]:
        """
        Get the currently cached data snapshot.

        Returns:
            Cached DataSnapshot or None if no data cached
        """
        with self._cache_lock:
            if self._cached_snapshot and not self._cached_snapshot.is_stale():
                return self._cached_snapshot
            else:
                logger.warning("Cached data is stale or missing")
                return None

    def invalidate_cache(self):
        """
        Invalidate the current cache.

        Useful for forcing a fresh fetch on the next cycle.
        """
        with self._cache_lock:
            self._cached_snapshot = None
            logger.info("Data cache invalidated")

    def get_subscriber_count(self) -> int:
        """Get the number of subscribed models"""
        with self._cache_lock:
            return len(self._subscribers)

    def get_cache_status(self) -> Dict:
        """
        Get cache status information.

        Returns:
            Dict with cache metadata
        """
        with self._cache_lock:
            if self._cached_snapshot:
                age_seconds = (datetime.now() - self._cached_snapshot.timestamp).total_seconds()
                return {
                    'cached': True,
                    'timestamp': self._cached_snapshot.timestamp.isoformat(),
                    'age_seconds': age_seconds,
                    'is_stale': self._cached_snapshot.is_stale(),
                    'symbol_count': len(self._cached_snapshot.symbols),
                    'subscribers': len(self._subscribers)
                }
            else:
                return {
                    'cached': False,
                    'timestamp': None,
                    'age_seconds': None,
                    'is_stale': True,
                    'symbol_count': 0,
                    'subscribers': len(self._subscribers)
                }

    def fetch_account_and_positions(self, trading_client) -> Optional[Dict]:
        """
        Fetch account and position data from Alpaca API and cache it.

        This method should be called once per cycle by the coordinator.
        All models can then access cached data without making redundant API calls.

        Args:
            trading_client: Alpaca TradingClient instance

        Returns:
            Dict with account, positions, and timestamp, or None if fetch failed
        """
        with self._cache_lock:
            try:
                fetch_start = datetime.now()
                logger.info("Fetching account and position data from Alpaca API...")

                # Fetch account info
                account = trading_client.get_account()
                logger.debug(f"Account fetched: equity=${account.equity}, buying_power=${account.buying_power}")

                # Fetch all positions
                positions = trading_client.get_all_positions()
                logger.debug(f"Positions fetched: {len(positions)} open positions")

                # Update cache
                self._cached_account = account
                self._cached_positions = positions
                self._account_cache_timestamp = datetime.now()

                duration = (datetime.now() - fetch_start).total_seconds()
                logger.info(
                    f"Account/positions cached in {duration:.2f}s "
                    f"(equity=${account.equity}, {len(positions)} positions)"
                )

                return {
                    'account': account,
                    'positions': positions,
                    'timestamp': self._account_cache_timestamp
                }

            except Exception as e:
                logger.error(f"Error fetching account/positions from Alpaca: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None

    def get_cached_account(self):
        """
        Get cached account data if fresh.

        Returns:
            Cached account object or None if stale/missing
        """
        with self._cache_lock:
            if self._cached_account is None or self._account_cache_timestamp is None:
                logger.debug("Account cache miss: no cached data")
                return None

            age_seconds = (datetime.now() - self._account_cache_timestamp).total_seconds()

            if age_seconds > self._account_cache_ttl:
                logger.debug(f"Account cache miss: data stale ({age_seconds:.1f}s > {self._account_cache_ttl}s)")
                return None

            logger.debug(f"Account cache hit (age: {age_seconds:.1f}s)")
            return self._cached_account

    def get_cached_positions(self):
        """
        Get cached position data if fresh.

        Returns:
            Cached positions list or None if stale/missing
        """
        with self._cache_lock:
            if self._cached_positions is None or self._account_cache_timestamp is None:
                logger.debug("Positions cache miss: no cached data")
                return None

            age_seconds = (datetime.now() - self._account_cache_timestamp).total_seconds()

            if age_seconds > self._account_cache_ttl:
                logger.debug(f"Positions cache miss: data stale ({age_seconds:.1f}s > {self._account_cache_ttl}s)")
                return None

            logger.debug(f"Positions cache hit (age: {age_seconds:.1f}s, {len(self._cached_positions)} positions)")
            return self._cached_positions

    def invalidate_account_cache(self):
        """
        Invalidate the account and position cache.

        Useful for forcing a fresh fetch on the next cycle or after trades are executed.
        """
        with self._cache_lock:
            self._cached_account = None
            self._cached_positions = None
            self._account_cache_timestamp = None
            logger.info("Account/position cache invalidated")

    def get_account_cache_status(self) -> Dict:
        """
        Get account cache status information.

        Returns:
            Dict with account cache metadata
        """
        with self._cache_lock:
            if self._cached_account and self._account_cache_timestamp:
                age_seconds = (datetime.now() - self._account_cache_timestamp).total_seconds()
                is_stale = age_seconds > self._account_cache_ttl

                return {
                    'cached': True,
                    'timestamp': self._account_cache_timestamp.isoformat(),
                    'age_seconds': age_seconds,
                    'is_stale': is_stale,
                    'ttl_seconds': self._account_cache_ttl,
                    'position_count': len(self._cached_positions) if self._cached_positions else 0,
                    'account_equity': float(self._cached_account.equity) if self._cached_account else None
                }
            else:
                return {
                    'cached': False,
                    'timestamp': None,
                    'age_seconds': None,
                    'is_stale': True,
                    'ttl_seconds': self._account_cache_ttl,
                    'position_count': 0,
                    'account_equity': None
                }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Simulated test (without actual data fetcher)
    class MockDataFetcher:
        def fetch_prices(self, universe, interval, lookback_days, use_cache):
            import pandas as pd
            # Return empty DataFrame for testing
            return pd.DataFrame()

    class MockFeatureBuilder:
        def build_features(self, price_df):
            return {}

        def build_market_snapshot(self, price_df, features_dict):
            return None

    feed = SharedDataFeed(
        data_fetcher=MockDataFetcher(),
        feature_builder=MockFeatureBuilder(),
        universe=["AAPL", "MSFT", "SPY"],
        interval="1d",
        lookback_days=90
    )

    # Test subscription
    feed.subscribe("model_1")
    feed.subscribe("model_2")

    print(f"\nSubscribers: {feed.get_subscriber_count()}")
    print(f"Cache Status: {feed.get_cache_status()}")
