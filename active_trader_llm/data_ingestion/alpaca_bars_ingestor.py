"""
Alpaca Bars Ingestor: Efficiently fetch OHLCV data from Alpaca with rate limiting.

Handles:
- Multi-symbol batching (200 symbols per request)
- Rate limit tracking and proactive throttling
- 429 response handling with retry
- Pagination for large datasets
- Caching to minimize API calls

Rate Limits:
- Standard: 200 requests/minute
- Unlimited: 1000 requests/minute
- Burst: 10 requests/second
- Multi-symbol: Max 200 symbols per request
"""

import pandas as pd
import logging
import time
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import sqlite3

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.common.exceptions import APIError
except ImportError:
    logging.warning("alpaca-py not installed. Install with: pip install alpaca-py")
    StockHistoricalDataClient = None

logger = logging.getLogger(__name__)


class AlpacaRateLimiter:
    """Track and manage Alpaca API rate limits"""

    def __init__(self, requests_per_minute: int = 200):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: 200 (standard) or 1000 (unlimited plan)
        """
        self.requests_per_minute = requests_per_minute
        self.requests_made = []
        self.last_reset_time = time.time()

    def can_make_request(self) -> bool:
        """Check if we can make a request without hitting rate limit"""
        now = time.time()

        # Remove requests older than 1 minute
        self.requests_made = [
            req_time for req_time in self.requests_made
            if now - req_time < 60
        ]

        return len(self.requests_made) < self.requests_per_minute

    def wait_if_needed(self):
        """Wait if approaching rate limit"""
        while not self.can_make_request():
            oldest_request = min(self.requests_made)
            wait_time = 60 - (time.time() - oldest_request) + 1
            logger.warning(f"Rate limit: waiting {wait_time:.1f}s")
            time.sleep(wait_time)

    def record_request(self):
        """Record that a request was made"""
        self.requests_made.append(time.time())

    def handle_429(self, retry_after: Optional[int] = None):
        """Handle 429 rate limit response"""
        wait_time = retry_after or 60
        logger.error(f"429 Rate Limit Hit. Waiting {wait_time}s")
        time.sleep(wait_time)
        # Clear request history after waiting
        self.requests_made = []


class AlpacaBarsIngestor:
    """
    Fetch OHLCV data from Alpaca with proper rate limiting.

    Designed for efficient bulk fetching of 5000+ stocks without crashing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        cache_db_path: str = "data/alpaca_cache.db",
        requests_per_minute: int = 200
    ):
        """
        Initialize Alpaca bars ingestor.

        Args:
            api_key: Alpaca API key (or uses ALPACA_API_KEY env var)
            secret_key: Alpaca secret key (or uses ALPACA_SECRET_KEY env var)
            cache_db_path: Path to SQLite cache database
            requests_per_minute: Rate limit (200 standard, 1000 unlimited)
        """
        if StockHistoricalDataClient is None:
            raise ImportError("alpaca-py required. Install: pip install alpaca-py")

        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')

        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not provided")

        self.client = StockHistoricalDataClient(self.api_key, self.secret_key)
        self.rate_limiter = AlpacaRateLimiter(requests_per_minute)

        # Cache setup
        self.cache_db_path = Path(cache_db_path)
        self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_cache_db()

    def _init_cache_db(self):
        """Initialize cache database"""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alpaca_bars (
                symbol TEXT,
                timestamp TEXT,
                timeframe TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                trade_count INTEGER,
                vwap REAL,
                fetched_at TEXT,
                PRIMARY KEY (symbol, timestamp, timeframe)
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_alpaca_symbol_timeframe
            ON alpaca_bars(symbol, timeframe, timestamp)
        ''')

        conn.commit()
        conn.close()

    def fetch_bars_batched(
        self,
        symbols: List[str],
        timeframe: str = "1Day",
        start: datetime = None,
        end: datetime = None,
        use_cache: bool = True,
        batch_size: int = 200
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch bars for many symbols using batching to avoid rate limits.

        Args:
            symbols: List of stock symbols (can be 5000+)
            timeframe: "1Day", "1Hour", "1Min", etc.
            start: Start date
            end: End date
            use_cache: Use cached data
            batch_size: Symbols per request (max 200 for Alpaca)

        Returns:
            Dict mapping symbol -> DataFrame with OHLCV data
        """
        if not start:
            start = datetime.now() - timedelta(days=90)
        if not end:
            end = datetime.now()

        # Map timeframe string to Alpaca TimeFrame
        timeframe_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, TimeFrame.Unit.Minute),
            "15Min": TimeFrame(15, TimeFrame.Unit.Minute),
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
            "1Week": TimeFrame.Week,
            "1Month": TimeFrame.Month
        }
        tf = timeframe_map.get(timeframe, TimeFrame.Day)

        all_data = {}
        total_batches = (len(symbols) + batch_size - 1) // batch_size

        logger.info(f"Fetching bars for {len(symbols)} symbols in {total_batches} batches")

        for batch_num, i in enumerate(range(0, len(symbols), batch_size), 1):
            batch = symbols[i:i + batch_size]
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} symbols)")

            # Check cache first
            if use_cache:
                cached_data = self._get_cached_bars(batch, timeframe, start, end)
                for symbol, df in cached_data.items():
                    all_data[symbol] = df
                    batch.remove(symbol)

                if not batch:
                    logger.info(f"Batch {batch_num}: All symbols in cache")
                    continue

            # Fetch from API with rate limiting
            try:
                # Wait if needed to avoid rate limit
                self.rate_limiter.wait_if_needed()

                request_params = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=tf,
                    start=start,
                    end=end,
                    limit=10000  # Max per request
                )

                logger.debug(f"Making API request for {len(batch)} symbols")
                bars_response = self.client.get_stock_bars(request_params)
                self.rate_limiter.record_request()

                # Convert to DataFrames
                for symbol in batch:
                    if symbol in bars_response:
                        df = bars_response[symbol].df
                        df.reset_index(inplace=True)
                        all_data[symbol] = df

                        # Cache the data
                        if use_cache:
                            self._cache_bars(symbol, timeframe, df)
                    else:
                        logger.warning(f"{symbol}: No data returned")

                logger.info(f"Batch {batch_num}: Fetched {len(all_data) - (i)} new symbols")

                # Small delay between batches for safety
                if batch_num < total_batches:
                    time.sleep(0.3)  # ~3 req/sec = 180 req/min (safety margin)

            except APIError as e:
                if e.status_code == 429:
                    # Rate limit hit
                    retry_after = int(e.response.headers.get('Retry-After', 60))
                    self.rate_limiter.handle_429(retry_after)

                    # Retry this batch
                    logger.warning(f"Retrying batch {batch_num} after rate limit wait")
                    # Recursively retry just this batch
                    retry_data = self.fetch_bars_batched(
                        batch, timeframe, start, end, use_cache=False, batch_size=batch_size
                    )
                    all_data.update(retry_data)
                else:
                    logger.error(f"API error in batch {batch_num}: {e}")
                    continue

            except Exception as e:
                logger.error(f"Error fetching batch {batch_num}: {e}")
                continue

        logger.info(f"Total symbols fetched: {len(all_data)}/{len(symbols)}")
        return all_data

    def _get_cached_bars(
        self,
        symbols: List[str],
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Retrieve cached bars for symbols"""
        conn = sqlite3.connect(self.cache_db_path)
        cached_data = {}

        for symbol in symbols:
            query = '''
                SELECT timestamp, open, high, low, close, volume, trade_count, vwap
                FROM alpaca_bars
                WHERE symbol = ? AND timeframe = ?
                AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
            '''

            df = pd.read_sql_query(
                query,
                conn,
                params=(symbol, timeframe, start.isoformat(), end.isoformat())
            )

            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                cached_data[symbol] = df

        conn.close()

        if cached_data:
            logger.info(f"Retrieved {len(cached_data)} symbols from cache")

        return cached_data

    def _cache_bars(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Cache bars to database"""
        if df.empty:
            return

        conn = sqlite3.connect(self.cache_db_path)
        df_cache = df.copy()
        df_cache['symbol'] = symbol
        df_cache['timeframe'] = timeframe
        df_cache['fetched_at'] = datetime.now().isoformat()

        # Convert timestamp to string
        if 'timestamp' in df_cache.columns:
            df_cache['timestamp'] = df_cache['timestamp'].astype(str)

        df_cache.to_sql('alpaca_bars', conn, if_exists='append', index=False)
        conn.close()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ingestor = AlpacaBarsIngestor()

    # Test with small batch
    test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    data = ingestor.fetch_bars_batched(
        symbols=test_symbols,
        timeframe="1Day",
        start=datetime.now() - timedelta(days=30),
        end=datetime.now()
    )

    for symbol, df in data.items():
        print(f"\n{symbol}: {len(df)} bars")
        print(df.head())

    # Simulate 5000 stock fetch (uncomment to test)
    # large_symbols = [f"SYM{i:04d}" for i in range(5000)]
    # start_time = time.time()
    # data = ingestor.fetch_bars_batched(large_symbols, timeframe="1Day")
    # elapsed = time.time() - start_time
    # print(f"\nFetched {len(data)} symbols in {elapsed:.1f} seconds")
    # print(f"API calls: ~{len(large_symbols) // 200}")
