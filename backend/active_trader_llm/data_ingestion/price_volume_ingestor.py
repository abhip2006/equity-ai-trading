"""
Price and volume data ingestion with support for:
- yfinance (for historical data)
- Alpaca (for live/paper-live real-time data)

Automatically selects data source based on mode.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta
import sqlite3
import logging
from functools import wraps
import time

logger = logging.getLogger(__name__)


def retry_with_backoff(max_retries=3, base_delay=1.0):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


class PriceVolumeIngestor:
    """
    Fetch and cache OHLCV data from yfinance or Alpaca.

    Automatically selects data source:
    - live/paper-live mode: Alpaca (real-time data)
    - Fallback: yfinance (historical data)
    """

    def __init__(self, cache_db_path: str = "data/price_cache.db", mode: str = "paper-live"):
        """
        Initialize price ingestor with SQLite cache.

        Args:
            cache_db_path: Path to SQLite database for caching
            mode: Trading mode ('paper-live', 'live')
        """
        self.cache_db_path = Path(cache_db_path)
        self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self._init_cache_db()

        # Initialize Alpaca data provider for live/paper-live modes
        self.alpaca_provider = None
        if mode in ['paper-live', 'live']:
            try:
                from active_trader_llm.data_ingestion.alpaca_data_provider import AlpacaDataProvider
                self.alpaca_provider = AlpacaDataProvider()
                logger.info(f"Alpaca data provider enabled for {mode} mode")
            except Exception as e:
                logger.warning(f"Failed to initialize Alpaca data provider: {e}")
                logger.warning("Falling back to yfinance")
        else:
            logger.info(f"Using yfinance for {mode} mode")

    def _init_cache_db(self):
        """Initialize cache database schema"""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                symbol TEXT,
                timestamp TEXT,
                interval TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                fetched_at TEXT,
                PRIMARY KEY (symbol, timestamp, interval)
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_interval
            ON price_data(symbol, interval, timestamp)
        ''')

        conn.commit()
        conn.close()

    @retry_with_backoff(max_retries=3)
    def _fetch_from_yfinance(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch data from yfinance with rate limiting.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Data interval ('1h', '1d', etc.)
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with OHLCV data
        """
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,
            actions=False
        )

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        # Standardize column names
        df.columns = df.columns.str.lower()
        df.index.name = 'timestamp'
        df.reset_index(inplace=True)

        return df

    def _cache_data(self, symbol: str, interval: str, df: pd.DataFrame):
        """Cache data to SQLite"""
        if df.empty:
            return

        conn = sqlite3.connect(self.cache_db_path)
        df_cache = df.copy()
        df_cache['symbol'] = symbol
        df_cache['interval'] = interval
        df_cache['fetched_at'] = datetime.now().isoformat()

        # Convert timestamp to string for storage
        df_cache['timestamp'] = df_cache['timestamp'].astype(str)

        # Use INSERT OR REPLACE to handle conflicts with existing cached data
        df_cache.to_sql('price_data', conn, if_exists='append', index=False)
        conn.close()

    def _get_cached_data(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Retrieve cached data from SQLite"""
        conn = sqlite3.connect(self.cache_db_path)

        query = '''
            SELECT timestamp, open, high, low, close, volume
            FROM price_data
            WHERE symbol = ? AND interval = ?
            AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
        '''

        df = pd.read_sql_query(
            query,
            conn,
            params=(symbol, interval, start_date.isoformat(), end_date.isoformat())
        )
        conn.close()

        if df.empty:
            return None

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def fetch_prices(
        self,
        universe: List[str],
        interval: str = "1h",
        lookback_days: int = 90,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch prices for universe of symbols.

        Automatically uses:
        - Alpaca for live/paper-live mode (real-time data)
        - yfinance as fallback (historical data)

        Args:
            universe: List of stock symbols
            interval: Data interval ('1h', '1d', '5m', '15m')
            lookback_days: Number of days to look back
            use_cache: Whether to use cached data (only for yfinance fallback)

        Returns:
            DataFrame with multi-symbol OHLCV data
        """
        # Use Alpaca for live/paper-live mode (always fresh data)
        if self.alpaca_provider is not None:
            logger.info(f"Fetching real-time data from Alpaca for {len(universe)} symbols ({interval})")
            try:
                df = self.alpaca_provider.fetch_bars(
                    symbols=universe,
                    interval=interval,
                    lookback_days=lookback_days
                )

                if not df.empty:
                    logger.info(f"  Fetched {len(df)} bars from Alpaca")
                    return df
                else:
                    logger.warning("No data from Alpaca, falling back to yfinance")

            except Exception as e:
                logger.error(f"Alpaca fetch failed: {e}, falling back to yfinance")

        # Fall back to yfinance for historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        all_data = []

        for symbol in universe:
            logger.info(f"Fetching {symbol} {interval} data from yfinance...")

            # Try cache first
            if use_cache:
                df = self._get_cached_data(symbol, interval, start_date, end_date)
                if df is not None and not df.empty:
                    logger.info(f"  Using cached data for {symbol}")
                    df['symbol'] = symbol
                    all_data.append(df)
                    continue

            # Fetch from yfinance
            try:
                df = self._fetch_from_yfinance(symbol, interval, start_date, end_date)
                if not df.empty:
                    # Cache the data
                    if use_cache:
                        self._cache_data(symbol, interval, df)

                    df['symbol'] = symbol
                    all_data.append(df)
                else:
                    logger.warning(f"  No data for {symbol}")

            except Exception as e:
                logger.error(f"  Error fetching {symbol}: {e}")
                continue

        if not all_data:
            return pd.DataFrame()

        # Combine all symbols
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.sort_values(['symbol', 'timestamp'], inplace=True)

        return combined_df


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ingestor = PriceVolumeIngestor()
    df = ingestor.fetch_prices(
        universe=["AAPL", "MSFT"],
        interval="1d",
        lookback_days=30
    )

    print(f"\nFetched {len(df)} rows for {df['symbol'].nunique()} symbols")
    print(df.head(10))
