#!/usr/bin/env python3
"""
Alpaca Data Provider: Fetch real-time market data from Alpaca

Uses Alpaca's data API for real-time price fetching in live/paper-live mode.
Replaces yfinance for current/recent data to ensure zero staleness.

Features:
- Real-time bar data (1min, 5min, 15min, 1h, 1d)
- Latest quotes for exact current prices
- Multi-symbol batch fetching
- Same pandas DataFrame format as yfinance for compatibility
"""

import logging
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional

try:
    from alpaca.data import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except ImportError:
    logging.warning("alpaca-py not installed. Install with: pip install alpaca-py")
    StockHistoricalDataClient = None

logger = logging.getLogger(__name__)


class AlpacaDataProvider:
    """
    Fetch real-time market data from Alpaca.

    Provides the same interface as yfinance but with real-time Alpaca data.
    """

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        """
        Initialize Alpaca data provider.

        Args:
            api_key: Alpaca API key (or uses ALPACA_API_KEY env var)
            secret_key: Alpaca secret key (or uses ALPACA_SECRET_KEY env var)
        """
        if StockHistoricalDataClient is None:
            raise ImportError("alpaca-py required. Install: pip install alpaca-py")

        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')

        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not provided")

        # Initialize data client (no paper vs live distinction for data API)
        self.client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key
        )

        logger.info("AlpacaDataProvider initialized")

    def _convert_interval_to_timeframe(self, interval: str) -> TimeFrame:
        """
        Convert interval string to Alpaca TimeFrame.

        Args:
            interval: Interval string (e.g., '1d', '1h', '5m')

        Returns:
            Alpaca TimeFrame object
        """
        interval_map = {
            '1m': TimeFrame(1, TimeFrameUnit.Minute),
            '5m': TimeFrame(5, TimeFrameUnit.Minute),
            '15m': TimeFrame(15, TimeFrameUnit.Minute),
            '1h': TimeFrame(1, TimeFrameUnit.Hour),
            '1d': TimeFrame(1, TimeFrameUnit.Day),
        }

        if interval not in interval_map:
            logger.warning(f"Unsupported interval '{interval}', defaulting to 1d")
            return TimeFrame(1, TimeFrameUnit.Day)

        return interval_map[interval]

    def fetch_bars(
        self,
        symbols: List[str],
        interval: str = '1d',
        lookback_days: int = 90,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch historical bars from Alpaca.

        Args:
            symbols: List of stock symbols
            interval: Bar interval ('1d', '1h', '5m', '15m', '1m')
            lookback_days: Number of days to look back (if start_date not provided)
            start_date: Optional start date (overrides lookback_days)
            end_date: Optional end date (defaults to now)

        Returns:
            pandas DataFrame with columns: symbol, timestamp, open, high, low, close, volume
        """
        if not symbols:
            logger.warning("No symbols provided to fetch_bars")
            return pd.DataFrame()

        # Calculate date range
        if end_date is None:
            end_date = datetime.now()

        if start_date is None:
            start_date = end_date - timedelta(days=lookback_days)

        # Convert interval to Alpaca timeframe
        timeframe = self._convert_interval_to_timeframe(interval)

        logger.info(f"Fetching Alpaca bars for {len(symbols)} symbols: {interval} from {start_date.date()} to {end_date.date()}")

        try:
            # Create request
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=timeframe,
                start=start_date,
                end=end_date
            )

            # Fetch bars
            bars = self.client.get_stock_bars(request)

            # Convert to DataFrame
            all_data = []

            for symbol in symbols:
                if symbol in bars:
                    symbol_bars = bars[symbol]

                    for bar in symbol_bars:
                        all_data.append({
                            'symbol': symbol,
                            'timestamp': bar.timestamp,
                            'open': float(bar.open),
                            'high': float(bar.high),
                            'low': float(bar.low),
                            'close': float(bar.close),
                            'volume': int(bar.volume)
                        })

            if not all_data:
                logger.warning(f"No bar data returned from Alpaca for symbols: {symbols}")
                return pd.DataFrame()

            df = pd.DataFrame(all_data)

            logger.info(f"Fetched {len(df)} bars from Alpaca for {len(symbols)} symbols")

            return df

        except Exception as e:
            logger.error(f"Error fetching bars from Alpaca: {e}", exc_info=True)
            return pd.DataFrame()

    def fetch_latest_quotes(self, symbols: List[str]) -> pd.DataFrame:
        """
        Fetch latest quotes (most current prices) from Alpaca.

        This provides the absolute latest prices, more current than bars.

        Args:
            symbols: List of stock symbols

        Returns:
            pandas DataFrame with columns: symbol, timestamp, bid, ask, bid_size, ask_size, last_price
        """
        if not symbols:
            logger.warning("No symbols provided to fetch_latest_quotes")
            return pd.DataFrame()

        logger.debug(f"Fetching latest quotes for {len(symbols)} symbols")

        try:
            # Create request
            request = StockLatestQuoteRequest(symbol_or_symbols=symbols)

            # Fetch quotes
            quotes = self.client.get_stock_latest_quote(request)

            # Convert to DataFrame
            all_data = []

            for symbol in symbols:
                if symbol in quotes:
                    quote = quotes[symbol]

                    all_data.append({
                        'symbol': symbol,
                        'timestamp': quote.timestamp,
                        'bid': float(quote.bid_price),
                        'ask': float(quote.ask_price),
                        'bid_size': int(quote.bid_size),
                        'ask_size': int(quote.ask_size),
                        # Use mid-price as "last_price"
                        'last_price': (float(quote.bid_price) + float(quote.ask_price)) / 2.0
                    })

            if not all_data:
                logger.warning(f"No quote data returned from Alpaca for symbols: {symbols}")
                return pd.DataFrame()

            df = pd.DataFrame(all_data)

            logger.debug(f"Fetched latest quotes for {len(df)} symbols")

            return df

        except Exception as e:
            logger.error(f"Error fetching latest quotes from Alpaca: {e}", exc_info=True)
            return pd.DataFrame()

    def fetch_current_prices(self, symbols: List[str]) -> dict:
        """
        Fetch current prices for symbols (convenience method).

        Args:
            symbols: List of stock symbols

        Returns:
            Dict mapping symbol -> current price
        """
        quotes_df = self.fetch_latest_quotes(symbols)

        if quotes_df.empty:
            return {}

        # Return dict of symbol -> last_price
        return dict(zip(quotes_df['symbol'], quotes_df['last_price']))


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    provider = AlpacaDataProvider()

    # Test fetching bars
    bars_df = provider.fetch_bars(
        symbols=['AAPL', 'MSFT', 'NVDA'],
        interval='1d',
        lookback_days=5
    )

    print("\nHistorical Bars:")
    print(bars_df.head(10))

    # Test fetching latest quotes
    quotes_df = provider.fetch_latest_quotes(['AAPL', 'MSFT', 'NVDA'])

    print("\nLatest Quotes:")
    print(quotes_df)

    # Test current prices
    prices = provider.fetch_current_prices(['AAPL', 'MSFT', 'NVDA'])

    print("\nCurrent Prices:")
    for symbol, price in prices.items():
        print(f"  {symbol}: ${price:.2f}")
