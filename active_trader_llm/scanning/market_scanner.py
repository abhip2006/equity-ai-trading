"""
Market Scanner - Dynamic Universe Discovery

Scans liquid, tradable stocks with sufficient volume and filters
based on configurable criteria (volume, price, volatility, market cap).
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ScannerConfig(BaseModel):
    """Scanner configuration parameters"""
    enabled: bool = True

    # Base universe to scan
    base_universe: str = "sp500"  # "sp500", "sp400", "sp600", "nasdaq100", "custom"
    custom_tickers: List[str] = []  # Used if base_universe = "custom"

    # Volume filters
    min_avg_volume: int = 1_000_000  # Minimum 20-day average volume
    min_dollar_volume: float = 10_000_000  # Minimum daily dollar volume ($10M)

    # Price filters
    min_price: float = 5.0  # Avoid penny stocks
    max_price: float = 1000.0  # Avoid extremely expensive stocks

    # Market cap filter (optional)
    min_market_cap: Optional[float] = None  # e.g., 1_000_000_000 for $1B+

    # Volatility filter (optional)
    min_atr_pct: Optional[float] = None  # e.g., 0.02 for 2%+ daily ATR
    max_atr_pct: Optional[float] = None  # e.g., 0.10 for <10% daily ATR

    # Universe size limits
    max_universe_size: int = 50  # Maximum symbols to analyze (cost control)

    # Scan frequency
    scan_interval_hours: int = 24  # Re-scan daily by default

    # Cache settings
    use_cache: bool = True
    cache_expiry_hours: int = 24


class ScanResult(BaseModel):
    """Result from market scan"""
    symbol: str
    price: float
    avg_volume_20d: float
    dollar_volume: float
    market_cap: Optional[float] = None
    atr_pct: Optional[float] = None
    sector: Optional[str] = None
    scan_timestamp: str


class MarketScanner:
    """
    Dynamically scans and filters tradable stocks based on liquidity,
    volume, and other configurable criteria.
    """

    def __init__(self, config: ScannerConfig):
        self.config = config
        self._cache: Dict[str, Tuple[List[str], datetime]] = {}
        logger.info(f"MarketScanner initialized with base_universe={config.base_universe}")

    def get_tradable_universe(self) -> List[str]:
        """
        Get list of tradable symbols based on scanner configuration.

        Returns:
            List of ticker symbols that pass all filters
        """
        if not self.config.enabled:
            logger.info("Scanner disabled, returning empty universe")
            return []

        # Check cache first
        if self.config.use_cache:
            cached_universe = self._get_cached_universe()
            if cached_universe:
                logger.info(f"Using cached universe with {len(cached_universe)} symbols")
                return cached_universe

        logger.info("Running market scan...")

        # Step 1: Get base universe
        base_tickers = self._get_base_universe()
        logger.info(f"Base universe: {len(base_tickers)} symbols from {self.config.base_universe}")

        # Step 2: Fetch and filter
        filtered_symbols = self._scan_and_filter(base_tickers)

        # Step 3: Limit to max size
        if len(filtered_symbols) > self.config.max_universe_size:
            logger.warning(
                f"Filtered universe ({len(filtered_symbols)}) exceeds max_universe_size "
                f"({self.config.max_universe_size}). Selecting top {self.config.max_universe_size} by dollar volume."
            )
            filtered_symbols = filtered_symbols[:self.config.max_universe_size]

        # Step 4: Cache results
        if self.config.use_cache:
            self._cache_universe(filtered_symbols)

        logger.info(f"Final tradable universe: {len(filtered_symbols)} symbols")
        return filtered_symbols

    def _get_base_universe(self) -> List[str]:
        """Get base list of tickers to scan"""
        if self.config.base_universe == "custom":
            return self.config.custom_tickers

        elif self.config.base_universe == "sp500":
            return self._get_sp500_tickers()

        elif self.config.base_universe == "nasdaq100":
            return self._get_nasdaq100_tickers()

        elif self.config.base_universe == "sp400":
            # S&P MidCap 400
            logger.warning("S&P 400 not yet implemented, using custom list")
            return self.config.custom_tickers

        elif self.config.base_universe == "sp600":
            # S&P SmallCap 600
            logger.warning("S&P 600 not yet implemented, using custom list")
            return self.config.custom_tickers

        else:
            logger.error(f"Unknown base_universe: {self.config.base_universe}")
            return []

    def _get_sp500_tickers(self) -> List[str]:
        """Get S&P 500 constituent tickers from Wikipedia"""
        try:
            # Wikipedia maintains an updated list
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()

            # Clean up tickers (some have dots instead of hyphens)
            tickers = [ticker.replace('.', '-') for ticker in tickers]

            logger.info(f"Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
            return tickers

        except Exception as e:
            logger.error(f"Failed to fetch S&P 500 tickers: {e}")
            # Fallback to a small subset
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "SPY", "QQQ"]

    def _get_nasdaq100_tickers(self) -> List[str]:
        """Get NASDAQ-100 constituent tickers"""
        try:
            url = "https://en.wikipedia.org/wiki/Nasdaq-100"
            tables = pd.read_html(url)
            nasdaq_table = tables[4]  # The constituents table
            tickers = nasdaq_table['Ticker'].tolist()

            logger.info(f"Fetched {len(tickers)} NASDAQ-100 tickers from Wikipedia")
            return tickers

        except Exception as e:
            logger.error(f"Failed to fetch NASDAQ-100 tickers: {e}")
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "QQQ"]

    def _scan_and_filter(self, tickers: List[str]) -> List[str]:
        """
        Scan tickers and filter by volume, price, and other criteria.

        Returns:
            List of symbols that pass all filters, sorted by dollar volume (descending)
        """
        scan_results: List[ScanResult] = []

        logger.info(f"Scanning {len(tickers)} symbols...")

        for i, symbol in enumerate(tickers):
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(tickers)} symbols scanned")

            try:
                result = self._scan_symbol(symbol)
                if result:
                    scan_results.append(result)

            except Exception as e:
                logger.debug(f"Failed to scan {symbol}: {e}")
                continue

        logger.info(f"Scan complete: {len(scan_results)}/{len(tickers)} symbols passed filters")

        # Sort by dollar volume (highest liquidity first)
        scan_results.sort(key=lambda x: x.dollar_volume, reverse=True)

        # Extract symbols
        filtered_symbols = [result.symbol for result in scan_results]

        return filtered_symbols

    def _scan_symbol(self, symbol: str) -> Optional[ScanResult]:
        """
        Scan individual symbol and check if it passes filters.

        Returns:
            ScanResult if symbol passes all filters, None otherwise
        """
        try:
            ticker = yf.Ticker(symbol)

            # Fetch recent data (20 days for volume average)
            hist = ticker.history(period="1mo", interval="1d")

            if hist.empty or len(hist) < 10:
                logger.debug(f"{symbol}: Insufficient data")
                return None

            # Get current price
            current_price = hist['Close'].iloc[-1]

            # Price filter
            if current_price < self.config.min_price or current_price > self.config.max_price:
                logger.debug(f"{symbol}: Price ${current_price:.2f} outside range")
                return None

            # Volume filters
            avg_volume = hist['Volume'].tail(20).mean()
            if avg_volume < self.config.min_avg_volume:
                logger.debug(f"{symbol}: Avg volume {avg_volume:,.0f} below minimum")
                return None

            dollar_volume = current_price * avg_volume
            if dollar_volume < self.config.min_dollar_volume:
                logger.debug(f"{symbol}: Dollar volume ${dollar_volume:,.0f} below minimum")
                return None

            # Market cap filter (optional)
            market_cap = None
            if self.config.min_market_cap:
                try:
                    info = ticker.info
                    market_cap = info.get('marketCap')
                    if market_cap and market_cap < self.config.min_market_cap:
                        logger.debug(f"{symbol}: Market cap ${market_cap:,.0f} below minimum")
                        return None
                except:
                    pass  # Skip if market cap not available

            # ATR/Volatility filter (optional)
            atr_pct = None
            if self.config.min_atr_pct or self.config.max_atr_pct:
                atr_pct = self._calculate_atr_pct(hist)
                if self.config.min_atr_pct and atr_pct < self.config.min_atr_pct:
                    logger.debug(f"{symbol}: ATR {atr_pct:.2%} below minimum")
                    return None
                if self.config.max_atr_pct and atr_pct > self.config.max_atr_pct:
                    logger.debug(f"{symbol}: ATR {atr_pct:.2%} above maximum")
                    return None

            # Get sector (optional)
            sector = None
            try:
                info = ticker.info
                sector = info.get('sector')
            except:
                pass

            # Symbol passes all filters
            return ScanResult(
                symbol=symbol,
                price=current_price,
                avg_volume_20d=avg_volume,
                dollar_volume=dollar_volume,
                market_cap=market_cap,
                atr_pct=atr_pct,
                sector=sector,
                scan_timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.debug(f"Error scanning {symbol}: {e}")
            return None

    def _calculate_atr_pct(self, hist: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR as percentage of price"""
        try:
            high = hist['High']
            low = hist['Low']
            close = hist['Close']

            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # ATR
            atr = tr.rolling(window=period).mean().iloc[-1]

            # ATR as percentage
            current_price = close.iloc[-1]
            atr_pct = atr / current_price

            return atr_pct

        except Exception as e:
            logger.debug(f"Failed to calculate ATR: {e}")
            return 0.0

    def _get_cached_universe(self) -> Optional[List[str]]:
        """Get cached universe if not expired"""
        cache_key = f"{self.config.base_universe}_{self.config.min_avg_volume}"

        if cache_key not in self._cache:
            return None

        cached_symbols, cached_time = self._cache[cache_key]

        # Check if cache expired
        expiry_time = timedelta(hours=self.config.cache_expiry_hours)
        if datetime.now() - cached_time > expiry_time:
            logger.info("Cache expired, will re-scan")
            return None

        return cached_symbols

    def _cache_universe(self, symbols: List[str]) -> None:
        """Cache universe with timestamp"""
        cache_key = f"{self.config.base_universe}_{self.config.min_avg_volume}"
        self._cache[cache_key] = (symbols, datetime.now())
        logger.info(f"Cached {len(symbols)} symbols")

    def get_scan_stats(self) -> Dict:
        """Get statistics about current universe"""
        universe = self.get_tradable_universe()

        if not universe:
            return {"universe_size": 0}

        # Fetch basic stats
        stats = {
            "universe_size": len(universe),
            "base_universe": self.config.base_universe,
            "filters": {
                "min_avg_volume": self.config.min_avg_volume,
                "min_price": self.config.min_price,
                "max_price": self.config.max_price,
            },
            "symbols": universe
        }

        return stats


def create_scanner_from_config(config_dict: Dict) -> MarketScanner:
    """
    Factory function to create scanner from configuration dictionary.

    Args:
        config_dict: Dictionary with scanner configuration

    Returns:
        Configured MarketScanner instance
    """
    scanner_config = ScannerConfig(**config_dict)
    return MarketScanner(scanner_config)
