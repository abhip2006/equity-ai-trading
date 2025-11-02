"""
Macro Data Ingestor: Fetches raw macro-economic data from yfinance.

Fetches volatility indices, treasury yields, commodities, and currency data.
Returns 100% raw numerical values - NO interpretations.
"""

import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import yfinance as yf

from active_trader_llm.feature_engineering.models import MacroSnapshot
from active_trader_llm.data_ingestion.nyse_breadth_ingestor import NYSEBreadthIngestor

logger = logging.getLogger(__name__)


class MacroDataIngestor:
    """
    Fetches macro-economic data from yfinance.

    Returns raw numerical values only - LLM provides all interpretation.
    Implements 5-minute caching to avoid rate limits.
    """

    # yfinance symbols for macro data
    SYMBOLS = {
        # Volatility indices
        'vix': '^VIX',           # S&P 500 volatility
        'vxn': '^VXN',           # Nasdaq-100 volatility
        'move': 'MOVE',          # Treasury volatility

        # Treasury yields
        'treasury_10y': '^TNX',  # 10-year yield
        'treasury_2y': '^IRX',   # 2-year (13-week) yield
        'treasury_30y': '^TYX',  # 30-year yield

        # Commodities
        'gold': 'GC=F',          # Gold futures
        'oil': 'CL=F',           # Crude oil futures

        # Currency
        'dollar': 'DX-Y.NYB',    # US Dollar Index
    }

    def __init__(self, cache_duration_seconds: int = 300):
        """
        Initialize macro data ingestor.

        Args:
            cache_duration_seconds: Cache duration in seconds (default 5 minutes)
        """
        self.cache_duration = timedelta(seconds=cache_duration_seconds)
        self._cache: Optional[MacroSnapshot] = None
        self._cache_timestamp: Optional[datetime] = None

        # Initialize NYSE breadth ingestor (24-hour cache for EOD data)
        self.breadth_ingestor = NYSEBreadthIngestor(
            cache_dir="data/breadth_cache",
            cache_duration_hours=24
        )

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if self._cache is None or self._cache_timestamp is None:
            return False
        return datetime.now() - self._cache_timestamp < self.cache_duration

    def _fetch_single_value(self, symbol: str, field_name: str) -> Optional[float]:
        """
        Fetch current value for a single symbol.

        Args:
            symbol: yfinance symbol (e.g., '^VIX')
            field_name: Human-readable field name for logging

        Returns:
            Current close price or None if fetch fails
        """
        try:
            ticker = yf.Ticker(symbol)
            # Get most recent data (1 day)
            hist = ticker.history(period="1d")

            if hist.empty:
                logger.warning(f"{field_name} ({symbol}): No data available")
                return None

            value = float(hist['Close'].iloc[-1])
            logger.debug(f"{field_name} ({symbol}): {value}")
            return value

        except Exception as e:
            logger.warning(f"Failed to fetch {field_name} ({symbol}): {e}")
            return None

    def fetch_volatility_indices(self) -> Dict[str, Optional[float]]:
        """
        Fetch volatility indices: VIX, VXN, MOVE.

        Returns:
            Dict with raw index values (NO interpretations)
        """
        logger.debug("Fetching volatility indices...")

        return {
            'vix': self._fetch_single_value(self.SYMBOLS['vix'], 'VIX'),
            'vxn': self._fetch_single_value(self.SYMBOLS['vxn'], 'VXN'),
            'move_index': self._fetch_single_value(self.SYMBOLS['move'], 'MOVE'),
        }

    def fetch_treasury_yields(self) -> Dict[str, Optional[float]]:
        """
        Fetch treasury yields and calculate spread.

        Returns:
            Dict with raw yield percentages and spread (simple subtraction)
        """
        logger.debug("Fetching treasury yields...")

        treasury_10y = self._fetch_single_value(self.SYMBOLS['treasury_10y'], '10Y Treasury')
        treasury_2y = self._fetch_single_value(self.SYMBOLS['treasury_2y'], '2Y Treasury')
        treasury_30y = self._fetch_single_value(self.SYMBOLS['treasury_30y'], '30Y Treasury')

        # Calculate yield curve spread (simple subtraction)
        yield_curve_spread = None
        if treasury_10y is not None and treasury_2y is not None:
            yield_curve_spread = treasury_10y - treasury_2y
            logger.debug(f"Yield Curve Spread (10Y-2Y): {yield_curve_spread:.2f}%")

        return {
            'treasury_10y': treasury_10y,
            'treasury_2y': treasury_2y,
            'treasury_30y': treasury_30y,
            'yield_curve_spread': yield_curve_spread,
        }

    def fetch_commodities(self) -> Dict[str, Optional[float]]:
        """
        Fetch commodity prices: Gold, Oil.

        Returns:
            Dict with raw prices (NO interpretations)
        """
        logger.debug("Fetching commodity prices...")

        return {
            'gold_price': self._fetch_single_value(self.SYMBOLS['gold'], 'Gold'),
            'oil_price': self._fetch_single_value(self.SYMBOLS['oil'], 'Crude Oil'),
        }

    def fetch_currency(self) -> Dict[str, Optional[float]]:
        """
        Fetch currency index: US Dollar Index (DXY).

        Returns:
            Dict with raw index value (NO interpretation)
        """
        logger.debug("Fetching currency data...")

        return {
            'dollar_index': self._fetch_single_value(self.SYMBOLS['dollar'], 'US Dollar Index'),
        }

    def fetch_nyse_breadth(self) -> Dict[str, Optional[int]]:
        """
        Fetch NYSE market breadth data from Unicorn Research.

        Returns:
            Dict with raw breadth counts and ratios (NO interpretations)
        """
        logger.debug("Fetching NYSE breadth data...")

        # Fetch from NYSE breadth ingestor
        breadth_data = self.breadth_ingestor.fetch_latest_breadth(use_cache=True)

        return breadth_data

    def fetch_all(self, use_cache: bool = True) -> MacroSnapshot:
        """
        Fetch all macro data and return MacroSnapshot.

        Args:
            use_cache: Use cached data if available (default True)

        Returns:
            MacroSnapshot with all available data (Optional fields may be None)
        """
        # Check cache
        if use_cache and self._is_cache_valid():
            logger.info("Using cached macro data")
            return self._cache

        logger.info("Fetching fresh macro data from yfinance and NYSE breadth...")

        # Fetch all data
        volatility = self.fetch_volatility_indices()
        yields = self.fetch_treasury_yields()
        commodities = self.fetch_commodities()
        currency = self.fetch_currency()
        breadth = self.fetch_nyse_breadth()

        # Combine into MacroSnapshot
        snapshot = MacroSnapshot(
            timestamp=datetime.now().isoformat(),
            # Volatility
            vix=volatility.get('vix'),
            vxn=volatility.get('vxn'),
            move_index=volatility.get('move_index'),
            # Yields
            treasury_10y=yields.get('treasury_10y'),
            treasury_2y=yields.get('treasury_2y'),
            treasury_30y=yields.get('treasury_30y'),
            yield_curve_spread=yields.get('yield_curve_spread'),
            # Commodities
            gold_price=commodities.get('gold_price'),
            oil_price=commodities.get('oil_price'),
            # Currency
            dollar_index=currency.get('dollar_index'),
            # NYSE Breadth
            nyse_advancing=breadth.get('nyse_advancing'),
            nyse_declining=breadth.get('nyse_declining'),
            nyse_unchanged=breadth.get('nyse_unchanged'),
            nyse_advancing_volume=breadth.get('nyse_advancing_volume'),
            nyse_declining_volume=breadth.get('nyse_declining_volume'),
            nyse_new_highs=breadth.get('nyse_new_highs'),
            nyse_new_lows=breadth.get('nyse_new_lows'),
            advance_decline_ratio=breadth.get('advance_decline_ratio'),
            up_volume_ratio=breadth.get('up_volume_ratio'),
        )

        # Update cache
        self._cache = snapshot
        self._cache_timestamp = datetime.now()

        # Log summary
        macro_fields = sum(1 for field in [
            snapshot.vix, snapshot.vxn, snapshot.move_index,
            snapshot.treasury_10y, snapshot.treasury_2y, snapshot.treasury_30y,
            snapshot.gold_price, snapshot.oil_price, snapshot.dollar_index
        ] if field is not None)

        breadth_fields = sum(1 for field in [
            snapshot.nyse_advancing, snapshot.nyse_declining, snapshot.nyse_unchanged,
            snapshot.nyse_advancing_volume, snapshot.nyse_declining_volume,
            snapshot.nyse_new_highs, snapshot.nyse_new_lows,
            snapshot.advance_decline_ratio, snapshot.up_volume_ratio
        ] if field is not None)

        logger.info(f"Macro data fetched: {macro_fields}/9 macro fields, {breadth_fields}/9 breadth fields available")

        return snapshot


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ingestor = MacroDataIngestor()
    snapshot = ingestor.fetch_all()

    print("\n=== Macro Snapshot ===")
    print(f"Timestamp: {snapshot.timestamp}")
    print(f"\nVolatility:")
    print(f"  VIX: {snapshot.vix}")
    print(f"  VXN: {snapshot.vxn}")
    print(f"  MOVE: {snapshot.move_index}")
    print(f"\nTreasury Yields:")
    print(f"  10Y: {snapshot.treasury_10y}%")
    print(f"  2Y: {snapshot.treasury_2y}%")
    print(f"  30Y: {snapshot.treasury_30y}%")
    print(f"  Curve Spread (10Y-2Y): {snapshot.yield_curve_spread}%")
    print(f"\nCommodities:")
    print(f"  Gold: ${snapshot.gold_price}")
    print(f"  Oil: ${snapshot.oil_price}")
    print(f"\nCurrency:")
    print(f"  Dollar Index: {snapshot.dollar_index}")
