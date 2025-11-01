"""
Universe Loader: Fetch and cache tradable stock universe from Alpaca API.

Loads 5000+ optionable stocks and caches in database to minimize API calls.
"""

import logging
from typing import List, Optional, Dict
from datetime import datetime
import os

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

from .scanner_db import ScannerDB, TradableStock

logger = logging.getLogger(__name__)


class UniverseLoader:
    """
    Manages loading and caching of tradable stock universe.

    Uses Alpaca API to fetch all optionable US stocks and caches them
    in the database to avoid repeated API calls.
    """

    def __init__(
        self,
        alpaca_api_key: Optional[str] = None,
        alpaca_secret_key: Optional[str] = None,
        alpaca_base_url: Optional[str] = None,
        db_path: str = "data/scanner.db"
    ):
        """
        Initialize universe loader.

        Args:
            alpaca_api_key: Alpaca API key (or uses ALPACA_API_KEY env var)
            alpaca_secret_key: Alpaca secret key (or uses ALPACA_SECRET_KEY env var)
            alpaca_base_url: Alpaca base URL (or uses ALPACA_BASE_URL env var)
            db_path: Path to scanner database
        """
        self.api_key = alpaca_api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = alpaca_secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.base_url = alpaca_base_url or os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

        self.db = ScannerDB(db_path=db_path)
        self.trading_client = None

        # Initialize Alpaca client if credentials available
        if self.api_key and self.secret_key:
            try:
                self.trading_client = TradingClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                    paper=True  # Always use paper for safety
                )
                logger.info("Alpaca trading client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Alpaca client: {e}")
                self.trading_client = None

    def load_tradable_universe(
        self,
        force_refresh: bool = False,
        refresh_hours: int = 24,
        optionable_only: bool = True
    ) -> List[TradableStock]:
        """
        Load tradable universe from cache or Alpaca API.

        Args:
            force_refresh: Force refresh from API regardless of cache age
            refresh_hours: Max cache age in hours before refresh
            optionable_only: Only return optionable stocks

        Returns:
            List of TradableStock objects
        """
        # Check cache age
        cache_age = self.db.get_universe_age_hours()

        if not force_refresh and cache_age is not None and cache_age < refresh_hours:
            logger.info(f"Using cached universe (age: {cache_age:.1f} hours)")
            return self.db.get_tradable_universe(optionable_only=optionable_only)

        # Refresh from API
        logger.info("Refreshing universe from Alpaca API...")

        if not self.trading_client:
            logger.error("Alpaca client not initialized. Cannot refresh universe.")
            # Fall back to cache if available
            cached = self.db.get_tradable_universe(optionable_only=optionable_only)
            if cached:
                logger.warning(f"Using stale cache ({cache_age:.1f} hours old)")
                return cached
            return []

        try:
            # Fetch all active US stocks
            search_params = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE
            )

            assets = self.trading_client.get_all_assets(search_params)
            logger.info(f"Fetched {len(assets)} assets from Alpaca")

            # Convert to TradableStock objects
            stocks = []
            for asset in assets:
                # Skip if optionable filter is on and stock is not optionable
                if optionable_only and not asset.options_trading:
                    continue

                # Skip if not tradable or fractionable (low liquidity indicators)
                if not asset.tradable or not asset.fractionable:
                    continue

                stock = TradableStock(
                    symbol=asset.symbol,
                    sector=asset.name if hasattr(asset, 'name') else "Unknown",  # Will update with proper sector later
                    market_cap=None,  # Not provided by assets endpoint
                    avg_volume_20d=None,  # Will be calculated later if needed
                    last_price=None,  # Will be fetched separately if needed
                    optionable=asset.options_trading if hasattr(asset, 'options_trading') else False,
                    updated_at=datetime.now().isoformat()
                )
                stocks.append(stock)

            logger.info(f"Filtered to {len(stocks)} tradable stocks")

            # Cache in database
            self.db.save_tradable_universe(stocks)

            return stocks

        except Exception as e:
            logger.error(f"Error refreshing universe from Alpaca: {e}")
            # Fall back to cache
            cached = self.db.get_tradable_universe(optionable_only=optionable_only)
            if cached:
                logger.warning(f"Using stale cache due to API error")
                return cached
            return []

    def get_sector_breakdown(self, universe: List[TradableStock]) -> Dict[str, List[str]]:
        """
        Group universe by sector.

        Args:
            universe: List of TradableStock objects

        Returns:
            Dictionary mapping sector -> list of symbols
        """
        sectors = {}
        for stock in universe:
            sector = stock.sector or "Unknown"
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(stock.symbol)

        return sectors

    def get_symbols_by_sector(self, sector: str, universe: List[TradableStock]) -> List[str]:
        """
        Get all symbols in a specific sector.

        Args:
            sector: Sector name
            universe: List of TradableStock objects

        Returns:
            List of symbols in that sector
        """
        return [stock.symbol for stock in universe if stock.sector == sector]

    def refresh_universe_metadata(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch additional metadata for symbols (sectors, market cap, etc.)

        This is a placeholder for enriching stock data with real sector information.
        In production, you would use Alpaca's market data API or another data source.

        Args:
            symbols: List of symbols to enrich

        Returns:
            Dictionary mapping symbol -> metadata dict
        """
        # TODO: Implement using Alpaca data API or yfinance as fallback
        # For now, return empty dict
        logger.warning("refresh_universe_metadata not fully implemented")
        return {}


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    loader = UniverseLoader()

    # Load universe (will use cache if fresh)
    universe = loader.load_tradable_universe(refresh_hours=24)
    print(f"\nLoaded {len(universe)} tradable stocks")

    # Get sector breakdown
    sectors = loader.get_sector_breakdown(universe)
    print(f"\nSectors found: {len(sectors)}")
    for sector, symbols in list(sectors.items())[:5]:
        print(f"  {sector}: {len(symbols)} stocks")

    # Sample stocks
    print(f"\nSample stocks:")
    for stock in universe[:5]:
        print(f"  {stock.symbol}: {stock.sector} (optionable={stock.optionable})")
