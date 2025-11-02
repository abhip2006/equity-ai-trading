"""
Universe Loader: Fetch and cache tradable stock universe from Alpaca API.

Loads 5000+ optionable stocks and caches in database to minimize API calls.
"""

import logging
from typing import List, Optional, Dict
from datetime import datetime
import os
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

from .scanner_db import ScannerDB, TradableStock

logger = logging.getLogger(__name__)


# GICS Sector mapping to standardized categories
SECTOR_MAPPING = {
    'Technology': 'Technology',
    'Communication Services': 'Communication',
    'Consumer Cyclical': 'Consumer Discretionary',
    'Consumer Defensive': 'Consumer Staples',
    'Financial Services': 'Financials',
    'Healthcare': 'Healthcare',
    'Industrials': 'Industrials',
    'Basic Materials': 'Materials',
    'Energy': 'Energy',
    'Real Estate': 'Real Estate',
    'Utilities': 'Utilities',
    'Financial': 'Financials',  # Alternative naming
}


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
        paper: bool = True,
        db_path: str = "data/scanner.db"
    ):
        """
        Initialize universe loader.

        Args:
            alpaca_api_key: Alpaca API key (or uses ALPACA_API_KEY env var)
            alpaca_secret_key: Alpaca secret key (or uses ALPACA_SECRET_KEY env var)
            alpaca_base_url: Alpaca base URL (or uses ALPACA_BASE_URL env var)
            paper: Use paper trading endpoint (default True for safety)
            db_path: Path to scanner database
        """
        self.api_key = alpaca_api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = alpaca_secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.base_url = alpaca_base_url or os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        self.paper = paper

        self.db = ScannerDB(db_path=db_path)
        self.trading_client = None

        # Initialize Alpaca client if credentials available
        if self.api_key and self.secret_key:
            try:
                self.trading_client = TradingClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                    paper=self.paper
                )
                mode = "PAPER" if self.paper else "LIVE"
                logger.info(f"Alpaca trading client initialized ({mode} mode)")
            except Exception as e:
                logger.warning(f"Failed to initialize Alpaca client: {e}")
                self.trading_client = None

    def load_tradable_universe(
        self,
        force_refresh: bool = False,
        refresh_hours: int = 24,
        optionable_only: bool = True,
        enrich_metadata: bool = False
    ) -> List[TradableStock]:
        """
        Load tradable universe from cache or Alpaca API.

        Args:
            force_refresh: Force refresh from API regardless of cache age
            refresh_hours: Max cache age in hours before refresh
            optionable_only: Only return optionable stocks
            enrich_metadata: Fetch real sector, market cap, volume, price via yfinance

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
                    sector="Unknown",  # Will enrich with real sector if requested
                    market_cap=None,  # Not provided by assets endpoint
                    avg_volume_20d=None,  # Will enrich if requested
                    last_price=None,  # Will enrich if requested
                    optionable=asset.options_trading if hasattr(asset, 'options_trading') else False,
                    updated_at=datetime.now().isoformat()
                )
                stocks.append(stock)

            logger.info(f"Filtered to {len(stocks)} tradable stocks")

            # Enrich with metadata if requested
            if enrich_metadata and stocks:
                logger.info("Enriching universe with metadata (sector, market cap, volume, price)...")
                symbols = [stock.symbol for stock in stocks]
                metadata = self.refresh_universe_metadata(symbols)

                # Merge enriched data
                enriched_count = 0
                for stock in stocks:
                    if stock.symbol in metadata:
                        stock_meta = metadata[stock.symbol]
                        stock.sector = stock_meta.get('sector', stock.sector)
                        stock.market_cap = stock_meta.get('market_cap', stock.market_cap)
                        stock.avg_volume_20d = stock_meta.get('avg_volume', stock.avg_volume_20d)
                        stock.last_price = stock_meta.get('last_price', stock.last_price)
                        enriched_count += 1

                logger.info(f"Enriched {enriched_count}/{len(stocks)} stocks ({enriched_count/len(stocks)*100:.1f}%)")

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
        unknown_count = 0

        for stock in universe:
            sector = stock.sector or "Unknown"
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(stock.symbol)

            if sector == "Unknown":
                unknown_count += 1

        # Warn if significant portion has unknown sector
        if unknown_count > 0:
            unknown_pct = unknown_count / len(universe) * 100 if universe else 0
            if unknown_pct > 10:
                logger.warning(
                    f"Sector data quality issue: {unknown_count}/{len(universe)} stocks ({unknown_pct:.1f}%) "
                    f"have 'Unknown' sector. Consider running with enrich_metadata=True to fetch real sectors."
                )

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

    def refresh_universe_metadata(self, symbols: List[str], batch_size: int = 50) -> Dict[str, Dict]:
        """
        Fetch additional metadata for symbols via yfinance.

        Fetches: sector, market cap, average volume, last price

        Args:
            symbols: List of symbols to enrich
            batch_size: Number of symbols to fetch per batch (default 50)

        Returns:
            Dictionary mapping symbol -> metadata dict with keys:
            - sector: Standardized sector name
            - market_cap: Market capitalization in USD
            - avg_volume: Average daily volume (20-day)
            - last_price: Most recent close price
        """
        logger.info(f"Fetching metadata for {len(symbols)} symbols via yfinance...")

        metadata = {}
        total_batches = (len(symbols) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(symbols), batch_size):
            batch = symbols[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} symbols)")

            try:
                # Fetch batch data using yfinance Tickers
                tickers = yf.Tickers(' '.join(batch))

                for symbol in batch:
                    try:
                        ticker = tickers.tickers.get(symbol)
                        if not ticker:
                            continue

                        info = ticker.info

                        # Extract sector and normalize
                        raw_sector = info.get('sector', 'Unknown')
                        normalized_sector = SECTOR_MAPPING.get(raw_sector, raw_sector)

                        metadata[symbol] = {
                            'sector': normalized_sector,
                            'market_cap': info.get('marketCap'),
                            'avg_volume': info.get('averageVolume') or info.get('volume'),
                            'last_price': info.get('regularMarketPrice') or info.get('currentPrice')
                        }

                    except Exception as e:
                        logger.debug(f"Failed to fetch metadata for {symbol}: {e}")
                        continue

                # Rate limiting - be respectful to yfinance
                if batch_idx + batch_size < len(symbols):
                    time.sleep(1)

            except Exception as e:
                logger.warning(f"Error fetching batch {batch_num}: {e}")
                continue

        success_rate = len(metadata) / len(symbols) * 100 if symbols else 0
        logger.info(f"Successfully fetched metadata for {len(metadata)}/{len(symbols)} symbols ({success_rate:.1f}%)")

        return metadata


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
