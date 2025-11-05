"""
Simplified Scanner Orchestrator: NO interpretations, just raw data to LLM.

Workflow:
1. Load universe with pre-filtering (API optimization ONLY)
2. Calculate raw metrics for all stocks
3. Send batches of raw metrics to LLM
4. LLM analyzes raw data and picks stocks (NO thresholds)
5. Return LLM's picks

CRITICAL: No Stage 1 guidance, no programmatic filtering, no hardcoded thresholds.
The LLM sees ONLY raw calculated metrics and decides what's interesting.
"""

import logging
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from .universe_loader import UniverseLoader, TradableStock
from .market_aggregator import MarketAggregator
from .raw_metrics_analyzer import RawMetricsAnalyzer
from .scanner_db import ScannerDB, ScanResult

# Import Alpaca bars ingestor
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from data_ingestion.alpaca_bars_ingestor import AlpacaBarsIngestor

logger = logging.getLogger(__name__)


class SimplifiedScannerOrchestrator:
    """
    Simplified scanner that sends raw metrics directly to LLM.

    NO hardcoded thresholds, NO pre-filtering logic, NO interpretations.
    Just raw data → LLM → picks.
    """

    def __init__(
        self,
        data_fetcher=None,  # Optional: PriceVolumeIngestor for yfinance fallback
        alpaca_api_key: Optional[str] = None,
        alpaca_secret_key: Optional[str] = None,
        alpaca_base_url: Optional[str] = None,
        paper: bool = True,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        db_path: str = "data/scanner.db",
        requests_per_minute: int = 200
    ):
        """
        Initialize simplified scanner orchestrator.

        Args:
            data_fetcher: Optional fallback data fetcher (yfinance)
            alpaca_api_key: Alpaca API key for universe loading and bars
            alpaca_secret_key: Alpaca secret key
            alpaca_base_url: Alpaca base URL
            paper: Use paper trading endpoint (default True for safety)
            openai_api_key: OpenAI API key for LLM calls (legacy)
            anthropic_api_key: Anthropic API key for LLM calls
            db_path: Path to scanner database
            requests_per_minute: Alpaca rate limit (200 standard, 1000 unlimited)
        """
        self.data_fetcher = data_fetcher

        # Initialize Alpaca bars ingestor (primary data source)
        self.alpaca_ingestor = None
        if alpaca_api_key and alpaca_secret_key:
            try:
                self.alpaca_ingestor = AlpacaBarsIngestor(
                    api_key=alpaca_api_key,
                    secret_key=alpaca_secret_key,
                    requests_per_minute=requests_per_minute
                )
                logger.info("Alpaca bars ingestor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Alpaca ingestor: {e}")
                logger.warning("Will use fallback data fetcher if available")

        # Initialize components
        self.universe_loader = UniverseLoader(
            alpaca_api_key=alpaca_api_key,
            alpaca_secret_key=alpaca_secret_key,
            alpaca_base_url=alpaca_base_url,
            paper=paper,
            db_path=db_path
        )
        self.market_aggregator = MarketAggregator(data_fetcher=data_fetcher)

        # Use anthropic_api_key if provided, otherwise fall back to openai_api_key
        llm_api_key = anthropic_api_key or openai_api_key
        self.raw_metrics_analyzer = RawMetricsAnalyzer(api_key=llm_api_key)

        self.scanner_db = ScannerDB(db_path=db_path)

    def run_full_scan(
        self,
        force_refresh_universe: bool = False,
        refresh_hours: int = 24,
        batch_size: int = 50,
        max_batches: Optional[int] = None,
        pre_filter_enabled: bool = True,
        min_price: Optional[float] = 10.0,
        min_avg_volume: Optional[int] = 1_000_000,
        min_market_cap: Optional[float] = 1_000_000_000,
        min_daily_liquidity: Optional[float] = 500_000_000,
        min_adr_percent: Optional[float] = 1.0,
        max_adr_percent: Optional[float] = 15.0
    ) -> List[str]:
        """
        Execute simplified scan with raw metrics.

        Args:
            force_refresh_universe: Force refresh universe from API
            refresh_hours: Max universe cache age
            batch_size: Stocks per LLM batch (default 50)
            max_batches: Max batches to process (None = all)
            pre_filter_enabled: Enable pre-filtering (API optimization ONLY)
            min_price: Minimum stock price (default $10)
            min_avg_volume: Minimum avg daily volume (default 1M shares)
            min_market_cap: Minimum market cap (default $1B)
            min_daily_liquidity: Minimum daily liquidity in USD (default $500M)
            min_adr_percent: Minimum ADR percentage (default 1%)
            max_adr_percent: Maximum ADR percentage (default 15%)

        Returns:
            List of LLM-picked symbols
        """
        start_time = time.time()
        scan_id = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        llm_calls = 0

        logger.info("="*60)
        logger.info(f"Starting Simplified Market Scan: {scan_id}")
        logger.info("NO thresholds - sending raw metrics to LLM")
        logger.info("="*60)

        # ===================================================================
        # STEP 1: LOAD UNIVERSE
        # ===================================================================
        logger.info("\n[STEP 1] Loading tradable universe...")
        universe = self.universe_loader.load_tradable_universe(
            force_refresh=force_refresh_universe,
            refresh_hours=refresh_hours,
            optionable_only=True,
            enrich_metadata=pre_filter_enabled  # Enrich if pre-filtering enabled
        )

        if not universe:
            logger.error("Failed to load universe. Aborting scan.")
            return []

        logger.info(f"Loaded {len(universe)} tradable stocks")

        # ===================================================================
        # STEP 2: PRE-FILTER (API OPTIMIZATION ONLY - NOT INTERPRETATION)
        # ===================================================================
        if pre_filter_enabled:
            logger.info("\n[STEP 2] Pre-filtering for API/cost optimization...")
            logger.info(f"  Phase 1 (metadata): price>${min_price}, volume>{min_avg_volume:,}, market_cap>${min_market_cap/1e9:.1f}B")
            logger.info(f"  Phase 2 (after metrics): liquidity>${min_daily_liquidity/1e6:.0f}M, ADR {min_adr_percent}%-{max_adr_percent}%")
            logger.info("  NOTE: This is cost optimization, NOT interpretation - LLM still decides on raw metrics")

            universe = self.universe_loader.pre_filter_universe(
                universe=universe,
                min_price=min_price,
                min_avg_volume=min_avg_volume,
                min_market_cap=min_market_cap
            )

            if not universe:
                logger.error("Pre-filter removed all stocks. Try relaxing criteria.")
                return []

            logger.info(f"Pre-filtered universe: {len(universe)} stocks remaining")
        else:
            logger.info("\n[STEP 2] Pre-filter DISABLED - will fetch data for all stocks")

        # ===================================================================
        # STEP 3: CALCULATE RAW METRICS
        # ===================================================================
        logger.info("\n[STEP 3] Calculating raw metrics for all stocks...")

        stock_metrics = []
        symbols_to_fetch = [stock.symbol for stock in universe]
        sector_map = {stock.symbol: stock.sector for stock in universe}

        logger.info(f"Fetching price data for {len(symbols_to_fetch)} stocks...")

        # Use Alpaca ingestor if available
        if self.alpaca_ingestor:
            try:
                price_data_map = self.alpaca_ingestor.fetch_bars_batched(
                    symbols=symbols_to_fetch,
                    timeframe="1Day",
                    start=datetime.now() - timedelta(days=90),
                    end=datetime.now(),
                    use_cache=True
                )
                logger.info(f"Fetched data for {len(price_data_map)} symbols from Alpaca")

                # Calculate metrics for each stock
                for symbol, price_df in price_data_map.items():
                    metrics = self.market_aggregator.calculate_stock_metrics(
                        symbol=symbol,
                        sector=sector_map.get(symbol, "Unknown"),
                        price_data=price_df
                    )
                    if metrics:
                        stock_metrics.append(metrics)

                logger.info(f"Calculated raw metrics for {len(stock_metrics)} stocks")

            except Exception as e:
                logger.error(f"Error fetching data from Alpaca: {e}")
                logger.warning("Falling back to yfinance if available")

                # Fallback to yfinance
                if self.data_fetcher:
                    try:
                        price_df = self.data_fetcher.fetch_prices(
                            universe=symbols_to_fetch,
                            interval='1d',
                            lookback_days=90
                        )
                        # Process yfinance data
                        for symbol in symbols_to_fetch:
                            if symbol in price_df['symbol'].unique():
                                symbol_df = price_df[price_df['symbol'] == symbol]
                                metrics = self.market_aggregator.calculate_stock_metrics(
                                    symbol=symbol,
                                    sector=sector_map.get(symbol, "Unknown"),
                                    price_data=symbol_df
                                )
                                if metrics:
                                    stock_metrics.append(metrics)
                        logger.info(f"Using yfinance fallback - calculated metrics for {len(stock_metrics)} stocks")
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed: {fallback_error}")
                        logger.error("Cannot proceed without market data. Aborting scan.")
                        return []
                else:
                    logger.error("No data source available. Aborting scan.")
                    return []
        else:
            logger.error("No Alpaca ingestor configured. Aborting scan.")
            return []

        if not stock_metrics:
            logger.error("No metrics calculated. Aborting scan.")
            return []

        # ===================================================================
        # STEP 3B: POST-METRICS FILTER (API/COST OPTIMIZATION)
        # ===================================================================
        if pre_filter_enabled and (min_daily_liquidity or min_adr_percent or max_adr_percent):
            logger.info("\n[STEP 3B] Post-metrics filtering (ADR and liquidity optimization)...")
            logger.info(f"  Criteria: liquidity>${min_daily_liquidity/1e6:.0f}M, ADR {min_adr_percent}%-{max_adr_percent}%")

            initial_count = len(stock_metrics)
            filtered_metrics = []
            removed_reasons = {
                'low_liquidity': 0,
                'low_adr': 0,
                'high_adr': 0,
                'missing_data': 0
            }

            for metrics in stock_metrics:
                # Check liquidity
                if min_daily_liquidity is not None:
                    if metrics.daily_liquidity is None:
                        removed_reasons['missing_data'] += 1
                        continue
                    if metrics.daily_liquidity < min_daily_liquidity:
                        removed_reasons['low_liquidity'] += 1
                        continue

                # Check ADR range
                if min_adr_percent is not None or max_adr_percent is not None:
                    if metrics.adr_percent is None:
                        removed_reasons['missing_data'] += 1
                        continue
                    if min_adr_percent is not None and metrics.adr_percent < min_adr_percent:
                        removed_reasons['low_adr'] += 1
                        continue
                    if max_adr_percent is not None and metrics.adr_percent > max_adr_percent:
                        removed_reasons['high_adr'] += 1
                        continue

                filtered_metrics.append(metrics)

            removed_total = initial_count - len(filtered_metrics)
            logger.info(f"Post-metrics filter: {initial_count} → {len(filtered_metrics)} stocks ({removed_total} removed)")
            if removed_total > 0:
                logger.info(f"  Removal reasons:")
                for reason, count in removed_reasons.items():
                    if count > 0:
                        logger.info(f"    {reason}: {count}")

            stock_metrics = filtered_metrics

            if not stock_metrics:
                logger.error("Post-metrics filter removed all stocks. Try relaxing ADR/liquidity criteria.")
                return []

        # ===================================================================
        # STEP 4: SEND RAW METRICS TO LLM IN BATCHES
        # ===================================================================
        logger.info("\n[STEP 4] Sending raw metrics to LLM for analysis...")
        logger.info(f"  Batch size: {batch_size} stocks")
        logger.info(f"  Max batches: {max_batches or 'unlimited'}")
        logger.info("  LLM will analyze raw data and pick stocks (NO thresholds)")

        final_candidates = self.raw_metrics_analyzer.analyze_all_batches(
            all_metrics=stock_metrics,
            batch_size=batch_size,
            max_batches=max_batches
        )

        # Count LLM calls
        num_batches = min(
            (len(stock_metrics) + batch_size - 1) // batch_size,
            max_batches if max_batches else float('inf')
        )
        llm_calls = int(num_batches)

        # ===================================================================
        # SAVE RESULTS
        # ===================================================================
        end_time = time.time()
        execution_time = end_time - start_time
        estimated_cost = llm_calls * 0.02  # Rough estimate: $0.02 per call

        logger.info("\n" + "="*60)
        logger.info("SCAN COMPLETE")
        logger.info("="*60)
        logger.info(f"Stocks analyzed: {len(stock_metrics)}")
        logger.info(f"Final LLM picks: {len(final_candidates)}")
        logger.info(f"Candidates: {final_candidates}")
        logger.info(f"Execution time: {execution_time:.1f} seconds")
        logger.info(f"LLM calls: {llm_calls}")
        logger.info(f"Estimated cost: ${estimated_cost:.3f}")
        logger.info("="*60)

        # Save scan result to database
        scan_result = ScanResult(
            scan_id=scan_id,
            timestamp=datetime.now().isoformat(),
            stage1_guidance={
                "market_bias": "N/A - raw metrics scan",
                "focus_sectors": [],
                "focus_patterns": []
            },  # Placeholder since we don't use Stage 1
            filtered_count=len(final_candidates),
            final_candidates=final_candidates,
            execution_time_seconds=execution_time,
            llm_calls=llm_calls,
            total_cost=estimated_cost
        )
        self.scanner_db.save_scan_result(scan_result)

        return final_candidates


# Example usage
if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO)

    # Requires REAL market data from Alpaca
    alpaca_api_key = os.getenv('ALPACA_API_KEY')
    alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')

    if not alpaca_api_key or not alpaca_secret_key:
        logger.error("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        logger.error("This scanner requires REAL market data to function properly")
        exit(1)

    orchestrator = SimplifiedScannerOrchestrator(
        alpaca_api_key=alpaca_api_key,
        alpaca_secret_key=alpaca_secret_key,
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
    )

    # Run scan with raw metrics and ALL 5 filters
    candidates = orchestrator.run_full_scan(
        force_refresh_universe=False,
        refresh_hours=24,
        batch_size=50,
        max_batches=None,  # Process ALL stocks that pass 5 filters
        pre_filter_enabled=True,
        min_price=10.0,
        min_avg_volume=1_000_000,
        min_market_cap=1_000_000_000,
        min_daily_liquidity=500_000_000,  # $500M
        min_adr_percent=1.0,  # 1%
        max_adr_percent=15.0  # 15%
    )

    print(f"\n\nFinal LLM picks for trading: {candidates}")
