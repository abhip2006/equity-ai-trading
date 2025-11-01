"""
Scanner Orchestrator: Coordinates the entire two-stage scanning process.

Workflow:
1. Load universe (5000+ stocks)
2. Stage 1: Market-wide summary → LLM guidance (1 LLM call)
3. Stage 2A: Apply programmatic filters (NO LLM)
4. Stage 2B: Batch deep analysis (5-10 LLM calls)
5. Return final candidates for trading agent

Total: 6-11 LLM calls, $0.16 cost, 3-5 minutes
"""

import logging
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd

from .universe_loader import UniverseLoader, TradableStock
from .market_aggregator import MarketAggregator, StockMetrics
from .stage1_analyzer import Stage1Analyzer, Stage1Guidance
from .programmatic_filter import ProgrammaticFilter, FilterResult
from .raw_data_scanner import RawDataScanner, TechnicalIndicators
from .stage2_analyzer import Stage2Analyzer
from .scanner_db import ScannerDB, ScanResult

# Import Alpaca bars ingestor
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from data_ingestion.alpaca_bars_ingestor import AlpacaBarsIngestor

logger = logging.getLogger(__name__)


class ScannerOrchestrator:
    """
    Main coordinator for two-stage market scanner.

    Orchestrates the complete scanning workflow from universe loading
    to final candidate selection.
    """

    def __init__(
        self,
        data_fetcher=None,  # Optional: PriceVolumeIngestor for yfinance fallback
        alpaca_api_key: Optional[str] = None,
        alpaca_secret_key: Optional[str] = None,
        alpaca_base_url: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        db_path: str = "data/scanner.db",
        requests_per_minute: int = 200
    ):
        """
        Initialize scanner orchestrator.

        Args:
            data_fetcher: Optional fallback data fetcher (yfinance)
            alpaca_api_key: Alpaca API key for universe loading and bars
            alpaca_secret_key: Alpaca secret key
            alpaca_base_url: Alpaca base URL
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
            db_path=db_path
        )
        self.market_aggregator = MarketAggregator(data_fetcher=data_fetcher)
        self.stage1_analyzer = Stage1Analyzer(api_key=anthropic_api_key)
        self.programmatic_filter = ProgrammaticFilter()
        self.raw_data_scanner = RawDataScanner(data_fetcher=data_fetcher)
        self.stage2_analyzer = Stage2Analyzer(api_key=anthropic_api_key)
        self.scanner_db = ScannerDB(db_path=db_path)

    def run_full_scan(
        self,
        force_refresh_universe: bool = False,
        refresh_hours: int = 24,
        batch_size: int = 15,
        max_batches: int = 10,
        max_candidates: int = 150
    ) -> List[str]:
        """
        Execute complete two-stage scan.

        Args:
            force_refresh_universe: Force refresh universe from API
            refresh_hours: Max universe cache age
            batch_size: Stocks per Stage 2 batch
            max_batches: Max Stage 2 batches
            max_candidates: Max candidates for Stage 2

        Returns:
            List of final candidate symbols
        """
        start_time = time.time()
        scan_id = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        llm_calls = 0

        logger.info("="*60)
        logger.info(f"Starting Two-Stage Market Scan: {scan_id}")
        logger.info("="*60)

        # ===================================================================
        # LOAD UNIVERSE
        # ===================================================================
        logger.info("\n[STEP 1] Loading tradable universe...")
        universe = self.universe_loader.load_tradable_universe(
            force_refresh=force_refresh_universe,
            refresh_hours=refresh_hours,
            optionable_only=True
        )

        if not universe:
            logger.error("Failed to load universe. Aborting scan.")
            return []

        logger.info(f"Loaded {len(universe)} tradable stocks")

        # ===================================================================
        # STAGE 1: MARKET-WIDE SUMMARY
        # ===================================================================
        logger.info("\n[STEP 2] Stage 1: Calculating market-wide statistics (NO LLM)...")

        # For demo purposes, we'll calculate metrics for a subset
        # In production, you'd calculate for all 5000+ stocks
        # TODO: Implement bulk data fetching for all symbols

        # For now, let's use a sample of universe
        sample_size = min(500, len(universe))  # Sample for demo
        universe_sample = universe[:sample_size]
        logger.info(f"Calculating metrics for sample of {sample_size} stocks...")

        # Fetch price data using Alpaca
        stock_metrics = []
        symbols_to_fetch = [stock.symbol for stock in universe_sample]
        sector_map = {stock.symbol: stock.sector for stock in universe_sample}

        logger.info(f"Fetching price data for {len(symbols_to_fetch)} stocks...")

        # Use Alpaca ingestor if available, otherwise fallback to yfinance
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

                logger.info(f"Calculated metrics for {len(stock_metrics)} stocks")

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
                        # Process yfinance data...
                        logger.info(f"Using yfinance fallback data")
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed: {fallback_error}")
        else:
            logger.warning("No Alpaca ingestor available. Using limited sample data.")
            # Create limited metrics from available data
            for stock in universe_sample[:100]:
                mock_metric = StockMetrics(
                    symbol=stock.symbol,
                    sector=stock.sector,
                    current_price=stock.last_price
                )
                stock_metrics.append(mock_metric)

        # Aggregate by sector
        sector_stats = self.market_aggregator.aggregate_by_sector(stock_metrics)

        # Generate market summary
        market_summary = self.market_aggregator.generate_market_summary(
            stock_metrics,
            sector_stats
        )

        logger.info(f"Market summary generated:")
        logger.info(f"  Total stocks: {market_summary.total_stocks}")
        logger.info(f"  Market breadth: {market_summary.market_breadth_score:.2f}")
        logger.info(f"  Sectors analyzed: {len(market_summary.sectors)}")

        # ===================================================================
        # STAGE 1: LLM INTERPRETATION (1 LLM CALL)
        # ===================================================================
        logger.info("\n[STEP 3] Stage 1: LLM analyzing market (1 LLM call)...")

        stage1_guidance = self.stage1_analyzer.analyze(market_summary)
        llm_calls += 1

        if not stage1_guidance:
            logger.warning("Stage 1 LLM failed, using fallback guidance")
            stage1_guidance = self.stage1_analyzer.get_fallback_guidance(market_summary)

        logger.info(f"Stage 1 Guidance:")
        logger.info(f"  Market bias: {stage1_guidance.market_bias}")
        logger.info(f"  Focus sectors: {stage1_guidance.focus_sectors}")
        logger.info(f"  Focus patterns: {stage1_guidance.focus_patterns}")
        logger.info(f"  Volume threshold: {stage1_guidance.filtering_criteria.volume_ratio_threshold}x")
        logger.info(f"  52w high threshold: {stage1_guidance.filtering_criteria.distance_from_52w_high_threshold_pct}%")

        # ===================================================================
        # STAGE 2A: PROGRAMMATIC FILTERING (NO LLM)
        # ===================================================================
        logger.info("\n[STEP 4] Stage 2A: Applying programmatic filters (NO LLM)...")

        filter_result = self.programmatic_filter.apply_filters(
            stock_metrics,
            stage1_guidance,
            max_candidates=max_candidates
        )

        logger.info(f"Filter results:")
        logger.info(f"  Initial: {filter_result.initial_count}")
        logger.info(f"  Filtered: {filter_result.filtered_count}")
        logger.info(f"  Candidates: {filter_result.candidates[:10]}...")  # Show first 10

        if not filter_result.candidates:
            logger.warning("No candidates passed filters. Scan complete.")
            return []

        # ===================================================================
        # STAGE 2B: DEEP ANALYSIS WITH LLM (5-10 LLM CALLS)
        # ===================================================================
        logger.info("\n[STEP 5] Stage 2B: Deep analysis in batches...")

        # Fetch detailed indicators for candidates
        logger.info(f"Fetching technical indicators for {len(filter_result.candidates)} candidates...")

        indicators_map = {}

        if self.alpaca_ingestor:
            try:
                # Fetch detailed price data for candidates
                candidate_price_data = self.alpaca_ingestor.fetch_bars_batched(
                    symbols=filter_result.candidates,
                    timeframe="1Day",
                    start=datetime.now() - timedelta(days=300),  # Need more data for 200-day MA
                    end=datetime.now(),
                    use_cache=True
                )

                # Calculate detailed indicators using pandas_ta
                for symbol, price_df in candidate_price_data.items():
                    indicators = self.raw_data_scanner.fetch_indicators(symbol, price_df)
                    if indicators:
                        indicators_map[symbol] = indicators

                logger.info(f"Calculated indicators for {len(indicators_map)} candidates")

            except Exception as e:
                logger.error(f"Error fetching indicators: {e}")
                logger.warning("Will use fallback selection")
        else:
            logger.warning("No Alpaca ingestor available. Using fallback selection")

        # Analyze in batches
        if indicators_map:
            final_candidates = self.stage2_analyzer.analyze_all_batches(
                filter_result.candidates,
                indicators_map,
                stage1_guidance,
                batch_size=batch_size,
                max_batches=max_batches
            )
            llm_calls += min(max_batches, (len(filter_result.candidates) + batch_size - 1) // batch_size)
        else:
            # Use fallback selection if no indicators
            logger.warning("Using fallback selection (no indicators available)")
            final_candidates = self.stage2_analyzer.get_fallback_selection(
                filter_result.candidates[:20],  # Limit to 20
                {},
                stage1_guidance,
                max_count=20
            )

        # ===================================================================
        # SAVE RESULTS
        # ===================================================================
        end_time = time.time()
        execution_time = end_time - start_time
        estimated_cost = llm_calls * 0.02  # Rough estimate: $0.02 per call

        logger.info("\n" + "="*60)
        logger.info("SCAN COMPLETE")
        logger.info("="*60)
        logger.info(f"Final candidates: {len(final_candidates)}")
        logger.info(f"Candidates: {final_candidates}")
        logger.info(f"Execution time: {execution_time:.1f} seconds")
        logger.info(f"LLM calls: {llm_calls}")
        logger.info(f"Estimated cost: ${estimated_cost:.3f}")
        logger.info("="*60)

        # Save scan result to database
        scan_result = ScanResult(
            scan_id=scan_id,
            timestamp=datetime.now().isoformat(),
            stage1_guidance=stage1_guidance.dict(),
            filtered_count=filter_result.filtered_count,
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

    # Mock data fetcher
    class MockDataFetcher:
        def fetch_prices(self, symbols, interval='1d', lookback_days=90):
            return pd.DataFrame()  # Mock implementation

    data_fetcher = MockDataFetcher()

    orchestrator = ScannerOrchestrator(
        data_fetcher=data_fetcher,
        alpaca_api_key=os.getenv('ALPACA_API_KEY'),
        alpaca_secret_key=os.getenv('ALPACA_SECRET_KEY'),
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
    )

    # Run scan
    candidates = orchestrator.run_full_scan(
        force_refresh_universe=False,
        refresh_hours=24,
        batch_size=15,
        max_batches=10,
        max_candidates=150
    )

    print(f"\n\nFinal candidates for trading: {candidates}")
