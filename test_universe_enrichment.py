#!/usr/bin/env python3
"""
Test Universe Enrichment
Tests that metadata enrichment works correctly with yfinance.
"""

import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from active_trader_llm.scanners.universe_loader import UniverseLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_small_universe_with_enrichment():
    """Test enriching a small universe of well-known stocks"""
    logger.info("=" * 80)
    logger.info("TEST: Small Universe with Metadata Enrichment")
    logger.info("=" * 80)

    # Initialize loader
    loader = UniverseLoader(paper=True)

    # Manually create a small test universe
    from active_trader_llm.scanners.scanner_db import TradableStock
    from datetime import datetime

    test_stocks = [
        TradableStock(
            symbol="AAPL",
            sector="Unknown",
            market_cap=None,
            avg_volume_20d=None,
            last_price=None,
            optionable=True,
            updated_at=datetime.now().isoformat()
        ),
        TradableStock(
            symbol="MSFT",
            sector="Unknown",
            market_cap=None,
            avg_volume_20d=None,
            last_price=None,
            optionable=True,
            updated_at=datetime.now().isoformat()
        ),
        TradableStock(
            symbol="GOOGL",
            sector="Unknown",
            market_cap=None,
            avg_volume_20d=None,
            last_price=None,
            optionable=True,
            updated_at=datetime.now().isoformat()
        ),
        TradableStock(
            symbol="NVDA",
            sector="Unknown",
            market_cap=None,
            avg_volume_20d=None,
            last_price=None,
            optionable=True,
            updated_at=datetime.now().isoformat()
        ),
        TradableStock(
            symbol="JPM",
            sector="Unknown",
            market_cap=None,
            avg_volume_20d=None,
            last_price=None,
            optionable=True,
            updated_at=datetime.now().isoformat()
        ),
    ]

    logger.info(f"\nBefore enrichment:")
    logger.info(f"  Total stocks: {len(test_stocks)}")
    for stock in test_stocks:
        logger.info(
            f"    {stock.symbol}: sector={stock.sector}, "
            f"market_cap={stock.market_cap}, "
            f"volume={stock.avg_volume_20d}, "
            f"price={stock.last_price}"
        )

    # Enrich metadata
    logger.info("\nEnriching metadata via yfinance...")
    symbols = [stock.symbol for stock in test_stocks]
    metadata = loader.refresh_universe_metadata(symbols)

    # Apply enrichment
    for stock in test_stocks:
        if stock.symbol in metadata:
            stock_meta = metadata[stock.symbol]
            stock.sector = stock_meta.get('sector', stock.sector)
            stock.market_cap = stock_meta.get('market_cap', stock.market_cap)
            stock.avg_volume_20d = stock_meta.get('avg_volume', stock.avg_volume_20d)
            stock.last_price = stock_meta.get('last_price', stock.last_price)

    logger.info(f"\nAfter enrichment:")
    logger.info(f"  Total stocks: {len(test_stocks)}")
    logger.info(f"  Enriched: {len(metadata)}/{len(test_stocks)} ({len(metadata)/len(test_stocks)*100:.1f}%)")

    for stock in test_stocks:
        market_cap_str = f"${stock.market_cap/1e9:.2f}B" if stock.market_cap else "N/A"
        volume_str = f"{stock.avg_volume_20d/1e6:.2f}M" if stock.avg_volume_20d else "N/A"
        price_str = f"${stock.last_price:.2f}" if stock.last_price else "N/A"

        logger.info(
            f"    {stock.symbol:6s}: {stock.sector:20s} | "
            f"Cap: {market_cap_str:10s} | Vol: {volume_str:10s} | Price: {price_str}"
        )

    # Test sector breakdown
    logger.info("\nSector Breakdown:")
    sectors = loader.get_sector_breakdown(test_stocks)
    for sector, symbols in sectors.items():
        logger.info(f"  {sector}: {', '.join(symbols)}")

    # Validate results
    logger.info("\nValidation:")
    errors = []

    for stock in test_stocks:
        if stock.sector == "Unknown":
            errors.append(f"{stock.symbol}: sector still 'Unknown'")
        if stock.market_cap is None:
            errors.append(f"{stock.symbol}: market_cap is None")
        if stock.avg_volume_20d is None:
            errors.append(f"{stock.symbol}: avg_volume_20d is None")
        if stock.last_price is None:
            errors.append(f"{stock.symbol}: last_price is None")

    if errors:
        logger.warning(f"Found {len(errors)} validation issues:")
        for error in errors:
            logger.warning(f"  - {error}")
    else:
        logger.info("All stocks successfully enriched!")

    return len(errors) == 0


def test_execution_mode():
    """Test that paper/live mode is threaded correctly"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Execution Mode Threading")
    logger.info("=" * 80)

    # Test paper mode
    loader_paper = UniverseLoader(paper=True)
    logger.info(f"Paper mode loader created: paper={loader_paper.paper}")

    # Test live mode
    loader_live = UniverseLoader(paper=False)
    logger.info(f"Live mode loader created: paper={loader_live.paper}")

    if loader_paper.paper and not loader_live.paper:
        logger.info("Execution mode threading: PASS")
        return True
    else:
        logger.error("Execution mode threading: FAIL")
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "=" * 80)
    logger.info("UNIVERSE ENRICHMENT TEST SUITE")
    logger.info("=" * 80)

    results = []

    # Test 1: Metadata enrichment
    try:
        results.append(("Metadata Enrichment", test_small_universe_with_enrichment()))
    except Exception as e:
        logger.error(f"Metadata enrichment test failed: {e}", exc_info=True)
        results.append(("Metadata Enrichment", False))

    # Test 2: Execution mode
    try:
        results.append(("Execution Mode", test_execution_mode()))
    except Exception as e:
        logger.error(f"Execution mode test failed: {e}", exc_info=True)
        results.append(("Execution Mode", False))

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"  {status}: {test_name}")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\nAll tests passed! Universe enrichment is working correctly.")
        logger.info("\nNext steps:")
        logger.info("1. The scanner can now populate real sector, market cap, volume, and price data")
        logger.info("2. Stage 1 sector breakdowns will use real sectors instead of 'Unknown'")
        logger.info("3. Paper/live mode is correctly threaded through the system")
    else:
        logger.error("\nSome tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
