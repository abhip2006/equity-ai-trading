#!/usr/bin/env python3
"""
Test script for market scanner functionality.

This script tests the scanner with different configurations to verify it works correctly.
"""

import logging
from active_trader_llm.scanning.market_scanner import MarketScanner, ScannerConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_scanner_custom():
    """Test scanner with a small custom universe"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Scanner with custom tickers (AAPL, MSFT, GOOGL, NVDA, TSLA)")
    logger.info("=" * 80)

    config = ScannerConfig(
        enabled=True,
        base_universe="custom",
        custom_tickers=["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMZN", "META", "SPY", "QQQ"],
        min_avg_volume=500_000,  # Lower threshold for testing
        min_dollar_volume=5_000_000,  # $5M
        min_price=5.0,
        max_price=1000.0,
        max_universe_size=10,
        use_cache=False  # Disable cache for testing
    )

    scanner = MarketScanner(config)
    universe = scanner.get_tradable_universe()

    logger.info(f"\nResults:")
    logger.info(f"  Filtered universe size: {len(universe)}")
    logger.info(f"  Symbols: {universe}")

    return universe


def test_scanner_sp500_small():
    """Test scanner with S&P 500 but limited to top 20"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Scanner with S&P 500 (top 20 by dollar volume)")
    logger.info("=" * 80)

    config = ScannerConfig(
        enabled=True,
        base_universe="sp500",
        min_avg_volume=1_000_000,  # 1M shares/day
        min_dollar_volume=10_000_000,  # $10M/day
        min_price=5.0,
        max_price=1000.0,
        max_universe_size=20,  # Limit to 20 for faster testing
        use_cache=False
    )

    scanner = MarketScanner(config)
    universe = scanner.get_tradable_universe()

    logger.info(f"\nResults:")
    logger.info(f"  Filtered universe size: {len(universe)}")
    logger.info(f"  Top 20 symbols by dollar volume: {universe[:20]}")

    return universe


def test_scanner_high_volume():
    """Test scanner with high volume requirements"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Scanner with high volume requirements (day trading profile)")
    logger.info("=" * 80)

    config = ScannerConfig(
        enabled=True,
        base_universe="sp500",
        min_avg_volume=5_000_000,  # 5M shares/day
        min_dollar_volume=50_000_000,  # $50M/day
        min_price=10.0,
        max_price=500.0,
        min_atr_pct=0.015,  # Minimum 1.5% daily ATR
        max_universe_size=30,
        use_cache=False
    )

    scanner = MarketScanner(config)
    universe = scanner.get_tradable_universe()

    logger.info(f"\nResults:")
    logger.info(f"  Filtered universe size: {len(universe)}")
    logger.info(f"  High-volume symbols: {universe[:30]}")

    return universe


def test_scanner_disabled():
    """Test that disabled scanner returns empty list"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Scanner disabled (should return empty)")
    logger.info("=" * 80)

    config = ScannerConfig(
        enabled=False,
        base_universe="sp500"
    )

    scanner = MarketScanner(config)
    universe = scanner.get_tradable_universe()

    logger.info(f"\nResults:")
    logger.info(f"  Filtered universe size: {len(universe)} (should be 0)")

    assert len(universe) == 0, "Disabled scanner should return empty universe"
    logger.info("  ✓ Test passed: disabled scanner returns empty")

    return universe


def test_scanner_stats():
    """Test scanner statistics"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Scanner statistics")
    logger.info("=" * 80)

    config = ScannerConfig(
        enabled=True,
        base_universe="custom",
        custom_tickers=["AAPL", "MSFT", "GOOGL"],
        min_avg_volume=500_000,
        max_universe_size=10,
        use_cache=False
    )

    scanner = MarketScanner(config)
    stats = scanner.get_scan_stats()

    logger.info(f"\nScanner Stats:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    return stats


if __name__ == "__main__":
    logger.info("Starting scanner tests...")
    logger.info("Note: These tests require internet connection to fetch market data via yfinance")

    try:
        # Run tests
        test_scanner_custom()
        test_scanner_sp500_small()
        test_scanner_high_volume()
        test_scanner_disabled()
        test_scanner_stats()

        logger.info("\n" + "=" * 80)
        logger.info("ALL TESTS COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
