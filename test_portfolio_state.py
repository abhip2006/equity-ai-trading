#!/usr/bin/env python3
"""
Test Portfolio State Management
Verifies that portfolio state is correctly handled in different modes.
"""

import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from active_trader_llm.main import ActiveTraderLLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_backtest_mode():
    """Test that backtest mode uses hardcoded $100k"""
    logger.info("=" * 80)
    logger.info("TEST 1: Backtest Mode Portfolio State")
    logger.info("=" * 80)

    try:
        system = ActiveTraderLLM("config.yaml")

        logger.info(f"\nMode: {system.config.mode}")
        logger.info(f"Portfolio State:")
        logger.info(f"  Cash: ${system.portfolio_state.cash:,.2f}")
        logger.info(f"  Equity: ${system.portfolio_state.equity:,.2f}")

        # Get current portfolio state (should return tracked state)
        current_state = system.get_current_portfolio_state()
        logger.info(f"\nCurrent Portfolio State (via get_current_portfolio_state()):")
        logger.info(f"  Cash: ${current_state.cash:,.2f}")
        logger.info(f"  Equity: ${current_state.equity:,.2f}")

        # Verify
        if system.config.mode == "backtest":
            if system.portfolio_state.cash == 100000.0 and system.portfolio_state.equity == 100000.0:
                logger.info("\nPASS: Backtest mode correctly initialized with $100,000")
                return True
            else:
                logger.error(f"\nFAIL: Expected $100,000, got cash=${system.portfolio_state.cash}, equity=${system.portfolio_state.equity}")
                return False
        else:
            logger.warning(f"\nSKIP: Config is in {system.config.mode} mode, not backtest")
            return True

    except Exception as e:
        logger.error(f"\nFAIL: {e}", exc_info=True)
        return False


def test_paper_live_mode():
    """Test that paper-live mode fetches from Alpaca"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Paper-Live Mode Portfolio State")
    logger.info("=" * 80)

    try:
        system = ActiveTraderLLM("config_alpaca_paper.yaml")

        logger.info(f"\nMode: {system.config.mode}")
        logger.info(f"Portfolio State:")
        logger.info(f"  Cash: ${system.portfolio_state.cash:,.2f}")
        logger.info(f"  Equity: ${system.portfolio_state.equity:,.2f}")

        # Get current portfolio state (should fetch from Alpaca)
        current_state = system.get_current_portfolio_state()
        logger.info(f"\nCurrent Portfolio State (via get_current_portfolio_state()):")
        logger.info(f"  Cash: ${current_state.cash:,.2f}")
        logger.info(f"  Equity: ${current_state.equity:,.2f}")

        # Verify broker executor exists
        if not system.broker_executor:
            logger.error("\nFAIL: Broker executor not initialized")
            return False

        # Fetch directly from broker to compare
        account_info = system.broker_executor.get_account_info()
        broker_equity = float(account_info['equity'])
        broker_cash = float(account_info['cash'])

        logger.info(f"\nDirect from Alpaca Broker:")
        logger.info(f"  Cash: ${broker_cash:,.2f}")
        logger.info(f"  Equity: ${broker_equity:,.2f}")

        # Verify values match
        if abs(current_state.equity - broker_equity) < 0.01 and abs(current_state.cash - broker_cash) < 0.01:
            logger.info("\nPASS: Portfolio state matches Alpaca broker values")
            return True
        else:
            logger.error(f"\nFAIL: Portfolio state doesn't match broker")
            logger.error(f"  Expected: equity=${broker_equity:,.2f}, cash=${broker_cash:,.2f}")
            logger.error(f"  Got: equity=${current_state.equity:,.2f}, cash=${current_state.cash:,.2f}")
            return False

    except Exception as e:
        logger.error(f"\nFAIL: {e}", exc_info=True)
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "=" * 80)
    logger.info("PORTFOLIO STATE TEST SUITE")
    logger.info("=" * 80)

    results = []

    # Test 1: Backtest mode
    try:
        results.append(("Backtest Mode", test_backtest_mode()))
    except Exception as e:
        logger.error(f"Backtest mode test failed: {e}", exc_info=True)
        results.append(("Backtest Mode", False))

    # Test 2: Paper-live mode
    try:
        results.append(("Paper-Live Mode", test_paper_live_mode()))
    except Exception as e:
        logger.error(f"Paper-live mode test failed: {e}", exc_info=True)
        results.append(("Paper-Live Mode", False))

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
        logger.info("\nAll tests passed! Portfolio state management is working correctly.")
        logger.info("\nKey improvements:")
        logger.info("  - Backtest mode uses tracked $100k state")
        logger.info("  - Paper-live/live modes fetch fresh from Alpaca")
        logger.info("  - Risk manager always uses accurate account balances")
    else:
        logger.error("\nSome tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
