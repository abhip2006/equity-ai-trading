#!/usr/bin/env python3
"""
Test script for Alpaca integration

This script verifies that:
1. Alpaca API credentials are configured
2. Connection to Alpaca is working
3. Account information can be retrieved
4. Positions can be queried
5. Order submission works (optional - commented out)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from active_trader_llm.execution.alpaca_broker import AlpacaBrokerExecutor
from active_trader_llm.trader.trader_agent import TradePlan


def test_alpaca_connection():
    """Test 1: Verify API connection"""
    print("=" * 60)
    print("TEST 1: Alpaca API Connection")
    print("=" * 60)

    try:
        executor = AlpacaBrokerExecutor(paper=True)
        print("✓ Successfully connected to Alpaca Paper Trading API")
        return executor
    except ValueError as e:
        print(f"✗ Failed: {e}")
        print("\nMake sure you have set your environment variables:")
        print("  export ALPACA_API_KEY=your_key_here")
        print("  export ALPACA_SECRET_KEY=your_secret_here")
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None


def test_account_info(executor):
    """Test 2: Get account information"""
    print("\n" + "=" * 60)
    print("TEST 2: Account Information")
    print("=" * 60)

    try:
        account = executor.get_account_info()
        print("✓ Successfully retrieved account info")
        print(f"\n  Account Number: {account.get('account_number', 'N/A')}")
        print(f"  Status: {account.get('status', 'N/A')}")
        print(f"  Portfolio Value: ${float(account.get('portfolio_value', 0)):,.2f}")
        print(f"  Cash: ${float(account.get('cash', 0)):,.2f}")
        print(f"  Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
        print(f"  Pattern Day Trader: {account.get('pattern_day_trader', 'N/A')}")
        return True
    except Exception as e:
        print(f"✗ Failed to get account info: {e}")
        return False


def test_positions(executor):
    """Test 3: Get current positions"""
    print("\n" + "=" * 60)
    print("TEST 3: Current Positions")
    print("=" * 60)

    try:
        positions = executor.get_positions()
        print(f"✓ Successfully retrieved positions")
        print(f"\n  Total Positions: {len(positions)}")

        if positions:
            print("\n  Details:")
            for pos in positions:
                print(f"    {pos.symbol}:")
                print(f"      Quantity: {pos.qty}")
                print(f"      Entry Price: ${pos.avg_entry_price:.2f}")
                print(f"      Current Price: ${pos.current_price:.2f}")
                print(f"      Market Value: ${pos.market_value:.2f}")
                print(f"      P&L: ${pos.unrealized_pl:.2f} ({pos.unrealized_pl_pct:.2%})")
        else:
            print("    No open positions")

        return True
    except Exception as e:
        print(f"✗ Failed to get positions: {e}")
        return False


def test_position_size_calculation(executor):
    """Test 4: Position size calculation"""
    print("\n" + "=" * 60)
    print("TEST 4: Position Size Calculation")
    print("=" * 60)

    try:
        # Create a sample trade plan
        plan = TradePlan(
            symbol="AAPL",
            strategy="momentum_breakout",
            direction="long",
            entry=175.0,
            stop_loss=170.0,
            take_profit=185.0,
            position_size_pct=0.05,  # 5% of portfolio
            time_horizon="3d",
            rationale=["Test position sizing"],
            risk_reward_ratio=2.0
        )

        shares = executor.calculate_position_size(plan)
        account = executor.get_account_info()
        equity = float(account.get('equity', 100000))
        allocation = equity * plan.position_size_pct

        print("✓ Position size calculated successfully")
        print(f"\n  Account Equity: ${equity:,.2f}")
        print(f"  Position Size %: {plan.position_size_pct * 100:.1f}%")
        print(f"  Dollar Allocation: ${allocation:,.2f}")
        print(f"  Entry Price: ${plan.entry:.2f}")
        print(f"  Shares to Buy: {shares}")
        print(f"  Actual Cost: ${shares * plan.entry:,.2f}")

        return True
    except Exception as e:
        print(f"✗ Failed position size calculation: {e}")
        return False


def test_order_submission(executor, dry_run=True):
    """Test 5: Order submission (dry run by default)"""
    print("\n" + "=" * 60)
    print("TEST 5: Order Submission (DRY RUN)")
    print("=" * 60)

    if dry_run:
        print("⚠ This is a dry run - no actual orders will be submitted")
        print("  To actually submit an order, set dry_run=False")
        print("\n  Example trade that would be submitted:")

        plan = TradePlan(
            symbol="SPY",
            strategy="momentum_breakout",
            direction="long",
            entry=450.0,
            stop_loss=445.0,
            take_profit=460.0,
            position_size_pct=0.02,  # 2% of portfolio
            time_horizon="3d",
            rationale=["Test order submission"],
            risk_reward_ratio=2.0
        )

        account = executor.get_account_info()
        equity = float(account.get('equity', 100000))
        shares = int((equity * plan.position_size_pct) / plan.entry)

        print(f"\n  Symbol: {plan.symbol}")
        print(f"  Direction: {plan.direction.upper()}")
        print(f"  Quantity: {shares} shares")
        print(f"  Entry: ${plan.entry:.2f}")
        print(f"  Stop Loss: ${plan.stop_loss:.2f}")
        print(f"  Take Profit: ${plan.take_profit:.2f}")
        print(f"  Risk/Reward: {plan.risk_reward_ratio:.2f}:1")

        print("\n✓ Order structure validated (not submitted)")
        return True
    else:
        print("⚠ LIVE ORDER SUBMISSION - This will place a real order!")
        response = input("  Are you sure you want to continue? (yes/no): ")

        if response.lower() != 'yes':
            print("✗ Order submission cancelled by user")
            return False

        try:
            plan = TradePlan(
                symbol="SPY",
                strategy="momentum_breakout",
                direction="long",
                entry=450.0,
                stop_loss=445.0,
                take_profit=460.0,
                position_size_pct=0.02,
                time_horizon="3d",
                rationale=["Test order submission"],
                risk_reward_ratio=2.0
            )

            result = executor.submit_trade(plan, order_type="market")

            if result.success:
                print(f"✓ Order submitted successfully!")
                print(f"  Order ID: {result.order_id}")
                print(f"  Status: {result.status}")
                if result.filled_price:
                    print(f"  Filled Price: ${result.filled_price:.2f}")
                    print(f"  Filled Qty: {result.filled_qty}")
                return True
            else:
                print(f"✗ Order failed: {result.error_message}")
                return False

        except Exception as e:
            print(f"✗ Error submitting order: {e}")
            return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ALPACA INTEGRATION TEST SUITE")
    print("=" * 60)

    # Test 1: Connection
    executor = test_alpaca_connection()
    if not executor:
        print("\n❌ Cannot proceed without valid Alpaca connection")
        return

    # Test 2: Account Info
    if not test_account_info(executor):
        print("\n⚠ Account info test failed, but continuing...")

    # Test 3: Positions
    if not test_positions(executor):
        print("\n⚠ Positions test failed, but continuing...")

    # Test 4: Position Size Calculation
    if not test_position_size_calculation(executor):
        print("\n⚠ Position size calculation failed, but continuing...")

    # Test 5: Order Submission (dry run)
    test_order_submission(executor, dry_run=True)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("✓ All tests completed")
    print("\nNext steps:")
    print("  1. Review the results above")
    print("  2. Check config_alpaca_paper.yaml configuration")
    print("  3. Run a test decision cycle:")
    print("     python -m active_trader_llm.main --config config_alpaca_paper.yaml --cycles 1")
    print("\n  See ALPACA_SETUP.md for detailed documentation")


if __name__ == "__main__":
    main()
