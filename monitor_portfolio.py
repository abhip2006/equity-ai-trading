#!/usr/bin/env python3
"""
Portfolio Monitoring Script for Live Trading

Real-time monitoring of positions, P&L, and portfolio status.
Compares local database positions with Alpaca broker positions.

Usage:
    python monitor_portfolio.py                    # One-time check
    python monitor_portfolio.py --watch            # Continuous monitoring
    python monitor_portfolio.py --watch --interval 60  # Check every 60 seconds
"""

import os
import sys
import time
import argparse
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List

# Load environment variables
load_dotenv()

from active_trader_llm.execution.position_manager import PositionManager
from active_trader_llm.execution.alpaca_broker import AlpacaBrokerExecutor


def format_currency(value: float) -> str:
    """Format currency with color coding"""
    if value > 0:
        return f"\033[92m${value:,.2f}\033[0m"  # Green
    elif value < 0:
        return f"\033[91m${value:,.2f}\033[0m"  # Red
    else:
        return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format percentage with color coding"""
    if value > 0:
        return f"\033[92m{value:+.2f}%\033[0m"  # Green
    elif value < 0:
        return f"\033[91m{value:+.2f}%\033[0m"  # Red
    else:
        return f"{value:.2f}%"


def print_header(text: str):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)


def check_portfolio(
    position_manager: PositionManager,
    broker: AlpacaBrokerExecutor,
    show_details: bool = True
):
    """
    Check and display portfolio status

    Args:
        position_manager: Local position manager
        broker: Alpaca broker executor
        show_details: Show detailed position breakdown
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}]")

    # Get Alpaca account info
    print_header("ALPACA ACCOUNT STATUS")
    account = broker.get_account_info()

    print(f"Account Status:    {account['status']}")
    print(f"Portfolio Value:   ${float(account['equity']):,.2f}")
    print(f"Cash Available:    ${float(account['cash']):,.2f}")
    print(f"Buying Power:      ${float(account['buying_power']):,.2f}")

    # Get Alpaca positions
    print_header("ALPACA BROKER POSITIONS")
    alpaca_positions = broker.get_positions()

    if alpaca_positions:
        print(f"Position Count: {len(alpaca_positions)}\n")
        print(f"{'Symbol':<8} {'Qty':>8} {'Avg Entry':>12} {'Current':>12} {'Market Value':>15} {'Unrealized P&L':>18}")
        print("-" * 80)

        total_market_value = 0.0
        total_unrealized_pnl = 0.0

        for pos in alpaca_positions:
            print(f"{pos.symbol:<8} {pos.qty:>8.0f} ${pos.avg_entry_price:>11.2f} "
                  f"${pos.current_price:>11.2f} ${pos.market_value:>14.2f} "
                  f"{format_currency(pos.unrealized_pl):>25} ({format_percentage(pos.unrealized_pl_pct)})")
            total_market_value += pos.market_value
            total_unrealized_pnl += pos.unrealized_pl

        print("-" * 80)
        print(f"{'TOTAL':>8} {' ':>8} {' ':>12} {' ':>12} "
              f"${total_market_value:>14.2f} {format_currency(total_unrealized_pnl):>25}")
    else:
        print("No positions in Alpaca broker")

    # Get local database positions
    print_header("LOCAL DATABASE POSITIONS")
    local_positions = position_manager.get_open_positions()

    if local_positions:
        print(f"Position Count: {len(local_positions)}\n")

        # Fetch current prices from Alpaca for unrealized P&L calculation
        current_prices = {}
        for pos in local_positions:
            try:
                # Get current price from Alpaca positions if available
                alpaca_pos = next((p for p in alpaca_positions if p.symbol == pos.symbol), None)
                if alpaca_pos:
                    current_prices[pos.symbol] = alpaca_pos.current_price
                else:
                    # Position not in Alpaca - might be pending or closed
                    current_prices[pos.symbol] = pos.entry_price
            except:
                current_prices[pos.symbol] = pos.entry_price

        # Calculate portfolio state
        portfolio = position_manager.get_portfolio_state(current_prices)

        print(f"{'Symbol':<8} {'Dir':>6} {'Shares':>8} {'Entry':>12} {'Stop':>12} {'Target':>12} {'Strategy':<20}")
        print("-" * 90)

        for pos in local_positions:
            print(f"{pos.symbol:<8} {pos.direction:>6} {pos.shares:>8} "
                  f"${pos.entry_price:>11.2f} ${pos.stop_loss:>11.2f} "
                  f"${pos.take_profit:>11.2f} {pos.strategy or 'N/A':<20}")

        if portfolio['open_positions']:
            print("\n" + "-" * 90)
            print(f"Total Exposure:     ${portfolio['total_exposure']:,.2f}")
            print(f"Total Unrealized:   {format_currency(portfolio['total_unrealized_pnl'])}")
    else:
        print("No positions in local database")

    # Get closed positions (today)
    print_header("CLOSED POSITIONS (TODAY)")
    from datetime import timedelta as td
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    closed_today = position_manager.get_closed_positions(since=today_start)

    if closed_today:
        print(f"Closed Count: {len(closed_today)}\n")
        print(f"{'Symbol':<8} {'Dir':>6} {'Entry':>12} {'Exit':>12} {'Reason':<15} {'Realized P&L':>18}")
        print("-" * 80)

        total_realized = 0.0
        wins = 0
        losses = 0

        for pos in closed_today:
            if pos.realized_pnl > 0:
                wins += 1
            elif pos.realized_pnl < 0:
                losses += 1

            print(f"{pos.symbol:<8} {pos.direction:>6} "
                  f"${pos.entry_price:>11.2f} ${pos.exit_price:>11.2f} "
                  f"{pos.exit_reason or 'N/A':<15} {format_currency(pos.realized_pnl):>25}")
            total_realized += pos.realized_pnl or 0.0

        print("-" * 80)
        print(f"{'TOTAL':>8} {' ':>6} {' ':>12} {' ':>12} {' ':<15} {format_currency(total_realized):>25}")
        print(f"\nWin Rate: {wins}/{len(closed_today)} ({100*wins/len(closed_today) if closed_today else 0:.1f}%)")
    else:
        print("No positions closed today")

    # Synchronization check
    print_header("SYNCHRONIZATION CHECK")

    alpaca_symbols = set(p.symbol for p in alpaca_positions)
    local_symbols = set(p.symbol for p in local_positions)

    in_both = alpaca_symbols & local_symbols
    only_alpaca = alpaca_symbols - local_symbols
    only_local = local_symbols - alpaca_symbols

    print(f"Positions in both systems:     {len(in_both)}")
    if in_both:
        print(f"  Symbols: {', '.join(sorted(in_both))}")

    if only_alpaca:
        print(f"\n⚠️  Positions only in Alpaca:     {len(only_alpaca)}")
        print(f"  Symbols: {', '.join(sorted(only_alpaca))}")
        print("  → These may need to be added to local database")

    if only_local:
        print(f"\n⚠️  Positions only in local DB:    {len(only_local)}")
        print(f"  Symbols: {', '.join(sorted(only_local))}")
        print("  → These may be pending orders or need to be closed in DB")

    if not only_alpaca and not only_local:
        print("✅ Systems are synchronized")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Monitor portfolio positions and P&L"
    )
    parser.add_argument(
        '--watch',
        action='store_true',
        help='Continuously monitor portfolio (refresh every interval)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=30,
        help='Refresh interval in seconds for watch mode (default: 30)'
    )
    parser.add_argument(
        '--db',
        type=str,
        default='data/positions.db',
        help='Path to positions database (default: data/positions.db)'
    )

    args = parser.parse_args()

    # Initialize components
    try:
        print("Initializing portfolio monitor...")
        position_manager = PositionManager(args.db)
        broker = AlpacaBrokerExecutor(paper=True)
        print("✅ Connected to Alpaca and local database\n")
    except Exception as e:
        print(f"❌ Error initializing: {e}")
        sys.exit(1)

    # Watch mode or one-time check
    if args.watch:
        print(f"Starting continuous monitoring (refresh every {args.interval}s)")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                check_portfolio(position_manager, broker)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
    else:
        check_portfolio(position_manager, broker)


if __name__ == "__main__":
    main()
