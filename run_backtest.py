#!/usr/bin/env python3
"""
Run Backtest - CLI Entry Point

Execute backtests of the ActiveTrader-LLM trading system.
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from active_trader_llm.backtest.backtest_engine import BacktestEngine
from active_trader_llm.backtest.report_generator import ReportGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_last_n_market_days(n: int = 5) -> tuple:
    """
    Get approximate start and end dates for last N market days.

    Args:
        n: Number of market days

    Returns:
        Tuple of (start_date, end_date) as strings
    """
    # Rough estimate: assume 5 trading days per week
    # Add extra days to ensure we get N market days
    buffer_days = int(n * 1.5)  # 50% buffer for weekends

    end_date = date.today() - timedelta(days=1)  # Yesterday
    start_date = end_date - timedelta(days=buffer_days)

    return str(start_date), str(end_date)


def main():
    parser = argparse.ArgumentParser(
        description='Run backtest for ActiveTrader-LLM trading system',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date for backtest (YYYY-MM-DD). If not provided, will backtest last 5 market days'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date for backtest (YYYY-MM-DD). If not provided, will use yesterday'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=5,
        help='Number of market days to backtest (used if start-date not provided)'
    )

    parser.add_argument(
        '--initial-capital',
        type=float,
        default=100000.0,
        help='Initial capital for backtest'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='TEST_RESULT.md',
        help='Output path for test result report'
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default='data/backtest.db',
        help='Path to backtest database'
    )

    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Optional name for this backtest run'
    )

    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip report generation (just run backtest)'
    )

    args = parser.parse_args()

    # Determine date range
    if args.start_date is None:
        logger.info(f"No start date provided, using last {args.days} market days")
        start_date, end_date = get_last_n_market_days(args.days)
    else:
        start_date = args.start_date
        end_date = args.end_date or str(date.today() - timedelta(days=1))

    logger.info("=" * 80)
    logger.info("ACTIVETRADER-LLM BACKTEST")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Initial capital: ${args.initial_capital:,.2f}")
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Report output: {args.output}")
    logger.info("=" * 80)

    try:
        # Initialize backtest engine
        engine = BacktestEngine(
            config_path=args.config,
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.initial_capital,
            backtest_db_path=args.db_path
        )

        # Run backtest
        run_id = engine.run(run_name=args.name)

        logger.info("\n" + "=" * 80)
        logger.info(f"Backtest completed successfully! Run ID: {run_id}")
        logger.info("=" * 80)

        # Generate report
        if not args.no_report:
            logger.info("\nGenerating report...")
            generator = ReportGenerator(args.db_path)
            generator.generate_report(run_id, output_path=args.output)

            logger.info(f"\n✅ Report generated: {args.output}")
            logger.info(f"📊 Database: {args.db_path}")
            logger.info(f"🆔 Run ID: {run_id}")

        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"\n❌ Backtest failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
