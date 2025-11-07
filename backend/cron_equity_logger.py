#!/usr/bin/env python3
"""
Standalone Equity Logger for Railway Cron Jobs

This script logs equity snapshots for all configured LLM trading models.
Designed to run as a Railway cron job every 5 minutes.

Schedule: */5 * * * * (every 5 minutes)
"""

import os
import sys
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv(Path(__file__).parent.parent / '.env')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database path - use volume mount if available, otherwise local
DATABASE_PATH = os.getenv('DATABASE_PATH', str(Path(__file__).parent.parent / 'data'))
EQUITY_LOG_DB = Path(DATABASE_PATH) / "equity_log.db"

# Ensure database directory exists
EQUITY_LOG_DB.parent.mkdir(parents=True, exist_ok=True)


def init_equity_log_db():
    """Initialize equity logging database with required schema"""
    try:
        conn = sqlite3.connect(str(EQUITY_LOG_DB))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                equity REAL NOT NULL,
                cash REAL,
                portfolio_value REAL,
                buying_power REAL,
                position_count INTEGER,
                UNIQUE(model_id, timestamp)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_equity_model_time
            ON equity_snapshots(model_id, timestamp DESC)
        """)

        conn.commit()
        conn.close()
        logger.info(f"Equity log database initialized at {EQUITY_LOG_DB}")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False


def log_equity_snapshot():
    """Log equity snapshots for all configured models"""
    models = ['gpt4', 'claude', 'gemini', 'deepseek', 'qwen', 'grok']
    timestamp = datetime.now().isoformat()

    logger.info(f"Starting equity snapshot logging at {timestamp}")

    try:
        conn = sqlite3.connect(str(EQUITY_LOG_DB))
        cursor = conn.cursor()

        logged_count = 0
        skipped_count = 0
        error_count = 0

        for model_id in models:
            try:
                # Get API credentials for this model
                api_key = os.getenv(f'ALPACA_{model_id.upper()}_API_KEY')
                secret_key = os.getenv(f'ALPACA_{model_id.upper()}_SECRET_KEY')

                if not api_key or not secret_key:
                    logger.debug(f"No credentials for {model_id}, skipping")
                    skipped_count += 1
                    continue

                # Import Alpaca SDK (lazy import to avoid errors if not installed)
                try:
                    from alpaca.trading.client import TradingClient
                except ImportError as e:
                    logger.error(f"Alpaca SDK not installed: {e}")
                    error_count += 1
                    continue

                # Get account info
                client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
                account = client.get_account()
                positions = client.get_all_positions()

                # Insert snapshot
                cursor.execute("""
                    INSERT OR IGNORE INTO equity_snapshots
                    (model_id, timestamp, equity, cash, portfolio_value, buying_power, position_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_id,
                    timestamp,
                    float(account.equity),
                    float(account.cash),
                    float(account.portfolio_value),
                    float(account.buying_power),
                    len(positions)
                ))

                logger.info(f"✓ Logged {model_id}: ${float(account.equity):.2f} ({len(positions)} positions)")
                logged_count += 1

            except Exception as e:
                logger.error(f"✗ Error logging equity for {model_id}: {e}")
                error_count += 1
                continue

        conn.commit()
        conn.close()

        # Summary
        logger.info(
            f"Equity snapshot complete: "
            f"{logged_count} logged, {skipped_count} skipped, {error_count} errors"
        )

        return logged_count > 0

    except Exception as e:
        logger.error(f"Critical error in equity logging: {e}")
        return False


def main():
    """Main entry point for cron job"""
    logger.info("=" * 60)
    logger.info("Equity Logger Cron Job Starting")
    logger.info("=" * 60)

    # Initialize database if needed
    if not init_equity_log_db():
        logger.error("Failed to initialize database, exiting")
        sys.exit(1)

    # Log equity snapshots
    success = log_equity_snapshot()

    if success:
        logger.info("Equity logging completed successfully")
        sys.exit(0)
    else:
        logger.warning("Equity logging completed with no data logged")
        sys.exit(0)  # Exit 0 even if no data, to avoid cron retry spam


if __name__ == "__main__":
    main()
