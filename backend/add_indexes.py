#!/usr/bin/env python3
"""
Add performance indexes to the battle database

This script adds optimized indexes to frequently queried tables:
- Positions table: compound index on (model_id, status) and single index on symbol
- Trades table: indexes on model_id and opened_at (descending)
- Metrics table: compound index on (model_id, timestamp)
- Decisions table: compound index on (model_id, timestamp)
- Equity curve table: index on model_id and timestamp

Expected performance improvement: 20-30% for common queries
"""

import sqlite3
import sys
from pathlib import Path


def add_indexes(db_path: str = "data/battle.db"):
    """
    Add performance indexes to battle database

    Args:
        db_path: Path to the battle.db database file
    """
    # Validate database exists
    db_file = Path(db_path)
    if not db_file.exists():
        print(f"Error: Database file not found at {db_path}")
        sys.exit(1)

    try:
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()

        # Define indexes with descriptive names
        indexes = [
            # Positions table indexes
            (
                "idx_positions_model_status",
                "CREATE INDEX IF NOT EXISTS idx_positions_model_status ON battle_positions(model_id, status)"
            ),
            (
                "idx_positions_symbol",
                "CREATE INDEX IF NOT EXISTS idx_positions_symbol ON battle_positions(symbol)"
            ),

            # Trades table indexes
            (
                "idx_trades_model",
                "CREATE INDEX IF NOT EXISTS idx_trades_model ON battle_trades(model_id)"
            ),
            (
                "idx_trades_timestamp",
                "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON battle_trades(opened_at DESC)"
            ),

            # Metrics table indexes
            (
                "idx_metrics_model_timestamp",
                "CREATE INDEX IF NOT EXISTS idx_metrics_model_timestamp ON battle_metrics(model_id, timestamp DESC)"
            ),

            # Decisions table indexes
            (
                "idx_decisions_model_timestamp",
                "CREATE INDEX IF NOT EXISTS idx_decisions_model_timestamp ON battle_decisions(model_id, timestamp DESC)"
            ),

            # Equity curve table indexes
            (
                "idx_equity_timestamp",
                "CREATE INDEX IF NOT EXISTS idx_equity_timestamp ON battle_equity_curve(model_id, timestamp DESC)"
            ),
        ]

        print("Adding indexes to battle database...")
        print(f"Database: {db_file.absolute()}")
        print("-" * 60)

        success_count = 0
        for name, sql in indexes:
            try:
                cursor.execute(sql)
                print(f"✓ Added index: {name}")
                success_count += 1
            except sqlite3.OperationalError as e:
                print(f"✗ Failed to add index {name}: {e}")
            except Exception as e:
                print(f"✗ Unexpected error for index {name}: {e}")

        conn.commit()
        conn.close()

        print("-" * 60)
        print(f"\nIndexing complete!")
        print(f"Successfully added {success_count}/{len(indexes)} indexes")

        if success_count == len(indexes):
            print("\nAll indexes added successfully!")
            print("Expected performance improvement: 20-30% for common queries")
            return 0
        else:
            print(f"\nWarning: {len(indexes) - success_count} index(es) failed to create")
            return 1

    except sqlite3.DatabaseError as e:
        print(f"Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Support passing database path as command line argument
    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/battle.db"
    exit_code = add_indexes(db_path)
    sys.exit(exit_code)
