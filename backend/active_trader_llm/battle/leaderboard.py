"""
Leaderboard: Rankings and competitive standings for trading battle.

This module handles:
- Model rankings across different timeframes
- Leaderboard entries with comprehensive metrics
- Ranking logic with tiebreakers
- Daily, weekly, monthly, and all-time leaderboards
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .metrics_engine import MetricsEngine, MetricsSnapshot

logger = logging.getLogger(__name__)


class LeaderboardEntry(BaseModel):
    """Single entry in the leaderboard with all metrics"""
    rank: int = Field(ge=1)
    model_id: str

    # Performance metrics
    total_return_pct: float
    total_return_dollars: float
    current_equity: float

    # Risk metrics
    sharpe_ratio: Optional[float] = None
    max_drawdown_pct: Optional[float] = None

    # Trading metrics
    total_trades: int = Field(ge=0)
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None

    # Position metrics
    open_positions_count: int = Field(ge=0)
    total_exposure_pct: float = Field(ge=0)

    # Metadata
    last_updated: datetime


class Leaderboard:
    """
    Manages leaderboard rankings for trading battle.

    Ranking logic:
    1. Primary: Total return percentage (higher is better)
    2. Tiebreaker 1: Sharpe ratio (higher is better)
    3. Tiebreaker 2: Max drawdown (lower is better)
    """

    def __init__(self, metrics_db_path: str, leaderboard_db_path: Optional[str] = None):
        """
        Initialize Leaderboard with database connections.

        Args:
            metrics_db_path: Path to metrics timeseries database
            leaderboard_db_path: Path to leaderboard database (defaults to same as metrics)
        """
        self.metrics_db_path = metrics_db_path
        self.leaderboard_db_path = leaderboard_db_path or metrics_db_path
        self.metrics_engine = MetricsEngine(metrics_db_path)

        self._ensure_database()
        logger.info(f"Leaderboard initialized with database: {self.leaderboard_db_path}")

    def _ensure_database(self):
        """Create leaderboard tables if they don't exist"""
        db_file = Path(self.leaderboard_db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.leaderboard_db_path)
        cursor = conn.cursor()

        # Leaderboard snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS leaderboard_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                timeframe TEXT NOT NULL CHECK(timeframe IN ('daily', 'weekly', 'monthly', 'all_time')),
                snapshot_time TIMESTAMP NOT NULL,
                rank INTEGER NOT NULL CHECK(rank >= 1),
                total_return_pct REAL NOT NULL,
                total_return_dollars REAL NOT NULL,
                current_equity REAL NOT NULL CHECK(current_equity > 0),
                sharpe_ratio REAL,
                max_drawdown_pct REAL,
                total_trades INTEGER NOT NULL CHECK(total_trades >= 0),
                win_rate REAL,
                profit_factor REAL,
                open_positions_count INTEGER NOT NULL CHECK(open_positions_count >= 0),
                total_exposure_pct REAL NOT NULL CHECK(total_exposure_pct >= 0),
                UNIQUE(model_id, timeframe, snapshot_time)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_leaderboard_timeframe_time
            ON leaderboard_snapshots(timeframe, snapshot_time DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_leaderboard_model_timeframe
            ON leaderboard_snapshots(model_id, timeframe, snapshot_time DESC)
        """)

        # Current standings table (updated in real-time)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS current_standings (
                model_id TEXT PRIMARY KEY,
                total_return_pct REAL NOT NULL,
                total_return_dollars REAL NOT NULL,
                current_equity REAL NOT NULL CHECK(current_equity > 0),
                sharpe_ratio REAL,
                max_drawdown_pct REAL,
                total_trades INTEGER NOT NULL CHECK(total_trades >= 0),
                win_rate REAL,
                profit_factor REAL,
                open_positions_count INTEGER NOT NULL CHECK(open_positions_count >= 0),
                total_exposure_pct REAL NOT NULL CHECK(total_exposure_pct >= 0),
                last_updated TIMESTAMP NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def update_standings(self, metrics: MetricsSnapshot):
        """
        Update current standings for a model.

        Args:
            metrics: MetricsSnapshot for the model
        """
        conn = sqlite3.connect(self.leaderboard_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO current_standings (
                model_id, total_return_pct, total_return_dollars, current_equity,
                sharpe_ratio, max_drawdown_pct, total_trades, win_rate, profit_factor,
                open_positions_count, total_exposure_pct, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
                total_return_pct = excluded.total_return_pct,
                total_return_dollars = excluded.total_return_dollars,
                current_equity = excluded.current_equity,
                sharpe_ratio = excluded.sharpe_ratio,
                max_drawdown_pct = excluded.max_drawdown_pct,
                total_trades = excluded.total_trades,
                win_rate = excluded.win_rate,
                profit_factor = excluded.profit_factor,
                open_positions_count = excluded.open_positions_count,
                total_exposure_pct = excluded.total_exposure_pct,
                last_updated = excluded.last_updated
        """, (
            metrics.model_id,
            metrics.total_return_pct,
            metrics.total_return_dollars,
            metrics.current_equity,
            metrics.sharpe_ratio,
            metrics.max_drawdown_pct,
            metrics.total_trades,
            metrics.win_rate,
            metrics.profit_factor,
            metrics.open_positions_count,
            metrics.total_exposure_pct,
            metrics.timestamp
        ))

        conn.commit()
        conn.close()

        logger.info(f"Updated standings for {metrics.model_id}: {metrics.total_return_pct:.2f}% return")

    def _sort_entries(self, entries: List[Dict]) -> List[Dict]:
        """
        Sort leaderboard entries by ranking logic.

        Ranking logic:
        1. Primary: Total return percentage (higher is better)
        2. Tiebreaker 1: Sharpe ratio (higher is better)
        3. Tiebreaker 2: Max drawdown (lower is better - less negative)

        Args:
            entries: List of entry dictionaries

        Returns:
            Sorted list with ranks assigned
        """
        def sort_key(entry):
            # Primary: total return (descending)
            primary = -entry['total_return_pct']

            # Tiebreaker 1: Sharpe ratio (descending, None treated as -inf)
            sharpe = entry.get('sharpe_ratio')
            tiebreak1 = -sharpe if sharpe is not None else float('inf')

            # Tiebreaker 2: Max drawdown (ascending - lower drawdown is better)
            drawdown = entry.get('max_drawdown_pct')
            tiebreak2 = drawdown if drawdown is not None else float('inf')

            return (primary, tiebreak1, tiebreak2)

        sorted_entries = sorted(entries, key=sort_key)

        # Assign ranks
        for i, entry in enumerate(sorted_entries, start=1):
            entry['rank'] = i

        return sorted_entries

    def get_rankings(self, timeframe: str = 'all_time') -> List[LeaderboardEntry]:
        """
        Get current leaderboard rankings.

        Args:
            timeframe: 'daily', 'weekly', 'monthly', or 'all_time'

        Returns:
            List of LeaderboardEntry objects sorted by rank
        """
        conn = sqlite3.connect(self.leaderboard_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # For now, use current_standings (future: filter by timeframe)
        cursor.execute("""
            SELECT * FROM current_standings
            ORDER BY total_return_pct DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        # Convert to dictionaries and sort
        entries = []
        for row in rows:
            entries.append({
                'model_id': row['model_id'],
                'total_return_pct': row['total_return_pct'],
                'total_return_dollars': row['total_return_dollars'],
                'current_equity': row['current_equity'],
                'sharpe_ratio': row['sharpe_ratio'],
                'max_drawdown_pct': row['max_drawdown_pct'],
                'total_trades': row['total_trades'],
                'win_rate': row['win_rate'],
                'profit_factor': row['profit_factor'],
                'open_positions_count': row['open_positions_count'],
                'total_exposure_pct': row['total_exposure_pct'],
                'last_updated': datetime.fromisoformat(row['last_updated']) if isinstance(row['last_updated'], str) else row['last_updated']
            })

        sorted_entries = self._sort_entries(entries)

        # Convert to LeaderboardEntry objects
        leaderboard = [
            LeaderboardEntry(**entry)
            for entry in sorted_entries
        ]

        return leaderboard

    def get_daily_leaderboard(self) -> List[LeaderboardEntry]:
        """Get daily leaderboard (last 24 hours)"""
        return self.get_rankings('daily')

    def get_weekly_leaderboard(self) -> List[LeaderboardEntry]:
        """Get weekly leaderboard (last 7 days)"""
        return self.get_rankings('weekly')

    def get_monthly_leaderboard(self) -> List[LeaderboardEntry]:
        """Get monthly leaderboard (last 30 days)"""
        return self.get_rankings('monthly')

    def get_all_time_leaderboard(self) -> List[LeaderboardEntry]:
        """Get all-time leaderboard"""
        return self.get_rankings('all_time')

    def get_model_rank(self, model_id: str, timeframe: str = 'all_time') -> Optional[int]:
        """
        Get rank for a specific model.

        Args:
            model_id: Model identifier
            timeframe: 'daily', 'weekly', 'monthly', or 'all_time'

        Returns:
            Rank (1-indexed) or None if model not found
        """
        rankings = self.get_rankings(timeframe)

        for entry in rankings:
            if entry.model_id == model_id:
                return entry.rank

        return None

    def snapshot_leaderboard(self, timeframe: str = 'all_time'):
        """
        Save current leaderboard as historical snapshot.

        Args:
            timeframe: Timeframe for this snapshot
        """
        rankings = self.get_rankings(timeframe)
        snapshot_time = datetime.now()

        conn = sqlite3.connect(self.leaderboard_db_path)
        cursor = conn.cursor()

        for entry in rankings:
            cursor.execute("""
                INSERT INTO leaderboard_snapshots (
                    model_id, timeframe, snapshot_time, rank,
                    total_return_pct, total_return_dollars, current_equity,
                    sharpe_ratio, max_drawdown_pct, total_trades,
                    win_rate, profit_factor, open_positions_count, total_exposure_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.model_id, timeframe, snapshot_time, entry.rank,
                entry.total_return_pct, entry.total_return_dollars, entry.current_equity,
                entry.sharpe_ratio, entry.max_drawdown_pct, entry.total_trades,
                entry.win_rate, entry.profit_factor, entry.open_positions_count,
                entry.total_exposure_pct
            ))

        conn.commit()
        conn.close()

        logger.info(f"Saved {timeframe} leaderboard snapshot with {len(rankings)} models")

    def get_historical_rankings(
        self,
        model_id: str,
        timeframe: str = 'all_time',
        limit: int = 30
    ) -> List[Dict]:
        """
        Get historical rank progression for a model.

        Args:
            model_id: Model identifier
            timeframe: Timeframe to query
            limit: Maximum number of snapshots

        Returns:
            List of dictionaries with timestamp and rank
        """
        conn = sqlite3.connect(self.leaderboard_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT snapshot_time, rank, total_return_pct
            FROM leaderboard_snapshots
            WHERE model_id = ? AND timeframe = ?
            ORDER BY snapshot_time DESC
            LIMIT ?
        """, (model_id, timeframe, limit))

        rows = cursor.fetchall()
        conn.close()

        history = [
            {
                'timestamp': datetime.fromisoformat(row['snapshot_time']) if isinstance(row['snapshot_time'], str) else row['snapshot_time'],
                'rank': row['rank'],
                'return_pct': row['total_return_pct']
            }
            for row in rows
        ]

        return history


# Example usage
if __name__ == "__main__":
    import os
    from metrics_engine import MetricsSnapshot

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Initialize leaderboard
    db_path = "/tmp/test_leaderboard.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    leaderboard = Leaderboard(db_path)

    # Create sample metrics for different models
    models_metrics = [
        MetricsSnapshot(
            model_id="claude-sonnet-3.5",
            timestamp=datetime.now(),
            current_equity=110000,
            initial_capital=100000,
            total_return_pct=10.0,
            total_return_dollars=10000,
            sharpe_ratio=1.8,
            max_drawdown_pct=5.0,
            max_drawdown_dollars=5000,
            total_trades=25,
            winning_trades=18,
            losing_trades=7,
            win_rate=72.0,
            profit_factor=2.5,
            open_positions_count=3,
            total_exposure_dollars=30000,
            total_exposure_pct=27.27,
            daily_pnl=500,
            daily_return_pct=0.45
        ),
        MetricsSnapshot(
            model_id="gpt-4o",
            timestamp=datetime.now(),
            current_equity=115000,
            initial_capital=100000,
            total_return_pct=15.0,
            total_return_dollars=15000,
            sharpe_ratio=1.5,
            max_drawdown_pct=8.0,
            max_drawdown_dollars=8000,
            total_trades=30,
            winning_trades=20,
            losing_trades=10,
            win_rate=66.67,
            profit_factor=2.0,
            open_positions_count=4,
            total_exposure_dollars=40000,
            total_exposure_pct=34.78,
            daily_pnl=800,
            daily_return_pct=0.70
        ),
        MetricsSnapshot(
            model_id="gemini-pro",
            timestamp=datetime.now(),
            current_equity=108000,
            initial_capital=100000,
            total_return_pct=8.0,
            total_return_dollars=8000,
            sharpe_ratio=2.0,
            max_drawdown_pct=3.0,
            max_drawdown_dollars=3000,
            total_trades=20,
            winning_trades=15,
            losing_trades=5,
            win_rate=75.0,
            profit_factor=3.0,
            open_positions_count=2,
            total_exposure_dollars=20000,
            total_exposure_pct=18.52,
            daily_pnl=300,
            daily_return_pct=0.28
        )
    ]

    # Update standings
    print("\n=== Updating Standings ===")
    for metrics in models_metrics:
        leaderboard.update_standings(metrics)
        print(f"Updated {metrics.model_id}: {metrics.total_return_pct}% return")

    # Get rankings
    print("\n=== Current Leaderboard ===")
    rankings = leaderboard.get_rankings()

    for entry in rankings:
        print(f"\n#{entry.rank} - {entry.model_id}")
        print(f"  Return: {entry.total_return_pct:.2f}% (${entry.total_return_dollars:,.2f})")
        print(f"  Sharpe: {entry.sharpe_ratio:.2f}" if entry.sharpe_ratio else "  Sharpe: N/A")
        print(f"  Max DD: {entry.max_drawdown_pct:.2f}%" if entry.max_drawdown_pct else "  Max DD: N/A")
        print(f"  Win Rate: {entry.win_rate:.2f}%" if entry.win_rate else "  Win Rate: N/A")
        print(f"  Trades: {entry.total_trades}")

    # Get specific rank
    print("\n=== Model Rank ===")
    rank = leaderboard.get_model_rank("claude-sonnet-3.5")
    print(f"claude-sonnet-3.5 rank: #{rank}")

    # Snapshot
    print("\n=== Saving Snapshot ===")
    leaderboard.snapshot_leaderboard('all_time')
    print("Snapshot saved")

    print("\n=== Demo Complete ===")
