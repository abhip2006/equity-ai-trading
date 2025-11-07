"""
Position Manager: Tracks and manages position lifecycle (open, monitor, close).

This module handles all position tracking and P&L calculations:
- Opening new positions
- Closing positions with realized P&L
- Tracking unrealized P&L for open positions
- Portfolio state calculations
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Position(BaseModel):
    """Position model matching the database schema"""
    id: Optional[int] = None
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)
    shares: int = Field(gt=0)
    position_size_pct: Optional[float] = Field(default=None, gt=0, le=100)
    strategy: Optional[str] = None
    opened_at: datetime
    closed_at: Optional[datetime] = None
    exit_price: Optional[float] = Field(default=None, gt=0)
    exit_reason: Optional[str] = None  # 'stop_loss', 'take_profit', 'manual', 'eod_close'
    realized_pnl: Optional[float] = None
    status: str = 'open'  # 'open' or 'closed'


class PositionManager:
    """
    Manages position lifecycle: open, monitor, close
    Tracks positions in SQLite database
    Calculates P&L (realized and unrealized)
    """

    def __init__(self, db_path: str):
        """
        Initialize PositionManager with database connection

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_database()
        logger.info(f"PositionManager initialized with database: {db_path}")

    def _ensure_database(self):
        """Create database and tables if they don't exist"""
        # Ensure parent directory exists
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        # Connect and create tables
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Read schema from SQL file if it exists, otherwise define inline
        schema_path = Path(__file__).parent.parent / 'database' / 'position_schema.sql'

        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            cursor.executescript(schema_sql)
            logger.info("Created database tables from position_schema.sql")
        else:
            # Fallback inline schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL CHECK(direction IN ('long', 'short')),
                    entry_price REAL NOT NULL CHECK(entry_price > 0),
                    stop_loss REAL NOT NULL CHECK(stop_loss > 0),
                    take_profit REAL NOT NULL CHECK(take_profit > 0),
                    shares INTEGER NOT NULL CHECK(shares > 0),
                    position_size_pct REAL CHECK(position_size_pct > 0 AND position_size_pct <= 100),
                    strategy TEXT,
                    opened_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    closed_at TIMESTAMP,
                    exit_price REAL CHECK(exit_price > 0),
                    exit_reason TEXT CHECK(exit_reason IN ('stop_loss', 'take_profit', 'manual', 'eod_close', NULL)),
                    realized_pnl REAL,
                    status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open', 'closed'))
                )
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_opened_at ON positions(opened_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol_status ON positions(symbol, status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy)")

            logger.info("Created database tables from inline schema")

        conn.commit()
        conn.close()

    def open_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        shares: int,
        strategy: Optional[str] = None,
        position_size_pct: Optional[float] = None,
        fill_timestamp: Optional[datetime] = None
    ) -> Position:
        """
        Record new position in database

        Args:
            symbol: Stock ticker symbol
            direction: 'long' or 'short'
            entry_price: Fill price for entry
            stop_loss: Stop loss price
            take_profit: Take profit price
            shares: Number of shares
            strategy: Strategy name (optional)
            position_size_pct: Position size as percentage of portfolio (optional)
            fill_timestamp: Timestamp of fill (defaults to now)

        Returns:
            Position object with database ID
        """
        opened_at = fill_timestamp or datetime.now()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO positions (
                symbol, direction, entry_price, stop_loss, take_profit,
                shares, position_size_pct, strategy, opened_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
        """, (
            symbol, direction, entry_price, stop_loss, take_profit,
            shares, position_size_pct, strategy, opened_at
        ))

        position_id = cursor.lastrowid
        conn.commit()
        conn.close()

        position = Position(
            id=position_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            shares=shares,
            position_size_pct=position_size_pct,
            strategy=strategy,
            opened_at=opened_at,
            status='open'
        )

        logger.info(f"Opened {direction} position: {symbol} @ ${entry_price:.2f} x {shares} shares (ID: {position_id})")
        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        exit_timestamp: Optional[datetime] = None
    ) -> Optional[Position]:
        """
        Close position and calculate realized P&L

        Args:
            symbol: Stock ticker symbol
            exit_price: Exit/fill price
            exit_reason: Reason for exit ('stop_loss', 'take_profit', 'manual', 'eod_close')
            exit_timestamp: Timestamp of exit (defaults to now)

        Returns:
            Updated Position object or None if no open position found
        """
        closed_at = exit_timestamp or datetime.now()

        # Get open position
        position = self.get_position(symbol)
        if not position:
            logger.warning(f"No open position found for {symbol}")
            return None

        # Calculate realized P&L
        realized_pnl = self.calculate_realized_pnl(position, exit_price)

        # Update database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE positions
            SET closed_at = ?,
                exit_price = ?,
                exit_reason = ?,
                realized_pnl = ?,
                status = 'closed'
            WHERE symbol = ? AND status = 'open'
        """, (closed_at, exit_price, exit_reason, realized_pnl, symbol))

        conn.commit()
        conn.close()

        # Return updated position
        position.closed_at = closed_at
        position.exit_price = exit_price
        position.exit_reason = exit_reason
        position.realized_pnl = realized_pnl
        position.status = 'closed'

        logger.info(
            f"Closed {position.direction} position: {symbol} @ ${exit_price:.2f} "
            f"| P&L: ${realized_pnl:.2f} | Reason: {exit_reason}"
        )

        return position

    def get_open_positions(self) -> List[Position]:
        """
        Get all open positions

        Returns:
            List of Position objects with status='open'
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM positions
            WHERE status = 'open'
            ORDER BY opened_at DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        positions = [self._row_to_position(row) for row in rows]
        logger.debug(f"Retrieved {len(positions)} open positions")

        return positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get specific open position by symbol

        Args:
            symbol: Stock ticker symbol

        Returns:
            Position object or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM positions
            WHERE symbol = ? AND status = 'open'
            LIMIT 1
        """, (symbol,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_position(row)
        return None

    def has_open_position(self, symbol: str) -> bool:
        """
        Check if position exists for symbol

        Args:
            symbol: Stock ticker symbol

        Returns:
            True if open position exists
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*) FROM positions
            WHERE symbol = ? AND status = 'open'
        """, (symbol,))

        count = cursor.fetchone()[0]
        conn.close()

        return count > 0

    def get_portfolio_state(self, current_prices: Dict[str, float], total_portfolio_value: Optional[float] = None) -> Dict:
        """
        Calculate current portfolio state including unrealized P&L

        Args:
            current_prices: Dictionary mapping symbol -> current price
            total_portfolio_value: Total portfolio equity for calculating exposure percentage (optional)

        Returns:
            Dictionary with portfolio metrics:
            - open_positions: List of positions with unrealized P&L
            - total_exposure: Sum of position values (dollars)
            - total_exposure_pct: Total exposure as percentage of portfolio (decimal, e.g. 0.05 = 5%)
            - total_unrealized_pnl: Sum of unrealized P&L
            - position_count: Number of open positions
        """
        open_positions = self.get_open_positions()

        portfolio_positions = []
        total_exposure = 0.0
        total_unrealized_pnl = 0.0

        for position in open_positions:
            current_price = current_prices.get(position.symbol)

            if current_price is None:
                logger.warning(f"No current price for {position.symbol}, skipping in portfolio state")
                continue

            # Calculate unrealized P&L
            unrealized_pnl = self.calculate_unrealized_pnl(position, current_price)

            # Calculate position value
            position_value = current_price * position.shares

            portfolio_positions.append({
                'symbol': position.symbol,
                'direction': position.direction,
                'shares': position.shares,
                'entry_price': position.entry_price,
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl,
                'position_value': position_value,
                'strategy': position.strategy,
                'opened_at': position.opened_at
            })

            total_exposure += position_value
            total_unrealized_pnl += unrealized_pnl

        # Calculate exposure percentage if portfolio value provided
        total_exposure_pct = 0.0
        if total_portfolio_value and total_portfolio_value > 0:
            total_exposure_pct = total_exposure / total_portfolio_value

        return {
            'open_positions': portfolio_positions,
            'total_exposure': total_exposure,
            'total_exposure_pct': total_exposure_pct,
            'total_unrealized_pnl': total_unrealized_pnl,
            'position_count': len(portfolio_positions)
        }

    def get_closed_positions(self, since: Optional[datetime] = None) -> List[Position]:
        """
        Get closed positions, optionally filtered by date

        Args:
            since: Only return positions closed after this datetime (optional)

        Returns:
            List of closed Position objects
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if since:
            cursor.execute("""
                SELECT * FROM positions
                WHERE status = 'closed' AND closed_at >= ?
                ORDER BY closed_at DESC
            """, (since,))
        else:
            cursor.execute("""
                SELECT * FROM positions
                WHERE status = 'closed'
                ORDER BY closed_at DESC
            """)

        rows = cursor.fetchall()
        conn.close()

        positions = [self._row_to_position(row) for row in rows]
        logger.debug(f"Retrieved {len(positions)} closed positions")

        return positions

    def calculate_unrealized_pnl(self, position: Position, current_price: float) -> float:
        """
        Calculate unrealized P&L for a position

        Args:
            position: Position object
            current_price: Current market price

        Returns:
            Unrealized P&L in dollars
        """
        if position.direction == 'long':
            # Long: profit when price goes up
            pnl = (current_price - position.entry_price) * position.shares
        else:
            # Short: profit when price goes down
            pnl = (position.entry_price - current_price) * position.shares

        return pnl

    def calculate_realized_pnl(self, position: Position, exit_price: float) -> float:
        """
        Calculate realized P&L for a closed position

        Args:
            position: Position object
            exit_price: Exit price

        Returns:
            Realized P&L in dollars
        """
        if position.direction == 'long':
            # Long: profit when exit > entry
            pnl = (exit_price - position.entry_price) * position.shares
        else:
            # Short: profit when exit < entry
            pnl = (position.entry_price - exit_price) * position.shares

        return pnl

    def clear_all_positions(self):
        """
        Clear all positions from the database.
        WARNING: This will permanently delete all position records!
        Use only for testing purposes.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM positions")
            deleted_count = cursor.rowcount
            conn.commit()
            logger.info(f"Cleared {deleted_count} positions from database")
        except Exception as e:
            logger.error(f"Error clearing positions: {e}")
            conn.rollback()
        finally:
            conn.close()

    def _row_to_position(self, row: sqlite3.Row) -> Position:
        """Convert database row to Position object"""
        return Position(
            id=row['id'],
            symbol=row['symbol'],
            direction=row['direction'],
            entry_price=row['entry_price'],
            stop_loss=row['stop_loss'],
            take_profit=row['take_profit'],
            shares=row['shares'],
            position_size_pct=row['position_size_pct'],
            strategy=row['strategy'],
            opened_at=datetime.fromisoformat(row['opened_at']) if isinstance(row['opened_at'], str) else row['opened_at'],
            closed_at=datetime.fromisoformat(row['closed_at']) if row['closed_at'] and isinstance(row['closed_at'], str) else row['closed_at'],
            exit_price=row['exit_price'],
            exit_reason=row['exit_reason'],
            realized_pnl=row['realized_pnl'],
            status=row['status']
        )


# Example usage
if __name__ == "__main__":
    import os
    from datetime import timedelta

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Initialize position manager
    db_path = "/tmp/test_positions.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    pm = PositionManager(db_path)

    # Example 1: Open a long position
    print("\n=== Opening Long Position ===")
    position1 = pm.open_position(
        symbol="AAPL",
        direction="long",
        entry_price=175.00,
        stop_loss=172.00,
        take_profit=180.00,
        shares=100,
        strategy="momentum_breakout",
        position_size_pct=5.0
    )
    print(f"Opened: {position1.symbol} | ID: {position1.id} | Status: {position1.status}")

    # Example 2: Open a short position
    print("\n=== Opening Short Position ===")
    position2 = pm.open_position(
        symbol="TSLA",
        direction="short",
        entry_price=250.00,
        stop_loss=255.00,
        take_profit=240.00,
        shares=50,
        strategy="mean_reversion",
        position_size_pct=3.0
    )
    print(f"Opened: {position2.symbol} | ID: {position2.id} | Status: {position2.status}")

    # Example 3: Check open positions
    print("\n=== Open Positions ===")
    open_positions = pm.get_open_positions()
    for pos in open_positions:
        print(f"{pos.symbol}: {pos.direction} @ ${pos.entry_price:.2f} x {pos.shares} shares")

    # Example 4: Calculate portfolio state
    print("\n=== Portfolio State ===")
    current_prices = {
        "AAPL": 177.50,  # Up $2.50
        "TSLA": 248.00   # Down $2.00
    }
    portfolio = pm.get_portfolio_state(current_prices)
    print(f"Position Count: {portfolio['position_count']}")
    print(f"Total Exposure: ${portfolio['total_exposure']:,.2f}")
    print(f"Total Unrealized P&L: ${portfolio['total_unrealized_pnl']:,.2f}")

    for pos in portfolio['open_positions']:
        print(f"  {pos['symbol']}: ${pos['unrealized_pnl']:,.2f} unrealized")

    # Example 5: Close a position
    print("\n=== Closing Position ===")
    closed_pos = pm.close_position(
        symbol="AAPL",
        exit_price=178.00,
        exit_reason="take_profit"
    )
    if closed_pos:
        print(f"Closed {closed_pos.symbol}: Realized P&L = ${closed_pos.realized_pnl:,.2f}")

    # Example 6: Check remaining open positions
    print("\n=== Remaining Open Positions ===")
    remaining = pm.get_open_positions()
    print(f"Open: {len(remaining)} positions")
    for pos in remaining:
        print(f"  {pos.symbol}: {pos.direction}")

    # Example 7: Check closed positions
    print("\n=== Closed Positions ===")
    closed = pm.get_closed_positions()
    for pos in closed:
        print(f"{pos.symbol}: {pos.exit_reason} | Realized P&L: ${pos.realized_pnl:,.2f}")

    print("\n=== Demo Complete ===")
