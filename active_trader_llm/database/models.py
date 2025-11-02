"""
Position tracking models and database initialization.

This module provides Pydantic models for position management and
database initialization utilities for the position tracking system.
"""

import sqlite3
import logging
from datetime import datetime
from typing import Optional, Literal
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class Position(BaseModel):
    """
    Complete position model matching the database schema.

    Represents both open and closed positions with full lifecycle tracking.
    """
    id: Optional[int] = None
    symbol: str = Field(..., description="Stock ticker symbol")
    direction: Literal['long', 'short'] = Field(..., description="Position direction")
    entry_price: float = Field(..., gt=0, description="Entry price per share")
    stop_loss: float = Field(..., gt=0, description="Stop loss price")
    take_profit: float = Field(..., gt=0, description="Take profit price")
    shares: int = Field(..., gt=0, description="Number of shares")
    position_size_pct: Optional[float] = Field(None, gt=0, le=100, description="Position size as percentage of capital")
    strategy: Optional[str] = Field(None, description="Trading strategy used")
    opened_at: datetime = Field(default_factory=datetime.now, description="When position was opened")
    closed_at: Optional[datetime] = Field(None, description="When position was closed")
    exit_price: Optional[float] = Field(None, gt=0, description="Exit price per share")
    exit_reason: Optional[Literal['stop_loss', 'take_profit', 'manual', 'eod_close']] = Field(
        None, description="Reason for closing position"
    )
    realized_pnl: Optional[float] = Field(None, description="Realized profit/loss in dollars")
    status: Literal['open', 'closed'] = Field(default='open', description="Position status")

    @field_validator('stop_loss')
    @classmethod
    def validate_stop_loss(cls, v: float, info) -> float:
        """Validate stop loss is appropriate for position direction."""
        if 'direction' in info.data and 'entry_price' in info.data:
            direction = info.data['direction']
            entry = info.data['entry_price']

            if direction == 'long' and v >= entry:
                raise ValueError(f"Long position stop loss ({v}) must be below entry price ({entry})")
            elif direction == 'short' and v <= entry:
                raise ValueError(f"Short position stop loss ({v}) must be above entry price ({entry})")
        return v

    @field_validator('take_profit')
    @classmethod
    def validate_take_profit(cls, v: float, info) -> float:
        """Validate take profit is appropriate for position direction."""
        if 'direction' in info.data and 'entry_price' in info.data:
            direction = info.data['direction']
            entry = info.data['entry_price']

            if direction == 'long' and v <= entry:
                raise ValueError(f"Long position take profit ({v}) must be above entry price ({entry})")
            elif direction == 'short' and v >= entry:
                raise ValueError(f"Short position take profit ({v}) must be below entry price ({entry})")
        return v

    @field_validator('status')
    @classmethod
    def validate_status_consistency(cls, v: str, info) -> str:
        """Validate that closed positions have required exit data."""
        if v == 'closed':
            if 'closed_at' in info.data and info.data['closed_at'] is None:
                raise ValueError("Closed positions must have closed_at timestamp")
            if 'exit_price' in info.data and info.data['exit_price'] is None:
                raise ValueError("Closed positions must have exit_price")
            if 'exit_reason' in info.data and info.data['exit_reason'] is None:
                raise ValueError("Closed positions must have exit_reason")
        return v

    def calculate_pnl(self) -> float:
        """
        Calculate position P&L.

        Returns:
            Realized P&L for closed positions, or unrealized P&L if exit_price provided
        """
        if self.exit_price is None:
            raise ValueError("Cannot calculate P&L without exit_price")

        if self.direction == 'long':
            pnl = (self.exit_price - self.entry_price) * self.shares
        else:  # short
            pnl = (self.entry_price - self.exit_price) * self.shares

        return pnl

    def risk_reward_ratio(self) -> float:
        """
        Calculate risk/reward ratio for the position.

        Returns:
            R:R ratio (potential reward divided by risk)
        """
        if self.direction == 'long':
            risk = self.entry_price - self.stop_loss
            reward = self.take_profit - self.entry_price
        else:  # short
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.take_profit

        if risk <= 0:
            raise ValueError("Invalid risk calculation - check stop loss placement")

        return reward / risk


class PositionCreate(BaseModel):
    """
    Model for creating/opening a new position.

    Used when the trader agent decides to enter a new position.
    """
    symbol: str = Field(..., description="Stock ticker symbol")
    direction: Literal['long', 'short'] = Field(..., description="Position direction")
    entry_price: float = Field(..., gt=0, description="Entry price per share")
    stop_loss: float = Field(..., gt=0, description="Stop loss price")
    take_profit: float = Field(..., gt=0, description="Take profit price")
    shares: int = Field(..., gt=0, description="Number of shares")
    position_size_pct: Optional[float] = Field(None, gt=0, le=100, description="Position size as percentage of capital")
    strategy: Optional[str] = Field(None, description="Trading strategy used")

    @field_validator('stop_loss')
    @classmethod
    def validate_stop_loss(cls, v: float, info) -> float:
        """Validate stop loss is appropriate for position direction."""
        if 'direction' in info.data and 'entry_price' in info.data:
            direction = info.data['direction']
            entry = info.data['entry_price']

            if direction == 'long' and v >= entry:
                raise ValueError(f"Long position stop loss ({v}) must be below entry price ({entry})")
            elif direction == 'short' and v <= entry:
                raise ValueError(f"Short position stop loss ({v}) must be above entry price ({entry})")
        return v

    @field_validator('take_profit')
    @classmethod
    def validate_take_profit(cls, v: float, info) -> float:
        """Validate take profit is appropriate for position direction."""
        if 'direction' in info.data and 'entry_price' in info.data:
            direction = info.data['direction']
            entry = info.data['entry_price']

            if direction == 'long' and v <= entry:
                raise ValueError(f"Long position take profit ({v}) must be above entry price ({entry})")
            elif direction == 'short' and v >= entry:
                raise ValueError(f"Short position take profit ({v}) must be below entry price ({entry})")
        return v

    def to_position(self) -> Position:
        """Convert PositionCreate to full Position model."""
        return Position(
            symbol=self.symbol,
            direction=self.direction,
            entry_price=self.entry_price,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            shares=self.shares,
            position_size_pct=self.position_size_pct,
            strategy=self.strategy,
            status='open'
        )


class PositionClose(BaseModel):
    """
    Model for closing an existing position.

    Used when the trader agent or risk manager decides to exit a position.
    """
    position_id: int = Field(..., gt=0, description="ID of position to close")
    exit_price: float = Field(..., gt=0, description="Exit price per share")
    exit_reason: Literal['stop_loss', 'take_profit', 'manual', 'eod_close'] = Field(
        ..., description="Reason for closing position"
    )
    closed_at: datetime = Field(default_factory=datetime.now, description="When position was closed")


def init_positions_db(db_path: str) -> None:
    """
    Initialize the positions database by creating tables from schema.

    Args:
        db_path: Path to SQLite database file

    Raises:
        sqlite3.Error: If database initialization fails
    """
    # Ensure parent directory exists
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    # Get path to schema file
    schema_path = Path(__file__).parent / 'position_schema.sql'

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    # Read schema
    with open(schema_path, 'r') as f:
        schema_sql = f.read()

    # Initialize database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Execute schema (supports multiple statements)
        cursor.executescript(schema_sql)

        conn.commit()
        logger.info(f"Database initialized successfully at {db_path}")

        # Verify tables were created
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        logger.info(f"Created tables: {[t[0] for t in tables]}")

        # Verify indexes were created
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = cursor.fetchall()
        logger.info(f"Created indexes: {[i[0] for i in indexes]}")

    except sqlite3.Error as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    finally:
        conn.close()


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test Position model
    print("=" * 60)
    print("Testing Position model")
    print("=" * 60)

    pos = Position(
        symbol="AAPL",
        direction="long",
        entry_price=175.0,
        stop_loss=172.0,
        take_profit=180.0,
        shares=100,
        position_size_pct=5.0,
        strategy="momentum_breakout"
    )

    print(f"\nCreated position: {pos.symbol} {pos.direction}")
    print(f"Entry: ${pos.entry_price}, SL: ${pos.stop_loss}, TP: ${pos.take_profit}")
    print(f"Risk/Reward: {pos.risk_reward_ratio():.2f}")

    # Test PositionCreate model
    print("\n" + "=" * 60)
    print("Testing PositionCreate model")
    print("=" * 60)

    pos_create = PositionCreate(
        symbol="MSFT",
        direction="short",
        entry_price=380.0,
        stop_loss=385.0,
        take_profit=370.0,
        shares=50,
        position_size_pct=3.0,
        strategy="reversal"
    )

    print(f"\nPosition to create: {pos_create.symbol} {pos_create.direction}")
    print(f"Entry: ${pos_create.entry_price}, SL: ${pos_create.stop_loss}, TP: ${pos_create.take_profit}")

    # Convert to full Position
    full_pos = pos_create.to_position()
    print(f"Converted to Position with status: {full_pos.status}")

    # Test PositionClose model
    print("\n" + "=" * 60)
    print("Testing PositionClose model")
    print("=" * 60)

    pos_close = PositionClose(
        position_id=1,
        exit_price=180.0,
        exit_reason="take_profit"
    )

    print(f"\nClosing position {pos_close.position_id}")
    print(f"Exit price: ${pos_close.exit_price}, Reason: {pos_close.exit_reason}")

    # Calculate P&L
    pos.exit_price = pos_close.exit_price
    pnl = pos.calculate_pnl()
    print(f"Realized P&L: ${pnl:.2f}")

    # Test database initialization
    print("\n" + "=" * 60)
    print("Testing database initialization")
    print("=" * 60)

    import tempfile
    import os

    # Create temp database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test_positions.db')
        print(f"\nInitializing test database at: {db_path}")

        init_positions_db(db_path)

        # Verify database was created
        assert os.path.exists(db_path), "Database file was not created"
        print(f"\nDatabase successfully created and verified!")

        # Test inserting a position
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO positions (
                symbol, direction, entry_price, stop_loss, take_profit,
                shares, position_size_pct, strategy, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pos.symbol, pos.direction, pos.entry_price, pos.stop_loss,
            pos.take_profit, pos.shares, pos.position_size_pct,
            pos.strategy, pos.status
        ))

        conn.commit()

        # Query it back
        cursor.execute("SELECT * FROM positions WHERE symbol = ?", (pos.symbol,))
        row = cursor.fetchone()

        print(f"\nInserted and retrieved position from database:")
        print(f"  ID: {row[0]}, Symbol: {row[1]}, Status: {row[14]}")

        conn.close()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
