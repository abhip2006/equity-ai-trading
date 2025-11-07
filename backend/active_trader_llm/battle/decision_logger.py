"""
Decision Logger: Transparency logging for all trading decisions.

This module handles:
- Logging every trade decision with reasoning
- Storing LLM reasoning and decision context
- Querying decisions by model, symbol, or timeframe
- Comparing decisions across models for same symbol/time
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Decision(BaseModel):
    """Individual trading decision with reasoning"""
    id: Optional[int] = None
    model_id: str
    symbol: str
    decision: str  # 'buy', 'sell', 'hold', 'close'
    reasoning: str  # Full LLM reasoning text
    confidence: Optional[float] = Field(default=None, ge=0, le=100)  # 0-100 percentage

    # Decision context
    timestamp: datetime
    current_price: Optional[float] = Field(default=None, gt=0)
    position_size: Optional[float] = Field(default=None, ge=0)  # Dollars or shares

    # Trade parameters (if buy decision)
    entry_price: Optional[float] = Field(default=None, gt=0)
    stop_loss: Optional[float] = Field(default=None, gt=0)
    take_profit: Optional[float] = Field(default=None, gt=0)
    direction: Optional[str] = None  # 'long' or 'short'

    # Execution tracking
    executed: bool = False
    execution_price: Optional[float] = Field(default=None, gt=0)
    execution_time: Optional[datetime] = None
    execution_status: Optional[str] = None  # 'filled', 'rejected', 'pending', 'cancelled'


class DecisionLogger:
    """
    Logs and retrieves trading decisions for transparency.

    Stores:
    - Every decision with full reasoning
    - Execution status
    - Indexed for efficient queries
    """

    def __init__(self, db_path: str):
        """
        Initialize DecisionLogger with database connection.

        Args:
            db_path: Path to decision log database
        """
        self.db_path = db_path
        self._ensure_database()
        logger.info(f"DecisionLogger initialized with database: {db_path}")

    def _ensure_database(self):
        """Create database tables if they don't exist"""
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Read schema if available
        schema_path = Path(__file__).parent / 'schema' / 'decision_schema.sql'

        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            cursor.executescript(schema_sql)
            logger.info("Created decision log database from schema file")
        else:
            # Fallback inline schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS decision_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    decision TEXT NOT NULL CHECK(decision IN ('buy', 'sell', 'hold', 'close')),
                    reasoning TEXT NOT NULL,
                    confidence REAL CHECK(confidence >= 0 AND confidence <= 100),
                    timestamp TIMESTAMP NOT NULL,
                    current_price REAL CHECK(current_price > 0),
                    position_size REAL CHECK(position_size >= 0),
                    entry_price REAL CHECK(entry_price > 0),
                    stop_loss REAL CHECK(stop_loss > 0),
                    take_profit REAL CHECK(take_profit > 0),
                    direction TEXT CHECK(direction IN ('long', 'short', NULL)),
                    executed BOOLEAN NOT NULL DEFAULT 0,
                    execution_price REAL CHECK(execution_price > 0),
                    execution_time TIMESTAMP,
                    execution_status TEXT CHECK(execution_status IN ('filled', 'rejected', 'pending', 'cancelled', NULL))
                )
            """)

            # Indexes for efficient queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_decision_model_time
                ON decision_log(model_id, timestamp DESC)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_decision_symbol_time
                ON decision_log(symbol, timestamp DESC)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_decision_model_symbol
                ON decision_log(model_id, symbol, timestamp DESC)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_decision_executed
                ON decision_log(executed, timestamp DESC)
            """)

            logger.info("Created decision log database with inline schema")

        conn.commit()
        conn.close()

    def log_decision(
        self,
        model_id: str,
        symbol: str,
        decision: str,
        reasoning: str,
        confidence: Optional[float] = None,
        current_price: Optional[float] = None,
        position_size: Optional[float] = None,
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        direction: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> int:
        """
        Log a trading decision.

        Args:
            model_id: Model making the decision
            symbol: Stock symbol
            decision: Decision type ('buy', 'sell', 'hold', 'close')
            reasoning: Full reasoning text from LLM
            confidence: Confidence level (0-100)
            current_price: Current market price
            position_size: Position size in dollars or shares
            entry_price: Entry price for buy decision
            stop_loss: Stop loss price
            take_profit: Take profit price
            direction: 'long' or 'short' for buy decision
            timestamp: Decision timestamp (defaults to now)

        Returns:
            Decision ID
        """
        decision_time = timestamp or datetime.now()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO decision_log (
                model_id, symbol, decision, reasoning, confidence, timestamp,
                current_price, position_size, entry_price, stop_loss, take_profit,
                direction, executed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (
            model_id, symbol, decision, reasoning, confidence, decision_time,
            current_price, position_size, entry_price, stop_loss, take_profit,
            direction
        ))

        decision_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(f"Logged {decision} decision for {symbol} by {model_id} (ID: {decision_id})")
        return decision_id

    def update_execution_status(
        self,
        decision_id: int,
        executed: bool,
        execution_price: Optional[float] = None,
        execution_status: str = 'filled',
        execution_time: Optional[datetime] = None
    ):
        """
        Update execution status for a decision.

        Args:
            decision_id: Decision ID to update
            executed: Whether decision was executed
            execution_price: Actual execution price
            execution_status: Execution status ('filled', 'rejected', 'pending', 'cancelled')
            execution_time: Execution timestamp (defaults to now)
        """
        exec_time = execution_time or datetime.now()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE decision_log
            SET executed = ?,
                execution_price = ?,
                execution_status = ?,
                execution_time = ?
            WHERE id = ?
        """, (executed, execution_price, execution_status, exec_time, decision_id))

        conn.commit()
        conn.close()

        logger.info(f"Updated execution status for decision {decision_id}: {execution_status}")

    def get_model_decisions(
        self,
        model_id: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
        decision_type: Optional[str] = None
    ) -> List[Decision]:
        """
        Get all decisions for a model.

        Args:
            model_id: Model identifier
            since: Only return decisions after this time
            limit: Maximum number of decisions
            decision_type: Filter by decision type ('buy', 'sell', 'hold', 'close')

        Returns:
            List of Decision objects
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM decision_log WHERE model_id = ?"
        params = [model_id]

        if since:
            query += " AND timestamp >= ?"
            params.append(since)

        if decision_type:
            query += " AND decision = ?"
            params.append(decision_type)

        query += " ORDER BY timestamp DESC"

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        decisions = [self._row_to_decision(row) for row in rows]
        logger.debug(f"Retrieved {len(decisions)} decisions for {model_id}")

        return decisions

    def get_symbol_decisions(
        self,
        symbol: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> Dict[str, List[Decision]]:
        """
        Get decisions for a symbol grouped by model.

        Args:
            symbol: Stock symbol
            since: Only return decisions after this time
            limit: Maximum number of decisions per model

        Returns:
            Dictionary mapping model_id -> List[Decision]
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM decision_log WHERE symbol = ?"
        params = [symbol]

        if since:
            query += " AND timestamp >= ?"
            params.append(since)

        query += " ORDER BY timestamp DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        # Group by model
        decisions_by_model: Dict[str, List[Decision]] = {}

        for row in rows:
            model_id = row['model_id']
            decision = self._row_to_decision(row)

            if model_id not in decisions_by_model:
                decisions_by_model[model_id] = []

            # Apply limit per model
            if limit is None or len(decisions_by_model[model_id]) < limit:
                decisions_by_model[model_id].append(decision)

        logger.debug(f"Retrieved decisions for {symbol} from {len(decisions_by_model)} models")

        return decisions_by_model

    def get_decisions_at_time(
        self,
        symbol: str,
        timestamp: datetime,
        window_minutes: int = 5
    ) -> Dict[str, Decision]:
        """
        Get decisions for a symbol near a specific time (for comparison).

        Args:
            symbol: Stock symbol
            timestamp: Target timestamp
            window_minutes: Time window around timestamp (±minutes)

        Returns:
            Dictionary mapping model_id -> Decision (closest to timestamp)
        """
        time_start = timestamp - timedelta(minutes=window_minutes)
        time_end = timestamp + timedelta(minutes=window_minutes)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM decision_log
            WHERE symbol = ?
              AND timestamp BETWEEN ? AND ?
            ORDER BY ABS(julianday(timestamp) - julianday(?))
        """, (symbol, time_start, time_end, timestamp))

        rows = cursor.fetchall()
        conn.close()

        # Get closest decision per model
        decisions_by_model: Dict[str, Decision] = {}

        for row in rows:
            model_id = row['model_id']
            if model_id not in decisions_by_model:
                decisions_by_model[model_id] = self._row_to_decision(row)

        logger.debug(f"Retrieved {len(decisions_by_model)} decisions for {symbol} near {timestamp}")

        return decisions_by_model

    def get_decision_stats(
        self,
        model_id: str,
        since: Optional[datetime] = None
    ) -> Dict:
        """
        Get decision statistics for a model.

        Args:
            model_id: Model identifier
            since: Only count decisions after this time

        Returns:
            Dictionary with decision counts and execution stats
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT
                decision,
                COUNT(*) as count,
                SUM(CASE WHEN executed = 1 THEN 1 ELSE 0 END) as executed_count,
                AVG(confidence) as avg_confidence
            FROM decision_log
            WHERE model_id = ?
        """
        params = [model_id]

        if since:
            query += " AND timestamp >= ?"
            params.append(since)

        query += " GROUP BY decision"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        stats = {
            'total_decisions': 0,
            'total_executed': 0,
            'by_type': {}
        }

        for row in rows:
            decision_type = row[0]
            count = row[1]
            executed_count = row[2]
            avg_confidence = row[3]

            stats['total_decisions'] += count
            stats['total_executed'] += executed_count

            stats['by_type'][decision_type] = {
                'count': count,
                'executed': executed_count,
                'avg_confidence': avg_confidence
            }

        return stats

    def _row_to_decision(self, row: sqlite3.Row) -> Decision:
        """Convert database row to Decision object"""
        return Decision(
            id=row['id'],
            model_id=row['model_id'],
            symbol=row['symbol'],
            decision=row['decision'],
            reasoning=row['reasoning'],
            confidence=row['confidence'],
            timestamp=datetime.fromisoformat(row['timestamp']) if isinstance(row['timestamp'], str) else row['timestamp'],
            current_price=row['current_price'],
            position_size=row['position_size'],
            entry_price=row['entry_price'],
            stop_loss=row['stop_loss'],
            take_profit=row['take_profit'],
            direction=row['direction'],
            executed=bool(row['executed']),
            execution_price=row['execution_price'],
            execution_time=datetime.fromisoformat(row['execution_time']) if row['execution_time'] and isinstance(row['execution_time'], str) else row['execution_time'],
            execution_status=row['execution_status']
        )


# Example usage
if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Initialize decision logger
    db_path = "/tmp/test_decisions.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    logger_obj = DecisionLogger(db_path)

    # Example 1: Log buy decision
    print("\n=== Logging Buy Decision ===")
    decision_id = logger_obj.log_decision(
        model_id="claude-sonnet-3.5",
        symbol="AAPL",
        decision="buy",
        reasoning="Strong momentum breakout above resistance at $175. Volume confirms bullish sentiment. RSI not overbought. Entry at current price with tight stop.",
        confidence=85.0,
        current_price=175.50,
        position_size=10000,
        entry_price=175.50,
        stop_loss=172.00,
        take_profit=180.00,
        direction="long"
    )
    print(f"Logged decision ID: {decision_id}")

    # Example 2: Log hold decision
    print("\n=== Logging Hold Decision ===")
    logger_obj.log_decision(
        model_id="gpt-4o",
        symbol="AAPL",
        decision="hold",
        reasoning="Watching for confirmation. Price near resistance but volume declining. Wait for clearer signal.",
        confidence=60.0,
        current_price=175.20,
        timestamp=datetime.now() - timedelta(minutes=2)
    )

    # Example 3: Update execution status
    print("\n=== Updating Execution Status ===")
    logger_obj.update_execution_status(
        decision_id=decision_id,
        executed=True,
        execution_price=175.52,
        execution_status="filled"
    )
    print(f"Updated execution for decision {decision_id}")

    # Example 4: Get model decisions
    print("\n=== Model Decisions ===")
    decisions = logger_obj.get_model_decisions("claude-sonnet-3.5", limit=10)
    for decision in decisions:
        print(f"\n{decision.symbol}: {decision.decision.upper()}")
        print(f"  Confidence: {decision.confidence}%")
        print(f"  Reasoning: {decision.reasoning[:100]}...")
        print(f"  Executed: {decision.executed}")

    # Example 5: Get symbol decisions across models
    print("\n=== Symbol Decisions (All Models) ===")
    symbol_decisions = logger_obj.get_symbol_decisions("AAPL")
    for model_id, decisions in symbol_decisions.items():
        print(f"\n{model_id}:")
        for decision in decisions:
            print(f"  {decision.decision} @ {decision.timestamp.strftime('%H:%M:%S')}")

    # Example 6: Get decision stats
    print("\n=== Decision Statistics ===")
    stats = logger_obj.get_decision_stats("claude-sonnet-3.5")
    print(f"Total decisions: {stats['total_decisions']}")
    print(f"Total executed: {stats['total_executed']}")
    for decision_type, type_stats in stats['by_type'].items():
        print(f"\n{decision_type.upper()}:")
        print(f"  Count: {type_stats['count']}")
        print(f"  Executed: {type_stats['executed']}")
        print(f"  Avg confidence: {type_stats['avg_confidence']:.1f}%")

    print("\n=== Demo Complete ===")
