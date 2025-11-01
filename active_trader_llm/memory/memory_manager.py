"""
Memory Manager: Layered memory for short-term trades and long-term strategy stats.

Inspired by FinMem-LLM architecture with short-term and long-term memory stores.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TradeMemory(BaseModel):
    """Short-term trade memory entry"""
    trade_id: str
    timestamp: str
    symbol: str
    strategy: str
    direction: str
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    outcome: Optional[str] = None
    context_snapshot: Dict = {}


class StrategyStats(BaseModel):
    """Long-term strategy performance statistics"""
    strategy_name: str
    regime: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    total_pnl: float = 0.0
    avg_rr: float = 0.0
    last_updated: str


class MemoryManager:
    """
    Manages layered memory for learning and adaptation.

    - Short-term memory: Recent N trades with full context
    - Long-term memory: Aggregated strategy performance by regime
    - Regime history: Daily regime snapshots
    """

    def __init__(self, db_path: str = "data/memory.db"):
        """Initialize memory manager with SQLite backend"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Short-term trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS short_term_memory (
                trade_id TEXT PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                strategy TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                outcome TEXT,
                context_snapshot TEXT,
                created_at TEXT
            )
        ''')

        # Long-term strategy stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT,
                regime TEXT,
                total_trades INTEGER,
                wins INTEGER,
                losses INTEGER,
                win_rate REAL,
                avg_return REAL,
                total_pnl REAL,
                avg_rr REAL,
                last_updated TEXT,
                UNIQUE(strategy_name, regime)
            )
        ''')

        # Regime history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS regime_history (
                date TEXT PRIMARY KEY,
                regime TEXT,
                breadth_score REAL,
                notes TEXT
            )
        ''')

        # Indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON short_term_memory(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_strategy ON short_term_memory(strategy)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON short_term_memory(timestamp)')

        conn.commit()
        conn.close()

    def add_trade(self, trade_memory: TradeMemory):
        """Add trade to short-term memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO short_term_memory
            (trade_id, timestamp, symbol, strategy, direction, entry_price, exit_price,
             pnl, outcome, context_snapshot, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_memory.trade_id,
            trade_memory.timestamp,
            trade_memory.symbol,
            trade_memory.strategy,
            trade_memory.direction,
            trade_memory.entry_price,
            trade_memory.exit_price,
            trade_memory.pnl,
            trade_memory.outcome,
            json.dumps(trade_memory.context_snapshot),
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

        logger.info(f"Added trade {trade_memory.trade_id} to short-term memory")

    def get_recent_trades(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        limit: int = 20
    ) -> List[TradeMemory]:
        """Retrieve recent trades from short-term memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM short_term_memory WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        trades = []
        for row in rows:
            trades.append(TradeMemory(
                trade_id=row[0],
                timestamp=row[1],
                symbol=row[2],
                strategy=row[3],
                direction=row[4],
                entry_price=row[5],
                exit_price=row[6],
                pnl=row[7],
                outcome=row[8],
                context_snapshot=json.loads(row[9]) if row[9] else {}
            ))

        return trades

    def update_strategy_stats(self, strategy_name: str, regime: str, trade_outcome: Dict):
        """Update long-term strategy statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get current stats
        cursor.execute('''
            SELECT total_trades, wins, losses, total_pnl, avg_rr
            FROM long_term_memory
            WHERE strategy_name = ? AND regime = ?
        ''', (strategy_name, regime))

        row = cursor.fetchone()

        if row:
            total_trades, wins, losses, total_pnl, avg_rr = row
        else:
            total_trades, wins, losses, total_pnl, avg_rr = 0, 0, 0, 0.0, 0.0

        # Update with new trade
        total_trades += 1
        if trade_outcome.get('pnl', 0) > 0:
            wins += 1
        else:
            losses += 1

        total_pnl += trade_outcome.get('pnl', 0.0)
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        avg_return = total_pnl / total_trades if total_trades > 0 else 0.0

        # Update R:R (rolling average)
        new_rr = trade_outcome.get('rr', 0.0)
        avg_rr = (avg_rr * (total_trades - 1) + new_rr) / total_trades if total_trades > 0 else 0.0

        # Insert or update
        cursor.execute('''
            INSERT OR REPLACE INTO long_term_memory
            (strategy_name, regime, total_trades, wins, losses, win_rate, avg_return, total_pnl, avg_rr, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            strategy_name, regime, total_trades, wins, losses,
            win_rate, avg_return, total_pnl, avg_rr,
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

        logger.info(f"Updated {strategy_name} stats for {regime}: {wins}/{total_trades} wins, {avg_return:.2f} avg return")

    def get_strategy_stats(self, strategy_name: Optional[str] = None, regime: Optional[str] = None) -> List[StrategyStats]:
        """Retrieve strategy performance statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM long_term_memory WHERE 1=1"
        params = []

        if strategy_name:
            query += " AND strategy_name = ?"
            params.append(strategy_name)

        if regime:
            query += " AND regime = ?"
            params.append(regime)

        query += " ORDER BY last_updated DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        stats = []
        for row in rows:
            stats.append(StrategyStats(
                strategy_name=row[1],
                regime=row[2],
                total_trades=row[3],
                wins=row[4],
                losses=row[5],
                win_rate=row[6],
                avg_return=row[7],
                total_pnl=row[8],
                avg_rr=row[9],
                last_updated=row[10]
            ))

        return stats

    def save_regime_snapshot(self, date: str, regime: str, breadth_score: float, notes: str):
        """Save daily regime snapshot"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO regime_history (date, regime, breadth_score, notes)
            VALUES (?, ?, ?, ?)
        ''', (date, regime, breadth_score, notes))

        conn.commit()
        conn.close()

    def get_regime_history(self, days: int = 30) -> List[Dict]:
        """Get recent regime history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(days=days)).date().isoformat()

        cursor.execute('''
            SELECT date, regime, breadth_score, notes
            FROM regime_history
            WHERE date >= ?
            ORDER BY date DESC
        ''', (cutoff,))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                'date': row[0],
                'regime': row[1],
                'breadth_score': row[2],
                'notes': row[3]
            }
            for row in rows
        ]

    def get_context_summary(self, symbol: str, max_trades: int = 10) -> str:
        """Generate context summary for a symbol"""
        recent_trades = self.get_recent_trades(symbol=symbol, limit=max_trades)

        if not recent_trades:
            return f"No recent trade history for {symbol}"

        wins = sum(1 for t in recent_trades if t.pnl and t.pnl > 0)
        total_pnl = sum(t.pnl for t in recent_trades if t.pnl)

        summary = f"{symbol}: {wins}/{len(recent_trades)} wins, ${total_pnl:.2f} P/L in last {len(recent_trades)} trades. "

        if recent_trades[0].outcome:
            summary += f"Last: {recent_trades[0].outcome}"

        return summary


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    memory = MemoryManager()

    # Add sample trade
    trade = TradeMemory(
        trade_id="trade_001",
        timestamp=datetime.now().isoformat(),
        symbol="AAPL",
        strategy="momentum_breakout",
        direction="long",
        entry_price=175.0,
        exit_price=178.5,
        pnl=350.0,
        outcome="win",
        context_snapshot={'regime': 'trending_bull', 'rsi': 65}
    )

    memory.add_trade(trade)

    # Update strategy stats
    memory.update_strategy_stats(
        "momentum_breakout",
        "trending_bull",
        {'pnl': 350.0, 'rr': 2.1}
    )

    # Retrieve stats
    stats = memory.get_strategy_stats(strategy_name="momentum_breakout")
    for s in stats:
        print(f"\n{s.strategy_name} in {s.regime}:")
        print(f"  Win rate: {s.win_rate:.1%}")
        print(f"  Avg return: ${s.avg_return:.2f}")
        print(f"  Total P/L: ${s.total_pnl:.2f}")
