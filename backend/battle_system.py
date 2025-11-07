#!/usr/bin/env python3
"""
Battle System Core Components
Orchestrates multi-model trading competition with real-time tracking
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """Model participation status"""
    ACTIVE = "active"
    PAUSED = "paused"
    INACTIVE = "inactive"


@dataclass
class ModelConfig:
    """Configuration for a competing model"""
    model_id: str
    display_name: str
    provider: str  # openai, anthropic, etc.
    model_name: str  # gpt-4, claude-3-opus, etc.
    initial_capital: float
    status: ModelStatus
    created_at: str
    metadata: Dict[str, Any] = None

    def to_dict(self):
        result = asdict(self)
        result['status'] = self.status.value
        return result


@dataclass
class Position:
    """Trading position"""
    model_id: str
    symbol: str
    direction: str  # long/short
    shares: int
    entry_price: float
    stop_loss: float
    take_profit: float
    opened_at: str
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class Trade:
    """Completed trade"""
    trade_id: str
    model_id: str
    symbol: str
    direction: str
    shares: int
    entry_price: float
    exit_price: float
    realized_pnl: float
    opened_at: str
    closed_at: str
    exit_reason: str
    reasoning: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class ModelMetrics:
    """Performance metrics for a model"""
    model_id: str
    current_equity: float
    total_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    current_positions: int
    timestamp: str

    def to_dict(self):
        return asdict(self)


class BattleDatabase:
    """Database manager for battle system"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize battle database schema"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS battle_models (
                model_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                initial_capital REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                metadata TEXT
            )
        """)

        # Positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS battle_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                shares INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                opened_at TIMESTAMP NOT NULL,
                closed_at TIMESTAMP,
                exit_price REAL,
                exit_reason TEXT,
                realized_pnl REAL,
                status TEXT NOT NULL DEFAULT 'open',
                reasoning TEXT,
                FOREIGN KEY (model_id) REFERENCES battle_models(model_id)
            )
        """)

        # Trades (completed) table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS battle_trades (
                trade_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                shares INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                opened_at TIMESTAMP NOT NULL,
                closed_at TIMESTAMP NOT NULL,
                exit_reason TEXT,
                reasoning TEXT,
                FOREIGN KEY (model_id) REFERENCES battle_models(model_id)
            )
        """)

        # Equity curve table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS battle_equity_curve (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                equity REAL NOT NULL,
                daily_pnl REAL,
                total_pnl REAL,
                FOREIGN KEY (model_id) REFERENCES battle_models(model_id)
            )
        """)

        # Metrics snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS battle_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                current_equity REAL NOT NULL,
                total_pnl REAL NOT NULL,
                total_trades INTEGER NOT NULL,
                winning_trades INTEGER NOT NULL,
                losing_trades INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                avg_win REAL NOT NULL,
                avg_loss REAL NOT NULL,
                profit_factor REAL NOT NULL,
                sharpe_ratio REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                current_positions INTEGER NOT NULL,
                FOREIGN KEY (model_id) REFERENCES battle_models(model_id)
            )
        """)

        # Decisions table (for comparison)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS battle_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                decision TEXT NOT NULL,
                reasoning TEXT,
                confidence REAL,
                market_data TEXT,
                FOREIGN KEY (model_id) REFERENCES battle_models(model_id)
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_model ON battle_positions(model_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_status ON battle_positions(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_model ON battle_trades(model_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_model ON battle_equity_curve(model_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_model ON battle_metrics(model_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_model_symbol ON battle_decisions(model_id, symbol, timestamp)")

        conn.commit()
        conn.close()
        logger.info(f"Battle database initialized at {self.db_path}")

    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def add_model(self, model: ModelConfig):
        """Register a new model"""
        conn = self.get_connection()
        cursor = conn.cursor()

        metadata_json = json.dumps(model.metadata) if model.metadata else None

        cursor.execute("""
            INSERT OR REPLACE INTO battle_models
            (model_id, display_name, provider, model_name, initial_capital, status, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model.model_id,
            model.display_name,
            model.provider,
            model.model_name,
            model.initial_capital,
            model.status.value,
            model.created_at,
            metadata_json
        ))

        conn.commit()
        conn.close()
        logger.info(f"Added model: {model.model_id}")

    def get_models(self) -> List[ModelConfig]:
        """Get all registered models"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM battle_models ORDER BY created_at ASC")
        rows = cursor.fetchall()
        conn.close()

        models = []
        for row in rows:
            metadata = json.loads(row['metadata']) if row['metadata'] else None
            models.append(ModelConfig(
                model_id=row['model_id'],
                display_name=row['display_name'],
                provider=row['provider'],
                model_name=row['model_name'],
                initial_capital=row['initial_capital'],
                status=ModelStatus(row['status']),
                created_at=row['created_at'],
                metadata=metadata
            ))

        return models

    def get_active_models(self) -> List[ModelConfig]:
        """Get only active models"""
        models = self.get_models()
        return [m for m in models if m.status == ModelStatus.ACTIVE]


class MetricsEngine:
    """Calculate and track model performance metrics"""

    def __init__(self, db: BattleDatabase):
        self.db = db

    def calculate_metrics(self, model_id: str) -> ModelMetrics:
        """Calculate comprehensive metrics for a model"""
        conn = self.db.get_connection()
        cursor = conn.cursor()

        # Get model initial capital
        cursor.execute("SELECT initial_capital FROM battle_models WHERE model_id = ?", (model_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Model not found: {model_id}")
        initial_capital = row['initial_capital']

        # Get all closed trades
        cursor.execute("""
            SELECT realized_pnl, exit_price, entry_price, shares
            FROM battle_trades
            WHERE model_id = ?
            ORDER BY closed_at ASC
        """, (model_id,))
        trades = cursor.fetchall()

        # Get current open positions
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM battle_positions
            WHERE model_id = ? AND status = 'open'
        """, (model_id,))
        current_positions = cursor.fetchone()['count']

        conn.close()

        # Calculate metrics
        total_trades = len(trades)
        if total_trades == 0:
            return ModelMetrics(
                model_id=model_id,
                current_equity=initial_capital,
                total_pnl=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                current_positions=current_positions,
                timestamp=datetime.now().isoformat()
            )

        # Basic metrics
        total_pnl = sum(t['realized_pnl'] for t in trades)
        winning_trades = [t for t in trades if t['realized_pnl'] > 0]
        losing_trades = [t for t in trades if t['realized_pnl'] < 0]

        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0

        avg_win = sum(t['realized_pnl'] for t in winning_trades) / win_count if win_count > 0 else 0.0
        avg_loss = abs(sum(t['realized_pnl'] for t in losing_trades) / loss_count) if loss_count > 0 else 0.0

        # Profit factor
        gross_profit = sum(t['realized_pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['realized_pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Calculate max drawdown
        equity_curve = [initial_capital]
        cumulative_pnl = 0
        for trade in trades:
            cumulative_pnl += trade['realized_pnl']
            equity_curve.append(initial_capital + cumulative_pnl)

        max_drawdown = 0.0
        peak = equity_curve[0]
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100 if peak > 0 else 0.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Simplified Sharpe ratio (using trade returns)
        if total_trades > 1:
            returns = [t['realized_pnl'] / initial_capital for t in trades]
            avg_return = sum(returns) / len(returns)
            std_dev = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            sharpe_ratio = (avg_return / std_dev * (252 ** 0.5)) if std_dev > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        current_equity = initial_capital + total_pnl

        return ModelMetrics(
            model_id=model_id,
            current_equity=current_equity,
            total_pnl=total_pnl,
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            current_positions=current_positions,
            timestamp=datetime.now().isoformat()
        )

    def save_metrics_snapshot(self, metrics: ModelMetrics):
        """Save metrics snapshot to database"""
        conn = self.db.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO battle_metrics
            (model_id, timestamp, current_equity, total_pnl, total_trades,
             winning_trades, losing_trades, win_rate, avg_win, avg_loss,
             profit_factor, sharpe_ratio, max_drawdown, current_positions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.model_id,
            metrics.timestamp,
            metrics.current_equity,
            metrics.total_pnl,
            metrics.total_trades,
            metrics.winning_trades,
            metrics.losing_trades,
            metrics.win_rate,
            metrics.avg_win,
            metrics.avg_loss,
            metrics.profit_factor,
            metrics.sharpe_ratio,
            metrics.max_drawdown,
            metrics.current_positions
        ))

        conn.commit()
        conn.close()


class Leaderboard:
    """Manage model rankings and leaderboards"""

    def __init__(self, db: BattleDatabase, metrics_engine: MetricsEngine):
        self.db = db
        self.metrics_engine = metrics_engine

    def get_rankings(self, timeframe: str = "all") -> List[Dict[str, Any]]:
        """
        Get model rankings for a timeframe

        Args:
            timeframe: 'daily', 'weekly', 'monthly', 'all'
        """
        models = self.db.get_active_models()
        rankings = []

        for model in models:
            metrics = self.metrics_engine.calculate_metrics(model.model_id)

            # Apply timeframe filter if needed
            if timeframe != "all":
                metrics = self._filter_by_timeframe(metrics, timeframe)

            rankings.append({
                "model_id": model.model_id,
                "display_name": model.display_name,
                "provider": model.provider,
                "model_name": model.model_name,
                "current_equity": metrics.current_equity,
                "total_pnl": metrics.total_pnl,
                "pnl_pct": (metrics.total_pnl / model.initial_capital * 100) if model.initial_capital > 0 else 0.0,
                "total_trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "profit_factor": metrics.profit_factor
            })

        # Sort by total P&L (descending)
        rankings.sort(key=lambda x: x['total_pnl'], reverse=True)

        # Add rank
        for i, ranking in enumerate(rankings, 1):
            ranking['rank'] = i

        return rankings

    def _filter_by_timeframe(self, metrics: ModelMetrics, timeframe: str) -> ModelMetrics:
        """Filter metrics by timeframe (simplified - returns same metrics for now)"""
        # TODO: Implement time-based filtering by querying trades within timeframe
        return metrics


class BattleOrchestrator:
    """Main orchestrator for battle system"""

    def __init__(self, db_path: Path):
        self.db = BattleDatabase(db_path)
        self.metrics_engine = MetricsEngine(self.db)
        self.leaderboard = Leaderboard(self.db, self.metrics_engine)
        self.start_time = datetime.now()

    def register_model(self, model: ModelConfig):
        """Register a new competing model"""
        self.db.add_model(model)
        logger.info(f"Registered model: {model.model_id}")

    def get_models(self) -> List[ModelConfig]:
        """Get all models"""
        return self.db.get_models()

    def get_system_status(self) -> Dict[str, Any]:
        """Get battle system status"""
        models = self.db.get_models()
        active_count = sum(1 for m in models if m.status == ModelStatus.ACTIVE)

        uptime = datetime.now() - self.start_time

        return {
            "status": "running",
            "uptime_seconds": uptime.total_seconds(),
            "total_models": len(models),
            "active_models": active_count,
            "current_cycle": datetime.now().isoformat(),
            "database_path": str(self.db.db_path)
        }

    def get_model_positions(self, model_id: str) -> List[Position]:
        """Get current positions for a model"""
        conn = self.db.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT *
            FROM battle_positions
            WHERE model_id = ? AND status = 'open'
            ORDER BY opened_at DESC
        """, (model_id,))

        rows = cursor.fetchall()
        conn.close()

        positions = []
        for row in rows:
            positions.append(Position(
                model_id=row['model_id'],
                symbol=row['symbol'],
                direction=row['direction'],
                shares=row['shares'],
                entry_price=row['entry_price'],
                stop_loss=row['stop_loss'],
                take_profit=row['take_profit'],
                opened_at=row['opened_at']
            ))

        return positions

    def get_model_trades(self, model_id: str, limit: int = 100) -> List[Trade]:
        """Get trade history for a model"""
        conn = self.db.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT *
            FROM battle_trades
            WHERE model_id = ?
            ORDER BY closed_at DESC
            LIMIT ?
        """, (model_id, limit))

        rows = cursor.fetchall()
        conn.close()

        trades = []
        for row in rows:
            trades.append(Trade(
                trade_id=row['trade_id'],
                model_id=row['model_id'],
                symbol=row['symbol'],
                direction=row['direction'],
                shares=row['shares'],
                entry_price=row['entry_price'],
                exit_price=row['exit_price'],
                realized_pnl=row['realized_pnl'],
                opened_at=row['opened_at'],
                closed_at=row['closed_at'],
                exit_reason=row['exit_reason'],
                reasoning=row['reasoning']
            ))

        return trades

    def get_equity_curve(self, model_id: str) -> List[Dict[str, Any]]:
        """Get equity curve data for a model"""
        conn = self.db.get_connection()
        cursor = conn.cursor()

        # Get initial capital
        cursor.execute("SELECT initial_capital FROM battle_models WHERE model_id = ?", (model_id,))
        row = cursor.fetchone()
        if not row:
            return []
        initial_capital = row['initial_capital']

        # Get all trades in chronological order
        cursor.execute("""
            SELECT closed_at, realized_pnl
            FROM battle_trades
            WHERE model_id = ?
            ORDER BY closed_at ASC
        """, (model_id,))

        trades = cursor.fetchall()
        conn.close()

        # Build equity curve
        equity_curve = [{"timestamp": self.start_time.isoformat(), "equity": initial_capital}]
        cumulative_pnl = 0

        for trade in trades:
            cumulative_pnl += trade['realized_pnl']
            equity_curve.append({
                "timestamp": trade['closed_at'],
                "equity": initial_capital + cumulative_pnl
            })

        return equity_curve
