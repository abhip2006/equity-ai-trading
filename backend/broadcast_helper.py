#!/usr/bin/env python3
"""
Helper functions to broadcast WebSocket events from the battle system
This module can be imported by the actual trading system to send real-time updates
"""

from typing import Dict, Any
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)


class BattleBroadcaster:
    """
    Utility class to broadcast battle events
    Should be initialized with the battle_manager from main.py
    """

    def __init__(self, battle_manager):
        self.battle_manager = battle_manager

    async def broadcast_new_trade(self, model_id: str, trade_data: Dict[str, Any]):
        """Broadcast a new completed trade"""
        message = {
            "type": "new_trade",
            "model_id": model_id,
            "data": trade_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.battle_manager.broadcast(message)
        logger.info(f"Broadcasted new_trade for {model_id}: {trade_data.get('symbol')}")

    async def broadcast_decision_made(self, model_id: str, decision_data: Dict[str, Any]):
        """Broadcast a trading decision"""
        message = {
            "type": "decision_made",
            "model_id": model_id,
            "data": decision_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.battle_manager.broadcast(message)
        logger.info(f"Broadcasted decision_made for {model_id}: {decision_data.get('symbol')}")

    async def broadcast_metrics_updated(self, model_id: str, metrics_data: Dict[str, Any]):
        """Broadcast updated metrics"""
        message = {
            "type": "metrics_updated",
            "model_id": model_id,
            "data": metrics_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.battle_manager.broadcast(message)
        logger.info(f"Broadcasted metrics_updated for {model_id}")

    async def broadcast_leaderboard_changed(self, leaderboard_data: Dict[str, Any]):
        """Broadcast leaderboard update"""
        message = {
            "type": "leaderboard_changed",
            "data": leaderboard_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.battle_manager.broadcast(message)
        logger.info("Broadcasted leaderboard_changed")

    async def broadcast_position_opened(self, model_id: str, position_data: Dict[str, Any]):
        """Broadcast new position opened"""
        message = {
            "type": "position_opened",
            "model_id": model_id,
            "data": position_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.battle_manager.broadcast(message)
        logger.info(f"Broadcasted position_opened for {model_id}: {position_data.get('symbol')}")

    async def broadcast_position_closed(self, model_id: str, position_data: Dict[str, Any]):
        """Broadcast position closed"""
        message = {
            "type": "position_closed",
            "model_id": model_id,
            "data": position_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.battle_manager.broadcast(message)
        logger.info(f"Broadcasted position_closed for {model_id}: {position_data.get('symbol')}")


# Example usage in trading system:
"""
from backend.broadcast_helper import BattleBroadcaster
from backend.main import battle_manager

# Initialize broadcaster
broadcaster = BattleBroadcaster(battle_manager)

# When a trade completes
await broadcaster.broadcast_new_trade(
    model_id="gpt4",
    trade_data={
        "trade_id": "trade_123",
        "symbol": "AAPL",
        "direction": "long",
        "realized_pnl": 500.0,
        "exit_reason": "take_profit"
    }
)

# When a decision is made
await broadcaster.broadcast_decision_made(
    model_id="claude3",
    decision_data={
        "symbol": "MSFT",
        "decision": "BUY",
        "reasoning": "Strong momentum breakout",
        "confidence": 0.85
    }
)

# When metrics are recalculated
await broadcaster.broadcast_metrics_updated(
    model_id="gpt4",
    metrics_data={
        "current_equity": 105000.0,
        "total_pnl": 5000.0,
        "win_rate": 65.5,
        "sharpe_ratio": 1.8
    }
)

# When leaderboard changes
await broadcaster.broadcast_leaderboard_changed(
    leaderboard_data={
        "rankings": [
            {"rank": 1, "model_id": "claude3", "total_pnl": 8000.0},
            {"rank": 2, "model_id": "gpt4", "total_pnl": 5000.0},
        ]
    }
)
"""
