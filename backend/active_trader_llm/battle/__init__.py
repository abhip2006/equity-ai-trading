"""
Battle Orchestration System

Coordinates multiple model instances competing simultaneously.

Components:
- ModelInstance: Isolated wrapper for each ActiveTraderLLM instance
- SharedDataFeed: Centralized market data fetching and caching
- CycleCoordinator: Synchronized trading cycle management
- BattleOrchestrator: Main coordinator for multi-model competition
- MetricsEngine: Calculate comprehensive performance metrics
- Leaderboard: Rankings and competitive standings
- DecisionLogger: Transparency logging for all decisions

Example usage:
    from active_trader_llm.battle import BattleOrchestrator

    model_configs = [
        {'model_id': 'gpt4o', 'provider': 'openai', 'model': 'gpt-4o', ...},
        {'model_id': 'claude_sonnet', 'provider': 'anthropic', 'model': 'claude-3-5-sonnet', ...},
        # ... more models
    ]

    orchestrator = BattleOrchestrator(
        model_configs=model_configs,
        base_config_path="config.yaml",
        universe=["AAPL", "MSFT", "GOOGL"],
        database_dir="data/battle"
    )

    orchestrator.initialize_models()
    orchestrator.start_competition(num_cycles=10)
    orchestrator.shutdown()
"""

from active_trader_llm.battle.model_instance import ModelInstance, ModelInstanceState
from active_trader_llm.battle.shared_data_feed import SharedDataFeed, DataSnapshot
from active_trader_llm.battle.cycle_coordinator import CycleCoordinator, CycleResult
from active_trader_llm.battle.battle_orchestrator import BattleOrchestrator

# Metrics and leaderboard components
from active_trader_llm.battle.metrics_engine import (
    MetricsEngine,
    MetricsSnapshot,
    EquitySnapshot,
    TradeRecord
)

from active_trader_llm.battle.leaderboard import (
    Leaderboard,
    LeaderboardEntry
)

from active_trader_llm.battle.decision_logger import (
    DecisionLogger,
    Decision
)

__all__ = [
    # Orchestration
    'ModelInstance',
    'ModelInstanceState',
    'SharedDataFeed',
    'DataSnapshot',
    'CycleCoordinator',
    'CycleResult',
    'BattleOrchestrator',

    # Metrics
    'MetricsEngine',
    'MetricsSnapshot',
    'EquitySnapshot',
    'TradeRecord',

    # Leaderboard
    'Leaderboard',
    'LeaderboardEntry',

    # Decision logging
    'DecisionLogger',
    'Decision',
]
