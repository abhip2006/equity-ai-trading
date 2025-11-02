"""
Backtest Module - Historical trading system testing.

This module provides comprehensive backtesting capabilities for the
ActiveTrader-LLM trading system, including:

- Historical walk-forward testing
- Realistic execution simulation
- Performance metrics calculation
- Report generation
"""

from active_trader_llm.backtest.backtest_engine import BacktestEngine
from active_trader_llm.backtest.execution_simulator import (
    ExecutionSimulator,
    ExecutionCostConfig,
    ExecutionResult
)
from active_trader_llm.backtest.performance_metrics import PerformanceMetricsCalculator
from active_trader_llm.backtest.report_generator import ReportGenerator

__all__ = [
    'BacktestEngine',
    'ExecutionSimulator',
    'ExecutionCostConfig',
    'ExecutionResult',
    'PerformanceMetricsCalculator',
    'ReportGenerator'
]
