#!/usr/bin/env python3
"""
Pydantic response models for battle API endpoints
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class ModelInfo(BaseModel):
    """Model information"""
    model_id: str
    display_name: str
    provider: str
    model_name: str
    status: str
    initial_capital: float
    created_at: str


class PositionResponse(BaseModel):
    """Single position"""
    model_id: str
    symbol: str
    direction: str
    shares: int
    entry_price: float
    stop_loss: float
    take_profit: float
    opened_at: str
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None


class TradeResponse(BaseModel):
    """Single trade"""
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


class EquityCurvePoint(BaseModel):
    """Single point on equity curve"""
    timestamp: str
    equity: float


class EquityCurveResponse(BaseModel):
    """Equity curve data"""
    model_id: str
    equity_curve: List[EquityCurvePoint]


class ModelMetricsResponse(BaseModel):
    """Comprehensive metrics for a model"""
    model_id: str
    current_equity: float
    total_pnl: float
    pnl_pct: float
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


class LeaderboardEntry(BaseModel):
    """Single entry in leaderboard"""
    rank: int
    model_id: str
    display_name: str
    provider: str
    model_name: str
    current_equity: float
    total_pnl: float
    pnl_pct: float
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float


class LeaderboardResponse(BaseModel):
    """Leaderboard rankings"""
    timeframe: str
    rankings: List[LeaderboardEntry]
    timestamp: str


class SystemStatusResponse(BaseModel):
    """Battle system status"""
    status: str
    uptime_seconds: float
    total_models: int
    active_models: int
    current_cycle: str
    database_path: str


class DecisionResponse(BaseModel):
    """Single decision from a model"""
    model_id: str
    symbol: str
    timestamp: str
    decision: str
    reasoning: Optional[str] = None
    confidence: Optional[float] = None


class DecisionComparisonResponse(BaseModel):
    """Compare decisions from all models for same symbol/timestamp"""
    symbol: str
    timestamp: str
    decisions: List[DecisionResponse]


class ModelDecisionsResponse(BaseModel):
    """List of decisions for a specific model"""
    model_id: str
    decisions: List[DecisionResponse]
    total: int


class ComparePositionsResponse(BaseModel):
    """All models' positions side-by-side"""
    timestamp: str
    positions_by_model: Dict[str, List[PositionResponse]]


class CompareEquityCurvesResponse(BaseModel):
    """All equity curves for charting"""
    equity_curves: Dict[str, List[EquityCurvePoint]]


class WebSocketMessage(BaseModel):
    """WebSocket event message"""
    type: str  # new_trade, decision_made, metrics_updated, leaderboard_changed, etc.
    model_id: Optional[str] = None
    data: Dict[str, Any]
    timestamp: str


class ModelsListResponse(BaseModel):
    """List of all models"""
    models: List[ModelInfo]
    count: int


class PositionsListResponse(BaseModel):
    """List of positions"""
    positions: List[PositionResponse]
    count: int


class TradesListResponse(BaseModel):
    """List of trades"""
    trades: List[TradeResponse]
    count: int
