"""
Configuration schema using Pydantic for validation.
"""

from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field, validator


class DataSourcesConfig(BaseModel):
    prices: str = "yfinance"
    interval: str = "1h"
    universe: List[str] = ["AAPL", "MSFT", "SPY", "QQQ"]
    lookback_days: int = 90


class RiskParameters(BaseModel):
    max_position_pct: float = 0.05
    max_concurrent_positions: int = 8
    max_daily_drawdown: float = 0.10
    max_sector_concentration: float = 0.30
    min_win_rate: float = 0.35
    min_net_pl: float = 0.0


class StrategyConfig(BaseModel):
    name: Literal["momentum_breakout", "mean_reversion", "pullback", "sector_rotation"]
    regime: str
    enabled: bool = True


class StrategySwitchingConfig(BaseModel):
    window_trades: int = 30
    min_win_rate: float = 0.35
    min_net_pl: float = 0.0
    hysteresis_trades_before_switch: int = 10


class ScheduleConfig(BaseModel):
    decision_interval: str = "1h"
    learning_update: str = "daily"


class CostControlConfig(BaseModel):
    combine_symbols_per_prompt: bool = True
    cache_data: bool = True
    minimize_call_frequency: bool = True
    max_cost_per_decision_usd: float = 0.05


class LLMConfig(BaseModel):
    provider: Literal["openai", "anthropic", "local"] = "anthropic"
    model: str = "claude-3-5-sonnet-20241022"
    api_key: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 2000


class Config(BaseModel):
    """Main configuration for ActiveTrader-LLM"""

    mode: Literal["backtest", "paper-live"] = "backtest"
    data_sources: DataSourcesConfig = Field(default_factory=DataSourcesConfig)
    risk_parameters: RiskParameters = Field(default_factory=RiskParameters)
    strategies: List[StrategyConfig] = Field(default_factory=list)
    strategy_switching: StrategySwitchingConfig = Field(default_factory=StrategySwitchingConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    cost_control: CostControlConfig = Field(default_factory=CostControlConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    # Storage
    database_path: str = "data/trading.db"
    log_path: str = "logs/trade_log.jsonl"

    # Feature flags
    enable_sentiment: bool = False
    enable_macro: bool = False

    @validator('strategies', pre=True, always=True)
    def set_default_strategies(cls, v):
        if not v:
            return [
                StrategyConfig(name="momentum_breakout", regime="trending_bull"),
                StrategyConfig(name="mean_reversion", regime="range"),
                StrategyConfig(name="pullback", regime="mild_trend"),
                StrategyConfig(name="sector_rotation", regime="mixed"),
            ]
        return v
