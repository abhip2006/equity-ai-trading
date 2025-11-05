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
    """
    Risk parameters for trading system.

    Position sizing limits removed - agent decides based on performance.
    Only emergency safety rail: optional daily drawdown circuit breaker.
    """
    # Emergency safety rail (optional)
    enforce_daily_drawdown: bool = True
    emergency_drawdown_limit: float = 0.20  # 20% daily loss triggers halt

    # Strategy performance thresholds (for strategy switching, not trade approval)
    min_win_rate: float = 0.35
    min_net_pl: float = 0.0


class TradeValidationConfig(BaseModel):
    """
    Trade plan validation configuration.

    Validates LLM-generated trade plans to prevent impossible or dangerous parameters.
    Checks: price logic, position sizing, risk-reward ratios, price sanity, stop distances.
    """
    enabled: bool = True
    max_position_pct: float = Field(default=100.0, gt=0, le=100, description="Maximum position size as % of capital")
    min_risk_reward_ratio: float = Field(default=0.0, ge=0, description="Minimum reward/risk ratio")
    max_price_deviation_pct: float = Field(default=50.0, gt=0, description="Max deviation from current price (%)")
    min_stop_distance_pct: float = Field(default=0.1, gt=0, description="Minimum stop distance from entry (%)")
    max_stop_distance_pct: float = Field(default=50.0, gt=0, description="Maximum stop distance from entry (%)")

    @validator('max_stop_distance_pct')
    def validate_stop_distances(cls, v, values):
        """Ensure max stop distance is greater than min"""
        if 'min_stop_distance_pct' in values and v <= values['min_stop_distance_pct']:
            raise ValueError(f"max_stop_distance_pct ({v}) must be greater than min_stop_distance_pct ({values['min_stop_distance_pct']})")
        return v


class PositionManagementConfig(BaseModel):
    """
    Position management configuration.

    Tracks positions across decision cycles and manages exits.
    """
    enabled: bool = True
    max_total_exposure_pct: float = Field(default=80.0, gt=0, le=100, description="Maximum total portfolio exposure")
    allow_duplicate_symbols: bool = Field(default=False, description="Prevent multiple positions in same symbol")
    auto_close_eod: bool = Field(default=True, description="Close all positions before market close")
    eod_close_time: str = Field(default="15:55", description="Time to close positions (3:55 PM)")


class ExecutionScheduleConfig(BaseModel):
    """
    Execution schedule configuration.

    Defines when to run decision cycles.
    """
    times: List[str] = Field(default_factory=lambda: ["09:30", "12:00", "16:00"], description="Times to run decision cycles")
    timezone: str = Field(default="America/New_York", description="Timezone for scheduling")
    monitor_exits_interval_seconds: int = Field(default=60, gt=0, description="How often to check exits between cycles")


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


class Stage1Config(BaseModel):
    """Stage 1: Market-wide summary configuration"""
    enabled: bool = True
    target_candidate_count: int = 100
    market_summary_sectors: int = 11


class Stage2Config(BaseModel):
    """Stage 2: Deep analysis configuration"""
    batch_size: int = 15
    max_batches: int = 10
    max_candidates: int = 150


class ScannerConfig(BaseModel):
    """Two-stage market scanner configuration"""
    enabled: bool = False
    universe_source: Literal["file", "alpaca_optionable", "tradable_universe_cache", "custom"] = "file"
    universe_file_path: Optional[str] = "data/stock_universe.txt"
    alpaca_api_url: Optional[str] = None
    refresh_universe_hours: int = 24
    stage1: Stage1Config = Field(default_factory=Stage1Config)
    stage2: Stage2Config = Field(default_factory=Stage2Config)


class MacroConfig(BaseModel):
    """
    Macro analysis configuration.

    Controls macro data fetching and caching settings.
    """
    cache_duration_seconds: int = Field(default=300, gt=0, description="Cache duration for macro data (default 5 minutes)")
    use_yfinance: bool = Field(default=True, description="Use yfinance for macro data fetching")


class ExecutionConfig(BaseModel):
    """
    Execution configuration for live trading.

    Controls how orders are submitted to the broker.
    """
    broker: Literal["alpaca", "simulated"] = "simulated"
    paper_trading: bool = True
    order_type: Literal["market", "limit"] = "market"
    time_in_force: Literal["day", "gtc", "ioc", "fok"] = "day"

    # Simulation costs
    commission_per_trade: float = 1.0  # Fixed commission per trade (USD)
    slippage_bps: float = 5.0  # Slippage in basis points (5 bps = 0.05%)

    # Alpaca-specific settings (credentials from environment variables)
    alpaca_api_key_env: str = "ALPACA_API_KEY"
    alpaca_secret_key_env: str = "ALPACA_SECRET_KEY"
    alpaca_base_url: Optional[str] = None  # Auto-selected based on paper_trading if None


class Config(BaseModel):
    """Main configuration for ActiveTrader-LLM"""

    mode: Literal["backtest", "paper-live", "live"] = "backtest"
    data_sources: DataSourcesConfig = Field(default_factory=DataSourcesConfig)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    risk_parameters: RiskParameters = Field(default_factory=RiskParameters)
    trade_validation: TradeValidationConfig = Field(default_factory=TradeValidationConfig)
    position_management: PositionManagementConfig = Field(default_factory=PositionManagementConfig)
    execution_schedule: ExecutionScheduleConfig = Field(default_factory=ExecutionScheduleConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    strategies: List[StrategyConfig] = Field(default_factory=list)
    strategy_switching: StrategySwitchingConfig = Field(default_factory=StrategySwitchingConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    cost_control: CostControlConfig = Field(default_factory=CostControlConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    macro_data: MacroConfig = Field(default_factory=MacroConfig)

    # Technical indicators and market breadth (raw dict for flexibility)
    technical_indicators: Dict[str, Any] = Field(default_factory=dict)
    market_breadth: Dict[str, Any] = Field(default_factory=dict)

    # Storage
    database_path: str = "data/trading.db"
    log_path: str = "logs/trade_log.jsonl"

    # Feature flags
    enable_sentiment: bool = False
    enable_macro: bool = False

    # REMOVED: Default strategy creation - LLM decides dynamically
    # @validator('strategies', pre=True, always=True)
    # def set_default_strategies(cls, v):
    #     if not v:
    #         return [
    #             StrategyConfig(name="momentum_breakout", regime="trending_bull"),
    #             StrategyConfig(name="mean_reversion", regime="range"),
    #             StrategyConfig(name="pullback", regime="mild_trend"),
    #             StrategyConfig(name="sector_rotation", regime="mixed"),
    #         ]
    #     return v
