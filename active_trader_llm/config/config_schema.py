"""
Configuration schema using Pydantic for validation.
"""

from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field, validator


class IndicatorConfig(BaseModel):
    """Technical indicator periods (customize for different trading styles)"""
    # RSI
    rsi_period: int = 14

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands
    bollinger_period: int = 20
    bollinger_std: float = 2.0

    # ATR
    atr_period: int = 14

    # Moving Averages
    ema_short: int = 50
    ema_long: int = 200
    sma_period: int = 20


class RegimeThresholds(BaseModel):
    """Market regime classification thresholds"""
    # Risk-off regime triggers
    risk_off_breadth: float = -0.5
    risk_off_rsi: float = 35.0

    # Trending bull regime triggers
    trending_bull_breadth: float = 0.5
    trending_bull_pct_above_ema200: float = 0.6

    # Trending bear regime trigger
    trending_bear_breadth: float = -0.3


class MacroThresholds(BaseModel):
    """Macro analyst VIX interpretation thresholds"""
    vix_risk_off: float = 25.0  # VIX above this = risk-off
    vix_risk_on: float = 12.0   # VIX below this = risk-on


class ScannerConfig(BaseModel):
    """Market scanner configuration for dynamic universe discovery"""
    enabled: bool = True

    # Base universe to scan
    base_universe: Literal["sp500", "sp400", "sp600", "nasdaq100", "custom"] = "sp500"
    custom_tickers: List[str] = []  # Used if base_universe = "custom"

    # Volume filters
    min_avg_volume: int = 1_000_000  # Minimum 20-day average volume
    min_dollar_volume: float = 10_000_000  # Minimum daily dollar volume ($10M)

    # Price filters
    min_price: float = 5.0  # Avoid penny stocks
    max_price: float = 1000.0  # Avoid extremely expensive stocks

    # Market cap filter (optional)
    min_market_cap: Optional[float] = None  # e.g., 1_000_000_000 for $1B+

    # Volatility filter (optional)
    min_atr_pct: Optional[float] = None  # e.g., 0.02 for 2%+ daily ATR
    max_atr_pct: Optional[float] = None  # e.g., 0.10 for <10% daily ATR

    # Universe size limits
    max_universe_size: int = 50  # Maximum symbols to analyze (cost control)

    # Scan frequency
    scan_interval_hours: int = 24  # Re-scan daily by default

    # Cache settings
    use_cache: bool = True
    cache_expiry_hours: int = 24


class DataSourcesConfig(BaseModel):
    prices: str = "yfinance"
    interval: str = "1h"
    universe: List[str] = ["AAPL", "MSFT", "SPY", "QQQ"]  # Used if scanner disabled
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

    # Strategy scoring weights (for selecting best strategy)
    win_rate_weight: float = 50.0
    avg_return_weight: float = 0.5
    avg_return_cap: float = 100.0  # Cap returns at this value for scoring
    avg_rr_weight: float = 10.0


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

    # Technical configuration
    indicators: IndicatorConfig = Field(default_factory=IndicatorConfig)
    regime_thresholds: RegimeThresholds = Field(default_factory=RegimeThresholds)
    macro_thresholds: MacroThresholds = Field(default_factory=MacroThresholds)

    # Market scanning
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)

    # Data and execution
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
