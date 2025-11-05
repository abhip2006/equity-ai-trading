"""
Battle configuration schema using Pydantic for validation.

This module defines the configuration structure for the LLM Battle Royale,
where multiple AI models compete simultaneously in paper trading.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, validator, field_validator
from datetime import datetime


class BattleMetadata(BaseModel):
    """Battle metadata and basic settings"""
    name: str = Field(..., min_length=1, description="Name of the battle round")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    mode: Literal["paper-live"] = Field(default="paper-live", description="Must be paper-live for battle mode")
    duration_days: int = Field(default=5, gt=0, description="Number of trading days to run battle")
    description: str = Field(default="", description="Optional battle description")

    @field_validator('start_date')
    @classmethod
    def validate_start_date(cls, v: str) -> str:
        """Validate start date format"""
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError(f"start_date must be in YYYY-MM-DD format, got: {v}")


class TradingHours(BaseModel):
    """Trading hours configuration"""
    start: str = Field(default="09:30", description="Market open time (HH:MM)")
    end: str = Field(default="16:00", description="Market close time (HH:MM)")
    timezone: str = Field(default="America/New_York", description="Timezone for trading hours")

    @field_validator('start', 'end')
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        """Validate time format HH:MM"""
        try:
            hours, minutes = v.split(':')
            h, m = int(hours), int(minutes)
            if not (0 <= h < 24 and 0 <= m < 60):
                raise ValueError
            return v
        except (ValueError, AttributeError):
            raise ValueError(f"Time must be in HH:MM format (00:00-23:59), got: {v}")


class PositionManagementConfig(BaseModel):
    """Position management configuration"""
    max_total_exposure_pct: float = Field(default=80.0, gt=0, le=100, description="Maximum total portfolio exposure")
    allow_duplicate_symbols: bool = Field(default=False, description="Prevent multiple positions in same symbol")
    auto_close_eod: bool = Field(default=True, description="Close all positions before market close")
    eod_close_time: str = Field(default="15:55", description="Time to close positions (HH:MM)")


class TradeValidationConfig(BaseModel):
    """Trade plan validation configuration"""
    enabled: bool = Field(default=True, description="Enable trade validation")
    max_position_pct: float = Field(default=100.0, gt=0, le=100, description="Maximum position size as % of capital")
    min_risk_reward_ratio: float = Field(default=0.0, ge=0, description="Minimum reward/risk ratio")
    max_price_deviation_pct: float = Field(default=50.0, gt=0, description="Max deviation from current price (%)")
    min_stop_distance_pct: float = Field(default=0.1, gt=0, description="Minimum stop distance from entry (%)")
    max_stop_distance_pct: float = Field(default=50.0, gt=0, description="Maximum stop distance from entry (%)")


class RiskParametersConfig(BaseModel):
    """Risk parameters configuration"""
    enforce_daily_drawdown: bool = Field(default=True, description="Enable emergency circuit breaker")
    emergency_drawdown_limit: float = Field(default=0.20, gt=0, le=1, description="Daily loss % that triggers halt")
    min_win_rate: float = Field(default=0.35, ge=0, le=1, description="Minimum win rate threshold")
    min_net_pl: float = Field(default=0.0, description="Minimum net P/L threshold")


class ExecutionConfig(BaseModel):
    """Execution configuration"""
    broker: Literal["alpaca"] = Field(default="alpaca", description="Broker to use (must be alpaca for battle)")
    paper_trading: bool = Field(default=True, description="Must be true for battle mode")
    order_type: Literal["market", "limit"] = Field(default="market", description="Order type")
    time_in_force: Literal["day", "gtc", "ioc", "fok"] = Field(default="day", description="Time in force")
    commission_per_trade: float = Field(default=0.0, ge=0, description="Commission per trade (USD)")
    slippage_bps: float = Field(default=2.0, ge=0, description="Slippage in basis points")


class TechnicalIndicatorsConfig(BaseModel):
    """Technical indicators configuration"""
    quality: Dict[str, Any] = Field(default_factory=dict)
    daily: Dict[str, Any] = Field(default_factory=dict)
    weekly: Dict[str, Any] = Field(default_factory=dict)


class MarketBreadthConfig(BaseModel):
    """Market breadth configuration"""
    enabled: bool = Field(default=True, description="Enable market breadth analysis")
    new_highs_lows: Dict[str, Any] = Field(default_factory=dict)


class ScannerConfig(BaseModel):
    """Scanner configuration"""
    enabled: bool = Field(default=False, description="Enable scanner (typically false for battle)")
    universe_source: str = Field(default="file", description="Universe source")
    universe_file_path: str = Field(default="data/stock_universe.txt", description="Path to universe file")


class SharedSettings(BaseModel):
    """
    Shared settings applied uniformly to all models for fair comparison.

    These settings ensure that all models:
    - Trade the same universe
    - Start with the same capital
    - Follow the same trading hours
    - Use the same risk parameters
    - Have identical execution settings
    """
    # Universe & Data
    universe_source: str = Field(default="file", description="Universe source")
    universe_file_path: str = Field(default="data/stock_universe.txt", description="Path to universe file")
    data_interval: str = Field(default="1d", description="Data interval (1h, 1d, 5m, 15m)")
    lookback_days: int = Field(default=90, gt=0, description="Historical data lookback period")

    # Capital & Risk
    initial_capital: float = Field(default=100000.0, gt=0, description="Starting capital per model")

    # Trading Hours & Cycle
    trading_hours: TradingHours = Field(default_factory=TradingHours)
    decision_cycle_times: List[str] = Field(
        default_factory=lambda: ["09:30", "12:00", "15:30"],
        description="Times to run decision cycles"
    )
    monitor_exits_interval_seconds: int = Field(default=60, gt=0, description="Exit monitoring interval")

    # Position Management
    position_management: PositionManagementConfig = Field(default_factory=PositionManagementConfig)

    # Trade Validation
    trade_validation: TradeValidationConfig = Field(default_factory=TradeValidationConfig)

    # Risk Parameters
    risk_parameters: RiskParametersConfig = Field(default_factory=RiskParametersConfig)

    # Execution Settings
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    # Technical Indicators
    technical_indicators: TechnicalIndicatorsConfig = Field(default_factory=TechnicalIndicatorsConfig)

    # Market Breadth
    market_breadth: MarketBreadthConfig = Field(default_factory=MarketBreadthConfig)

    # Scanner
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)

    # Feature Flags
    enable_sentiment: bool = Field(default=False, description="Enable sentiment analysis")
    enable_macro: bool = Field(default=False, description="Enable macro analysis")

    @field_validator('initial_capital')
    @classmethod
    def validate_initial_capital(cls, v: float) -> float:
        """Ensure reasonable capital amount"""
        if v < 1000:
            raise ValueError(f"initial_capital must be at least $1,000, got: ${v:,.2f}")
        return v


class LLMConfig(BaseModel):
    """LLM configuration for a specific model"""
    provider: Literal["openai", "anthropic", "google"] = Field(..., description="LLM provider")
    model: str = Field(..., min_length=1, description="Model name/ID")
    api_key_env: str = Field(..., min_length=1, description="Environment variable for API key")
    temperature: float = Field(default=0.3, ge=0, le=2, description="Temperature for generation")
    max_tokens: int = Field(default=2000, gt=0, description="Maximum tokens to generate")
    base_url: Optional[str] = Field(default=None, description="Custom base URL for API (optional)")


class AlpacaConfig(BaseModel):
    """Alpaca broker configuration for a specific model"""
    api_key_env: str = Field(..., min_length=1, description="Environment variable for Alpaca API key")
    secret_key_env: str = Field(..., min_length=1, description="Environment variable for Alpaca secret key")
    base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Alpaca API base URL"
    )

    @field_validator('base_url')
    @classmethod
    def validate_paper_url(cls, v: str) -> str:
        """Ensure using paper trading URL for battle"""
        if "paper-api" not in v:
            raise ValueError(f"Battle mode requires paper trading URL, got: {v}")
        return v


class ModelConfig(BaseModel):
    """
    Configuration for a single model in the battle.

    Each model must have:
    - Unique LLM configuration
    - Separate Alpaca paper account
    - Isolated database and logs
    """
    enabled: bool = Field(default=True, description="Enable this model in the battle")
    name: str = Field(..., min_length=1, description="Display name for the model")
    description: str = Field(default="", description="Optional model description")

    # LLM Configuration
    llm: LLMConfig = Field(..., description="LLM configuration")

    # Alpaca Configuration
    alpaca: AlpacaConfig = Field(..., description="Alpaca broker configuration")

    # Storage Paths
    database_path: str = Field(..., min_length=1, description="Path to model's database file")
    log_path: str = Field(..., min_length=1, description="Path to model's log file")

    @field_validator('database_path', 'log_path')
    @classmethod
    def validate_paths(cls, v: str) -> str:
        """Ensure paths are not empty and don't conflict"""
        if not v or v.strip() == "":
            raise ValueError("Path cannot be empty")
        return v


class MonitoringAlerts(BaseModel):
    """Alert threshold configuration"""
    drawdown_warning_pct: float = Field(default=10.0, gt=0, description="Drawdown % to trigger warning")
    drawdown_critical_pct: float = Field(default=15.0, gt=0, description="Drawdown % to trigger critical alert")
    min_acceptable_win_rate: float = Field(default=0.30, ge=0, le=1, description="Minimum acceptable win rate")
    position_concentration_warning_pct: float = Field(
        default=40.0, gt=0, le=100,
        description="Single position % to trigger warning"
    )

    @field_validator('drawdown_critical_pct')
    @classmethod
    def validate_drawdown_levels(cls, v: float, info) -> float:
        """Ensure critical drawdown is greater than warning"""
        if 'drawdown_warning_pct' in info.data and v <= info.data['drawdown_warning_pct']:
            raise ValueError(
                f"drawdown_critical_pct ({v}) must be greater than "
                f"drawdown_warning_pct ({info.data['drawdown_warning_pct']})"
            )
        return v


class MonitoringConfig(BaseModel):
    """Monitoring and reporting configuration"""
    # Performance Tracking
    metrics_update_interval_seconds: int = Field(
        default=300, gt=0,
        description="Metrics update interval in seconds"
    )

    # Leaderboard
    leaderboard_file: str = Field(
        default="data/battle/leaderboard.json",
        description="Path to leaderboard file"
    )

    # Alerts
    alerts: MonitoringAlerts = Field(default_factory=MonitoringAlerts)

    # Reporting
    daily_report_time: str = Field(default="16:30", description="Time to generate daily report (HH:MM)")
    report_output_dir: str = Field(default="reports/battle", description="Directory for battle reports")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    console_log: bool = Field(default=True, description="Enable console logging")
    file_log: bool = Field(default=True, description="Enable file logging")
    battle_summary_log: str = Field(
        default="logs/battle/battle_summary.log",
        description="Path to battle summary log"
    )


class DevSettings(BaseModel):
    """Development and testing settings"""
    dry_run: bool = Field(default=False, description="Simulate without real API calls")
    test_mode: bool = Field(default=False, description="Use reduced capital and shorter cycles")
    test_mode_capital: float = Field(default=10000.0, gt=0, description="Capital for test mode")
    use_mock_data: bool = Field(default=False, description="Use mock data for testing")
    parallel_execution: bool = Field(default=True, description="Run models simultaneously vs sequentially")
    max_parallel_workers: int = Field(default=6, gt=0, le=10, description="Maximum parallel workers")
    enable_graceful_shutdown: bool = Field(default=True, description="Enable graceful shutdown")
    shutdown_timeout_seconds: int = Field(default=30, gt=0, description="Shutdown timeout in seconds")


class BattleConfig(BaseModel):
    """
    Top-level battle configuration.

    Defines:
    - Battle metadata (name, dates, mode)
    - Shared settings (applied to all models)
    - Individual model configurations
    - Monitoring and alerting
    - Development settings

    Validation ensures:
    - All required API keys are defined
    - Database paths don't conflict
    - All models use paper trading
    - Trading hours are valid
    """
    battle: BattleMetadata = Field(..., description="Battle metadata")
    shared_settings: SharedSettings = Field(..., description="Settings shared across all models")
    models: Dict[str, ModelConfig] = Field(..., min_length=1, description="Model configurations")
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    dev_settings: DevSettings = Field(default_factory=DevSettings)

    @field_validator('models')
    @classmethod
    def validate_models(cls, v: Dict[str, ModelConfig]) -> Dict[str, ModelConfig]:
        """Validate model configurations"""
        if not v:
            raise ValueError("At least one model must be configured")

        # Check for database path conflicts
        db_paths = set()
        log_paths = set()
        api_key_envs = set()

        for model_id, config in v.items():
            # Check database path uniqueness
            if config.database_path in db_paths:
                raise ValueError(
                    f"Model '{model_id}' has duplicate database_path: {config.database_path}"
                )
            db_paths.add(config.database_path)

            # Check log path uniqueness
            if config.log_path in log_paths:
                raise ValueError(
                    f"Model '{model_id}' has duplicate log_path: {config.log_path}"
                )
            log_paths.add(config.log_path)

            # Check Alpaca API key uniqueness (each model needs its own account)
            alpaca_key_pair = (config.alpaca.api_key_env, config.alpaca.secret_key_env)
            if alpaca_key_pair in api_key_envs:
                raise ValueError(
                    f"Model '{model_id}' has duplicate Alpaca credentials. "
                    f"Each model must have a separate Alpaca paper account."
                )
            api_key_envs.add(alpaca_key_pair)

        return v

    @field_validator('battle')
    @classmethod
    def validate_battle_mode(cls, v: BattleMetadata) -> BattleMetadata:
        """Ensure battle mode is paper-live"""
        if v.mode != "paper-live":
            raise ValueError(f"Battle mode must be 'paper-live', got: {v.mode}")
        return v

    def get_enabled_models(self) -> Dict[str, ModelConfig]:
        """Get only enabled models"""
        return {
            model_id: config
            for model_id, config in self.models.items()
            if config.enabled
        }

    def get_all_required_env_vars(self) -> List[str]:
        """Get list of all required environment variables"""
        env_vars = set()

        for config in self.models.values():
            if config.enabled:
                # LLM API key
                env_vars.add(config.llm.api_key_env)

                # Alpaca credentials
                env_vars.add(config.alpaca.api_key_env)
                env_vars.add(config.alpaca.secret_key_env)

        return sorted(list(env_vars))

    def validate_env_vars_present(self, env_dict: Dict[str, str]) -> List[str]:
        """
        Validate that all required environment variables are present.

        Args:
            env_dict: Dictionary of environment variables (typically os.environ)

        Returns:
            List of missing environment variable names
        """
        required = self.get_all_required_env_vars()
        missing = [var for var in required if var not in env_dict or not env_dict[var]]
        return missing

    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        extra = 'forbid'  # Raise error on unknown fields
