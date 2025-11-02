"""
Type-safe Pydantic models for indicator outputs.
"""

from pydantic import BaseModel, Field
from typing import Dict, Literal, Optional


class FeatureSet(BaseModel):
    """
    Features for a single symbol at a point in time.

    Contains both daily and weekly indicators.
    """
    symbol: str
    timestamp: str

    # Daily indicators
    daily_indicators: Dict[str, float] = Field(
        description="Daily timeframe indicators"
    )

    # Weekly indicators (calculated from daily data)
    weekly_indicators: Dict[str, float] = Field(
        description="Weekly timeframe indicators"
    )

    # Raw OHLCV data
    ohlcv: Dict[str, float] = Field(
        description="Open, High, Low, Close, Volume"
    )

    # Convenience accessors
    def get_daily(self, indicator_name: str, default: Optional[float] = None) -> Optional[float]:
        """Get daily indicator value"""
        return self.daily_indicators.get(indicator_name, default)

    def get_weekly(self, indicator_name: str, default: Optional[float] = None) -> Optional[float]:
        """Get weekly indicator value"""
        return self.weekly_indicators.get(indicator_name, default)

    @property
    def close(self) -> float:
        return self.ohlcv['close']

    @property
    def volume(self) -> int:
        return int(self.ohlcv['volume'])

    @property
    def high(self) -> float:
        return self.ohlcv['high']

    @property
    def low(self) -> float:
        return self.ohlcv['low']

    @property
    def open(self) -> float:
        return self.ohlcv['open']

    @property
    def price(self) -> float:
        """Current price (alias for close)"""
        return self.close


class MarketSnapshot(BaseModel):
    """Market-level breadth and regime information"""
    timestamp: str
    regime_hint: Literal["risk_off", "range", "trending_bull", "trending_bear"]
    breadth_score: float = Field(..., ge=-1.0, le=1.0)
    advance_decline_ratio: float
    new_highs: int
    new_lows: int
    up_volume_ratio: float

    # Aggregated indicator stats
    avg_rsi: Optional[float] = None
    pct_above_sma200_daily: Optional[float] = None
    pct_above_sma50_weekly: Optional[float] = None


class MacroSnapshot(BaseModel):
    """
    Raw macro-economic data snapshot.

    Contains ONLY raw numerical values - NO interpretations.
    LLM interprets these values to determine macro environment.
    """
    timestamp: str

    # Volatility indices (raw index values)
    vix: Optional[float] = Field(None, description="S&P 500 Volatility Index")
    vxn: Optional[float] = Field(None, description="Nasdaq-100 Volatility Index")
    move_index: Optional[float] = Field(None, description="Treasury Volatility Index (MOVE)")

    # Treasury yields (raw percentages)
    treasury_10y: Optional[float] = Field(None, description="10-Year Treasury Yield (%)")
    treasury_2y: Optional[float] = Field(None, description="2-Year Treasury Yield (%)")
    treasury_30y: Optional[float] = Field(None, description="30-Year Treasury Yield (%)")
    yield_curve_spread: Optional[float] = Field(None, description="10Y-2Y Spread (simple subtraction)")

    # Commodities (raw prices)
    gold_price: Optional[float] = Field(None, description="Gold Futures Price ($/oz)")
    oil_price: Optional[float] = Field(None, description="Crude Oil Futures Price ($/barrel)")

    # Currency (raw index value)
    dollar_index: Optional[float] = Field(None, description="US Dollar Index (DXY)")

    # NYSE Market Breadth (raw counts - full market, not just universe)
    nyse_advancing: Optional[int] = Field(None, description="NYSE advancing issues count")
    nyse_declining: Optional[int] = Field(None, description="NYSE declining issues count")
    nyse_unchanged: Optional[int] = Field(None, description="NYSE unchanged issues count")
    nyse_advancing_volume: Optional[int] = Field(None, description="Volume in advancing stocks")
    nyse_declining_volume: Optional[int] = Field(None, description="Volume in declining stocks")
    nyse_new_highs: Optional[int] = Field(None, description="52-week new highs count")
    nyse_new_lows: Optional[int] = Field(None, description="52-week new lows count")

    # Calculated breadth metrics (simple math, no interpretation)
    advance_decline_ratio: Optional[float] = Field(None, description="advancing / declining")
    up_volume_ratio: Optional[float] = Field(None, description="adv_volume / total_volume")
