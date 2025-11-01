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
