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
    """
    Market-level breadth - RAW DATA ONLY (no calculations or interpretations).

    LLM interprets these raw values to determine market environment.
    """
    timestamp: str

    # Raw advance/decline counts
    stocks_advancing: int = Field(description="Count of stocks above 200-day SMA")
    stocks_declining: int = Field(description="Count of stocks below 200-day SMA")
    total_stocks: int = Field(description="Total stocks in universe")

    # Raw new highs/lows counts
    new_highs: int = Field(description="Stocks at/near 52-week highs")
    new_lows: int = Field(description="Stocks at/near 52-week lows")

    # Raw volume data
    up_volume: int = Field(description="Total volume in advancing stocks")
    down_volume: int = Field(description="Total volume in declining stocks")
    total_volume: int = Field(description="Total market volume")

    # Aggregated indicator stats (raw averages)
    avg_rsi: Optional[float] = Field(None, description="Average RSI across universe")
    pct_above_sma200_daily: Optional[float] = Field(None, description="% of stocks above 200-day SMA")
    pct_above_sma50_weekly: Optional[float] = Field(None, description="% of stocks above 50-week SMA")


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
