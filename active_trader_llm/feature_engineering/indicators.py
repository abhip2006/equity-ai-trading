"""
Technical indicators and market breadth feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Literal
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class FeatureSet(BaseModel):
    """Feature set for a single symbol at a point in time"""
    symbol: str
    timestamp: str
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    ema_50: float
    ema_200: float
    sma_20: float
    atr_14: float
    bollinger_upper: float
    bollinger_middle: float
    bollinger_lower: float
    bollinger_bandwidth: float
    volume: int
    close: float
    high: float
    low: float
    open: float


class MarketSnapshot(BaseModel):
    """Aggregated market features and regime"""
    timestamp: str
    regime_hint: Literal["risk_off", "range", "trending_bull", "trending_bear"]
    breadth_score: float = Field(..., ge=-1.0, le=1.0)
    sector_strengths: Dict[str, float] = Field(default_factory=dict)
    vix: float = 15.0
    advance_decline_ratio: float = 1.0
    new_highs: int = 0
    new_lows: int = 0
    up_volume_ratio: float = 0.5


class FeatureEngineer:
    """Compute technical indicators and market breadth features"""

    @staticmethod
    def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Handle division by zero when loss = 0 (strong uptrend with no down days)
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        # Fill NaN values with neutral RSI (50) when calculation undefined
        rsi = rsi.fillna(50.0)
        return rsi

    @staticmethod
    def compute_macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def compute_bollinger_bands(
        prices: pd.Series,
        period: int = 20,
        num_std: float = 2.0
    ) -> tuple:
        """Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper = middle + (std * num_std)
        lower = middle - (std * num_std)

        return upper, middle, lower

    @staticmethod
    def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    @staticmethod
    def compute_ema(prices: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()

    @staticmethod
    def compute_sma(prices: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return prices.rolling(window=period).mean()

    def build_features(self, price_df: pd.DataFrame) -> Dict[str, FeatureSet]:
        """
        Compute all features for each symbol.

        Args:
            price_df: DataFrame with columns: symbol, timestamp, open, high, low, close, volume

        Returns:
            Dictionary mapping symbol -> FeatureSet for latest timestamp
        """
        features = {}

        for symbol in price_df['symbol'].unique():
            symbol_df = price_df[price_df['symbol'] == symbol].copy()
            symbol_df.sort_values('timestamp', inplace=True)

            if len(symbol_df) < 200:  # Need enough data for EMA200
                logger.warning(f"Insufficient data for {symbol}: {len(symbol_df)} bars")
                continue

            # Compute indicators
            symbol_df['rsi'] = self.compute_rsi(symbol_df['close'])

            macd, signal, hist = self.compute_macd(symbol_df['close'])
            symbol_df['macd'] = macd
            symbol_df['macd_signal'] = signal
            symbol_df['macd_hist'] = hist

            symbol_df['ema_50'] = self.compute_ema(symbol_df['close'], 50)
            symbol_df['ema_200'] = self.compute_ema(symbol_df['close'], 200)
            symbol_df['sma_20'] = self.compute_sma(symbol_df['close'], 20)

            symbol_df['atr_14'] = self.compute_atr(
                symbol_df['high'],
                symbol_df['low'],
                symbol_df['close']
            )

            bb_upper, bb_middle, bb_lower = self.compute_bollinger_bands(symbol_df['close'])
            symbol_df['bollinger_upper'] = bb_upper
            symbol_df['bollinger_middle'] = bb_middle
            symbol_df['bollinger_lower'] = bb_lower
            # Avoid division by zero when middle band is zero (rare but possible with corrupted data)
            symbol_df['bollinger_bandwidth'] = (bb_upper - bb_lower) / bb_middle.replace(0, np.nan)
            symbol_df['bollinger_bandwidth'] = symbol_df['bollinger_bandwidth'].fillna(0.1)

            # Get latest row
            latest = symbol_df.iloc[-1]

            features[symbol] = FeatureSet(
                symbol=symbol,
                timestamp=str(latest['timestamp']),
                rsi=float(latest['rsi']) if pd.notna(latest['rsi']) else 50.0,
                macd=float(latest['macd']) if pd.notna(latest['macd']) else 0.0,
                macd_signal=float(latest['macd_signal']) if pd.notna(latest['macd_signal']) else 0.0,
                macd_hist=float(latest['macd_hist']) if pd.notna(latest['macd_hist']) else 0.0,
                ema_50=float(latest['ema_50']) if pd.notna(latest['ema_50']) else latest['close'],
                ema_200=float(latest['ema_200']) if pd.notna(latest['ema_200']) else latest['close'],
                sma_20=float(latest['sma_20']) if pd.notna(latest['sma_20']) else latest['close'],
                atr_14=float(latest['atr_14']) if pd.notna(latest['atr_14']) else 1.0,
                bollinger_upper=float(latest['bollinger_upper']) if pd.notna(latest['bollinger_upper']) else latest['close'],
                bollinger_middle=float(latest['bollinger_middle']) if pd.notna(latest['bollinger_middle']) else latest['close'],
                bollinger_lower=float(latest['bollinger_lower']) if pd.notna(latest['bollinger_lower']) else latest['close'],
                bollinger_bandwidth=float(latest['bollinger_bandwidth']) if pd.notna(latest['bollinger_bandwidth']) else 0.1,
                volume=int(latest['volume']),
                close=float(latest['close']),
                high=float(latest['high']),
                low=float(latest['low']),
                open=float(latest['open'])
            )

        return features

    def compute_breadth_features(self, price_df: pd.DataFrame, features: Dict[str, FeatureSet]) -> MarketSnapshot:
        """
        Compute market breadth and regime features.

        Args:
            price_df: Raw price data
            features: Computed features for all symbols

        Returns:
            MarketSnapshot with breadth indicators and regime hint
        """
        if not features:
            return MarketSnapshot(
                timestamp=str(pd.Timestamp.now()),
                regime_hint="range",
                breadth_score=0.0
            )

        latest_timestamp = list(features.values())[0].timestamp

        # Advance/Decline calculation (simplified)
        advances = sum(1 for f in features.values() if f.close > f.ema_50)
        declines = len(features) - advances

        # Proper handling of zero declines (all stocks advancing)
        if declines > 0:
            ad_ratio = advances / declines
        elif advances > 0:
            ad_ratio = float('inf')  # All advancing, no declines
        else:
            ad_ratio = 1.0  # No stocks, neutral

        # Breadth score: -1 (bearish) to +1 (bullish)
        breadth_score = (advances - declines) / len(features)

        # Calculate actual new highs/lows from price history
        # A stock is at a "new high" if current close is within 2% of its period high
        new_highs = 0
        new_lows = 0

        for symbol in features.keys():
            symbol_data = price_df[price_df['symbol'] == symbol].tail(252)  # ~1 year of data or less
            if len(symbol_data) > 20:  # Need minimum data
                recent_high = symbol_data['high'].max()
                recent_low = symbol_data['low'].min()
                current_close = features[symbol].close

                # Within 2% of period high = new high, within 2% of period low = new low
                if current_close >= recent_high * 0.98:
                    new_highs += 1
                if current_close <= recent_low * 1.02:
                    new_lows += 1

        # Up/Down volume (simplified)
        up_volume = sum(f.volume for f in features.values() if f.close > f.open)
        total_volume = sum(f.volume for f in features.values())

        # Proper handling of zero volume
        up_volume_ratio = up_volume / total_volume if total_volume > 0 else 0.5

        # Regime determination
        avg_rsi = np.mean([f.rsi for f in features.values()])
        avg_macd_hist = np.mean([f.macd_hist for f in features.values()])
        pct_above_ema200 = sum(1 for f in features.values() if f.close > f.ema_200) / len(features)

        if breadth_score < -0.5 or avg_rsi < 35:
            regime = "risk_off"
        elif breadth_score > 0.5 and avg_macd_hist > 0 and pct_above_ema200 > 0.6:
            regime = "trending_bull"
        elif breadth_score < -0.3 and avg_macd_hist < 0:
            regime = "trending_bear"
        else:
            regime = "range"

        return MarketSnapshot(
            timestamp=latest_timestamp,
            regime_hint=regime,
            breadth_score=breadth_score,
            advance_decline_ratio=ad_ratio,
            new_highs=new_highs,
            new_lows=new_lows,
            up_volume_ratio=up_volume_ratio
        )


# Example usage
if __name__ == "__main__":
    # Test with sample data
    dates = pd.date_range('2024-01-01', periods=250, freq='D')
    sample_df = pd.DataFrame({
        'symbol': ['AAPL'] * 250,
        'timestamp': dates,
        'open': np.random.randn(250).cumsum() + 150,
        'high': np.random.randn(250).cumsum() + 152,
        'low': np.random.randn(250).cumsum() + 148,
        'close': np.random.randn(250).cumsum() + 150,
        'volume': np.random.randint(1000000, 10000000, 250)
    })

    engineer = FeatureEngineer()
    features = engineer.build_features(sample_df)

    print(f"Computed features for {len(features)} symbols")
    for symbol, feat in features.items():
        print(f"\n{symbol}:")
        print(f"  RSI: {feat.rsi:.2f}")
        print(f"  MACD: {feat.macd:.4f}")
        print(f"  EMA50: {feat.ema_50:.2f}, EMA200: {feat.ema_200:.2f}")
