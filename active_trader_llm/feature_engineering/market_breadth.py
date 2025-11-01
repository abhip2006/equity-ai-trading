"""
Market-level breadth features and regime detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

from .models import FeatureSet, MarketSnapshot

logger = logging.getLogger(__name__)


class MarketBreadthAnalyzer:
    """Analyzes market-level breadth and determines regime"""

    def __init__(self, config: Dict):
        self.config = config
        self.hl_config = config['new_highs_lows']
        self.regime_config = config['regime_thresholds']

    def analyze(self, price_df: pd.DataFrame, features: Dict[str, FeatureSet]) -> MarketSnapshot:
        """Calculate market breadth features"""
        if not features:
            logger.error("Cannot calculate market breadth: No features provided")
            logger.error("Market breadth requires real feature data for all symbols")
            raise ValueError("Market breadth analysis requires feature data - cannot proceed with empty features")

        timestamp = list(features.values())[0].timestamp

        # Calculate advance/decline (using daily SMA 200)
        advances, declines = self._calculate_advance_decline(features)
        ad_ratio = self._safe_ratio(advances, declines)

        # Breadth score
        total = len(features)
        breadth_score = (advances - declines) / total if total > 0 else 0.0

        # New highs/lows
        new_highs, new_lows = self._calculate_new_highs_lows(price_df, features)

        # Volume metrics
        up_volume_ratio = self._calculate_volume_ratio(features)

        # Aggregated stats
        avg_rsi = self._avg_indicator(features, 'RSI_14_daily')
        pct_above_sma200 = self._pct_above_ma(features, 'SMA_200_daily')
        pct_above_sma50_weekly = self._pct_above_ma(features, 'SMA_50week')

        # Determine regime
        regime = self._determine_regime(breadth_score, avg_rsi, pct_above_sma200)

        return MarketSnapshot(
            timestamp=timestamp,
            regime_hint=regime,
            breadth_score=breadth_score,
            advance_decline_ratio=ad_ratio,
            new_highs=new_highs,
            new_lows=new_lows,
            up_volume_ratio=up_volume_ratio,
            avg_rsi=avg_rsi,
            pct_above_sma200_daily=pct_above_sma200,
            pct_above_sma50_weekly=pct_above_sma50_weekly
        )

    def _calculate_advance_decline(self, features: Dict[str, FeatureSet]) -> tuple:
        """Count stocks above/below daily SMA 200"""
        advances = sum(1 for f in features.values()
                      if f.close > f.get_daily('SMA_200_daily', f.close))
        declines = len(features) - advances
        return advances, declines

    def _safe_ratio(self, numerator: float, denominator: float) -> float:
        """Division by zero protection"""
        if denominator > 0:
            return numerator / denominator
        elif numerator > 0:
            return float('inf')
        else:
            return 1.0

    def _calculate_new_highs_lows(self, price_df: pd.DataFrame,
                                   features: Dict[str, FeatureSet]) -> tuple:
        """Calculate new highs/lows"""
        lookback = self.hl_config['lookback_days']
        min_bars = self.hl_config['min_data_bars']
        high_thresh = self.hl_config['high_threshold_pct']
        low_thresh = self.hl_config['low_threshold_pct']

        new_highs = 0
        new_lows = 0

        for symbol, feat in features.items():
            symbol_data = price_df[price_df['symbol'] == symbol].tail(lookback)

            if len(symbol_data) < min_bars:
                continue

            period_high = symbol_data['high'].max()
            period_low = symbol_data['low'].min()
            current_close = feat.close

            if current_close >= period_high * high_thresh:
                new_highs += 1
            if current_close <= period_low * low_thresh:
                new_lows += 1

        return new_highs, new_lows

    def _calculate_volume_ratio(self, features: Dict[str, FeatureSet]) -> float:
        """Calculate up volume ratio"""
        up_volume = sum(f.volume for f in features.values() if f.close > f.ohlcv['open'])
        total_volume = sum(f.volume for f in features.values())

        if total_volume > 0:
            return up_volume / total_volume
        else:
            return 0.5

    def _avg_indicator(self, features: Dict[str, FeatureSet],
                      indicator_name: str) -> Optional[float]:
        """Average daily indicator value"""
        values = [f.get_daily(indicator_name) for f in features.values()]
        values = [v for v in values if v is not None]
        return np.mean(values) if values else None

    def _pct_above_ma(self, features: Dict[str, FeatureSet],
                     ma_name: str) -> Optional[float]:
        """Percentage above moving average"""
        # Determine if daily or weekly
        is_weekly = 'week' in ma_name

        count = 0
        for f in features.values():
            ma_value = f.get_weekly(ma_name) if is_weekly else f.get_daily(ma_name)
            if ma_value and f.close > ma_value:
                count += 1

        return count / len(features) if features else None

    def _determine_regime(self, breadth: float, rsi: Optional[float],
                         pct_sma200: Optional[float]) -> str:
        """
        Determine market regime based on available indicators.

        Breadth is the PRIMARY signal. RSI and pct_sma200 are confirmatory.
        If confirmatory indicators are missing, regime is determined from breadth alone.
        """
        t = self.regime_config

        # Warn if key indicators are missing - regime will be breadth-only
        if rsi is None:
            logger.warning("Average RSI unavailable - regime determined from breadth only")
        if pct_sma200 is None:
            logger.warning("Pct above SMA200 unavailable - regime determined from breadth only")

        # Determine regime using available indicators (NO fabricated values)

        # Risk-off check: breadth OR rsi (if available)
        risk_off_breadth = breadth < t['risk_off_breadth']
        risk_off_rsi = (rsi is not None and rsi < t['risk_off_rsi'])
        if risk_off_breadth or risk_off_rsi:
            return "risk_off"

        # Trending bull check: breadth AND pct_sma200 (if available)
        bull_breadth = breadth > t['trending_bull_breadth']
        bull_sma200 = (pct_sma200 is not None and pct_sma200 > t['trending_bull_pct_above_ema200'])
        # Only confirm trending_bull if we have pct_sma200 data OR breadth is very strong
        if bull_breadth:
            if pct_sma200 is None:
                # Require stronger breadth confirmation without pct_sma200
                if breadth > t['trending_bull_breadth'] * 1.2:
                    return "trending_bull"
            elif bull_sma200:
                return "trending_bull"

        # Trending bear check: breadth only (no confirmatory needed)
        if breadth < t['trending_bear_breadth']:
            return "trending_bear"

        # Default: range
        return "range"
