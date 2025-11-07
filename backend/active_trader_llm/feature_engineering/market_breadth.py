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
        # regime_config no longer needed - LLM interprets raw data

    def analyze(self, price_df: pd.DataFrame, features: Dict[str, FeatureSet]) -> MarketSnapshot:
        """
        Calculate RAW market breadth data (no interpretations or scores).

        LLM receives raw counts and determines market environment itself.
        """
        if not features:
            logger.error("Cannot calculate market breadth: No features provided")
            logger.error("Market breadth requires real feature data for all symbols")
            raise ValueError("Market breadth analysis requires feature data - cannot proceed with empty features")

        timestamp = list(features.values())[0].timestamp

        # Calculate raw advance/decline counts (using daily SMA 200)
        advances, declines = self._calculate_advance_decline(features)
        total = len(features)

        # Calculate raw new highs/lows counts
        new_highs, new_lows = self._calculate_new_highs_lows(price_df, features)

        # Calculate raw volume data
        up_volume, down_volume, total_volume = self._calculate_volume_data(features)

        # Aggregated stats (raw averages, no thresholds)
        avg_rsi = self._avg_indicator(features, 'RSI_14_daily')
        pct_above_sma200 = self._pct_above_ma(features, 'SMA_200_daily')
        pct_above_sma50_weekly = self._pct_above_ma(features, 'SMA_50week')

        return MarketSnapshot(
            timestamp=timestamp,
            stocks_advancing=advances,
            stocks_declining=declines,
            total_stocks=total,
            new_highs=new_highs,
            new_lows=new_lows,
            up_volume=up_volume,
            down_volume=down_volume,
            total_volume=total_volume,
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

    def _calculate_volume_data(self, features: Dict[str, FeatureSet]) -> tuple:
        """
        Calculate raw volume data (no ratios).

        Returns:
            (up_volume, down_volume, total_volume) tuple of integers
        """
        up_volume = sum(f.volume for f in features.values() if f.close > f.ohlcv['open'])
        down_volume = sum(f.volume for f in features.values() if f.close <= f.ohlcv['open'])
        total_volume = sum(f.volume for f in features.values())

        return int(up_volume), int(down_volume), int(total_volume)

    def _avg_indicator(self, features: Dict[str, FeatureSet],
                      indicator_name: str) -> float:
        """Average daily indicator value"""
        values = [f.get_daily(indicator_name) for f in features.values()]
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else 0.0

    def _pct_above_ma(self, features: Dict[str, FeatureSet],
                     ma_name: str) -> float:
        """Percentage above moving average"""
        # Determine if daily or weekly
        is_weekly = 'week' in ma_name

        count = 0
        for f in features.values():
            ma_value = f.get_weekly(ma_name) if is_weekly else f.get_daily(ma_name)
            if ma_value and f.close > ma_value:
                count += 1

        return count / len(features) if features else 0.0

