"""
Main calculation engine for technical indicators.
"""

import pandas as pd
from typing import Dict, List, Tuple
import logging

from .base import BaseIndicator
from .daily_indicators import DailyEMA, DailySMA, RSI, ATR
from .weekly_indicators import WeeklyEMA, WeeklySMA

logger = logging.getLogger(__name__)


class IndicatorCalculator:
    """
    Main engine for calculating indicators.

    Supports both daily and weekly timeframes.
    """

    # Indicator registry
    DAILY_INDICATORS = {
        'ema_5': DailyEMA,
        'ema_10': DailyEMA,
        'ema_20': DailyEMA,
        'sma_50': DailySMA,
        'sma_200': DailySMA,
        'rsi': RSI,
        'atr': ATR,
    }

    WEEKLY_INDICATORS = {
        'ema_10': WeeklyEMA,
        'sma_21': WeeklySMA,
        'sma_30': WeeklySMA,
        'sma_50': WeeklySMA,
    }

    def __init__(self, config: Dict):
        self.config = config
        self.quality_config = config['quality']
        self.daily_indicators = self._load_indicators('daily', self.DAILY_INDICATORS)
        self.weekly_indicators = self._load_indicators('weekly', self.WEEKLY_INDICATORS)

    def _load_indicators(self, timeframe: str, indicator_map: Dict) -> List[BaseIndicator]:
        """Load enabled indicators for a timeframe"""
        indicators = []

        if timeframe not in self.config:
            return indicators

        timeframe_config = self.config[timeframe]

        for indicator_name, indicator_config in timeframe_config.items():
            if not indicator_config.get('enabled', False):
                continue

            indicator_class = indicator_map.get(indicator_name)
            if not indicator_class:
                logger.warning(f"Unknown {timeframe} indicator: {indicator_name}")
                continue

            indicator = indicator_class(indicator_name, indicator_config)
            indicators.append(indicator)
            logger.info(f"Loaded {timeframe}: {indicator_name}")

        return indicators

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators (daily + weekly)"""
        # Validate minimum data
        min_bars = self.quality_config['min_bars_required']
        if len(df) < min_bars:
            raise ValueError(f"Insufficient data: {len(df)} < {min_bars} bars")

        # Calculate daily indicators
        for indicator in self.daily_indicators:
            if not indicator.validate_requirements(df):
                logger.warning(f"Skipping {indicator.name}: requirements not met")
                continue

            try:
                results = indicator.calculate(df)
                for col_name, values in results.items():
                    df[col_name] = values
            except Exception as e:
                logger.error(f"Error calculating {indicator.name}: {e}")
                raise

        # Calculate weekly indicators
        for indicator in self.weekly_indicators:
            if not indicator.validate_requirements(df):
                logger.warning(f"Skipping {indicator.name}: requirements not met")
                continue

            try:
                results = indicator.calculate(df)
                for col_name, values in results.items():
                    df[col_name] = values
            except Exception as e:
                logger.error(f"Error calculating {indicator.name}: {e}")
                raise

        return df

    def validate_complete(self, df: pd.DataFrame) -> bool:
        """Validate all indicators present (no NaN in latest row)"""
        if self.quality_config.get('reject_incomplete', True):
            latest = df.iloc[-1]

            # Get all expected columns
            expected_cols = []
            for indicator in self.daily_indicators + self.weekly_indicators:
                expected_cols.extend(indicator.get_column_names())

            # Check each column
            for col in expected_cols:
                if col not in df.columns:
                    logger.warning(f"Missing column: {col}")
                    return False

                if pd.isna(latest[col]):
                    logger.warning(f"NaN in latest row for: {col}")
                    return False

        return True

    def extract_latest(self, df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Extract latest indicator values.

        Returns:
            (daily_indicators, weekly_indicators) tuple of dicts
        """
        latest = df.iloc[-1]

        daily_values = {}
        for indicator in self.daily_indicators:
            for col in indicator.get_column_names():
                if col in df.columns:
                    daily_values[col] = float(latest[col])

        weekly_values = {}
        for indicator in self.weekly_indicators:
            for col in indicator.get_column_names():
                if col in df.columns:
                    weekly_values[col] = float(latest[col])

        return daily_values, weekly_values
