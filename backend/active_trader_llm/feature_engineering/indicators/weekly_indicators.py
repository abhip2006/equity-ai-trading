"""
Weekly timeframe indicators calculated from daily data.
"""

import pandas as pd
import tradingview_indicators as ta
from typing import Dict, List
from .base import BaseIndicator


class WeeklyEMA(BaseIndicator):
    """
    Exponential Moving Average - Weekly

    Calculated from daily data using converted period.
    Example: 10-week EMA = 50-day EMA (10 weeks * 5 trading days)
    """

    def calculate(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        period_weeks = self.config['period']
        period_days = self.config['conversion']  # Converted to daily bars

        ema_values = ta.ema(df['close'], period_days)
        return {f'EMA_{period_weeks}week': ema_values}

    def get_column_names(self) -> List[str]:
        return [f'EMA_{self.config["period"]}week']

    def validate_requirements(self, df: pd.DataFrame) -> bool:
        period_days = self.config['conversion']
        return 'close' in df.columns and len(df) >= period_days * 2


class WeeklySMA(BaseIndicator):
    """
    Simple Moving Average - Weekly

    Calculated from daily data using converted period.
    Example: 30-week SMA = 150-day SMA (30 weeks * 5 trading days)
    """

    def calculate(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        period_weeks = self.config['period']
        period_days = self.config['conversion']  # Converted to daily bars

        sma_values = ta.sma(df['close'], period_days)
        return {f'SMA_{period_weeks}week': sma_values}

    def get_column_names(self) -> List[str]:
        return [f'SMA_{self.config["period"]}week']

    def validate_requirements(self, df: pd.DataFrame) -> bool:
        period_days = self.config['conversion']
        return 'close' in df.columns and len(df) >= period_days * 2
