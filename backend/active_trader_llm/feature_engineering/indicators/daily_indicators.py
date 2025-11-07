"""
Daily timeframe indicators using tradingview-indicators.
"""

import pandas as pd
import numpy as np
import tradingview_indicators as ta
from typing import Dict, List
from .base import BaseIndicator


class DailyEMA(BaseIndicator):
    """Exponential Moving Average - Daily"""

    def calculate(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        period = self.config['period']
        ema_values = ta.ema(df['close'], period)
        return {f'EMA_{period}_daily': ema_values}

    def get_column_names(self) -> List[str]:
        return [f'EMA_{self.config["period"]}_daily']

    def validate_requirements(self, df: pd.DataFrame) -> bool:
        period = self.config['period']
        return 'close' in df.columns and len(df) >= period * 2


class DailySMA(BaseIndicator):
    """Simple Moving Average - Daily"""

    def calculate(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        period = self.config['period']
        sma_values = ta.sma(df['close'], period)
        return {f'SMA_{period}_daily': sma_values}

    def get_column_names(self) -> List[str]:
        return [f'SMA_{self.config["period"]}_daily']

    def validate_requirements(self, df: pd.DataFrame) -> bool:
        period = self.config['period']
        return 'close' in df.columns and len(df) >= period * 2


class RSI(BaseIndicator):
    """Relative Strength Index - Daily"""

    def calculate(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        period = self.config['period']
        rsi_values = ta.RSI(df['close'], period)
        return {f'RSI_{period}_daily': rsi_values}

    def get_column_names(self) -> List[str]:
        return [f'RSI_{self.config["period"]}_daily']

    def validate_requirements(self, df: pd.DataFrame) -> bool:
        period = self.config['period']
        return 'close' in df.columns and len(df) >= period * 2


class ATR(BaseIndicator):
    """Average True Range - Daily (manual implementation)"""

    def calculate(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        period = self.config['period']

        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift(1))
        low_close_prev = np.abs(df['low'] - df['close'].shift(1))

        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

        # Calculate ATR as exponential moving average of true range
        atr_values = true_range.ewm(span=period, adjust=False).mean()

        return {f'ATR_{period}_daily': atr_values}

    def get_column_names(self) -> List[str]:
        return [f'ATR_{self.config["period"]}_daily']

    def validate_requirements(self, df: pd.DataFrame) -> bool:
        required = ['high', 'low', 'close']
        period = self.config['period']
        return all(col in df.columns for col in required) and len(df) >= period * 2
