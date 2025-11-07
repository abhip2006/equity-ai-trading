"""
Base classes for indicator system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd


class BaseIndicator(ABC):
    """Abstract base class for all indicators"""

    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate indicator values.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            Dictionary mapping column names to Series
        """
        pass

    @abstractmethod
    def get_column_names(self) -> List[str]:
        """Return list of output column names"""
        pass

    @abstractmethod
    def validate_requirements(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame meets requirements"""
        pass
