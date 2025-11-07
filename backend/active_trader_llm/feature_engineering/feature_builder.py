"""
Main interface for building features - replaces old indicators.py
"""

import pandas as pd
from typing import Dict
import logging

from .indicators.calculator import IndicatorCalculator
from .market_breadth import MarketBreadthAnalyzer
from .models import FeatureSet, MarketSnapshot

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Main class for calculating features.

    Replaces the old FeatureEngineer class.

    Usage:
        config = load_config('config.yaml')
        builder = FeatureBuilder(config)
        features = builder.build_features(price_df)
        snapshot = builder.build_market_snapshot(price_df, features)
    """

    def __init__(self, config: Dict):
        self.indicators_config = config['technical_indicators']
        self.breadth_config = config.get('market_breadth', {})

        # Initialize calculator
        self.calculator = IndicatorCalculator(self.indicators_config)

        # Initialize breadth analyzer if enabled
        if self.breadth_config.get('enabled', True):
            self.breadth_analyzer = MarketBreadthAnalyzer(self.breadth_config)
        else:
            self.breadth_analyzer = None

    def build_features(self, price_df: pd.DataFrame) -> Dict[str, FeatureSet]:
        """
        Calculate indicators for all symbols.

        Args:
            price_df: DataFrame with columns: symbol, timestamp, open, high, low, close, volume

        Returns:
            Dict mapping symbol -> FeatureSet (only symbols with complete indicators)
        """
        features = {}
        total_symbols = price_df['symbol'].nunique()
        excluded_count = 0

        for symbol in price_df['symbol'].unique():
            try:
                # Extract symbol data
                symbol_df = price_df[price_df['symbol'] == symbol].copy()
                symbol_df.sort_values('timestamp', inplace=True)

                # Calculate all indicators
                symbol_df = self.calculator.calculate_all(symbol_df)

                # Validate (no NaN tolerance)
                if not self.calculator.validate_complete(symbol_df):
                    logger.warning(f"Excluding {symbol}: incomplete indicators")
                    excluded_count += 1
                    continue

                # Extract latest row
                latest = symbol_df.iloc[-1]

                # Get indicator values
                daily_values, weekly_values = self.calculator.extract_latest(symbol_df)

                # Build FeatureSet
                features[symbol] = FeatureSet(
                    symbol=symbol,
                    timestamp=str(latest['timestamp']),
                    daily_indicators=daily_values,
                    weekly_indicators=weekly_values,
                    ohlcv={
                        'open': float(latest['open']),
                        'high': float(latest['high']),
                        'low': float(latest['low']),
                        'close': float(latest['close']),
                        'volume': int(latest['volume'])
                    }
                )

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                excluded_count += 1
                continue

        logger.info(
            f"Built features for {len(features)}/{total_symbols} symbols "
            f"({excluded_count} excluded)"
        )

        return features

    def build_market_snapshot(self, price_df: pd.DataFrame,
                              features: Dict[str, FeatureSet]) -> MarketSnapshot:
        """Calculate market breadth and regime"""
        if self.breadth_analyzer is None:
            logger.error("Cannot build market snapshot: Market breadth analyzer is disabled")
            logger.error("Enable market_breadth in config or the system cannot determine market regime")
            raise ValueError("Market breadth analysis is required but disabled in config")

        return self.breadth_analyzer.analyze(price_df, features)
