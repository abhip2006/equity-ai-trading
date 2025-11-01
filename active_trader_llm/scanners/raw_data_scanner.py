"""
Raw Data Scanner: Fetch real technical indicators for Stage 2 deep analysis.

All indicators calculated by pandas_ta library (NO LLM calculations).
LLM will only receive pre-calculated values to interpret.

ANTI-HALLUCINATION DESIGN:
- pandas_ta calculates all technical indicators
- Alpaca/yfinance provides real price data
- LLM never sees raw OHLCV, only formatted indicator values
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    import pandas_ta as ta
except ImportError:
    logging.warning("pandas_ta not installed. Install with: pip install pandas-ta")
    ta = None

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TechnicalIndicators(BaseModel):
    """Pre-calculated technical indicators for a symbol"""
    symbol: str
    current_price: float

    # Trend indicators (calculated by pandas_ta)
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    ema_200: Optional[float] = None

    # Momentum indicators
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None

    # Volatility indicators
    atr_14: Optional[float] = None
    bbands_upper: Optional[float] = None
    bbands_middle: Optional[float] = None
    bbands_lower: Optional[float] = None
    bbands_width: Optional[float] = None

    # Volume indicators
    volume_sma_20: Optional[float] = None
    current_volume: Optional[float] = None
    volume_ratio: Optional[float] = None  # current / 20-day avg

    # Price positioning
    pct_from_sma20: Optional[float] = None
    pct_from_ema50: Optional[float] = None
    pct_from_ema200: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    pct_from_52w_high: Optional[float] = None

    # Price action
    price_change_1d_pct: Optional[float] = None
    price_change_5d_pct: Optional[float] = None
    price_change_20d_pct: Optional[float] = None


class RawDataScanner:
    """
    Fetches price data and calculates technical indicators using pandas_ta.

    NO LLM INVOLVED - pure calculation.
    """

    def __init__(self, data_fetcher=None):
        """
        Initialize raw data scanner.

        Args:
            data_fetcher: Object with fetch_prices method (e.g., PriceVolumeIngestor)
        """
        self.data_fetcher = data_fetcher

        if ta is None:
            raise ImportError("pandas_ta is required. Install with: pip install pandas-ta")

    def fetch_indicators(
        self,
        symbol: str,
        price_data: pd.DataFrame
    ) -> Optional[TechnicalIndicators]:
        """
        Calculate all technical indicators for a symbol using pandas_ta.

        Args:
            symbol: Stock symbol
            price_data: DataFrame with columns: timestamp, open, high, low, close, volume

        Returns:
            TechnicalIndicators object or None if insufficient data
        """
        try:
            if price_data.empty or len(price_data) < 200:
                logger.warning(f"{symbol}: Insufficient data ({len(price_data)} bars)")
                return None

            # Ensure data is sorted
            df = price_data.sort_values('timestamp').copy()

            # Current values
            current_price = float(df['close'].iloc[-1])
            current_volume = float(df['volume'].iloc[-1]) if 'volume' in df.columns else None

            # Calculate indicators using pandas_ta
            # Moving Averages
            df['SMA_20'] = ta.sma(df['close'], length=20)
            df['SMA_50'] = ta.sma(df['close'], length=50)
            df['EMA_20'] = ta.ema(df['close'], length=20)
            df['EMA_50'] = ta.ema(df['close'], length=50)
            df['EMA_200'] = ta.ema(df['close'], length=200)

            # RSI
            df['RSI_14'] = ta.rsi(df['close'], length=14)

            # MACD
            macd_result = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd_result is not None:
                df = pd.concat([df, macd_result], axis=1)

            # ATR
            atr_result = ta.atr(df['high'], df['low'], df['close'], length=14)
            if atr_result is not None:
                df['ATR_14'] = atr_result

            # Bollinger Bands
            bbands_result = ta.bbands(df['close'], length=20, std=2)
            if bbands_result is not None:
                df = pd.concat([df, bbands_result], axis=1)

            # Volume SMA
            if 'volume' in df.columns:
                df['Volume_SMA_20'] = ta.sma(df['volume'], length=20)

            # Extract latest values
            last_row = df.iloc[-1]

            sma_20 = last_row.get('SMA_20')
            sma_50 = last_row.get('SMA_50')
            ema_20 = last_row.get('EMA_20')
            ema_50 = last_row.get('EMA_50')
            ema_200 = last_row.get('EMA_200')

            rsi_14 = last_row.get('RSI_14')
            macd = last_row.get('MACD_12_26_9')
            macd_signal = last_row.get('MACDs_12_26_9')
            macd_hist = last_row.get('MACDh_12_26_9')

            atr_14 = last_row.get('ATR_14')
            bbands_upper = last_row.get('BBU_20_2.0')
            bbands_middle = last_row.get('BBM_20_2.0')
            bbands_lower = last_row.get('BBL_20_2.0')
            bbands_width = last_row.get('BBB_20_2.0')

            volume_sma_20 = last_row.get('Volume_SMA_20')

            # Calculate positioning percentages
            pct_from_sma20 = ((current_price - sma_20) / sma_20 * 100) if pd.notna(sma_20) else None
            pct_from_ema50 = ((current_price - ema_50) / ema_50 * 100) if pd.notna(ema_50) else None
            pct_from_ema200 = ((current_price - ema_200) / ema_200 * 100) if pd.notna(ema_200) else None

            # 52-week high/low
            high_52w = df['high'].iloc[-252:].max() if len(df) >= 252 else df['high'].max()
            low_52w = df['low'].iloc[-252:].min() if len(df) >= 252 else df['low'].min()
            pct_from_52w_high = ((current_price - high_52w) / high_52w * 100)

            # Volume ratio
            volume_ratio = (current_volume / volume_sma_20) if (current_volume and pd.notna(volume_sma_20) and volume_sma_20 > 0) else None

            # Price changes
            price_1d_ago = df['close'].iloc[-2] if len(df) >= 2 else current_price
            price_5d_ago = df['close'].iloc[-6] if len(df) >= 6 else current_price
            price_20d_ago = df['close'].iloc[-21] if len(df) >= 21 else current_price

            price_change_1d_pct = ((current_price - price_1d_ago) / price_1d_ago * 100)
            price_change_5d_pct = ((current_price - price_5d_ago) / price_5d_ago * 100)
            price_change_20d_pct = ((current_price - price_20d_ago) / price_20d_ago * 100)

            # Convert to float and handle NaN
            def to_float(val):
                if pd.isna(val):
                    return None
                return float(val)

            return TechnicalIndicators(
                symbol=symbol,
                current_price=current_price,
                sma_20=to_float(sma_20),
                sma_50=to_float(sma_50),
                ema_20=to_float(ema_20),
                ema_50=to_float(ema_50),
                ema_200=to_float(ema_200),
                rsi_14=to_float(rsi_14),
                macd=to_float(macd),
                macd_signal=to_float(macd_signal),
                macd_hist=to_float(macd_hist),
                atr_14=to_float(atr_14),
                bbands_upper=to_float(bbands_upper),
                bbands_middle=to_float(bbands_middle),
                bbands_lower=to_float(bbands_lower),
                bbands_width=to_float(bbands_width),
                volume_sma_20=to_float(volume_sma_20),
                current_volume=to_float(current_volume),
                volume_ratio=to_float(volume_ratio),
                pct_from_sma20=to_float(pct_from_sma20),
                pct_from_ema50=to_float(pct_from_ema50),
                pct_from_ema200=to_float(pct_from_ema200),
                high_52w=to_float(high_52w),
                low_52w=to_float(low_52w),
                pct_from_52w_high=to_float(pct_from_52w_high),
                price_change_1d_pct=to_float(price_change_1d_pct),
                price_change_5d_pct=to_float(price_change_5d_pct),
                price_change_20d_pct=to_float(price_change_20d_pct)
            )

        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return None

    def fetch_batch_indicators(
        self,
        symbols: List[str],
        price_data_map: Dict[str, pd.DataFrame]
    ) -> Dict[str, TechnicalIndicators]:
        """
        Calculate indicators for a batch of symbols.

        Args:
            symbols: List of symbols
            price_data_map: Dict mapping symbol -> price DataFrame

        Returns:
            Dict mapping symbol -> TechnicalIndicators
        """
        results = {}

        for symbol in symbols:
            if symbol not in price_data_map:
                logger.warning(f"{symbol}: No price data available")
                continue

            indicators = self.fetch_indicators(symbol, price_data_map[symbol])
            if indicators:
                results[symbol] = indicators

        logger.info(f"Calculated indicators for {len(results)}/{len(symbols)} symbols")
        return results

    def format_for_llm(self, indicators: TechnicalIndicators) -> str:
        """
        Format pre-calculated indicators as text for LLM interpretation.

        Args:
            indicators: TechnicalIndicators object

        Returns:
            Formatted string for LLM prompt
        """
        lines = [
            f"STOCK: {indicators.symbol}",
            f"Price: ${indicators.current_price:.2f}",
            "",
            "Moving Averages:",
            f"  - SMA20: ${indicators.sma_20:.2f} (price {indicators.pct_from_sma20:+.1f}% from SMA)" if indicators.sma_20 else "  - SMA20: N/A",
            f"  - EMA50: ${indicators.ema_50:.2f} (price {indicators.pct_from_ema50:+.1f}% from EMA)" if indicators.ema_50 else "  - EMA50: N/A",
            f"  - EMA200: ${indicators.ema_200:.2f} (price {indicators.pct_from_ema200:+.1f}% from EMA)" if indicators.ema_200 else "  - EMA200: N/A",
            "",
            "Momentum:",
            f"  - RSI(14): {indicators.rsi_14:.1f}" if indicators.rsi_14 else "  - RSI(14): N/A",
            f"  - MACD: {indicators.macd:.3f}, Signal: {indicators.macd_signal:.3f}, Hist: {indicators.macd_hist:.3f}" if indicators.macd else "  - MACD: N/A",
            f"  - 1d change: {indicators.price_change_1d_pct:+.2f}%" if indicators.price_change_1d_pct else "  - 1d change: N/A",
            f"  - 5d change: {indicators.price_change_5d_pct:+.2f}%" if indicators.price_change_5d_pct else "  - 5d change: N/A",
            "",
            "Volatility:",
            f"  - ATR(14): ${indicators.atr_14:.2f}" if indicators.atr_14 else "  - ATR(14): N/A",
            f"  - Bollinger: Upper=${indicators.bbands_upper:.2f}, Mid=${indicators.bbands_middle:.2f}, Lower=${indicators.bbands_lower:.2f}" if indicators.bbands_upper else "  - Bollinger: N/A",
            "",
            "Volume:",
            f"  - Current vs 20d avg: {indicators.volume_ratio:.2f}x" if indicators.volume_ratio else "  - Volume ratio: N/A",
            "",
            "Price Levels:",
            f"  - 52w High: ${indicators.high_52w:.2f} (current {indicators.pct_from_52w_high:+.1f}% from high)" if indicators.high_52w else "  - 52w High: N/A",
            f"  - 52w Low: ${indicators.low_52w:.2f}" if indicators.low_52w else "  - 52w Low: N/A"
        ]

        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create sample price data
    dates = pd.date_range(end=datetime.now(), periods=300, freq='D')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(300).cumsum() + 100,
        'high': np.random.randn(300).cumsum() + 102,
        'low': np.random.randn(300).cumsum() + 98,
        'close': np.random.randn(300).cumsum() + 100,
        'volume': np.random.randint(1e6, 10e6, 300)
    })

    scanner = RawDataScanner()

    # Calculate indicators
    indicators = scanner.fetch_indicators("AAPL", sample_data)

    if indicators:
        print("\nTechnical Indicators (calculated by pandas_ta):")
        print(f"  Current price: ${indicators.current_price:.2f}")
        print(f"  RSI(14): {indicators.rsi_14:.2f}")
        print(f"  MACD: {indicators.macd:.3f}")
        print(f"  ATR(14): ${indicators.atr_14:.2f}")
        print(f"  Volume ratio: {indicators.volume_ratio:.2f}x")
        print(f"  % from 52w high: {indicators.pct_from_52w_high:.2f}%")

        print("\n" + "="*50)
        print("Formatted for LLM:")
        print("="*50)
        print(scanner.format_for_llm(indicators))
