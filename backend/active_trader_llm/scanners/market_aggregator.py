"""
Market Aggregator: Calculate market-wide statistics for Stage 1.

All calculations are programmatic (NO LLM). Computes:
- Per-stock metrics: 5d price change, volume ratio, MA positioning, 52w high distance
- Sector-level aggregates: avg change, breadth, volume patterns
- Market summary for LLM interpretation
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class StockMetrics(BaseModel):
    """Calculated metrics for a single stock"""
    symbol: str
    sector: str
    price_change_5d_pct: Optional[float] = None
    volume_ratio: Optional[float] = None  # Current vs 20-day avg
    position_vs_ma50: Optional[str] = None  # "above", "below", "near"
    position_vs_ma200: Optional[str] = None
    distance_from_52w_high_pct: Optional[float] = None
    current_price: Optional[float] = None
    avg_volume_20d: Optional[float] = None
    daily_liquidity: Optional[float] = None  # price × avg_volume in USD
    adr_percent: Optional[float] = None  # Average Daily Range as % of close (20-day)


class SectorStats(BaseModel):
    """Aggregated statistics for a sector"""
    sector: str
    stock_count: int
    avg_price_change_5d_pct: float
    pct_above_ma50: float
    pct_above_ma200: float
    breadth_score: float  # -1 to +1
    avg_volume_ratio: float
    avg_distance_from_52w_high_pct: float


class MarketSummary(BaseModel):
    """Market-wide summary for Stage 1 LLM"""
    timestamp: str
    total_stocks: int
    sectors: List[SectorStats]
    market_breadth_score: float  # Overall market breadth
    avg_price_change_5d_pct: float
    pct_stocks_above_ma50: float
    pct_stocks_above_ma200: float
    high_volume_count: int  # Count of stocks with >2x avg volume
    near_52w_high_count: int  # Count within 5% of 52w high


class MarketAggregator:
    """
    Calculates market-wide statistics programmatically.

    NO LLM CALLS - pure calculation.
    """

    def __init__(self, data_fetcher=None):
        """
        Initialize market aggregator.

        Args:
            data_fetcher: Object with fetch_prices method (e.g., PriceVolumeIngestor)
        """
        self.data_fetcher = data_fetcher

    def calculate_stock_metrics(
        self,
        symbol: str,
        sector: str,
        price_data: pd.DataFrame
    ) -> Optional[StockMetrics]:
        """
        Calculate metrics for a single stock (NO LLM).

        Args:
            symbol: Stock symbol
            sector: Sector name
            price_data: DataFrame with OHLCV data (columns: timestamp, close, volume)

        Returns:
            StockMetrics object or None if insufficient data
        """
        try:
            if price_data.empty or len(price_data) < 20:
                return None

            # Sort by timestamp
            df = price_data.sort_values('timestamp')

            # Current price
            current_price = df['close'].iloc[-1]

            # 5-day price change
            if len(df) >= 5:
                price_5d_ago = df['close'].iloc[-6]  # -6 because -1 is current
                price_change_5d_pct = ((current_price - price_5d_ago) / price_5d_ago) * 100
            else:
                price_change_5d_pct = None

            # Volume ratio (current vs 20-day average)
            if len(df) >= 20 and 'volume' in df.columns:
                avg_volume_20d = df['volume'].iloc[-20:].mean()
                current_volume = df['volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else None
            else:
                avg_volume_20d = None
                volume_ratio = None

            # Moving averages
            if len(df) >= 50:
                ma50 = df['close'].iloc[-50:].mean()
                diff_from_ma50 = ((current_price - ma50) / ma50) * 100
                if diff_from_ma50 > 2:
                    position_vs_ma50 = "above"
                elif diff_from_ma50 < -2:
                    position_vs_ma50 = "below"
                else:
                    position_vs_ma50 = "near"
            else:
                position_vs_ma50 = None

            if len(df) >= 200:
                ma200 = df['close'].iloc[-200:].mean()
                diff_from_ma200 = ((current_price - ma200) / ma200) * 100
                if diff_from_ma200 > 2:
                    position_vs_ma200 = "above"
                elif diff_from_ma200 < -2:
                    position_vs_ma200 = "below"
                else:
                    position_vs_ma200 = "near"
            else:
                position_vs_ma200 = None

            # Distance from 52-week high
            high_52w = df['high'].iloc[-252:].max() if len(df) >= 252 else df['high'].max()
            distance_from_52w_high_pct = ((current_price - high_52w) / high_52w) * 100

            # Daily liquidity (price × avg_volume in USD)
            daily_liquidity = None
            if avg_volume_20d is not None and current_price is not None:
                daily_liquidity = current_price * avg_volume_20d

            # Average Daily Range (ADR) as % of close (20-day average)
            adr_percent = None
            if len(df) >= 20 and 'high' in df.columns and 'low' in df.columns:
                # Calculate daily range for last 20 days
                daily_ranges = ((df['high'].iloc[-20:] - df['low'].iloc[-20:]) / df['close'].iloc[-20:]) * 100
                adr_percent = daily_ranges.mean()

            return StockMetrics(
                symbol=symbol,
                sector=sector,
                price_change_5d_pct=price_change_5d_pct,
                volume_ratio=volume_ratio,
                position_vs_ma50=position_vs_ma50,
                position_vs_ma200=position_vs_ma200,
                distance_from_52w_high_pct=distance_from_52w_high_pct,
                current_price=current_price,
                avg_volume_20d=avg_volume_20d,
                daily_liquidity=daily_liquidity,
                adr_percent=adr_percent
            )

        except Exception as e:
            logger.warning(f"Error calculating metrics for {symbol}: {e}")
            return None

    def aggregate_by_sector(
        self,
        stock_metrics: List[StockMetrics]
    ) -> List[SectorStats]:
        """
        Aggregate stock metrics by sector (NO LLM).

        Args:
            stock_metrics: List of StockMetrics objects

        Returns:
            List of SectorStats objects
        """
        sectors = {}

        for stock in stock_metrics:
            sector = stock.sector or "Unknown"
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(stock)

        sector_stats = []

        for sector, stocks in sectors.items():
            # Filter out None values for each metric
            price_changes = [s.price_change_5d_pct for s in stocks if s.price_change_5d_pct is not None]
            volume_ratios = [s.volume_ratio for s in stocks if s.volume_ratio is not None]
            distances_52w = [s.distance_from_52w_high_pct for s in stocks if s.distance_from_52w_high_pct is not None]

            above_ma50 = len([s for s in stocks if s.position_vs_ma50 == "above"])
            above_ma200 = len([s for s in stocks if s.position_vs_ma200 == "above"])
            total_with_ma50 = len([s for s in stocks if s.position_vs_ma50 is not None])
            total_with_ma200 = len([s for s in stocks if s.position_vs_ma200 is not None])

            # Calculate breadth score (-1 to +1)
            # Positive if more stocks above MA200 than below
            if total_with_ma200 > 0:
                breadth_score = (above_ma200 / total_with_ma200) * 2 - 1
            else:
                breadth_score = 0.0

            sector_stats.append(SectorStats(
                sector=sector,
                stock_count=len(stocks),
                avg_price_change_5d_pct=np.mean(price_changes) if price_changes else 0.0,
                pct_above_ma50=(above_ma50 / total_with_ma50 * 100) if total_with_ma50 > 0 else 0.0,
                pct_above_ma200=(above_ma200 / total_with_ma200 * 100) if total_with_ma200 > 0 else 0.0,
                breadth_score=breadth_score,
                avg_volume_ratio=np.mean(volume_ratios) if volume_ratios else 1.0,
                avg_distance_from_52w_high_pct=np.mean(distances_52w) if distances_52w else 0.0
            ))

        # Sort by stock count (largest sectors first)
        sector_stats.sort(key=lambda x: x.stock_count, reverse=True)

        return sector_stats

    def generate_market_summary(
        self,
        stock_metrics: List[StockMetrics],
        sector_stats: List[SectorStats]
    ) -> MarketSummary:
        """
        Generate market-wide summary for LLM (NO LLM in this function).

        Args:
            stock_metrics: List of StockMetrics
            sector_stats: List of SectorStats

        Returns:
            MarketSummary object
        """
        # Market-wide aggregates
        price_changes = [s.price_change_5d_pct for s in stock_metrics if s.price_change_5d_pct is not None]
        volume_ratios = [s.volume_ratio for s in stock_metrics if s.volume_ratio is not None]
        distances_52w = [s.distance_from_52w_high_pct for s in stock_metrics if s.distance_from_52w_high_pct is not None]

        above_ma50_count = len([s for s in stock_metrics if s.position_vs_ma50 == "above"])
        above_ma200_count = len([s for s in stock_metrics if s.position_vs_ma200 == "above"])
        total_with_ma50 = len([s for s in stock_metrics if s.position_vs_ma50 is not None])
        total_with_ma200 = len([s for s in stock_metrics if s.position_vs_ma200 is not None])

        # High volume stocks (>2x average)
        high_volume_count = len([s for s in stock_metrics if s.volume_ratio and s.volume_ratio > 2.0])

        # Near 52-week high (within 5%)
        near_52w_high_count = len([s for s in stock_metrics if s.distance_from_52w_high_pct and s.distance_from_52w_high_pct > -5.0])

        # Overall breadth score
        if total_with_ma200 > 0:
            market_breadth_score = (above_ma200_count / total_with_ma200) * 2 - 1
        else:
            market_breadth_score = 0.0

        return MarketSummary(
            timestamp=datetime.now().isoformat(),
            total_stocks=len(stock_metrics),
            sectors=sector_stats,
            market_breadth_score=market_breadth_score,
            avg_price_change_5d_pct=np.mean(price_changes) if price_changes else 0.0,
            pct_stocks_above_ma50=(above_ma50_count / total_with_ma50 * 100) if total_with_ma50 > 0 else 0.0,
            pct_stocks_above_ma200=(above_ma200_count / total_with_ma200 * 100) if total_with_ma200 > 0 else 0.0,
            high_volume_count=high_volume_count,
            near_52w_high_count=near_52w_high_count
        )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create sample price data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'close': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'volume': np.random.randint(1e6, 10e6, 100)
    })

    aggregator = MarketAggregator()

    # Calculate metrics for sample stock
    metrics = aggregator.calculate_stock_metrics("AAPL", "Technology", sample_data)
    if metrics:
        print(f"\nSample stock metrics:")
        print(f"  5d change: {metrics.price_change_5d_pct:.2f}%")
        print(f"  Volume ratio: {metrics.volume_ratio:.2f}x")
        print(f"  Position vs MA50: {metrics.position_vs_ma50}")
        print(f"  Distance from 52w high: {metrics.distance_from_52w_high_pct:.2f}%")

    # Create sample list for aggregation
    sample_metrics = [metrics] if metrics else []

    if sample_metrics:
        # Aggregate by sector
        sector_stats = aggregator.aggregate_by_sector(sample_metrics)
        print(f"\nSector stats:")
        for stats in sector_stats:
            print(f"  {stats.sector}: {stats.stock_count} stocks, breadth={stats.breadth_score:.2f}")

        # Generate market summary
        summary = aggregator.generate_market_summary(sample_metrics, sector_stats)
        print(f"\nMarket summary:")
        print(f"  Total stocks: {summary.total_stocks}")
        print(f"  Market breadth: {summary.market_breadth_score:.2f}")
        print(f"  Avg 5d change: {summary.avg_price_change_5d_pct:.2f}%")
