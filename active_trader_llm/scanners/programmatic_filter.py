"""
Programmatic Filter: Apply Stage 1 guidance to filter stocks (NO LLM).

Takes Stage 1 guidance and stock metrics, applies filtering criteria
to reduce 5000+ stocks to 50-200 candidates for Stage 2 deep analysis.
"""

import logging
from typing import List
from pydantic import BaseModel

from .stage1_analyzer import Stage1Guidance
from .market_aggregator import StockMetrics

logger = logging.getLogger(__name__)


class FilterResult(BaseModel):
    """Result of programmatic filtering"""
    candidates: List[str]  # Symbols that passed filters
    filtered_count: int
    initial_count: int
    filter_stats: dict


class ProgrammaticFilter:
    """
    Applies Stage 1 guidance to filter stocks programmatically.

    NO LLM CALLS - pure boolean logic based on guidance thresholds.
    """

    def __init__(self):
        """Initialize programmatic filter"""
        pass

    def apply_filters(
        self,
        stock_metrics: List[StockMetrics],
        guidance: Stage1Guidance,
        max_candidates: int = 150
    ) -> FilterResult:
        """
        Apply filtering criteria from Stage 1 guidance.

        Args:
            stock_metrics: List of StockMetrics from market_aggregator
            guidance: Stage1Guidance from stage1_analyzer
            max_candidates: Maximum candidates to return

        Returns:
            FilterResult with candidate symbols
        """
        initial_count = len(stock_metrics)
        logger.info(f"Applying filters to {initial_count} stocks...")

        # Filter stats for tracking
        filter_stats = {
            'passed_volume': 0,
            'passed_52w_high': 0,
            'passed_momentum': 0,
            'passed_sector': 0,
            'passed_liquidity': 0,
            'passed_adr': 0,
            'passed_all': 0
        }

        candidates = []

        for stock in stock_metrics:
            # Filter 1: Volume ratio
            if stock.volume_ratio is None or stock.volume_ratio < guidance.filtering_criteria.volume_ratio_threshold:
                continue
            filter_stats['passed_volume'] += 1

            # Filter 2: Distance from 52-week high
            if stock.distance_from_52w_high_pct is None:
                continue
            # Guidance threshold is max distance (e.g., 7% = within 7% of high)
            # Stock distance is negative (e.g., -8% = 8% below high)
            if stock.distance_from_52w_high_pct < -guidance.filtering_criteria.distance_from_52w_high_threshold_pct:
                continue
            filter_stats['passed_52w_high'] += 1

            # Filter 3: Minimum price change
            if stock.price_change_5d_pct is None or stock.price_change_5d_pct < guidance.filtering_criteria.min_price_change_5d_pct:
                continue
            filter_stats['passed_momentum'] += 1

            # Filter 4: Focus sectors
            if guidance.focus_sectors and stock.sector not in guidance.focus_sectors:
                # If focus sectors specified, only include those
                continue
            filter_stats['passed_sector'] += 1

            # Filter 5: Daily liquidity (> $500M)
            if stock.daily_liquidity is None or stock.daily_liquidity < 500_000_000:
                continue
            filter_stats['passed_liquidity'] += 1

            # Filter 6: ADR range (1% to 15%)
            if stock.adr_percent is None or stock.adr_percent < 1.0 or stock.adr_percent > 15.0:
                continue
            filter_stats['passed_adr'] += 1

            # Passed all filters
            filter_stats['passed_all'] += 1
            candidates.append(stock.symbol)

        # Limit to max candidates (if we have too many, take top by momentum)
        if len(candidates) > max_candidates:
            logger.info(f"Limiting from {len(candidates)} to {max_candidates} candidates")

            # Create dict of symbol -> momentum for sorting
            momentum_map = {
                stock.symbol: stock.price_change_5d_pct or 0.0
                for stock in stock_metrics
                if stock.symbol in candidates
            }

            # Sort by momentum and take top N
            candidates = sorted(
                candidates,
                key=lambda sym: momentum_map.get(sym, 0.0),
                reverse=True
            )[:max_candidates]

        logger.info(f"Filtered {initial_count} → {len(candidates)} candidates")
        logger.info(f"Filter pass rates: volume={filter_stats['passed_volume']}, "
                   f"52w_high={filter_stats['passed_52w_high']}, "
                   f"momentum={filter_stats['passed_momentum']}, "
                   f"sector={filter_stats['passed_sector']}, "
                   f"liquidity={filter_stats['passed_liquidity']}, "
                   f"adr={filter_stats['passed_adr']}, "
                   f"all={filter_stats['passed_all']}")

        return FilterResult(
            candidates=candidates,
            filtered_count=len(candidates),
            initial_count=initial_count,
            filter_stats=filter_stats
        )

    def apply_additional_filters(
        self,
        stock_metrics: List[StockMetrics],
        min_price: float = 5.0,
        max_price: float = 1000.0,
        ma_filter: str = "above_ma50"  # "above_ma50", "above_ma200", "any"
    ) -> List[str]:
        """
        Apply additional optional filters.

        Args:
            stock_metrics: List of StockMetrics
            min_price: Minimum stock price
            max_price: Maximum stock price
            ma_filter: Moving average filter ("above_ma50", "above_ma200", "any")

        Returns:
            List of symbols that pass filters
        """
        candidates = []

        for stock in stock_metrics:
            # Price range filter
            if stock.current_price is None:
                continue
            if stock.current_price < min_price or stock.current_price > max_price:
                continue

            # MA filter
            if ma_filter == "above_ma50":
                if stock.position_vs_ma50 != "above":
                    continue
            elif ma_filter == "above_ma200":
                if stock.position_vs_ma200 != "above":
                    continue

            candidates.append(stock.symbol)

        return candidates

    def rank_candidates(
        self,
        stock_metrics: List[StockMetrics],
        candidates: List[str],
        ranking_method: str = "momentum"  # "momentum", "volume", "composite"
    ) -> List[str]:
        """
        Rank candidates by specified method.

        Args:
            stock_metrics: List of StockMetrics
            candidates: List of candidate symbols
            ranking_method: How to rank ("momentum", "volume", "composite")

        Returns:
            Ranked list of symbols (best first)
        """
        # Create metric map
        metrics_map = {stock.symbol: stock for stock in stock_metrics if stock.symbol in candidates}

        if ranking_method == "momentum":
            # Rank by 5-day price change
            ranked = sorted(
                candidates,
                key=lambda sym: metrics_map[sym].price_change_5d_pct or 0.0,
                reverse=True
            )

        elif ranking_method == "volume":
            # Rank by volume ratio
            ranked = sorted(
                candidates,
                key=lambda sym: metrics_map[sym].volume_ratio or 1.0,
                reverse=True
            )

        elif ranking_method == "composite":
            # Composite score: (momentum * 0.6) + (volume_ratio * 0.4)
            def composite_score(sym):
                stock = metrics_map[sym]
                momentum = stock.price_change_5d_pct or 0.0
                volume = stock.volume_ratio or 1.0
                return (momentum * 0.6) + (volume * 0.4)

            ranked = sorted(candidates, key=composite_score, reverse=True)

        else:
            logger.warning(f"Unknown ranking method: {ranking_method}. Returning unsorted.")
            ranked = candidates

        return ranked


# Example usage
if __name__ == "__main__":
    from .stage1_analyzer import FilteringCriteria, Stage1Guidance

    logging.basicConfig(level=logging.INFO)

    # Create sample guidance
    sample_guidance = Stage1Guidance(
        market_bias="bullish",
        focus_sectors=["Technology", "Healthcare"],
        focus_patterns=["breakouts", "momentum"],
        filtering_criteria=FilteringCriteria(
            volume_ratio_threshold=2.5,
            distance_from_52w_high_threshold_pct=7.0,
            min_price_change_5d_pct=3.0
        ),
        target_count=100,
        reasoning="Strong bullish market favoring momentum plays in tech"
    )

    # Create sample stock metrics
    sample_metrics = [
        StockMetrics(
            symbol="AAPL",
            sector="Technology",
            price_change_5d_pct=4.5,
            volume_ratio=3.2,
            position_vs_ma50="above",
            position_vs_ma200="above",
            distance_from_52w_high_pct=-3.5,
            current_price=175.0,
            avg_volume_20d=50e6
        ),
        StockMetrics(
            symbol="XYZ",
            sector="Energy",
            price_change_5d_pct=2.0,
            volume_ratio=1.8,
            position_vs_ma50="below",
            position_vs_ma200="below",
            distance_from_52w_high_pct=-15.0,
            current_price=45.0,
            avg_volume_20d=1e6
        ),
        StockMetrics(
            symbol="MSFT",
            sector="Technology",
            price_change_5d_pct=3.5,
            volume_ratio=2.8,
            position_vs_ma50="above",
            position_vs_ma200="above",
            distance_from_52w_high_pct=-5.0,
            current_price=350.0,
            avg_volume_20d=25e6
        )
    ]

    # Apply filters
    filter_engine = ProgrammaticFilter()
    result = filter_engine.apply_filters(sample_metrics, sample_guidance)

    print(f"\nFilter Results:")
    print(f"  Initial: {result.initial_count}")
    print(f"  Filtered: {result.filtered_count}")
    print(f"  Candidates: {result.candidates}")
    print(f"  Filter stats: {result.filter_stats}")

    # Rank candidates
    ranked = filter_engine.rank_candidates(sample_metrics, result.candidates, "momentum")
    print(f"\nRanked by momentum: {ranked}")
