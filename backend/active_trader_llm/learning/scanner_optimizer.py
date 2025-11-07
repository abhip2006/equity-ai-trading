"""
Scanner Optimizer: Learn optimal filtering thresholds from historical scanner performance.

Analyzes which Stage 1 filtering criteria led to profitable trades and
suggests adjustments to improve future scans.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ThresholdRecommendation(BaseModel):
    """Recommended threshold adjustment"""
    metric: str
    current_value: float
    recommended_value: float
    confidence: float
    reasoning: str


class ScannerOptimizerReport(BaseModel):
    """Scanner optimization analysis report"""
    timestamp: str
    scan_performance_30d: Dict
    threshold_recommendations: List[ThresholdRecommendation]
    overall_assessment: str


class ScannerOptimizer:
    """
    Analyzes scanner performance and suggests threshold optimizations.

    Uses historical trade outcomes to determine which filtering criteria
    are most effective at identifying profitable opportunities.
    """

    def __init__(self, memory_manager):
        """
        Initialize scanner optimizer.

        Args:
            memory_manager: MemoryManager instance for accessing historical data
        """
        self.memory = memory_manager

    def analyze_scan_performance(self, days: int = 30) -> ScannerOptimizerReport:
        """
        Analyze scanner performance over time period.

        Args:
            days: Number of days to analyze

        Returns:
            ScannerOptimizerReport with recommendations
        """
        logger.info(f"Analyzing scanner performance over last {days} days...")

        # Get scan statistics from memory
        scan_stats = self.memory.get_scan_statistics(days=days)

        # Analyze performance patterns
        recommendations = self._generate_recommendations(scan_stats)

        # Generate overall assessment
        overall = self._generate_assessment(scan_stats, recommendations)

        return ScannerOptimizerReport(
            timestamp=datetime.now().isoformat(),
            scan_performance_30d=scan_stats,
            threshold_recommendations=recommendations,
            overall_assessment=overall
        )

    def _generate_recommendations(self, scan_stats: Dict) -> List[ThresholdRecommendation]:
        """
        Generate threshold adjustment recommendations based on performance.

        Args:
            scan_stats: Scanner performance statistics

        Returns:
            List of ThresholdRecommendation objects
        """
        recommendations = []

        win_rate = scan_stats.get('win_rate', 0.0)
        total_trades = scan_stats.get('total_scanned_trades', 0)

        # Recommendation 1: Volume ratio threshold
        if total_trades >= 10:
            if win_rate < 40.0:
                # Low win rate: Be more selective, increase volume threshold
                recommendations.append(ThresholdRecommendation(
                    metric="volume_ratio_threshold",
                    current_value=2.5,
                    recommended_value=3.0,
                    confidence=0.7,
                    reasoning=f"Win rate ({win_rate:.1f}%) below target. Increasing volume threshold to filter for stronger conviction."
                ))
            elif win_rate > 60.0:
                # High win rate: Can be less selective, decrease volume threshold
                recommendations.append(ThresholdRecommendation(
                    metric="volume_ratio_threshold",
                    current_value=2.5,
                    recommended_value=2.0,
                    confidence=0.6,
                    reasoning=f"Win rate ({win_rate:.1f}%) above target. Decreasing volume threshold to capture more opportunities."
                ))

        # Recommendation 2: 52-week high threshold
        if total_trades >= 10:
            if win_rate < 40.0:
                # Low win rate: Look for stocks closer to highs
                recommendations.append(ThresholdRecommendation(
                    metric="distance_from_52w_high_threshold_pct",
                    current_value=7.0,
                    recommended_value=5.0,
                    confidence=0.6,
                    reasoning="Tightening 52-week high threshold to focus on stronger momentum stocks."
                ))

        # Recommendation 3: Minimum price change
        avg_pnl = scan_stats.get('avg_pnl', 0.0)
        if total_trades >= 10:
            if avg_pnl < 0:
                # Negative average P/L: Require stronger momentum
                recommendations.append(ThresholdRecommendation(
                    metric="min_price_change_5d_pct",
                    current_value=3.0,
                    recommended_value=4.0,
                    confidence=0.7,
                    reasoning=f"Average P/L (${avg_pnl:.2f}) is negative. Requiring stronger 5-day momentum."
                ))

        # If no specific recommendations, suggest maintaining current thresholds
        if not recommendations and total_trades >= 10:
            recommendations.append(ThresholdRecommendation(
                metric="all_thresholds",
                current_value=0.0,
                recommended_value=0.0,
                confidence=0.8,
                reasoning=f"Current thresholds performing well (win rate: {win_rate:.1f}%). Maintain current settings."
            ))

        return recommendations

    def _generate_assessment(self, scan_stats: Dict, recommendations: List[ThresholdRecommendation]) -> str:
        """
        Generate overall scanner performance assessment.

        Args:
            scan_stats: Scanner statistics
            recommendations: List of recommendations

        Returns:
            Assessment string
        """
        total_trades = scan_stats.get('total_scanned_trades', 0)
        win_rate = scan_stats.get('win_rate', 0.0)
        total_pnl = scan_stats.get('total_pnl', 0.0)

        if total_trades == 0:
            return "Insufficient data: No scanner-generated trades executed yet."

        if total_trades < 10:
            return f"Limited data: Only {total_trades} scanner trades. Need more data for reliable optimization."

        # Assess performance
        if win_rate >= 60.0 and total_pnl > 0:
            performance = "EXCELLENT"
        elif win_rate >= 50.0 and total_pnl > 0:
            performance = "GOOD"
        elif win_rate >= 40.0:
            performance = "FAIR"
        else:
            performance = "POOR"

        assessment = f"Scanner Performance: {performance}\n"
        assessment += f"  - {total_trades} trades executed from scanner\n"
        assessment += f"  - {win_rate:.1f}% win rate\n"
        assessment += f"  - ${total_pnl:.2f} total P/L\n"

        if recommendations:
            assessment += f"\nRecommendations: {len(recommendations)} threshold adjustments suggested"

        return assessment

    def apply_recommendations(self, recommendations: List[ThresholdRecommendation], config_dict: Dict) -> Dict:
        """
        Apply threshold recommendations to configuration.

        Args:
            recommendations: List of recommendations
            config_dict: Current configuration dictionary

        Returns:
            Updated configuration dictionary
        """
        updated_config = config_dict.copy()

        for rec in recommendations:
            if rec.metric == "all_thresholds":
                # No changes needed
                continue

            # Navigate to scanner.stage1 filtering criteria
            if 'scanner' not in updated_config:
                logger.warning("Scanner config not found")
                continue

            # This would be implemented based on your config structure
            # For now, log the recommendation
            logger.info(f"Recommendation: {rec.metric} = {rec.recommended_value} (was {rec.current_value})")
            logger.info(f"  Reasoning: {rec.reasoning}")

        return updated_config

    def should_trigger_optimization(self, days: int = 30) -> bool:
        """
        Check if optimization should be triggered.

        Args:
            days: Days to analyze

        Returns:
            True if optimization should run, False otherwise
        """
        scan_stats = self.memory.get_scan_statistics(days=days)
        total_trades = scan_stats.get('total_scanned_trades', 0)

        # Trigger if we have at least 20 trades
        # This ensures enough data for meaningful optimization
        return total_trades >= 20


# Example usage
if __name__ == "__main__":
    from memory.memory_manager import MemoryManager

    logging.basicConfig(level=logging.INFO)

    # Initialize
    memory = MemoryManager()
    optimizer = ScannerOptimizer(memory)

    # Check if optimization should run
    if optimizer.should_trigger_optimization(days=30):
        # Run analysis
        report = optimizer.analyze_scan_performance(days=30)

        print("\n" + "="*60)
        print("SCANNER OPTIMIZATION REPORT")
        print("="*60)
        print(f"\nTimestamp: {report.timestamp}")
        print(f"\nPerformance (30 days):")
        for key, value in report.scan_performance_30d.items():
            print(f"  {key}: {value}")

        print(f"\nThreshold Recommendations:")
        for rec in report.threshold_recommendations:
            print(f"\n  {rec.metric}:")
            print(f"    Current: {rec.current_value}")
            print(f"    Recommended: {rec.recommended_value}")
            print(f"    Confidence: {rec.confidence:.1%}")
            print(f"    Reasoning: {rec.reasoning}")

        print(f"\n{report.overall_assessment}")
        print("="*60)
    else:
        print("Not enough data for optimization (need 20+ scanner trades)")
