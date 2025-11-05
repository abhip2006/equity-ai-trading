"""
Stage 1 Analyzer: LLM interprets market-wide statistics and returns filtering guidance.

This is the ONLY LLM call in Stage 1 (1 call total).
The LLM receives pre-calculated market data and decides filtering thresholds.
"""

import json
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from active_trader_llm.llm import get_llm_client, LLMMessage

from .market_aggregator import MarketSummary

logger = logging.getLogger(__name__)


class FilteringCriteria(BaseModel):
    """Programmatic filtering thresholds (decided by LLM)"""
    volume_ratio_threshold: float = Field(..., description="Min volume ratio (e.g., 2.5 = 2.5x average)")
    distance_from_52w_high_threshold_pct: float = Field(..., description="Max distance from 52w high % (e.g., 7 = within 7%)")
    min_price_change_5d_pct: float = Field(..., description="Min 5-day price change % (e.g., 3 = +3% minimum)")


class Stage1Guidance(BaseModel):
    """Stage 1 LLM output: Market interpretation + filtering guidance"""
    market_bias: str = Field(..., description="bullish, bearish, or neutral")
    focus_sectors: List[str] = Field(..., description="Sectors to focus on (e.g., ['Technology', 'Healthcare'])")
    focus_patterns: List[str] = Field(..., description="Patterns to look for (e.g., ['breakouts', 'momentum', 'mean_reversion'])")
    filtering_criteria: FilteringCriteria
    target_count: int = Field(..., description="Target number of candidates for Stage 2")
    reasoning: str = Field(..., description="Brief explanation of market conditions and filtering choices")


class Stage1Analyzer:
    """
    LLM-powered Stage 1 market interpreter.

    Receives pre-calculated market statistics and returns filtering guidance.
    NO calculation happens in LLM - only interpretation.
    """

    SYSTEM_PROMPT = """You are a Market Strategist analyzing market-wide conditions to guide stock screening.

Your task is to interpret pre-calculated market statistics and decide:
1. Overall market bias (bullish/bearish/neutral)
2. Which sectors show strength
3. What patterns to look for (breakouts, momentum, mean reversion, etc.)
4. Filtering thresholds to identify candidates

CRITICAL: Return ONLY valid JSON matching this exact schema, NO prose:
{
    "market_bias": "bullish|bearish|neutral",
    "focus_sectors": ["sector1", "sector2", ...],
    "focus_patterns": ["breakouts", "momentum", "mean_reversion", "pullback"],
    "filtering_criteria": {
        "volume_ratio_threshold": <number>,
        "distance_from_52w_high_threshold_pct": <number>,
        "min_price_change_5d_pct": <number>
    },
    "target_count": <number>,
    "reasoning": "Brief explanation of market conditions and why these filters make sense"
}

You are NOT calculating anything. You are interpreting already-calculated data.
Determine appropriate thresholds based on current market conditions - do NOT use fixed rules."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        provider: str = "openai",
        temperature: float = 0.3
    ):
        """
        Initialize Stage 1 Analyzer.

        Args:
            api_key: API key for the LLM provider
            model: LLM model to use (gpt-4o, gpt-4-turbo, gpt-3.5-turbo, claude-3-5-sonnet-20241022)
            provider: LLM provider (openai, anthropic, local)
            temperature: Temperature for LLM (0.3 for consistency)
        """
        self.client = get_llm_client(
            provider=provider,
            model=model,
            api_key=api_key
        )
        self.model = model
        self.provider = provider
        self.temperature = temperature

    def _build_market_prompt(self, market_summary: MarketSummary) -> str:
        """
        Build prompt with pre-calculated market statistics.

        Args:
            market_summary: MarketSummary object from market_aggregator

        Returns:
            Formatted prompt string
        """
        prompt = f"""Analyze these market-wide statistics and provide filtering guidance.

MARKET OVERVIEW (pre-calculated):
- Total stocks analyzed: {market_summary.total_stocks}
- Market breadth score: {market_summary.market_breadth_score:.2f}
- Average 5d price change: {market_summary.avg_price_change_5d_pct:.2f}%
- Stocks above MA50: {market_summary.pct_stocks_above_ma50:.1f}%
- Stocks above MA200: {market_summary.pct_stocks_above_ma200:.1f}%
- High volume stocks: {market_summary.high_volume_count}
- Near 52-week high: {market_summary.near_52w_high_count}

SECTOR BREAKDOWN (top sectors by stock count):
"""

        # Include top 11 sectors (or fewer if not available)
        for sector in market_summary.sectors[:11]:
            prompt += f"""
{sector.sector} ({sector.stock_count} stocks):
  - Avg 5d change: {sector.avg_price_change_5d_pct:.2f}%
  - Above MA50: {sector.pct_above_ma50:.1f}%
  - Above MA200: {sector.pct_above_ma200:.1f}%
  - Breadth score: {sector.breadth_score:.2f}
  - Avg volume ratio: {sector.avg_volume_ratio:.2f}x
  - Avg distance from 52w high: {sector.avg_distance_from_52w_high_pct:.2f}%
"""

        prompt += "\nBased on these PRE-CALCULATED statistics, provide your filtering guidance as JSON:"

        return prompt

    def analyze(self, market_summary: MarketSummary) -> Optional[Stage1Guidance]:
        """
        Analyze market summary and return filtering guidance (1 LLM call).

        Args:
            market_summary: MarketSummary from market_aggregator

        Returns:
            Stage1Guidance object or None on error
        """
        prompt = self._build_market_prompt(market_summary)

        try:
            logger.info("Calling LLM for Stage 1 market analysis...")

            # Use unified LLM client
            messages = [
                LLMMessage(role="system", content=self.SYSTEM_PROMPT),
                LLMMessage(role="user", content=prompt)
            ]

            response = self.client.generate(
                messages=messages,
                temperature=self.temperature,
                max_tokens=2000
            )

            # Extract JSON from response
            content = response.content

            # Try to parse JSON
            try:
                guidance_dict = json.loads(content)
                guidance = Stage1Guidance(**guidance_dict)
                logger.info(f"Stage 1 guidance received: {guidance.market_bias} bias, focus on {guidance.focus_sectors}")
                return guidance

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Stage 1 JSON response: {e}")
                logger.debug(f"Raw response: {content}")
                return None

        except Exception as e:
            logger.error(f"Error in Stage 1 LLM call: {e}")
            return None


# Example usage
if __name__ == "__main__":
    import os
    from .market_aggregator import SectorStats

    logging.basicConfig(level=logging.INFO)

    # Create sample market summary
    sample_summary = MarketSummary(
        timestamp="2024-01-15T10:00:00",
        total_stocks=5000,
        sectors=[
            SectorStats(
                sector="Technology",
                stock_count=800,
                avg_price_change_5d_pct=3.5,
                pct_above_ma50=65.0,
                pct_above_ma200=58.0,
                breadth_score=0.6,
                avg_volume_ratio=2.3,
                avg_distance_from_52w_high_pct=-8.5
            ),
            SectorStats(
                sector="Healthcare",
                stock_count=600,
                avg_price_change_5d_pct=1.2,
                pct_above_ma50=52.0,
                pct_above_ma200=48.0,
                breadth_score=0.2,
                avg_volume_ratio=1.8,
                avg_distance_from_52w_high_pct=-12.3
            )
        ],
        market_breadth_score=0.45,
        avg_price_change_5d_pct=2.1,
        pct_stocks_above_ma50=58.5,
        pct_stocks_above_ma200=52.3,
        high_volume_count=450,
        near_52w_high_count=320
    )

    analyzer = Stage1Analyzer(
        api_key=os.getenv('OPENAI_API_KEY'),
        provider="openai",
        model="gpt-3.5-turbo"
    )

    # Try LLM analysis
    guidance = analyzer.analyze(sample_summary)

    if guidance:
        print(f"\nStage 1 Guidance:")
        print(f"  Market bias: {guidance.market_bias}")
        print(f"  Focus sectors: {guidance.focus_sectors}")
        print(f"  Focus patterns: {guidance.focus_patterns}")
        print(f"  Volume threshold: {guidance.filtering_criteria.volume_ratio_threshold}x")
        print(f"  52w high threshold: {guidance.filtering_criteria.distance_from_52w_high_threshold_pct}%")
        print(f"  Min 5d change: {guidance.filtering_criteria.min_price_change_5d_pct}%")
        print(f"  Target count: {guidance.target_count}")
        print(f"  Reasoning: {guidance.reasoning}")
    else:
        print("\nStage 1 LLM failed - no guidance available (fallback removed)")
