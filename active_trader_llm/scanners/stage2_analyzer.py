"""
Stage 2 Analyzer: LLM interprets pre-calculated indicators in batches.

Batches 15 stocks per LLM call (5-10 calls total for 50-150 candidates).
LLM receives ONLY pre-calculated indicators, never raw price data.

ANTI-HALLUCINATION DESIGN:
- All indicators calculated by pandas_ta (in raw_data_scanner.py)
- LLM only interprets formatted indicator values
- No opportunity for LLM to invent numbers
"""

import json
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from anthropic import Anthropic

from .raw_data_scanner import TechnicalIndicators, RawDataScanner
from .stage1_analyzer import Stage1Guidance

logger = logging.getLogger(__name__)


class StockEvaluation(BaseModel):
    """Evaluation of a single stock"""
    symbol: str
    favorable: bool
    pattern_match: List[str]  # e.g., ["breakout", "momentum"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str


class Stage2Result(BaseModel):
    """Result of Stage 2 batch analysis"""
    favorable_symbols: List[str]
    evaluations: Dict[str, StockEvaluation]  # symbol -> evaluation
    batch_summary: str


class Stage2Analyzer:
    """
    LLM-powered Stage 2 deep analysis in batches.

    Analyzes pre-calculated indicators to identify high-probability setups.
    Batches 15 stocks per call to minimize LLM costs.
    """

    SYSTEM_PROMPT = """You are a Technical Trading Analyst identifying high-probability setups.

You receive PRE-CALCULATED technical indicators (from pandas_ta library, NOT from you).
Your task is to INTERPRET these indicators and identify favorable trading opportunities.

CRITICAL: Return ONLY valid JSON matching this exact schema, NO prose:
{
    "favorable_symbols": ["SYMBOL1", "SYMBOL2", ...],
    "evaluations": {
        "SYMBOL1": {
            "symbol": "SYMBOL1",
            "favorable": true,
            "pattern_match": ["breakout", "momentum"],
            "confidence": 0.75,
            "reasoning": "Strong breakout above EMA50 with 3.2x volume confirming momentum"
        },
        ...
    },
    "batch_summary": "Brief summary of market conditions in this batch"
}

CRITICAL RULES:
1. You are NOT calculating indicators - they are already calculated
2. You are INTERPRETING pre-calculated values
3. Be selective - only mark favorable if genuinely good setup
4. Consider Stage 1 guidance (market bias, focus patterns)
5. Look for confluence of multiple indicators"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022", temperature: float = 0.3):
        """
        Initialize Stage 2 Analyzer.

        Args:
            api_key: Anthropic API key
            model: Claude model to use
            temperature: Temperature for LLM (0.3 for consistency)
        """
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.scanner = RawDataScanner()

    def _build_batch_prompt(
        self,
        batch_indicators: Dict[str, TechnicalIndicators],
        guidance: Stage1Guidance
    ) -> str:
        """
        Build prompt for a batch of stocks with pre-calculated indicators.

        Args:
            batch_indicators: Dict of symbol -> TechnicalIndicators
            guidance: Stage1Guidance from Stage 1

        Returns:
            Formatted prompt string
        """
        prompt = f"""Analyze this batch of stocks and identify favorable setups.

STAGE 1 CONTEXT:
- Market bias: {guidance.market_bias}
- Focus patterns: {', '.join(guidance.focus_patterns)}
- Focus sectors: {', '.join(guidance.focus_sectors)}

PRE-CALCULATED INDICATORS (from pandas_ta, NOT from you):

"""

        # Add each stock's indicators
        for symbol, indicators in batch_indicators.items():
            prompt += self.scanner.format_for_llm(indicators)
            prompt += "\n\n" + "="*50 + "\n\n"

        prompt += """Based on these PRE-CALCULATED indicators, identify which stocks are favorable.

Return your analysis as JSON:"""

        return prompt

    def analyze_batch(
        self,
        batch_symbols: List[str],
        indicators_map: Dict[str, TechnicalIndicators],
        guidance: Stage1Guidance
    ) -> Optional[Stage2Result]:
        """
        Analyze a batch of stocks (1 LLM call).

        Args:
            batch_symbols: List of symbols in this batch (max 15)
            indicators_map: Dict mapping all symbols -> TechnicalIndicators
            guidance: Stage1Guidance from Stage 1

        Returns:
            Stage2Result or None on error
        """
        # Filter indicators to just this batch
        batch_indicators = {
            sym: indicators_map[sym]
            for sym in batch_symbols
            if sym in indicators_map
        }

        if not batch_indicators:
            logger.warning(f"No indicators available for batch: {batch_symbols}")
            return None

        prompt = self._build_batch_prompt(batch_indicators, guidance)

        try:
            logger.info(f"Calling LLM for Stage 2 batch analysis ({len(batch_indicators)} stocks)...")

            response = self.client.messages.create(
                model=self.model,
                max_tokens=3000,
                temperature=self.temperature,
                system=self.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract JSON from response
            content = response.content[0].text

            # Try to parse JSON
            try:
                result_dict = json.loads(content)
                result = Stage2Result(**result_dict)
                logger.info(f"Stage 2 batch analysis: {len(result.favorable_symbols)}/{len(batch_indicators)} favorable")
                return result

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Stage 2 JSON response: {e}")
                logger.debug(f"Raw response: {content}")
                return None

        except Exception as e:
            logger.error(f"Error in Stage 2 LLM call: {e}")
            return None

    def analyze_all_batches(
        self,
        candidates: List[str],
        indicators_map: Dict[str, TechnicalIndicators],
        guidance: Stage1Guidance,
        batch_size: int = 15,
        max_batches: int = 10
    ) -> List[str]:
        """
        Analyze all candidates in batches (5-10 LLM calls).

        Args:
            candidates: List of candidate symbols from Stage 2A filter
            indicators_map: Dict of all indicators
            guidance: Stage1Guidance from Stage 1
            batch_size: Stocks per batch (default 15)
            max_batches: Maximum batches to process (default 10)

        Returns:
            List of all favorable symbols across all batches
        """
        all_favorable = []

        # Split into batches
        batches = [
            candidates[i:i + batch_size]
            for i in range(0, len(candidates), batch_size)
        ]

        # Limit to max_batches
        batches = batches[:max_batches]

        logger.info(f"Analyzing {len(candidates)} candidates in {len(batches)} batches")

        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)}: {len(batch)} stocks")

            result = self.analyze_batch(batch, indicators_map, guidance)

            if result:
                all_favorable.extend(result.favorable_symbols)
                logger.info(f"Batch {i+1}: {len(result.favorable_symbols)} favorable")
            else:
                logger.warning(f"Batch {i+1} failed, skipping")

        logger.info(f"Total favorable stocks across all batches: {len(all_favorable)}")
        return all_favorable

    def get_fallback_selection(
        self,
        candidates: List[str],
        indicators_map: Dict[str, TechnicalIndicators],
        guidance: Stage1Guidance,
        max_count: int = 20
    ) -> List[str]:
        """
        Rule-based fallback if LLM fails.

        Args:
            candidates: List of candidates
            indicators_map: Dict of indicators
            guidance: Stage1Guidance
            max_count: Max stocks to return

        Returns:
            List of symbols
        """
        logger.warning("Using fallback rule-based selection (LLM failed)")

        scored = []

        for symbol in candidates:
            if symbol not in indicators_map:
                continue

            ind = indicators_map[symbol]
            score = 0.0

            # Momentum score
            if ind.price_change_5d_pct and ind.price_change_5d_pct > 3.0:
                score += 1.0
            if ind.rsi_14 and 50 < ind.rsi_14 < 70:
                score += 1.0
            if ind.macd_hist and ind.macd_hist > 0:
                score += 1.0

            # Trend score
            if ind.pct_from_ema50 and ind.pct_from_ema50 > 0:
                score += 1.0
            if ind.pct_from_ema200 and ind.pct_from_ema200 > 0:
                score += 0.5

            # Volume score
            if ind.volume_ratio and ind.volume_ratio > 2.0:
                score += 1.5

            scored.append((symbol, score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Take top N
        return [sym for sym, score in scored[:max_count]]


# Example usage
if __name__ == "__main__":
    import os
    from .stage1_analyzer import FilteringCriteria

    logging.basicConfig(level=logging.INFO)

    # Sample guidance
    sample_guidance = Stage1Guidance(
        market_bias="bullish",
        focus_sectors=["Technology"],
        focus_patterns=["breakouts", "momentum"],
        filtering_criteria=FilteringCriteria(
            volume_ratio_threshold=2.5,
            distance_from_52w_high_threshold_pct=7.0,
            min_price_change_5d_pct=3.0
        ),
        target_count=100,
        reasoning="Bullish market favoring tech breakouts"
    )

   

    analyzer = Stage2Analyzer(api_key=os.getenv('ANTHROPIC_API_KEY'))

    # Analyze batch
    result = analyzer.analyze_batch(
        batch_symbols=["AAPL"],
        indicators_map=sample_indicators,
        guidance=sample_guidance
    )

    if result:
        print(f"\nStage 2 Result:")
        print(f"  Favorable: {result.favorable_symbols}")
        print(f"  Summary: {result.batch_summary}")
        for sym, eval in result.evaluations.items():
            print(f"\n  {sym}:")
            print(f"    Favorable: {eval.favorable}")
            print(f"    Patterns: {eval.pattern_match}")
            print(f"    Confidence: {eval.confidence:.2f}")
            print(f"    Reasoning: {eval.reasoning}")
