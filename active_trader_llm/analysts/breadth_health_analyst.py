"""
Breadth Health Analyst: Determines market regime and overall market health.
"""

import json
import logging
from typing import Dict, Literal, Optional, List
from pydantic import BaseModel, Field
from anthropic import Anthropic

logger = logging.getLogger(__name__)


class BreadthHealthSignal(BaseModel):
    """Breadth health analyst output schema"""
    analyst: str = "BreadthHealth"
    regime: Literal["trending_bull", "trending_bear", "range", "risk_off"]
    breadth_score: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    notes: List[str]


class BreadthHealthAnalyst:
    """
    Analyzes market breadth and determines regime.

    Uses both quantitative metrics and LLM interpretation.
    """

    SYSTEM_PROMPT = """You are BreadthHealthAnalyst, an expert in market breadth analysis and regime identification.

Analyze market-wide metrics to determine the current trading regime.

CRITICAL: Return ONLY valid JSON matching this exact schema:
{
    "analyst": "BreadthHealth",
    "regime": "trending_bull|trending_bear|range|risk_off",
    "breadth_score": -1.0 to 1.0,
    "confidence": 0.0 to 1.0,
    "notes": ["observation 1", "observation 2", "observation 3"]
}

Regime Definitions:
- trending_bull: Strong breadth, advancing issues, momentum positive
- trending_bear: Weak breadth, declining issues, momentum negative
- range: Mixed breadth, choppy price action, no clear trend
- risk_off: Market stress, breadth deteriorating rapidly, defensive posture

Use breadth metrics holistically. Confidence reflects clarity of regime."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        """Initialize Breadth Health Analyst"""
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def _build_analysis_prompt(self, market_snapshot: Dict) -> str:
        """Build analysis prompt from market snapshot"""
        prompt = f"""Analyze current market breadth and determine regime.

MARKET BREADTH METRICS:
- Breadth Score: {market_snapshot['breadth_score']:.2f} (-1=bearish, +1=bullish)
- Advance/Decline Ratio: {market_snapshot.get('advance_decline_ratio', 1.0):.2f}
- New Highs: {market_snapshot.get('new_highs', 0)}
- New Lows: {market_snapshot.get('new_lows', 0)}
- Up Volume Ratio: {market_snapshot.get('up_volume_ratio', 0.5):.2f}
- VIX: {market_snapshot.get('vix', 15.0):.1f}

CURRENT REGIME HINT: {market_snapshot.get('regime_hint', 'unknown')}

Determine the most appropriate regime and provide clear rationale.
Generate regime JSON:"""

        return prompt

    def _validate_regime(
        self,
        signal: BreadthHealthSignal,
        market_snapshot: Dict
    ) -> tuple[bool, Optional[str]]:
        """
        Validate LLM regime classification against actual breadth data to prevent hallucinations.

        Args:
            signal: Signal from LLM
            market_snapshot: Actual market data (LIVE DATA)

        Returns:
            Tuple of (is_valid, error_message)
        """
        breadth_score = market_snapshot.get('breadth_score', 0.0)
        vix = market_snapshot.get('vix', 15.0)
        ad_ratio = market_snapshot.get('advance_decline_ratio', 1.0)

        # 1. Validate breadth_score in valid range
        if signal.breadth_score < -1.0 or signal.breadth_score > 1.0:
            return False, f"Breadth score {signal.breadth_score} outside valid range [-1.0, 1.0]"

        # 2. Validate breadth_score matches input data (allow small deviation for interpretation)
        if abs(signal.breadth_score - breadth_score) > 0.3:
            return False, f"Breadth score {signal.breadth_score} deviates too much from actual {breadth_score}"

        # 3. Validate regime matches the actual data
        if signal.regime == "trending_bull":
            # Trending bull requires positive breadth and decent A/D ratio
            if breadth_score < 0.2:
                return False, f"Regime 'trending_bull' inconsistent with breadth_score {breadth_score:.2f} (should be > 0.2)"
            if ad_ratio < 1.0:
                return False, f"Regime 'trending_bull' inconsistent with A/D ratio {ad_ratio:.2f} (should be > 1.0)"

        elif signal.regime == "trending_bear":
            # Trending bear requires negative breadth
            if breadth_score > -0.2:
                return False, f"Regime 'trending_bear' inconsistent with breadth_score {breadth_score:.2f} (should be < -0.2)"

        elif signal.regime == "risk_off":
            # Risk off requires very negative breadth OR high VIX
            if breadth_score > -0.3 and vix < 20:
                return False, f"Regime 'risk_off' inconsistent with breadth_score {breadth_score:.2f} and VIX {vix:.1f}"

        # 4. Validate confidence range
        if signal.confidence < 0.0 or signal.confidence > 1.0:
            return False, f"Confidence {signal.confidence} outside valid range [0.0, 1.0]"

        # 5. Validate notes are not empty
        if not signal.notes or len(signal.notes) == 0:
            return False, "Signal must include notes"

        return True, None

    def analyze(self, market_snapshot: Dict) -> BreadthHealthSignal:
        """
        Analyze market breadth and determine regime.

        Args:
            market_snapshot: Market-wide breadth and health metrics

        Returns:
            BreadthHealthSignal with regime determination
        """
        prompt = self._build_analysis_prompt(market_snapshot)

        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=400,
                    temperature=0.2,
                    system=self.SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}]
                )

                content = response.content[0].text.strip()

                # Handle markdown code blocks
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()

                signal_dict = json.loads(content)
                signal = BreadthHealthSignal(**signal_dict)

                # CRITICAL: Validate against live breadth data to prevent hallucinations
                is_valid, error_msg = self._validate_regime(signal, market_snapshot)

                if not is_valid:
                    logger.warning(f"Regime validation failed: {error_msg}")

                    if attempt < max_retries - 1:
                        logger.info(f"Retrying regime analysis (attempt {attempt + 2}/{max_retries})")
                        prompt += f"\n\nPREVIOUS ATTEMPT FAILED: {error_msg}\nGenerate a regime that matches the actual breadth metrics provided."
                        continue
                    else:
                        logger.error(f"Failed to generate valid regime after {max_retries} attempts, using rule-based fallback")
                        regime = self._rule_based_regime(market_snapshot)
                        return BreadthHealthSignal(
                            regime=regime,
                            breadth_score=market_snapshot.get('breadth_score', 0.0),
                            confidence=0.5,
                            notes=["Fallback to rule-based regime after validation failures"]
                        )

                logger.info(f"Market regime: {signal.regime} (confidence: {signal.confidence:.2f})")
                logger.info(f"  Notes: {', '.join(signal.notes[:2])}")

                return signal

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse breadth analysis: {e}")
                logger.error(f"Response: {content}")

                if attempt < max_retries - 1:
                    prompt += f"\n\nPREVIOUS ATTEMPT RETURNED INVALID JSON: {e}\nReturn ONLY valid JSON matching the schema."
                    continue
                else:
                    # Fallback to rule-based regime
                    regime = self._rule_based_regime(market_snapshot)
                    return BreadthHealthSignal(
                        regime=regime,
                        breadth_score=market_snapshot.get('breadth_score', 0.0),
                        confidence=0.5,
                        notes=["Fallback to rule-based regime determination"]
                    )

            except Exception as e:
                logger.error(f"Error in breadth analysis: {e}")
                regime = self._rule_based_regime(market_snapshot)
                return BreadthHealthSignal(
                    regime=regime,
                    breadth_score=market_snapshot.get('breadth_score', 0.0),
                    confidence=0.3,
                    notes=[f"Error: {str(e)}"]
                )

        # Should not reach here, but safety fallback
        regime = self._rule_based_regime(market_snapshot)
        return BreadthHealthSignal(
            regime=regime,
            breadth_score=market_snapshot.get('breadth_score', 0.0),
            confidence=0.3,
            notes=["Max retries exceeded"]
        )

    def _rule_based_regime(self, market_snapshot: Dict) -> Literal["trending_bull", "trending_bear", "range", "risk_off"]:
        """Simple rule-based regime determination as fallback"""
        breadth = market_snapshot.get('breadth_score', 0.0)
        vix = market_snapshot.get('vix', 15.0)
        ad_ratio = market_snapshot.get('advance_decline_ratio', 1.0)

        if vix > 25 or breadth < -0.6:
            return "risk_off"
        elif breadth > 0.5 and ad_ratio > 1.5:
            return "trending_bull"
        elif breadth < -0.3 and ad_ratio < 0.7:
            return "trending_bear"
        else:
            return "range"


# Example usage
if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO)

    sample_market = {
        'breadth_score': 0.35,
        'advance_decline_ratio': 1.8,
        'new_highs': 45,
        'new_lows': 12,
        'up_volume_ratio': 0.65,
        'vix': 14.2,
        'regime_hint': 'trending_bull'
    }

    analyst = BreadthHealthAnalyst(api_key=os.getenv("ANTHROPIC_API_KEY"))
    signal = analyst.analyze(sample_market)

    print(f"\nRegime: {signal.regime}")
    print(f"Breadth Score: {signal.breadth_score:.2f}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Notes: {signal.notes}")
