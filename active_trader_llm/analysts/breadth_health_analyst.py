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

Analyze the raw market breadth metrics provided and determine which regime best describes current conditions.
Base your determination on the data itself without relying on fixed thresholds.
Use breadth metrics holistically. Confidence reflects clarity of regime."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        """Initialize Breadth Health Analyst"""
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def _build_analysis_prompt(self, market_snapshot: Dict) -> str:
        """Build analysis prompt from market snapshot"""
        # Get VIX value or None if not provided
        vix = market_snapshot.get('vix')
        vix_str = f"{vix:.1f}" if vix is not None else "NOT AVAILABLE"

        # Log warning if VIX is missing
        if vix is None:
            logger.warning("VIX data not provided - regime analysis may be less accurate")

        prompt = f"""Analyze current market breadth and determine regime.

MARKET BREADTH METRICS:
- Breadth Score: {market_snapshot['breadth_score']:.2f}
- Advance/Decline Ratio: {market_snapshot.get('advance_decline_ratio', 1.0):.2f}
- New Highs: {market_snapshot.get('new_highs', 0)}
- New Lows: {market_snapshot.get('new_lows', 0)}
- Up Volume Ratio: {market_snapshot.get('up_volume_ratio', 0.5):.2f}
- VIX: {vix_str}

CURRENT REGIME HINT: {market_snapshot.get('regime_hint', 'unknown')}

{"NOTE: VIX data unavailable - rely more heavily on breadth metrics for regime determination." if vix is None else ""}

Determine the most appropriate regime and provide clear rationale.
Generate regime JSON:"""

        return prompt

    def analyze(self, market_snapshot: Dict) -> Optional[BreadthHealthSignal]:
        """
        Analyze market breadth and determine regime.

        Returns None if LLM fails (refuses to use hardcoded regime thresholds).

        Args:
            market_snapshot: Market-wide breadth and health metrics

        Returns:
            BreadthHealthSignal with regime determination
        """
        prompt = self._build_analysis_prompt(market_snapshot)

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

            logger.info(f"Market regime: {signal.regime} (confidence: {signal.confidence:.2f})")
            logger.info(f"  Notes: {', '.join(signal.notes[:2])}")

            return signal

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse breadth analysis: {e}")
            logger.error(f"Response: {content}")
            logger.error("Breadth analysis LLM failed - refusing to use hardcoded regime thresholds")
            return None

        except Exception as e:
            logger.error(f"Error in breadth analysis: {e}")
            logger.error("Breadth analysis LLM failed - refusing to use hardcoded regime thresholds")
            return None


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
