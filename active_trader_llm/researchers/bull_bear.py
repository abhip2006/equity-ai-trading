"""
Bull and Bear Researchers: Debate trading thesis from different perspectives.
"""

import json
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from anthropic import Anthropic

logger = logging.getLogger(__name__)


class BullThesis(BaseModel):
    """Bullish researcher output schema"""
    researcher: str = "Bullish"
    symbol: str
    thesis: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)


class BearThesis(BaseModel):
    """Bearish researcher output schema"""
    researcher: str = "Bearish"
    symbol: str
    thesis: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)


class ResearcherDebate:
    """
    Bull and Bear researchers debate trading opportunities.

    Uses analyst outputs and memory to construct opposing viewpoints.
    """

    SYSTEM_PROMPT = """You are two expert researchers debating a trading opportunity: BullResearcher and BearResearcher.

Your task is to provide BOTH perspectives in a single JSON response with two objects.

CRITICAL: Return ONLY valid JSON array with EXACTLY two objects:
[
    {
        "researcher": "Bullish",
        "symbol": "<TICKER>",
        "thesis": ["bull argument 1", "bull argument 2", "bull argument 3"],
        "confidence": 0.0-1.0
    },
    {
        "researcher": "Bearish",
        "symbol": "<TICKER>",
        "thesis": ["bear argument 1", "bear argument 2", "bear argument 3"],
        "confidence": 0.0-1.0
    }
]

Guidelines for Bull Researcher:
- Focus on positive technical signals (momentum, support levels)
- Highlight favorable breadth/regime conditions
- Consider bullish catalysts and sentiment
- Confidence reflects strength of bullish case

Guidelines for Bear Researcher:
- Focus on negative technical signals (resistance, divergences)
- Highlight unfavorable breadth/regime conditions
- Consider bearish risks and headwinds
- Confidence reflects strength of bearish case

Both researchers should:
- Use analyst outputs as evidence
- Consider recent trade history from memory
- Be intellectually honest about risks/opportunities
- Provide 3-5 specific points each"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        """Initialize researcher debate"""
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def _build_debate_prompt(
        self,
        symbol: str,
        analyst_outputs: Dict,
        market_snapshot: Dict,
        memory_summary: Optional[str] = None
    ) -> str:
        """Build debate prompt with analyst context"""
        prompt = f"""Debate the trading opportunity for {symbol}.

ANALYST OUTPUTS:
"""
        # Technical analyst
        if 'technical' in analyst_outputs:
            tech = analyst_outputs['technical']
            prompt += f"""
Technical Analyst:
  Signal: {tech.get('signal', 'neutral')}
  Confidence: {tech.get('confidence', 0.0):.2f}
  Reasons: {', '.join(tech.get('reasons', []))}
"""

        # Breadth analyst
        if 'breadth' in analyst_outputs:
            breadth = analyst_outputs['breadth']
            prompt += f"""
Breadth Health:
  Regime: {breadth.get('regime', 'unknown')}
  Breadth Score: {breadth.get('breadth_score', 0.0):.2f}
  Notes: {', '.join(breadth.get('notes', [])[:2])}
"""

        # Sentiment (if available)
        if 'sentiment' in analyst_outputs:
            sent = analyst_outputs['sentiment']
            prompt += f"""
Sentiment:
  Score: {sent.get('sentiment_score', 0.0):.2f}
  Drivers: {', '.join(sent.get('drivers', []))}
"""

        # Macro (if available)
        if 'macro' in analyst_outputs:
            macro = analyst_outputs['macro']
            prompt += f"""
Macro:
  Bias: {macro.get('bias', 'neutral')}
  Context: {', '.join(macro.get('market_context', []))}
"""

        if memory_summary:
            prompt += f"\nRECENT TRADING CONTEXT:\n{memory_summary}\n"

        prompt += "\nProvide BOTH bull and bear theses as JSON array:"

        return prompt

    def _validate_debate(
        self,
        bull: BullThesis,
        bear: BearThesis,
        analyst_outputs: Dict
    ) -> tuple[bool, Optional[str]]:
        """
        Validate debate against analyst outputs to prevent hallucinations.

        Args:
            bull: Bull thesis from LLM
            bear: Bear thesis from LLM
            analyst_outputs: Actual analyst data (LIVE DATA)

        Returns:
            Tuple of (is_valid, error_message)
        """
        # 1. Validate confidence ranges
        if bull.confidence < 0.0 or bull.confidence > 1.0:
            return False, f"Bull confidence {bull.confidence} outside valid range [0.0, 1.0]"
        if bear.confidence < 0.0 or bear.confidence > 1.0:
            return False, f"Bear confidence {bear.confidence} outside valid range [0.0, 1.0]"

        # 2. Validate thesis points are not empty
        if not bull.thesis or len(bull.thesis) == 0:
            return False, "Bull thesis must include at least one point"
        if not bear.thesis or len(bear.thesis) == 0:
            return False, "Bear thesis must include at least one point"

        # 3. Validate thesis points are not trivially short
        if any(len(point) < 10 for point in bull.thesis):
            return False, "Bull thesis points too short (min 10 chars)"
        if any(len(point) < 10 for point in bear.thesis):
            return False, "Bear thesis points too short (min 10 chars)"

        # 4. Sanity check: if technical signal is strongly directional,
        # the opposing thesis shouldn't be overconfident
        tech_signal = analyst_outputs.get('technical', {}).get('signal', 'neutral')
        tech_confidence = analyst_outputs.get('technical', {}).get('confidence', 0.5)

        if tech_signal == 'long' and tech_confidence > 0.8:
            # Strong bullish technical, bear shouldn't be overconfident
            if bear.confidence > 0.8:
                return False, f"Bear overconfident ({bear.confidence:.2f}) despite strong bullish technical signal ({tech_confidence:.2f})"

        elif tech_signal == 'short' and tech_confidence > 0.8:
            # Strong bearish technical, bull shouldn't be overconfident
            if bull.confidence > 0.8:
                return False, f"Bull overconfident ({bull.confidence:.2f}) despite strong bearish technical signal ({tech_confidence:.2f})"

        # 5. Confidences should be somewhat balanced (not both >0.9 or both <0.1)
        if bull.confidence > 0.9 and bear.confidence > 0.9:
            return False, f"Both bull ({bull.confidence:.2f}) and bear ({bear.confidence:.2f}) overconfident"
        if bull.confidence < 0.1 and bear.confidence < 0.1:
            return False, f"Both bull ({bull.confidence:.2f}) and bear ({bear.confidence:.2f}) underconfident"

        return True, None

    def debate(
        self,
        symbol: str,
        analyst_outputs: Dict,
        market_snapshot: Dict,
        memory_summary: Optional[str] = None
    ) -> tuple[BullThesis, BearThesis]:
        """
        Generate bull and bear debate for a symbol.

        Args:
            symbol: Stock symbol
            analyst_outputs: Dict of analyst name -> output
            market_snapshot: Market context
            memory_summary: Optional recent trade context

        Returns:
            Tuple of (BullThesis, BearThesis)
        """
        prompt = self._build_debate_prompt(symbol, analyst_outputs, market_snapshot, memory_summary)

        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=800,
                    temperature=0.4,
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

                debate_list = json.loads(content)

                if not isinstance(debate_list, list) or len(debate_list) != 2:
                    raise ValueError("Expected JSON array with 2 objects")

                bull_dict = next(d for d in debate_list if d.get('researcher') == 'Bullish')
                bear_dict = next(d for d in debate_list if d.get('researcher') == 'Bearish')

                bull = BullThesis(**bull_dict)
                bear = BearThesis(**bear_dict)

                # CRITICAL: Validate against analyst data to prevent hallucinations
                is_valid, error_msg = self._validate_debate(bull, bear, analyst_outputs)

                if not is_valid:
                    logger.warning(f"Debate validation failed for {symbol}: {error_msg}")

                    if attempt < max_retries - 1:
                        logger.info(f"Retrying debate (attempt {attempt + 2}/{max_retries})")
                        prompt += f"\n\nPREVIOUS ATTEMPT FAILED: {error_msg}\nGenerate a valid debate based on the actual analyst signals provided."
                        continue
                    else:
                        logger.error(f"Failed to generate valid debate after {max_retries} attempts, using fallback")
                        return self._fallback_debate(symbol, analyst_outputs)

                logger.info(f"Debate for {symbol}:")
                logger.info(f"  Bull confidence: {bull.confidence:.2f}")
                logger.info(f"  Bear confidence: {bear.confidence:.2f}")

                return bull, bear

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse debate for {symbol}: {e}")
                logger.error(f"Response: {content}")

                if attempt < max_retries - 1:
                    prompt += f"\n\nPREVIOUS ATTEMPT RETURNED INVALID JSON: {e}\nReturn ONLY valid JSON array with 2 objects."
                    continue
                else:
                    return self._fallback_debate(symbol, analyst_outputs)

            except Exception as e:
                logger.error(f"Error in debate for {symbol}: {e}")

                if attempt < max_retries - 1:
                    continue
                else:
                    return self._fallback_debate(symbol, analyst_outputs)

        return self._fallback_debate(symbol, analyst_outputs)

    def _fallback_debate(self, symbol: str, analyst_outputs: Dict) -> tuple[BullThesis, BearThesis]:
        """Generate simple rule-based debate as fallback"""
        tech_signal = analyst_outputs.get('technical', {}).get('signal', 'neutral')

        if tech_signal == 'long':
            bull = BullThesis(
                symbol=symbol,
                thesis=["Technical indicators favor upside", "Momentum positive"],
                confidence=0.6
            )
            bear = BearThesis(
                symbol=symbol,
                thesis=["Could face resistance", "Market conditions uncertain"],
                confidence=0.4
            )
        elif tech_signal == 'short':
            bull = BullThesis(
                symbol=symbol,
                thesis=["Potential oversold bounce", "Support levels nearby"],
                confidence=0.4
            )
            bear = BearThesis(
                symbol=symbol,
                thesis=["Technical weakness evident", "Downtrend intact"],
                confidence=0.6
            )
        else:
            bull = BullThesis(
                symbol=symbol,
                thesis=["Mixed signals - wait for clarity"],
                confidence=0.3
            )
            bear = BearThesis(
                symbol=symbol,
                thesis=["Mixed signals - avoid risk"],
                confidence=0.3
            )

        return bull, bear

    def debate_batch(
        self,
        symbols_data: Dict[str, Dict],
        market_snapshot: Dict,
        memory_summaries: Optional[Dict[str, str]] = None
    ) -> Dict[str, tuple[BullThesis, BearThesis]]:
        """
        Debate multiple symbols.

        Args:
            symbols_data: Dict of symbol -> analyst outputs
            market_snapshot: Market context
            memory_summaries: Optional dict of symbol -> memory summary

        Returns:
            Dict of symbol -> (BullThesis, BearThesis)
        """
        results = {}

        for symbol, analyst_outputs in symbols_data.items():
            memory_summary = memory_summaries.get(symbol) if memory_summaries else None
            results[symbol] = self.debate(symbol, analyst_outputs, market_snapshot, memory_summary)

        return results


# Example usage
if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO)

    sample_analyst_outputs = {
        'technical': {
            'signal': 'long',
            'confidence': 0.7,
            'reasons': ['RSI recovery from oversold', 'MACD crossover', 'Price above EMA50']
        },
        'breadth': {
            'regime': 'trending_bull',
            'breadth_score': 0.45,
            'notes': ['Strong advance/decline', 'New highs expanding']
        }
    }

    sample_market = {
        'regime_hint': 'trending_bull',
        'breadth_score': 0.45
    }

    debate = ResearcherDebate(api_key=os.getenv("ANTHROPIC_API_KEY"))
    bull, bear = debate.debate("AAPL", sample_analyst_outputs, sample_market)

    print(f"\nBull Thesis ({bull.confidence:.2f}):")
    for point in bull.thesis:
        print(f"  - {point}")

    print(f"\nBear Thesis ({bear.confidence:.2f}):")
    for point in bear.thesis:
        print(f"  - {point}")
