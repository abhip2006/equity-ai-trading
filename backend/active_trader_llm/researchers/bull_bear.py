"""
Bull and Bear Researchers: Debate trading thesis from different perspectives.
"""

import json
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from active_trader_llm.llm import get_llm_client, LLMMessage

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

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-nano",
        provider: str = "openai"
    ):
        """
        Initialize researcher debate

        Args:
            api_key: API key for the LLM provider
            model: LLM model to use (gpt-4.1-nano recommended, also supports: gpt-4o, claude-3-5-sonnet-20241022)
            provider: LLM provider (openai, anthropic, local)
        """
        self.client = get_llm_client(
            provider=provider,
            model=model,
            api_key=api_key
        )
        self.model = model
        self.provider = provider

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

    def debate(
        self,
        symbol: str,
        analyst_outputs: Dict,
        market_snapshot: Dict,
        memory_summary: Optional[str] = None
    ) -> Optional[tuple[BullThesis, BearThesis]]:
        """
        Generate bull and bear debate for a symbol.

        Returns None if LLM fails (refuses to fabricate debate).

        Args:
            symbol: Stock symbol
            analyst_outputs: Dict of analyst name -> output
            market_snapshot: Market context
            memory_summary: Optional recent trade context

        Returns:
            Tuple of (BullThesis, BearThesis)
        """
        prompt = self._build_debate_prompt(symbol, analyst_outputs, market_snapshot, memory_summary)

        try:
            # Use unified LLM client
            messages = [
                LLMMessage(role="system", content=self.SYSTEM_PROMPT),
                LLMMessage(role="user", content=prompt)
            ]

            response = self.client.generate(
                messages=messages,
                temperature=0.4,
                max_tokens=800
            )

            content = response.content

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

            logger.info(f"Debate for {symbol}:")
            logger.info(f"  Bull confidence: {bull.confidence:.2f}")
            logger.info(f"  Bear confidence: {bear.confidence:.2f}")

            return bull, bear

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse debate for {symbol}: {e}")
            logger.error(f"Response: {content}")
            logger.error("Debate LLM failed - refusing to fabricate synthetic debate arguments")
            return None

        except Exception as e:
            logger.error(f"Error in debate for {symbol}: {e}")
            logger.error("Debate LLM failed - refusing to fabricate synthetic debate arguments")
            return None

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

    debate = ResearcherDebate(
        api_key=os.getenv("OPENAI_API_KEY"),
        provider="openai",
        model="gpt-4.1-nano"
    )
    bull, bear = debate.debate("AAPL", sample_analyst_outputs, sample_market)

    print(f"\nBull Thesis ({bull.confidence:.2f}):")
    for point in bull.thesis:
        print(f"  - {point}")

    print(f"\nBear Thesis ({bear.confidence:.2f}):")
    for point in bear.thesis:
        print(f"  - {point}")
