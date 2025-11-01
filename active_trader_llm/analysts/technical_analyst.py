"""
Technical Analyst Agent: Analyzes technical indicators to generate trading signals.
"""

import json
import logging
from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field
from anthropic import Anthropic

logger = logging.getLogger(__name__)


class TechnicalSignal(BaseModel):
    """Technical analyst output schema"""
    analyst: str = "Technical"
    symbol: str
    signal: Literal["long", "short", "neutral"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasons: list[str]
    horizon: str = "1-3 days"


class TechnicalAnalyst:
    """
    LLM-powered technical analysis agent.

    Analyzes price action, indicators, and market context to generate signals.
    """

    SYSTEM_PROMPT = """You are TechnicalAnalyst, an expert in technical analysis for active stock trading.

Your task is to analyze technical indicators and generate trading signals with clear reasoning.

CRITICAL: Return ONLY valid JSON matching this exact schema, NO prose or explanations:
{
    "analyst": "Technical",
    "symbol": "<TICKER>",
    "signal": "long|short|neutral",
    "confidence": 0.0-1.0,
    "reasons": ["reason 1", "reason 2", "reason 3"],
    "horizon": "1-3 days"
}

Analyze the pre-calculated indicators and determine your trading signal. Be concise in reasons (3-5 bullet points max).

Avoid overfitting to single indicators. Look for confluence."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize Technical Analyst.

        Args:
            api_key: Anthropic API key (or use environment variable)
            model: Claude model to use
        """
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def _build_analysis_prompt(
        self,
        symbol: str,
        features: Dict,
        market_snapshot: Dict,
        memory_context: Optional[str] = None
    ) -> str:
        """Build the analysis prompt with all context"""
        prompt = f"""Analyze {symbol} and generate a trading signal.

TECHNICAL INDICATORS:
- RSI: {features['rsi']:.2f}
- MACD: {features['macd']:.4f}, Signal: {features['macd_signal']:.4f}, Histogram: {features['macd_hist']:.4f}
- EMA50: {features['ema_50']:.2f}, EMA200: {features['ema_200']:.2f}
- Current Price: {features['close']:.2f}
- ATR(14): {features['atr_14']:.2f}
- Bollinger Bands: Upper={features['bollinger_upper']:.2f}, Middle={features['bollinger_middle']:.2f}, Lower={features['bollinger_lower']:.2f}
- Bollinger Bandwidth: {features['bollinger_bandwidth']:.4f}

MARKET CONTEXT:
- Regime: {market_snapshot['regime_hint']}
- Breadth Score: {market_snapshot['breadth_score']:.2f}
- A/D Ratio: {market_snapshot.get('advance_decline_ratio', 1.0):.2f}
"""

        if memory_context:
            prompt += f"\nRECENT CONTEXT (from memory):\n{memory_context}\n"

        prompt += "\nGenerate signal JSON:"

        return prompt

    def analyze(
        self,
        symbol: str,
        features: Dict,
        market_snapshot: Dict,
        memory_context: Optional[str] = None
    ) -> TechnicalSignal:
        """
        Analyze symbol and generate trading signal.

        Args:
            symbol: Stock symbol
            features: Technical indicator features
            market_snapshot: Market regime and breadth data
            memory_context: Optional recent trading context from memory

        Returns:
            TechnicalSignal with decision and reasoning
        """
        prompt = self._build_analysis_prompt(symbol, features, market_snapshot, memory_context)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.3,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )

            # Extract JSON from response
            content = response.content[0].text.strip()

            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            signal_dict = json.loads(content)
            signal = TechnicalSignal(**signal_dict)

            logger.info(f"Technical analysis for {symbol}: {signal.signal} (confidence: {signal.confidence:.2f})")
            return signal

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response for {symbol}: {e}")
            logger.error(f"Response content: {content}")
            # Return neutral signal as fallback
            return TechnicalSignal(
                symbol=symbol,
                signal="neutral",
                confidence=0.0,
                reasons=["Error parsing LLM response"]
            )
        except Exception as e:
            logger.error(f"Error in technical analysis for {symbol}: {e}")
            return TechnicalSignal(
                symbol=symbol,
                signal="neutral",
                confidence=0.0,
                reasons=[f"Analysis error: {str(e)}"]
            )

    def analyze_batch(
        self,
        symbols_data: Dict[str, Dict],
        market_snapshot: Dict,
        memory_contexts: Optional[Dict[str, str]] = None
    ) -> Dict[str, TechnicalSignal]:
        """
        Analyze multiple symbols (can batch for cost efficiency).

        Args:
            symbols_data: Dict of symbol -> features
            market_snapshot: Market context
            memory_contexts: Optional dict of symbol -> memory context

        Returns:
            Dict of symbol -> TechnicalSignal
        """
        results = {}

        for symbol, features in symbols_data.items():
            memory_ctx = memory_contexts.get(symbol) if memory_contexts else None
            results[symbol] = self.analyze(symbol, features, market_snapshot, memory_ctx)

        return results


# Example usage
if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO)

    # Sample data
    sample_features = {
        'rsi': 38.5,
        'macd': -0.2,
        'macd_signal': -0.15,
        'macd_hist': -0.05,
        'ema_50': 173.4,
        'ema_200': 176.2,
        'close': 175.0,
        'atr_14': 2.15,
        'bollinger_upper': 180.0,
        'bollinger_middle': 175.0,
        'bollinger_lower': 170.0,
        'bollinger_bandwidth': 0.12
    }

    sample_market = {
        'regime_hint': 'range',
        'breadth_score': -0.15,
        'advance_decline_ratio': 0.8
    }

    analyst = TechnicalAnalyst(api_key=os.getenv("ANTHROPIC_API_KEY"))
    signal = analyst.analyze("AAPL", sample_features, sample_market)

    print(f"\nSignal: {signal.signal}")
    print(f"Confidence: {signal.confidence}")
    print(f"Reasons: {signal.reasons}")
