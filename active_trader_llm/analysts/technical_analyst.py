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
    # Price and risk data for trader agent
    price: float = Field(..., description="Current price")
    atr: float = Field(..., description="ATR for position sizing")


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
        # Extract indicators from nested structure
        daily = features.get('daily_indicators', {})
        weekly = features.get('weekly_indicators', {})
        ohlcv = features.get('ohlcv', {})

        # Get price data
        close = ohlcv.get('close', 0.0)
        volume = ohlcv.get('volume', 0)

        # Get daily indicators (using actual indicators we calculate)
        rsi = daily.get('rsi')
        atr = daily.get('atr')
        ema_5 = daily.get('ema_5')
        ema_10 = daily.get('ema_10')
        ema_20 = daily.get('ema_20')
        sma_50 = daily.get('sma_50')
        sma_200 = daily.get('sma_200')

        # Get weekly indicators
        weekly_ema_10 = weekly.get('ema_10')
        weekly_sma_21 = weekly.get('sma_21')
        weekly_sma_50 = weekly.get('sma_50')

        # Build prompt with available indicators
        prompt = f"""Analyze {symbol} and generate a trading signal.

TECHNICAL INDICATORS (Daily):
- Current Price: ${close:.2f}
- Volume: {volume:,}"""

        if rsi is not None:
            prompt += f"\n- RSI(14): {rsi:.2f}"
        if atr is not None:
            prompt += f"\n- ATR(14): ${atr:.2f}"

        prompt += "\n\nMOVING AVERAGES (Daily):"
        if ema_5 is not None:
            prompt += f"\n- EMA(5): ${ema_5:.2f}"
        if ema_10 is not None:
            prompt += f"\n- EMA(10): ${ema_10:.2f}"
        if ema_20 is not None:
            prompt += f"\n- EMA(20): ${ema_20:.2f}"
        if sma_50 is not None:
            prompt += f"\n- SMA(50): ${sma_50:.2f}"
            if close > 0:
                pct_from_sma50 = ((close - sma_50) / sma_50) * 100
                prompt += f" ({pct_from_sma50:+.1f}% from price)"
        if sma_200 is not None:
            prompt += f"\n- SMA(200): ${sma_200:.2f}"
            if close > 0:
                pct_from_sma200 = ((close - sma_200) / sma_200) * 100
                prompt += f" ({pct_from_sma200:+.1f}% from price)"

        prompt += "\n\nWEEKLY INDICATORS:"
        if weekly_ema_10 is not None:
            prompt += f"\n- EMA(10): ${weekly_ema_10:.2f}"
        if weekly_sma_21 is not None:
            prompt += f"\n- SMA(21): ${weekly_sma_21:.2f}"
        if weekly_sma_50 is not None:
            prompt += f"\n- SMA(50): ${weekly_sma_50:.2f}"

        prompt += f"""

MARKET CONTEXT:
- Regime: {market_snapshot.get('regime_hint', 'unknown')}
- Breadth Score: {market_snapshot.get('breadth_score', 0.0):.2f}
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
    ) -> Optional[TechnicalSignal]:
        """
        Analyze symbol and generate trading signal.

        Returns None if critical data (price, ATR) is missing or LLM fails.

        Args:
            symbol: Stock symbol
            features: Technical indicator features
            market_snapshot: Market regime and breadth data
            memory_context: Optional recent trading context from memory

        Returns:
            TechnicalSignal with decision and reasoning
        """
        # Extract price and ATR (critical data - abort if missing)
        ohlcv = features.get('ohlcv', {})
        daily = features.get('daily_indicators', {})

        price = ohlcv.get('close')
        atr = daily.get('atr')

        # Abort if critical data is missing
        if price is None or price == 0.0:
            logger.error(f"{symbol}: Price data missing or zero - cannot analyze without price")
            return None
        if atr is None:
            logger.error(f"{symbol}: ATR data missing - cannot determine position sizing without ATR")
            return None

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

            # Add price and ATR to the signal
            signal_dict['price'] = price
            signal_dict['atr'] = atr

            signal = TechnicalSignal(**signal_dict)

            logger.info(f"Technical analysis for {symbol}: {signal.signal} (confidence: {signal.confidence:.2f})")
            return signal

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response for {symbol}: {e}")
            logger.error(f"Response content: {content}")
            logger.error("Technical analysis LLM failed - refusing to fabricate neutral signal")
            return None
        except Exception as e:
            logger.error(f"Error in technical analysis for {symbol}: {e}")
            logger.error("Technical analysis LLM failed - refusing to fabricate neutral signal")
            return None

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

    # Sample data (using new FeatureSet structure)
    sample_features = {
        'symbol': 'AAPL',
        'timestamp': '2025-01-15',
        'daily_indicators': {
            'rsi': 38.5,
            'atr': 2.15,
            'ema_5': 174.0,
            'ema_10': 173.5,
            'ema_20': 173.0,
            'sma_50': 173.4,
            'sma_200': 176.2
        },
        'weekly_indicators': {
            'ema_10': 175.0,
            'sma_21': 174.5,
            'sma_50': 173.8
        },
        'ohlcv': {
            'open': 174.5,
            'high': 176.0,
            'low': 174.0,
            'close': 175.0,
            'volume': 50000000
        }
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
    print(f"Price: ${signal.price:.2f}")
    print(f"ATR: ${signal.atr:.2f}")
    print(f"Reasons: {signal.reasons}")
