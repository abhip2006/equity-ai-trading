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

Guidelines:
- RSI <30 = oversold (potential long), RSI >70 = overbought (potential short)
- MACD crossover (hist > 0) = bullish, crossunder (hist < 0) = bearish
- Price above EMA50 and EMA200 = uptrend, below both = downtrend
- Bollinger bandwidth expansion = increased volatility/opportunity
- ATR shows volatility level for stop-loss sizing
- Align with market regime when relevant (trending_bull favors longs, etc.)
- Confidence should reflect conviction based on indicator alignment
- Be concise in reasons (3-5 bullet points max)

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
- Breadth Score: {market_snapshot['breadth_score']:.2f} (-1=bearish, +1=bullish)
- A/D Ratio: {market_snapshot.get('advance_decline_ratio', 1.0):.2f}
"""

        if memory_context:
            prompt += f"\nRECENT CONTEXT (from memory):\n{memory_context}\n"

        prompt += "\nGenerate signal JSON:"

        return prompt

    def _validate_signal(
        self,
        signal: TechnicalSignal,
        features: Dict,
        market_snapshot: Dict
    ) -> tuple[bool, Optional[str]]:
        """
        Validate LLM signal against actual indicator values to prevent hallucinations.

        Args:
            signal: Signal from LLM
            features: Actual technical indicator values (LIVE DATA)
            market_snapshot: Actual market data (LIVE DATA)

        Returns:
            Tuple of (is_valid, error_message)
        """
        rsi = features['rsi']
        macd_hist = features['macd_hist']
        close = features['close']
        ema_50 = features['ema_50']
        ema_200 = features['ema_200']

        # Calculate basic indicator states from LIVE data
        is_oversold = rsi < 30
        is_overbought = rsi > 70
        macd_bullish = macd_hist > 0
        macd_bearish = macd_hist < 0
        price_above_ema50 = close > ema_50
        price_above_ema200 = close > ema_200

        # 1. Check for obvious contradictions
        if signal.signal == "long":
            # Long signal should have some bullish indicators
            bullish_count = sum([
                is_oversold,
                macd_bullish,
                price_above_ema50,
                price_above_ema200
            ])

            # If claiming long but all indicators bearish, reject
            if bullish_count == 0 and signal.confidence > 0.5:
                return False, f"Long signal with {signal.confidence:.2f} confidence but all indicators bearish (RSI={rsi:.1f}, MACD_hist={macd_hist:.4f}, price below EMAs)"

        elif signal.signal == "short":
            # Short signal should have some bearish indicators
            bearish_count = sum([
                is_overbought,
                macd_bearish,
                not price_above_ema50,
                not price_above_ema200
            ])

            # If claiming short but all indicators bullish, reject
            if bearish_count == 0 and signal.confidence > 0.5:
                return False, f"Short signal with {signal.confidence:.2f} confidence but all indicators bullish (RSI={rsi:.1f}, MACD_hist={macd_hist:.4f}, price above EMAs)"

        # 2. Validate confidence is reasonable (not overconfident with mixed signals)
        if signal.confidence > 0.9:
            # Very high confidence requires strong alignment
            if signal.signal == "long":
                if rsi > 60 or macd_bearish or not price_above_ema50:
                    return False, f"Overconfident long signal ({signal.confidence:.2f}) with mixed indicators"
            elif signal.signal == "short":
                if rsi < 40 or macd_bullish or price_above_ema50:
                    return False, f"Overconfident short signal ({signal.confidence:.2f}) with mixed indicators"

        # 3. Validate confidence range
        if signal.confidence < 0.0 or signal.confidence > 1.0:
            return False, f"Confidence {signal.confidence} outside valid range [0.0, 1.0]"

        # 4. Validate reasons are not empty
        if not signal.reasons or len(signal.reasons) == 0:
            return False, "Signal must include reasoning"

        return True, None

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

        max_retries = 2
        for attempt in range(max_retries):
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

                # CRITICAL: Validate against live indicator data to prevent hallucinations
                is_valid, error_msg = self._validate_signal(signal, features, market_snapshot)

                if not is_valid:
                    logger.warning(f"Signal validation failed for {symbol}: {error_msg}")

                    if attempt < max_retries - 1:
                        logger.info(f"Retrying analysis (attempt {attempt + 2}/{max_retries})")
                        prompt += f"\n\nPREVIOUS ATTEMPT FAILED: {error_msg}\nGenerate a signal that matches the actual indicator values provided."
                        continue
                    else:
                        logger.error(f"Failed to generate valid signal after {max_retries} attempts, using neutral")
                        return TechnicalSignal(
                            symbol=symbol,
                            signal="neutral",
                            confidence=0.3,
                            reasons=["Unable to generate valid signal matching indicators"]
                        )

                logger.info(f"Technical analysis for {symbol}: {signal.signal} (confidence: {signal.confidence:.2f})")
                return signal

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response for {symbol}: {e}")
                logger.error(f"Response content: {content}")

                if attempt < max_retries - 1:
                    prompt += f"\n\nPREVIOUS ATTEMPT RETURNED INVALID JSON: {e}\nReturn ONLY valid JSON matching the schema."
                    continue
                else:
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

        return TechnicalSignal(
            symbol=symbol,
            signal="neutral",
            confidence=0.0,
            reasons=["Max retries exceeded"]
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
