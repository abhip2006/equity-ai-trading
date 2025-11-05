"""
Macro Analyst: Analyzes macro-economic context using LLM.

Interprets raw macro data (volatility, yields, commodities, currency)
to determine market environment and provide macro context.
"""

import json
import logging
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from active_trader_llm.llm import get_llm_client, LLMMessage

from active_trader_llm.feature_engineering.models import MacroSnapshot

logger = logging.getLogger(__name__)


class MacroSignal(BaseModel):
    """Macro analyst output schema"""
    analyst: str = "Macro"
    market_environment: Literal["risk_on", "risk_off", "neutral", "transitioning"]
    tailwinds: List[str] = Field(description="Positive macro factors supporting risk assets")
    headwinds: List[str] = Field(description="Negative macro factors creating headwinds")
    key_observations: List[str] = Field(description="Key data points and observations")
    confidence: float = Field(..., ge=0.0, le=1.0)


class MacroAnalyst:
    """
    LLM-powered macro analysis agent.

    Analyzes raw macro-economic data to determine market environment.
    Uses OpenAI to interpret volatility, yields, commodities, and currency data.
    """

    SYSTEM_PROMPT = """You are MacroAnalyst, an expert in macro-economic analysis and market environment assessment.

Your task is to analyze raw macro-economic data and determine the current market environment.

CRITICAL: Return ONLY valid JSON matching this exact schema, NO prose or explanations:
{
    "analyst": "Macro",
    "market_environment": "risk_on|risk_off|neutral|transitioning",
    "tailwinds": ["positive factor 1", "positive factor 2"],
    "headwinds": ["negative factor 1", "negative factor 2"],
    "key_observations": ["observation 1", "observation 2", "observation 3"],
    "confidence": 0.0-1.0
}

Guidelines for analysis:
- You receive RAW numerical values only - interpret them yourself
- NO hardcoded thresholds - use professional judgment based on context
- Consider ALL data holistically, not individual metrics in isolation
- Identify both supportive (tailwinds) and challenging (headwinds) factors
- Be concise in observations (3-5 points max)
- Confidence reflects clarity of environment (not data completeness)

Market Environment Definitions:
- risk_on: Favorable conditions for equities (low vol, stable/rising yields, strong commodities)
- risk_off: Flight to safety (high vol, falling yields, weak commodities, strong dollar)
- neutral: Mixed signals, no clear directional bias
- transitioning: Environment in flux between regimes"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        provider: str = "openai"
    ):
        """
        Initialize Macro Analyst.

        Args:
            api_key: API key for the LLM provider
            model: LLM model to use (gpt-3.5-turbo, gpt-4o, claude-3-5-sonnet-20241022)
            provider: LLM provider (openai, anthropic, local)
        """
        self.client = get_llm_client(
            provider=provider,
            model=model,
            api_key=api_key
        )
        self.model = model
        self.provider = provider
        logger.info(f"MacroAnalyst initialized with {provider}/{model}")

    def _build_analysis_prompt(self, macro_snapshot: MacroSnapshot) -> str:
        """
        Build analysis prompt with raw macro data.

        Args:
            macro_snapshot: Raw macro-economic data

        Returns:
            Formatted prompt with 100% raw numerical values
        """
        prompt = f"""Analyze the current macro-economic environment using these RAW data points:

VOLATILITY ENVIRONMENT:"""

        # Volatility indices
        if macro_snapshot.vix is not None:
            prompt += f"\n- VIX (S&P 500 Volatility): {macro_snapshot.vix:.2f}"
        else:
            prompt += "\n- VIX: NOT AVAILABLE"

        if macro_snapshot.vxn is not None:
            prompt += f"\n- VXN (Nasdaq Volatility): {macro_snapshot.vxn:.2f}"
        else:
            prompt += "\n- VXN: NOT AVAILABLE"

        if macro_snapshot.move_index is not None:
            prompt += f"\n- MOVE Index (Treasury Volatility): {macro_snapshot.move_index:.2f}"
        else:
            prompt += "\n- MOVE Index: NOT AVAILABLE"

        prompt += "\n\nINTEREST RATE ENVIRONMENT:"

        # Treasury yields
        if macro_snapshot.treasury_10y is not None:
            prompt += f"\n- 10-Year Treasury Yield: {macro_snapshot.treasury_10y:.2f}%"
        else:
            prompt += "\n- 10-Year Treasury: NOT AVAILABLE"

        if macro_snapshot.treasury_2y is not None:
            prompt += f"\n- 2-Year Treasury Yield: {macro_snapshot.treasury_2y:.2f}%"
        else:
            prompt += "\n- 2-Year Treasury: NOT AVAILABLE"

        if macro_snapshot.treasury_30y is not None:
            prompt += f"\n- 30-Year Treasury Yield: {macro_snapshot.treasury_30y:.2f}%"
        else:
            prompt += "\n- 30-Year Treasury: NOT AVAILABLE"

        if macro_snapshot.yield_curve_spread is not None:
            prompt += f"\n- Yield Curve Spread (10Y-2Y): {macro_snapshot.yield_curve_spread:+.2f}%"
        else:
            prompt += "\n- Yield Curve Spread: NOT AVAILABLE"

        prompt += "\n\nCOMMODITIES:"

        # Commodities
        if macro_snapshot.gold_price is not None:
            prompt += f"\n- Gold: ${macro_snapshot.gold_price:.2f}/oz"
        else:
            prompt += "\n- Gold: NOT AVAILABLE"

        if macro_snapshot.oil_price is not None:
            prompt += f"\n- Crude Oil: ${macro_snapshot.oil_price:.2f}/barrel"
        else:
            prompt += "\n- Crude Oil: NOT AVAILABLE"

        prompt += "\n\nCURRENCY:"

        # Currency
        if macro_snapshot.dollar_index is not None:
            prompt += f"\n- US Dollar Index (DXY): {macro_snapshot.dollar_index:.2f}"
        else:
            prompt += "\n- US Dollar Index: NOT AVAILABLE"

        prompt += "\n\nNYSE MARKET BREADTH:"

        # NYSE breadth - advancing/declining
        if macro_snapshot.nyse_advancing is not None and macro_snapshot.nyse_declining is not None:
            prompt += f"\n- Advancing Issues: {macro_snapshot.nyse_advancing:,}"
            prompt += f"\n- Declining Issues: {macro_snapshot.nyse_declining:,}"
            if macro_snapshot.nyse_unchanged is not None:
                prompt += f"\n- Unchanged Issues: {macro_snapshot.nyse_unchanged:,}"
            if macro_snapshot.advance_decline_ratio is not None:
                prompt += f"\n- Advance/Decline Ratio: {macro_snapshot.advance_decline_ratio:.2f}"
        else:
            prompt += "\n- Advancing/Declining Issues: NOT AVAILABLE"

        # NYSE breadth - volume
        if macro_snapshot.nyse_advancing_volume is not None and macro_snapshot.nyse_declining_volume is not None:
            prompt += f"\n- Advancing Volume: {macro_snapshot.nyse_advancing_volume:,}"
            prompt += f"\n- Declining Volume: {macro_snapshot.nyse_declining_volume:,}"
            if macro_snapshot.up_volume_ratio is not None:
                prompt += f"\n- Up Volume Ratio: {macro_snapshot.up_volume_ratio:.2%}"
        else:
            prompt += "\n- Volume Breadth: NOT AVAILABLE"

        # NYSE breadth - new highs/lows
        if macro_snapshot.nyse_new_highs is not None and macro_snapshot.nyse_new_lows is not None:
            prompt += f"\n- 52-Week New Highs: {macro_snapshot.nyse_new_highs:,}"
            prompt += f"\n- 52-Week New Lows: {macro_snapshot.nyse_new_lows:,}"
        else:
            prompt += "\n- New Highs/Lows: NOT AVAILABLE"

        prompt += "\n\nBased on these RAW values, determine the macro environment and generate your analysis JSON:"

        return prompt

    def analyze(self, macro_snapshot: MacroSnapshot) -> Optional[MacroSignal]:
        """
        Analyze macro-economic environment using LLM.

        Args:
            macro_snapshot: Raw macro-economic data

        Returns:
            MacroSignal with environment assessment, or None if LLM fails
        """
        # Check if we have ANY data
        if all(field is None for field in [
            macro_snapshot.vix, macro_snapshot.vxn, macro_snapshot.move_index,
            macro_snapshot.treasury_10y, macro_snapshot.treasury_2y, macro_snapshot.treasury_30y,
            macro_snapshot.gold_price, macro_snapshot.oil_price, macro_snapshot.dollar_index
        ]):
            logger.error("MacroAnalyst: No macro data available - cannot analyze")
            return None

        prompt = self._build_analysis_prompt(macro_snapshot)

        try:
            # Use unified LLM client
            messages = [
                LLMMessage(role="system", content=self.SYSTEM_PROMPT),
                LLMMessage(role="user", content=prompt)
            ]

            response = self.client.generate(
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )

            # Extract JSON from response
            content = response.content

            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            signal_dict = json.loads(content)
            signal = MacroSignal(**signal_dict)

            logger.info(f"Macro environment: {signal.market_environment} (confidence: {signal.confidence:.2f})")
            logger.info(f"  Tailwinds: {', '.join(signal.tailwinds[:2])}")
            logger.info(f"  Headwinds: {', '.join(signal.headwinds[:2])}")

            return signal

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Response content: {content}")
            logger.error("Macro analysis LLM failed - refusing to fabricate neutral environment")
            return None
        except Exception as e:
            logger.error(f"Error in macro analysis: {e}")
            logger.error("Macro analysis LLM failed - refusing to fabricate neutral environment")
            return None


# Example usage
if __name__ == "__main__":
    import os
    from data_ingestion.macro_data_ingestor import MacroDataIngestor

    logging.basicConfig(level=logging.INFO)

    # Fetch real macro data
    ingestor = MacroDataIngestor()
    snapshot = ingestor.fetch_all()

    print("\n=== Testing MacroAnalyst ===")
    print(f"Fetched data timestamp: {snapshot.timestamp}")
    print(f"VIX: {snapshot.vix}, 10Y Yield: {snapshot.treasury_10y}%")

    # Analyze with LLM
    analyst = MacroAnalyst(
        api_key=os.getenv("OPENAI_API_KEY"),
        provider="openai",
        model="gpt-3.5-turbo"
    )
    signal = analyst.analyze(snapshot)

    if signal:
        print(f"\n=== Macro Signal ===")
        print(f"Environment: {signal.market_environment}")
        print(f"Confidence: {signal.confidence:.2f}")
        print(f"\nTailwinds:")
        for tw in signal.tailwinds:
            print(f"  + {tw}")
        print(f"\nHeadwinds:")
        for hw in signal.headwinds:
            print(f"  - {hw}")
        print(f"\nKey Observations:")
        for obs in signal.key_observations:
            print(f"  • {obs}")
    else:
        print("Failed to generate macro signal")
