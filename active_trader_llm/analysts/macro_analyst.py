"""
Macro Analyst: Analyzes macro economic context (optional module).
"""

import json
import logging
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MacroSignal(BaseModel):
    """Macro analyst output schema"""
    analyst: str = "Macro"
    market_context: List[str]
    bias: Literal["risk_on", "risk_off", "neutral"]
    confidence: float = Field(..., ge=0.0, le=1.0)


class MacroAnalyst:
    """
    Analyzes macro economic context (optional module).

    IMPORTANT: This is currently a stub. Do NOT enable macro analysis in production
    until a real data source is configured. This analyzer will refuse to return
    fabricated neutral bias to prevent misleading the trading system.

    To implement:
    - Integrate VIX from yfinance (^VIX)
    - Connect FRED API for treasury yields
    - Use currency/commodity data (DXY, oil, gold)
    - Monitor Fed policy signals
    """

    def __init__(self, enabled: bool = False):
        """
        Initialize macro analyst.

        Args:
            enabled: Must be explicitly set to True to enable (not recommended without real data source)
        """
        if enabled:
            logger.warning("MacroAnalyst enabled but no real data source configured!")
            logger.warning("This analyzer is a STUB and should NOT be used in production")
        logger.info("MacroAnalyst initialized (stub mode - will not return fabricated data)")

    def analyze(self, macro_data: Optional[Dict] = None) -> Optional[MacroSignal]:
        """
        Analyze macro economic context.

        Args:
            macro_data: Optional pre-fetched macro indicators (REQUIRED for real analysis)

        Returns:
            None - this stub will not fabricate macro context
        """
        # STUB: Refuse to return fabricated neutral bias
        # In production, integrate with real data sources:
        # - VIX from yfinance (^VIX)
        # - Treasury yields from FRED API
        # - Currency/commodity data (DXY, oil, gold)
        # - Fed policy signals

        # STUB: Refuse to operate without LLM-based analysis
        # This stub will NOT use hardcoded VIX thresholds
        logger.error("Macro analysis requested but MacroAnalyst is a stub")
        logger.error("This module requires LLM-based interpretation of macro data")
        logger.error("Please disable macro analysis or implement LLM reasoning")
        logger.error("TODO: Implement LLM-based macro analysis that interprets VIX, yields, etc.")

        # Return None instead of using hardcoded VIX thresholds
        return None


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    analyst = MacroAnalyst()
    signal = analyst.analyze({'vix': 18.5})

    print(f"\nBias: {signal.bias}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Context: {signal.market_context}")
