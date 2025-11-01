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
    Analyzes macro economic context (optional, stub for now).

    Can be extended with:
    - Interest rates (Fed funds, 10Y yields)
    - VIX and volatility
    - Currency strength (DXY)
    - Commodity prices (oil, gold)
    """

    def __init__(self, vix_risk_off: float = 25.0, vix_risk_on: float = 12.0):
        """
        Initialize macro analyst.

        Args:
            vix_risk_off: VIX threshold for risk-off bias (default 25.0)
            vix_risk_on: VIX threshold for risk-on bias (default 12.0)
        """
        self.vix_risk_off = vix_risk_off
        self.vix_risk_on = vix_risk_on
        logger.info(f"MacroAnalyst initialized (VIX thresholds: risk_off>{vix_risk_off}, risk_on<{vix_risk_on})")

    def analyze(self, macro_data: Optional[Dict] = None) -> MacroSignal:
        """
        Analyze macro economic context.

        Args:
            macro_data: Optional pre-fetched macro indicators
                Required keys: 'vix' (float, >0)
                Optional keys: 'treasury_10y', 'dxy', 'oil_price'

        Returns:
            MacroSignal with market bias

        Raises:
            ValueError: If macro_data is missing or invalid (STUB MODE - REQUIRES REAL DATA)
        """
        # CRITICAL: This is a STUB implementation that requires real market data
        # In production, integrate:
        # - VIX from yfinance (^VIX)
        # - Treasury yields from FRED API
        # - Currency/commodity data
        # - Fed policy signals

        if not macro_data:
            logger.error("❌ MACRO ANALYST STUB: No macro_data provided. This analyst REQUIRES live data integration.")
            logger.error("   To fix: Fetch VIX from yfinance and pass via macro_data={'vix': <value>}")
            raise ValueError("MacroAnalyst requires live macro_data. Cannot operate in stub mode without data.")

        vix = macro_data.get('vix')

        # Validate VIX is present and reasonable
        if vix is None:
            logger.error("❌ MACRO ANALYST: 'vix' key missing from macro_data")
            raise ValueError("macro_data must include 'vix' key with live VIX value")

        if not isinstance(vix, (int, float)) or vix <= 0 or vix > 100:
            logger.error(f"❌ MACRO ANALYST: Invalid VIX value {vix} (must be 0-100)")
            raise ValueError(f"Invalid VIX value: {vix}. Must be positive float between 0-100")

        # LIVE DATA VALIDATION: Check if VIX looks like default/fake data
        if vix == 15.0:
            logger.warning("⚠️  MACRO ANALYST: VIX=15.0 exactly - this may be default/fake data")

        # Classify based on LIVE VIX data using configurable thresholds
        if vix > self.vix_risk_off:
            bias = "risk_off"
            context = [f"VIX elevated at {vix:.1f} (>{self.vix_risk_off} threshold)", "Market stress signals"]
            confidence = min(0.5 + (vix - self.vix_risk_off) / 50, 0.9)  # Higher confidence for higher VIX
        elif vix < self.vix_risk_on:
            bias = "risk_on"
            context = [f"VIX low at {vix:.1f} (<{self.vix_risk_on} threshold)", "Low volatility environment"]
            confidence = min(0.5 + (self.vix_risk_on - vix) / self.vix_risk_on, 0.9)
        else:
            bias = "neutral"
            context = [f"VIX normal at {vix:.1f} ({self.vix_risk_on}-{self.vix_risk_off} range)", "Balanced volatility conditions"]
            confidence = 0.6

        logger.info(f"Macro analysis: {bias} bias (VIX: {vix:.1f}, confidence: {confidence:.2f})")

        return MacroSignal(
            market_context=context,
            bias=bias,
            confidence=confidence
        )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    analyst = MacroAnalyst()
    signal = analyst.analyze({'vix': 18.5})

    print(f"\nBias: {signal.bias}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Context: {signal.market_context}")
