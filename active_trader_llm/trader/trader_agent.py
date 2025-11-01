"""
Trader Agent: Synthesizes analyst and researcher outputs into concrete trade plans.
"""

import json
import logging
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from anthropic import Anthropic

logger = logging.getLogger(__name__)


class TradePlan(BaseModel):
    """Trade plan schema"""
    symbol: str
    strategy: Literal["momentum_breakout", "mean_reversion", "pullback", "sector_rotation"]
    direction: Literal["long", "short"]
    entry: float
    stop_loss: float
    take_profit: float
    position_size_pct: float = Field(..., ge=0.0, le=0.10)
    time_horizon: Literal["1d", "3d", "1w"]
    rationale: List[str]
    risk_reward_ratio: float = 0.0


class TraderAgent:
    """
    Synthesizes multi-agent signals into trade plans.

    Selects appropriate strategy based on regime and recent performance.
    """

    SYSTEM_PROMPT = """You are TraderAgent, an expert at synthesizing analysis into concrete trade plans.

Your task is to combine analyst signals, researcher debates, strategy library, and risk parameters into executable trade plans.

CRITICAL: Return ONLY valid JSON matching this schema:
{
    "symbol": "<TICKER>",
    "strategy": "momentum_breakout|mean_reversion|pullback|sector_rotation",
    "direction": "long|short",
    "entry": 0.0,
    "stop_loss": 0.0,
    "take_profit": 0.0,
    "position_size_pct": 0.0-0.10,
    "time_horizon": "1d|3d|1w",
    "rationale": ["reason 1", "reason 2", "reason 3"]
}

Strategy Selection Guidelines:
- momentum_breakout: Use in trending_bull regime when price breaking above resistance with volume
- mean_reversion: Use in range regime when price at extremes (RSI <30 or >70)
- pullback: Use in mild_trend when price pulls back to EMA50 in uptrend
- sector_rotation: Use when breadth mixed, rotate to strongest sectors

Entry/Exit Sizing:
- Entry should be near current price with slight buffer
- Stop loss: Use ATR-based stops (1.5-2x ATR below entry for longs)
- Take profit: Aim for minimum 1.5:1 risk/reward, prefer 2:1+
- Position size: Respect max_position_pct from risk params (default 0.05)

Decision Making:
- Align strategy with current regime
- Higher conviction when analysts + researchers agree
- Reduce size or pass if conflicting signals
- Consider recent strategy performance from memory

Be disciplined and selective. Not every signal requires a trade."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        """Initialize trader agent"""
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def _build_decision_prompt(
        self,
        symbol: str,
        analyst_outputs: Dict,
        researcher_outputs: tuple,
        strategy_lib: List[Dict],
        risk_params: Dict,
        memory_context: Optional[str] = None
    ) -> str:
        """Build trading decision prompt"""
        bull, bear = researcher_outputs

        prompt = f"""Synthesize analysis into a trade plan for {symbol}.

ANALYST SIGNALS:
"""
        # Technical
        if 'technical' in analyst_outputs:
            tech = analyst_outputs['technical']
            prompt += f"""Technical: {tech.get('signal', 'neutral')} (confidence: {tech.get('confidence', 0):.2f})
  Reasons: {', '.join(tech.get('reasons', []))}
"""

        # Breadth
        if 'breadth' in analyst_outputs:
            breadth = analyst_outputs['breadth']
            prompt += f"""Breadth: Regime={breadth.get('regime', 'unknown')}, Score={breadth.get('breadth_score', 0):.2f}
"""

        prompt += f"""
RESEARCHER DEBATE:
Bull Thesis (confidence: {bull.confidence:.2f}):
  {chr(10).join(f"  - {point}" for point in bull.thesis)}

Bear Thesis (confidence: {bear.confidence:.2f}):
  {chr(10).join(f"  - {point}" for point in bear.thesis)}

STRATEGY LIBRARY:
"""
        for strat in strategy_lib:
            prompt += f"  - {strat['name']}: suitable for {strat['regime']} regime\n"

        prompt += f"""
RISK PARAMETERS:
  Max Position Size: {risk_params.get('max_position_pct', 0.05)*100:.1f}%
  Max Concurrent: {risk_params.get('max_concurrent_positions', 8)} positions

CURRENT PRICE DATA:
  Price: {analyst_outputs.get('technical', {}).get('price', 0.0):.2f}
  ATR: {analyst_outputs.get('technical', {}).get('atr', 1.0):.2f}
"""

        if memory_context:
            prompt += f"\nRECENT PERFORMANCE:\n{memory_context}\n"

        prompt += "\nGenerate trade plan JSON (or return null if no trade warranted):"

        return prompt

    def decide(
        self,
        symbol: str,
        analyst_outputs: Dict,
        researcher_outputs: tuple,
        strategy_lib: List[Dict],
        risk_params: Dict,
        memory_context: Optional[str] = None
    ) -> Optional[TradePlan]:
        """
        Generate trade plan for a symbol.

        Args:
            symbol: Stock symbol
            analyst_outputs: Dict of analyst results
            researcher_outputs: (BullThesis, BearThesis) tuple
            strategy_lib: Available strategies
            risk_params: Risk parameters
            memory_context: Recent strategy performance

        Returns:
            TradePlan or None if no trade
        """
        prompt = self._build_decision_prompt(
            symbol, analyst_outputs, researcher_outputs,
            strategy_lib, risk_params, memory_context
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=600,
                temperature=0.3,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.content[0].text.strip()

            # Handle markdown
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            # Check for null/no-trade
            if content.lower() in ['null', 'none', '{}']:
                logger.info(f"No trade recommended for {symbol}")
                return None

            plan_dict = json.loads(content)
            plan = TradePlan(**plan_dict)

            # Calculate risk/reward
            if plan.direction == "long":
                risk = plan.entry - plan.stop_loss
                reward = plan.take_profit - plan.entry
            else:
                risk = plan.stop_loss - plan.entry
                reward = plan.entry - plan.take_profit

            plan.risk_reward_ratio = reward / risk if risk > 0 else 0.0

            logger.info(f"Trade plan for {symbol}: {plan.direction} {plan.strategy} (R:R {plan.risk_reward_ratio:.2f})")

            return plan

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse trade plan for {symbol}: {e}")
            logger.error(f"Content: {content}")
            return None

        except Exception as e:
            logger.error(f"Error generating trade plan for {symbol}: {e}")
            return None


# Example usage
if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO)

    sample_analyst = {
        'technical': {
            'signal': 'long',
            'confidence': 0.75,
            'reasons': ['RSI recovering', 'MACD cross', 'Above EMA50'],
            'price': 175.0,
            'atr': 2.5
        },
        'breadth': {
            'regime': 'trending_bull',
            'breadth_score': 0.5
        }
    }

    from active_trader_llm.researchers.bull_bear import BullThesis, BearThesis

    bull = BullThesis(
        symbol="AAPL",
        thesis=["Momentum resuming", "Support held"],
        confidence=0.7
    )
    bear = BearThesis(
        symbol="AAPL",
        thesis=["Near resistance", "Could consolidate"],
        confidence=0.4
    )

    strategies = [
        {"name": "momentum_breakout", "regime": "trending_bull"},
        {"name": "mean_reversion", "regime": "range"}
    ]

    risk_params = {
        "max_position_pct": 0.05,
        "max_concurrent_positions": 8
    }

    trader = TraderAgent(api_key=os.getenv("ANTHROPIC_API_KEY"))
    plan = trader.decide("AAPL", sample_analyst, (bull, bear), strategies, risk_params)

    if plan:
        print(f"\nTrade Plan:")
        print(f"  Strategy: {plan.strategy}")
        print(f"  Direction: {plan.direction}")
        print(f"  Entry: ${plan.entry:.2f}")
        print(f"  Stop: ${plan.stop_loss:.2f}")
        print(f"  Target: ${plan.take_profit:.2f}")
        print(f"  Size: {plan.position_size_pct*100:.1f}%")
        print(f"  R:R: {plan.risk_reward_ratio:.2f}")
    else:
        print("\nNo trade generated")
