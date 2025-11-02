"""
Trader Agent: Synthesizes analyst and researcher outputs into concrete trade plans.
"""

import json
import logging
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from active_trader_llm.trader.trade_plan_validator import TradePlanValidator, ValidationConfig

logger = logging.getLogger(__name__)


class TradePlan(BaseModel):
    """Trade plan schema - Fully autonomous LLM decision"""
    action: Literal["open_long", "open_short", "close", "hold", "pass"]
    symbol: str
    direction: Literal["long", "short"] = "long"  # Derived from action
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_size_pct: float = Field(default=0.0, ge=0.0, le=0.10)
    time_horizon: Literal["1d", "3d", "1w"] = "3d"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    invalidation_condition: str = ""
    rationale: str = ""  # Changed from List[str] to single string for chain-of-thought
    risk_reward_ratio: float = 0.0


class TraderAgent:
    """
    Synthesizes multi-agent signals into concrete trade plans.

    Uses LLM reasoning to make fully dynamic trading decisions.
    """

    SYSTEM_PROMPT = """You are an autonomous equity trader making real-time trading decisions based on technical data.

Your task: Analyze raw price/indicator data, assess existing positions, and decide on new trades.

ANALYSIS APPROACH:
1. Review existing positions - decide HOLD or CLOSE based on invalidation conditions
2. Scan for new opportunities - analyze price action, indicators, market structure
3. Use chain-of-thought reasoning - explain your analysis step-by-step
4. Make disciplined decisions - not every signal requires a trade

OUTPUT FORMAT - Return valid JSON:
{
    "action": "open_long|open_short|close|hold|pass",
    "symbol": "<TICKER>",
    "entry": 0.0,
    "stop_loss": 0.0,
    "take_profit": 0.0,
    "position_size_pct": 0.0-0.10,
    "time_horizon": "1d|3d|1w",
    "confidence": 0.0-1.0,
    "invalidation_condition": "Specific price/indicator condition that would invalidate this trade",
    "rationale": "Chain-of-thought explaining your full reasoning process"
}

RISK GUIDELINES:
- Position size: Typically 2-5% per trade, max 10% for high-conviction setups
- Stop loss: Use ATR-based stops (1.5-2.5x ATR) or key support/resistance
- Take profit: Aim for 1.5:1 to 3:1 risk/reward based on market structure
- Invalidation: Define clear conditions where your thesis is wrong
- Portfolio: Max 80% total exposure across all positions

Be systematic, disciplined, and adaptive to changing market conditions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: bool = True
    ):
        """
        Initialize trader agent

        Args:
            api_key: OpenAI API key
            model: LLM model to use (gpt-4o, gpt-4-turbo, gpt-3.5-turbo)
            validation_config: Trade plan validation configuration
            enable_validation: Whether to enable trade plan validation
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.enable_validation = enable_validation
        self.validator = TradePlanValidator(validation_config) if enable_validation else None

    def _build_decision_prompt(
        self,
        symbol: str,
        analyst_outputs: Dict,
        researcher_outputs: tuple,
        risk_params: Dict,
        memory_context: Optional[str] = None,
        existing_position: Optional[Dict] = None,
        account_state: Optional[Dict] = None
    ) -> str:
        """
        Build fully autonomous trading prompt (modeled after nof1.ai approach)
        Provides raw indicator time series instead of pre-interpreted signals
        """

        # Extract raw technical data
        tech = analyst_outputs.get('technical', {})
        features = tech.get('features', {})

        # Get time series data (last 10 bars if available)
        price_series = features.get('price_series', [tech.get('price', 0.0)])
        ema5_series = features.get('ema_5_series', [])
        ema10_series = features.get('ema_10_series', [])
        ema20_series = features.get('ema_20_series', [])
        rsi_series = features.get('rsi_series', [tech.get('rsi', 50.0)])
        atr_series = features.get('atr_series', [tech.get('atr', 1.0)])
        volume_series = features.get('volume_series', [])

        # Current values
        current_price = price_series[-1] if price_series else 0.0
        current_ema5 = ema5_series[-1] if ema5_series else 0.0
        current_ema10 = ema10_series[-1] if ema10_series else 0.0
        current_ema20 = ema20_series[-1] if ema20_series else 0.0
        current_rsi = rsi_series[-1] if rsi_series else 50.0
        current_atr = atr_series[-1] if atr_series else 1.0

        prompt = f"""=== TRADING DECISION FOR {symbol} ===

ALL DATA ORDERED: OLDEST → NEWEST

--- RAW TECHNICAL DATA ---
Current: price={current_price:.2f}, ema5={current_ema5:.2f}, ema10={current_ema10:.2f}, ema20={current_ema20:.2f}, rsi={current_rsi:.1f}, atr={current_atr:.2f}

Price series (last 10 bars): {[round(p, 2) for p in price_series[-10:]]}
EMA5 series: {[round(e, 2) for e in ema5_series[-10:]] if ema5_series else 'N/A'}
EMA10 series: {[round(e, 2) for e in ema10_series[-10:]] if ema10_series else 'N/A'}
EMA20 series: {[round(e, 2) for e in ema20_series[-10:]] if ema20_series else 'N/A'}
RSI series (14-period): {[round(r, 1) for r in rsi_series[-10:]]}
ATR series (14-period): {[round(a, 2) for a in atr_series[-10:]] if atr_series else 'N/A'}
Volume series: {[int(v) for v in volume_series[-10:]] if volume_series else 'N/A'}
"""

        # Market regime context
        if 'breadth' in analyst_outputs:
            breadth = analyst_outputs['breadth']
            prompt += f"""
--- MARKET BREADTH CONTEXT ---
Regime: {breadth.get('regime', 'unknown')}
Breadth Score: {breadth.get('breadth_score', 0):.2f}
New Highs: {breadth.get('new_highs', 0)}
New Lows: {breadth.get('new_lows', 0)}
"""

        # Existing position context (if any)
        if existing_position:
            prompt += f"""
--- EXISTING POSITION IN {symbol} ---
Direction: {existing_position.get('direction')}
Entry Price: ${existing_position.get('entry_price', 0):.2f}
Current Price: ${current_price:.2f}
Stop Loss: ${existing_position.get('stop_loss', 0):.2f}
Take Profit: ${existing_position.get('take_profit', 0):.2f}
Shares: {existing_position.get('shares', 0)}
Unrealized P&L: ${existing_position.get('unrealized_pnl', 0):.2f}
Invalidation Condition: {existing_position.get('invalidation_condition', 'Not specified')}

DECISION REQUIRED: Should you HOLD or CLOSE this position? Check if invalidation triggered.
"""

        # Account state
        if account_state:
            prompt += f"""
--- ACCOUNT STATE ---
Available Cash: ${account_state.get('cash', 0):,.2f}
Total Equity: ${account_state.get('equity', 0):,.2f}
Current Positions: {account_state.get('position_count', 0)}
Total Exposure: ${account_state.get('total_exposure', 0):,.2f} ({account_state.get('exposure_pct', 0)*100:.1f}%)
"""

        # Risk parameters
        prompt += f"""
--- RISK PARAMETERS ---
Max Position Size: {risk_params.get('max_position_pct', 0.05)*100:.1f}%
Max Total Exposure: 80%
Max Concurrent Positions: {risk_params.get('max_concurrent_positions', 8)}
"""

        # Performance context (if available)
        if memory_context:
            prompt += f"""
--- RECENT PERFORMANCE ---
{memory_context}
"""

        prompt += """
--- YOUR TASK ---
Analyze the raw data above and make a trading decision.

1. If there's an EXISTING POSITION: Decide HOLD or CLOSE based on invalidation condition
2. If NO POSITION: Decide whether to OPEN NEW (long/short) or PASS
3. Use CHAIN-OF-THOUGHT reasoning: Explain your analysis step-by-step
4. Be disciplined: Not every setup warrants a trade

Return JSON with your decision and full reasoning."""

        return prompt

    def decide(
        self,
        symbol: str,
        analyst_outputs: Dict,
        researcher_outputs: tuple,
        risk_params: Dict,
        memory_context: Optional[str] = None,
        existing_position: Optional[Dict] = None,
        account_state: Optional[Dict] = None
    ) -> Optional[TradePlan]:
        """
        Generate fully autonomous trade plan for a symbol.

        Args:
            symbol: Stock symbol
            analyst_outputs: Dict with raw technical data (not pre-interpreted)
            researcher_outputs: (BullThesis, BearThesis) tuple (kept for compatibility)
            risk_params: Risk parameters
            memory_context: Recent performance context
            existing_position: Current position in this symbol (if any)
            account_state: Current account cash, equity, exposure

        Returns:
            TradePlan with action (open_long/open_short/close/hold/pass) or None
        """
        prompt = self._build_decision_prompt(
            symbol, analyst_outputs, researcher_outputs,
            risk_params, memory_context, existing_position, account_state
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=600,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )

            content = response.choices[0].message.content.strip()

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

            # Handle action field to derive direction
            action = plan_dict.get('action', 'pass')
            if action == 'open_long':
                plan_dict['direction'] = 'long'
            elif action == 'open_short':
                plan_dict['direction'] = 'short'
            elif action in ['close', 'hold', 'pass']:
                # These actions don't require full trade plan validation
                plan_dict.setdefault('direction', 'long')  # Default value
                plan_dict.setdefault('entry', 0.0)
                plan_dict.setdefault('stop_loss', 0.0)
                plan_dict.setdefault('take_profit', 0.0)
                plan_dict.setdefault('position_size_pct', 0.0)

            plan = TradePlan(**plan_dict)

            # Calculate risk/reward only for open actions
            if plan.action in ['open_long', 'open_short']:
                if plan.direction == "long":
                    risk = plan.entry - plan.stop_loss
                    reward = plan.take_profit - plan.entry
                else:
                    risk = plan.stop_loss - plan.entry
                    reward = plan.entry - plan.take_profit

                plan.risk_reward_ratio = reward / risk if risk > 0 else 0.0
            else:
                plan.risk_reward_ratio = 0.0

            # Validate trade plan only for new positions
            if plan.action in ['open_long', 'open_short']:
                if self.enable_validation and self.validator:
                    current_price = analyst_outputs.get('technical', {}).get('price', 0.0)

                    is_valid, error_msg = self.validator.validate_trade_plan(
                        action=plan.direction,
                        entry=plan.entry,
                        stop_loss=plan.stop_loss,
                        take_profit=plan.take_profit,
                        position_pct=plan.position_size_pct * 100.0,  # Convert to percentage
                        current_price=current_price,
                        symbol=symbol
                    )

                    if not is_valid:
                        logger.error(f"Trade plan REJECTED for {symbol}: {error_msg}")
                        return None

                logger.info(
                    f"Trade plan for {symbol}: {plan.action} @ {plan.entry:.2f} "
                    f"(size: {plan.position_size_pct*100:.1f}%, R:R {plan.risk_reward_ratio:.2f}, confidence: {plan.confidence:.2f})"
                )
            else:
                logger.info(f"Trade decision for {symbol}: {plan.action}")

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

    risk_params = {
        "max_position_pct": 0.05,
        "max_concurrent_positions": 8
    }

    trader = TraderAgent(api_key=os.getenv("OPENAI_API_KEY"))
    plan = trader.decide("AAPL", sample_analyst, (bull, bear), risk_params)

    if plan:
        print(f"\nTrade Plan:")
        print(f"  Direction: {plan.direction}")
        print(f"  Entry: ${plan.entry:.2f}")
        print(f"  Stop: ${plan.stop_loss:.2f}")
        print(f"  Target: ${plan.take_profit:.2f}")
        print(f"  Size: {plan.position_size_pct*100:.1f}%")
        print(f"  R:R: {plan.risk_reward_ratio:.2f}")
    else:
        print("\nNo trade generated")
