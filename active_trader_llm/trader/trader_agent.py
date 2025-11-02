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

    SYSTEM_PROMPT = """You are an ACTIVE equity trader focused on generating capital gains through systematic trading.

Your PRIMARY GOAL: Grow capital through disciplined, active trading. You are paid to TAKE TRADES and generate returns, not sit in cash.

TRADING PHILOSOPHY:
- You are an ACTIVE trader - your job is to find and execute profitable opportunities
- Even in challenging markets, skilled traders find asymmetric setups
- Risk-off markets create opportunities (volatility = opportunity for nimble traders)
- Every day should have potential trades if you look hard enough
- Sitting in cash means missing growth - ACTIVELY SEEK OPPORTUNITIES

ANALYSIS APPROACH:
1. Review existing positions - decide HOLD or CLOSE based on invalidation conditions
2. Actively scan for new opportunities - analyze price action, indicators, market structure
3. Use chain-of-thought reasoning - explain your analysis step-by-step
4. Be opportunistic - look for edges and asymmetric risk/reward setups
5. Your success is measured by P&L generation, not by avoiding losses

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

RISK MANAGEMENT (for executing trades):
- Position size: Typically 3-7% per trade, up to 10% for high-conviction setups
- Stop loss: Use ATR-based stops (1.5-2.5x ATR) or key support/resistance
- Take profit: Aim for 1.5:1 to 3:1 risk/reward based on market structure
- Invalidation: Define clear conditions where your thesis is wrong
- Portfolio: Target 60-80% total exposure - use your capital to generate returns
- Diversification: Multiple positions (3-6) across different setups reduces single-trade risk

ACTIVE TRADING MINDSET:
- Your edge comes from finding setups others miss
- Volatility and uncertainty create the best opportunities
- Risk management enables you to TAKE MORE TRADES, not fewer
- Capital sitting in cash earns 0% - put it to work strategically"""

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
        Build fully autonomous trading prompt with simplified data.
        Provides only current indicator values and daily close price history.
        """

        # Extract raw feature data (passed separately from technical signal)
        features = analyst_outputs.get('features', {})
        tech = analyst_outputs.get('technical', {})
        price_series = analyst_outputs.get('price_series', [])

        # Get feature components
        daily_indicators = features.get('daily_indicators', {})
        weekly_indicators = features.get('weekly_indicators', {})
        ohlcv = features.get('ohlcv', {})

        # Get current price
        current_price = ohlcv.get('close', tech.get('price', 0.0))

        # Extract current indicator values from daily_indicators dict
        ema_5_daily = daily_indicators.get('EMA_5_daily', 0.0)
        ema_10_daily = daily_indicators.get('EMA_10_daily', 0.0)
        ema_20_daily = daily_indicators.get('EMA_20_daily', 0.0)
        sma_50_daily = daily_indicators.get('SMA_50_daily', 0.0)
        sma_200_daily = daily_indicators.get('SMA_200_daily', 0.0)

        # Extract current indicator values from weekly_indicators dict
        ema_10_weekly = weekly_indicators.get('EMA_10week', 0.0)
        sma_21_weekly = weekly_indicators.get('SMA_21week', 0.0)
        sma_30_weekly = weekly_indicators.get('SMA_30week', 0.0)

        # Volume data
        current_volume = ohlcv.get('volume', 0)
        avg_volume = features.get('avg_volume', current_volume)  # 20-day average
        volume_pct_diff = ((current_volume - avg_volume) / avg_volume * 100) if avg_volume > 0 else 0.0

        prompt = f"""=== TRADING DECISION FOR {symbol} ===

ALL DATA ORDERED: OLDEST → NEWEST

--- RAW TECHNICAL DATA ---
Current Price: ${current_price:.2f}

Daily Close Price History (last 20 bars): {[round(p, 2) for p in price_series[-20:]]}

Daily Indicators (current values):
- 5 EMA: ${ema_5_daily:.2f}
- 10 EMA: ${ema_10_daily:.2f}
- 20 EMA: ${ema_20_daily:.2f}
- 50 SMA: ${sma_50_daily:.2f}
- 200 SMA: ${sma_200_daily:.2f}

Weekly Indicators (current values):
- 10 EMA: ${ema_10_weekly:.2f}
- 21 SMA: ${sma_21_weekly:.2f}
- 30 SMA: ${sma_30_weekly:.2f}

Volume:
- Current Daily Volume: {int(current_volume):,}
- 20-Day Avg Volume: {int(avg_volume):,}
- % Difference from Avg: {volume_pct_diff:+.1f}%
"""

        # Market breadth context (RAW DATA ONLY)
        if 'breadth' in analyst_outputs:
            breadth = analyst_outputs['breadth']
            prompt += f"""
--- MARKET BREADTH DATA (Raw Counts) ---
Stocks Advancing (above 200-day SMA): {breadth.get('stocks_advancing', 0)}
Stocks Declining (below 200-day SMA): {breadth.get('stocks_declining', 0)}
Total Stocks in Universe: {breadth.get('total_stocks', 0)}

New 52-Week Highs: {breadth.get('new_highs', 0)}
New 52-Week Lows: {breadth.get('new_lows', 0)}

Up Volume (stocks closing higher): {breadth.get('up_volume', 0):,}
Down Volume (stocks closing lower): {breadth.get('down_volume', 0):,}
Total Market Volume: {breadth.get('total_volume', 0):,}

Average RSI Across Universe: {breadth.get('avg_rsi', 0):.1f}
% of Stocks Above 200-day SMA: {breadth.get('pct_above_sma200_daily', 0)*100:.1f}%
% of Stocks Above 50-week SMA: {breadth.get('pct_above_sma50_weekly', 0)*100:.1f}%
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
Analyze the raw data above and make an ACTIVE trading decision.

REMEMBER: Your goal is to GROW CAPITAL through active trading. You are compensated for P&L generation, not for sitting in cash.

1. If there's an EXISTING POSITION: Decide HOLD or CLOSE based on invalidation condition
2. If NO POSITION: ACTIVELY SEEK opportunities to OPEN NEW positions (long/short)
3. Use CHAIN-OF-THOUGHT reasoning: Explain your analysis step-by-step
4. Be opportunistic: Look for asymmetric risk/reward setups where potential gain exceeds risk
5. Consider: Even in risk-off markets, skilled traders find profitable setups (mean reversion, oversold bounces, short squeezes, etc.)

BIAS TOWARD ACTION: If you see a reasonable setup with 1.5:1+ R/R, TAKE IT. Passing should be rare.

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
