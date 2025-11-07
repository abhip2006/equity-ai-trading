"""
Trader Agent: Synthesizes analyst and researcher outputs into concrete trade plans.
"""

import json
import logging
import re
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from active_trader_llm.llm import get_llm_client, LLMMessage
from active_trader_llm.trader.trade_plan_validator import TradePlanValidator, ValidationConfig

logger = logging.getLogger(__name__)


class TradePlan(BaseModel):
    """Trade plan schema - Fully autonomous LLM decision"""
    action: Literal["open_long", "open_short", "close", "hold", "pass", "adjust_stop", "adjust_target", "adjust_both"]
    symbol: str
    direction: Literal["long", "short"] = "long"  # Derived from action
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_size_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    time_horizon: Literal["1d", "3d", "1w"] = "3d"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    invalidation_condition: str = ""
    rationale: str = ""  # Changed from List[str] to single string for chain-of-thought
    risk_reward_ratio: float = 0.0
    position_id: Optional[int] = None  # Link to existing position for adjustments


class TraderAgent:
    """
    Synthesizes multi-agent signals into concrete trade plans.

    Uses LLM reasoning to make fully dynamic trading decisions.
    """

    SYSTEM_PROMPT = """You are an active equity trader generating P/L through trading decisions.

YOUR MISSION: GROW CAPITAL through active trading (long or short). You are measured on P/L generation.

TRADING APPROACH:
- Take trades to generate returns
- Use proper position sizing, stops, and invalidation conditions
- Analyze price action, indicators, and market structure
- React quickly to new opportunities and changing market conditions

DECISION PROCESS (with chain-of-thought reasoning):
1. **EXISTING POSITIONS ANALYSIS** - Review each open position:
   - Current price vs entry, stop, and target
   - Evaluate if setup is still working or showing weakness
   - Decide: HOLD, CLOSE, or ADJUST (stops/targets)

2. **NEW OPPORTUNITIES ANALYSIS** - Evaluate new trade setups:
   - Identify high-probability setups from raw data
   - Assess risk/reward and conviction level
   - Determine if opportunity warrants action

3. **EXECUTION DECISION** - Final action with reasoning

POSITION MANAGEMENT ACTIONS:
- "hold" - Keep position unchanged (thesis intact, stops/targets appropriate)
- "close" - Close entire position (thesis invalidated or target reached)
- "adjust_stop" - Modify stop loss only (trail profits, reduce risk)
- "adjust_target" - Modify take profit only (extend target if momentum accelerating)
- "adjust_both" - Modify both stop and target (market conditions changed)

WHEN TO ADJUST vs HOLD vs CLOSE:
- **ADJUST**: Thesis still valid but risk/reward has evolved
  * Trail stops to lock in profits as price moves favorably
  * Widen stops if volatility increased but thesis intact
  * Extend targets if momentum accelerating beyond initial expectations
  * Tighten stops if market conditions deteriorating but position still valid
- **HOLD**: Thesis intact, current stops/targets remain appropriate
  * Position working as planned
  * No material change in risk/reward dynamics
- **CLOSE**: Thesis invalidated or objective met
  * Technical breakdown / invalidation condition triggered
  * Target reached
  * Better opportunity identified (capital reallocation)

OUTPUT FORMAT - Return valid JSON:
{
    "action": "open_long|open_short|close|hold|pass|adjust_stop|adjust_target|adjust_both",
    "symbol": "<TICKER>",
    "entry": 0.0,
    "stop_loss": 0.0,  // For adjustments: new stop level. For new positions: initial stop
    "take_profit": 0.0,  // For adjustments: new target level. For new positions: initial target
    "position_size_pct": 0.0-1.0,
    "time_horizon": "1d|3d|1w",
    "confidence": 0.0-1.0,
    "invalidation_condition": "Specific price/indicator condition that would invalidate this trade",
    "rationale": "CHAIN-OF-THOUGHT ANALYSIS:\\n\\nEXISTING POSITIONS (if any):\\n- Brief status of each position\\n- Adjustment rationale if modifying stops/targets\\n\\nNEW OPPORTUNITY:\\n- Price action and technical setup\\n- Risk/reward and confidence\\n\\nDECISION:\\n- Final action and reasoning"
}

RISK MANAGEMENT:
- Determine position sizes based on your analysis and performance
- Use stops based on your analysis
- Define clear invalidation conditions for each trade"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-nano",
        provider: str = "openai",
        base_url: Optional[str] = None,
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: bool = True
    ):
        """
        Initialize trader agent

        Args:
            api_key: API key for the LLM provider
            model: LLM model to use (gpt-4.1-nano recommended, also supports: gpt-4o, gpt-4-turbo, claude-3-5-sonnet-20241022)
            provider: LLM provider (openai, anthropic, local)
            base_url: Optional base URL for API endpoint
            validation_config: Trade plan validation configuration
            enable_validation: Whether to enable trade plan validation
        """
        self.client = get_llm_client(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=60.0
        )
        self.model = model
        self.provider = provider
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
        account_state: Optional[Dict] = None,
        all_open_positions: Optional[List[Dict]] = None
    ) -> str:
        """
        Build fully autonomous trading prompt with simplified data.
        Provides only current indicator values and daily close price history.

        Args:
            all_open_positions: List of ALL open positions for comparative analysis
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

Daily Close Prices (last 10): {[round(p, 2) for p in price_series[-10:]]}

Daily Indicators:
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
        if 'breadth' in analyst_outputs and analyst_outputs['breadth'] is not None:
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

        # Macro-economic data (RAW DATA ONLY)
        if 'macro' in analyst_outputs:
            macro = analyst_outputs['macro']
            prompt += "\n--- MACRO-ECONOMIC DATA (Raw Values) ---\n"

            # Volatility environment
            if macro.get('vix') is not None:
                prompt += f"VIX (S&P 500 Volatility): {macro.get('vix'):.2f}\n"
            if macro.get('vxn') is not None:
                prompt += f"VXN (Nasdaq Volatility): {macro.get('vxn'):.2f}\n"
            if macro.get('move_index') is not None:
                prompt += f"MOVE Index (Treasury Volatility): {macro.get('move_index'):.2f}\n"

            # Interest rates
            if macro.get('treasury_10y') is not None or macro.get('treasury_2y') is not None:
                prompt += "\n"
            if macro.get('treasury_10y') is not None:
                prompt += f"10-Year Treasury Yield: {macro.get('treasury_10y'):.2f}%\n"
            if macro.get('treasury_2y') is not None:
                prompt += f"2-Year Treasury Yield: {macro.get('treasury_2y'):.2f}%\n"
            if macro.get('yield_curve_spread') is not None:
                prompt += f"Yield Curve Spread (10Y-2Y): {macro.get('yield_curve_spread'):+.2f}%\n"

            # Commodities & currency
            if any(macro.get(k) is not None for k in ['gold_price', 'oil_price', 'dollar_index']):
                prompt += "\n"
            if macro.get('gold_price') is not None:
                prompt += f"Gold: ${macro.get('gold_price'):.2f}/oz\n"
            if macro.get('oil_price') is not None:
                prompt += f"Crude Oil: ${macro.get('oil_price'):.2f}/barrel\n"
            if macro.get('dollar_index') is not None:
                prompt += f"US Dollar Index (DXY): {macro.get('dollar_index'):.2f}\n"

            # NYSE breadth (if available)
            if macro.get('nyse_advancing') is not None and macro.get('nyse_declining') is not None:
                prompt += f"\nNYSE Advancing: {macro.get('nyse_advancing'):,}\n"
                prompt += f"NYSE Declining: {macro.get('nyse_declining'):,}\n"
                if macro.get('nyse_new_highs') is not None:
                    prompt += f"NYSE New Highs: {macro.get('nyse_new_highs'):,}\n"
                if macro.get('nyse_new_lows') is not None:
                    prompt += f"NYSE New Lows: {macro.get('nyse_new_lows'):,}\n"

        # Open positions (compact format)
        if all_open_positions and len(all_open_positions) > 0:
            prompt += f"\n--- OPEN POSITIONS ({len(all_open_positions)}) ---\n"
            for pos in all_open_positions:
                entry = pos.get('entry_price', 0)
                current = pos.get('current_price', 0)
                pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
                prompt += f"{pos.get('symbol')}: {pos.get('direction')} {pos.get('shares')} @ ${entry:.2f} → ${current:.2f} ({pnl_pct:+.2f}%) | Stop: ${pos.get('stop_loss', 0):.2f} Target: ${pos.get('take_profit', 0):.2f} | Days: {pos.get('days_held', 0)}\n"

            if existing_position:
                prompt += f"\nYou hold {symbol}. Evaluate if setup still working.\n"

        # Account state (compact)
        if account_state:
            prompt += f"\n--- ACCOUNT ---\nCash: ${account_state.get('cash', 0):,.0f} | Equity: ${account_state.get('equity', 0):,.0f} | Positions: {account_state.get('position_count', 0)} | Exposure: {account_state.get('exposure_pct', 0)*100:.1f}%\n"

        # Performance context (if available)
        if memory_context:
            prompt += f"""
--- RECENT PERFORMANCE ---
{memory_context}
"""

        prompt += f"""
--- YOUR TASK ---
Analyze {symbol} and make a trading decision. Generate P/L through active trading.

Return JSON with your action and chain-of-thought reasoning."""

        return prompt

    def _build_batch_decision_prompt(
        self,
        symbols: List[str],
        analyst_outputs: Dict,
        risk_params: Dict,
        account_state: Optional[Dict] = None
    ) -> str:
        """
        Build batch trading prompt for multiple stocks.

        Analyzes multiple stocks in a single API call to reduce costs.

        Args:
            symbols: List of stock symbols to analyze
            analyst_outputs: Dict of symbol -> analyst data
            risk_params: Risk parameters
            account_state: Current account cash, equity, exposure

        Returns:
            Prompt string for batch analysis
        """
        prompt = f"""=== BATCH TRADING DECISION FOR {len(symbols)} STOCKS ===

ALL DATA ORDERED: OLDEST → NEWEST

You are analyzing {len(symbols)} stocks in this batch. For each stock, decide whether to take a trade or pass.

"""

        # Add data for each symbol in batch
        for i, symbol in enumerate(symbols, 1):
            data = analyst_outputs.get(symbol, {})
            features = data.get('features', {})
            daily_indicators = features.get('daily_indicators', {})
            weekly_indicators = features.get('weekly_indicators', {})
            ohlcv = features.get('ohlcv', {})
            price_series = data.get('price_series', [])

            # Get current price and indicators
            current_price = ohlcv.get('close', 0.0)
            ema_5_daily = daily_indicators.get('EMA_5_daily', 0.0)
            ema_10_daily = daily_indicators.get('EMA_10_daily', 0.0)
            ema_20_daily = daily_indicators.get('EMA_20_daily', 0.0)
            sma_50_daily = daily_indicators.get('SMA_50_daily', 0.0)
            sma_200_daily = daily_indicators.get('SMA_200_daily', 0.0)
            ema_10_weekly = weekly_indicators.get('EMA_10week', 0.0)

            # Volume data
            current_volume = ohlcv.get('volume', 0)
            avg_volume = features.get('avg_volume', current_volume)
            volume_pct_diff = ((current_volume - avg_volume) / avg_volume * 100) if avg_volume > 0 else 0.0

            prompt += f"""
--- STOCK {i}/{len(symbols)}: {symbol} ---
Price: ${current_price:.2f}
Recent Prices (last 10 bars): {[round(p, 2) for p in price_series[-10:]]}

Daily MAs: 5EMA=${ema_5_daily:.2f}, 10EMA=${ema_10_daily:.2f}, 20EMA=${ema_20_daily:.2f}, 50SMA=${sma_50_daily:.2f}, 200SMA=${sma_200_daily:.2f}
Weekly: 10EMA=${ema_10_weekly:.2f}
Volume: {int(current_volume):,} (Avg: {int(avg_volume):,}, {volume_pct_diff:+.1f}%)

"""

        # Add market breadth (once, shared across all stocks)
        if symbols and symbols[0] in analyst_outputs:
            breadth = analyst_outputs[symbols[0]].get('breadth', {})
            if breadth:
                prompt += f"""
--- MARKET CONTEXT (Shared) ---
Stocks Above 200-day SMA: {breadth.get('pct_above_sma200_daily', 0)*100:.1f}%
New 52-Week Highs: {breadth.get('new_highs', 0)}
New 52-Week Lows: {breadth.get('new_lows', 0)}
"""

        # Account state
        if account_state:
            prompt += f"""
--- ACCOUNT STATE ---
Cash: ${account_state.get('cash', 0):,.2f}
Equity: ${account_state.get('equity', 0):,.2f}
Current Positions: {account_state.get('position_count', 0)}
Exposure: {account_state.get('exposure_pct', 0)*100:.1f}%
"""

        prompt += f"""
--- YOUR TASK ---
Analyze each of the {len(symbols)} stocks above and decide which ones (if any) present good trading opportunities.

For each stock, you can:
- OPEN_LONG: Buy the stock
- OPEN_SHORT: Short the stock
- PASS: No trade (not a good setup)

Return a JSON array with one object per stock. Each object must include:
- symbol: The stock ticker
- action: "open_long", "open_short", or "pass"
- entry: Entry price (0.0 if pass)
- stop_loss: Stop loss price (0.0 if pass)
- take_profit: Take profit target (0.0 if pass)
- position_size_pct: Position size as decimal 0.0-1.0 (0.0 if pass)
- time_horizon: "1d", "3d", or "1w"
- confidence: Your confidence 0.0-1.0
- invalidation_condition: What would invalidate this trade
- rationale: Brief reasoning (2-3 sentences max per stock)

Return ONLY the JSON array, no markdown or additional text.

Example response format:
[
  {{"symbol": "STOCK1", "action": "open_long", "entry": 150.0, "stop_loss": 145.0, "take_profit": 160.0, "position_size_pct": 0.03, "time_horizon": "3d", "confidence": 0.7, "invalidation_condition": "Break below $145", "rationale": "Strong momentum with volume confirmation. Above all key MAs."}},
  {{"symbol": "STOCK2", "action": "pass", "entry": 0.0, "stop_loss": 0.0, "take_profit": 0.0, "position_size_pct": 0.0, "time_horizon": "1d", "confidence": 0.0, "invalidation_condition": "", "rationale": "Weak setup, no clear edge."}},
  {{"symbol": "STOCK3", "action": "open_short", "entry": 200.0, "stop_loss": 205.0, "take_profit": 190.0, "position_size_pct": 0.02, "time_horizon": "1w", "confidence": 0.6, "invalidation_condition": "Break above $205", "rationale": "Bearish breakdown with high volume."}}
]
"""

        return prompt

    def decide(
        self,
        symbol: str,
        analyst_outputs: Dict,
        researcher_outputs: tuple,
        risk_params: Dict,
        memory_context: Optional[str] = None,
        existing_position: Optional[Dict] = None,
        account_state: Optional[Dict] = None,
        all_open_positions: Optional[List[Dict]] = None
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
            all_open_positions: List of ALL open positions for comparative analysis

        Returns:
            TradePlan with action (open_long/open_short/close/hold/pass) or None
        """
        prompt = self._build_decision_prompt(
            symbol, analyst_outputs, researcher_outputs,
            risk_params, memory_context, existing_position, account_state,
            all_open_positions
        )

        try:
            # Use unified LLM client
            messages = [
                LLMMessage(role="system", content=self.SYSTEM_PROMPT),
                LLMMessage(role="user", content=prompt)
            ]

            response = self.client.generate(
                messages=messages,
                temperature=0.6,
                max_tokens=900
            )

            content = response.content

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

            # Clean JSON: remove trailing commas before closing braces/brackets
            # This fixes common LLM JSON formatting errors
            content = re.sub(r',(\s*[}\]])', r'\1', content)

            plan_dict = json.loads(content)

            # Handle action field to derive direction
            action = plan_dict.get('action', 'pass')
            if action == 'open_long':
                plan_dict['direction'] = 'long'
            elif action == 'open_short':
                plan_dict['direction'] = 'short'
            elif action in ['close', 'hold', 'pass', 'adjust_stop', 'adjust_target', 'adjust_both']:
                # These actions don't require full trade plan validation
                # For adjustments, direction will be determined from existing position
                plan_dict.setdefault('direction', existing_position.get('direction', 'long') if existing_position else 'long')
                plan_dict.setdefault('entry', existing_position.get('entry_price', 0.0) if existing_position else 0.0)
                # stop_loss and take_profit should come from LLM for adjustments
                if action in ['adjust_stop', 'adjust_both']:
                    # Validate stop_loss is provided
                    if 'stop_loss' not in plan_dict or plan_dict['stop_loss'] == 0.0:
                        logger.error(f"adjust_stop action requires stop_loss value")
                        return None
                else:
                    plan_dict.setdefault('stop_loss', 0.0)

                if action in ['adjust_target', 'adjust_both']:
                    # Validate take_profit is provided
                    if 'take_profit' not in plan_dict or plan_dict['take_profit'] == 0.0:
                        logger.error(f"adjust_target action requires take_profit value")
                        return None
                else:
                    plan_dict.setdefault('take_profit', 0.0)

                plan_dict.setdefault('position_size_pct', 0.0)
                # Store position ID for adjustments
                if existing_position:
                    plan_dict['position_id'] = existing_position.get('id')

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
                    current_price = analyst_outputs.get('features', {}).get('ohlcv', {}).get('close', 0.0)

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
            elif plan.action in ['adjust_stop', 'adjust_target', 'adjust_both']:
                adj_details = []
                if plan.action in ['adjust_stop', 'adjust_both']:
                    adj_details.append(f"stop: ${plan.stop_loss:.2f}")
                if plan.action in ['adjust_target', 'adjust_both']:
                    adj_details.append(f"target: ${plan.take_profit:.2f}")
                logger.info(f"Position adjustment for {symbol}: {plan.action} ({', '.join(adj_details)})")
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

    def decide_batch(
        self,
        symbols: List[str],
        analyst_outputs: Dict,
        risk_params: Dict,
        account_state: Optional[Dict] = None
    ) -> List[TradePlan]:
        """
        Generate trade plans for multiple stocks in a single API call.

        This method reduces API costs by analyzing multiple stocks together.
        If batch processing fails, it will fallback to single-stock analysis.

        Args:
            symbols: List of stock symbols to analyze
            analyst_outputs: Dict of symbol -> analyst data
            risk_params: Risk parameters
            account_state: Current account cash, equity, exposure

        Returns:
            List of TradePlan objects (one per symbol that has a trade)
        """
        if not symbols:
            return []

        logger.info(f"Batch analyzing {len(symbols)} stocks in single API call...")

        prompt = self._build_batch_decision_prompt(
            symbols, analyst_outputs, risk_params, account_state
        )

        try:
            # Single API call for entire batch using unified client
            messages = [
                LLMMessage(role="system", content=self.SYSTEM_PROMPT),
                LLMMessage(role="user", content=prompt)
            ]

            response = self.client.generate(
                messages=messages,
                temperature=0.6,
                max_tokens=2000 + (len(symbols) * 200)  # Scale tokens with batch size
            )

            content = response.content

            # Handle markdown
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            # Clean JSON: remove trailing commas
            content = re.sub(r',(\s*[}\]])', r'\1', content)

            # Parse JSON array
            plans_data = json.loads(content)

            if not isinstance(plans_data, list):
                raise ValueError("Response is not a JSON array")

            # Convert to TradePlan objects
            trade_plans = []
            for plan_dict in plans_data:
                try:
                    # Skip "pass" actions
                    action = plan_dict.get('action', 'pass')
                    if action == 'pass':
                        symbol = plan_dict.get('symbol', 'UNKNOWN')
                        logger.info(f"Batch: {symbol} - PASS (no trade)")
                        continue

                    # Handle direction
                    if action == 'open_long':
                        plan_dict['direction'] = 'long'
                    elif action == 'open_short':
                        plan_dict['direction'] = 'short'
                    else:
                        logger.warning(f"Unknown action: {action}")
                        continue

                    plan = TradePlan(**plan_dict)

                    # Calculate risk/reward
                    if plan.direction == "long":
                        risk = plan.entry - plan.stop_loss
                        reward = plan.take_profit - plan.entry
                    else:
                        risk = plan.stop_loss - plan.entry
                        reward = plan.entry - plan.take_profit

                    plan.risk_reward_ratio = reward / risk if risk > 0 else 0.0

                    # Validate trade plan (only for new positions)
                    if self.enable_validation and self.validator:
                        current_price = analyst_outputs.get(plan.symbol, {}).get('features', {}).get('ohlcv', {}).get('close', 0.0)

                        is_valid, error_msg = self.validator.validate_trade_plan(
                            action=plan.direction,
                            entry=plan.entry,
                            stop_loss=plan.stop_loss,
                            take_profit=plan.take_profit,
                            position_pct=plan.position_size_pct * 100.0,
                            current_price=current_price,
                            symbol=plan.symbol
                        )

                        if not is_valid:
                            logger.warning(f"Batch: {plan.symbol} - Trade plan REJECTED: {error_msg}")
                            continue

                    trade_plans.append(plan)
                    logger.info(
                        f"Batch: {plan.symbol} - {plan.action} @ {plan.entry:.2f} "
                        f"(size: {plan.position_size_pct*100:.1f}%, R:R {plan.risk_reward_ratio:.2f})"
                    )

                except Exception as e:
                    symbol = plan_dict.get('symbol', 'UNKNOWN')
                    logger.error(f"Error processing plan for {symbol} in batch: {e}")
                    continue

            logger.info(f"Batch processing complete: {len(trade_plans)} trades from {len(symbols)} stocks")
            return trade_plans

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse batch response: {e}")
            logger.error(f"Content: {content[:500]}...")
            logger.warning("Batch processing failed - falling back to single-stock analysis")

            # Fallback to single-stock processing
            trade_plans = []
            for symbol in symbols:
                plan = self.decide(
                    symbol,
                    analyst_outputs.get(symbol, {}),
                    (None, None),
                    risk_params,
                    memory_context=None,
                    existing_position=None,
                    account_state=account_state
                )
                if plan and plan.action in ['open_long', 'open_short']:
                    trade_plans.append(plan)

            return trade_plans

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            logger.warning("Falling back to single-stock analysis")

            # Fallback to single-stock processing
            trade_plans = []
            for symbol in symbols:
                try:
                    plan = self.decide(
                        symbol,
                        analyst_outputs.get(symbol, {}),
                        (None, None),
                        risk_params,
                        memory_context=None,
                        existing_position=None,
                        account_state=account_state
                    )
                    if plan and plan.action in ['open_long', 'open_short']:
                        trade_plans.append(plan)
                except Exception as sym_e:
                    logger.error(f"Error processing {symbol}: {sym_e}")
                    continue

            return trade_plans


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

    trader = TraderAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        provider="openai",
        model="gpt-4.1-nano"
    )
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
