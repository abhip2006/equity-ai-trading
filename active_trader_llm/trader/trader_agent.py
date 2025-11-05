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
    action: Literal["open_long", "open_short", "close", "hold", "pass"]
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
- Actively manage capital allocation between existing and new opportunities

DECISION PROCESS (with chain-of-thought reasoning):
1. **EXISTING POSITIONS ANALYSIS** - Review each open position systematically:
   - Check current price vs entry, stop, and target
   - Evaluate if original thesis remains valid
   - Compare performance vs initial expectations
   - Decide: HOLD (thesis intact) or CLOSE (thesis invalidated)

2. **NEW OPPORTUNITIES ANALYSIS** - Evaluate new trade setups:
   - Identify high-probability setups from raw data
   - Assess risk/reward and conviction level
   - Compare strength vs existing holdings

3. **CAPITAL ALLOCATION DECISION** - When cash is insufficient for new opportunity:
   - Compare new opportunity strength vs existing positions
   - Consider closing weakest positions to reallocate capital
   - Only close if new opportunity is significantly stronger
   - Document comparative reasoning

4. **EXECUTION DECISION** - Final action with full reasoning

CRITICAL POSITION MANAGEMENT RULES:

When reviewing EXISTING positions:

DO NOT CLOSE positions due to:
- Minor price fluctuations or small unrealized losses (<5%)
- Price slightly below entry (normal intraday noise)
- Temporary consolidation or sideways movement
- Short-term weakness without technical breakdown

ONLY CLOSE positions when TRUE INVALIDATION occurs:
- Clear break of major support/resistance levels (not just entry price)
- Complete failure of the original thesis (e.g., failed breakout, trend reversal confirmed)
- Stop-loss level has been breached or is about to be breached
- Significant adverse technical development (e.g., death cross, breakdown below key moving averages)

CAPITAL REALLOCATION: When comparing new opportunity vs existing positions:
- Only close existing position if new setup is meaningfully stronger (higher confidence, better R/R)
- Consider days held and momentum of existing positions
- Avoid churning - transaction costs matter
- Document WHY new opportunity justifies closing existing position

Remember: Mechanical stops and targets are in place for a reason. Let them do their job.
Your role is to identify THESIS-LEVEL invalidations and strategic capital reallocation opportunities.

OUTPUT FORMAT - Return valid JSON:
{
    "action": "open_long|open_short|close|hold|pass",
    "symbol": "<TICKER>",
    "entry": 0.0,
    "stop_loss": 0.0,
    "take_profit": 0.0,
    "position_size_pct": 0.0-1.0,
    "time_horizon": "1d|3d|1w",
    "confidence": 0.0-1.0,
    "invalidation_condition": "Specific price/indicator condition that would invalidate this trade",
    "rationale": "CHAIN-OF-THOUGHT ANALYSIS with multiple steps separated by \\n\\n. REQUIRED STRUCTURE:\\n\\nEXISTING POSITIONS ANALYSIS:\\n- [For each open position] Symbol: Entry price, current price, P&L%, thesis status, decision (HOLD/CLOSE)\\n\\nNEW OPPORTUNITY ANALYSIS:\\n- Price Action: Current setup and key levels\\n- Moving Averages: Trend structure and alignment\\n- Momentum: RSI, volume, strength indicators\\n- Market Context: Breadth, sector rotation, macro\\n- Risk/Reward: Entry, stop, target, R:R ratio\\n- Conviction Level: Why this setup has edge\\n\\nCAPITAL ALLOCATION DECISION:\\n- Available cash vs required capital\\n- If insufficient: Compare new opportunity vs weakest existing position\\n- Reallocation decision: Close existing? Why/Why not?\\n\\nFINAL DECISION:\\n- Action and reasoning summary"
}

CRITICAL: The rationale field MUST contain detailed step-by-step analysis with the structure above. Include EXISTING POSITIONS ANALYSIS even if empty. Do NOT provide a single-line summary.

RISK MANAGEMENT:
- Determine position sizes based on your analysis and performance
- Use stops based on your analysis
- Define clear invalidation conditions for each trade
- Let mechanical stops handle downside protection
- Let mechanical targets handle profit-taking"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        provider: str = "openai",
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: bool = True
    ):
        """
        Initialize trader agent

        Args:
            api_key: API key for the LLM provider
            model: LLM model to use (gpt-4o, gpt-4-turbo, gpt-3.5-turbo, claude-3-5-sonnet-20241022)
            provider: LLM provider (openai, anthropic, local)
            validation_config: Trade plan validation configuration
            enable_validation: Whether to enable trade plan validation
        """
        self.client = get_llm_client(
            provider=provider,
            model=model,
            api_key=api_key,
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

        # ALL open positions context (for comparative analysis)
        if all_open_positions and len(all_open_positions) > 0:
            prompt += f"""
--- ALL CURRENT OPEN POSITIONS ({len(all_open_positions)} positions) ---
"""
            for pos in all_open_positions:
                pos_symbol = pos.get('symbol', 'UNKNOWN')
                entry_price = pos.get('entry_price', 0)
                current_pos_price = pos.get('current_price', 0)
                pnl = pos.get('unrealized_pnl', 0)
                pnl_pct = ((current_pos_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                days_held = pos.get('days_held', 0)

                prompt += f"""
{pos_symbol}:
  Direction: {pos.get('direction', 'long')}
  Entry: ${entry_price:.2f} → Current: ${current_pos_price:.2f} ({pnl_pct:+.2f}%)
  Unrealized P&L: ${pnl:+,.2f}
  Stop: ${pos.get('stop_loss', 0):.2f} | Target: ${pos.get('take_profit', 0):.2f}
  Shares: {pos.get('shares', 0)} | Days Held: {days_held}
  Original Rationale: {pos.get('original_rationale', 'Not available')[:200]}...
  Invalidation: {pos.get('invalidation_condition', 'Not specified')}
"""

            # Specific call-out if we're analyzing a position we already hold
            if existing_position:
                prompt += f"""
NOTE: You currently hold {symbol}. Evaluate if thesis remains valid or if invalidation has occurred.
"""
        elif existing_position:
            # Fallback to single position display
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

        # Performance context (if available)
        if memory_context:
            prompt += f"""
--- RECENT PERFORMANCE ---
{memory_context}
"""

        prompt += """
--- YOUR TASK ---
Analyze the raw data above and make a trading decision using systematic chain-of-thought reasoning.

Your goal: Generate P/L through active trading (long or short) while optimizing capital allocation.

ANALYSIS STEPS (include in rationale):

1. **EXISTING POSITIONS ANALYSIS** (if any open positions exist):
   - For each position: Check thesis validity, price action, and invalidation conditions
   - Decision for each: HOLD (thesis intact) or CLOSE (thesis failed)

2. **NEW OPPORTUNITY ANALYSIS** (for {symbol}):
   - Evaluate the setup: Price action, moving averages, momentum, volume
   - Determine: Is this a high-probability trade? What's the edge?
   - Risk/Reward: Entry, stop, target levels

3. **CAPITAL ALLOCATION DECISION**:
   - Available cash vs required capital for new position
   - If insufficient cash: Compare new opportunity strength vs existing positions
   - Should we close any existing positions to pursue this opportunity?
   - Only recommend closure if new setup is meaningfully stronger

4. **FINAL DECISION**:
   - Action: open_long, open_short, close, hold, or pass
   - Full reasoning: Why this decision maximizes expected value

CRITICAL: Use the structured rationale format specified in the OUTPUT FORMAT above.
Include all sections: EXISTING POSITIONS ANALYSIS, NEW OPPORTUNITY ANALYSIS, CAPITAL ALLOCATION DECISION, FINAL DECISION.

Return JSON with your decision and complete chain-of-thought reasoning."""

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
        model="gpt-3.5-turbo"
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
