# New Fully Autonomous Trading Prompt - Example

## SYSTEM PROMPT (sent once per conversation):
```
You are an autonomous equity trader making real-time trading decisions based on technical data.

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

Be systematic, disciplined, and adaptive to changing market conditions.
```

---

## USER PROMPT (sent for each symbol):

```
=== TRADING DECISION FOR AAPL ===

ALL DATA ORDERED: OLDEST → NEWEST

--- RAW TECHNICAL DATA ---
Current: price=175.50, ema5=174.20, ema10=173.80, ema20=172.50, rsi=42.3, atr=3.20

Price series (last 10 bars): [170.5, 171.2, 172.0, 171.5, 172.8, 173.2, 174.0, 174.5, 175.0, 175.5]
EMA5 series: [171.8, 172.1, 172.5, 172.8, 173.2, 173.5, 173.8, 174.0, 174.1, 174.2]
EMA10 series: [171.2, 171.5, 171.9, 172.3, 172.7, 173.0, 173.3, 173.5, 173.7, 173.8]
EMA20 series: [170.5, 170.8, 171.2, 171.5, 171.8, 172.0, 172.2, 172.3, 172.4, 172.5]
RSI series (14-period): [28.5, 32.1, 35.6, 38.2, 39.5, 40.1, 41.2, 41.5, 42.0, 42.3]
ATR series (14-period): [3.5, 3.4, 3.3, 3.3, 3.2, 3.2, 3.1, 3.2, 3.2, 3.2]
Volume series: [2500000, 2800000, 3200000, 2900000, 2650000, 3100000, 3300000, 3000000, 2900000, 2700000]

--- MARKET BREADTH CONTEXT ---
Regime: trending_bull
Breadth Score: 0.52
New Highs: 145
New Lows: 23

--- ACCOUNT STATE ---
Available Cash: $50,000.00
Total Equity: $100,000.00
Current Positions: 2
Total Exposure: $30,000.00 (30.0%)

--- RISK PARAMETERS ---
Max Position Size: 5.0%
Max Total Exposure: 80%
Max Concurrent Positions: 8

--- RECENT PERFORMANCE ---
Last 20 trades: Win Rate 55.0% (11W/9L)
Total P&L: $1,250.00
Avg per trade: $62.50

--- YOUR TASK ---
Analyze the raw data above and make a trading decision.

1. If there's an EXISTING POSITION: Decide HOLD or CLOSE based on invalidation condition
2. If NO POSITION: Decide whether to OPEN NEW (long/short) or PASS
3. Use CHAIN-OF-THOUGHT reasoning: Explain your analysis step-by-step
4. Be disciplined: Not every setup warrants a trade

Return JSON with your decision and full reasoning.
```

---

## Example LLM Response (New Format):

```json
{
  "action": "open_long",
  "symbol": "AAPL",
  "entry": 176.00,
  "stop_loss": 170.60,
  "take_profit": 186.80,
  "position_size_pct": 0.04,
  "time_horizon": "3d",
  "confidence": 0.72,
  "invalidation_condition": "Price closes below EMA20 at $172.50 or RSI drops back below 35, indicating failed breakout attempt",
  "rationale": "CHAIN-OF-THOUGHT ANALYSIS:\n\nStep 1 - Price Action Review:\nPrice series shows clear uptrend: 170.5 → 175.5 over last 10 bars. This is +2.9% move with higher lows (171.5 was only pullback). Current price 175.50 is at session high.\n\nStep 2 - Moving Average Structure:\n• Price > EMA5 (175.50 > 174.20) ✓\n• Price > EMA10 (175.50 > 173.80) ✓  \n• Price > EMA20 (175.50 > 172.50) ✓\n• EMA5 > EMA10 > EMA20 = Classic bullish alignment\n\nAll EMAs are rising (look at series progression). This is textbook bullish trend structure.\n\nStep 3 - Momentum Analysis:\nRSI recovered from oversold 28.5 → 42.3. Key observations:\n• RSI was deeply oversold (< 30), now recovering\n• Currently at 42.3 = neutral zone, not overbought\n• Steady climb in RSI series indicates building momentum\n• Room to run before hitting overbought (70+)\n\nStep 4 - Volatility & Risk:\nATR stable at 3.20 (last 5 bars: 3.2, 3.1, 3.2, 3.2, 3.2). Low volatility suggests controlled move, not panic buying.\n\nStep 5 - Volume Confirmation:\nVolume spiked to 3,300,000 during the breakout (6th bar), then normalized. Initial surge confirms institutional participation. Current lighter volume (2,700,000) is healthy consolidation.\n\nStep 6 - Market Context:\nBreadth regime: trending_bull (0.52 score)\nNew Highs: 145 vs New Lows: 23 = Strong market\nThis macro backdrop supports upside moves in quality names.\n\nStep 7 - Entry Strategy:\nEntry: $176.00 (slight buffer above current $175.50)\n• Waits for breakout confirmation above $175.50\n• Not chasing; requires follow-through\n\nStop: $170.60 (2.0x ATR = $6.40 below entry)\n• Just below recent consolidation low at $171.50\n• Gives trade breathing room\n• Clear invalidation if broken\n\nTarget: $186.80 (next resistance zone)\n• Risk: $5.40 (176.00 - 170.60)\n• Reward: $10.80 (186.80 - 176.00)\n• R/R: 2.0:1 ✓\n\nStep 8 - Position Sizing:\nAccount equity: $100,000\nCurrent exposure: $30,000 (30%)\nProposed position: 4% = $4,000\nNew total exposure: 34% (well below 80% limit)\n\nSize rationale:\n• 4% is conservative for 0.72 confidence\n• Approaching prior resistance, not full conviction\n• Leaves room for additional positions\n\nStep 9 - Risk Assessment:\nInvalidation condition: Price closes below EMA20 ($172.50) OR RSI drops below 35\nWhy? This would indicate:\n• Failed breakout attempt\n• Loss of bullish MA structure\n• Momentum fading\n\nConfidence: 0.72\nWhy not higher?\n• No weekly/monthly context provided\n• Unknown distance to major resistance\n• Mid-trend entry (not ideal base breakout)\n\nWhy not lower?\n• All technical factors align bullishly\n• Market breadth supportive\n• Clean price structure\n\nDECISION: OPEN_LONG at $176.00 with 4% allocation\nExpected hold: 3 days (time_horizon: 3d)\nMonitor: Daily close relative to EMA20 and RSI momentum"
}
```

---

## Key Differences vs Old Prompt:

### Old Prompt (Pre-interpreted):
- ❌ "Technical Analyst says: LONG (confidence 0.75)"
- ❌ "Reasons: RSI recovering, MACD cross, Above EMA50"
- ❌ "Bull Thesis: Momentum resuming, Support held"

### New Prompt (Raw Data):
- ✅ Shows actual price sequence: [170.5, 171.2, 172.0...]
- ✅ Shows EMA sequences: [171.8, 172.1, 172.5...]
- ✅ Shows RSI progression: [28.5, 32.1, 35.6...]
- ✅ LLM forms own interpretation
- ✅ LLM generates full chain-of-thought reasoning
- ✅ LLM defines invalidation conditions

---

## What Makes This "Fully Autonomous"?

1. **Zero Pre-interpretation** - LLM sees raw numbers, not labels like "oversold" or "breakout"
2. **Pattern Discovery** - LLM identifies trends in time series (rising EMAs, recovering RSI)
3. **Confidence Scoring** - LLM assigns own confidence based on analysis
4. **Invalidation Rules** - LLM defines specific conditions where thesis is wrong
5. **Full Reasoning** - LLM explains entire thought process step-by-step
6. **Position Management** - Can decide to hold/close existing positions
7. **Portfolio Awareness** - Sees account state, can size appropriately

This matches the **nof1.ai competition format** where the LLM is truly autonomous!
