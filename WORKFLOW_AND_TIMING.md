# ActiveTrader-LLM Workflow and Timing Guide

## Overview

The ActiveTrader-LLM system is a multi-agent trading system that makes decisions at configurable intervals (default: **1 hour**). Each decision cycle involves multiple agents running in sequence to analyze markets, debate perspectives, and generate trade plans.

---

## Complete Decision Cycle Workflow

### Total Cycle Time: ~15-25 seconds per symbol
*(Depends on number of symbols, API latency, and market data complexity)*

---

## Phase 1: Data Ingestion & Feature Engineering
**Duration: 3-5 seconds per symbol**

### Step 1.1: Market Data Fetch (2-3 seconds)
```
Component: main.py → FeatureEngineer
Data Source: yfinance (free API)
```

**What happens:**
- Fetches OHLCV data for all symbols in universe
- Fetches VIX data (volatility index)
- Fetches advance/decline ratio data
- Downloads 200+ bars for indicator calculation

**Timing breakdown:**
- Single symbol: ~0.5-1 second
- 5 symbols: ~2-3 seconds (parallel requests)
- 10 symbols: ~3-5 seconds

**Configurable parameters:**
- `decision_interval_minutes`: 60 (how often to run)
- Minimum data bars needed: 200 (for EMA-200)

### Step 1.2: Technical Indicator Calculation (0.5-1 second)
```
Component: FeatureEngineer.build_features()
Processing: Pure Python/Pandas computation
```

**Indicators calculated (per symbol):**
- RSI (Relative Strength Index) - uses `rsi_period: 14`
- MACD (Moving Average Convergence Divergence) - uses `macd_fast: 12, macd_slow: 26, macd_signal: 9`
- Bollinger Bands - uses `bollinger_period: 20, bollinger_std: 2.0`
- ATR (Average True Range) - uses `atr_period: 14`
- EMAs (Exponential Moving Averages) - uses `ema_short: 50, ema_long: 200`
- SMA (Simple Moving Average) - uses `sma_period: 20`

**Timing:**
- Per symbol: ~50-100ms
- 10 symbols: ~0.5-1 second

### Step 1.3: Market Regime Classification (0.2-0.5 seconds)
```
Component: FeatureEngineer.determine_regime()
Processing: Statistical analysis
```

**Metrics calculated:**
- Breadth score (advance/decline ratio)
- Percentage of stocks above EMA-200
- Average RSI across universe
- Volatility (ATR) levels

**Regime categories:**
- `trending_bull` - breadth > 0.5, RSI > 55, VIX < 20
- `trending_bear` - breadth < -0.3, RSI < 45
- `range` - neutral breadth, low volatility
- `risk_off` - breadth < -0.5, VIX > 25

**Timing:** ~200-500ms total

---

## Phase 2: Analyst Analysis (LLM Agents)
**Duration: 4-8 seconds (3 analysts run in parallel)**

### Step 2.1: Technical Analyst
```
Component: TechnicalAnalyst.analyze()
LLM Model: claude-3-5-sonnet-20241022
API: Anthropic
```

**Input:**
- All calculated indicators (RSI, MACD, Bollinger, EMAs)
- Current price and volume
- Recent price action

**Process:**
1. Constructs prompt with indicator values (~500 tokens)
2. Sends to Claude API
3. Receives signal analysis (~300-500 tokens)
4. **Validates against live data** (hallucination check)
5. Retries once if validation fails

**Output:**
- Signal: "long", "short", or "neutral"
- Confidence: 0.0 to 1.0
- Reasoning: text explanation

**Timing:**
- LLM API call: 1.5-3 seconds
- Validation: ~50ms
- With retry (if needed): 3-5 seconds

**Validation checks:**
- Long signal must have bullish indicators (RSI < 70, MACD > 0, or price > EMA-50)
- Short signal must have bearish indicators
- Confidence must match indicator strength

### Step 2.2: Breadth & Health Analyst
```
Component: BreadthHealthAnalyst.analyze()
LLM Model: claude-3-5-sonnet-20241022
API: Anthropic
```

**Input:**
- Market breadth metrics
- VIX (volatility index)
- Advance/decline ratio
- Percentage above EMA-200

**Process:**
1. Analyzes overall market health
2. Classifies market regime
3. **Validates against thresholds**
4. Falls back to rule-based if LLM fails

**Output:**
- Regime: "trending_bull", "trending_bear", "range", "risk_off"
- Health score: 0.0 to 1.0
- Explanation

**Timing:**
- LLM API call: 1.5-3 seconds
- Validation: ~50ms
- Fallback (if needed): ~100ms (rule-based)

**Validation checks:**
- Bull regime requires breadth > 0.2
- Risk-off requires breadth < -0.3 OR VIX > 25

### Step 2.3: Macro Analyst (if enabled)
```
Component: MacroAnalyst.analyze()
No LLM - Rule-based logic
```

**Input:**
- VIX level (live data required)
- Configurable thresholds:
  - `vix_risk_off: 25.0`
  - `vix_risk_on: 12.0`

**Process:**
- Simple threshold-based classification
- Fails loudly if no live VIX data

**Output:**
- Bias: "risk_on", "neutral", "risk_off"
- VIX level

**Timing:** ~10ms (no API call)

**Note:** Sentiment Analyst disabled by default (requires external API)

---

## Phase 3: Researcher Debate
**Duration: 3-5 seconds**

### Step 3.1: Bull & Bear Researchers
```
Component: BullBearResearchers.debate()
LLM Model: claude-3-5-sonnet-20241022
API: Anthropic (2 separate calls)
```

**Input:**
- All analyst outputs (Technical, Breadth, Macro)
- Current market snapshot
- Symbol information

**Process:**
1. **Bull Researcher** argues for long position (1.5-2.5s)
2. **Bear Researcher** argues for short/neutral (1.5-2.5s)
3. Both run in parallel
4. **Validates confidence levels** against technical signals
5. Retries if overconfident without supporting data

**Output:**
- Bull argument + confidence (0.0-1.0)
- Bear argument + confidence (0.0-1.0)
- Net sentiment balance

**Timing:**
- 2 parallel LLM calls: 1.5-2.5 seconds each
- Total: ~2-3 seconds (parallel)
- Validation: ~50ms
- With retries: 4-5 seconds

**Validation checks:**
- If technical signal strongly bullish (confidence > 0.8), bear confidence capped at 0.8
- If technical signal strongly bearish, bull confidence capped at 0.8
- Prevents unrealistic overconfidence

---

## Phase 4: Trade Plan Generation
**Duration: 2-4 seconds**

### Step 4.1: Trader Agent Decision
```
Component: TraderAgent.decide()
LLM Model: claude-3-5-sonnet-20241022
API: Anthropic
```

**Input:**
- All analyst outputs (Technical, Breadth, Macro)
- Bull/Bear debate results
- Current positions and portfolio state
- Market regime
- Risk parameters

**Process:**
1. Synthesizes all inputs into comprehensive prompt (~1000 tokens)
2. Generates trade plan via Claude API
3. **Validates trade plan against live data** (critical hallucination check)
4. Retries once with error feedback if validation fails

**Output (TradePlan object):**
```python
{
    "action": "enter_long" | "enter_short" | "exit" | "hold",
    "symbol": "AAPL",
    "direction": "long" | "short",
    "entry": 175.50,      # Price level
    "stop_loss": 172.00,  # Stop price
    "take_profit": 182.00, # Target price
    "size_pct": 0.15,     # 15% of portfolio
    "reasoning": "..."
}
```

**Timing:**
- LLM API call: 2-3 seconds
- Validation: ~100ms
- With retry: 4-5 seconds

**Critical Validation Checks:**
1. **Entry price within 5% of current price** (prevents hallucination like entry=$200 when stock at $100)
2. **Stop loss on correct side** (long stop < entry, short stop > entry)
3. **Stop distance 0.5x-3.0x ATR** (prevents too-tight or too-loose stops)
4. **Risk/reward ratio 0.5x-5.0x** (prevents unrealistic targets)
5. **Position size ≤ max per position** (prevents over-concentration)

**Example validation:**
```
Current price: $175.50
ATR: $3.50

VALID trade plan:
- Entry: $175.80 (within 5% ✓)
- Stop: $172.00 (below entry ✓, 1.09x ATR ✓)
- Target: $181.00 (above entry ✓, R:R = 1.37 ✓)

INVALID trade plan (would be rejected):
- Entry: $180.00 (2.5% above current, too far ✗)
- Stop: $178.00 (only 0.57x ATR, too tight ✗)
```

---

## Phase 5: Risk Management
**Duration: 0.5-2 seconds**

### Step 5.1: Position Sizing
```
Component: RiskManager.evaluate_trade()
Processing: Pure Python calculations
```

**Checks performed:**
1. **Portfolio heat** (total risk across all positions)
   - Max: `max_portfolio_heat: 0.06` (6% total)
2. **Per-position size**
   - Max: `max_position_size: 0.20` (20% per symbol)
3. **Sector concentration** (uses live yfinance lookup)
   - Max: `max_sector_concentration: 0.40` (40% per sector)
4. **Correlation** (if enabled)
   - Max correlation between positions

**Timing:**
- Risk calculations: ~50-100ms
- Sector lookup (yfinance): ~500ms-1s per new symbol (cached after first lookup)
- Total: 0.5-2 seconds

**Critical Fix Applied:**
- ✅ Now uses **live yfinance sector lookup** instead of hardcoded "Technology"
- ✅ Includes ETF/index mapping for SPY, QQQ, XLE, etc.
- ✅ Sector diversification now works correctly

### Step 5.2: Trade Approval
```
Component: RiskManager.evaluate_trade()
Output: Approved/Rejected + adjustments
```

**Possible outcomes:**
1. **Approved** - Trade passes all checks
2. **Approved with reduced size** - Reduce to meet heat/concentration limits
3. **Rejected** - Violates hard limits (with reason)

**Timing:** ~50ms

---

## Phase 6: Strategy Monitoring
**Duration: 0.1-0.5 seconds**

### Step 6.1: Performance Tracking
```
Component: StrategyMonitor.record_result()
Processing: Database write + statistics
```

**What's tracked:**
- Win rate per strategy
- Average return per strategy
- Average risk/reward ratio
- Strategy performance by regime

**Timing:** ~100-200ms per trade result

### Step 6.2: Strategy Switching
```
Component: StrategyMonitor.select_best_strategy()
Processing: Scoring algorithm
```

**When triggered:**
- Every N trades: `performance_window: 20`
- If win rate drops below: `min_win_rate: 0.40` (40%)
- If drawdown exceeds: `max_drawdown: 0.15` (15%)

**Scoring formula (configurable):**
```python
score = (
    win_rate * 50.0 +              # win_rate_weight
    min(avg_return, 100) * 0.5 +   # avg_return_weight
    avg_risk_reward * 10.0         # avg_rr_weight
)
```

**Timing:** ~50-100ms

---

## Complete Timing Summary

### Single Symbol Decision Cycle

| Phase | Component | Duration | API Calls |
|-------|-----------|----------|-----------|
| 1.1 | Market Data Fetch | 0.5-1s | yfinance |
| 1.2 | Indicator Calculation | 50-100ms | - |
| 1.3 | Regime Classification | 200-500ms | - |
| 2.1 | Technical Analyst | 1.5-3s | Claude API |
| 2.2 | Breadth Analyst | 1.5-3s | Claude API |
| 2.3 | Macro Analyst | 10ms | - |
| 3.1 | Bull/Bear Debate | 2-3s | 2x Claude API (parallel) |
| 4.1 | Trader Agent | 2-4s | Claude API |
| 5.1 | Risk Management | 0.5-2s | yfinance (cached) |
| 6.1 | Strategy Monitoring | 100-200ms | - |
| **TOTAL** | **~9-15 seconds** | **5 LLM calls** |

### With Retries (worst case)
- Technical Analyst retry: +2s
- Breadth Analyst retry: +2s
- Debate retry: +3s
- Trader Agent retry: +3s
- **Worst case total: ~25 seconds**

### Multi-Symbol Portfolio (10 symbols)

**Parallel processing:**
- Data fetch: 3-5s (parallel yfinance requests)
- Feature engineering: 0.5-1s per symbol = 5-10s total
- Analysts run per symbol: 9-15s × 10 = **90-150 seconds**

**Optimization strategies:**
1. Process symbols in parallel (requires API rate limit management)
2. Shared breadth/macro analysis (only run once)
3. Batch indicator calculations

**Realistic multi-symbol timing:**
- 5 symbols: ~45-75 seconds (run 2-3 in parallel)
- 10 symbols: ~90-150 seconds (run 2-3 in parallel)

---

## Decision Interval Schedule

### Default Configuration
```yaml
decision_interval_minutes: 60
```

**Schedule example (9:30 AM market open):**
- 9:30 AM - Market opens
- 10:30 AM - First decision cycle (after 1 hour)
- 11:30 AM - Second decision cycle
- 12:30 PM - Third decision cycle
- 1:30 PM - Fourth decision cycle
- 2:30 PM - Fifth decision cycle
- 3:30 PM - Sixth decision cycle
- 4:00 PM - Market closes

**Total decisions per day:** 6 (with 60-minute interval)

### Alternative Intervals

**Scalping (15 minutes):**
```yaml
decision_interval_minutes: 15
```
- Decisions per day: ~24
- Requires faster execution
- Higher API costs (~120 LLM calls/day)

**Day Trading (30 minutes):**
```yaml
decision_interval_minutes: 30
```
- Decisions per day: ~12
- Balanced approach
- Moderate API costs (~60 LLM calls/day)

**Swing Trading (240 minutes / 4 hours):**
```yaml
decision_interval_minutes: 240
```
- Decisions per day: ~2
- Lower API costs (~10 LLM calls/day)
- Suitable for longer-term positions

---

## API Cost Estimation

### Claude API Costs
**Model: claude-3-5-sonnet-20241022**
- Input: $3 per million tokens
- Output: $15 per million tokens

**Per decision cycle (single symbol):**
- Technical Analyst: ~500 input + 400 output = ~$0.0075
- Breadth Analyst: ~400 input + 300 output = ~$0.0057
- Bull Researcher: ~600 input + 400 output = ~$0.0078
- Bear Researcher: ~600 input + 400 output = ~$0.0078
- Trader Agent: ~1000 input + 500 output = ~$0.0105
- **Total per cycle: ~$0.039 (4 cents)**

**Daily costs (60-minute interval, 6 cycles):**
- 1 symbol: $0.23/day = ~$7/month
- 5 symbols: $1.17/day = ~$35/month
- 10 symbols: $2.34/day = ~$70/month

**Daily costs (15-minute interval, 24 cycles):**
- 1 symbol: $0.94/day = ~$28/month
- 5 symbols: $4.68/day = ~$140/month

### Market Data Costs
**yfinance (free API):**
- $0/month
- Rate limits: ~2000 requests/hour
- Sufficient for small portfolios

---

## Performance Optimizations

### Current Optimizations ✅
1. **Parallel Bull/Bear debate** (saves 1.5-2s)
2. **Sector caching** (yfinance lookups cached)
3. **Rule-based fallbacks** (breadth analyst, macro analyst)
4. **Validation before LLM** (skip LLM if data invalid)

### Potential Future Optimizations
1. **Batch LLM calls** (analyze multiple symbols in one prompt)
2. **Prompt caching** (Anthropic feature, saves ~50% on repeated prompts)
3. **Indicator caching** (reuse calculations across intervals)
4. **Async processing** (run all symbols in parallel)
5. **Use Claude Haiku** for non-critical analyses (10x cheaper)

**Estimated speedup with all optimizations:**
- Current: 90-150s for 10 symbols
- Optimized: 30-50s for 10 symbols (3x faster)

---

## Safety Mechanisms & Timing

### Anti-Hallucination Validation
**Total validation time: ~300ms per cycle**

1. **Entry price check** (~50ms)
   - Compares LLM entry vs current price
   - Rejects if deviation > 5%

2. **Technical signal check** (~50ms)
   - Compares LLM signal vs actual RSI/MACD/EMA values
   - Rejects contradictions

3. **Regime validation** (~50ms)
   - Compares LLM regime vs breadth/VIX thresholds
   - Rejects inconsistencies

4. **Trade plan validation** (~100ms)
   - Checks stop/target placement
   - Validates ATR multiples
   - Checks risk/reward ratios

5. **Confidence validation** (~50ms)
   - Compares debate confidence vs technical signals
   - Prevents overconfidence

**Impact:** <2% overhead, prevents catastrophic errors

### Error Handling & Retries
**Retry timing:**
- First attempt fails validation: retry immediately
- Second attempt fails: abort and hold
- Max retry time: +3-4 seconds

**Failure modes:**
1. **Data fetch failure** → Skip cycle, log error
2. **LLM timeout** (>30s) → Skip cycle, log error
3. **Validation failure** → Retry once, then hold
4. **Risk rejection** → Log and hold (no retry)

---

## Real-World Execution Timeline

### Example: AAPL Decision at 10:30 AM

```
10:30:00.000 - Cycle starts
10:30:00.100 - Fetch AAPL data (yfinance)
10:30:01.200 - Fetch VIX data
10:30:01.400 - Fetch breadth data
10:30:01.650 - Calculate RSI, MACD, Bollinger, EMAs
10:30:01.900 - Determine market regime
10:30:02.000 - [START] Technical Analyst LLM call
10:30:04.200 - [END] Technical Analyst (signal: long, confidence: 0.75)
10:30:04.250 - Validate technical signal ✓
10:30:04.300 - [START] Breadth Analyst LLM call
10:30:06.500 - [END] Breadth Analyst (regime: trending_bull)
10:30:06.550 - Validate regime ✓
10:30:06.600 - [START] Bull/Bear debate (parallel)
10:30:09.100 - [END] Debate (bull: 0.72, bear: 0.45)
10:30:09.150 - Validate debate ✓
10:30:09.200 - [START] Trader Agent LLM call
10:30:11.800 - [END] Trader Agent (action: enter_long, entry: 175.80)
10:30:11.900 - Validate trade plan ✓
10:30:12.100 - Lookup AAPL sector (Technology) [cached]
10:30:12.200 - Calculate portfolio heat (4.2% < 6% ✓)
10:30:12.300 - Calculate sector exposure (35% < 40% ✓)
10:30:12.400 - APPROVED: Enter AAPL long, size: 15%
10:30:12.500 - Update strategy stats
10:30:12.600 - [READY] Send order to broker
10:30:12.700 - Cycle complete (12.7 seconds)
```

**Next cycle:** 11:30:00 AM (60 minutes later)

---

## Configuration Impact on Timing

### Faster Decision Intervals
```yaml
decision_interval_minutes: 15  # 4x more frequent
```
**Impact:**
- 4x more LLM API calls
- 4x higher costs
- More responsive to intraday moves
- Higher API rate limit requirements

### More Symbols
```yaml
symbols: [AAPL, MSFT, GOOGL, TSLA, NVDA, AMD, SPY, QQQ, META, AMZN]  # 10 symbols
```
**Impact:**
- 10x more processing time (if sequential)
- 3-4x more time (if 2-3 parallel)
- 10x higher costs
- Requires parallelization

### Longer Indicator Periods
```yaml
indicators:
  ema_long: 200  # Requires 200+ bars
  rsi_period: 21  # Slower RSI
```
**Impact:**
- Longer data history needed
- Slightly slower indicator calculation (+100-200ms)
- More stable signals (less noise)

---

## Monitoring & Logging

### Key Metrics to Track
1. **Cycle duration** (should be <15s for single symbol)
2. **LLM response time** (should be <3s per call)
3. **Validation failure rate** (should be <5%)
4. **Data fetch failures** (should be <1%)
5. **Risk rejections** (track reasons)

### Performance Alerts
- Cycle duration >30s: investigate API latency
- Validation failures >10%: check LLM prompts
- Data fetch failures >5%: check yfinance connectivity

---

## Summary

**Key Takeaways:**

1. **Single symbol decision:** 9-15 seconds (typical), up to 25 seconds (with retries)
2. **LLM calls per cycle:** 5 (Technical, Breadth, Bull, Bear, Trader)
3. **Cost per decision:** ~$0.04 (4 cents)
4. **Decision frequency:** 60 minutes (default), configurable
5. **Daily decisions:** 6 cycles × $0.04 = ~$0.24/day/symbol

**Bottlenecks:**
1. LLM API latency (1.5-3s per call) - 70% of total time
2. Market data fetching (0.5-1s) - 10% of total time
3. Risk management lookups (0.5-2s) - 15% of total time

**Reliability:**
- Validation catches hallucinations in <300ms
- Retry logic adds robustness
- Fallback mechanisms prevent hard failures
- All critical paths have error handling

**Scalability:**
- 1-5 symbols: run sequentially (45-75s total)
- 5-10 symbols: run 2-3 in parallel (90-150s total)
- 10+ symbols: requires full parallelization + API rate limit management
