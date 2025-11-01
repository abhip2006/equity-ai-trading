# Hardcoded Values Audit - Complete Inventory

**Date**: 2025-11-01
**Status**: 🟡 Many hardcoded values remain (non-critical but should be configurable)

---

## Executive Summary

After fixing all hallucination risks, **32 hardcoded values** remain in the codebase. None are critical for safety (hallucinations are fixed), but they limit flexibility for different trading strategies and market conditions.

**Impact**: Medium - System works but cannot be easily tuned for different:
- Trading styles (scalping vs swing trading)
- Risk profiles (conservative vs aggressive)
- Market conditions (bull vs bear markets)

---

## Category 1: Technical Indicator Periods (7 values)

**File**: `active_trader_llm/feature_engineering/indicators.py`

### Current Hardcoded Values

```python
# Line 54 - RSI period
def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:

# Lines 67-69 - MACD parameters
def compute_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> tuple:

# Lines 84-85 - Bollinger Bands
def compute_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> tuple:

# Line 97 - ATR period
def compute_atr(high, low, close, period: int = 14) -> pd.Series:

# Lines 146-148 - Moving average periods
symbol_df['ema_50'] = self.compute_ema(symbol_df['close'], 50)
symbol_df['ema_200'] = self.compute_ema(symbol_df['close'], 200)
symbol_df['sma_20'] = self.compute_sma(symbol_df['close'], 20)
```

### Why This Matters

Different trading strategies need different periods:
- **Scalping**: RSI(7), MACD(5/13/5), EMA(20/50)
- **Day trading**: RSI(14), MACD(12/26/9), EMA(50/200) ← Current defaults
- **Swing trading**: RSI(21), MACD(24/52/9), EMA(100/200)

### Recommended Fix

**Option A - Add to config.yaml**:
```yaml
indicators:
  rsi_period: 14
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  bollinger_period: 20
  bollinger_std: 2.0
  atr_period: 14
  ema_short: 50
  ema_long: 200
  sma_period: 20
```

**Option B - Load from config in FeatureEngineer**:
```python
class FeatureEngineer:
    def __init__(self, indicator_config: Dict):
        self.rsi_period = indicator_config.get('rsi_period', 14)
        self.macd_fast = indicator_config.get('macd_fast', 12)
        self.macd_slow = indicator_config.get('macd_slow', 26)
        # etc.

    def build_features(self, price_df):
        symbol_df['rsi'] = self.compute_rsi(symbol_df['close'], self.rsi_period)
        # etc.
```

**Priority**: 🟡 Medium - Allows strategy customization

---

## Category 2: Regime Classification Thresholds (6 values)

**File**: `active_trader_llm/feature_engineering/indicators.py`

### Current Hardcoded Values

```python
# Lines 231-238 - Regime determination thresholds
if breadth_score < -0.5 or avg_rsi < 35:
    regime = "risk_off"
elif breadth_score > 0.5 and avg_macd_hist > 0 and pct_above_ema200 > 0.6:
    regime = "trending_bull"
elif breadth_score < -0.3 and avg_macd_hist < 0:
    regime = "trending_bear"
else:
    regime = "range"
```

**Values**:
- `breadth_score < -0.5` → risk_off
- `avg_rsi < 35` → risk_off
- `breadth_score > 0.5` → trending_bull
- `pct_above_ema200 > 0.6` → trending_bull (60% of stocks above 200-day EMA)
- `breadth_score < -0.3` → trending_bear

### Why This Matters

Different market environments need different sensitivity:
- **Bull market 2020-2021**: Raise thresholds (0.7 for trending_bull)
- **Bear market 2022**: Lower thresholds (-0.4 for risk_off)
- **Choppy 2024**: Wider range zone (-0.2 to 0.2)

### Recommended Fix

Add to config:
```yaml
regime_thresholds:
  risk_off:
    breadth_score: -0.5
    min_rsi: 35
  trending_bull:
    breadth_score: 0.5
    pct_above_ema200: 0.6
  trending_bear:
    breadth_score: -0.3
  range:
    # Anything not matching above
```

**Priority**: 🟡 Medium - Allows market adaptation

---

## Category 3: Risk Management Parameters (6 values)

**File**: `active_trader_llm/config/config_schema.py`

### Current Hardcoded Defaults

```python
# Lines 17-22 - Risk parameters
class RiskParameters(BaseModel):
    max_position_pct: float = 0.05           # 5% max per position
    max_concurrent_positions: int = 8         # Max 8 positions
    max_daily_drawdown: float = 0.10         # 10% daily loss limit
    max_sector_concentration: float = 0.30    # 30% max in one sector
    min_win_rate: float = 0.35               # 35% min win rate
    min_net_pl: float = 0.0                  # $0 min P/L
```

### Why This Matters

**Current defaults are for a MODERATE risk profile**. Different traders need different limits:

| Profile | max_position_pct | max_concurrent | max_daily_dd |
|---------|-----------------|----------------|--------------|
| Conservative | 2% | 5 | 5% |
| **Current** | **5%** | **8** | **10%** |
| Aggressive | 10% | 15 | 15% |

### Status

✅ **GOOD NEWS**: These already have config defaults in `config_schema.py`
⚠️ **PROBLEM**: Defaults are baked in - unclear these are defaults

### Recommended Fix

**Option A - Make required (no defaults)**:
```python
class RiskParameters(BaseModel):
    max_position_pct: float  # No default - user must specify
    max_concurrent_positions: int  # No default - user must specify
    max_daily_drawdown: float  # No default - user must specify
```

**Option B - Keep defaults but document them**:
```python
class RiskParameters(BaseModel):
    # MODERATE RISK PROFILE (5% position, 8 concurrent, 10% daily DD)
    max_position_pct: float = 0.05
    max_concurrent_positions: int = 8
    max_daily_drawdown: float = 0.10
```

**Priority**: 🟡 Medium - Already configurable, just improve documentation

---

## Category 4: Validation Thresholds (4 values)

**File**: `active_trader_llm/trader/trader_agent.py`

### Current Hardcoded Values

```python
# Line 155 - Entry price deviation limit
max_entry_deviation = current_price * 0.05  # 5% max

# Lines 178-179 - ATR multiple range
if atr_multiple < 0.5 or atr_multiple > 3.0:  # 0.5x to 3x ATR

# Line 200 - Position size range
if plan.position_size_pct < 0.005 or plan.position_size_pct > 0.10:  # 0.5% to 10%
```

### Why This Matters

These are **anti-hallucination safeguards** I just added. They prevent:
- LLM suggesting entry too far from current price
- LLM using unreasonably tight/wide stops
- LLM suggesting tiny or huge positions

**Should these be configurable?**

**Argument FOR keeping hardcoded**:
- These are safety limits, not strategy parameters
- Making them configurable could allow dangerous values
- Users might set entry_deviation=50% and get bad trades

**Argument FOR making configurable**:
- Volatile stocks (crypto, biotech) may need wider limits
- Different markets have different norms
- Advanced users may want control

### Recommended Fix

**Option A - Keep hardcoded** (safer):
```python
# Anti-hallucination safety limits (DO NOT MODIFY)
SAFETY_LIMITS = {
    'max_entry_deviation_pct': 0.05,  # 5%
    'min_atr_multiple': 0.5,
    'max_atr_multiple': 3.0,
    'min_position_pct': 0.005,  # 0.5%
    'max_position_pct': 0.10,   # 10%
}
```

**Option B - Make configurable with warnings**:
```python
# Load from config with validation
max_entry_deviation = config.get('max_entry_deviation_pct', 0.05)
if max_entry_deviation > 0.10:
    logger.warning("⚠️  Entry deviation >10% is dangerous!")
```

**Priority**: 🟢 Low - Keep as-is for safety

---

## Category 5: Strategy Switching Parameters (4 values)

**File**: `active_trader_llm/config/config_schema.py` and `learning/strategy_monitor.py`

### Current Hardcoded Values

```python
# config_schema.py lines 32-35
class StrategySwitchingConfig(BaseModel):
    window_trades: int = 30               # Evaluate last 30 trades
    min_win_rate: float = 0.35           # 35% min to continue
    min_net_pl: float = 0.0              # $0 min P/L
    hysteresis_trades_before_switch: int = 10  # Wait 10 trades before switching back

# learning/strategy_monitor.py lines 224-226 - Scoring weights
score = (
    stats.win_rate * 50 +                     # Win rate weight = 50
    min(stats.avg_return, 100) * 0.5 +        # Return weight = 0.5 (capped at 100)
    stats.avg_rr * 10                         # R:R weight = 10
)
```

### Why This Matters

Strategy switching sensitivity affects:
- **Conservative**: window=50, min_wr=0.40, hysteresis=20 (slower to switch)
- **Current**: window=30, min_wr=0.35, hysteresis=10 (moderate)
- **Aggressive**: window=15, min_wr=0.30, hysteresis=5 (quick to switch)

Scoring weights determine what matters more:
- Current formula: Win rate >>> R:R > Avg return
- Alternative: Prioritize returns over win rate

### Status

✅ **Already configurable** in config_schema.py for window/min_wr/hysteresis
❌ **Scoring weights hardcoded** in strategy_monitor.py

### Recommended Fix

Add scoring weights to config:
```yaml
strategy_switching:
  window_trades: 30
  min_win_rate: 0.35
  min_net_pl: 0.0
  hysteresis_trades_before_switch: 10
  scoring_weights:
    win_rate: 50
    avg_return: 0.5
    avg_rr: 10
```

**Priority**: 🟡 Medium - Allows strategy tuning

---

## Category 6: Macro/Breadth Analyst Thresholds (3 values)

**File**: `active_trader_llm/analysts/macro_analyst.py` and `breadth_health_analyst.py`

### Current Hardcoded Values

```python
# macro_analyst.py lines 79-90 - VIX thresholds
if vix > 25:                    # Risk-off threshold
    bias = "risk_off"
elif vix < 12:                  # Risk-on threshold
    bias = "risk_on"
else:
    bias = "neutral"

# breadth_health_analyst.py lines 105-118 - Regime validation thresholds
if signal.regime == "trending_bull":
    if breadth_score < 0.2:     # Min breadth for bull
        return False
    if ad_ratio < 1.0:          # Min A/D ratio for bull
        return False

if signal.regime == "risk_off":
    if breadth_score > -0.3 and vix < 20:  # Risk-off requires breadth<-0.3 OR vix>20
        return False
```

### Why This Matters

VIX interpretation varies by market environment:
- **2019 (low vol)**: VIX>20 is elevated
- **2020 (COVID)**: VIX>40 is elevated
- **2024 (normal)**: VIX>25 is elevated ← Current

### Recommended Fix

Add to config:
```yaml
macro_thresholds:
  vix_risk_off: 25
  vix_risk_on: 12

breadth_thresholds:
  trending_bull_min_breadth: 0.2
  trending_bull_min_ad_ratio: 1.0
  risk_off_max_breadth: -0.3
  risk_off_min_vix: 20
```

**Priority**: 🟡 Medium - Allows market adaptation

---

## Category 7: CRITICAL BUG - Hardcoded Sector (1 value)

**File**: `active_trader_llm/risk/risk_manager.py`

### Current Hardcoded Value

```python
# Line 154 - ALL STOCKS HARDCODED AS TECHNOLOGY
def _calculate_sector_exposure(self, portfolio_state, trade_plan):
    sector = "Technology"  # Would lookup via yfinance or mapping
```

### Why This Is a Problem

**SECTOR CONCENTRATION LIMITS DON'T WORK!**

If you set `max_sector_concentration: 0.30` (30%), it won't help because:
- AAPL → "Technology"
- MSFT → "Technology"
- SPY → "Technology" (WRONG!)
- QQQ → "Technology" (WRONG!)
- ANY stock → "Technology"

You could have 100% of portfolio in tech and system thinks it's diversified!

### Recommended Fix

**Option A - Static sector mapping**:
```python
SECTOR_MAP = {
    'AAPL': 'Technology',
    'MSFT': 'Technology',
    'GOOGL': 'Technology',
    'SPY': 'Index',
    'QQQ': 'Technology Index',
    'XLE': 'Energy',
    'XLF': 'Financials',
    # Add your watchlist here
}

def _calculate_sector_exposure(self, portfolio_state, trade_plan):
    sector = SECTOR_MAP.get(trade_plan.symbol, 'Unknown')
```

**Option B - Dynamic lookup via yfinance** (better):
```python
import yfinance as yf

def _calculate_sector_exposure(self, portfolio_state, trade_plan):
    try:
        ticker = yf.Ticker(trade_plan.symbol)
        sector = ticker.info.get('sector', 'Unknown')
    except Exception as e:
        logger.warning(f"Failed to get sector for {trade_plan.symbol}: {e}")
        sector = 'Unknown'

    # Calculate exposure for this actual sector
    sector_exposure = sum(
        pos['size_pct'] for pos in portfolio_state.positions
        if pos.get('sector') == sector
    )
    sector_exposure += trade_plan.position_size_pct

    return sector_exposure
```

**Option C - Load from config**:
```yaml
sector_mappings:
  AAPL: Technology
  MSFT: Technology
  SPY: Index
  QQQ: Technology Index
  JPM: Financials
  XOM: Energy
```

**Priority**: 🔴 **HIGH** - This is a bug that breaks sector diversification

---

## Category 8: LLM Prompt Hardcoded Values (2 values)

**File**: `active_trader_llm/trader/trader_agent.py`

### Current Hardcoded Values

```python
# Line 60 in SYSTEM_PROMPT
"Use ATR-based stops (1.5-2x ATR below entry for longs)"
```

This tells the LLM to use 1.5-2x ATR, but we validate 0.5-3x ATR (different range!).

### Why This Matters

Mismatch between what we tell LLM vs what we validate:
- Prompt says: "1.5-2x ATR"
- Validation allows: "0.5-3x ATR"

If you want tighter stops (1x ATR), LLM won't suggest them.

### Recommended Fix

**Option A - Make prompt match validation**:
```python
"Use ATR-based stops (0.5-3x ATR, typically 1.5-2x for normal volatility)"
```

**Option B - Make both configurable**:
```python
# In config
stop_loss_atr_range:
  min: 0.5
  max: 3.0
  suggested_min: 1.5
  suggested_max: 2.0

# In prompt
f"Use ATR-based stops ({config.suggested_min}-{config.suggested_max}x ATR)"

# In validation
if atr_multiple < config.min or atr_multiple > config.max:
```

**Priority**: 🟡 Medium - Minor inconsistency

---

## Category 9: Data Source Defaults (3 values)

**File**: `active_trader_llm/config/config_schema.py`

### Current Hardcoded Values

```python
# Lines 10-13
class DataSourcesConfig(BaseModel):
    prices: str = "yfinance"
    interval: str = "1h"
    universe: List[str] = ["AAPL", "MSFT", "SPY", "QQQ"]
    lookback_days: int = 90
```

### Why This Matters

Default watchlist is just 4 symbols. Different traders want:
- **Index trader**: ["SPY", "QQQ", "DIA", "IWM"]
- **Tech trader**: ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]
- **Sector rotation**: 11 sector ETFs
- **Crypto trader**: ["BTC-USD", "ETH-USD", ...]

### Status

✅ **Already configurable** via config.yaml

### Recommended Fix

Just document better:
```python
# Default universe is a SAMPLE - customize via config.yaml
universe: List[str] = ["AAPL", "MSFT", "SPY", "QQQ"]
```

**Priority**: 🟢 Low - Already configurable

---

## Summary Table

| Category | Count | Priority | Already Configurable? |
|----------|-------|----------|---------------------|
| Technical Indicator Periods | 7 | 🟡 Medium | ❌ No |
| Regime Thresholds | 6 | 🟡 Medium | ❌ No |
| Risk Parameters | 6 | 🟡 Medium | ✅ Yes (but unclear) |
| Validation Limits | 4 | 🟢 Low | ❌ No (intentional for safety) |
| Strategy Switching | 4 | 🟡 Medium | ⚠️ Partial (weights not configurable) |
| Macro/Breadth Thresholds | 3 | 🟡 Medium | ❌ No |
| **Sector Mapping** | **1** | **🔴 HIGH** | **❌ No - BUG** |
| LLM Prompt Values | 2 | 🟡 Medium | ❌ No |
| Data Source Defaults | 3 | 🟢 Low | ✅ Yes |
| **TOTAL** | **32** | | |

---

## Recommended Action Plan

### Phase 1 - Fix Critical Bug (1 hour)

**Fix hardcoded sector mapping** in `risk_manager.py`:
```python
# Add static mapping or yfinance lookup
# This is a BUG - sector limits don't work without it
```

### Phase 2 - Make Indicator Periods Configurable (2-3 hours)

Add indicator config:
```yaml
indicators:
  rsi_period: 14
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  ema_short: 50
  ema_long: 200
```

Modify `FeatureEngineer.__init__()` to accept config.

### Phase 3 - Make Thresholds Configurable (2-3 hours)

Add threshold configs for:
- Regime classification
- Macro analyst VIX levels
- Breadth analyst limits

### Phase 4 - Document Existing Configurability (30 min)

Add comments explaining:
- Risk parameters already configurable
- Data sources already configurable
- How to customize via config.yaml

---

## What Should Stay Hardcoded?

**Keep hardcoded** (for safety):
1. ✅ Anti-hallucination validation limits (entry deviation, ATR range, position size range)
2. ✅ Confidence range [0, 1]
3. ✅ Data validation rules

**Make configurable**:
1. ❌ Technical indicator periods
2. ❌ Regime classification thresholds
3. ❌ Strategy switching weights
4. ❌ Sector mappings (CRITICAL BUG)

---

## Files That Need Changes

1. `active_trader_llm/feature_engineering/indicators.py` - Indicator periods
2. `active_trader_llm/risk/risk_manager.py` - **CRITICAL: Sector mapping**
3. `active_trader_llm/learning/strategy_monitor.py` - Scoring weights
4. `active_trader_llm/analysts/macro_analyst.py` - VIX thresholds
5. `active_trader_llm/config/config_schema.py` - Add new config sections

---

## Conclusion

**32 hardcoded values remain**, but only **1 is a critical bug** (sector mapping).

The others are "defaults that should be configurable" - they work fine for moderate day trading but limit flexibility for other strategies.

**Immediate action needed**:
1. 🔴 Fix sector mapping bug (risk limits don't work)
2. 🟡 Make indicator periods configurable (enables strategy customization)
3. 🟡 Make regime thresholds configurable (enables market adaptation)

**Can wait**:
- Validation limits (keep hardcoded for safety)
- Data source defaults (already configurable)
- Risk parameters (already configurable, just improve docs)
