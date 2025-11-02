# Trading Agent Architecture Analysis - ActiveTrader-LLM

## Overview
ActiveTrader-LLM is a **multi-agent LLM-based trading system** that synthesizes analysis from multiple agents into executable trade plans. The system is designed for active stock trading with reasoning-based decision making rather than rule-based strategies.

---

## 1. TRADING AGENT LOCATION & STRUCTURE

### File Path
**Primary Location:** `/Users/abhinavpenagalapati/equity-ai-trading/active_trader_llm/trader/`

### Core Files
- **trader_agent.py** - Main trader agent (316 lines)
- **trade_plan_validator.py** - Validation logic for trade plans (297 lines)
- **__init__.py** - Module initialization

### Key Classes

#### TraderAgent Class
- **Location:** `active_trader_llm/trader/trader_agent.py`
- **Purpose:** Synthesizes multi-agent signals into concrete trade plans
- **Key Methods:**
  - `decide()` - Generate trade plan for a symbol
  - `_build_decision_prompt()` - Constructs the prompt for the LLM
  
- **Model:** Uses Claude 3.5 Sonnet (claude-3-5-sonnet-20241022)
- **Temperature:** 0.3 (low creativity, deterministic)
- **Max Tokens:** 600

#### TradePlan Schema
```python
class TradePlan(BaseModel):
    symbol: str
    strategy: Literal["momentum_breakout", "mean_reversion", "pullback", "sector_rotation"]
    direction: Literal["long", "short"]
    entry: float
    stop_loss: float
    take_profit: float
    position_size_pct: float  # 0.0 to 10.0%
    time_horizon: Literal["1d", "3d", "1w"]
    rationale: List[str]      # Reasoning for the trade
    risk_reward_ratio: float  # Calculated after plan generation
```

---

## 2. HOW THE TRADER RECEIVES DECISIONS

### Data Flow (Complete Pipeline)

```
1. DATA INGESTION
   ↓
2. FEATURE ENGINEERING
   ↓
3. ANALYST SIGNALS (3 types)
   ├─ Technical Analyst
   ├─ Breadth Health Analyst
   └─ Optional: Sentiment & Macro
   ↓
4. RESEARCHER DEBATE (Bull vs Bear)
   ↓
5. TRADER AGENT DECISION (Synthesizes all above)
   ↓
6. RISK VALIDATION
   ↓
7. EXECUTION & LOGGING
```

### Input Sources to TraderAgent.decide()

The `decide()` method receives:

```python
def decide(
    symbol: str,                          # Stock ticker
    analyst_outputs: Dict,                # Dict of analyst results
    researcher_outputs: tuple,            # (BullThesis, BearThesis)
    strategy_lib: List[Dict],             # Available strategies
    risk_params: Dict,                    # Risk parameters
    memory_context: Optional[str] = None  # Recent performance
) -> Optional[TradePlan]
```

#### Analyst Outputs Structure
```python
analyst_outputs = {
    'technical': {
        'signal': 'long|short|neutral',
        'confidence': 0.0-1.0,
        'reasons': [...],
        'price': float,    # Current price
        'atr': float,      # ATR for position sizing
        'horizon': str
    },
    'breadth': {
        'regime': str,  # trending_bull, range, etc
        'breadth_score': float,
        'notes': [...]
    },
    'sentiment': {...},  # Optional
    'macro': {...}       # Optional
}
```

#### Researcher Outputs Structure
```python
# Tuple of (BullThesis, BearThesis)
BullThesis(
    symbol="AAPL",
    thesis=["Bull point 1", "Bull point 2", ...],
    confidence=0.7
)

BearThesis(
    symbol="AAPL",
    thesis=["Bear point 1", "Bear point 2", ...],
    confidence=0.4
)
```

### Main Entry Point Orchestration
**File:** `active_trader_llm/main.py` - `ActiveTraderLLM` class

**Workflow in `run_decision_cycle()`:**

```python
Step 0: Run market scanner (if enabled)
  └─ Returns universe of stocks to trade

Step 1: Fetch market data (yfinance or Alpaca)
  └─ Historical OHLCV data

Step 2: Compute technical features
  └─ Indicators (EMA, SMA, RSI, ATR) + market snapshot

Step 3: Run analyst agents (per symbol)
  ├─ TechnicalAnalyst.analyze()
  ├─ BreadthHealthAnalyst.analyze()
  └─ Optional sentiment/macro

Step 4: Run researcher debate (per symbol)
  └─ ResearcherDebate.debate()

Step 5: Generate trade plans
  └─ TraderAgent.decide() for each symbol

Step 6: Risk validation
  └─ RiskManager.evaluate() filters/approves plans

Step 7: Execution logging
  └─ JSONLogger records all decisions

Step 8: Strategy learning (optional)
  └─ StrategyMonitor.monitor_and_switch()
```

---

## 3. TRADE EXECUTION LOGIC

### Current State
**Trade execution is NOT yet implemented.** The system currently:
- Generates trade plans (JSON format)
- Validates them
- Logs decisions to JSONL files
- **Simulates execution** in backtest mode only

### Trade Plan Validation (Built-in Safety)

**File:** `active_trader_llm/trader/trade_plan_validator.py`

The `TradePlanValidator` performs comprehensive checks:

1. **Required Fields Check**
   - Entry > 0
   - Stop loss > 0
   - Take profit > 0
   - Position size > 0

2. **Price Logic Validation**
   - **LONG:** stop_loss < entry < take_profit
   - **SHORT:** take_profit < entry < stop_loss

3. **Position Size Bounds**
   - Max 10.0% of capital (configurable)

4. **Risk-Reward Ratio**
   - Minimum 1.5:1 (configurable)
   - Calculated as: reward / risk

5. **Price Sanity Check**
   - Entry price within 5% of current price (prevents stale prices)

6. **Stop Distance Validation**
   - Minimum 0.5% (prevents noise stops)
   - Maximum 15.0% (prevents excessive risk)

**Validation Config:**
```python
@dataclass
class ValidationConfig:
    max_position_pct: float = 10.0
    min_risk_reward_ratio: float = 1.5
    max_price_deviation_pct: float = 5.0
    min_stop_distance_pct: float = 0.5
    max_stop_distance_pct: float = 15.0
```

### Risk Manager

**File:** `active_trader_llm/risk/risk_manager.py`

**Philosophy:** MINIMAL INTERFERENCE - Position sizing removed entirely

```python
class RiskManager:
    """
    Only emergency safety rail: optional daily drawdown circuit breaker.
    
    Position sizing limits REMOVED - agent decides based on performance.
    """
    
    def evaluate(trade_plan, portfolio_state) -> RiskDecision:
        # Check only: daily drawdown circuit breaker
        if enforce_daily_drawdown:
            if current_dd_pct >= emergency_drawdown_limit:
                REJECT trade
        
        # Otherwise: APPROVE unchanged
        return RiskDecision(approved=True, modifications={})
```

**Risk Parameters:**
```yaml
risk_parameters:
  enforce_daily_drawdown: true
  emergency_drawdown_limit: 0.20  # 20% daily loss stops trading
  min_win_rate: 0.35
  min_net_pl: 0.0
```

---

## 4. CONFIGURATION SYSTEM

### Configuration Architecture

**Primary Config File:** `/Users/abhinavpenagalapati/equity-ai-trading/config.yaml`

**Schema Definition:** `active_trader_llm/config/config_schema.py`

**Loader:** `active_trader_llm/config/loader.py`

### Configuration Structure

```python
class Config(BaseModel):
    # Execution mode
    mode: Literal["backtest", "paper-live"]
    
    # Data sources
    data_sources: DataSourcesConfig
        - prices: str (yfinance, alpaca)
        - interval: str (1h, 1d, 5m, 15m)
        - universe: List[str]  # Stock symbols
        - lookback_days: int
    
    # Market scanner (optional two-stage system)
    scanner: ScannerConfig
        - enabled: bool
        - stage1: Stage1Config (market-wide summary)
        - stage2: Stage2Config (deep analysis)
    
    # Risk parameters
    risk_parameters: RiskParameters
        - enforce_daily_drawdown: bool
        - emergency_drawdown_limit: float
    
    # Trade validation
    trade_validation: TradeValidationConfig
        - enabled: bool
        - max_position_pct: float
        - min_risk_reward_ratio: float
        - max_price_deviation_pct: float
        - min_stop_distance_pct: float
        - max_stop_distance_pct: float
    
    # Strategies
    strategies: List[StrategyConfig]
    strategy_switching: StrategySwitchingConfig
    
    # LLM settings
    llm: LLMConfig
        - provider: str (anthropic, openai, local)
        - model: str
        - api_key: str
        - temperature: float
        - max_tokens: int
    
    # Technical indicators
    technical_indicators: Dict
        - quality: data requirements
        - daily: EMA, SMA, RSI, ATR periods
        - weekly: Weekly-converted indicators
    
    # Market breadth
    market_breadth: Dict
        - new_highs_lows config
        - regime thresholds
    
    # Logging and storage
    database_path: str
    log_path: str
```

### Loading Configuration

```python
# From YAML or JSON
config = load_config('config.yaml')

# Access nested values
mode = config.mode  # 'backtest' or 'paper-live'
universe = config.data_sources.universe
max_position = config.trade_validation.max_position_pct
```

---

## 5. DATA FLOW FROM ANALYSTS TO TRADER

### Stage 1: Data Ingestion

**Files:** 
- `active_trader_llm/data_ingestion/price_volume_ingestor.py` - Primary (yfinance-based)
- `active_trader_llm/data_ingestion/alpaca_bars_ingestor.py` - Alpaca integration

**Ingestor.fetch_prices()** returns:
```python
DataFrame with columns: symbol, open, high, low, close, volume
Index: DatetimeIndex
Lookback: 90 days (configurable)
```

### Stage 2: Feature Engineering

**File:** `active_trader_llm/feature_engineering/feature_builder.py`

Computes:
- **Daily Indicators:** EMA(5,10,20), SMA(50,200), RSI(14), ATR(14)
- **Weekly Indicators:** EMA(50 days), SMA(105,150,250 days)
- **Market Snapshot:** Regime detection, breadth scoring

Returns:
```python
features_dict = {
    'AAPL': FeatureSet(
        daily_indicators={'rsi': 65.2, 'atr': 2.5, ...},
        weekly_indicators={'ema_10': 170.0, ...},
        ohlcv={'close': 175.0, 'volume': 50000000}
    ),
    ...
}

market_snapshot = MarketSnapshot(
    regime_hint='trending_bull',
    breadth_score=0.45,
    pct_above_ema200=0.65
)
```

### Stage 3: Technical Analyst

**File:** `active_trader_llm/analysts/technical_analyst.py`

```python
signal = technical_analyst.analyze(
    symbol='AAPL',
    features=features['AAPL'],
    market_snapshot=market_snapshot,
    memory_context="Recent context..."
)

# Returns TechnicalSignal:
{
    'signal': 'long',
    'confidence': 0.75,
    'reasons': ['RSI recovering', 'MACD cross', 'Above EMA50'],
    'price': 175.0,
    'atr': 2.5,
    'horizon': '1-3 days'
}
```

**Analyst System Prompt includes:**
- Indicator interpretation guidelines
- Confluence rules (look for multiple signals agreeing)
- Recent trade memory context
- Market regime considerations

### Stage 4: Breadth Health Analyst

**File:** `active_trader_llm/analysts/breadth_health_analyst.py`

Analyzes market-wide health:
```python
signal = breadth_analyst.analyze(market_snapshot)

# Returns:
{
    'regime': 'trending_bull',  # trending_bull, range, trending_bear, risk_off
    'breadth_score': 0.45,      # -1.0 to 1.0
    'confidence': 0.8,
    'notes': ['Strong new highs', 'Advance/decline ratio high']
}
```

### Stage 5: Researcher Debate

**File:** `active_trader_llm/researchers/bull_bear.py`

Takes analyst signals and generates opposing viewpoints:

```python
bull, bear = researcher_debate.debate(
    symbol='AAPL',
    analyst_outputs={
        'technical': {...},
        'breadth': {...}
    },
    market_snapshot=market_snapshot,
    memory_summary="Recent trade context..."
)

# Returns tuple of:
BullThesis(
    thesis=['Momentum resuming', 'Support held'],
    confidence=0.7
)

BearThesis(
    thesis=['Near resistance', 'Could consolidate'],
    confidence=0.4
)
```

### Stage 6: Trade Plan Generation

**File:** `active_trader_llm/trader/trader_agent.py`

Trader synthesizes all signals:

```python
plan = trader_agent.decide(
    symbol='AAPL',
    analyst_outputs={
        'technical': {...},
        'breadth': {...}
    },
    researcher_outputs=(bull_thesis, bear_thesis),
    strategy_lib=[
        {'name': 'momentum_breakout', 'regime': 'trending_bull'},
        {'name': 'mean_reversion', 'regime': 'range'},
        ...
    ],
    risk_params={'max_position_pct': 0.05, ...},
    memory_context="Last 5 trades summary..."
)

# Returns TradePlan or None
TradePlan(
    symbol='AAPL',
    strategy='momentum_breakout',
    direction='long',
    entry=175.0,
    stop_loss=172.0,
    take_profit=180.0,
    position_size_pct=0.05,
    time_horizon='3d',
    rationale=['Momentum breakout...'],
    risk_reward_ratio=1.67
)
```

---

## 6. BROKER INTEGRATIONS

### Current Integration Status

**PRIMARY DATA SOURCE:** Alpaca (read-only, market data only)

**File:** `active_trader_llm/data_ingestion/alpaca_bars_ingestor.py`

Features:
- Efficient multi-symbol batching (200 symbols per request)
- Rate limit tracking and throttling
- Automatic 429 retry handling
- Pagination support
- Caching to minimize API calls

**Rate Limits:**
- Standard: 200 requests/minute
- Unlimited: 1000 requests/minute
- Multi-symbol: Max 200 symbols per request

### Trade Execution

**Status:** NOT YET IMPLEMENTED

Current implementation only:
1. Generates trade plans (JSON)
2. Validates them
3. Logs to JSONL files
4. Simulates execution in backtest mode

**Next Steps Required:**
- Alpaca account integration (Paper Trading API)
- Order placement logic
- Position tracking
- Trade exit (stop loss, take profit) management
- Real-time order monitoring
- Slippage simulation for paper trading

---

## 7. KEY CLASSES AND METHODS

### TraderAgent

| Method | Purpose | Input | Output |
|--------|---------|-------|--------|
| `__init__()` | Initialize with API key and model | api_key, model, validation_config | TraderAgent instance |
| `decide()` | Generate trade plan | symbol, analyst_outputs, researcher_outputs, strategy_lib, risk_params | TradePlan or None |
| `_build_decision_prompt()` | Construct LLM prompt | symbol, all analyst/researcher data | str (prompt) |

### TradePlanValidator

| Method | Purpose | Checks |
|--------|---------|--------|
| `validate_trade_plan()` | Main validation method | All 6 validation rules |
| `_validate_required_fields()` | Check non-zero values | Entry, stop, target, size > 0 |
| `_validate_long_prices()` | Check long logic | stop < entry < target |
| `_validate_short_prices()` | Check short logic | target < entry < stop |
| `_validate_position_size()` | Check size bounds | size <= max_position_pct |
| `_validate_risk_reward()` | Check R:R ratio | reward/risk >= min_ratio |
| `_validate_price_sanity()` | Check current price deviation | entry within max_deviation_pct |
| `_validate_stop_distance()` | Check stop width | min <= distance <= max |

### RiskManager

| Method | Purpose |
|--------|---------|
| `evaluate()` | Single trade plan evaluation |
| `batch_evaluate()` | Multiple plans evaluation |

### ActiveTraderLLM (Main Orchestrator)

| Method | Purpose |
|--------|---------|
| `__init__()` | Initialize all components |
| `run_decision_cycle()` | Execute one complete decision cycle (7 steps) |
| `run_learning_update()` | End-of-day strategy stats update |

---

## 8. STRATEGY SELECTION LOGIC

### Strategy Library

The system supports 4 strategies, each suited to different regimes:

```yaml
strategies:
  - name: momentum_breakout
    regime: trending_bull
    enabled: true
    
  - name: mean_reversion
    regime: range
    enabled: true
    
  - name: pullback
    regime: mild_trend
    enabled: true
    
  - name: sector_rotation
    regime: mixed
    enabled: true
```

### Regime Detection

Breadth analyst determines current market regime:
- **trending_bull:** >50% breadth score + >60% above EMA200
- **range:** -0.5 < breadth < 0.5
- **trending_bear:** breadth < -0.3
- **risk_off:** breadth < -0.5 + high VIX

### Strategy Selection by Trader

Trader selects strategy based on:
1. Current market regime (from breadth analyst)
2. Signal strength (analyst confidence)
3. Recent strategy performance (from memory)
4. Analyst/researcher agreement level

---

## 9. MEMORY SYSTEM

**File:** `active_trader_llm/memory/memory_manager.py`

### Layered Memory Architecture

**Short-term Memory:**
- Recent N trades with full context
- Used for intra-day pattern recognition
- Guides researcher debate with recent outcomes

**Long-term Memory:**
- Strategy performance statistics
- Per-regime, per-strategy metrics
- Used for strategy switching decisions

### Metrics Tracked

```python
class StrategyStats:
    strategy_name: str
    regime: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_return: float
    total_pnl: float
    avg_rr: float
```

### Strategy Switching

**File:** `active_trader_llm/learning/strategy_monitor.py`

Monitors strategy performance and recommends switches:
```python
config:
  window_trades: 30          # Evaluate last 30 trades
  min_win_rate: 0.35         # Must achieve 35% win rate
  min_net_pl: 0.0            # Must be profitable
  hysteresis_trades_before_switch: 10  # Cooldown period
```

---

## 10. EXECUTION MODES

### Backtest Mode
```yaml
mode: backtest
```
- Reads historical data
- Simulates execution with 0.02 slippage
- Logs all decisions
- No actual trades

### Paper-Live Mode
```yaml
mode: paper-live
```
- Ready for integration with Alpaca Paper Trading
- Would place real orders (in paper account)
- Real-time position tracking
- Simulated fills with realistic slippage

---

## 11. KEY INSIGHTS & PATTERNS

### Design Philosophy
1. **Reasoning over Rules:** Uses LLMs to reason about trades, not fixed rules
2. **Multi-Agent Debate:** Bull/bear researchers provide balanced perspective
3. **Validation-First:** Comprehensive trade plan validation prevents LLM errors
4. **Memory-Driven:** System learns from recent performance and adjusts
5. **Minimal Risk Constraints:** Trusts LLM agent with sizing (validation only)

### Error Handling
- **Technical analyst fails?** Symbol skipped (no data)
- **Researcher debate fails?** Symbol skipped (no synthetic debates)
- **Trade plan validation fails?** Plan rejected (safety rail)
- **Risk manager rejects?** Emergency drawdown only (minimal interference)

### Extensibility Points
1. Add new analysts (sentiment, macro, options)
2. Add new validation rules in `TradePlanValidator`
3. Add new strategies in config
4. Implement broker-specific execution modules
5. Customize regime detection thresholds

---

## Summary Table

| Component | Location | Purpose | Input | Output |
|-----------|----------|---------|-------|--------|
| TraderAgent | trader/trader_agent.py | Synthesize to trade plans | Analyst/researcher signals | TradePlan |
| TradePlanValidator | trader/trade_plan_validator.py | Validate LLM output | TradePlan | Approval/rejection |
| RiskManager | risk/risk_manager.py | Filter by risk constraints | TradePlan + portfolio | Approval/modification |
| TechnicalAnalyst | analysts/technical_analyst.py | Generate technical signals | Features + market snapshot | TechnicalSignal |
| BreadthAnalyst | analysts/breadth_health_analyst.py | Detect market regime | Market snapshot | BreadthSignal |
| ResearcherDebate | researchers/bull_bear.py | Debate opportunity | Analyst signals | Bull/Bear thesis |
| MemoryManager | memory/memory_manager.py | Store trade history | Trade results | Context for future decisions |
| StrategyMonitor | learning/strategy_monitor.py | Evaluate strategy performance | Trade history | Strategy switch recommendations |
| Config | config/config_schema.py | Define all parameters | YAML file | Config object |
| Main Orchestrator | main.py | Run complete cycle | Config | Trade decisions logged |

