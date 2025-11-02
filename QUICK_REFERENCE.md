# Trading Agent - Quick Reference Guide

## Core Locations

| What | Where |
|------|-------|
| Main Trader Agent | `/active_trader_llm/trader/trader_agent.py` |
| Trade Validation | `/active_trader_llm/trader/trade_plan_validator.py` |
| Risk Manager | `/active_trader_llm/risk/risk_manager.py` |
| Main Orchestrator | `/active_trader_llm/main.py` |
| Configuration | `/config.yaml` + `config/config_schema.py` |

## Decision Flow

```
Data → Features → Analyst Signals → Researcher Debate → Trade Plan → Validation → Risk Check → Execution
```

## Key Method Signature

```python
trader_agent.decide(
    symbol: str,                      # "AAPL"
    analyst_outputs: Dict,            # Technical, breadth, sentiment, macro
    researcher_outputs: tuple,        # (BullThesis, BearThesis)
    strategy_lib: List[Dict],         # Available strategies
    risk_params: Dict,                # max_position_pct, etc
    memory_context: Optional[str]     # Recent trade context
) -> Optional[TradePlan]
```

## Trade Plan Output

```json
{
    "symbol": "AAPL",
    "strategy": "momentum_breakout|mean_reversion|pullback|sector_rotation",
    "direction": "long|short",
    "entry": 175.0,
    "stop_loss": 172.0,
    "take_profit": 180.0,
    "position_size_pct": 0.05,
    "time_horizon": "1d|3d|1w",
    "rationale": ["reason 1", "reason 2", "reason 3"],
    "risk_reward_ratio": 1.67
}
```

## Validation Rules (Auto-Applied)

1. All prices must be positive
2. LONG: stop < entry < target
3. SHORT: target < entry < stop
4. Position size ≤ 10% (configurable)
5. Risk/reward ≥ 1.5:1 (configurable)
6. Entry within 5% of current price (configurable)
7. Stop distance: 0.5-15% (configurable)

**If any rule fails:** Trade plan is REJECTED

## Risk Manager Philosophy

- **Minimal Interference:** Agent decides position sizing
- **Only Safety Rail:** Emergency daily drawdown circuit breaker (optional)
- **Default Limits:** None - validation layer handles sanity checks

## Configuration Quick Reference

```yaml
# Execution mode
mode: backtest | paper-live

# Trading universe
data_sources:
  universe: [AAPL, MSFT, SPY, QQQ]
  interval: 1h | 1d | 5m | 15m
  lookback_days: 90

# Risk parameters
risk_parameters:
  enforce_daily_drawdown: true
  emergency_drawdown_limit: 0.20  # 20% loss stops trading

# Trade validation
trade_validation:
  enabled: true
  max_position_pct: 10.0
  min_risk_reward_ratio: 1.5
  max_price_deviation_pct: 5.0
  min_stop_distance_pct: 0.5
  max_stop_distance_pct: 15.0

# LLM settings
llm:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  temperature: 0.3
  max_tokens: 2000
```

## Running the System

```bash
# One decision cycle
python -m active_trader_llm.main --config config.yaml --cycles 1

# Multiple cycles
python -m active_trader_llm.main --config config.yaml --cycles 5

# With learning update
python -m active_trader_llm.main --config config.yaml --cycles 5 --learning
```

## Error Handling

| Failure | Outcome |
|---------|---------|
| Data not available | Symbol skipped |
| Technical analysis fails | Symbol skipped |
| Researcher debate fails | Symbol skipped (no synthetic debates) |
| Trade validation fails | Trade rejected (safety) |
| Risk manager rejects | Trade rejected (emergency circuit) |

## Integration Points for Extensions

1. **Add Analyst:** Create new analyzer in `analysts/`
2. **Add Validation Rule:** Extend `TradePlanValidator`
3. **Add Strategy:** Add to `config.yaml` strategies list
4. **Add Regime:** Modify breadth analyst detection
5. **Broker Integration:** Implement execution in trade logging phase

## Memory System

- **Short-term:** Recent trades with full context
- **Long-term:** Strategy stats per regime
- **Used for:** Researcher debate context + strategy switching

## Strategy Switching

Evaluated every 30 trades in current regime:
- **Trigger:** Win rate < 35% OR net_pl < 0.0
- **Cooldown:** 10 trades before next switch
- **Options:** momentum_breakout, mean_reversion, pullback, sector_rotation

## Important Design Choices

1. **No position size hard limits** - Agent decides with validation safety
2. **No max concurrent positions limit** - Only breadth-regime matching
3. **No sector concentration limits** - Full agent discretion
4. **Validation > Risk Manager** - Prevents impossible trades before risking capital
5. **Memory-driven strategy selection** - Adapts to market conditions

## Files to Modify for Integration

| Goal | Files |
|------|-------|
| Change validation rules | `trader/trade_plan_validator.py` |
| Change risk parameters | `config.yaml` or `risk/risk_manager.py` |
| Change strategy logic | `config.yaml` + trader system prompt |
| Add broker integration | `data_ingestion/` for execution |
| Customize indicators | `feature_engineering/feature_builder.py` |
| Adjust regime detection | `analysts/breadth_health_analyst.py` |

## Key LLM Models Used

- **Trader Agent:** Claude 3.5 Sonnet (synthesis)
- **Technical Analyst:** Claude 3.5 Sonnet (analysis)
- **Breadth Analyst:** Claude 3.5 Sonnet (regime detection)
- **Researcher Debate:** Claude 3.5 Sonnet (opposing views)

All use temperature=0.3 for deterministic, reasoning-based outputs.

## Broker Status

- **Data Source:** Alpaca (read-only, market data)
- **Paper Trading API:** Not yet implemented
- **Live Trading:** Not yet implemented
- **Current State:** Simulated backtest execution only

