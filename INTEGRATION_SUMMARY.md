# Alpaca Trading API Integration - Summary

## Overview

Your ActiveTrader-LLM trading agent is now fully integrated with Alpaca's trading API, enabling it to place real trades based on the decisions made by your multi-agent system.

---

## What Was Implemented

### 1. **Alpaca Broker Executor Module**
**File**: [active_trader_llm/execution/alpaca_broker.py](active_trader_llm/execution/alpaca_broker.py)

A comprehensive broker integration that handles:
- ✅ Bracket orders with automatic stop loss and take profit
- ✅ Market and limit order types
- ✅ Position size calculation based on portfolio percentage
- ✅ Position tracking and monitoring
- ✅ Order status checking
- ✅ Paper trading and live trading modes
- ✅ Error handling and retry logic

**Key Features**:
```python
# Automatically submits bracket orders
executor.submit_trade(plan)  # Creates 3 orders:
# 1. Main entry order
# 2. Stop loss order (auto-exits on loss)
# 3. Take profit order (auto-exits on win)
```

### 2. **Configuration Schema Updates**
**File**: [active_trader_llm/config/config_schema.py](active_trader_llm/config/config_schema.py)

Added new `ExecutionConfig` class with:
- Broker selection (alpaca, simulated)
- Paper trading toggle
- Order type configuration
- Time-in-force settings
- API credential management

### 3. **Main Pipeline Integration**
**File**: [active_trader_llm/main.py](active_trader_llm/main.py)

Modified execution flow to support three modes:
1. **Backtest**: Simulated execution (no broker)
2. **Paper-Live**: Alpaca paper trading (fake money)
3. **Live**: Real money trading via Alpaca

**Execution Flow**:
```
Decision Cycle → Trade Plans → Risk Validation → Alpaca Execution → Position Tracking
```

### 4. **Enhanced Logging**
**File**: [active_trader_llm/utils/logging_json.py](active_trader_llm/utils/logging_json.py)

Updated to track:
- Broker order IDs
- Order status (filled, partial, pending)
- Execution method (alpaca_paper, alpaca_live)

### 5. **Configuration Files**

#### **config_alpaca_paper.yaml**
Ready-to-use configuration for paper trading:
```yaml
mode: paper-live
execution:
  broker: alpaca
  paper_trading: true  # Safe testing mode
  order_type: market
  time_in_force: day
```

### 6. **Documentation**

#### **ALPACA_SETUP.md**
Comprehensive guide covering:
- API key setup
- Environment configuration
- Running paper trading
- Going live safely
- Monitoring trades
- Troubleshooting

#### **test_alpaca_integration.py**
Test suite to verify:
- API connection
- Account access
- Position retrieval
- Order submission

---

## How It Works

### Trade Execution Flow

```
1. Analysts analyze market data
   ↓
2. Researchers debate (bull vs bear thesis)
   ↓
3. Trader Agent generates TradePlan
   {
     symbol: "AAPL",
     direction: "long",
     entry: 175.00,
     stop_loss: 170.00,
     take_profit: 185.00,
     position_size_pct: 0.05  // 5% of portfolio
   }
   ↓
4. Risk Manager validates plan
   ↓
5. AlpacaBrokerExecutor calculates shares
   (portfolio_value × position_size_pct) / entry_price
   ↓
6. Submit bracket order to Alpaca
   - Main order: Buy 100 shares @ market
   - Stop loss: Sell 100 shares @ $170
   - Take profit: Sell 100 shares @ $185
   ↓
7. Alpaca executes and manages the orders
   ↓
8. System tracks position in database
   ↓
9. Exit monitor checks for stop/target hits
```

### Automatic Risk Management

Every trade includes:
- **Stop Loss**: Limits downside risk
- **Take Profit**: Locks in gains
- **Position Sizing**: Based on portfolio percentage (agent-decided up to 10%)
- **Bracket Orders**: Exit orders placed simultaneously with entry

---

## File Structure

```
equity-ai-trading/
├── active_trader_llm/
│   ├── execution/
│   │   ├── __init__.py
│   │   └── alpaca_broker.py          # ← NEW: Alpaca integration
│   ├── config/
│   │   └── config_schema.py          # ← UPDATED: Added ExecutionConfig
│   ├── utils/
│   │   └── logging_json.py           # ← UPDATED: Broker order tracking
│   └── main.py                        # ← UPDATED: Integrated broker execution
├── config_alpaca_paper.yaml           # ← NEW: Paper trading config
├── ALPACA_SETUP.md                    # ← NEW: Setup guide
├── INTEGRATION_SUMMARY.md             # ← NEW: This file
└── test_alpaca_integration.py         # ← NEW: Test suite
```

---

## Quick Start

### 1. Set Up Alpaca API Keys

```bash
# Get API keys from https://alpaca.markets
export ALPACA_API_KEY="your_paper_api_key"
export ALPACA_SECRET_KEY="your_paper_secret_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
```

### 2. Test the Integration

```bash
python test_alpaca_integration.py
```

Expected output:
```
✓ Successfully connected to Alpaca Paper Trading API
✓ Successfully retrieved account info
  Portfolio Value: $100,000.00
  Cash: $100,000.00
  Buying Power: $100,000.00
✓ Successfully retrieved positions
  Total Positions: 0
✓ Position size calculated successfully
✓ Order structure validated (not submitted)
```

### 3. Run Paper Trading

```bash
python -m active_trader_llm.main --config config_alpaca_paper.yaml --cycles 1
```

This will:
1. Analyze configured stocks (AAPL, MSFT, GOOGL, AMZN, NVDA)
2. Generate trade plans
3. Submit bracket orders to Alpaca paper trading
4. Log all executions

### 4. Monitor Results

**Check Alpaca Dashboard**:
- Paper Trading: https://app.alpaca.markets/paper/dashboard/overview

**Check Local Logs**:
```bash
cat logs/trade_log.jsonl | grep '"type": "execution"' | jq .
```

---

## Safety Features

### Built-in Protections

1. **Trade Plan Validation**
   - Validates entry/stop/target prices
   - Ensures minimum risk/reward ratio (1.5:1)
   - Prevents excessive position sizes (max 10%)

2. **Risk Manager**
   - Emergency drawdown circuit breaker (20% daily loss)
   - Position concentration limits
   - Duplicate symbol prevention

3. **Bracket Orders**
   - Automatic stop loss on every trade
   - Automatic take profit
   - No manual intervention needed

4. **Paper Trading Default**
   - Config defaults to `paper_trading: true`
   - Prevents accidental live trading

---

## Configuration Options

### Execution Modes

| Mode | Broker | Real Money? | Use Case |
|------|--------|-------------|----------|
| `backtest` | None | No | Simulated execution, historical testing |
| `paper-live` | Alpaca | No | Live market, fake money (recommended for testing) |
| `live` | Alpaca | **Yes** | Real money trading ⚠️ |

### Order Types

| Type | Execution | Use Case |
|------|-----------|----------|
| `market` | Immediate at current price | Fast execution, accepts slippage |
| `limit` | Only at specified price or better | Price control, may not fill |

### Time in Force

| TIF | Behavior | Use Case |
|-----|----------|----------|
| `day` | Expires at market close | Intraday trading |
| `gtc` | Good 'til cancelled | Swing trading |
| `ioc` | Immediate or cancel | All-or-nothing fills |
| `fok` | Fill or kill | Large orders |

---

## Example Trade

Here's what happens when the system generates a trade:

### 1. Trade Plan Generated
```python
TradePlan(
    symbol="AAPL",
    strategy="momentum_breakout",
    direction="long",
    entry=175.00,
    stop_loss=170.00,
    take_profit=185.00,
    position_size_pct=0.05,  # 5% of portfolio
    time_horizon="3d",
    rationale=["RSI breakout", "Volume surge", "Above EMA50"],
    risk_reward_ratio=2.0
)
```

### 2. Position Sizing
```
Account Equity: $100,000
Position Size: 5%
Dollar Allocation: $5,000
Entry Price: $175.00
Shares: 28 (= $5,000 / $175)
```

### 3. Bracket Order Submitted
```
Main Order:    BUY 28 shares AAPL @ MARKET
Stop Loss:     SELL 28 shares AAPL @ $170.00 STOP
Take Profit:   SELL 28 shares AAPL @ $185.00 LIMIT
```

### 4. Alpaca Execution
```
Order ID: abc123-def456-ghi789
Status: filled
Filled Price: $175.02
Filled Qty: 28 shares
```

### 5. Automatic Exit Management

Alpaca monitors the position and automatically:
- Sells at $170 if stop loss is hit (loss: $140)
- Sells at $185 if take profit is hit (gain: $280)

---

## Monitoring & Maintenance

### Daily Checklist

```bash
# 1. Check positions
python -c "from active_trader_llm.execution.alpaca_broker import AlpacaBrokerExecutor; e = AlpacaBrokerExecutor(paper=True); [print(f'{p.symbol}: ${p.unrealized_pl:.2f}') for p in e.get_positions()]"

# 2. Review recent trades
tail -100 logs/trade_log.jsonl | grep execution

# 3. Check for errors
tail -100 logs/trade_log.jsonl | grep error

# 4. Account summary
python -c "from active_trader_llm.execution.alpaca_broker import AlpacaBrokerExecutor; e = AlpacaBrokerExecutor(paper=True); a = e.get_account_info(); print(f\"Portfolio: ${a['portfolio_value']}, Cash: ${a['cash']}\")"
```

### Weekly Review

1. Analyze win rate and P&L
2. Review strategy performance
3. Adjust position sizing if needed
4. Update universe based on scanner results

---

## Next Steps

### Recommended Path to Live Trading

1. **Week 1-2**: Paper trading with fixed universe
   - Monitor executions daily
   - Verify stop losses work
   - Check position sizing

2. **Week 3-4**: Paper trading with scanner enabled
   - Test with larger universe
   - Monitor cost per decision
   - Verify strategy switching

3. **Month 2**: Extended paper trading
   - Run through different market conditions
   - Track win rate and drawdown
   - Build confidence in system

4. **Going Live**: Start small
   - Use 10% of intended capital
   - Trade only 1-2 symbols initially
   - Gradually scale up

---

## Troubleshooting

### Common Issues

**Problem**: "Alpaca API credentials not provided"
```bash
# Solution: Set environment variables
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"
```

**Problem**: "Order rejected: Insufficient buying power"
```yaml
# Solution: Reduce position size in config
trade_validation:
  max_position_pct: 5.0  # Lower from 10.0
```

**Problem**: Orders not filling
```yaml
# Solution: Use market orders instead of limit
execution:
  order_type: market  # Changed from limit
```

---

## Technical Details

### Dependencies Added
- `alpaca-py==0.43.0` - Official Alpaca Python SDK

### API Endpoints Used
- **Paper Trading**: `https://paper-api.alpaca.markets`
- **Live Trading**: `https://api.alpaca.markets`

### Rate Limits
- Standard: 200 requests/minute
- Unlimited: 1000 requests/minute

The system respects rate limits automatically.

---

## Support & Resources

- **Setup Guide**: [ALPACA_SETUP.md](ALPACA_SETUP.md)
- **Test Suite**: [test_alpaca_integration.py](test_alpaca_integration.py)
- **Alpaca Docs**: https://docs.alpaca.markets/
- **Alpaca Forum**: https://forum.alpaca.markets/

---

## Summary

✅ **Complete Integration**: Your trading agent can now execute real trades through Alpaca

✅ **Safety First**: Multiple layers of protection (validation, risk management, bracket orders)

✅ **Paper Trading Ready**: Test thoroughly before risking real money

✅ **Production Ready**: When ready, switch to live trading with minimal config changes

✅ **Well Documented**: Comprehensive guides and test tools provided

**You're all set!** Start with paper trading and monitor the results closely. Good luck! 🚀
