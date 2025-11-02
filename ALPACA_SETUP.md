# Alpaca Trading Integration Setup Guide

This guide explains how to integrate your ActiveTrader-LLM system with Alpaca for live trading.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Getting Alpaca API Keys](#getting-alpaca-api-keys)
- [Environment Setup](#environment-setup)
- [Configuration](#configuration)
- [Running Paper Trading](#running-paper-trading)
- [Going Live](#going-live)
- [Monitoring Trades](#monitoring-trades)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

1. **Alpaca Account**: Sign up at [alpaca.markets](https://alpaca.markets/)
2. **Python 3.8+**: Ensure you have Python installed
3. **alpaca-py SDK**: Already installed in this project

```bash
pip install alpaca-py
```

---

## Getting Alpaca API Keys

### Paper Trading Keys (Recommended for Testing)

1. Log in to your [Alpaca dashboard](https://app.alpaca.markets/)
2. Navigate to **Your API Keys** in the sidebar
3. Under **Paper Trading**, click **Generate New Key**
4. Copy your **API Key** and **Secret Key**
5. Keep these secure - they provide full access to your paper trading account

### Live Trading Keys (Use with Caution)

1. In the Alpaca dashboard, switch to **Live Trading**
2. Complete any required account verification
3. Navigate to **Your API Keys** under Live Trading
4. Click **Generate New Key**
5. Copy your **Live API Key** and **Secret Key**

⚠️ **WARNING**: Live keys control real money. Never commit them to version control!

---

## Environment Setup

### 1. Create `.env` File

In your project root, create a `.env` file:

```bash
# Alpaca API Credentials (Paper Trading)
ALPACA_API_KEY=your_paper_api_key_here
ALPACA_SECRET_KEY=your_paper_secret_key_here

# Anthropic API Key (for LLM agents)
ANTHROPIC_API_KEY=your_anthropic_key_here

# Optional: Specify base URL (auto-detected if not set)
# ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### 2. Load Environment Variables

Add to your shell profile (`.bashrc`, `.zshrc`, etc.):

```bash
# Load environment variables
export $(cat .env | xargs)
```

Or use Python's `python-dotenv`:

```bash
pip install python-dotenv
```

Then in your code:
```python
from dotenv import load_dotenv
load_dotenv()
```

### 3. Secure Your `.env` File

Add to `.gitignore`:
```
.env
*.env
```

---

## Configuration

### Use the Provided Configuration

We've created `config_alpaca_paper.yaml` for you:

```yaml
mode: paper-live  # Use paper trading mode

execution:
  broker: alpaca
  paper_trading: true  # IMPORTANT: Keep true for testing
  order_type: market
  time_in_force: day

  # API credentials from environment variables
  alpaca_api_key_env: ALPACA_API_KEY
  alpaca_secret_key_env: ALPACA_SECRET_KEY
  alpaca_base_url: null  # Auto-selected
```

### Configuration Options

#### Modes
- `backtest`: Simulated execution only (no broker)
- `paper-live`: Alpaca paper trading (recommended for testing)
- `live`: Real money trading ⚠️

#### Order Types
- `market`: Execute immediately at current market price
- `limit`: Execute only at specified price or better

#### Time in Force
- `day`: Order expires at end of trading day
- `gtc`: Good 'til cancelled (stays active until filled or cancelled)
- `ioc`: Immediate or cancel
- `fok`: Fill or kill (all or nothing)

---

## Running Paper Trading

### Step 1: Verify API Connection

Test your API credentials:

```bash
cd active_trader_llm
python -c "from execution.alpaca_broker import AlpacaBrokerExecutor; executor = AlpacaBrokerExecutor(paper=True); print(executor.get_account_info())"
```

You should see your account info printed.

### Step 2: Run a Decision Cycle

```bash
python -m active_trader_llm.main --config config_alpaca_paper.yaml --cycles 1
```

This will:
1. Analyze the configured universe (AAPL, MSFT, etc.)
2. Generate trade plans
3. Submit orders to Alpaca paper trading
4. Log all executions

### Step 3: Monitor Your Trades

Check your Alpaca paper trading dashboard:
- [Paper Trading Dashboard](https://app.alpaca.markets/paper/dashboard/overview)

Or programmatically:

```python
from execution.alpaca_broker import AlpacaBrokerExecutor

executor = AlpacaBrokerExecutor(paper=True)

# Get all positions
positions = executor.get_positions()
for pos in positions:
    print(f"{pos.symbol}: {pos.qty} shares @ ${pos.avg_entry_price:.2f}")
    print(f"  P&L: ${pos.unrealized_pl:.2f} ({pos.unrealized_pl_pct:.2%})")
```

### Step 4: Review Logs

All trades are logged to `logs/trade_log.jsonl`:

```bash
# View execution logs
cat logs/trade_log.jsonl | grep '"type": "execution"' | jq .
```

---

## Going Live

⚠️ **WARNING**: Live trading involves real money. Only proceed if you:
1. Have thoroughly tested in paper trading
2. Understand the risks
3. Are comfortable with potential losses
4. Have verified all strategies

### Steps for Live Trading

1. **Update API Keys** in `.env`:
```bash
ALPACA_API_KEY=your_LIVE_api_key_here
ALPACA_SECRET_KEY=your_LIVE_secret_key_here
```

2. **Create Live Configuration** (`config_alpaca_live.yaml`):
```yaml
mode: live
execution:
  broker: alpaca
  paper_trading: false  # LIVE TRADING
  # ... rest of config
```

3. **Start Small**:
   - Test with a small position size
   - Limit your universe to 1-2 symbols initially
   - Monitor closely for the first few cycles

4. **Run Live System**:
```bash
python -m active_trader_llm.main --config config_alpaca_live.yaml --cycles 1
```

---

## Monitoring Trades

### Real-time Monitoring

The system logs all actions. Monitor with:

```bash
# Tail logs in real-time
tail -f logs/trade_log.jsonl

# Filter for errors
tail -f logs/trade_log.jsonl | grep '"type": "error"'

# Filter for executions
tail -f logs/trade_log.jsonl | grep '"type": "execution"'
```

### Checking Positions

```python
from execution.alpaca_broker import AlpacaBrokerExecutor

executor = AlpacaBrokerExecutor(paper=True)  # or False for live

# Get account info
account = executor.get_account_info()
print(f"Portfolio Value: ${account['portfolio_value']:,.2f}")
print(f"Cash: ${account['cash']:,.2f}")
print(f"Buying Power: ${account['buying_power']:,.2f}")

# Get all positions
positions = executor.get_positions()
print(f"\nOpen Positions: {len(positions)}")
for pos in positions:
    print(f"{pos.symbol}: {pos.qty} shares @ ${pos.avg_entry_price:.2f}, P&L: ${pos.unrealized_pl:.2f}")
```

### Checking Orders

```python
# Get order status
order_id = "your-order-id-here"
status = executor.get_order_status(order_id)
print(status)
```

---

## Bracket Orders

All trades automatically include:
- **Stop Loss**: Automatic exit if price moves against you
- **Take Profit**: Automatic exit at target price

Example trade plan execution:
```
Entry: $175.00
Stop Loss: $170.00 (risk: $5.00)
Take Profit: $185.00 (reward: $10.00)
Risk/Reward: 2:1
```

When submitted to Alpaca, this becomes a **bracket order** with:
1. Main order: Buy 100 shares at market
2. Stop loss order: Sell 100 shares if price drops to $170
3. Take profit order: Sell 100 shares if price rises to $185

---

## Troubleshooting

### Common Issues

#### 1. "Alpaca API credentials not provided"

**Solution**: Ensure environment variables are set:
```bash
echo $ALPACA_API_KEY
echo $ALPACA_SECRET_KEY
```

If empty, reload your `.env` file:
```bash
export $(cat .env | xargs)
```

#### 2. "Failed to connect to Alpaca"

**Possible causes**:
- Invalid API keys
- Using live keys with `paper=True` or vice versa
- Network issues

**Solution**:
```python
# Verify keys match mode
from execution.alpaca_broker import AlpacaBrokerExecutor
executor = AlpacaBrokerExecutor(paper=True)  # Should use paper keys
```

#### 3. "Order rejected: Insufficient buying power"

**Solution**: Your account doesn't have enough cash for the trade. Either:
- Reduce `max_position_pct` in config
- Add more funds to your account
- Close existing positions

#### 4. "Market is closed"

**Solution**: Alpaca only accepts orders during market hours (9:30 AM - 4:00 PM ET).

For after-hours testing, use `time_in_force: gtc` to queue orders for next market open.

#### 5. Import Error: "No module named 'execution.alpaca_broker'"

**Solution**: Run from project root:
```bash
cd /path/to/equity-ai-trading
python -m active_trader_llm.main --config config_alpaca_paper.yaml
```

---

## Best Practices

### 1. Always Test in Paper Trading First
Never deploy directly to live trading without extensive paper trading validation.

### 2. Start with Conservative Position Sizes
Keep `max_position_pct` at 5% or lower until you're confident in the system.

### 3. Monitor Daily
Check positions and P&L daily, especially during the first weeks.

### 4. Review Logs Regularly
```bash
# Weekly review of all trades
cat logs/trade_log.jsonl | jq 'select(.type=="execution")' > weekly_trades.json
```

### 5. Enable Trade Validation
Keep `trade_validation.enabled: true` in your config to prevent bad trades.

### 6. Use Emergency Stop Loss
Set `risk_parameters.emergency_drawdown_limit: 0.20` (20%) to halt trading on large losses.

### 7. Version Control Your Config
Track changes to your configuration:
```bash
git add config_alpaca_paper.yaml
git commit -m "Updated position sizing"
```

---

## Additional Resources

- [Alpaca API Documentation](https://docs.alpaca.markets/)
- [alpaca-py SDK Reference](https://alpaca.markets/sdks/python/)
- [Alpaca Paper Trading](https://docs.alpaca.markets/docs/paper-trading)
- [Order Types Guide](https://docs.alpaca.markets/docs/orders-at-alpaca)

---

## Support

If you encounter issues:

1. Check the [Alpaca Community Forum](https://forum.alpaca.markets/)
2. Review logs in `logs/trade_log.jsonl`
3. Test with the example script in `active_trader_llm/execution/alpaca_broker.py`

---

## Safety Checklist

Before going live, verify:

- [ ] Tested extensively in paper trading (minimum 30 days recommended)
- [ ] Position sizing is appropriate for your account
- [ ] Stop losses are properly set
- [ ] Emergency drawdown limit is configured
- [ ] You understand all configuration options
- [ ] You've reviewed the code and understand what it does
- [ ] You're prepared for potential losses
- [ ] You have a plan to monitor and adjust

**Remember**: Past performance in paper trading does not guarantee future results in live trading. Only risk capital you can afford to lose.
