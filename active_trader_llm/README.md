# ActiveTrader-LLM

**Cost-efficient, reasoning-based LLM agent for active stock trading** (hours–days horizon)

Multi-agent workflow with memory-driven learning and strategy switching. Inspired by TauricResearch/TradingAgents (multi-agent) and pipiku915/FinMem-LLM-StockTrading (layered memory).

---

## Overview

ActiveTrader-LLM implements a sophisticated multi-agent trading system that combines:

- **Multiple Analyst Agents**: Technical, Breadth Health, Sentiment, Macro
- **Researcher Debate**: Bull vs Bear thesis generation
- **Trader Agent**: Synthesizes signals into trade plans
- **Risk Manager**: Validates and adjusts positions
- **Memory System**: Short-term trades + long-term strategy stats
- **Strategy Monitor**: Tracks performance and triggers switches

### Key Features

- Transparent reasoning with JSON logs for every decision
- Cost control for LLM calls (max $0.05 per decision)
- Continuous learning with memory and strategy switching
- Supports backtest and paper-live modes
- No HFT - operates on 1h/1d timeframes

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/abhip2006/equity-ai-trading.git
cd equity-ai-trading/active_trader_llm

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export ANTHROPIC_API_KEY=your_api_key_here
```

### Configuration

Edit `config.yaml` to customize:
- Trading universe (default: AAPL, MSFT, SPY, QQQ)
- Risk parameters (position size, drawdown limits)
- Strategy selection
- LLM provider and model

### Run Your First Backtest

```bash
# Run a single decision cycle
python main.py --config ../config.yaml --cycles 1

# Run with learning update
python main.py --config ../config.yaml --cycles 5 --learning
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Multi-Agent Pipeline                       │
└─────────────────────────────────────────────────────────────┘

1. DATA INGESTION
   ├─ yfinance (1h/1d bars)
   ├─ SQLite caching
   └─ Retry logic with backoff

2. FEATURE ENGINEERING
   ├─ Technical indicators (RSI, MACD, EMA, Bollinger, ATR)
   ├─ Market breadth (A/D, new highs/lows, volume)
   └─ Regime detection

3. ANALYST LAYER
   ├─ TechnicalAnalyst → Signal + confidence + reasons
   ├─ BreadthHealthAnalyst → Regime determination
   ├─ SentimentAnalyst (optional) → Sentiment score
   └─ MacroAnalyst (optional) → Macro bias

4. RESEARCHER DEBATE
   ├─ BullResearcher → Bullish thesis
   └─ BearResearcher → Bearish thesis

5. TRADER AGENT
   ├─ Strategy selection (momentum/reversion/pullback/rotation)
   ├─ Entry/stop/target calculation
   └─ Position sizing

6. RISK MANAGER
   ├─ Position size limits
   ├─ Concurrent position limits
   ├─ Daily drawdown check
   ├─ Sector concentration
   └─ Min risk/reward validation

7. MEMORY & LEARNING
   ├─ Short-term: Recent N trades
   ├─ Long-term: Strategy performance by regime
   └─ Regime history

8. STRATEGY MONITOR
   ├─ Rolling performance metrics
   ├─ Degradation detection
   └─ Automated strategy switching
```

---

## Project Structure

```
active_trader_llm/
├── config/
│   ├── config_schema.py       # Pydantic configuration models
│   └── loader.py              # YAML/JSON config loader
│
├── data_ingestion/
│   └── price_volume_ingestor.py  # yfinance data fetching + caching
│
├── feature_engineering/
│   └── indicators.py          # Technical indicators + breadth features
│
├── analysts/
│   ├── technical_analyst.py   # LLM-powered technical analysis
│   ├── breadth_health_analyst.py  # Market regime determination
│   ├── sentiment_analyst.py   # Sentiment analysis (stub)
│   └── macro_analyst.py       # Macro context (stub)
│
├── researchers/
│   └── bull_bear.py          # Bull vs Bear debate agents
│
├── trader/
│   └── trader_agent.py       # Trade plan synthesis
│
├── risk/
│   └── risk_manager.py       # Risk validation and adjustment
│
├── memory/
│   └── memory_manager.py     # Layered memory (short/long-term)
│
├── learning/
│   └── strategy_monitor.py   # Performance tracking + switching
│
├── utils/
│   └── logging_json.py       # Structured JSON logging
│
├── main.py                   # Main orchestration script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Workflow

### Decision Cycle

1. **Fetch Data**: Pull latest OHLCV for universe from yfinance (with caching)
2. **Compute Features**: Calculate indicators (RSI, MACD, etc.) and breadth metrics
3. **Run Analysts**: Each analyst produces structured JSON output
4. **Debate**: Bull and Bear researchers argue the case
5. **Trade Plan**: Trader synthesizes into concrete plan with entry/stop/target
6. **Risk Check**: Risk manager validates against portfolio constraints
7. **Log Decision**: All reasoning logged to JSON for transparency
8. **(Execution)**: In paper-live mode, execute via Alpaca API

### Learning Update (Daily/EOD)

1. **Update Memory**: Log completed trades to short-term memory
2. **Update Stats**: Aggregate performance by strategy + regime
3. **Monitor**: Check if current strategy has degraded
4. **Switch**: If degraded and better alternative exists, switch strategy
5. **Log Switch**: Record reasoning for switch decision

---

## Configuration

### Key Parameters

**Risk Management**:
```yaml
risk_parameters:
  max_position_pct: 0.05        # Max 5% per position
  max_concurrent_positions: 8   # Max 8 open positions
  max_daily_drawdown: 0.10      # 10% daily stop
  min_risk_reward: 1.5          # Minimum R:R ratio
```

**Strategy Switching**:
```yaml
strategy_switching:
  window_trades: 30             # Evaluate last 30 trades
  min_win_rate: 0.35           # Min 35% win rate
  min_net_pl: 0.0              # Min break-even
  hysteresis_trades_before_switch: 10  # Cooldown period
```

**Cost Control**:
```yaml
cost_control:
  combine_symbols_per_prompt: true
  cache_data: true
  max_cost_per_decision_usd: 0.05
```

---

## Strategies

### momentum_breakout
- **Regime**: trending_bull
- **Entry**: Break above N-day high with volume
- **Exit**: ATR-based stop, 1.5+ R:R target

### mean_reversion
- **Regime**: range
- **Entry**: RSI<30 or Bollinger lower pierce
- **Exit**: Mid-band target, tight stop

### pullback
- **Regime**: mild_trend
- **Entry**: Price pulls back to EMA50 in uptrend
- **Exit**: Prior swing high, ATR stop

### sector_rotation
- **Regime**: mixed
- **Entry**: Overweight strongest sector
- **Exit**: Rotate on strength reversal

Strategies auto-switch based on performance degradation within each regime.

---

## JSON Logging

All decisions, executions, and outcomes logged in `logs/trade_log.jsonl`:

```json
{
  "type": "decision",
  "trade_id": "uuid",
  "symbol": "AAPL",
  "analyst_outputs": {...},
  "researcher_outputs": {...},
  "trade_plan": {...},
  "risk_decision": {...}
}
```

```json
{
  "type": "outcome",
  "trade_id": "uuid",
  "pnl": 315.50,
  "return_pct": 1.8,
  "exit_reason": "take_profit",
  "lessons": ["momentum strategy worked in trending_bull"]
}
```

Analyze logs with:
```python
from active_trader_llm.utils.logging_json import JSONLogger

logger = JSONLogger("logs/trade_log.jsonl")
decisions = logger.read_logs(entry_type='decision', limit=50)
```

---

## Memory System

### Short-Term Memory
- Stores last N trades with full context
- Used to inform future decisions on same symbol
- Helps avoid repeating recent mistakes

### Long-Term Memory
- Aggregated strategy performance by regime
- Win rate, avg return, Sharpe proxy, R:R
- Guides strategy selection

### Regime History
- Daily snapshots of market regime
- Tracks regime persistence and transitions

---

## Cost Optimization

Target: **< $0.05 per decision**

Techniques:
1. **Batch symbols per LLM call** when possible
2. **Cache price data** locally in SQLite
3. **Minimize call frequency** (hourly/daily, not tick-level)
4. **Use efficient models** (Claude Haiku for simple tasks, Sonnet for complex)
5. **Structured outputs** reduce token usage vs prose

Example costs (Anthropic Claude):
- Technical analysis per symbol: ~500 tokens → $0.0015
- Researcher debate: ~800 tokens → $0.0024
- Trade plan synthesis: ~600 tokens → $0.0018
- **Total per cycle (4 symbols)**: ~$0.02-0.03 ✓

---

## Performance Metrics

Target KPIs:
- **Net P/L**: Positive over 3-month simulation
- **Max Drawdown**: < 10%
- **Win Rate Improvement**: +10% after 50 trades (via learning)
- **Decision Cost**: <= $0.05 per decision
- **Reasoning Coverage**: 100% decisions logged

Track with:
```python
from active_trader_llm.learning.strategy_monitor import StrategyMonitor

monitor = StrategyMonitor(memory, config)
metrics = monitor.compute_metrics("momentum_breakout", "trending_bull")
print(f"Win Rate: {metrics.win_rate:.1%}")
print(f"Sharpe: {metrics.sharpe_proxy:.2f}")
```

---

## Extending the System

### Add a New Analyst

```python
# analysts/new_analyst.py
class NewAnalyst:
    def analyze(self, symbol, features, market_snapshot):
        # Your analysis logic
        return AnalystOutput(...)

# Update main.py
self.new_analyst = NewAnalyst()
```

### Add a New Strategy

```yaml
# config.yaml
strategies:
  - name: my_custom_strategy
    regime: custom_regime
    enabled: true
```

Implement entry/exit logic in trader prompts or as deterministic rules.

### Enable Sentiment/Macro

```yaml
enable_sentiment: true
enable_macro: true
```

Extend stub implementations in `analysts/sentiment_analyst.py` and `analysts/macro_analyst.py` with:
- FinBERT for news
- Reddit/Twitter APIs
- FRED for macro data

---

## Testing

```bash
# Run with test data
python main.py --config config.yaml --cycles 1

# Check logs
cat logs/trade_log.jsonl | jq '.type'

# View trades
sqlite3 data/trading.db "SELECT * FROM short_term_memory ORDER BY timestamp DESC LIMIT 10;"
```

---

## Troubleshooting

**No data fetched**:
- Check internet connection
- Verify yfinance is working: `python -c "import yfinance; print(yfinance.Ticker('AAPL').history())"`

**LLM errors**:
- Verify API key: `echo $ANTHROPIC_API_KEY`
- Check rate limits
- Review `logs/trade_log.jsonl` for error entries

**No trades generated**:
- Markets may genuinely have no opportunities
- Check analyst confidence levels
- Review risk manager rejections in logs

---

## Roadmap

- [ ] Full backtest engine with historical simulation
- [ ] Paper-live mode with Alpaca integration
- [ ] Web dashboard for monitoring (FastAPI + React)
- [ ] Enhanced sentiment (FinBERT, social media)
- [ ] Macro indicators (VIX, yields, currencies)
- [ ] Options strategies integration
- [ ] Multi-asset support (futures, forex)
- [ ] Advanced risk models (correlation, beta hedging)

---

## License

TBD

## Acknowledgments

Inspired by:
- **TauricResearch/TradingAgents**: Multi-agent architecture
- **pipiku915/FinMem-LLM-StockTrading**: Layered memory system

---

## Contact

For questions or contributions, please open an issue on GitHub.

**Repository**: https://github.com/abhip2006/equity-ai-trading
