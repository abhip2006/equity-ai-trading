# Two-Stage Market Scanner

A cost-efficient, anti-hallucination market scanner that analyzes 5000+ stocks with only 6-11 LLM calls ($0.16/scan, 3-5 minutes).

## Architecture Overview

The scanner separates **calculation** from **interpretation**, ensuring the LLM never generates technical data—only interprets pre-calculated values.

### Design Principles

1. **LLM Interprets, Never Calculates**: All technical indicators are calculated by `pandas_ta` library, not LLM
2. **Two-Stage Filtering**: Broad filter → Deep analysis minimizes expensive LLM calls
3. **Batched Processing**: 15 stocks per LLM call optimizes cost/performance
4. **Anti-Hallucination**: LLM receives formatted indicator values, never raw OHLCV data

---

## Two-Stage Workflow

### Stage 1: Market-Wide Summary (1 LLM Call)

**What Happens:**
1. **Load Universe**: Fetch 5000+ optionable stocks from Alpaca API (cached for 24h)
2. **Calculate Metrics** (NO LLM):
   - For each stock: 5d price change, volume ratio, MA positioning, 52w high distance
   - Aggregate by sector: breadth, momentum, volume patterns
3. **LLM Analysis** (1 call):
   - Interprets market statistics
   - Returns filtering guidance:
     ```json
     {
       "market_bias": "bullish",
       "focus_sectors": ["Technology", "Healthcare"],
       "focus_patterns": ["breakouts", "momentum"],
       "filtering_criteria": {
         "volume_ratio_threshold": 2.5,
         "distance_from_52w_high_threshold_pct": 7,
         "min_price_change_5d_pct": 3
       }
     }
     ```

**Key Point**: LLM decides thresholds based on market conditions, not hardcoded values!

### Stage 2: Filter & Deep Analysis (5-10 LLM Calls)

#### Part A: Programmatic Filtering (NO LLM)
- Apply Stage 1 guidance thresholds:
  - Volume > 2.5x average?
  - Within 7% of 52-week high?
  - 5-day momentum > 3%?
  - In focus sectors?
- Result: ~50-200 candidates

#### Part B: Batch Deep Analysis (5-10 LLM calls)
- **Fetch Real Indicators** (NO LLM):
  - Use `pandas_ta` to calculate RSI, MACD, ATR, BBands, etc.
  - Calculate moving averages programmatically
- **Batch to LLM** (15 stocks per call):
  ```
  STOCK #1: NVDA
  Price: $485.23
  Moving Averages:
    - SMA20: $478.12 (price +1.5% from MA)
    - EMA50: $465.34 (price +4.3% from MA)
  Momentum:
    - RSI(14): 67.3
    - MACD Hist: +0.45
  Volume: 2.1x average
  ```
- **LLM Interprets** and returns:
  ```json
  {
    "favorable_symbols": ["NVDA", "MSFT"],
    "reasoning": {
      "NVDA": "Breakout above EMA50 with 2.1x volume confirming momentum"
    }
  }
  ```

---

## Anti-Hallucination Design

| What LLM Does NOT Do | What Provides Real Data |
|---------------------|------------------------|
| ❌ Calculate RSI, MACD, ATR | ✅ `pandas_ta` library |
| ❌ Determine stock sectors | ✅ Alpaca API |
| ❌ Generate price data | ✅ Alpaca/yfinance APIs |
| ❌ Calculate moving averages | ✅ Programmatic calculation |
| ❌ Invent support/resistance | ✅ Calculated from OHLCV |

**What LLM DOES:**
- ✅ Interpret market-wide statistics → decide filtering thresholds
- ✅ Identify favorable patterns from pre-calculated indicators
- ✅ Select high-probability setups based on technical confluence

---

## Module Structure

```
active_trader_llm/scanners/
├── scanner_db.py              # Database: TradableUniverse, ScanResults
├── universe_loader.py          # Load & cache 5000+ stocks from Alpaca
├── market_aggregator.py        # Calculate market statistics (NO LLM)
├── stage1_analyzer.py          # Stage 1 LLM: Market → Filtering guidance
├── programmatic_filter.py      # Stage 2A: Apply filters (NO LLM)
├── raw_data_scanner.py         # Fetch indicators via pandas_ta (NO LLM)
├── stage2_analyzer.py          # Stage 2B: Batch analysis (5-10 LLM calls)
└── scanner_orchestrator.py     # Main coordinator
```

### Database Schema

**TradableUniverse Table:**
- Caches 5000+ optionable stocks
- Fields: symbol, sector, market_cap, avg_volume_20d, optionable, updated_at
- Refreshed every 24 hours

**ScanResults Table:**
- Historical scan tracking
- Fields: scan_id, timestamp, stage1_guidance, filtered_count, final_candidates, execution_time, llm_calls, total_cost

---

## Usage

### Quick Start

```python
from scanners.scanner_orchestrator import ScannerOrchestrator
from data_ingestion.price_volume_ingestor import PriceVolumeIngestor

# Initialize
data_fetcher = PriceVolumeIngestor()
orchestrator = ScannerOrchestrator(
    data_fetcher=data_fetcher,
    alpaca_api_key="your_alpaca_key",
    alpaca_secret_key="your_alpaca_secret",
    anthropic_api_key="your_anthropic_key"
)

# Run scan
candidates = orchestrator.run_full_scan(
    force_refresh_universe=False,
    refresh_hours=24,
    batch_size=15,
    max_batches=10,
    max_candidates=150
)

print(f"Final candidates: {candidates}")
```

### Configuration

In [config.yaml](../../config.yaml):

```yaml
scanner:
  enabled: true  # Set to true to use scanner mode
  universe_source: alpaca_optionable
  refresh_universe_hours: 24
  stage1:
    target_candidate_count: 100
  stage2:
    batch_size: 15
    max_batches: 10
    max_candidates: 150
```

### Environment Variables

Copy [.env.example](../../.env.example) to `.env`:

```bash
# Required
ANTHROPIC_API_KEY=your_anthropic_key_here
ALPACA_API_KEY=your_alpaca_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

---

## Performance Metrics

**Target Performance:**
- **Stocks Analyzed**: 5000+
- **LLM Calls**: 6-11 total
  - Stage 1: 1 call
  - Stage 2: 5-10 calls (15 stocks/call)
- **Cost**: ~$0.16 per scan
- **Time**: 3-5 minutes
- **Final Candidates**: 10-30 high-probability setups

**vs. Traditional Approach:**
- ❌ 5000+ LLM calls = $20+ / scan
- ❌ 2-4 hours execution time
- ❌ High hallucination risk

---

## Implementation Status

### ✅ Completed (Core Modules)

- ✅ Database schema ([scanner_db.py](scanner_db.py))
- ✅ Universe loading & caching ([universe_loader.py](universe_loader.py))
- ✅ Market aggregation ([market_aggregator.py](market_aggregator.py))
- ✅ Stage 1 LLM analyzer ([stage1_analyzer.py](stage1_analyzer.py))
- ✅ Programmatic filtering ([programmatic_filter.py](programmatic_filter.py))
- ✅ Raw data scanning with pandas_ta ([raw_data_scanner.py](raw_data_scanner.py))
- ✅ Stage 2 LLM batch analyzer ([stage2_analyzer.py](stage2_analyzer.py))
- ✅ Scanner orchestrator ([scanner_orchestrator.py](scanner_orchestrator.py))
- ✅ Configuration schemas
- ✅ Environment setup

### 🔄 Pending (Integration Tasks)

- ⏳ Integrate with `price_volume_ingestor.py` for bulk data fetching
- ⏳ Add Alpaca API integration to `price_volume_ingestor.py`
- ⏳ Modify `main.py` to support scanner mode
- ⏳ Extend `memory_manager.py` to track scan performance
- ⏳ Create `scanner_optimizer.py` for learning from results
- ⏳ Unit tests for all scanner components
- ⏳ Integration tests for full workflow

---

## Next Steps

### 1. Complete Data Integration

**Update [price_volume_ingestor.py](../data_ingestion/price_volume_ingestor.py):**
- Add Alpaca API support alongside yfinance
- Implement bulk fetching for 5000+ symbols
- Add efficient caching strategy

**Example:**
```python
def fetch_bulk_prices(self, symbols: List[str], interval='1d'):
    """Fetch prices for large symbol lists efficiently"""
    # Batch into chunks of 100
    # Use Alpaca bulk bars API
    # Cache results
```

### 2. Integrate with Main Agent

**Update [main.py](../main.py):**
```python
from scanners.scanner_orchestrator import ScannerOrchestrator

# Add scanner mode
if config.scanner.enabled:
    universe = orchestrator.run_full_scan()
else:
    universe = config.data_sources.universe

# Continue with existing agent workflow
for symbol in universe:
    # Run TechnicalAnalyst, Researchers, Trader, RiskManager
```

### 3. Add Learning & Optimization

**Create [learning/scanner_optimizer.py](../learning/):**
- Track which Stage 1 thresholds led to profitable trades
- Optimize filtering criteria based on historical performance
- Suggest threshold adjustments per market regime

### 4. Testing

**Create [tests/test_scanners/](../../tests/):**
- Unit tests for each scanner module
- Mock LLM responses for deterministic testing
- Integration test for full scan workflow
- Performance benchmarks

---

## Example Output

```
============================================================
Starting Two-Stage Market Scan: scan_20240115_140532
============================================================

[STEP 1] Loading tradable universe...
Loaded 5247 tradable stocks

[STEP 2] Stage 1: Calculating market-wide statistics (NO LLM)...
Market summary generated:
  Total stocks: 5247
  Market breadth: 0.45
  Sectors analyzed: 11

[STEP 3] Stage 1: LLM analyzing market (1 LLM call)...
Stage 1 Guidance:
  Market bias: bullish
  Focus sectors: ['Technology', 'Healthcare']
  Focus patterns: ['breakouts', 'momentum']
  Volume threshold: 2.5x
  52w high threshold: 7%

[STEP 4] Stage 2A: Applying programmatic filters (NO LLM)...
Filter results:
  Initial: 5247
  Filtered: 127

[STEP 5] Stage 2B: Deep analysis in batches...
Calculating indicators for 127 candidates...
Batch 1/9: 15 stocks → 8 favorable
Batch 2/9: 15 stocks → 6 favorable
...
Batch 9/9: 7 stocks → 4 favorable

============================================================
SCAN COMPLETE
============================================================
Final candidates: 42
Candidates: ['NVDA', 'MSFT', 'AAPL', 'GOOGL', ...]
Execution time: 245 seconds
LLM calls: 10
Estimated cost: $0.16
============================================================
```

---

## Troubleshooting

**Issue: "Alpaca client not initialized"**
- Ensure `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` are set in `.env`
- Verify API keys are valid at https://alpaca.markets

**Issue: "pandas_ta not installed"**
```bash
cd active_trader_llm
pip install pandas-ta==0.3.14b
```

**Issue: Stage 1 LLM fails**
- Fallback guidance will be used automatically
- Check `ANTHROPIC_API_KEY` is valid
- Review logs for API errors

**Issue: No candidates found**
- Market conditions may be unfavorable
- Try adjusting filters in [config.yaml](../../config.yaml)
- Check Stage 1 guidance thresholds in logs

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   STAGE 1: MARKET SUMMARY                │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐         ┌───────────────────┐         │
│  │ Universe     │ 5000+   │ Market            │  NO LLM │
│  │ Loader       ├────────>│ Aggregator        │         │
│  │ (Alpaca API) │ stocks  │ (Calculate stats) │         │
│  └──────────────┘         └─────────┬─────────┘         │
│                                      │                    │
│                                      v                    │
│                            ┌─────────────────┐           │
│                            │ Stage1 Analyzer │  1 LLM    │
│                            │ (LLM interprets)│  CALL     │
│                            └────────┬────────┘           │
│                                     │                     │
│                                     v                     │
│                         Filtering Guidance                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              STAGE 2: FILTER & DEEP ANALYSIS             │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────┐        ┌──────────────────┐        │
│  │ Programmatic    │  NO    │ 50-200           │        │
│  │ Filter          │  LLM   │ Candidates       │        │
│  │ (Apply guidance)│        │                  │        │
│  └─────────────────┘        └────────┬─────────┘        │
│                                       │                   │
│                                       v                   │
│                         ┌──────────────────────┐         │
│                         │ Raw Data Scanner     │  NO LLM │
│                         │ (pandas_ta indicators)│        │
│                         └──────────┬───────────┘         │
│                                    │                      │
│                                    v                      │
│                         ┌──────────────────┐             │
│                         │ Stage2 Analyzer  │  5-10 LLM   │
│                         │ (Batch: 15/call) │  CALLS      │
│                         └──────────┬───────┘             │
│                                    │                      │
│                                    v                      │
│                            Final Candidates               │
│                            (10-30 stocks)                 │
└─────────────────────────────────────────────────────────┘
```

---

## License

Part of the ActiveTrader-LLM project. See main [README](../../README.md) for details.
