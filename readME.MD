# Equity AI Trading Platform

AI-powered equity and options trading system built on Alpaca Paper Trading API.

## Overview

This platform implements an autonomous trading system for equity markets with a focus on options trading strategies. The system leverages the Alpaca trading API for market data, order execution, and portfolio management, with a modern tech stack built on FastAPI and Next.js.

**Note**: This is a redesign of the ai-trading-platform with a completely reimagined agent architecture. The Alpaca integration and core infrastructure are preserved while the agent decision-making logic will be built from scratch.

---

## Alpaca Integration

### API Configuration

The platform uses Alpaca Paper Trading with the following setup:

**Environment Variables**:
```bash
ALPACA_API_KEY=your_paper_trading_api_key_here
ALPACA_SECRET_KEY=your_paper_trading_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_WS_URL=wss://stream.data.alpaca.markets/v2/iex
```

**Python SDKs**:
- `alpaca-py==0.18.0` - Official Alpaca Python SDK
- `alpaca-trade-api==3.1.1` - Legacy API support

### Alpaca Clients

Three primary clients are initialized:

1. **Trading Client** - Orders and account management
   ```python
   TradingClient(api_key, secret_key, paper=True)
   ```

2. **Stock Data Client** - Historical and real-time market data
   ```python
   StockHistoricalDataClient(api_key, secret_key)
   ```

3. **Options Data Client** - Options chains and Greeks
   ```python
   OptionHistoricalDataClient(api_key, secret_key)
   ```

### Core API Capabilities

#### Account Management
- `get_account()` - Fetch account status, cash, equity, buying power, margins

#### Position Management
- `get_positions()` / `get_all_positions()` - Current positions with P&L
- Special handling for options with missing prices via bid/ask snapshots

#### Order Execution
- `place_market_order(symbol, qty, side, time_in_force)` - Market orders
- `place_limit_order(symbol, qty, side, limit_price, time_in_force)` - Limit orders
- `place_option_order(option_symbol, qty, side, order_type, limit_price)` - Options orders
- `place_option_spread_order(legs)` - Multi-leg spreads

#### Order Management
- `get_order(order_id)` - Fetch order details
- `get_all_orders(status)` - Filter by status (open/closed)
- `cancel_order(order_id)` - Cancel pending orders

#### Market Data - Stocks
- `get_latest_quote(symbol)` - Real-time bid/ask quotes
- `get_bars(symbol, timeframe, start, end, limit)` - Historical OHLCV bars
  - Supports: 1Min, 5Min, 15Min, 1Hour, 1Day timeframes
  - Returns: timestamp, open, high, low, close, volume, trade_count, vwap

#### Market Data - Options
- `get_option_chain(underlying_symbol, expiration_date)` - Full options chain
- `get_option_quote(option_symbol)` - Option quote data
- `get_option_snapshot(option_symbol)` - Complete option data including Greeks
  - Returns: delta, gamma, theta, vega, rho, implied_volatility

#### Asset Discovery
- `get_all_optionable_stocks(only_tradable, return_with_sectors)` - Tradable universe
  - Fetches all active US equities (~5000+ stocks)
  - Optional sector/industry classification
- `get_asset_info(symbol)` - Asset details (name, sector, industry, exchange)

#### WebSocket Streaming
- `start_websocket_stream(symbols, handlers)` - Real-time market data
- `stop_websocket_stream()` - Stop streaming

### OCC Option Symbol Format

The platform parses OCC-formatted option symbols:

**Format**: `UNDERLYING + YYMMDD + C/P + STRIKE*1000`

**Example**: `SPY251121P00550000` = SPY Put, Nov 21 2025, $550 strike

Custom parser extracts:
- Underlying symbol
- Expiration date
- Option type (call/put)
- Strike price

### Infrastructure Resilience

**Circuit Breaker Pattern**:
```python
alpaca_circuit_breaker = CircuitBreaker(
    name="alpaca_api",
    failure_threshold=5,  # Open after 5 failures
    timeout_seconds=60    # Retry after 60 seconds
)
```

**Retry Logic**:
- All API calls decorated with `@retry_on_infrastructure_failure`
- Configurable max attempts with exponential backoff
- Only retries network/timeout failures, not business logic errors

---

## Technology Stack

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **Database**: PostgreSQL 14+ with TimescaleDB
- **Cache/Queue**: Redis
- **Task Queue**: Celery
- **ORM**: SQLAlchemy 2.0
- **Trading API**: Alpaca Paper Trading
- **WebSocket**: python-socketio, websockets

### Frontend
- **Framework**: Next.js 14 (React 18)
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **HTTP Client**: Axios
- **UI Components**: Radix UI, Lucide Icons

### Data Processing
- **Market Data**: pandas, numpy
- **Indicators**: pandas-ta, scipy
- **Options Pricing**: Custom Greeks calculations

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (Next.js)                       │
│  Dashboard │ Charts │ Trade History │ Real-time Updates     │
└─────────────────────────────────────────────────────────────┘
                          │ HTTP/WebSocket
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Backend (FastAPI)                           │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            API Layer (FastAPI Routes)                 │  │
│  │  /account  /trades  /portfolio  /models  /strategies │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Service Layer                            │  │
│  │  • AlpacaService  • TradingEngine                    │  │
│  │  • WebSocketHandler • CRUD Operations                │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Tools & Utilities                          │  │
│  │  • OptionsTools      • TradingView Indicators        │  │
│  │  • MarketScanners    • Quant Strategies              │  │
│  │  • Web Research      • Options Indicators            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
┌──────────────────┐              ┌─────────────────────┐
│   PostgreSQL     │              │  Alpaca Markets API │
│   TimescaleDB    │              │  • Trading Client   │
│                  │              │  • Stock Data       │
│  • Trades        │              │  • Options Data     │
│  • Signals       │              │  • WebSocket Stream │
│  • Portfolio     │              └─────────────────────┘
│  • Metrics       │
└──────────────────┘
```

---

## Project Structure

```
equity-ai-trading/
├── backend/                    # Python FastAPI backend
│   ├── alembic/               # Database migrations
│   ├── app/
│   │   ├── api/               # FastAPI route handlers
│   │   │   ├── routes/        # Account, trading endpoints
│   │   │   ├── models.py      # Model management endpoints
│   │   │   ├── portfolio.py   # Portfolio endpoints
│   │   │   ├── strategies.py  # Strategy endpoints
│   │   │   └── trades.py      # Trade history endpoints
│   │   ├── core/              # Core configuration
│   │   │   ├── config.py      # Settings & environment vars
│   │   │   ├── database.py    # SQLAlchemy setup
│   │   │   ├── logging.py     # Logging configuration
│   │   │   └── celery_app.py  # Celery task queue
│   │   ├── models/            # Database ORM models
│   │   │   └── orm_models.py  # SQLAlchemy models
│   │   ├── services/          # Business logic layer
│   │   │   ├── alpaca_client.py      # Alpaca API integration
│   │   │   ├── trading_engine.py     # Trade execution engine
│   │   │   ├── websocket_handler.py  # WebSocket streaming
│   │   │   └── crud.py              # Database CRUD operations
│   │   ├── tools/            # Trading tools
│   │   │   ├── options_tools.py         # Options trading wrapper
│   │   │   ├── tradingview_indicators.py # TradingView indicators
│   │   │   ├── options_indicators.py    # Options-specific indicators
│   │   │   ├── market_scanners.py       # Market screening
│   │   │   └── quant_strategies.py      # Quantitative models
│   │   └── utils/            # Utility functions
│   ├── requirements.txt      # Python dependencies
│   └── .env.example          # Environment template
│
├── frontend/                 # Next.js 14 React frontend
│   ├── app/                  # Next.js app directory
│   ├── components/           # React components
│   ├── lib/                  # Utility libraries
│   └── package.json          # Node dependencies
│
└── docker-compose.yml        # Service orchestration
```

---

## Database Schema

### Core Models

**Trade** - Trade execution records
- Order details (symbol, side, quantity, price)
- Alpaca order ID linkage
- Entry/exit prices and P&L tracking
- Option details (strike, expiration, type, Greeks)

**Strategy** - Trading strategy definitions
- Strategy parameters and configuration
- Performance tracking

**TradingSignal** - Generated trading signals
- Signal metadata and reasoning
- Confidence scores

**PortfolioSnapshot** - Portfolio state snapshots
- Time-series portfolio values
- Position tracking over time

**PerformanceMetric** - Performance analytics
- Sharpe ratio, max drawdown, win rate
- Strategy-level metrics

**TradableUniverse** - Cached tradable assets
- Pre-filtered liquid stocks (~2,000 from 12,000+)
- Liquidity metrics (volume, market cap, ATR)
- Updated weekly to reduce API calls

---

## API Endpoints

### Account Endpoints
- `GET /api/account/status` - Account balance, positions, P&L
- `GET /api/account/positions` - Current open positions

### Trading Endpoints
- `GET /api/trades/recent?limit=20` - Recent trade history

### Portfolio Endpoints
- `GET /api/portfolio` - Portfolio summary and metrics

### WebSocket Endpoint
- `WS /ws` - Real-time updates for market data, trades, positions

---

## Data Structures

### Account Object
```python
{
    'account_number': str,
    'status': 'ACTIVE',
    'currency': 'USD',
    'cash': float,
    'portfolio_value': float,
    'buying_power': float,
    'equity': float,
    'initial_margin': float,
    'maintenance_margin': float
}
```

### Position Object
```python
{
    'symbol': str,
    'qty': float,
    'side': 'long' | 'short',
    'avg_entry_price': float,
    'current_price': float,
    'market_value': float,
    'unrealized_pl': float,
    'unrealized_plpc': float,  # Percentage
    'cost_basis': float
}
```

### Option Contract
```python
{
    'symbol': str,  # OCC format
    'underlying_symbol': str,
    'strike': float,
    'expiration': date,
    'type': 'call' | 'put',
    'bid': float,
    'ask': float,
    'delta': float,
    'gamma': float,
    'theta': float,
    'vega': float,
    'implied_volatility': float,
    'days_to_expiration': int,
    'moneyness': 'ITM' | 'ATM' | 'OTM'
}
```

---

## Core Dependencies

### Backend (Python)
```
fastapi==0.109.0
uvicorn[standard]==0.27.0
sqlalchemy==2.0.25
psycopg2-binary==2.9.9
alembic==1.13.1
redis==5.0.1
celery==5.3.6
websockets==12.0

# Alpaca Trading
alpaca-py==0.18.0
alpaca-trade-api==3.1.1

# Data Processing
pandas==2.2.0
numpy==1.26.3
pandas-ta==0.3.14b
scipy==1.11.4

# HTTP & Utilities
httpx==0.26.0
python-dotenv==1.0.0
pytz==2024.1
```

### Frontend (Node.js)
```
next@14.1.0
react@18.2.0
recharts@2.10.3
axios@1.6.5
tailwindcss@3.3.0
```

---

## Configuration

### Environment Variables

**Required**:
```bash
# Alpaca API
ALPACA_API_KEY=your_paper_trading_api_key
ALPACA_SECRET_KEY=your_paper_trading_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_WS_URL=wss://stream.data.alpaca.markets/v2/iex

# Database
POSTGRES_USER=trading_user
POSTGRES_PASSWORD=trading_password
POSTGRES_DB=equity_ai_trading
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
DATABASE_URL=postgresql://user:pass@localhost:5432/equity_ai_trading

# Redis
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Application
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
SECRET_KEY=generate_secure_random_key

# Trading Configuration
DEFAULT_CAPITAL=100000
MAX_POSITION_SIZE_PCT=0.05
STOP_LOSS_PCT=0.05
MAX_DAILY_LOSS_PCT=0.10
```

### Docker Compose Services
1. **postgres** - TimescaleDB for time-series data
2. **redis** - Cache and message broker
3. **backend** - FastAPI application
4. **celery-worker** - Background task processing
5. **celery-beat** - Scheduled task runner
6. **frontend** - Next.js dashboard

---

## Basic Workflow

### 1. System Startup
- FastAPI initializes database connections
- WebSocket handler connects to Alpaca market data stream
- Trading engine starts monitoring loop

### 2. Account Status Flow
```
Frontend Request → /api/account/status
                → alpaca_service.get_account()
                → Alpaca Trading Client API
                → Returns: cash, equity, buying_power, positions
                → Frontend updates dashboard
```

### 3. Market Data Flow
```
Trading Engine → alpaca_service.get_bars(symbol, timeframe)
              → Alpaca Stock Data Client
              → Returns: OHLCV data
              → Stored in database
              → Broadcast via WebSocket to frontend
```

### 4. Trade Execution Flow
```
Signal Generation → Validation
                 → alpaca_service.place_market_order()
                 → Alpaca Trading Client
                 → Order submitted
                 → Monitor order status
                 → Update database when filled
                 → WebSocket notification to frontend
```

### 5. Options Trading Flow
```
1. Get Options Chain:
   → alpaca_service.get_option_chain()
   → Returns: strikes, expirations, types

2. Enrich with Greeks:
   → alpaca_service.get_option_snapshot()
   → Returns: delta, gamma, theta, vega, IV

3. Execute Trade:
   → alpaca_service.place_option_order()
   → Monitor fill status
   → Update position tracking
```

---

## Performance Optimizations

1. **Connection Pooling**: Database pool_size=10, max_overflow=20
2. **Circuit Breaker**: Prevents cascading failures on Alpaca API issues
3. **Retry Logic**: Exponential backoff on transient failures
4. **Greeks Fetching**: Limited to first 30 contracts to prevent API hangs
5. **Caching**: Redis for frequently accessed data

---

## Getting Started

Documentation for setup and deployment coming soon...

## License

TBD
