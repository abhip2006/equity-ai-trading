#!/usr/bin/env python3
"""
FastAPI Backend for Trading Dashboard
Serves real-time trading data from SQLite databases and JSON logs
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Set
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging
import asyncio
import os
from dotenv import load_dotenv
from collections import defaultdict
import time

# Import battle system components
from battle_system import BattleOrchestrator, ModelConfig, ModelStatus
from battle_models import (
    LeaderboardResponse,
    LeaderboardEntry,
    ModelsListResponse,
    ModelInfo,
    SystemStatusResponse,
    PositionsListResponse,
    PositionResponse,
    TradesListResponse,
    TradeResponse,
    EquityCurveResponse,
    EquityCurvePoint,
    ModelMetricsResponse,
    ComparePositionsResponse,
    DecisionComparisonResponse,
    DecisionResponse,
    ModelDecisionsResponse,
    CompareEquityCurvesResponse,
    WebSocketMessage
)

# Load environment variables from project root
import pathlib
load_dotenv(pathlib.Path(__file__).parent.parent / '.env')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Trading Dashboard API", version="1.0.0")

# Get CORS origins from environment or use defaults
allowed_origins_env = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5173')
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(',')]

# CORS middleware to allow dashboard to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response Cache Configuration
CACHE_TTLS = {
    'models': 60,           # Models list changes rarely
    'equity_curve': 30,     # Historical data
    'metrics': 5,           # Real-time but expensive to compute
    'positions': 3,         # Balance freshness vs load
    'trades': 15,           # Historical data
    'leaderboard': 15,      # Rankings update periodically
    'stats': 30,            # Aggregate stats
}


class SimpleCache:
    """Simple in-memory cache with TTL support."""
    def __init__(self):
        self._cache: Dict[str, tuple[Any, datetime]] = {}

    def get(self, key: str, ttl_seconds: int) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now() - timestamp < timedelta(seconds=ttl_seconds):
                return value
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: Any):
        """Store value in cache with current timestamp."""
        self._cache[key] = (value, datetime.now())

    def invalidate(self, pattern: str):
        """Invalidate cache keys matching pattern."""
        keys_to_delete = [k for k in self._cache.keys() if pattern in k]
        for k in keys_to_delete:
            del self._cache[k]


# Global cache instance
response_cache = SimpleCache()

# Alpaca data cache - stores fetched position data to handle rate limiting
alpaca_data_cache = SimpleCache()
alpaca_cache_lock = asyncio.Lock()  # Prevent concurrent cache updates


# Equity Logging Database
# Support Railway volume mount via DATABASE_PATH env variable
DATA_DIR = Path(os.getenv('DATABASE_PATH', Path(__file__).parent.parent / "data"))
EQUITY_LOG_DB = DATA_DIR / "equity_log.db"
EQUITY_LOG_DB.parent.mkdir(parents=True, exist_ok=True)


def init_equity_log_db():
    """Initialize equity logging database"""
    conn = sqlite3.connect(str(EQUITY_LOG_DB))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS equity_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            equity REAL NOT NULL,
            cash REAL,
            portfolio_value REAL,
            buying_power REAL,
            position_count INTEGER,
            UNIQUE(model_id, timestamp)
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_equity_model_time
        ON equity_snapshots(model_id, timestamp DESC)
    """)

    conn.commit()
    conn.close()
    logger.info(f"Equity log database initialized at {EQUITY_LOG_DB}")


async def log_equity_snapshot():
    """Background task to log equity snapshots every 5 minutes"""
    models = ['gpt4', 'claude', 'gemini', 'deepseek', 'qwen']

    while True:
        try:
            conn = sqlite3.connect(str(EQUITY_LOG_DB))
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()

            for model_id in models:
                try:
                    # Get API credentials for this model
                    api_key = os.getenv(f'ALPACA_{model_id.upper()}_API_KEY')
                    secret_key = os.getenv(f'ALPACA_{model_id.upper()}_SECRET_KEY')

                    if not api_key or not secret_key:
                        logger.debug(f"No credentials for {model_id}, skipping equity snapshot")
                        continue

                    # Import Alpaca SDK
                    from alpaca.trading.client import TradingClient

                    # Get account info
                    client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
                    account = client.get_account()
                    positions = client.get_all_positions()

                    # Insert snapshot
                    cursor.execute("""
                        INSERT OR IGNORE INTO equity_snapshots
                        (model_id, timestamp, equity, cash, portfolio_value, buying_power, position_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        model_id,
                        timestamp,
                        float(account.equity),
                        float(account.cash),
                        float(account.portfolio_value),
                        float(account.buying_power),
                        len(positions)
                    ))

                    logger.info(f"Logged equity snapshot for {model_id}: ${float(account.equity):.2f}")

                except Exception as e:
                    logger.error(f"Error logging equity for {model_id}: {e}")
                    continue

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error in equity logging task: {e}")

        # Wait 5 minutes before next snapshot
        await asyncio.sleep(300)


# Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# Rate Limiting Middleware
class RateLimiter:
    """Simple in-memory rate limiter."""
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed based on rate limit."""
        now = time.time()
        minute_ago = now - 60

        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > minute_ago
        ]

        # Check if under limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False

        # Add current request
        self.requests[client_id].append(now)
        return True


# Initialize rate limiter (60 requests per minute per IP)
rate_limiter = RateLimiter(requests_per_minute=int(os.getenv('RATE_LIMIT_RPM', '60')))


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to requests."""
    # Skip rate limiting for health check
    if request.url.path == "/":
        return await call_next(request)

    # Get client IP
    client_ip = request.client.host if request.client else "unknown"

    # Check rate limit
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded. Please try again later."}
        )

    return await call_next(request)


# Optional API Key Authentication Middleware
API_KEY = os.getenv('API_KEY')  # Optional API key from environment

if API_KEY:
    @app.middleware("http")
    async def api_key_middleware(request: Request, call_next):
        """Validate API key if configured."""
        # Skip API key check for health check
        if request.url.path == "/":
            return await call_next(request)

        # Check for API key in header
        provided_key = request.headers.get('X-API-Key')

        if not provided_key or provided_key != API_KEY:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing API key"}
            )

        return await call_next(request)

    logger.info("API key authentication enabled")
else:
    logger.warning("No API key configured - API is publicly accessible")

# Database paths
# Use DATA_DIR defined earlier (supports Railway volume mount)
BASE_DIR = Path(__file__).parent.parent
POSITIONS_DB = DATA_DIR / "positions.db"
TRADING_DB = DATA_DIR / "trading.db"
TRADE_LOG = BASE_DIR / "logs" / "trade_log.jsonl"
BATTLE_DB = DATA_DIR / "battle.db"

# Ensure logs directory exists
TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)

# Initialize battle orchestrator
battle_orchestrator = BattleOrchestrator(BATTLE_DB)

# WebSocket connection manager for battle updates
class BattleConnectionManager:
    """Manage WebSocket connections for battle updates"""
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Battle WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"Battle WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.active_connections.discard(conn)

battle_manager = BattleConnectionManager()

# Import stream manager
# Temporarily disabled to prevent Alpaca rate limiting
STREAMING_ENABLED = False
logger.info("Alpaca streaming disabled to prevent rate limiting")
# try:
#     from alpaca_stream import get_stream_manager
#     STREAMING_ENABLED = True
#     logger.info("Alpaca streaming enabled")
# except ImportError as e:
#     STREAMING_ENABLED = False
#     logger.warning(f"Alpaca streaming not available: {e}")


def get_db_connection(db_path: Path):
    """Create database connection"""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Trading Dashboard API",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/positions")
async def get_positions():
    """Get all open positions"""
    try:
        conn = get_db_connection(POSITIONS_DB)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                symbol,
                direction,
                entry_price,
                stop_loss,
                take_profit,
                shares,
                opened_at,
                position_size_pct,
                strategy
            FROM positions
            WHERE status = 'open'
            ORDER BY opened_at DESC
        """)

        positions = []
        for row in cursor.fetchall():
            positions.append({
                "symbol": row[0],
                "direction": row[1],
                "entry_price": row[2],
                "stop_loss": row[3],
                "take_profit": row[4],
                "shares": row[5],
                "opened_at": row[6],
                "position_size_pct": row[7],
                "strategy": row[8]
            })

        conn.close()
        return {"positions": positions, "count": len(positions)}

    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trades")
async def get_trades(limit: int = 100):
    """Get recent trades from log file"""
    try:
        if not TRADE_LOG.exists():
            return {"trades": [], "count": 0}

        trades = []
        with open(TRADE_LOG, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("type") == "decision":
                        trades.append({
                            "trade_id": entry.get("trade_id"),
                            "timestamp": entry.get("timestamp"),
                            "symbol": entry.get("symbol"),
                            "trade_plan": entry.get("trade_plan", {}),
                            "risk_decision": entry.get("risk_decision", {})
                        })
                except json.JSONDecodeError:
                    continue

        # Return most recent trades
        trades = trades[-limit:]
        trades.reverse()

        return {"trades": trades, "count": len(trades)}

    except Exception as e:
        logger.error(f"Error fetching trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/executions")
async def get_executions(limit: int = 100):
    """Get recent executions from log file"""
    try:
        if not TRADE_LOG.exists():
            return {"executions": [], "count": 0}

        executions = []
        with open(TRADE_LOG, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("type") == "execution":
                        executions.append({
                            "trade_id": entry.get("trade_id"),
                            "timestamp": entry.get("timestamp"),
                            "symbol": entry.get("symbol"),
                            "direction": entry.get("direction"),
                            "filled_price": entry.get("filled_price"),
                            "filled_qty": entry.get("filled_qty"),
                            "slippage": entry.get("slippage"),
                            "execution_method": entry.get("execution_method")
                        })
                except json.JSONDecodeError:
                    continue

        # Return most recent executions
        executions = executions[-limit:]
        executions.reverse()

        return {"executions": executions, "count": len(executions)}

    except Exception as e:
        logger.error(f"Error fetching executions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/performance")
async def get_performance():
    """Calculate portfolio performance metrics"""
    try:
        conn = get_db_connection(POSITIONS_DB)
        cursor = conn.cursor()

        # Get closed positions for P&L calculation
        cursor.execute("""
            SELECT
                symbol,
                direction,
                entry_price,
                exit_price,
                shares,
                realized_pnl,
                closed_at
            FROM positions
            WHERE status = 'closed'
            ORDER BY closed_at DESC
            LIMIT 100
        """)

        closed_positions = cursor.fetchall()

        # Calculate metrics
        total_trades = len(closed_positions)
        winning_trades = sum(1 for p in closed_positions if p[5] and p[5] > 0)
        losing_trades = sum(1 for p in closed_positions if p[5] and p[5] < 0)
        total_pnl = sum(p[5] for p in closed_positions if p[5])

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Get open positions count
        cursor.execute("SELECT COUNT(*) FROM positions WHERE status = 'open'")
        open_positions = cursor.fetchone()[0]

        conn.close()

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "open_positions": open_positions,
            "avg_win": round(total_pnl / winning_trades, 2) if winning_trades > 0 else 0,
            "avg_loss": round(abs(sum(p[5] for p in closed_positions if p[5] and p[5] < 0)) / losing_trades, 2) if losing_trades > 0 else 0
        }

    except Exception as e:
        logger.error(f"Error calculating performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio")
async def get_portfolio():
    """Get current portfolio summary"""
    try:
        # In paper-live mode, we'd fetch from Alpaca
        # For now, calculate from positions
        conn = get_db_connection(POSITIONS_DB)
        cursor = conn.cursor()

        # Get open positions
        cursor.execute("""
            SELECT
                symbol,
                shares,
                entry_price
            FROM positions
            WHERE status = 'open'
        """)

        positions = cursor.fetchall()

        # Calculate total exposure
        total_exposure = sum(p[1] * p[2] for p in positions)

        # Initial capital (from config)
        initial_capital = 100000.0

        # Get closed positions P&L
        cursor.execute("SELECT SUM(realized_pnl) FROM positions WHERE status = 'closed'")
        result = cursor.fetchone()
        total_realized_pnl = result[0] if result[0] else 0

        # Calculate current equity
        equity = initial_capital + total_realized_pnl
        cash = equity - total_exposure

        conn.close()

        return {
            "equity": round(equity, 2),
            "cash": round(cash, 2),
            "total_exposure": round(total_exposure, 2),
            "exposure_pct": round((total_exposure / equity * 100) if equity > 0 else 0, 2),
            "position_count": len(positions),
            "initial_capital": initial_capital,
            "total_pnl": round(total_realized_pnl, 2),
            "pnl_pct": round((total_realized_pnl / initial_capital * 100) if initial_capital > 0 else 0, 2)
        }

    except Exception as e:
        logger.error(f"Error fetching portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/equity-curve")
async def get_equity_curve():
    """Get equity curve data for charting"""
    try:
        conn = get_db_connection(POSITIONS_DB)
        cursor = conn.cursor()

        # Get all closed positions ordered by exit time
        cursor.execute("""
            SELECT
                closed_at,
                realized_pnl
            FROM positions
            WHERE status = 'closed' AND closed_at IS NOT NULL
            ORDER BY closed_at ASC
        """)

        positions = cursor.fetchall()
        conn.close()

        # Calculate cumulative equity
        initial_capital = 100000.0
        equity_curve = [{"timestamp": datetime.now().isoformat(), "equity": initial_capital}]

        cumulative_pnl = 0
        for timestamp, pnl in positions:
            if pnl:
                cumulative_pnl += pnl
                equity_curve.append({
                    "timestamp": timestamp,
                    "equity": round(initial_capital + cumulative_pnl, 2)
                })

        return {"equity_curve": equity_curve}

    except Exception as e:
        logger.error(f"Error generating equity curve: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Get comprehensive statistics"""
    try:
        portfolio = await get_portfolio()
        performance = await get_performance()
        positions = await get_positions()

        return {
            "portfolio": portfolio,
            "performance": performance,
            "positions": positions
        }

    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def fetch_and_cache_alpaca_data():
    """Background task that fetches Alpaca data for all models and caches it"""
    models = ['gpt4', 'claude', 'gemini', 'deepseek', 'qwen']

    while True:
        try:
            await asyncio.sleep(30)  # Wait 30 seconds between fetches

            async with alpaca_cache_lock:
                from active_trader_llm.execution.alpaca_broker import AlpacaBrokerExecutor
                base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
                is_paper = 'paper' in base_url.lower()

                for model_id in models:
                    try:
                        api_key = os.getenv(f'ALPACA_{model_id.upper()}_API_KEY')
                        secret_key = os.getenv(f'ALPACA_{model_id.upper()}_SECRET_KEY')

                        if not api_key or not secret_key:
                            logger.debug(f"Skipping {model_id}: credentials not configured")
                            continue

                        # Temporarily set environment variables
                        original_key = os.environ.get('ALPACA_API_KEY')
                        original_secret = os.environ.get('ALPACA_SECRET_KEY')
                        os.environ['ALPACA_API_KEY'] = api_key
                        os.environ['ALPACA_SECRET_KEY'] = secret_key

                        try:
                            executor = AlpacaBrokerExecutor(paper=is_paper)
                            account_info = executor.get_account_info()
                            positions = executor.get_positions()

                            # Calculate total unrealized P&L
                            total_pl = sum(p.unrealized_pl for p in positions)

                            # Convert positions to dict format
                            positions_data = [
                                {
                                    "symbol": p.symbol,
                                    "qty": p.qty,
                                    "avg_entry_price": p.avg_entry_price,
                                    "current_price": p.current_price,
                                    "market_value": p.market_value,
                                    "unrealized_pl": p.unrealized_pl,
                                    "unrealized_pl_pct": p.unrealized_pl_pct
                                } for p in positions
                            ]

                            # Cache the data
                            cache_data = {
                                "success": True,
                                "model_id": model_id,
                                "account": {
                                    "account_number": account_info.get("account_number"),
                                    "status": account_info.get("status"),
                                    "portfolio_value": account_info.get("portfolio_value"),
                                    "cash": account_info.get("cash"),
                                    "buying_power": account_info.get("buying_power"),
                                    "equity": account_info.get("equity")
                                },
                                "positions": positions_data,
                                "summary": {
                                    "total_positions": len(positions),
                                    "total_unrealized_pl": total_pl,
                                    "account_type": "Paper Trading" if is_paper else "Live Trading"
                                },
                                "timestamp": datetime.now().isoformat(),
                                "cached": True
                            }

                            alpaca_data_cache.set(f"positions_{model_id}", cache_data)
                            logger.debug(f"Cached data for {model_id}: {len(positions)} positions")

                        finally:
                            # Restore original credentials
                            if original_key:
                                os.environ['ALPACA_API_KEY'] = original_key
                            if original_secret:
                                os.environ['ALPACA_SECRET_KEY'] = original_secret

                    except Exception as e:
                        logger.error(f"Error fetching data for {model_id}: {e}")
                        continue

                logger.info(f"Alpaca data cache updated for {len(models)} models")

        except asyncio.CancelledError:
            logger.info("Alpaca data caching task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in Alpaca data caching task: {e}")
            await asyncio.sleep(30)  # Wait before retry


@app.get("/api/alpaca-account")
async def get_alpaca_account():
    """Get Alpaca account information and configuration"""
    try:
        # Import Alpaca client
        from alpaca.trading.client import TradingClient

        # Get credentials from environment
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

        if not api_key or not secret_key:
            return {
                "connected": False,
                "error": "Alpaca credentials not configured"
            }

        # Determine if paper trading
        is_paper = 'paper' in base_url.lower()

        # Connect to Alpaca
        client = TradingClient(api_key, secret_key, paper=is_paper)
        account = client.get_account()

        # Mask API key for security (show first 6 and last 4 characters)
        masked_api_key = f"{api_key[:6]}{'•' * 20}{api_key[-4:]}" if len(api_key) > 10 else "••••••••••"

        # Determine which model this account belongs to
        model_name = "Primary Trader"
        if os.getenv('ANTHROPIC_API_KEY'):
            model_name = "Claude Sonnet 4.5"
        elif os.getenv('OPENAI_API_KEY'):
            model_name = "GPT-4 Turbo"

        return {
            "connected": True,
            "model_name": model_name,
            "account_type": "Paper Trading" if is_paper else "Live Trading",
            "api_key": masked_api_key,
            "base_url": base_url,
            "account": {
                "account_number": account.account_number,
                "status": str(account.status),
                "currency": account.currency,
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "last_equity": float(account.last_equity),
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "transfers_blocked": account.transfers_blocked,
                "account_blocked": account.account_blocked,
                "created_at": str(account.created_at)
            }
        }

    except ImportError:
        return {
            "connected": False,
            "error": "Alpaca SDK not installed (pip install alpaca-py)"
        }
    except Exception as e:
        logger.error(f"Error fetching Alpaca account: {e}")
        return {
            "connected": False,
            "error": str(e)
        }


@app.get("/api/alpaca-positions")
async def get_alpaca_positions(model_id: str = Query(None, description="Optional model ID to fetch specific account")):
    """Get current Alpaca positions and account summary for default or specific model account

    Uses caching to handle Alpaca API rate limits. Returns cached data if API is rate limited.
    """
    # Check cache first (TTL: 120 seconds as fallback)
    cache_key = f"positions_{model_id or 'default'}"
    cached_data = alpaca_data_cache.get(cache_key, 120)

    try:
        # Import Alpaca client
        from active_trader_llm.execution.alpaca_broker import AlpacaBrokerExecutor

        # Get base URL from environment
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        is_paper = 'paper' in base_url.lower()

        # If model_id specified, use model-specific credentials
        if model_id:
            api_key = os.getenv(f'ALPACA_{model_id.upper()}_API_KEY')
            secret_key = os.getenv(f'ALPACA_{model_id.upper()}_SECRET_KEY')

            if not api_key or not secret_key:
                # Return cached data if available
                if cached_data:
                    logger.info(f"Returning cached data for {model_id} (credentials not configured)")
                    return cached_data

                return {
                    "success": False,
                    "error": f"Credentials not found for model {model_id}. Set ALPACA_{model_id.upper()}_API_KEY and ALPACA_{model_id.upper()}_SECRET_KEY in .env"
                }

            # Temporarily set environment variables for this model
            original_key = os.environ.get('ALPACA_API_KEY')
            original_secret = os.environ.get('ALPACA_SECRET_KEY')
            os.environ['ALPACA_API_KEY'] = api_key
            os.environ['ALPACA_SECRET_KEY'] = secret_key

            try:
                executor = AlpacaBrokerExecutor(paper=is_paper)
                account_info = executor.get_account_info()
                positions = executor.get_positions()
            finally:
                # Restore original credentials
                if original_key:
                    os.environ['ALPACA_API_KEY'] = original_key
                if original_secret:
                    os.environ['ALPACA_SECRET_KEY'] = original_secret
        else:
            # Use default credentials
            executor = AlpacaBrokerExecutor(paper=is_paper)
            account_info = executor.get_account_info()
            positions = executor.get_positions()

        # Calculate total unrealized P&L
        total_pl = sum(p.unrealized_pl for p in positions)

        # Convert positions to dict format
        positions_data = [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_entry_price": p.avg_entry_price,
                "current_price": p.current_price,
                "market_value": p.market_value,
                "unrealized_pl": p.unrealized_pl,
                "unrealized_pl_pct": p.unrealized_pl_pct
            } for p in positions
        ]

        # Build response
        response_data = {
            "success": True,
            "model_id": model_id or "default",
            "account": {
                "account_number": account_info.get("account_number"),
                "status": account_info.get("status"),
                "portfolio_value": account_info.get("portfolio_value"),
                "cash": account_info.get("cash"),
                "buying_power": account_info.get("buying_power"),
                "equity": account_info.get("equity")
            },
            "positions": positions_data,
            "summary": {
                "total_positions": len(positions),
                "total_unrealized_pl": total_pl,
                "account_type": "Paper Trading" if is_paper else "Live Trading"
            },
            "timestamp": datetime.now().isoformat(),
            "cached": False
        }

        # Update cache with fresh data
        alpaca_data_cache.set(cache_key, response_data)

        return response_data

    except ImportError as e:
        logger.error(f"Import error: {e}")
        # Return cached data if available
        if cached_data:
            logger.info(f"Returning cached data for {model_id or 'default'} (import error)")
            return cached_data

        return {
            "success": False,
            "error": "Alpaca SDK not installed or module not found"
        }
    except Exception as e:
        error_msg = str(e)

        # Check if this is a rate limit error
        is_rate_limited = "429" in error_msg or "too many requests" in error_msg.lower() or "rate limit" in error_msg.lower()

        if is_rate_limited:
            logger.warning(f"Rate limited for {model_id or 'default'}: {error_msg}")
        else:
            logger.error(f"Error fetching Alpaca positions for {model_id or 'default'}: {error_msg}")

        # Return cached data if available
        if cached_data:
            logger.info(f"Returning cached data for {model_id or 'default'} ({error_msg[:50]}...)")
            return cached_data

        return {
            "success": False,
            "error": error_msg,
            "rate_limited": is_rate_limited
        }


@app.get("/api/alpaca-equity-curves")
async def get_alpaca_equity_curves(timeframe: str = Query("1W", description="Timeframe: 1D, 1W, 1M, or All")):
    """Get historical portfolio values (equity curves) from logged snapshots

    Uses caching to prevent rate limiting from excessive dashboard requests.
    Cache TTL: 60 seconds
    """
    # Check cache first (60-second TTL)
    cache_key = f"equity_curves_{timeframe}"
    cached_data = alpaca_data_cache.get(cache_key, 60)

    if cached_data:
        logger.debug(f"Returning cached equity curves for {timeframe}")
        # Mark as cached before returning
        cached_data["cached"] = True
        return cached_data

    try:
        # Calculate time window based on timeframe
        now = datetime.now()
        time_windows = {
            "1D": now - timedelta(days=1),
            "1W": now - timedelta(weeks=1),
            "1M": now - timedelta(days=30),
            "All": now - timedelta(days=365)
        }

        start_time = time_windows.get(timeframe, now - timedelta(weeks=1))

        # Models to fetch
        models = ['gpt4', 'claude', 'gemini', 'deepseek', 'qwen']
        all_curves = []

        conn = sqlite3.connect(str(EQUITY_LOG_DB))
        cursor = conn.cursor()

        for model_id in models:
            try:
                # Query equity snapshots for this model within timeframe
                cursor.execute("""
                    SELECT timestamp, equity
                    FROM equity_snapshots
                    WHERE model_id = ?
                    AND datetime(timestamp) >= datetime(?)
                    ORDER BY timestamp ASC
                """, (model_id, start_time.isoformat()))

                rows = cursor.fetchall()

                data_points = []
                for timestamp, equity in rows:
                    data_points.append({
                        "timestamp": timestamp,
                        "equity": float(equity)
                    })

                # If no logged data, try to get current equity from Alpaca
                if not data_points:
                    try:
                        from alpaca.trading.client import TradingClient
                        api_key = os.getenv(f'ALPACA_{model_id.upper()}_API_KEY')
                        secret_key = os.getenv(f'ALPACA_{model_id.upper()}_SECRET_KEY')

                        if api_key and secret_key:
                            client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
                            account = client.get_account()
                            data_points.append({
                                "timestamp": datetime.now().isoformat(),
                                "equity": float(account.equity)
                            })
                    except Exception as e:
                        logger.debug(f"Could not fetch current equity for {model_id}: {e}")

                all_curves.append({
                    "model_name": model_id,
                    "data": data_points
                })

            except Exception as e:
                logger.error(f"Error fetching equity curve for {model_id}: {e}")
                all_curves.append({
                    "model_name": model_id,
                    "data": []
                })

        conn.close()

        response_data = {
            "success": True,
            "timeframe": timeframe,
            "curves": all_curves,
            "timestamp": datetime.now().isoformat(),
            "cached": False
        }

        # Cache the response
        alpaca_data_cache.set(cache_key, response_data)
        logger.debug(f"Cached equity curves for {timeframe} (total curves: {len(all_curves)})")

        return response_data

    except Exception as e:
        logger.error(f"Error fetching equity curves: {e}")
        return {
            "success": False,
            "error": str(e),
            "curves": []
        }


# Cache for LLM models data to prevent rate limiting
_llm_models_cache = {"data": None, "timestamp": 0}
_LLM_MODELS_CACHE_TTL = 60  # Cache for 60 seconds

# WebSocket connection manager for LLM models
class LLMModelsConnectionManager:
    """Manage WebSocket connections for LLM model updates"""
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"LLM Models WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"LLM Models WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to LLM models client: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

llm_models_manager = LLMModelsConnectionManager()

@app.get("/api/llm-models")
async def get_llm_models():
    """Get all configured LLM models with their Alpaca account status"""

    # Check cache first
    current_time = time.time()
    if _llm_models_cache["data"] is not None and (current_time - _llm_models_cache["timestamp"]) < _LLM_MODELS_CACHE_TTL:
        logger.info("Returning cached LLM models data")
        return _llm_models_cache["data"]

    models = []

    # Define models from battle config
    model_configs = [
        {
            "id": "gpt4",
            "name": "GPT-4 Turbo",
            "provider": "OpenAI",
            "api_key_env": "ALPACA_GPT4_API_KEY",
            "secret_key_env": "ALPACA_GPT4_SECRET_KEY"
        },
        {
            "id": "claude",
            "name": "Claude Sonnet 4.5",
            "provider": "Anthropic",
            "api_key_env": "ALPACA_CLAUDE_API_KEY",
            "secret_key_env": "ALPACA_CLAUDE_SECRET_KEY"
        },
        {
            "id": "grok",
            "name": "Grok-2",
            "provider": "XAI",
            "api_key_env": "ALPACA_GROK_API_KEY",
            "secret_key_env": "ALPACA_GROK_SECRET_KEY"
        },
        {
            "id": "deepseek",
            "name": "DeepSeek V3",
            "provider": "DeepSeek",
            "api_key_env": "ALPACA_DEEPSEEK_API_KEY",
            "secret_key_env": "ALPACA_DEEPSEEK_SECRET_KEY"
        },
        {
            "id": "gemini",
            "name": "Gemini 2.0 Flash",
            "provider": "Google",
            "api_key_env": "ALPACA_GEMINI_API_KEY",
            "secret_key_env": "ALPACA_GEMINI_SECRET_KEY"
        },
        {
            "id": "qwen",
            "name": "Qwen Max",
            "provider": "Alibaba",
            "api_key_env": "ALPACA_QWEN_API_KEY",
            "secret_key_env": "ALPACA_QWEN_SECRET_KEY"
        },
    ]

    try:
        from alpaca.trading.client import TradingClient

        for config in model_configs:
            api_key = os.getenv(config["api_key_env"])
            secret_key = os.getenv(config["secret_key_env"])

            model_info = {
                "model_id": config["id"],
                "model_name": config["name"],
                "provider": config["provider"],
                "connected": False,
                "account": None,
                "error": None
            }

            # If API keys exist, try to connect
            if api_key and secret_key:
                try:
                    client = TradingClient(api_key, secret_key, paper=True)
                    account = client.get_account()
                    model_info["connected"] = True
                    model_info["account"] = {
                        "portfolio_value": float(account.portfolio_value),
                        "cash": float(account.cash),
                        "buying_power": float(account.buying_power),
                        "equity": float(account.equity),
                        "position_count": 0,  # Will be populated if positions exist
                    }
                except Exception as e:
                    model_info["error"] = str(e)
                    logger.warning(f"Could not connect {config['name']}: {e}")
            else:
                model_info["error"] = "API keys not configured"

            models.append(model_info)

        # Cache the result
        result = {"models": models, "total": len(models), "connected": sum(1 for m in models if m["connected"])}
        _llm_models_cache["data"] = result
        _llm_models_cache["timestamp"] = current_time
        logger.info(f"Cached LLM models data: {result['connected']}/{result['total']} connected")
        return result

    except ImportError:
        return {"error": "Alpaca SDK not installed", "models": [], "total": 0, "connected": 0}
    except Exception as e:
        logger.error(f"Error fetching LLM models: {e}")
        return {"error": str(e), "models": [], "total": 0, "connected": 0}


@app.websocket("/ws/llm-models")
async def llm_models_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time LLM model account updates

    Sends updates every 30 seconds with latest account data for all models.
    Uses cached data to avoid Alpaca rate limits.
    """
    await llm_models_manager.connect(websocket)

    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to LLM models stream",
            "timestamp": datetime.now().isoformat()
        })

        # Send initial data
        initial_data = await get_llm_models()
        await websocket.send_json({
            "type": "models_update",
            "data": initial_data,
            "timestamp": datetime.now().isoformat()
        })

        # Keep connection alive and send periodic updates
        last_update = time.time()
        UPDATE_INTERVAL = 30  # Update every 30 seconds

        while True:
            try:
                # Check if it's time to send an update
                current_time = time.time()
                if current_time - last_update >= UPDATE_INTERVAL:
                    # Get latest data (will use cache if still fresh)
                    models_data = await get_llm_models()
                    await websocket.send_json({
                        "type": "models_update",
                        "data": models_data,
                        "timestamp": datetime.now().isoformat()
                    })
                    last_update = current_time
                    logger.debug(f"Sent LLM models update: {models_data['connected']}/{models_data['total']} connected")

                # Wait a bit before next check (non-blocking)
                await asyncio.sleep(5)

            except WebSocketDisconnect:
                logger.info("LLM models WebSocket client disconnected")
                break
            except Exception as e:
                logger.error(f"Error in LLM models WebSocket loop: {e}")
                break

    except Exception as e:
        logger.error(f"LLM models WebSocket error: {e}")

    finally:
        llm_models_manager.disconnect(websocket)


# ============================================================================
# BATTLE SYSTEM ENDPOINTS
# ============================================================================

@app.get("/api/battle/leaderboard", response_model=LeaderboardResponse)
async def get_battle_leaderboard(timeframe: str = Query("all", regex="^(daily|weekly|monthly|all)$")):
    """
    Get model leaderboard rankings

    Timeframes:
    - daily: Today's performance
    - weekly: Last 7 days
    - monthly: Last 30 days
    - all: All-time performance
    """
    cache_key = f"leaderboard_{timeframe}"
    cached = response_cache.get(cache_key, CACHE_TTLS['leaderboard'])
    if cached:
        return cached

    try:
        rankings = battle_orchestrator.leaderboard.get_rankings(timeframe)

        leaderboard_entries = [
            LeaderboardEntry(
                rank=r['rank'],
                model_id=r['model_id'],
                display_name=r['display_name'],
                provider=r['provider'],
                model_name=r['model_name'],
                current_equity=round(r['current_equity'], 2),
                total_pnl=round(r['total_pnl'], 2),
                pnl_pct=round(r['pnl_pct'], 2),
                total_trades=r['total_trades'],
                win_rate=round(r['win_rate'], 2),
                sharpe_ratio=round(r['sharpe_ratio'], 2),
                max_drawdown=round(r['max_drawdown'], 2),
                profit_factor=round(r['profit_factor'], 2)
            )
            for r in rankings
        ]

        result = LeaderboardResponse(
            timeframe=timeframe,
            rankings=leaderboard_entries,
            timestamp=datetime.now().isoformat()
        )
        response_cache.set(cache_key, result)
        return result

    except Exception as e:
        logger.error(f"Error fetching leaderboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/battle/alpaca-leaderboard", response_model=LeaderboardResponse)
async def get_alpaca_leaderboard(timeframe: str = Query("all", regex="^(daily|weekly|monthly|all)$")):
    """
    Get real-time leaderboard from actual Alpaca accounts

    Fetches live account data for all 6 models and calculates rankings based on
    actual portfolio equity from their paper trading accounts.

    Uses caching to prevent rate limiting from excessive dashboard requests.
    Cache TTL: 60 seconds

    Note: All models start with $100,000 initial capital.
    """
    # Check cache first (60-second TTL)
    cache_key = f"leaderboard_{timeframe}"
    cached_data = alpaca_data_cache.get(cache_key, 60)

    if cached_data:
        logger.debug(f"Returning cached leaderboard for {timeframe}")
        return cached_data

    try:
        from alpaca.trading.client import TradingClient

        # Define model configurations matching battle_config.yaml
        model_configs = [
            {
                "model_id": "gpt4",
                "display_name": "GPT-4 Turbo",
                "provider": "OpenAI",
                "model_name": "gpt-4-turbo",
                "api_key_env": "ALPACA_GPT4_API_KEY",
                "secret_key_env": "ALPACA_GPT4_SECRET_KEY"
            },
            {
                "model_id": "claude",
                "display_name": "Claude Sonnet 4.5",
                "provider": "Anthropic",
                "model_name": "claude-sonnet-4-5-20250929",
                "api_key_env": "ALPACA_CLAUDE_API_KEY",
                "secret_key_env": "ALPACA_CLAUDE_SECRET_KEY"
            },
            {
                "model_id": "gemini",
                "display_name": "Gemini 2.0 Flash",
                "provider": "Google",
                "model_name": "gemini-2.0-flash-exp",
                "api_key_env": "ALPACA_GEMINI_API_KEY",
                "secret_key_env": "ALPACA_GEMINI_SECRET_KEY"
            },
            {
                "model_id": "deepseek",
                "display_name": "DeepSeek V3",
                "provider": "DeepSeek",
                "model_name": "deepseek-chat",
                "api_key_env": "ALPACA_DEEPSEEK_API_KEY",
                "secret_key_env": "ALPACA_DEEPSEEK_SECRET_KEY"
            },
            {
                "model_id": "qwen",
                "display_name": "Qwen Max",
                "provider": "Alibaba",
                "model_name": "qwen/qwen-2.5-72b-instruct",
                "api_key_env": "ALPACA_QWEN_API_KEY",
                "secret_key_env": "ALPACA_QWEN_SECRET_KEY"
            },
            {
                "model_id": "grok",
                "display_name": "Grok-2",
                "provider": "XAI",
                "model_name": "grok-2-1212",
                "api_key_env": "ALPACA_GROK_API_KEY",
                "secret_key_env": "ALPACA_GROK_SECRET_KEY"
            }
        ]

        initial_capital = 100000.0  # All models start with $100k
        rankings_data = []

        # Fetch account data for each model
        for config in model_configs:
            api_key = os.getenv(config["api_key_env"])
            secret_key = os.getenv(config["secret_key_env"])

            # Skip if credentials not configured
            if not api_key or not secret_key:
                logger.warning(f"Skipping {config['model_id']}: credentials not configured")
                # Add placeholder entry with zero values
                rankings_data.append({
                    "model_id": config["model_id"],
                    "display_name": config["display_name"],
                    "provider": config["provider"],
                    "model_name": config["model_name"],
                    "current_equity": initial_capital,
                    "total_pnl": 0.0,
                    "pnl_pct": 0.0,
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "profit_factor": 0.0
                })
                continue

            try:
                # Connect to Alpaca and fetch account data
                client = TradingClient(api_key, secret_key, paper=True)
                account = client.get_account()

                # Calculate metrics from account data
                current_equity = float(account.equity)
                total_pnl = current_equity - initial_capital
                pnl_pct = (total_pnl / initial_capital) * 100

                # Note: Trade-specific metrics (win_rate, sharpe, etc.) require trade history
                # For now, set these to 0.0 as no trades have been executed yet
                rankings_data.append({
                    "model_id": config["model_id"],
                    "display_name": config["display_name"],
                    "provider": config["provider"],
                    "model_name": config["model_name"],
                    "current_equity": current_equity,
                    "total_pnl": total_pnl,
                    "pnl_pct": pnl_pct,
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "profit_factor": 0.0
                })

                logger.info(f"Fetched {config['model_id']} account: equity=${current_equity:.2f}, P/L=${total_pnl:.2f}")

            except Exception as e:
                logger.error(f"Error fetching Alpaca account for {config['model_id']}: {e}")
                # Add placeholder entry with initial capital
                rankings_data.append({
                    "model_id": config["model_id"],
                    "display_name": config["display_name"],
                    "provider": config["provider"],
                    "model_name": config["model_name"],
                    "current_equity": initial_capital,
                    "total_pnl": 0.0,
                    "pnl_pct": 0.0,
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "profit_factor": 0.0
                })

        # Sort by total P/L (descending) and assign ranks
        rankings_data.sort(key=lambda x: x["total_pnl"], reverse=True)

        leaderboard_entries = [
            LeaderboardEntry(
                rank=idx + 1,
                model_id=r["model_id"],
                display_name=r["display_name"],
                provider=r["provider"],
                model_name=r["model_name"],
                current_equity=round(r["current_equity"], 2),
                total_pnl=round(r["total_pnl"], 2),
                pnl_pct=round(r["pnl_pct"], 2),
                total_trades=r["total_trades"],
                win_rate=round(r["win_rate"], 2),
                sharpe_ratio=round(r["sharpe_ratio"], 2),
                max_drawdown=round(r["max_drawdown"], 2),
                profit_factor=round(r["profit_factor"], 2)
            )
            for idx, r in enumerate(rankings_data)
        ]

        response_data = LeaderboardResponse(
            timeframe=timeframe,
            rankings=leaderboard_entries,
            timestamp=datetime.now().isoformat()
        )

        # Cache the response
        alpaca_data_cache.set(cache_key, response_data)
        logger.debug(f"Cached leaderboard for {timeframe} (total entries: {len(leaderboard_entries)})")

        return response_data

    except ImportError:
        logger.error("Alpaca SDK not installed")
        raise HTTPException(status_code=500, detail="Alpaca SDK not installed (pip install alpaca-py)")
    except Exception as e:
        logger.error(f"Error fetching Alpaca leaderboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/battle/models", response_model=ModelsListResponse)
async def get_battle_models():
    """Get all competing models with status"""
    cache_key = "battle_models"
    cached = response_cache.get(cache_key, CACHE_TTLS['models'])
    if cached:
        return cached

    try:
        models = battle_orchestrator.get_models()

        model_infos = [
            ModelInfo(
                model_id=m.model_id,
                display_name=m.display_name,
                provider=m.provider,
                model_name=m.model_name,
                status=m.status.value,
                initial_capital=m.initial_capital,
                created_at=m.created_at
            )
            for m in models
        ]

        result = ModelsListResponse(
            models=model_infos,
            count=len(model_infos)
        )
        response_cache.set(cache_key, result)
        return result

    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/battle/status", response_model=SystemStatusResponse)
async def get_battle_status():
    """Get battle system health and status"""
    try:
        status = battle_orchestrator.get_system_status()
        return SystemStatusResponse(**status)

    except Exception as e:
        logger.error(f"Error fetching battle status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/battle/model/{model_id}/positions", response_model=PositionsListResponse)
async def get_model_positions(model_id: str):
    """Get current positions for a specific model"""
    cache_key = f"positions_{model_id}"
    cached = response_cache.get(cache_key, CACHE_TTLS['positions'])
    if cached:
        return cached

    try:
        # Verify model exists
        models = battle_orchestrator.get_models()
        if not any(m.model_id == model_id for m in models):
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

        positions = battle_orchestrator.get_model_positions(model_id)

        position_responses = [
            PositionResponse(
                model_id=p.model_id,
                symbol=p.symbol,
                direction=p.direction,
                shares=p.shares,
                entry_price=p.entry_price,
                stop_loss=p.stop_loss,
                take_profit=p.take_profit,
                opened_at=p.opened_at,
                current_price=p.current_price,
                unrealized_pnl=p.unrealized_pnl
            )
            for p in positions
        ]

        result = PositionsListResponse(
            positions=position_responses,
            count=len(position_responses)
        )
        response_cache.set(cache_key, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching model positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/battle/model/{model_id}/trades", response_model=TradesListResponse)
async def get_model_trades(model_id: str, limit: int = Query(100, ge=1, le=1000)):
    """Get trade history for a specific model"""
    cache_key = f"trades_{model_id}_{limit}"
    cached = response_cache.get(cache_key, CACHE_TTLS['trades'])
    if cached:
        return cached

    try:
        # Verify model exists
        models = battle_orchestrator.get_models()
        if not any(m.model_id == model_id for m in models):
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

        trades = battle_orchestrator.get_model_trades(model_id, limit)

        trade_responses = [
            TradeResponse(
                trade_id=t.trade_id,
                model_id=t.model_id,
                symbol=t.symbol,
                direction=t.direction,
                shares=t.shares,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                realized_pnl=round(t.realized_pnl, 2),
                opened_at=t.opened_at,
                closed_at=t.closed_at,
                exit_reason=t.exit_reason,
                reasoning=t.reasoning
            )
            for t in trades
        ]

        result = TradesListResponse(
            trades=trade_responses,
            count=len(trade_responses)
        )
        response_cache.set(cache_key, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching model trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/battle/model/{model_id}/decisions", response_model=ModelDecisionsResponse)
async def get_model_decisions(
    model_id: str,
    limit: int = Query(10, ge=1, le=100, description="Maximum number of decisions to return"),
    timeframe: Optional[int] = Query(None, ge=1, description="Hours to look back (optional)")
):
    """Get recent decisions/reasoning for a specific model"""
    cache_key = f"decisions_{model_id}_{limit}_{timeframe}"
    cached = response_cache.get(cache_key, CACHE_TTLS['trades'])
    if cached:
        return cached

    try:
        # Verify model exists
        models = battle_orchestrator.get_models()
        if not any(m.model_id == model_id for m in models):
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

        conn = battle_orchestrator.db.get_connection()
        cursor = conn.cursor()

        # Build query with optional timeframe filter
        query = """
            SELECT model_id, symbol, timestamp, decision, reasoning, confidence
            FROM battle_decisions
            WHERE model_id = ?
        """
        params = [model_id]

        if timeframe:
            cutoff_time = datetime.now() - timedelta(hours=timeframe)
            query += " AND timestamp >= ?"
            params.append(cutoff_time.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        decisions = [
            DecisionResponse(
                model_id=row['model_id'],
                symbol=row['symbol'],
                timestamp=row['timestamp'],
                decision=row['decision'],
                reasoning=row['reasoning'],
                confidence=row['confidence']
            )
            for row in rows
        ]

        result = ModelDecisionsResponse(
            model_id=model_id,
            decisions=decisions,
            total=len(decisions)
        )
        response_cache.set(cache_key, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching model decisions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/battle/model/{model_id}/equity_curve", response_model=EquityCurveResponse)
async def get_model_equity_curve(model_id: str):
    """Get equity curve time series data for a model"""
    cache_key = f"equity_curve_{model_id}"
    cached = response_cache.get(cache_key, CACHE_TTLS['equity_curve'])
    if cached:
        return cached

    try:
        # Verify model exists
        models = battle_orchestrator.get_models()
        if not any(m.model_id == model_id for m in models):
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

        equity_curve = battle_orchestrator.get_equity_curve(model_id)

        curve_points = [
            EquityCurvePoint(
                timestamp=point['timestamp'],
                equity=round(point['equity'], 2)
            )
            for point in equity_curve
        ]

        result = EquityCurveResponse(
            model_id=model_id,
            equity_curve=curve_points
        )
        response_cache.set(cache_key, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching equity curve: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/battle/model/{model_id}/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics(model_id: str):
    """Get comprehensive metrics snapshot for a model"""
    cache_key = f"metrics_{model_id}"
    cached = response_cache.get(cache_key, CACHE_TTLS['metrics'])
    if cached:
        return cached

    try:
        # Verify model exists
        models = battle_orchestrator.get_models()
        model = next((m for m in models if m.model_id == model_id), None)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

        metrics = battle_orchestrator.metrics_engine.calculate_metrics(model_id)

        pnl_pct = (metrics.total_pnl / model.initial_capital * 100) if model.initial_capital > 0 else 0.0

        result = ModelMetricsResponse(
            model_id=metrics.model_id,
            current_equity=round(metrics.current_equity, 2),
            total_pnl=round(metrics.total_pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            total_trades=metrics.total_trades,
            winning_trades=metrics.winning_trades,
            losing_trades=metrics.losing_trades,
            win_rate=round(metrics.win_rate, 2),
            avg_win=round(metrics.avg_win, 2),
            avg_loss=round(metrics.avg_loss, 2),
            profit_factor=round(metrics.profit_factor, 2),
            sharpe_ratio=round(metrics.sharpe_ratio, 2),
            max_drawdown=round(metrics.max_drawdown, 2),
            current_positions=metrics.current_positions,
            timestamp=metrics.timestamp
        )
        response_cache.set(cache_key, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/battle/compare/positions", response_model=ComparePositionsResponse)
async def compare_positions():
    """Get all models' positions side-by-side for comparison"""
    try:
        models = battle_orchestrator.db.get_active_models()

        positions_by_model = {}
        for model in models:
            positions = battle_orchestrator.get_model_positions(model.model_id)
            positions_by_model[model.model_id] = [
                PositionResponse(
                    model_id=p.model_id,
                    symbol=p.symbol,
                    direction=p.direction,
                    shares=p.shares,
                    entry_price=p.entry_price,
                    stop_loss=p.stop_loss,
                    take_profit=p.take_profit,
                    opened_at=p.opened_at,
                    current_price=p.current_price,
                    unrealized_pnl=p.unrealized_pnl
                )
                for p in positions
            ]

        return ComparePositionsResponse(
            timestamp=datetime.now().isoformat(),
            positions_by_model=positions_by_model
        )

    except Exception as e:
        logger.error(f"Error comparing positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/battle/compare/decisions", response_model=DecisionComparisonResponse)
async def compare_decisions(
    symbol: str = Query(..., description="Stock symbol"),
    timestamp: str = Query(..., description="ISO timestamp")
):
    """Compare all models' decisions for same stock at same time"""
    try:
        conn = battle_orchestrator.db.get_connection()
        cursor = conn.cursor()

        # Query decisions for all models at the given timestamp
        cursor.execute("""
            SELECT model_id, symbol, timestamp, decision, reasoning, confidence
            FROM battle_decisions
            WHERE symbol = ? AND timestamp = ?
            ORDER BY model_id
        """, (symbol, timestamp))

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"No decisions found for {symbol} at {timestamp}"
            )

        decisions = [
            DecisionResponse(
                model_id=row['model_id'],
                symbol=row['symbol'],
                timestamp=row['timestamp'],
                decision=row['decision'],
                reasoning=row['reasoning'],
                confidence=row['confidence']
            )
            for row in rows
        ]

        return DecisionComparisonResponse(
            symbol=symbol,
            timestamp=timestamp,
            decisions=decisions
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing decisions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/battle/compare/equity_curves", response_model=CompareEquityCurvesResponse)
async def compare_equity_curves():
    """Get all equity curves for charting comparison"""
    try:
        models = battle_orchestrator.db.get_active_models()

        equity_curves = {}
        for model in models:
            curve_data = battle_orchestrator.get_equity_curve(model.model_id)
            equity_curves[model.model_id] = [
                EquityCurvePoint(
                    timestamp=point['timestamp'],
                    equity=round(point['equity'], 2)
                )
                for point in curve_data
            ]

        return CompareEquityCurvesResponse(equity_curves=equity_curves)

    except Exception as e:
        logger.error(f"Error comparing equity curves: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/battle/compare/reasoning")
async def compare_reasoning(
    symbol: str = Query(..., description="Stock symbol"),
    timestamp: str = Query(..., description="ISO timestamp")
):
    """
    Compare reasoning from all models for same symbol/timestamp
    Returns more detailed comparison than /compare/decisions
    """
    try:
        conn = battle_orchestrator.db.get_connection()
        cursor = conn.cursor()

        # Get decisions with full reasoning
        cursor.execute("""
            SELECT
                bd.model_id,
                bd.symbol,
                bd.timestamp,
                bd.decision,
                bd.reasoning,
                bd.confidence,
                bd.market_data,
                bm.display_name,
                bm.provider,
                bm.model_name
            FROM battle_decisions bd
            JOIN battle_models bm ON bd.model_id = bm.model_id
            WHERE bd.symbol = ? AND bd.timestamp = ?
            ORDER BY bd.model_id
        """, (symbol, timestamp))

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"No decisions found for {symbol} at {timestamp}"
            )

        comparisons = []
        for row in rows:
            market_data = json.loads(row['market_data']) if row['market_data'] else None

            comparisons.append({
                "model_id": row['model_id'],
                "display_name": row['display_name'],
                "provider": row['provider'],
                "model_name": row['model_name'],
                "decision": row['decision'],
                "reasoning": row['reasoning'],
                "confidence": row['confidence'],
                "market_data": market_data
            })

        return {
            "symbol": symbol,
            "timestamp": timestamp,
            "comparisons": comparisons
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing reasoning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/battle")
async def battle_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time battle updates

    Event types:
    - new_trade: A model completed a trade
    - decision_made: A model made a trading decision
    - metrics_updated: Model metrics recalculated
    - leaderboard_changed: Rankings updated
    - position_opened: New position opened
    - position_closed: Position closed
    """
    await battle_manager.connect(websocket)

    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to battle stream",
            "timestamp": datetime.now().isoformat()
        })

        # Send current leaderboard
        rankings = battle_orchestrator.leaderboard.get_rankings("all")
        await websocket.send_json({
            "type": "leaderboard_snapshot",
            "data": {"rankings": rankings},
            "timestamp": datetime.now().isoformat()
        })

        # Keep connection alive and handle client messages
        while True:
            try:
                data = await websocket.receive_text()
                logger.debug(f"Received from battle client: {data}")

                # Handle client requests (e.g., subscribe to specific model)
                try:
                    message = json.loads(data)
                    if message.get("action") == "subscribe_model":
                        model_id = message.get("model_id")
                        # Send model-specific updates
                        logger.info(f"Client subscribed to model: {model_id}")
                except json.JSONDecodeError:
                    pass

            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"Battle WebSocket error: {e}")

    finally:
        battle_manager.disconnect(websocket)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()

    if not STREAMING_ENABLED:
        await websocket.send_json({
            "type": "error",
            "message": "Streaming not enabled"
        })
        await websocket.close()
        return

    stream_manager = get_stream_manager()
    stream_manager.add_client(websocket)

    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to live stream",
            "timestamp": datetime.now().isoformat()
        })

        # Send current account info
        account_info = await stream_manager.get_account_info()
        if account_info:
            await websocket.send_json({
                "type": "account_update",
                "data": account_info,
                "timestamp": datetime.now().isoformat()
            })

        # Send current positions
        positions = await stream_manager.get_current_positions()
        await websocket.send_json({
            "type": "positions_snapshot",
            "data": positions,
            "timestamp": datetime.now().isoformat()
        })

        # Keep connection alive and handle messages
        while True:
            try:
                data = await websocket.receive_text()
                # Echo back or handle client requests
                logger.debug(f"Received from client: {data}")
            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")

    finally:
        stream_manager.remove_client(websocket)


@app.on_event("startup")
async def startup_event():
    """Start Alpaca streaming and equity logging on server startup"""
    # Initialize equity logging database
    init_equity_log_db()

    # Start equity logging background task
    asyncio.create_task(log_equity_snapshot())
    logger.info("Equity logging background task started (5-minute intervals)")

    # Start Alpaca data caching background task
    asyncio.create_task(fetch_and_cache_alpaca_data())
    logger.info("Alpaca data caching task started (30-second intervals)")

    if STREAMING_ENABLED:
        logger.info("Starting Alpaca stream...")
        stream_manager = get_stream_manager()

        # Get list of symbols to track from open positions
        try:
            conn = get_db_connection(POSITIONS_DB)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM positions WHERE status = 'open'")
            symbols = [row[0] for row in cursor.fetchall()]
            conn.close()

            if not symbols:
                logger.info("No open positions to track")
                symbols = []

            # Start streaming in background task
            asyncio.create_task(stream_manager.start_streaming(symbols))
            logger.info(f"Alpaca stream started for {len(symbols)} symbols")

        except Exception as e:
            logger.error(f"Error starting Alpaca stream: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop Alpaca streaming on server shutdown"""
    if STREAMING_ENABLED:
        logger.info("Stopping Alpaca stream...")
        stream_manager = get_stream_manager()
        await stream_manager.stop_streaming()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
