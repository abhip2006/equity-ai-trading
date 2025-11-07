#!/usr/bin/env python3
"""
Alpaca Live Streaming Client
Streams real-time updates from Alpaca: positions, orders, trades, account updates
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Set, Optional
import sqlite3
from pathlib import Path

from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
from alpaca.trading.enums import OrderSide

logger = logging.getLogger(__name__)

# Database paths
BASE_DIR = Path(__file__).parent.parent
POSITIONS_DB = BASE_DIR / "data" / "positions.db"


class AlpacaStreamManager:
    """Manages Alpaca streaming connections and database updates"""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        """
        Initialize Alpaca streaming manager

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Use paper trading account (default: True)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper

        # Initialize clients
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        self.stream = TradingStream(api_key, secret_key, paper=paper)

        # Track active connections
        self.connected_clients: Set = set()
        self.is_streaming = False

        logger.info(f"AlpacaStreamManager initialized (paper={paper})")

    def get_db_connection(self):
        """Create database connection"""
        conn = sqlite3.connect(str(POSITIONS_DB))
        conn.row_factory = sqlite3.Row
        return conn

    async def handle_trade_update(self, data):
        """Handle trade (order fill) updates from Alpaca"""
        try:
            logger.info(f"Trade update received: {data}")

            order = data.order
            event = data.event

            # Prepare update data
            update = {
                "type": "trade_update",
                "timestamp": datetime.now().isoformat(),
                "event": event,
                "symbol": order.symbol,
                "order_id": str(order.id),
                "side": order.side.value,
                "qty": float(order.qty) if order.qty else 0,
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                "status": order.status.value,
            }

            # Update database based on event
            if event == "fill":
                await self._update_position_on_fill(order)
            elif event == "canceled" or event == "rejected":
                logger.warning(f"Order {order.id} {event}: {order.symbol}")

            # Broadcast to connected clients
            await self.broadcast(update)

        except Exception as e:
            logger.error(f"Error handling trade update: {e}", exc_info=True)

    async def _update_position_on_fill(self, order):
        """Update position in database when order is filled"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()

            symbol = order.symbol
            filled_qty = float(order.filled_qty) if order.filled_qty else 0
            filled_price = float(order.filled_avg_price) if order.filled_avg_price else 0
            side = order.side

            if filled_qty == 0 or filled_price == 0:
                logger.warning(f"Invalid fill data for {symbol}")
                return

            # Check if we have an open position for this symbol
            cursor.execute("""
                SELECT id, shares, entry_price
                FROM positions
                WHERE symbol = ? AND status = 'open'
                LIMIT 1
            """, (symbol,))

            existing = cursor.fetchone()

            if side == OrderSide.BUY:
                if existing:
                    # Update existing position (adding to it)
                    new_shares = existing[1] + int(filled_qty)
                    new_avg_price = ((existing[1] * existing[2]) + (filled_qty * filled_price)) / new_shares

                    cursor.execute("""
                        UPDATE positions
                        SET shares = ?, entry_price = ?
                        WHERE id = ?
                    """, (new_shares, new_avg_price, existing[0]))

                    logger.info(f"Updated position: {symbol} +{filled_qty} shares @ {filled_price}")
                else:
                    logger.info(f"New position opened via Alpaca fill: {symbol} (not tracked in our system)")
                    # Note: New positions opened directly via Alpaca won't have stop/target data
                    # They would need to be added by the main trading system

            elif side == OrderSide.SELL:
                if existing:
                    # Closing or reducing position
                    new_shares = existing[1] - int(filled_qty)

                    if new_shares <= 0:
                        # Close position
                        pnl = (filled_price - existing[2]) * existing[1]

                        cursor.execute("""
                            UPDATE positions
                            SET status = 'closed',
                                closed_at = ?,
                                exit_price = ?,
                                exit_reason = 'alpaca_fill',
                                realized_pnl = ?
                            WHERE id = ?
                        """, (datetime.now().isoformat(), filled_price, pnl, existing[0]))

                        logger.info(f"Closed position: {symbol} @ {filled_price}, P&L: ${pnl:.2f}")
                    else:
                        # Partial close
                        cursor.execute("""
                            UPDATE positions
                            SET shares = ?
                            WHERE id = ?
                        """, (new_shares, existing[0]))

                        logger.info(f"Reduced position: {symbol} -{filled_qty} shares @ {filled_price}")

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error updating position on fill: {e}", exc_info=True)

    async def handle_quote_update(self, data):
        """Handle real-time quote updates"""
        try:
            # Only broadcast, don't store every quote
            update = {
                "type": "quote",
                "timestamp": datetime.now().isoformat(),
                "symbol": data.symbol,
                "bid": float(data.bid_price),
                "ask": float(data.ask_price),
                "bid_size": int(data.bid_size),
                "ask_size": int(data.ask_size),
            }

            await self.broadcast(update)

        except Exception as e:
            logger.error(f"Error handling quote update: {e}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected WebSocket clients"""
        if not self.connected_clients:
            return

        # Send to all connected clients
        disconnected = set()
        for client in self.connected_clients:
            try:
                await client.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.add(client)

        # Remove disconnected clients
        self.connected_clients -= disconnected

    async def start_streaming(self, symbols: Optional[list] = None):
        """Start streaming from Alpaca"""
        try:
            # Subscribe to trade updates (order fills, etc)
            self.stream.subscribe_trade_updates(self.handle_trade_update)

            # Subscribe to quotes for tracked symbols
            if symbols:
                for symbol in symbols:
                    self.stream.subscribe_quotes(self.handle_quote_update, symbol)
                    logger.info(f"Subscribed to quotes: {symbol}")

            # Start streaming
            logger.info("Starting Alpaca stream...")
            self.is_streaming = True
            await self.stream._run_forever()

        except Exception as e:
            logger.error(f"Error in Alpaca stream: {e}", exc_info=True)
            self.is_streaming = False

    async def stop_streaming(self):
        """Stop streaming"""
        try:
            logger.info("Stopping Alpaca stream...")
            self.is_streaming = False
            await self.stream.stop()
        except Exception as e:
            logger.error(f"Error stopping stream: {e}")

    def add_client(self, client):
        """Add WebSocket client"""
        self.connected_clients.add(client)
        logger.info(f"Client connected. Total clients: {len(self.connected_clients)}")

    def remove_client(self, client):
        """Remove WebSocket client"""
        self.connected_clients.discard(client)
        logger.info(f"Client disconnected. Total clients: {len(self.connected_clients)}")

    async def get_current_positions(self):
        """Fetch current positions from Alpaca"""
        try:
            positions = self.trading_client.get_all_positions()

            result = []
            for pos in positions:
                result.append({
                    "symbol": pos.symbol,
                    "qty": float(pos.qty),
                    "side": pos.side.value,
                    "avg_entry_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "market_value": float(pos.market_value),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                })

            return result

        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []

    async def get_account_info(self):
        """Fetch account information from Alpaca"""
        try:
            account = self.trading_client.get_account()

            return {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "last_equity": float(account.last_equity) if account.last_equity else float(account.equity),
            }

        except Exception as e:
            logger.error(f"Error fetching account: {e}")
            return None


# Global stream manager instance
stream_manager: Optional[AlpacaStreamManager] = None


def get_stream_manager() -> AlpacaStreamManager:
    """Get or create stream manager"""
    global stream_manager

    if stream_manager is None:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise ValueError("Alpaca credentials not found in environment")

        stream_manager = AlpacaStreamManager(api_key, secret_key, paper=True)

    return stream_manager
