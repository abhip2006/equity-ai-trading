"""
Alpaca Broker Executor: Execute trades via Alpaca Trading API.

Supports:
- Bracket orders with stop loss and take profit
- Market and limit order types
- Position tracking
- Order status monitoring
- Paper trading and live trading modes
"""

import logging
import os
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        TakeProfitRequest,
        StopLossRequest
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, OrderStatus
    from alpaca.common.exceptions import APIError
except ImportError:
    logging.warning("alpaca-py not installed. Install with: pip install alpaca-py")
    TradingClient = None

from active_trader_llm.trader.trader_agent import TradePlan

logger = logging.getLogger(__name__)


class OrderResult(BaseModel):
    """Result of an order submission"""
    success: bool
    order_id: Optional[str] = None
    filled_price: Optional[float] = None
    filled_qty: Optional[float] = None
    status: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: str


class PositionInfo(BaseModel):
    """Position information from broker"""
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_pl_pct: float


class AlpacaBrokerExecutor:
    """
    Execute trades via Alpaca Trading API.

    Features:
    - Bracket orders with automatic stop loss and take profit
    - Support for both market and limit orders
    - Paper trading mode for testing
    - Position and order tracking
    - Error handling and retry logic
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
        base_url: Optional[str] = None,
        extended_hours: bool = False
    ):
        """
        Initialize Alpaca broker executor.

        Args:
            api_key: Alpaca API key (or uses ALPACA_API_KEY env var)
            secret_key: Alpaca secret key (or uses ALPACA_SECRET_KEY env var)
            paper: Use paper trading (default True for safety)
            base_url: Optional custom base URL (auto-selected if None)
            extended_hours: Enable extended hours (pre-market and after-hours) trading
        """
        if TradingClient is None:
            raise ImportError("alpaca-py required. Install: pip install alpaca-py")

        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')

        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not provided")

        # Initialize trading client
        # If base_url not specified, paper=True uses paper trading automatically
        if base_url:
            self.client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                url_override=base_url
            )
        else:
            self.client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=paper
            )

        self.paper = paper
        self.extended_hours = extended_hours
        self.mode = "PAPER" if paper else "LIVE"

        logger.info(f"AlpacaBrokerExecutor initialized in {self.mode} mode")
        if self.extended_hours:
            logger.info("Extended hours trading ENABLED (pre-market: 4:00-9:30 AM, after-hours: 4:00-8:00 PM ET)")

        # Verify connection
        try:
            account = self.client.get_account()
            logger.info(f"Account status: {account.status}")
            logger.info(f"Buying power: ${float(account.buying_power):,.2f}")
            logger.info(f"Portfolio value: ${float(account.portfolio_value):,.2f}")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise

    def calculate_position_size(
        self,
        plan: TradePlan,
        account_equity: Optional[float] = None
    ) -> int:
        """
        Calculate position size in shares based on trade plan.

        Args:
            plan: Trade plan with position_size_pct
            account_equity: Current account equity (fetched if None)

        Returns:
            Number of shares to buy/sell
        """
        if account_equity is None:
            logger.debug(f"{plan.symbol}: Fetching account info (no cache provided)")
            account = self.client.get_account()
            account_equity = float(account.equity)
        else:
            logger.debug(f"{plan.symbol}: Using cached account equity: ${account_equity:,.2f}")

        # Calculate dollar amount to allocate
        dollar_amount = account_equity * plan.position_size_pct

        # Calculate shares (round down to avoid over-allocation)
        shares = int(dollar_amount / plan.entry)

        # Ensure at least 1 share
        shares = max(1, shares)

        logger.info(f"{plan.symbol}: Position size = {shares} shares "
                   f"(${dollar_amount:.2f} / ${plan.entry:.2f})")

        return shares

    def submit_trade(
        self,
        plan: TradePlan,
        order_type: str = "market",
        time_in_force: str = "gtc",
        cached_account_equity: Optional[float] = None
    ) -> OrderResult:
        """
        Submit a trade based on trade plan.

        Args:
            plan: Trade plan from trader agent
            order_type: "market" or "limit" (default: market)
            time_in_force: "day", "gtc", "ioc", "fok" (default: gtc - keeps bracket orders active)
            cached_account_equity: Cached account equity to avoid API call (optional)

        Returns:
            OrderResult with execution details
        """
        timestamp = datetime.now().isoformat()

        try:
            # Calculate position size (use cached equity if provided)
            qty = self.calculate_position_size(plan, account_equity=cached_account_equity)

            # Determine order side
            side = OrderSide.BUY if plan.direction == "long" else OrderSide.SELL

            # Map time in force
            tif_map = {
                "day": TimeInForce.DAY,
                "gtc": TimeInForce.GTC,
                "ioc": TimeInForce.IOC,
                "fok": TimeInForce.FOK
            }
            tif = tif_map.get(time_in_force.lower(), TimeInForce.DAY)

            # Create bracket order with stop loss and take profit
            logger.info(f"\n{'='*60}")
            logger.info(f"Submitting {self.mode} order for {plan.symbol}")
            logger.info(f"Direction: {plan.direction.upper()}")
            logger.info(f"Action: {plan.action}")
            logger.info(f"Quantity: {qty} shares")
            logger.info(f"Entry: ${plan.entry:.2f}")
            logger.info(f"Stop Loss: ${plan.stop_loss:.2f}")
            logger.info(f"Take Profit: ${plan.take_profit:.2f}")
            logger.info(f"Risk/Reward: {plan.risk_reward_ratio:.2f}")
            logger.info(f"{'='*60}")

            if order_type == "market":
                order_request = MarketOrderRequest(
                    symbol=plan.symbol,
                    qty=qty,
                    side=side,
                    time_in_force=tif,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(
                        limit_price=plan.take_profit
                    ),
                    stop_loss=StopLossRequest(
                        stop_price=plan.stop_loss
                    ),
                    extended_hours=self.extended_hours
                )
            else:  # limit order
                order_request = LimitOrderRequest(
                    symbol=plan.symbol,
                    limit_price=plan.entry,
                    qty=qty,
                    side=side,
                    time_in_force=tif,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(
                        limit_price=plan.take_profit
                    ),
                    stop_loss=StopLossRequest(
                        stop_price=plan.stop_loss
                    ),
                    extended_hours=self.extended_hours
                )

            # Submit order
            order = self.client.submit_order(order_data=order_request)

            logger.info(f"Order submitted successfully!")
            logger.info(f"Order ID: {order.id}")
            logger.info(f"Status: {order.status}")

            # For market orders, wait for fill (they usually fill within seconds)
            filled_price = None
            filled_qty = None

            if order_type == "market":
                # Poll for fill status (market orders typically fill in 1-3 seconds)
                import time
                max_wait_seconds = 10
                poll_interval = 0.5
                elapsed = 0

                logger.info(f"Waiting for market order to fill...")

                while elapsed < max_wait_seconds:
                    # Refresh order status
                    current_order = self.client.get_order_by_id(order.id)

                    if current_order.filled_avg_price and current_order.filled_qty:
                        filled_price = float(current_order.filled_avg_price)
                        filled_qty = float(current_order.filled_qty)
                        logger.info(f"✓ Order FILLED: {filled_qty} shares @ ${filled_price:.2f}")
                        break

                    # Check if order is in terminal state without fill
                    order_status = str(current_order.status).lower()
                    if order_status in ['cancelled', 'expired', 'rejected', 'replaced']:
                        logger.warning(f"Order reached terminal state '{order_status}' without fill")
                        break

                    time.sleep(poll_interval)
                    elapsed += poll_interval

                if filled_qty is None:
                    logger.warning(f"Market order did not fill within {max_wait_seconds}s - may fill later")
            else:
                # Limit orders - check if immediately filled
                if order.filled_avg_price:
                    filled_price = float(order.filled_avg_price)
                    filled_qty = float(order.filled_qty) if order.filled_qty else None
                    if filled_qty:
                        logger.info(f"Filled: {filled_qty} shares @ ${filled_price:.2f}")

            return OrderResult(
                success=True,
                order_id=str(order.id),
                filled_price=filled_price,
                filled_qty=filled_qty,
                status=str(order.status),
                timestamp=timestamp
            )

        except APIError as e:
            error_msg = f"Alpaca API error: {e}"
            logger.error(error_msg)
            return OrderResult(
                success=False,
                error_message=error_msg,
                timestamp=timestamp
            )

        except Exception as e:
            error_msg = f"Error submitting order: {e}"
            logger.error(error_msg, exc_info=True)
            return OrderResult(
                success=False,
                error_message=error_msg,
                timestamp=timestamp
            )

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of an order.

        Args:
            order_id: Alpaca order ID

        Returns:
            Order details dict or None if error
        """
        try:
            order = self.client.get_order_by_id(order_id)

            return {
                'order_id': str(order.id),
                'symbol': order.symbol,
                'status': str(order.status),
                'filled_qty': float(order.filled_qty) if order.filled_qty else 0.0,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                'side': str(order.side),
                'type': str(order.type),
                'created_at': str(order.created_at),
                'updated_at': str(order.updated_at)
            }

        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return None

    def get_positions(self) -> List[PositionInfo]:
        """
        Get all current positions.

        Returns:
            List of PositionInfo objects
        """
        try:
            positions = self.client.get_all_positions()

            position_list = []
            for pos in positions:
                position_list.append(PositionInfo(
                    symbol=pos.symbol,
                    qty=float(pos.qty),
                    avg_entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    market_value=float(pos.market_value),
                    unrealized_pl=float(pos.unrealized_pl),
                    unrealized_pl_pct=float(pos.unrealized_plpc)
                ))

            return position_list

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """
        Get position for a specific symbol.

        Args:
            symbol: Stock symbol

        Returns:
            PositionInfo or None if no position
        """
        try:
            pos = self.client.get_open_position(symbol)

            return PositionInfo(
                symbol=pos.symbol,
                qty=float(pos.qty),
                avg_entry_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                market_value=float(pos.market_value),
                unrealized_pl=float(pos.unrealized_pl),
                unrealized_pl_pct=float(pos.unrealized_plpc)
            )

        except Exception as e:
            # Position doesn't exist - this is normal
            logger.debug(f"No position for {symbol}: {e}")
            return None

    def close_position(self, symbol: str) -> OrderResult:
        """
        Close an existing position.

        Args:
            symbol: Stock symbol to close

        Returns:
            OrderResult with execution details
        """
        timestamp = datetime.now().isoformat()

        try:
            logger.info(f"Closing position: {symbol}")

            order = self.client.close_position(symbol)

            logger.info(f"Position close order submitted: {order.id}")

            return OrderResult(
                success=True,
                order_id=str(order.id),
                status=str(order.status),
                timestamp=timestamp
            )

        except Exception as e:
            error_msg = f"Error closing position: {e}"
            logger.error(error_msg)
            return OrderResult(
                success=False,
                error_message=error_msg,
                timestamp=timestamp
            )

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Alpaca order ID

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.cancel_order_by_id(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True

        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    def get_account_info(self, use_cache: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get account information.

        Args:
            use_cache: Optional cached account dict to return instead of fetching

        Returns:
            Dict with account details
        """
        if use_cache is not None:
            logger.debug("Using cached account info (cache hit)")
            return use_cache

        try:
            logger.debug("Fetching account info from API (cache miss)")
            account = self.client.get_account()

            return {
                'account_number': account.account_number,
                'status': account.status,
                'currency': account.currency,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'last_equity': float(account.last_equity),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked,
                'created_at': str(account.created_at)
            }

        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}

    def get_account_info_cached(self, cached_account: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return formatted account info from cached account dict without API call.

        Useful when SharedDataFeed or coordinator has already fetched account info.

        Args:
            cached_account: Already fetched account dict

        Returns:
            Formatted account info dict (passthrough of cached data)
        """
        logger.debug("Returning cached account info (no API call)")
        return cached_account

    def get_bracket_orders(self, symbol: str) -> Dict[str, Optional[str]]:
        """
        Get bracket order IDs (stop loss and take profit) for a position.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with 'stop_order_id' and 'take_profit_order_id' (None if not found)
        """
        try:
            # Get all open orders for this symbol
            from alpaca.trading.enums import QueryOrderStatus
            orders = self.client.get_orders(
                status=QueryOrderStatus.OPEN,
                symbols=[symbol]
            )

            stop_order_id = None
            take_profit_order_id = None

            for order in orders:
                # Stop loss orders have stop_price
                if order.stop_price is not None and order.limit_price is None:
                    stop_order_id = str(order.id)
                    logger.debug(f"{symbol}: Found stop loss order {stop_order_id} @ ${float(order.stop_price):.2f}")

                # Take profit orders have limit_price but no stop_price
                elif order.limit_price is not None and order.stop_price is None:
                    take_profit_order_id = str(order.id)
                    logger.debug(f"{symbol}: Found take profit order {take_profit_order_id} @ ${float(order.limit_price):.2f}")

            return {
                'stop_order_id': stop_order_id,
                'take_profit_order_id': take_profit_order_id
            }

        except Exception as e:
            logger.error(f"Error getting bracket orders for {symbol}: {e}")
            return {'stop_order_id': None, 'take_profit_order_id': None}

    def modify_stop_loss(
        self,
        symbol: str,
        new_stop: float,
        reason: str = "LLM adjustment"
    ) -> OrderResult:
        """
        Modify stop loss on existing position's bracket order.

        Uses Alpaca's replace_order() API to update the stop loss order.

        Args:
            symbol: Stock symbol
            new_stop: New stop loss price
            reason: Reason for adjustment (for logging)

        Returns:
            OrderResult with success status
        """
        timestamp = datetime.now().isoformat()

        try:
            # Get current position to determine quantity and direction
            position = self.get_position(symbol)
            if not position:
                error_msg = f"No position found for {symbol}"
                logger.error(error_msg)
                return OrderResult(
                    success=False,
                    error_message=error_msg,
                    timestamp=timestamp
                )

            # Get bracket orders
            bracket_orders = self.get_bracket_orders(symbol)
            stop_order_id = bracket_orders['stop_order_id']

            if not stop_order_id:
                error_msg = f"No stop loss order found for {symbol}"
                logger.error(error_msg)
                return OrderResult(
                    success=False,
                    error_message=error_msg,
                    timestamp=timestamp
                )

            # Get current stop order details
            old_order = self.client.get_order_by_id(stop_order_id)
            old_stop = float(old_order.stop_price) if old_order.stop_price else None

            logger.info(f"\n{'='*60}")
            logger.info(f"Modifying stop loss for {symbol}")
            logger.info(f"Reason: {reason}")
            logger.info(f"Old stop: ${old_stop:.2f if old_stop else 0.0}")
            logger.info(f"New stop: ${new_stop:.2f}")
            logger.info(f"Position: {position.qty} shares @ ${position.avg_entry_price:.2f}")
            logger.info(f"{'='*60}")

            # Replace stop order with new stop price
            try:
                from alpaca.trading.requests import ReplaceOrderRequest
                new_order = self.client.replace_order_by_id(
                    order_id=stop_order_id,
                    order_data=ReplaceOrderRequest(
                        qty=abs(int(position.qty)),
                        time_in_force=old_order.time_in_force,
                        stop_price=new_stop
                    )
                )

                logger.info(f"Stop loss updated successfully!")
                logger.info(f"New order ID: {new_order.id}")

                return OrderResult(
                    success=True,
                    order_id=str(new_order.id),
                    status=str(new_order.status),
                    timestamp=timestamp
                )

            except ImportError:
                # Fallback: Cancel old order and create new one
                logger.warning("ReplaceOrderRequest not available, using cancel+create method")
                self.cancel_order(stop_order_id)

                # Determine side (opposite of position direction)
                side = OrderSide.SELL if position.qty > 0 else OrderSide.BUY

                # Submit new stop loss order
                stop_request = StopLossRequest(stop_price=new_stop)
                from alpaca.trading.requests import StopOrderRequest
                new_stop_order = self.client.submit_order(
                    StopOrderRequest(
                        symbol=symbol,
                        qty=abs(int(position.qty)),
                        side=side,
                        stop_price=new_stop,
                        time_in_force=old_order.time_in_force
                    )
                )

                logger.info(f"New stop loss order created: {new_stop_order.id}")

                return OrderResult(
                    success=True,
                    order_id=str(new_stop_order.id),
                    status=str(new_stop_order.status),
                    timestamp=timestamp
                )

        except Exception as e:
            error_msg = f"Error modifying stop loss: {e}"
            logger.error(error_msg, exc_info=True)
            return OrderResult(
                success=False,
                error_message=error_msg,
                timestamp=timestamp
            )

    def modify_take_profit(
        self,
        symbol: str,
        new_target: float,
        reason: str = "LLM adjustment"
    ) -> OrderResult:
        """
        Modify take profit on existing position's bracket order.

        Uses Alpaca's replace_order() API to update the take profit order.

        Args:
            symbol: Stock symbol
            new_target: New take profit price
            reason: Reason for adjustment (for logging)

        Returns:
            OrderResult with success status
        """
        timestamp = datetime.now().isoformat()

        try:
            # Get current position
            position = self.get_position(symbol)
            if not position:
                error_msg = f"No position found for {symbol}"
                logger.error(error_msg)
                return OrderResult(
                    success=False,
                    error_message=error_msg,
                    timestamp=timestamp
                )

            # Get bracket orders
            bracket_orders = self.get_bracket_orders(symbol)
            take_profit_order_id = bracket_orders['take_profit_order_id']

            if not take_profit_order_id:
                error_msg = f"No take profit order found for {symbol}"
                logger.error(error_msg)
                return OrderResult(
                    success=False,
                    error_message=error_msg,
                    timestamp=timestamp
                )

            # Get current take profit order details
            old_order = self.client.get_order_by_id(take_profit_order_id)
            old_target = float(old_order.limit_price) if old_order.limit_price else None

            logger.info(f"\n{'='*60}")
            logger.info(f"Modifying take profit for {symbol}")
            logger.info(f"Reason: {reason}")
            logger.info(f"Old target: ${old_target:.2f if old_target else 0.0}")
            logger.info(f"New target: ${new_target:.2f}")
            logger.info(f"Position: {position.qty} shares @ ${position.avg_entry_price:.2f}")
            logger.info(f"{'='*60}")

            # Replace take profit order with new target price
            try:
                from alpaca.trading.requests import ReplaceOrderRequest
                new_order = self.client.replace_order_by_id(
                    order_id=take_profit_order_id,
                    order_data=ReplaceOrderRequest(
                        qty=abs(int(position.qty)),
                        time_in_force=old_order.time_in_force,
                        limit_price=new_target
                    )
                )

                logger.info(f"Take profit updated successfully!")
                logger.info(f"New order ID: {new_order.id}")

                return OrderResult(
                    success=True,
                    order_id=str(new_order.id),
                    status=str(new_order.status),
                    timestamp=timestamp
                )

            except ImportError:
                # Fallback: Cancel old order and create new one
                logger.warning("ReplaceOrderRequest not available, using cancel+create method")
                self.cancel_order(take_profit_order_id)

                # Determine side (opposite of position direction)
                side = OrderSide.SELL if position.qty > 0 else OrderSide.BUY

                # Submit new limit order as take profit
                new_take_profit_order = self.client.submit_order(
                    LimitOrderRequest(
                        symbol=symbol,
                        qty=abs(int(position.qty)),
                        side=side,
                        limit_price=new_target,
                        time_in_force=old_order.time_in_force
                    )
                )

                logger.info(f"New take profit order created: {new_take_profit_order.id}")

                return OrderResult(
                    success=True,
                    order_id=str(new_take_profit_order.id),
                    status=str(new_take_profit_order.status),
                    timestamp=timestamp
                )

        except Exception as e:
            error_msg = f"Error modifying take profit: {e}"
            logger.error(error_msg, exc_info=True)
            return OrderResult(
                success=False,
                error_message=error_msg,
                timestamp=timestamp
            )

    def modify_bracket_order(
        self,
        symbol: str,
        new_stop: Optional[float] = None,
        new_target: Optional[float] = None,
        reason: str = "LLM adjustment"
    ) -> Dict[str, OrderResult]:
        """
        Modify both stop loss and take profit in one call.

        Args:
            symbol: Stock symbol
            new_stop: New stop loss price (None to skip)
            new_target: New take profit price (None to skip)
            reason: Reason for adjustment (for logging)

        Returns:
            Dict with 'stop' and 'target' OrderResult objects
        """
        results = {}

        if new_stop is not None:
            results['stop'] = self.modify_stop_loss(symbol, new_stop, reason)
        else:
            results['stop'] = OrderResult(
                success=True,
                timestamp=datetime.now().isoformat()
            )

        if new_target is not None:
            results['target'] = self.modify_take_profit(symbol, new_target, reason)
        else:
            results['target'] = OrderResult(
                success=True,
                timestamp=datetime.now().isoformat()
            )

        return results


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example: Create a test trade plan
    from active_trader_llm.trader.trader_agent import TradePlan

    test_plan = TradePlan(
        symbol="AAPL",
        strategy="momentum_breakout",
        direction="long",
        entry=175.0,
        stop_loss=170.0,
        take_profit=185.0,
        position_size_pct=0.05,  # 5% of portfolio
        time_horizon="3d",
        rationale=["Test trade for integration"],
        risk_reward_ratio=2.0
    )

    # Initialize executor in PAPER mode (safe for testing)
    executor = AlpacaBrokerExecutor(paper=True)

    # Get account info
    account_info = executor.get_account_info()
    print("\nAccount Info:")
    print(f"  Portfolio Value: ${account_info['portfolio_value']:,.2f}")
    print(f"  Buying Power: ${account_info['buying_power']:,.2f}")
    print(f"  Cash: ${account_info['cash']:,.2f}")

    # Submit trade (uncomment to actually submit)
    # result = executor.submit_trade(test_plan, order_type="market")
    # print(f"\nOrder Result:")
    # print(f"  Success: {result.success}")
    # if result.success:
    #     print(f"  Order ID: {result.order_id}")
    #     print(f"  Status: {result.status}")
    # else:
    #     print(f"  Error: {result.error_message}")

    # Get all positions
    positions = executor.get_positions()
    print(f"\nCurrent Positions: {len(positions)}")
    for pos in positions:
        print(f"  {pos.symbol}: {pos.qty} shares @ ${pos.avg_entry_price:.2f}")
        print(f"    Current: ${pos.current_price:.2f}, P&L: ${pos.unrealized_pl:.2f} ({pos.unrealized_pl_pct:.2%})")
