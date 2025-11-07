"""
Simulated Broker: Pure simulation for testing without external APIs.

Simulates trade execution with:
- Realistic slippage modeling
- Commission costs
- Market price fills
- Position size calculation
- Order result tracking
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel

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


class SimulatedBroker:
    """
    Pure simulation broker for testing.

    Simulates order execution without external API calls.
    Applies realistic slippage and commission costs.
    """

    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission_per_trade: float = 1.0,
        slippage_bps: float = 5.0
    ):
        """
        Initialize simulated broker.

        Args:
            initial_cash: Starting cash balance
            commission_per_trade: Fixed commission per trade ($)
            slippage_bps: Slippage in basis points (default 5 = 0.05%)
        """
        self.cash = initial_cash
        self.equity = initial_cash
        self.commission_per_trade = commission_per_trade
        self.slippage_bps = slippage_bps
        self.order_counter = 0

        logger.info(f"SimulatedBroker initialized with ${initial_cash:,.2f}")
        logger.info(f"Commission: ${commission_per_trade:.2f} per trade")
        logger.info(f"Slippage: {slippage_bps} bps")

    def update_cash(self, cash: float, equity: float):
        """
        Update cash and equity from external source.

        Args:
            cash: New cash balance
            equity: New total equity
        """
        self.cash = cash
        self.equity = equity

    def calculate_position_size(
        self,
        plan: TradePlan,
        account_equity: Optional[float] = None
    ) -> int:
        """
        Calculate position size in shares based on trade plan.

        Args:
            plan: Trade plan with position_size_pct
            account_equity: Current account equity (uses self.equity if None)

        Returns:
            Number of shares to buy/sell
        """
        if account_equity is None:
            account_equity = self.equity

        # Calculate dollar amount to allocate
        dollar_amount = account_equity * plan.position_size_pct

        # Calculate shares (round down to avoid over-allocation)
        shares = int(dollar_amount / plan.entry)

        # Ensure at least 1 share
        shares = max(1, shares)

        logger.info(f"{plan.symbol}: Position size = {shares} shares "
                   f"(${dollar_amount:.2f} / ${plan.entry:.2f})")

        return shares

    def apply_slippage(self, price: float, direction: str) -> float:
        """
        Apply slippage to entry price.

        Args:
            price: Original price
            direction: "long" (buy) or "short" (sell)

        Returns:
            Price with slippage applied
        """
        slippage_factor = self.slippage_bps / 10000.0

        if direction == "long":
            # Buy: slippage increases price
            return price * (1 + slippage_factor)
        else:
            # Short: slippage decreases price (worse entry)
            return price * (1 - slippage_factor)

    def submit_trade(
        self,
        plan: TradePlan,
        order_type: str = "market",
        time_in_force: str = "day"
    ) -> OrderResult:
        """
        Simulate a trade based on trade plan.

        Args:
            plan: Trade plan from trader agent
            order_type: "market" or "limit" (always executes in simulation)
            time_in_force: Ignored in simulation

        Returns:
            OrderResult with execution details
        """
        timestamp = datetime.now().isoformat()
        self.order_counter += 1
        order_id = f"SIM-{self.order_counter:06d}"

        try:
            # Calculate position size
            qty = self.calculate_position_size(plan)

            # Apply slippage to entry price
            filled_price = self.apply_slippage(plan.entry, plan.direction)

            # Calculate total cost
            position_value = filled_price * qty
            total_cost = position_value + self.commission_per_trade

            # Check if we have enough cash
            if total_cost > self.cash:
                error_msg = f"Insufficient cash: need ${total_cost:.2f}, have ${self.cash:.2f}"
                logger.warning(error_msg)
                return OrderResult(
                    success=False,
                    error_message=error_msg,
                    timestamp=timestamp
                )

            # Simulate successful fill
            logger.info(f"\n{'='*60}")
            logger.info(f"Simulating MARKET order for {plan.symbol}")
            logger.info(f"Direction: {plan.direction.upper()}")
            logger.info(f"Quantity: {qty} shares")
            logger.info(f"Entry (planned): ${plan.entry:.2f}")
            logger.info(f"Entry (filled): ${filled_price:.2f} (slippage: ${filled_price - plan.entry:.4f})")
            logger.info(f"Stop Loss: ${plan.stop_loss:.2f}")
            logger.info(f"Take Profit: ${plan.take_profit:.2f}")
            logger.info(f"Risk/Reward: {plan.risk_reward_ratio:.2f}")
            logger.info(f"Position Value: ${position_value:.2f}")
            logger.info(f"Commission: ${self.commission_per_trade:.2f}")
            logger.info(f"Total Cost: ${total_cost:.2f}")
            logger.info(f"{'='*60}")

            # Cash deduction handled externally by position manager
            # We don't actually modify self.cash here to avoid double-counting

            return OrderResult(
                success=True,
                order_id=order_id,
                filled_price=filled_price,
                filled_qty=float(qty),
                status="filled",
                timestamp=timestamp
            )

        except Exception as e:
            error_msg = f"Error simulating order: {e}"
            logger.error(error_msg, exc_info=True)
            return OrderResult(
                success=False,
                error_message=error_msg,
                timestamp=timestamp
            )

    def get_account_info(self) -> Dict[str, Any]:
        """
        Get simulated account information.

        Returns:
            Dict with account details
        """
        return {
            'account_number': 'SIMULATED',
            'status': 'ACTIVE',
            'currency': 'USD',
            'buying_power': self.cash,
            'cash': self.cash,
            'portfolio_value': self.equity,
            'equity': self.equity,
            'last_equity': self.equity,
            'pattern_day_trader': False,
            'trading_blocked': False,
            'transfers_blocked': False,
            'account_blocked': False,
            'created_at': datetime.now().isoformat()
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

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
        rationale=["Test trade for simulation"],
        risk_reward_ratio=2.0
    )

    # Initialize simulated broker
    broker = SimulatedBroker(
        initial_cash=100000.0,
        commission_per_trade=1.0,
        slippage_bps=5.0
    )

    # Get account info
    account_info = broker.get_account_info()
    print("\nSimulated Account Info:")
    print(f"  Portfolio Value: ${account_info['portfolio_value']:,.2f}")
    print(f"  Buying Power: ${account_info['buying_power']:,.2f}")
    print(f"  Cash: ${account_info['cash']:,.2f}")

    # Submit trade
    result = broker.submit_trade(test_plan)
    print(f"\nOrder Result:")
    print(f"  Success: {result.success}")
    if result.success:
        print(f"  Order ID: {result.order_id}")
        print(f"  Filled Price: ${result.filled_price:.2f}")
        print(f"  Filled Qty: {result.filled_qty}")
        print(f"  Status: {result.status}")
    else:
        print(f"  Error: {result.error_message}")
