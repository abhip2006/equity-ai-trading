"""
Realistic Execution Simulator for Backtesting.

Simulates real-world execution constraints including:
- Volume-based slippage
- Commission structure
- Market hours validation
- Partial fills
- Gap handling
"""

import logging
from datetime import datetime, time
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
import pandas as pd

logger = logging.getLogger(__name__)


class ExecutionCostConfig(BaseModel):
    """Configuration for execution cost modeling."""
    # Commission structure
    commission_per_share: float = Field(default=0.005, description="Commission per share (e.g., $0.005)")
    min_commission: float = Field(default=1.0, description="Minimum commission per trade")
    max_commission_pct: float = Field(default=0.5, description="Max commission as % of trade value")

    # Slippage modeling
    base_slippage_bps: float = Field(default=2.0, description="Base slippage in basis points")
    volume_impact_factor: float = Field(default=0.1, description="Impact of trade size relative to volume")

    # Market hours (Eastern Time)
    market_open: time = Field(default=time(9, 30), description="Market open time")
    market_close: time = Field(default=time(16, 0), description="Market close time")

    # Execution constraints
    max_position_pct_of_adv: float = Field(default=5.0, description="Max position as % of avg daily volume")
    allow_premarket: bool = Field(default=False, description="Allow pre-market trading")
    allow_afterhours: bool = Field(default=False, description="Allow after-hours trading")


class ExecutionResult(BaseModel):
    """Result of simulated trade execution."""
    success: bool = Field(..., description="Whether execution succeeded")
    filled_price: Optional[float] = Field(None, description="Price at which order filled")
    filled_qty: Optional[int] = Field(None, description="Number of shares filled")
    slippage: Optional[float] = Field(None, description="Slippage in dollars per share")
    slippage_bps: Optional[float] = Field(None, description="Slippage in basis points")
    commission: Optional[float] = Field(None, description="Commission charged")
    total_cost: Optional[float] = Field(None, description="Total execution cost (slippage + commission)")
    rejection_reason: Optional[str] = Field(None, description="Reason for rejection if not filled")
    timestamp: datetime = Field(default_factory=datetime.now, description="Execution timestamp")
    is_partial_fill: bool = Field(default=False, description="Whether this is a partial fill")


class ExecutionSimulator:
    """
    Simulates realistic trade execution for backtesting.

    Models real-world execution constraints and costs.
    """

    def __init__(self, config: Optional[ExecutionCostConfig] = None):
        """
        Initialize execution simulator.

        Args:
            config: Execution cost configuration
        """
        self.config = config or ExecutionCostConfig()
        logger.info("ExecutionSimulator initialized")
        logger.info(f"  Commission: ${self.config.commission_per_share}/share (min ${self.config.min_commission})")
        logger.info(f"  Base slippage: {self.config.base_slippage_bps} bps")

    def simulate_execution(
        self,
        symbol: str,
        direction: Literal['long', 'short'],
        quantity: int,
        entry_price: float,
        execution_time: Optional[datetime] = None,
        avg_daily_volume: Optional[int] = None,
        current_spread_bps: Optional[float] = None
    ) -> ExecutionResult:
        """
        Simulate trade execution with realistic costs.

        Args:
            symbol: Stock symbol
            direction: 'long' or 'short'
            quantity: Number of shares to execute
            entry_price: Target entry price
            execution_time: When execution occurs (for market hours check)
            avg_daily_volume: Average daily volume for the symbol
            current_spread_bps: Current bid-ask spread in basis points

        Returns:
            ExecutionResult with fill details and costs
        """
        execution_time = execution_time or datetime.now()

        # Check market hours
        if not self._is_market_hours(execution_time):
            return ExecutionResult(
                success=False,
                rejection_reason=f"Outside market hours ({execution_time.time()})"
            )

        # Check volume constraints
        if avg_daily_volume is not None:
            max_shares = int(avg_daily_volume * (self.config.max_position_pct_of_adv / 100))
            if quantity > max_shares:
                logger.warning(
                    f"{symbol}: Order size {quantity} exceeds {self.config.max_position_pct_of_adv}% "
                    f"of ADV ({max_shares} shares)"
                )
                # Allow partial fill
                quantity = max_shares

        # Calculate slippage
        slippage_per_share = self._calculate_slippage(
            entry_price=entry_price,
            quantity=quantity,
            avg_daily_volume=avg_daily_volume,
            current_spread_bps=current_spread_bps,
            direction=direction
        )

        # Calculate fill price
        if direction == 'long':
            filled_price = entry_price + slippage_per_share
        else:  # short
            filled_price = entry_price - slippage_per_share

        # Calculate slippage in basis points
        slippage_bps = (abs(slippage_per_share) / entry_price) * 10000

        # Calculate commission
        commission = self._calculate_commission(quantity, filled_price)

        # Total execution cost
        total_cost = abs(slippage_per_share * quantity) + commission

        return ExecutionResult(
            success=True,
            filled_price=round(filled_price, 2),
            filled_qty=quantity,
            slippage=round(slippage_per_share, 4),
            slippage_bps=round(slippage_bps, 2),
            commission=round(commission, 2),
            total_cost=round(total_cost, 2),
            timestamp=execution_time,
            is_partial_fill=(avg_daily_volume is not None and
                           quantity < int(avg_daily_volume * (self.config.max_position_pct_of_adv / 100)))
        )

    def _calculate_slippage(
        self,
        entry_price: float,
        quantity: int,
        avg_daily_volume: Optional[int],
        current_spread_bps: Optional[float],
        direction: Literal['long', 'short']
    ) -> float:
        """
        Calculate realistic slippage based on order characteristics.

        Slippage components:
        1. Base slippage (crossing the spread)
        2. Volume impact (larger orders move the market)
        3. Spread component (wider spreads = more slippage)

        Args:
            entry_price: Target entry price
            quantity: Number of shares
            avg_daily_volume: Average daily volume
            current_spread_bps: Bid-ask spread in bps
            direction: Trade direction

        Returns:
            Slippage in dollars per share
        """
        # Start with base slippage
        slippage_bps = self.config.base_slippage_bps

        # Add spread component (half spread for crossing)
        if current_spread_bps is not None:
            slippage_bps += current_spread_bps / 2

        # Add volume impact
        if avg_daily_volume is not None and avg_daily_volume > 0:
            order_pct_of_adv = (quantity / avg_daily_volume) * 100
            # Square root impact model (larger orders have non-linear impact)
            volume_impact_bps = self.config.volume_impact_factor * (order_pct_of_adv ** 0.5)
            slippage_bps += volume_impact_bps

        # Convert basis points to dollars
        slippage_dollars = (slippage_bps / 10000) * entry_price

        return slippage_dollars

    def _calculate_commission(self, quantity: int, fill_price: float) -> float:
        """
        Calculate commission based on share quantity and value.

        Uses per-share commission with min/max constraints.

        Args:
            quantity: Number of shares
            fill_price: Fill price per share

        Returns:
            Commission in dollars
        """
        # Base commission
        commission = quantity * self.config.commission_per_share

        # Apply minimum
        commission = max(commission, self.config.min_commission)

        # Apply maximum (% of trade value)
        trade_value = quantity * fill_price
        max_commission = trade_value * (self.config.max_commission_pct / 100)
        commission = min(commission, max_commission)

        return commission

    def _is_market_hours(self, execution_time: datetime) -> bool:
        """
        Check if execution time is within market hours.

        Args:
            execution_time: Time to check

        Returns:
            True if within market hours
        """
        exec_time = execution_time.time()

        # Regular market hours
        if self.config.market_open <= exec_time <= self.config.market_close:
            return True

        # Pre-market (4:00 AM - 9:30 AM ET)
        if self.config.allow_premarket and time(4, 0) <= exec_time < self.config.market_open:
            return True

        # After-hours (4:00 PM - 8:00 PM ET)
        if self.config.allow_afterhours and self.config.market_close < exec_time <= time(20, 0):
            return True

        return False

    def simulate_exit(
        self,
        symbol: str,
        direction: Literal['long', 'short'],
        quantity: int,
        exit_price: float,
        exit_reason: Literal['stop_loss', 'take_profit', 'manual', 'eod_close'],
        execution_time: Optional[datetime] = None
    ) -> ExecutionResult:
        """
        Simulate position exit with appropriate modeling.

        Stop losses and take profits may have different slippage characteristics.

        Args:
            symbol: Stock symbol
            direction: Position direction
            quantity: Shares to close
            exit_price: Target exit price
            exit_reason: Reason for exit
            execution_time: When exit occurs

        Returns:
            ExecutionResult
        """
        # Stop losses typically have worse slippage (urgency, market moving against you)
        if exit_reason == 'stop_loss':
            # Increase slippage for stop loss exits
            config_copy = self.config.model_copy()
            config_copy.base_slippage_bps *= 1.5  # 50% more slippage

            temp_simulator = ExecutionSimulator(config_copy)
            return temp_simulator.simulate_execution(
                symbol=symbol,
                direction='short' if direction == 'long' else 'long',  # Opposite direction for exit
                quantity=quantity,
                entry_price=exit_price,
                execution_time=execution_time
            )

        # Normal exit (take profit, manual, EOD)
        return self.simulate_execution(
            symbol=symbol,
            direction='short' if direction == 'long' else 'long',  # Opposite direction for exit
            quantity=quantity,
            entry_price=exit_price,
            execution_time=execution_time
        )


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Testing ExecutionSimulator")
    print("=" * 60)

    # Create simulator with default config
    simulator = ExecutionSimulator()

    # Test 1: Small order with low volume impact
    print("\n" + "=" * 60)
    print("Test 1: Small AAPL order (100 shares)")
    print("=" * 60)

    # Use market hours time (10:00 AM)
    market_time = datetime.now().replace(hour=10, minute=0, second=0)

    result = simulator.simulate_execution(
        symbol="AAPL",
        direction="long",
        quantity=100,
        entry_price=175.0,
        execution_time=market_time,
        avg_daily_volume=50_000_000,  # 50M avg volume
        current_spread_bps=2.0  # 2 bps spread
    )

    print(f"Success: {result.success}")
    print(f"Filled: {result.filled_qty} shares @ ${result.filled_price}")
    print(f"Slippage: ${result.slippage} ({result.slippage_bps} bps)")
    print(f"Commission: ${result.commission}")
    print(f"Total cost: ${result.total_cost}")

    # Test 2: Large order with volume impact
    print("\n" + "=" * 60)
    print("Test 2: Large order (10,000 shares)")
    print("=" * 60)

    result = simulator.simulate_execution(
        symbol="SMCI",
        direction="long",
        quantity=10000,
        entry_price=50.0,
        execution_time=market_time,
        avg_daily_volume=1_000_000,  # 1M avg volume (1% of ADV)
        current_spread_bps=5.0  # Wider spread
    )

    print(f"Success: {result.success}")
    print(f"Filled: {result.filled_qty} shares @ ${result.filled_price}")
    print(f"Slippage: ${result.slippage} ({result.slippage_bps} bps)")
    print(f"Commission: ${result.commission}")
    print(f"Total cost: ${result.total_cost}")

    # Test 3: Stop loss exit (higher slippage)
    print("\n" + "=" * 60)
    print("Test 3: Stop loss exit")
    print("=" * 60)

    result = simulator.simulate_exit(
        symbol="AAPL",
        direction="long",
        quantity=100,
        exit_price=172.0,
        exit_reason="stop_loss",
        execution_time=market_time
    )

    print(f"Success: {result.success}")
    print(f"Filled: {result.filled_qty} shares @ ${result.filled_price}")
    print(f"Slippage: ${result.slippage} ({result.slippage_bps} bps)")
    print(f"Total cost: ${result.total_cost}")

    # Test 4: After-hours rejection
    print("\n" + "=" * 60)
    print("Test 4: After-hours execution (should reject)")
    print("=" * 60)

    after_hours_time = datetime.now().replace(hour=18, minute=0)
    result = simulator.simulate_execution(
        symbol="AAPL",
        direction="long",
        quantity=100,
        entry_price=175.0,
        execution_time=after_hours_time
    )

    print(f"Success: {result.success}")
    print(f"Rejection reason: {result.rejection_reason}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
