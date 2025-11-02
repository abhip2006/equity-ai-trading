"""
ExitMonitor - Continuous monitoring of stop-loss and take-profit conditions

This module provides the ExitMonitor class which continuously monitors open positions
for exit conditions (stop-loss, take-profit) and executes exits automatically when
conditions are met.
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ExitMonitor:
    """
    Monitors open positions for exit conditions (stop-loss, take-profit)
    Executes exits automatically when conditions are met
    """

    def __init__(self, position_manager):
        """
        Initialize the ExitMonitor

        Args:
            position_manager: PositionManager instance to manage positions
        """
        self.position_manager = position_manager
        logger.info("ExitMonitor initialized")

    def check_exits(self, current_prices: Dict[str, float]) -> List:
        """
        Check all open positions for exit conditions

        Args:
            current_prices: Dict mapping symbol -> current price

        Returns:
            List of positions that were closed
        """
        logger.debug(f"Checking exits for {len(current_prices)} symbols")

        # Get all open positions
        open_positions = self.position_manager.get_open_positions()
        closed_positions = []

        # For each position, check if it should be exited
        for position in open_positions:
            symbol = position.symbol

            # Skip if we don't have a current price for this symbol
            if symbol not in current_prices:
                logger.warning(f"No current price available for {symbol}, skipping exit check")
                continue

            current_price = current_prices[symbol]

            # Check if position should be exited
            should_exit, reason = self.should_exit_position(position, current_price)

            if should_exit:
                logger.info(f"Exit condition met for {symbol}: {reason}")
                # Execute the exit
                closed_position = self.execute_exit(position, current_price, reason)
                closed_positions.append(closed_position)

        if closed_positions:
            logger.info(f"Closed {len(closed_positions)} positions")

        return closed_positions

    def should_exit_position(self, position, current_price: float) -> Tuple[bool, Optional[str]]:
        """
        Determine if position should be exited

        Args:
            position: Position object to check
            current_price: Current market price

        Returns:
            (should_exit: bool, reason: Optional[str])
            reason can be 'stop_loss', 'take_profit', or None
        """
        # Check if position has stop_loss and take_profit attributes
        if not hasattr(position, 'stop_loss') or not hasattr(position, 'take_profit'):
            logger.warning(f"Position {position.symbol} missing stop_loss or take_profit")
            return (False, None)

        # Get position direction
        is_long = position.side.lower() == 'long' or position.quantity > 0

        if is_long:
            # Long position exit conditions
            # Stop loss: price drops to or below stop_loss
            if position.stop_loss is not None and current_price <= position.stop_loss:
                pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                logger.info(
                    f"Stop loss triggered for {position.symbol}: "
                    f"price={current_price:.2f}, stop={position.stop_loss:.2f}, "
                    f"PnL={pnl_pct:.2f}%"
                )
                return (True, 'stop_loss')

            # Take profit: price rises to or above take_profit
            if position.take_profit is not None and current_price >= position.take_profit:
                pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                logger.info(
                    f"Take profit triggered for {position.symbol}: "
                    f"price={current_price:.2f}, target={position.take_profit:.2f}, "
                    f"PnL={pnl_pct:.2f}%"
                )
                return (True, 'take_profit')
        else:
            # Short position exit conditions
            # Stop loss: price rises to or above stop_loss
            if position.stop_loss is not None and current_price >= position.stop_loss:
                pnl_pct = ((position.entry_price - current_price) / position.entry_price) * 100
                logger.info(
                    f"Stop loss triggered for {position.symbol}: "
                    f"price={current_price:.2f}, stop={position.stop_loss:.2f}, "
                    f"PnL={pnl_pct:.2f}%"
                )
                return (True, 'stop_loss')

            # Take profit: price drops to or below take_profit
            if position.take_profit is not None and current_price <= position.take_profit:
                pnl_pct = ((position.entry_price - current_price) / position.entry_price) * 100
                logger.info(
                    f"Take profit triggered for {position.symbol}: "
                    f"price={current_price:.2f}, target={position.take_profit:.2f}, "
                    f"PnL={pnl_pct:.2f}%"
                )
                return (True, 'take_profit')

        # No exit condition met
        return (False, None)

    def execute_exit(self, position, current_price: float, reason: str):
        """
        Execute position exit

        Args:
            position: Position to close
            current_price: Exit price
            reason: Exit reason ('stop_loss', 'take_profit', 'manual', 'eod_close')

        Returns:
            Closed position with P&L
        """
        logger.info(
            f"Executing exit for {position.symbol}: "
            f"reason={reason}, price={current_price:.2f}"
        )

        # Call position_manager.close_position()
        closed_position = self.position_manager.close_position(
            position=position,
            exit_price=current_price,
            exit_reason=reason
        )

        # Calculate P&L for logging
        if hasattr(closed_position, 'pnl') and hasattr(closed_position, 'pnl_percent'):
            logger.info(
                f"Position closed: {position.symbol} | "
                f"Reason: {reason} | "
                f"Entry: ${position.entry_price:.2f} | "
                f"Exit: ${current_price:.2f} | "
                f"P&L: ${closed_position.pnl:.2f} ({closed_position.pnl_percent:.2f}%)"
            )
        else:
            logger.info(
                f"Position closed: {position.symbol} | "
                f"Reason: {reason} | "
                f"Entry: ${position.entry_price:.2f} | "
                f"Exit: ${current_price:.2f}"
            )

        return closed_position

    def close_all_positions(self, current_prices: Dict[str, float], reason: str = 'eod_close') -> List:
        """
        Close all open positions (for end-of-day cleanup)

        Args:
            current_prices: Current prices for all positions
            reason: Why closing (default 'eod_close')

        Returns:
            List of closed positions
        """
        logger.info(f"Closing all positions: reason={reason}")

        # Get all open positions
        open_positions = self.position_manager.get_open_positions()
        closed_positions = []

        # Close each position
        for position in open_positions:
            symbol = position.symbol

            # Get current price, use entry price as fallback if not available
            if symbol in current_prices:
                current_price = current_prices[symbol]
            else:
                logger.warning(
                    f"No current price for {symbol}, using entry price for close"
                )
                current_price = position.entry_price

            # Execute exit
            try:
                closed_position = self.execute_exit(position, current_price, reason)
                closed_positions.append(closed_position)
            except Exception as e:
                logger.error(
                    f"Error closing position {symbol}: {e}",
                    exc_info=True
                )

        logger.info(f"Closed {len(closed_positions)} positions (reason: {reason})")
        return closed_positions
