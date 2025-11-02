#!/usr/bin/env python3
"""
Test suite for ExitMonitor

Tests stop-loss and take-profit detection, exit execution, and bulk close operations.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Optional


# Mock classes for testing (will be replaced when actual implementation exists)
class Position:
    """Represents a trading position"""
    def __init__(self, symbol, side, entry_price, quantity, entry_time, stop_loss=None, take_profit=None):
        self.symbol = symbol
        self.side = side  # 'long' or 'short'
        self.entry_price = Decimal(str(entry_price))
        self.quantity = Decimal(str(quantity))
        self.entry_time = entry_time
        self.stop_loss = Decimal(str(stop_loss)) if stop_loss else None
        self.take_profit = Decimal(str(take_profit)) if take_profit else None

    def is_stop_triggered(self, current_price):
        """Check if stop-loss is triggered"""
        if self.stop_loss is None:
            return False

        current = Decimal(str(current_price))
        if self.side == 'long':
            return current <= self.stop_loss
        else:  # short
            return current >= self.stop_loss

    def is_target_reached(self, current_price):
        """Check if take-profit target is reached"""
        if self.take_profit is None:
            return False

        current = Decimal(str(current_price))
        if self.side == 'long':
            return current >= self.take_profit
        else:  # short
            return current <= self.take_profit


class ExitSignal:
    """Represents an exit signal for a position"""
    def __init__(self, symbol, exit_type, exit_price, reason):
        self.symbol = symbol
        self.exit_type = exit_type  # 'stop_loss', 'take_profit', 'manual'
        self.exit_price = Decimal(str(exit_price))
        self.reason = reason
        self.timestamp = datetime.now()


class ExitMonitor:
    """Monitors positions for exit conditions"""
    def __init__(self, position_manager=None):
        self.position_manager = position_manager
        self.exit_signals = []

    def check_exits(self, positions: List[Position], current_prices: Dict[str, float]) -> List[ExitSignal]:
        """Check all positions for exit conditions"""
        signals = []

        for position in positions:
            if position.symbol not in current_prices:
                continue

            current_price = current_prices[position.symbol]

            # Check stop-loss
            if position.is_stop_triggered(current_price):
                signal = ExitSignal(
                    symbol=position.symbol,
                    exit_type='stop_loss',
                    exit_price=position.stop_loss,
                    reason=f'Stop-loss triggered at {position.stop_loss}'
                )
                signals.append(signal)
                continue

            # Check take-profit
            if position.is_target_reached(current_price):
                signal = ExitSignal(
                    symbol=position.symbol,
                    exit_type='take_profit',
                    exit_price=position.take_profit,
                    reason=f'Take-profit reached at {position.take_profit}'
                )
                signals.append(signal)

        return signals

    def execute_exit(self, position: Position, exit_price: float, exit_type: str, reason: str) -> ExitSignal:
        """Execute an exit for a position"""
        signal = ExitSignal(
            symbol=position.symbol,
            exit_type=exit_type,
            exit_price=exit_price,
            reason=reason
        )
        self.exit_signals.append(signal)
        return signal

    def close_all(self, positions: List[Position], current_prices: Dict[str, float], reason: str = 'Manual close') -> List[ExitSignal]:
        """Close all positions at current prices"""
        signals = []

        for position in positions:
            if position.symbol in current_prices:
                exit_price = current_prices[position.symbol]
                signal = self.execute_exit(position, exit_price, 'manual', reason)
                signals.append(signal)

        return signals

    def get_exit_signals(self) -> List[ExitSignal]:
        """Get all exit signals generated"""
        return self.exit_signals

    def clear_signals(self):
        """Clear all exit signals"""
        self.exit_signals = []


# Test fixtures
@pytest.fixture
def exit_monitor():
    """Create a fresh ExitMonitor instance"""
    return ExitMonitor()


@pytest.fixture
def sample_time():
    """Sample timestamp for testing"""
    return datetime(2024, 1, 15, 10, 30, 0)


@pytest.fixture
def long_position(sample_time):
    """Create a sample long position"""
    return Position(
        symbol='AAPL',
        side='long',
        entry_price=150.0,
        quantity=100,
        entry_time=sample_time,
        stop_loss=145.0,
        take_profit=160.0
    )


@pytest.fixture
def short_position(sample_time):
    """Create a sample short position"""
    return Position(
        symbol='TSLA',
        side='short',
        entry_price=200.0,
        quantity=50,
        entry_time=sample_time,
        stop_loss=210.0,
        take_profit=180.0
    )


# Tests for stop-loss detection (long positions)
def test_long_stop_loss_triggered(exit_monitor, long_position):
    """Test stop-loss detection for long position when triggered"""
    current_prices = {'AAPL': 144.0}  # Below stop-loss of 145
    signals = exit_monitor.check_exits([long_position], current_prices)

    assert len(signals) == 1
    assert signals[0].symbol == 'AAPL'
    assert signals[0].exit_type == 'stop_loss'
    assert signals[0].exit_price == Decimal('145.0')


def test_long_stop_loss_at_exact_level(exit_monitor, long_position):
    """Test stop-loss detection for long position at exact level"""
    current_prices = {'AAPL': 145.0}  # Exactly at stop-loss
    signals = exit_monitor.check_exits([long_position], current_prices)

    assert len(signals) == 1
    assert signals[0].exit_type == 'stop_loss'


def test_long_stop_loss_not_triggered(exit_monitor, long_position):
    """Test stop-loss not triggered when price above stop"""
    current_prices = {'AAPL': 148.0}  # Above stop-loss
    signals = exit_monitor.check_exits([long_position], current_prices)

    assert len(signals) == 0


# Tests for stop-loss detection (short positions)
def test_short_stop_loss_triggered(exit_monitor, short_position):
    """Test stop-loss detection for short position when triggered"""
    current_prices = {'TSLA': 212.0}  # Above stop-loss of 210
    signals = exit_monitor.check_exits([short_position], current_prices)

    assert len(signals) == 1
    assert signals[0].symbol == 'TSLA'
    assert signals[0].exit_type == 'stop_loss'
    assert signals[0].exit_price == Decimal('210.0')


def test_short_stop_loss_at_exact_level(exit_monitor, short_position):
    """Test stop-loss detection for short position at exact level"""
    current_prices = {'TSLA': 210.0}  # Exactly at stop-loss
    signals = exit_monitor.check_exits([short_position], current_prices)

    assert len(signals) == 1
    assert signals[0].exit_type == 'stop_loss'


def test_short_stop_loss_not_triggered(exit_monitor, short_position):
    """Test stop-loss not triggered when price below stop"""
    current_prices = {'TSLA': 205.0}  # Below stop-loss
    signals = exit_monitor.check_exits([short_position], current_prices)

    assert len(signals) == 0


# Tests for take-profit detection (long positions)
def test_long_take_profit_reached(exit_monitor, long_position):
    """Test take-profit detection for long position when reached"""
    current_prices = {'AAPL': 162.0}  # Above take-profit of 160
    signals = exit_monitor.check_exits([long_position], current_prices)

    assert len(signals) == 1
    assert signals[0].symbol == 'AAPL'
    assert signals[0].exit_type == 'take_profit'
    assert signals[0].exit_price == Decimal('160.0')


def test_long_take_profit_at_exact_level(exit_monitor, long_position):
    """Test take-profit detection for long position at exact level"""
    current_prices = {'AAPL': 160.0}  # Exactly at take-profit
    signals = exit_monitor.check_exits([long_position], current_prices)

    assert len(signals) == 1
    assert signals[0].exit_type == 'take_profit'


def test_long_take_profit_not_reached(exit_monitor, long_position):
    """Test take-profit not reached when price below target"""
    current_prices = {'AAPL': 155.0}  # Below take-profit
    signals = exit_monitor.check_exits([long_position], current_prices)

    assert len(signals) == 0


# Tests for take-profit detection (short positions)
def test_short_take_profit_reached(exit_monitor, short_position):
    """Test take-profit detection for short position when reached"""
    current_prices = {'TSLA': 175.0}  # Below take-profit of 180
    signals = exit_monitor.check_exits([short_position], current_prices)

    assert len(signals) == 1
    assert signals[0].symbol == 'TSLA'
    assert signals[0].exit_type == 'take_profit'
    assert signals[0].exit_price == Decimal('180.0')


def test_short_take_profit_at_exact_level(exit_monitor, short_position):
    """Test take-profit detection for short position at exact level"""
    current_prices = {'TSLA': 180.0}  # Exactly at take-profit
    signals = exit_monitor.check_exits([short_position], current_prices)

    assert len(signals) == 1
    assert signals[0].exit_type == 'take_profit'


def test_short_take_profit_not_reached(exit_monitor, short_position):
    """Test take-profit not reached when price above target"""
    current_prices = {'TSLA': 190.0}  # Above take-profit
    signals = exit_monitor.check_exits([short_position], current_prices)

    assert len(signals) == 0


# Tests for multiple positions
def test_multiple_positions_mixed_signals(exit_monitor, long_position, short_position, sample_time):
    """Test checking multiple positions with different outcomes"""
    # Create another position with no signals
    neutral_position = Position('GOOGL', 'long', 140.0, 100, sample_time, 135.0, 150.0)

    positions = [long_position, short_position, neutral_position]
    current_prices = {
        'AAPL': 162.0,   # Long take-profit triggered
        'TSLA': 212.0,   # Short stop-loss triggered
        'GOOGL': 142.0   # No signal
    }

    signals = exit_monitor.check_exits(positions, current_prices)

    assert len(signals) == 2
    signal_types = {s.symbol: s.exit_type for s in signals}
    assert signal_types['AAPL'] == 'take_profit'
    assert signal_types['TSLA'] == 'stop_loss'


def test_position_without_price_data(exit_monitor, long_position):
    """Test position when current price is not available"""
    current_prices = {}  # No price for AAPL
    signals = exit_monitor.check_exits([long_position], current_prices)

    assert len(signals) == 0


# Tests for exit execution
def test_execute_manual_exit(exit_monitor, long_position):
    """Test executing a manual exit"""
    signal = exit_monitor.execute_exit(
        position=long_position,
        exit_price=155.0,
        exit_type='manual',
        reason='User requested exit'
    )

    assert signal.symbol == 'AAPL'
    assert signal.exit_type == 'manual'
    assert signal.exit_price == Decimal('155.0')
    assert signal.reason == 'User requested exit'
    assert len(exit_monitor.get_exit_signals()) == 1


def test_execute_stop_loss_exit(exit_monitor, long_position):
    """Test executing a stop-loss exit"""
    signal = exit_monitor.execute_exit(
        position=long_position,
        exit_price=145.0,
        exit_type='stop_loss',
        reason='Stop-loss triggered'
    )

    assert signal.exit_type == 'stop_loss'
    assert signal.exit_price == Decimal('145.0')


def test_execute_take_profit_exit(exit_monitor, long_position):
    """Test executing a take-profit exit"""
    signal = exit_monitor.execute_exit(
        position=long_position,
        exit_price=160.0,
        exit_type='take_profit',
        reason='Take-profit reached'
    )

    assert signal.exit_type == 'take_profit'
    assert signal.exit_price == Decimal('160.0')


# Tests for closing all positions
def test_close_all_positions(exit_monitor, long_position, short_position):
    """Test closing all positions at once"""
    positions = [long_position, short_position]
    current_prices = {'AAPL': 155.0, 'TSLA': 195.0}

    signals = exit_monitor.close_all(positions, current_prices, 'End of day close')

    assert len(signals) == 2
    assert all(s.exit_type == 'manual' for s in signals)
    assert all(s.reason == 'End of day close' for s in signals)

    symbols = {s.symbol for s in signals}
    assert symbols == {'AAPL', 'TSLA'}


def test_close_all_with_missing_prices(exit_monitor, long_position, short_position):
    """Test closing positions when some prices are missing"""
    positions = [long_position, short_position]
    current_prices = {'AAPL': 155.0}  # No price for TSLA

    signals = exit_monitor.close_all(positions, current_prices)

    assert len(signals) == 1
    assert signals[0].symbol == 'AAPL'


# Tests for signal management
def test_get_exit_signals(exit_monitor, long_position):
    """Test retrieving all exit signals"""
    exit_monitor.execute_exit(long_position, 155.0, 'manual', 'Test exit 1')
    exit_monitor.execute_exit(long_position, 160.0, 'take_profit', 'Test exit 2')

    signals = exit_monitor.get_exit_signals()
    assert len(signals) == 2


def test_clear_signals(exit_monitor, long_position):
    """Test clearing exit signals"""
    exit_monitor.execute_exit(long_position, 155.0, 'manual', 'Test exit')
    assert len(exit_monitor.get_exit_signals()) == 1

    exit_monitor.clear_signals()
    assert len(exit_monitor.get_exit_signals()) == 0


# Tests for edge cases
def test_position_without_stop_or_target(exit_monitor, sample_time):
    """Test position with no stop-loss or take-profit"""
    position = Position('AAPL', 'long', 150.0, 100, sample_time)  # No stops
    current_prices = {'AAPL': 100.0}  # Price dropped significantly

    signals = exit_monitor.check_exits([position], current_prices)
    assert len(signals) == 0  # No signals since no stops defined


def test_stop_loss_priority_over_take_profit(exit_monitor, sample_time):
    """Test that stop-loss is checked before take-profit"""
    # This shouldn't happen in practice, but tests priority
    position = Position('AAPL', 'long', 150.0, 100, sample_time, 155.0, 160.0)
    # Stop at 155, target at 160
    current_prices = {'AAPL': 165.0}  # Above both

    signals = exit_monitor.check_exits([position], current_prices)

    # Should trigger stop-loss first since it's checked first
    assert len(signals) == 1
    assert signals[0].exit_type == 'stop_loss'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
