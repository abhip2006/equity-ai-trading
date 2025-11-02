#!/usr/bin/env python3
"""
Test suite for PositionManager

Tests position opening, closing, P&L calculations, and portfolio state.
"""

import pytest
from datetime import datetime
from decimal import Decimal


# Mock PositionManager for testing (will be replaced when actual implementation exists)
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
        self.exit_price = None
        self.exit_time = None
        self.realized_pl = None

    def unrealized_pl(self, current_price):
        """Calculate unrealized P&L"""
        current_price = Decimal(str(current_price))
        if self.side == 'long':
            return (current_price - self.entry_price) * self.quantity
        else:  # short
            return (self.entry_price - current_price) * self.quantity

    def close(self, exit_price, exit_time):
        """Close the position"""
        self.exit_price = Decimal(str(exit_price))
        self.exit_time = exit_time
        self.realized_pl = self.unrealized_pl(exit_price)
        return self.realized_pl

    def is_open(self):
        """Check if position is still open"""
        return self.exit_price is None


class PositionManager:
    """Manages trading positions"""
    def __init__(self, initial_capital=100000):
        self.initial_capital = Decimal(str(initial_capital))
        self.positions = {}  # symbol -> Position
        self.closed_positions = []

    def open_position(self, symbol, side, entry_price, quantity, entry_time, stop_loss=None, take_profit=None):
        """Open a new position"""
        if symbol in self.positions:
            raise ValueError(f"Position already exists for {symbol}")

        position = Position(symbol, side, entry_price, quantity, entry_time, stop_loss, take_profit)
        self.positions[symbol] = position
        return position

    def close_position(self, symbol, exit_price, exit_time):
        """Close an existing position"""
        if symbol not in self.positions:
            raise ValueError(f"No open position for {symbol}")

        position = self.positions[symbol]
        realized_pl = position.close(exit_price, exit_time)
        self.closed_positions.append(position)
        del self.positions[symbol]
        return realized_pl

    def get_position(self, symbol):
        """Get a position by symbol"""
        return self.positions.get(symbol)

    def get_all_positions(self):
        """Get all open positions"""
        return list(self.positions.values())

    def close_all_positions(self, exit_price_map, exit_time):
        """Close all open positions"""
        results = {}
        symbols = list(self.positions.keys())
        for symbol in symbols:
            if symbol in exit_price_map:
                pl = self.close_position(symbol, exit_price_map[symbol], exit_time)
                results[symbol] = pl
        return results

    def calculate_portfolio_state(self, current_prices):
        """Calculate current portfolio state"""
        total_unrealized_pl = Decimal('0')
        total_realized_pl = Decimal('0')
        total_exposure = Decimal('0')

        # Calculate unrealized P&L and exposure for open positions
        for symbol, position in self.positions.items():
            current_price = Decimal(str(current_prices.get(symbol, position.entry_price)))
            unrealized = position.unrealized_pl(current_price)
            total_unrealized_pl += unrealized
            total_exposure += position.quantity * current_price

        # Calculate realized P&L from closed positions
        for position in self.closed_positions:
            if position.realized_pl:
                total_realized_pl += position.realized_pl

        current_capital = self.initial_capital + total_realized_pl + total_unrealized_pl
        exposure_pct = (total_exposure / self.initial_capital * 100) if self.initial_capital > 0 else Decimal('0')

        return {
            'current_capital': float(current_capital),
            'total_realized_pl': float(total_realized_pl),
            'total_unrealized_pl': float(total_unrealized_pl),
            'total_exposure': float(total_exposure),
            'exposure_pct': float(exposure_pct),
            'open_positions': len(self.positions),
            'closed_positions': len(self.closed_positions)
        }


# Test fixtures
@pytest.fixture
def position_manager():
    """Create a fresh PositionManager instance"""
    return PositionManager(initial_capital=100000)


@pytest.fixture
def sample_time():
    """Sample timestamp for testing"""
    return datetime(2024, 1, 15, 10, 30, 0)


# Tests for opening positions
def test_open_long_position(position_manager, sample_time):
    """Test opening a long position"""
    position = position_manager.open_position(
        symbol='AAPL',
        side='long',
        entry_price=150.0,
        quantity=100,
        entry_time=sample_time,
        stop_loss=145.0,
        take_profit=160.0
    )

    assert position.symbol == 'AAPL'
    assert position.side == 'long'
    assert position.entry_price == Decimal('150.0')
    assert position.quantity == Decimal('100')
    assert position.stop_loss == Decimal('145.0')
    assert position.take_profit == Decimal('160.0')
    assert position.is_open()


def test_open_short_position(position_manager, sample_time):
    """Test opening a short position"""
    position = position_manager.open_position(
        symbol='TSLA',
        side='short',
        entry_price=200.0,
        quantity=50,
        entry_time=sample_time,
        stop_loss=210.0,
        take_profit=180.0
    )

    assert position.symbol == 'TSLA'
    assert position.side == 'short'
    assert position.entry_price == Decimal('200.0')
    assert position.quantity == Decimal('50')
    assert position.is_open()


def test_open_duplicate_position_raises_error(position_manager, sample_time):
    """Test that opening duplicate position raises error"""
    position_manager.open_position('AAPL', 'long', 150.0, 100, sample_time)

    with pytest.raises(ValueError, match="Position already exists"):
        position_manager.open_position('AAPL', 'long', 155.0, 100, sample_time)


# Tests for closing positions
def test_close_long_position_profit(position_manager, sample_time):
    """Test closing a long position with profit"""
    position_manager.open_position('AAPL', 'long', 150.0, 100, sample_time)

    exit_time = datetime(2024, 1, 15, 15, 0, 0)
    realized_pl = position_manager.close_position('AAPL', 160.0, exit_time)

    assert realized_pl == Decimal('1000.0')  # (160 - 150) * 100
    assert 'AAPL' not in position_manager.positions
    assert len(position_manager.closed_positions) == 1


def test_close_long_position_loss(position_manager, sample_time):
    """Test closing a long position with loss"""
    position_manager.open_position('AAPL', 'long', 150.0, 100, sample_time)

    exit_time = datetime(2024, 1, 15, 15, 0, 0)
    realized_pl = position_manager.close_position('AAPL', 145.0, exit_time)

    assert realized_pl == Decimal('-500.0')  # (145 - 150) * 100


def test_close_short_position_profit(position_manager, sample_time):
    """Test closing a short position with profit"""
    position_manager.open_position('TSLA', 'short', 200.0, 50, sample_time)

    exit_time = datetime(2024, 1, 15, 15, 0, 0)
    realized_pl = position_manager.close_position('TSLA', 180.0, exit_time)

    assert realized_pl == Decimal('1000.0')  # (200 - 180) * 50


def test_close_short_position_loss(position_manager, sample_time):
    """Test closing a short position with loss"""
    position_manager.open_position('TSLA', 'short', 200.0, 50, sample_time)

    exit_time = datetime(2024, 1, 15, 15, 0, 0)
    realized_pl = position_manager.close_position('TSLA', 210.0, exit_time)

    assert realized_pl == Decimal('-500.0')  # (200 - 210) * 50


def test_close_nonexistent_position_raises_error(position_manager, sample_time):
    """Test that closing nonexistent position raises error"""
    with pytest.raises(ValueError, match="No open position"):
        position_manager.close_position('AAPL', 150.0, sample_time)


# Tests for P&L calculations
def test_unrealized_pl_long_profit(position_manager, sample_time):
    """Test unrealized P&L calculation for long position in profit"""
    position = position_manager.open_position('AAPL', 'long', 150.0, 100, sample_time)
    unrealized = position.unrealized_pl(155.0)

    assert unrealized == Decimal('500.0')  # (155 - 150) * 100


def test_unrealized_pl_long_loss(position_manager, sample_time):
    """Test unrealized P&L calculation for long position in loss"""
    position = position_manager.open_position('AAPL', 'long', 150.0, 100, sample_time)
    unrealized = position.unrealized_pl(145.0)

    assert unrealized == Decimal('-500.0')  # (145 - 150) * 100


def test_unrealized_pl_short_profit(position_manager, sample_time):
    """Test unrealized P&L calculation for short position in profit"""
    position = position_manager.open_position('TSLA', 'short', 200.0, 50, sample_time)
    unrealized = position.unrealized_pl(190.0)

    assert unrealized == Decimal('500.0')  # (200 - 190) * 50


def test_unrealized_pl_short_loss(position_manager, sample_time):
    """Test unrealized P&L calculation for short position in loss"""
    position = position_manager.open_position('TSLA', 'short', 200.0, 50, sample_time)
    unrealized = position.unrealized_pl(210.0)

    assert unrealized == Decimal('-500.0')  # (200 - 210) * 50


# Tests for querying positions
def test_get_position(position_manager, sample_time):
    """Test retrieving a specific position"""
    position_manager.open_position('AAPL', 'long', 150.0, 100, sample_time)

    position = position_manager.get_position('AAPL')
    assert position is not None
    assert position.symbol == 'AAPL'


def test_get_nonexistent_position_returns_none(position_manager):
    """Test that getting nonexistent position returns None"""
    position = position_manager.get_position('AAPL')
    assert position is None


def test_get_all_positions(position_manager, sample_time):
    """Test retrieving all open positions"""
    position_manager.open_position('AAPL', 'long', 150.0, 100, sample_time)
    position_manager.open_position('TSLA', 'short', 200.0, 50, sample_time)

    positions = position_manager.get_all_positions()
    assert len(positions) == 2
    symbols = {p.symbol for p in positions}
    assert symbols == {'AAPL', 'TSLA'}


def test_close_all_positions(position_manager, sample_time):
    """Test closing all open positions at once"""
    position_manager.open_position('AAPL', 'long', 150.0, 100, sample_time)
    position_manager.open_position('TSLA', 'short', 200.0, 50, sample_time)

    exit_time = datetime(2024, 1, 15, 15, 0, 0)
    results = position_manager.close_all_positions(
        {'AAPL': 160.0, 'TSLA': 180.0},
        exit_time
    )

    assert len(results) == 2
    assert results['AAPL'] == Decimal('1000.0')  # Long profit
    assert results['TSLA'] == Decimal('1000.0')  # Short profit
    assert len(position_manager.positions) == 0
    assert len(position_manager.closed_positions) == 2


# Tests for portfolio state calculation
def test_portfolio_state_no_positions(position_manager):
    """Test portfolio state with no positions"""
    state = position_manager.calculate_portfolio_state({})

    assert state['current_capital'] == 100000.0
    assert state['total_realized_pl'] == 0.0
    assert state['total_unrealized_pl'] == 0.0
    assert state['total_exposure'] == 0.0
    assert state['exposure_pct'] == 0.0
    assert state['open_positions'] == 0
    assert state['closed_positions'] == 0


def test_portfolio_state_open_positions(position_manager, sample_time):
    """Test portfolio state with open positions"""
    position_manager.open_position('AAPL', 'long', 150.0, 100, sample_time)
    position_manager.open_position('TSLA', 'short', 200.0, 50, sample_time)

    current_prices = {'AAPL': 155.0, 'TSLA': 190.0}
    state = position_manager.calculate_portfolio_state(current_prices)

    # AAPL: (155 - 150) * 100 = 500
    # TSLA: (200 - 190) * 50 = 500
    assert state['total_unrealized_pl'] == 1000.0
    assert state['total_realized_pl'] == 0.0
    assert state['current_capital'] == 101000.0
    assert state['open_positions'] == 2


def test_portfolio_state_closed_positions(position_manager, sample_time):
    """Test portfolio state with closed positions"""
    position_manager.open_position('AAPL', 'long', 150.0, 100, sample_time)
    exit_time = datetime(2024, 1, 15, 15, 0, 0)
    position_manager.close_position('AAPL', 160.0, exit_time)

    state = position_manager.calculate_portfolio_state({})

    assert state['total_realized_pl'] == 1000.0
    assert state['total_unrealized_pl'] == 0.0
    assert state['current_capital'] == 101000.0
    assert state['open_positions'] == 0
    assert state['closed_positions'] == 1


def test_portfolio_state_mixed_positions(position_manager, sample_time):
    """Test portfolio state with both open and closed positions"""
    # Open and close AAPL for profit
    position_manager.open_position('AAPL', 'long', 150.0, 100, sample_time)
    exit_time = datetime(2024, 1, 15, 15, 0, 0)
    position_manager.close_position('AAPL', 160.0, exit_time)

    # Open TSLA still open
    position_manager.open_position('TSLA', 'short', 200.0, 50, sample_time)

    current_prices = {'TSLA': 190.0}
    state = position_manager.calculate_portfolio_state(current_prices)

    assert state['total_realized_pl'] == 1000.0  # AAPL
    assert state['total_unrealized_pl'] == 500.0  # TSLA
    assert state['current_capital'] == 101500.0
    assert state['open_positions'] == 1
    assert state['closed_positions'] == 1


def test_portfolio_exposure_calculation(position_manager, sample_time):
    """Test portfolio exposure calculation"""
    # Open position worth $15,000 at current price
    position_manager.open_position('AAPL', 'long', 150.0, 100, sample_time)

    current_prices = {'AAPL': 150.0}
    state = position_manager.calculate_portfolio_state(current_prices)

    assert state['total_exposure'] == 15000.0
    assert state['exposure_pct'] == 15.0  # 15k / 100k = 15%


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
