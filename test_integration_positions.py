#!/usr/bin/env python3
"""
Integration tests for position management system

Tests full cycle workflows, duplicate position handling, portfolio limits,
and multi-cycle operations.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional


# Mock classes for integration testing
class Position:
    """Represents a trading position"""
    def __init__(self, symbol, side, entry_price, quantity, entry_time, stop_loss=None, take_profit=None):
        self.symbol = symbol
        self.side = side
        self.entry_price = Decimal(str(entry_price))
        self.quantity = Decimal(str(quantity))
        self.entry_time = entry_time
        self.stop_loss = Decimal(str(stop_loss)) if stop_loss else None
        self.take_profit = Decimal(str(take_profit)) if take_profit else None
        self.exit_price = None
        self.exit_time = None
        self.realized_pl = None

    def unrealized_pl(self, current_price):
        current_price = Decimal(str(current_price))
        if self.side == 'long':
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def close(self, exit_price, exit_time):
        self.exit_price = Decimal(str(exit_price))
        self.exit_time = exit_time
        self.realized_pl = self.unrealized_pl(exit_price)
        return self.realized_pl

    def is_open(self):
        return self.exit_price is None

    def is_stop_triggered(self, current_price):
        if self.stop_loss is None:
            return False
        current = Decimal(str(current_price))
        if self.side == 'long':
            return current <= self.stop_loss
        else:
            return current >= self.stop_loss

    def is_target_reached(self, current_price):
        if self.take_profit is None:
            return False
        current = Decimal(str(current_price))
        if self.side == 'long':
            return current >= self.take_profit
        else:
            return current <= self.take_profit


class PositionManager:
    """Manages trading positions"""
    def __init__(self, initial_capital=100000, max_exposure_pct=80.0, allow_duplicates=False):
        self.initial_capital = Decimal(str(initial_capital))
        self.max_exposure_pct = Decimal(str(max_exposure_pct))
        self.allow_duplicates = allow_duplicates
        self.positions = {}
        self.closed_positions = []

    def open_position(self, symbol, side, entry_price, quantity, entry_time, stop_loss=None, take_profit=None):
        if not self.allow_duplicates and symbol in self.positions:
            raise ValueError(f"Position already exists for {symbol}")

        position = Position(symbol, side, entry_price, quantity, entry_time, stop_loss, take_profit)
        self.positions[symbol] = position
        return position

    def close_position(self, symbol, exit_price, exit_time):
        if symbol not in self.positions:
            raise ValueError(f"No open position for {symbol}")
        position = self.positions[symbol]
        realized_pl = position.close(exit_price, exit_time)
        self.closed_positions.append(position)
        del self.positions[symbol]
        return realized_pl

    def get_position(self, symbol):
        return self.positions.get(symbol)

    def get_all_positions(self):
        return list(self.positions.values())

    def close_all_positions(self, exit_price_map, exit_time):
        results = {}
        symbols = list(self.positions.keys())
        for symbol in symbols:
            if symbol in exit_price_map:
                pl = self.close_position(symbol, exit_price_map[symbol], exit_time)
                results[symbol] = pl
        return results

    def calculate_portfolio_state(self, current_prices):
        total_unrealized_pl = Decimal('0')
        total_realized_pl = Decimal('0')
        total_exposure = Decimal('0')

        for symbol, position in self.positions.items():
            current_price = Decimal(str(current_prices.get(symbol, position.entry_price)))
            unrealized = position.unrealized_pl(current_price)
            total_unrealized_pl += unrealized
            total_exposure += position.quantity * current_price

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

    def check_exposure_limit(self, current_prices):
        """Check if current exposure exceeds limit"""
        state = self.calculate_portfolio_state(current_prices)
        return state['exposure_pct'] <= float(self.max_exposure_pct)


class ExitMonitor:
    """Monitors positions for exit conditions"""
    def __init__(self, position_manager):
        self.position_manager = position_manager

    def check_exits(self, positions, current_prices):
        """Check positions and return symbols to exit"""
        exits = []
        for position in positions:
            if position.symbol not in current_prices:
                continue
            current_price = current_prices[position.symbol]
            if position.is_stop_triggered(current_price):
                exits.append({
                    'symbol': position.symbol,
                    'type': 'stop_loss',
                    'price': float(position.stop_loss)
                })
            elif position.is_target_reached(current_price):
                exits.append({
                    'symbol': position.symbol,
                    'type': 'take_profit',
                    'price': float(position.take_profit)
                })
        return exits

    def execute_exits(self, exit_signals, exit_time):
        """Execute exit signals"""
        results = []
        for signal in exit_signals:
            try:
                pl = self.position_manager.close_position(
                    signal['symbol'],
                    signal['price'],
                    exit_time
                )
                results.append({
                    'symbol': signal['symbol'],
                    'type': signal['type'],
                    'pl': float(pl)
                })
            except ValueError:
                pass  # Position already closed
        return results


class TradingCycle:
    """Represents a complete trading cycle"""
    def __init__(self, position_manager, exit_monitor):
        self.position_manager = position_manager
        self.exit_monitor = exit_monitor
        self.cycle_count = 0

    def run_cycle(self, current_time, current_prices, new_trades=None):
        """Run a complete trading cycle"""
        self.cycle_count += 1
        results = {
            'cycle': self.cycle_count,
            'time': current_time,
            'exits': [],
            'entries': [],
            'errors': []
        }

        # Step 1: Check and execute exits
        positions = self.position_manager.get_all_positions()
        exit_signals = self.exit_monitor.check_exits(positions, current_prices)
        if exit_signals:
            exit_results = self.exit_monitor.execute_exits(exit_signals, current_time)
            results['exits'] = exit_results

        # Step 2: Enter new trades
        if new_trades:
            for trade in new_trades:
                try:
                    position = self.position_manager.open_position(
                        symbol=trade['symbol'],
                        side=trade['side'],
                        entry_price=trade['price'],
                        quantity=trade['quantity'],
                        entry_time=current_time,
                        stop_loss=trade.get('stop_loss'),
                        take_profit=trade.get('take_profit')
                    )
                    results['entries'].append({
                        'symbol': trade['symbol'],
                        'side': trade['side'],
                        'price': trade['price']
                    })
                except ValueError as e:
                    results['errors'].append({
                        'symbol': trade['symbol'],
                        'error': str(e)
                    })

        # Step 3: Portfolio state
        results['portfolio'] = self.position_manager.calculate_portfolio_state(current_prices)

        return results


# Test fixtures
@pytest.fixture
def position_manager():
    """Create position manager with standard settings"""
    return PositionManager(
        initial_capital=100000,
        max_exposure_pct=80.0,
        allow_duplicates=False
    )


@pytest.fixture
def exit_monitor(position_manager):
    """Create exit monitor"""
    return ExitMonitor(position_manager)


@pytest.fixture
def trading_cycle(position_manager, exit_monitor):
    """Create trading cycle"""
    return TradingCycle(position_manager, exit_monitor)


# Integration tests
def test_full_cycle_with_position_opening(trading_cycle):
    """Test complete cycle with opening a new position"""
    current_time = datetime(2024, 1, 15, 9, 30, 0)
    current_prices = {'AAPL': 150.0}
    new_trades = [{
        'symbol': 'AAPL',
        'side': 'long',
        'price': 150.0,
        'quantity': 100,
        'stop_loss': 145.0,
        'take_profit': 160.0
    }]

    results = trading_cycle.run_cycle(current_time, current_prices, new_trades)

    assert results['cycle'] == 1
    assert len(results['entries']) == 1
    assert results['entries'][0]['symbol'] == 'AAPL'
    assert results['portfolio']['open_positions'] == 1
    assert results['portfolio']['exposure_pct'] == 15.0  # 15k / 100k


def test_duplicate_position_rejection(trading_cycle):
    """Test that duplicate positions are rejected"""
    current_time = datetime(2024, 1, 15, 9, 30, 0)
    current_prices = {'AAPL': 150.0}

    # First trade - should succeed
    trade = {
        'symbol': 'AAPL',
        'side': 'long',
        'price': 150.0,
        'quantity': 100,
        'stop_loss': 145.0,
        'take_profit': 160.0
    }
    results1 = trading_cycle.run_cycle(current_time, current_prices, [trade])
    assert len(results1['entries']) == 1
    assert len(results1['errors']) == 0

    # Second trade - should fail
    current_time = datetime(2024, 1, 15, 10, 30, 0)
    results2 = trading_cycle.run_cycle(current_time, current_prices, [trade])
    assert len(results2['entries']) == 0
    assert len(results2['errors']) == 1
    assert 'already exists' in results2['errors'][0]['error']


def test_portfolio_concentration_limits(position_manager):
    """Test portfolio exposure limits"""
    current_time = datetime(2024, 1, 15, 9, 30, 0)

    # Open position for 50% of capital
    position_manager.open_position('AAPL', 'long', 150.0, 333, current_time)
    current_prices = {'AAPL': 150.0}

    # Check exposure (should be ~50%)
    state = position_manager.calculate_portfolio_state(current_prices)
    assert 49.0 <= state['exposure_pct'] <= 51.0
    assert position_manager.check_exposure_limit(current_prices)

    # Try to add another large position (total would exceed 80%)
    position_manager.open_position('TSLA', 'long', 200.0, 200, current_time)
    current_prices['TSLA'] = 200.0

    state = position_manager.calculate_portfolio_state(current_prices)
    # Total exposure: (333*150 + 200*200) / 100000 = 89.95%
    assert state['exposure_pct'] > 80.0
    assert not position_manager.check_exposure_limit(current_prices)


def test_multi_cycle_workflow_with_exits(trading_cycle):
    """Test multiple cycles with entries and exits"""
    # Cycle 1: Open position
    cycle1_time = datetime(2024, 1, 15, 9, 30, 0)
    cycle1_prices = {'AAPL': 150.0}
    cycle1_trades = [{
        'symbol': 'AAPL',
        'side': 'long',
        'price': 150.0,
        'quantity': 100,
        'stop_loss': 145.0,
        'take_profit': 160.0
    }]

    results1 = trading_cycle.run_cycle(cycle1_time, cycle1_prices, cycle1_trades)
    assert len(results1['entries']) == 1
    assert results1['portfolio']['open_positions'] == 1

    # Cycle 2: Price moves up, check position (no exit yet)
    cycle2_time = datetime(2024, 1, 15, 12, 0, 0)
    cycle2_prices = {'AAPL': 155.0}

    results2 = trading_cycle.run_cycle(cycle2_time, cycle2_prices)
    assert len(results2['exits']) == 0
    assert results2['portfolio']['open_positions'] == 1
    assert results2['portfolio']['total_unrealized_pl'] == 500.0  # (155-150)*100

    # Cycle 3: Take profit hit
    cycle3_time = datetime(2024, 1, 15, 15, 0, 0)
    cycle3_prices = {'AAPL': 162.0}

    results3 = trading_cycle.run_cycle(cycle3_time, cycle3_prices)
    assert len(results3['exits']) == 1
    assert results3['exits'][0]['type'] == 'take_profit'
    assert results3['exits'][0]['pl'] == 1000.0  # (160-150)*100
    assert results3['portfolio']['open_positions'] == 0
    assert results3['portfolio']['closed_positions'] == 1
    assert results3['portfolio']['total_realized_pl'] == 1000.0


def test_stop_loss_triggering_in_cycle(trading_cycle):
    """Test stop-loss triggering during a cycle"""
    # Open position
    cycle1_time = datetime(2024, 1, 15, 9, 30, 0)
    cycle1_prices = {'AAPL': 150.0}
    cycle1_trades = [{
        'symbol': 'AAPL',
        'side': 'long',
        'price': 150.0,
        'quantity': 100,
        'stop_loss': 145.0,
        'take_profit': 160.0
    }]

    results1 = trading_cycle.run_cycle(cycle1_time, cycle1_prices, cycle1_trades)
    assert results1['portfolio']['open_positions'] == 1

    # Price drops below stop
    cycle2_time = datetime(2024, 1, 15, 10, 30, 0)
    cycle2_prices = {'AAPL': 143.0}

    results2 = trading_cycle.run_cycle(cycle2_time, cycle2_prices)
    assert len(results2['exits']) == 1
    assert results2['exits'][0]['type'] == 'stop_loss'
    assert results2['exits'][0]['pl'] == -500.0  # (145-150)*100
    assert results2['portfolio']['open_positions'] == 0


def test_multiple_positions_in_cycle(trading_cycle):
    """Test managing multiple positions across cycles"""
    cycle1_time = datetime(2024, 1, 15, 9, 30, 0)
    cycle1_prices = {'AAPL': 150.0, 'TSLA': 200.0, 'GOOGL': 140.0}
    cycle1_trades = [
        {'symbol': 'AAPL', 'side': 'long', 'price': 150.0, 'quantity': 100, 'stop_loss': 145.0, 'take_profit': 160.0},
        {'symbol': 'TSLA', 'side': 'short', 'price': 200.0, 'quantity': 50, 'stop_loss': 210.0, 'take_profit': 180.0},
        {'symbol': 'GOOGL', 'side': 'long', 'price': 140.0, 'quantity': 100, 'stop_loss': 135.0, 'take_profit': 150.0}
    ]

    results1 = trading_cycle.run_cycle(cycle1_time, cycle1_prices, cycle1_trades)
    assert len(results1['entries']) == 3
    assert results1['portfolio']['open_positions'] == 3

    # Cycle 2: AAPL hits target, TSLA hits stop, GOOGL continues
    cycle2_time = datetime(2024, 1, 15, 15, 0, 0)
    cycle2_prices = {'AAPL': 162.0, 'TSLA': 212.0, 'GOOGL': 142.0}

    results2 = trading_cycle.run_cycle(cycle2_time, cycle2_prices)
    assert len(results2['exits']) == 2
    exit_types = {e['symbol']: e['type'] for e in results2['exits']}
    assert exit_types['AAPL'] == 'take_profit'
    assert exit_types['TSLA'] == 'stop_loss'
    assert results2['portfolio']['open_positions'] == 1


def test_close_all_at_end_of_day(position_manager, exit_monitor):
    """Test closing all positions at end of day"""
    current_time = datetime(2024, 1, 15, 9, 30, 0)

    # Open multiple positions
    position_manager.open_position('AAPL', 'long', 150.0, 100, current_time, 145.0, 160.0)
    position_manager.open_position('TSLA', 'short', 200.0, 50, current_time, 210.0, 180.0)
    position_manager.open_position('GOOGL', 'long', 140.0, 100, current_time, 135.0, 150.0)

    assert len(position_manager.get_all_positions()) == 3

    # Close all at EOD
    eod_time = datetime(2024, 1, 15, 15, 55, 0)
    eod_prices = {'AAPL': 155.0, 'TSLA': 195.0, 'GOOGL': 145.0}

    results = position_manager.close_all_positions(eod_prices, eod_time)

    assert len(results) == 3
    assert position_manager.get_all_positions() == []
    assert len(position_manager.closed_positions) == 3

    # Verify P&L
    state = position_manager.calculate_portfolio_state({})
    # AAPL: (155-150)*100 = 500
    # TSLA: (200-195)*50 = 250
    # GOOGL: (145-140)*100 = 500
    assert state['total_realized_pl'] == 1250.0


def test_cycle_with_no_trades(trading_cycle):
    """Test cycle with no new trades"""
    current_time = datetime(2024, 1, 15, 9, 30, 0)
    current_prices = {'AAPL': 150.0}

    results = trading_cycle.run_cycle(current_time, current_prices)

    assert results['cycle'] == 1
    assert len(results['entries']) == 0
    assert len(results['exits']) == 0
    assert results['portfolio']['open_positions'] == 0


def test_partial_position_close(trading_cycle):
    """Test closing only some positions when conditions met"""
    cycle1_time = datetime(2024, 1, 15, 9, 30, 0)
    cycle1_prices = {'AAPL': 150.0, 'TSLA': 200.0}
    cycle1_trades = [
        {'symbol': 'AAPL', 'side': 'long', 'price': 150.0, 'quantity': 100, 'stop_loss': 145.0, 'take_profit': 160.0},
        {'symbol': 'TSLA', 'side': 'long', 'price': 200.0, 'quantity': 50, 'stop_loss': 190.0, 'take_profit': 220.0}
    ]

    results1 = trading_cycle.run_cycle(cycle1_time, cycle1_prices, cycle1_trades)
    assert results1['portfolio']['open_positions'] == 2

    # Only AAPL hits target
    cycle2_time = datetime(2024, 1, 15, 15, 0, 0)
    cycle2_prices = {'AAPL': 162.0, 'TSLA': 205.0}

    results2 = trading_cycle.run_cycle(cycle2_time, cycle2_prices)
    assert len(results2['exits']) == 1
    assert results2['exits'][0]['symbol'] == 'AAPL'
    assert results2['portfolio']['open_positions'] == 1
    assert results2['portfolio']['closed_positions'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
