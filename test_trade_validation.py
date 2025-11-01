"""
Unit tests for trade plan validation.

Tests all validation rules:
- Required fields validation
- Position size bounds
- Price logic for long/short positions
- Risk-reward ratios
- Price sanity checks
- Stop distance validation
"""

import pytest
from active_trader_llm.trader.trade_plan_validator import (
    TradePlanValidator,
    ValidationConfig,
    TradeValidationError
)


class TestTradePlanValidator:
    """Test suite for TradePlanValidator"""

    @pytest.fixture
    def validator(self):
        """Create validator with default config"""
        config = ValidationConfig(
            max_position_pct=10.0,
            min_risk_reward_ratio=1.5,
            max_price_deviation_pct=5.0,
            min_stop_distance_pct=0.5,
            max_stop_distance_pct=15.0
        )
        return TradePlanValidator(config)

    # Valid Trade Plans

    def test_valid_long_trade(self, validator):
        """Test valid long trade passes validation"""
        is_valid, error = validator.validate_trade_plan(
            action="long",
            entry=100.0,
            stop_loss=95.0,  # 5% below entry
            take_profit=110.0,  # 10% above entry (2:1 R:R)
            position_pct=5.0,
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid is True
        assert error is None

    def test_valid_short_trade(self, validator):
        """Test valid short trade passes validation"""
        is_valid, error = validator.validate_trade_plan(
            action="short",
            entry=100.0,
            stop_loss=105.0,  # 5% above entry
            take_profit=90.0,  # 10% below entry (2:1 R:R)
            position_pct=5.0,
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid is True
        assert error is None

    # Invalid Action

    def test_invalid_action(self, validator):
        """Test invalid action is rejected"""
        is_valid, error = validator.validate_trade_plan(
            action="sideways",
            entry=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            position_pct=5.0,
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid is False
        assert "Invalid action" in error

    # Position Size Validation

    def test_position_size_exceeds_100_percent(self, validator):
        """Test position size > 100% is rejected"""
        is_valid, error = validator.validate_trade_plan(
            action="long",
            entry=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            position_pct=150.0,
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid is False
        assert "exceeds 100% of capital" in error

    def test_position_size_exceeds_max(self, validator):
        """Test position size > max_position_pct is rejected"""
        is_valid, error = validator.validate_trade_plan(
            action="long",
            entry=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            position_pct=15.0,  # Exceeds 10% max
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid is False
        assert "exceeds maximum allowed" in error

    def test_position_size_zero(self, validator):
        """Test zero position size is rejected"""
        is_valid, error = validator.validate_trade_plan(
            action="long",
            entry=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            position_pct=0.0,
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid is False
        assert "Position percentage must be positive" in error

    def test_position_size_negative(self, validator):
        """Test negative position size is rejected"""
        is_valid, error = validator.validate_trade_plan(
            action="long",
            entry=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            position_pct=-5.0,
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid is False
        assert "Position percentage must be positive" in error

    # Long Position Price Logic

    def test_long_stop_above_entry(self, validator):
        """Test long with stop > entry is rejected (inverse logic)"""
        is_valid, error = validator.validate_trade_plan(
            action="long",
            entry=100.0,
            stop_loss=105.0,  # WRONG: Above entry
            take_profit=110.0,
            position_pct=5.0,
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid is False
        assert "Stop loss" in error and "must be BELOW entry" in error
        assert "inverse logic" in error.lower()

    def test_long_target_below_entry(self, validator):
        """Test long with target < entry is rejected (inverse logic)"""
        is_valid, error = validator.validate_trade_plan(
            action="long",
            entry=100.0,
            stop_loss=95.0,
            take_profit=90.0,  # WRONG: Below entry
            position_pct=5.0,
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid is False
        assert "Take profit" in error and "must be ABOVE entry" in error
        assert "inverse logic" in error.lower()

    def test_long_stop_equals_entry(self, validator):
        """Test long with stop = entry is rejected"""
        is_valid, error = validator.validate_trade_plan(
            action="long",
            entry=100.0,
            stop_loss=100.0,  # Equal to entry
            take_profit=110.0,
            position_pct=5.0,
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid is False
        assert "must be BELOW entry" in error

    # Short Position Price Logic

    def test_short_stop_below_entry(self, validator):
        """Test short with stop < entry is rejected (inverse logic)"""
        is_valid, error = validator.validate_trade_plan(
            action="short",
            entry=100.0,
            stop_loss=95.0,  # WRONG: Below entry
            take_profit=90.0,
            position_pct=5.0,
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid is False
        assert "Stop loss" in error and "must be ABOVE entry" in error
        assert "inverse logic" in error.lower()

    def test_short_target_above_entry(self, validator):
        """Test short with target > entry is rejected (inverse logic)"""
        is_valid, error = validator.validate_trade_plan(
            action="short",
            entry=100.0,
            stop_loss=105.0,
            take_profit=110.0,  # WRONG: Above entry
            position_pct=5.0,
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid is False
        assert "Take profit" in error and "must be BELOW entry" in error
        assert "inverse logic" in error.lower()

    # Risk-Reward Validation

    def test_risk_reward_too_low(self, validator):
        """Test R:R below minimum is rejected"""
        is_valid, error = validator.validate_trade_plan(
            action="long",
            entry=100.0,
            stop_loss=95.0,  # Risk: $5
            take_profit=103.0,  # Reward: $3 (R:R = 0.6 < 1.5)
            position_pct=5.0,
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid is False
        assert "Risk-reward ratio" in error
        assert "below minimum" in error

    def test_risk_reward_exactly_minimum(self, validator):
        """Test R:R exactly at minimum passes"""
        is_valid, error = validator.validate_trade_plan(
            action="long",
            entry=100.0,
            stop_loss=95.0,  # Risk: $5
            take_profit=107.5,  # Reward: $7.5 (R:R = 1.5)
            position_pct=5.0,
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid is True
        assert error is None

    # Price Sanity Checks

    def test_entry_far_from_current_price(self, validator):
        """Test entry deviating too much from current price is rejected"""
        is_valid, error = validator.validate_trade_plan(
            action="long",
            entry=110.0,  # 10% above current (exceeds 5% max)
            stop_loss=105.0,
            take_profit=120.0,
            position_pct=5.0,
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid is False
        assert "entry" in error.lower()
        assert "deviates" in error

    def test_stop_far_from_entry_caught_by_stop_distance(self, validator):
        """Test stop too far from entry is caught by stop distance check, not price sanity"""
        is_valid, error = validator.validate_trade_plan(
            action="long",
            entry=100.0,
            stop_loss=80.0,  # 20% below entry (exceeds 15% max stop distance)
            take_profit=140.0,  # Need higher target for R:R > 1.5 (risk=$20, reward=$40, R:R=2.0)
            position_pct=5.0,
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid is False
        assert "Stop distance" in error
        assert "too wide" in error

    def test_target_far_from_entry_is_allowed(self, validator):
        """Test target far from entry is allowed (only entry needs to be near current price)"""
        is_valid, error = validator.validate_trade_plan(
            action="long",
            entry=100.0,  # Entry at current price
            stop_loss=98.0,  # 2% stop
            take_profit=150.0,  # 50% target is fine (validated by R:R, not price deviation)
            position_pct=5.0,
            current_price=100.0,
            symbol="TEST"
        )
        # This should pass - target can be far as long as R:R is good and stop distance is ok
        assert is_valid is True

    # Stop Distance Validation

    def test_stop_too_tight(self, validator):
        """Test stop distance < min is rejected (too tight)"""
        is_valid, error = validator.validate_trade_plan(
            action="long",
            entry=100.0,
            stop_loss=99.8,  # 0.2% below entry (less than 0.5% min)
            take_profit=105.0,
            position_pct=5.0,
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid is False
        assert "Stop distance" in error
        assert "too tight" in error

    def test_stop_too_wide_from_entry(self, validator):
        """Test stop distance > max is rejected (too wide)"""
        is_valid, error = validator.validate_trade_plan(
            action="long",
            entry=101.0,  # Slightly above current to avoid entry deviation check
            stop_loss=84.0,  # 16.8% below entry (exceeds 15% max stop distance)
            take_profit=130.0,
            position_pct=5.0,
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid is False
        assert "Stop distance" in error
        assert "too wide" in error

    # Zero/Negative Price Validation

    def test_zero_entry_price(self, validator):
        """Test zero entry price is rejected"""
        is_valid, error = validator.validate_trade_plan(
            action="long",
            entry=0.0,
            stop_loss=95.0,
            take_profit=110.0,
            position_pct=5.0,
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid is False
        assert "Entry price must be positive" in error

    def test_negative_stop_loss(self, validator):
        """Test negative stop loss is rejected"""
        is_valid, error = validator.validate_trade_plan(
            action="long",
            entry=100.0,
            stop_loss=-5.0,
            take_profit=110.0,
            position_pct=5.0,
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid is False
        assert "Stop loss must be positive" in error

    # Edge Cases

    def test_empty_action(self, validator):
        """Test empty action string is rejected"""
        is_valid, error = validator.validate_trade_plan(
            action="",
            entry=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            position_pct=5.0,
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid is False
        assert "Action is required" in error

    def test_current_price_zero_skips_sanity_check(self, validator):
        """Test that zero current price skips sanity check"""
        # This should pass because sanity check is skipped when current_price = 0
        is_valid, error = validator.validate_trade_plan(
            action="long",
            entry=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            position_pct=5.0,
            current_price=0.0,  # Zero current price
            symbol="TEST"
        )
        # Should still pass other validations
        assert is_valid is True

    # Configuration Tests

    def test_custom_max_position_pct(self):
        """Test custom max position percentage"""
        config = ValidationConfig(max_position_pct=5.0)
        validator = TradePlanValidator(config)

        is_valid, error = validator.validate_trade_plan(
            action="long",
            entry=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            position_pct=7.0,  # Exceeds custom 5% max
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid is False
        assert "exceeds maximum allowed 5.0%" in error

    def test_custom_min_risk_reward(self):
        """Test custom minimum risk-reward ratio"""
        config = ValidationConfig(min_risk_reward_ratio=2.0)
        validator = TradePlanValidator(config)

        is_valid, error = validator.validate_trade_plan(
            action="long",
            entry=100.0,
            stop_loss=95.0,  # Risk: $5
            take_profit=107.5,  # Reward: $7.5 (R:R = 1.5 < 2.0)
            position_pct=5.0,
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid is False
        assert "below minimum 2.00" in error

    def test_case_insensitive_action(self, validator):
        """Test action is case insensitive"""
        # Test uppercase
        is_valid1, _ = validator.validate_trade_plan(
            action="LONG",
            entry=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            position_pct=5.0,
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid1 is True

        # Test mixed case
        is_valid2, _ = validator.validate_trade_plan(
            action="Short",
            entry=100.0,
            stop_loss=105.0,
            take_profit=90.0,
            position_pct=5.0,
            current_price=100.0,
            symbol="TEST"
        )
        assert is_valid2 is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
