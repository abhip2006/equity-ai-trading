"""
Trade Plan Validator

Validates trade plans to ensure they follow basic trading logic and risk management rules.
Prevents accepting impossible or dangerous trade parameters from LLM outputs.
"""

from typing import Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """
    Configuration for trade plan validation.

    IMPORTANT: Validator should only catch LLM output errors (broken logic),
    not make trading decisions. Let the LLM agent decide position sizes,
    risk-reward ratios, and trade parameters autonomously.
    """
    max_position_pct: float = 100.0  # LLM decides position size (only prevent >100%)
    min_risk_reward_ratio: float = 0.0  # DISABLED - LLM decides if trade is worth taking
    max_price_deviation_pct: float = 50.0  # Very permissive - allow limit orders far from price
    min_stop_distance_pct: float = 0.1  # Very tight - only prevent same-price stops
    max_stop_distance_pct: float = 50.0  # Very wide - LLM decides appropriate stop distance


class TradeValidationError(Exception):
    """Custom exception for trade plan validation errors"""
    pass


class TradePlanValidator:
    """
    Validates trade plans to catch LLM output errors (broken logic).

    DOES NOT make trading decisions - the LLM agent decides position sizes,
    risk-reward ratios, stop distances, etc.

    Safety checks ONLY:
    1. Required fields are present and non-zero
    2. Price logic is correct (stops/targets on correct side of entry)
    3. Position size doesn't exceed 100% of capital
    4. Catches obviously broken outputs (stop = entry, negative prices, etc.)

    All other trading decisions (R:R ratios, stop distances, price levels) are
    left to the LLM agent's judgment.
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()

    def validate_trade_plan(
        self,
        action: str,
        entry: float,
        stop_loss: float,
        take_profit: float,
        position_pct: float,
        current_price: float,
        symbol: str = "UNKNOWN"
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a complete trade plan.

        Args:
            action: "long" or "short"
            entry: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            position_pct: Position size as percentage (0-100)
            current_price: Current market price
            symbol: Symbol being traded (for logging)

        Returns:
            Tuple of (is_valid, error_message)
            If valid: (True, None)
            If invalid: (False, "detailed error message")
        """
        try:
            # 1. Validate required fields are present and positive
            self._validate_required_fields(action, entry, stop_loss, take_profit, position_pct)

            # 2. Validate position size bounds
            self._validate_position_size(position_pct)

            # 3. Validate price logic based on direction
            if action.lower() == "long":
                self._validate_long_prices(entry, stop_loss, take_profit)
            elif action.lower() == "short":
                self._validate_short_prices(entry, stop_loss, take_profit)
            else:
                raise TradeValidationError(f"Invalid action '{action}'. Must be 'long' or 'short'")

            # 4. Validate risk-reward ratio
            self._validate_risk_reward(action, entry, stop_loss, take_profit)

            # 5. Validate prices are reasonable vs current market price
            self._validate_price_sanity(entry, stop_loss, take_profit, current_price)

            # 6. Validate stop distance is reasonable
            self._validate_stop_distance(action, entry, stop_loss)

            logger.info(f"[{symbol}] Trade plan validation PASSED")
            return True, None

        except TradeValidationError as e:
            error_msg = f"[{symbol}] Trade plan validation FAILED: {str(e)}"
            logger.error(error_msg)
            logger.error(f"[{symbol}] Rejected plan details: action={action}, entry={entry:.2f}, "
                        f"stop={stop_loss:.2f}, target={take_profit:.2f}, size={position_pct:.1f}%, "
                        f"current_price={current_price:.2f}")
            return False, str(e)
        except Exception as e:
            error_msg = f"[{symbol}] Unexpected validation error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def _validate_required_fields(
        self,
        action: str,
        entry: float,
        stop_loss: float,
        take_profit: float,
        position_pct: float
    ):
        """Validate all required fields are present and have valid values"""
        if not action or action.strip() == "":
            raise TradeValidationError("Action is required")

        if entry <= 0:
            raise TradeValidationError(f"Entry price must be positive, got {entry}")

        if stop_loss <= 0:
            raise TradeValidationError(f"Stop loss must be positive, got {stop_loss}")

        if take_profit <= 0:
            raise TradeValidationError(f"Take profit must be positive, got {take_profit}")

        if position_pct <= 0:
            raise TradeValidationError(f"Position percentage must be positive, got {position_pct}")

    def _validate_position_size(self, position_pct: float):
        """Validate position size is within acceptable bounds (only sanity check for >100%)"""
        if position_pct > 100.0:
            raise TradeValidationError(
                f"Position size {position_pct:.1f}% exceeds 100% of capital"
            )

    def _validate_long_prices(self, entry: float, stop_loss: float, take_profit: float):
        """Validate price logic for long positions: stop < entry < target"""
        if stop_loss >= entry:
            raise TradeValidationError(
                f"LONG: Stop loss ({stop_loss:.2f}) must be BELOW entry ({entry:.2f}). "
                f"Got stop ABOVE/EQUAL entry - inverse logic detected!"
            )

        if take_profit <= entry:
            raise TradeValidationError(
                f"LONG: Take profit ({take_profit:.2f}) must be ABOVE entry ({entry:.2f}). "
                f"Got target BELOW/EQUAL entry - inverse logic detected!"
            )

        if stop_loss >= take_profit:
            raise TradeValidationError(
                f"LONG: Stop loss ({stop_loss:.2f}) must be below take profit ({take_profit:.2f}). "
                f"Impossible trade geometry detected!"
            )

    def _validate_short_prices(self, entry: float, stop_loss: float, take_profit: float):
        """Validate price logic for short positions: target < entry < stop"""
        if stop_loss <= entry:
            raise TradeValidationError(
                f"SHORT: Stop loss ({stop_loss:.2f}) must be ABOVE entry ({entry:.2f}). "
                f"Got stop BELOW/EQUAL entry - inverse logic detected!"
            )

        if take_profit >= entry:
            raise TradeValidationError(
                f"SHORT: Take profit ({take_profit:.2f}) must be BELOW entry ({entry:.2f}). "
                f"Got target ABOVE/EQUAL entry - inverse logic detected!"
            )

        if take_profit >= stop_loss:
            raise TradeValidationError(
                f"SHORT: Take profit ({take_profit:.2f}) must be below stop loss ({stop_loss:.2f}). "
                f"Impossible trade geometry detected!"
            )

    def _validate_risk_reward(
        self,
        action: str,
        entry: float,
        stop_loss: float,
        take_profit: float
    ):
        """
        Validate risk and reward are positive (catches broken logic).

        Does NOT enforce minimum R:R ratio - the LLM decides if trade is worth taking.
        """
        if action.lower() == "long":
            risk = entry - stop_loss
            reward = take_profit - entry
        else:  # short
            risk = stop_loss - entry
            reward = entry - take_profit

        # Only check that risk/reward are positive (catches broken price logic)
        if risk <= 0:
            raise TradeValidationError(
                f"Risk must be positive, got {risk:.2f}. "
                f"This indicates stop is on wrong side of entry."
            )

        if reward <= 0:
            raise TradeValidationError(
                f"Reward must be positive, got {reward:.2f}. "
                f"This indicates target is on wrong side of entry."
            )

        # Calculate R:R for logging, but don't enforce minimum (LLM decides)
        risk_reward_ratio = reward / risk

        # Only enforce minimum if explicitly configured (default is 0.0 = disabled)
        if self.config.min_risk_reward_ratio > 0 and risk_reward_ratio < self.config.min_risk_reward_ratio:
            logger.warning(
                f"R:R ratio {risk_reward_ratio:.2f} is below configured minimum "
                f"{self.config.min_risk_reward_ratio:.2f}, but allowing LLM's decision. "
                f"Risk=${risk:.2f}, Reward=${reward:.2f}"
            )

    def _validate_price_sanity(
        self,
        entry: float,
        stop_loss: float,
        take_profit: float,
        current_price: float
    ):
        """
        Validate entry price is within reasonable range of current market price.

        Note: We only check entry price here. Stop and target prices are validated
        via stop_distance checks, which allow them to be further from current price
        based on the trade's risk profile.
        """
        if current_price <= 0:
            logger.warning("Current price is zero or negative, skipping sanity check")
            return

        max_deviation = self.config.max_price_deviation_pct / 100.0

        # Only validate entry price is near current price
        # Stop and target are validated via stop distance checks
        entry_deviation = abs(entry - current_price) / current_price

        if entry_deviation > max_deviation:
            raise TradeValidationError(
                f"entry price {entry:.2f} deviates {entry_deviation*100:.1f}% "
                f"from current price {current_price:.2f}, exceeding maximum allowed "
                f"{self.config.max_price_deviation_pct:.1f}%. Entry should be near current market price."
            )

    def _validate_stop_distance(self, action: str, entry: float, stop_loss: float):
        """Validate stop distance is reasonable (not too tight, not too wide)"""
        if action.lower() == "long":
            stop_distance_pct = (entry - stop_loss) / entry * 100.0
        else:  # short
            stop_distance_pct = (stop_loss - entry) / entry * 100.0

        if stop_distance_pct < self.config.min_stop_distance_pct:
            raise TradeValidationError(
                f"Stop distance {stop_distance_pct:.2f}% is too tight, "
                f"minimum is {self.config.min_stop_distance_pct:.1f}%. "
                f"Risk getting stopped out by noise."
            )

        if stop_distance_pct > self.config.max_stop_distance_pct:
            raise TradeValidationError(
                f"Stop distance {stop_distance_pct:.2f}% is too wide, "
                f"maximum is {self.config.max_stop_distance_pct:.1f}%. "
                f"Risk per trade is excessive."
            )

    def validate_position_adjustment(
        self,
        adjustment_type: str,  # "stop", "target", or "both"
        direction: str,  # "long" or "short"
        entry_price: float,
        current_price: float,
        old_stop: Optional[float],
        new_stop: Optional[float],
        old_target: Optional[float],
        new_target: Optional[float],
        symbol: str = "UNKNOWN"
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a position adjustment (stop/target modification).

        Different validation rules than new positions:
        - Stops can only be moved in favorable direction or to protect profits
        - Targets can be adjusted freely
        - Must maintain minimum distance from current price
        - Allow wider stops if LLM determines increased volatility

        Args:
            adjustment_type: "stop", "target", or "both"
            direction: "long" or "short"
            entry_price: Original entry price
            current_price: Current market price
            old_stop: Current stop loss (None for target-only adjustment)
            new_stop: New stop loss (None for target-only adjustment)
            old_target: Current take profit (None for stop-only adjustment)
            new_target: New take profit (None for stop-only adjustment)
            symbol: Symbol being adjusted (for logging)

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Validate stop adjustment
            if adjustment_type in ["stop", "both"] and new_stop is not None:
                self._validate_stop_adjustment(
                    direction, entry_price, current_price, old_stop, new_stop
                )

            # Validate target adjustment
            if adjustment_type in ["target", "both"] and new_target is not None:
                self._validate_target_adjustment(
                    direction, entry_price, current_price, new_target
                )

            # Validate new stop/target geometry
            if new_stop is not None and new_target is not None:
                if direction.lower() == "long":
                    self._validate_long_prices(entry_price, new_stop, new_target)
                else:
                    self._validate_short_prices(entry_price, new_stop, new_target)

            logger.info(f"[{symbol}] Position adjustment validation PASSED")
            return True, None

        except TradeValidationError as e:
            error_msg = f"[{symbol}] Position adjustment validation FAILED: {str(e)}"
            logger.error(error_msg)
            logger.error(f"[{symbol}] Rejected adjustment: type={adjustment_type}, direction={direction}, "
                        f"entry={entry_price:.2f}, current={current_price:.2f}, "
                        f"old_stop={old_stop:.2f if old_stop else 'N/A'}, "
                        f"new_stop={new_stop:.2f if new_stop else 'N/A'}, "
                        f"old_target={old_target:.2f if old_target else 'N/A'}, "
                        f"new_target={new_target:.2f if new_target else 'N/A'}")
            return False, str(e)
        except Exception as e:
            error_msg = f"[{symbol}] Unexpected adjustment validation error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def _validate_stop_adjustment(
        self,
        direction: str,
        entry_price: float,
        current_price: float,
        old_stop: Optional[float],
        new_stop: float
    ):
        """
        Validate stop loss adjustment.

        Rules:
        - New stop must be on correct side of entry (basic sanity)
        - Stops can be tightened (reduced risk) or widened if LLM determines necessary
        - Trailing stops (moving stop up/down as price moves) are allowed and encouraged
        - Must maintain minimum distance from current price to avoid immediate stop-out
        """
        if new_stop <= 0:
            raise TradeValidationError(f"New stop loss must be positive, got {new_stop}")

        if direction.lower() == "long":
            # Long position: stop must be below entry
            if new_stop >= entry_price:
                raise TradeValidationError(
                    f"LONG: New stop ({new_stop:.2f}) must be BELOW entry ({entry_price:.2f})"
                )

            # Check minimum distance from current price (prevent immediate stop-out)
            distance_from_current = (current_price - new_stop) / current_price * 100.0
            if distance_from_current < self.config.min_stop_distance_pct:
                raise TradeValidationError(
                    f"LONG: New stop ({new_stop:.2f}) too close to current price ({current_price:.2f}). "
                    f"Distance: {distance_from_current:.2f}%, minimum: {self.config.min_stop_distance_pct:.1f}%"
                )

            # Allow any adjustment (tighter or wider) - LLM decides based on volatility/market conditions
            if old_stop and new_stop < old_stop:
                logger.info(f"LONG: Moving stop DOWN ${old_stop:.2f} → ${new_stop:.2f} (increasing risk - LLM decision)")
            elif old_stop and new_stop > old_stop:
                profit_protection = (new_stop - entry_price) / entry_price * 100.0
                if profit_protection > 0:
                    logger.info(f"LONG: Trailing stop UP ${old_stop:.2f} → ${new_stop:.2f} (locking in {profit_protection:.1f}% profit)")
                else:
                    logger.info(f"LONG: Tightening stop UP ${old_stop:.2f} → ${new_stop:.2f} (reducing risk)")

        else:  # short
            # Short position: stop must be above entry
            if new_stop <= entry_price:
                raise TradeValidationError(
                    f"SHORT: New stop ({new_stop:.2f}) must be ABOVE entry ({entry_price:.2f})"
                )

            # Check minimum distance from current price
            distance_from_current = (new_stop - current_price) / current_price * 100.0
            if distance_from_current < self.config.min_stop_distance_pct:
                raise TradeValidationError(
                    f"SHORT: New stop ({new_stop:.2f}) too close to current price ({current_price:.2f}). "
                    f"Distance: {distance_from_current:.2f}%, minimum: {self.config.min_stop_distance_pct:.1f}%"
                )

            # Allow any adjustment - LLM decides
            if old_stop and new_stop > old_stop:
                logger.info(f"SHORT: Moving stop UP ${old_stop:.2f} → ${new_stop:.2f} (increasing risk - LLM decision)")
            elif old_stop and new_stop < old_stop:
                profit_protection = (entry_price - new_stop) / entry_price * 100.0
                if profit_protection > 0:
                    logger.info(f"SHORT: Trailing stop DOWN ${old_stop:.2f} → ${new_stop:.2f} (locking in {profit_protection:.1f}% profit)")
                else:
                    logger.info(f"SHORT: Tightening stop DOWN ${old_stop:.2f} → ${new_stop:.2f} (reducing risk)")

    def _validate_target_adjustment(
        self,
        direction: str,
        entry_price: float,
        current_price: float,
        new_target: float
    ):
        """
        Validate take profit adjustment.

        Targets can be adjusted more freely than stops:
        - Can be extended if momentum accelerating
        - Can be pulled in if taking profits early
        - Must be on correct side of entry
        """
        if new_target <= 0:
            raise TradeValidationError(f"New take profit must be positive, got {new_target}")

        if direction.lower() == "long":
            # Long: target must be above entry
            if new_target <= entry_price:
                raise TradeValidationError(
                    f"LONG: New target ({new_target:.2f}) must be ABOVE entry ({entry_price:.2f})"
                )

            # Target can be anywhere above entry - LLM decides based on setup
            logger.info(f"LONG: Adjusting target to ${new_target:.2f} (potential gain: {((new_target - entry_price) / entry_price * 100):.1f}%)")

        else:  # short
            # Short: target must be below entry
            if new_target >= entry_price:
                raise TradeValidationError(
                    f"SHORT: New target ({new_target:.2f}) must be BELOW entry ({entry_price:.2f})"
                )

            logger.info(f"SHORT: Adjusting target to ${new_target:.2f} (potential gain: {((entry_price - new_target) / entry_price * 100):.1f}%)")


def create_validator_from_config(config_dict: dict) -> TradePlanValidator:
    """
    Create a validator from a config dictionary.

    Defaults are very permissive - validator only catches broken LLM outputs,
    not trading decisions.

    Args:
        config_dict: Dictionary with validation config settings

    Returns:
        Configured TradePlanValidator instance
    """
    validation_config = ValidationConfig(
        max_position_pct=config_dict.get("max_position_pct", 100.0),
        min_risk_reward_ratio=config_dict.get("min_risk_reward_ratio", 0.0),  # Disabled by default
        max_price_deviation_pct=config_dict.get("max_price_deviation_pct", 50.0),
        min_stop_distance_pct=config_dict.get("min_stop_distance_pct", 0.1),
        max_stop_distance_pct=config_dict.get("max_stop_distance_pct", 50.0)
    )

    return TradePlanValidator(validation_config)
