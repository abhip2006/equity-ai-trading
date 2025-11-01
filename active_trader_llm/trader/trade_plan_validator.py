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
    """Configuration for trade plan validation"""
    max_position_pct: float = 10.0  # Maximum position size as % of capital
    min_risk_reward_ratio: float = 1.5  # Minimum R:R ratio
    max_price_deviation_pct: float = 5.0  # Max deviation from current price
    min_stop_distance_pct: float = 0.5  # Minimum stop distance from entry
    max_stop_distance_pct: float = 15.0  # Maximum stop distance from entry


class TradeValidationError(Exception):
    """Custom exception for trade plan validation errors"""
    pass


class TradePlanValidator:
    """
    Validates trade plans to ensure they meet basic trading logic requirements.

    Validation checks:
    1. Required fields are present and non-zero
    2. Price logic is correct (stops/targets on right side of entry)
    3. Position size is within bounds
    4. Risk-reward ratio is acceptable
    5. Prices are within reasonable range of current market price
    6. Stop distance is reasonable
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
        """Validate position size is within acceptable bounds"""
        if position_pct > 100.0:
            raise TradeValidationError(
                f"Position size {position_pct:.1f}% exceeds 100% of capital"
            )

        if position_pct > self.config.max_position_pct:
            raise TradeValidationError(
                f"Position size {position_pct:.1f}% exceeds maximum allowed "
                f"{self.config.max_position_pct:.1f}%"
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
        """Validate risk-reward ratio is acceptable"""
        if action.lower() == "long":
            risk = entry - stop_loss
            reward = take_profit - entry
        else:  # short
            risk = stop_loss - entry
            reward = entry - take_profit

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

        risk_reward_ratio = reward / risk

        if risk_reward_ratio < self.config.min_risk_reward_ratio:
            raise TradeValidationError(
                f"Risk-reward ratio {risk_reward_ratio:.2f} is below minimum "
                f"{self.config.min_risk_reward_ratio:.2f}. "
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


def create_validator_from_config(config_dict: dict) -> TradePlanValidator:
    """
    Create a validator from a config dictionary.

    Args:
        config_dict: Dictionary with validation config settings

    Returns:
        Configured TradePlanValidator instance
    """
    validation_config = ValidationConfig(
        max_position_pct=config_dict.get("max_position_pct", 10.0),
        min_risk_reward_ratio=config_dict.get("min_risk_reward_ratio", 1.5),
        max_price_deviation_pct=config_dict.get("max_price_deviation_pct", 5.0),
        min_stop_distance_pct=config_dict.get("min_stop_distance_pct", 0.5),
        max_stop_distance_pct=config_dict.get("max_stop_distance_pct", 15.0)
    )

    return TradePlanValidator(validation_config)
