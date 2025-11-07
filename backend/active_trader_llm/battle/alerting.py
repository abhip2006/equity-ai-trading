"""
Alert Manager: Centralized alerting system for battle monitoring.

Manages multiple notification channels and alert thresholds:
- Console alerts (real-time)
- File alerts (persistent log)
- Email alerts (future)
- Webhook alerts (future)
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from .notifications import (
    Alert,
    AlertSeverity,
    AlertType,
    BaseNotifier,
    ConsoleNotifier,
    FileNotifier
)

logger = logging.getLogger(__name__)


class AlertThresholds:
    """Alert threshold configuration"""
    def __init__(
        self,
        drawdown_warning_pct: float = 10.0,
        drawdown_critical_pct: float = 20.0,
        win_rate_warning_pct: float = 40.0,
        error_rate_warning_pct: float = 10.0,
        position_concentration_pct: float = 40.0,
        unusual_trade_count: int = 10,
        unusual_trade_window_minutes: int = 5
    ):
        """
        Initialize alert thresholds.

        Args:
            drawdown_warning_pct: Drawdown warning threshold (%)
            drawdown_critical_pct: Drawdown critical threshold (%)
            win_rate_warning_pct: Minimum acceptable win rate (%)
            error_rate_warning_pct: Maximum acceptable error rate (%)
            position_concentration_pct: Max single position size (%)
            unusual_trade_count: Number of trades that is unusual
            unusual_trade_window_minutes: Time window for unusual trading
        """
        self.drawdown_warning_pct = drawdown_warning_pct
        self.drawdown_critical_pct = drawdown_critical_pct
        self.win_rate_warning_pct = win_rate_warning_pct
        self.error_rate_warning_pct = error_rate_warning_pct
        self.position_concentration_pct = position_concentration_pct
        self.unusual_trade_count = unusual_trade_count
        self.unusual_trade_window_minutes = unusual_trade_window_minutes


class AlertManager:
    """
    Centralized alert management system.

    Handles:
    - Multiple notification channels
    - Alert threshold monitoring
    - Alert deduplication
    - Alert history tracking
    """

    def __init__(
        self,
        thresholds: AlertThresholds,
        notifiers: Optional[List[BaseNotifier]] = None,
        dedup_window_minutes: int = 5
    ):
        """
        Initialize alert manager.

        Args:
            thresholds: Alert threshold configuration
            notifiers: List of notification channels
            dedup_window_minutes: Time window for deduplicating alerts
        """
        self.thresholds = thresholds
        self.notifiers = notifiers or []
        self.dedup_window = timedelta(minutes=dedup_window_minutes)

        # Alert history for deduplication
        self.recent_alerts: Dict[str, datetime] = {}

        logger.info(f"AlertManager initialized with {len(self.notifiers)} notifiers")

    def add_notifier(self, notifier: BaseNotifier):
        """Add a notification channel"""
        self.notifiers.append(notifier)
        logger.info(f"Added notifier: {notifier.__class__.__name__}")

    def send_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        model_id: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> bool:
        """
        Send an alert through all enabled channels.

        Args:
            alert_type: Type of alert
            severity: Alert severity level
            model_id: Model that triggered the alert
            message: Alert message
            metadata: Additional alert data
            force: If True, bypass deduplication

        Returns:
            True if alert was sent, False if deduplicated
        """
        # Check for duplicate alerts
        if not force and self._is_duplicate_alert(alert_type, model_id):
            logger.debug(f"Skipping duplicate alert: {alert_type.value} for {model_id}")
            return False

        # Create alert
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            model_id=model_id,
            message=message,
            metadata=metadata or {}
        )

        # Send through all notifiers
        sent_count = 0
        for notifier in self.notifiers:
            if notifier.is_enabled():
                try:
                    if notifier.send_alert(alert):
                        sent_count += 1
                except Exception as e:
                    logger.error(f"Failed to send alert via {notifier.__class__.__name__}: {e}")

        # Record alert for deduplication
        alert_key = f"{alert_type.value}:{model_id}"
        self.recent_alerts[alert_key] = datetime.now()

        logger.info(f"Alert sent via {sent_count}/{len(self.notifiers)} notifiers: {alert}")
        return True

    def _is_duplicate_alert(self, alert_type: AlertType, model_id: str) -> bool:
        """Check if this alert was recently sent"""
        alert_key = f"{alert_type.value}:{model_id}"
        last_sent = self.recent_alerts.get(alert_key)

        if last_sent is None:
            return False

        # Check if alert is within dedup window
        time_since_last = datetime.now() - last_sent
        return time_since_last < self.dedup_window

    def check_drawdown(self, model_id: str, drawdown_pct: float, equity: float):
        """
        Check drawdown thresholds and send alerts if exceeded.

        Args:
            model_id: Model identifier
            drawdown_pct: Current drawdown percentage
            equity: Current equity value
        """
        if drawdown_pct >= self.thresholds.drawdown_critical_pct:
            self.send_alert(
                alert_type=AlertType.CRITICAL_DRAWDOWN,
                severity=AlertSeverity.EMERGENCY,
                model_id=model_id,
                message=f"CRITICAL DRAWDOWN: {drawdown_pct:.1f}% (threshold: {self.thresholds.drawdown_critical_pct:.1f}%)",
                metadata={
                    'drawdown_pct': drawdown_pct,
                    'threshold_pct': self.thresholds.drawdown_critical_pct,
                    'equity': equity
                }
            )
        elif drawdown_pct >= self.thresholds.drawdown_warning_pct:
            self.send_alert(
                alert_type=AlertType.WARNING_DRAWDOWN,
                severity=AlertSeverity.WARNING,
                model_id=model_id,
                message=f"Warning: Drawdown at {drawdown_pct:.1f}% (threshold: {self.thresholds.drawdown_warning_pct:.1f}%)",
                metadata={
                    'drawdown_pct': drawdown_pct,
                    'threshold_pct': self.thresholds.drawdown_warning_pct,
                    'equity': equity
                }
            )

    def check_error_rate(self, model_id: str, error_count: int, total_cycles: int):
        """
        Check error rate and send alert if too high.

        Args:
            model_id: Model identifier
            error_count: Number of errors
            total_cycles: Total decision cycles
        """
        if total_cycles == 0:
            return

        error_rate_pct = (error_count / total_cycles) * 100

        if error_rate_pct >= self.thresholds.error_rate_warning_pct:
            self.send_alert(
                alert_type=AlertType.HIGH_ERROR_RATE,
                severity=AlertSeverity.WARNING,
                model_id=model_id,
                message=f"High error rate: {error_rate_pct:.1f}% ({error_count}/{total_cycles} cycles)",
                metadata={
                    'error_rate_pct': error_rate_pct,
                    'error_count': error_count,
                    'total_cycles': total_cycles,
                    'threshold_pct': self.thresholds.error_rate_warning_pct
                }
            )

    def check_win_rate(self, model_id: str, win_rate_pct: float, total_trades: int):
        """
        Check win rate and send alert if too low.

        Args:
            model_id: Model identifier
            win_rate_pct: Current win rate percentage
            total_trades: Total number of trades
        """
        # Only check if we have enough trades for statistical significance
        if total_trades < 10:
            return

        if win_rate_pct < self.thresholds.win_rate_warning_pct:
            self.send_alert(
                alert_type=AlertType.LOW_WIN_RATE,
                severity=AlertSeverity.WARNING,
                model_id=model_id,
                message=f"Low win rate: {win_rate_pct:.1f}% (threshold: {self.thresholds.win_rate_warning_pct:.1f}%)",
                metadata={
                    'win_rate_pct': win_rate_pct,
                    'threshold_pct': self.thresholds.win_rate_warning_pct,
                    'total_trades': total_trades
                }
            )

    def check_position_concentration(
        self,
        model_id: str,
        position_value: float,
        total_equity: float,
        symbol: str
    ):
        """
        Check if a single position is too concentrated.

        Args:
            model_id: Model identifier
            position_value: Value of the position
            total_equity: Total account equity
            symbol: Stock symbol
        """
        if total_equity == 0:
            return

        concentration_pct = (position_value / total_equity) * 100

        if concentration_pct >= self.thresholds.position_concentration_pct:
            self.send_alert(
                alert_type=AlertType.POSITION_CONCENTRATION,
                severity=AlertSeverity.WARNING,
                model_id=model_id,
                message=f"High position concentration: {symbol} at {concentration_pct:.1f}% of equity",
                metadata={
                    'symbol': symbol,
                    'concentration_pct': concentration_pct,
                    'threshold_pct': self.thresholds.position_concentration_pct,
                    'position_value': position_value,
                    'equity': total_equity
                }
            )

    def alert_model_crash(self, model_id: str, error_message: str):
        """Alert when a model crashes"""
        self.send_alert(
            alert_type=AlertType.MODEL_CRASH,
            severity=AlertSeverity.CRITICAL,
            model_id=model_id,
            message=f"Model crashed: {error_message}",
            metadata={'error': error_message},
            force=True  # Always send crash alerts
        )

    def alert_model_recovery(self, model_id: str, downtime_seconds: int):
        """Alert when a model recovers from error state"""
        self.send_alert(
            alert_type=AlertType.MODEL_RECOVERY,
            severity=AlertSeverity.INFO,
            model_id=model_id,
            message=f"Model recovered after {downtime_seconds}s downtime",
            metadata={'downtime_seconds': downtime_seconds}
        )

    def alert_circuit_breaker(self, model_id: str, reason: str):
        """Alert when circuit breaker is triggered"""
        self.send_alert(
            alert_type=AlertType.CIRCUIT_BREAKER,
            severity=AlertSeverity.EMERGENCY,
            model_id=model_id,
            message=f"CIRCUIT BREAKER TRIGGERED: {reason}",
            metadata={'reason': reason},
            force=True  # Always send circuit breaker alerts
        )

    def cleanup_old_alerts(self, max_age_hours: int = 24):
        """Remove old alerts from deduplication tracking"""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = [
            key for key, timestamp in self.recent_alerts.items()
            if timestamp < cutoff
        ]
        for key in to_remove:
            del self.recent_alerts[key]

        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old alerts from tracking")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create alert manager with default thresholds
    thresholds = AlertThresholds()
    manager = AlertManager(thresholds=thresholds)

    # Add notification channels
    manager.add_notifier(ConsoleNotifier(enabled=True, use_colors=True))
    manager.add_notifier(FileNotifier(file_path="data/battle/alerts.log", enabled=True))

    print("\n=== Testing Alert Manager ===\n")

    # Test drawdown alerts
    print("Testing drawdown monitoring...")
    manager.check_drawdown("gpt4o", drawdown_pct=12.5, equity=87500)
    manager.check_drawdown("claude_sonnet", drawdown_pct=22.0, equity=78000)

    # Test error rate alerts
    print("\nTesting error rate monitoring...")
    manager.check_error_rate("grok", error_count=3, total_cycles=20)

    # Test win rate alerts
    print("\nTesting win rate monitoring...")
    manager.check_win_rate("deepseek", win_rate_pct=35.0, total_trades=20)

    # Test model crash alert
    print("\nTesting model crash alert...")
    manager.alert_model_crash("gemini", "Connection timeout after 3 retries")

    # Test position concentration
    print("\nTesting position concentration...")
    manager.check_position_concentration("qwen", position_value=45000, total_equity=100000, symbol="NVDA")

    # Test circuit breaker
    print("\nTesting circuit breaker alert...")
    manager.alert_circuit_breaker("gpt4o", "Daily drawdown limit exceeded")

    print("\n=== Alert Manager Test Complete ===")
