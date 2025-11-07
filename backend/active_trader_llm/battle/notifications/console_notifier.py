"""
Console Notifier: Sends alerts to console/stdout with color formatting.
"""

import logging
from typing import Dict, Any
from .base_notifier import BaseNotifier, Alert, AlertSeverity

logger = logging.getLogger(__name__)


class ConsoleNotifier(BaseNotifier):
    """
    Console notification channel.

    Sends alerts to stdout with severity-based formatting and colors.
    """

    # ANSI color codes
    COLORS = {
        AlertSeverity.INFO: '\033[94m',      # Blue
        AlertSeverity.WARNING: '\033[93m',   # Yellow
        AlertSeverity.CRITICAL: '\033[91m',  # Red
        AlertSeverity.EMERGENCY: '\033[95m'  # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def __init__(self, enabled: bool = True, use_colors: bool = True):
        """
        Initialize console notifier.

        Args:
            enabled: Whether this notifier is active
            use_colors: Whether to use ANSI color codes
        """
        super().__init__(enabled)
        self.use_colors = use_colors

    def send_alert(self, alert: Alert) -> bool:
        """
        Send alert to console.

        Args:
            alert: Alert object to send

        Returns:
            True (console alerts always succeed)
        """
        if not self.enabled:
            return False

        try:
            # Build formatted message
            if self.use_colors:
                color = self.COLORS.get(alert.severity, '')
                message = (
                    f"{color}{self.BOLD}[{alert.severity.value}]{self.RESET} "
                    f"{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"{color}{alert.model_id}{self.RESET} | "
                    f"{alert.alert_type.value}: {alert.message}"
                )
            else:
                message = str(alert)

            # Print to console
            print(message)

            # Also log it
            log_level = self._get_log_level(alert.severity)
            logger.log(log_level, f"{alert.model_id} | {alert.alert_type.value}: {alert.message}")

            # Print metadata if available
            if alert.metadata and self.use_colors:
                metadata_str = self._format_metadata(alert.metadata)
                if metadata_str:
                    print(f"  {metadata_str}")

            return True

        except Exception as e:
            logger.error(f"Failed to send console alert: {e}")
            return False

    def _get_log_level(self, severity: AlertSeverity) -> int:
        """Map alert severity to logging level"""
        mapping = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.ERROR,
            AlertSeverity.EMERGENCY: logging.CRITICAL
        }
        return mapping.get(severity, logging.INFO)

    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata for console display"""
        if not metadata:
            return ""

        parts = []
        for key, value in metadata.items():
            if isinstance(value, float):
                # Format floats nicely
                if 'pct' in key.lower() or 'rate' in key.lower():
                    parts.append(f"{key}={value:.2f}%")
                else:
                    parts.append(f"{key}={value:.2f}")
            elif isinstance(value, int):
                parts.append(f"{key}={value}")
            else:
                parts.append(f"{key}={value}")

        return " | ".join(parts)


# Example usage
if __name__ == "__main__":
    from .base_notifier import AlertType

    # Create console notifier
    notifier = ConsoleNotifier(enabled=True, use_colors=True)

    # Test alerts at different severity levels
    test_alerts = [
        Alert(
            alert_type=AlertType.MODEL_CRASH,
            severity=AlertSeverity.CRITICAL,
            model_id="gpt4o",
            message="Model crashed during decision cycle",
            metadata={'error_count': 3, 'last_error': 'Connection timeout'}
        ),
        Alert(
            alert_type=AlertType.CRITICAL_DRAWDOWN,
            severity=AlertSeverity.EMERGENCY,
            model_id="claude_sonnet",
            message="Drawdown exceeds critical threshold",
            metadata={'drawdown_pct': 22.5, 'threshold_pct': 20.0}
        ),
        Alert(
            alert_type=AlertType.LOW_WIN_RATE,
            severity=AlertSeverity.WARNING,
            model_id="grok",
            message="Win rate below acceptable threshold",
            metadata={'win_rate_pct': 35.0, 'threshold_pct': 40.0, 'trades': 20}
        ),
        Alert(
            alert_type=AlertType.MODEL_RECOVERY,
            severity=AlertSeverity.INFO,
            model_id="gpt4o",
            message="Model recovered and resumed normal operation",
            metadata={'downtime_seconds': 45}
        )
    ]

    print("\n=== Console Notifier Test ===\n")
    for alert in test_alerts:
        notifier.send_alert(alert)
        print()  # Extra line for readability
