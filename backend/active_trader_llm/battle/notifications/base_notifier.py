"""
Base Notifier: Abstract base class for alert notification channels.

All notification channels (console, file, email, etc.) inherit from this.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any
from enum import Enum


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class AlertType(Enum):
    """Types of alerts"""
    MODEL_CRASH = "model_crash"
    HIGH_ERROR_RATE = "high_error_rate"
    CRITICAL_DRAWDOWN = "critical_drawdown"
    WARNING_DRAWDOWN = "warning_drawdown"
    LOW_WIN_RATE = "low_win_rate"
    UNUSUAL_TRADING = "unusual_trading"
    POSITION_CONCENTRATION = "position_concentration"
    CIRCUIT_BREAKER = "circuit_breaker"
    MODEL_RECOVERY = "model_recovery"


class Alert:
    """Alert data structure"""
    def __init__(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        model_id: str,
        message: str,
        metadata: Dict[str, Any] = None,
        timestamp: datetime = None
    ):
        self.alert_type = alert_type
        self.severity = severity
        self.model_id = model_id
        self.message = message
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'model_id': self.model_id,
            'message': self.message,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }

    def __str__(self) -> str:
        """String representation of alert"""
        return (
            f"[{self.severity.value}] {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
            f"| {self.model_id} | {self.alert_type.value}: {self.message}"
        )


class BaseNotifier(ABC):
    """
    Abstract base class for notification channels.

    All notifiers must implement the send_alert method.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize base notifier.

        Args:
            enabled: Whether this notifier is active
        """
        self.enabled = enabled

    @abstractmethod
    def send_alert(self, alert: Alert) -> bool:
        """
        Send an alert through this notification channel.

        Args:
            alert: Alert object to send

        Returns:
            True if alert was sent successfully, False otherwise
        """
        pass

    def is_enabled(self) -> bool:
        """Check if this notifier is enabled"""
        return self.enabled

    def enable(self):
        """Enable this notifier"""
        self.enabled = True

    def disable(self):
        """Disable this notifier"""
        self.enabled = False
