"""
File Notifier: Sends alerts to a log file with rotation support.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .base_notifier import BaseNotifier, Alert

logger = logging.getLogger(__name__)


class FileNotifier(BaseNotifier):
    """
    File notification channel.

    Writes alerts to a JSONL (JSON Lines) file for easy parsing and analysis.
    Supports log rotation to prevent files from growing too large.
    """

    def __init__(
        self,
        file_path: str,
        enabled: bool = True,
        max_file_size_mb: int = 10,
        backup_count: int = 5
    ):
        """
        Initialize file notifier.

        Args:
            file_path: Path to alert log file
            enabled: Whether this notifier is active
            max_file_size_mb: Maximum file size before rotation (in MB)
            backup_count: Number of backup files to keep
        """
        super().__init__(enabled)
        self.file_path = Path(file_path)
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.backup_count = backup_count

        # Create directory if it doesn't exist
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create file if it doesn't exist
        if not self.file_path.exists():
            self.file_path.touch()

        logger.info(f"FileNotifier initialized: {self.file_path}")

    def send_alert(self, alert: Alert) -> bool:
        """
        Send alert to file.

        Args:
            alert: Alert object to send

        Returns:
            True if alert was written successfully, False otherwise
        """
        if not self.enabled:
            return False

        try:
            # Check if rotation is needed
            self._rotate_if_needed()

            # Write alert as JSON line
            with open(self.file_path, 'a') as f:
                alert_dict = alert.to_dict()
                f.write(json.dumps(alert_dict) + '\n')

            return True

        except Exception as e:
            logger.error(f"Failed to send file alert: {e}")
            return False

    def _rotate_if_needed(self):
        """Rotate log file if it exceeds max size"""
        try:
            if not self.file_path.exists():
                return

            file_size = self.file_path.stat().st_size

            if file_size >= self.max_file_size_bytes:
                logger.info(f"Rotating alert log file (size: {file_size / 1024 / 1024:.2f} MB)")
                self._rotate_files()

        except Exception as e:
            logger.error(f"Failed to rotate log file: {e}")

    def _rotate_files(self):
        """Rotate log files, keeping backup_count backups"""
        # Remove oldest backup if it exists
        oldest_backup = self.file_path.with_suffix(f'.log.{self.backup_count}')
        if oldest_backup.exists():
            oldest_backup.unlink()

        # Shift all backups
        for i in range(self.backup_count - 1, 0, -1):
            old_backup = self.file_path.with_suffix(f'.log.{i}')
            new_backup = self.file_path.with_suffix(f'.log.{i + 1}')
            if old_backup.exists():
                old_backup.rename(new_backup)

        # Move current log to .1
        if self.file_path.exists():
            first_backup = self.file_path.with_suffix('.log.1')
            self.file_path.rename(first_backup)

        # Create new empty log
        self.file_path.touch()

    def get_recent_alerts(self, limit: int = 100) -> list:
        """
        Get recent alerts from the log file.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of alert dictionaries
        """
        try:
            alerts = []

            if not self.file_path.exists():
                return alerts

            # Read file in reverse to get most recent first
            with open(self.file_path, 'r') as f:
                lines = f.readlines()

            # Parse most recent lines
            for line in reversed(lines[-limit:]):
                try:
                    alert_dict = json.loads(line.strip())
                    alerts.append(alert_dict)
                except json.JSONDecodeError:
                    continue

            return alerts

        except Exception as e:
            logger.error(f"Failed to read recent alerts: {e}")
            return []

    def get_alerts_by_model(self, model_id: str, limit: int = 100) -> list:
        """
        Get alerts for a specific model.

        Args:
            model_id: Model identifier
            limit: Maximum number of alerts to return

        Returns:
            List of alert dictionaries for the specified model
        """
        all_alerts = self.get_recent_alerts(limit * 2)  # Get more to filter
        model_alerts = [a for a in all_alerts if a.get('model_id') == model_id]
        return model_alerts[:limit]

    def get_alerts_by_severity(self, severity: str, limit: int = 100) -> list:
        """
        Get alerts by severity level.

        Args:
            severity: Severity level (INFO, WARNING, CRITICAL, EMERGENCY)
            limit: Maximum number of alerts to return

        Returns:
            List of alert dictionaries with the specified severity
        """
        all_alerts = self.get_recent_alerts(limit * 2)
        severity_alerts = [a for a in all_alerts if a.get('severity') == severity]
        return severity_alerts[:limit]


# Example usage
if __name__ == "__main__":
    from .base_notifier import AlertType, AlertSeverity

    # Create file notifier
    notifier = FileNotifier(
        file_path="data/battle/alerts.log",
        enabled=True,
        max_file_size_mb=1,  # Small for testing
        backup_count=3
    )

    # Test alerts
    test_alerts = [
        Alert(
            alert_type=AlertType.MODEL_CRASH,
            severity=AlertSeverity.CRITICAL,
            model_id="gpt4o",
            message="Model crashed during decision cycle",
            metadata={'error_count': 3}
        ),
        Alert(
            alert_type=AlertType.CRITICAL_DRAWDOWN,
            severity=AlertSeverity.EMERGENCY,
            model_id="claude_sonnet",
            message="Drawdown exceeds critical threshold",
            metadata={'drawdown_pct': 22.5}
        ),
        Alert(
            alert_type=AlertType.LOW_WIN_RATE,
            severity=AlertSeverity.WARNING,
            model_id="grok",
            message="Win rate below threshold",
            metadata={'win_rate_pct': 35.0}
        )
    ]

    print("\n=== File Notifier Test ===")
    print(f"Writing alerts to: {notifier.file_path}\n")

    for alert in test_alerts:
        success = notifier.send_alert(alert)
        print(f"Alert sent: {success} - {alert}")

    # Read back recent alerts
    print("\n=== Recent Alerts ===")
    recent = notifier.get_recent_alerts(limit=5)
    for alert in recent:
        print(f"  {alert['severity']} | {alert['model_id']} | {alert['message']}")

    # Filter by model
    print("\n=== Alerts for gpt4o ===")
    gpt4_alerts = notifier.get_alerts_by_model("gpt4o", limit=5)
    for alert in gpt4_alerts:
        print(f"  {alert['severity']} | {alert['message']}")
