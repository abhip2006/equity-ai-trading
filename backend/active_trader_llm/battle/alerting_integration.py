"""
Alerting Integration: Integrates alerting system with battle orchestrator.

This module bridges the alerting/monitoring system with the battle orchestrator,
providing automatic health checks and alert generation during battle execution.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .alerting import AlertManager, AlertThresholds
from .health_monitor import HealthMonitor, ModelHealthReport
from .alerts_db import AlertsDatabase
from .notifications import (
    ConsoleNotifier,
    FileNotifier,
    Alert,
    AlertType,
    AlertSeverity
)

logger = logging.getLogger(__name__)


class BattleAlertingSystem:
    """
    Integrated alerting system for battle orchestrator.

    Provides:
    - Automatic health monitoring
    - Alert generation and distribution
    - Persistent alert storage
    - Model performance tracking
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize battle alerting system.

        Args:
            config: Alerting configuration from battle_config.yaml
        """
        self.config = config
        self.enabled = config.get('enabled', True)

        if not self.enabled:
            logger.info("Alerting system disabled by configuration")
            return

        # Initialize components
        self._init_thresholds()
        self._init_notifiers()
        self._init_alert_manager()
        self._init_health_monitor()
        self._init_database()

        logger.info("BattleAlertingSystem initialized")

    def _init_thresholds(self):
        """Initialize alert thresholds from config"""
        threshold_config = self.config.get('thresholds', {})

        self.thresholds = AlertThresholds(
            drawdown_warning_pct=threshold_config.get('drawdown_warning_pct', 10.0),
            drawdown_critical_pct=threshold_config.get('drawdown_critical_pct', 20.0),
            win_rate_warning_pct=threshold_config.get('win_rate_warning_pct', 40.0),
            error_rate_warning_pct=threshold_config.get('error_rate_warning_pct', 10.0),
            position_concentration_pct=threshold_config.get('position_concentration_pct', 40.0),
            unusual_trade_count=threshold_config.get('unusual_trade_count', 10),
            unusual_trade_window_minutes=threshold_config.get('unusual_trade_window_minutes', 5)
        )

        logger.info("Alert thresholds configured")

    def _init_notifiers(self):
        """Initialize notification channels from config"""
        self.notifiers = []

        channels = self.config.get('channels', ['console', 'file'])

        # Console notifier
        if 'console' in channels:
            console_config = self.config.get('console_notifier', {})
            notifier = ConsoleNotifier(
                enabled=True,
                use_colors=console_config.get('use_colors', True)
            )
            self.notifiers.append(notifier)
            logger.info("Console notifier enabled")

        # File notifier
        if 'file' in channels:
            file_config = self.config.get('file_notifier', {})
            notifier = FileNotifier(
                file_path=file_config.get('log_path', 'data/battle/alerts.log'),
                enabled=True,
                max_file_size_mb=file_config.get('max_file_size_mb', 10),
                backup_count=file_config.get('backup_count', 5)
            )
            self.notifiers.append(notifier)
            logger.info("File notifier enabled")

        # Future: Email, Webhook, Slack, etc.

    def _init_alert_manager(self):
        """Initialize alert manager"""
        dedup_window = self.config.get('dedup_window_minutes', 5)

        self.alert_manager = AlertManager(
            thresholds=self.thresholds,
            notifiers=self.notifiers,
            dedup_window_minutes=dedup_window
        )

        logger.info("Alert manager initialized")

    def _init_health_monitor(self):
        """Initialize health monitor"""
        health_config = self.config.get('health_check', {})

        self.health_monitor = HealthMonitor(
            max_response_time_seconds=health_config.get('max_response_time_seconds', 300.0),
            max_error_rate_pct=health_config.get('max_error_rate_pct', 10.0),
            max_consecutive_errors=health_config.get('max_consecutive_errors', 3),
            max_time_since_decision_minutes=health_config.get('max_time_since_decision_minutes', 30.0)
        )

        logger.info("Health monitor initialized")

    def _init_database(self):
        """Initialize alerts database"""
        db_path = self.config.get('database_path', 'data/battle/alerts.db')
        self.alerts_db = AlertsDatabase(db_path)
        logger.info(f"Alerts database initialized: {db_path}")

    def check_model_health(self, model_id: str, model_state: Dict[str, Any]) -> ModelHealthReport:
        """
        Check health of a model and generate alerts if needed.

        Args:
            model_id: Model identifier
            model_state: Current model state

        Returns:
            ModelHealthReport
        """
        if not self.enabled:
            return None

        # Get health report
        report = self.health_monitor.check_model_health(model_id, model_state)

        # Generate alerts based on health issues
        for issue in report.issues:
            # Determine alert type and severity based on issue
            if "error state" in issue.lower():
                self.alert_manager.alert_model_crash(model_id, issue)
            elif "high error rate" in issue.lower():
                self.alert_manager.check_error_rate(
                    model_id,
                    report.error_count,
                    report.cycles_completed
                )
            elif "slow response" in issue.lower():
                self.alert_manager.send_alert(
                    alert_type=AlertType.HIGH_ERROR_RATE,
                    severity=AlertSeverity.WARNING,
                    model_id=model_id,
                    message=f"Slow response time: {report.avg_response_time_seconds:.1f}s",
                    metadata={'response_time': report.avg_response_time_seconds}
                )

        return report

    def check_metrics(self, model_id: str, metrics: Dict[str, Any]):
        """
        Check performance metrics and generate alerts if needed.

        Args:
            model_id: Model identifier
            metrics: Performance metrics (from MetricsEngine)
        """
        if not self.enabled:
            return

        # Check drawdown
        max_drawdown_pct = metrics.get('max_drawdown_pct')
        equity = metrics.get('current_equity')
        if max_drawdown_pct is not None and equity is not None:
            self.alert_manager.check_drawdown(model_id, max_drawdown_pct, equity)

        # Check win rate
        win_rate = metrics.get('win_rate')
        total_trades = metrics.get('total_trades', 0)
        if win_rate is not None:
            self.alert_manager.check_win_rate(model_id, win_rate, total_trades)

        # Check position concentration (would need position-level data)
        # Future enhancement

    def check_all_models_health(
        self,
        model_states: Dict[str, Dict[str, Any]]
    ) -> Dict[str, ModelHealthReport]:
        """
        Check health of all models.

        Args:
            model_states: Dictionary mapping model_id to model state

        Returns:
            Dictionary mapping model_id to health report
        """
        if not self.enabled:
            return {}

        reports = {}

        for model_id, state in model_states.items():
            report = self.check_model_health(model_id, state)
            reports[model_id] = report

        # Get system-wide health
        system_health = self.health_monitor.get_system_health(list(reports.values()))

        # Log system health
        logger.info(
            f"System Health: {system_health['status']} "
            f"(Score: {system_health['overall_health_score']:.1f})"
        )

        # Alert on critical system health
        if system_health['status'] == 'critical':
            for issue in system_health['issues']:
                logger.critical(f"SYSTEM HEALTH: {issue}")

        return reports

    def store_alert_in_db(self, alert: Alert):
        """Store alert in database for historical analysis"""
        if not self.enabled:
            return

        try:
            self.alerts_db.store_alert(alert)
        except Exception as e:
            logger.error(f"Failed to store alert in database: {e}")

    def get_recent_alerts(
        self,
        limit: int = 100,
        model_id: Optional[str] = None,
        severity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent alerts from database.

        Args:
            limit: Maximum number of alerts
            model_id: Filter by model
            severity: Filter by severity

        Returns:
            List of alert dictionaries
        """
        if not self.enabled:
            return []

        return self.alerts_db.get_recent_alerts(
            limit=limit,
            model_id=model_id,
            severity=severity
        )

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get comprehensive alert statistics"""
        if not self.enabled:
            return {}

        return self.alerts_db.get_statistics()

    def cleanup(self):
        """Cleanup old alerts and resources"""
        if not self.enabled:
            return

        self.alert_manager.cleanup_old_alerts(max_age_hours=24)
        logger.info("Alerting system cleanup complete")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example config
    config = {
        'enabled': True,
        'channels': ['console', 'file'],
        'database_path': 'data/battle/alerts.db',
        'file_notifier': {
            'log_path': 'data/battle/alerts.log',
            'max_file_size_mb': 10,
            'backup_count': 5
        },
        'console_notifier': {
            'use_colors': True
        },
        'thresholds': {
            'drawdown_warning_pct': 10.0,
            'drawdown_critical_pct': 20.0,
            'win_rate_warning_pct': 40.0,
            'error_rate_warning_pct': 10.0
        },
        'dedup_window_minutes': 5,
        'health_check': {
            'enabled': True,
            'max_response_time_seconds': 300.0,
            'max_consecutive_errors': 3
        }
    }

    # Initialize system
    print("\n=== Testing Battle Alerting System ===\n")
    alerting = BattleAlertingSystem(config)

    # Test model health check
    print("Testing model health check...")
    model_state = {
        'status': 'error',
        'cycles_completed': 10,
        'error_count': 8,
        'last_error': 'Connection timeout',
        'last_cycle_time': datetime.now().isoformat()
    }

    report = alerting.check_model_health("gpt4o", model_state)
    print(f"  Health Status: {report.status.value}")
    print(f"  Health Score: {report.health_score:.1f}")
    print(f"  Issues: {report.issues}")

    # Test metrics check
    print("\nTesting metrics check...")
    metrics = {
        'max_drawdown_pct': 22.5,
        'current_equity': 77500,
        'win_rate': 35.0,
        'total_trades': 20
    }

    alerting.check_metrics("claude_sonnet", metrics)

    # Get statistics
    print("\nGetting alert statistics...")
    stats = alerting.get_alert_statistics()
    print(f"  Total alerts: {stats.get('total_alerts', 0)}")
    print(f"  By severity: {stats.get('by_severity', {})}")

    print("\n=== Test Complete ===")
