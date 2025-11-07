"""
Health Monitor: Monitors model health and trading system status.

Tracks:
- Model response times
- Error rates
- Trading activity
- System resource usage
- Decision cycle health
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class ModelHealthReport:
    """Health report for a single model"""
    model_id: str
    status: HealthStatus
    timestamp: datetime

    # Response metrics
    is_responding: bool
    last_response_time: Optional[datetime]
    avg_response_time_seconds: Optional[float]

    # Error metrics
    error_count: int
    error_rate_pct: float
    last_error: Optional[str]
    consecutive_errors: int

    # Activity metrics
    cycles_completed: int
    last_decision_time: Optional[datetime]
    time_since_last_decision_seconds: Optional[float]

    # Health indicators
    health_score: float  # 0-100
    issues: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return {
            'model_id': self.model_id,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'is_responding': self.is_responding,
            'last_response_time': self.last_response_time.isoformat() if self.last_response_time else None,
            'avg_response_time_seconds': self.avg_response_time_seconds,
            'error_count': self.error_count,
            'error_rate_pct': self.error_rate_pct,
            'last_error': self.last_error,
            'consecutive_errors': self.consecutive_errors,
            'cycles_completed': self.cycles_completed,
            'last_decision_time': self.last_decision_time.isoformat() if self.last_decision_time else None,
            'time_since_last_decision_seconds': self.time_since_last_decision_seconds,
            'health_score': self.health_score,
            'issues': self.issues,
            'warnings': self.warnings
        }


class HealthMonitor:
    """
    Monitors health of all trading models.

    Performs periodic health checks:
    - Response time monitoring
    - Error rate tracking
    - Activity monitoring
    - Anomaly detection
    """

    def __init__(
        self,
        max_response_time_seconds: float = 300.0,  # 5 minutes
        max_error_rate_pct: float = 10.0,
        max_consecutive_errors: int = 3,
        max_time_since_decision_minutes: float = 30.0
    ):
        """
        Initialize health monitor.

        Args:
            max_response_time_seconds: Maximum acceptable response time
            max_error_rate_pct: Maximum acceptable error rate
            max_consecutive_errors: Maximum consecutive errors before critical
            max_time_since_decision_minutes: Maximum time since last decision
        """
        self.max_response_time = max_response_time_seconds
        self.max_error_rate = max_error_rate_pct
        self.max_consecutive_errors = max_consecutive_errors
        self.max_time_since_decision = timedelta(minutes=max_time_since_decision_minutes)

        # Health tracking
        self.model_health_history: Dict[str, List[ModelHealthReport]] = {}

        logger.info("HealthMonitor initialized")
        logger.info(f"  Max response time: {max_response_time_seconds}s")
        logger.info(f"  Max error rate: {max_error_rate_pct}%")
        logger.info(f"  Max consecutive errors: {max_consecutive_errors}")

    def check_model_health(
        self,
        model_id: str,
        model_state: Dict[str, Any],
        response_times: Optional[List[float]] = None
    ) -> ModelHealthReport:
        """
        Check health of a specific model.

        Args:
            model_id: Model identifier
            model_state: Current model state from ModelInstance
            response_times: Recent response times (in seconds)

        Returns:
            ModelHealthReport with comprehensive health assessment
        """
        now = datetime.now()

        # Extract state information
        status = model_state.get('status', 'unknown')
        cycles_completed = model_state.get('cycles_completed', 0)
        error_count = model_state.get('error_count', 0)
        last_error = model_state.get('last_error')
        last_cycle_time_str = model_state.get('last_cycle_time')

        # Parse last cycle time
        last_decision_time = None
        time_since_last_decision = None
        if last_cycle_time_str:
            try:
                last_decision_time = datetime.fromisoformat(last_cycle_time_str)
                time_since_last_decision = (now - last_decision_time).total_seconds()
            except (ValueError, TypeError):
                pass

        # Calculate response metrics
        avg_response_time = None
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)

        # Calculate error rate
        error_rate_pct = 0.0
        if cycles_completed > 0:
            error_rate_pct = (error_count / cycles_completed) * 100

        # Determine if model is responding
        is_responding = (
            status in ['running', 'initialized'] and
            error_count < self.max_consecutive_errors
        )

        # Get consecutive errors (simplified - would need history for accurate count)
        consecutive_errors = error_count if status == 'error' else 0

        # Perform health checks
        issues = []
        warnings = []
        health_score = 100.0

        # Check 1: Model status
        if status == 'error':
            issues.append(f"Model in error state: {last_error}")
            health_score -= 30
        elif status == 'shutdown':
            issues.append("Model is shut down")
            health_score = 0

        # Check 2: Error rate
        if error_rate_pct >= self.max_error_rate:
            issues.append(f"High error rate: {error_rate_pct:.1f}%")
            health_score -= 20
        elif error_rate_pct >= self.max_error_rate * 0.5:
            warnings.append(f"Elevated error rate: {error_rate_pct:.1f}%")
            health_score -= 10

        # Check 3: Consecutive errors
        if consecutive_errors >= self.max_consecutive_errors:
            issues.append(f"Too many consecutive errors: {consecutive_errors}")
            health_score -= 25

        # Check 4: Response time
        if avg_response_time and avg_response_time >= self.max_response_time:
            issues.append(f"Slow response time: {avg_response_time:.1f}s")
            health_score -= 15
        elif avg_response_time and avg_response_time >= self.max_response_time * 0.7:
            warnings.append(f"Elevated response time: {avg_response_time:.1f}s")
            health_score -= 5

        # Check 5: Time since last decision
        if time_since_last_decision:
            max_seconds = self.max_time_since_decision.total_seconds()
            if time_since_last_decision >= max_seconds:
                issues.append(f"No decision in {time_since_last_decision/60:.1f} minutes")
                health_score -= 20
            elif time_since_last_decision >= max_seconds * 0.7:
                warnings.append(f"No recent decision ({time_since_last_decision/60:.1f} minutes)")
                health_score -= 10

        # Check 6: Activity level
        if cycles_completed == 0:
            warnings.append("No cycles completed yet")
            health_score -= 5

        # Determine overall health status
        if health_score >= 80:
            overall_status = HealthStatus.HEALTHY
        elif health_score >= 60:
            overall_status = HealthStatus.DEGRADED
        elif health_score >= 30:
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.CRITICAL

        # Create health report
        report = ModelHealthReport(
            model_id=model_id,
            status=overall_status,
            timestamp=now,
            is_responding=is_responding,
            last_response_time=last_decision_time,
            avg_response_time_seconds=avg_response_time,
            error_count=error_count,
            error_rate_pct=error_rate_pct,
            last_error=last_error,
            consecutive_errors=consecutive_errors,
            cycles_completed=cycles_completed,
            last_decision_time=last_decision_time,
            time_since_last_decision_seconds=time_since_last_decision,
            health_score=health_score,
            issues=issues,
            warnings=warnings
        )

        # Store in history
        if model_id not in self.model_health_history:
            self.model_health_history[model_id] = []
        self.model_health_history[model_id].append(report)

        # Keep only recent history (last 100 reports)
        if len(self.model_health_history[model_id]) > 100:
            self.model_health_history[model_id] = self.model_health_history[model_id][-100:]

        return report

    def get_system_health(self, model_reports: List[ModelHealthReport]) -> Dict[str, Any]:
        """
        Get overall system health based on all models.

        Args:
            model_reports: List of model health reports

        Returns:
            System health summary
        """
        if not model_reports:
            return {
                'status': HealthStatus.CRITICAL.value,
                'healthy_count': 0,
                'degraded_count': 0,
                'unhealthy_count': 0,
                'critical_count': 0,
                'overall_health_score': 0.0,
                'issues': ['No models reporting']
            }

        # Count models by status
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.CRITICAL: 0
        }

        for report in model_reports:
            status_counts[report.status] += 1

        # Calculate overall health score
        total_score = sum(report.health_score for report in model_reports)
        avg_score = total_score / len(model_reports)

        # Determine system status
        critical_count = status_counts[HealthStatus.CRITICAL]
        unhealthy_count = status_counts[HealthStatus.UNHEALTHY]
        degraded_count = status_counts[HealthStatus.DEGRADED]

        if critical_count > len(model_reports) / 2:
            system_status = HealthStatus.CRITICAL
        elif critical_count + unhealthy_count > len(model_reports) / 2:
            system_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0 or critical_count + unhealthy_count > 0:
            system_status = HealthStatus.DEGRADED
        else:
            system_status = HealthStatus.HEALTHY

        # Collect all issues
        all_issues = []
        for report in model_reports:
            for issue in report.issues:
                all_issues.append(f"{report.model_id}: {issue}")

        return {
            'status': system_status.value,
            'healthy_count': status_counts[HealthStatus.HEALTHY],
            'degraded_count': status_counts[HealthStatus.DEGRADED],
            'unhealthy_count': status_counts[HealthStatus.UNHEALTHY],
            'critical_count': status_counts[HealthStatus.CRITICAL],
            'overall_health_score': avg_score,
            'total_models': len(model_reports),
            'issues': all_issues
        }

    def get_health_history(self, model_id: str, limit: int = 10) -> List[ModelHealthReport]:
        """
        Get recent health history for a model.

        Args:
            model_id: Model identifier
            limit: Maximum number of reports to return

        Returns:
            List of recent health reports
        """
        history = self.model_health_history.get(model_id, [])
        return history[-limit:]


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create health monitor
    monitor = HealthMonitor(
        max_response_time_seconds=300.0,
        max_error_rate_pct=10.0,
        max_consecutive_errors=3
    )

    print("\n=== Testing Health Monitor ===\n")

    # Test healthy model
    print("1. Checking healthy model...")
    healthy_state = {
        'status': 'running',
        'cycles_completed': 50,
        'error_count': 2,
        'last_error': None,
        'last_cycle_time': datetime.now().isoformat()
    }
    report1 = monitor.check_model_health("gpt4o", healthy_state, response_times=[45.2, 52.1, 48.9])
    print(f"   Status: {report1.status.value}")
    print(f"   Health Score: {report1.health_score:.1f}")
    print(f"   Issues: {report1.issues}")
    print(f"   Warnings: {report1.warnings}")

    # Test degraded model
    print("\n2. Checking degraded model...")
    degraded_state = {
        'status': 'running',
        'cycles_completed': 30,
        'error_count': 5,
        'last_error': 'API rate limit',
        'last_cycle_time': (datetime.now() - timedelta(minutes=15)).isoformat()
    }
    report2 = monitor.check_model_health("claude_sonnet", degraded_state, response_times=[180.5, 220.3])
    print(f"   Status: {report2.status.value}")
    print(f"   Health Score: {report2.health_score:.1f}")
    print(f"   Issues: {report2.issues}")
    print(f"   Warnings: {report2.warnings}")

    # Test unhealthy model
    print("\n3. Checking unhealthy model...")
    unhealthy_state = {
        'status': 'error',
        'cycles_completed': 10,
        'error_count': 8,
        'last_error': 'Connection timeout',
        'last_cycle_time': (datetime.now() - timedelta(minutes=45)).isoformat()
    }
    report3 = monitor.check_model_health("grok", unhealthy_state, response_times=[350.0, 420.0])
    print(f"   Status: {report3.status.value}")
    print(f"   Health Score: {report3.health_score:.1f}")
    print(f"   Issues: {report3.issues}")
    print(f"   Warnings: {report3.warnings}")

    # Get system health
    print("\n4. Checking overall system health...")
    system_health = monitor.get_system_health([report1, report2, report3])
    print(f"   System Status: {system_health['status']}")
    print(f"   Overall Health Score: {system_health['overall_health_score']:.1f}")
    print(f"   Healthy: {system_health['healthy_count']}")
    print(f"   Degraded: {system_health['degraded_count']}")
    print(f"   Unhealthy: {system_health['unhealthy_count']}")
    print(f"   Critical: {system_health['critical_count']}")
    print(f"   Issues: {system_health['issues']}")

    print("\n=== Health Monitor Test Complete ===")
