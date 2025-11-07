"""
Alerts Database: Persistent storage for battle alerts.

Stores alerts in SQLite database with querying capabilities.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

from .notifications import Alert

logger = logging.getLogger(__name__)


class AlertsDatabase:
    """
    Database manager for battle alerts.

    Handles:
    - Persistent alert storage
    - Alert querying and filtering
    - Alert acknowledgment and resolution
    - Historical analysis
    """

    def __init__(self, db_path: str):
        """
        Initialize alerts database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_database()
        logger.info(f"AlertsDatabase initialized: {db_path}")

    def _ensure_database(self):
        """Create database and tables if they don't exist"""
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Read schema if available
        schema_path = Path(__file__).parent / 'schema' / 'alerts_schema.sql'

        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            cursor.executescript(schema_sql)
            logger.info("Created alerts database from schema file")
        else:
            # Fallback inline schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metadata TEXT,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    acknowledged BOOLEAN DEFAULT 0,
                    resolved BOOLEAN DEFAULT 0
                )
            """)
            logger.info("Created alerts database with inline schema")

        conn.commit()
        conn.close()

    def store_alert(self, alert: Alert) -> int:
        """
        Store an alert in the database.

        Args:
            alert: Alert object to store

        Returns:
            Alert ID (database row ID)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Convert metadata to JSON
            metadata_json = json.dumps(alert.metadata) if alert.metadata else None

            cursor.execute("""
                INSERT INTO alerts (alert_type, severity, model_id, message, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_type.value,
                alert.severity.value,
                alert.model_id,
                alert.message,
                metadata_json,
                alert.timestamp
            ))

            alert_id = cursor.lastrowid
            conn.commit()

            logger.debug(f"Stored alert {alert_id}: {alert.alert_type.value} for {alert.model_id}")
            return alert_id

        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_recent_alerts(
        self,
        limit: int = 100,
        model_id: Optional[str] = None,
        severity: Optional[str] = None,
        alert_type: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent alerts with optional filtering.

        Args:
            limit: Maximum number of alerts to return
            model_id: Filter by model ID
            severity: Filter by severity level
            alert_type: Filter by alert type
            since: Only return alerts after this time

        Returns:
            List of alert dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM alerts WHERE 1=1"
        params = []

        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)

        if severity:
            query += " AND severity = ?"
            params.append(severity)

        if alert_type:
            query += " AND alert_type = ?"
            params.append(alert_type)

        if since:
            query += " AND timestamp >= ?"
            params.append(since)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        # Convert rows to dictionaries
        alerts = []
        for row in rows:
            alert_dict = dict(row)
            # Parse metadata JSON
            if alert_dict['metadata']:
                try:
                    alert_dict['metadata'] = json.loads(alert_dict['metadata'])
                except json.JSONDecodeError:
                    alert_dict['metadata'] = {}
            return alerts

        return alerts

    def get_alert_counts_by_model(self, since: Optional[datetime] = None) -> Dict[str, int]:
        """
        Get alert counts grouped by model.

        Args:
            since: Only count alerts after this time

        Returns:
            Dictionary mapping model_id to alert count
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT model_id, COUNT(*) as count FROM alerts"
        params = []

        if since:
            query += " WHERE timestamp >= ?"
            params.append(since)

        query += " GROUP BY model_id"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return {row[0]: row[1] for row in rows}

    def get_alert_counts_by_severity(self, since: Optional[datetime] = None) -> Dict[str, int]:
        """
        Get alert counts grouped by severity.

        Args:
            since: Only count alerts after this time

        Returns:
            Dictionary mapping severity to alert count
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT severity, COUNT(*) as count FROM alerts"
        params = []

        if since:
            query += " WHERE timestamp >= ?"
            params.append(since)

        query += " GROUP BY severity"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return {row[0]: row[1] for row in rows}

    def acknowledge_alert(self, alert_id: int, acknowledged_by: str = "system"):
        """
        Mark an alert as acknowledged.

        Args:
            alert_id: Alert ID to acknowledge
            acknowledged_by: Who acknowledged the alert
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE alerts
            SET acknowledged = 1,
                acknowledged_at = ?,
                acknowledged_by = ?
            WHERE id = ?
        """, (datetime.now(), acknowledged_by, alert_id))

        conn.commit()
        conn.close()

        logger.info(f"Acknowledged alert {alert_id}")

    def resolve_alert(self, alert_id: int, resolution_notes: Optional[str] = None):
        """
        Mark an alert as resolved.

        Args:
            alert_id: Alert ID to resolve
            resolution_notes: Optional notes about resolution
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE alerts
            SET resolved = 1,
                resolved_at = ?,
                resolution_notes = ?
            WHERE id = ?
        """, (datetime.now(), resolution_notes, alert_id))

        conn.commit()
        conn.close()

        logger.info(f"Resolved alert {alert_id}")

    def get_statistics(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get comprehensive alert statistics.

        Args:
            since: Only include alerts after this time

        Returns:
            Dictionary with alert statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build time filter
        time_filter = ""
        params = []
        if since:
            time_filter = "WHERE timestamp >= ?"
            params.append(since)

        # Total counts
        cursor.execute(f"SELECT COUNT(*) FROM alerts {time_filter}", params)
        total_alerts = cursor.fetchone()[0]

        # Counts by severity
        cursor.execute(f"SELECT severity, COUNT(*) FROM alerts {time_filter} GROUP BY severity", params)
        severity_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # Counts by type
        cursor.execute(f"SELECT alert_type, COUNT(*) FROM alerts {time_filter} GROUP BY alert_type", params)
        type_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # Counts by model
        cursor.execute(f"SELECT model_id, COUNT(*) FROM alerts {time_filter} GROUP BY model_id", params)
        model_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # Acknowledgment stats
        ack_filter = time_filter if time_filter else "WHERE"
        ack_filter = f"{ack_filter} {'AND' if time_filter else ''} acknowledged = 1"
        cursor.execute(f"SELECT COUNT(*) FROM alerts {ack_filter}", params)
        acknowledged_count = cursor.fetchone()[0]

        # Resolution stats
        res_filter = time_filter if time_filter else "WHERE"
        res_filter = f"{res_filter} {'AND' if time_filter else ''} resolved = 1"
        cursor.execute(f"SELECT COUNT(*) FROM alerts {res_filter}", params)
        resolved_count = cursor.fetchone()[0]

        conn.close()

        return {
            'total_alerts': total_alerts,
            'acknowledged': acknowledged_count,
            'resolved': resolved_count,
            'by_severity': severity_counts,
            'by_type': type_counts,
            'by_model': model_counts
        }


# Example usage
if __name__ == "__main__":
    from .notifications import AlertType, AlertSeverity

    logging.basicConfig(level=logging.INFO)

    # Create database
    db = AlertsDatabase("data/battle/alerts.db")

    # Create test alerts
    test_alerts = [
        Alert(
            alert_type=AlertType.MODEL_CRASH,
            severity=AlertSeverity.CRITICAL,
            model_id="gpt4o",
            message="Model crashed",
            metadata={'error': 'timeout'}
        ),
        Alert(
            alert_type=AlertType.CRITICAL_DRAWDOWN,
            severity=AlertSeverity.EMERGENCY,
            model_id="claude_sonnet",
            message="Critical drawdown",
            metadata={'drawdown_pct': 22.5}
        ),
        Alert(
            alert_type=AlertType.LOW_WIN_RATE,
            severity=AlertSeverity.WARNING,
            model_id="grok",
            message="Low win rate",
            metadata={'win_rate_pct': 35.0}
        )
    ]

    print("\n=== Storing Alerts ===")
    for alert in test_alerts:
        alert_id = db.store_alert(alert)
        print(f"Stored alert {alert_id}: {alert.message}")

    # Get statistics
    print("\n=== Alert Statistics ===")
    stats = db.get_statistics()
    print(f"Total: {stats['total_alerts']}")
    print(f"By severity: {stats['by_severity']}")
    print(f"By type: {stats['by_type']}")
    print(f"By model: {stats['by_model']}")

    print("\n=== Test Complete ===")
