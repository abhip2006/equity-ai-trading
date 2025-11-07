# Battle Alerting System - Technical Architecture

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BATTLE ORCHESTRATOR                                  │
│                                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  GPT-4   │  │ Claude   │  │  Grok    │  │ DeepSeek │  │  Gemini  │    │
│  │  Turbo   │  │ Sonnet   │  │    2     │  │    V3    │  │ 2.0 Flash│    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │             │             │             │             │            │
│       └─────────────┴─────────────┴─────────────┴─────────────┘            │
│                                   │                                         │
└───────────────────────────────────┼─────────────────────────────────────────┘
                                    │
                                    │ Model States & Metrics
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      BATTLE ALERTING SYSTEM                                  │
│                   (BattleAlertingSystem)                                     │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                       HEALTH MONITOR                                │    │
│  │                     (HealthMonitor)                                 │    │
│  │                                                                     │    │
│  │  • Response time tracking                                          │    │
│  │  • Error rate monitoring                                           │    │
│  │  • Activity monitoring                                             │    │
│  │  • Health score calculation (0-100)                                │    │
│  │  • System-wide health aggregation                                  │    │
│  │                                                                     │    │
│  │  Output: ModelHealthReport                                         │    │
│  └────────────────────────┬───────────────────────────────────────────┘    │
│                           │                                                 │
│                           ▼                                                 │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                       ALERT MANAGER                                 │    │
│  │                      (AlertManager)                                 │    │
│  │                                                                     │    │
│  │  ┌──────────────────────────────────────────────────────────┐     │    │
│  │  │              THRESHOLD CHECKING                           │     │    │
│  │  │                                                           │     │    │
│  │  │  • Drawdown (warning: 10%, critical: 20%)                │     │    │
│  │  │  • Win rate (minimum: 40%)                               │     │    │
│  │  │  • Error rate (maximum: 10%)                             │     │    │
│  │  │  • Position concentration (maximum: 40%)                 │     │    │
│  │  │  • Unusual trading activity (>10 trades/5min)            │     │    │
│  │  └──────────────────────────────────────────────────────────┘     │    │
│  │                           │                                        │    │
│  │                           ▼                                        │    │
│  │  ┌──────────────────────────────────────────────────────────┐     │    │
│  │  │              ALERT GENERATION                             │     │    │
│  │  │                                                           │     │    │
│  │  │  Alert Type + Severity + Message + Metadata              │     │    │
│  │  └──────────────────────────────────────────────────────────┘     │    │
│  │                           │                                        │    │
│  │                           ▼                                        │    │
│  │  ┌──────────────────────────────────────────────────────────┐     │    │
│  │  │              DEDUPLICATION                                │     │    │
│  │  │                                                           │     │    │
│  │  │  • Check if alert sent recently (5 min window)           │     │    │
│  │  │  • In-memory cache of recent alerts                      │     │    │
│  │  │  • Prevent alert spam                                    │     │    │
│  │  └──────────────────────────────────────────────────────────┘     │    │
│  │                                                                     │    │
│  │  Output: Alert object                                              │    │
│  └────────────────────────┬───────────────────────────────────────────┘    │
│                           │                                                 │
│                           ▼                                                 │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                   NOTIFICATION CHANNELS                             │    │
│  │                                                                     │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐         │    │
│  │  │   Console     │  │     File      │  │    Future     │         │    │
│  │  │   Notifier    │  │   Notifier    │  │  (Email, SMS) │         │    │
│  │  │               │  │               │  │               │         │    │
│  │  │ • Color-coded │  │ • JSONL log   │  │ • SMTP        │         │    │
│  │  │ • Real-time   │  │ • Rotation    │  │ • Twilio      │         │    │
│  │  │ • Formatted   │  │ • Queryable   │  │ • Webhook     │         │    │
│  │  └───────────────┘  └───────────────┘  └───────────────┘         │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                      ALERTS DATABASE                                │    │
│  │                     (AlertsDatabase)                                │    │
│  │                                                                     │    │
│  │  SQLite Database: data/battle/alerts.db                            │    │
│  │                                                                     │    │
│  │  • Persistent storage                                              │    │
│  │  • Historical queries                                              │    │
│  │  • Statistical analysis                                            │    │
│  │  • Alert acknowledgment                                            │    │
│  │  • Alert resolution tracking                                       │    │
│  │                                                                     │    │
│  │  Views:                                                            │    │
│  │  • alert_summary_by_model                                          │    │
│  │  • alert_summary_by_type                                           │    │
│  │  • recent_critical_alerts                                          │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### 1. BattleAlertingSystem (Integration Layer)

**Purpose:** Orchestrates all alerting components and integrates with battle system

**Responsibilities:**
- Initialize all components from configuration
- Coordinate health checks and metric checks
- Provide unified API for battle orchestrator
- Manage component lifecycle

**Key Methods:**
```python
check_model_health(model_id, model_state) -> ModelHealthReport
check_metrics(model_id, metrics)
check_all_models_health(model_states) -> Dict[str, ModelHealthReport]
get_recent_alerts(limit, model_id, severity) -> List[Alert]
get_alert_statistics() -> Dict[str, Any]
```

### 2. HealthMonitor (Health Assessment)

**Purpose:** Evaluate model health and generate health reports

**Responsibilities:**
- Track response times
- Monitor error rates
- Detect stale models (no recent activity)
- Calculate health scores (0-100)
- Identify health issues and warnings
- Aggregate system-wide health

**Health Scoring Algorithm:**
```python
Base Score: 100 points

Deductions:
  - Model in error state: -30
  - High error rate (≥10%): -20
  - Elevated error rate (≥5%): -10
  - Too many consecutive errors (≥3): -25
  - Slow response time (≥300s): -15
  - Elevated response time (≥210s): -5
  - No recent decision (≥30min): -20
  - Delayed decision (≥21min): -10
  - No cycles completed: -5

Final Status:
  100-80: HEALTHY
  79-60: DEGRADED
  59-30: UNHEALTHY
  <30: CRITICAL
```

### 3. AlertManager (Alert Coordination)

**Purpose:** Central alert generation and distribution

**Responsibilities:**
- Check thresholds against current values
- Generate alerts when thresholds exceeded
- Deduplicate alerts within time window
- Distribute alerts to notification channels
- Store alerts in database

**Threshold Checks:**
```python
check_drawdown(model_id, drawdown_pct, equity)
  → If ≥20%: CRITICAL_DRAWDOWN (EMERGENCY)
  → If ≥10%: WARNING_DRAWDOWN (WARNING)

check_error_rate(model_id, error_count, total_cycles)
  → If ≥10%: HIGH_ERROR_RATE (WARNING)

check_win_rate(model_id, win_rate_pct, total_trades)
  → If <40% (and ≥10 trades): LOW_WIN_RATE (WARNING)

check_position_concentration(model_id, position_value, equity, symbol)
  → If ≥40%: POSITION_CONCENTRATION (WARNING)
```

**Special Alerts:**
```python
alert_model_crash(model_id, error_message)
  → MODEL_CRASH (CRITICAL)

alert_circuit_breaker(model_id, reason)
  → CIRCUIT_BREAKER (EMERGENCY)

alert_model_recovery(model_id, downtime_seconds)
  → MODEL_RECOVERY (INFO)
```

### 4. Notification Channels (Alert Delivery)

#### ConsoleNotifier
- Real-time console output
- ANSI color coding by severity
- Formatted metadata display
- Logging integration

#### FileNotifier
- JSONL format (one JSON per line)
- Automatic log rotation
- Configurable file size limits
- Queryable alert history

#### BaseNotifier (Abstract)
- Common interface for all notifiers
- Enable/disable functionality
- Future: Email, SMS, Webhook, Slack

### 5. AlertsDatabase (Persistence Layer)

**Purpose:** Store and query alerts for historical analysis

**Database Schema:**
```sql
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY,
    alert_type TEXT,        -- model_crash, critical_drawdown, etc.
    severity TEXT,          -- INFO, WARNING, CRITICAL, EMERGENCY
    model_id TEXT,
    message TEXT,
    metadata TEXT,          -- JSON blob
    timestamp TIMESTAMP,
    acknowledged BOOLEAN,
    acknowledged_at TIMESTAMP,
    acknowledged_by TEXT,
    resolved BOOLEAN,
    resolved_at TIMESTAMP,
    resolution_notes TEXT
);
```

**Indexes:**
- (model_id, timestamp DESC)
- (alert_type, timestamp DESC)
- (severity, timestamp DESC)
- (resolved, timestamp DESC)

**Views:**
- `alert_summary_by_model` - Count and severity breakdown per model
- `alert_summary_by_type` - Count and affected models per alert type
- `recent_critical_alerts` - Last 24 hours of critical/emergency alerts

## Data Flow

### Alert Generation Flow

```
1. Model State Update
   ↓
2. HealthMonitor.check_model_health()
   ↓
3. Generate ModelHealthReport
   ↓
4. Check health issues
   ↓
5. AlertManager.send_alert()
   ↓
6. Deduplication check
   ↓
7. Create Alert object
   ↓
8. Send to all enabled notifiers
   ├→ ConsoleNotifier.send_alert() → stdout
   ├→ FileNotifier.send_alert() → JSONL file
   └→ Future notifiers
   ↓
9. AlertsDatabase.store_alert() → SQLite
   ↓
10. Update deduplication cache
```

### Metrics Check Flow

```
1. Metrics Update (from MetricsEngine)
   ↓
2. AlertManager.check_metrics()
   ↓
3. Check drawdown threshold
   ├→ If ≥20%: Critical alert
   └→ If ≥10%: Warning alert
   ↓
4. Check win rate threshold
   └→ If <40%: Warning alert
   ↓
5. Check position concentration
   └→ If ≥40%: Warning alert
   ↓
6. Alerts generated via send_alert()
```

### System Health Flow

```
1. Collect all model states
   ↓
2. For each model:
   └→ HealthMonitor.check_model_health()
   ↓
3. Aggregate all health reports
   ↓
4. HealthMonitor.get_system_health()
   ↓
5. Calculate system status:
   ├→ >50% critical → System CRITICAL
   ├→ >50% unhealthy/critical → System UNHEALTHY
   ├→ Any degraded/unhealthy → System DEGRADED
   └→ All healthy → System HEALTHY
   ↓
6. Return system health summary
```

## Configuration Schema

```yaml
alerting:
  # Master switch
  enabled: true

  # Active channels
  channels:
    - console
    - file
    # Future: email, slack, webhook, sms

  # Database location
  database_path: "data/battle/alerts.db"

  # File notifier config
  file_notifier:
    log_path: "data/battle/alerts.log"
    max_file_size_mb: 10
    backup_count: 5

  # Console notifier config
  console_notifier:
    use_colors: true

  # Alert thresholds
  thresholds:
    drawdown_warning_pct: 10.0
    drawdown_critical_pct: 20.0
    win_rate_warning_pct: 40.0
    error_rate_warning_pct: 10.0
    position_concentration_pct: 40.0
    unusual_trade_count: 10
    unusual_trade_window_minutes: 5

  # Deduplication
  dedup_window_minutes: 5

  # Health monitoring
  health_check:
    enabled: true
    check_interval_seconds: 60
    max_response_time_seconds: 300
    max_consecutive_errors: 3
    max_time_since_decision_minutes: 30
```

## Alert Types & Severities

| Alert Type | Severity | Trigger Condition | Force Send |
|-----------|----------|-------------------|------------|
| MODEL_CRASH | CRITICAL | Model enters error state | Yes |
| HIGH_ERROR_RATE | WARNING | Error rate ≥ 10% | No |
| MODEL_RECOVERY | INFO | Model recovers from error | No |
| CRITICAL_DRAWDOWN | EMERGENCY | Drawdown ≥ 20% | No |
| WARNING_DRAWDOWN | WARNING | Drawdown ≥ 10% | No |
| LOW_WIN_RATE | WARNING | Win rate < 40% (≥10 trades) | No |
| POSITION_CONCENTRATION | WARNING | Single position ≥ 40% equity | No |
| CIRCUIT_BREAKER | EMERGENCY | Emergency shutdown | Yes |
| UNUSUAL_TRADING | WARNING | >10 trades in 5 minutes | No |

**Force Send:** Bypasses deduplication (always sent)

## Performance Characteristics

### Timing (per cycle, 6 models)

- Health check: ~60ms (10ms per model)
- Alert generation: <30ms (5ms per alert)
- Database write: <60ms (10ms per alert)
- File write: <30ms (5ms per alert)
- **Total overhead: <200ms**

### Memory Usage

- AlertManager cache: <1 KB
- Recent alerts tracking: ~5 KB
- Health report history: ~50 KB (100 reports × 6 models)
- **Total memory: <100 KB**

### Database Growth

- Average alert size: ~500 bytes
- 100 alerts/hour: ~50 KB/hour
- 1000 alerts/day: ~500 KB/day
- **30 days: ~15 MB**

## Error Handling

### Component Failures

1. **Notifier failure:**
   - Log error
   - Continue with other notifiers
   - Don't block alert generation

2. **Database failure:**
   - Log error
   - Alerts still sent to notifiers
   - Retry on next alert

3. **Health check failure:**
   - Log error
   - Return degraded health status
   - Continue monitoring

### Graceful Degradation

- If alerting disabled: System continues without alerts
- If database fails: Alerts still shown in console/file
- If notifier fails: Other notifiers still work
- If threshold config missing: Use defaults

## Extension Points

### Adding New Alert Types

```python
# 1. Add to AlertType enum (base_notifier.py)
class AlertType(Enum):
    # ... existing types
    NEW_ALERT_TYPE = "new_alert_type"

# 2. Add checking method (alerting.py)
def check_new_condition(self, model_id, value):
    if value > threshold:
        self.send_alert(
            alert_type=AlertType.NEW_ALERT_TYPE,
            severity=AlertSeverity.WARNING,
            model_id=model_id,
            message=f"New condition triggered: {value}",
            metadata={'value': value}
        )

# 3. Call from integration (alerting_integration.py)
alerting.alert_manager.check_new_condition(model_id, value)
```

### Adding New Notifiers

```python
# 1. Create notifier class (notifications/email_notifier.py)
from .base_notifier import BaseNotifier, Alert

class EmailNotifier(BaseNotifier):
    def __init__(self, smtp_host, smtp_port, recipients):
        super().__init__(enabled=True)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.recipients = recipients

    def send_alert(self, alert: Alert) -> bool:
        # Send email via SMTP
        pass

# 2. Register in integration (alerting_integration.py)
if 'email' in channels:
    email_config = self.config.get('email_notifier', {})
    notifier = EmailNotifier(
        smtp_host=email_config['smtp_host'],
        smtp_port=email_config['smtp_port'],
        recipients=email_config['recipients']
    )
    self.notifiers.append(notifier)
```

## Testing Strategy

### Unit Tests
- Each component has standalone test code
- Run via: `python <component>.py`

### Integration Tests
- Full system test: `python test_alerting_system.py`
- Tests all components together
- Verifies configuration loading

### Production Tests
- Monitor first battle run
- Verify alerts triggered appropriately
- Check database growth rate
- Confirm no performance impact

## Security Considerations

1. **No sensitive data in alerts**
   - No API keys
   - No account numbers
   - No PII

2. **Local storage only**
   - Database not networked
   - Files in secure directory
   - No external transmission (yet)

3. **Future considerations**
   - Encrypt email payloads
   - Authenticate webhook calls
   - Rate limit alert generation

## Operational Metrics

Track these metrics for alerting system health:

- Alert generation rate (alerts/hour)
- Alert deduplication rate (%)
- Database size growth (MB/day)
- Average alert latency (ms)
- Notifier success rate (%)
- Health check duration (ms)

---

**Last Updated:** 2025-01-03
**Version:** 1.0.0
