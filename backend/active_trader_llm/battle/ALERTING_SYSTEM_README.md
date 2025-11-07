# Battle Alerting and Monitoring System

Comprehensive alerting and health monitoring system for the LLM Battle Royale.

## Overview

The alerting system monitors all models in real-time and sends alerts when issues are detected:
- Model crashes and errors
- Performance degradation
- Trading anomalies
- Circuit breaker triggers
- System health issues

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Battle Orchestrator                        │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Model 1    │  │   Model 2    │  │   Model 3    │     │
│  │   (GPT-4)    │  │  (Claude)    │  │   (Grok)     │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                  │              │
└─────────┼─────────────────┼──────────────────┼──────────────┘
          │                 │                  │
          └─────────────────┼──────────────────┘
                           │
                           ▼
          ┌────────────────────────────────────┐
          │   BattleAlertingSystem              │
          │                                     │
          │  ┌──────────────────────────────┐  │
          │  │     HealthMonitor            │  │
          │  │  - Response time tracking    │  │
          │  │  - Error rate monitoring     │  │
          │  │  - Activity tracking         │  │
          │  └──────────────────────────────┘  │
          │                                     │
          │  ┌──────────────────────────────┐  │
          │  │     AlertManager             │  │
          │  │  - Threshold checking        │  │
          │  │  - Alert generation          │  │
          │  │  - Deduplication             │  │
          │  └──────────────────────────────┘  │
          │                                     │
          │  ┌──────────────────────────────┐  │
          │  │     AlertsDatabase           │  │
          │  │  - Persistent storage        │  │
          │  │  - Historical analysis       │  │
          │  └──────────────────────────────┘  │
          └────────────────┬───────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
    ┌──────────┐                   ┌──────────┐
    │ Console  │                   │   File   │
    │ Notifier │                   │ Notifier │
    └──────────┘                   └──────────┘
    Real-time                      Persistent
    alerts                         log file
```

## Components

### 1. Alert Manager (`alerting.py`)

Central coordinator for all alerts.

**Responsibilities:**
- Check thresholds (drawdown, win rate, error rate, etc.)
- Generate alerts when thresholds are exceeded
- Distribute alerts to notification channels
- Deduplicate alerts within time window

**Key Methods:**
```python
alert_manager.check_drawdown(model_id, drawdown_pct, equity)
alert_manager.check_error_rate(model_id, error_count, total_cycles)
alert_manager.check_win_rate(model_id, win_rate_pct, total_trades)
alert_manager.alert_model_crash(model_id, error_message)
alert_manager.alert_circuit_breaker(model_id, reason)
```

### 2. Health Monitor (`health_monitor.py`)

Monitors model health and generates health reports.

**Tracks:**
- Response time (is model responding?)
- Error rate (how many failures?)
- Last decision time (when was last activity?)
- Consecutive errors (is model stuck?)

**Health Scoring:**
- **100-80**: Healthy (green)
- **79-60**: Degraded (yellow)
- **59-30**: Unhealthy (orange)
- **<30**: Critical (red)

**Key Methods:**
```python
report = health_monitor.check_model_health(model_id, model_state)
system_health = health_monitor.get_system_health(all_reports)
```

### 3. Notification Channels

#### Console Notifier (`notifications/console_notifier.py`)
- Real-time alerts to stdout
- Color-coded by severity
- Formatted metadata display

#### File Notifier (`notifications/file_notifier.py`)
- Persistent JSONL log file
- Automatic log rotation
- Queryable alert history

#### Base Notifier (`notifications/base_notifier.py`)
- Abstract base class
- Extensible for email, webhooks, Slack, etc.

### 4. Alerts Database (`alerts_db.py`)

SQLite database for persistent alert storage.

**Features:**
- Historical alert tracking
- Filtering by model, severity, type
- Alert acknowledgment and resolution
- Statistical analysis

**Schema:**
```sql
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY,
    alert_type TEXT,      -- model_crash, critical_drawdown, etc.
    severity TEXT,        -- INFO, WARNING, CRITICAL, EMERGENCY
    model_id TEXT,
    message TEXT,
    metadata TEXT,        -- JSON blob
    timestamp TIMESTAMP,
    acknowledged BOOLEAN,
    resolved BOOLEAN
);
```

### 5. Alerting Integration (`alerting_integration.py`)

Bridges alerting system with battle orchestrator.

**Provides:**
- Configuration loading
- Component initialization
- Automatic health checks
- Metrics monitoring integration

## Alert Types

### Model Health Alerts

1. **MODEL_CRASH**
   - **Severity:** CRITICAL
   - **Trigger:** Model enters error state
   - **Action:** Immediate notification

2. **HIGH_ERROR_RATE**
   - **Severity:** WARNING
   - **Trigger:** Error rate > 10%
   - **Action:** Monitor for degradation

3. **MODEL_RECOVERY**
   - **Severity:** INFO
   - **Trigger:** Model recovers from error state
   - **Action:** Log recovery time

### Performance Alerts

4. **CRITICAL_DRAWDOWN**
   - **Severity:** EMERGENCY
   - **Trigger:** Drawdown > 20%
   - **Action:** Circuit breaker consideration

5. **WARNING_DRAWDOWN**
   - **Severity:** WARNING
   - **Trigger:** Drawdown > 10%
   - **Action:** Increased monitoring

6. **LOW_WIN_RATE**
   - **Severity:** WARNING
   - **Trigger:** Win rate < 40% (after 10+ trades)
   - **Action:** Review model strategy

### Position Alerts

7. **POSITION_CONCENTRATION**
   - **Severity:** WARNING
   - **Trigger:** Single position > 40% of equity
   - **Action:** Review risk management

### System Alerts

8. **CIRCUIT_BREAKER**
   - **Severity:** EMERGENCY
   - **Trigger:** Daily loss limit exceeded
   - **Action:** Halt trading for model

9. **UNUSUAL_TRADING**
   - **Severity:** WARNING
   - **Trigger:** >10 trades in 5 minutes
   - **Action:** Check for model malfunction

## Configuration

### In `battle_config.yaml`:

```yaml
alerting:
  enabled: true

  channels:
    - console
    - file

  database_path: "data/battle/alerts.db"

  file_notifier:
    log_path: "data/battle/alerts.log"
    max_file_size_mb: 10
    backup_count: 5

  console_notifier:
    use_colors: true

  thresholds:
    drawdown_warning_pct: 10.0
    drawdown_critical_pct: 20.0
    win_rate_warning_pct: 40.0
    error_rate_warning_pct: 10.0
    position_concentration_pct: 40.0
    unusual_trade_count: 10
    unusual_trade_window_minutes: 5

  dedup_window_minutes: 5

  health_check:
    enabled: true
    check_interval_seconds: 60
    max_response_time_seconds: 300
    max_consecutive_errors: 3
    max_time_since_decision_minutes: 30
```

## Usage

### Integration with Battle Orchestrator

```python
from active_trader_llm.battle.alerting_integration import BattleAlertingSystem

# Initialize alerting system
alerting_config = battle_config['alerting']
alerting_system = BattleAlertingSystem(alerting_config)

# During battle cycle:

# 1. Check model health
for model in model_instances:
    model_state = model.get_current_state()
    health_report = alerting_system.check_model_health(
        model.model_id,
        model_state
    )

    # Handle critical health
    if health_report.status == HealthStatus.CRITICAL:
        logger.critical(f"{model.model_id} in CRITICAL health")
        # Consider pausing model

# 2. Check metrics
metrics = metrics_engine.get_all_metrics(model_id, ...)
alerting_system.check_metrics(model_id, metrics.to_dict())

# 3. Check all models' health
all_states = {m.model_id: m.get_current_state() for m in models}
health_reports = alerting_system.check_all_models_health(all_states)

# 4. Get alert statistics
stats = alerting_system.get_alert_statistics()
```

### Querying Alert History

```python
# Get recent critical alerts
critical_alerts = alerting_system.get_recent_alerts(
    limit=20,
    severity='CRITICAL'
)

# Get alerts for specific model
gpt4_alerts = alerting_system.get_recent_alerts(
    limit=50,
    model_id='gpt4o'
)

# Get statistics
stats = alerting_system.get_alert_statistics()
print(f"Total alerts: {stats['total_alerts']}")
print(f"By severity: {stats['by_severity']}")
print(f"By model: {stats['by_model']}")
```

### Manual Alert Triggering

```python
# Send custom alert
alerting_system.alert_manager.send_alert(
    alert_type=AlertType.UNUSUAL_TRADING,
    severity=AlertSeverity.WARNING,
    model_id="gpt4o",
    message="Detected 12 trades in 3 minutes",
    metadata={'trade_count': 12, 'window_minutes': 3}
)
```

## Alert Deduplication

Alerts are deduplicated within a configurable time window (default: 5 minutes).

**Example:**
- T=0:00 - "gpt4o critical drawdown" → **SENT**
- T=0:02 - "gpt4o critical drawdown" → **SKIPPED** (duplicate)
- T=0:06 - "gpt4o critical drawdown" → **SENT** (window expired)

This prevents alert spam while ensuring critical issues are not missed.

## Health Monitoring

### Health Check Cycle

```
Every 60 seconds (configurable):
  For each model:
    1. Check response time
    2. Check error rate
    3. Check last activity time
    4. Check consecutive errors
    5. Calculate health score
    6. Generate health report
    7. Trigger alerts if needed
```

### Health Report Example

```python
ModelHealthReport(
    model_id='gpt4o',
    status=HealthStatus.DEGRADED,
    health_score=65.0,
    is_responding=True,
    error_count=5,
    error_rate_pct=8.3,
    cycles_completed=60,
    avg_response_time_seconds=180.5,
    issues=[
        'Elevated error rate: 8.3%',
        'Slow response time: 180.5s'
    ],
    warnings=[
        'No recent decision (15 minutes)'
    ]
)
```

## Testing

### Run Individual Components

```bash
# Test base notifier
python active_trader_llm/battle/notifications/base_notifier.py

# Test console notifier
python active_trader_llm/battle/notifications/console_notifier.py

# Test file notifier
python active_trader_llm/battle/notifications/file_notifier.py

# Test alert manager
python active_trader_llm/battle/alerting.py

# Test health monitor
python active_trader_llm/battle/health_monitor.py

# Test alerts database
python active_trader_llm/battle/alerts_db.py

# Test integration
python active_trader_llm/battle/alerting_integration.py
```

## Future Enhancements

### Additional Notification Channels

1. **Email Notifier**
   - Send email alerts for critical issues
   - Configurable recipients
   - HTML formatting

2. **Webhook Notifier**
   - POST alerts to custom endpoints
   - Integrate with external monitoring systems

3. **Slack Notifier**
   - Send alerts to Slack channel
   - Rich message formatting
   - Interactive buttons

4. **SMS Notifier**
   - Text message alerts for emergencies
   - Twilio integration

### Advanced Features

1. **Alert Routing**
   - Different channels for different severities
   - Model-specific notification rules

2. **Alert Aggregation**
   - Batch similar alerts
   - Summary notifications

3. **Predictive Alerts**
   - Machine learning-based anomaly detection
   - Predict issues before they occur

4. **Dashboard Integration**
   - Real-time alert feed in web dashboard
   - Alert acknowledgment UI
   - Historical alert charts

## Database Queries

### Useful SQL Queries

```sql
-- Recent critical alerts
SELECT * FROM alerts
WHERE severity IN ('CRITICAL', 'EMERGENCY')
ORDER BY timestamp DESC
LIMIT 20;

-- Alert count by model
SELECT model_id, COUNT(*) as alert_count
FROM alerts
GROUP BY model_id
ORDER BY alert_count DESC;

-- Most common alert types
SELECT alert_type, COUNT(*) as count
FROM alerts
GROUP BY alert_type
ORDER BY count DESC;

-- Alerts in last hour
SELECT * FROM alerts
WHERE timestamp > datetime('now', '-1 hour')
ORDER BY timestamp DESC;

-- Unresolved critical alerts
SELECT * FROM alerts
WHERE severity IN ('CRITICAL', 'EMERGENCY')
  AND resolved = 0
ORDER BY timestamp DESC;
```

## Troubleshooting

### Issue: No alerts being sent

**Check:**
1. Is alerting enabled in config? (`alerting.enabled: true`)
2. Are notification channels configured? (`alerting.channels`)
3. Are thresholds being exceeded?
4. Check logs for initialization errors

### Issue: Too many duplicate alerts

**Solution:**
- Increase `dedup_window_minutes` in config
- Check if alerts are being force-sent (bypassing deduplication)

### Issue: Alerts database growing too large

**Solution:**
- Implement database cleanup (remove old resolved alerts)
- Archive old alerts to separate database
- Adjust retention policy

### Issue: Health checks too slow

**Solution:**
- Increase `check_interval_seconds`
- Reduce number of health metrics tracked
- Optimize database queries

## Performance Considerations

- Health checks are lightweight (< 10ms per model)
- Alert generation is async-safe
- Database writes are non-blocking
- File rotation happens automatically
- Deduplication uses in-memory cache

## Security

- No sensitive data in alert messages
- Database stored locally (not networked)
- File logs in secure directory
- Future: Encrypt email/webhook payloads

---

**Created:** 2025-01-03
**Version:** 1.0.0
**Author:** Battle System Team
