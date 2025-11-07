# Battle Alerting System - Quick Start Guide

## 5-Minute Setup

### 1. Install (Already Done)

All files are already in place:
```bash
active_trader_llm/battle/
├── alerting.py              # Alert manager
├── health_monitor.py        # Health monitoring
├── alerts_db.py            # Database layer
├── alerting_integration.py # Integration layer
└── notifications/          # Notification channels
    ├── base_notifier.py
    ├── console_notifier.py
    └── file_notifier.py
```

### 2. Configure (Already Done)

Configuration is in `battle_config.yaml`:
```yaml
alerting:
  enabled: true
  channels: [console, file]
  database_path: "data/battle/alerts.db"
  # ... (see full config in battle_config.yaml)
```

### 3. Initialize in Battle Orchestrator

```python
from active_trader_llm.battle.alerting_integration import BattleAlertingSystem
import yaml

# Load config
with open('battle_config.yaml') as f:
    config = yaml.safe_load(f)

# Initialize alerting
alerting = BattleAlertingSystem(config['alerting'])
```

### 4. Use in Battle Loop

```python
# During each cycle:

# Check model health
for model in model_instances:
    health = alerting.check_model_health(
        model.model_id,
        model.get_current_state()
    )

    if health.status == HealthStatus.CRITICAL:
        logger.critical(f"{model.model_id} CRITICAL!")

# Check metrics after each decision
metrics = metrics_engine.get_all_metrics(...)
alerting.check_metrics(model.model_id, metrics.to_dict())

# Periodic system health check (every 5 minutes)
if time_to_check_system():
    all_states = {m.model_id: m.get_current_state() for m in models}
    system_health = alerting.check_all_models_health(all_states)
```

## Common Operations

### Check Model Health

```python
model_state = {
    'status': 'running',
    'cycles_completed': 50,
    'error_count': 2,
    'last_error': None,
    'last_cycle_time': datetime.now().isoformat()
}

report = alerting.check_model_health("gpt4o", model_state)

print(f"Health: {report.status.value}")
print(f"Score: {report.health_score}")
print(f"Issues: {report.issues}")
```

### Check Metrics

```python
metrics = {
    'max_drawdown_pct': 12.5,
    'current_equity': 87500,
    'win_rate': 45.0,
    'total_trades': 20
}

alerting.check_metrics("claude_sonnet", metrics)
# Alerts triggered automatically if thresholds exceeded
```

### Send Manual Alert

```python
from active_trader_llm.battle.notifications import AlertType, AlertSeverity

alerting.alert_manager.send_alert(
    alert_type=AlertType.UNUSUAL_TRADING,
    severity=AlertSeverity.WARNING,
    model_id="gpt4o",
    message="Detected unusual trading pattern",
    metadata={'trades': 15, 'window_min': 5}
)
```

### Query Alert History

```python
# Recent alerts
recent = alerting.get_recent_alerts(limit=20)

# Alerts for specific model
gpt4_alerts = alerting.get_recent_alerts(
    limit=50,
    model_id='gpt4o'
)

# Only critical alerts
critical = alerting.get_recent_alerts(
    limit=20,
    severity='CRITICAL'
)

# Statistics
stats = alerting.get_alert_statistics()
print(f"Total: {stats['total_alerts']}")
print(f"By model: {stats['by_model']}")
```

## Alert Types Reference

| Type | Severity | When It Triggers |
|------|----------|------------------|
| MODEL_CRASH | CRITICAL | Model enters error state |
| HIGH_ERROR_RATE | WARNING | Error rate ≥ 10% |
| MODEL_RECOVERY | INFO | Model recovers |
| CRITICAL_DRAWDOWN | EMERGENCY | Drawdown ≥ 20% |
| WARNING_DRAWDOWN | WARNING | Drawdown ≥ 10% |
| LOW_WIN_RATE | WARNING | Win rate < 40% |
| POSITION_CONCENTRATION | WARNING | Position ≥ 40% equity |
| CIRCUIT_BREAKER | EMERGENCY | Emergency shutdown |
| UNUSUAL_TRADING | WARNING | >10 trades in 5min |

## Health Status Levels

- **HEALTHY** (80-100): All systems normal
- **DEGRADED** (60-79): Minor issues, monitoring needed
- **UNHEALTHY** (30-59): Significant issues, action needed
- **CRITICAL** (<30): Severe issues, immediate action required

## Notification Channels

### Console (Enabled by default)
- Real-time colored output
- Shows in terminal immediately
- Good for development/monitoring

### File (Enabled by default)
- Persistent JSONL log
- Location: `data/battle/alerts.log`
- Automatic rotation at 10MB
- Queryable history

### Database (Always enabled)
- Location: `data/battle/alerts.db`
- Queryable with SQL
- Historical analysis
- Alert acknowledgment/resolution

## Configuration Cheat Sheet

```yaml
alerting:
  enabled: true                    # Master switch
  channels: [console, file]        # Active channels
  database_path: "data/battle/alerts.db"

  thresholds:
    drawdown_warning_pct: 10.0     # Warning at 10% DD
    drawdown_critical_pct: 20.0    # Critical at 20% DD
    win_rate_warning_pct: 40.0     # Warn if WR < 40%
    error_rate_warning_pct: 10.0   # Warn if errors ≥ 10%

  dedup_window_minutes: 5          # Don't repeat alerts within 5min

  health_check:
    enabled: true
    check_interval_seconds: 60     # Check every minute
    max_response_time_seconds: 300 # 5 min timeout
    max_consecutive_errors: 3      # Critical after 3 errors
```

## Testing

### Run Full Test
```bash
python test_alerting_system.py
```

### Test Individual Components
```bash
python active_trader_llm/battle/alerting.py
python active_trader_llm/battle/health_monitor.py
python active_trader_llm/battle/alerts_db.py
```

## Troubleshooting

### No alerts appearing

1. Check if enabled: `alerting.enabled` in config
2. Check channels: `alerting.channels` list
3. Verify thresholds are being exceeded
4. Check logs for errors

### Too many duplicate alerts

Increase deduplication window:
```yaml
dedup_window_minutes: 10  # Increase from 5 to 10
```

### Database too large

Run cleanup periodically:
```python
# Delete old resolved alerts
import sqlite3
conn = sqlite3.connect('data/battle/alerts.db')
conn.execute("""
    DELETE FROM alerts
    WHERE resolved = 1
    AND timestamp < datetime('now', '-30 days')
""")
conn.commit()
```

## Files & Locations

- **Config:** `battle_config.yaml` (alerting section)
- **Database:** `data/battle/alerts.db`
- **Log file:** `data/battle/alerts.log`
- **Code:** `active_trader_llm/battle/alerting*.py`
- **Docs:** `active_trader_llm/battle/ALERTING_*.md`

## Example: Full Integration

```python
# battle_orchestrator.py

from active_trader_llm.battle.alerting_integration import BattleAlertingSystem
from active_trader_llm.battle.health_monitor import HealthStatus
import yaml

class BattleOrchestrator:
    def __init__(self, config_path):
        # ... existing init ...

        # Load config
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Initialize alerting
        self.alerting = BattleAlertingSystem(config.get('alerting', {}))

    def start_competition(self, num_cycles):
        for cycle in range(num_cycles):
            # Run cycle
            self.cycle_coordinator.trigger_cycle()

            # Check health after cycle
            for model in self.model_instances:
                health = self.alerting.check_model_health(
                    model.model_id,
                    model.get_current_state()
                )

                # Handle critical health
                if health.status == HealthStatus.CRITICAL:
                    logger.critical(
                        f"{model.model_id} in critical state: {health.issues}"
                    )
                    # Optionally pause model

                # Check metrics
                metrics = self.get_model_metrics(model)
                self.alerting.check_metrics(model.model_id, metrics)

            # System-wide health check every 5 cycles
            if cycle % 5 == 0:
                all_states = {
                    m.model_id: m.get_current_state()
                    for m in self.model_instances
                }
                system_health = self.alerting.check_all_models_health(all_states)

                logger.info(
                    f"System health: {system_health['status']} "
                    f"(Score: {system_health['overall_health_score']:.1f})"
                )
```

## Next Steps

1. **Run test:** `python test_alerting_system.py`
2. **Integrate:** Add to `BattleOrchestrator`
3. **Monitor:** Watch alerts during first battle
4. **Tune:** Adjust thresholds based on observations
5. **Extend:** Add email/webhook notifiers as needed

## Quick Reference

```python
# Initialize
alerting = BattleAlertingSystem(config['alerting'])

# Check health
report = alerting.check_model_health(model_id, model_state)

# Check metrics
alerting.check_metrics(model_id, metrics_dict)

# System health
reports = alerting.check_all_models_health(all_states)

# Query alerts
alerts = alerting.get_recent_alerts(limit=100)

# Statistics
stats = alerting.get_alert_statistics()

# Manual alert
alerting.alert_manager.send_alert(type, severity, model_id, message)
```

---

**Ready to use! No additional setup required.**
