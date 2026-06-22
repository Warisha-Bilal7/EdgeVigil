# Example Incident Runbook

## CPU Leak on Linux Host
- **Symptoms**: Gradual linear increase in CPU usage over 24 hours.
- **Root Cause**: Memory leak in daemon process or runaway thread.
- **Resolution**: Restart service, check log files.
