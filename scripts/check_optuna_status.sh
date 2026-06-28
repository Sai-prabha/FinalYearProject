#!/bin/bash
# Quick status check without long monitoring

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Optuna Status Check - $(date '+%H:%M:%S')"
echo "============================================"

# Check workers
WORKERS=$(ps aux | grep "optuna_v4_12_pure_sharpe.py" | grep -v grep | grep -v monitor | wc -l | tr -d ' ')
echo "Workers running: $WORKERS"

# Check database
if [ -f "reports/optuna_v4_12.db" ]; then
    STATS=$(sqlite3 reports/optuna_v4_12.db "SELECT COUNT(*) as total, SUM(CASE WHEN state='COMPLETE' THEN 1 ELSE 0 END) as done, SUM(CASE WHEN state='RUNNING' THEN 1 ELSE 0 END) as running FROM trials;" 2>/dev/null)
    echo "Trials: $STATS (total|completed|running)"
    
    BEST=$(sqlite3 reports/optuna_v4_12.db "SELECT MAX(value) FROM trial_values WHERE value > -900;" 2>/dev/null)
    if [ ! -z "$BEST" ] && [ "$BEST" != "" ]; then
        echo "Best Sharpe: $BEST ✅"
    else
        echo "Best Sharpe: None yet (all rejected)"
    fi
fi

echo "============================================"

