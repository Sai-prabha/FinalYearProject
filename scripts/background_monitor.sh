#!/bin/bash
# Background monitoring that logs to file (no timeout issues)

LOG_FILE="reports/optuna_monitoring.log"
cd /Users/sai.p/VSCode/Workspace\(main\)/FinalYearProject

echo "Starting background monitoring - logging to $LOG_FILE"
echo "Press Ctrl+C to stop"

while true; do
    {
        echo ""
        echo "================================"
        echo "Update: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "================================"
        
        # Check workers
        WORKERS=$(ps aux | grep "optuna_v4_12_pure_sharpe.py" | grep -v grep | grep -v monitor | wc -l | tr -d ' ')
        echo "Workers: $WORKERS running"
        
        if [ "$WORKERS" = "0" ]; then
            echo "⚠️  All workers stopped!"
            break
        fi
        
        # Database stats
        if [ -f "reports/optuna_v4_12.db" ]; then
            sqlite3 reports/optuna_v4_12.db << 'SQL'
SELECT 'Trials: ' || COUNT(*) || ' total, ' || 
       SUM(CASE WHEN state='COMPLETE' THEN 1 ELSE 0 END) || ' complete, ' ||
       SUM(CASE WHEN state='RUNNING' THEN 1 ELSE 0 END) || ' running'
FROM trials;

SELECT 'Best Sharpe: ' || COALESCE(CAST(MAX(value) AS TEXT), 'None yet')
FROM trial_values WHERE value > -900;
SQL
        fi
        
    } >> "$LOG_FILE" 2>&1
    
    sleep 300  # Check every 5 minutes
done

echo "Monitoring stopped" >> "$LOG_FILE"

