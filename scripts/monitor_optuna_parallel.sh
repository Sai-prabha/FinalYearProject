#!/bin/bash
# Monitor parallel Optuna workers

echo "========================================================================"
echo "OPTUNA PARALLEL WORKERS - PROGRESS MONITOR"
echo "========================================================================"
echo ""

# Check running processes
PIDS=$(ps aux | grep "optuna_v4_12_pure_sharpe.py" | grep -v grep | grep -v monitor | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "❌ No Optuna workers running"
    echo ""
    echo "Start workers with:"
    echo "  bash scripts/run_optuna_parallel.sh"
    exit 1
fi

# Count workers
N_WORKERS=$(echo "$PIDS" | wc -l | tr -d ' ')
echo "✅ Workers Running: $N_WORKERS"
echo ""

# Show each worker
echo "Worker Status:"
echo "------------------------------------------------------------------------"
for PID in $PIDS; do
    CPU=$(ps aux | grep "^[^ ]*[ ]*$PID" | awk '{print $3}')
    MEM=$(ps aux | grep "^[^ ]*[ ]*$PID" | awk '{print $4}')
    TIME=$(ps aux | grep "^[^ ]*[ ]*$PID" | awk '{print $10}')
    WORKER_ID=$(ps aux | grep "^[^ ]*[ ]*$PID" | grep -o "worker-id [0-9]*" | awk '{print $2}')
    
    if [ -z "$WORKER_ID" ]; then
        WORKER_ID="?"
    fi
    
    echo "  Worker $WORKER_ID (PID $PID): CPU ${CPU}%, MEM ${MEM}%, Time $TIME"
done
echo ""

# Database stats
if [ -f "reports/optuna_v4_12.db" ]; then
    echo "📊 Database Statistics:"
    echo "------------------------------------------------------------------------"
    
    sqlite3 reports/optuna_v4_12.db << 'EOF'
.mode column
SELECT 
    COUNT(*) as total_trials,
    SUM(CASE WHEN state = 'COMPLETE' THEN 1 ELSE 0 END) as completed,
    SUM(CASE WHEN state = 'RUNNING' THEN 1 ELSE 0 END) as running,
    SUM(CASE WHEN state = 'FAIL' THEN 1 ELSE 0 END) as failed
FROM trials;
EOF
    
    echo ""
    
    # Get best trial
    BEST=$(sqlite3 reports/optuna_v4_12.db "SELECT MAX(tv.value) FROM trial_values tv JOIN trials t ON tv.trial_id = t.trial_id WHERE tv.value > -900;" 2>/dev/null)
    
    if [ ! -z "$BEST" ] && [ "$BEST" != "" ]; then
        echo "🏆 Best Sharpe So Far: $BEST"
    else
        echo "⏳ No valid trials yet (all rejected by constraints)"
    fi
    
    echo ""
    
    # Recent completions
    echo "📈 Recent Trial Results (last 5):"
    echo "------------------------------------------------------------------------"
    sqlite3 reports/optuna_v4_12.db << 'EOF'
.mode column
SELECT 
    t.number as trial,
    ROUND(tv.value, 4) as sharpe,
    t.state,
    strftime('%H:%M', t.datetime_complete) as time
FROM trials t
LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
WHERE t.state = 'COMPLETE'
ORDER BY t.number DESC
LIMIT 5;
EOF
    
    echo ""
    
    # Rejection reasons
    echo "⚠️  Rejection Summary:"
    echo "------------------------------------------------------------------------"
    sqlite3 reports/optuna_v4_12.db << 'EOF'
SELECT 
    SUBSTR(value_json, 2, 15) as reason,
    COUNT(*) as count
FROM trial_user_attributes
WHERE key = 'rejection_reason'
GROUP BY SUBSTR(value_json, 2, 15)
ORDER BY count DESC;
EOF
    
fi

echo ""
echo "========================================================================"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Commands:"
echo "  View worker logs:  tail -f reports/optuna_parallel_logs/worker_*.log"
echo "  Stop all workers:  pkill -f optuna_v4_12_pure_sharpe.py"
echo "  Refresh monitor:   bash scripts/monitor_optuna_parallel.sh"
echo "========================================================================"

