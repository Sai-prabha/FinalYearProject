#!/bin/bash
# Quick status check for classifier Optuna workers

cd /Users/sai.p/VSCode/Workspace\(main\)/FinalYearProject
DB="reports/optuna_v4_12_cls.db"

echo "========================================================================"
echo "Optuna V4.12 CLASSIFIER Status - $(date '+%H:%M:%S')"
echo "========================================================================"

# Workers
WORKERS=$(ps aux | grep "optuna_v4_12_classifier.py" | grep -v grep | wc -l | tr -d ' ')
echo "Workers running: $WORKERS"

if [ "$WORKERS" -gt 0 ]; then
    ps aux | grep "optuna_v4_12_classifier.py" | grep -v grep | awk '{printf "  PID %s: CPU %s%%, MEM %s%%\n", $2, $3, $4}'
fi

# Database
if [ -f "$DB" ]; then
    echo ""
    echo "Trials:"
    sqlite3 "$DB" "SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN state='COMPLETE' THEN 1 ELSE 0 END) as done,
        SUM(CASE WHEN state='RUNNING' THEN 1 ELSE 0 END) as running
    FROM trials;" 2>/dev/null | awk -F'|' '{printf "  Total: %s | Completed: %s | Running: %s\n", $1, $2, $3}'
    
    # Best Sharpe
    BEST=$(sqlite3 "$DB" "SELECT MAX(value) FROM trial_values WHERE value > -900;" 2>/dev/null)
    if [ ! -z "$BEST" ] && [ "$BEST" != "" ]; then
        echo ""
        echo "🏆 Best Sharpe: $BEST"
        
        # Get best trial details
        sqlite3 "$DB" "
            SELECT 'Trial #' || t.number || ': Sharpe=' || ROUND(tv.value, 4) ||
                   ' | Trades=' || COALESCE(tua1.value_json, '?') ||
                   ' | Neutral=' || COALESCE(ROUND(CAST(tua2.value_json AS REAL), 1), '?') || '%'
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            LEFT JOIN trial_user_attributes tua1 ON t.trial_id = tua1.trial_id AND tua1.key = 'TradeCount'
            LEFT JOIN trial_user_attributes tua2 ON t.trial_id = tua2.trial_id AND tua2.key = 'NeutralPct'
            WHERE tv.value > -900
            ORDER BY tv.value DESC
            LIMIT 5;
        " 2>/dev/null | while read line; do echo "  $line"; done
    else
        echo ""
        echo "No valid trials yet"
    fi
    
    # Valid vs rejected
    VALID=$(sqlite3 "$DB" "SELECT COUNT(*) FROM trial_values WHERE value > -900;" 2>/dev/null)
    TOTAL=$(sqlite3 "$DB" "SELECT COUNT(*) FROM trials WHERE state='COMPLETE';" 2>/dev/null)
    if [ "$TOTAL" -gt 0 ]; then
        echo ""
        echo "Valid: $VALID / $TOTAL ($(echo "scale=1; $VALID * 100 / $TOTAL" | bc)%)"
    fi
    
    # Rejection reasons
    REJECTED=$(sqlite3 "$DB" "SELECT SUBSTR(value_json, 2, 40) as reason, COUNT(*) as cnt 
        FROM trial_user_attributes WHERE key='rejection_reason' 
        GROUP BY reason ORDER BY cnt DESC LIMIT 3;" 2>/dev/null)
    if [ ! -z "$REJECTED" ]; then
        echo ""
        echo "Top rejections:"
        echo "$REJECTED" | while IFS='|' read reason cnt; do
            echo "  $reason: $cnt"
        done
    fi
else
    echo ""
    echo "No database yet (workers still starting up)"
fi

echo ""
echo "========================================================================"

