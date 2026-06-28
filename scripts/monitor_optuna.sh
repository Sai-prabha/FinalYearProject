#!/bin/bash
# Monitor Optuna Pseudo-Huber optimization progress

LOG_FILE="pseudohuber_optimization.log"
PID=22508

echo "="
echo "PSEUDO-HUBER OPTIMIZATION MONITOR"
echo "="

# Check if process is running
if ps -p $PID > /dev/null 2>&1; then
    echo "✓ Status: RUNNING (PID: $PID)"
    
    # Runtime
    START_TIME=$(ps -p $PID -o lstart= | xargs -I {} date -j -f "%a %b %d %T %Y" "{}" +%s 2>/dev/null || echo "")
    if [ -n "$START_TIME" ]; then
        CURRENT_TIME=$(date +%s)
        ELAPSED=$((CURRENT_TIME - START_TIME))
        MINS=$((ELAPSED / 60))
        SECS=$((ELAPSED % 60))
        echo "✓ Runtime: ${MINS}m ${SECS}s"
    fi
    
    # Progress from log
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "Latest progress:"
        tail -20 "$LOG_FILE" | grep -E "Trial|%|Sharpe|it/s" | tail -5
    fi
    
    echo ""
    echo "ℹ️  Estimated completion: ~30-45 minutes total"
    echo "ℹ️  To view full log: tail -f $LOG_FILE"
else
    echo "✗ Status: NOT RUNNING"
    echo ""
    
    # Check if completed
    if [ -f "reports/v4_pseudohuber_test/optuna_results.json" ]; then
        echo "✅ OPTIMIZATION COMPLETE!"
        echo ""
        echo "Results:"
        cat reports/v4_pseudohuber_test/optuna_results.json | python3 -m json.tool 2>/dev/null || cat reports/v4_pseudohuber_test/optuna_results.json
    else
        echo "❌ Process stopped but no results found"
        echo "Check log: tail -100 $LOG_FILE"
    fi
fi

echo ""
echo "="

