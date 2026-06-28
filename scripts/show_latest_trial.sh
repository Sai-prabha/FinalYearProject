#!/bin/bash

# Show latest completed trial from optimization

LOG_FILE="optimization_pseudohuber.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    exit 1
fi

# Extract latest trial completion
LATEST_TRIAL=$(grep "Trial.*finished" "$LOG_FILE" | tail -1)

if [ -z "$LATEST_TRIAL" ]; then
    echo "⏳ No trials completed yet. First trial in progress..."
    echo ""
    echo "Process status:"
    ps aux | grep "optuna_tune_v4_balanced_smart" | grep -v grep | head -1
    exit 0
fi

# Extract trial number and score
TRIAL_NUM=$(echo "$LATEST_TRIAL" | grep -oP "Trial \K[0-9]+")
SCORE=$(echo "$LATEST_TRIAL" | grep -oP "value: \K[0-9.-]+")
BEST_TRIAL=$(echo "$LATEST_TRIAL" | grep -oP "Best is trial \K[0-9]+")
BEST_VALUE=$(echo "$LATEST_TRIAL" | grep -oP "Best is trial [0-9]+ with value: \K[0-9.-]+")

# Count total completed
TOTAL_COMPLETED=$(grep -c "Trial.*finished" "$LOG_FILE")
TOTAL_TRIALS=200

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📊 LATEST TRIAL COMPLETION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Trial #$TRIAL_NUM completed"
echo "   Score: $SCORE"
echo ""
echo "🏆 Best so far: Trial #$BEST_TRIAL (Score: $BEST_VALUE)"
echo ""
echo "📈 Progress: $TOTAL_COMPLETED / $TOTAL_TRIALS trials ($(($TOTAL_COMPLETED * 100 / $TOTAL_TRIALS))%)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"





