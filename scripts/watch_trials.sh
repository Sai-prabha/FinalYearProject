#!/bin/bash

# Watch for new trial completions and display updates

LOG_FILE="optimization_pseudohuber.log"
LAST_TRIAL_COUNT=0

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🔍 WATCHING FOR NEW TRIAL COMPLETIONS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Press Ctrl+C to stop watching"
echo ""

while true; do
    CURRENT_COUNT=$(grep -c "Trial.*finished" "$LOG_FILE" 2>/dev/null || echo "0")
    
    if [ "$CURRENT_COUNT" -gt "$LAST_TRIAL_COUNT" ]; then
        # New trial completed!
        NEW_TRIALS=$((CURRENT_COUNT - LAST_TRIAL_COUNT))
        
        for i in $(seq 1 $NEW_TRIALS); do
            LATEST=$(grep "Trial.*finished" "$LOG_FILE" | tail -$i | head -1)
            
            TRIAL_NUM=$(echo "$LATEST" | grep -oP "Trial \K[0-9]+" 2>/dev/null || echo "?")
            SCORE=$(echo "$LATEST" | grep -oP "value: \K[0-9.-]+" 2>/dev/null || echo "?")
            BEST_TRIAL=$(echo "$LATEST" | grep -oP "Best is trial \K[0-9]+" 2>/dev/null || echo "?")
            BEST_VALUE=$(echo "$LATEST" | grep -oP "Best is trial [0-9]+ with value: \K[0-9.-]+" 2>/dev/null || echo "?")
            
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "  🎉 NEW TRIAL COMPLETED!"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            echo "✅ Trial #$TRIAL_NUM completed"
            echo "   Score: $SCORE"
            echo ""
            echo "🏆 Best so far: Trial #$BEST_TRIAL (Score: $BEST_VALUE)"
            echo ""
            echo "📈 Progress: $CURRENT_COUNT / 200 trials ($(($CURRENT_COUNT * 100 / 200))%)"
            echo ""
            echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
            echo ""
        done
        
        LAST_TRIAL_COUNT=$CURRENT_COUNT
    fi
    
    sleep 30  # Check every 30 seconds
done





