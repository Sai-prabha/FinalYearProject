#!/bin/bash

# Optuna Balanced Long/Short Optimization Progress Monitor
# Usage: ./check_optuna_progress.sh

LOG_FILE="optimization_pseudohuber.log"
PID=8880

echo "================================================================================"
echo "  OPTUNA OPTIMIZATION PROGRESS - Balanced Long/Short Strategy"
echo "================================================================================"
echo ""

# Check if process is running
if ps -p $PID > /dev/null 2>&1; then
    echo "✅ Status: RUNNING (PID: $PID)"
else
    echo "⚠️  Status: NOT RUNNING (completed or stopped)"
    if [ -f "$LOG_FILE" ]; then
        if grep -q "🎉 OPTIMIZATION COMPLETE" "$LOG_FILE"; then
            echo "   ✅ Optimization completed successfully!"
        else
            echo "   ⚠️  Process may have crashed - check log for errors"
        fi
    fi
fi
echo ""

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    exit 1
fi

# Count trials
echo "📊 Trial Progress:"
echo "--------------------------------------------------------------------------------"

# Count completed trials
COMPLETED_TRIALS=$(grep -c "Trial.*finished" "$LOG_FILE" 2>/dev/null || echo "0")
COMPLETED_TRIALS=${COMPLETED_TRIALS//[^0-9]/}  # Remove non-numeric chars
COMPLETED_TRIALS=${COMPLETED_TRIALS:-0}  # Default to 0 if empty
TOTAL_TRIALS=400  # 200 MSE + 200 Pseudo-Huber

echo "   Completed: $COMPLETED_TRIALS / $TOTAL_TRIALS trials"

# Calculate percentage (with safety check)
if [ "$COMPLETED_TRIALS" -gt 0 ] 2>/dev/null; then
    PERCENT=$((COMPLETED_TRIALS * 100 / TOTAL_TRIALS))
else
    PERCENT=0
fi
echo "   Progress: $PERCENT%"

# Progress bar
FILLED=$((PERCENT / 2))
EMPTY=$((50 - FILLED))
printf "   ["
printf "%${FILLED}s" | tr ' ' '█'
printf "%${EMPTY}s" | tr ' ' '░'
printf "]\n"

# Estimate time remaining
if [ "$COMPLETED_TRIALS" -gt 0 ] 2>/dev/null; then
    # Rough estimate: 2.5 minutes per trial
    REMAINING_TRIALS=$((TOTAL_TRIALS - COMPLETED_TRIALS))
    REMAINING_MINUTES=$((REMAINING_TRIALS * 2))
    REMAINING_HOURS=$((REMAINING_MINUTES / 60))
    REMAINING_MINS=$((REMAINING_MINUTES % 60))
    echo "   Estimated remaining: ${REMAINING_HOURS}h ${REMAINING_MINS}m"
fi
echo ""

# Current phase
echo "📍 Current Phase:"
echo "--------------------------------------------------------------------------------"
if grep -q "🚀 OPTIMIZING WITH MSE" "$LOG_FILE" && ! grep -q "MSE OPTIMIZATION COMPLETE" "$LOG_FILE"; then
    MSE_TRIALS=$(grep -c "Trial.*finished" "$LOG_FILE" 2>/dev/null || echo "0")
    MSE_TRIALS=${MSE_TRIALS//[^0-9]/}
    MSE_TRIALS=${MSE_TRIALS:-0}
    echo "   Phase 1: MSE Optimization ($MSE_TRIALS/200 trials)"
elif grep -q "MSE OPTIMIZATION COMPLETE" "$LOG_FILE" && ! grep -q "PSEUDO-HUBER OPTIMIZATION COMPLETE" "$LOG_FILE"; then
    PH_START=$(grep -n "🚀 OPTIMIZING WITH PSEUDO-HUBER" "$LOG_FILE" | cut -d: -f1)
    if [ -n "$PH_START" ]; then
        PH_TRIALS=$(tail -n +$PH_START "$LOG_FILE" | grep -c "Trial.*finished" 2>/dev/null || echo "0")
        PH_TRIALS=${PH_TRIALS//[^0-9]/}
        PH_TRIALS=${PH_TRIALS:-0}
        echo "   Phase 2: Pseudo-Huber Optimization ($PH_TRIALS/200 trials)"
    fi
else
    echo "   Phase: Starting up..."
fi
echo ""

# Best results so far
echo "🏆 Best Results So Far:"
echo "--------------------------------------------------------------------------------"

# Extract best trial info from each phase
if grep -q "Best Trial" "$LOG_FILE"; then
    # Get the most recent best trial section
    BEST_SECTION=$(grep -A 10 "🏆 Best Trial:" "$LOG_FILE" | tail -20)
    
    if echo "$BEST_SECTION" | grep -q "Sharpe:"; then
        BEST_SHARPE=$(echo "$BEST_SECTION" | grep "Sharpe:" | tail -1 | awk '{print $2}')
        BEST_PNL=$(echo "$BEST_SECTION" | grep "Total PnL:" | tail -1 | awk '{print $3}')
        BEST_TRADES=$(echo "$BEST_SECTION" | grep "Trades:" | tail -1 | awk '{print $2}')
        BEST_LONG=$(echo "$BEST_SECTION" | grep "Long%:" | tail -1 | awk '{print $2}')
        BEST_SHORT=$(echo "$BEST_SECTION" | grep "Short%:" | tail -1 | awk '{print $5}')
        
        echo "   Sharpe: $BEST_SHARPE"
        echo "   Total PnL: $BEST_PNL"
        echo "   Trades: $BEST_TRADES"
        echo "   Balance: $BEST_LONG long / $BEST_SHORT short"
    else
        echo "   (No best trial recorded yet)"
    fi
else
    echo "   (Trials still initializing...)"
fi
echo ""

# Latest trial info
echo "📝 Latest Trial:"
echo "--------------------------------------------------------------------------------"
LATEST_TRIAL=$(tail -50 "$LOG_FILE" | grep -E "(Trial [0-9]+ finished|Sharpe:|Penalties:|Trade)" | tail -5)
if [ -n "$LATEST_TRIAL" ]; then
    echo "$LATEST_TRIAL" | sed 's/^/   /'
else
    echo "   (No trial data yet)"
fi
echo ""

# Errors/warnings
echo "⚠️  Recent Warnings/Errors:"
echo "--------------------------------------------------------------------------------"
ERRORS=$(tail -100 "$LOG_FILE" | grep -i -E "(error|warning|failed|exception)" | tail -3)
if [ -n "$ERRORS" ]; then
    echo "$ERRORS" | sed 's/^/   /'
else
    echo "   ✅ No recent errors"
fi
echo ""

# Final status if complete
if grep -q "🎉 OPTIMIZATION COMPLETE" "$LOG_FILE"; then
    echo "================================================================================"
    echo "  ✅ OPTIMIZATION COMPLETE!"
    echo "================================================================================"
    echo ""
    echo "📂 Results saved to:"
    echo "   • reports/v4_balanced_longshort_optimization/optimization_results.json"
    echo "   • reports/v4_balanced_longshort_mse/all_trials.csv"
    echo "   • reports/v4_balanced_longshort_pseudo-huber/all_trials.csv"
    echo ""
    echo "Next steps:"
    echo "   1. Review optimization_results.json for best parameters"
    echo "   2. Update scripts/run_v4_11_optimized.py with best params"
    echo "   3. Run backtest to validate"
    echo ""
fi

echo "================================================================================"
echo "  Log file: $LOG_FILE"
echo "  Last updated: $(date)"
echo "================================================================================"

