#!/bin/bash
# Monitor V4.12 Optimization Progress

LOG_FILE="reports/v4_12_optimization_run.log"
JSON_LOG="reports/v4_12_optimization_log.jsonl"

echo "=================================================="
echo "V4.12 OPTIMIZATION PROGRESS MONITOR"
echo "=================================================="
echo ""

# Check if optimization is running
if pgrep -f "optimize_v4_12_to_target.py" > /dev/null; then
    echo "✅ Optimization is RUNNING"
else
    echo "⚠️  Optimization is NOT running"
fi

echo ""
echo "📊 Latest Results from Log:"
echo "--------------------------------------------------"

# Show last iteration results from log
if [ -f "$LOG_FILE" ]; then
    echo "Last 50 lines of log:"
    tail -50 "$LOG_FILE"
else
    echo "Log file not found yet: $LOG_FILE"
fi

echo ""
echo "=================================================="
echo "📈 SHARPE SUMMARY (from JSON log):"
echo "=================================================="

# Parse JSON log for Sharpe values
if [ -f "$JSON_LOG" ]; then
    echo ""
    python3 << 'EOF'
import json
import sys

try:
    with open('reports/v4_12_optimization_log.jsonl', 'r') as f:
        results = []
        for line in f:
            if line.strip():
                data = json.loads(line)
                iteration = data.get('iteration', 0)
                name = data.get('config_name', 'Unknown')
                sharpe = data.get('backtest_metrics', {}).get('Sharpe', 0)
                trade_count = data.get('backtest_metrics', {}).get('TradeCount', 0)
                max_dd = data.get('backtest_metrics', {}).get('MaxDrawdown', 0)
                results.append((iteration, name, sharpe, trade_count, max_dd))
        
        if results:
            print(f"{'Iter':<6} {'Config Name':<30} {'Sharpe':<10} {'Trades':<8} {'MaxDD':<10}")
            print("-" * 80)
            for iter, name, sharpe, trades, dd in results:
                print(f"{iter:<6} {name:<30} {sharpe:<10.4f} {trades:<8} {dd*100:<9.2f}%")
            
            best = max(results, key=lambda x: x[2])
            print("")
            print(f"🏆 BEST SO FAR: {best[1]} with Sharpe {best[2]:.4f}")
        else:
            print("No results yet")
except FileNotFoundError:
    print("JSON log file not found yet")
except Exception as e:
    print(f"Error parsing results: {e}")
EOF

else
    echo "JSON log not found yet: $JSON_LOG"
fi

echo ""
echo "=================================================="
echo "To view live log: tail -f reports/v4_12_optimization_run.log"
echo "=================================================="

