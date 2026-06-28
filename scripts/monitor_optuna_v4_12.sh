#!/bin/bash
# Monitor Optuna V4.12 Pure Sharpe optimization progress

echo "========================================================================"
echo "OPTUNA V4.12 PURE SHARPE - PROGRESS MONITOR"
echo "========================================================================"
echo ""

# Check if process is running
PID=$(ps aux | grep "optuna_v4_12_pure_sharpe.py" | grep -v grep | grep -v monitor | awk '{print $2}' | head -1)

if [ -z "$PID" ]; then
    echo "❌ Optimization process NOT running"
    echo ""
    echo "Check if it completed or failed:"
    echo "  tail -100 reports/optuna_v4_12_full_run.log"
    exit 1
fi

# Process info
CPU=$(ps aux | grep $PID | grep -v grep | awk '{print $3}')
MEM=$(ps aux | grep $PID | grep -v grep | awk '{print $4}')
TIME=$(ps aux | grep $PID | grep -v grep | awk '{print $10}')

echo "✅ Process Status: RUNNING"
echo "   PID:        $PID"
echo "   CPU:        ${CPU}%"
echo "   Memory:     ${MEM}%"
echo "   CPU Time:   $TIME"
echo ""

# Check log for latest progress
echo "📊 Latest Progress:"
echo "------------------------------------------------------------------------"
tail -20 reports/optuna_v4_12_full_run.log | grep -E "(Trial|Sharpe|%|Best)" | tail -10
echo ""

# Check database for trial count
if [ -f "reports/optuna_v4_12.db" ]; then
    echo "📈 Database Status:"
    echo "------------------------------------------------------------------------"
    # Use Python to query SQLite
    python3 << 'EOF'
import sqlite3
import sys
from pathlib import Path

db_path = "reports/optuna_v4_12.db"
if not Path(db_path).exists():
    print("   Database not found")
    sys.exit(0)

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get study info
    cursor.execute("""
        SELECT study_name, direction 
        FROM studies 
        WHERE study_name = 'v4_12_pure_sharpe'
    """)
    study = cursor.fetchone()
    
    if study:
        print(f"   Study: {study[0]}")
        print(f"   Direction: {study[1]}")
        
        # Get trial stats
        cursor.execute("""
            SELECT study_id FROM studies WHERE study_name = 'v4_12_pure_sharpe'
        """)
        study_id = cursor.fetchone()[0]
        
        cursor.execute(f"""
            SELECT COUNT(*), 
                   MIN(value), 
                   MAX(value), 
                   AVG(value)
            FROM trials 
            WHERE study_id = {study_id}
              AND value IS NOT NULL
              AND value > -900
        """)
        stats = cursor.fetchone()
        
        if stats and stats[0] > 0:
            print(f"   Completed trials: {stats[0]}")
            print(f"   Best Sharpe:      {stats[2]:.4f}")
            print(f"   Worst Sharpe:     {stats[1]:.4f}")
            print(f"   Average Sharpe:   {stats[3]:.4f}")
        else:
            print("   No valid trials yet")
        
        # Get rejection stats
        cursor.execute(f"""
            SELECT COUNT(*)
            FROM trials 
            WHERE study_id = {study_id}
              AND value IS NOT NULL
              AND value <= -900
        """)
        rejected = cursor.fetchone()[0]
        print(f"   Rejected trials:  {rejected}")
    else:
        print("   Study not found in database")
    
    conn.close()
except Exception as e:
    print(f"   Error reading database: {e}")
EOF
fi

echo ""
echo "========================================================================"
echo "Started: $(grep 'Started:' reports/optuna_v4_12_full_run.log | tail -1)"
echo "Current: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "To view live log:"
echo "  tail -f reports/optuna_v4_12_full_run.log"
echo "========================================================================"

