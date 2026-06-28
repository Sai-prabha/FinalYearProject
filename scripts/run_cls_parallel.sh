#!/bin/bash
# Launch parallel XGBClassifier Optuna workers

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PYTHON="/opt/anaconda3/envs/ml_env/bin/python"
SCRIPT="scripts/optuna_v4_12_classifier.py"
STUDY="v4_12_classifier"
TRIALS_PER_WORKER=50
N_WORKERS=3
LOG_DIR="reports/optuna_cls_logs"

echo "========================================================================"
echo "OPTUNA V4.12 CLASSIFIER - PARALLEL LAUNCHER"
echo "========================================================================"
echo ""
echo "Config:"
echo "  Script:   $SCRIPT"
echo "  Study:    $STUDY"
echo "  Workers:  $N_WORKERS"
echo "  Trials:   $TRIALS_PER_WORKER per worker (${N_WORKERS}x${TRIALS_PER_WORKER} = $((N_WORKERS * TRIALS_PER_WORKER)) total)"
echo ""

# Kill any existing workers
echo "Checking for existing workers..."
pkill -f "optuna_v4_12_classifier.py" 2>/dev/null && echo "  Killed existing workers" || echo "  No existing workers"
sleep 2

# Create log directory
mkdir -p "$LOG_DIR"

echo ""
echo "Launching $N_WORKERS workers..."
echo ""

PIDS=""
for i in $(seq 1 $N_WORKERS); do
    LOG_FILE="$LOG_DIR/worker_${i}.log"
    nohup $PYTHON $SCRIPT \
        --n-trials $TRIALS_PER_WORKER \
        --study-name $STUDY \
        --worker-id $i \
        > "$LOG_FILE" 2>&1 &
    PID=$!
    PIDS="$PIDS $PID"
    echo "  Worker $i: PID $PID → $LOG_FILE"
done

echo ""
echo "========================================================================"
echo "All $N_WORKERS workers launched!"
echo "========================================================================"
echo ""
echo "Monitor:"
echo "  bash scripts/check_cls_status.sh"
echo ""
echo "Worker logs:"
for i in $(seq 1 $N_WORKERS); do
    echo "  tail -f $LOG_DIR/worker_${i}.log"
done
echo ""
echo "Stop all:"
echo "  pkill -f optuna_v4_12_classifier.py"
echo "========================================================================"

