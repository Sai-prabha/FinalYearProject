#!/bin/bash
# Run Optuna optimization with multiple parallel workers

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="/opt/anaconda3/envs/ml_env/bin/python"

# Configuration
STUDY_NAME="v4_12_pure_sharpe_fast"
N_WORKERS=3
TRIALS_PER_WORKER=50
TOTAL_TRIALS=$((N_WORKERS * TRIALS_PER_WORKER))

echo "========================================================================"
echo "OPTUNA PARALLEL OPTIMIZATION LAUNCHER"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Study name:        $STUDY_NAME"
echo "  Number of workers: $N_WORKERS"
echo "  Trials per worker: $TRIALS_PER_WORKER"
echo "  Total trials:      $TOTAL_TRIALS"
echo ""
echo "Speed estimate:"
echo "  Time per trial:    ~5 minutes (with reduced folds)"
echo "  Total time:        ~$(echo "scale=1; $TRIALS_PER_WORKER * 5 / 60" | bc) hours per worker"
echo "  With parallelism:  ~$(echo "scale=1; $TRIALS_PER_WORKER * 5 / 60" | bc) hours (all workers finish together)"
echo ""
echo "========================================================================"
echo ""

# Kill any existing optuna processes
echo "🔍 Checking for existing Optuna processes..."
EXISTING_PIDS=$(ps aux | grep "optuna_v4_12_pure_sharpe.py" | grep -v grep | awk '{print $2}')
if [ ! -z "$EXISTING_PIDS" ]; then
    echo "⚠️  Found existing processes: $EXISTING_PIDS"
    read -p "Kill them? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kill $EXISTING_PIDS
        sleep 2
        echo "✓ Killed existing processes"
    else
        echo "⚠️  Continuing with existing processes running (may cause conflicts)"
    fi
fi

# Create logs directory
mkdir -p "$PROJECT_DIR/reports/optuna_parallel_logs"

# Launch workers
echo ""
echo "🚀 Launching $N_WORKERS parallel workers..."
echo ""

for i in $(seq 1 $N_WORKERS); do
    LOG_FILE="$PROJECT_DIR/reports/optuna_parallel_logs/worker_${i}.log"
    
    echo "Starting Worker $i..."
    echo "  Log: reports/optuna_parallel_logs/worker_${i}.log"
    
    nohup $PYTHON_BIN "$SCRIPT_DIR/optuna_v4_12_pure_sharpe.py" \
        --n-trials $TRIALS_PER_WORKER \
        --study-name "$STUDY_NAME" \
        --worker-id $i \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo "  PID: $PID"
    echo ""
    
    # Small delay to avoid database conflicts at startup
    sleep 2
done

echo "========================================================================"
echo "✅ All workers launched!"
echo "========================================================================"
echo ""
echo "Monitor progress:"
echo "  bash scripts/monitor_optuna_parallel.sh"
echo ""
echo "View individual worker logs:"
for i in $(seq 1 $N_WORKERS); do
    echo "  tail -f reports/optuna_parallel_logs/worker_${i}.log"
done
echo ""
echo "Stop all workers:"
echo "  pkill -f optuna_v4_12_pure_sharpe.py"
echo ""
echo "Check worker processes:"
echo "  ps aux | grep optuna_v4_12_pure_sharpe.py | grep -v grep"
echo ""
echo "========================================================================"

