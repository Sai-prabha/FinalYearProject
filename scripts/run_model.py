"""Unified pipeline runner for V4 and V5 - XGBoost focus only.

LSTM removed for simplicity - focus on proven XGBoost models.
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.report_utils import find_latest_csv, select_best_tau, write_selected_params


def _get_float(best: dict, keys, default=None):
    for key in keys:
        if key in best:
            val = best.get(key)
            try:
                f = float(val)
                if not math.isnan(f):
                    return f
            except (TypeError, ValueError):
                continue
    return default


def run_cmd(cmd, cwd=None):
    print(f"[runner] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified pipeline runner for v4.1 and v5 (XGBoost only)")
    parser.add_argument("--pipeline", choices=["v4", "v5"], required=True)
    parser.add_argument("--months", type=int, default=12, help="Months of history for v4 prepare_data")
    parser.add_argument("--days", type=int, default=14, help="Days of live data for v5 build")
    parser.add_argument("--out", default="reports/v4.1", help="Output directory for v4 artifacts")
    parser.add_argument("--out-v5", default="reports/v5", help="Output directory for v5 artifacts")
    parser.add_argument("--wait-live", type=int, default=0, help="Minutes to wait before running v5 build step")
    parser.add_argument("--tau-grid", default="0.8,1.0,1.2", help="TAU multipliers for v5 sweep")
    parser.add_argument("--prob-gate-grid", default="0.55,0.60,0.65", help="Probability gates for v5 sweep")
    parser.add_argument("--cooldown", type=int, default=10, help="Cooldown setting for backtests")
    parser.add_argument("--cost-bps", type=float, default=0.0, help="Trading cost in basis points (default: 0)")
    return parser.parse_args()


def _extract_params(best: dict, args: argparse.Namespace) -> dict:
    """Extract parameters from best result with fallback defaults."""
    return {
        "tau_bps": _get_float(best, ["TauBps", "tau_bps"], 0.0) or 0.0,
        "tau_mult": _get_float(best, ["TauBase", "TauMultiplier", "tau_mult"], None),
        "prob_gate": _get_float(best, ["ProbGate", "prob_gate"], 0.60) or 0.60,
        "cooldown": int(_get_float(best, ["Cooldown", "cooldown"], args.cooldown)),
        "cost_bps": _get_float(best, ["CostBps", "cost_bps"], args.cost_bps) or args.cost_bps,
    }


def pipeline_v4(args: argparse.Namespace) -> None:
    """Run V4 pipeline - XGBoost only."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("V4 PIPELINE: XGBOOST")
    print("="*80)
    print(f"Output directory: {out_dir}")
    print("Focus: XGBoost only (LSTM removed)")
    print("="*80 + "\n")

    # Prepare data once
    run_cmd([sys.executable, "-m", "src.prepare_data", "--months", str(args.months)])
    
    # Use run_v4_9_3.py for production model
    run_cmd([sys.executable, "scripts/run_v4_9_3.py"])
    
    print("\n" + "="*80)
    print("V4 PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nReports generated in: reports/v4.9.3/xgboost/")
    print("\nXGBoost (Production):")
    print(f"  - reports/v4.9.3/xgboost/equity_curve.png")
    print(f"  - reports/v4.9.3/xgboost/feature_importance.png")
    print(f"  - reports/v4.9.3/xgboost/metrics_summary.csv")
    print("="*80)


def pipeline_v5(args: argparse.Namespace) -> None:
    """Run V5 pipeline - XGBoost only."""
    out_dir = Path(args.out_v5)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("V5 PIPELINE: XGBOOST (CROSS-EXCHANGE)")
    print("="*80)

    if args.wait_live > 0:
        print(f"[runner] Waiting {args.wait_live} minutes for live data...")
        time.sleep(args.wait_live * 60)

    run_cmd([sys.executable, "-m", "src.data.build_cross_exchange", "--days", str(args.days), "--live-dir", "data/live", "--out", "data/processed_v5"])
    run_cmd([sys.executable, "-m", "scripts.tune_tau_v5", "--days", str(args.days), "--grid", args.tau_grid, "--prob-gate", args.prob_gate_grid, "--cooldown", str(args.cooldown), "--out", str(out_dir)])

    csv_path = find_latest_csv(out_dir, pattern="tau_sweep_v5*.csv")
    best = select_best_tau(csv_path)
    params = _extract_params(best, args)
    config_path = write_selected_params(out_dir, params)

    run_cmd([sys.executable, "-m", "src.train_and_backtest_v5", "--use-config", str(config_path), "--report-dir", str(out_dir)])

    print("\nRUN SUMMARY")
    print(" pipeline: v5")
    print(json.dumps(params, indent=2))
    print(f" artifacts directory: {out_dir}")


def main() -> None:
    args = parse_args()
    if args.pipeline == "v4":
        pipeline_v4(args)
    else:
        pipeline_v5(args)


if __name__ == "__main__":
    main()
