import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.backtest import dynamic_tau, generate_positions_v5, pnl_from_positions_v5
    from src.data.build_cross_exchange import DEFAULT_OUT as DEFAULT_PROCESSED
    from src.features import build_features_v5
    from src.models import (
        get_feature_importance_plot_v5,
        make_xgb_cls,
        make_xgb_reg,
        walk_forward_cls_v5,
        walk_forward_reg_v5,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback for direct execution
    from backtest import dynamic_tau, generate_positions_v5, pnl_from_positions_v5
    from data.build_cross_exchange import DEFAULT_OUT as DEFAULT_PROCESSED
    from features import build_features_v5
    from models import (
        get_feature_importance_plot_v5,
        make_xgb_cls,
        make_xgb_reg,
        walk_forward_cls_v5,
        walk_forward_reg_v5,
    )

from scripts.tune_tau import _pretty_tau_table  # reuse helper

REPORT_DIR = Path("reports/v5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TAU/Prob sweep for Version 5")
    parser.add_argument("--days", type=int, default=14, help="Lookback days for dataset build")
    parser.add_argument("--grid", default="0.8,1.0,1.2,1.5,2.0", help="Comma-separated TAU multipliers")
    parser.add_argument("--prob-gate", default="0.55,0.60,0.65", help="Prob thresholds to evaluate")
    parser.add_argument("--cooldown", type=int, default=10, help="Cooldown in decision steps")
    parser.add_argument("--processed", default=str(DEFAULT_PROCESSED), help="Processed v5 directory")
    parser.add_argument("--out", default=str(REPORT_DIR), help="Output directory for reports")
    return parser.parse_args()


def load_dataset(processed_dir: Path, days: int) -> pd.DataFrame:
    path = processed_dir / "cross_ex_ratio_1m.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset {path} not found")
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df["ts"], utc=True)
    df["ts"] = ts.dt.tz_localize(None)
    cutoff = df["ts"].max() - pd.Timedelta(days=days)
    df = df[df["ts"] >= cutoff].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Processed dataset is empty for the given horizon")
    return df


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed)
    report_dir = Path(args.out)
    report_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(processed_dir, args.days)
    X, y_reg, y_cls, meta = build_features_v5(data)

    preds_reg, _, reg_metrics, te_idx_reg, clean_idx_reg = walk_forward_reg_v5(X, y_reg)
    print("[v5] Regression metrics:", reg_metrics)

    probs_cls, cls_labels, cls_metrics, te_idx_cls, _ = walk_forward_cls_v5(X, y_cls)
    print("[v5] Classification metrics:", cls_metrics)

    time_index = pd.to_datetime(meta["time_index"], utc=True).tz_convert(None)
    decision_times_reg = time_index[np.array(te_idx_reg, dtype=int)]
    preds_reg_series = pd.Series(preds_reg, index=decision_times_reg)

    if probs_cls.size:
        decision_times_cls = time_index[np.array(te_idx_cls, dtype=int)]
        probs_series = pd.Series(probs_cls, index=decision_times_cls)
    else:
        decision_times_cls = None
        probs_series = None

    fwd_returns = pd.Series(y_reg.values, index=time_index)

    vol_series = data.set_index(pd.to_datetime(data["ts"], utc=True).tz_convert(None))["ratio_mean_ret"].fillna(0)
    spread_series = data.set_index(pd.to_datetime(data["ts"], utc=True).tz_convert(None))["spread"].fillna(0)
    tau_base = dynamic_tau(vol_series, spread_series)

    tau_multipliers = parse_float_list(args.grid)
    prob_gates = parse_float_list(args.prob_gate)

    rows = []
    for tau_mult in tau_multipliers:
        tau_scaled = tau_base.loc[decision_times_reg].ffill().fillna(tau_base.median()) * tau_mult
        for gate in prob_gates:
            prob_reindexed = None
            if probs_series is not None:
                prob_reindexed = probs_series.reindex(decision_times_reg).ffill()
            positions = generate_positions_v5(
                decision_times=decision_times_reg,
                preds_reg=preds_reg_series,
                probs_cls=prob_reindexed,
                tau_series=tau_scaled,
                prob_gate=gate,
                cooldown=args.cooldown,
            )
            pnl_components = pnl_from_positions_v5(positions, fwd_returns.loc[decision_times_reg])
            metrics_dict = dict(pnl_components["metrics"])
            trade_count = metrics_dict.pop("TradeCount", 0)
            metrics_dict.update(
                {
                    "TauMultiplier": tau_mult,
                    "ProbGate": gate,
                    "Trades": trade_count,
                }
            )
            rows.append(metrics_dict)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(report_dir / "tau_sweep_v5.csv", index=False)
    _pretty_tau_table(results_df, "TAU Sweep (bps, v5)", report_dir / "tau_sweep_v5.md")
    print(f"[v5] Saved TAU sweep to {report_dir}")

    fi_model = make_xgb_reg()
    fi_model.fit(X, y_reg)
    get_feature_importance_plot_v5(fi_model, meta["feature_names"], report_dir / "feature_importance_v5.png")

    metrics_summary = pd.DataFrame(
        [
            {"model": "xgb_reg_v5", **reg_metrics},
            {"model": "xgb_cls_v5", **cls_metrics},
        ]
    )
    metrics_summary.to_csv(report_dir / "metrics_summary_v5.csv", index=False)


if __name__ == "__main__":
    main()
