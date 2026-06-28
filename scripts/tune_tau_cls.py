import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from features import build_features
from models import evaluate_classification, make_xgb_cls
from backtest import (
    future_k_bar_return,
    positions_from_proba,
    pnl_from_positions,
    ratio_returns_1m,
    summarize_performance,
)

K_AHEAD = 10
COOLDOWN = 20
MIN_HOLD = 4
THRESH_GRID = [(0.55, 0.60), (0.58, 0.63), (0.60, 0.65), (0.62, 0.68)]
REPORT_CSV = "reports/prob_gating_sweep_v4.csv"
REPORT_MD = "reports/prob_gating_sweep_v4.md"


def df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    rows = ["| " + " | ".join(fmt(row[c]) for c in cols) + " |" for _, row in df.iterrows()]
    return "\n".join([head, sep] + rows)


def main():
    os.makedirs("reports", exist_ok=True)
    df = pd.read_parquet("data/processed/btc_eth_ratio_1m.parquet")

    X, y_bps, kept_idx, meta = build_features(df, k_ahead=K_AHEAD, label="bps")
    target_cls = pd.Series(meta.get("y_cls"), dtype=float) if "y_cls" in meta else y_bps

    proba_up, _, metrics, te_idx, _ = evaluate_classification(make_xgb_cls, X, target_cls)
    print("Classification metrics:", metrics)

    time_clean = df.loc[kept_idx, "open_time"].reset_index(drop=True)
    ratio_clean = df.loc[kept_idx, "R"].reset_index(drop=True)
    vol_series = ratio_returns_1m(ratio_clean)
    vol_series.index = time_clean

    decision_times = time_clean.iloc[np.array(te_idx, dtype=int)]
    prob_series = pd.Series(proba_up, index=decision_times)

    fwd_log = future_k_bar_return(ratio_clean, k=K_AHEAD)
    fwd_log.index = time_clean

    rows = []
    for entry, flip in THRESH_GRID:
        positions = positions_from_proba(
            prob_series,
            entry_threshold=entry,
            flip_threshold=flip,
            min_hold=MIN_HOLD,
            cooldown=COOLDOWN,
        )
        pnl_components = pnl_from_positions(positions, fwd_log, vol_series)
        metrics_summary = summarize_performance(pnl_components, positions, fwd_log)
        metrics_summary.update(
            {
                "entry_thresh": entry,
                "flip_thresh": flip,
                "Trades": metrics_summary.pop("TradeCount"),
            }
        )
        rows.append(metrics_summary)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(REPORT_CSV, index=False)
    with open(REPORT_MD, "w") as f:
        f.write(df_to_md(out_df))
    print(f"Saved {REPORT_CSV} and {REPORT_MD}")


if __name__ == "__main__":
    main()
