#!/usr/bin/env python3
"""
Is P(up) informative enough to clear transaction costs, and at what horizon?

For each cached proba bar, compute forward log-returns of the ratio at several
horizons, bucket by proba, and report mean directional forward return per
bucket. The answer determines viable hold times and entry thresholds:
a bucket is tradeable only if |mean forward return| comfortably exceeds the
round-trip cost (~0.18% taker / ~0.08% maker).

Usage: python scripts/proba_informativeness.py <probas.parquet> [--label X]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

HORIZONS = [15, 60, 240, 720, 1440]  # bars (minutes)
EDGES = [0.0, 0.40, 0.45, 0.475, 0.50, 0.525, 0.55, 0.60, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", help="probas parquet from fast_backtest.py")
    ap.add_argument("--label", default="informativeness")
    args = ap.parse_args()

    df = pd.read_parquet(args.cache)
    df = df[df["valid"]].reset_index(drop=True)
    logr = np.log(df["ratio"])

    rows = []
    for h in HORIZONS:
        fwd = (logr.shift(-h) - logr) * 100  # forward % return
        buck = pd.cut(df["proba"], EDGES, right=False)
        g = fwd.groupby(buck, observed=True)
        for interval, stats in g.agg(["count", "mean", "std"]).iterrows():
            # Directional: SHORT buckets flip sign so positive = model-aligned edge
            lo = interval.left
            aligned = stats["mean"] if lo >= 0.5 else -stats["mean"]
            rows.append({
                "horizon_bars": h,
                "proba_bucket": str(interval),
                "n": int(stats["count"]),
                "fwd_mean_pct": round(float(stats["mean"]), 4),
                "aligned_edge_pct": round(float(aligned), 4),
                "fwd_std_pct": round(float(stats["std"]), 4),
                "t_stat": round(float(aligned / (stats["std"] / np.sqrt(stats["count"]))), 2)
                if stats["count"] > 1 and stats["std"] > 0 else 0.0,
            })

    out = pd.DataFrame(rows)
    out_dir = ROOT / "reports" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.label}.csv"
    out.to_csv(out_path, index=False)

    print(f"Bars: {len(df):,}  |  taker RT cost ≈ 0.18%, maker RT ≈ 0.08%\n")
    for h in HORIZONS:
        sub = out[out["horizon_bars"] == h]
        print(f"— horizon {h} bars —")
        for _, r in sub.iterrows():
            flag = " ***" if abs(r["aligned_edge_pct"]) > 0.08 and abs(r["t_stat"]) > 2 else ""
            print(f"  p∈{r['proba_bucket']:<15} n={r['n']:>7,} aligned_edge={r['aligned_edge_pct']:+.4f}% "
                  f"t={r['t_stat']:+.2f}{flag}")
        print()
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
