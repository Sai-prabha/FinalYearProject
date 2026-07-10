#!/usr/bin/env python3
"""
Sensitivity sweep around the v4.18 conviction-gate config on BOTH windows.

Purpose: confirm the chosen (short_thr, max_hold) sits on a robustness
plateau rather than a lucky point. Reports net expectancy / PF / PnL per
combo per window. Not a tuner — the config was chosen from the
informativeness tables; this just checks the neighbourhood.

Usage: python scripts/sweep_v418.py
"""
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from api.version_config import V418_CONFIG
from api.feature_calculator import V416SignalGenerator
from scripts.fast_backtest import compute_metrics, replay, CACHE_DIR, WARMUP

WINDOWS = {
    "tune2025": CACHE_DIR / "probas_1759168800000_1765756800000.parquet",
    "holdout2026": CACHE_DIR / "probas_1765648800000_1782864000000.parquet",
}
FEE_BPS = 4.5

SHORT_THRS = [0.44, 0.45, 0.46]
MAX_HOLDS = [120, 240, 360]


def main():
    caches = {name: pd.read_parquet(p) for name, p in WINDOWS.items()}
    rows = []
    for st in SHORT_THRS:
        for mh in MAX_HOLDS:
            cfg = replace(V418_CONFIG, entry_threshold_short=st, max_hold_bars=mh)
            for wname, cached in caches.items():
                trades, equity, _ = replay(
                    None, cached, FEE_BPS,
                    sig_gen=V416SignalGenerator(cfg=cfg),
                )
                m = compute_metrics(trades, equity, net=True)
                rows.append({
                    "short_thr": st, "max_hold": mh, "window": wname,
                    "n": m.get("n_trades", 0),
                    "net_exp_pct": m.get("expectancy_pct"),
                    "net_pf": m.get("profit_factor"),
                    "net_pnl": m.get("total_pnl_dollar"),
                    "max_dd_pct": m.get("max_drawdown_pct"),
                    "win_rate": m.get("win_rate_pct"),
                    "wl_ratio": m.get("win_loss_ratio"),
                })

    df = pd.DataFrame(rows)
    out = ROOT / "reports" / "eval" / "sweep_v418.csv"
    df.to_csv(out, index=False)
    for wname in WINDOWS:
        print(f"\n== {wname} (NET, fee {FEE_BPS}bps/side x4) ==")
        sub = df[df["window"] == wname]
        print(sub.drop(columns="window").to_string(index=False))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
