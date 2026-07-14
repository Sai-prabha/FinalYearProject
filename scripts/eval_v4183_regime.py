#!/usr/bin/env python3
"""H_regime evaluation — pre-registered in V4183_NEXT_CANDIDATE.md.

Replays the UNCHANGED v4.18 decision layer over the existing cached
old-model probas, with entry-blocking regime overlays:
  V1 trend-block  : SHORT entries only when ratio < SMA1440
  V2 toxic-vol    : no entries when RV240 > trailing-30d 90th percentile
  V3 = V1 and V2
Selection on tune2025 only; ONE holdout2026 run for the selected variant
through the frozen gate. Entries are blocked by feeding a neutral proba
(0.5) while FLAT in a blocked regime; in-position bars always see the true
proba so exits are untouched.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from api.feature_calculator import V416SignalGenerator
from api.version_config import V418_CONFIG
from fast_backtest import CACHE_DIR, WARMUP, compute_metrics
from train_v4183 import top1_removed_sign_ok

FEE_BPS = 4.5
SMA_N, RV_N, PCT_N, PCT_Q = 1440, 240, 43_200, 0.90
LEDGER = ROOT / "reports" / "experiments.jsonl"
EVAL_DIR = ROOT / "reports" / "eval" / "v4183"

WINDOWS = {
    "tune2025": ("probas_1759168800000_1765756800000.parquet",),
    "holdout2026": ("probas_1765648800000_1782864000000.parquet",),
}


def load(window: str) -> pd.DataFrame:
    df = pd.read_parquet(CACHE_DIR / WINDOWS[window][0])
    r = df["ratio"]
    sma = r.rolling(SMA_N, min_periods=SMA_N).mean()
    rv = np.log(r / r.shift(1)).rolling(RV_N, min_periods=RV_N).std()
    p90 = rv.rolling(PCT_N, min_periods=RV_N * 4).quantile(PCT_Q)
    df["trend_block_short"] = (r >= sma) | sma.isna()      # unknown regime ⇒ block (conservative)
    df["vol_block"] = (rv > p90) | p90.isna()
    return df


def replay_gated(df: pd.DataFrame, variant: str, fee_bps: float = FEE_BPS):
    """v4.18 layer + entry-block overlay. Cost handling identical to fast_backtest.replay."""
    gen = V416SignalGenerator(cfg=V418_CONFIG)
    rt_cost = 4 * fee_bps / 10_000.0
    equity = [gen.balance]
    blocked_entries = 0

    t = df["time"].to_numpy()
    ratio = df["ratio"].to_numpy()
    proba = df["proba"].to_numpy()
    valid = df["valid"].to_numpy()
    tb = df["trend_block_short"].to_numpy()
    vb = df["vol_block"].to_numpy()

    for i in range(WARMUP, len(df)):
        if not valid[i]:
            continue
        p = float(proba[i])
        if gen.position == 0:
            block = False
            if variant in ("V1", "V3") and tb[i] and p <= gen._short_thr:
                block = True
            if variant in ("V2", "V3") and vb[i] and (p <= gen._short_thr or p >= gen._long_thr):
                block = True
            if block:
                blocked_entries += 1
                p = 0.5  # neutral: no entry this bar; exits unaffected (flat)
        n_before = len(gen.trades)
        gen.update(p, float(ratio[i]), int(t[i]))
        for trade in gen.trades[n_before:]:
            notional = abs(trade["pnl_dollar"] / (trade["pnl_pct"] / 100.0)) \
                if trade["pnl_pct"] != 0 else gen.balance * trade["position_size_pct"] / 100.0
            cost = notional * rt_cost
            trade["fee_dollar"] = cost
            trade["pnl_dollar_net"] = trade["pnl_dollar"] - cost
            trade["pnl_pct_net"] = trade["pnl_pct"] - rt_cost * 100.0
        # fees reduce equity exactly as in fast_backtest.replay
        for trade in gen.trades[n_before:]:
            gen.balance -= trade["fee_dollar"]
            gen.total_pnl -= trade["fee_dollar"]
        equity.append(gen.balance)

    return gen.trades, np.asarray(equity), blocked_entries


def main() -> None:
    tune = load("tune2025")
    results = {}

    # Baseline: ungated v4.18 on the same cache (sanity anchor vs PROGRESS numbers)
    tr0, eq0, _ = replay_gated(tune, variant="NONE")
    results["baseline_tune"] = compute_metrics(tr0, eq0, net=True)

    picks = []
    for v in ("V1", "V2", "V3"):
        tr, eq, blocked = replay_gated(tune, v)
        m = compute_metrics(tr, eq, net=True)
        results[f"{v}_tune"] = {**m, "blocked_entry_bars": blocked}
        print(f"tune {v}: n={m.get('n_trades')} exp={m.get('expectancy_pct')}% "
              f"pnl=${m.get('total_pnl_dollar')} dd={m.get('max_drawdown_pct')}% blocked_bars={blocked}")
        if m.get("n_trades", 0) >= 15 and m.get("expectancy_pct", -9) > -0.176:
            picks.append((m["total_pnl_dollar"], v, m))

    if not picks:
        verdict = "GATE NOT REACHED — no variant qualified on tune2025; holdout untouched"
        gate = {"selected": None}
    else:
        picks.sort(reverse=True)
        _, sel, m_tune = picks[0]
        hold = load("holdout2026")
        tr_h, eq_h, blocked_h = replay_gated(hold, sel)
        m_h = compute_metrics(tr_h, eq_h, net=True)
        results["selected"] = sel
        results[f"{sel}_holdout"] = {**m_h, "blocked_entry_bars": blocked_h}
        gate = {
            "selected": sel,
            "holdout_net_exp_pos": m_h.get("expectancy_pct", -1) > 0,
            "holdout_pnl_beats_v418": m_h.get("total_pnl_dollar", -1e9) > -22.0,
            "holdout_n_ge_20": m_h.get("n_trades", 0) >= 20,
            "holdout_dd_floor": bool(m_h.get("max_drawdown_pct", -100) > -7.4),
            "top1_removed_sign_ok": top1_removed_sign_ok(tr_h),
        }
        passed = all(v for k, v in gate.items() if k != "selected")
        verdict = ("GATE PASSED — eligible for live implementation + shadow" if passed
                   else "GATE FAILED — regime variant not activated")
        print(f"\nholdout {sel}: n={m_h.get('n_trades')} exp={m_h.get('expectancy_pct')}% "
              f"pnl=${m_h.get('total_pnl_dollar')} pf={m_h.get('profit_factor')} "
              f"dd={m_h.get('max_drawdown_pct')}% blocked_bars={blocked_h}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "summary_regime.json").write_text(json.dumps(
        {"gate": gate, "verdict": verdict, "results": results}, indent=2, default=str))
    with open(LEDGER, "a") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": "v4.18.3-h3-regime-gate",
            "hypothesis": "short-tail edge is conditional: block shorts above the 24h ratio mean (V1) / in toxic vol (V2/V3)",
            "spec": "V4183_NEXT_CANDIDATE.md (definitions frozen pre-evaluation)",
            "gate": gate, "verdict": verdict,
            "artifacts": "reports/eval/v4183/summary_regime.json",
        }, default=str) + "\n")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
