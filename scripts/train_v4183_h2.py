#!/usr/bin/env python3
"""v4.18.3 hypothesis #2 (pre-registered before running, ledger row appended):

Train the 240-bar-label model THROUGH 2025-12-15 (PROGRESS.md recommended
path #2 — capture the late-2025 regime), then apply the SAME decision rule
shape selected by hypothesis #1 with zero new freedom:
  short_thr = 2% quantile of the model's own probas over the last 60 train
  days, longs disabled, max_hold 240, v4.18 guardrails.
ONE evaluation on holdout2026. Same gate as V4183_CANDIDATE.md §3.
"""

import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from api.version_config import V418_CONFIG
from api.feature_calculator import V416SignalGenerator
from fast_backtest import CACHE_DIR, MODEL_DIR, compute_probas, replay, compute_metrics
from train_v4183 import ms, top1_removed_sign_ok, HORIZON_BARS, FEE_BPS, LEDGER

OUT_DIR = ROOT / "models" / "v4_18_3"
EVAL_DIR = ROOT / "reports" / "eval" / "v4183"


def load_klines(symbol: str) -> pd.DataFrame:
    """Concat the cached train + tune parquets into Feb→Dec-15 2025."""
    a = pd.read_parquet(CACHE_DIR / f"klines_{symbol}_{ms('2025-01-25')}_{ms('2025-10-01')}.parquet")
    b = pd.read_parquet(CACHE_DIR / f"klines_{symbol}_1759168800000_1765756800000.parquet")
    df = pd.concat([a, b]).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df


def main() -> None:
    old_model = XGBClassifier()
    old_model.load_model(str(MODEL_DIR / "model.json"))
    feature_names = json.loads((MODEL_DIR / "feature_names.json").read_text())
    xgb_params = json.loads((MODEL_DIR / "config.json").read_text())["xgb_params"]

    print("== training frame Feb 2025 → Dec 15 2025 ==")
    btc, eth = load_klines("BTCUSDT"), load_klines("ETHUSDT")
    cached, frame = compute_probas(btc, eth, old_model, feature_names, return_frame=True)
    ratio = cached["ratio"].to_numpy()
    times = cached["time"].to_numpy()
    fwd = np.full(len(ratio), np.nan)
    fwd[:-HORIZON_BARS] = np.log(ratio[HORIZON_BARS:] / ratio[:-HORIZON_BARS])

    t0 = ms("2025-02-01") // 1000
    t1 = 1765756800 - HORIZON_BARS * 60  # train end = holdout start − horizon (no peek)
    mask = cached["valid"].to_numpy() & ~np.isnan(fwd) & (times >= t0) & (times <= t1)
    X, y = frame[mask], (fwd[mask] > 0).astype(int)
    print(f"   samples={len(X):,} pos_rate={y.mean():.4f}")

    model = XGBClassifier(**xgb_params, eval_metric="auc")
    model.fit(X, y, verbose=False)
    auc_in = roc_auc_score(y, model.predict_proba(X)[:, 1])
    print(f"   AUC in-sample={auc_in:.4f}")

    # Deterministic threshold rule — no selection, no holdout contact
    tail_mask = mask & (times >= (t1 - 60 * 86_400))
    tail_probas = model.predict_proba(frame[tail_mask])[:, 1]
    short_thr = round(float(np.quantile(tail_probas, 0.02)), 4)
    print(f"   short_thr (2% of last-60d train probas) = {short_thr}")

    cfg = replace(V418_CONFIG, entry_threshold_short=short_thr, entry_threshold_long=1.01)

    print("== ONE holdout2026 evaluation ==")
    hb = pd.read_parquet(CACHE_DIR / f"klines_BTCUSDT_1765648800000_1782864000000.parquet")
    he = pd.read_parquet(CACHE_DIR / f"klines_ETHUSDT_1765648800000_1782864000000.parquet")
    hold = compute_probas(hb, he, model, feature_names)
    trades, eq, _ = replay("v4.18.3-h2", hold, FEE_BPS, sig_gen=V416SignalGenerator(cfg=cfg))
    m = compute_metrics(trades, eq, net=True)

    gate = {
        "selected": {"short_thr": short_thr, "long_thr": 1.01},
        "holdout_net_exp_pos": m.get("expectancy_pct", -1) > 0,
        "holdout_pnl_beats_v418": m.get("total_pnl_dollar", -1e9) > -22.0,
        "holdout_n_ge_20": m.get("n_trades", 0) >= 20,
        "holdout_dd_floor": bool(m.get("max_drawdown_pct", -100) > -7.4),
        "top1_removed_sign_ok": top1_removed_sign_ok(trades),
    }
    passed = all(v for k, v in gate.items() if k != "selected")
    verdict = "GATE PASSED — eligible for shadow" if passed else "GATE FAILED — v4.18.3 not activated"

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "summary_h2.json").write_text(json.dumps(
        {"gate": gate, "verdict": verdict, "holdout2026": m,
         "train": {"end": "2025-12-15 minus 240 bars", "auc_in_sample": round(float(auc_in), 4),
                   "samples": int(len(X))}},
        indent=2, default=str))
    with open(LEDGER, "a") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": "v4.18.3-h2-retrain-through-dec2025",
            "hypothesis": "240-bar labels + training through 2025-12 captures the 2026 regime",
            "spec": "scripts/train_v4183_h2.py docstring (deterministic threshold rule, one holdout eval)",
            "gate": gate, "verdict": verdict,
            "artifacts": "reports/eval/v4183/summary_h2.json",
        }, default=str) + "\n")

    print(f"   holdout: n={m.get('n_trades')} exp={m.get('expectancy_pct')}% "
          f"pnl=${m.get('total_pnl_dollar')} pf={m.get('profit_factor')} dd={m.get('max_drawdown_pct')}%")
    print(json.dumps(gate, indent=2, default=str))
    print(f"VERDICT: {verdict}")

    if passed:
        # Only a PASSING model may occupy models/v4_18_3/
        model.save_model(str(OUT_DIR / "model.json"))
        (OUT_DIR / "feature_names.json").write_text(json.dumps(feature_names))
        (OUT_DIR / "config.json").write_text(json.dumps({
            "version": "v4.18.3", "trained_at": datetime.now(timezone.utc).isoformat(),
            "k_ahead": HORIZON_BARS, "train_window": ["2025-02-01", "2025-12-15-minus-240bars"],
            "xgb_params": xgb_params, "auc_in_sample": round(float(auc_in), 4),
            "decision": {"short_thr": short_thr, "long_thr": 1.01, "max_hold_bars": 240},
        }, indent=2))
        print(f"   model written to {OUT_DIR}")


if __name__ == "__main__":
    main()
