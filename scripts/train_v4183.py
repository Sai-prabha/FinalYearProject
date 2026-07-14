#!/usr/bin/env python3
"""Train + evaluate the v4.18.3 candidate — pre-registered in V4183_CANDIDATE.md.

One variable changes vs v4.14: the label horizon (10-bar -> 240-bar forward
sign of the ratio move). Same 50 features, same XGB hyperparams, same train
span. Threshold selection uses tune2025 ONLY; holdout2026 is evaluated once.

Run:  .venv/bin/python scripts/train_v4183.py
Artifacts: models/v4_18_3/, data/backtest/probas_v4183_*.parquet,
           reports/eval/v4183/, reports/experiments.jsonl (append-only).
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
from fast_backtest import (
    CACHE_DIR, MODEL_DIR, compute_probas, fetch_klines, replay, compute_metrics,
)

HORIZON_BARS = 240
FEE_BPS = 4.5
OUT_DIR = ROOT / "models" / "v4_18_3"
EVAL_DIR = ROOT / "reports" / "eval" / "v4183"
LEDGER = ROOT / "reports" / "experiments.jsonl"

def ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


WINDOWS = {
    # label: (fetch_start_ms, fetch_end_ms) — tune/holdout use the EXACT spans
    # already cached by fast_backtest runs so no refetch happens.
    "train":       (ms("2025-01-25"), ms("2025-10-01")),
    "tune2025":    (1759168800000, 1765756800000),
    "holdout2026": (1765648800000, 1782864000000),
}


def load_window(label: str, model, feature_names, want_frame=False):
    start_ms, end_ms = WINDOWS[label]
    btc = fetch_klines("BTCUSDT", start_ms, end_ms)
    eth = fetch_klines("ETHUSDT", start_ms, end_ms)
    if want_frame:
        cached, frame = compute_probas(btc, eth, model, feature_names, return_frame=True)
        return cached, frame
    cache = CACHE_DIR / f"probas_v4183_{label}.parquet"
    if cache.exists():
        return pd.read_parquet(cache), None
    cached = compute_probas(btc, eth, model, feature_names)
    cached.to_parquet(cache)
    return cached, None


def top1_removed_sign_ok(trades) -> bool:
    """Gate 5: expectancy sign must not depend on the single best trade."""
    if len(trades) < 2:
        return False
    p = sorted((t["pnl_pct_net"] for t in trades), reverse=True)
    full = float(np.mean(p))
    trimmed = float(np.mean(p[1:]))
    return (full > 0) == (trimmed > 0)


def main() -> None:
    old_model = XGBClassifier()
    old_model.load_model(str(MODEL_DIR / "model.json"))
    feature_names = json.loads((MODEL_DIR / "feature_names.json").read_text())
    v414_cfg = json.loads((MODEL_DIR / "config.json").read_text())
    xgb_params = v414_cfg["xgb_params"]

    # ── 1. Training frame + 240-bar labels ────────────────────────────────
    print("== building training frame (Feb–Oct 2025) ==")
    cached_tr, frame_tr = load_window("train", old_model, feature_names, want_frame=True)
    ratio = cached_tr["ratio"].to_numpy()
    times = cached_tr["time"].to_numpy()
    fwd = np.full(len(ratio), np.nan)
    fwd[:-HORIZON_BARS] = np.log(ratio[HORIZON_BARS:] / ratio[:-HORIZON_BARS])

    t0 = ms("2025-02-01") // 1000
    t1 = ms("2025-10-01") // 1000 - HORIZON_BARS * 60  # labels must not peek past train end
    mask = (
        cached_tr["valid"].to_numpy()
        & ~np.isnan(fwd)
        & (times >= t0) & (times <= t1)
    )
    X = frame_tr[mask]
    y = (fwd[mask] > 0).astype(int)
    print(f"   samples={len(X):,}  pos_rate={y.mean():.4f} "
          f"(overlapping 240-bar labels — OOS replay is the real judge)")

    # ── 2. Train (exact v4.14 hyperparams — no search) ────────────────────
    model = XGBClassifier(**xgb_params, eval_metric="auc")
    model.fit(X, y, verbose=False)
    auc_in = roc_auc_score(y, model.predict_proba(X)[:, 1])
    cut = int(len(X) * 0.85)  # chronological last-15% slice, report-only
    auc_tail = roc_auc_score(y[cut:], model.predict_proba(X[cut:])[:, 1]) if len(set(y[cut:])) > 1 else float("nan")
    print(f"   AUC in-sample={auc_in:.4f}  last-15%={auc_tail:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(OUT_DIR / "model.json"))
    (OUT_DIR / "feature_names.json").write_text(json.dumps(feature_names))
    (OUT_DIR / "config.json").write_text(json.dumps({
        "version": "v4.18.3",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "base": "v4_14_production hyperparams + features, 240-bar labels",
        "k_ahead": HORIZON_BARS,
        "train_window": ["2025-02-01", "2025-10-01"],
        "xgb_params": xgb_params,
        "auc_in_sample": round(float(auc_in), 4),
        "auc_last15pct": round(float(auc_tail), 4),
        "label": "sign of 240-bar forward log ratio return, no deadband",
        "strategy_params": v414_cfg["strategy_params"],  # server-boot compatibility
    }, indent=2))

    # ── 3. New-model probas on tune + holdout ─────────────────────────────
    print("== computing candidate probas on tune2025 / holdout2026 ==")
    new_model = XGBClassifier()
    new_model.load_model(str(OUT_DIR / "model.json"))
    tune, _ = load_window("tune2025", new_model, feature_names)
    hold, _ = load_window("holdout2026", new_model, feature_names)
    q = tune.loc[tune["valid"], "proba"].quantile([0.02, 0.05, 0.10, 0.90, 0.95, 0.98])
    print("   tune proba quantiles:", {f"{k:.0%}": round(v, 4) for k, v in q.items()})

    def run(cfg, cached):
        return replay("v4.18.3", cached, FEE_BPS, sig_gen=V416SignalGenerator(cfg=cfg))

    results = {}

    # ── 4. Ablation: v4.18 decision layer untouched, new weights ──────────
    tr_t, eq_t, _ = run(V418_CONFIG, tune)
    tr_h, eq_h, _ = run(V418_CONFIG, hold)
    results["ablation_v418_layer"] = {
        "tune2025": compute_metrics(tr_t, eq_t, net=True),
        "holdout2026": compute_metrics(tr_h, eq_h, net=True),
    }

    # ── 5. Threshold derivation — tune2025 ONLY (quantile grid) ───────────
    grid = []
    for s_q in (0.02, 0.05, 0.10):
        for l_q in (None, 0.90, 0.95, 0.98):
            grid.append({
                "short_thr": round(float(q[s_q]), 4),
                "long_thr": round(float(q[l_q]), 4) if l_q else 1.01,
                "sq": s_q, "lq": l_q,
            })
    best = None
    for g in grid:
        cfg = replace(V418_CONFIG,
                      entry_threshold_short=g["short_thr"],
                      entry_threshold_long=g["long_thr"])
        trades, eq, _ = run(cfg, tune)
        m = compute_metrics(trades, eq, net=True)
        g["tune"] = m
        ok = m.get("n_trades", 0) >= 15
        if ok and (best is None or m["total_pnl_dollar"] > best["tune"]["total_pnl_dollar"]):
            best = g
    results["tune_grid"] = grid

    if best is None:
        verdict = "GATE FAILED (no tune config with n>=15 trades)"
        chosen_cfg = None
        gate = {"selected": None}
    else:
        chosen_cfg = replace(V418_CONFIG,
                             entry_threshold_short=best["short_thr"],
                             entry_threshold_long=best["long_thr"])
        # ── 6. ONE holdout evaluation of the frozen candidate ─────────────
        trades_h, eq_hh, _ = run(chosen_cfg, hold)
        m_h = compute_metrics(trades_h, eq_hh, net=True)
        results["candidate"] = {"thresholds": {k: best[k] for k in ("short_thr", "long_thr")},
                                "tune2025": best["tune"], "holdout2026": m_h}
        gate = {
            "selected": results["candidate"]["thresholds"],
            "holdout_net_exp_pos": m_h.get("expectancy_pct", -1) > 0,
            "holdout_pnl_beats_v418": m_h.get("total_pnl_dollar", -1e9) > -22.0,
            "tune_exp_beats_v418": best["tune"].get("expectancy_pct", -1) > -0.176,
            "holdout_n_ge_20": m_h.get("n_trades", 0) >= 20,
            "holdout_dd_floor": m_h.get("max_drawdown_pct", -100) > -7.4,
            "top1_removed_sign_ok": top1_removed_sign_ok(trades_h),
        }
        passed = all(v for k, v in gate.items() if k != "selected")
        verdict = "GATE PASSED — eligible for shadow" if passed else "GATE FAILED — v4.18.3 not activated"

    # ── 7. Artifacts + append-only experiment ledger ──────────────────────
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "summary.json").write_text(json.dumps(
        {"gate": gate, "verdict": verdict, "results": results}, indent=2, default=str))
    with open(LEDGER, "a") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": "v4.18.3-retrain-240bar-labels",
            "hypothesis": "240-bar labels give the fixed feature set a tradeable 4h edge that clears taker costs",
            "spec": "V4183_CANDIDATE.md (pre-registered)",
            "gate": gate, "verdict": verdict,
            "artifacts": "reports/eval/v4183/summary.json",
        }, default=str) + "\n")

    print(json.dumps({"gate": gate, "verdict": verdict}, indent=2, default=str))
    print("\n== ablation (v4.18 layer, new weights) ==")
    for w in ("tune2025", "holdout2026"):
        m = results["ablation_v418_layer"][w]
        print(f"   {w}: n={m.get('n_trades')} exp={m.get('expectancy_pct')}% pnl=${m.get('total_pnl_dollar')} dd={m.get('max_drawdown_pct')}%")
    if best:
        m = results["candidate"]["holdout2026"]
        print(f"\n== candidate holdout: n={m.get('n_trades')} exp={m.get('expectancy_pct')}% "
              f"pnl=${m.get('total_pnl_dollar')} pf={m.get('profit_factor')} dd={m.get('max_drawdown_pct')}%")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
