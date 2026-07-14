#!/usr/bin/env python3
"""H_calm — conditional fitting + disciplined Optuna HPO.
Pre-registered in V4183_CALM_META_OPTUNA.md (2026-07-14). Frozen there:
regime/window/label definitions, the 8-dim search space, TPE seed 42,
n_trials=40, no pruner, the min-over-two-tune-halves net-expectancy
objective with hard floors, the H_B logit meta family (no search,
cutoff grid {0.40..0.60}), selection on tune2025 only, ONE holdout2026
evaluation through the unchanged gate.

The holdout parquet is loaded ONLY inside run_holdout(), which executes
once, after selection. Nothing in the Optuna objective can see it.

Run:  .venv/bin/python scripts/tune_v4183_calm.py
Artifacts: reports/eval/v4183/{optuna_calm.db,optuna_trials_calm.jsonl,
           summary_calm.json}; reports/experiments.jsonl (append-only);
           models/v4_18_3/ ONLY on gate pass.
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
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from api.feature_calculator import V416SignalGenerator
from api.version_config import V418_CONFIG
from fast_backtest import CACHE_DIR, MODEL_DIR, WARMUP, compute_probas, fetch_klines, compute_metrics
from train_v4183 import ms, top1_removed_sign_ok

FEE_BPS = 4.5
HORIZON_BARS = 240
SMA_N, RV_N, PCT_N, PCT_Q = 1440, 240, 43_200, 0.90
N_TRIALS = 40                      # pre-registered; do not raise after results
META_CUTOFFS = (0.40, 0.45, 0.50, 0.55, 0.60)
EVAL_DIR = ROOT / "reports" / "eval" / "v4183"
LEDGER = ROOT / "reports" / "experiments.jsonl"
TRIAL_LOG = EVAL_DIR / "optuna_trials_calm.jsonl"

WINDOWS = {
    "train":       (ms("2025-01-25"), ms("2025-10-01")),
    "tune2025":    (1759168800000, 1765756800000),
    "holdout2026": (1765648800000, 1782864000000),
}


# ── Data plumbing ──────────────────────────────────────────────────────────

def add_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Frozen H3 regime series: calm = RV240 <= trailing-30d p90 (unknown ⇒ toxic)."""
    r = df["ratio"]
    rv = np.log(r / r.shift(1)).rolling(RV_N, min_periods=RV_N).std()
    p90 = rv.rolling(PCT_N, min_periods=RV_N * 4).quantile(PCT_Q)
    sma = r.rolling(SMA_N, min_periods=SMA_N).mean()
    df["vol_block"] = ((rv > p90) | p90.isna()).to_numpy()
    df["vol_headroom"] = (rv / p90).to_numpy()
    df["trend_gap"] = np.log(r / sma).to_numpy()
    return df


def load_window(label: str, old_model, feature_names):
    """(cached old-model probas + regime cols, feature frame) — both parquet-cached."""
    start_ms, end_ms = WINDOWS[label]
    pc = CACHE_DIR / f"probas_old_{label}_calm.parquet"
    fc = CACHE_DIR / f"frame_{label}_calm.parquet"
    if pc.exists() and fc.exists():
        return pd.read_parquet(pc), pd.read_parquet(fc)
    btc = fetch_klines("BTCUSDT", start_ms, end_ms)
    eth = fetch_klines("ETHUSDT", start_ms, end_ms)
    cached, frame = compute_probas(btc, eth, old_model, feature_names, return_frame=True)
    cached = add_regime(cached)
    cached.to_parquet(pc)
    frame.to_parquet(fc)
    return cached, frame


# ── Replay: calm entry gate (H3 semantics) + optional meta gate ────────────

def replay_calm(cached: pd.DataFrame, probas: np.ndarray, cfg,
                meta=None, fee_bps: float = FEE_BPS):
    """v4.18 layer, entries blocked while flat in toxic vol; exits untouched.
    meta = dict(model, mu, sd, cutoff, thr): logit acceptance gate on calm
    entry signals. Fee handling identical to fast_backtest.replay."""
    gen = V416SignalGenerator(cfg=cfg)
    rt_cost = 4 * fee_bps / 10_000.0
    equity = [gen.balance]
    blocked_vol = blocked_meta = 0
    thr = cfg.entry_threshold_short

    t = cached["time"].to_numpy()
    ratio = cached["ratio"].to_numpy()
    valid = cached["valid"].to_numpy()
    vb = cached["vol_block"].to_numpy()
    vh = cached["vol_headroom"].to_numpy()
    tg = cached["trend_gap"].to_numpy()

    for i in range(WARMUP, len(cached)):
        if not valid[i]:
            continue
        p = float(probas[i])
        if gen.position == 0:
            if vb[i]:
                blocked_vol += 1
                p = 0.5
            elif meta is not None and p <= thr:
                x = (np.array([thr - p, vh[i], tg[i]]) - meta["mu"]) / meta["sd"]
                if float(meta["model"].predict_proba(x.reshape(1, -1))[:, 1][0]) < meta["cutoff"]:
                    blocked_meta += 1
                    p = 0.5
        n_before = len(gen.trades)
        gen.update(p, float(ratio[i]), int(t[i]))
        for trade in gen.trades[n_before:]:
            notional = abs(trade["pnl_dollar"] / (trade["pnl_pct"] / 100.0)) \
                if trade["pnl_pct"] != 0 else gen.balance * trade["position_size_pct"] / 100.0
            trade["fee_dollar"] = notional * rt_cost
            trade["pnl_dollar_net"] = trade["pnl_dollar"] - trade["fee_dollar"]
            trade["pnl_pct_net"] = trade["pnl_pct"] - rt_cost * 100.0
            gen.balance -= trade["fee_dollar"]
            gen.total_pnl -= trade["fee_dollar"]
        equity.append(gen.balance)

    return gen.trades, np.asarray(equity), {"blocked_vol": blocked_vol, "blocked_meta": blocked_meta}


def score_tune(trades, equity, t_mid: int):
    """Pre-registered objective: min(net exp half1, net exp half2), hard floors."""
    n = len(trades)
    peak, dd = equity[0], 0.0
    for v in equity:
        peak = max(peak, v)
        dd = min(dd, (v - peak) / peak)
    h1 = [tr["pnl_pct_net"] for tr in trades if tr["entry_time"] < t_mid]
    h2 = [tr["pnl_pct_net"] for tr in trades if tr["entry_time"] >= t_mid]
    detail = {"n": n, "n_h1": len(h1), "n_h2": len(h2), "max_dd_pct": round(dd * 100, 3)}
    if n < 15 or len(h1) < 3 or len(h2) < 3:
        return -1000.0 - max(0, 15 - n), detail
    if dd * 100 <= -6.0:
        return -1000.0 - abs(dd * 100), detail
    detail["exp_h1"], detail["exp_h2"] = round(float(np.mean(h1)), 4), round(float(np.mean(h2)), 4)
    return min(detail["exp_h1"], detail["exp_h2"]), detail


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    old_model = XGBClassifier()
    old_model.load_model(str(MODEL_DIR / "model.json"))
    feature_names = json.loads((MODEL_DIR / "feature_names.json").read_text())
    base_params = json.loads((MODEL_DIR / "config.json").read_text())["xgb_params"]

    print("== loading windows (train + tune only; holdout stays sealed) ==")
    tr_cached, tr_frame = load_window("train", old_model, feature_names)
    tu_cached, tu_frame = load_window("tune2025", old_model, feature_names)
    tu_probas_old = tu_cached["proba"].to_numpy()
    tu_valid_calm = tu_cached["valid"].to_numpy() & ~tu_cached["vol_block"].to_numpy()
    replayed_times = tu_cached["time"].to_numpy()[WARMUP:]
    t_mid = int(replayed_times[len(replayed_times) // 2])

    # Sanity anchor: base calm-gated v4.18 must reproduce H3's V2 tune numbers
    tr0, eq0, blk0 = replay_calm(tu_cached, tu_probas_old, V418_CONFIG)
    m0 = compute_metrics(tr0, eq0, net=True)
    print(f"   anchor v4.18+calm tune: n={m0.get('n_trades')} exp={m0.get('expectancy_pct')}% "
          f"(H3 recorded n=50, -0.116%)")

    # ── H_A training data: calm train bars, 240-bar labels ────────────────
    ratio_tr = tr_cached["ratio"].to_numpy()
    times_tr = tr_cached["time"].to_numpy()
    fwd = np.full(len(ratio_tr), np.nan)
    fwd[:-HORIZON_BARS] = np.log(ratio_tr[HORIZON_BARS:] / ratio_tr[:-HORIZON_BARS])
    t0 = ms("2025-02-01") // 1000
    t1 = ms("2025-10-01") // 1000 - HORIZON_BARS * 60
    mask = (tr_cached["valid"].to_numpy() & ~tr_cached["vol_block"].to_numpy()
            & ~np.isnan(fwd) & (times_tr >= t0) & (times_tr <= t1))
    X, y = tr_frame[mask], (fwd[mask] > 0).astype(int)
    print(f"   H_A calm train samples={len(X):,} pos_rate={y.mean():.4f}")

    def build_model(params: dict) -> XGBClassifier:
        m = XGBClassifier(**{**base_params, **params, "random_state": 42},
                          eval_metric="logloss", n_jobs=-1)
        m.fit(X, y, verbose=False)
        return m

    def eval_ha(model: XGBClassifier, short_q: float):
        pr = np.full(len(tu_cached), np.nan)
        v = tu_cached["valid"].to_numpy()
        pr[v] = model.predict_proba(tu_frame[v])[:, 1]
        thr = float(np.quantile(pr[tu_valid_calm], short_q))
        cfg = replace(V418_CONFIG, entry_threshold_short=round(thr, 4), entry_threshold_long=1.01)
        trades, eq, blk = replay_calm(tu_cached, pr, cfg)
        score, detail = score_tune(trades, eq, t_mid)
        return score, {**detail, "thr": round(thr, 4), **blk}, cfg

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_categorical("max_depth", [2, 3, 4]),
            "n_estimators": trial.suggest_int("n_estimators", 20, 120, step=10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
        }
        short_q = trial.suggest_categorical("short_q", [0.02, 0.05, 0.10])
        score, detail, _ = eval_ha(build_model(params), short_q)
        for k, v in detail.items():
            trial.set_user_attr(k, v)
        with open(TRIAL_LOG, "a") as f:
            f.write(json.dumps({"trial": trial.number, "params": {**params, "short_q": short_q},
                                "score": score, **detail}) + "\n")
        return score

    print(f"== Optuna study: TPE seed 42, n_trials={N_TRIALS}, no pruner ==")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name="v4183-h4-calm", direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=f"sqlite:///{EVAL_DIR / 'optuna_calm.db'}", load_if_exists=True)
    remaining = N_TRIALS - len(study.trials)   # crash-resume must not exceed the budget
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    best = study.best_trial
    print(f"   H_A best: trial {best.number} score={best.value:.4f} attrs={best.user_attrs}")

    # ── H_B: logit meta-acceptance (no search) ─────────────────────────────
    print("== H_B: meta training on train-window calm-gated trades ==")
    tr_trades, _, _ = replay_calm(tr_cached, tr_cached["proba"].to_numpy(), V418_CONFIG)
    by_time = {int(t): i for i, t in enumerate(tr_cached["time"].to_numpy())}
    thr_b = V418_CONFIG.entry_threshold_short
    vh_tr = tr_cached["vol_headroom"].to_numpy()
    tg_tr = tr_cached["trend_gap"].to_numpy()
    pr_tr = tr_cached["proba"].to_numpy()
    rows, labels = [], []
    for trd in tr_trades:
        i = by_time.get(int(trd["entry_time"]))
        if i is None or np.isnan(vh_tr[i]) or np.isnan(tg_tr[i]):
            continue
        rows.append([thr_b - pr_tr[i], vh_tr[i], tg_tr[i]])
        labels.append(1 if trd["pnl_pct_net"] > 0 else 0)
    Xm, ym = np.array(rows), np.array(labels)
    print(f"   meta samples={len(Xm)} pos_rate={ym.mean():.3f}")
    hb_results = []
    if len(Xm) >= 30 and 0 < ym.mean() < 1:
        mu, sd = Xm.mean(axis=0), Xm.std(axis=0) + 1e-12
        logit = LogisticRegression(C=1.0, max_iter=1000).fit((Xm - mu) / sd, ym)
        for cut in META_CUTOFFS:
            meta = {"model": logit, "mu": mu, "sd": sd, "cutoff": cut, "thr": thr_b}
            trades, eq, blk = replay_calm(tu_cached, tu_probas_old, V418_CONFIG, meta=meta)
            score, detail = score_tune(trades, eq, t_mid)
            hb_results.append({"cutoff": cut, "score": score, **detail, **blk})
            print(f"   H_B cut={cut}: score={score:.4f} {detail}")
    else:
        print("   H_B skipped: insufficient/degenerate meta sample")
        logit = mu = sd = None

    # ── Selection (tune only) ──────────────────────────────────────────────
    hb_best = max(hb_results, key=lambda r: r["score"]) if hb_results else None
    winner = None
    if best.value > -1000 or (hb_best and hb_best["score"] > -1000):
        winner = "H_A" if best.value >= (hb_best["score"] if hb_best else -1e9) else "H_B"
    print(f"== selection: winner={winner} "
          f"(H_A={best.value:.4f}, H_B={hb_best['score'] if hb_best else None}) ==")

    summary = {
        "pre_registration": "V4183_CALM_META_OPTUNA.md",
        "anchor_v418_calm_tune": m0,
        "ha_best_trial": {"number": best.number, "value": best.value,
                          "params": best.params, "attrs": best.user_attrs},
        "hb_results": hb_results, "winner": winner,
    }

    # ── ONE holdout shot (only function that touches the holdout window) ──
    def run_holdout():
        ho_cached, ho_frame = load_window("holdout2026", old_model, feature_names)
        if winner == "H_A":
            model = build_model({k: v for k, v in best.params.items() if k != "short_q"})
            pr = np.full(len(ho_cached), np.nan)
            v = ho_cached["valid"].to_numpy()
            pr[v] = model.predict_proba(ho_frame[v])[:, 1]
            cfg = replace(V418_CONFIG, entry_threshold_short=best.user_attrs["thr"],
                          entry_threshold_long=1.01)
            return replay_calm(ho_cached, pr, cfg) + (model,)
        meta = {"model": logit, "mu": mu, "sd": sd, "cutoff": hb_best["cutoff"], "thr": thr_b}
        return replay_calm(ho_cached, ho_cached["proba"].to_numpy(), V418_CONFIG, meta=meta) + (None,)

    if winner is None:
        verdict = "GATE NOT REACHED — no family cleared tune floors; holdout untouched"
        gate = {"selected": None}
    else:
        trades_h, eq_h, blk_h, model_h = run_holdout()
        m_h = compute_metrics(trades_h, eq_h, net=True)
        summary["holdout"] = {**m_h, **blk_h}
        gate = {
            "selected": winner,
            "config": best.params if winner == "H_A" else {"cutoff": hb_best["cutoff"]},
            "holdout_net_exp_pos": m_h.get("expectancy_pct", -1) > 0,
            "holdout_pnl_beats_v418": m_h.get("total_pnl_dollar", -1e9) > -22.0,
            "holdout_n_ge_20": m_h.get("n_trades", 0) >= 20,
            "holdout_dd_floor": bool(m_h.get("max_drawdown_pct", -100) > -7.4),
            "top1_removed_sign_ok": top1_removed_sign_ok(trades_h),
        }
        passed = all(v for k, v in gate.items() if k not in ("selected", "config"))
        verdict = ("GATE PASSED — eligible for shadow wiring" if passed
                   else "GATE FAILED — v4.18.3 not activated")
        print(f"\nholdout {winner}: n={m_h.get('n_trades')} exp={m_h.get('expectancy_pct')}% "
              f"pnl=${m_h.get('total_pnl_dollar')} pf={m_h.get('profit_factor')} "
              f"dd={m_h.get('max_drawdown_pct')}% {blk_h}")
        if passed and winner == "H_A":
            out = ROOT / "models" / "v4_18_3"
            out.mkdir(parents=True, exist_ok=True)
            model_h.save_model(str(out / "model.json"))
            (out / "feature_names.json").write_text(json.dumps(feature_names))
            (out / "config.json").write_text(json.dumps({
                "version": "v4.18.3", "trained_at": datetime.now(timezone.utc).isoformat(),
                "base": "calm-conditional retrain, Optuna-tuned (study v4183-h4-calm)",
                "optuna": {"trial": best.number, "params": best.params, "score": best.value},
                "xgb_params": {**base_params, **{k: v for k, v in best.params.items() if k != "short_q"}},
            }, indent=2))
        # gate fail ⇒ nothing persisted to models/ (weights lived only in memory)

    summary["gate"], summary["verdict"] = gate, verdict
    (EVAL_DIR / "summary_calm.json").write_text(json.dumps(summary, indent=2, default=str))
    with open(LEDGER, "a") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": "v4.18.3-h4-calm-fit",
            "hypothesis": "signal quality inside calm regimes improves via conditional fitting (H_A calm retrain + 40-trial Optuna) or meta-acceptance (H_B logit)",
            "spec": "V4183_CALM_META_OPTUNA.md (pre-registered)",
            "gate": gate, "verdict": verdict,
            "artifacts": "reports/eval/v4183/{summary_calm.json,optuna_calm.db,optuna_trials_calm.jsonl}",
        }, default=str) + "\n")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
