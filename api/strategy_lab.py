"""
Strategy Lab — research jobs, walk-forward ranking, deploy-candidate gating,
shadow registration and research scheduling.

Design record: STRATEGY_LAB.md (repo root). The short version:

- A "strategy" is a StrategyConfig variant over the frozen v4.14 proba stream —
  exactly how v4.15…v4.18 already differ. Search = small, explainable config
  families replayed through scripts/fast_backtest.replay (parity-verified
  against the live path) with realistic fees.
- Evaluation: continuous validation replay sliced into 5 contiguous folds for
  regime consistency + a reserved holdout replayed cold-start, only for the
  top-5 gate passers (nobody mines the holdout).
- "Deploy-candidate" is defined by explicit gates; every failure is recorded
  as a machine-readable reason for the UI's "why rejected" explainer.
- Shadow registration reuses the existing shadow slot (never a second scheme);
  the execution path structurally never reads shadow state.

This module never imports model_server (model_server imports us) and never
touches the broker.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Callable, Dict, List, Optional, Tuple

from api.version_config import StrategyConfig, get_strategy_config

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "v4_14_production"
RESEARCH_DIR = ROOT / "data" / "research"
CANDIDATES_JSON = RESEARCH_DIR / "candidates.json"
STATE_JSON = RESEARCH_DIR / "state.json"
AUDIT_JSONL = RESEARCH_DIR / "audit.jsonl"
SHADOW_REGISTRATION_JSON = ROOT / "data" / "live" / "shadow_registration.json"

WARMUP = 1000            # must match scripts.fast_backtest.WARMUP
STARTING_BALANCE = 1000.0
HOLDOUT_DAYS = 56        # reserved out-of-sample tail (8 weeks)
N_FOLDS = 5
FEE_GRID = (4.5, 2.0, 0.0)   # taker / maker-ish / frictionless sensitivity band
PRIMARY_FEE = 4.5
MIN_DATASET_DAYS = HOLDOUT_DAYS + 60
_EULER_GAMMA = 0.5772156649015329

# ── Search space ────────────────────────────────────────────────────────────
# Small and explainable by design: every trial raises the deflated-Sharpe bar
# for all trials, so families only grow with a written rationale.

FAMILIES: Dict[str, Dict] = {
    "conviction": {
        "base": "v4.18",
        "rationale": (
            "Trade only the probability tails, hold to a time horizon. The only "
            "edge that has ever cleared taker fees here is the p<0.45 tail at a "
            "~4h horizon (reports/eval 2026-07-08)."
        ),
        "grid": {
            "side": ["short", "both"],
            "tail": [0.44, 0.45, 0.46],
            "max_hold": [120, 240, 480],
        },
    },
    "band": {
        "base": "v4.15",
        "rationale": (
            "Control group: symmetric entry/exit band churn (v4.15/16 lineage), "
            "known cost-hostile at taker fees. These exist to prove the gates "
            "reject fragile strategies."
        ),
        "grid": {
            "entry": [0.525, 0.535, 0.545],
            "exit": [0.51, 0.505],
            "min_hold": [25, 40],
        },
    },
}

BASELINE_VERSIONS = ["v4.15", "v4.16", "v4.17", "v4.18"]


def _spec_id(family: str, params: Dict) -> str:
    digest = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:6]
    return f"{family}-{digest}"


def _spec_label(family: str, params: Dict) -> str:
    if family == "conviction":
        hours = params["max_hold"] / 60
        side = f"short p≤{params['tail']}" if params["side"] == "short" \
            else f"p≤{params['tail']} / p≥{round(1 - params['tail'], 4)}"
        return f"conviction {side} · {hours:g}h hold"
    return f"band {params['entry']}/{params['exit']} · hold {params['min_hold']}"


def build_candidate_specs() -> List[Dict]:
    """Enumerate the full (small) search space plus baseline reference rows."""
    specs: List[Dict] = []
    for family, fam in FAMILIES.items():
        keys = list(fam["grid"].keys())
        combos: List[Dict] = [{}]
        for k in keys:
            combos = [dict(c, **{k: v}) for c in combos for v in fam["grid"][k]]
        for params in combos:
            specs.append({
                "id": _spec_id(family, params),
                "role": "candidate",
                "family": family,
                "base_version": fam["base"],
                "params": params,
                "label": _spec_label(family, params),
            })
    for v in BASELINE_VERSIONS:
        specs.append({
            "id": v, "role": "baseline", "family": "baseline",
            "base_version": v, "params": {}, "label": f"{v} (registered)",
        })
    return specs


def spec_config(spec: Dict) -> StrategyConfig:
    """Rebuild the StrategyConfig for a candidate spec (deterministic)."""
    base = get_strategy_config(spec["base_version"])
    p = spec["params"]
    if spec["family"] == "conviction":
        long_thr = round(1 - p["tail"], 4) if p["side"] == "both" else 1.01
        return dataclasses.replace(
            base,
            entry_threshold_short=p["tail"],
            entry_threshold_long=long_thr,
            max_hold_bars=p["max_hold"],
        )
    if spec["family"] == "band":
        return dataclasses.replace(
            base,
            entry_threshold=p["entry"],
            exit_threshold=p["exit"],
            min_hold=p["min_hold"],
        )
    return base  # baseline rows


def _make_gen(spec: Dict):
    """Fresh signal generator for a spec. Baselines use the exact live builder."""
    from scripts.fast_backtest import make_signal_gen
    from api.feature_calculator import V416SignalGenerator
    if spec["role"] == "baseline":
        return make_signal_gen(spec["base_version"])
    return V416SignalGenerator(cfg=spec_config(spec))


# ── Dataset ─────────────────────────────────────────────────────────────────

def _load_model():
    from xgboost import XGBClassifier
    m = XGBClassifier()
    m.load_model(str(MODEL_DIR / "model.json"))
    names = json.loads((MODEL_DIR / "feature_names.json").read_text())
    return m, names


def build_dataset(lookback_days: int, model=None, feature_names=None,
                  fetch_missing: bool = True, now: Optional[datetime] = None):
    """(time, ratio, proba, valid) frame for the last N days ending at the last
    complete UTC day. Reuses every cached proba parquet that overlaps; computes
    (and caches) only the missing tail from public Binance klines."""
    import pandas as pd
    from scripts.fast_backtest import CACHE_DIR, compute_probas, fetch_klines

    now = now or datetime.now(timezone.utc)
    end_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=lookback_days)
    start_s = int(start_dt.timestamp())
    end_s = int(end_dt.timestamp())
    head_s = start_s - (WARMUP + 800) * 60  # replay warm-up runway

    frames = []
    for p in sorted(CACHE_DIR.glob("probas_*.parquet")):
        try:
            s_ms, e_ms = (int(x) for x in p.stem.split("_")[1:3])
        except (ValueError, IndexError):
            continue
        if e_ms // 1000 < head_s or s_ms // 1000 > end_s:
            continue
        frames.append(pd.read_parquet(p))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["time", "ratio", "proba", "valid"])
    if len(df):
        df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
        df = df[df["time"] >= head_s].reset_index(drop=True)

    last_s = int(df["time"].max()) if len(df) else None
    tail_missing = last_s is None or last_s < end_s - 86400
    if tail_missing and fetch_missing:
        if model is None:
            model, feature_names = _load_model()
        fetch_start_s = (last_s - 1800 * 60) if last_s else head_s
        logger.info(f"Strategy lab: fetching klines {fetch_start_s} → {end_s}")
        btc = fetch_klines("BTCUSDT", fetch_start_s * 1000, end_s * 1000)
        eth = fetch_klines("ETHUSDT", fetch_start_s * 1000, end_s * 1000)
        fresh = compute_probas(btc, eth, model, feature_names)
        fresh.to_parquet(CACHE_DIR / f"probas_{fetch_start_s * 1000}_{end_s * 1000}.parquet")
        df = pd.concat([df, fresh], ignore_index=True)
        df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
        df = df[df["time"] >= head_s].reset_index(drop=True)

    if not len(df):
        raise RuntimeError("No proba data available — cache empty and fetch disabled/failed")
    coverage_days = (df["time"].max() - df["time"].min()) / 86400
    if coverage_days < MIN_DATASET_DAYS:
        raise RuntimeError(
            f"Dataset covers only {coverage_days:.0f} days; "
            f"need ≥ {MIN_DATASET_DAYS} (holdout {HOLDOUT_DAYS} + 60 validation)")
    return df


# ── Evaluation ──────────────────────────────────────────────────────────────

def _equity_max_dd_pct(equity) -> float:
    peak, max_dd = equity[0], 0.0
    for v in equity:
        peak = max(peak, v)
        max_dd = min(max_dd, (v - peak) / peak)
    return round(max_dd * 100, 3)


def _fold_slices(trades: List[Dict], t_start: float, t_end: float, n_folds: int) -> List[Dict]:
    edges = [t_start + (t_end - t_start) * i / n_folds for i in range(n_folds + 1)]
    folds = []
    for i in range(n_folds):
        lo, hi = edges[i], edges[i + 1]
        pcts = [t["pnl_pct_net"] for t in trades if lo <= (t.get("exit_time") or 0) < hi]
        exp = round(sum(pcts) / len(pcts), 4) if pcts else None
        folds.append({
            "i": i + 1,
            "from": int(lo), "to": int(hi),
            "n": len(pcts),
            "exp_pct": exp,
            "positive": bool(pcts) and exp > 0,
        })
    return folds


def expected_max_sharpe(sharpes: List[float]) -> float:
    """E[max SR] across N independent trials under the null (Bailey & López de
    Prado deflated-Sharpe benchmark). 0 when there is nothing to deflate."""
    n = len(sharpes)
    if n < 2:
        return 0.0
    mean = sum(sharpes) / n
    var = sum((s - mean) ** 2 for s in sharpes) / (n - 1)
    if var <= 0:
        return 0.0
    nd = NormalDist()
    return math.sqrt(var) * (
        (1 - _EULER_GAMMA) * nd.inv_cdf(1 - 1 / n)
        + _EULER_GAMMA * nd.inv_cdf(1 - 1 / (n * math.e))
    )


def _score(exp_pct: Optional[float], n: int, consistency: float, dd_pct: float) -> float:
    """Robust ranking score: shrunk edge × fold consistency × drawdown factor.
    Fragile or inconsistent edges score ≈ 0; the gates decide, the score ranks.
    The consistency floor keeps losing candidates ordered by how badly they
    lose instead of collapsing them all to zero."""
    if exp_pct is None:
        return 0.0
    shrunk = exp_pct * n / (n + 30)          # 30-trade prior toward zero
    dd_factor = max(0.05, 1 + dd_pct / 100)  # dd_pct is negative
    return round(shrunk * max(consistency, 0.2) * dd_factor, 6) + 0.0


def _neighbors(spec: Dict, all_specs: List[Dict]) -> List[str]:
    """IDs of grid points differing in exactly one param by one grid step."""
    fam = FAMILIES.get(spec["family"])
    if not fam:
        return []
    out = []
    for other in all_specs:
        if other["family"] != spec["family"] or other["id"] == spec["id"]:
            continue
        diffs = [k for k in fam["grid"] if other["params"][k] != spec["params"][k]]
        if len(diffs) != 1:
            continue
        k = diffs[0]
        vals = fam["grid"][k]
        if abs(vals.index(other["params"][k]) - vals.index(spec["params"][k])) == 1:
            out.append(other["id"])
    return out


def evaluate_run(dataset, specs: List[Dict], live_version: str,
                 progress: Optional[Callable[[str, int, int], None]] = None) -> Dict:
    """Replay every spec over the validation span (+ fee sensitivity), gate,
    rank, and holdout-test the top-5 gate passers. Pure computation — no I/O."""
    import numpy as np
    from scripts.fast_backtest import compute_metrics, replay

    times = dataset["time"].to_numpy()
    end_s = float(times[-1])
    holdout_start_s = end_s - HOLDOUT_DAYS * 86400
    h_idx = int(np.searchsorted(times, holdout_start_s))
    if h_idx <= WARMUP:
        raise RuntimeError("Dataset too small for the holdout split")
    val = dataset.iloc[:h_idx].reset_index(drop=True)
    hold = dataset.iloc[h_idx - WARMUP:].reset_index(drop=True)
    val_t_start = float(times[WARMUP])

    total = len(specs)
    rows: List[Dict] = []
    for k, spec in enumerate(specs):
        if progress:
            progress(f"replaying {spec['label']}", k, total)
        fees = {}
        primary = None
        for fee in FEE_GRID:
            trades, equity, _ = replay(spec["id"], val, fee, sig_gen=_make_gen(spec))
            m = compute_metrics(trades, equity, net=True)
            fees[f"{fee:g}"] = m.get("expectancy_pct")
            if fee == PRIMARY_FEE:
                primary = (trades, equity, m)
        trades, equity, m = primary
        folds = _fold_slices(trades, val_t_start, holdout_start_s, N_FOLDS)
        consistency = sum(1 for f in folds if f["positive"]) / N_FOLDS
        dd_pct = _equity_max_dd_pct(equity)
        n = m.get("n_trades", 0)
        exp = m.get("expectancy_pct")
        dols = [t["pnl_dollar_net"] for t in trades]
        gross_profit = sum(d for d in dols if d > 0)
        concentration = (max(dols) / gross_profit) if (dols and gross_profit > 0) else None
        rows.append({
            **spec,
            "metrics": {
                "n_trades": n,
                "exp_pct": exp,
                "profit_factor": m.get("profit_factor"),
                "sharpe": m.get("sharpe"),
                "max_dd_pct": dd_pct,
                "total_pnl_dollar": m.get("total_pnl_dollar"),
                "win_rate_pct": m.get("win_rate_pct"),
                "avg_bars_held": m.get("avg_bars_held"),
            },
            "folds": folds,
            "consistency": consistency,
            "fees": fees,
            "concentration": round(concentration, 3) if concentration is not None else None,
            "score": _score(exp, n, consistency, dd_pct),
        })

    # Deflated-Sharpe benchmark across the actual family trials of this run
    trial_sharpes = [r["metrics"]["sharpe"] or 0.0 for r in rows if r["role"] == "candidate"]
    sr_benchmark = round(expected_max_sharpe(trial_sharpes), 3)

    by_id = {r["id"]: r for r in rows}
    baseline_row = by_id.get(live_version)
    baseline_score = baseline_row["score"] if baseline_row else 0.0

    # Gates (candidates only; every failure becomes a "why rejected" reason)
    for r in rows:
        if r["role"] != "candidate":
            continue
        m = r["metrics"]
        neigh = _neighbors(r, rows)
        neigh_scores = sorted(by_id[i]["score"] for i in neigh if i in by_id)
        neigh_median = neigh_scores[len(neigh_scores) // 2] if neigh_scores else None
        real_folds = [f for f in r["folds"] if f["n"] >= 3]
        worst_fold = min((f["exp_pct"] for f in real_folds), default=None)
        gates = {
            "edge": {"pass": (m["exp_pct"] or 0) > 0, "value": m["exp_pct"],
                     "need": "net expectancy > 0 after fees"},
            "trades": {"pass": m["n_trades"] >= 30, "value": m["n_trades"],
                       "need": "≥ 30 closed trades in validation"},
            "consistency": {"pass": r["consistency"] >= 0.6, "value": r["consistency"],
                            "need": "≥ 3/5 folds net-positive"},
            "worst_fold": {"pass": worst_fold is not None and worst_fold > -0.10,
                           "value": worst_fold, "need": "no fold below −0.10%/trade"},
            "concentration": {"pass": r["concentration"] is not None and r["concentration"] < 0.4,
                              "value": r["concentration"],
                              "need": "best trade < 40% of gross profit"},
            "dsr": {"pass": (m["sharpe"] or 0) > sr_benchmark, "value": m["sharpe"],
                    "need": f"Sharpe > {sr_benchmark} (E[max] of {len(trial_sharpes)} trials)"},
            "neighborhood": {"pass": neigh_median is None or neigh_median > 0,
                             "value": neigh_median,
                             "need": "median neighbor score > 0 (plateau, not cliff)"},
        }
        r["gates"] = gates
        r["pre_holdout_pass"] = all(g["pass"] for g in gates.values())

    # Holdout: cold-start replay, ONLY for the top-5 pre-holdout passers
    eligible = sorted(
        (r for r in rows if r["role"] == "candidate" and r.get("pre_holdout_pass")),
        key=lambda r: r["score"], reverse=True)
    for j, r in enumerate(eligible):
        if j < 5:
            if progress:
                progress(f"holdout {r['label']}", total, total)
            trades, equity, _ = replay(r["id"], hold, PRIMARY_FEE, sig_gen=_make_gen(r))
            hm = compute_metrics(trades, equity, net=True)
            ok = (hm.get("expectancy_pct") or 0) > 0 and (hm.get("profit_factor") or 0) >= 1.0
            r["holdout"] = {
                "evaluated": True, "pass": ok,
                "n_trades": hm.get("n_trades", 0),
                "exp_pct": hm.get("expectancy_pct"),
                "profit_factor": hm.get("profit_factor"),
                "max_dd_pct": _equity_max_dd_pct(equity) if len(equity) else None,
            }
        else:
            r["holdout"] = {"evaluated": False, "pass": False,
                            "note": "outside top-5 — holdout stays unmined"}
    for r in rows:
        if r["role"] == "candidate" and "holdout" not in r:
            r["holdout"] = {"evaluated": False, "pass": False}

    # Status + reasons
    for r in rows:
        if r["role"] != "candidate":
            r["lifecycle"] = "BASELINE"
            r["status_reasons"] = []
            continue
        reasons = [f"{name}: {g['need']} (got {g['value']})"
                   for name, g in r["gates"].items() if not g["pass"]]
        if r["pre_holdout_pass"] and not r["holdout"]["pass"]:
            reasons.append(
                "holdout: " + (r["holdout"].get("note")
                               or f"failed out-of-sample (exp {r['holdout'].get('exp_pct')}%, "
                                  f"PF {r['holdout'].get('profit_factor')})"))
        if not reasons:
            r["lifecycle"] = "DEPLOY_CANDIDATE" if r["score"] > baseline_score else "MATCH"
            if r["lifecycle"] == "MATCH":
                r["status_reasons"] = [
                    f"passes all gates but does not beat the live baseline "
                    f"({live_version} score {baseline_score})"]
            else:
                r["status_reasons"] = []
        else:
            r["lifecycle"] = "REJECTED"
            r["status_reasons"] = reasons

    ranked = sorted(rows, key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(ranked):
        r["rank"] = i + 1

    return {
        "trials": len(trial_sharpes),
        "sr_benchmark": sr_benchmark,
        "baseline_version": live_version,
        "baseline_score": baseline_score,
        "dataset": {
            "from": int(times[0]), "to": int(end_s),
            "bars": int(len(dataset)),
            "holdout_from": int(holdout_start_s),
            "folds": N_FOLDS,
            "fee_bps_per_side": PRIMARY_FEE,
        },
        "candidates": ranked,
    }


# ── Persistence + audit ─────────────────────────────────────────────────────

def _audit(event: str, **fields) -> None:
    try:
        RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
        row = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        with open(AUDIT_JSONL, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except Exception as e:  # audit must never take the pipeline down
        logger.warning(f"Strategy lab audit write failed: {e}")


def load_candidates() -> Optional[Dict]:
    if not CANDIDATES_JSON.exists():
        return None
    try:
        return json.loads(CANDIDATES_JSON.read_text())
    except Exception as e:
        logger.warning(f"candidates.json unreadable: {e}")
        return None


_STICKY_LIFECYCLES = {"SHADOW_RUNNING", "SHADOW_REJECTED", "PROMOTED"}


def _persist_run(result: Dict, run_id: str, trigger: str, depth: str) -> Dict:
    """Merge fresh evaluation with prior lifecycle state and write the store.
    Candidates already in shadow (or beyond) keep their lifecycle — a metrics
    refresh must not silently un-shadow a running evaluation."""
    prev = load_candidates() or {}
    prev_by_id = {c["id"]: c for c in prev.get("candidates", [])}
    for c in result["candidates"]:
        old = prev_by_id.get(c["id"])
        if old and old.get("lifecycle") in _STICKY_LIFECYCLES:
            c["backtest_verdict"] = c["lifecycle"]  # what this run would have said
            c["lifecycle"] = old["lifecycle"]
            c["shadow"] = old.get("shadow")
    store = {
        "run_id": run_id,
        "generated": datetime.now(timezone.utc).isoformat(),
        "trigger": trigger,
        "depth": depth,
        **result,
    }
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATES_JSON.write_text(json.dumps(store, indent=2, default=str))
    return store


def _save_state(**fields) -> None:
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    state.update(fields)
    STATE_JSON.write_text(json.dumps(state, indent=2, default=str))


def load_state() -> Dict:
    if STATE_JSON.exists():
        try:
            return json.loads(STATE_JSON.read_text())
        except Exception:
            pass
    return {}


# ── Research job (one at a time, worker thread) ─────────────────────────────

_job_lock = threading.Lock()
_job: Dict = {"state": "idle"}


def job_snapshot() -> Dict:
    with _job_lock:
        return dict(_job)


def _set_job(**fields) -> None:
    with _job_lock:
        _job.update(fields)


DEPTHS = {"daily": 180, "deep": 270}  # lookback days


def start_research_job(trigger: str, depth: str = "deep",
                       model=None, feature_names=None, live_version: str = "v4.15") -> Dict:
    """Kick off a research run in a worker thread. Raises RuntimeError if one
    is already running (single-job guard — research never competes with itself)."""
    if depth not in DEPTHS:
        raise ValueError(f"depth must be one of {sorted(DEPTHS)}")
    with _job_lock:
        if _job.get("state") == "running":
            raise RuntimeError("A research job is already running")
        run_id = "r" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        _job.clear()
        _job.update({
            "state": "running", "run_id": run_id, "trigger": trigger, "depth": depth,
            "phase": "starting", "done": 0, "total": 0,
            "started": datetime.now(timezone.utc).isoformat(), "finished": None, "error": None,
        })
    _audit("run_started", run_id=run_id, trigger=trigger, depth=depth)
    t = threading.Thread(
        target=_run_research_sync,
        args=(run_id, trigger, depth, model, feature_names, live_version),
        daemon=True, name=f"strategy-lab-{run_id}")
    t.start()
    return job_snapshot()


def _run_research_sync(run_id: str, trigger: str, depth: str,
                       model, feature_names, live_version: str) -> None:
    t0 = time.time()
    try:
        _set_job(phase="building dataset")
        dataset = build_dataset(DEPTHS[depth], model=model, feature_names=feature_names)
        specs = build_candidate_specs()
        _set_job(total=len(specs))

        def progress(phase: str, done: int, total: int) -> None:
            _set_job(phase=phase, done=done, total=total)

        result = evaluate_run(dataset, specs, live_version, progress=progress)
        _set_job(phase="ranking + persisting")
        store = _persist_run(result, run_id, trigger, depth)
        duration = round(time.time() - t0, 1)
        counts = {}
        for c in store["candidates"]:
            counts[c["lifecycle"]] = counts.get(c["lifecycle"], 0) + 1
        top = [{"id": c["id"], "label": c["label"], "score": c["score"],
                "lifecycle": c["lifecycle"]}
               for c in store["candidates"][:5]]
        _save_state(last_success={
            "ts": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
            "duration_s": duration, "trials": store["trials"],
            "depth": depth, "trigger": trigger, "counts": counts, "top": top,
        })
        _audit("run_completed", run_id=run_id, duration_s=duration,
               trials=store["trials"], counts=counts, top=top)
        _set_job(state="done", phase="done", finished=datetime.now(timezone.utc).isoformat())
    except Exception as e:
        logger.exception(f"Research run {run_id} failed")
        _audit("run_failed", run_id=run_id, error=str(e))
        _set_job(state="failed", error=str(e),
                 finished=datetime.now(timezone.utc).isoformat())


# ── Scheduler ───────────────────────────────────────────────────────────────
# Crypto has no session close; the natural boundaries are the UTC day (daily
# kline close = data completeness) and the weekly liquidity trough. BTC's
# lowest volume/volatility sits in the 00:00–06:00 UTC window (vol minimum
# ≈ 05:00 UTC) and weekends are quieter than weekdays — see STRATEGY_LAB.md.

DAILY_UTC = (4, 30)      # daily refresh 04:30 UTC
DEEP_UTC = (5, 0)        # Saturday deep scan 05:00 UTC
DEEP_WEEKDAY = 5         # Saturday
MIN_HOURS_BETWEEN_RUNS = 20

_sched: Dict = {"auto": None, "next_run": None, "next_kind": None}


def next_scheduled_run(now: datetime) -> Tuple[datetime, str]:
    """Next research window ≥ now: Saturday 05:00 UTC deep scan, 04:30 UTC
    daily refresh every other day (pure function — unit-tested)."""
    for offset in range(8):
        day = (now + timedelta(days=offset)).date()
        deep = day.weekday() == DEEP_WEEKDAY
        h, m = DEEP_UTC if deep else DAILY_UTC
        at = datetime(day.year, day.month, day.day, h, m, tzinfo=timezone.utc)
        if at > now:
            return at, ("deep" if deep else "daily")
    raise AssertionError("unreachable")


def scheduler_snapshot() -> Dict:
    snap = dict(_sched)
    snap["last_success"] = load_state().get("last_success")
    return snap


async def scheduler_loop(get_model: Callable[[], Tuple], get_live_version: Callable[[], str]):
    """Background research scheduler. RESEARCH_AUTO=0 disables it entirely."""
    import asyncio
    auto = os.environ.get("RESEARCH_AUTO", "1") != "0"
    _sched["auto"] = auto
    if not auto:
        logger.info("Strategy lab scheduler disabled (RESEARCH_AUTO=0)")
        return
    logger.info("Strategy lab scheduler active: daily 04:30 UTC, deep scan Sat 05:00 UTC")
    while True:
        now = datetime.now(timezone.utc)
        nxt, kind = next_scheduled_run(now)
        _sched.update(next_run=nxt.isoformat(), next_kind=kind)
        await asyncio.sleep(max(1.0, (nxt - now).total_seconds()))
        last = load_state().get("last_success")
        recent = False
        if last:
            try:
                age_h = (datetime.now(timezone.utc)
                         - datetime.fromisoformat(last["ts"])).total_seconds() / 3600
                recent = age_h < MIN_HOURS_BETWEEN_RUNS
            except Exception:
                pass
        if kind == "daily" and recent:
            logger.info("Strategy lab: skipping daily run (last success < 20h ago)")
        else:
            try:
                model, feature_names = get_model()
                start_research_job(trigger=f"scheduler:{kind}", depth=kind if kind in DEPTHS else "deep",
                                   model=model, feature_names=feature_names,
                                   live_version=get_live_version())
            except RuntimeError as e:
                logger.info(f"Strategy lab: scheduled run skipped ({e})")
            except Exception:
                logger.exception("Strategy lab: scheduled run failed to start")
        await asyncio.sleep(90)  # step past the trigger minute


# ── Shadow registration + promotion readiness ───────────────────────────────

def get_candidate(cand_id: str) -> Optional[Dict]:
    store = load_candidates()
    if not store:
        return None
    return next((c for c in store["candidates"] if c["id"] == cand_id), None)


def set_lifecycle(cand_id: str, lifecycle: str, reason: Optional[str] = None) -> None:
    store = load_candidates()
    if not store:
        return
    for c in store["candidates"]:
        if c["id"] == cand_id:
            c["lifecycle"] = lifecycle
            if lifecycle == "SHADOW_RUNNING":
                c["shadow"] = {"since": datetime.now(timezone.utc).isoformat()}
    CANDIDATES_JSON.write_text(json.dumps(store, indent=2, default=str))
    _audit("lifecycle", candidate=cand_id, lifecycle=lifecycle, reason=reason)


def shadow_version_name(cand_id: str) -> str:
    return f"lab-{cand_id}"


def write_shadow_registration(cand: Dict, trigger: str) -> Dict:
    reg = {
        "candidate_id": cand["id"],
        "label": cand["label"],
        "version": shadow_version_name(cand["id"]),
        "family": cand["family"],
        "base_version": cand["base_version"],
        "params": cand["params"],
        "registered_ts": datetime.now(timezone.utc).isoformat(),
        "trigger": trigger,
    }
    SHADOW_REGISTRATION_JSON.parent.mkdir(parents=True, exist_ok=True)
    SHADOW_REGISTRATION_JSON.write_text(json.dumps(reg, indent=2))
    _audit("shadow_registered", candidate=cand["id"], version=reg["version"], trigger=trigger)
    return reg


def read_shadow_registration() -> Optional[Dict]:
    if not SHADOW_REGISTRATION_JSON.exists():
        return None
    try:
        return json.loads(SHADOW_REGISTRATION_JSON.read_text())
    except Exception as e:
        logger.warning(f"shadow_registration.json unreadable: {e}")
        return None


def clear_shadow_registration(reason: str) -> None:
    reg = read_shadow_registration()
    if SHADOW_REGISTRATION_JSON.exists():
        SHADOW_REGISTRATION_JSON.unlink()
    if reg:
        _audit("shadow_registration_cleared", candidate=reg.get("candidate_id"), reason=reason)


def registration_config(reg: Dict) -> StrategyConfig:
    """Rebuild the StrategyConfig for a persisted registration (restart path)."""
    return spec_config({
        "family": reg["family"], "base_version": reg["base_version"],
        "params": reg["params"], "role": "candidate",
    })


READINESS_MIN_DAYS = 14
READINESS_MIN_TRADES = 5


def promotion_readiness(since_ts: Optional[float], shadow_stats: Dict,
                        primary_window_stats: Dict, now_ts: Optional[float] = None) -> Dict:
    """Continuous promotion-readiness evaluation for whatever runs in the shadow
    slot. Readiness is a signal, never an action — going live stays behind the
    deliberate /model/promote flow."""
    now_ts = now_ts or time.time()
    days = (now_ts - since_ts) / 86400 if since_ts else 0.0
    closed = shadow_stats.get("n") or 0
    exp = shadow_stats.get("expectancy_pct")
    p_exp = primary_window_stats.get("expectancy_pct")
    beats = exp is not None and (p_exp is None or exp >= p_exp)
    checks = [
        {"name": "observation", "pass": days >= READINESS_MIN_DAYS,
         "value": round(days, 1), "need": f"≥ {READINESS_MIN_DAYS} days in shadow"},
        {"name": "sample", "pass": closed >= READINESS_MIN_TRADES,
         "value": closed, "need": f"≥ {READINESS_MIN_TRADES} closed shadow trades"},
        {"name": "edge", "pass": exp is not None and exp >= 0,
         "value": exp, "need": "shadow net expectancy ≥ 0"},
        {"name": "vs_live", "pass": beats,
         "value": {"shadow": exp, "primary": p_exp},
         "need": "shadow expectancy ≥ primary over the matched window"},
    ]
    return {"ready": all(c["pass"] for c in checks), "checks": checks}
