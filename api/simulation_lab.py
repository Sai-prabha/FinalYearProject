"""Simulation campaigns — named, repeatable replay grids for decision support.

A campaign replays chosen strategies (strategy-lab spec ids or baseline
versions) across chosen pairs and auto-named scenario windows (full window,
highest-vol week, quietest week) at a fee band. Presentation is
decision-oriented: per-cell metrics plus a per-strategy verdict —
ROBUST | MIXED | CONFIRMS_REJECT — and whether that agrees with the current
Strategy Lab lifecycle.

Governance (RESEARCH_TOOLS.md §4): campaigns are post-hoc checks — results
NEVER feed back into ranking or lifecycle. This module never imports the
broker or the control plane, and shares a mutual-exclusion check with the
research runner so only one heavy replay job runs at a time.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# dynamic module references (not value imports) so the test suite's tiny-lab
# monkeypatches (WARMUP, FAMILIES, …) apply here too
from api import strategy_lab
from api.strategy_lab import DEFAULT_PAIR, PAIR_UNIVERSE, build_candidate_specs, build_dataset

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = ROOT / "data" / "simulation"
CAMPAIGNS_JSON = SIM_DIR / "campaigns.json"

DEFAULT_FEES = (4.5, 2.0)
PRIMARY_FEE = 4.5
DEFAULT_WINDOW_DAYS = 120
MAX_WINDOW_DAYS = 270
MAX_STRATEGIES = 8
MAX_PAIRS = 3
SCENARIO_WEEK_S = 7 * 86400
EQUITY_POINTS = 160

_job_lock = threading.Lock()
_job: Dict = {"state": "idle"}


def job_snapshot() -> Dict:
    with _job_lock:
        return dict(_job)


def _set_job(**fields) -> None:
    with _job_lock:
        _job.update(fields)


def _busy_reason() -> Optional[str]:
    if job_snapshot().get("state") == "running":
        return "A simulation campaign is already running"
    return None


# research runner refuses to start while a campaign runs, and vice versa
strategy_lab.external_busy_checks.append(_busy_reason)


# ── Store ────────────────────────────────────────────────────────────────────

def load_campaigns() -> List[Dict]:
    if not CAMPAIGNS_JSON.exists():
        return []
    try:
        return json.loads(CAMPAIGNS_JSON.read_text())
    except Exception as e:
        logger.warning(f"campaigns.json unreadable: {e}")
        return []


def _save_campaigns(items: List[Dict]) -> None:
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    CAMPAIGNS_JSON.write_text(json.dumps(items, indent=2, default=str))


def _update_campaign(cid: str, **fields) -> None:
    items = load_campaigns()
    for c in items:
        if c["id"] == cid:
            c.update(fields)
    _save_campaigns(items)


def get_campaign(cid: str) -> Optional[Dict]:
    return next((c for c in load_campaigns() if c["id"] == cid), None)


def load_results(cid: str) -> Optional[Dict]:
    path = SIM_DIR / f"{cid}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.warning(f"simulation results {cid} unreadable: {e}")
        return None


# ── Scenarios ────────────────────────────────────────────────────────────────

def detect_scenarios(dataset) -> List[Dict]:
    """Auto-named scenario windows — chosen by rule, not cherry-picked:
    the full window, the highest-realized-vol week, and the quietest week."""
    import numpy as np
    t = dataset["time"].to_numpy()
    rets = dataset["ratio"].pct_change().to_numpy()
    start_s, end_s = float(t[strategy_lab.WARMUP]), float(t[-1])
    scenarios = [{"name": "full-window", "from": int(start_s), "to": int(end_s),
                  "reason": "entire campaign window"}]
    # weekly buckets over the evaluable span
    n_weeks = int((end_s - start_s) // SCENARIO_WEEK_S)
    if n_weeks >= 3:
        vols = []
        for w in range(n_weeks):
            lo = start_s + w * SCENARIO_WEEK_S
            hi = lo + SCENARIO_WEEK_S
            mask = (t >= lo) & (t < hi)
            if mask.sum() < 2000:  # want a mostly-complete week of minutes
                continue
            vols.append((float(np.nanstd(rets[mask])), int(lo), int(hi)))
        if len(vols) >= 2:
            vols.sort()
            quiet, spike = vols[0], vols[-1]
            scenarios.append({"name": "vol-spike-week", "from": spike[1], "to": spike[2],
                              "reason": f"highest realized-vol week (σ {spike[0] * 100:.3f}%/min)"})
            scenarios.append({"name": "quiet-week", "from": quiet[1], "to": quiet[2],
                              "reason": f"quietest week (σ {quiet[0] * 100:.3f}%/min)"})
    return scenarios


def _slice_scenario(dataset, from_s: int, to_s: int):
    """Slice with WARMUP runway before the window so replays start warm."""
    import numpy as np
    t = dataset["time"].to_numpy()
    i0 = int(np.searchsorted(t, from_s))
    i1 = int(np.searchsorted(t, to_s, side="right"))
    return dataset.iloc[max(0, i0 - strategy_lab.WARMUP):i1].reset_index(drop=True)


def _downsample(equity, points: int = EQUITY_POINTS) -> List[float]:
    if len(equity) <= points:
        return [round(float(v), 2) for v in equity]
    step = len(equity) / points
    return [round(float(equity[int(i * step)]), 2) for i in range(points)] + \
           [round(float(equity[-1]), 2)]


# ── Campaign lifecycle ───────────────────────────────────────────────────────

def create_campaign(strategy_ids: List[str], label: Optional[str] = None,
                    pairs: Optional[List[str]] = None,
                    window_days: int = DEFAULT_WINDOW_DAYS,
                    fees: Optional[List[float]] = None,
                    created_by: str = "anonymous", created_via: str = "api",
                    model=None, feature_names=None) -> Dict:
    """Validate, persist, and start a campaign in a worker thread."""
    specs_by_id = {s["id"]: s for s in build_candidate_specs()}
    unknown = [s for s in strategy_ids if s not in specs_by_id]
    if unknown:
        raise ValueError(f"unknown strategy ids: {unknown}")
    if not strategy_ids:
        raise ValueError("strategy_ids must not be empty")
    if len(strategy_ids) > MAX_STRATEGIES:
        raise ValueError(f"at most {MAX_STRATEGIES} strategies per campaign")
    pairs = list(dict.fromkeys(pairs or [DEFAULT_PAIR]))
    bad = [p for p in pairs if p not in PAIR_UNIVERSE]
    if bad:
        raise ValueError(f"unknown pairs {bad} — universe: {sorted(PAIR_UNIVERSE)}")
    if len(pairs) > MAX_PAIRS:
        raise ValueError(f"at most {MAX_PAIRS} pairs per campaign")
    window_days = int(window_days)
    if not (strategy_lab.MIN_DATASET_DAYS <= window_days <= MAX_WINDOW_DAYS):
        raise ValueError(f"window_days must be {strategy_lab.MIN_DATASET_DAYS}–{MAX_WINDOW_DAYS}")
    fees = [float(f) for f in (fees or DEFAULT_FEES)]
    if PRIMARY_FEE not in fees:
        fees.insert(0, PRIMARY_FEE)  # verdicts are judged at taker cost

    busy = _busy_reason() or (
        "A research job is already running"
        if strategy_lab.job_snapshot().get("state") == "running" else None)
    if busy:
        raise RuntimeError(busy)

    cid = "sim-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    meta = {
        "id": cid,
        "label": label or f"campaign {cid}",
        "strategy_ids": strategy_ids,
        "pairs": pairs,
        "window_days": window_days,
        "fees": fees,
        "created_by": created_by,
        "created_via": created_via,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "finished_at": None,
        "error": None,
    }
    items = load_campaigns()
    items.insert(0, meta)
    _save_campaigns(items)
    with _job_lock:
        _job.clear()
        _job.update({"state": "running", "id": cid, "phase": "starting",
                     "done": 0, "total": 0})
    strategy_lab._audit("simulation_started", campaign=cid, label=meta["label"],
                        strategies=strategy_ids, pairs=pairs,
                        window_days=window_days, actor=created_by, via=created_via)
    t = threading.Thread(target=_run_campaign_sync,
                         args=(cid, [specs_by_id[s] for s in strategy_ids], pairs,
                               window_days, fees, model, feature_names),
                         daemon=True, name=f"simulation-{cid}")
    t.start()
    return meta


def _run_campaign_sync(cid: str, specs: List[Dict], pairs: List[str],
                       window_days: int, fees: List[float],
                       model, feature_names) -> None:
    from scripts.fast_backtest import compute_metrics, replay
    t0 = time.time()
    try:
        datasets = {}
        for pk in pairs:
            _set_job(phase=f"building dataset · {pk}")
            datasets[pk] = build_dataset(window_days, model=model,
                                         feature_names=feature_names, pair_key=pk)
        scenario_map = {pk: detect_scenarios(ds) for pk, ds in datasets.items()}
        total = sum(len(scenario_map[pk]) * len(fees) for pk in pairs) * len(specs)
        _set_job(total=total)

        lab_store = strategy_lab.load_candidates() or {}
        lab_by_id = {c["id"]: c for c in lab_store.get("candidates", [])}

        cells: List[Dict] = []
        verdicts: List[Dict] = []
        done = 0
        for spec in specs:
            per_scenario_taker: List[float] = []
            for pk in pairs:
                for scen in scenario_map[pk]:
                    sliced = _slice_scenario(datasets[pk], scen["from"], scen["to"])
                    for fee in fees:
                        _set_job(phase=f"{spec['label']} · {pk} · {scen['name']} · {fee:g}bps",
                                 done=done)
                        done += 1
                        trades, equity, _ = replay(spec["id"], sliced, fee,
                                                   sig_gen=strategy_lab._make_gen(spec))
                        m = compute_metrics(trades, equity, net=True)
                        exp = m.get("expectancy_pct")
                        if fee == PRIMARY_FEE:
                            per_scenario_taker.append(exp if exp is not None else 0.0)
                        cells.append({
                            "strategy_id": spec["id"],
                            "pair": pk,
                            "scenario": scen["name"],
                            "fee_bps": fee,
                            "n_trades": m.get("n_trades", 0),
                            "exp_pct": exp,
                            "profit_factor": m.get("profit_factor"),
                            "max_dd_pct": strategy_lab._equity_max_dd_pct(equity) if len(equity) else None,
                            "sharpe": m.get("sharpe"),
                            "total_pnl_dollar": m.get("total_pnl_dollar"),
                            "equity": _downsample(equity) if fee == PRIMARY_FEE else None,
                        })
            pos = sum(1 for e in per_scenario_taker if e > 0)
            if pos == len(per_scenario_taker) and per_scenario_taker:
                verdict = "ROBUST"
            elif pos == 0:
                verdict = "CONFIRMS_REJECT"
            else:
                verdict = "MIXED"
            lab_row = lab_by_id.get(spec["id"])
            lab_lifecycle = lab_row.get("lifecycle") if lab_row else None
            consistent = None
            # baselines aren't gated — agreement only means something for
            # candidates that actually carry a lab verdict
            if lab_lifecycle in ("REJECTED", "MATCH", "DEPLOY_CANDIDATE"):
                consistent = (verdict == "CONFIRMS_REJECT") == (lab_lifecycle == "REJECTED")
            verdicts.append({
                "strategy_id": spec["id"],
                "label": spec["label"],
                "family": spec["family"],
                "verdict": verdict,
                "positive_scenarios": pos,
                "total_scenarios": len(per_scenario_taker),
                "lab_lifecycle": lab_lifecycle,
                "consistent_with_lab": consistent,
            })

        result = {
            "id": cid,
            "generated": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(time.time() - t0, 1),
            "pairs": pairs,
            "fees": fees,
            "scenarios": {pk: scenario_map[pk] for pk in pairs},
            "verdicts": verdicts,
            "cells": cells,
        }
        SIM_DIR.mkdir(parents=True, exist_ok=True)
        (SIM_DIR / f"{cid}.json").write_text(json.dumps(result, indent=2, default=str))
        _update_campaign(cid, status="done",
                         finished_at=datetime.now(timezone.utc).isoformat())
        strategy_lab._audit("simulation_completed", campaign=cid,
                            duration_s=result["duration_s"],
                            verdicts={v["strategy_id"]: v["verdict"] for v in verdicts})
        _set_job(state="done", phase="done", done=done)
    except Exception as e:
        logger.exception(f"Simulation campaign {cid} failed")
        _update_campaign(cid, status="failed", error=str(e),
                         finished_at=datetime.now(timezone.utc).isoformat())
        strategy_lab._audit("simulation_failed", campaign=cid, error=str(e))
        _set_job(state="failed", error=str(e))
