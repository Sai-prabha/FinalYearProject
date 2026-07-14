"""Timing advisor — when is it worth running strategy research, and why.

Advisory only (RESEARCH_TOOLS.md §2): it recommends and explains, it never
starts anything. Every reason is a measurable condition:

- **Data completeness**: just after the UTC daily close, yesterday's candles
  are final — replays end on a complete day (00:10–01:30 UTC window).
- **Weekly liquidity trough**: Saturday morning UTC is the quietest window
  (same rationale as the scheduler's Sat 05:00 deep scan) — cheap time for a
  long multi-pair or deep run (05:00–08:00 UTC).
- **Volatility regime shift**: realized vol of the live pair's ratio over the
  last 24h vs its 30-day baseline. A ratio > 1.5 (elevated) or < 0.67 (quiet)
  since the last successful run means the leaderboard was ranked under a
  different regime — re-ranking is informative NOW.
- **Staleness**: no successful research run in 7+ days.

Alert-fatigue rule: few windows, each with an explicit reason and a
confidence; thresholds are multiples of the pair's own baseline, not fixed
percentages.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DAILY_WINDOW = ((0, 10), (1, 30))     # post-UTC-close completeness window
DEEP_WINDOW = ((5, 0), (8, 0))        # Saturday liquidity trough
DEEP_WEEKDAY = 5
STALENESS_DAYS = 7
VOL_HIGH = 1.5
VOL_LOW = 0.67
VOL_BASELINE_DAYS = 30
HORIZON_HOURS = 48

_cache_lock = threading.Lock()
_cache: Dict = {"ts": 0.0, "result": None}


# ── Volatility state from the cached proba stream (no network) ──────────────

def ratio_vol_state(now_ts: Optional[float] = None) -> Optional[Dict]:
    """Realized vol of the default pair's ratio: last 24h vs 30d baseline.
    Reads the existing proba parquet cache only — degrades to None without it."""
    try:
        import pandas as pd
        from scripts.fast_backtest import CACHE_DIR
    except Exception:
        return None
    now_ts = now_ts or time.time()
    start_s = now_ts - VOL_BASELINE_DAYS * 86400
    frames = []
    for p in sorted(CACHE_DIR.glob("probas_*.parquet")):
        try:
            s_ms, e_ms = (int(x) for x in p.stem.split("_")[1:3])
        except (ValueError, IndexError):
            continue  # pair-prefixed caches — the advisor watches the live pair
        if e_ms / 1000 < start_s:
            continue
        try:
            frames.append(pd.read_parquet(p, columns=["time", "ratio"]))
        except Exception:
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).drop_duplicates("time").sort_values("time")
    df = df[df["time"] >= start_s]
    if len(df) < 3000:  # need a real baseline (~2+ days of minutes)
        return None
    rets = df["ratio"].pct_change().to_numpy()[1:]
    t = df["time"].to_numpy()[1:]
    # anchor the "recent" window to the DATA's end, not the wall clock — the
    # cache ends at the last complete UTC day by construction
    data_end = float(t[-1])
    last24 = rets[t >= data_end - 86400]
    if len(last24) < 300 or len(rets) < 2 * len(last24):
        return None
    import numpy as np
    r24 = float(np.nanstd(last24))
    r30 = float(np.nanstd(rets))
    if not (r30 > 0 and math.isfinite(r24) and math.isfinite(r30)):
        return None
    ratio = r24 / r30
    regime = "elevated" if ratio > VOL_HIGH else "quiet" if ratio < VOL_LOW else "normal"
    return {
        "r24h_pct": round(r24 * 100, 4),
        "r30d_pct": round(r30 * 100, 4),
        "ratio": round(ratio, 3),
        "regime": regime,
        "data_to": int(df["time"].max()),
    }


# ── Windows ─────────────────────────────────────────────────────────────────

def _clock_windows(now: datetime) -> List[Dict]:
    """Upcoming (and currently open) recurring windows in the next 48h."""
    out = []
    for offset in range(3):
        day = (now + timedelta(days=offset)).date()
        specs = [("daily-close", DAILY_WINDOW,
                  "yesterday's UTC candles are complete — replays end on a full day", 0.55)]
        if day.weekday() == DEEP_WEEKDAY:
            specs.append(("weekly-trough", DEEP_WINDOW,
                          "Saturday liquidity trough — quiet window for a deep or multi-pair run", 0.7))
        for kind, ((h1, m1), (h2, m2)), reason, conf in specs:
            start = datetime(day.year, day.month, day.day, h1, m1, tzinfo=timezone.utc)
            end = datetime(day.year, day.month, day.day, h2, m2, tzinfo=timezone.utc)
            if end <= now or start > now + timedelta(hours=HORIZON_HOURS):
                continue
            out.append({"from": start.isoformat(), "to": end.isoformat(),
                        "kind": kind, "reason": reason, "confidence": conf})
    return sorted(out, key=lambda w: w["from"])


def advise(now: Optional[datetime] = None,
           last_success: Optional[Dict] = None,
           next_scheduled: Optional[Dict] = None,
           vol: Optional[Dict] = "auto") -> Dict:
    """Full advisory document. ``vol`` may be injected for tests."""
    now = now or datetime.now(timezone.utc)
    if vol == "auto":
        vol = ratio_vol_state(now.timestamp())

    windows = _clock_windows(now)
    reasons: List[str] = []
    confidence = 0.0

    last_ts: Optional[datetime] = None
    if last_success and last_success.get("ts"):
        try:
            last_ts = datetime.fromisoformat(last_success["ts"])
        except ValueError:
            pass

    # staleness — always-on trigger, strongest signal
    if last_ts is None:
        reasons.append("no successful research run recorded yet")
        confidence = max(confidence, 0.8)
    else:
        age_days = (now - last_ts).total_seconds() / 86400
        if age_days >= STALENESS_DAYS:
            reasons.append(f"last successful run was {age_days:.1f} days ago (≥ {STALENESS_DAYS}d)")
            confidence = max(confidence, 0.8)

    # volatility regime shift since the last run
    if vol and vol.get("regime") in ("elevated", "quiet") and last_ts is not None:
        shift = (f"24h realized vol is {vol['ratio']}× the 30d baseline "
                 f"({vol['regime']} regime)")
        reasons.append(shift + " — the leaderboard was ranked under a different regime")
        confidence = max(confidence, 0.6)
        windows.insert(0, {
            "from": now.isoformat(),
            "to": (now + timedelta(hours=2)).isoformat(),
            "kind": "regime-shift", "reason": shift, "confidence": 0.6,
        })

    # inside a recurring window right now?
    now_iso = now.isoformat()
    for w in windows:
        if w["from"] <= now_iso <= w["to"] and w["kind"] in ("daily-close", "weekly-trough"):
            reasons.append(w["reason"])
            confidence = max(confidence, w["confidence"])

    # scheduler proximity: if the scheduler will fire within 2h, don't nag
    sched_soon = False
    if next_scheduled and next_scheduled.get("next_run"):
        try:
            nxt = datetime.fromisoformat(next_scheduled["next_run"])
            sched_soon = timedelta(0) <= (nxt - now) <= timedelta(hours=2)
        except ValueError:
            pass
    if sched_soon and reasons:
        reasons.append("note: the scheduler runs within 2h and will cover this")
        confidence = min(confidence, 0.4)

    return {
        "now": {
            "recommended": bool(reasons) and not sched_soon,
            "reasons": reasons,
            "confidence": round(confidence, 2),
        },
        "windows": windows,
        "vol": vol,
        "last_success": last_success,
        "next_scheduled": next_scheduled,
        "generated": now_iso,
    }


def advise_cached(max_age_s: float = 300, **kwargs) -> Dict:
    """Cheap accessor for the WS payload — recomputes at most every 5 min."""
    with _cache_lock:
        if _cache["result"] is not None and time.time() - _cache["ts"] < max_age_s:
            return _cache["result"]
    result = advise(**kwargs)
    with _cache_lock:
        _cache.update(ts=time.time(), result=result)
    return result
