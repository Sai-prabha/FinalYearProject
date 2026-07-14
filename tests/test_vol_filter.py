"""Toxic-vol regime entry filter (H_regime V2) — V4183_NEXT_CANDIDATE.md.

Default OFF ⇒ zero behavior change. Enabled ⇒ blocks NEW entries in
high-vol regimes, never blocks exits, conservative on unknown history.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.feature_calculator import V416SignalGenerator
from api.version_config import V418_CONFIG

FAST = dict(vol_filter_window=20, vol_filter_pct_window=400, vol_filter_quantile=0.90)
T0 = 1_700_000_000


def drive(gen, ratios, start_bar=0):
    for i, r in enumerate(ratios):
        gen.update(0.5, r, T0 + (start_bar + i) * 60)


def calm_ratios(n, base=35.0, amp=0.0001, seed=1):
    rng = np.random.default_rng(seed)
    return base * np.cumprod(1 + rng.normal(0, amp, n))


def cooling_ratios(n=200, base=35.0, seed=1):
    """High vol early, low vol late — the trailing window is unambiguously
    below its own p90 (a flat-vol series sits above it ~10% of the time by
    construction, which is regime-noise, not a bug)."""
    rng = np.random.default_rng(seed)
    amps = np.concatenate([np.full(n - 50, 0.002), np.full(50, 0.0001)])
    return base * np.cumprod(1 + rng.normal(0, 1, n) * amps)


def test_default_off_is_zero_behavior_change():
    cfg = replace(V418_CONFIG, **FAST)                 # enabled stays False
    gen = V416SignalGenerator(cfg=cfg)
    drive(gen, calm_ratios(150))
    assert gen._vol_regime_toxic() is False            # off ⇒ never toxic
    gen.update(0.40, 35.0, T0 + 200 * 60)              # p ≤ short_thr ⇒ entry
    assert gen.position == -1


def test_insufficient_history_blocks_conservatively():
    cfg = replace(V418_CONFIG, vol_filter_enabled=True, **FAST)
    gen = V416SignalGenerator(cfg=cfg)
    drive(gen, calm_ratios(30))                        # < 4×window history
    assert gen._vol_regime_toxic() is True
    gen.update(0.40, 35.0, T0 + 40 * 60)
    assert gen.position == 0                           # entry refused
    assert "Toxic-vol regime" in gen._get_blocked_by(0.40, True)


def test_calm_allows_and_spike_blocks_entries():
    cfg = replace(V418_CONFIG, vol_filter_enabled=True, **FAST)
    gen = V416SignalGenerator(cfg=cfg)
    cool = cooling_ratios(200)
    drive(gen, cool)
    assert gen._vol_regime_toxic() is False            # cooled regime ⇒ tradable

    # Volatility spike: 25 bars of ±1% swings pushes RV20 over its p90
    spiky = cool[-1] * np.cumprod(1 + np.tile([0.01, -0.01], 13)[:25])
    drive(gen, spiky, start_bar=200)
    assert gen._vol_regime_toxic() is True
    gen.update(0.40, float(spiky[-1]), T0 + 300 * 60)
    assert gen.position == 0                           # blocked in toxic regime


def test_exits_never_blocked_by_regime():
    cfg = replace(V418_CONFIG, vol_filter_enabled=True, **FAST)
    gen = V416SignalGenerator(cfg=cfg)
    cool = cooling_ratios(200)
    drive(gen, cool)
    gen.update(0.40, float(cool[-1]), T0 + 210 * 60)
    assert gen.position == -1                          # entered while calm

    # Toxic spike while IN position: max-hold/time exits must still fire.
    spiky = 35.0 * np.cumprod(1 + np.tile([0.01, -0.01], 20)[:40])
    for i, r in enumerate(spiky):
        gen.update(0.5, float(r), T0 + (211 + i) * 60)
    hold_cfg_bars = cfg.max_hold_bars
    # Drive past max_hold with neutral proba; the position must close even
    # though the regime is toxic (blocking applies to entries only).
    for i in range(hold_cfg_bars + 10):
        gen.update(0.5, float(spiky[-1]), T0 + (260 + i) * 60)
    assert gen.position == 0
    assert len(gen.trades) == 1


def test_stride_quantile_tracks_pandas_reference():
    import pandas as pd
    rng = np.random.default_rng(7)
    rets = rng.normal(0, 0.0005, 2000) * np.where(rng.random(2000) < 0.1, 8, 1)
    w = 20
    cfg = replace(V418_CONFIG, vol_filter_enabled=True, **FAST)
    gen = V416SignalGenerator(cfg=cfg)
    gen._vf_returns.extend(rets[-cfg.vol_filter_pct_window:])
    ours = gen._vf_quantile(np.asarray(gen._vf_returns), w)
    ref = (pd.Series(rets[-cfg.vol_filter_pct_window:]).rolling(w).std()
           .dropna().quantile(0.90))
    assert abs(ours - ref) / ref < 0.15                # same distribution, stride-sampled
