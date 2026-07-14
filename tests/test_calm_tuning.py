"""Tests for the H_calm tuning machinery (scripts/tune_v4183_calm.py):
objective floors/math, regime conservatism, and the meta-acceptance gate.
Optuna is a research-only dep — skip cleanly where it isn't installed."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("optuna")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from api.version_config import V418_CONFIG
from tune_v4183_calm import WARMUP, add_regime, replay_calm, score_tune


def _trades(pnls, t_start=0, step=60):
    return [{"pnl_pct_net": p, "entry_time": t_start + i * step} for i, p in enumerate(pnls)]


class TestScoreTune:
    def test_rejects_below_n_floor(self):
        score, d = score_tune(_trades([0.5] * 10), np.array([1000.0, 1005.0]), t_mid=300)
        assert score <= -1000 and d["n"] == 10

    def test_rejects_empty_half(self):
        # 20 trades all in half 1
        score, _ = score_tune(_trades([0.1] * 20), np.array([1000.0, 1001.0]), t_mid=10**9)
        assert score <= -1000

    def test_rejects_dd_breach(self):
        eq = np.array([1000.0, 1200.0, 1100.0])  # -8.3% from peak
        score, _ = score_tune(_trades([0.1] * 20, step=60), np.array(eq), t_mid=600)
        assert score <= -1000

    def test_min_half_math(self):
        # half1 (entry_time < 600): 10 trades at +0.2; half2: 10 at +0.05
        tr = _trades([0.2] * 10, t_start=0) + _trades([0.05] * 10, t_start=600)
        score, d = score_tune(tr, np.array([1000.0, 1010.0]), t_mid=600)
        assert score == pytest.approx(0.05)
        assert d["exp_h1"] == pytest.approx(0.2) and d["exp_h2"] == pytest.approx(0.05)


def _cooling_frame(n=2600, seed=7):
    """High-vol first half, near-zero vol tail ⇒ calm regime late, unknown early."""
    rng = np.random.default_rng(seed)
    steps = np.where(np.arange(n) < n // 2, rng.normal(0, 3e-3, n), rng.normal(0, 1e-5, n))
    ratio = 30.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame({"time": np.arange(n) * 60, "ratio": ratio,
                         "proba": np.full(n, 0.5), "valid": np.ones(n, bool)})


class TestAddRegime:
    def test_unknown_history_is_toxic(self):
        df = add_regime(_cooling_frame())
        assert df["vol_block"].to_numpy()[: 240].all()  # RV undefined ⇒ block

    def test_calm_tail_unblocks(self):
        df = add_regime(_cooling_frame())
        assert not df["vol_block"].to_numpy()[-200:].any()  # cooled ⇒ calm


class _StubLogit:
    """predict_proba fixed at 0.5 — cutoff extremes decide acceptance."""
    def predict_proba(self, X):
        return np.tile([0.5, 0.5], (len(X), 1))


class TestMetaGate:
    def _frame(self):
        df = add_regime(_cooling_frame(n=2600))
        p = np.full(len(df), 0.5)
        p[1300:] = 0.30  # persistent SHORT entry signal in the calm tail
        return df, p

    def test_cutoff_above_one_blocks_everything(self):
        df, p = self._frame()
        meta = {"model": _StubLogit(), "mu": np.zeros(3), "sd": np.ones(3), "cutoff": 1.1}
        trades, _, blk = replay_calm(df, p, V418_CONFIG, meta=meta)
        assert trades == [] and blk["blocked_meta"] > 0

    def test_cutoff_zero_is_identity(self):
        df, p = self._frame()
        base_trades, base_eq, _ = replay_calm(df, p, V418_CONFIG)
        meta = {"model": _StubLogit(), "mu": np.zeros(3), "sd": np.ones(3), "cutoff": 0.0}
        trades, eq, blk = replay_calm(df, p, V418_CONFIG, meta=meta)
        assert blk["blocked_meta"] == 0
        assert len(trades) == len(base_trades)
        assert eq[-1] == pytest.approx(base_eq[-1])
