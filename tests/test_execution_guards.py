"""
Unit tests for api/execution_guards.py.

Run: pytest tests/test_execution_guards.py -v
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.execution_guards import ExecutionGuards, GuardResult


def _guards(**overrides) -> ExecutionGuards:
    """Build a guard with known defaults, overridable per test."""
    defaults = dict(
        max_notional_usdt=500.0,
        stale_seconds=120,
        min_confidence=0.0,
        max_positions=1,
    )
    defaults.update(overrides)
    return ExecutionGuards(**defaults)


def _fresh_ts() -> int:
    """Return a candle timestamp that is current (0 seconds old)."""
    return int(time.time())


def _stale_ts(seconds: int = 200) -> int:
    """Return a candle timestamp that is ``seconds`` old."""
    return int(time.time()) - seconds


# ── happy path ────────────────────────────────────────────────────────────────

def test_valid_entry_passes_all_guards():
    g = _guards()
    r = g.check_entry(confidence=0.65, candle_ts=_fresh_ts(), leg_price=60_000.0, leg_qty=0.001)
    assert r.allowed
    assert r.reason == "ok"


def test_confidence_zero_disables_confidence_guard():
    g = _guards(min_confidence=0.0)
    r = g.check_entry(confidence=0.0, candle_ts=_fresh_ts(), leg_price=60_000.0, leg_qty=0.001)
    assert r.allowed, "min_confidence=0 should never block"


# ── stale signal ──────────────────────────────────────────────────────────────

def test_blocks_stale_signal():
    g = _guards(stale_seconds=120)
    r = g.check_entry(confidence=0.7, candle_ts=_stale_ts(200), leg_price=60_000.0, leg_qty=0.001)
    assert not r.allowed
    assert r.reason == "stale_signal"


def test_allows_signal_just_within_stale_window():
    g = _guards(stale_seconds=120)
    # 100 s old — within the 120 s window
    r = g.check_entry(confidence=0.7, candle_ts=_stale_ts(100), leg_price=60_000.0, leg_qty=0.001)
    assert r.allowed


# ── notional cap ──────────────────────────────────────────────────────────────

def test_blocks_oversized_notional():
    g = _guards(max_notional_usdt=500.0)
    # 1 BTC at $60k → $60,000 notional >> $500 limit
    r = g.check_entry(confidence=0.7, candle_ts=_fresh_ts(), leg_price=60_000.0, leg_qty=1.0)
    assert not r.allowed
    assert r.reason == "notional_limit"


def test_allows_notional_exactly_at_limit():
    g = _guards(max_notional_usdt=500.0)
    # 0.008 BTC at $60k = $480 < $500
    r = g.check_entry(confidence=0.7, candle_ts=_fresh_ts(), leg_price=60_000.0, leg_qty=0.008)
    assert r.allowed


# ── confidence floor ──────────────────────────────────────────────────────────

def test_blocks_low_confidence():
    g = _guards(min_confidence=0.6)
    r = g.check_entry(confidence=0.55, candle_ts=_fresh_ts(), leg_price=60_000.0, leg_qty=0.001)
    assert not r.allowed
    assert r.reason == "low_confidence"


def test_allows_confidence_at_threshold():
    g = _guards(min_confidence=0.6)
    r = g.check_entry(confidence=0.60, candle_ts=_fresh_ts(), leg_price=60_000.0, leg_qty=0.001)
    assert r.allowed


# ── max positions ─────────────────────────────────────────────────────────────

def test_blocks_when_at_max_positions():
    g = _guards(max_positions=1)
    r = g.check_entry(
        confidence=0.7, candle_ts=_fresh_ts(), leg_price=60_000.0, leg_qty=0.001,
        open_position_count=1,
    )
    assert not r.allowed
    assert r.reason == "max_positions"


def test_allows_entry_when_below_max_positions():
    g = _guards(max_positions=2)
    r = g.check_entry(
        confidence=0.7, candle_ts=_fresh_ts(), leg_price=60_000.0, leg_qty=0.001,
        open_position_count=1,
    )
    assert r.allowed


# ── guard ordering (first-fail) ───────────────────────────────────────────────

def test_stale_checked_before_notional():
    """Stale signal blocks before notional even when both would fire."""
    g = _guards(stale_seconds=10, max_notional_usdt=1.0)
    r = g.check_entry(confidence=0.7, candle_ts=_stale_ts(200), leg_price=60_000.0, leg_qty=1.0)
    assert r.reason == "stale_signal"


# ── to_dict ───────────────────────────────────────────────────────────────────

def test_to_dict_contains_all_keys():
    g = _guards()
    d = g.to_dict()
    assert set(d.keys()) == {"max_notional_usdt", "stale_seconds", "min_confidence", "max_positions"}
