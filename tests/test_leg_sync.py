"""
Unit tests for FT-2 leg-sync logic in api/model_server.py.

Covers:
- _build_exec_event: correct outcome/all_ok/flat-list semantics
- unwind side selection: BUY leg → SELL unwind, SELL leg → BUY unwind
- FAILED_STATUSES constant
- Policy invariants (exit failure → final_pos=prev_pos, etc.)

Run: pytest tests/test_leg_sync.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.model_server import _build_exec_event, _FAILED_STATUSES, _TERMINAL_STATUSES


# ── _FAILED_STATUSES / _TERMINAL_STATUSES ─────────────────────────────────────

def test_failed_statuses_subset_of_terminal():
    assert _FAILED_STATUSES <= _TERMINAL_STATUSES


def test_filled_is_not_a_failure():
    assert "FILLED" not in _FAILED_STATUSES


def test_new_is_not_a_failure():
    # MARKET orders start as NEW on Binance — should NOT trigger unwind
    assert "NEW" not in _FAILED_STATUSES


# ── _build_exec_event ─────────────────────────────────────────────────────────

def _leg(symbol, side, status, order_id="ord1"):
    return {"symbol": symbol, "side": side, "status": status, "qty": 0.001,
            "filled_qty": 0.001, "avg_price": 60000.0, "order_id": order_id, "error": None}


def test_build_exec_event_ok():
    btc = _leg("BTCUSDT", "BUY", "FILLED", "ord1")
    eth = _leg("ETHUSDT", "SELL", "FILLED", "ord2")
    ev = _build_exec_event(0, 1, 1, [], [btc, eth], [], "OK")
    assert ev["all_ok"] is True
    assert ev["outcome"] == "OK"
    assert ev["final_pos"] == 1
    assert ev["prev_pos"] == 0
    assert ev["new_pos"] == 1
    assert len(ev["legs"]) == 2           # flat list = exit + entry + unwind
    assert ev["exit_legs"] == []
    assert ev["entry_legs"] == [btc, eth]
    assert ev["unwind_legs"] == []
    assert ev["reconciled"] is False


def test_build_exec_event_partial_unwind_ok():
    btc = _leg("BTCUSDT", "BUY", "FILLED")
    eth = _leg("ETHUSDT", "SELL", "REJECTED")
    unwind = _leg("BTCUSDT", "SELL", "FILLED", "uw1")
    ev = _build_exec_event(0, 1, 0, [], [btc, eth], [unwind], "ENTRY_PARTIAL_UNWIND_OK")
    assert ev["all_ok"] is False
    assert ev["outcome"] == "ENTRY_PARTIAL_UNWIND_OK"
    assert ev["final_pos"] == 0           # stayed flat
    assert len(ev["legs"]) == 3           # btc entry + eth entry + btc unwind
    assert len(ev["unwind_legs"]) == 1


def test_build_exec_event_exit_partial_failure():
    btc = _leg("BTCUSDT", "SELL", "REJECTED")
    eth = _leg("ETHUSDT", "BUY", "FILLED")
    ev = _build_exec_event(1, 0, 1, [btc, eth], [], [], "EXIT_PARTIAL_FAILURE")
    assert ev["all_ok"] is False
    assert ev["final_pos"] == 1           # prev_pos preserved for retry
    assert ev["transition"] == "LONG→FLAT"


def test_build_exec_event_all_rejected():
    btc = _leg("BTCUSDT", "BUY", "REJECTED")
    eth = _leg("ETHUSDT", "SELL", "REJECTED")
    ev = _build_exec_event(0, 1, 0, [], [btc, eth], [], "ENTRY_ALL_REJECTED")
    assert ev["all_ok"] is False
    assert ev["final_pos"] == 0


def test_transition_label_reversal():
    ev = _build_exec_event(-1, 1, 1, [], [], [], "OK")
    assert ev["transition"] == "SHORT→LONG"


# ── Unwind side selection (pure logic, no async) ───────────────────────────────

def _unwind_side_for(filled_side: str) -> str:
    """Mirror the logic in _unwind_one_leg without the async call."""
    return "SELL" if filled_side == "BUY" else "BUY"


def test_buy_leg_unwinds_with_sell():
    assert _unwind_side_for("BUY") == "SELL"


def test_sell_leg_unwinds_with_buy():
    assert _unwind_side_for("SELL") == "BUY"


# ── Policy invariants ─────────────────────────────────────────────────────────

def test_exit_partial_failure_final_pos_equals_prev():
    """final_pos must equal prev_pos on exit failure so retry fires next tick."""
    prev = 1
    ev = _build_exec_event(prev, 0, prev, [], [], [], "EXIT_PARTIAL_FAILURE")
    assert ev["final_pos"] == ev["prev_pos"]


def test_entry_partial_unwind_final_pos_is_flat():
    """After a partial entry + unwind, we must end up flat (not at new_pos)."""
    ev = _build_exec_event(0, 1, 0, [], [], [], "ENTRY_PARTIAL_UNWIND_OK")
    assert ev["final_pos"] == 0
    assert ev["new_pos"] == 1   # intended target was non-zero
    assert ev["final_pos"] != ev["new_pos"]
