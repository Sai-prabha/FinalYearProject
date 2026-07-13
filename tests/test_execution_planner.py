"""Broker-aware transition planner + balance-based sizing (Mission: execution quality).

Covers:
  * _size_legs — balance math, fixed fallback, safety-cap / max-notional
    clamps, below-minimum skips
  * _plan_transition — exits only for real exchange exposure (at real size),
    drift reconciliation when the exchange is flat, satisfied-skip when the
    exchange already holds the target, model-assumed fallback (paper)
  * _execute_broker_position_change — RECONCILED / SKIPPED outcomes, sizing
    detail on entry legs, exit-failure semantics preserved
  * _maybe_auto_execute — in-flight lock dedupes the dual-candle double-fire

Run: pytest tests/test_execution_planner.py
"""
import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import model_server as ms
from api.broker_client import OrderResponse, PaperBroker
from api.broker_config import BrokerConfig

PRICES = {"BTCUSDT": 62500.0, "ETHUSDT": 1750.0}
FILTERS = {
    "BTCUSDT": {"step_size": 0.0001, "min_qty": 0.0001, "min_notional": 50.0},
    "ETHUSDT": {"step_size": 0.001, "min_qty": 0.001, "min_notional": 20.0},
}


class StubBroker(PaperBroker):
    """Paper broker with scriptable exchange truth and order recording."""

    def __init__(self, positions=None, usdt=None, fail_positions=False,
                 reject_reduce_only=False, order_delay=0.0):
        self._positions = positions if positions is not None else []
        self._usdt = usdt
        self._fail = fail_positions
        self._reject_ro = reject_reduce_only
        self._delay = order_delay
        self.placed = []

    def try_get_open_positions(self):
        return None if self._fail else list(self._positions)

    def get_balance(self):
        if self._usdt is None:
            return {"assets": []}
        return {"assets": [{"asset": "USDT", "balance": self._usdt, "available": self._usdt}]}

    def get_symbol_filters(self, symbol):
        return FILTERS[symbol]

    def place_order(self, req):
        if self._delay:
            time.sleep(self._delay)
        self.placed.append(req)
        if self._reject_ro and req.reduce_only:
            return OrderResponse(broker_order_id="", status="REJECTED",
                                 message="ReduceOnly Order is rejected.")
        return OrderResponse(broker_order_id=f"stub-{len(self.placed)}", status="FILLED",
                             filled_qty=req.quantity, avg_price=req.price or 100.0)


def _pos(symbol, signed):
    return {"symbol": symbol, "side": "LONG" if signed > 0 else "SHORT",
            "size": abs(signed), "signed_size": signed, "entry_price": 100.0, "mark_price": 100.0}


@pytest.fixture
def wire(monkeypatch):
    """Install a stub broker + config into model_server; silence persistence/WS."""

    def _wire(broker, **cfg):
        monkeypatch.setattr(ms, "broker_client", broker)
        monkeypatch.setattr(ms, "broker_config", BrokerConfig(auto_execute=True, **cfg))
        monkeypatch.setattr(ms, "execution_guards", None)
        monkeypatch.setattr(ms, "_broker_position", 0)
        monkeypatch.setattr(ms, "_persist_exec_event", lambda e: None)

        async def _noop(*a, **k):
            return None

        monkeypatch.setattr(ms, "broadcast_to_clients", _noop)
        monkeypatch.setattr(ms, "_reconcile_and_broadcast", _noop)
        return broker

    return _wire


# ── Sizing ────────────────────────────────────────────────────────────────


def test_balance_sizing_math(wire):
    wire(StubBroker(usdt=5000.0))
    sized = ms._size_legs(PRICES)
    # 5000 × 2% = 100 USDT per leg; floored to exchange step
    assert sized["BTCUSDT"]["qty"] == pytest.approx(0.0016)
    assert sized["ETHUSDT"]["qty"] == pytest.approx(0.057)
    for s in sized.values():
        assert s["skip"] is None
        assert s["sizing"]["basis"] == "balance"
        assert s["sizing"]["equity"] == 5000.0
        assert s["sizing"]["risk_fraction"] == 0.02


def test_sizing_scales_with_equity(wire):
    broker = wire(StubBroker(usdt=5000.0))
    q1 = ms._size_legs(PRICES)["BTCUSDT"]["qty"]
    broker._usdt = 10000.0
    q2 = ms._size_legs(PRICES)["BTCUSDT"]["qty"]
    assert q2 == pytest.approx(2 * q1)


def test_sizing_clamped_by_caps(wire, monkeypatch):
    wire(StubBroker(usdt=10_000_000.0))
    monkeypatch.setattr(ms, "execution_guards", ms.ExecutionGuards())
    sized = ms._size_legs(PRICES)
    # raw 200k USDT → symbol cap 0.01 BTC → 625 USDT → max-notional 500 → 0.008
    assert sized["BTCUSDT"]["qty"] == pytest.approx(0.008)
    assert "max notional" in sized["BTCUSDT"]["sizing"]["clamped_by"]


def test_sizing_below_minimums_skips(wire):
    wire(StubBroker(usdt=100.0))
    sized = ms._size_legs(PRICES)  # 2 USDT per leg
    assert sized["BTCUSDT"]["skip"] is not None
    assert sized["ETHUSDT"]["skip"] is not None
    assert "min" in sized["ETHUSDT"]["skip"]


def test_sizing_fixed_fallback_when_no_balance(wire):
    wire(StubBroker(usdt=None))  # balance query yields no USDT row
    sized = ms._size_legs(PRICES)
    assert sized["BTCUSDT"]["qty"] == 0.001
    assert sized["ETHUSDT"]["qty"] == 0.05
    assert sized["BTCUSDT"]["sizing"]["basis"] == "fixed-fallback"


def test_sizing_mode_fixed(wire):
    wire(StubBroker(usdt=5000.0), sizing_mode="fixed")
    sized = ms._size_legs(PRICES)
    assert sized["BTCUSDT"]["qty"] == 0.001
    assert sized["BTCUSDT"]["sizing"]["basis"] == "fixed"


# ── Planning ──────────────────────────────────────────────────────────────


def test_plan_flat_to_long(wire):
    wire(StubBroker(usdt=5000.0))
    sized = ms._size_legs(PRICES)
    plan = asyncio.run(ms._plan_transition(0, 1, sized))
    assert plan["source"] == "exchange"
    assert plan["exits"] == []
    assert {(e["symbol"], e["side"]) for e in plan["entries"]} == {("BTCUSDT", "BUY"), ("ETHUSDT", "SELL")}


def test_plan_exit_uses_real_exchange_size(wire):
    # Exchange holds 0.0023 BTC / -0.061 ETH (≠ any configured default)
    wire(StubBroker(positions=[_pos("BTCUSDT", 0.0023), _pos("ETHUSDT", -0.061)], usdt=5000.0))
    sized = ms._size_legs(PRICES)
    plan = asyncio.run(ms._plan_transition(1, 0, sized))
    exits = {e["symbol"]: e for e in plan["exits"]}
    assert exits["BTCUSDT"]["qty"] == pytest.approx(0.0023) and exits["BTCUSDT"]["side"] == "SELL"
    assert exits["ETHUSDT"]["qty"] == pytest.approx(0.061) and exits["ETHUSDT"]["side"] == "BUY"
    assert all(e["reduce_only"] for e in plan["exits"])
    assert plan["entries"] == []


def test_plan_never_reduces_a_flat_exchange(wire):
    # Model thinks LONG; exchange is flat (another writer closed it).
    wire(StubBroker(positions=[], usdt=5000.0))
    sized = ms._size_legs(PRICES)
    plan = asyncio.run(ms._plan_transition(1, 0, sized))
    assert plan["exits"] == [] and plan["entries"] == []
    assert {l["skip_kind"] for l in plan["skipped"]} == {"drift"}


def test_plan_satisfied_by_existing_exposure(wire):
    wire(StubBroker(positions=[_pos("BTCUSDT", 0.002), _pos("ETHUSDT", -0.06)], usdt=5000.0))
    sized = ms._size_legs(PRICES)
    plan = asyncio.run(ms._plan_transition(0, 1, sized))
    assert plan["exits"] == [] and plan["entries"] == []
    assert {l["skip_kind"] for l in plan["skipped"]} == {"satisfied"}


def test_plan_falls_back_to_model_state_when_query_fails(wire):
    wire(StubBroker(fail_positions=True, usdt=None))
    sized = ms._size_legs(PRICES)
    plan = asyncio.run(ms._plan_transition(1, 0, sized))
    assert plan["source"] == "model-assumed"
    exits = {e["symbol"]: e["qty"] for e in plan["exits"]}
    assert exits == {"BTCUSDT": 0.001, "ETHUSDT": 0.05}  # configured defaults


# ── Execution outcomes ────────────────────────────────────────────────────


def test_exec_reconciled_on_drift(wire):
    broker = wire(StubBroker(positions=[], usdt=5000.0))
    ms._broker_position = 1
    asyncio.run(ms._execute_broker_position_change(1, 0, 1_760_000_000, PRICES))
    ev = ms._last_execution_event
    assert ev["outcome"] == "RECONCILED" and ev["all_ok"]
    assert broker.placed == []  # no order ever reached the broker
    assert ms._broker_position == 0


def test_exec_skipped_when_not_tradable(wire):
    broker = wire(StubBroker(usdt=100.0))
    asyncio.run(ms._execute_broker_position_change(0, 1, 1_760_000_000, PRICES))
    ev = ms._last_execution_event
    assert ev["outcome"] == "SKIPPED" and ev["all_ok"]
    assert broker.placed == []
    assert ms._broker_position == 0  # honest: nothing is on the exchange
    assert all(l["status"] == "SKIPPED" for l in ev["legs"])


def test_exec_entry_carries_sizing_detail(wire):
    broker = wire(StubBroker(usdt=5000.0))
    asyncio.run(ms._execute_broker_position_change(0, 1, 1_760_000_000, PRICES))
    ev = ms._last_execution_event
    assert ev["outcome"] == "OK" and ms._broker_position == 1
    assert sorted(r.quantity for r in broker.placed) == [pytest.approx(0.0016), pytest.approx(0.057)]
    for leg in ev["entry_legs"]:
        assert leg["sizing"]["basis"] == "balance"
        assert leg["sizing"]["equity"] == 5000.0


def test_exec_exit_failure_semantics_preserved(wire):
    wire(StubBroker(positions=[_pos("BTCUSDT", 0.0016), _pos("ETHUSDT", -0.057)],
                    usdt=5000.0, reject_reduce_only=True))
    ms._broker_position = 1
    asyncio.run(ms._execute_broker_position_change(1, 0, 1_760_000_000, PRICES))
    ev = ms._last_execution_event
    assert ev["outcome"] == "EXIT_PARTIAL_FAILURE" and not ev["all_ok"]
    assert ms._broker_position == 1  # retry on next tick


# ── In-flight lock ────────────────────────────────────────────────────────


def test_double_fire_is_deduped(wire):
    broker = wire(StubBroker(usdt=5000.0, order_delay=0.05))

    async def scenario():
        # Two candle ticks (BTC + ETH close) land almost simultaneously.
        r1, r2 = await asyncio.gather(
            ms._maybe_auto_execute(1, 1_760_000_000, PRICES),
            ms._maybe_auto_execute(1, 1_760_000_000, PRICES),
        )
        return r1, r2

    r1, r2 = asyncio.run(scenario())
    assert sorted([r1, r2]) == [False, True]  # exactly one executed
    assert len(broker.placed) == 2  # one dual-leg entry, not four legs


def test_no_refire_when_already_at_target(wire):
    broker = wire(StubBroker(usdt=5000.0))
    ms._broker_position = 1
    ran = asyncio.run(ms._maybe_auto_execute(1, 1_760_000_000, PRICES))
    assert ran is False and broker.placed == []
