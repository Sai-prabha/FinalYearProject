"""Reconciliation engine tests — hand-computed fixtures per TRADE_RECONCILIATION.md.

The scenario builders double as the synthetic broker-fill generator for
simulation tooling: build_trade / build_event / build_fill compose any
execution reality against any model intent.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.reconciliation import (
    BROKER_ONLY,
    BrokerHistoryStore,
    CRITICAL,
    EXPLAINED_COSTS,
    FILLS_MISSING,
    INFO,
    MATCHED,
    MODEL_ONLY,
    MODEL_ONLY_BREAK,
    PARTIAL_EXECUTION,
    PENDING,
    SIGN_MISMATCH,
    TIMING_DRIFT,
    UNVERIFIED_PAPER,
    WARNING,
    eligible_at,
    parse_client_id,
    reconcile,
)

T0 = 1_700_000_000  # entry signal candle (epoch s)
T1 = 1_700_003_600  # exit signal candle


# ── Scenario builders (synthetic broker reality generator) ────────────────

def build_trade(direction="LONG", entry=T0, exit_=T1, pnl_pct=1.0, pnl_dollar=10.0):
    return {
        "direction": direction, "entry_time": entry, "exit_time": exit_,
        "entry_price": 35.0, "exit_price": 35.35 if direction == "LONG" else 34.65,
        "pnl_pct": pnl_pct, "pnl_dollar": pnl_dollar, "bars_held": 60,
        "entry_probability": 0.61, "entry_strength": 0.22, "reason": "Signal change",
        "model_version": "v4.16", "position_size_pct": 66.0,
    }


def leg(symbol, side, qty, order_id, decision, status="FILLED", reduce_only=False):
    return {
        "symbol": symbol, "side": side, "qty": qty, "reduce_only": reduce_only,
        "status": status, "order_id": order_id, "filled_qty": qty, "avg_price": 0.0,
        "error": None, "decision_price": decision,
        "client_id": f"v415-{T0 if not reduce_only else T1}-X-{symbol[:3]}",
    }


def build_event(signal_ts, prev_pos, new_pos, entry_legs=None, exit_legs=None,
                outcome="OK", timestamp=None):
    entry_legs = entry_legs or []
    exit_legs = exit_legs or []
    return {
        "timestamp": timestamp or f"2026-07-14T00:00:{signal_ts % 60:02d}+00:00",
        "signal_ts": signal_ts, "prev_pos": prev_pos, "new_pos": new_pos,
        "final_pos": new_pos, "outcome": outcome, "all_ok": True,
        "legs": entry_legs + exit_legs,
        "entry_legs": entry_legs, "exit_legs": exit_legs,
        "unwind_legs": [], "skipped_legs": [],
    }


def build_fill(order_id, symbol, side, price, qty, realized_pnl=0.0, commission=0.02,
               fill_id=None, time_ms=None):
    return {
        "kind": "fill", "id": fill_id or int(order_id) * 10, "symbol": symbol,
        "order_id": str(order_id), "side": side, "price": price, "qty": qty,
        "quote_qty": price * qty, "realized_pnl": realized_pnl,
        "commission": commission, "commission_asset": "USDT", "maker": False,
        "time_ms": time_ms or T0 * 1000,
    }


def clean_long_roundtrip():
    """LONG ratio round-trip, expected +1.00 gross, slippage −0.095, fees −0.12."""
    trade = build_trade()
    entry_ev = build_event(T0, 0, 1, entry_legs=[
        leg("BTCUSDT", "BUY", 0.001, "101", 50_000.0),
        leg("ETHUSDT", "SELL", 0.05, "102", 2_000.0),
    ])
    exit_ev = build_event(T1, 1, 0, exit_legs=[
        leg("BTCUSDT", "SELL", 0.001, "201", 50_500.0, reduce_only=True),
        leg("ETHUSDT", "BUY", 0.05, "202", 1_990.0, reduce_only=True),
    ])
    fills = [
        build_fill("101", "BTCUSDT", "BUY", 50_010.0, 0.001),
        build_fill("102", "ETHUSDT", "SELL", 1_999.5, 0.05, commission=0.04),
        build_fill("201", "BTCUSDT", "SELL", 50_490.0, 0.001, realized_pnl=0.48, time_ms=T1 * 1000),
        build_fill("202", "ETHUSDT", "BUY", 1_991.0, 0.05, realized_pnl=0.425, commission=0.04, time_ms=T1 * 1000),
    ]
    return trade, [entry_ev, exit_ev], fills


def run(trades, events, fills, funding=None, **kw):
    kw.setdefault("now_s", T1 + 100_000)
    kw.setdefault("sync_age_s", 60.0)
    return reconcile(trades, events, fills, funding or [], **kw)


# ── Attribution math (hand-computed) ──────────────────────────────────────

def test_clean_roundtrip_matches_with_exact_attribution():
    trade, events, fills = clean_long_roundtrip()
    out = run([trade], events, fills)
    row = out["trades"][0]
    a = row["attribution"]
    assert abs(a["expected_gross"] - 1.0) < 1e-6
    assert abs(a["realized_gross"] - 0.905) < 1e-6
    assert abs(a["slippage"] - (-0.095)) < 1e-6
    assert abs(a["fees"] - (-0.12)) < 1e-6
    assert a["funding"] == 0.0
    assert abs(a["residual"]) < 1e-9            # buckets fully explain the delta
    assert abs(a["realized_net"] - 0.785) < 1e-6
    assert row["status"] == MATCHED
    assert row["severity"] == INFO
    assert out["summary"]["explained_rate"] == 1.0
    assert out["summary"]["critical_count"] == 0


def test_funding_flows_into_attribution():
    trade, events, fills = clean_long_roundtrip()
    funding = [
        {"kind": "funding", "id": "f1", "symbol": "BTCUSDT", "income": -0.03,
         "time_ms": (T0 + 600) * 1000},
        {"kind": "funding", "id": "f2", "symbol": "BTCUSDT", "income": -0.5,
         "time_ms": (T1 + 999) * 1000},  # outside the holding window — excluded
    ]
    out = run([trade], events, fills, funding)
    a = out["trades"][0]["attribution"]
    assert abs(a["funding"] - (-0.03)) < 1e-9
    assert abs(a["realized_net"] - (0.785 - 0.03)) < 1e-6


def test_costs_flip_is_explained_not_a_break():
    """Model win / broker net loss purely from costs ⇒ healthy EXPLAINED_COSTS."""
    trade, events, fills = clean_long_roundtrip()
    # Small edge: exit decisions barely above entry ⇒ expected +0.10 gross
    events[1]["exit_legs"][0]["decision_price"] = 50_080.0
    events[1]["exit_legs"][1]["decision_price"] = 1_999.6
    fills[2] = build_fill("201", "BTCUSDT", "SELL", 50_070.0, 0.001,
                          realized_pnl=0.06, time_ms=T1 * 1000)
    fills[3] = build_fill("202", "ETHUSDT", "BUY", 1_999.7, 0.05,
                          realized_pnl=-0.01, commission=0.04, time_ms=T1 * 1000)
    out = run([trade], events, fills)
    row = out["trades"][0]
    a = row["attribution"]
    assert a["expected_gross"] > 0
    assert a["realized_net"] < 0                 # fees flipped the outcome
    assert abs(a["residual"]) <= a["tolerance"]
    assert row["status"] == EXPLAINED_COSTS
    assert row["severity"] == INFO               # NOT presented as a defect


def test_sign_mismatch_unexplained_is_critical():
    trade, events, fills = clean_long_roundtrip()
    # Broker reports a big loss the buckets cannot explain
    fills[2] = build_fill("201", "BTCUSDT", "SELL", 50_490.0, 0.001,
                          realized_pnl=-2.5, time_ms=T1 * 1000)
    out = run([trade], events, fills)
    row = out["trades"][0]
    assert row["status"] == SIGN_MISMATCH
    assert row["severity"] == CRITICAL
    assert out["summary"]["sign_mismatch_count"] == 1


def test_partial_fill_flagged_warning():
    trade, events, fills = clean_long_roundtrip()
    fills[0] = build_fill("101", "BTCUSDT", "BUY", 50_010.0, 0.0004)  # 40% filled
    out = run([trade], events, fills)
    row = out["trades"][0]
    assert row["status"] == PARTIAL_EXECUTION
    assert row["severity"] == WARNING


def test_model_only_healthy_vs_break_depends_on_eligibility():
    trade = build_trade()
    healthy = run([trade], [], [])
    assert healthy["trades"][0]["status"] == MODEL_ONLY
    assert healthy["trades"][0]["severity"] == INFO

    broken = run([trade], [], [], eligibility=[(T0 - 500, True)])
    assert broken["trades"][0]["status"] == MODEL_ONLY_BREAK
    assert broken["trades"][0]["severity"] == CRITICAL
    assert broken["summary"]["unmatched_model"] == 1


def test_paper_mode_is_unverified_not_broken():
    trade = build_trade()
    entry_ev = build_event(T0, 0, 1, entry_legs=[
        leg("BTCUSDT", "BUY", 0.001, "paper-abc", 50_000.0),
        leg("ETHUSDT", "SELL", 0.05, "paper-def", 2_000.0),
    ])
    exit_ev = build_event(T1, 1, 0, exit_legs=[
        leg("BTCUSDT", "SELL", 0.001, "paper-ghi", 50_500.0, reduce_only=True),
        leg("ETHUSDT", "BUY", 0.05, "paper-jkl", 1_990.0, reduce_only=True),
    ])
    out = run([trade], [entry_ev, exit_ev], [])
    row = out["trades"][0]
    assert row["status"] == UNVERIFIED_PAPER
    assert row["severity"] == INFO


def test_pending_grace_then_fills_missing():
    trade, events, _ = clean_long_roundtrip()
    fresh = reconcile([trade], events, [], [], now_s=T1 + 30, sync_age_s=10.0)
    assert fresh["trades"][0]["status"] == PENDING

    stale = reconcile([trade], events, [], [], now_s=T1 + 100_000, sync_age_s=10.0)
    assert stale["trades"][0]["status"] == FILLS_MISSING
    assert stale["trades"][0]["severity"] == CRITICAL


def test_orphan_fills_manual_vs_unknown():
    orphan_manual = build_fill("900", "BTCUSDT", "BUY", 50_000.0, 0.002)
    orphan_unknown = build_fill("901", "ETHUSDT", "SELL", 2_000.0, 0.1, fill_id=9011)
    out = run([], [], [orphan_manual, orphan_unknown], manual_order_ids={"900"})
    assert len(out["orphans"]) == 2
    by_id = {o["order_id"]: o for o in out["orphans"]}
    assert by_id["900"]["severity"] == WARNING and by_id["900"]["origin"] == "manual-api"
    assert by_id["901"]["severity"] == CRITICAL and by_id["901"]["origin"] == "unknown"
    assert all(o["status"] == BROKER_ONLY for o in out["orphans"])
    assert out["summary"]["unmatched_broker"] == 2


def test_window_fallback_marks_timing_drift():
    trade, events, fills = clean_long_roundtrip()
    # Legacy rows: no signal_ts / client_id — only a wall timestamp 40 s off
    for ev, t in ((events[0], T0), (events[1], T1)):
        ev["signal_ts"] = None
        for l in ev["legs"]:
            l.pop("client_id", None)
        from datetime import datetime, timezone
        ev["timestamp"] = datetime.fromtimestamp(t + 40, tz=timezone.utc).isoformat()
    out = run([trade], events, fills)
    row = out["trades"][0]
    assert row["status"] == TIMING_DRIFT
    assert row["linkage"]["entry"]["method"] == "window"


def test_reversal_event_serves_exit_and_entry():
    """LONG→SHORT reversal: one event closes trade A and opens trade B."""
    trade_a = build_trade("LONG", T0, T1)
    trade_b = build_trade("SHORT", T1, T1 + 3600)
    entry_a = build_event(T0, 0, 1, entry_legs=[
        leg("BTCUSDT", "BUY", 0.001, "101", 50_000.0),
        leg("ETHUSDT", "SELL", 0.05, "102", 2_000.0),
    ])
    reversal = build_event(T1, 1, -1,
        exit_legs=[
            leg("BTCUSDT", "SELL", 0.001, "201", 50_500.0, reduce_only=True),
            leg("ETHUSDT", "BUY", 0.05, "202", 1_990.0, reduce_only=True),
        ],
        entry_legs=[
            leg("BTCUSDT", "SELL", 0.001, "301", 50_500.0),
            leg("ETHUSDT", "BUY", 0.05, "302", 1_990.0),
        ])
    exit_b = build_event(T1 + 3600, -1, 0, exit_legs=[
        leg("BTCUSDT", "BUY", 0.001, "401", 50_400.0, reduce_only=True),
        leg("ETHUSDT", "SELL", 0.05, "402", 1_995.0, reduce_only=True),
    ])
    fills = [
        build_fill("101", "BTCUSDT", "BUY", 50_000.0, 0.001, commission=0.0),
        build_fill("102", "ETHUSDT", "SELL", 2_000.0, 0.05, commission=0.0),
        build_fill("201", "BTCUSDT", "SELL", 50_500.0, 0.001, realized_pnl=0.5, commission=0.0),
        build_fill("202", "ETHUSDT", "BUY", 1_990.0, 0.05, realized_pnl=0.5, commission=0.0),
        build_fill("301", "BTCUSDT", "SELL", 50_500.0, 0.001, commission=0.0),
        build_fill("302", "ETHUSDT", "BUY", 1_990.0, 0.05, commission=0.0),
        build_fill("401", "BTCUSDT", "BUY", 50_400.0, 0.001, realized_pnl=0.1, commission=0.0),
        build_fill("402", "ETHUSDT", "SELL", 1_995.0, 0.05, realized_pnl=0.25, commission=0.0),
    ]
    out = run([trade_a, trade_b], [entry_a, reversal, exit_b], fills)
    by_id = {r["id"]: r for r in out["trades"]}
    a = by_id[f"rt-{T0}-{T1}"]
    b = by_id[f"rt-{T1}-{T1 + 3600}"]
    assert a["status"] == MATCHED and abs(a["attribution"]["realized_gross"] - 1.0) < 1e-9
    assert b["status"] == MATCHED and abs(b["attribution"]["realized_gross"] - 0.35) < 1e-9
    assert out["summary"]["unmatched_broker"] == 0  # reversal legs all accounted


def test_position_drift_is_a_critical_break():
    out = run([], [], [], position_check={
        "model_pos": 1, "broker_pos": 1, "exchange_pos": 0, "status": "DRIFT",
    })
    assert out["breaks"][0]["status"] == "POSITION_DRIFT"
    assert out["breaks"][0]["severity"] == CRITICAL


def test_helpers():
    assert parse_client_id("v415-1700000000-L-BTC") == 1_700_000_000
    assert parse_client_id("manual-xyz") is None
    pts = [(100.0, True), (200.0, False), (300.0, True)]
    assert eligible_at(pts, 50) is False
    assert eligible_at(pts, 150) is True
    assert eligible_at(pts, 250) is False
    assert eligible_at(pts, 350) is True


# ── Store: sync idempotency, late data, restart survival ─────────────────

class FakeBroker:
    """Deterministic broker returning canned pages; counts calls."""

    def __init__(self, fills_by_symbol, income=None):
        self.fills_by_symbol = fills_by_symbol
        self.income = income or []
        self.calls = []

    def get_user_trades(self, symbol, start_ms=None, from_id=None, limit=1000):
        self.calls.append(("fills", symbol, start_ms, from_id))
        rows = self.fills_by_symbol.get(symbol, [])
        if from_id is not None:
            rows = [r for r in rows if r["id"] >= from_id]
        return rows

    def get_income(self, income_type=None, start_ms=None, limit=1000):
        self.calls.append(("income", income_type, start_ms))
        return [r for r in self.income if start_ms is None or r["time"] >= start_ms]


def _raw_fill(fid, symbol="BTCUSDT", price=50_000.0, qty=0.001, t=None):
    return {"id": fid, "symbol": symbol, "orderId": fid * 7, "side": "BUY",
            "price": str(price), "qty": str(qty), "quoteQty": str(price * qty),
            "realizedPnl": "0", "commission": "0.02", "commissionAsset": "USDT",
            "maker": False, "time": t or int(time.time() * 1000)}


def test_store_sync_idempotent_and_late_data(tmp_path):
    store = BrokerHistoryStore(tmp_path)
    broker = FakeBroker({"BTCUSDT": [_raw_fill(1), _raw_fill(2)]},
                        income=[{"symbol": "BTCUSDT", "income": "-0.01",
                                 "time": int(time.time() * 1000), "tranId": 555,
                                 "incomeType": "FUNDING_FEE"}])
    r1 = store.sync(broker)
    assert r1["added_fills"] == 2 and r1["added_funding"] == 1
    r2 = store.sync(broker)                     # same data again — no dupes
    assert r2["added_fills"] == 0 and r2["added_funding"] == 0

    # Late-arriving fill lands on the next sync
    broker.fills_by_symbol["BTCUSDT"].append(_raw_fill(3))
    r3 = store.sync(broker)
    assert r3["added_fills"] == 1

    # Restart survival: a fresh store over the same dir sees everything
    fills, funding = BrokerHistoryStore(tmp_path).load()
    assert len(fills) == 3 and len(funding) == 1
    # And reconciliation over reloaded data is deterministic
    out1 = reconcile([], [], fills, funding)
    out2 = reconcile([], [], fills, funding)
    assert out1["summary"]["unmatched_broker"] == out2["summary"]["unmatched_broker"] == 3


def test_store_unsupported_paper_mode(tmp_path):
    class PaperLike:
        def get_user_trades(self, *a, **k):
            return None

        def get_income(self, *a, **k):
            return None

    store = BrokerHistoryStore(tmp_path)
    assert store.sync(PaperLike()) == {"unsupported": True}
    assert store.last_sync_age_s() is None      # never claims a sync happened
