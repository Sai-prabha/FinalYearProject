"""Incident regression: model held a position with no execution trace.

Root causes pinned here:
1. Auto-exec used to run only when broker mode == "demo" — paper mode skipped
   the whole execution block silently while the UI said "auto-execute ON".
2. Nothing surfaced a model↔broker position mismatch (restart mid-position,
   auto-exec enabled late, broker degraded at transition time).

Run: .venv/bin/python -m pytest tests/test_incident_paper_routing.py -q
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import model_server as ms
from api.broker_client import PaperBroker
from api.broker_config import BrokerConfig


def test_auto_exec_eligible_in_paper_and_demo(monkeypatch):
    monkeypatch.setattr(ms, "broker_config", BrokerConfig(auto_execute=True))

    paper = PaperBroker()
    monkeypatch.setattr(ms, "broker_client", paper)
    assert ms._auto_exec_eligible() is True  # THE incident: this was False

    monkeypatch.setattr(paper, "mode", "demo", raising=False)
    assert ms._auto_exec_eligible() is True

    monkeypatch.setattr(ms, "broker_config", BrokerConfig(auto_execute=False))
    assert ms._auto_exec_eligible() is False

    monkeypatch.setattr(ms, "broker_config", BrokerConfig(auto_execute=True))
    monkeypatch.setattr(ms, "broker_client", None)
    assert ms._auto_exec_eligible() is False


def test_paper_transition_executes_and_persists_event(tmp_path, monkeypatch):
    """FLAT→LONG in paper mode goes through the real leg flow and leaves a trace."""
    monkeypatch.setattr(ms, "broker_client", PaperBroker())
    monkeypatch.setattr(ms, "broker_config", BrokerConfig(auto_execute=True))
    monkeypatch.setattr(ms, "_broker_position", 0)
    monkeypatch.setattr(ms, "EXEC_EVENTS_JSONL", tmp_path / "exec_events.jsonl")

    asyncio.run(ms._execute_broker_position_change(0, 1, 1_760_000_000))

    assert ms._broker_position == 1
    ev = ms._last_execution_event
    assert ev["transition"] == "FLAT→LONG" and ev["all_ok"]
    assert {(l["symbol"], l["side"]) for l in ev["legs"]} == {("BTCUSDT", "BUY"), ("ETHUSDT", "SELL")}
    # persisted → survives restart, visible via GET /broker/executions
    rows = [json.loads(l) for l in open(tmp_path / "exec_events.jsonl")]
    assert rows and rows[0]["transition"] == "FLAT→LONG"


def test_position_drift_surfaced(monkeypatch):
    monkeypatch.setattr(ms, "broker_client", PaperBroker())
    monkeypatch.setattr(ms, "broker_config", BrokerConfig(auto_execute=True))
    monkeypatch.setattr(ms, "signal_gen", ms._make_signal_gen("v4.16"))
    monkeypatch.setattr(ms, "_broker_position", 0)

    ms.signal_gen.position = 1  # model LONG, broker never told
    drift = ms._broker_summary()["position_drift"]
    assert drift == {"model": 1, "broker": 0, "drifted": True}

    ms.signal_gen.position = 0
    assert ms._broker_summary()["position_drift"]["drifted"] is False


def test_shadow_status_sides_gain_recent_stats(tmp_path, monkeypatch):
    primary_file = tmp_path / "trade_history.json"
    trades = [
        {"direction": "LONG", "pnl_pct": 0.1 * i, "pnl_dollar": float(i), "entry_time": i, "exit_time": i + 1}
        for i in range(1, 31)  # 30 trades → recent window must clip to 20
    ]
    primary_file.write_text(json.dumps(trades))
    shadow_file = tmp_path / "shadow.json"
    shadow_file.write_text("[]")

    monkeypatch.setattr(ms, "SHADOW_MODEL_VERSION", "v4.18")
    monkeypatch.setattr(ms, "SHADOW_TRADES_JSON", shadow_file)
    monkeypatch.setattr(ms, "TRADE_HISTORY_JSON", primary_file)
    monkeypatch.setattr(ms, "signal_gen", ms._make_signal_gen("v4.16"))
    monkeypatch.setattr(ms, "shadow_signal_gen", ms._make_signal_gen("v4.18"))

    out = asyncio.run(ms.shadow_status(authorization=None))
    assert out["primary"]["stats"]["n"] == 30
    assert out["primary"]["recent_stats"]["n"] == 20
    # recent window = trades 11..30 → total pnl 11+...+30 = 410
    assert out["primary"]["recent_stats"]["total_pnl_dollar"] == 410.0
    assert out["shadow"]["recent_stats"]["n"] == 0
