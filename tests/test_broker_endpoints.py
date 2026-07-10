"""Broker observability endpoints: executions history, activity log, balance
degradation, and broker-health surfacing.

Run: .venv/bin/python -m pytest tests/test_broker_endpoints.py -q
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import model_server as ms
from api.broker_client import PaperBroker


def _run(coro):
    return asyncio.run(coro)


def test_exec_events_persist_and_dedup(tmp_path, monkeypatch):
    path = tmp_path / "exec_events.jsonl"
    monkeypatch.setattr(ms, "EXEC_EVENTS_JSONL", path)

    e1 = {"timestamp": "2026-07-10T10:00:00+00:00", "outcome": "OK", "reconciled": False}
    e2 = {"timestamp": "2026-07-10T11:00:00+00:00", "outcome": "EXIT_PARTIAL_FAILURE", "reconciled": False}
    ms._persist_exec_event(e1)
    ms._persist_exec_event(e2)
    ms._persist_exec_event({**e1, "reconciled": True})  # reconciled snapshot of e1

    out = _run(ms.get_broker_executions(limit=50, authorization=None))
    assert out["count"] == 2
    # newest first
    assert out["executions"][0]["timestamp"].startswith("2026-07-10T11")
    # deduped: e1's reconciled snapshot wins
    e1_row = out["executions"][1]
    assert e1_row["reconciled"] is True


def test_exec_events_tolerate_malformed_lines(tmp_path, monkeypatch):
    path = tmp_path / "exec_events.jsonl"
    path.write_text('{"timestamp": "t1", "outcome": "OK"}\nNOT-JSON\n{"timestamp": "t2", "outcome": "OK"}\n')
    monkeypatch.setattr(ms, "EXEC_EVENTS_JSONL", path)
    out = _run(ms.get_broker_executions(limit=10, authorization=None))
    assert out["count"] == 2


def test_activity_filters_polling_noise(tmp_path, monkeypatch):
    log = tmp_path / "broker.jsonl"
    rows = [
        {"ts": "1", "action": "get_balance", "request": {}, "response": {}},
        {"ts": "2", "action": "place_order", "request": {"symbol": "BTCUSDT"}, "response": {"status": "FILLED"}},
        {"ts": "3", "action": "get_order_status", "request": {}, "response": {}},
        {"ts": "4", "action": "cancel_order", "request": {"order_id": "x"}, "response": {"ok": True}},
    ]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(ms, "JSONL_LOG_PATH", log)

    out = _run(ms.get_broker_activity(limit=10, authorization=None))
    assert out["count"] == 2
    assert [r["action"] for r in out["activity"]] == ["cancel_order", "place_order"]  # newest first


def test_balance_surfaces_broker_error(monkeypatch):
    class DeadKeyBroker(PaperBroker):
        mode = "demo"
        def get_balance(self):
            return {"assets": [], "raw": {"code": -2015, "msg": "Invalid API-key"}}

    monkeypatch.setattr(ms, "broker_client", DeadKeyBroker())
    out = _run(ms.get_broker_balance(authorization=None))
    assert out["ok"] is False and "-2015" in out["error"]
    assert out["assets"] == []  # existing shape preserved


def test_balance_ok_and_paper_fallback_reason(monkeypatch):
    pb = PaperBroker()
    monkeypatch.setattr(ms, "broker_client", pb)
    out = _run(ms.get_broker_balance(authorization=None))
    assert out["ok"] is True and out["error"] is None

    pb2 = PaperBroker()
    pb2.init_error = "Testnet auth failed (-2015)"
    monkeypatch.setattr(ms, "broker_client", pb2)
    out2 = _run(ms.get_broker_balance(authorization=None))
    assert out2["ok"] is False and "paper" in out2["error"]
    # health also visible in the broker summary used by /status + WS
    assert ms._broker_summary()["init_error"] == "Testnet auth failed (-2015)"
