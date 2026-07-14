"""Execution control plane: authoritative GET, explicit-SET PATCH, idempotency,
optimistic concurrency, audit trail, and the writer gate.

Run: .venv/bin/python -m pytest tests/test_execution_control.py -q
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import broker_config as bc
from api import execution_control as ec
from api import model_server as ms
from api.broker_config import BrokerConfig


def _run(coro):
    return asyncio.run(coro)


class _Broker:
    def __init__(self, mode="demo"):
        self.mode = mode
        self.init_error = None


@pytest.fixture
def ctl(tmp_path, monkeypatch):
    """Fresh control plane in tmp: no meta file, broker_config auto_execute=False."""
    monkeypatch.setattr(ec, "EXEC_DIR", tmp_path / "execution")
    monkeypatch.setattr(ec, "CONTROL_PATH", tmp_path / "execution" / "control.json")
    monkeypatch.setattr(ec, "AUDIT_PATH", tmp_path / "execution" / "audit.jsonl")
    monkeypatch.setattr(bc, "LIVE_DATA_DIR", tmp_path / "live")
    monkeypatch.setattr(bc, "CONFIG_PATH", tmp_path / "live" / "broker_config.json")
    monkeypatch.setattr(ms, "broker_config", BrokerConfig(auto_execute=False))
    monkeypatch.setattr(ms, "broker_client", _Broker("demo"))
    monkeypatch.setattr(ms, "_non_writer_warned", False)
    # deterministic writer identity: local observer unless a test overrides
    monkeypatch.delenv("EXECUTION_WRITER", raising=False)
    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
    monkeypatch.delenv("RAILWAY_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAILWAY_REPLICA_ID", raising=False)
    return tmp_path


def _audit_rows(tmp_path):
    path = tmp_path / "execution" / "audit.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _patch(auto_execute, expected_version=None, request_id=None, surface="meridian"):
    return _run(ms.patch_execution_control(
        ms.ExecutionControlPatch(
            auto_execute=auto_execute,
            expected_version=expected_version,
            request_id=request_id,
        ),
        authorization=None,
        x_control_surface=surface,
    ))


# ── GET ────────────────────────────────────────────────────────────────────


def test_get_returns_authoritative_state(ctl):
    out = _run(ms.get_execution_control(authorization=None))
    assert out["auto_execute"] is False
    assert out["version"] == 1
    assert out["mode"] == "demo"
    assert out["writer"]["backend"] == "local"
    assert out["writer"]["is_writer"] is False
    assert out["writer"]["role"] == "observer"
    assert out["writer"]["instance_id"]


# ── PATCH: explicit set ────────────────────────────────────────────────────


def test_patch_set_true(ctl):
    out = _patch(True, expected_version=1)
    assert out["auto_execute"] is True
    assert out["version"] == 2
    assert out["updated_via"] == "meridian"
    assert out["updated_by"] == "anonymous"
    assert out["updated_at"]
    assert ms.broker_config.auto_execute is True
    # value persisted through the normal broker-config store
    assert json.loads((ctl / "live" / "broker_config.json").read_text())["auto_execute"] is True


def test_patch_set_false(ctl):
    _patch(True, expected_version=1)
    out = _patch(False, expected_version=2, surface="vercel")
    assert out["auto_execute"] is False
    assert out["version"] == 3
    assert out["updated_via"] == "vercel"
    assert ms.broker_config.auto_execute is False


def test_repeated_identical_patch_is_idempotent(ctl):
    first = _patch(True, expected_version=1, request_id="req-1")
    n_audit = len(_audit_rows(ctl))
    # retry / duplicate submission: same desired value, current version
    again = _patch(True, expected_version=first["version"], request_id="req-1")
    assert again["auto_execute"] is True
    assert again["version"] == first["version"]  # no bump
    assert len(_audit_rows(ctl)) == n_audit      # no extra side effects
    # even without expected_version, setting the same value is a no-op
    third = _patch(True)
    assert third["version"] == first["version"]


def test_stale_version_rejected_with_current_state(ctl):
    _patch(True, expected_version=1)
    with pytest.raises(HTTPException) as exc:
        _patch(False, expected_version=1)  # stale: version is now 2
    assert exc.value.status_code == 409
    current = exc.value.detail["current"]
    assert current["auto_execute"] is True
    assert current["version"] == 2
    assert ms.broker_config.auto_execute is True  # state unchanged
    conflict = _audit_rows(ctl)[-1]
    assert conflict["outcome"] == "conflict"
    assert conflict["expected_version"] == 1
    assert conflict["current_version"] == 2


def test_audit_record_written_on_change(ctl):
    _patch(True, expected_version=1, request_id="abc-123")
    row = _audit_rows(ctl)[-1]
    assert row["event"] == "auto_execute_changed"
    assert row["prev"] is False and row["new"] is True
    assert row["actor"] == "anonymous"
    assert row["via"] == "meridian"
    assert row["request_id"] == "abc-123"
    assert row["version"] == 2
    assert row["outcome"] == "applied"
    assert row["instance"] and row["ts"]


def test_legacy_broker_config_routes_through_control_plane(ctl):
    out = _run(ms.update_broker_config_route(
        ms.BrokerConfigUpdate(auto_execute=True), authorization=None,
    ))
    assert out["auto_execute"] is True
    state = _run(ms.get_execution_control(authorization=None))
    assert state["version"] == 2
    assert state["updated_via"] == "legacy-api"
    assert _audit_rows(ctl)[-1]["via"] == "legacy-api"


# ── Writer gate ────────────────────────────────────────────────────────────


def test_non_writer_cannot_auto_execute_in_demo(ctl, monkeypatch):
    monkeypatch.setattr(ms, "broker_config", BrokerConfig(auto_execute=True))
    assert ms.broker_client.mode == "demo"
    assert ec.is_writer() is False
    assert ms._auto_exec_eligible() is False


def test_writer_can_auto_execute_in_demo(ctl, monkeypatch):
    monkeypatch.setattr(ms, "broker_config", BrokerConfig(auto_execute=True))
    monkeypatch.setenv("EXECUTION_WRITER", "1")
    assert ms._auto_exec_eligible() is True


def test_paper_rehearsal_allowed_on_non_writer(ctl, monkeypatch):
    monkeypatch.setattr(ms, "broker_config", BrokerConfig(auto_execute=True))
    monkeypatch.setattr(ms, "broker_client", _Broker("paper"))
    assert ec.is_writer() is False
    assert ms._auto_exec_eligible() is True


def test_railway_env_makes_writer(ctl, monkeypatch):
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    info = ec.writer_info()
    assert info == {
        "backend": "railway",
        "instance_id": info["instance_id"],
        "is_writer": True,
        "role": "writer",
    }
    # explicit override wins over environment detection
    monkeypatch.setenv("EXECUTION_WRITER", "0")
    assert ec.is_writer() is False


def test_writer_identity_in_broker_summary(ctl):
    writer = ms._broker_summary()["writer"]
    assert writer["role"] == "observer"
    assert writer["backend"] == "local"
