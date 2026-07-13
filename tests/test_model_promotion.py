"""Model promotion semantics: a shadow candidate goes live only through an
explicit, guarded, audited action — never accidentally.

Run: .venv/bin/python -m pytest tests/test_model_promotion.py -q
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import model_server as ms


def _setup(monkeypatch, tmp_path, primary="v4.16", shadow="v4.18"):
    monkeypatch.setattr(ms, "MODEL_VERSION", primary)
    monkeypatch.setattr(ms, "SHADOW_MODEL_VERSION", shadow)
    monkeypatch.setattr(ms, "signal_gen", ms._make_signal_gen(primary))
    monkeypatch.setattr(ms, "shadow_signal_gen", ms._make_signal_gen(shadow))
    monkeypatch.setattr(ms, "_broker_position", 0)
    monkeypatch.setattr(ms, "LIVE_DATA_DIR", tmp_path)
    monkeypatch.setattr(ms, "PROMOTIONS_JSONL", tmp_path / "promotions.jsonl")


def _promote(version, confirm=True):
    return asyncio.run(ms.promote_model(ms.PromoteRequest(version=version, confirm=confirm), authorization=None))


def test_promote_requires_shadow_candidate(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(ms, "shadow_signal_gen", None)
    with pytest.raises(HTTPException) as e:
        _promote("v4.18")
    assert e.value.status_code == 400


def test_promote_rejects_unevaluated_version(monkeypatch, tmp_path):
    # Only the version that has been running in shadow can go live
    _setup(monkeypatch, tmp_path)
    with pytest.raises(HTTPException) as e:
        _promote("v4.17")
    assert e.value.status_code == 400


def test_promote_requires_explicit_confirm(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    with pytest.raises(HTTPException) as e:
        _promote("v4.18", confirm=False)
    assert e.value.status_code == 400
    assert ms.MODEL_VERSION == "v4.16"  # nothing changed


def test_promote_refuses_open_position(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    ms.signal_gen.position = 1
    with pytest.raises(HTTPException) as e:
        _promote("v4.18")
    assert e.value.status_code == 409


def test_promote_refuses_broker_drift(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(ms, "_broker_position", -1)  # model flat, broker short
    with pytest.raises(HTTPException) as e:
        _promote("v4.18")
    assert e.value.status_code == 409


def test_promote_success_carries_portfolio_and_audits(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    ms.signal_gen.balance = 950.0
    ms.signal_gen.trades = [{"direction": "SHORT", "pnl_dollar": -50.0}]
    ms.signal_gen.wins, ms.signal_gen.losses = 0, 1

    out = _promote("v4.18")

    assert out["ok"] is True and out["from"] == "v4.16" and out["to"] == "v4.18"
    assert out["persisted"] is False and "MODEL_VERSION=v4.18" in out["note"]
    # live version moved, shadow retired
    assert ms.MODEL_VERSION == "v4.18"
    assert ms.SHADOW_MODEL_VERSION == "" and ms.shadow_signal_gen is None
    # the new generator is the candidate strategy with the old portfolio
    assert ms.signal_gen.cfg.entry_threshold_short == 0.45  # v4.18 conviction gate
    assert ms.signal_gen.balance == 950.0
    assert len(ms.signal_gen.trades) == 1 and ms.signal_gen.losses == 1
    assert ms.signal_gen.position == 0
    # audit record written
    rec = json.loads((tmp_path / "promotions.jsonl").read_text().strip())
    assert rec["from"] == "v4.16" and rec["to"] == "v4.18"


def test_promoted_version_visible_in_version_endpoint(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    _promote("v4.18")
    out = asyncio.run(ms.version())
    assert out["current"] == "v4.18"
    # configured (env) version unchanged — the mismatch is the "update Railway
    # env to persist" signal for operators and the UI
    assert out["configured"] == ms.CONFIGURED_MODEL_VERSION != "v4.18"
