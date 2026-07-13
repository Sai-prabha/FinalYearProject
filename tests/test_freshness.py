"""Freshness contract: /status.freshness tells clients explicitly whether the
candle data is stale, so no UI ever has to invent its own threshold.

Run: .venv/bin/python -m pytest tests/test_freshness.py -q
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import model_server as ms


class _StubCalc:
    def __init__(self, btc_time, eth_time):
        self._dq = {
            "btc_candles": 1500,
            "eth_candles": 1500,
            "ready": True,
            "last_btc_time": btc_time,
            "last_eth_time": eth_time,
            "synced": btc_time == eth_time,
        }

    def get_data_quality_status(self):
        return self._dq


def test_fresh_when_last_candle_recent(monkeypatch):
    now = time.time()
    monkeypatch.setattr(ms, "feature_calc", _StubCalc(now - 30, now - 30))
    f = ms._freshness()
    assert f["stale"] is False
    assert 0 <= f["age_seconds"] < ms.STALE_AFTER_SECONDS
    assert f["stale_after_seconds"] == ms.STALE_AFTER_SECONDS


def test_staleness_bounded_by_older_leg(monkeypatch):
    # A fresh BTC candle cannot compensate for a dead ETH feed
    now = time.time()
    monkeypatch.setattr(ms, "feature_calc", _StubCalc(now - 30, now - 400))
    f = ms._freshness()
    assert f["stale"] is True
    assert f["age_seconds"] >= 400
    assert f["last_candle_ts"] == int(now - 400) or abs(f["last_candle_ts"] - (now - 400)) < 1


def test_no_data_is_stale_never_fresh(monkeypatch):
    monkeypatch.setattr(ms, "feature_calc", None)
    f = ms._freshness()
    assert f["stale"] is True
    assert f["age_seconds"] is None
    assert f["last_candle_ts"] is None


def test_status_exposes_freshness_and_configured_version(monkeypatch):
    import asyncio

    now = time.time()
    monkeypatch.setattr(ms, "feature_calc", _StubCalc(now - 10, now - 10))
    out = asyncio.run(ms.status())
    assert out["freshness"]["stale"] is False
    assert out["configured_version"] == ms.CONFIGURED_MODEL_VERSION
    assert out["model_version"] == ms.MODEL_VERSION
