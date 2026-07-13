"""Broker adapter unit tests: order translation, response parsing, and the
mode-selection factory matrix (paper / demo / degraded). No network — the
Binance client is constructed against monkeypatched transport internals.

Run: .venv/bin/python -m pytest tests/test_broker_adapter.py -q
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.broker_client import (
    BinanceFuturesBroker,
    OrderRequest,
    make_broker_client,
)


def _demo_env(monkeypatch, **extra):
    env = {
        "BINANCE_ENV": "demo",
        "BINANCE_API_KEY": "test-key",
        "BINANCE_API_SECRET": "test-secret",
        "BINANCE_BASE_URL": "https://testnet.binancefuture.com",
        **extra,
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)


# ── order translation ─────────────────────────────────────────────────────


def test_market_order_params():
    p = BinanceFuturesBroker._build_order_params(
        OrderRequest(symbol="btcusdt", side="BUY", quantity=0.001)
    )
    assert p == {"symbol": "BTCUSDT", "side": "BUY", "type": "MARKET", "quantity": 0.001}


def test_limit_order_requires_price_and_gets_gtc():
    req = OrderRequest(symbol="ETHUSDT", side="SELL", order_type="LIMIT", quantity=0.05, price=2500.0)
    p = BinanceFuturesBroker._build_order_params(req)
    assert p["price"] == 2500.0 and p["timeInForce"] == "GTC"
    with pytest.raises(ValueError, match="LIMIT order requires price"):
        BinanceFuturesBroker._build_order_params(
            OrderRequest(symbol="ETHUSDT", side="SELL", order_type="LIMIT", quantity=0.05)
        )


def test_reduce_only_and_client_id_flags():
    req = OrderRequest(symbol="BTCUSDT", side="SELL", quantity=0.001, reduce_only=True, client_id="meridian-1")
    p = BinanceFuturesBroker._build_order_params(req)
    assert p["reduceOnly"] == "true" and p["newClientOrderId"] == "meridian-1"


# ── response parsing ──────────────────────────────────────────────────────


def test_parse_fill_response():
    r = BinanceFuturesBroker._parse_order_response(
        {"orderId": 123456, "status": "FILLED", "executedQty": "0.001", "avgPrice": "97500.10"}
    )
    assert r.broker_order_id == "123456"
    assert r.status == "FILLED"
    assert r.filled_qty == 0.001
    assert r.avg_price == 97500.10


def test_parse_binance_error_envelope_is_rejected_with_reason():
    r = BinanceFuturesBroker._parse_order_response({"code": -2019, "msg": "Margin is insufficient."})
    assert r.status == "REJECTED"
    assert "Margin is insufficient" in r.message


# ── mode-selection factory matrix ─────────────────────────────────────────


def test_factory_paper_when_env_not_demo(monkeypatch):
    monkeypatch.delenv("BINANCE_ENV", raising=False)
    b = make_broker_client()
    assert b.mode == "paper" and b.init_error is None  # intended paper, not degraded


def test_factory_demo_missing_keys_degrades_with_reason(monkeypatch):
    monkeypatch.setenv("BINANCE_ENV", "demo")
    for k in ("BINANCE_API_KEY", "BINANCE_API_SECRET", "BINANCE_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    b = make_broker_client()
    assert b.mode == "paper"
    assert b.init_error is not None and "BINANCE_API_KEY" in b.init_error


def test_factory_demo_bad_keys_degrades_with_auth_reason(monkeypatch):
    _demo_env(monkeypatch)
    monkeypatch.setattr(BinanceFuturesBroker, "_sync_time", lambda self: None)
    monkeypatch.setattr(
        BinanceFuturesBroker,
        "_signed_request",
        lambda self, method, path, params=None, retried=False: (
            401,
            {"code": -2015, "msg": "Invalid API-key, IP, or permissions for action."},
        ),
    )
    b = make_broker_client()
    assert b.mode == "paper"
    assert "Testnet auth failed" in b.init_error and "-2015" in b.init_error


def test_factory_demo_healthy_keys_yields_binance_broker(monkeypatch):
    _demo_env(monkeypatch)
    monkeypatch.setattr(BinanceFuturesBroker, "_sync_time", lambda self: None)
    monkeypatch.setattr(
        BinanceFuturesBroker,
        "_signed_request",
        lambda self, method, path, params=None, retried=False: (
            200,
            [{"asset": "USDT", "balance": "4997.0", "availableBalance": "4997.0"}],
        ),
    )
    b = make_broker_client()
    assert isinstance(b, BinanceFuturesBroker)
    assert b.mode == "demo" and b.init_error is None


def test_demo_order_success_and_rejection_via_transport(monkeypatch):
    """End-to-end through place_order with the HTTP layer stubbed."""
    _demo_env(monkeypatch)
    monkeypatch.setattr(BinanceFuturesBroker, "_sync_time", lambda self: None)

    responses = {
        "ok": (200, {"orderId": 42, "status": "FILLED", "executedQty": "0.001", "avgPrice": "97000"}),
        "reject": (400, {"code": -1013, "msg": "Filter failure: LOT_SIZE"}),
    }
    which = {"key": "ok"}
    monkeypatch.setattr(
        BinanceFuturesBroker,
        "_signed_request",
        lambda self, method, path, params=None, retried=False: responses[which["key"]],
    )

    broker = BinanceFuturesBroker()
    ok = broker.place_order(OrderRequest(symbol="BTCUSDT", side="BUY", quantity=0.001))
    assert ok.status == "FILLED" and ok.broker_order_id == "42" and ok.avg_price == 97000

    which["key"] = "reject"
    bad = broker.place_order(OrderRequest(symbol="BTCUSDT", side="BUY", quantity=0.001))
    assert bad.status == "REJECTED" and "LOT_SIZE" in bad.message
