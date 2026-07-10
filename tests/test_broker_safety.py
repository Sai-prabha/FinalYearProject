"""Self-check for broker safety rails. Run: .venv/bin/python tests/test_broker_safety.py"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.broker_client import (
    BinanceFuturesBroker,
    BrokerConfigError,
    KILL_SWITCH_PATH,
    OrderRequest,
    PaperBroker,
    check_order_safety,
    make_broker_client,
)


def order(symbol="BTCUSDT", qty=0.001):
    return OrderRequest(symbol=symbol, side="BUY", quantity=qty)


def main():
    KILL_SWITCH_PATH.unlink(missing_ok=True)

    # 1. Normal tiny order passes the gate
    assert check_order_safety(order()) is None

    # 2. Symbol allowlist
    assert "allowlist" in check_order_safety(order(symbol="DOGEUSDT"))

    # 3. Quantity caps (defaults: 0.01 BTC, 0.5 ETH)
    assert "exceeds max" in check_order_safety(order(qty=0.02))
    assert check_order_safety(order(symbol="ETHUSDT", qty=0.4)) is None
    assert "exceeds max" in check_order_safety(order(symbol="ETHUSDT", qty=0.6))

    # 4. Kill switch blocks everything, PaperBroker included
    KILL_SWITCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    KILL_SWITCH_PATH.touch()
    try:
        assert "KILL SWITCH" in check_order_safety(order())
        resp = PaperBroker().place_order(order())
        assert resp.status == "REJECTED" and "KILL SWITCH" in resp.message
    finally:
        KILL_SWITCH_PATH.unlink()

    # 5. Testnet host pin: production URL must be unconstructable
    env = {"BINANCE_API_KEY": "x", "BINANCE_API_SECRET": "y",
           "BINANCE_BASE_URL": "https://fapi.binance.com"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        try:
            BinanceFuturesBroker()
            raise AssertionError("production host was accepted!")
        except BrokerConfigError as e:
            assert "not an approved testnet host" in str(e)
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    # 6. Factory: anything but BINANCE_ENV=demo yields PaperBroker
    os.environ.pop("BINANCE_ENV", None)
    assert make_broker_client().mode == "paper"
    os.environ["BINANCE_ENV"] = "live"
    assert make_broker_client().mode == "paper"
    os.environ.pop("BINANCE_ENV", None)

    print("test_broker_safety: ALL 6 CHECKS PASSED")


if __name__ == "__main__":
    main()
