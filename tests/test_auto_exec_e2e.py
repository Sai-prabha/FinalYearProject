"""E2E check of the auto-execute transition path (signal → dual-leg orders).

Drives model_server._execute_broker_position_change through all three
transition shapes with the real PaperBroker + safety gate + JSONL logging —
the exact code path live testnet execution uses (only the broker differs).

Run: .venv/bin/python tests/test_auto_exec_e2e.py
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import model_server as ms
from api.broker_client import JSONL_LOG_PATH, PaperBroker
from api.broker_config import BrokerConfig


async def main():
    ms.broker_client = PaperBroker()
    ms.broker_config = BrokerConfig(auto_execute=True)
    ms._broker_position = 0

    n_before = sum(1 for _ in open(JSONL_LOG_PATH)) if JSONL_LOG_PATH.exists() else 0

    # FLAT → SHORT (entry: SELL BTC + BUY ETH)
    await ms._execute_broker_position_change(0, -1, 1_760_000_000)
    ev = ms._last_execution_event
    assert ev["transition"] == "FLAT→SHORT" and len(ev["legs"]) == 2, ev
    assert {(l["symbol"], l["side"]) for l in ev["legs"]} == {("BTCUSDT", "SELL"), ("ETHUSDT", "BUY")}
    assert ev["all_ok"] and ms._broker_position == -1

    # SHORT → LONG (reversal: 2 reduce-only exits + 2 entries)
    await ms._execute_broker_position_change(-1, 1, 1_760_000_060)
    ev = ms._last_execution_event
    assert ev["transition"] == "SHORT→LONG" and len(ev["legs"]) == 4, ev
    assert sum(l["reduce_only"] for l in ev["legs"]) == 2

    # LONG → FLAT (reduce-only exit)
    await ms._execute_broker_position_change(1, 0, 1_760_000_120)
    ev = ms._last_execution_event
    assert ev["transition"] == "LONG→FLAT" and all(l["reduce_only"] for l in ev["legs"])
    assert ms._broker_position == 0

    # give the fire-and-forget reconciliation tasks a beat, then check the log
    await asyncio.sleep(0.2)
    lines = [json.loads(l) for l in open(JSONL_LOG_PATH)][n_before:]
    orders = [l for l in lines if l["action"] == "place_order"]
    assert len(orders) == 8, f"expected 8 legs logged, got {len(orders)}"
    assert all(o["mode"] == "paper" for o in orders)

    print("test_auto_exec_e2e: ALL TRANSITIONS OK (8 legs placed+logged, positions tracked)")


if __name__ == "__main__":
    asyncio.run(main())
