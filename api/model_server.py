"""
FastAPI Model Server for Live Trading Signal Streaming.

Supports version switching between strategy versions (v4.15, v4.16, etc.)
via the MODEL_VERSION environment variable or --model-version CLI flag.

Startup sequence:
  1. Load serialized XGBClassifier from models/v4_14_production/
  2. Instantiate the correct signal generator for the selected version
  3. Restore portfolio state from persisted trade history
  4. Fetch 1000 historical 1m candles (BTC + ETH) from REST API
  5. Connect to WebSocket for live BTC + ETH 1m klines
  6. On each closed candle: calculate features -> model.predict_proba() ->
     hysteresis + circuit breaker + SL/TP -> stream signal to frontend via WS
"""

import asyncio
import csv
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Set
from contextlib import asynccontextmanager

import secrets

import bcrypt
import numpy as np
import requests
import websockets
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from xgboost import XGBClassifier

from .feature_calculator import V414FeatureCalculator, V414SignalGenerator, V416SignalGenerator
from .trade_stats import compute_trade_stats
from .version_config import get_strategy_config, list_versions
from .broker_client import (
    JSONL_LOG_PATH,
    BrokerClient,
    OrderRequest,
    OrderResponse,
    make_broker_client,
)
from .broker_config import (
    BrokerConfig,
    apply_partial_update,
    load_broker_config,
    save_broker_config,
)
from .execution_guards import ExecutionGuards
from .kill_switch import KillSwitch

# ── Auth config (optional; set AUTH_REQUIRED=true on Railway for production) ──
AUTH_REQUIRED = os.environ.get("AUTH_REQUIRED", "false").lower() == "true"
ADMIN_USERNAME = "adm1nFYP"
ADMIN_PASSWORD_HASH = os.environ.get("ADMIN_PASSWORD_HASH", "")
# Production must supply JWT_SECRET; a random per-boot secret is only
# acceptable when auth is off (dev), where tokens are decorative anyway.
_jwt_secret_env = os.environ.get("JWT_SECRET", "")
if not _jwt_secret_env and AUTH_REQUIRED:
    raise RuntimeError(
        "JWT_SECRET environment variable must be set when AUTH_REQUIRED=true. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )
JWT_SECRET = _jwt_secret_env or secrets.token_hex(32)
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24
SUBSCRIBER_API_KEY = os.environ.get("SUBSCRIBER_API_KEY", "")


# ── Numpy → native Python converter ───────────────────────────────────────

def _sanitize_for_json(obj):
    """Recursively convert numpy scalars/arrays to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "v4_14_production"
MODEL_PATH = MODEL_DIR / "model.json"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"
CONFIG_PATH = MODEL_DIR / "config.json"

# ── Trade persistence paths ──────────────────────────────────────────────
LIVE_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "live"
TRADE_HISTORY_JSON = LIVE_DATA_DIR / "trade_history.json"
KILL_SWITCH_JSON = LIVE_DATA_DIR / "kill_switch.json"
TRADE_HISTORY_CSV = LIVE_DATA_DIR / "trade_history.csv"
# Append-only log of auto-execute events (one row on submit, one on
# reconcile). Read back by GET /broker/executions, deduped by timestamp.
EXEC_EVENTS_JSONL = LIVE_DATA_DIR / "exec_events.jsonl"

# ── Data API ──────────────────────────────────────────────────────────────
REST_BASE = "https://api.binance.com/api/v3"
WS_STREAM_BASE = "wss://stream.binance.com:9443/stream"
WARMUP_CANDLES = 1000

# ── Version selection ──────────────────────────────────────────────────────
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v4.15")

# Shadow candidate version — when set (and different from MODEL_VERSION), a
# second signal generator runs side-by-side on the exact same probability
# stream. It persists simulated trades to its own file and NEVER touches the
# live trade history, portfolio state, or broker. Unset = feature off, zero
# behavior change.
SHADOW_MODEL_VERSION = os.environ.get("SHADOW_MODEL_VERSION", "").strip()
if SHADOW_MODEL_VERSION == MODEL_VERSION:
    SHADOW_MODEL_VERSION = ""

# Shadow trades live in their own file, keyed by version, so the real
# trade_history.json is never written by the shadow path.
SHADOW_TRADES_JSON = (
    LIVE_DATA_DIR / f"shadow_{SHADOW_MODEL_VERSION.replace('.', '_')}_trades.json"
    if SHADOW_MODEL_VERSION else None
)

# ── Global state ───────────────────────────────────────────────────────────
model: XGBClassifier = None
feature_names: list = []
config: dict = {}
feature_calc: V414FeatureCalculator = None
# signal_gen is typed as a union — both generators share the same .update() interface
signal_gen = None  # V414SignalGenerator | V416SignalGenerator

active_connections: Set[WebSocket] = set()

# Track last known trade count for persistence
_last_trade_count: int = 0

# Anchors the baseline-vs-candidate matched comparison window when no shadow
# trade exists yet (the window otherwise starts at the earliest shadow trade).
_SERVER_STARTED_TS = time.time()

# Shadow candidate generator (None unless SHADOW_MODEL_VERSION is set)
shadow_signal_gen = None  # V414SignalGenerator | V416SignalGenerator
_shadow_last_trade_count: int = 0
_last_shadow_signal: Optional[Dict] = None

# Broker integration state
broker_client: Optional[BrokerClient] = None
broker_config: Optional[BrokerConfig] = None
# Last position the broker has been instructed to hold (-1, 0, +1).
# Updated after every attempt so transient outages don't cause order storms.
_broker_position: int = 0
# Most recent auto-execute event — included in every WS broadcast after it fires.
_last_execution_event: Optional[Dict] = None
execution_guards: Optional[ExecutionGuards] = None
kill_switch: Optional[KillSwitch] = None

latest_signal_data: Dict = {
    "timestamp": None,
    "ratio": None,
    "features": None,
    "signal": None,
    "data_quality": None,
}


# ── Trade persistence ────────────────────────────────────────────────────

def _ensure_live_dir():
    """Create data/live/ directory if it doesn't exist."""
    LIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _persist_trade(trade: Dict):
    """Append a single trade to JSON and CSV files."""
    _ensure_live_dir()

    # ── JSON append ──
    trades = []
    if TRADE_HISTORY_JSON.exists():
        try:
            with open(TRADE_HISTORY_JSON, "r") as f:
                trades = json.load(f)
        except (json.JSONDecodeError, Exception):
            trades = []

    trades.append(trade)
    with open(TRADE_HISTORY_JSON, "w") as f:
        json.dump(trades, f, indent=2, default=str)

    # ── CSV append ──
    csv_exists = TRADE_HISTORY_CSV.exists()
    fieldnames = [
        "direction", "entry_price", "exit_price", "entry_time", "exit_time",
        "pnl_pct", "pnl_dollar", "bars_held", "position_size_pct",
        "stop_loss", "take_profit", "entry_probability", "entry_strength", "reason",
    ]
    with open(TRADE_HISTORY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not csv_exists:
            writer.writeheader()
        writer.writerow(trade)

    logger.info(f"Trade persisted: {trade['direction']} PnL={trade['pnl_pct']:.3f}%")


def _persist_exec_event(event: Dict) -> None:
    """Append an execution event snapshot to the JSONL history.

    Called twice per event (on submit and after fill reconciliation); the
    read path keeps the newest row per event timestamp. Never raises.
    """
    try:
        _ensure_live_dir()
        with open(EXEC_EVENTS_JSONL, "a") as f:
            f.write(json.dumps(_sanitize_for_json(event), default=str) + "\n")
    except Exception as e:
        logger.warning(f"Exec event persist failed: {e}")


def _read_jsonl_tail(path: Path, limit: int) -> list:
    """Read up to ``limit`` newest valid JSON rows from a JSONL file.

    Malformed lines are skipped — an interrupted write must not take the
    whole history endpoint down. Returns rows oldest→newest.
    """
    if not path.exists():
        return []
    rows: list = []
    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        logger.warning(f"JSONL read failed ({path.name}): {e}")
        return []
    for line in lines[-max(limit * 3, limit):]:  # slack for dedup/filter passes
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _make_signal_gen(version: str):
    """Construct the signal generator for a version string.

    Used for both the primary and the shadow generator so they are built
    identically. v4.15 runs on the frozen V414SignalGenerator with production
    config params; every other registered version runs on the config-driven
    V416SignalGenerator. Unknown versions raise (fail fast).
    """
    if version == "v4.15":
        strategy = config["strategy_params"]
        return V414SignalGenerator(
            entry_threshold=strategy["entry_threshold"],
            exit_threshold=strategy["exit_threshold"],
            min_hold=strategy["min_hold"],
            cooldown=strategy["cooldown"],
            cb_lookback=strategy["cb_lookback"],
            cb_threshold=strategy["cb_threshold"],
        )
    return V416SignalGenerator(cfg=get_strategy_config(version))


def _check_and_persist_shadow_trades():
    """Persist new shadow trades to the shadow-only JSON file (never the real history)."""
    global _shadow_last_trade_count
    if shadow_signal_gen is None or SHADOW_TRADES_JSON is None:
        return
    current = len(shadow_signal_gen.trades)
    if current <= _shadow_last_trade_count:
        return
    _ensure_live_dir()
    existing = []
    if SHADOW_TRADES_JSON.exists():
        try:
            with open(SHADOW_TRADES_JSON, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.extend(shadow_signal_gen.trades[_shadow_last_trade_count:])
    with open(SHADOW_TRADES_JSON, "w") as f:
        json.dump(_sanitize_for_json(existing), f, indent=2, default=str)
    _shadow_last_trade_count = current
    logger.info(f"SHADOW trade persisted ({SHADOW_MODEL_VERSION}): total={current}")


def _shadow_snapshot() -> Optional[Dict]:
    """Compact shadow-generator state for /status, /shadow/status and WS payloads."""
    if shadow_signal_gen is None:
        return None
    g = shadow_signal_gen
    wins = getattr(g, "wins", 0)
    losses = getattr(g, "losses", 0)
    closed = wins + losses
    return {
        "version": SHADOW_MODEL_VERSION,
        "position": g.position,
        "balance": getattr(g, "balance", None),
        "total_pnl": getattr(g, "total_pnl", None),
        "total_trades": len(g.trades),
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / closed) if closed else 0.0,
        "last_signal": _last_shadow_signal,
    }


def _check_and_persist_new_trades():
    """Check if signal_gen has new trades and persist them."""
    global _last_trade_count
    if signal_gen is None:
        return

    current_count = len(signal_gen.trades)
    if current_count > _last_trade_count:
        # Persist new trades
        for trade in signal_gen.trades[_last_trade_count:]:
            _persist_trade(trade)
            if kill_switch is not None:
                kill_switch.record_pnl(float(trade.get("pnl_dollar", 0.0) or 0.0))
        _last_trade_count = current_count


async def _place_leg_order(
    symbol: str,
    side: str,
    qty: float,
    reduce_only: bool,
    tag: str,
) -> dict:
    """Place a single MARKET order for one leg of a ratio trade.

    Returns a result dict with keys: symbol, side, qty, reduce_only, status,
    order_id, filled_qty, avg_price, error.  Never raises.
    """
    client_id = f"v415-{tag}-{symbol[:3]}"[:36]
    req = OrderRequest(
        symbol=symbol,
        side=side,
        order_type="MARKET",
        quantity=qty,
        reduce_only=reduce_only,
        client_id=client_id,
    )
    base = {"symbol": symbol, "side": side, "qty": qty, "reduce_only": reduce_only}
    try:
        resp = await asyncio.to_thread(broker_client.place_order, req)
        if resp.status == "REJECTED":
            logger.warning(
                f"Broker leg REJECTED {symbol} {side} qty={qty} reduce={reduce_only}: {resp.message}"
            )
            return {**base, "status": "REJECTED", "order_id": "", "filled_qty": 0.0, "avg_price": 0.0, "error": resp.message}
        logger.info(
            f"Broker leg {symbol} {side} qty={qty} reduce={reduce_only} "
            f"→ {resp.status} id={resp.broker_order_id} filled={resp.filled_qty} avg={resp.avg_price}"
        )
        return {**base, "status": resp.status, "order_id": resp.broker_order_id, "filled_qty": resp.filled_qty, "avg_price": resp.avg_price, "error": None}
    except Exception as e:
        logger.error(f"Broker leg {symbol} {side} raised: {e}", exc_info=True)
        return {**base, "status": "ERROR", "order_id": "", "filled_qty": 0.0, "avg_price": 0.0, "error": str(e)}


async def _place_entry_orders(direction: int, timestamp: int) -> list:
    """Place dual-leg MARKET entry orders for a new ratio position.

    LONG  ratio (BTC/ETH up)  → BUY  BTCUSDT + SELL ETHUSDT
    SHORT ratio (BTC/ETH down) → SELL BTCUSDT + BUY  ETHUSDT
    Returns list of per-leg result dicts.
    """
    btc_side = "BUY" if direction == 1 else "SELL"
    eth_side = "SELL" if direction == 1 else "BUY"
    label = "L" if direction == 1 else "S"
    tag = f"{timestamp}-{label}"

    btc = await _place_leg_order("BTCUSDT", btc_side, broker_config.default_btc_qty, False, tag)
    eth = await _place_leg_order("ETHUSDT", eth_side, broker_config.default_eth_qty, False, tag)
    return [btc, eth]


async def _place_exit_orders(direction: int, timestamp: int) -> list:
    """Place reduce-only MARKET exit orders to close an existing ratio position.

    Closes LONG  → SELL BTCUSDT + BUY  ETHUSDT (reduce_only)
    Closes SHORT → BUY  BTCUSDT + SELL ETHUSDT (reduce_only)
    Returns list of per-leg result dicts.
    """
    btc_side = "SELL" if direction == 1 else "BUY"
    eth_side = "BUY" if direction == 1 else "SELL"
    label = "XL" if direction == 1 else "XS"
    tag = f"{timestamp}-{label}"

    btc = await _place_leg_order("BTCUSDT", btc_side, broker_config.default_btc_qty, True, tag)
    eth = await _place_leg_order("ETHUSDT", eth_side, broker_config.default_eth_qty, True, tag)
    return [btc, eth]


_TERMINAL_STATUSES = {"FILLED", "REJECTED", "ERROR", "CANCELED", "EXPIRED"}
# Statuses that mean "the exchange never accepted this order" — no fill to unwind.
_FAILED_STATUSES = frozenset({"REJECTED", "ERROR"})


async def _unwind_one_leg(leg: dict) -> dict:
    """Reverse a filled entry leg with a reduce-only MARKET order.

    Used when a multi-leg entry partially fails: if leg A fills but leg B is
    rejected, we send a reduce-only SELL/BUY on leg A's symbol to close the
    unhedged position.  Reuses ``_place_leg_order`` so errors are caught and
    returned as status="ERROR" rather than raised.
    """
    close_side = "SELL" if leg["side"] == "BUY" else "BUY"
    tag = f"unwind-{(leg.get('order_id') or 'x')[:12]}"
    result = await _place_leg_order(
        leg["symbol"], close_side, leg["qty"], reduce_only=True, tag=tag
    )
    result["unwind"] = True  # mark so downstream readers can distinguish
    return result


def _build_exec_event(
    prev_pos: int,
    new_pos: int,
    final_pos: int,
    exit_legs: list,
    entry_legs: list,
    unwind_legs: list,
    outcome: str,
) -> dict:
    """Build the execution event dict stored in ``_last_execution_event``.

    ``outcome`` is one of: "OK" | "EXIT_PARTIAL_FAILURE" | "ENTRY_ALL_REJECTED"
      | "ENTRY_PARTIAL_UNWIND_OK" | "ENTRY_PARTIAL_UNWIND_FAILED".

    The flat ``legs`` list (exit + entry + unwind) is kept for backward
    compatibility with ``_reconcile_legs`` and WS consumers.
    """
    _pl = {-1: "SHORT", 0: "FLAT", 1: "LONG"}
    all_legs = exit_legs + entry_legs + unwind_legs
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prev_pos": prev_pos,
        "new_pos": new_pos,
        "final_pos": final_pos,
        "transition": f"{_pl.get(prev_pos, '?')}→{_pl.get(new_pos, '?')}",
        "outcome": outcome,
        "all_ok": outcome == "OK",
        "legs": all_legs,
        "exit_legs": exit_legs,
        "entry_legs": entry_legs,
        "unwind_legs": unwind_legs,
        "reconciled": False,
    }


async def _reconcile_legs(legs: list) -> None:
    """Poll Binance concurrently for confirmed fill data on all submitted legs.

    Each leg is polled independently with back-off: 0.5 s, 1.5 s, 3.5 s cumulative.
    Legs without an order_id or already in a terminal state are skipped.
    Paper broker returns None from get_order_status and is also skipped.
    Total worst-case wall time ≈ 3.5 s regardless of how many legs there are.
    """
    POLL_DELAYS = [0.5, 1.0, 2.0]  # gaps between polls; cumulative max ≈ 3.5 s

    async def _poll_one(leg: dict) -> None:
        order_id = leg.get("order_id", "")
        if not order_id or leg.get("status") in _TERMINAL_STATUSES:
            return
        for delay in POLL_DELAYS:
            await asyncio.sleep(delay)
            try:
                fill = await asyncio.to_thread(
                    broker_client.get_order_status, leg["symbol"], order_id
                )
            except Exception as e:
                logger.warning(f"Reconcile poll error {leg['symbol']} #{order_id}: {e}")
                return
            if fill is None:
                return  # paper broker — no polling possible
            leg["status"] = fill["status"]
            leg["filled_qty"] = fill["filled_qty"]
            leg["avg_price"] = fill["avg_price"]
            leg["reconciled"] = True
            if fill["status"] in _TERMINAL_STATUSES:
                logger.info(
                    f"Reconciled {leg['symbol']} #{order_id}: "
                    f"{fill['status']} qty={fill['filled_qty']} avg={fill['avg_price']:.2f}"
                )
                return

    await asyncio.gather(*(_poll_one(leg) for leg in legs))


async def _reconcile_and_broadcast() -> None:
    """Background task: reconcile leg fills then push an exec_reconciled WS event."""
    global _last_execution_event
    if _last_execution_event is None:
        return
    legs = _last_execution_event.get("legs", [])
    await _reconcile_legs(legs)
    # all_ok reflects the outcome determined at execution time; reconciliation
    # only updates fill details, it does not override a partial-failure outcome.
    _last_execution_event["all_ok"] = _last_execution_event.get("outcome") == "OK"
    _last_execution_event["reconciled"] = True
    _persist_exec_event(_last_execution_event)
    # Broadcast a lightweight message so the frontend doesn't wait for the next candle
    await broadcast_to_clients({
        "type": "exec_reconciled",
        "last_exec": _last_execution_event,
    })
    logger.info(
        "Reconciliation broadcast: %d legs, outcome=%s, all_ok=%s",
        len(legs),
        _last_execution_event.get("outcome"),
        _last_execution_event["all_ok"],
    )


def _auto_exec_eligible() -> bool:
    """Whether the per-candle auto-exec block runs at all.

    True for BOTH demo (real testnet orders) and paper (simulated fills):
    the guard/kill-switch/leg/event chain must behave identically in both, so
    paper is a faithful rehearsal and the model never holds a position with no
    execution trace. (The incident this fixes: paper mode used to skip the
    block silently while the UI said "auto-execute ON".)
    """
    return (
        broker_client is not None
        and broker_config is not None
        and broker_client.mode in ("demo", "paper")
        and broker_config.auto_execute
    )


async def _execute_broker_position_change(prev_pos: int, new_pos: int, timestamp: int) -> None:
    """Execute broker orders for a model position transition with leg-sync safety.

    Staged flow:
      Step 1 — Exit the previous position (if any).
               If any exit leg is rejected/errored, abort immediately and keep
               _broker_position = prev_pos so the next signal tick retries.
      Step 2 — Enter the new position (if any).
               If ALL entry legs fail:  stay flat (final_pos = 0).
               If SOME entry legs fill and SOME fail (partial entry):
                 fire reduce-only MARKET orders to unwind the filled leg(s),
                 then stay flat (final_pos = 0).
               If ALL entry legs succeed: advance to new_pos.

    ``_broker_position`` is only advanced to new_pos when all legs succeed.
    In all failure/partial cases it is left at a consistent known-safe value
    so the next position-change trigger will re-attempt the right transition
    without an explicit retry loop.

    Outcomes stored in ``_last_execution_event.outcome``:
      "OK"                         — all legs filled, position advanced
      "EXIT_PARTIAL_FAILURE"       — exit failed; old position may still be open
      "ENTRY_ALL_REJECTED"         — all entry legs rejected; stayed flat
      "ENTRY_PARTIAL_UNWIND_OK"    — partial entry; unwind succeeded; stayed flat
      "ENTRY_PARTIAL_UNWIND_FAILED"— partial entry; unwind failed; manual fix needed
    """
    global _broker_position, _last_execution_event

    _pl = {-1: "SHORT", 0: "FLAT", 1: "LONG"}
    exit_legs: list = []
    entry_legs: list = []
    unwind_legs: list = []
    final_pos = prev_pos  # conservative default — only advanced on confirmed success

    # ── Step 1: Exit previous position ───────────────────────────────────────
    if prev_pos != 0:
        exit_legs = await _place_exit_orders(prev_pos, timestamp)
        exit_failures = [l for l in exit_legs if l["status"] in _FAILED_STATUSES]

        if exit_failures:
            # At least one exit leg was rejected.  The old position may be
            # partially or fully open on the exchange.  Keep _broker_position
            # at prev_pos — the next signal tick will retry the exit cleanly.
            logger.error(
                "EXIT PARTIAL FAILURE [%s→%s]: %d/%d leg(s) failed (%s); "
                "keeping _broker_position=%d to allow retry on next signal",
                _pl.get(prev_pos, "?"), _pl.get(new_pos, "?"),
                len(exit_failures), len(exit_legs),
                ", ".join(f"{l['symbol']} {l['status']}" for l in exit_failures),
                prev_pos,
            )
            _broker_position = prev_pos
            _last_execution_event = _build_exec_event(
                prev_pos, new_pos, prev_pos, exit_legs, [], [], "EXIT_PARTIAL_FAILURE"
            )
            _persist_exec_event(_last_execution_event)
            await broadcast_to_clients({"type": "exec_event", "last_exec": _last_execution_event})
            asyncio.create_task(_reconcile_and_broadcast())
            return

        # All exit legs accepted — old position is closed (fills confirmed async)
        final_pos = 0

    # ── Step 2: Enter new position ────────────────────────────────────────────
    if new_pos != 0:
        entry_legs = await _place_entry_orders(new_pos, timestamp)
        entry_failures = [l for l in entry_legs if l["status"] in _FAILED_STATUSES]
        entry_filled   = [l for l in entry_legs if l["status"] not in _FAILED_STATUSES]

        if entry_failures and entry_filled:
            # Partial entry: some legs went through, some were rejected.
            # Unwind the filled legs in parallel to avoid an unhedged position.
            logger.error(
                "ENTRY PARTIAL FAILURE [%s→%s]: %d/%d leg(s) failed; "
                "unwinding %d filled leg(s) to stay flat",
                _pl.get(prev_pos, "?"), _pl.get(new_pos, "?"),
                len(entry_failures), len(entry_legs), len(entry_filled),
            )
            unwind_legs = list(await asyncio.gather(
                *(_unwind_one_leg(l) for l in entry_filled)
            ))
            unwind_failures = [l for l in unwind_legs if l["status"] in _FAILED_STATUSES]
            if unwind_failures:
                logger.critical(
                    "UNWIND FAILED for %d leg(s) — manual broker intervention required: %s",
                    len(unwind_failures),
                    ", ".join(f"{l['symbol']} {l['side']}" for l in unwind_failures),
                )
                outcome = "ENTRY_PARTIAL_UNWIND_FAILED"
            else:
                logger.info(
                    "Unwind succeeded for %d leg(s) — position stays flat",
                    len(unwind_legs),
                )
                outcome = "ENTRY_PARTIAL_UNWIND_OK"
            final_pos = 0  # stay flat regardless; unwind log shows what needs attention

        elif entry_failures:
            # All entry legs rejected — no fill occurred, stay flat cleanly
            logger.error(
                "ENTRY ALL REJECTED [%s→%s]: staying FLAT",
                _pl.get(prev_pos, "?"), _pl.get(new_pos, "?"),
            )
            outcome = "ENTRY_ALL_REJECTED"
            final_pos = 0

        else:
            # All entry legs accepted
            outcome = "OK"
            final_pos = new_pos

    else:
        # Exit-only (new_pos == 0) with all exits succeeding
        outcome = "OK"
        final_pos = 0

    _broker_position = final_pos
    _last_execution_event = _build_exec_event(
        prev_pos, new_pos, final_pos, exit_legs, entry_legs, unwind_legs, outcome
    )
    _persist_exec_event(_last_execution_event)
    logger.info(
        "AUTO-EXEC [%s] %s→%s confirmed→%s (%d exit / %d entry / %d unwind legs): %s",
        outcome,
        _pl.get(prev_pos, "?"), _pl.get(new_pos, "?"), _pl.get(final_pos, "?"),
        len(exit_legs), len(entry_legs), len(unwind_legs),
        ", ".join(f"{l['symbol']} {l['side']} {l['status']}" for l in exit_legs + entry_legs + unwind_legs),
    )
    # Background reconciliation enriches fill details then re-broadcasts
    asyncio.create_task(_reconcile_and_broadcast())


# ── REST warm-up ─────────────────────────────────────────────────────────

def fetch_historical_candles(symbol: str, interval: str = "1m", limit: int = 1000) -> list:
    """Fetch historical klines from REST API."""
    url = f"{REST_BASE}/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    logger.info(f"Fetching {limit} {interval} candles for {symbol}...")

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    candles = []
    for k in data:
        candles.append({
            "time": int(k[0]) // 1000,
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "taker_buy_volume": float(k[9]),
        })

    logger.info(f"  Got {len(candles)} candles for {symbol}")
    return candles


# ── WebSocket data loop ──────────────────────────────────────────────────

async def connect_to_data_stream():
    """Connect to WebSocket and stream BTC/ETH 1m klines."""
    streams = ["btcusdt@kline_1m", "ethusdt@kline_1m"]
    url = f"{WS_STREAM_BASE}?streams={'/'.join(streams)}"

    logger.info(f"Connecting to data WebSocket: {url}")

    while True:
        try:
            async with websockets.connect(url) as ws:
                logger.info("Connected to data WebSocket — streaming live data")

                async for message in ws:
                    try:
                        data = json.loads(message)
                        await process_kline_message(data)
                    except Exception as e:
                        logger.error(f"Error processing message: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Data WebSocket error: {e}")
            logger.info("Reconnecting in 5 seconds...")
            await asyncio.sleep(5)


async def process_kline_message(data: Dict):
    """Process incoming kline message."""
    if "data" not in data:
        return

    stream_data = data["data"]
    if stream_data.get("e") != "kline":
        return

    kline = stream_data["k"]
    symbol_raw = stream_data["s"]

    if symbol_raw == "BTCUSDT":
        symbol = "BTC"
    elif symbol_raw == "ETHUSDT":
        symbol = "ETH"
    else:
        return

    candle = {
        "time": int(kline["t"]) // 1000,
        "open": float(kline["o"]),
        "high": float(kline["h"]),
        "low": float(kline["l"]),
        "close": float(kline["c"]),
        "volume": float(kline["v"]),
        "taker_buy_volume": float(kline.get("V", float(kline["v"]) * 0.5)),
    }

    is_closed = kline.get("x", False)
    feature_calc.add_candle(symbol, candle)

    # Only run full prediction on CLOSED candles to avoid lookahead
    if not is_closed:
        return

    data_quality = feature_calc.get_data_quality_status()
    if not data_quality["ready"]:
        return

    # Calculate features
    t0 = time.time()
    features_df = feature_calc.calculate_features()
    calc_ms = (time.time() - t0) * 1000

    if features_df is None:
        logger.warning("Feature calc returned None (NaN values)")
        return

    # Run prediction
    proba = model.predict_proba(features_df)[:, 1]
    proba_up = float(proba[0])

    # Current ratio
    btc_close = feature_calc.btc_candles[-1]["close"]
    eth_close = feature_calc.eth_candles[-1]["close"]
    current_ratio = btc_close / eth_close

    # Capture pre-update model position for change detection
    _pre_update_pos = signal_gen.position

    # Generate signal
    sig = signal_gen.update(proba_up, current_ratio, candle["time"])

    # Persist new trades to disk
    _check_and_persist_new_trades()

    # Shadow candidate: same proba/ratio/timestamp, isolated state, own file,
    # no broker interaction. Runs AFTER the primary path so a shadow bug can
    # never affect the live signal.
    if shadow_signal_gen is not None:
        global _last_shadow_signal
        try:
            shadow_sig = shadow_signal_gen.update(proba_up, current_ratio, candle["time"])
            _last_shadow_signal = {
                "direction": shadow_sig["direction"],
                "triggered": shadow_sig["triggered"],
                "probability": shadow_sig["probability"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            _check_and_persist_shadow_trades()
            if shadow_signal_gen.position != signal_gen.position:
                logger.info(
                    "SHADOW DIVERGENCE | primary(%s)=%d shadow(%s)=%d proba=%.4f",
                    MODEL_VERSION, signal_gen.position,
                    SHADOW_MODEL_VERSION, shadow_signal_gen.position, proba_up,
                )
        except Exception as e:
            logger.error(f"Shadow generator error (primary unaffected): {e}", exc_info=True)

    # Structured decision log — emitted every time a signal fires
    if sig["triggered"]:
        logger.info(
            "DECISION | direction=%s probability=%.4f strength=%.4f "
            "broker_pos=%d signal_pos=%d auto_execute=%s mode=%s",
            sig["direction"],
            proba_up,
            sig.get("strength", 0),
            _broker_position,
            signal_gen.position,
            broker_config.auto_execute if broker_config else False,
            broker_client.mode if broker_client else "none",
        )

    # Auto-execute: send dual-leg broker orders whenever model and broker
    # positions differ. Runs every candle (not just on transitions), so drift
    # from a restart mid-position / late auto-exec enable / broker recovery
    # self-heals on the next candle instead of persisting silently.
    if _auto_exec_eligible():
        _post_update_pos = signal_gen.position
        if _post_update_pos != _broker_position:
            blocked = False
            # Kill switch: tripped blocks new entries; exits to flat always proceed
            if _post_update_pos != 0 and kill_switch is not None and not kill_switch.entries_allowed:
                blocked = True
                logger.warning(
                    "AUTO-EXEC BLOCKED [kill_switch_tripped]: %d\u2192%d \u2014 position unchanged",
                    _broker_position, _post_update_pos,
                )
            # Exits to flat always proceed; entries/reversals go through guards
            if not blocked and _post_update_pos != 0 and execution_guards is not None:
                guard = execution_guards.check_entry(
                    confidence=proba_up,
                    candle_ts=candle["time"],
                    leg_price=btc_close,
                    leg_qty=broker_config.default_btc_qty,
                    # ponytail: ratio strategy holds at most 1 position; reversals
                    # replace the old position so open_position_count stays 0.
                    open_position_count=0,
                )
                blocked = not guard.allowed
                if blocked:
                    logger.warning(
                        "AUTO-EXEC BLOCKED [%s]: %d→%d — position unchanged",
                        guard.reason, _broker_position, _post_update_pos,
                    )
            if not blocked:
                await _execute_broker_position_change(
                    _broker_position, _post_update_pos, candle["time"]
                )
    # Build ALL features for frontend display & export (all 50)
    feature_values = {}
    for col in feature_names:
        feature_values[col] = float(features_df[col].iloc[0])

    # Prepare message
    message = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ratio": current_ratio,
        "btc_price": btc_close,
        "eth_price": eth_close,
        "features": feature_values,
        "signal": {
            "direction": sig["direction"],
            "strength": sig["strength"],
            "probability": sig["probability"],
            "triggered": sig["triggered"],
            "blocked_by": sig["blocked_by"],
            "reasoning": sig["reasoning"],
            "circuit_breaker_active": sig["circuit_breaker_active"],
            "entry_threshold": signal_gen.entry_threshold,
            "exit_threshold": signal_gen.exit_threshold,
        },
        "position_meta": sig["position_meta"],
        "portfolio": sig["portfolio"],
        "data_quality": data_quality,
        "model_info": {
            "version": MODEL_VERSION,
            "n_features": len(feature_names),
            "calc_time_ms": round(calc_ms, 1),
            "broker": _broker_summary(),
            "last_exec": _last_execution_event,
            "shadow": _shadow_snapshot(),
        },
    }

    # Sanitize numpy types so json.dumps never fails
    message = _sanitize_for_json(message)

    # Update global state
    global latest_signal_data
    latest_signal_data = message

    # Broadcast to connected frontend clients
    await broadcast_to_clients(message)

    # Log significant events
    if sig["triggered"]:
        logger.info(
            f"SIGNAL TRIGGERED: {sig['direction']} | "
            f"P(up)={proba_up:.4f} | R={current_ratio:.4f}"
        )


async def broadcast_to_clients(message: Dict):
    """Broadcast message to all connected WebSocket clients."""
    if not active_connections:
        return

    connections = active_connections.copy()
    for conn in connections:
        try:
            await conn.send_json(message)
        except Exception:
            active_connections.discard(conn)


# ── Startup / Shutdown ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, warm up feature calculator, start data WS."""
    global model, feature_names, config, feature_calc, signal_gen, _last_trade_count

    version = MODEL_VERSION
    logger.info("=" * 80)
    logger.info(f"{version.upper()} MODEL SERVER - STARTING")
    logger.info("=" * 80)

    # 1. Load model (shared across all versions)
    logger.info(f"Loading {version} production model...")
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"

    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    with open(FEATURE_NAMES_PATH) as f:
        feature_names = json.load(f)

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    logger.info(f"  Model: {len(feature_names)} features, {config['best_n_estimators']} trees")

    # 2. Initialize components — version-aware signal generator
    feature_calc = V414FeatureCalculator(
        feature_names=feature_names,
        max_history=1500,
    )

    signal_gen = _make_signal_gen(version)
    logger.info(f"  Signal generator: {type(signal_gen).__name__} ({version})")

    # Shadow candidate generator — isolated instance, own persistence file
    global shadow_signal_gen, _shadow_last_trade_count
    if SHADOW_MODEL_VERSION:
        shadow_signal_gen = _make_signal_gen(SHADOW_MODEL_VERSION)
        if SHADOW_TRADES_JSON and SHADOW_TRADES_JSON.exists():
            try:
                with open(SHADOW_TRADES_JSON, "r") as f:
                    shadow_trades = json.load(f)
                if shadow_trades:
                    shadow_signal_gen.restore_from_trades(shadow_trades)
                    _shadow_last_trade_count = len(shadow_trades)
            except Exception as e:
                logger.warning(f"  Could not restore shadow trades: {e}")
        logger.info(
            f"  SHADOW mode: {SHADOW_MODEL_VERSION} "
            f"({type(shadow_signal_gen).__name__}) → {SHADOW_TRADES_JSON}"
        )

    # Ensure trade persistence directory exists
    _ensure_live_dir()

    # Restore portfolio state from persisted trade history so balance
    # survives server restarts instead of resetting to $1000.
    if TRADE_HISTORY_JSON.exists():
        try:
            with open(TRADE_HISTORY_JSON, "r") as f:
                persisted_trades = json.load(f)
            if persisted_trades:
                signal_gen.restore_from_trades(persisted_trades)
                _last_trade_count = len(persisted_trades)
                logger.info(f"  Restored {len(persisted_trades)} trades from disk")
            else:
                _last_trade_count = 0
        except Exception as e:
            logger.warning(f"  Could not restore trade history: {e}")
            _last_trade_count = 0
    else:
        _last_trade_count = 0

    logger.info(f"  Trade history: {TRADE_HISTORY_JSON}")

    # 2.5. Broker integration — runtime config + (Binance Testnet | paper) client
    global broker_client, broker_config, _broker_position, execution_guards, kill_switch
    broker_config = load_broker_config()
    broker_client = make_broker_client()
    # signal_gen.position is always 0 after restore_from_trades; safe sentinel.
    _broker_position = 0
    logger.info(
        f"  Broker: mode={broker_client.mode} "
        f"auto_execute={broker_config.auto_execute} "
        f"symbol={broker_config.default_symbol} "
        f"qty={broker_config.default_qty} "
        f"btc_qty={broker_config.default_btc_qty} "
        f"eth_qty={broker_config.default_eth_qty}"
    )

    # 2.6. Execution guardrails (configured from env vars)
    execution_guards = ExecutionGuards()
    logger.info(f"  Execution guards: {execution_guards.to_dict()}")

    # 2.7. Session max-loss kill switch (state survives restarts)
    kill_switch = KillSwitch.load(KILL_SWITCH_JSON)
    logger.info(f"  Kill switch: {kill_switch.to_dict()}")

    if broker_client.mode == "demo":
        logger.info("=" * 80)
        logger.info("PAPER TRADING DEMO MODE — Binance Testnet, no real funds at risk")
        logger.info("=" * 80)
    else:
        logger.info("SIMULATED PAPER MODE — all orders synthetic, no broker connection")

    # 3. Warm up with historical candles
    logger.info(f"Warming up with {WARMUP_CANDLES} historical candles...")
    try:
        btc_candles = fetch_historical_candles("BTCUSDT", "1m", WARMUP_CANDLES)
        eth_candles = fetch_historical_candles("ETHUSDT", "1m", WARMUP_CANDLES)
        feature_calc.seed_historical("BTC", btc_candles)
        feature_calc.seed_historical("ETH", eth_candles)

        status = feature_calc.get_data_quality_status()
        logger.info(f"  BTC: {status['btc_candles']} candles | ETH: {status['eth_candles']} candles")
        logger.info(f"  Ready: {status['ready']}")

        # Verify features work
        test_features = feature_calc.calculate_features()
        if test_features is not None:
            test_proba = model.predict_proba(test_features)[:, 1]
            logger.info(f"  Initial P(up) = {float(test_proba[0]):.4f}")
        else:
            logger.warning("  Initial feature calc returned None — will retry with live data")

    except Exception as e:
        logger.error(f"Warm-up failed: {e}. Server will attempt warm-up from live data.")

    # 4. Start data WebSocket task
    data_task = asyncio.create_task(connect_to_data_stream())
    logger.info("=" * 80)
    logger.info(f"{version.upper()} MODEL SERVER - READY")
    logger.info("=" * 80)

    yield

    # Shutdown
    logger.info("Shutting down model server...")
    data_task.cancel()
    try:
        await data_task
    except asyncio.CancelledError:
        pass


# ── Auth helpers ────────────────────────────────────────────────────────────

def _verify_password(plain: str, hashed: str) -> bool:
    """Verify password against bcrypt hash."""
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def _create_jwt(username: str) -> str:
    """Create JWT token for authenticated user."""
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS)
    return jwt.encode(
        {"sub": username, "exp": expire},
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )


def _verify_jwt(token: str) -> Optional[str]:
    """Verify JWT and return username, or None if invalid."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


def _get_token_from_request(authorization: Optional[str] = None) -> Optional[str]:
    """Extract Bearer token from Authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return authorization[7:].strip()


def _require_auth_or_skip(authorization: Optional[str] = None):
    """If AUTH_REQUIRED, validate JWT. Otherwise allow."""
    if not AUTH_REQUIRED:
        return
    token = _get_token_from_request(authorization)
    if not token or not _verify_jwt(token):
        raise HTTPException(status_code=401, detail="Invalid or missing token")


def _verify_api_key(api_key: Optional[str] = None, x_api_key: Optional[str] = None):
    """Verify subscriber API key for /api/predict."""
    key = api_key or x_api_key
    if not SUBSCRIBER_API_KEY or key != SUBSCRIBER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Broker helpers ─────────────────────────────────────────────────────────


def _broker_summary() -> Dict:
    """Compact broker snapshot exposed in /status and WS payloads."""
    return {
        "mode": broker_client.mode if broker_client is not None else "unknown",
        # Why the demo broker fell back to paper (e.g. expired testnet keys);
        # None when the broker is running as configured.
        "init_error": getattr(broker_client, "init_error", None),
        "auto_execute": broker_config.auto_execute if broker_config is not None else False,
        "default_symbol": broker_config.default_symbol if broker_config is not None else "BTCUSDT",
        "default_qty": broker_config.default_qty if broker_config is not None else 0.001,
        "default_btc_qty": broker_config.default_btc_qty if broker_config is not None else 0.001,
        "default_eth_qty": broker_config.default_eth_qty if broker_config is not None else 0.05,
        "guards": execution_guards.to_dict() if execution_guards is not None else None,
        "kill_switch": kill_switch.to_dict() if kill_switch is not None else None,
        # Model vs broker position (-1/0/+1). Drift means the broker was never
        # told about the model's current position (entry happened while
        # auto-exec was off / broker unavailable / before a restart). With
        # auto-exec on it self-heals on the next candle; surfaced so the
        # operator sees it rather than trusting silence.
        "position_drift": {
            "model": signal_gen.position if signal_gen is not None else 0,
            "broker": _broker_position,
            "drifted": (signal_gen.position if signal_gen is not None else 0) != _broker_position,
        },
    }


# ── FastAPI app ────────────────────────────────────────────────────────────

app = FastAPI(
    title=f"{MODEL_VERSION} Trading Model Server",
    description="Real-time BTC/ETH ratio trading signals via XGBClassifier",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.pairpredictions.com",
        "https://pairpredictions.com",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app|http://localhost:\d+|http://127\.0\.0\.1:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    username: str
    password: str


@app.get("/")
async def root():
    return {
        "status": "running",
        "service": f"{MODEL_VERSION} Trading Model Server",
        "version": MODEL_VERSION,
        "model": "XGBClassifier",
    }


@app.post("/api/login")
async def api_login(req: LoginRequest):
    """Authenticate user and return JWT."""
    if not AUTH_REQUIRED:
        return {"token": _create_jwt(ADMIN_USERNAME), "user": ADMIN_USERNAME}
    if not ADMIN_PASSWORD_HASH:
        raise HTTPException(status_code=503, detail="Auth not configured")
    if req.username != ADMIN_USERNAME or not _verify_password(req.password, ADMIN_PASSWORD_HASH):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"token": _create_jwt(req.username), "user": req.username}


@app.get("/api/me")
async def api_me(authorization: Optional[str] = Header(None)):
    """Validate JWT and return user info."""
    if not AUTH_REQUIRED:
        return {"user": ADMIN_USERNAME, "authenticated": True}
    token = _get_token_from_request(authorization)
    username = _verify_jwt(token) if token else None
    if not username:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return {"user": username, "authenticated": True}


@app.post("/api/logout")
async def api_logout():
    """Logout (client should discard token)."""
    return {"status": "ok"}


@app.get("/api/predict")
async def api_predict(
    request: Request,
    api_key: Optional[str] = Query(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """Return latest prediction. Auth: X-API-Key header or ?api_key= query param."""
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        key = auth_header[7:].strip()
    else:
        key = api_key or x_api_key
    _verify_api_key(api_key=key, x_api_key=key)
    if not latest_signal_data.get("timestamp"):
        raise HTTPException(status_code=503, detail="No prediction available yet")
    return _sanitize_for_json(latest_signal_data)


@app.get("/version")
async def version():
    """Return current model version and available versions."""
    return {
        "current": MODEL_VERSION,
        "available": list_versions(),
    }


@app.get("/status")
async def status():
    return {
        "data_quality": feature_calc.get_data_quality_status() if feature_calc else None,
        "latest_signal": latest_signal_data.get("signal"),
        "last_update": latest_signal_data.get("timestamp"),
        "active_clients": len(active_connections),
        "model_version": MODEL_VERSION,
        "n_features": len(feature_names),
        "broker": _broker_summary(),
        "last_exec": _last_execution_event,
        "shadow": _shadow_snapshot(),
    }


@app.get("/shadow/status")
async def shadow_status(authorization: Optional[str] = Header(None)):
    """Shadow candidate vs primary comparison. enabled=false when no shadow is configured."""
    _require_auth_or_skip(authorization)
    if shadow_signal_gen is None:
        return {"enabled": False, "primary_version": MODEL_VERSION}

    shadow_trades = []
    if SHADOW_TRADES_JSON and SHADOW_TRADES_JSON.exists():
        try:
            with open(SHADOW_TRADES_JSON, "r") as f:
                shadow_trades = json.load(f)
        except Exception:
            shadow_trades = []

    def _side(gen, trades):
        wins = getattr(gen, "wins", 0)
        losses = getattr(gen, "losses", 0)
        closed = wins + losses
        return {
            "position": gen.position,
            "balance": getattr(gen, "balance", None),
            "total_pnl": getattr(gen, "total_pnl", None),
            "total_trades": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / closed) if closed else 0.0,
            "recent_trades": trades[-20:],
            "stats": compute_trade_stats(trades, getattr(gen, "starting_balance", 1000.0)),
            # Degradation lens: same engine over only the last 20 closed trades,
            # so recent form can be compared against lifetime honestly (n is
            # reported; small-sample metrics stay suppressed).
            "recent_stats": compute_trade_stats(trades[-20:], getattr(gen, "starting_balance", 1000.0)),
        }

    primary_trades = []
    if TRADE_HISTORY_JSON.exists():
        try:
            with open(TRADE_HISTORY_JSON, "r") as f:
                primary_trades = json.load(f)
        except Exception:
            primary_trades = []

    # Matched comparison window: lifetime baseline stats aren't comparable to a
    # candidate that only started observing recently, so the primary side also
    # gets stats restricted to the shadow's observation window (earliest shadow
    # trade, or this server boot when the shadow hasn't traded yet).
    window_since = _SERVER_STARTED_TS
    if shadow_trades:
        first_shadow = min((t.get("entry_time") or window_since) for t in shadow_trades)
        window_since = min(window_since, first_shadow)
    primary_in_window = [t for t in primary_trades if (t.get("exit_time") or 0) >= window_since]

    primary_side = _side(signal_gen, primary_trades)
    primary_side["window_stats"] = compute_trade_stats(
        primary_in_window, getattr(signal_gen, "starting_balance", 1000.0)
    )

    return _sanitize_for_json({
        "enabled": True,
        "primary_version": MODEL_VERSION,
        "shadow_version": SHADOW_MODEL_VERSION,
        "window_since": window_since,
        "primary": primary_side,
        "shadow": _side(shadow_signal_gen, shadow_trades),
        "last_shadow_signal": _last_shadow_signal,
    })


@app.get("/trades")
async def get_trades(authorization: Optional[str] = Header(None)):
    """Return persisted trade history from JSON file."""
    _require_auth_or_skip(authorization)
    if TRADE_HISTORY_JSON.exists():
        try:
            with open(TRADE_HISTORY_JSON, "r") as f:
                trades = json.load(f)
            return {"trades": trades, "count": len(trades)}
        except Exception:
            return {"trades": [], "count": 0}
    return {"trades": [], "count": 0}


@app.delete("/trades/clear")
async def clear_trades(authorization: Optional[str] = Header(None)):
    """Clear all persisted trade history and reset signal generator state."""
    _require_auth_or_skip(authorization)
    global _last_trade_count, latest_signal_data

    # Clear JSON file
    _ensure_live_dir()
    with open(TRADE_HISTORY_JSON, "w") as f:
        json.dump([], f)

    # Clear CSV file
    if TRADE_HISTORY_CSV.exists():
        TRADE_HISTORY_CSV.unlink()

    # Reset signal generator in-memory state
    if signal_gen is not None:
        signal_gen.trades = []
        signal_gen.balance = signal_gen.starting_balance
        signal_gen.total_pnl = 0.0
        signal_gen.wins = 0
        signal_gen.losses = 0
    _last_trade_count = 0

    # Purge stale trades from the cached WebSocket broadcast so
    # reconnecting clients don't receive old portfolio data.
    if latest_signal_data.get("portfolio"):
        latest_signal_data["portfolio"]["recent_trades"] = []
        latest_signal_data["portfolio"]["total_trades"] = 0
        latest_signal_data["portfolio"]["wins"] = 0
        latest_signal_data["portfolio"]["losses"] = 0
        latest_signal_data["portfolio"]["win_rate"] = 0
        latest_signal_data["portfolio"]["total_pnl"] = 0.0
        latest_signal_data["portfolio"]["total_pnl_pct"] = 0.0
        latest_signal_data["portfolio"]["balance"] = signal_gen.starting_balance if signal_gen else 1000.0

    logger.info("Trade history cleared (JSON + CSV + in-memory state + cached broadcast)")
    return {"status": "cleared"}


@app.get("/features/importance")
async def features_importance(authorization: Optional[str] = Header(None)):
    """Return all 50 feature names with their XGBoost importance scores."""
    _require_auth_or_skip(authorization)
    if model is None or not feature_names:
        return {"features": [], "error": "Model not loaded"}

    try:
        # Get feature importance from the model (gain-based)
        importance_map = model.get_booster().get_score(importance_type="gain")

        # The booster uses f0, f1, ... naming - map back to feature names
        result = []
        for idx, name in enumerate(feature_names):
            # XGBoost feature key format
            key = f"f{idx}"
            score = importance_map.get(key, 0.0)
            result.append({
                "rank": 0,  # will be set after sorting
                "name": name,
                "importance": float(score),
            })

        # Sort by importance descending and assign ranks
        result.sort(key=lambda x: x["importance"], reverse=True)
        for i, item in enumerate(result):
            item["rank"] = i + 1

        return {"features": result, "count": len(result)}
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}", exc_info=True)
        return {"features": [], "error": str(e)}


# ── Broker control ────────────────────────────────────────────────────────


class BrokerConfigUpdate(BaseModel):
    """Partial update payload for POST /broker/config."""

    auto_execute: Optional[bool] = None
    default_symbol: Optional[str] = None
    default_qty: Optional[float] = None
    default_btc_qty: Optional[float] = None
    default_eth_qty: Optional[float] = None


def _broker_or_503() -> BrokerClient:
    if broker_client is None:
        raise HTTPException(status_code=503, detail="Broker not initialized")
    return broker_client


@app.get("/broker/config")
async def get_broker_config(authorization: Optional[str] = Header(None)):
    """Return the current runtime broker config + mode."""
    _require_auth_or_skip(authorization)
    return _broker_summary()


@app.post("/broker/config")
async def update_broker_config_route(
    update: BrokerConfigUpdate,
    authorization: Optional[str] = Header(None),
):
    """Update auto_execute / default_symbol / default_qty at runtime."""
    _require_auth_or_skip(authorization)
    global broker_config
    if broker_config is None:
        raise HTTPException(status_code=503, detail="Broker config not initialized")

    try:
        new_cfg = apply_partial_update(broker_config, update.model_dump(exclude_none=True))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid broker config: {e}")

    save_broker_config(new_cfg)
    broker_config = new_cfg
    logger.info(f"Broker config updated: {new_cfg.model_dump()}")
    return _broker_summary()


@app.get("/broker/balance")
async def get_broker_balance(authorization: Optional[str] = Header(None)):
    """Account balances. Additive fields: ``ok`` and ``error`` expose a broker
    error envelope (e.g. expired testnet keys) instead of an ambiguous empty
    list. Existing keys (``mode``, ``assets``, ``raw``) are unchanged."""
    _require_auth_or_skip(authorization)
    bc = _broker_or_503()
    balance = await asyncio.to_thread(bc.get_balance)
    raw = balance.get("raw")
    error = None
    if isinstance(raw, dict) and isinstance(raw.get("code"), int) and raw["code"] < 0:
        error = f"{raw.get('code')}: {raw.get('msg', 'unknown broker error')}"
    init_error = getattr(bc, "init_error", None)
    if error is None and init_error:
        error = f"broker degraded to paper: {init_error}"
    return {"mode": bc.mode, "ok": error is None, "error": error, **balance}


# Broker-log actions that constitute order activity (polling reads like
# get_balance/get_order_status are logged too but are noise for operators).
_ACTIVITY_ACTIONS = {
    "place_order",
    "place_test_order",
    "cancel_order",
    "bracket_stop_market",
    "bracket_take_profit_market",
}


@app.get("/broker/executions")
async def get_broker_executions(
    limit: int = Query(50, ge=1, le=200),
    authorization: Optional[str] = Header(None),
):
    """Auto-execute event history (persisted across restarts).

    Each event appears once, newest first, with its most recent snapshot
    (reconciled fills win over the submit-time row).
    """
    _require_auth_or_skip(authorization)
    rows = _read_jsonl_tail(EXEC_EVENTS_JSONL, limit)
    latest_by_ts: Dict[str, dict] = {}
    for row in rows:  # oldest→newest, so later (reconciled) rows overwrite
        ts = str(row.get("timestamp", ""))
        if ts:
            latest_by_ts[ts] = row
    events = sorted(latest_by_ts.values(), key=lambda r: r.get("timestamp", ""), reverse=True)
    events = events[:limit]
    return {"executions": events, "count": len(events)}


@app.get("/broker/activity")
async def get_broker_activity(
    limit: int = Query(50, ge=1, le=200),
    authorization: Optional[str] = Header(None),
):
    """Order-level broker interactions (orders, brackets, cancels), newest
    first, from the append-only broker log. Secrets are redacted at write
    time. Read-only view — nothing here mutates broker or history state."""
    _require_auth_or_skip(authorization)
    rows = _read_jsonl_tail(JSONL_LOG_PATH, limit * 4)  # log includes non-activity rows
    activity = [r for r in rows if r.get("action") in _ACTIVITY_ACTIONS]
    activity.reverse()  # newest first
    activity = activity[:limit]
    return {"activity": activity, "count": len(activity)}


@app.get("/broker/positions")
async def get_broker_positions(authorization: Optional[str] = Header(None)):
    _require_auth_or_skip(authorization)
    bc = _broker_or_503()
    positions = await asyncio.to_thread(bc.get_open_positions)
    return {"mode": bc.mode, "positions": positions}


@app.post("/trade")
async def post_trade(
    req: OrderRequest,
    authorization: Optional[str] = Header(None),
):
    """Place a real order through the configured broker (demo or paper).

    For demo mode, polls for fill confirmation before returning so callers
    get confirmed filled_qty and avg_price rather than the initial NEW status.
    """
    _require_auth_or_skip(authorization)
    if (
        kill_switch is not None
        and not kill_switch.entries_allowed
        and not getattr(req, "reduce_only", False)
    ):
        raise HTTPException(
            status_code=423,
            detail="Session kill switch tripped. Only reduce-only exits are allowed; re-arm or disarm to trade.",
        )
    bc = _broker_or_503()
    resp = await asyncio.to_thread(bc.place_order, req)
    result = resp.model_dump()

    # Enrich with confirmed fill data when the broker supports status polling
    if resp.broker_order_id and resp.status not in _TERMINAL_STATUSES:
        for delay in [0.5, 1.0, 2.0]:
            await asyncio.sleep(delay)
            try:
                fill = await asyncio.to_thread(bc.get_order_status, req.symbol, resp.broker_order_id)
            except Exception:
                break
            if fill is None:
                break  # paper broker
            result["status"] = fill["status"]
            result["filled_qty"] = fill["filled_qty"]
            result["avg_price"] = fill["avg_price"]
            if fill["status"] in _TERMINAL_STATUSES:
                break

    return result


@app.post("/broker/kill-switch")
async def post_kill_switch(
    payload: dict,
    authorization: Optional[str] = Header(None),
):
    """Arm, disarm, or re-arm the session max-loss kill switch.

    Body: {"action": "arm" | "disarm" | "rearm", "limit_usdt": float (arm only)}
    All transitions are explicit, logged as KILL_SWITCH lines, and persisted.
    """
    _require_auth_or_skip(authorization)
    if kill_switch is None:
        raise HTTPException(status_code=503, detail="Kill switch not initialized")
    action = str(payload.get("action", "")).lower()
    try:
        if action == "arm":
            kill_switch.arm(payload.get("limit_usdt"))
        elif action == "disarm":
            kill_switch.disarm()
        elif action == "rearm":
            kill_switch.rearm()
        else:
            raise HTTPException(status_code=400, detail=f"invalid action: {action!r}")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return _broker_summary()


@app.post("/trade/test")
async def post_trade_test(
    req: OrderRequest,
    authorization: Optional[str] = Header(None),
):
    """Send the order through Binance's test endpoint (no real fill)."""
    _require_auth_or_skip(authorization)
    bc = _broker_or_503()
    resp = await asyncio.to_thread(bc.place_test_order, req)
    return resp.model_dump()


@app.delete("/broker/order/{order_id}")
async def delete_broker_order(
    order_id: str,
    symbol: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    _require_auth_or_skip(authorization)
    bc = _broker_or_503()
    ok = await asyncio.to_thread(bc.cancel_order, order_id, symbol)
    return {"ok": ok, "order_id": order_id, "symbol": symbol}


@app.websocket("/ws/signals")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming trading signals to frontend."""
    # Validate JWT from query param when AUTH_REQUIRED
    if AUTH_REQUIRED:
        token = websocket.scope.get("query_string", b"").decode()
        params = dict(p.split("=", 1) for p in token.split("&") if "=" in p)
        jwt_token = params.get("token", "")
        if not jwt_token or not _verify_jwt(jwt_token):
            await websocket.close(code=4001)
            return
    await websocket.accept()
    active_connections.add(websocket)

    logger.info(f"Client connected. Total: {len(active_connections)}")

    try:
        # Send latest data immediately
        if latest_signal_data["timestamp"]:
            await websocket.send_json(latest_signal_data)

        while True:
            try:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
                elif data == "get_latest":
                    if latest_signal_data["timestamp"]:
                        await websocket.send_json(latest_signal_data)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WS error: {e}")
                break
    finally:
        active_connections.discard(websocket)
        logger.info(f"Client disconnected. Total: {len(active_connections)}")


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting {MODEL_VERSION} Model Server on http://127.0.0.1:8888")
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="info")
