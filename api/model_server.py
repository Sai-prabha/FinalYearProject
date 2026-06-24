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
from .version_config import get_strategy_config, list_versions
from .broker_client import (
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

# ── Auth config (optional; set AUTH_REQUIRED=true on Railway for production) ──
AUTH_REQUIRED = os.environ.get("AUTH_REQUIRED", "false").lower() == "true"
ADMIN_USERNAME = "adm1nFYP"
ADMIN_PASSWORD_HASH = os.environ.get("ADMIN_PASSWORD_HASH", "")
JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret-change-in-production")
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
TRADE_HISTORY_CSV = LIVE_DATA_DIR / "trade_history.csv"

# ── Data API ──────────────────────────────────────────────────────────────
REST_BASE = "https://api.binance.com/api/v3"
WS_STREAM_BASE = "wss://stream.binance.com:9443/stream"
WARMUP_CANDLES = 1000

# ── Version selection ──────────────────────────────────────────────────────
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v4.15")

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

# Broker integration state
broker_client: Optional[BrokerClient] = None
broker_config: Optional[BrokerConfig] = None
# Last position the broker has been instructed to hold (-1, 0, +1).
# Updated after every attempt so transient outages don't cause order storms.
_broker_position: int = 0
# Most recent auto-execute event — included in every WS broadcast after it fires.
_last_execution_event: Optional[Dict] = None

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
    # Recompute all_ok with final statuses
    _last_execution_event["all_ok"] = all(
        leg["status"] not in ("REJECTED", "ERROR", "CANCELED", "EXPIRED")
        for leg in legs
    )
    _last_execution_event["reconciled"] = True
    # Broadcast a lightweight message so the frontend doesn't wait for the next candle
    await broadcast_to_clients({
        "type": "exec_reconciled",
        "last_exec": _last_execution_event,
    })
    logger.info(
        "Reconciliation broadcast: %d legs, all_ok=%s",
        len(legs),
        _last_execution_event["all_ok"],
    )


async def _execute_broker_position_change(prev_pos: int, new_pos: int, timestamp: int) -> None:
    """Execute broker orders for a model position transition and advance _broker_position.

    Handles three cases:
      0 → ±1 : entry orders for both legs
      ±1 → 0 : reduce-only exit orders for both legs
      ±1 → ∓1: exit old position then open new (reversal)

    ``_broker_position`` is updated regardless of order outcomes so a
    transient broker outage does not trigger an infinite order loop.
    Results are stored in ``_last_execution_event`` and a background reconciliation
    task enriches fill data (filled_qty, avg_price) then re-broadcasts to WS clients.
    """
    global _broker_position, _last_execution_event

    legs: list = []
    if prev_pos != 0:
        legs.extend(await _place_exit_orders(prev_pos, timestamp))

    if new_pos != 0:
        legs.extend(await _place_entry_orders(new_pos, timestamp))

    _broker_position = new_pos

    pos_label = {-1: "SHORT", 0: "FLAT", 1: "LONG"}
    all_ok = all(leg["status"] not in ("REJECTED", "ERROR") for leg in legs)
    _last_execution_event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prev_pos": prev_pos,
        "new_pos": new_pos,
        "transition": f"{pos_label.get(prev_pos,'?')}→{pos_label.get(new_pos,'?')}",
        "legs": legs,
        "all_ok": all_ok,
        "reconciled": False,
    }
    status_str = "OK" if all_ok else "PARTIAL/FAILED"
    logger.info(
        f"AUTO-EXEC [{status_str}] {pos_label.get(prev_pos,'?')}→{pos_label.get(new_pos,'?')} "
        f"({len(legs)} legs): "
        + ", ".join(f"{l['symbol']} {l['side']} {l['status']}" for l in legs)
    )
    # Fire reconciliation in background — updates legs and re-broadcasts within ~3.5 s
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

    # Auto-execute: detect position transitions and send dual-leg broker orders
    if (
        broker_client is not None
        and broker_config is not None
        and broker_client.mode == "demo"
        and broker_config.auto_execute
    ):
        _post_update_pos = signal_gen.position
        if _post_update_pos != _broker_position:
            await _execute_broker_position_change(
                _broker_position, _post_update_pos, candle["time"]
            )
    elif (
        broker_config is not None
        and broker_config.auto_execute
        and broker_client is not None
        and broker_client.mode != "demo"
    ):
        # Warn once per candle so the operator knows orders are not being sent.
        logger.warning(
            "auto_execute=True but broker mode=%r — restart with BINANCE_ENV=demo "
            "and credentials to send real Testnet orders",
            broker_client.mode,
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

    if version == "v4.16":
        strategy_cfg = get_strategy_config("v4.16")
        signal_gen = V416SignalGenerator(cfg=strategy_cfg)
        logger.info(f"  Signal generator: V416SignalGenerator (asymmetric TP/SL, Kelly sizing)")
    else:
        # v4.15 (default) — use V414SignalGenerator with production config
        strategy = config["strategy_params"]
        signal_gen = V414SignalGenerator(
            entry_threshold=strategy["entry_threshold"],
            exit_threshold=strategy["exit_threshold"],
            min_hold=strategy["min_hold"],
            cooldown=strategy["cooldown"],
            cb_lookback=strategy["cb_lookback"],
            cb_threshold=strategy["cb_threshold"],
        )
        logger.info(f"  Signal generator: V414SignalGenerator (v4.15 baseline)")

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
    global broker_client, broker_config, _broker_position
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
        "auto_execute": broker_config.auto_execute if broker_config is not None else False,
        "default_symbol": broker_config.default_symbol if broker_config is not None else "BTCUSDT",
        "default_qty": broker_config.default_qty if broker_config is not None else 0.001,
        "default_btc_qty": broker_config.default_btc_qty if broker_config is not None else 0.001,
        "default_eth_qty": broker_config.default_eth_qty if broker_config is not None else 0.05,
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
    """Authenticate user and return JWT. Credentials: adm1nFYP / FYP2026!"""
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
    }


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
    _require_auth_or_skip(authorization)
    bc = _broker_or_503()
    balance = await asyncio.to_thread(bc.get_balance)
    return {"mode": bc.mode, **balance}


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
