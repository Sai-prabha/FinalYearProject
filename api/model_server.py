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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set
from contextlib import asynccontextmanager

import numpy as np
import requests
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from xgboost import XGBClassifier

from .feature_calculator import V414FeatureCalculator, V414SignalGenerator, V416SignalGenerator
from .version_config import get_strategy_config, list_versions


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

    # Generate signal
    sig = signal_gen.update(proba_up, current_ratio, candle["time"])

    # Persist new trades to disk
    _check_and_persist_new_trades()

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


# ── FastAPI app ────────────────────────────────────────────────────────────

app = FastAPI(
    title=f"{MODEL_VERSION} Trading Model Server",
    description="Real-time BTC/ETH ratio trading signals via XGBClassifier",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "status": "running",
        "service": f"{MODEL_VERSION} Trading Model Server",
        "version": MODEL_VERSION,
        "model": "XGBClassifier",
    }


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
    }


@app.get("/trades")
async def get_trades():
    """Return persisted trade history from JSON file."""
    if TRADE_HISTORY_JSON.exists():
        try:
            with open(TRADE_HISTORY_JSON, "r") as f:
                trades = json.load(f)
            return {"trades": trades, "count": len(trades)}
        except Exception:
            return {"trades": [], "count": 0}
    return {"trades": [], "count": 0}


@app.delete("/trades/clear")
async def clear_trades():
    """Clear all persisted trade history and reset signal generator state."""
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
async def features_importance():
    """Return all 50 feature names with their XGBoost importance scores."""
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


@app.websocket("/ws/signals")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming trading signals to frontend."""
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
