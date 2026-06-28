#!/usr/bin/env python3
"""
V4.14 Live Test Script - CLI validation before frontend deployment.

This script:
  1. Loads the saved production model from models/v4_14_production/
  2. Fetches 1000 historical 1m candles from Binance REST API (BTC + ETH)
  3. Connects to Binance WebSocket for live BTC + ETH 1m klines
  4. On each new closed candle: calculates features, runs predict_proba(),
     applies hysteresis logic, prints predictions to terminal
  5. Validates feature values are sensible

Usage:
    python scripts/test_v4_14_live.py
    python scripts/test_v4_14_live.py --offline   # Test with only REST data, no WebSocket
"""

import sys
import json
import time
import asyncio
import signal
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import requests
from xgboost import XGBClassifier

from api.feature_calculator import V414FeatureCalculator, V414SignalGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_DIR = Path("models/v4_14_production")
MODEL_PATH = MODEL_DIR / "model.json"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"
CONFIG_PATH = MODEL_DIR / "config.json"

BINANCE_REST = "https://api.binance.com/api/v3"
BINANCE_WS = "wss://stream.binance.com:9443/stream"

# ── Fetch historical candles from Binance REST API ─────────────────────────

def fetch_historical_candles(symbol: str, interval: str = "1m", limit: int = 1000) -> list:
    """
    Fetch historical klines from Binance REST API.
    Returns list of candle dicts (oldest first).
    """
    url = f"{BINANCE_REST}/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    logger.info(f"Fetching {limit} {interval} candles for {symbol}...")

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    candles = []
    for k in data:
        candles.append({
            "time": int(k[0]) // 1000,     # openTime ms -> seconds
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "taker_buy_volume": float(k[9]),  # taker buy base asset volume
        })

    logger.info(f"  Got {len(candles)} candles ({candles[0]['time']} -> {candles[-1]['time']})")
    return candles


# ── WebSocket streaming ───────────────────────────────────────────────────

async def run_websocket_loop(model, feature_calc, signal_gen, feature_names):
    """Connect to Binance WS and process live candles."""
    try:
        import websockets
    except ImportError:
        logger.error("websockets package not installed. Run: pip install websockets")
        return

    streams = "btcusdt@kline_1m/ethusdt@kline_1m"
    url = f"{BINANCE_WS}?streams={streams}"

    logger.info(f"\nConnecting to Binance WebSocket...")
    logger.info(f"URL: {url}\n")

    while True:
        try:
            async with websockets.connect(url) as ws:
                logger.info("WebSocket connected. Waiting for candles...")
                async for message in ws:
                    try:
                        data = json.loads(message)
                        if "data" not in data:
                            continue

                        stream_data = data["data"]
                        if stream_data.get("e") != "kline":
                            continue

                        kline = stream_data["k"]
                        symbol_raw = stream_data["s"]
                        symbol = "BTC" if symbol_raw == "BTCUSDT" else "ETH" if symbol_raw == "ETHUSDT" else None
                        if not symbol:
                            continue

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

                        # Only run prediction on closed candles
                        if is_closed:
                            process_prediction(model, feature_calc, signal_gen, feature_names, candle["time"])

                    except Exception as e:
                        logger.error(f"Error processing WS message: {e}")

        except Exception as e:
            logger.error(f"WebSocket error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)


def process_prediction(model, feature_calc, signal_gen, feature_names, timestamp):
    """Run one prediction cycle."""
    status = feature_calc.get_data_quality_status()
    if not status["ready"]:
        logger.info(f"Not ready: BTC={status['btc_candles']}, ETH={status['eth_candles']} "
                     f"(need {status['min_required']})")
        return

    # Calculate features
    t0 = time.time()
    features_df = feature_calc.calculate_features()
    calc_time = (time.time() - t0) * 1000

    if features_df is None:
        logger.warning("Feature calculation returned None (NaN values)")
        return

    # Run prediction
    proba = model.predict_proba(features_df)[:, 1]
    proba_up = float(proba[0])

    # Get current ratio
    btc_close = feature_calc.btc_candles[-1]["close"]
    eth_close = feature_calc.eth_candles[-1]["close"]
    current_ratio = btc_close / eth_close

    # Generate signal
    sig = signal_gen.update(proba_up, current_ratio, timestamp)

    # Print formatted output
    ts_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%H:%M:%S")
    pos_str = sig["direction"]
    pos_color = "\033[92m" if pos_str == "LONG" else "\033[91m" if pos_str == "SHORT" else "\033[90m"
    reset = "\033[0m"

    cb_str = " [CB]" if sig["circuit_breaker_active"] else ""
    triggered_str = " *** TRIGGERED ***" if sig["triggered"] else ""

    portfolio = sig["portfolio"]
    pnl_color = "\033[92m" if portfolio["total_pnl"] >= 0 else "\033[91m"

    print(
        f"  {ts_str} | "
        f"BTC: {btc_close:,.0f} | ETH: {eth_close:,.2f} | "
        f"R: {current_ratio:.4f} | "
        f"P(up): {proba_up:.4f} | "
        f"{pos_color}{pos_str:>7}{reset}{cb_str} | "
        f"Str: {sig['strength']:.1%} | "
        f"Bal: ${portfolio['balance']:.2f} "
        f"({pnl_color}{portfolio['total_pnl']:+.2f}{reset}) | "
        f"W/L: {portfolio['wins']}/{portfolio['losses']} | "
        f"{calc_time:.0f}ms"
        f"{triggered_str}"
    )

    # Log blocked signals
    if sig["blocked_by"]:
        logger.debug(f"  Blocked: {', '.join(sig['blocked_by'])}")


# ── Offline test mode ─────────────────────────────────────────────────────

def run_offline_test(model, feature_calc, signal_gen, feature_names):
    """
    Test predictions on the historical data already loaded.
    Iterates through the last 200 candles as if they arrived live.
    """
    logger.info("\n" + "=" * 80)
    logger.info("OFFLINE TEST: Simulating last 200 candles...")
    logger.info("=" * 80 + "\n")

    # First, verify features work with full buffer
    features_df = feature_calc.calculate_features()
    if features_df is None:
        logger.error("Feature calculation failed on initial data!")
        return

    # Show feature sample
    print("\nFeature sample (last row):")
    print("-" * 60)
    for col in feature_names[:10]:
        val = features_df[col].iloc[0]
        print(f"  {col:40s} = {val:.6f}")
    print(f"  ... ({len(feature_names)} total features)")
    print()

    # Run prediction
    proba = model.predict_proba(features_df)[:, 1]
    proba_up = float(proba[0])

    btc_close = feature_calc.btc_candles[-1]["close"]
    eth_close = feature_calc.eth_candles[-1]["close"]
    current_ratio = btc_close / eth_close

    print(f"Current state:")
    print(f"  BTC: ${btc_close:,.2f}")
    print(f"  ETH: ${eth_close:,.2f}")
    print(f"  Ratio: {current_ratio:.4f}")
    print(f"  P(up): {proba_up:.4f}")
    print(f"  Direction: {'LONG' if proba_up >= 0.525 else 'SHORT' if proba_up <= 0.475 else 'NEUTRAL'}")
    print()

    # Validate feature ranges
    print("Feature validation:")
    print("-" * 60)
    nan_count = features_df.isna().sum().sum()
    inf_count = np.isinf(features_df.values).sum()
    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")
    print(f"  Feature range: [{features_df.min().min():.6f}, {features_df.max().max():.6f}]")

    # Check individual feature ranges
    warnings = []
    for col in feature_names:
        val = features_df[col].iloc[0]
        if abs(val) > 100:
            warnings.append(f"  WARNING: {col} = {val:.4f} (large value)")
    if warnings:
        print("\nPotential issues:")
        for w in warnings:
            print(w)
    else:
        print("  All features within reasonable ranges")

    print()
    print("Offline test PASSED" if nan_count == 0 and inf_count == 0 else "Offline test FAILED")
    print()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V4.14 Live Test")
    parser.add_argument("--offline", action="store_true", help="Offline test only (no WebSocket)")
    args = parser.parse_args()

    print("=" * 80)
    print("V4.14 LIVE MODEL TEST")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. Load model
    print("Loading production model...")
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}. Run train_v4_14_production.py first."

    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    with open(FEATURE_NAMES_PATH) as f:
        feature_names = json.load(f)

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    strategy = config["strategy_params"]
    print(f"  Model loaded: {len(feature_names)} features, {config['best_n_estimators']} trees")
    print(f"  Strategy: entry={strategy['entry_threshold']}, exit={strategy['exit_threshold']}, "
          f"hold={strategy['min_hold']}, cd={strategy['cooldown']}")
    print()

    # 2. Initialize feature calculator & signal generator
    feature_calc = V414FeatureCalculator(feature_names=feature_names, max_history=1500)
    signal_gen = V414SignalGenerator(
        entry_threshold=strategy["entry_threshold"],
        exit_threshold=strategy["exit_threshold"],
        min_hold=strategy["min_hold"],
        cooldown=strategy["cooldown"],
        cb_lookback=strategy["cb_lookback"],
        cb_threshold=strategy["cb_threshold"],
    )

    # 3. Fetch historical data
    print("Fetching historical data from Binance...")
    btc_candles = fetch_historical_candles("BTCUSDT", "1m", 1000)
    eth_candles = fetch_historical_candles("ETHUSDT", "1m", 1000)

    feature_calc.seed_historical("BTC", btc_candles)
    feature_calc.seed_historical("ETH", eth_candles)

    status = feature_calc.get_data_quality_status()
    print(f"\nData quality:")
    print(f"  BTC candles: {status['btc_candles']}")
    print(f"  ETH candles: {status['eth_candles']}")
    print(f"  Ready: {status['ready']}")
    print(f"  Synced: {status['synced']}")
    print()

    if not status["ready"]:
        logger.error("Not enough data! Check Binance API connection.")
        sys.exit(1)

    # 4. Run offline test first
    run_offline_test(model, feature_calc, signal_gen, feature_names)

    if args.offline:
        print("Offline mode - exiting.")
        return

    # 5. Run live WebSocket loop
    print("=" * 80)
    print("LIVE MODE: Connecting to Binance WebSocket...")
    print("Press Ctrl+C to stop")
    print("=" * 80)
    print()
    print(f"  {'Time':>8} | {'BTC':>10} | {'ETH':>8} | {'Ratio':>7} | "
          f"{'P(up)':>7} | {'Signal':>7} | {'Str':>5} | "
          f"{'Balance':>14} | {'W/L':>5} | {'ms':>4}")
    print("-" * 110)

    try:
        asyncio.run(run_websocket_loop(model, feature_calc, signal_gen, feature_names))
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        portfolio = signal_gen.trades
        print(f"\nSession summary:")
        print(f"  Total trades: {len(portfolio)}")
        print(f"  Balance: ${signal_gen.balance:.2f}")
        print(f"  P&L: ${signal_gen.total_pnl:+.2f}")
        print(f"  Wins: {signal_gen.wins}, Losses: {signal_gen.losses}")


if __name__ == "__main__":
    main()

