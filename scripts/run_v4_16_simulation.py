#!/usr/bin/env python3
"""
V4.16 Offline Live-Trading Simulation
======================================

Replays the exact live-trading pipeline over a historical window so that
v4.16 (and v4.15 as baseline) can be evaluated on the same market data
without any look-ahead bias.

Pipeline (mirrors api/model_server.py):
  1. Download BTC + ETH 1m candles from Binance REST API
     (with 1000 warmup bars before the simulation start).
  2. Load the production XGBClassifier (models/v4_14_production/).
  3. Feed candles one-at-a-time through V414FeatureCalculator
     (identical to the live feature engine).
  4. On each closed candle (after warmup):
        features -> model.predict_proba() -> P(up)
        P(up)  + ratio -> V414SignalGenerator (v4.15)
        P(up)  + ratio -> V416SignalGenerator (v4.16)
  5. Export trade-level CSVs in the same schema used by the live server.

Assumptions & Approximations
  - Candle data comes from Binance public REST API (BTCUSDT, ETHUSDT).
  - No order-book replay or fill simulation — trades execute at the
    close price of the candle (same assumption as the live system).
  - No explicit transaction-cost deduction (consistent with v4.15 live).
  - Starting balance: $1000.

Usage:
    python scripts/run_v4_16_simulation.py \\
        --start "2026-02-08 22:40:00" \\
        --end   "2026-02-14 21:48:00"
"""

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from xgboost import XGBClassifier

# ── Project paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.feature_calculator import V414FeatureCalculator, V414SignalGenerator, V416SignalGenerator
from api.version_config import V415_CONFIG, V416_CONFIG, get_strategy_config

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
MODEL_DIR = ROOT / "models" / "v4_14_production"
MODEL_PATH = MODEL_DIR / "model.json"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"
CONFIG_PATH = MODEL_DIR / "config.json"

REPORT_DIR = ROOT / "reports" / "v4.16"
CACHE_DIR = ROOT / "data" / "sim_cache"

REST_BASE = "https://api.binance.com/api/v3"
WARMUP_CANDLES = 1000       # same as model_server.py


# ══════════════════════════════════════════════════════════════════════════
# Data Download
# ══════════════════════════════════════════════════════════════════════════

def _dt_to_ms(dt_str: str) -> int:
    """Parse 'YYYY-MM-DD HH:MM:SS' as UTC and return epoch milliseconds."""
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ms_to_dt(ms: int) -> datetime:
    """Convert epoch-ms to UTC datetime."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def fetch_klines_range(
    symbol: str,
    start_ms: int,
    end_ms: int,
    interval: str = "1m",
    limit_per_req: int = 1000,
    sleep_sec: float = 0.5,
) -> List[Dict]:
    """
    Download historical klines from Binance REST API between [start_ms, end_ms].

    Paginates automatically (max 1000 rows per request).
    Returns list of candle dicts with keys:
        time, open, high, low, close, volume, taker_buy_volume
    """
    url = f"{REST_BASE}/klines"
    all_candles: List[Dict] = []
    cursor = start_ms

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": limit_per_req,
        }

        for attempt in range(5):
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    wait = float(resp.headers.get("Retry-After", 10))
                    logger.warning(f"Rate limited, sleeping {wait}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                wait = min(2 ** attempt, 30)
                logger.warning(f"Request error ({e}), retrying in {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Failed to fetch {symbol} candles after 5 attempts")

        data = resp.json()
        if not data:
            break

        for k in data:
            all_candles.append({
                "time": int(k[0]) // 1000,        # open-time in seconds
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "taker_buy_volume": float(k[9]),
            })

        # Advance cursor past the last returned candle's open-time
        last_open_ms = int(data[-1][0])
        if last_open_ms <= cursor:
            break  # no progress
        cursor = last_open_ms + 60_000  # next minute

        if len(data) < limit_per_req:
            break  # reached the end

        time.sleep(sleep_sec)

    logger.info(f"  Downloaded {len(all_candles)} {interval} candles for {symbol}")
    return all_candles


def download_or_load_cache(
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> List[Dict]:
    """Download candle data with local parquet caching."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{symbol}_{start_ms}_{end_ms}"
    cache_path = CACHE_DIR / f"{tag}.parquet"

    if cache_path.exists():
        logger.info(f"  Loading cached {symbol} data from {cache_path}")
        df = pd.read_parquet(cache_path)
        return df.to_dict("records")

    candles = fetch_klines_range(symbol, start_ms, end_ms)
    if candles:
        pd.DataFrame(candles).to_parquet(cache_path, index=False)
        logger.info(f"  Cached {len(candles)} candles to {cache_path}")
    return candles


# ══════════════════════════════════════════════════════════════════════════
# LiveReplayEngine
# ══════════════════════════════════════════════════════════════════════════

class LiveReplayEngine:
    """
    Offline live-trading replay engine.

    Replays the exact pipeline from api/model_server.py:
      1. Seeds V414FeatureCalculator with warmup candles.
      2. Iterates over the simulation window bar-by-bar.
      3. At each bar: add candle -> compute features -> predict -> signal.
      4. Runs both v4.15 (V414SignalGenerator) and v4.16 (V416SignalGenerator).

    No look-ahead: each bar is only processed after it is "closed".
    """

    def __init__(
        self,
        model: XGBClassifier,
        feature_names: List[str],
        config: Dict,
        starting_balance: float = 1000.0,
    ):
        self.model = model
        self.feature_names = feature_names
        self.config = config
        self.starting_balance = starting_balance

        # Feature calculator (same as live)
        self.feature_calc = V414FeatureCalculator(
            feature_names=feature_names,
            max_history=1500,
        )

        # V4.15 signal generator (baseline)
        strategy = config["strategy_params"]
        self.sig_gen_v415 = V414SignalGenerator(
            entry_threshold=strategy["entry_threshold"],
            exit_threshold=strategy["exit_threshold"],
            min_hold=strategy["min_hold"],
            cooldown=strategy["cooldown"],
            cb_lookback=strategy["cb_lookback"],
            cb_threshold=strategy["cb_threshold"],
            starting_balance=starting_balance,
        )

        # V4.16 signal generator (improved)
        self.sig_gen_v416 = V416SignalGenerator(cfg=get_strategy_config("v4.16"))

        # Track equity curves (balance after each bar in sim window)
        self.equity_v415: List[float] = [starting_balance]
        self.equity_v416: List[float] = [starting_balance]
        self.timestamps: List[int] = []

        # Counters
        self._bars_processed = 0
        self._bars_with_features = 0

    def seed_warmup(self, btc_candles: List[Dict], eth_candles: List[Dict]) -> None:
        """Seed the feature calculator with warmup candles (oldest first)."""
        self.feature_calc.seed_historical("BTC", btc_candles)
        self.feature_calc.seed_historical("ETH", eth_candles)
        status = self.feature_calc.get_data_quality_status()
        logger.info(
            f"  Warmup complete: BTC={status['btc_candles']} ETH={status['eth_candles']} "
            f"ready={status['ready']}"
        )

    def process_bar(self, btc_candle: Dict, eth_candle: Dict) -> Optional[Dict]:
        """
        Process a single 1-minute bar (closed candle).

        Mirrors the logic in model_server.py::process_kline_message().
        Returns the signal dict for logging, or None if features not ready.
        """
        # 1. Add candles to feature calculator
        self.feature_calc.add_candle("BTC", btc_candle)
        self.feature_calc.add_candle("ETH", eth_candle)

        self._bars_processed += 1

        # 2. Check data quality
        status = self.feature_calc.get_data_quality_status()
        if not status["ready"]:
            return None

        # 3. Compute features
        features_df = self.feature_calc.calculate_features()
        if features_df is None:
            return None

        self._bars_with_features += 1

        # 4. Model prediction
        proba = self.model.predict_proba(features_df)[:, 1]
        proba_up = float(proba[0])

        # 5. Current ratio
        current_ratio = btc_candle["close"] / eth_candle["close"]
        timestamp = btc_candle["time"]

        # 6. Feed into both signal generators
        sig_v415 = self.sig_gen_v415.update(proba_up, current_ratio, timestamp)
        sig_v416 = self.sig_gen_v416.update(proba_up, current_ratio, timestamp)

        # 7. Track equity
        self.equity_v415.append(self.sig_gen_v415.balance)
        self.equity_v416.append(self.sig_gen_v416.balance)
        self.timestamps.append(timestamp)

        return {
            "timestamp": timestamp,
            "ratio": current_ratio,
            "proba_up": proba_up,
            "v415_direction": sig_v415["direction"],
            "v416_direction": sig_v416["direction"],
        }

    def run(
        self,
        btc_warmup: List[Dict],
        eth_warmup: List[Dict],
        btc_sim: List[Dict],
        eth_sim: List[Dict],
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Run the full simulation.

        Args:
            btc_warmup, eth_warmup: warmup candles (oldest first)
            btc_sim, eth_sim: simulation-window candles (oldest first, same length)

        Returns:
            (v415_trades, v416_trades)
        """
        # Seed warmup
        logger.info("Seeding warmup candles...")
        self.seed_warmup(btc_warmup, eth_warmup)

        # Verify features are working after warmup
        test_features = self.feature_calc.calculate_features()
        if test_features is not None:
            test_proba = self.model.predict_proba(test_features)[:, 1]
            logger.info(f"  Warmup check: P(up) = {float(test_proba[0]):.4f}")
        else:
            logger.warning("  Warmup check: features returned None (may need more bars)")

        # Bar-by-bar replay
        n_bars = min(len(btc_sim), len(eth_sim))
        logger.info(f"Replaying {n_bars} bars...")

        for i in range(n_bars):
            self.process_bar(btc_sim[i], eth_sim[i])

            # Progress logging every 1000 bars
            if (i + 1) % 1000 == 0 or i == n_bars - 1:
                logger.info(
                    f"  Bar {i+1}/{n_bars} | "
                    f"v4.15: ${self.sig_gen_v415.balance:.2f} "
                    f"({len(self.sig_gen_v415.trades)} trades) | "
                    f"v4.16: ${self.sig_gen_v416.balance:.2f} "
                    f"({len(self.sig_gen_v416.trades)} trades)"
                )

        # Close any open positions at the final bar
        if btc_sim and eth_sim:
            final_ratio = btc_sim[-1]["close"] / eth_sim[-1]["close"]
            final_ts = btc_sim[-1]["time"]

            if self.sig_gen_v415.position != 0 and self.sig_gen_v415.entry_ratio > 0:
                self.sig_gen_v415._close_position(final_ratio, final_ts, "End of simulation")
            if self.sig_gen_v416.position != 0 and self.sig_gen_v416.entry_ratio > 0:
                self.sig_gen_v416._close_position(final_ratio, final_ts, "End of simulation")

        logger.info(
            f"Simulation complete: {self._bars_processed} bars processed, "
            f"{self._bars_with_features} with valid features"
        )

        return self.sig_gen_v415.trades, self.sig_gen_v416.trades


# ══════════════════════════════════════════════════════════════════════════
# CSV Export
# ══════════════════════════════════════════════════════════════════════════

CSV_FIELDNAMES = [
    "direction", "entry_price", "exit_price", "entry_time", "exit_time",
    "pnl_pct", "pnl_dollar", "bars_held", "position_size_pct",
    "stop_loss", "take_profit", "entry_probability", "entry_strength",
    "reason", "model_version",
]


def export_trades_csv(trades: List[Dict], path: Path, model_version: str) -> None:
    """Export trades to CSV in the same schema as the live server."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for t in trades:
            row = dict(t)
            row.setdefault("model_version", model_version)
            writer.writerow(row)

    logger.info(f"  Exported {len(trades)} trades to {path}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V4.16 Offline Live-Trading Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start", default="2026-02-08 22:40:00",
        help="Simulation start (UTC), default: 2026-02-08 22:40:00",
    )
    parser.add_argument(
        "--end", default="2026-02-14 21:48:00",
        help="Simulation end (UTC), default: 2026-02-14 21:48:00",
    )
    parser.add_argument(
        "--balance", type=float, default=1000.0,
        help="Starting balance in dollars (default: 1000)",
    )
    parser.add_argument(
        "--report-dir", type=str, default=None,
        help="Override output directory (default: reports/v4.16/)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Skip cache and re-download candles from Binance",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("V4.16 OFFLINE LIVE-TRADING SIMULATION")
    print("=" * 80)
    print(f"  Start:   {args.start} UTC")
    print(f"  End:     {args.end} UTC")
    print(f"  Balance: ${args.balance:,.2f}")
    print()

    # ── 1. Parse time boundaries ────────────────────────────────────────
    sim_start_ms = _dt_to_ms(args.start)
    sim_end_ms = _dt_to_ms(args.end)

    # Warmup: 1000 bars * 60s/bar = 60000s before sim start
    warmup_start_ms = sim_start_ms - (WARMUP_CANDLES * 60 * 1000)

    logger.info("Step 1: Downloading candle data from Binance...")
    logger.info(f"  Warmup from: {_ms_to_dt(warmup_start_ms).strftime('%Y-%m-%d %H:%M')} UTC")
    logger.info(f"  Sim window:  {_ms_to_dt(sim_start_ms).strftime('%Y-%m-%d %H:%M')} - "
                f"{_ms_to_dt(sim_end_ms).strftime('%Y-%m-%d %H:%M')} UTC")

    # ── 2. Download or load cached data ─────────────────────────────────
    if args.no_cache:
        # Clear cache for this range
        for tag_sym in ["BTCUSDT", "ETHUSDT"]:
            for suffix in ["warmup", "sim"]:
                p = CACHE_DIR / f"{tag_sym}_{warmup_start_ms}_{sim_start_ms}.parquet"
                if p.exists():
                    p.unlink()

    # Download warmup candles
    btc_warmup = download_or_load_cache("BTCUSDT", warmup_start_ms, sim_start_ms)
    eth_warmup = download_or_load_cache("ETHUSDT", warmup_start_ms, sim_start_ms)

    # Download simulation-window candles
    btc_sim = download_or_load_cache("BTCUSDT", sim_start_ms, sim_end_ms)
    eth_sim = download_or_load_cache("ETHUSDT", sim_start_ms, sim_end_ms)

    logger.info(
        f"  Data loaded: warmup BTC={len(btc_warmup)} ETH={len(eth_warmup)} | "
        f"sim BTC={len(btc_sim)} ETH={len(eth_sim)}"
    )

    if not btc_sim or not eth_sim:
        logger.error("No simulation candles downloaded. Aborting.")
        sys.exit(1)

    # ── 3. Load production model ────────────────────────────────────────
    logger.info("Step 2: Loading production XGBClassifier...")
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
    assert FEATURE_NAMES_PATH.exists(), f"Feature names not found: {FEATURE_NAMES_PATH}"

    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    with open(FEATURE_NAMES_PATH) as f:
        feature_names = json.load(f)
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    logger.info(f"  Model: {len(feature_names)} features, {config['best_n_estimators']} trees")

    # ── 4. Run replay ───────────────────────────────────────────────────
    logger.info("Step 3: Running bar-by-bar replay...")
    engine = LiveReplayEngine(
        model=model,
        feature_names=feature_names,
        config=config,
        starting_balance=args.balance,
    )

    trades_v415, trades_v416 = engine.run(
        btc_warmup=btc_warmup,
        eth_warmup=eth_warmup,
        btc_sim=btc_sim,
        eth_sim=eth_sim,
    )

    # ── 5. Export results ───────────────────────────────────────────────
    logger.info("Step 4: Exporting results...")
    report_dir = Path(args.report_dir) if args.report_dir else REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    # Build filename tag from the window
    tag = (args.start.replace(" ", "_").replace(":", "-")
           + "_to_"
           + args.end.replace(" ", "_").replace(":", "-"))

    v415_csv = report_dir / f"results_v4_15_sim_{tag}.csv"
    v416_csv = report_dir / f"results_v4_16_sim_{tag}.csv"

    export_trades_csv(trades_v415, v415_csv, model_version="v4.15")
    export_trades_csv(trades_v416, v416_csv, model_version="v4.16")

    # Also save equity curves for the comparison script
    eq_path = report_dir / f"equity_curves_sim_{tag}.csv"
    eq_df = pd.DataFrame({
        "timestamp": engine.timestamps,
        "equity_v415": engine.equity_v415[1:len(engine.timestamps) + 1],
        "equity_v416": engine.equity_v416[1:len(engine.timestamps) + 1],
    })
    eq_df.to_csv(eq_path, index=False)
    logger.info(f"  Equity curves saved to {eq_path}")

    # ── 6. Print summary ────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 80)

    for label, trades, sig_gen in [
        ("V4.15 (baseline)", trades_v415, engine.sig_gen_v415),
        ("V4.16 (improved)", trades_v416, engine.sig_gen_v416),
    ]:
        n = len(trades)
        if n == 0:
            print(f"\n  {label}: No trades")
            continue

        pnl_pcts = np.array([t["pnl_pct"] for t in trades])
        pnl_dollars = np.array([t["pnl_dollar"] for t in trades])
        is_win = pnl_dollars > 0
        wins = int(is_win.sum())
        losses = n - wins
        win_rate = wins / n * 100

        gross_profit = float(pnl_dollars[is_win].sum()) if wins > 0 else 0
        gross_loss = abs(float(pnl_dollars[~is_win].sum())) if losses > 0 else 1e-8
        profit_factor = gross_profit / gross_loss

        avg_win = float(pnl_pcts[is_win].mean()) if wins > 0 else 0
        avg_loss = float(pnl_pcts[~is_win].mean()) if losses > 0 else 0
        rr = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        # Max drawdown from equity curve
        equity = engine.equity_v415 if "15" in label else engine.equity_v416
        peak = equity[0]
        max_dd = 0.0
        for eq in equity:
            peak = max(peak, eq)
            dd = (eq - peak) / peak
            max_dd = min(max_dd, dd)

        # Sharpe (per-trade, annualized)
        bars_held = np.array([t["bars_held"] for t in trades])
        avg_hold = float(bars_held.mean())
        est_cycle = avg_hold + 15
        trades_per_year = 105120.0 / max(est_cycle, 1)
        mean_ret = float(pnl_pcts.mean())
        std_ret = float(pnl_pcts.std()) if n > 1 else 1e-8
        sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 1e-8 else 0

        print(f"\n  {label}")
        print(f"    Trades:         {n} (W={wins}, L={losses})")
        print(f"    Win Rate:       {win_rate:.1f}%")
        print(f"    Total PnL:      ${sig_gen.balance - args.balance:+.2f}")
        print(f"    Final Balance:  ${sig_gen.balance:.2f}")
        print(f"    Profit Factor:  {profit_factor:.4f}")
        print(f"    R:R Ratio:      {rr:.4f}")
        print(f"    Sharpe (ann.):  {sharpe:.4f}")
        print(f"    Max Drawdown:   {max_dd*100:.4f}%")
        print(f"    Avg Hold (bars):{avg_hold:.1f}")

    print()
    print(f"Output files in {report_dir}/:")
    print(f"  {v415_csv.name}")
    print(f"  {v416_csv.name}")
    print(f"  {eq_path.name}")
    print("=" * 80)


if __name__ == "__main__":
    main()

