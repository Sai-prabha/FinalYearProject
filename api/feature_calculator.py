"""
V4.15 Live Feature Calculator for BTC/ETH Ratio Trading Strategy.

Mirrors the feature engineering from src/features.py build_features() so that
the production XGBClassifier receives identical inputs in live mode as it saw
during training.

Key design:
  - Maintains a rolling buffer of ~1500 BTC+ETH candles (including buy/sell volumes).
  - On each new candle, recomputes all features over the buffer (~1500 rows, <50 ms).
  - Returns a single-row DataFrame with exactly the 50 selected features.
"""

import json
import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants matching src/features.py ──────────────────────────────────────

ZSCORE_CLIP_SIGMA = 5.0
ROLL_WINDOWS_SHORT = (5, 10, 30)
ROLL_WINDOWS_LONG = (60, 120)
ALL_WINDOWS = ROLL_WINDOWS_SHORT + ROLL_WINDOWS_LONG
LAG_RETURNS = (1, 2, 3, 5, 10)

# Minimum candles required before features are fully populated.
# Longest window is 720 (trend_sma_720) + 60 (slope shift) = 780.
# vol_regime_z uses _zscore(vol, 240) which itself needs 240 on top of the vol window.
# Practically 1000 candles is more than enough.
MIN_CANDLES_READY = 800


# ── Helper functions (copied from src/features.py for parity) ──────────────

def _zscore(s: pd.Series, window: int, clip_sigma: float = None) -> pd.Series:
    roll = s.rolling(window, min_periods=window)
    mean = roll.mean()
    std = roll.std()
    result = np.where(std > 1e-8, (s - mean) / std, np.nan)
    if clip_sigma is not None:
        result = np.clip(result, -clip_sigma, clip_sigma)
    return pd.Series(result, index=s.index)


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window, min_periods=window).mean()
    roll_down = down.rolling(window, min_periods=window).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


# ── Main Feature Calculator ────────────────────────────────────────────────

class V414FeatureCalculator:
    """
    Stateful live feature calculator for V4.15 XGBClassifier.

    Feed it candles (BTC + ETH) via ``add_candle()``.  When enough history
    is available, ``calculate_features()`` returns a single-row DataFrame
    with the 50 selected features in the correct column order.
    """

    def __init__(
        self,
        feature_names: List[str],
        max_history: int = 1500,
    ):
        self.feature_names = feature_names
        self.max_history = max_history

        # Candle history stored as list-of-dicts, converted to DF on demand
        self.btc_candles: deque = deque(maxlen=max_history)
        self.eth_candles: deque = deque(maxlen=max_history)

        # Track latest timestamps
        self.last_btc_time: int = 0
        self.last_eth_time: int = 0

    # ------------------------------------------------------------------ IO

    def add_candle(self, symbol: str, candle: Dict) -> None:
        """
        Add (or update) a candle.

        ``candle`` keys: time, open, high, low, close, volume,
        and optionally taker_buy_volume (kline index 9).
        """
        row = {
            "time": int(candle["time"]),
            "open": float(candle["open"]),
            "high": float(candle["high"]),
            "low": float(candle["low"]),
            "close": float(candle["close"]),
            "volume": float(candle["volume"]),
            "taker_buy_volume": float(candle.get("taker_buy_volume", candle["volume"] * 0.5)),
        }

        buf = self.btc_candles if symbol == "BTC" else self.eth_candles

        # Update in-place if same timestamp (partial candle update from WS)
        if buf and buf[-1]["time"] == row["time"]:
            buf[-1] = row
        else:
            buf.append(row)

        if symbol == "BTC":
            self.last_btc_time = row["time"]
        else:
            self.last_eth_time = row["time"]

    def seed_historical(self, symbol: str, candles: List[Dict]) -> None:
        """Bulk-load historical candles (oldest first)."""
        for c in candles:
            self.add_candle(symbol, c)
        logger.info(f"Seeded {len(candles)} {symbol} candles")

    # -------------------------------------------------------------- Status

    def get_data_quality_status(self) -> Dict:
        n_btc = len(self.btc_candles)
        n_eth = len(self.eth_candles)
        return {
            "btc_candles": n_btc,
            "eth_candles": n_eth,
            "ready": n_btc >= MIN_CANDLES_READY and n_eth >= MIN_CANDLES_READY,
            "min_required": MIN_CANDLES_READY,
            "last_btc_time": self.last_btc_time,
            "last_eth_time": self.last_eth_time,
            "synced": abs(self.last_btc_time - self.last_eth_time) <= 60,
        }

    # ------------------------------------------------------------ Features

    def calculate_features(self) -> Optional[pd.DataFrame]:
        """
        Compute all features and return a single-row DataFrame with the
        50 selected feature columns (same order as training).

        Returns None if not enough data.
        """
        status = self.get_data_quality_status()
        if not status["ready"]:
            return None

        # 1. Build a merged DataFrame that mirrors the training parquet -----
        btc_df = pd.DataFrame(list(self.btc_candles))
        eth_df = pd.DataFrame(list(self.eth_candles))

        merged = pd.merge(
            btc_df, eth_df,
            on="time", suffixes=("_btc", "_eth"),
            how="inner",
        )

        if len(merged) < MIN_CANDLES_READY:
            return None

        merged = merged.sort_values("time").reset_index(drop=True)

        # Clean the merged data before feature computation
        merged = self._clean_merged_data(merged)

        # Reconstruct columns matching training data layout
        df = pd.DataFrame({
            "open_time": pd.to_datetime(merged["time"], unit="s", utc=True).dt.tz_localize(None),
            "btc_close": merged["close_btc"],
            "eth_close": merged["close_eth"],
            "R": merged["close_btc"] / merged["close_eth"],
            # Buy / sell decomposition from taker volumes
            "btc_buy_ratio": merged["taker_buy_volume_btc"] / (merged["volume_btc"] + 1e-12),
            "btc_net_buy_pressure": (2 * merged["taker_buy_volume_btc"] / (merged["volume_btc"] + 1e-12)) - 1,
            "eth_buy_ratio": merged["taker_buy_volume_eth"] / (merged["volume_eth"] + 1e-12),
            "eth_net_buy_pressure": (2 * merged["taker_buy_volume_eth"] / (merged["volume_eth"] + 1e-12)) - 1,
        })
        df["buy_pressure_divergence"] = df["btc_buy_ratio"] - df["eth_buy_ratio"]

        # 2. Compute ALL feature families (same as build_features) ----------
        ratio = df["R"].astype(float)
        rret = np.log(ratio / ratio.shift(1))

        feats: Dict[str, pd.Series] = {}

        # Lag returns
        for lag in LAG_RETURNS:
            feats[f"r_lag_{lag}"] = rret.shift(lag)

        # Rolling features
        feats.update(self._rolling_features(rret))

        # Trend features
        feats.update(self._trend_features(ratio))

        # Z-scores of returns
        feats["rret_z_30"] = _zscore(rret.fillna(0), 30, clip_sigma=ZSCORE_CLIP_SIGMA)
        feats["rret_z_60"] = _zscore(rret.fillna(0), 60, clip_sigma=ZSCORE_CLIP_SIGMA)

        # RSI
        feats["rsi_14"] = _rsi(rret.fillna(0), 14)

        # Cross-asset features
        feats.update(self._cross_asset_features(df))

        # Volume pressure features
        feats.update(self._volume_pressure_features(df))

        # Time features
        feats.update(self._time_features(df))

        # Regime features
        feats.update(self._regime_features(df, ratio))

        # 3. Assemble into a DataFrame and extract last row -----------------
        feature_frame = pd.DataFrame(feats, index=df.index).astype("float32")

        # Only keep columns the model needs
        available = [c for c in self.feature_names if c in feature_frame.columns]
        missing = [c for c in self.feature_names if c not in feature_frame.columns]
        if missing:
            logger.warning(f"Missing features (will be NaN): {missing}")

        row = feature_frame[available].iloc[[-1]].copy()

        # Fill any missing columns with NaN
        for col in missing:
            row[col] = np.nan

        # Reorder to match training column order exactly
        row = row[self.feature_names]

        # Reject if any NaN
        if row.isna().any(axis=1).iloc[0]:
            nan_cols = row.columns[row.isna().iloc[0]].tolist()
            logger.warning(f"NaN features ({len(nan_cols)}): {nan_cols[:5]}...")
            return None

        return row

    # ── Data cleaning ────────────────────────────────────────────────────

    @staticmethod
    def _clean_merged_data(merged: pd.DataFrame) -> pd.DataFrame:
        """
        Clean merged BTC+ETH data before feature computation.

        - Forward-fill gaps (missing 1m candles)
        - Clip extreme single-bar price spikes (>3 std from 30-bar mean)
        - Replace zero/negative volumes with rolling median
        """
        n_before = len(merged)

        # ── 1. Gap detection & forward-fill ──
        # Identify time jumps > 60s and insert forward-filled rows
        times = merged["time"].values
        expected_gap = 60  # 1 minute in seconds
        gap_mask = np.diff(times) > expected_gap * 2  # gap larger than 2 minutes
        n_gaps = int(gap_mask.sum())
        if n_gaps > 0:
            logger.info(f"Data cleaning: {n_gaps} time gaps detected, forward-filling")
            # Build a complete time index and reindex
            full_times = np.arange(times[0], times[-1] + expected_gap, expected_gap)
            merged = merged.set_index("time").reindex(full_times).ffill().reset_index()
            merged = merged.rename(columns={"index": "time"})

        # ── 2. Price spike filtering ──
        # Clip extreme moves in close prices (>3 std from 30-bar rolling mean)
        for col in ["close_btc", "close_eth"]:
            if col not in merged.columns:
                continue
            series = merged[col].astype(float)
            pct_change = series.pct_change().abs()
            rolling_std = pct_change.rolling(30, min_periods=10).std()
            rolling_mean = pct_change.rolling(30, min_periods=10).mean()
            spike_threshold = rolling_mean + 3.0 * rolling_std
            spikes = pct_change > spike_threshold
            # Don't flag first 30 rows or NaN thresholds
            spikes = spikes & spike_threshold.notna() & (pct_change.index >= 30)
            n_spikes = int(spikes.sum())
            if n_spikes > 0:
                logger.warning(
                    f"Data cleaning: {n_spikes} price spikes in {col}, "
                    f"clipping to previous value"
                )
                merged.loc[spikes, col] = np.nan
                merged[col] = merged[col].ffill()

        # ── 3. Volume validation ──
        # Replace zero or negative volumes with rolling median
        for col in ["volume_btc", "volume_eth"]:
            if col not in merged.columns:
                continue
            vol = merged[col].astype(float)
            bad_vol = vol <= 0
            n_bad = int(bad_vol.sum())
            if n_bad > 0:
                rolling_med = vol.rolling(30, min_periods=5).median()
                merged.loc[bad_vol, col] = rolling_med[bad_vol]
                # If still NaN (first few rows), use global median
                still_bad = merged[col].isna() | (merged[col] <= 0)
                if still_bad.any():
                    merged.loc[still_bad, col] = vol[vol > 0].median()
                logger.warning(
                    f"Data cleaning: {n_bad} zero/negative volumes in {col}, replaced"
                )

        n_after = len(merged)
        if n_after != n_before:
            logger.info(
                f"Data cleaning: rows {n_before} -> {n_after} "
                f"(+{n_after - n_before} gap-filled)"
            )

        return merged

    # ── Feature computation helpers (mirrors src/features.py) ─────────────

    @staticmethod
    def _rolling_features(rret: pd.Series) -> Dict[str, pd.Series]:
        feats: Dict[str, pd.Series] = {}
        std_map: Dict[int, pd.Series] = {}

        for w in ALL_WINDOWS:
            roll = rret.rolling(w, min_periods=w)
            feats[f"r_roll_mean_{w}"] = roll.mean()
            std_series = roll.std()
            feats[f"r_roll_std_{w}"] = std_series
            std_map[w] = std_series

        if 60 in std_map and 10 in std_map:
            feats["std_ratio_60_10"] = pd.Series(
                np.where(std_map[10] > 1e-8, std_map[60] / std_map[10], np.nan),
                index=rret.index,
            )
        if 120 in std_map and 30 in std_map:
            feats["std_ratio_120_30"] = pd.Series(
                np.where(std_map[30] > 1e-8, std_map[120] / std_map[30], np.nan),
                index=rret.index,
            )
        return feats

    @staticmethod
    def _trend_features(ratio: pd.Series) -> Dict[str, pd.Series]:
        feats: Dict[str, pd.Series] = {}
        for window in [240, 480, 720]:
            sma = ratio.rolling(window, min_periods=window).mean()
            feats[f"trend_sma_{window}"] = ratio / sma - 1
            trend_slope = (sma - sma.shift(60)) / sma.shift(60)
            feats[f"trend_slope_{window}"] = trend_slope

        for window in [120, 240]:
            high = ratio.rolling(window).max()
            low = ratio.rolling(window).min()
            range_pct = (high - low) / low
            position_in_range = (ratio - low) / (high - low + 1e-8)
            feats[f"trend_strength_{window}"] = range_pct * np.abs(position_in_range - 0.5) * 2
        return feats

    @staticmethod
    def _cross_asset_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
        feats: Dict[str, pd.Series] = {}
        btc = df["btc_close"].astype(float)
        eth = df["eth_close"].astype(float)
        btc_r = np.log(btc / btc.shift(1))
        eth_r = np.log(eth / eth.shift(1))

        feats["corr_30"] = btc_r.rolling(30, min_periods=30).corr(eth_r)
        feats["corr_60"] = btc_r.rolling(60, min_periods=60).corr(eth_r)
        feats["corr_delta_30"] = feats["corr_30"].diff()
        feats["corr_delta_60"] = feats["corr_60"].diff()
        feats["vol_spread_30"] = btc_r.rolling(30, min_periods=30).std() - eth_r.rolling(30, min_periods=30).std()
        feats["vol_spread_60"] = btc_r.rolling(60, min_periods=60).std() - eth_r.rolling(60, min_periods=60).std()

        feats["vol_corr_interact_30"] = feats["corr_30"] * feats["vol_spread_30"]
        feats["vol_corr_interact_60"] = feats["corr_60"] * feats["vol_spread_60"]
        return feats

    @staticmethod
    def _volume_pressure_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
        feats: Dict[str, pd.Series] = {}

        if "btc_buy_ratio" in df.columns:
            br = df["btc_buy_ratio"].astype(float)
            feats["btc_buy_ratio_10m"] = br.rolling(10, min_periods=5).mean()
            feats["btc_buy_momentum_10m"] = br.diff(10)
            feats["btc_buy_ratio_z_30"] = _zscore(br, 30, clip_sigma=ZSCORE_CLIP_SIGMA)
            feats["btc_buy_ratio_z_60"] = _zscore(br, 60, clip_sigma=ZSCORE_CLIP_SIGMA)

        if "btc_net_buy_pressure" in df.columns:
            bp = df["btc_net_buy_pressure"].astype(float)
            feats["btc_net_pressure_10m"] = bp.rolling(10, min_periods=5).mean()
            feats["btc_net_pressure_30m"] = bp.rolling(30, min_periods=15).mean()
            feats["btc_net_pressure_momentum"] = bp.diff(5)

        if "eth_buy_ratio" in df.columns:
            er = df["eth_buy_ratio"].astype(float)
            feats["eth_buy_ratio_10m"] = er.rolling(10, min_periods=5).mean()
            feats["eth_buy_momentum_10m"] = er.diff(10)
            feats["eth_buy_ratio_z_30"] = _zscore(er, 30, clip_sigma=ZSCORE_CLIP_SIGMA)
            feats["eth_buy_ratio_z_60"] = _zscore(er, 60, clip_sigma=ZSCORE_CLIP_SIGMA)

        if "eth_net_buy_pressure" in df.columns:
            ep = df["eth_net_buy_pressure"].astype(float)
            feats["eth_net_pressure_10m"] = ep.rolling(10, min_periods=5).mean()
            feats["eth_net_pressure_30m"] = ep.rolling(30, min_periods=15).mean()
            feats["eth_net_pressure_momentum"] = ep.diff(5)

        if "buy_pressure_divergence" in df.columns:
            div = df["buy_pressure_divergence"].astype(float)
            feats["buy_pressure_div_10m"] = div.rolling(10, min_periods=5).mean()
            feats["buy_pressure_div_30m"] = div.rolling(30, min_periods=15).mean()
            feats["buy_pressure_div_momentum"] = div.diff(5)

        if "btc_buy_ratio" in df.columns and "eth_buy_ratio" in df.columns:
            feats["buy_ratio_spread"] = df["btc_buy_ratio"].astype(float) - df["eth_buy_ratio"].astype(float)
            feats["buy_ratio_spread_10m"] = feats["buy_ratio_spread"].rolling(10, min_periods=5).mean()

        return feats

    @staticmethod
    def _time_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
        feats: Dict[str, pd.Series] = {}
        if "open_time" not in df.columns:
            return feats

        ts = pd.to_datetime(df["open_time"])
        hour = ts.dt.hour
        feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        dow = ts.dt.dayofweek
        feats["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        feats["dow_cos"] = np.cos(2 * np.pi * dow / 7)

        feats["is_weekend"] = (dow >= 5).astype(float)
        feats["us_trading_hours"] = ((hour >= 13) & (hour <= 21)).astype(float)
        feats["eu_trading_hours"] = ((hour >= 7) & (hour <= 15)).astype(float)
        feats["asia_trading_hours"] = ((hour >= 0) & (hour <= 8)).astype(float)
        return feats

    @staticmethod
    def _regime_features(df: pd.DataFrame, ratio: pd.Series) -> Dict[str, pd.Series]:
        feats: Dict[str, pd.Series] = {}
        rret = np.log(ratio / ratio.shift(1))

        for window in [30, 60, 120]:
            vol = rret.rolling(window, min_periods=window // 2).std()
            feats[f"vol_regime_{window}"] = vol
            feats[f"vol_regime_z_{window}"] = _zscore(vol, 240, clip_sigma=ZSCORE_CLIP_SIGMA)

        for window in [30, 60]:
            high = ratio.rolling(window).max()
            low = ratio.rolling(window).min()
            range_pct = (high - low) / (ratio + 1e-8)
            feats[f"trend_strength_{window}"] = range_pct

        for window in [60, 120, 240]:
            sma = ratio.rolling(window, min_periods=window // 2).mean()
            feats[f"distance_from_sma_{window}"] = (ratio - sma) / (sma + 1e-8)

        recent_vol = rret.rolling(10).std()
        past_vol = rret.rolling(30).std()
        feats["vol_clustering"] = recent_vol / (past_vol + 1e-8)
        return feats


# ── V4.15 Signal Generator (hysteresis + circuit breaker) ──────────────────

class V414SignalGenerator:
    """
    Mirrors the position logic from ``scripts/run_v4_14_classifier.py`` (upgraded to v4.15).

    Probability-based hysteresis with:
      - entry_threshold / exit_threshold
      - Dynamic min_hold with SL/TP early exit
      - cooldown
      - Rolling-PnL circuit breaker
      - Dynamic position sizing based on signal confidence
    """

    # Dynamic leverage sizing: target $ gain when TP is hit
    TARGET_WIN_FRAC = 0.005   # 0.5% of balance per TP hit
    MIN_LEVERAGE = 0.50       # Floor: 50% of balance
    MAX_LEVERAGE = 3.0        # Cap: 3x leverage

    # Dynamic min-hold: absolute floor even under SL/TP override
    ABSOLUTE_MIN_HOLD = 5

    # Default SL/TP (overridden dynamically by volatility)
    DEFAULT_STOP_LOSS_PCT = -0.20   # -0.20%
    DEFAULT_TAKE_PROFIT_PCT = 0.30  # +0.30%

    # Trailing take-profit thresholds
    TRAILING_BREAKEVEN_FRAC = 0.50   # Move SL to breakeven at 50% of TP
    TRAILING_LOCK_FRAC = 0.75        # Trail SL at 50% of unrealized at 75% of TP
    TRAILING_LOCK_RATIO = 0.50       # Lock this fraction of unrealized profit

    def __init__(
        self,
        entry_threshold: float = 0.525,
        exit_threshold: float = 0.51,
        min_hold: int = 25,
        cooldown: int = 15,
        cb_lookback: int = 500,
        cb_threshold: float = -0.03,
        starting_balance: float = 1000.0,
    ):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.min_hold = min_hold
        self.cooldown = cooldown
        self.cb_lookback = cb_lookback
        self.cb_threshold = cb_threshold

        # Position state
        self.position: int = 0          # -1, 0, +1
        self.bars_since_change: int = 10_000  # start high so first trade allowed
        self.bar_count: int = 0

        # Circuit breaker state
        self.pnl_history: deque = deque(maxlen=cb_lookback)
        self.cb_active: bool = False

        # Portfolio tracking
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.entry_ratio: float = 0.0
        self.entry_time: int = 0
        self.entry_proba: float = 0.5
        self.entry_strength: float = 0.0
        self.current_position_size: float = 0.0  # fraction of balance
        self.current_stop_loss: float = 0.0       # ratio price level
        self.current_take_profit: float = 0.0     # ratio price level
        self.total_pnl: float = 0.0
        self.wins: int = 0
        self.losses: int = 0
        self.trades: list = []
        self.current_position_label: Optional[str] = None  # 'LONG'/'SHORT'/None

        # Recent volatility tracking for dynamic SL/TP
        self._recent_returns: deque = deque(maxlen=30)

    # ── Dynamic helpers ──────────────────────────────────────────────────

    def _signal_strength(self, proba_up: float) -> float:
        """Map probability to 0-1 strength (distance from 0.5)."""
        return float(abs(proba_up - 0.5) * 2.0)

    def _position_size_frac(self, strength: float, tp_pct: float = 0.003) -> float:
        """
        Dynamic leverage: calculate position size so that hitting TP yields
        a meaningful dollar gain (~TARGET_WIN_FRAC of balance).

        Args:
            strength: signal strength 0-1
            tp_pct: take-profit as a fraction (e.g. 0.003 for 0.30%)
        """
        if tp_pct <= 1e-8:
            return self.MIN_LEVERAGE
        required = self.TARGET_WIN_FRAC / tp_pct  # e.g. 0.005 / 0.003 = 1.67x
        # Scale by signal strength (stronger -> closer to required, weaker -> reduced)
        scaled = self.MIN_LEVERAGE + (required - self.MIN_LEVERAGE) * min(strength, 1.0)
        return max(self.MIN_LEVERAGE, min(scaled, self.MAX_LEVERAGE))

    def _recent_volatility(self) -> float:
        """Annualized vol from recent 30-bar returns (used for SL/TP scaling)."""
        if len(self._recent_returns) < 10:
            return 0.001  # default small vol
        arr = np.array(self._recent_returns)
        return float(np.std(arr)) if np.std(arr) > 1e-8 else 0.001

    def _compute_sl_tp(self, direction: int, entry_ratio: float, strength: float) -> Tuple[float, float, float]:
        """
        Compute dynamic stop-loss and take-profit price levels.

        SL/TP widths scale with recent volatility:
          - Higher vol -> wider SL/TP to avoid noise
          - Higher confidence -> tighter TP (capture quickly), wider SL (hold conviction)

        Returns (stop_loss_price, take_profit_price, tp_pct_fraction).
        """
        vol = self._recent_volatility()
        vol_mult = max(vol * 100, 0.05)  # vol as pct, floor 0.05%

        # Base SL/TP in ratio pct
        sl_pct = max(abs(self.DEFAULT_STOP_LOSS_PCT), vol_mult * 1.5) / 100.0
        tp_pct = max(abs(self.DEFAULT_TAKE_PROFIT_PCT), vol_mult * 2.0) / 100.0

        # Confidence adjustment: stronger signal -> slightly wider SL, tighter TP
        sl_pct *= (1.0 + strength * 0.3)   # hold conviction longer
        tp_pct *= (1.0 - strength * 0.15)  # take profit sooner on strong signals

        if direction == 1:  # LONG
            stop_loss = entry_ratio * (1.0 - sl_pct)
            take_profit = entry_ratio * (1.0 + tp_pct)
        else:  # SHORT
            stop_loss = entry_ratio * (1.0 + sl_pct)
            take_profit = entry_ratio * (1.0 - tp_pct)

        return stop_loss, take_profit, tp_pct

    def _update_trailing_sl(self, current_ratio: float) -> None:
        """
        Update stop-loss using trailing logic when trade is in profit.

        - At 50% of TP target reached: move SL to breakeven (entry price)
        - At 75% of TP target reached: trail SL at 50% of unrealized profit
        """
        if self.position == 0 or self.entry_ratio <= 0:
            return
        if self.current_take_profit <= 0:
            return

        # Compute unrealized movement toward TP
        if self.position == 1:  # LONG
            tp_distance = self.current_take_profit - self.entry_ratio
            current_profit = current_ratio - self.entry_ratio
        else:  # SHORT
            tp_distance = self.entry_ratio - self.current_take_profit
            current_profit = self.entry_ratio - current_ratio

        if tp_distance <= 0:
            return

        progress = current_profit / tp_distance  # 0 to 1+ toward TP

        if progress >= self.TRAILING_LOCK_FRAC:
            # Trail SL at TRAILING_LOCK_RATIO of unrealized profit
            locked_profit = current_profit * self.TRAILING_LOCK_RATIO
            if self.position == 1:
                new_sl = self.entry_ratio + locked_profit
                self.current_stop_loss = max(self.current_stop_loss, new_sl)
            else:
                new_sl = self.entry_ratio - locked_profit
                self.current_stop_loss = min(self.current_stop_loss, new_sl)
        elif progress >= self.TRAILING_BREAKEVEN_FRAC:
            # Move SL to breakeven
            if self.position == 1:
                self.current_stop_loss = max(self.current_stop_loss, self.entry_ratio)
            else:
                self.current_stop_loss = min(self.current_stop_loss, self.entry_ratio)

    def _effective_min_hold(self, current_ratio: float, proba_up: float) -> int:
        """
        Return dynamic min-hold that can be reduced when SL/TP conditions met.

        Early exit allowed (min hold reduced to ABSOLUTE_MIN_HOLD) when:
          1. Unrealized PnL hits take-profit threshold
          2. Unrealized PnL hits stop-loss threshold
          3. Signal flips strongly to the opposite direction
        """
        if self.position == 0 or self.entry_ratio <= 0:
            return self.min_hold

        # Already past min hold — no restriction
        if self.bars_since_change >= self.min_hold:
            return self.min_hold

        # Must respect absolute minimum
        if self.bars_since_change < self.ABSOLUTE_MIN_HOLD:
            return self.min_hold

        # Check SL/TP price levels
        if self.position == 1:  # LONG
            hit_tp = current_ratio >= self.current_take_profit > 0
            hit_sl = current_ratio <= self.current_stop_loss > 0
        else:  # SHORT
            hit_tp = current_ratio <= self.current_take_profit > 0 if self.current_take_profit > 0 else False
            hit_sl = current_ratio >= self.current_stop_loss > 0 if self.current_stop_loss > 0 else False

        if hit_tp or hit_sl:
            return self.ABSOLUTE_MIN_HOLD

        # Strong opposite signal override
        if self.position == 1 and proba_up <= (1.0 - self.entry_threshold - 0.01):
            return self.ABSOLUTE_MIN_HOLD
        if self.position == -1 and proba_up >= (self.entry_threshold + 0.01):
            return self.ABSOLUTE_MIN_HOLD

        return self.min_hold

    # ── Main update ─────────────────────────────────────────────────────

    def update(self, proba_up: float, current_ratio: float, timestamp: int) -> Dict:
        """
        Process one bar: decide position, track PnL, check circuit breaker.

        Returns a signal dict suitable for the frontend.
        """
        # Track returns for volatility estimation
        if self.position != 0 and self.entry_ratio > 0:
            self._recent_returns.append(np.log(current_ratio / self.entry_ratio) / max(self.bars_since_change, 1))
        elif len(self._recent_returns) == 0:
            self._recent_returns.append(0.0)

        prev = self.position
        desired = prev

        # Dynamic min-hold (may be reduced by SL/TP or strong signal flip)
        effective_mh = self._effective_min_hold(current_ratio, proba_up)
        can_change = self.bars_since_change >= effective_mh

        # Update trailing stop-loss before checking SL/TP
        self._update_trailing_sl(current_ratio)

        # Determine SL/TP exit reason if applicable
        early_exit_reason = None
        if self.position != 0 and self.entry_ratio > 0 and can_change and self.bars_since_change < self.min_hold:
            if self.position == 1:
                if self.current_take_profit > 0 and current_ratio >= self.current_take_profit:
                    early_exit_reason = "Take profit"
                elif self.current_stop_loss > 0 and current_ratio <= self.current_stop_loss:
                    early_exit_reason = "Stop loss"
            else:
                if self.current_take_profit > 0 and current_ratio <= self.current_take_profit:
                    early_exit_reason = "Take profit"
                elif self.current_stop_loss > 0 and current_ratio >= self.current_stop_loss:
                    early_exit_reason = "Stop loss"

        # Also check SL/TP for trades past min_hold
        if self.position != 0 and self.entry_ratio > 0 and can_change and not early_exit_reason:
            if self.position == 1:
                if self.current_take_profit > 0 and current_ratio >= self.current_take_profit:
                    early_exit_reason = "Take profit"
                elif self.current_stop_loss > 0 and current_ratio <= self.current_stop_loss:
                    early_exit_reason = "Stop loss"
            else:
                if self.current_take_profit > 0 and current_ratio <= self.current_take_profit:
                    early_exit_reason = "Take profit"
                elif self.current_stop_loss > 0 and current_ratio >= self.current_stop_loss:
                    early_exit_reason = "Stop loss"

        # ── Compute unrealized PnL for signal-exit guard ──
        _unrealized_pnl_pct = 0.0
        if self.position != 0 and self.entry_ratio > 0:
            if self.position == 1:
                _unrealized_pnl_pct = (current_ratio - self.entry_ratio) / self.entry_ratio
            else:
                _unrealized_pnl_pct = (self.entry_ratio - current_ratio) / self.entry_ratio

        # ── Hysteresis logic ──
        if prev == 0:
            if proba_up >= self.entry_threshold:
                desired = 1
            elif proba_up <= (1.0 - self.entry_threshold):
                desired = -1
        elif prev == 1:
            if can_change:
                if early_exit_reason:
                    desired = 0  # SL/TP exit
                elif proba_up <= (1.0 - self.entry_threshold):
                    desired = -1
                elif proba_up < self.exit_threshold:
                    desired = 0
        elif prev == -1:
            if can_change:
                if early_exit_reason:
                    desired = 0  # SL/TP exit
                elif proba_up >= self.entry_threshold:
                    desired = 1
                elif proba_up > (1.0 - self.exit_threshold):
                    desired = 0

        # ── Signal-exit guard: prevent exiting at a loss on weak signals ──
        # If exit is due to signal change (not SL/TP) and we're underwater,
        # only allow if the opposite signal is strong (beyond entry threshold)
        if desired != prev and prev != 0 and not early_exit_reason:
            is_strong_flip = False
            if prev == 1 and proba_up <= (1.0 - self.entry_threshold):
                is_strong_flip = True  # Strong SHORT signal
            elif prev == -1 and proba_up >= self.entry_threshold:
                is_strong_flip = True  # Strong LONG signal

            if _unrealized_pnl_pct < 0 and not is_strong_flip:
                # Underwater and weak signal — hold position to avoid locking in loss
                desired = prev

        # Cooldown filter (only applies to normal signal changes, not SL/TP)
        if desired != prev and self.bars_since_change <= self.cooldown and not early_exit_reason:
            desired = prev

        # ── Circuit breaker ──
        if len(self.pnl_history) >= self.cb_lookback // 2:
            rolling_pnl = sum(self.pnl_history)
            self.cb_active = rolling_pnl < self.cb_threshold
        else:
            self.cb_active = False

        if self.cb_active:
            desired = 0  # Force neutral

        # ── Apply position change ──
        close_reason = early_exit_reason or "Signal change"
        if desired != prev:
            # Close existing position
            if prev != 0 and self.entry_ratio > 0:
                self._close_position(current_ratio, timestamp, close_reason)
            # Open new position with dynamic sizing
            if desired != 0:
                strength = self._signal_strength(proba_up)
                self._open_position(desired, current_ratio, timestamp, proba_up, strength)
            self.bars_since_change = 0
        else:
            self.bars_since_change += 1

        self.position = desired
        self.bar_count += 1

        # Track bar PnL for circuit breaker
        bar_pnl = 0.0
        if self.position != 0 and self.entry_ratio > 0:
            ratio_return = np.log(current_ratio / self.entry_ratio)
            bar_pnl = self.position * ratio_return
        self.pnl_history.append(bar_pnl)

        # Build reasoning
        reasoning = self._build_reasoning(proba_up, can_change, early_exit_reason)

        # Direction label
        direction = "LONG" if desired == 1 else "SHORT" if desired == -1 else "NEUTRAL"
        triggered = desired != prev and desired != 0

        # Unrealised P&L (using actual position size)
        unrealized_pnl = 0.0
        unrealized_pnl_pct = 0.0
        if self.position != 0 and self.entry_ratio > 0:
            if self.position == 1:
                unrealized_pnl_pct = (current_ratio - self.entry_ratio) / self.entry_ratio
            else:
                unrealized_pnl_pct = (self.entry_ratio - current_ratio) / self.entry_ratio
            unrealized_pnl = self.balance * self.current_position_size * unrealized_pnl_pct

        win_rate = self.wins / (self.wins + self.losses) * 100 if (self.wins + self.losses) > 0 else 0

        return {
            "direction": direction,
            "strength": self._signal_strength(proba_up),
            "probability": float(proba_up),
            "triggered": triggered,
            "blocked_by": self._get_blocked_by(proba_up, can_change),
            "reasoning": reasoning,
            "circuit_breaker_active": self.cb_active,
            "position_meta": {
                "stop_loss": self.current_stop_loss if self.position != 0 else 0.0,
                "take_profit": self.current_take_profit if self.position != 0 else 0.0,
                "position_size_pct": self.current_position_size * 100 if self.position != 0 else 0.0,
                "effective_min_hold": effective_mh,
                "bars_held": self.bars_since_change,
            },
            "portfolio": {
                "balance": self.balance,
                "starting_balance": self.starting_balance,
                "total_pnl": self.total_pnl,
                "total_pnl_pct": (self.balance - self.starting_balance) / self.starting_balance * 100,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct * 100,
                "position": self.current_position_label,
                "entry_price": self.entry_ratio,
                "total_trades": len(self.trades),
                "wins": self.wins,
                "losses": self.losses,
                "win_rate": win_rate,
                "recent_trades": self.trades[-20:],
            },
        }

    def _open_position(self, direction: int, ratio: float, timestamp: int,
                       proba_up: float = 0.5, strength: float = 0.0):
        self.entry_ratio = ratio
        self.entry_time = timestamp
        self.entry_proba = proba_up
        self.entry_strength = strength
        self.current_position_label = "LONG" if direction == 1 else "SHORT"

        # Dynamic SL/TP levels (also returns tp_pct for leverage calculation)
        self.current_stop_loss, self.current_take_profit, tp_pct = self._compute_sl_tp(
            direction, ratio, strength
        )

        # Dynamic leverage sizing: scale position so TP hit = meaningful $ gain
        self.current_position_size = self._position_size_frac(strength, tp_pct)

        logger.info(
            f"OPEN {self.current_position_label} @ {ratio:.4f} | "
            f"leverage={self.current_position_size:.2f}x | "
            f"SL={self.current_stop_loss:.4f} TP={self.current_take_profit:.4f}"
        )

    def _close_position(self, exit_ratio: float, timestamp: int, reason: str = "Signal change"):
        if self.entry_ratio <= 0:
            return
        prev_pos = self.position
        if prev_pos == 1:
            pnl_pct = (exit_ratio - self.entry_ratio) / self.entry_ratio
        else:
            pnl_pct = (self.entry_ratio - exit_ratio) / self.entry_ratio

        trade_size = self.balance * self.current_position_size
        pnl_dollar = trade_size * pnl_pct
        self.balance += pnl_dollar
        self.total_pnl += pnl_dollar

        if pnl_dollar > 0:
            self.wins += 1
        else:
            self.losses += 1

        bars_held = self.bars_since_change

        self.trades.append({
            "direction": "LONG" if prev_pos == 1 else "SHORT",
            "entry_price": self.entry_ratio,
            "exit_price": exit_ratio,
            "entry_time": self.entry_time,
            "exit_time": timestamp,
            "pnl_pct": pnl_pct * 100,
            "pnl_dollar": pnl_dollar,
            "bars_held": bars_held,
            "position_size_pct": self.current_position_size * 100,
            "stop_loss": self.current_stop_loss,
            "take_profit": self.current_take_profit,
            "entry_probability": self.entry_proba,
            "entry_strength": self.entry_strength,
            "reason": reason,
        })

        logger.info(
            f"CLOSE {self.current_position_label} @ {exit_ratio:.4f} | "
            f"PnL={pnl_pct*100:.3f}% (${pnl_dollar:.2f}) | "
            f"Bars={bars_held} | Reason={reason}"
        )

        self.entry_ratio = 0.0
        self.entry_time = 0
        self.entry_proba = 0.5
        self.entry_strength = 0.0
        self.current_position_size = 0.0
        self.current_stop_loss = 0.0
        self.current_take_profit = 0.0
        self.current_position_label = None

    def _build_reasoning(self, proba_up: float, can_change: bool,
                         early_exit_reason: Optional[str] = None) -> list:
        lines = []
        lines.append(f"P(up) = {proba_up:.4f}")
        if proba_up >= self.entry_threshold:
            lines.append(f"Above entry threshold ({self.entry_threshold}) -> LONG signal")
        elif proba_up <= (1 - self.entry_threshold):
            lines.append(f"Below entry threshold ({1 - self.entry_threshold:.3f}) -> SHORT signal")
        else:
            lines.append(f"Within neutral band ({1 - self.entry_threshold:.3f} - {self.entry_threshold})")

        if self.position != 0:
            pos_label = "LONG" if self.position == 1 else "SHORT"
            lines.append(f"Current position: {pos_label}")
            lines.append(f"Bars held: {self.bars_since_change}")
            if self.current_position_size > 0:
                lines.append(f"Size: {self.current_position_size*100:.1f}%")
            if not can_change:
                eff_mh = self._effective_min_hold(0, proba_up)  # approximate
                lines.append(f"Min hold not met ({self.bars_since_change}/{eff_mh})")

        if early_exit_reason:
            lines.append(f"Early exit: {early_exit_reason}")

        if self.cb_active:
            lines.append("CIRCUIT BREAKER ACTIVE - forced NEUTRAL")

        if self.bars_since_change <= self.cooldown and self.position != 0:
            lines.append(f"Cooldown active ({self.bars_since_change}/{self.cooldown})")

        return lines

    def _get_blocked_by(self, proba_up: float, can_change: bool) -> list:
        blocked = []
        if self.cb_active:
            blocked.append("Circuit breaker")
        if not can_change and self.position != 0:
            eff_mh = self._effective_min_hold(0, proba_up)
            blocked.append(f"Min hold ({self.bars_since_change}/{eff_mh})")
        if self.bars_since_change <= self.cooldown:
            blocked.append(f"Cooldown ({self.bars_since_change}/{self.cooldown})")
        return blocked

    # ── State restoration ─────────────────────────────────────────────────

    def restore_from_trades(self, trades: list) -> None:
        """
        Restore portfolio state from persisted trade history.

        Called on server restart to reconstruct balance, wins/losses, and
        trade list from the JSON trade log on disk, so the frontend always
        sees the correct cumulative balance regardless of how many times
        the server has been restarted.
        """
        self.trades = []
        self.balance = self.starting_balance
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0

        for trade in trades:
            pnl_dollar = float(trade.get("pnl_dollar", 0))
            self.balance += pnl_dollar
            self.total_pnl += pnl_dollar
            if pnl_dollar > 0:
                self.wins += 1
            else:
                self.losses += 1
            self.trades.append(trade)

        logger.info(
            f"Restored {len(trades)} trades: "
            f"balance=${self.balance:.2f}, "
            f"total_pnl=${self.total_pnl:.2f}, "
            f"W/L={self.wins}/{self.losses}"
        )


# ══════════════════════════════════════════════════════════════════════════
# V4.16 Signal Generator — config-driven, improved R:R & risk management
# ══════════════════════════════════════════════════════════════════════════

class V416SignalGenerator:
    """
    V4.16 signal generator with improved risk-adjusted returns.

    Key differences from V414SignalGenerator:
      1. Config-driven — reads all parameters from a StrategyConfig dataclass
      2. Asymmetric TP/SL — wider TP with tighter SL → R:R ≥ 1.2
      3. Vol-scaled Kelly sizing — risk a fixed fraction of equity per trade
      4. Time-of-day filter — raises threshold during historically poor hours
      5. Minimum signal strength gate — rejects low-conviction trades
      6. More aggressive trailing stops — lock profits earlier

    The update() method returns the exact same dict schema as V414SignalGenerator
    so the model server and frontend require zero changes.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: a StrategyConfig (from api.version_config) or compatible object.
        """
        # Import here to avoid circular import at module level
        from api.version_config import StrategyConfig  # noqa: F811
        if not isinstance(cfg, StrategyConfig):
            raise TypeError(f"Expected StrategyConfig, got {type(cfg).__name__}")

        self.cfg = cfg

        # Asymmetric conviction thresholds (v4.18+). Fall back to the
        # symmetric entry_threshold band when not configured.
        self._long_thr = cfg.entry_threshold_long if cfg.entry_threshold_long is not None \
            else cfg.entry_threshold
        self._short_thr = cfg.entry_threshold_short if cfg.entry_threshold_short is not None \
            else 1.0 - cfg.entry_threshold

        # Mirror the same param names used in V414SignalGenerator
        self.entry_threshold = cfg.entry_threshold
        self.exit_threshold = cfg.exit_threshold
        self.min_hold = cfg.min_hold
        self.cooldown = cfg.cooldown
        self.cb_lookback = cfg.cb_lookback
        self.cb_threshold = cfg.cb_threshold

        # Position state
        self.position: int = 0          # -1, 0, +1
        self.bars_since_change: int = 10_000
        self.bar_count: int = 0

        # Circuit breaker
        self.pnl_history: deque = deque(maxlen=cfg.cb_lookback)
        self.cb_active: bool = False

        # Portfolio tracking
        self.starting_balance = 1000.0
        self.balance = 1000.0
        self.entry_ratio: float = 0.0
        self.entry_time: int = 0
        self.entry_proba: float = 0.5
        self.entry_strength: float = 0.0
        self.current_position_size: float = 0.0
        self.current_stop_loss: float = 0.0
        self.current_take_profit: float = 0.0
        self.total_pnl: float = 0.0
        self.wins: int = 0
        self.losses: int = 0
        self.trades: list = []
        self.current_position_label: Optional[str] = None

        # Volatility tracking
        self._recent_returns: deque = deque(maxlen=30)

        # ── V4.16 live-hardening state ──
        # Directional bias breaker: track recent trade directions
        self._recent_directions: deque = deque(maxlen=cfg.direction_bias_lookback)

        # Loss streak cooldown: track consecutive losses
        self._consecutive_losses: int = 0

        # Drawdown-scaled sizing: track peak balance
        self._peak_balance: float = self.starting_balance

    # ── Dynamic helpers ──────────────────────────────────────────────────

    def _signal_strength(self, proba_up: float) -> float:
        """Map probability to 0-1 strength (distance from 0.5)."""
        return float(abs(proba_up - 0.5) * 2.0)

    def _recent_volatility(self) -> float:
        """Annualised vol from recent 30-bar returns."""
        if len(self._recent_returns) < 10:
            return 0.001
        arr = np.array(self._recent_returns)
        return float(np.std(arr)) if np.std(arr) > 1e-8 else 0.001

    # ── Time-of-day filter ───────────────────────────────────────────────

    def _should_trade(self, proba_up: float, timestamp: int) -> bool:
        """
        Return True if the trade passes the time-of-day filter.

        During penalty hours, require a higher probability to enter.
        """
        if not self.cfg.time_filter_enabled:
            return True

        try:
            from datetime import datetime, timezone
            hour = datetime.fromtimestamp(timestamp, tz=timezone.utc).hour
        except (OSError, ValueError):
            return True  # fail open

        if hour in self.cfg.time_filter_penalty_hours:
            # Need extra conviction during penalty hours
            boosted_threshold = self.entry_threshold + self.cfg.time_filter_extra_threshold
            is_long = proba_up >= boosted_threshold
            is_short = proba_up <= (1.0 - boosted_threshold)
            return is_long or is_short

        return True

    # ── Position sizing: vol-scaled Kelly ────────────────────────────────

    def _drawdown_scaler(self) -> float:
        """Return a [floor, 1.0] multiplier based on current drawdown from peak."""
        if not self.cfg.drawdown_scaling_enabled or self._peak_balance <= 0:
            return 1.0
        dd = (self.balance - self._peak_balance) / self._peak_balance  # negative when in DD
        if dd >= self.cfg.drawdown_scaling_start:
            return 1.0  # Not in meaningful drawdown
        # Linear scale from 1.0 at start down to floor at cb_threshold (or 3x start)
        dd_range = self.cfg.drawdown_scaling_start * 3.0  # full floor at 3x start
        progress = (dd - self.cfg.drawdown_scaling_start) / (dd_range - self.cfg.drawdown_scaling_start)
        progress = max(0.0, min(1.0, progress))
        return 1.0 - progress * (1.0 - self.cfg.drawdown_scaling_floor)

    def _position_size_frac(self, strength: float, sl_pct: float) -> float:
        """
        Vol-scaled Kelly: risk a fixed fraction of equity at the stop-loss.

        position_size = max_risk / sl_distance
        Then scale by signal strength and clamp to [min_leverage, max_leverage].
        Applies drawdown scaling when equity is below peak.
        """
        if self.cfg.position_sizing_mode == "vol_scaled_kelly":
            if sl_pct <= 1e-8:
                return self.cfg.min_leverage
            # How much of balance to allocate so that SL hit = max_risk_per_trade loss
            raw_size = self.cfg.max_risk_per_trade / sl_pct
            # Scale by strength (stronger signals get closer to full Kelly)
            scaled = self.cfg.min_leverage + (raw_size - self.cfg.min_leverage) * min(strength, 1.0)
            size = max(self.cfg.min_leverage, min(scaled, self.cfg.max_leverage))
        else:
            # Fallback: v4.15-compatible target_win_frac mode
            tp_pct = sl_pct * 2.0  # rough estimate
            if tp_pct <= 1e-8:
                return self.cfg.min_leverage
            required = self.cfg.target_win_frac / tp_pct
            scaled = self.cfg.min_leverage + (required - self.cfg.min_leverage) * min(strength, 1.0)
            size = max(self.cfg.min_leverage, min(scaled, self.cfg.max_leverage))

        # Apply drawdown scaling
        size *= self._drawdown_scaler()
        return max(self.cfg.min_leverage * self.cfg.drawdown_scaling_floor, size)

    # ── Asymmetric SL/TP ─────────────────────────────────────────────────

    def _compute_sl_tp(self, direction: int, entry_ratio: float,
                       strength: float) -> Tuple[float, float, float, float]:
        """
        Compute dynamic stop-loss and take-profit using asymmetric widths.

        Returns (stop_loss_price, take_profit_price, sl_pct_fraction, tp_pct_fraction).
        """
        vol = self._recent_volatility()
        vol_mult = max(vol * 100, 0.05)  # vol as %, floor 0.05%

        # Asymmetric base widths
        sl_pct = max(abs(self.cfg.default_stop_loss_pct), vol_mult * self.cfg.sl_vol_mult) / 100.0
        tp_pct = max(abs(self.cfg.default_take_profit_pct), vol_mult * self.cfg.tp_vol_mult) / 100.0

        # Confidence adjustment
        sl_pct *= (1.0 + strength * self.cfg.sl_strength_scale)
        tp_pct *= (1.0 - strength * self.cfg.tp_strength_scale)

        if direction == 1:  # LONG
            stop_loss = entry_ratio * (1.0 - sl_pct)
            take_profit = entry_ratio * (1.0 + tp_pct)
        else:  # SHORT
            stop_loss = entry_ratio * (1.0 + sl_pct)
            take_profit = entry_ratio * (1.0 - tp_pct)

        return stop_loss, take_profit, sl_pct, tp_pct

    # ── Trailing stop-loss ───────────────────────────────────────────────

    def _update_trailing_sl(self, current_ratio: float) -> None:
        """Update stop-loss using trailing logic when trade is in profit."""
        if self.position == 0 or self.entry_ratio <= 0:
            return
        if self.current_take_profit <= 0:
            return

        if self.position == 1:
            tp_distance = self.current_take_profit - self.entry_ratio
            current_profit = current_ratio - self.entry_ratio
        else:
            tp_distance = self.entry_ratio - self.current_take_profit
            current_profit = self.entry_ratio - current_ratio

        if tp_distance <= 0:
            return

        progress = current_profit / tp_distance

        if progress >= self.cfg.trailing_lock_frac:
            locked_profit = current_profit * self.cfg.trailing_lock_ratio
            if self.position == 1:
                new_sl = self.entry_ratio + locked_profit
                self.current_stop_loss = max(self.current_stop_loss, new_sl)
            else:
                new_sl = self.entry_ratio - locked_profit
                self.current_stop_loss = min(self.current_stop_loss, new_sl)
        elif progress >= self.cfg.trailing_breakeven_frac:
            if self.position == 1:
                self.current_stop_loss = max(self.current_stop_loss, self.entry_ratio)
            else:
                self.current_stop_loss = min(self.current_stop_loss, self.entry_ratio)

    # ── Dynamic min-hold ─────────────────────────────────────────────────

    def _effective_min_hold(self, current_ratio: float, proba_up: float) -> int:
        """Return dynamic min-hold — can be reduced by SL/TP or strong flip."""
        if self.position == 0 or self.entry_ratio <= 0:
            return self.min_hold
        if self.bars_since_change >= self.min_hold:
            return self.min_hold
        if self.bars_since_change < self.cfg.absolute_min_hold:
            return self.min_hold

        # SL/TP price check
        if self.position == 1:
            hit_tp = self.current_take_profit > 0 and current_ratio >= self.current_take_profit
            hit_sl = self.current_stop_loss > 0 and current_ratio <= self.current_stop_loss
        else:
            hit_tp = self.current_take_profit > 0 and current_ratio <= self.current_take_profit
            hit_sl = self.current_stop_loss > 0 and current_ratio >= self.current_stop_loss

        if hit_tp or hit_sl:
            return self.cfg.absolute_min_hold

        # Strong opposite signal
        if self.position == 1 and proba_up <= (self._short_thr - 0.01):
            return self.cfg.absolute_min_hold
        if self.position == -1 and proba_up >= (self._long_thr + 0.01):
            return self.cfg.absolute_min_hold

        return self.min_hold

    # ── Trade open / close ───────────────────────────────────────────────

    def _open_position(self, direction: int, ratio: float, timestamp: int,
                       proba_up: float = 0.5, strength: float = 0.0):
        self.entry_ratio = ratio
        self.entry_time = timestamp
        self.entry_proba = proba_up
        self.entry_strength = strength
        self.current_position_label = "LONG" if direction == 1 else "SHORT"

        sl, tp, sl_pct, tp_pct = self._compute_sl_tp(direction, ratio, strength)
        self.current_stop_loss = sl
        self.current_take_profit = tp

        # Position sizing uses SL distance for Kelly mode
        self.current_position_size = self._position_size_frac(strength, sl_pct)

        logger.info(
            f"OPEN {self.current_position_label} @ {ratio:.4f} | "
            f"size={self.current_position_size:.2f}x | "
            f"SL={sl:.4f} TP={tp:.4f} (R:R={tp_pct/sl_pct:.2f}:1)"
        )

    def _close_position(self, exit_ratio: float, timestamp: int,
                        reason: str = "Signal change"):
        if self.entry_ratio <= 0:
            return
        prev_pos = self.position
        if prev_pos == 1:
            pnl_pct = (exit_ratio - self.entry_ratio) / self.entry_ratio
        else:
            pnl_pct = (self.entry_ratio - exit_ratio) / self.entry_ratio

        trade_size = self.balance * self.current_position_size
        pnl_dollar = trade_size * pnl_pct
        self.balance += pnl_dollar
        self.total_pnl += pnl_dollar

        if pnl_dollar > 0:
            self.wins += 1
            self._consecutive_losses = 0
        else:
            self.losses += 1
            self._consecutive_losses += 1

        bars_held = self.bars_since_change
        trade_direction = "LONG" if prev_pos == 1 else "SHORT"

        # Track direction for bias breaker
        self._recent_directions.append(trade_direction)

        self.trades.append({
            "direction": trade_direction,
            "entry_price": self.entry_ratio,
            "exit_price": exit_ratio,
            "entry_time": self.entry_time,
            "exit_time": timestamp,
            "pnl_pct": pnl_pct * 100,
            "pnl_dollar": pnl_dollar,
            "bars_held": bars_held,
            "position_size_pct": self.current_position_size * 100,
            "stop_loss": self.current_stop_loss,
            "take_profit": self.current_take_profit,
            "entry_probability": self.entry_proba,
            "entry_strength": self.entry_strength,
            "reason": reason,
            "model_version": "v4.16",
        })

        logger.info(
            f"CLOSE {self.current_position_label} @ {exit_ratio:.4f} | "
            f"PnL={pnl_pct*100:.3f}% (${pnl_dollar:.2f}) | "
            f"Bars={bars_held} | Reason={reason}"
        )

        self.entry_ratio = 0.0
        self.entry_time = 0
        self.entry_proba = 0.5
        self.entry_strength = 0.0
        self.current_position_size = 0.0
        self.current_stop_loss = 0.0
        self.current_take_profit = 0.0
        self.current_position_label = None

    # ── Reasoning / blocked_by (identical to V414) ───────────────────────

    def _build_reasoning(self, proba_up: float, can_change: bool,
                         early_exit_reason: Optional[str] = None) -> list:
        lines = []
        lines.append(f"P(up) = {proba_up:.4f}")
        if proba_up >= self.entry_threshold:
            lines.append(f"Above entry threshold ({self.entry_threshold}) -> LONG signal")
        elif proba_up <= (1 - self.entry_threshold):
            lines.append(f"Below entry threshold ({1 - self.entry_threshold:.3f}) -> SHORT signal")
        else:
            lines.append(f"Within neutral band ({1 - self.entry_threshold:.3f} - {self.entry_threshold})")

        if self.position != 0:
            pos_label = "LONG" if self.position == 1 else "SHORT"
            lines.append(f"Current position: {pos_label}")
            lines.append(f"Bars held: {self.bars_since_change}")
            if self.current_position_size > 0:
                lines.append(f"Size: {self.current_position_size*100:.1f}%")
            if not can_change:
                eff_mh = self._effective_min_hold(0, proba_up)
                lines.append(f"Min hold not met ({self.bars_since_change}/{eff_mh})")

        if early_exit_reason:
            lines.append(f"Early exit: {early_exit_reason}")

        if self.cb_active:
            lines.append("CIRCUIT BREAKER ACTIVE - forced NEUTRAL")

        if self.bars_since_change <= self.cooldown and self.position != 0:
            lines.append(f"Cooldown active ({self.bars_since_change}/{self.cooldown})")

        # V4.16 safeguard reasoning
        if self._consecutive_losses >= self.cfg.loss_streak_threshold:
            lines.append(f"Loss streak: {self._consecutive_losses} (threshold tightened)")

        dd_scaler = self._drawdown_scaler()
        if dd_scaler < 1.0:
            lines.append(f"Drawdown scaling: {dd_scaler:.0%} position size")

        if (self.cfg.direction_bias_enabled
                and len(self._recent_directions) >= self.cfg.direction_bias_lookback
                and len(set(self._recent_directions)) == 1):
            lines.append(f"Direction bias: last {len(self._recent_directions)} trades all {self._recent_directions[-1]}")

        return lines

    def _get_blocked_by(self, proba_up: float, can_change: bool) -> list:
        blocked = []
        if self.cb_active:
            blocked.append("Circuit breaker")
        if not can_change and self.position != 0:
            eff_mh = self._effective_min_hold(0, proba_up)
            blocked.append(f"Min hold ({self.bars_since_change}/{eff_mh})")
        if self.bars_since_change <= self.cooldown:
            blocked.append(f"Cooldown ({self.bars_since_change}/{self.cooldown})")

        # V4.16 safeguards
        if (self.cfg.direction_bias_enabled
                and len(self._recent_directions) >= self.cfg.direction_bias_lookback
                and len(set(self._recent_directions)) == 1):
            blocked.append(f"Direction bias ({self._recent_directions[-1]})")

        if self._consecutive_losses >= self.cfg.loss_streak_threshold:
            blocked.append(f"Loss streak ({self._consecutive_losses})")

        return blocked

    # ── Main update ──────────────────────────────────────────────────────

    def update(self, proba_up: float, current_ratio: float, timestamp: int) -> Dict:
        """
        Process one bar — identical interface to V414SignalGenerator.update().
        """
        # Track returns for volatility estimation
        if self.position != 0 and self.entry_ratio > 0:
            self._recent_returns.append(
                np.log(current_ratio / self.entry_ratio) / max(self.bars_since_change, 1)
            )
        elif len(self._recent_returns) == 0:
            self._recent_returns.append(0.0)

        prev = self.position
        desired = prev

        # Dynamic min-hold
        effective_mh = self._effective_min_hold(current_ratio, proba_up)
        can_change = self.bars_since_change >= effective_mh

        # Update trailing stop
        self._update_trailing_sl(current_ratio)

        # Check SL/TP exit
        early_exit_reason = None
        if self.position != 0 and self.entry_ratio > 0 and can_change:
            if self.position == 1:
                if self.current_take_profit > 0 and current_ratio >= self.current_take_profit:
                    early_exit_reason = "Take profit"
                elif self.current_stop_loss > 0 and current_ratio <= self.current_stop_loss:
                    early_exit_reason = "Stop loss"
            else:
                if self.current_take_profit > 0 and current_ratio <= self.current_take_profit:
                    early_exit_reason = "Take profit"
                elif self.current_stop_loss > 0 and current_ratio >= self.current_stop_loss:
                    early_exit_reason = "Stop loss"

        # Time-based exit: position has been held to its horizon (v4.18+)
        if (self.position != 0 and not early_exit_reason
                and self.cfg.max_hold_bars > 0
                and self.bars_since_change >= self.cfg.max_hold_bars):
            early_exit_reason = "Max hold"

        # Unrealized PnL
        _unrealized_pnl_pct = 0.0
        if self.position != 0 and self.entry_ratio > 0:
            if self.position == 1:
                _unrealized_pnl_pct = (current_ratio - self.entry_ratio) / self.entry_ratio
            else:
                _unrealized_pnl_pct = (self.entry_ratio - current_ratio) / self.entry_ratio

        # ── Hysteresis logic (asymmetric-aware) ──
        if prev == 0:
            if proba_up >= self._long_thr:
                desired = 1
            elif proba_up <= self._short_thr:
                desired = -1
        elif prev == 1:
            if can_change:
                if early_exit_reason:
                    desired = 0
                elif proba_up <= self._short_thr:
                    desired = -1
                elif proba_up < self.exit_threshold:
                    desired = 0
        elif prev == -1:
            if can_change:
                if early_exit_reason:
                    desired = 0
                elif proba_up >= self._long_thr:
                    desired = 1
                elif proba_up > (1.0 - self.exit_threshold):
                    desired = 0

        # ── Signal-exit guard ──
        if desired != prev and prev != 0 and not early_exit_reason:
            is_strong_flip = False
            if prev == 1 and proba_up <= self._short_thr:
                is_strong_flip = True
            elif prev == -1 and proba_up >= self._long_thr:
                is_strong_flip = True

            if _unrealized_pnl_pct < 0 and not is_strong_flip:
                desired = prev

        # Cooldown
        if desired != prev and self.bars_since_change <= self.cooldown and not early_exit_reason:
            desired = prev

        # ── V4.16-specific: Signal strength gate ──
        if desired != 0 and prev == 0:
            strength = self._signal_strength(proba_up)
            if strength < self.cfg.min_signal_strength:
                desired = 0  # Reject weak signals

        # ── V4.16-specific: Time-of-day filter ──
        if desired != 0 and prev == 0:
            if not self._should_trade(proba_up, timestamp):
                desired = 0

        # ── V4.16-specific: Directional bias breaker ──
        if (desired != 0 and prev == 0
                and self.cfg.direction_bias_enabled
                and len(self._recent_directions) >= self.cfg.direction_bias_lookback):
            desired_dir = "LONG" if desired == 1 else "SHORT"
            if all(d == desired_dir for d in self._recent_directions):
                desired = 0  # Block same-direction entry

        # ── V4.16-specific: Loss streak cooldown ──
        if desired != 0 and prev == 0 and self._consecutive_losses >= self.cfg.loss_streak_threshold:
            extra = min(
                (self._consecutive_losses - self.cfg.loss_streak_threshold + 1) * self.cfg.loss_streak_extra_threshold,
                self.cfg.loss_streak_max_extra,
            )
            tightened_entry = self.entry_threshold + extra
            is_long = proba_up >= tightened_entry
            is_short = proba_up <= (1.0 - tightened_entry)
            if not (is_long or is_short):
                desired = 0  # Signal not strong enough during loss streak

        # ── Circuit breaker ──
        if len(self.pnl_history) >= self.cb_lookback // 2:
            rolling_pnl = sum(self.pnl_history)
            self.cb_active = rolling_pnl < self.cb_threshold
        else:
            self.cb_active = False

        if self.cb_active:
            desired = 0

        # ── Apply position change ──
        close_reason = early_exit_reason or "Signal change"
        if desired != prev:
            if prev != 0 and self.entry_ratio > 0:
                self._close_position(current_ratio, timestamp, close_reason)
            if desired != 0:
                strength = self._signal_strength(proba_up)
                self._open_position(desired, current_ratio, timestamp, proba_up, strength)
            self.bars_since_change = 0
        else:
            self.bars_since_change += 1

        self.position = desired
        self.bar_count += 1

        # Update peak balance for drawdown scaling
        if self.balance > self._peak_balance:
            self._peak_balance = self.balance

        # Track bar PnL for circuit breaker
        bar_pnl = 0.0
        if self.position != 0 and self.entry_ratio > 0:
            ratio_return = np.log(current_ratio / self.entry_ratio)
            bar_pnl = self.position * ratio_return
        self.pnl_history.append(bar_pnl)

        # Build output
        reasoning = self._build_reasoning(proba_up, can_change, early_exit_reason)

        direction = "LONG" if desired == 1 else "SHORT" if desired == -1 else "NEUTRAL"
        triggered = desired != prev and desired != 0

        unrealized_pnl = 0.0
        unrealized_pnl_pct = 0.0
        if self.position != 0 and self.entry_ratio > 0:
            if self.position == 1:
                unrealized_pnl_pct = (current_ratio - self.entry_ratio) / self.entry_ratio
            else:
                unrealized_pnl_pct = (self.entry_ratio - current_ratio) / self.entry_ratio
            unrealized_pnl = self.balance * self.current_position_size * unrealized_pnl_pct

        win_rate = self.wins / (self.wins + self.losses) * 100 if (self.wins + self.losses) > 0 else 0

        return {
            "direction": direction,
            "strength": self._signal_strength(proba_up),
            "probability": float(proba_up),
            "triggered": triggered,
            "blocked_by": self._get_blocked_by(proba_up, can_change),
            "reasoning": reasoning,
            "circuit_breaker_active": self.cb_active,
            "position_meta": {
                "stop_loss": self.current_stop_loss if self.position != 0 else 0.0,
                "take_profit": self.current_take_profit if self.position != 0 else 0.0,
                "position_size_pct": self.current_position_size * 100 if self.position != 0 else 0.0,
                "effective_min_hold": effective_mh,
                "bars_held": self.bars_since_change,
            },
            "portfolio": {
                "balance": self.balance,
                "starting_balance": self.starting_balance,
                "total_pnl": self.total_pnl,
                "total_pnl_pct": (self.balance - self.starting_balance) / self.starting_balance * 100,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct * 100,
                "position": self.current_position_label,
                "entry_price": self.entry_ratio,
                "total_trades": len(self.trades),
                "wins": self.wins,
                "losses": self.losses,
                "win_rate": win_rate,
                "recent_trades": self.trades[-20:],
            },
        }

    # ── State restoration ─────────────────────────────────────────────────

    def restore_from_trades(self, trades: list) -> None:
        """Restore portfolio state from persisted trade history.

        Rebuilds balance, wins/losses, direction bias history, loss streak
        counter, and peak balance so all v4.16 safeguards work correctly
        after a server restart.
        """
        self.trades = []
        self.balance = self.starting_balance
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0
        self._peak_balance = self.starting_balance
        self._consecutive_losses = 0
        self._recent_directions.clear()

        for trade in trades:
            pnl_dollar = float(trade.get("pnl_dollar", 0))
            self.balance += pnl_dollar
            self.total_pnl += pnl_dollar

            if pnl_dollar > 0:
                self.wins += 1
                self._consecutive_losses = 0
            else:
                self.losses += 1
                self._consecutive_losses += 1

            # Track peak balance for drawdown scaling
            if self.balance > self._peak_balance:
                self._peak_balance = self.balance

            # Rebuild direction history for bias breaker
            direction = trade.get("direction", "")
            if direction in ("LONG", "SHORT"):
                self._recent_directions.append(direction)

            self.trades.append(trade)

        logger.info(
            f"[V4.16] Restored {len(trades)} trades: "
            f"balance=${self.balance:.2f}, "
            f"total_pnl=${self.total_pnl:.2f}, "
            f"W/L={self.wins}/{self.losses}, "
            f"loss_streak={self._consecutive_losses}, "
            f"peak=${self._peak_balance:.2f}, "
            f"recent_dirs={list(self._recent_directions)[-4:]}"
        )
