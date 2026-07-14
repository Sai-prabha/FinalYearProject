"""Microstructure feature pipeline — offline, deterministic, read-only.

Reads the parquet segments written by microstructure_ingest and computes
research features (MICROSTRUCTURE.md table). Pure functions of stored rows:
no network, no wall clock — same inputs, same outputs. Nothing in any live
model path imports this module; it exists for training scripts, Strategy
Lab experiments, and candidate pre-registrations.

Alignment recipe for kline-based training (see MICROSTRUCTURE.md):
    f = load_microstructure_features("BTCUSDT", t0, t1)
    f1m = f.resample("1min").last().shift(1)   # known at bar open
    joined = klines.join(f1m, on="time")
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    from .microstructure_ingest import MICRO_DIR
except ImportError:  # script use: `sys.path.insert(ROOT)` + `import api.x`
    from api.microstructure_ingest import MICRO_DIR

RV_SHORT_S = 60
RV_LONG_S = 300
REGIME_WINDOW_S = 86_400          # trailing 1 day of 1s rows
REGIME_MIN_PERIODS = 3_600        # conservative: unknown regime until 1h seen
REGIME_Q = 0.90


def _day_range(start_ts: float, end_ts: float) -> List[str]:
    d0 = datetime.fromtimestamp(start_ts, tz=timezone.utc).date()
    d1 = datetime.fromtimestamp(end_ts, tz=timezone.utc).date()
    out, d = [], d0
    while d <= d1:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def _read_segments(kind: str, symbol: str, start_ts: float, end_ts: float,
                   root: Path = MICRO_DIR) -> pd.DataFrame:
    parts = []
    for day in _day_range(start_ts, end_ts):
        d = root / kind / symbol / day
        if not d.exists():
            continue
        for f in sorted(d.glob("*.parquet")):
            parts.append(pd.read_parquet(f))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    ts_col = "ts_s" if "ts_s" in df.columns else (
        "event_ms" if "event_ms" in df.columns else "ts_ms")
    secs = df[ts_col] if ts_col == "ts_s" else df[ts_col] / 1000.0
    df = df[(secs >= start_ts) & (secs < end_ts)]
    return df.sort_values(ts_col).drop_duplicates(ts_col, keep="last").reset_index(drop=True)


def load_lob(symbol: str, start_ts: float, end_ts: float,
             root: Path = MICRO_DIR) -> pd.DataFrame:
    return _read_segments("lob", symbol, start_ts, end_ts, root)


def load_trades_1s(symbol: str, start_ts: float, end_ts: float,
                   root: Path = MICRO_DIR) -> pd.DataFrame:
    return _read_segments("trades1s", symbol, start_ts, end_ts, root)


# ── Feature computations (pure) ────────────────────────────────────────────

def compute_ofi(best_bid: np.ndarray, bid_sz: np.ndarray,
                best_ask: np.ndarray, ask_sz: np.ndarray) -> np.ndarray:
    """Best-level order-flow imbalance between consecutive snapshots
    (Cont–Kukanov–Stoikov 2014). First element is 0 (no predecessor)."""
    pb, sb = best_bid[:-1], bid_sz[:-1]
    pa, sa = best_ask[:-1], ask_sz[:-1]
    b, qb = best_bid[1:], bid_sz[1:]
    a, qa = best_ask[1:], ask_sz[1:]
    e_bid = np.where(b > pb, qb, np.where(b == pb, qb - sb, -sb))
    e_ask = np.where(a < pa, qa, np.where(a == pa, qa - sa, -sa))
    ofi = np.zeros(len(best_bid))
    ofi[1:] = e_bid - e_ask
    return ofi


def compute_lob_features(lob: pd.DataFrame) -> pd.DataFrame:
    """1s-grid book features from raw LOB rows."""
    if lob.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"ts_s": (lob["event_ms"] // 1000).astype(int)})
    out["mid"] = lob["mid"].to_numpy()
    out["spread"] = lob["spread"].to_numpy()
    out["spread_bps"] = (lob["spread"] / lob["mid"] * 10_000).to_numpy()
    b0, a0 = lob["bid_sz_0"].to_numpy(), lob["ask_sz_0"].to_numpy()
    out["imb_top1"] = b0 / np.where(b0 + a0 > 0, b0 + a0, np.nan)
    bd, ad = lob["bid_depth20"].to_numpy(), lob["ask_depth20"].to_numpy()
    out["imb_top20"] = bd / np.where(bd + ad > 0, bd + ad, np.nan)
    out["ofi_1s"] = compute_ofi(lob["best_bid"].to_numpy(), b0,
                                lob["best_ask"].to_numpy(), a0)
    logret = np.log(out["mid"] / out["mid"].shift(1))
    out["rv_60s"] = logret.rolling(RV_SHORT_S, min_periods=RV_SHORT_S).std()
    out["rv_300s"] = logret.rolling(RV_LONG_S, min_periods=RV_LONG_S).std()
    return out


def compute_trade_features(tr: pd.DataFrame) -> pd.DataFrame:
    if tr.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"ts_s": tr["ts_s"].astype(int)})
    out["signed_flow_1s"] = (tr["buy_qty"] - tr["sell_qty"]).to_numpy()
    tot = (tr["buy_qty"] + tr["sell_qty"]).to_numpy()
    out["aggression"] = tr["buy_qty"].to_numpy() / np.where(tot > 0, tot, np.nan)
    out["n_trades_1s"] = tr["n_trades"].to_numpy()
    return out


def label_regime(df: pd.DataFrame) -> pd.Series:
    """quiet / normal / toxic from rv_300s and spread_bps vs their own
    trailing quantiles (self-normalizing, same philosophy as the shipped
    RV240/p90 vol filter). Unknown history ⇒ 'toxic' (conservative)."""
    rv, sp = df["rv_300s"], df["spread_bps"]
    rv_hi = rv.rolling(REGIME_WINDOW_S, min_periods=REGIME_MIN_PERIODS).quantile(REGIME_Q)
    sp_hi = sp.rolling(REGIME_WINDOW_S, min_periods=REGIME_MIN_PERIODS).quantile(REGIME_Q)
    rv_lo = rv.rolling(REGIME_WINDOW_S, min_periods=REGIME_MIN_PERIODS).quantile(0.5)
    sp_lo = sp.rolling(REGIME_WINDOW_S, min_periods=REGIME_MIN_PERIODS).quantile(0.5)
    toxic = (rv > rv_hi) | (sp > sp_hi) | rv_hi.isna() | sp_hi.isna()
    quiet = (rv < rv_lo) & (sp < sp_lo) & ~toxic
    out = pd.Series("normal", index=df.index)
    out[toxic] = "toxic"
    out[quiet] = "quiet"
    return out


def load_microstructure_features(symbol: str, start_ts: float, end_ts: float,
                                 feature_set: str = "basic",
                                 root: Path = MICRO_DIR) -> pd.DataFrame:
    """Standardized feature frame on the 1s grid, indexed by UTC timestamp.
    Deterministic: same stored segments in, same frame out."""
    if feature_set != "basic":
        raise ValueError(f"unknown feature_set {feature_set!r}")
    lobf = compute_lob_features(load_lob(symbol, start_ts, end_ts, root))
    trf = compute_trade_features(load_trades_1s(symbol, start_ts, end_ts, root))
    if lobf.empty and trf.empty:
        return pd.DataFrame()
    if lobf.empty or trf.empty:
        df = lobf if trf.empty else trf
    else:
        df = lobf.merge(trf, on="ts_s", how="outer").sort_values("ts_s").reset_index(drop=True)
    if "rv_300s" in df.columns and "spread_bps" in df.columns:
        df["regime"] = label_regime(df)
    df.index = pd.to_datetime(df["ts_s"], unit="s", utc=True)
    df.index.name = "time"
    return df
