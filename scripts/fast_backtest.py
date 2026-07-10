#!/usr/bin/env python3
"""
Fast strategy backtest with cached model probabilities and transaction costs.

Why this exists: the per-bar harness (backtest_v417_candidate.py) recomputes the
full feature frame every bar (~hours per window). But P(up) does not depend on
the strategy config, so we compute features ONCE vectorized over the whole
window, batch-predict, cache (time, proba, ratio) to parquet, then replay any
number of signal-generator configs over the cache in seconds.

Feature parity: all features are causal rolling ops with min_periods=window
(max lookback 780 bars), so full-frame values equal the live trailing-1500-bar
buffer values. `--parity N` verifies this by rebuilding N random bars through
the real live path (V414FeatureCalculator.calculate_features) and comparing
probabilities.

Costs: `--fee-bps-per-side` (default 4.5 = Binance USDM taker 0.045%) is
charged per leg per side. A ratio round trip = 2 legs x 2 sides = 4 fills.
Cost is deducted from the signal generator's balance at each trade close so
compounding, drawdown scaling and equity curves see net PnL.

Usage:
    python scripts/fast_backtest.py --start 2025-10-01 --end 2025-12-15 \
        --versions v4.15,v4.16,v4.17 --label tune2025
    python scripts/fast_backtest.py --start 2025-12-15 --end 2026-07-01 \
        --versions v4.15,v4.16,v4.17 --label holdout2026
    python scripts/fast_backtest.py --start 2025-10-01 --end 2025-12-15 --parity 40
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import requests
from xgboost import XGBClassifier

from api.feature_calculator import (
    ZSCORE_CLIP_SIGMA,
    LAG_RETURNS,
    V414FeatureCalculator,
    V414SignalGenerator,
    V416SignalGenerator,
    _zscore,
    _rsi,
)
from api.version_config import get_strategy_config

MODEL_DIR = ROOT / "models" / "v4_14_production"
CACHE_DIR = ROOT / "data" / "backtest"
BINANCE_REST = "https://api.binance.com/api/v3/klines"
WARMUP = 1000
STARTING_BALANCE = 1000.0


# ── Data ───────────────────────────────────────────────────────────────────

def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch 1m klines, cached to parquet per (symbol, window)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"klines_{symbol}_{start_ms}_{end_ms}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        print(f"  {symbol}: {len(df):,} candles (cached)")
        return df

    rows, cursor = [], start_ms
    print(f"  Fetching {symbol} ...", end="", flush=True)
    while cursor < end_ms:
        params = {"symbol": symbol, "interval": "1m", "startTime": cursor,
                  "endTime": end_ms, "limit": 1000}
        for attempt in range(4):
            try:
                r = requests.get(BINANCE_REST, params=params, timeout=15)
                r.raise_for_status()
                batch = r.json()
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        if not batch:
            break
        for k in batch:
            rows.append((int(k[0]) // 1000, float(k[1]), float(k[2]), float(k[3]),
                         float(k[4]), float(k[5]), float(k[9])))
        cursor = int(batch[-1][0]) + 60_000
        print(".", end="", flush=True)
        if len(batch) < 1000:
            break
        time.sleep(0.12)

    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close",
                                     "volume", "taker_buy_volume"])
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df.to_parquet(cache)
    print(f" {len(df):,} candles")
    return df


# ── Vectorized features + batch predict ──────────────────────────────────

def compute_probas(btc: pd.DataFrame, eth: pd.DataFrame,
                   model: XGBClassifier, feature_names: list) -> pd.DataFrame:
    """Full-frame replica of V414FeatureCalculator.calculate_features().

    Returns DataFrame(time, ratio, proba, valid) for every aligned candle.
    Reuses the calculator's static helpers so the math stays single-sourced.
    """
    merged = pd.merge(btc, eth, on="time", suffixes=("_btc", "_eth"), how="inner")
    merged = merged.sort_values("time").reset_index(drop=True)
    orig_times = merged["time"].to_numpy()
    merged = V414FeatureCalculator._clean_merged_data(merged)

    df = pd.DataFrame({
        "open_time": pd.to_datetime(merged["time"], unit="s", utc=True).dt.tz_localize(None),
        "btc_close": merged["close_btc"],
        "eth_close": merged["close_eth"],
        "R": merged["close_btc"] / merged["close_eth"],
        "btc_buy_ratio": merged["taker_buy_volume_btc"] / (merged["volume_btc"] + 1e-12),
        "btc_net_buy_pressure": (2 * merged["taker_buy_volume_btc"] / (merged["volume_btc"] + 1e-12)) - 1,
        "eth_buy_ratio": merged["taker_buy_volume_eth"] / (merged["volume_eth"] + 1e-12),
        "eth_net_buy_pressure": (2 * merged["taker_buy_volume_eth"] / (merged["volume_eth"] + 1e-12)) - 1,
    })
    df["buy_pressure_divergence"] = df["btc_buy_ratio"] - df["eth_buy_ratio"]

    ratio = df["R"].astype(float)
    rret = np.log(ratio / ratio.shift(1))

    feats = {}
    for lag in LAG_RETURNS:
        feats[f"r_lag_{lag}"] = rret.shift(lag)
    feats.update(V414FeatureCalculator._rolling_features(rret))
    feats.update(V414FeatureCalculator._trend_features(ratio))
    feats["rret_z_30"] = _zscore(rret.fillna(0), 30, clip_sigma=ZSCORE_CLIP_SIGMA)
    feats["rret_z_60"] = _zscore(rret.fillna(0), 60, clip_sigma=ZSCORE_CLIP_SIGMA)
    feats["rsi_14"] = _rsi(rret.fillna(0), 14)
    feats.update(V414FeatureCalculator._cross_asset_features(df))
    feats.update(V414FeatureCalculator._volume_pressure_features(df))
    feats.update(V414FeatureCalculator._time_features(df))
    feats.update(V414FeatureCalculator._regime_features(df, ratio))

    frame = pd.DataFrame(feats, index=df.index).astype("float32")
    for col in feature_names:
        if col not in frame.columns:
            frame[col] = np.nan
    frame = frame[feature_names]

    valid = ~frame.isna().any(axis=1)
    probas = np.full(len(frame), np.nan, dtype=float)
    if valid.any():
        probas[valid.to_numpy()] = model.predict_proba(frame[valid])[:, 1]

    out = pd.DataFrame({
        "time": merged["time"].astype(int),
        "ratio": (merged["close_btc"] / merged["close_eth"]).astype(float),
        "proba": probas,
        "valid": valid.to_numpy(),
    })
    # Only real candles count as tradeable bars (gap-filled rows are synthetic)
    out = out[out["time"].isin(orig_times)].reset_index(drop=True)
    return out


def parity_check(btc: pd.DataFrame, eth: pd.DataFrame, cached: pd.DataFrame,
                 model: XGBClassifier, feature_names: list, n_samples: int) -> float:
    """Rebuild N random bars through the real live per-bar path; return max |Δproba|."""
    rng = np.random.default_rng(7)
    eligible = cached[cached["valid"]].index
    eligible = eligible[eligible >= 1500]
    picks = rng.choice(eligible, size=min(n_samples, len(eligible)), replace=False)
    btc_by_t = {int(r.time): r._asdict() for r in btc.itertuples(index=False)}
    eth_by_t = {int(r.time): r._asdict() for r in eth.itertuples(index=False)}
    times = cached["time"].to_numpy()

    worst = 0.0
    for idx in sorted(picks):
        t = int(times[idx])
        fc = V414FeatureCalculator(feature_names=feature_names, max_history=1500)
        window_times = times[max(0, idx - 1499): idx + 1]
        for wt in window_times:
            wt = int(wt)
            if wt in btc_by_t and wt in eth_by_t:
                fc.add_candle("BTC", btc_by_t[wt])
                fc.add_candle("ETH", eth_by_t[wt])
        row = fc.calculate_features()
        if row is None:
            continue
        p_live = float(model.predict_proba(row)[:, 1][0])
        p_vec = float(cached.loc[idx, "proba"])
        worst = max(worst, abs(p_live - p_vec))
    return worst


# ── Replay with costs ──────────────────────────────────────────────────────

def make_signal_gen(version: str):
    cfg = get_strategy_config(version)
    if version == "v4.15":
        return V414SignalGenerator(
            entry_threshold=cfg.entry_threshold, exit_threshold=cfg.exit_threshold,
            min_hold=cfg.min_hold, cooldown=cfg.cooldown,
            cb_lookback=cfg.cb_lookback, cb_threshold=cfg.cb_threshold,
            starting_balance=STARTING_BALANCE)
    return V416SignalGenerator(cfg=cfg)


def replay(version: str, cached: pd.DataFrame, fee_bps_per_side: float, sig_gen=None):
    """Run one config over cached probas. Fees: 4 fills per round trip.

    Pass ``sig_gen`` to replay an ad-hoc config (see sweep_v418.py);
    otherwise it is built from the registered ``version`` string.
    """
    if sig_gen is None:
        sig_gen = make_signal_gen(version)
    round_trip_cost = 4 * fee_bps_per_side / 10_000.0
    equity = [sig_gen.balance]
    n_seen = 0

    times = cached["time"].to_numpy()
    ratios = cached["ratio"].to_numpy()
    probas = cached["proba"].to_numpy()
    valids = cached["valid"].to_numpy()

    for i in range(WARMUP, len(cached)):
        if not valids[i]:
            continue
        n_before = len(sig_gen.trades)
        sig_gen.update(float(probas[i]), float(ratios[i]), int(times[i]))
        for trade in sig_gen.trades[n_before:]:
            # Charge round-trip fees on the traded notional at close
            notional = abs(trade["pnl_dollar"] / (trade["pnl_pct"] / 100.0)) \
                if trade["pnl_pct"] != 0 else sig_gen.balance * trade["position_size_pct"] / 100.0
            cost = notional * round_trip_cost
            trade["fee_dollar"] = cost
            trade["pnl_dollar_net"] = trade["pnl_dollar"] - cost
            trade["pnl_pct_net"] = trade["pnl_pct"] - round_trip_cost * 100.0
            sig_gen.balance -= cost
            sig_gen.total_pnl -= cost
        equity.append(sig_gen.balance)
        n_seen += 1

    return sig_gen.trades, np.asarray(equity), n_seen


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(trades: list, equity: np.ndarray, net: bool) -> dict:
    if not trades:
        return {"n_trades": 0}
    pk, dk = ("pnl_pct_net", "pnl_dollar_net") if net else ("pnl_pct", "pnl_dollar")
    pcts = np.array([t[pk] for t in trades])
    dols = np.array([t[dk] for t in trades])
    bars = np.array([t["bars_held"] for t in trades])
    win = dols > 0
    n, wins = len(trades), int(win.sum())
    avg_win = float(pcts[win].mean()) if wins else 0.0
    avg_loss = float(pcts[~win].mean()) if wins < n else 0.0
    gp = float(dols[win].sum()) if wins else 0.0
    gl = abs(float(dols[~win].sum())) if wins < n else 1e-8

    peak, max_dd = equity[0], 0.0
    for v in equity:
        peak = max(peak, v)
        max_dd = min(max_dd, (v - peak) / peak)

    mean_r, std_r = float(pcts.mean()), float(pcts.std()) if n > 1 else 1e-8
    cycle = float(bars.mean()) + 15
    tpy = 105120.0 / max(cycle, 1)
    downside = pcts[pcts < 0]
    dstd = float(np.sqrt(np.mean(downside ** 2))) if len(downside) else 1e-8

    by_reason = {}
    for t in trades:
        r = t.get("reason", "?")
        by_reason.setdefault(r, []).append(t[pk])
    reason_stats = {r: {"n": len(v), "avg_pct": round(float(np.mean(v)), 4)}
                    for r, v in sorted(by_reason.items())}
    longs = [t[pk] for t in trades if t["direction"] == "LONG"]
    shorts = [t[pk] for t in trades if t["direction"] == "SHORT"]

    return {
        "n_trades": n,
        "win_rate_pct": round(wins / n * 100, 2),
        "expectancy_pct": round(mean_r, 4),
        "avg_win_pct": round(avg_win, 4),
        "avg_loss_pct": round(avg_loss, 4),
        "win_loss_ratio": round(abs(avg_win / avg_loss), 3) if avg_loss else None,
        "profit_factor": round(gp / gl, 3),
        "total_pnl_dollar": round(float(dols.sum()), 2),
        "final_equity": round(float(equity[-1]), 2),
        "max_drawdown_pct": round(max_dd * 100, 3),
        "sharpe": round((mean_r / std_r) * math.sqrt(tpy), 2) if std_r > 1e-8 else 0.0,
        "sortino": round((mean_r / dstd) * math.sqrt(tpy), 2) if dstd > 1e-8 else 0.0,
        "avg_bars_held": round(float(bars.mean()), 1),
        "long_n": len(longs), "long_avg_pct": round(float(np.mean(longs)), 4) if longs else None,
        "short_n": len(shorts), "short_avg_pct": round(float(np.mean(shorts)), 4) if shorts else None,
        "exit_reasons": reason_stats,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="UTC date, e.g. 2025-10-01")
    ap.add_argument("--end", required=True)
    ap.add_argument("--versions", default="v4.15,v4.16")
    ap.add_argument("--fee-bps-per-side", type=float, default=4.5,
                    help="taker fee per leg per side in bps (default 4.5 = 0.045%%)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--parity", type=int, default=0,
                    help="verify N random bars against the live per-bar path, then exit")
    args = ap.parse_args()

    start_ms = int(datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc).timestamp() * 1000)
    # Pull warm-up candles before the window so trading starts at --start
    warm_ms = start_ms - (WARMUP + 800) * 60_000
    label = args.label or f"{args.start}_{args.end}"
    out_dir = ROOT / "reports" / "eval" / label
    out_dir.mkdir(parents=True, exist_ok=True)

    model = XGBClassifier()
    model.load_model(str(MODEL_DIR / "model.json"))
    feature_names = json.loads((MODEL_DIR / "feature_names.json").read_text())

    print(f"Window {args.start} → {args.end} (+{WARMUP + 800} warm-up bars)")
    btc = fetch_klines("BTCUSDT", warm_ms, end_ms)
    eth = fetch_klines("ETHUSDT", warm_ms, end_ms)

    proba_cache = CACHE_DIR / f"probas_{warm_ms}_{end_ms}.parquet"
    if proba_cache.exists():
        cached = pd.read_parquet(proba_cache)
        print(f"Probas: {len(cached):,} bars (cached)")
    else:
        t0 = time.time()
        cached = compute_probas(btc, eth, model, feature_names)
        cached.to_parquet(proba_cache)
        print(f"Probas: {len(cached):,} bars computed in {time.time() - t0:.1f}s "
              f"({int((~cached['valid']).sum())} invalid)")

    if args.parity:
        worst = parity_check(btc, eth, cached, model, feature_names, args.parity)
        print(f"PARITY: max |proba_vectorized - proba_live_path| over {args.parity} bars = {worst:.2e}")
        assert worst < 1e-4, "Parity check FAILED — vectorized path diverges from live path"
        print("PARITY OK (< 1e-4)")
        return

    results = {}
    for version in args.versions.split(","):
        version = version.strip()
        t0 = time.time()
        trades, equity, n_bars = replay(version, cached, args.fee_bps_per_side)
        gross = compute_metrics(trades, equity, net=False)
        # Equity curve already includes fees, so net metrics reuse it
        net = compute_metrics(trades, equity, net=True)
        results[version] = {"gross": gross, "net": net, "bars_replayed": n_bars}
        pd.DataFrame(trades).to_csv(out_dir / f"trades_{version.replace('.', '_')}.csv", index=False)
        print(f"\n== {version} ==  ({n_bars:,} bars, {time.time() - t0:.1f}s replay)")
        for tag, m in (("GROSS", gross), ("NET  ", net)):
            if m["n_trades"] == 0:
                print(f"  {tag}: no trades")
                continue
            print(f"  {tag}: n={m['n_trades']} win%={m['win_rate_pct']} exp={m['expectancy_pct']}% "
                  f"avgW={m['avg_win_pct']}% avgL={m['avg_loss_pct']}% W/L={m['win_loss_ratio']} "
                  f"PF={m['profit_factor']} PnL=${m['total_pnl_dollar']} DD={m['max_drawdown_pct']}% "
                  f"Sharpe={m['sharpe']}")

    meta = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "window": {"start": args.start, "end": args.end},
        "fee_bps_per_side": args.fee_bps_per_side,
        "fills_per_round_trip": 4,
        "starting_balance": STARTING_BALANCE,
        "results": results,
    }
    (out_dir / "scorecard.json").write_text(json.dumps(meta, indent=2))
    print(f"\nArtifacts → {out_dir}")


if __name__ == "__main__":
    main()
