#!/usr/bin/env python3
"""
V4.17 Candidate vs V4.16 Baseline — Self-Contained Backtest

Uses the production XGBoost model + V414FeatureCalculator to replay
Binance 1m klines (fetched via REST) and compare the two signal generators.

Hypothesis under test:
  Raising min_hold 25→40 and lowering exit_threshold 0.51→0.505 will
  convert low-value signal-change exits into higher-value TP/SL exits,
  improving per-trade PnL and Sharpe without worsening drawdown.

Usage (from project root, with .venv active):
    python scripts/backtest_v417_candidate.py

Outputs:
  - Side-by-side metric table to stdout
  - reports/v417/comparison.csv
  - reports/v417/trades_v416.csv
  - reports/v417/trades_v417.csv
"""

import sys
import time
import json
import math
import logging
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import requests
from xgboost import XGBClassifier

from api.feature_calculator import V414FeatureCalculator, V416SignalGenerator
from api.version_config import get_strategy_config

logging.basicConfig(level=logging.WARNING)  # suppress feature-calc info spam


# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIR   = ROOT / "models" / "v4_14_production"
REPORT_DIR  = ROOT / "reports" / "v417"

# Fetch window: Oct–Dec 2025 out-of-sample period
# Model trained on Feb–Oct 2025, so this is genuinely out-of-sample
FETCH_START_MS = int(datetime(2025, 10, 1, tzinfo=timezone.utc).timestamp() * 1000)
FETCH_END_MS   = int(datetime(2025, 12, 15, tzinfo=timezone.utc).timestamp() * 1000)

BINANCE_REST = "https://api.binance.com/api/v3/klines"
LIMIT_PER_CALL = 1000   # Binance max
STARTING_BALANCE = 1000.0
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list:
    """Fetch 1m klines from Binance REST in paginated chunks."""
    all_candles = []
    cursor = start_ms
    print(f"  Fetching {symbol} klines {datetime.fromtimestamp(start_ms/1000).date()} → "
          f"{datetime.fromtimestamp(end_ms/1000).date()} ...", end="", flush=True)

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": cursor,
            "endTime": end_ms,
            "limit": LIMIT_PER_CALL,
        }
        for attempt in range(3):
            try:
                r = requests.get(BINANCE_REST, params=params, timeout=15)
                r.raise_for_status()
                batch = r.json()
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

        if not batch:
            break

        for row in batch:
            all_candles.append({
                "time": int(row[0]) // 1000,          # ms → s
                "open":  float(row[1]),
                "high":  float(row[2]),
                "low":   float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "taker_buy_volume": float(row[9]),
            })

        last_open_ms = int(batch[-1][0])
        cursor = last_open_ms + 60_000  # advance by 1 minute

        print(".", end="", flush=True)
        if len(batch) < LIMIT_PER_CALL:
            break
        time.sleep(0.15)  # be polite to Binance rate limits

    print(f" {len(all_candles):,} candles")
    return all_candles


# ─────────────────────────────────────────────────────────────────────────────
# Backtest engine
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(cfg_name: str, btc_candles: list, eth_candles: list,
                 model: XGBClassifier, feature_names: list):
    """
    Replay bar-by-bar using V414FeatureCalculator + production model +
    V416SignalGenerator (which handles both v4.16 and v4.17 configs).

    Returns (trades, equity_curve).
    """
    cfg = get_strategy_config(cfg_name)
    sig_gen = V416SignalGenerator(cfg=cfg)

    # Seed feature calculator with first 1000 candles as warm-up
    WARMUP = 1000
    fc = V414FeatureCalculator(feature_names=feature_names, max_history=1500)

    # Align candles by timestamp (inner join)
    btc_by_t = {c["time"]: c for c in btc_candles}
    eth_by_t = {c["time"]: c for c in eth_candles}
    common_times = sorted(set(btc_by_t) & set(eth_by_t))

    if len(common_times) < WARMUP + 100:
        raise RuntimeError(f"Not enough aligned candles: {len(common_times)}")

    # Warm up feature calculator (no trading during warm-up)
    for t in common_times[:WARMUP]:
        fc.add_candle("BTC", btc_by_t[t])
        fc.add_candle("ETH", eth_by_t[t])

    equity_curve = [STARTING_BALANCE]
    features_computed = 0
    features_skipped = 0

    for t in common_times[WARMUP:]:
        fc.add_candle("BTC", btc_by_t[t])
        fc.add_candle("ETH", eth_by_t[t])

        features_df = fc.calculate_features()
        if features_df is None:
            features_skipped += 1
            continue

        proba_up = float(model.predict_proba(features_df)[:, 1][0])
        current_ratio = float(btc_by_t[t]["close"]) / float(eth_by_t[t]["close"])

        sig = sig_gen.update(proba_up, current_ratio, t)
        equity_curve.append(sig_gen.balance)
        features_computed += 1

    return sig_gen.trades, np.array(equity_curve), features_computed, features_skipped


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(trades: list, equity_curve: np.ndarray) -> dict:
    if not trades:
        return {"n_trades": 0}

    pnl_pcts   = np.array([t["pnl_pct"] for t in trades])
    pnl_dollars = np.array([t["pnl_dollar"] for t in trades])
    bars_held  = np.array([t["bars_held"] for t in trades])
    is_win     = pnl_dollars > 0

    n      = len(trades)
    wins   = int(is_win.sum())
    losses = n - wins
    win_rate = wins / n * 100

    avg_win_pct  = float(pnl_pcts[is_win].mean())  if wins   > 0 else 0.0
    avg_loss_pct = float(pnl_pcts[~is_win].mean()) if losses > 0 else 0.0
    rr_ratio     = abs(avg_win_pct / avg_loss_pct)  if avg_loss_pct != 0 else float("inf")

    gross_profit = float(pnl_dollars[is_win].sum())       if wins   > 0 else 0.0
    gross_loss   = abs(float(pnl_dollars[~is_win].sum())) if losses > 0 else 1e-8
    profit_factor = gross_profit / gross_loss

    # Max drawdown from equity curve
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    # Annualised Sharpe (per-trade approximation)
    mean_ret = float(pnl_pcts.mean())
    std_ret  = float(pnl_pcts.std()) if n > 1 else 1e-8
    avg_hold = float(bars_held.mean())
    cycle    = avg_hold + 15  # hold + cooldown
    trades_per_year = 105120.0 / max(cycle, 1)
    sharpe = (mean_ret / std_ret) * math.sqrt(trades_per_year) if std_ret > 1e-8 else 0.0

    downside     = pnl_pcts[pnl_pcts < 0]
    downside_std = float(np.sqrt(np.mean(downside ** 2))) if len(downside) > 0 else 1e-8
    sortino = (mean_ret / downside_std) * math.sqrt(trades_per_year) if downside_std > 1e-8 else 0.0

    reasons: dict = {}
    for t in trades:
        r = t.get("reason", "Unknown")
        reasons[r] = reasons.get(r, 0) + 1

    # Direction breakdown
    long_trades  = [t for t in trades if t.get("direction") == "LONG"]
    short_trades = [t for t in trades if t.get("direction") == "SHORT"]
    long_pnl  = sum(t["pnl_pct"] for t in long_trades)
    short_pnl = sum(t["pnl_pct"] for t in short_trades)

    return {
        "n_trades":        n,
        "wins":            wins,
        "losses":          losses,
        "win_rate":        round(win_rate, 2),
        "avg_pnl_pct":     round(mean_ret, 4),
        "avg_win_pct":     round(avg_win_pct, 4),
        "avg_loss_pct":    round(avg_loss_pct, 4),
        "rr_ratio":        round(rr_ratio, 4),
        "profit_factor":   round(profit_factor, 4),
        "sharpe":          round(sharpe, 4),
        "sortino":         round(sortino, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "final_balance":   round(float(equity_curve[-1]), 2),
        "total_pnl_pct":   round((float(equity_curve[-1]) - STARTING_BALANCE) / STARTING_BALANCE * 100, 4),
        "avg_hold_bars":   round(float(bars_held.mean()), 1),
        "exit_reasons":    reasons,
        "long_count":      len(long_trades),
        "short_count":     len(short_trades),
        "long_pnl_pct":    round(long_pnl, 4),
        "short_pnl_pct":   round(short_pnl, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap helpers
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(trades: list, metric_fn, n_iter: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED):
    rng = np.random.RandomState(seed)
    n = len(trades)
    if n < 10:
        return {"mean": 0, "ci_lower": 0, "ci_upper": 0}
    vals = []
    for _ in range(n_iter):
        idx = rng.randint(0, n, size=n)
        sample = [trades[i] for i in idx]
        vals.append(metric_fn(sample))
    vals = np.array(vals)
    return {
        "mean":     round(float(np.mean(vals)), 4),
        "ci_lower": round(float(np.percentile(vals, 2.5)), 4),
        "ci_upper": round(float(np.percentile(vals, 97.5)), 4),
    }

def _pf(trades):
    wins = sum(t["pnl_dollar"] for t in trades if t["pnl_dollar"] > 0)
    loss = abs(sum(t["pnl_dollar"] for t in trades if t["pnl_dollar"] < 0))
    return wins / max(loss, 1e-8)

def _rr(trades):
    w = [t["pnl_pct"] for t in trades if t["pnl_pct"] > 0]
    l = [t["pnl_pct"] for t in trades if t["pnl_pct"] < 0]
    return abs(np.mean(w) / np.mean(l)) if w and l else 0.0

def _wr(trades):
    return sum(1 for t in trades if t["pnl_dollar"] > 0) / max(len(trades), 1) * 100

def _sharpe(trades):
    pnls = np.array([t["pnl_pct"] for t in trades])
    bars = np.array([t["bars_held"] for t in trades])
    if len(pnls) < 2:
        return 0.0
    std = pnls.std()
    if std < 1e-8:
        return 0.0
    cycle = bars.mean() + 15
    tpy = 105120.0 / max(cycle, 1)
    return float(pnls.mean() / std * math.sqrt(tpy))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("V4.17 CANDIDATE vs V4.16 BASELINE — BACKTEST")
    print("Oct–Dec 2025 out-of-sample Binance 1m data")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # ── Load model ──────────────────────────────────────────────────────────
    print("\nStep 1: Loading production model...")
    model = XGBClassifier()
    model.load_model(str(MODEL_DIR / "model.json"))
    with open(MODEL_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    print(f"  {len(feature_names)} features, model loaded from {MODEL_DIR.name}")

    # ── Fetch data ──────────────────────────────────────────────────────────
    print("\nStep 2: Fetching Binance 1m klines (Oct–Dec 2025)...")
    btc_candles = fetch_klines("BTCUSDT", FETCH_START_MS, FETCH_END_MS)
    eth_candles = fetch_klines("ETHUSDT", FETCH_START_MS, FETCH_END_MS)
    print(f"  BTC: {len(btc_candles):,} candles | ETH: {len(eth_candles):,} candles")

    # ── Run backtests ───────────────────────────────────────────────────────
    print("\nStep 3: Running bar-by-bar backtests...")

    print("  [v4.16 baseline]")
    trades_416, eq_416, computed_416, skipped_416 = run_backtest(
        "v4.16", btc_candles, eth_candles, model, feature_names
    )
    print(f"    {len(trades_416)} trades | {computed_416:,} bars with features | {skipped_416} skipped")

    print("  [v4.17 candidate]")
    trades_417, eq_417, computed_417, skipped_417 = run_backtest(
        "v4.17", btc_candles, eth_candles, model, feature_names
    )
    print(f"    {len(trades_417)} trades | {computed_417:,} bars with features | {skipped_417} skipped")

    # ── Compute metrics ─────────────────────────────────────────────────────
    print("\nStep 4: Computing metrics...")
    m416 = compute_metrics(trades_416, eq_416)
    m417 = compute_metrics(trades_417, eq_417)

    # ── Side-by-side table ──────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("BASELINE VS CANDIDATE RESULTS")
    print("=" * 80)

    rows = [
        ("n_trades",         "Trade Count"),
        ("wins",             "Wins"),
        ("losses",           "Losses"),
        ("win_rate",         "Win Rate %"),
        ("avg_pnl_pct",      "Avg PnL %"),
        ("avg_win_pct",      "Avg Win %"),
        ("avg_loss_pct",     "Avg Loss %"),
        ("rr_ratio",         "R:R Ratio"),
        ("profit_factor",    "Profit Factor"),
        ("sharpe",           "Sharpe (ann.)"),
        ("sortino",          "Sortino (ann.)"),
        ("max_drawdown_pct", "Max DD %"),
        ("total_pnl_pct",    "Total PnL %"),
        ("final_balance",    "Final Balance $"),
        ("avg_hold_bars",    "Avg Hold (bars)"),
        ("long_count",       "LONG trades"),
        ("short_count",      "SHORT trades"),
        ("long_pnl_pct",     "LONG total PnL %"),
        ("short_pnl_pct",    "SHORT total PnL %"),
    ]

    print(f"\n  {'Metric':<24} {'v4.16 (base)':>14} {'v4.17 (cand)':>14} {'Delta':>12}")
    print("  " + "─" * 68)
    for key, label in rows:
        b = m416.get(key, 0)
        c = m417.get(key, 0)
        delta = c - b
        # Direction: higher is better for most, lower for DD and avg_loss
        marker = ""
        if key in ("max_drawdown_pct", "avg_loss_pct", "losses"):
            marker = " ▼" if delta < 0 else (" ▲" if delta > 0 else "")
            marker = " ✓" if delta < 0 else (" ✗" if delta > 0 else "")
        elif key not in ("n_trades",):
            marker = " ✓" if delta > 0 else (" ✗" if delta < 0 else "")
        print(f"  {label:<24} {b:>14.4f} {c:>14.4f} {delta:>+12.4f}{marker}")

    # Exit reason breakdown
    print(f"\n  EXIT DISTRIBUTION CHANGE")
    print(f"  {'Reason':<22} {'v4.16':>8} {'v4.17':>8} {'Delta':>8}")
    print("  " + "─" * 50)
    all_reasons = sorted(
        set(list(m416.get("exit_reasons", {}).keys()) +
            list(m417.get("exit_reasons", {}).keys()))
    )
    for r in all_reasons:
        b = m416.get("exit_reasons", {}).get(r, 0)
        c = m417.get("exit_reasons", {}).get(r, 0)
        print(f"  {r:<22} {b:>8} {c:>8} {c-b:>+8}")

    # % breakdown
    total_416 = max(m416.get("n_trades", 1), 1)
    total_417 = max(m417.get("n_trades", 1), 1)
    print(f"\n  Exit %:")
    for r in all_reasons:
        b = m416.get("exit_reasons", {}).get(r, 0)
        c = m417.get("exit_reasons", {}).get(r, 0)
        bp = b / total_416 * 100
        cp = c / total_417 * 100
        print(f"  {r:<22} {bp:>7.1f}%  {cp:>7.1f}%  {cp-bp:>+7.1f}pp")

    # ── Bootstrap confidence intervals ──────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"BOOTSTRAP CONFIDENCE INTERVALS ({N_BOOTSTRAP} resamples, 95% CI)")
    print("=" * 80)

    for name, trades in [("v4.16", trades_416), ("v4.17", trades_417)]:
        if not trades:
            continue
        bs_pf = bootstrap_ci(trades, _pf)
        bs_rr = bootstrap_ci(trades, _rr)
        bs_wr = bootstrap_ci(trades, _wr)
        bs_sh = bootstrap_ci(trades, _sharpe)
        print(f"\n  {name}:")
        print(f"    Profit Factor  {bs_pf['mean']:.4f}  [{bs_pf['ci_lower']:.4f}, {bs_pf['ci_upper']:.4f}]")
        print(f"    R:R Ratio      {bs_rr['mean']:.4f}  [{bs_rr['ci_lower']:.4f}, {bs_rr['ci_upper']:.4f}]")
        print(f"    Win Rate       {bs_wr['mean']:.2f}%  [{bs_wr['ci_lower']:.2f}%, {bs_wr['ci_upper']:.2f}%]")
        print(f"    Sharpe (ann.)  {bs_sh['mean']:.4f}  [{bs_sh['ci_lower']:.4f}, {bs_sh['ci_upper']:.4f}]")

    # ── Risk trade-offs summary ──────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("RISK TRADEOFFS")
    print("=" * 80)

    dd_change = m417.get("max_drawdown_pct", 0) - m416.get("max_drawdown_pct", 0)
    sl_416 = m416.get("exit_reasons", {}).get("Stop loss", 0)
    sl_417 = m417.get("exit_reasons", {}).get("Stop loss", 0)
    sc_416 = m416.get("exit_reasons", {}).get("Signal change", 0)
    sc_417 = m417.get("exit_reasons", {}).get("Signal change", 0)
    tp_416 = m416.get("exit_reasons", {}).get("Take profit", 0)
    tp_417 = m417.get("exit_reasons", {}).get("Take profit", 0)

    print(f"\n  Signal-change exits: {sc_416} → {sc_417}  ({sc_417-sc_416:+d})  "
          f"{'✓ reduced' if sc_417 < sc_416 else '✗ increased'}")
    print(f"  Take-profit exits:   {tp_416} → {tp_417}  ({tp_417-tp_416:+d})  "
          f"{'✓ increased' if tp_417 > tp_416 else '✗ decreased'}")
    print(f"  Stop-loss exits:     {sl_416} → {sl_417}  ({sl_417-sl_416:+d})  "
          f"{'✓ acceptable' if sl_417 <= sl_416 * 1.3 else '✗ large rise'}")
    print(f"  Max drawdown:        {m416.get('max_drawdown_pct',0):.3f}% → "
          f"{m417.get('max_drawdown_pct',0):.3f}%  ({dd_change:+.3f}pp)  "
          f"{'✓ improved' if dd_change > 0 else ('✗ worse' if dd_change < -0.5 else '~ neutral')}")
    print(f"  Avg hold:            {m416.get('avg_hold_bars',0):.1f} → "
          f"{m417.get('avg_hold_bars',0):.1f} bars  "
          f"(expected to rise with higher min_hold)")
    print(f"  Trade count:         {m416.get('n_trades',0)} → "
          f"{m417.get('n_trades',0)}  "
          f"(expect similar — min_hold affects duration not entry rate)")

    # ── Recommendation ───────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("RECOMMENDATION")
    print("=" * 80)

    pnl_up   = m417.get("total_pnl_pct", 0) > m416.get("total_pnl_pct", 0)
    sharpe_up = m417.get("sharpe", 0) > m416.get("sharpe", 0)
    dd_ok    = (m417.get("max_drawdown_pct", -99) >= m416.get("max_drawdown_pct", -99) - 0.5)
    sl_ok    = sl_417 <= sl_416 * 1.5  # allow up to 50% more SLs (fewer SC exits means more time in market)
    sc_down  = sc_417 < sc_416

    promote_signal = sum([pnl_up, sharpe_up, dd_ok, sl_ok, sc_down])

    print(f"\n  Criteria check:")
    print(f"    Total PnL improved:        {'✓ YES' if pnl_up else '✗ NO'}")
    print(f"    Sharpe improved:           {'✓ YES' if sharpe_up else '✗ NO'}")
    print(f"    Drawdown within 0.5pp:     {'✓ YES' if dd_ok else '✗ NO'}")
    print(f"    SL exits not up >50%:      {'✓ YES' if sl_ok else '✗ NO'}")
    print(f"    Signal-change exits down:  {'✓ YES' if sc_down else '✗ NO'}")

    if promote_signal >= 4:
        verdict = "PROMOTE — evidence clearly favours v4.17"
    elif promote_signal >= 3:
        verdict = "MARGINAL — monitor 2–3 live sessions before promoting"
    else:
        verdict = "DO NOT PROMOTE — hypothesis not supported"

    print(f"\n  {promote_signal}/5 criteria met → {verdict}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    print(f"\nSaving reports to {REPORT_DIR}...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    import csv, io

    comparison_rows = []
    for key, label in rows:
        comparison_rows.append({
            "metric": label,
            "v4.16": m416.get(key, 0),
            "v4.17": m417.get(key, 0),
            "delta": m417.get(key, 0) - m416.get(key, 0),
        })

    with open(REPORT_DIR / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "v4.16", "v4.17", "delta"])
        w.writeheader()
        w.writerows(comparison_rows)

    if trades_416:
        import json as _json
        with open(REPORT_DIR / "trades_v416.csv", "w", newline="") as f:
            fieldnames = list(trades_416[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(trades_416)

    if trades_417:
        with open(REPORT_DIR / "trades_v417.csv", "w", newline="") as f:
            fieldnames = list(trades_417[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(trades_417)

    print("  comparison.csv, trades_v416.csv, trades_v417.csv saved")
    print(f"\n{'=' * 80}")
    print("BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
