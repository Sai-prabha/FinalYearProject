#!/usr/bin/env python3
"""
V4.16 vs V4.15 Side-by-Side Backtest Comparison

Runs both signal generators bar-by-bar through the historical data to
produce a like-for-like comparison including SL/TP, trailing stops,
position sizing, time filters, and circuit breaker dynamics.

Generates:
  - Side-by-side metrics table (console + CSV)
  - Equity curves (overlaid PNG)
  - Trade-level detail CSVs
  - Bootstrap confidence intervals

Usage:
    python scripts/backtest_v416_comparison.py
"""

import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor

from src.features import build_features
from src.models import evaluate_classification, select_features_by_importance
from src.backtest import future_k_bar_return, ratio_returns_1m
from src.utils import as_dtindex

from api.feature_calculator import V414SignalGenerator, V416SignalGenerator
from api.version_config import V415_CONFIG, V416_CONFIG, get_strategy_config


# ══════════════════════════════════════════════════════════════════════════
# Parameters (shared with v4.14 training)
# ══════════════════════════════════════════════════════════════════════════

K_AHEAD = 10
MAX_FEATURES = 50
DATA_PATH = "data/processed/btc_eth_ratio_1m.parquet"
REPORT_DIR = ROOT / "reports" / "v4.16"

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 3,
    "learning_rate": 0.04,
    "min_child_weight": 7,
    "subsample": 0.75,
    "colsample_bytree": 0.5589686229585592,
    "gamma": 2.5,
    "reg_alpha": 0.05828398920416312,
    "reg_lambda": 0.5,
    "scale_pos_weight": 1.0,
}

# Bootstrap parameters
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def run_signal_generator(sig_gen, proba_up_arr, ratio_arr, timestamps_arr):
    """
    Run a signal generator bar-by-bar and collect trade-level results.

    Returns: (trades_list, equity_curve, signal_records)
    """
    equity_curve = [sig_gen.balance]

    for i in range(len(proba_up_arr)):
        sig = sig_gen.update(
            proba_up=float(proba_up_arr[i]),
            current_ratio=float(ratio_arr[i]),
            timestamp=int(timestamps_arr[i]),
        )
        equity_curve.append(sig_gen.balance)

    return sig_gen.trades, np.array(equity_curve)


def compute_trade_metrics(trades, starting_balance=1000.0):
    """Compute standard metrics from a list of trade dicts."""
    if not trades:
        return {}

    pnl_pcts = np.array([t["pnl_pct"] for t in trades])
    pnl_dollars = np.array([t["pnl_dollar"] for t in trades])
    bars_held = np.array([t["bars_held"] for t in trades])
    is_win = pnl_dollars > 0

    n = len(trades)
    wins = int(is_win.sum())
    losses = n - wins
    win_rate = wins / n * 100

    avg_win_pct = float(pnl_pcts[is_win].mean()) if wins > 0 else 0
    avg_loss_pct = float(pnl_pcts[~is_win].mean()) if losses > 0 else 0
    rr_ratio = abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else float("inf")

    gross_profit = float(pnl_dollars[is_win].sum()) if wins > 0 else 0
    gross_loss = abs(float(pnl_dollars[~is_win].sum())) if losses > 0 else 1e-8
    profit_factor = gross_profit / gross_loss

    # Equity curve from trades
    balance = starting_balance
    peak = balance
    max_dd = 0
    for t in trades:
        balance += t["pnl_dollar"]
        peak = max(peak, balance)
        dd = (balance - peak) / peak
        max_dd = min(max_dd, dd)

    final_balance = balance
    total_pnl_dollar = final_balance - starting_balance

    # Sharpe/Sortino (per-trade)
    mean_ret = float(pnl_pcts.mean())
    std_ret = float(pnl_pcts.std()) if n > 1 else 1e-8
    avg_hold = float(bars_held.mean())
    estimated_cycle = avg_hold + 15
    trades_per_year = 105120.0 / max(estimated_cycle, 1)

    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 1e-8 else 0
    downside = pnl_pcts[pnl_pcts < 0]
    downside_std = float(np.sqrt(np.mean(downside ** 2))) if len(downside) > 0 else 1e-8
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 1e-8 else 0

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        r = t.get("reason", "Unknown")
        reasons[r] = reasons.get(r, 0) + 1

    return {
        "n_trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 2),
        "avg_pnl_pct": round(mean_ret, 4),
        "avg_win_pct": round(avg_win_pct, 4),
        "avg_loss_pct": round(avg_loss_pct, 4),
        "rr_ratio": round(rr_ratio, 4),
        "profit_factor": round(profit_factor, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "final_balance": round(final_balance, 2),
        "total_pnl_dollar": round(total_pnl_dollar, 2),
        "avg_hold_bars": round(avg_hold, 1),
        "exit_reasons": reasons,
    }


def bootstrap_metric(trades, metric_fn, n_iter=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Bootstrap resample trades to get CI for a metric."""
    rng = np.random.RandomState(seed)
    n = len(trades)
    if n < 5:
        return {"mean": 0, "ci_lower": 0, "ci_upper": 0}

    values = []
    for _ in range(n_iter):
        idx = rng.randint(0, n, size=n)
        sample_trades = [trades[i] for i in idx]
        values.append(metric_fn(sample_trades))

    values = np.array(values)
    return {
        "mean": round(float(np.mean(values)), 4),
        "ci_lower": round(float(np.percentile(values, 2.5)), 4),
        "ci_upper": round(float(np.percentile(values, 97.5)), 4),
    }


def _profit_factor_from_trades(trades):
    pnls = [t["pnl_dollar"] for t in trades]
    wins_sum = sum(p for p in pnls if p > 0)
    losses_sum = abs(sum(p for p in pnls if p < 0))
    return wins_sum / max(losses_sum, 1e-8)


def _rr_from_trades(trades):
    pnl_pcts = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnl_pcts if p > 0]
    losses = [p for p in pnl_pcts if p < 0]
    avg_w = np.mean(wins) if wins else 0
    avg_l = abs(np.mean(losses)) if losses else 1e-8
    return avg_w / avg_l


def _win_rate_from_trades(trades):
    wins = sum(1 for t in trades if t["pnl_dollar"] > 0)
    return wins / max(len(trades), 1) * 100


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("V4.16 vs V4.15 BACKTEST COMPARISON")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ── 1. Load data & build features ────────────────────────────────────
    print("Step 1: Loading data and building features...")
    df = pd.read_parquet(DATA_PATH)
    X, y_bps, kept_idx, meta = build_features(df, k_ahead=K_AHEAD, label="bps")
    y_bps = y_bps.astype("float32")

    time_clean = as_dtindex(meta.get("time_clean", X.index))
    ratio_clean = df.loc[kept_idx, "R"].reset_index(drop=True)

    print(f"  {X.shape[0]:,} samples, {X.shape[1]} raw features")

    # ── 2. Feature selection ─────────────────────────────────────────────
    print("\nStep 2: Feature selection...")

    def make_simple_reg():
        return XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            objective="reg:squarederror", n_jobs=-1, random_state=42,
        )

    X_selected, _ = select_features_by_importance(
        X, y_bps, make_model=make_simple_reg, max_features=MAX_FEATURES, verbose=False,
    )
    print(f"  {X_selected.shape[1]} features selected")

    # ── 3. Walk-forward CV to get probabilities ──────────────────────────
    print("\nStep 3: Walk-forward classification (this may take a few minutes)...")

    def make_cls():
        return XGBClassifier(
            objective="binary:logistic", eval_metric="auc",
            early_stopping_rounds=30, n_jobs=1, random_state=42,
            tree_method="hist", **XGB_PARAMS,
        )

    proba_up, ys, cls_metrics, te_idx, _ = evaluate_classification(
        make_cls, X_selected, y_bps
    )
    print(f"  AUC: {cls_metrics.get('AUC', 0):.4f}")
    print(f"  Test samples: {len(proba_up):,}")

    # Align ratio and timestamps with test indices
    te_idx_arr = np.array(te_idx, dtype=int)
    decision_times = time_clean[te_idx_arr]
    ratio_test = ratio_clean.iloc[te_idx_arr].values
    timestamps_test = np.array([int(dt.timestamp()) for dt in decision_times])

    # ── 4. Run both signal generators ────────────────────────────────────
    print("\nStep 4: Running signal generators bar-by-bar...")

    # V4.15
    print("  Running V4.15 (baseline)...")
    v415_strategy = {
        "entry_threshold": V415_CONFIG.entry_threshold,
        "exit_threshold": V415_CONFIG.exit_threshold,
        "min_hold": V415_CONFIG.min_hold,
        "cooldown": V415_CONFIG.cooldown,
        "cb_lookback": V415_CONFIG.cb_lookback,
        "cb_threshold": V415_CONFIG.cb_threshold,
    }
    sig_gen_415 = V414SignalGenerator(**v415_strategy)
    trades_415, equity_415 = run_signal_generator(
        sig_gen_415, proba_up, ratio_test, timestamps_test,
    )
    print(f"    {len(trades_415)} trades")

    # V4.16
    print("  Running V4.16 (improved)...")
    v416_cfg = get_strategy_config("v4.16")
    sig_gen_416 = V416SignalGenerator(cfg=v416_cfg)
    trades_416, equity_416 = run_signal_generator(
        sig_gen_416, proba_up, ratio_test, timestamps_test,
    )
    print(f"    {len(trades_416)} trades")

    # ── 5. Compute metrics ───────────────────────────────────────────────
    print("\nStep 5: Computing metrics...")
    metrics_415 = compute_trade_metrics(trades_415)
    metrics_416 = compute_trade_metrics(trades_416)

    # ── 6. Side-by-side comparison table ─────────────────────────────────
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 80)

    comparison_keys = [
        ("n_trades", "Trade Count"),
        ("wins", "Wins"),
        ("losses", "Losses"),
        ("win_rate", "Win Rate %"),
        ("avg_pnl_pct", "Avg PnL %"),
        ("avg_win_pct", "Avg Win %"),
        ("avg_loss_pct", "Avg Loss %"),
        ("rr_ratio", "R:R Ratio"),
        ("profit_factor", "Profit Factor"),
        ("sharpe", "Sharpe (ann.)"),
        ("sortino", "Sortino (ann.)"),
        ("max_drawdown_pct", "Max DD %"),
        ("final_balance", "Final Balance"),
        ("total_pnl_dollar", "Total PnL $"),
        ("avg_hold_bars", "Avg Hold (bars)"),
    ]

    # Validation targets
    targets = {
        "rr_ratio": (">=", 1.2),
        "win_rate": (">=", 60.0),
        "max_drawdown_pct": (">=", -1.5),  # drawdown is negative, so >= is correct
        "profit_factor": (">=", 1.6),
        "avg_pnl_pct": (">=", 0.05),
    }

    print(f"\n  {'Metric':<22} {'v4.15':>12} {'v4.16':>12} {'Delta':>12} {'Target':>12} {'Pass':>6}")
    print("  " + "-" * 78)

    validation_results = {}

    for key, label in comparison_keys:
        v15 = metrics_415.get(key, 0)
        v16 = metrics_416.get(key, 0)
        delta = v16 - v15

        target_str = ""
        pass_str = ""
        if key in targets:
            op, target_val = targets[key]
            if op == ">=":
                passed = v16 >= target_val
            else:
                passed = v16 <= target_val
            target_str = f"{op}{target_val}"
            pass_str = "YES" if passed else "NO"
            validation_results[key] = passed

        print(f"  {label:<22} {v15:>12.4f} {v16:>12.4f} {delta:>+12.4f} {target_str:>12} {pass_str:>6}")

    # Exit reason breakdown
    print(f"\n  Exit Reasons:")
    all_reasons = set(list(metrics_415.get("exit_reasons", {}).keys()) +
                      list(metrics_416.get("exit_reasons", {}).keys()))
    for reason in sorted(all_reasons):
        v15 = metrics_415.get("exit_reasons", {}).get(reason, 0)
        v16 = metrics_416.get("exit_reasons", {}).get(reason, 0)
        print(f"    {reason:<20} v4.15: {v15:>4}   v4.16: {v16:>4}")

    # ── 7. Bootstrap confidence intervals ────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"BOOTSTRAP CONFIDENCE INTERVALS ({N_BOOTSTRAP} iterations)")
    print("=" * 80)

    if trades_416:
        bs_pf_416 = bootstrap_metric(trades_416, _profit_factor_from_trades)
        bs_rr_416 = bootstrap_metric(trades_416, _rr_from_trades)
        bs_wr_416 = bootstrap_metric(trades_416, _win_rate_from_trades)

        print(f"\n  V4.16 Profit Factor:  {bs_pf_416['mean']:.4f}  "
              f"[{bs_pf_416['ci_lower']:.4f}, {bs_pf_416['ci_upper']:.4f}]")
        print(f"  V4.16 R:R Ratio:      {bs_rr_416['mean']:.4f}  "
              f"[{bs_rr_416['ci_lower']:.4f}, {bs_rr_416['ci_upper']:.4f}]")
        print(f"  V4.16 Win Rate:       {bs_wr_416['mean']:.2f}%  "
              f"[{bs_wr_416['ci_lower']:.2f}%, {bs_wr_416['ci_upper']:.2f}%]")

    if trades_415:
        bs_pf_415 = bootstrap_metric(trades_415, _profit_factor_from_trades)
        bs_rr_415 = bootstrap_metric(trades_415, _rr_from_trades)
        bs_wr_415 = bootstrap_metric(trades_415, _win_rate_from_trades)

        print(f"\n  V4.15 Profit Factor:  {bs_pf_415['mean']:.4f}  "
              f"[{bs_pf_415['ci_lower']:.4f}, {bs_pf_415['ci_upper']:.4f}]")
        print(f"  V4.15 R:R Ratio:      {bs_rr_415['mean']:.4f}  "
              f"[{bs_rr_415['ci_lower']:.4f}, {bs_rr_415['ci_upper']:.4f}]")
        print(f"  V4.15 Win Rate:       {bs_wr_415['mean']:.2f}%  "
              f"[{bs_wr_415['ci_lower']:.2f}%, {bs_wr_415['ci_upper']:.2f}%]")

    # ── 8. Validation summary ────────────────────────────────────────────
    all_passed = all(validation_results.values()) if validation_results else False
    print(f"\n{'=' * 80}")
    print(f"VALIDATION: {'ALL CRITERIA MET ✓' if all_passed else 'SOME CRITERIA NOT MET ✗'}")
    print("=" * 80)
    for key, passed in validation_results.items():
        label = [l for k, l in comparison_keys if k == key][0]
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}: {metrics_416.get(key, 0):.4f} ({targets[key][0]}{targets[key][1]})")

    # ── 9. Save reports ──────────────────────────────────────────────────
    print(f"\nStep 9: Saving reports to {REPORT_DIR}...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Comparison CSV
    comparison_rows = []
    for key, label in comparison_keys:
        comparison_rows.append({
            "metric": label,
            "v4.15": metrics_415.get(key, 0),
            "v4.16": metrics_416.get(key, 0),
            "delta": metrics_416.get(key, 0) - metrics_415.get(key, 0),
        })
    pd.DataFrame(comparison_rows).to_csv(REPORT_DIR / "comparison_report.csv", index=False)
    print("  comparison_report.csv saved")

    # Trade-level CSVs
    if trades_415:
        pd.DataFrame(trades_415).to_csv(REPORT_DIR / "trades_v415_backtest.csv", index=False)
    if trades_416:
        pd.DataFrame(trades_416).to_csv(REPORT_DIR / "trades_v416_backtest.csv", index=False)
    print("  Trade CSVs saved")

    # Equity curve plot
    fig, ax = plt.subplots(figsize=(14, 7))

    # Use bar indices for x-axis since equity curves may have different lengths
    x_415 = np.arange(len(equity_415))
    x_416 = np.arange(len(equity_416))

    ax.plot(x_415, equity_415, linewidth=1.5, color="#2E86C1", alpha=0.7,
            label=f"v4.15 (PF={metrics_415.get('profit_factor', 0):.2f}, "
                  f"WR={metrics_415.get('win_rate', 0):.1f}%)")
    ax.plot(x_416, equity_416, linewidth=2, color="#E74C3C",
            label=f"v4.16 (PF={metrics_416.get('profit_factor', 0):.2f}, "
                  f"WR={metrics_416.get('win_rate', 0):.1f}%)")

    ax.set_title(
        f"V4.16 vs V4.15 Equity Comparison\n"
        f"v4.15: {metrics_415.get('n_trades', 0)} trades, "
        f"R:R={metrics_415.get('rr_ratio', 0):.2f} | "
        f"v4.16: {metrics_416.get('n_trades', 0)} trades, "
        f"R:R={metrics_416.get('rr_ratio', 0):.2f}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Bar Index", fontsize=12)
    ax.set_ylabel("Balance ($)", fontsize=12)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "equity_comparison.png", dpi=160)
    plt.close()
    print("  equity_comparison.png saved")

    print(f"\n{'=' * 80}")
    print("BACKTEST COMPARISON COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

