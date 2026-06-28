#!/usr/bin/env python3
"""
V4.14 Classifier Backtest - Optimised for Higher Sharpe

Improvements over V4.13 based on Optuna correlation analysis and equity curve
diagnosis:

1. MODEL: Revert to depth 3 (corr -0.408 with Sharpe), more estimators (+0.382),
   slower learning rate, moderate gamma, stronger regularisation
2. STRATEGY: Higher entry threshold, longer min_hold (+0.375), longer cooldown
   (+0.362) to reduce over-trading (V4.13 had 41 trades/day)
3. NEW - DRAWDOWN CIRCUIT BREAKER: Two-pass approach — after generating positions,
   compute rolling PnL and force neutral during sustained drawdown periods.
   This targets the Aug-Sept drawdown that took V4.13 from 1.6x to 0.4x equity.

Usage:
    python scripts/run_v4_14_classifier.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor

from src.features import build_features
from src.models import (
    evaluate_classification,
    get_feature_importance_grouped,
    select_features_by_importance,
)
from src.backtest import (
    future_k_bar_return,
    pnl_from_positions,
    ratio_returns_1m,
    summarize_performance,
)
from src.utils import as_dtindex
from src.utils.trade_excel_export import create_trading_decisions_excel


# ============================================================================
# V4.14 PARAMETERS
# ============================================================================

K_AHEAD = 10
MAX_FEATURES = 50

# XGBClassifier hyperparameters — tuned from Optuna correlation analysis
XGB_PARAMS = {
    "n_estimators": 500,                    # was 350 (corr +0.382: more trees = better)
    "max_depth": 3,                         # was 4 (corr -0.408: shallower = better Sharpe)
    "learning_rate": 0.04,                  # was 0.073 (slower to match more trees)
    "min_child_weight": 7,                  # was 5 (more conservative splitting)
    "subsample": 0.75,                      # was 0.713 (corr +0.310: slightly higher)
    "colsample_bytree": 0.5589686229585592, # unchanged from Trial 84
    "gamma": 2.5,                           # was 1.5 (moderate pruning; top trials had 3.4-4.4)
    "reg_alpha": 0.05828398920416312,       # unchanged from Trial 84
    "reg_lambda": 0.5,                      # was 0.236 (stronger L2 regularisation)
    "scale_pos_weight": 1.0,               # keep neutral (V4.13 fix that worked)
}

# Strategy parameters — less frequent, higher conviction trades
ENTRY_THRESHOLD = 0.525   # was 0.52 (more selective entries)
EXIT_THRESHOLD = 0.51     # was 0.505 (wider hysteresis band to let trades develop)
COOLDOWN = 15             # was 8 (corr +0.362: longer = better)
MIN_HOLD = 25             # was 15 (corr +0.375: longer = better)

# Circuit breaker parameters
CB_LOOKBACK = 500         # Rolling window in bars (~8.3 hours) to measure regime
CB_THRESHOLD = -0.03      # If rolling PnL < -3%, force neutral (regime is adverse)

# Report directory
DATA_PATH = "data/processed/btc_eth_ratio_1m.parquet"
REPORT_DIR = Path("reports/v4.14/xgboost")

# V4.13 baseline for comparison
V413_BASELINE = {
    "Sharpe": 2.6305,
    "Sortino": 2.1454,
    "TotalPnL": 153.72,
    "MaxDrawdown": -129.96,
    "TradeCount": 2981,
    "LongTime": 25.9,
    "ShortTime": 17.7,
    "NeutralTime": 56.4,
}


# ============================================================================
# HYSTERESIS POSITION GENERATION
# ============================================================================

def generate_positions_classifier(
    proba_up: np.ndarray,
    decision_times: pd.DatetimeIndex,
    entry_threshold: float = ENTRY_THRESHOLD,
    exit_threshold: float = EXIT_THRESHOLD,
    min_hold: int = MIN_HOLD,
    cooldown: int = COOLDOWN,
) -> pd.Series:
    """
    Generate positions using hysteresis logic with entry/exit thresholds.
    """
    pos = np.zeros(len(proba_up), dtype=int)
    last_change = -10_000

    for i in range(len(proba_up)):
        prob = proba_up[i]
        prev = pos[i - 1] if i > 0 else 0
        desired = prev
        can_change = (i - last_change) >= min_hold

        if prev == 0:  # NEUTRAL -> enter only on strong signal
            if prob >= entry_threshold:
                desired = 1
            elif prob <= (1.0 - entry_threshold):
                desired = -1

        elif prev == 1:  # LONG -> exit when signal weakens
            if can_change:
                if prob <= (1.0 - entry_threshold):
                    desired = -1    # Flip to SHORT
                elif prob < exit_threshold:
                    desired = 0     # Exit to NEUTRAL

        elif prev == -1:  # SHORT -> exit when signal weakens
            if can_change:
                if prob >= entry_threshold:
                    desired = 1     # Flip to LONG
                elif prob > (1.0 - exit_threshold):
                    desired = 0     # Exit to NEUTRAL

        # Cooldown
        if desired != prev and (i - last_change) <= cooldown:
            desired = prev

        pos[i] = desired
        if desired != prev:
            last_change = i

    return pd.Series(pos, index=decision_times, name="position")


# ============================================================================
# CIRCUIT BREAKER (Two-Pass Drawdown Filter)
# ============================================================================

def apply_circuit_breaker(
    positions: pd.Series,
    fwd_returns: pd.Series,
    lookback: int = CB_LOOKBACK,
    threshold: float = CB_THRESHOLD,
) -> pd.Series:
    """
    Two-pass drawdown circuit breaker.

    Pass 1: Compute rolling PnL using the raw positions.
    Pass 2: Zero out positions during periods where rolling PnL < threshold.

    This forces the strategy to go neutral during sustained adverse regimes,
    preventing the deep drawdowns seen in V4.13 (Aug-Sept period).
    """
    # Compute bar-level PnL from positions
    aligned_returns = fwd_returns.reindex(positions.index).fillna(0.0)
    bar_pnl = positions * aligned_returns

    # Rolling sum over lookback window
    rolling_pnl = bar_pnl.rolling(lookback, min_periods=lookback // 2).sum()

    # Create circuit breaker mask: True = regime is adverse, go neutral
    cb_active = rolling_pnl < threshold

    # Apply: zero out positions where CB is active
    filtered_positions = positions.copy()
    filtered_positions[cb_active] = 0

    return filtered_positions, cb_active


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 80)
    print("V4.14 CLASSIFIER BACKTEST - OPTIMISED FOR HIGHER SHARPE")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data: {DATA_PATH}")
    print(f"Reports: {REPORT_DIR}")
    print()
    print("  KEY CHANGES FROM V4.13:")
    print("    max_depth:        4     -> 3 (shallower = better Sharpe, corr -0.408)")
    print("    n_estimators:     350   -> 500 (more trees, corr +0.382)")
    print("    learning_rate:    0.073 -> 0.04 (slower for more trees)")
    print("    min_child_weight: 5     -> 7 (more conservative)")
    print("    gamma:            1.5   -> 2.5 (moderate pruning)")
    print("    reg_lambda:       0.236 -> 0.5 (stronger L2)")
    print("    entry_threshold:  0.52  -> 0.525 (more selective)")
    print("    exit_threshold:   0.505 -> 0.51 (wider hysteresis)")
    print("    min_hold:         15    -> 25 (longer holds, corr +0.375)")
    print("    cooldown:         8     -> 15 (longer cooldown, corr +0.362)")
    print("    NEW: Drawdown circuit breaker (500-bar lookback, -3% threshold)")
    print()

    # ------------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------------
    print("=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)

    df = pd.read_parquet(DATA_PATH)
    X, y_bps, kept_idx, meta = build_features(df, k_ahead=K_AHEAD, label="bps")
    y_bps = y_bps.astype("float32")

    time_clean = as_dtindex(meta.get("time_clean", X.index))
    ratio_clean = df.loc[kept_idx, "R"].reset_index(drop=True)

    print(f"  Data loaded: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"  Time range: {time_clean[0]} -> {time_clean[-1]}")
    print()

    # ------------------------------------------------------------------
    # 2. FEATURE SELECTION
    # ------------------------------------------------------------------
    print("=" * 80)
    print("STEP 2: FEATURE SELECTION")
    print("=" * 80)

    def make_simple_reg():
        return XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
        )

    X_selected, feature_importance = select_features_by_importance(
        X,
        y_bps,
        make_model=make_simple_reg,
        max_features=MAX_FEATURES,
        verbose=True,
    )
    print(f"  Features: {X.shape[1]} -> {X_selected.shape[1]}")
    print()

    # ------------------------------------------------------------------
    # 3. TRAIN XGBCLASSIFIER WITH WALK-FORWARD CV
    # ------------------------------------------------------------------
    print("=" * 80)
    print("STEP 3: TRAINING XGBCLASSIFIER (Walk-Forward CV)")
    print("=" * 80)
    print(f"  XGB Params:")
    for k, v in XGB_PARAMS.items():
        print(f"    {k}: {v}")
    print(f"  Strategy Params:")
    print(f"    entry_threshold: {ENTRY_THRESHOLD}")
    print(f"    exit_threshold: {EXIT_THRESHOLD}")
    print(f"    cooldown: {COOLDOWN}")
    print(f"    min_hold: {MIN_HOLD}")
    print(f"  Circuit Breaker:")
    print(f"    lookback: {CB_LOOKBACK} bars")
    print(f"    threshold: {CB_THRESHOLD}")
    print()

    def make_cls():
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            early_stopping_rounds=30,
            n_jobs=1,
            random_state=42,
            tree_method="hist",
            **XGB_PARAMS,
        )

    proba_up, ys, cls_metrics, te_idx, _ = evaluate_classification(
        make_cls, X_selected, y_bps
    )

    print(f"\n  Model trained:")
    print(f"    AUC: {cls_metrics.get('AUC', 0):.4f}")
    print(f"    Test samples: {len(proba_up):,}")
    print(f"    Proba mean: {proba_up.mean():.4f}")
    print(f"    Proba std: {proba_up.std():.4f}")
    print(f"    Pct above entry ({ENTRY_THRESHOLD:.4f}): "
          f"{(proba_up >= ENTRY_THRESHOLD).mean() * 100:.1f}%")
    print(f"    Pct below entry ({1 - ENTRY_THRESHOLD:.4f}): "
          f"{(proba_up <= (1 - ENTRY_THRESHOLD)).mean() * 100:.1f}%")
    print()

    # ------------------------------------------------------------------
    # 4. GENERATE POSITIONS (Raw)
    # ------------------------------------------------------------------
    print("=" * 80)
    print("STEP 4: GENERATING POSITIONS (Hysteresis Logic)")
    print("=" * 80)

    decision_times = time_clean[np.array(te_idx, dtype=int)]
    raw_positions = generate_positions_classifier(
        proba_up, decision_times,
        entry_threshold=ENTRY_THRESHOLD,
        exit_threshold=EXIT_THRESHOLD,
        min_hold=MIN_HOLD,
        cooldown=COOLDOWN,
    )

    # Raw position diagnostics
    raw_switches = (raw_positions.diff() != 0).sum()
    raw_long = (raw_positions == 1).sum() / len(raw_positions) * 100
    raw_short = (raw_positions == -1).sum() / len(raw_positions) * 100
    raw_neutral = (raw_positions == 0).sum() / len(raw_positions) * 100

    print(f"  Raw Position Summary (before circuit breaker):")
    print(f"    Total bars: {len(raw_positions):,}")
    print(f"    Switches: {raw_switches}")
    print(f"    Long: {raw_long:.1f}% | Short: {raw_short:.1f}% | Neutral: {raw_neutral:.1f}%")
    print()

    # ------------------------------------------------------------------
    # 5. APPLY CIRCUIT BREAKER
    # ------------------------------------------------------------------
    print("=" * 80)
    print("STEP 5: APPLYING DRAWDOWN CIRCUIT BREAKER")
    print("=" * 80)

    vol_series = ratio_returns_1m(ratio_clean)
    vol_series.index = time_clean

    fwd_log = future_k_bar_return(ratio_clean, k=K_AHEAD)
    fwd_log.index = time_clean

    positions, cb_active = apply_circuit_breaker(
        raw_positions, fwd_log,
        lookback=CB_LOOKBACK,
        threshold=CB_THRESHOLD,
    )

    cb_bars = cb_active.sum()
    cb_pct = cb_bars / len(positions) * 100

    # Final position diagnostics
    switches = (positions.diff() != 0).sum()
    long_pct = (positions == 1).sum() / len(positions) * 100
    short_pct = (positions == -1).sum() / len(positions) * 100
    neutral_pct = (positions == 0).sum() / len(positions) * 100

    print(f"  Circuit breaker activated: {cb_bars:,} bars ({cb_pct:.1f}%)")
    print(f"  Positions zeroed by CB: {(raw_positions != 0)[cb_active].sum():,}")
    print()
    print(f"  Final Position Summary (after circuit breaker):")
    print(f"    Switches: {switches}")
    print(f"    Long: {long_pct:.1f}% | Short: {short_pct:.1f}% | Neutral: {neutral_pct:.1f}%")
    n_days = len(positions) / (60 * 24)
    print(f"    Trades/day: {switches / n_days:.2f}")
    print()

    # ------------------------------------------------------------------
    # 6. CALCULATE PNL AND METRICS
    # ------------------------------------------------------------------
    print("=" * 80)
    print("STEP 6: CALCULATING PNL AND METRICS")
    print("=" * 80)

    pnl_components = pnl_from_positions(positions, fwd_log, vol_series)
    metrics = summarize_performance(pnl_components, positions, fwd_log)

    # Also compute raw (no CB) metrics for comparison
    raw_pnl_components = pnl_from_positions(raw_positions, fwd_log, vol_series)
    raw_metrics = summarize_performance(raw_pnl_components, raw_positions, fwd_log)

    print(f"\n  Performance (with circuit breaker):")
    print(f"    Sharpe:        {metrics['Sharpe']:.4f}")
    print(f"    Sortino:       {metrics['Sortino']:.4f}")
    print(f"    Total PnL:     {metrics['TotalPnL'] * 100:.2f}%")
    print(f"    Max Drawdown:  {metrics['MaxDrawdown'] * 100:.2f}%")
    print(f"    Trade Count:   {metrics['TradeCount']}")
    print(f"    Hit Rate:      {metrics['HitRate']:.1%}")
    print(f"    Hit Rate Long: {metrics['HitRateLong']:.1%}")
    print(f"    Hit Rate Short:{metrics['HitRateShort']:.1%}")
    print(f"    Long Time:     {metrics.get('LongTime%', 0):.1f}%")
    print(f"    Short Time:    {metrics.get('ShortTime%', 0):.1f}%")
    print(f"    Neutral Time:  {metrics.get('NeutralTime%', 0):.1f}%")
    print()
    print(f"  Performance (WITHOUT circuit breaker):")
    print(f"    Sharpe:        {raw_metrics['Sharpe']:.4f}")
    print(f"    Total PnL:     {raw_metrics['TotalPnL'] * 100:.2f}%")
    print(f"    Max Drawdown:  {raw_metrics['MaxDrawdown'] * 100:.2f}%")
    print(f"    CB Impact:     Sharpe {metrics['Sharpe'] - raw_metrics['Sharpe']:+.4f}, "
          f"DD {(metrics['MaxDrawdown'] - raw_metrics['MaxDrawdown']) * 100:+.2f}%")
    print()

    # ------------------------------------------------------------------
    # 7. SAVE REPORTS
    # ------------------------------------------------------------------
    print("=" * 80)
    print("STEP 7: SAVING REPORTS")
    print("=" * 80)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Equity Curve (with and without CB overlay) ---
    pnl = pnl_components["pnl"]
    equity = (1.0 + pnl).cumprod()

    raw_pnl = raw_pnl_components["pnl"]
    raw_equity = (1.0 + raw_pnl).cumprod()

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(raw_equity.index, raw_equity.values, linewidth=1.2, color="#BBBBBB",
            alpha=0.6, label=f"Without CB (Sharpe {raw_metrics['Sharpe']:.2f})")
    ax.plot(equity.index, equity.values, linewidth=2, color="#8E44AD",
            label=f"With CB (Sharpe {metrics['Sharpe']:.2f})")

    # Shade CB active periods
    cb_active_reindexed = cb_active.reindex(equity.index).fillna(False)
    if cb_active_reindexed.any():
        for start, end in _get_contiguous_ranges(cb_active_reindexed):
            ax.axvspan(start, end, alpha=0.08, color='red')

    ax.set_title(
        f"V4.14 CLASSIFIER (Optimised + Circuit Breaker)\n"
        f"Sharpe {metrics['Sharpe']:.2f} | PnL {metrics['TotalPnL'] * 100:.1f}% | "
        f"Trades {metrics['TradeCount']} | CB active {cb_pct:.1f}%",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Equity (normalized to 1.0)", fontsize=12)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "equity_curve.png", dpi=160)
    plt.close()
    print(f"  Equity curve saved (with CB overlay)")

    # --- Equity PnL CSV ---
    pd.DataFrame({"pnl": pnl}).to_csv(REPORT_DIR / "equity_curve_pnl.csv")
    print(f"  Equity PnL CSV saved")

    # --- Trade Breakdown ---
    pd.DataFrame(
        {
            "pnl": pnl,
            "position": positions.reindex(pnl.index),
            "cb_active": cb_active.reindex(pnl.index),
        }
    ).to_csv(REPORT_DIR / "trade_breakdown.csv")
    print(f"  Trade breakdown saved")

    # --- Metrics Summary ---
    metrics_row = {
        "model": "xgboost_v4.14_classifier_optimised",
        "AUC": cls_metrics.get("AUC", 0),
        **metrics,
        "entry_threshold": ENTRY_THRESHOLD,
        "exit_threshold": EXIT_THRESHOLD,
        "cooldown": COOLDOWN,
        "min_hold": MIN_HOLD,
        "cb_lookback": CB_LOOKBACK,
        "cb_threshold": CB_THRESHOLD,
        "cb_active_pct": cb_pct,
        "raw_sharpe_no_cb": raw_metrics["Sharpe"],
        "raw_dd_no_cb": raw_metrics["MaxDrawdown"],
    }
    metrics_df = pd.DataFrame([metrics_row])
    metrics_df.to_csv(REPORT_DIR / "metrics_summary.csv", index=False)
    print(f"  Metrics summary saved")

    # --- Feature Importance ---
    print("\n  Computing feature importance...")
    cls_full = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
        **XGB_PARAMS,
    )
    y_dir = (y_bps > 0).astype(int)
    cls_full.fit(X_selected, y_dir)

    grouped_fi, color_map = get_feature_importance_grouped(
        cls_full, X_selected.columns
    )

    if not grouped_fi.empty:
        print("\n  TOP 10 FEATURES:")
        for i, (_, row) in enumerate(grouped_fi.head(10).iterrows(), 1):
            feat_name = row.get("display_name", row.get("base_family", "Unknown"))
            print(f"    {i}. {feat_name}: {row['importance']:.4f} [{row['group']}]")

        top_features = grouped_fi.head(20).copy()
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = [color_map.get(group, "#bdbdbd") for group in top_features["group"]]
        y_pos = np.arange(len(top_features))
        ax.barh(
            y_pos,
            top_features["importance"],
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_yticks(y_pos)
        labels = (
            top_features["display_name"]
            if "display_name" in top_features.columns
            else top_features.index
        )
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Importance (Gain)", fontsize=11)
        ax.set_title(
            "V4.14 Classifier - Feature Importance by Asset Type",
            fontsize=13,
            fontweight="bold",
        )
        ax.invert_yaxis()

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=color_map[group], label=group, alpha=0.8)
            for group in sorted(color_map.keys())
            if group in top_features["group"].values
        ]
        ax.legend(
            handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9
        )

        plt.tight_layout()
        plt.savefig(REPORT_DIR / "feature_importance.png", dpi=160)
        plt.close()
        print(f"  Feature importance saved")

    # --- Trading Decisions Excel ---
    print("\n  Generating trading decisions Excel...")

    btc_prices = df.loc[kept_idx, "btc_close"].reset_index(drop=True)
    btc_prices.index = time_clean
    eth_prices = df.loc[kept_idx, "eth_close"].reset_index(drop=True)
    eth_prices.index = time_clean
    ratio_prices = ratio_clean.copy()
    ratio_prices.index = time_clean

    proba_series = pd.Series(proba_up, index=decision_times, name="proba_up")
    forecast_series = pd.Series(
        (proba_up - 0.5) / 100.0, index=decision_times, name="forecast"
    )
    confidence_series = pd.Series(
        np.abs(proba_up - 0.5) * 2.0, index=decision_times, name="confidence"
    )
    tau_series = pd.Series(
        np.full(len(decision_times), (ENTRY_THRESHOLD - 0.5) / 100.0),
        index=decision_times,
        name="tau",
    )

    create_trading_decisions_excel(
        positions=positions,
        pnl_components=pnl_components,
        forecast_series=forecast_series,
        confidence_series=confidence_series,
        tau_series=tau_series,
        btc_prices=btc_prices,
        eth_prices=eth_prices,
        ratio_prices=ratio_prices,
        output_path=REPORT_DIR / "trading_decisions.xlsx",
        strategy_params={
            "entry_threshold": ENTRY_THRESHOLD,
            "exit_threshold": EXIT_THRESHOLD,
            "cooldown": COOLDOWN,
            "min_hold": MIN_HOLD,
            "cb_lookback": CB_LOOKBACK,
            "cb_threshold": CB_THRESHOLD,
            "model": "XGBClassifier",
            "approach": "Probability hysteresis + drawdown circuit breaker (V4.14)",
        },
    )

    # ------------------------------------------------------------------
    # 8. COMPARISON WITH V4.13
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("V4.14 vs V4.13 COMPARISON")
    print("=" * 80)
    print()
    print(f"  {'Metric':<20} {'V4.13':>12} {'V4.14':>12} {'Delta':>12}")
    print(f"  {'-'*56}")

    v414_sharpe = metrics['Sharpe']
    v414_pnl = metrics['TotalPnL'] * 100
    v414_dd = metrics['MaxDrawdown'] * 100
    v414_trades = metrics['TradeCount']
    v414_long = metrics.get('LongTime%', 0)
    v414_short = metrics.get('ShortTime%', 0)
    v414_neutral = metrics.get('NeutralTime%', 0)

    def fmt_delta(new_val, old_val, fmt=".2f"):
        d = new_val - old_val
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:{fmt}}"

    print(f"  {'Sharpe':<20} {V413_BASELINE['Sharpe']:>12.4f} {v414_sharpe:>12.4f} {fmt_delta(v414_sharpe, V413_BASELINE['Sharpe'], '.4f'):>12}")
    print(f"  {'Sortino':<20} {V413_BASELINE['Sortino']:>12.4f} {metrics['Sortino']:>12.4f} {fmt_delta(metrics['Sortino'], V413_BASELINE['Sortino'], '.4f'):>12}")
    print(f"  {'Total PnL %':<20} {V413_BASELINE['TotalPnL']:>11.2f}% {v414_pnl:>11.2f}% {fmt_delta(v414_pnl, V413_BASELINE['TotalPnL']):>12}")
    print(f"  {'Max Drawdown %':<20} {V413_BASELINE['MaxDrawdown']:>11.2f}% {v414_dd:>11.2f}% {fmt_delta(v414_dd, V413_BASELINE['MaxDrawdown']):>12}")
    print(f"  {'Trade Count':<20} {V413_BASELINE['TradeCount']:>12.0f} {v414_trades:>12.0f} {fmt_delta(v414_trades, V413_BASELINE['TradeCount'], '.0f'):>12}")
    print(f"  {'Long Time %':<20} {V413_BASELINE['LongTime']:>11.1f}% {v414_long:>11.1f}% {fmt_delta(v414_long, V413_BASELINE['LongTime'], '.1f'):>12}")
    print(f"  {'Short Time %':<20} {V413_BASELINE['ShortTime']:>11.1f}% {v414_short:>11.1f}% {fmt_delta(v414_short, V413_BASELINE['ShortTime'], '.1f'):>12}")
    print(f"  {'Neutral Time %':<20} {V413_BASELINE['NeutralTime']:>11.1f}% {v414_neutral:>11.1f}% {fmt_delta(v414_neutral, V413_BASELINE['NeutralTime'], '.1f'):>12}")
    print()

    # Assessment
    improvements = []
    regressions = []

    if v414_sharpe > V413_BASELINE['Sharpe']:
        improvements.append(f"Sharpe improved: {V413_BASELINE['Sharpe']:.4f} -> {v414_sharpe:.4f} ({(v414_sharpe/V413_BASELINE['Sharpe']-1)*100:+.1f}%)")
    else:
        regressions.append(f"Sharpe decreased: {V413_BASELINE['Sharpe']:.4f} -> {v414_sharpe:.4f}")

    if abs(v414_dd) < abs(V413_BASELINE['MaxDrawdown']):
        improvements.append(f"Drawdown improved: {V413_BASELINE['MaxDrawdown']:.1f}% -> {v414_dd:.1f}%")
    else:
        regressions.append(f"Drawdown worsened: {V413_BASELINE['MaxDrawdown']:.1f}% -> {v414_dd:.1f}%")

    if v414_trades < V413_BASELINE['TradeCount']:
        improvements.append(f"Trade count reduced: {V413_BASELINE['TradeCount']} -> {v414_trades} (less churning)")
    else:
        regressions.append(f"Trade count increased: {V413_BASELINE['TradeCount']} -> {v414_trades}")

    if improvements:
        print("  IMPROVEMENTS:")
        for imp in improvements:
            print(f"    + {imp}")
    if regressions:
        print("  REGRESSIONS:")
        for reg in regressions:
            print(f"    - {reg}")

    # ------------------------------------------------------------------
    # 9. FINAL SUMMARY
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("V4.14 CLASSIFIER BACKTEST COMPLETE")
    print("=" * 80)
    print()
    print(f"  Final Results:")
    print(f"    Sharpe:        {metrics['Sharpe']:.4f}")
    print(f"    Sortino:       {metrics['Sortino']:.4f}")
    print(f"    Total PnL:     {metrics['TotalPnL'] * 100:.2f}%")
    print(f"    Max Drawdown:  {metrics['MaxDrawdown'] * 100:.2f}%")
    print(f"    Trades:        {metrics['TradeCount']}")
    print(f"    Hit Rate:      {metrics['HitRate']:.1%}")
    print(f"    AUC:           {cls_metrics.get('AUC', 0):.4f}")
    print(f"    Long/Short/Neutral: {metrics.get('LongTime%', 0):.1f}% / "
          f"{metrics.get('ShortTime%', 0):.1f}% / "
          f"{metrics.get('NeutralTime%', 0):.1f}%")
    print(f"    CB Active:     {cb_pct:.1f}%")
    print()
    print(f"  Reports saved to: {REPORT_DIR}/")
    print(f"    - equity_curve.png (with CB overlay)")
    print(f"    - equity_curve_pnl.csv")
    print(f"    - metrics_summary.csv")
    print(f"    - trade_breakdown.csv")
    print(f"    - feature_importance.png")
    print(f"    - trading_decisions.xlsx")
    print()
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


def _get_contiguous_ranges(mask: pd.Series):
    """Helper to get start/end of contiguous True regions for shading."""
    changes = mask.astype(int).diff().fillna(0)
    starts = mask.index[changes == 1]
    ends = mask.index[changes == -1]

    if mask.iloc[0]:
        starts = starts.insert(0, mask.index[0])
    if mask.iloc[-1]:
        ends = ends.append(pd.DatetimeIndex([mask.index[-1]]))

    for s, e in zip(starts, ends):
        yield s, e


if __name__ == "__main__":
    main()

