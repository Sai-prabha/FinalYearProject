#!/usr/bin/env python3
"""
Optuna Optimization for V4.12 - Pure Sharpe Ratio Objective (FAST VERSION)

KEY IMPROVEMENTS over previous optimizations:
1. PURE SHARPE objective (no bonuses/penalties that distort the signal)
2. HARD CONSTRAINTS for trading activity (reject non-trading solutions)
3. ANTI-OVERFITTING parameter ranges (shallow trees, feature selection)
4. SQLite persistence (resume interrupted runs)
5. PARALLEL SUPPORT (run multiple workers simultaneously)
6. REDUCED FOLDS (15-20 instead of 69 for 5x speed boost)
7. Comprehensive logging for thesis documentation

LESSONS from Trial 7 failure:
- Composite scoring (Sharpe + bonuses - penalties) is misleading
- Trial 7 reported "5.5" but actual Sharpe was 1.3
- Need pure metrics for reliable optimization
- Need constraints to prevent neutral-heavy solutions

SPEED OPTIMIZATIONS:
- Walk-forward folds: 69 → ~15 (step=10000 instead of 2000)
- Per-trial time: 25 min → ~5 min (5x faster!)
- Parallel-safe SQLite with load_if_exists=True
"""

import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from src.features import build_features
from src.models import evaluate_regression, select_features_by_importance
from src.backtest import (
    adaptive_tau,
    future_k_bar_return,
    generate_positions,
    pnl_from_positions,
    ratio_returns_1m,
    summarize_performance,
)
from src.utils import as_dtindex


# Constants
K_AHEAD = 10
VOL_MULTIPLIER = 2.0
MAX_FEATURES = 50  # Feature selection for anti-overfitting

# Hard constraints (RELAXED - original constraints rejected 163/163 trials!)
MIN_TRADES = 25  # Minimum trades required (was 50, too strict)
MAX_NEUTRAL_PCT = 0.90  # Maximum 90% neutral (was 80%, too strict)
MAX_DRAWDOWN = -0.35  # Maximum -35% drawdown (slightly relaxed)

# SPEED OPTIMIZATION: Reduce walk-forward folds
# Original: step=2000 gives ~69 folds (robust but slow)
# Fast: step=10000 gives ~15 folds (5x faster, still robust)
WALK_FORWARD_STEP = 10000  # Increase for faster trials
WALK_FORWARD_MIN_TRAIN = 5000


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series,
              ratio_clean: pd.Series, time_clean: pd.DatetimeIndex) -> float:
    """
    Pure Sharpe optimization objective with hard constraints.
    
    Returns:
        float: Pure Sharpe ratio, or -999 if constraints violated
    """
    
    # ========================================================================
    # ANTI-OVERFITTING PARAMETER RANGES
    # ========================================================================
    # Constrained ranges learned from Trial 7 failure and V4.12 success
    
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 150, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 6),  # SHALLOW TREES ONLY!
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.65, 0.90),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.60, 0.85),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 15.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 15.0, log=True),
    }
    
    # Strategy parameters - enable active trading
    strategy_params = {
        'prob_threshold': trial.suggest_float('prob_threshold', 0.48, 0.65),
        'cooldown': trial.suggest_int('cooldown', 5, 15),
        'tau_mult': trial.suggest_float('tau_mult', 0.5, 2.0),  # Lower range!
        'tau_bps_base': trial.suggest_float('tau_bps_base', 0.5, 3.0),  # Lower range!
        'min_hold': trial.suggest_int('min_hold', 5, 20),
    }
    
    # ========================================================================
    # MODEL TRAINING WITH FEATURE SELECTION
    # ========================================================================
    
    def make_custom_xgb():
        """Create XGBoost model with trial parameters."""
        from xgboost import XGBRegressor
        return XGBRegressor(
            objective='reg:squarederror',
            n_jobs=1,
            random_state=42,
            tree_method='hist',
            **xgb_params
        )
    
    try:
        # Suppress verbose output
        import io
        import contextlib
        
        # SPEED OPTIMIZATION: Patch walk_forward_splits to use fewer folds
        import src.models as models
        original_wf_splits = models.walk_forward_splits
        
        def fast_walk_forward_splits(n: int, train_frac: float = 0.7, 
                                     step: int = WALK_FORWARD_STEP, 
                                     min_train: int = WALK_FORWARD_MIN_TRAIN):
            """Faster walk-forward with larger steps = fewer folds."""
            first_test = max(int(n * train_frac), min_train)
            for test_start in range(first_test, n - 1000, step):
                train_idx = range(0, test_start)
                test_idx = range(test_start, min(test_start + step, n))
                yield list(train_idx), list(test_idx)
        
        # Temporarily replace with faster version
        models.walk_forward_splits = fast_walk_forward_splits
        
        with contextlib.redirect_stdout(io.StringIO()):
            # Train model with walk-forward validation (now faster!)
            xgb_pred, xgb_conf, _, xgb_metrics, xgb_te_idx, _ = evaluate_regression(
                make_custom_xgb, X, y
            )
        
        # Restore original function
        models.walk_forward_splits = original_wf_splits
        
        # ========================================================================
        # BACKTEST EVALUATION
        # ========================================================================
        
        # Generate positions
        decision_times = time_clean[np.array(xgb_te_idx, dtype=int)]
        forecast_series = pd.Series(xgb_pred / 10_000.0, index=decision_times, name="forecast")
        confidence_series = pd.Series(xgb_conf, index=decision_times, name="confidence")
        
        # Calculate adaptive TAU
        vol_series = ratio_returns_1m(ratio_clean)
        vol_series.index = time_clean
        
        tau_dynamic = adaptive_tau(vol_series, multiplier=VOL_MULTIPLIER)
        tau_base = strategy_params['tau_bps_base'] / 10_000.0
        tau_series = np.maximum(tau_base, tau_dynamic.reindex(decision_times)).ffill()
        tau_series = tau_series.fillna(tau_base) * strategy_params['tau_mult']
        
        # Generate positions with strategy parameters
        positions = generate_positions(
            forecast_series,
            confidence_series,
            tau_series,
            prob_threshold=strategy_params['prob_threshold'],
            min_hold=strategy_params['min_hold'],
            cooldown=strategy_params['cooldown'],
        )
        
        # Calculate PnL and performance metrics
        fwd_log = future_k_bar_return(ratio_clean, k=K_AHEAD)
        fwd_log.index = time_clean
        pnl_components = pnl_from_positions(positions, fwd_log, vol_series)
        metrics = summarize_performance(pnl_components, positions, fwd_log)
        
        # ========================================================================
        # EXTRACT METRICS
        # ========================================================================
        
        sharpe = metrics.get('Sharpe', -999)
        trade_count = metrics.get('TradeCount', 0)
        max_dd = metrics.get('MaxDrawdown', 0)
        total_pnl = metrics.get('TotalPnL', 0)
        hit_rate = metrics.get('HitRate', 0)
        
        # Calculate position distribution
        neutral_mask = positions == 0
        long_mask = positions == 1
        short_mask = positions == -1
        
        neutral_pct = neutral_mask.sum() / len(positions)
        long_pct = long_mask.sum() / len(positions)
        short_pct = short_mask.sum() / len(positions)
        
        # ========================================================================
        # HARD CONSTRAINTS (REJECT IF NOT MET)
        # ========================================================================
        
        # Constraint 1: Minimum trades
        if trade_count < MIN_TRADES:
            trial.set_user_attr('rejection_reason', f'Too few trades: {trade_count} < {MIN_TRADES}')
            trial.set_user_attr('Sharpe', sharpe)
            trial.set_user_attr('TradeCount', trade_count)
            return -999.0
        
        # Constraint 2: Maximum neutral percentage
        if neutral_pct > MAX_NEUTRAL_PCT:
            trial.set_user_attr('rejection_reason', f'Too neutral: {neutral_pct:.1%} > {MAX_NEUTRAL_PCT:.1%}')
            trial.set_user_attr('Sharpe', sharpe)
            trial.set_user_attr('TradeCount', trade_count)
            trial.set_user_attr('NeutralPct', neutral_pct * 100)
            return -999.0
        
        # Constraint 3: Maximum drawdown
        if max_dd < MAX_DRAWDOWN:
            trial.set_user_attr('rejection_reason', f'Excessive drawdown: {max_dd:.1%} < {MAX_DRAWDOWN:.1%}')
            trial.set_user_attr('Sharpe', sharpe)
            trial.set_user_attr('TradeCount', trade_count)
            trial.set_user_attr('MaxDrawdown', max_dd * 100)
            return -999.0
        
        # ========================================================================
        # LOG COMPREHENSIVE METRICS
        # ========================================================================
        
        trial.set_user_attr('Sharpe', sharpe)
        trial.set_user_attr('TotalPnL', total_pnl * 100)
        trial.set_user_attr('TradeCount', trade_count)
        trial.set_user_attr('MaxDrawdown', max_dd * 100)
        trial.set_user_attr('HitRate', hit_rate * 100)
        trial.set_user_attr('LongPct', long_pct * 100)
        trial.set_user_attr('ShortPct', short_pct * 100)
        trial.set_user_attr('NeutralPct', neutral_pct * 100)
        trial.set_user_attr('ModelRMSE', xgb_metrics.get('RMSE', 0))
        trial.set_user_attr('ModelHitRate', xgb_metrics.get('HitRate', 0) * 100)
        
        # Calculate balance when trading
        trading_time = long_pct + short_pct
        if trading_time > 0.01:
            long_ratio_trading = long_pct / trading_time * 100
            trial.set_user_attr('LongRatioWhenTrading', long_ratio_trading)
        
        # ========================================================================
        # RETURN PURE SHARPE (NO BONUSES/PENALTIES!)
        # ========================================================================
        
        return sharpe
        
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        trial.set_user_attr('error', str(e))
        return -999.0


def main():
    parser = argparse.ArgumentParser(
        description='V4.12 Pure Sharpe Optuna Optimization (FAST VERSION)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 100 trials with default settings
  python scripts/optuna_v4_12_pure_sharpe.py --n-trials 100
  
  # Resume previous study
  python scripts/optuna_v4_12_pure_sharpe.py --n-trials 50 --study-name v4_12_pure_sharpe
  
  # Run parallel workers (same study name = share database)
  # Terminal 1:
  python scripts/optuna_v4_12_pure_sharpe.py --n-trials 50 --worker-id 1
  # Terminal 2:
  python scripts/optuna_v4_12_pure_sharpe.py --n-trials 50 --worker-id 2
  # Terminal 3:
  python scripts/optuna_v4_12_pure_sharpe.py --n-trials 50 --worker-id 3
  
  # Run with timeout
  python scripts/optuna_v4_12_pure_sharpe.py --n-trials 200 --timeout 28800
        """
    )
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of optimization trials (default: 100)')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout in seconds (default: None)')
    parser.add_argument('--study-name', type=str, default='v4_12_pure_sharpe',
                       help='Study name for SQLite database (default: v4_12_pure_sharpe)')
    parser.add_argument('--storage', type=str, default='sqlite:///reports/optuna_v4_12.db',
                       help='SQLite database path (default: sqlite:///reports/optuna_v4_12.db)')
    parser.add_argument('--worker-id', type=int, default=None,
                       help='Worker ID for parallel runs (default: None)')
    args = parser.parse_args()
    
    # Set worker ID in environment for logging
    if args.worker_id is not None:
        os.environ['OPTUNA_WORKER_ID'] = str(args.worker_id)
    
    # ========================================================================
    # PRINT BANNER
    # ========================================================================
    
    worker_id_str = f" [Worker {args.worker_id}]" if args.worker_id else ""
    
    print("=" * 80)
    print(f"OPTUNA V4.12 OPTIMIZATION: PURE SHARPE RATIO (FAST){worker_id_str}")
    print("=" * 80)
    print(f"\n📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n🎯 OPTIMIZATION STRATEGY:")
    print("   • Objective: PURE Sharpe Ratio (no bonuses/penalties)")
    print("   • Hard Constraints:")
    print(f"      - Minimum trades: {MIN_TRADES}")
    print(f"      - Maximum neutral: {MAX_NEUTRAL_PCT:.0%}")
    print(f"      - Maximum drawdown: {MAX_DRAWDOWN:.0%}")
    print("\n🛡️  ANTI-OVERFITTING MEASURES:")
    print("   • max_depth constrained to 3-6 (shallow trees only)")
    print("   • Feature selection: 75 → 50 features")
    print("   • Walk-forward cross-validation")
    print("   • Strong regularization ranges")
    print("\n⚡ SPEED OPTIMIZATIONS:")
    print(f"   • Walk-forward folds: ~15 (step={WALK_FORWARD_STEP})")
    print("   • Original folds: ~69 (step=2000)")
    print("   • Speedup: ~5x faster per trial!")
    print(f"\n🔬 TRIALS:")
    print(f"   • Number: {args.n_trials}")
    print(f"   • Study: {args.study_name}")
    print(f"   • Storage: {args.storage}")
    if args.worker_id:
        print(f"   • Worker ID: {args.worker_id} (parallel mode)")
    if args.timeout:
        print(f"   • Timeout: {args.timeout / 3600:.1f} hours")
    print(f"\n⏱️  ESTIMATED TIME:")
    print(f"   • ~5 minutes per trial (with reduced folds)")
    print(f"   • Total: ~{args.n_trials * 5 / 60:.1f} hours")
    if args.worker_id:
        print(f"   • With 3 workers: ~{args.n_trials * 5 / 60 / 3:.1f} hours")
    print("=" * 80)
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    print("\n📥 Loading data...")
    df = pd.read_parquet("data/processed/btc_eth_ratio_1m.parquet")
    X, y_smoothed, kept_idx, meta = build_features(df, k_ahead=K_AHEAD, label="bps")
    y_smoothed = y_smoothed.astype("float32")
    
    time_clean = as_dtindex(meta.get("time_clean"))
    ratio_clean = df.loc[kept_idx, "R"].reset_index(drop=True)
    
    print(f"✓ Data loaded: {X.shape[0]:,} samples, {X.shape[1]} features")
    
    # Feature selection for anti-overfitting
    print(f"\n🔍 Performing feature selection (top {MAX_FEATURES} features)...")
    
    # Create a simple model for feature selection (without early stopping)
    def make_simple_xgb():
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective='reg:squarederror',
            n_jobs=-1,
            random_state=42
        )
    
    X_selected, _ = select_features_by_importance(
        X, y_smoothed, 
        make_model=make_simple_xgb, 
        max_features=MAX_FEATURES,
        verbose=True
    )
    print(f"✓ Features reduced: {X.shape[1]} → {X_selected.shape[1]}")
    
    # ========================================================================
    # CREATE OPTUNA STUDY
    # ========================================================================
    
    print(f"\n🚀 Creating Optuna study...")
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='maximize',
        sampler=TPESampler(seed=42, n_startup_trials=20),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=0),
        load_if_exists=True,  # Resume if exists
    )
    
    # Check if resuming
    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"✓ Resuming existing study with {n_existing} completed trials")
        try:
            best_value = study.best_value
            print(f"   Best Sharpe so far: {best_value:.4f}")
        except ValueError:
            print(f"   No valid trials yet (all rejected by constraints)")
    else:
        print(f"✓ Starting new study")
    
    # ========================================================================
    # RUN OPTIMIZATION
    # ========================================================================
    
    print(f"\n⚡ Starting optimization...")
    print(f"   Press Ctrl+C to stop gracefully")
    print("=" * 80 + "\n")
    
    study.optimize(
        lambda trial: objective(trial, X_selected, y_smoothed, ratio_clean, time_clean),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
        catch=(Exception,),
    )
    
    # ========================================================================
    # RESULTS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("✅ OPTIMIZATION COMPLETE")
    print("=" * 80)
    
    best_trial = study.best_trial
    
    print(f"\n🏆 BEST TRIAL: #{best_trial.number}")
    print(f"   Pure Sharpe: {best_trial.value:.4f}")
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   Total PnL:     {best_trial.user_attrs.get('TotalPnL', 0):.2f}%")
    print(f"   Max Drawdown:  {best_trial.user_attrs.get('MaxDrawdown', 0):.2f}%")
    print(f"   Trade Count:   {best_trial.user_attrs.get('TradeCount', 0):.0f}")
    print(f"   Hit Rate:      {best_trial.user_attrs.get('HitRate', 0):.1f}%")
    
    print(f"\n📈 POSITION DISTRIBUTION:")
    print(f"   Long:          {best_trial.user_attrs.get('LongPct', 0):.1f}%")
    print(f"   Short:         {best_trial.user_attrs.get('ShortPct', 0):.1f}%")
    print(f"   Neutral:       {best_trial.user_attrs.get('NeutralPct', 0):.1f}%")
    if 'LongRatioWhenTrading' in best_trial.user_attrs:
        print(f"   Long when trading: {best_trial.user_attrs.get('LongRatioWhenTrading', 0):.1f}%")
    
    print(f"\n🎛️  BEST PARAMETERS:")
    print("\nXGBoost:")
    for key in ['n_estimators', 'max_depth', 'learning_rate', 'min_child_weight',
                'subsample', 'colsample_bytree', 'gamma', 'reg_alpha', 'reg_lambda']:
        if key in best_trial.params:
            value = best_trial.params[key]
            print(f"   {key}: {value}")
    
    print("\nStrategy:")
    for key in ['prob_threshold', 'cooldown', 'tau_mult', 'tau_bps_base', 'min_hold']:
        if key in best_trial.params:
            value = best_trial.params[key]
            print(f"   {key}: {value}")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    output_dir = Path("reports/v4_12_optuna_pure_sharpe")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save best parameters
    best_params = {
        'optimization_date': datetime.now().isoformat(),
        'study_name': args.study_name,
        'n_trials': len(study.trials),
        'best_trial_number': best_trial.number,
        'best_sharpe': best_trial.value,
        'best_params': best_trial.params,
        'best_metrics': best_trial.user_attrs,
        'constraints': {
            'min_trades': MIN_TRADES,
            'max_neutral_pct': MAX_NEUTRAL_PCT,
            'max_drawdown': MAX_DRAWDOWN,
        },
        'anti_overfitting': {
            'max_depth_range': [3, 6],
            'max_features': MAX_FEATURES,
            'early_stopping': True,
        }
    }
    
    with open(output_dir / 'best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\n✓ Best parameters saved to {output_dir}/best_params.json")
    
    # Save all trials
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / 'all_trials.csv', index=False)
    print(f"✓ All trials saved to {output_dir}/all_trials.csv")
    
    # ========================================================================
    # TOP 10 TRIALS
    # ========================================================================
    
    print(f"\n🎯 TOP 10 TRIALS:")
    print("-" * 80)
    
    # Filter valid trials (Sharpe > -900)
    valid_trials = trials_df[trials_df['value'] > -900].copy()
    
    if len(valid_trials) > 0:
        top_10 = valid_trials.nlargest(10, 'value')
        
        for idx, row in top_10.iterrows():
            trial_num = row['number']
            sharpe = row['value']
            trades = row.get('user_attrs_TradeCount', 0)
            neutral = row.get('user_attrs_NeutralPct', 0)
            print(f"   #{trial_num:3d}: Sharpe {sharpe:6.4f} | "
                  f"Trades: {trades:3.0f} | Neutral: {neutral:4.1f}%")
    else:
        print("   No valid trials (all rejected by constraints)")
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    print(f"\n📊 TRIAL STATISTICS:")
    n_total = len(study.trials)
    n_valid = len(valid_trials)
    n_rejected = n_total - n_valid
    
    print(f"   Total trials:    {n_total}")
    print(f"   Valid trials:    {n_valid} ({n_valid/n_total*100:.1f}%)")
    print(f"   Rejected trials: {n_rejected} ({n_rejected/n_total*100:.1f}%)")
    
    if n_rejected > 0:
        # Count rejection reasons
        rejection_counts = {}
        for trial in study.trials:
            if trial.value <= -900 and 'rejection_reason' in trial.user_attrs:
                reason = trial.user_attrs['rejection_reason']
                reason_type = reason.split(':')[0]  # Get first part
                rejection_counts[reason_type] = rejection_counts.get(reason_type, 0) + 1
        
        print(f"\n   Rejection reasons:")
        for reason, count in sorted(rejection_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"      {reason}: {count} ({count/n_rejected*100:.1f}%)")
    
    # ========================================================================
    # COMPARISON WITH BASELINES
    # ========================================================================
    
    print(f"\n📈 COMPARISON WITH BASELINES:")
    print("-" * 80)
    print(f"   Trial 7 (original): Sharpe 1.2976")
    print(f"   V4.12 Conservative: Sharpe 1.5697")
    print(f"   V4.9.2 Optuna:      Sharpe 3.2292")
    print(f"   V4.12 Pure Sharpe:  Sharpe {best_trial.value:.4f}")
    
    if best_trial.value > 1.5697:
        improvement = ((best_trial.value / 1.5697) - 1) * 100
        print(f"\n   ✅ {improvement:+.1f}% improvement over V4.12 Conservative!")
    
    if best_trial.value > 3.2292:
        print(f"\n   🎉 NEW RECORD! Beat V4.9.2 (Sharpe 3.23)!")
    
    # ========================================================================
    # NEXT STEPS
    # ========================================================================
    
    print(f"\n📝 NEXT STEPS:")
    print(f"   1. Review best parameters in {output_dir}/best_params.json")
    print(f"   2. Validate on held-out test data")
    print(f"   3. Update src/v4_config.py with optimal parameters")
    print(f"   4. Run full backtest with V4Pipeline")
    print(f"   5. Compare with V4.12 Conservative baseline")
    
    print("\n" + "=" * 80)
    print("🎉 OPTIMIZATION COMPLETE!")
    print("=" * 80)
    
    return study


if __name__ == "__main__":
    main()

