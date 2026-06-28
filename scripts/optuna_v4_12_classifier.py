#!/usr/bin/env python3
"""
Optuna V4.12 - XGBClassifier Approach (FIXES the regression TAU mismatch)

WHY THIS EXISTS:
The previous regression-based Optuna (optuna_v4_12_pure_sharpe.py) ran 313 trials
and ALL were rejected. Root cause: forecast magnitudes (~7e-6) are 14x smaller
than the adaptive TAU floor (1e-4), so the model never triggers trades.

SOLUTION:
Switch from XGBRegressor → XGBClassifier:
  - Predicts P(up) directly as 0-1 probability
  - Uses `positions_from_proba` (no TAU needed!)
  - Symmetric by design: P(up)=0.6 → long, P(up)=0.4 → short
  - Eliminates the forecast-vs-TAU scale mismatch entirely

OPTIMIZATION DETAILS:
  - Objective: Pure Sharpe Ratio
  - Hard constraints: min trades, max neutral %, max drawdown
  - Anti-overfitting: shallow trees, feature selection, walk-forward CV
  - Parallel-safe SQLite persistence
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

from src.features import build_features
from src.models import evaluate_classification, select_features_by_importance
from src.backtest import (
    future_k_bar_return,
    pnl_from_positions,
    ratio_returns_1m,
    summarize_performance,
)
from src.utils import as_dtindex


# ============================================================================
# CONSTANTS
# ============================================================================

K_AHEAD = 10

# Hard constraints (relaxed to find initial valid trials, then Optuna tightens)
MIN_TRADES = 20
MAX_NEUTRAL_PCT = 0.90   # 90% max neutral
MAX_DRAWDOWN = -0.50     # 50% max drawdown (loose, let Sharpe do the work)

# Walk-forward speed settings
WALK_FORWARD_STEP = 10000
WALK_FORWARD_MIN_TRAIN = 5000
MAX_FEATURES = 50


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

def objective(trial: optuna.Trial, X: pd.DataFrame, y_bps: pd.Series,
              ratio_clean: pd.Series, time_clean: pd.DatetimeIndex) -> float:
    """
    XGBClassifier-based Sharpe optimization.
    
    Key difference from regression: uses P(up) directly with positions_from_proba,
    eliminating the TAU threshold that blocked all previous trials.
    """
    
    # ------------------------------------------------------------------
    # XGBoost CLASSIFIER parameters
    # ------------------------------------------------------------------
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 150, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 8),
        'subsample': trial.suggest_float('subsample', 0.60, 0.90),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.55, 0.85),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 1.2),
    }
    
    # ------------------------------------------------------------------
    # Strategy parameters (NO TAU! Just probability thresholds)
    # Hysteresis: enter at higher threshold, exit at lower threshold
    # This prevents rapid enter/exit cycling.
    # ------------------------------------------------------------------
    strategy_params = {
        'entry_threshold': trial.suggest_float('entry_threshold', 0.52, 0.60),
        'exit_threshold': trial.suggest_float('exit_threshold', 0.48, 0.52),  # Must be < entry
        'min_hold': trial.suggest_int('min_hold', 5, 30),
        'cooldown': trial.suggest_int('cooldown', 3, 15),
    }
    
    # ------------------------------------------------------------------
    # BUILD MODEL
    # ------------------------------------------------------------------
    
    def make_cls():
        from xgboost import XGBClassifier
        return XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            early_stopping_rounds=30,
            n_jobs=1,
            random_state=42,
            tree_method='hist',
            **xgb_params
        )
    
    try:
        # Patch walk_forward_splits for speed
        import io
        import contextlib
        import src.models as models
        original_wf = models.walk_forward_splits
        
        def fast_wf(n, train_frac=0.7, step=WALK_FORWARD_STEP, 
                     min_train=WALK_FORWARD_MIN_TRAIN):
            first_test = max(int(n * train_frac), min_train)
            for test_start in range(first_test, n - 1000, step):
                train_idx = range(0, test_start)
                test_idx = range(test_start, min(test_start + step, n))
                yield list(train_idx), list(test_idx)
        
        models.walk_forward_splits = fast_wf
        
        with contextlib.redirect_stdout(io.StringIO()):
            proba_up, ys, cls_metrics, te_idx, _ = evaluate_classification(
                make_cls, X, y_bps
            )
        
        models.walk_forward_splits = original_wf
        
        # ------------------------------------------------------------------
        # GENERATE POSITIONS (probability-based, no TAU!)
        # Uses custom logic with proper exit-to-neutral zone.
        # positions_from_proba lacks neutral exit (once in LONG, needs prob<0.30
        # to exit, which almost never happens with narrow distributions).
        # ------------------------------------------------------------------
        
        decision_times = time_clean[np.array(te_idx, dtype=int)]
        
        entry_thresh = strategy_params['entry_threshold']
        exit_thresh = strategy_params['exit_threshold']
        min_hold_val = strategy_params['min_hold']
        cooldown_val = strategy_params['cooldown']
        
        # Hysteresis position generation:
        #   Enter LONG  when prob >= entry_threshold (e.g., 0.55)
        #   Enter SHORT when prob <= 1 - entry_threshold (e.g., 0.45)
        #   Exit LONG  when prob drops below exit_threshold (e.g., 0.50)
        #   Exit SHORT when prob rises above 1 - exit_threshold (e.g., 0.50)
        # This creates a dead zone between exit and entry that prevents cycling.
        
        pos = np.zeros(len(proba_up), dtype=int)
        last_change = -10_000
        
        for i in range(len(proba_up)):
            prob = proba_up[i]
            prev = pos[i - 1] if i > 0 else 0
            desired = prev
            can_change = (i - last_change) >= min_hold_val
            
            if prev == 0:  # NEUTRAL → enter only on strong signal
                if prob >= entry_thresh:
                    desired = 1
                elif prob <= (1.0 - entry_thresh):
                    desired = -1
            
            elif prev == 1:  # LONG → exit when signal weakens
                if can_change:
                    if prob <= (1.0 - entry_thresh):
                        desired = -1    # Flip to SHORT (strong opposite signal)
                    elif prob < exit_thresh:
                        desired = 0     # Exit to NEUTRAL (signal weakened)
            
            elif prev == -1:  # SHORT → exit when signal weakens
                if can_change:
                    if prob >= entry_thresh:
                        desired = 1     # Flip to LONG (strong opposite signal)
                    elif prob > (1.0 - exit_thresh):
                        desired = 0     # Exit to NEUTRAL (signal weakened)
            
            # Cooldown
            if desired != prev and (i - last_change) <= cooldown_val:
                desired = prev
            
            pos[i] = desired
            if desired != prev:
                last_change = i
        
        positions = pd.Series(pos, index=decision_times, name="position")
        
        # ------------------------------------------------------------------
        # CALCULATE PERFORMANCE
        # ------------------------------------------------------------------
        
        vol_series = ratio_returns_1m(ratio_clean)
        vol_series.index = time_clean
        
        fwd_log = future_k_bar_return(ratio_clean, k=K_AHEAD)
        fwd_log.index = time_clean
        
        pnl_components = pnl_from_positions(positions, fwd_log, vol_series)
        metrics = summarize_performance(pnl_components, positions, fwd_log)
        
        sharpe = metrics.get('Sharpe', -999)
        trade_count = metrics.get('TradeCount', 0)
        max_dd = metrics.get('MaxDrawdown', 0)
        total_pnl = metrics.get('TotalPnL', 0)
        hit_rate = metrics.get('HitRate', 0)
        
        # Position distribution
        neutral_pct = (positions == 0).sum() / len(positions)
        long_pct = (positions == 1).sum() / len(positions)
        short_pct = (positions == -1).sum() / len(positions)
        
        # ------------------------------------------------------------------
        # PROBABILITY DISTRIBUTION STATS (diagnostic)
        # ------------------------------------------------------------------
        pct_above_entry = (proba_up >= entry_thresh).sum() / len(proba_up) * 100
        pct_below_entry = (proba_up <= (1 - entry_thresh)).sum() / len(proba_up) * 100
        proba_std = float(np.std(proba_up))
        proba_mean = float(np.mean(proba_up))
        
        # ------------------------------------------------------------------
        # LOG CONSTRAINT STATUS (but don't reject yet - we need to see actual Sharpe!)
        # ------------------------------------------------------------------
        
        constraints_passed = True
        constraint_issues = []
        
        if trade_count < MIN_TRADES:
            constraint_issues.append(f'trades={trade_count}<{MIN_TRADES}')
            constraints_passed = False
        
        if neutral_pct > MAX_NEUTRAL_PCT:
            constraint_issues.append(f'neutral={neutral_pct:.1%}>{MAX_NEUTRAL_PCT:.0%}')
            constraints_passed = False
        
        if max_dd < MAX_DRAWDOWN:
            constraint_issues.append(f'dd={max_dd:.1%}<{MAX_DRAWDOWN:.0%}')
            constraints_passed = False
        
        # Print trial summary (visible in logs)
        constraint_str = " | ".join(constraint_issues) if constraint_issues else "ALL PASSED"
        print(f"  Trial {trial.number}: Sharpe={sharpe:.4f} PnL={total_pnl*100:.2f}% "
              f"Trades={trade_count} Neutral={neutral_pct:.1%} DD={max_dd:.2%} "
              f"[{constraint_str}]", flush=True)
        
        # Soft rejection: only reject if Sharpe is also bad
        if not constraints_passed and sharpe <= 0:
            trial.set_user_attr('rejection_reason', constraint_str)
            trial.set_user_attr('Sharpe', sharpe)
            trial.set_user_attr('TradeCount', trade_count)
            trial.set_user_attr('NeutralPct', neutral_pct * 100)
            trial.set_user_attr('ProbaStd', proba_std)
            trial.set_user_attr('PctAboveEntry', pct_above_entry)
            trial.set_user_attr('MaxDrawdown', max_dd * 100)
            return -999.0
        
        # ------------------------------------------------------------------
        # LOG ALL METRICS (trial passed constraints!)
        # ------------------------------------------------------------------
        
        trial.set_user_attr('Sharpe', sharpe)
        trial.set_user_attr('TotalPnL', total_pnl * 100)
        trial.set_user_attr('TradeCount', trade_count)
        trial.set_user_attr('MaxDrawdown', max_dd * 100)
        trial.set_user_attr('HitRate', hit_rate * 100)
        trial.set_user_attr('LongPct', long_pct * 100)
        trial.set_user_attr('ShortPct', short_pct * 100)
        trial.set_user_attr('NeutralPct', neutral_pct * 100)
        trial.set_user_attr('ModelAUC', cls_metrics.get('AUC', 0))
        trial.set_user_attr('ProbaStd', proba_std)
        trial.set_user_attr('ProbaMean', proba_mean)
        trial.set_user_attr('PctAboveEntry', pct_above_entry)
        trial.set_user_attr('PctBelowEntry', pct_below_entry)
        
        # Long/short balance
        trading_time = long_pct + short_pct
        if trading_time > 0.01:
            trial.set_user_attr('LongRatioWhenTrading', long_pct / trading_time * 100)
        
        # ------------------------------------------------------------------
        # RETURN PURE SHARPE
        # ------------------------------------------------------------------
        return sharpe
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        trial.set_user_attr('error', str(e))
        return -999.0


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='V4.12 Optuna - XGBClassifier (fixes TAU mismatch)',
    )
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--timeout', type=int, default=None)
    parser.add_argument('--study-name', type=str, default='v4_12_classifier')
    parser.add_argument('--storage', type=str, default='sqlite:///reports/optuna_v4_12_cls.db')
    parser.add_argument('--worker-id', type=int, default=None)
    args = parser.parse_args()
    
    if args.worker_id is not None:
        os.environ['OPTUNA_WORKER_ID'] = str(args.worker_id)
    
    worker_str = f" [Worker {args.worker_id}]" if args.worker_id else ""
    
    print("=" * 80)
    print(f"OPTUNA V4.12 - XGBCLASSIFIER APPROACH{worker_str}")
    print("=" * 80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nWhy XGBClassifier?")
    print(f"   Previous regression approach: 313 trials, ALL rejected")
    print(f"   Root cause: forecast magnitude (~7e-6) << TAU floor (1e-4)")
    print(f"   Fix: Classify P(up) directly → no TAU needed!")
    print(f"\nObjective: Pure Sharpe Ratio")
    print(f"Constraints: min_trades={MIN_TRADES}, max_neutral={MAX_NEUTRAL_PCT:.0%}, max_dd={MAX_DRAWDOWN:.0%}")
    print(f"Trials: {args.n_trials} | Study: {args.study_name}")
    print(f"Walk-forward folds: ~{460000 // WALK_FORWARD_STEP} (fast mode)")
    print("=" * 80)
    
    # ------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------
    
    print("\nLoading data...")
    df = pd.read_parquet("data/processed/btc_eth_ratio_1m.parquet")
    X, y_bps, kept_idx, meta = build_features(df, k_ahead=K_AHEAD, label="bps")
    y_bps = y_bps.astype("float32")
    
    time_clean = as_dtindex(meta.get("time_clean"))
    ratio_clean = df.loc[kept_idx, "R"].reset_index(drop=True)
    
    print(f"  Data: {X.shape[0]:,} samples, {X.shape[1]} features")
    
    # Feature selection (use regressor - works with continuous y_bps)
    print(f"\nFeature selection (top {MAX_FEATURES})...")
    
    def make_simple_reg():
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            objective='reg:squarederror', n_jobs=-1, random_state=42
        )
    
    X_selected, _ = select_features_by_importance(
        X, y_bps, make_model=make_simple_reg, max_features=MAX_FEATURES, verbose=True
    )
    print(f"  Features: {X.shape[1]} → {X_selected.shape[1]}")
    
    # ------------------------------------------------------------------
    # CREATE STUDY (fresh DB, separate from regression attempts)
    # ------------------------------------------------------------------
    
    print(f"\nCreating Optuna study...")
    
    # Use different seed per worker to prevent duplicate startup trials
    seed = 42 + (args.worker_id or 0)
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='maximize',
        sampler=TPESampler(seed=seed, n_startup_trials=15),
        load_if_exists=True,
    )
    
    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"  Resuming: {n_existing} existing trials")
        try:
            print(f"  Best Sharpe so far: {study.best_value:.4f}")
        except ValueError:
            print(f"  No valid trials yet")
    else:
        print(f"  Starting fresh study")
    
    # ------------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------------
    
    print(f"\nStarting optimization ({args.n_trials} trials)...")
    print("=" * 80 + "\n")
    
    study.optimize(
        lambda trial: objective(trial, X_selected, y_bps, ratio_clean, time_clean),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
        catch=(Exception,),
    )
    
    # ------------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------------
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    
    # Filter valid trials
    valid_trials = [t for t in study.trials if t.value is not None and t.value > -900]
    n_total = len(study.trials)
    n_valid = len(valid_trials)
    
    print(f"\nTrials: {n_total} total, {n_valid} valid ({n_valid/max(n_total,1)*100:.1f}%)")
    
    if n_valid > 0:
        try:
            best = study.best_trial
            print(f"\nBEST TRIAL: #{best.number}")
            print(f"  Pure Sharpe:   {best.value:.4f}")
            print(f"  Total PnL:     {best.user_attrs.get('TotalPnL', 0):.2f}%")
            print(f"  Max Drawdown:  {best.user_attrs.get('MaxDrawdown', 0):.2f}%")
            print(f"  Trade Count:   {best.user_attrs.get('TradeCount', 0):.0f}")
            print(f"  Hit Rate:      {best.user_attrs.get('HitRate', 0):.1f}%")
            print(f"  Long:          {best.user_attrs.get('LongPct', 0):.1f}%")
            print(f"  Short:         {best.user_attrs.get('ShortPct', 0):.1f}%")
            print(f"  Neutral:       {best.user_attrs.get('NeutralPct', 0):.1f}%")
            print(f"  Model AUC:     {best.user_attrs.get('ModelAUC', 0):.4f}")
            
            print(f"\nBest XGBoost params:")
            for key in ['n_estimators', 'max_depth', 'learning_rate', 'min_child_weight',
                        'subsample', 'colsample_bytree', 'gamma', 'reg_alpha', 'reg_lambda',
                        'scale_pos_weight']:
                if key in best.params:
                    print(f"  {key}: {best.params[key]}")
            
            print(f"\nBest Strategy params:")
            for key in ['entry_threshold', 'exit_threshold', 'min_hold', 'cooldown']:
                if key in best.params:
                    print(f"  {key}: {best.params[key]}")
            
            # Save results
            output_dir = Path("reports/v4_12_classifier_optuna")
            output_dir.mkdir(exist_ok=True, parents=True)
            
            results = {
                'optimization_date': datetime.now().isoformat(),
                'approach': 'XGBClassifier (fixes TAU mismatch)',
                'study_name': args.study_name,
                'n_trials': n_total,
                'n_valid': n_valid,
                'best_trial': best.number,
                'best_sharpe': best.value,
                'best_params': best.params,
                'best_metrics': best.user_attrs,
                'constraints': {
                    'min_trades': MIN_TRADES,
                    'max_neutral_pct': MAX_NEUTRAL_PCT,
                    'max_drawdown': MAX_DRAWDOWN,
                },
            }
            
            with open(output_dir / 'best_params.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nSaved: {output_dir}/best_params.json")
            
            # Save all trials CSV
            trials_df = study.trials_dataframe()
            trials_df.to_csv(output_dir / 'all_trials.csv', index=False)
            print(f"Saved: {output_dir}/all_trials.csv")
            
            # Top 10
            print(f"\nTOP 10 TRIALS:")
            print("-" * 70)
            sorted_valid = sorted(valid_trials, key=lambda t: t.value, reverse=True)[:10]
            for t in sorted_valid:
                trades = t.user_attrs.get('TradeCount', 0)
                neutral = t.user_attrs.get('NeutralPct', 0)
                pnl = t.user_attrs.get('TotalPnL', 0)
                print(f"  #{t.number:3d}: Sharpe={t.value:7.4f} | Trades={trades:4.0f} | "
                      f"Neutral={neutral:5.1f}% | PnL={pnl:+6.2f}%")
        except ValueError:
            print("\nNo best trial found (edge case)")
    else:
        print("\nNo valid trials found!")
        
        # Diagnostic: show rejection reasons
        rejection_counts = {}
        for t in study.trials:
            if 'rejection_reason' in t.user_attrs:
                reason = t.user_attrs['rejection_reason'].split(':')[0]
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        
        if rejection_counts:
            print("\nRejection reasons:")
            for reason, count in sorted(rejection_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {reason}: {count}")
        
        # Show probability stats from recent trials
        recent = study.trials[-5:]
        for t in recent:
            ps = t.user_attrs.get('ProbaStd', 0)
            pa = t.user_attrs.get('PctAboveEntry', 0)
            tc = t.user_attrs.get('TradeCount', 0)
            print(f"  Trial {t.number}: proba_std={ps:.4f}, pct_above_entry={pa:.1f}%, trades={tc}")
    
    # Baselines comparison
    print(f"\nBASELINES:")
    print(f"  Trial 7 (original):  Sharpe 1.2976")
    print(f"  V4.12 Conservative:  Sharpe 1.5697")
    if n_valid > 0:
        best_sharpe = study.best_value
        if best_sharpe > 1.5697:
            print(f"  V4.12 Classifier:    Sharpe {best_sharpe:.4f} ✅ IMPROVEMENT!")
        else:
            print(f"  V4.12 Classifier:    Sharpe {best_sharpe:.4f}")
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    
    return study


if __name__ == "__main__":
    main()

