#!/usr/bin/env python3
"""
Validate Optuna Best Parameters on Held-Out Test Data

This script validates the best parameters from Optuna optimization
on a held-out test set to ensure they generalize well.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.features import build_features
from src.backtest import (
    adaptive_tau,
    future_k_bar_return,
    generate_positions,
    pnl_from_positions,
    ratio_returns_1m,
    summarize_performance,
)
from src.utils import as_dtindex


K_AHEAD = 10


def validate_params(params_dict: dict, X: pd.DataFrame, y: pd.Series,
                    ratio_clean: pd.Series, time_clean: pd.DatetimeIndex,
                    test_split: float = 0.8) -> dict:
    """
    Validate parameters on held-out test data.
    
    Args:
        params_dict: Dictionary with 'xgb_params' and 'strategy_params'
        X, y: Features and target
        ratio_clean: BTC/ETH ratio
        time_clean: Timestamps
        test_split: Train on first X%, test on remaining
    
    Returns:
        Dictionary with validation metrics
    """
    
    # Split data
    split_idx = int(len(X) * test_split)
    
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    time_train = time_clean[:split_idx]
    time_test = time_clean[split_idx:]
    ratio_train = ratio_clean.iloc[:split_idx]
    ratio_test = ratio_clean.iloc[split_idx:]
    
    print(f"\n📊 Data Split:")
    print(f"   Train: {len(X_train):,} samples ({test_split*100:.0f}%)")
    print(f"   Test:  {len(X_test):,} samples ({(1-test_split)*100:.0f}%)")
    
    # Extract parameters
    xgb_params = params_dict['xgb_params']
    strategy_params = params_dict['strategy_params']
    
    # Train model on training data
    print(f"\n🔧 Training model on training data...")
    model = XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42,
        tree_method='hist',
        **xgb_params
    )
    
    model.fit(X_train, y_train)
    
    # Predict on test data
    print(f"🔮 Predicting on test data...")
    y_pred = model.predict(X_test)
    
    # Calculate model metrics
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    hit_rate = (np.sign(y_pred) == np.sign(y_test)).mean()
    
    print(f"\n📈 Model Performance:")
    print(f"   RMSE:     {rmse:.2f}")
    print(f"   Hit Rate: {hit_rate:.1%}")
    
    # Backtest on test data
    print(f"\n💰 Backtesting on test data...")
    
    forecast_series = pd.Series(y_pred / 10_000.0, index=time_test, name="forecast")
    
    # Simple confidence (can be improved)
    confidence_series = pd.Series(0.6, index=time_test, name="confidence")
    
    # Calculate adaptive TAU
    vol_series = ratio_returns_1m(ratio_test)
    vol_series.index = time_test
    
    tau_dynamic = adaptive_tau(vol_series, multiplier=2.0)
    tau_base = strategy_params['tau_bps_base'] / 10_000.0
    tau_series = np.maximum(tau_base, tau_dynamic.reindex(time_test)).ffill()
    tau_series = tau_series.fillna(tau_base) * strategy_params['tau_mult']
    
    # Generate positions
    positions = generate_positions(
        forecast_series,
        confidence_series,
        tau_series,
        prob_threshold=strategy_params['prob_threshold'],
        min_hold=strategy_params.get('min_hold', 10),
        cooldown=strategy_params['cooldown'],
    )
    
    # Calculate PnL
    fwd_log = future_k_bar_return(ratio_test, k=K_AHEAD)
    fwd_log.index = time_test
    pnl_components = pnl_from_positions(positions, fwd_log, vol_series)
    metrics = summarize_performance(pnl_components, positions, fwd_log)
    
    # Position statistics
    neutral_pct = (positions == 0).sum() / len(positions)
    long_pct = (positions == 1).sum() / len(positions)
    short_pct = (positions == -1).sum() / len(positions)
    
    print(f"\n🎯 Backtest Results:")
    print(f"   Sharpe:        {metrics['Sharpe']:.4f}")
    print(f"   Total PnL:     {metrics['TotalPnL']*100:.2f}%")
    print(f"   Max Drawdown:  {metrics['MaxDrawdown']*100:.2f}%")
    print(f"   Trade Count:   {metrics['TradeCount']}")
    print(f"   Hit Rate:      {metrics['HitRate']:.1%}")
    print(f"\n📍 Position Distribution:")
    print(f"   Long:          {long_pct:.1%}")
    print(f"   Short:         {short_pct:.1%}")
    print(f"   Neutral:       {neutral_pct:.1%}")
    
    return {
        'model_metrics': {
            'RMSE': float(rmse),
            'HitRate': float(hit_rate),
        },
        'backtest_metrics': {
            'Sharpe': float(metrics['Sharpe']),
            'TotalPnL': float(metrics['TotalPnL']),
            'MaxDrawdown': float(metrics['MaxDrawdown']),
            'TradeCount': int(metrics['TradeCount']),
            'BacktestHitRate': float(metrics['HitRate']),
        },
        'position_stats': {
            'LongPct': float(long_pct),
            'ShortPct': float(short_pct),
            'NeutralPct': float(neutral_pct),
        },
        'data_split': {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'test_split': test_split,
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Validate Optuna best parameters on held-out test data'
    )
    parser.add_argument('--params-file', type=str,
                       default='reports/v4_12_optuna_pure_sharpe/best_params.json',
                       help='Path to best parameters JSON file')
    parser.add_argument('--test-split', type=float, default=0.8,
                       help='Train on first X%% of data (default: 0.8)')
    parser.add_argument('--output', type=str,
                       default='reports/v4_12_optuna_pure_sharpe/validation_results.json',
                       help='Output file for validation results')
    args = parser.parse_args()
    
    print("=" * 80)
    print("OPTUNA BEST PARAMETERS VALIDATION")
    print("=" * 80)
    print(f"\n📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📁 Parameters: {args.params_file}")
    print(f"📊 Test Split: {args.test_split*100:.0f}% train / {(1-args.test_split)*100:.0f}% test")
    print("=" * 80)
    
    # Load parameters
    print(f"\n📥 Loading best parameters...")
    with open(args.params_file, 'r') as f:
        params_data = json.load(f)
    
    # Extract XGB and strategy params
    best_params = params_data['best_params']
    
    # Separate parameters
    xgb_param_keys = ['n_estimators', 'max_depth', 'learning_rate', 'min_child_weight',
                      'subsample', 'colsample_bytree', 'gamma', 'reg_alpha', 'reg_lambda']
    strategy_param_keys = ['prob_threshold', 'cooldown', 'tau_mult', 'tau_bps_base', 'min_hold']
    
    xgb_params = {k: best_params[k] for k in xgb_param_keys if k in best_params}
    strategy_params = {k: best_params[k] for k in strategy_param_keys if k in best_params}
    
    print(f"✓ Loaded parameters from trial #{params_data['best_trial_number']}")
    print(f"   Training Sharpe: {params_data['best_sharpe']:.4f}")
    
    # Load data
    print(f"\n📥 Loading data...")
    df = pd.read_parquet("data/processed/btc_eth_ratio_1m.parquet")
    X, y_smoothed, kept_idx, meta = build_features(df, k_ahead=K_AHEAD, label="bps")
    y_smoothed = y_smoothed.astype("float32")
    
    time_clean = as_dtindex(meta.get("time_clean"))
    ratio_clean = df.loc[kept_idx, "R"].reset_index(drop=True)
    
    print(f"✓ Data loaded: {X.shape[0]:,} samples, {X.shape[1]} features")
    
    # Validate
    validation_results = validate_params(
        {'xgb_params': xgb_params, 'strategy_params': strategy_params},
        X, y_smoothed, ratio_clean, time_clean,
        test_split=args.test_split
    )
    
    # Add metadata
    validation_results['validation_date'] = datetime.now().isoformat()
    validation_results['params_file'] = args.params_file
    validation_results['training_sharpe'] = params_data['best_sharpe']
    validation_results['parameters'] = {
        'xgb_params': xgb_params,
        'strategy_params': strategy_params,
    }
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\n✓ Validation results saved to {output_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    training_sharpe = params_data['best_sharpe']
    test_sharpe = validation_results['backtest_metrics']['Sharpe']
    
    print(f"\n📈 Sharpe Comparison:")
    print(f"   Training (CV):  {training_sharpe:.4f}")
    print(f"   Test (Held-out): {test_sharpe:.4f}")
    
    if test_sharpe > training_sharpe * 0.9:
        print(f"   ✅ Good generalization! (Test >= 90% of training)")
    elif test_sharpe > training_sharpe * 0.7:
        print(f"   ⚠️  Moderate generalization (Test >= 70% of training)")
    else:
        print(f"   ❌ Poor generalization! (Test < 70% of training)")
        print(f"   Consider: More regularization, simpler model, or different features")
    
    print("\n" + "=" * 80)
    print("✅ VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

