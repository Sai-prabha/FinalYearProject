"""Optuna hyperparameter optimization for V4 model.

Tunes both XGBoost parameters and trading strategy parameters to maximize Sharpe ratio.
"""
import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import build_features
from src.models import make_xgb_reg, evaluate_regression
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
MIN_HOLD = 3
VOL_MULTIPLIER = 2.5


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, 
              ratio_clean: pd.Series, time_clean: pd.DatetimeIndex) -> float:
    """Optimization objective: maximize Sharpe ratio."""
    
    # XGBoost hyperparameters
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
    }
    
    # Trading strategy parameters
    prob_threshold = trial.suggest_float('prob_threshold', 0.50, 0.75)
    cooldown_val = trial.suggest_int('cooldown', 5, 25)
    tau_mult = trial.suggest_float('tau_mult', 0.5, 5.0)
    tau_bps_base = trial.suggest_float('tau_bps_base', 0.5, 5.0)
    
    # Create custom XGBoost model maker with trial parameters
    def make_custom_xgb():
        from xgboost import XGBRegressor
        return XGBRegressor(
            objective='reg:squarederror',
            n_jobs=1,
            random_state=42,
            **xgb_params
        )
    
    try:
        # Train model with walk-forward validation
        xgb_pred, xgb_conf, _, xgb_metrics, xgb_te_idx, _ = evaluate_regression(
            make_custom_xgb, X, y
        )
        
        # Generate positions
        decision_times_reg = time_clean[np.array(xgb_te_idx, dtype=int)]
        forecast_series = pd.Series(xgb_pred / 10_000.0, index=decision_times_reg, name="forecast")
        regression_conf = pd.Series(xgb_conf, index=decision_times_reg, name="confidence")
        
        vol_series = ratio_returns_1m(ratio_clean)
        vol_series.index = time_clean
        
        tau_dynamic = adaptive_tau(vol_series, multiplier=VOL_MULTIPLIER)
        tau_base = tau_bps_base / 10_000.0
        tau_series = np.maximum(tau_base, tau_dynamic.reindex(decision_times_reg)).ffill()
        tau_series = tau_series.fillna(tau_base) * tau_mult
        
        positions = generate_positions(
            forecast_series,
            regression_conf,
            tau_series,
            prob_threshold=prob_threshold,
            min_hold=MIN_HOLD,
            cooldown=cooldown_val,
        )
        
        # Calculate PnL (zero costs)
        fwd_log = future_k_bar_return(ratio_clean, k=K_AHEAD)
        fwd_log.index = time_clean
        pnl_components = pnl_from_positions(positions, fwd_log, vol_series)
        metrics = summarize_performance(pnl_components, positions, fwd_log)
        
        sharpe = metrics.get('Sharpe', -10.0)
        
        # Penalize if too few trades (less than 100) or too many (more than 10000)
        trade_count = metrics.get('TradeCount', 0)
        if trade_count < 100:
            sharpe -= (100 - trade_count) * 0.01
        elif trade_count > 10000:
            sharpe -= (trade_count - 10000) * 0.0001
        
        # Report intermediate values
        trial.set_user_attr('hit_rate', metrics.get('HitRate', 0.0))
        trial.set_user_attr('total_pnl', metrics.get('TotalPnL', 0.0))
        trial.set_user_attr('trade_count', trade_count)
        trial.set_user_attr('max_drawdown', metrics.get('MaxDrawdown', 0.0))
        
        return sharpe
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return -10.0


def main(n_trials: int = 50, output_dir: str = "reports/v4.9.2"):
    """Run Optuna optimization."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pd.read_parquet("data/processed/btc_eth_ratio_1m.parquet")
    X, _, kept_idx, meta = build_features(df, k_ahead=K_AHEAD, label="bps")
    y_smoothed = df.loc[kept_idx, "y_bps_smoothed"].astype("float32").reset_index(drop=True)
    
    time_clean = as_dtindex(meta.get("time_clean", X.index if hasattr(X, "index") else None))
    if time_clean is None:
        raise RuntimeError("Could not infer time_clean from features meta or X.index")
    
    ratio_clean = df.loc[kept_idx, "R"].reset_index(drop=True)
    
    print(f"Data loaded: X shape {X.shape}, y len {len(y_smoothed)}")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='v4_xgboost_optimization'
    )
    
    # Optimize
    print(f"Starting optimization with {n_trials} trials...")
    study.optimize(
        lambda trial: objective(trial, X, y_smoothed, ratio_clean, time_clean),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    
    # Save results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nBest Sharpe Ratio: {study.best_value:.4f}")
    print("\nBest Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    print("\nBest Trial Metrics:")
    best_trial = study.best_trial
    for key, value in best_trial.user_attrs.items():
        print(f"  {key}: {value}")
    
    # Save to JSON
    results = {
        'best_sharpe': study.best_value,
        'best_params': study.best_params,
        'best_metrics': best_trial.user_attrs,
        'n_trials': n_trials,
    }
    
    with open(output_path / "optuna_best_params.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save study history
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_path / "optuna_trials.csv", index=False)
    
    print(f"\nResults saved to {output_path}/")
    print("  - optuna_best_params.json")
    print("  - optuna_trials.csv")
    
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna optimization for V4 model")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--output-dir", default="reports/v4.9.2", help="Output directory")
    args = parser.parse_args()
    
    main(n_trials=args.n_trials, output_dir=args.output_dir)


