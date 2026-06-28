"""Optuna optimization for XGBoost with BALANCED LONG/SHORT strategy.

CRITICAL DIFFERENCE from original optimization:
- Original (v4.9.3): Allows neutral positions, goes neutral ~95% of time
- This version: FORCES 50/50 long/short exposure (never neutral)

This requires different hyperparameters and strategy parameters.
We test both MSE and Pseudo-Huber objectives.
"""

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner

from src.features import build_features
from src.models import evaluate_regression
from src.backtest import (
    adaptive_tau, future_k_bar_return,
    pnl_from_positions, ratio_returns_1m, summarize_performance,
)
from src.utils import as_dtindex


K_AHEAD = 10
VOL_MULTIPLIER = 2.5


def generate_positions_balanced(forecast: pd.Series, confidence: pd.Series, tau_series: pd.Series,
                                conf_thresh: float, min_hold: int, cooldown: int) -> pd.Series:
    """Balanced long/short strategy (never neutral) with adaptive tau."""
    pos = np.zeros(len(forecast), dtype=int)
    last_change = -10_000
    
    for i in range(len(forecast)):
        fcast = forecast.iloc[i]
        conf = confidence.iloc[i]
        tau = tau_series.iloc[i]
        
        if i == 0:
            pos[i] = 1 if fcast > 0 else -1
            last_change = 0
            continue
        
        prev = pos[i-1]
        desired = prev
        
        # Switch on strong signal
        if prev == 1 and fcast < -tau and conf >= conf_thresh:
            if (i - last_change) >= min_hold:
                desired = -1
        elif prev == -1 and fcast > tau and conf >= conf_thresh:
            if (i - last_change) >= min_hold:
                desired = 1
        # Weaker signal after longer hold
        elif (i - last_change) >= (min_hold * 3):
            weak_tau = tau / 2.5
            if prev == 1 and fcast < -weak_tau and conf >= (conf_thresh - 0.05):
                desired = -1
            elif prev == -1 and fcast > weak_tau and conf >= (conf_thresh - 0.05):
                desired = 1
        
        # Cooldown check
        if desired != prev and (i - last_change) <= cooldown:
            desired = prev
        
        pos[i] = desired
        if desired != prev:
            last_change = i
    
    return pd.Series(pos, index=pd.to_datetime(forecast.index, utc=True).tz_convert(None), name="position")


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series,
              ratio_clean: pd.Series, time_clean: pd.DatetimeIndex, 
              use_pseudohuber: bool) -> float:
    """Optimization objective for balanced long/short strategy."""
    
    # XGBoost hyperparameters
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'gamma': trial.suggest_float('gamma', 0.0, 6.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 30.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 20.0, log=True),
    }
    
    if use_pseudohuber:
        xgb_params['huber_slope'] = trial.suggest_float('huber_slope', 0.3, 2.0)
    
    # Strategy parameters - CRITICAL for balanced long/short
    # These ranges are MUCH WIDER than neutral-allowed strategy
    strategy_params = {
        'prob_threshold': trial.suggest_float('prob_threshold', 0.50, 0.70),
        'cooldown': trial.suggest_int('cooldown', 5, 60),
        'tau_mult': trial.suggest_float('tau_mult', 0.3, 3.0),  # Key param!
        'tau_bps_base': trial.suggest_float('tau_bps_base', 0.3, 5.0),
        'min_hold': trial.suggest_int('min_hold', 5, 120),  # Also optimize this
    }
    
    # Create model
    def make_custom_xgb():
        from xgboost import XGBRegressor
        objective_name = 'reg:pseudohubererror' if use_pseudohuber else 'reg:squarederror'
        return XGBRegressor(
            objective=objective_name,
            n_jobs=1,
            random_state=42,
            tree_method='hist',
            **xgb_params
        )
    
    try:
        # Train model (suppress output)
        import io
        import contextlib
        
        with contextlib.redirect_stdout(io.StringIO()):
            xgb_pred, xgb_conf, _, xgb_metrics, xgb_te_idx, _ = evaluate_regression(
                make_custom_xgb, X, y
            )
        
        # Backtest with BALANCED strategy
        decision_times = time_clean[np.array(xgb_te_idx, dtype=int)]
        forecast_series = pd.Series(xgb_pred / 10_000.0, index=decision_times)
        confidence_series = pd.Series(xgb_conf, index=decision_times)
        
        vol_series = ratio_returns_1m(ratio_clean)
        vol_series.index = time_clean
        
        tau_dynamic = adaptive_tau(vol_series, multiplier=VOL_MULTIPLIER)
        tau_base = strategy_params['tau_bps_base'] / 10_000.0
        tau_series = np.maximum(tau_base, tau_dynamic.reindex(decision_times)).ffill()
        tau_series = tau_series.fillna(tau_base) * strategy_params['tau_mult']
        
        # Generate BALANCED positions (never neutral)
        positions = generate_positions_balanced(
            forecast_series, confidence_series, tau_series,
            conf_thresh=strategy_params['prob_threshold'],
            min_hold=strategy_params['min_hold'],
            cooldown=strategy_params['cooldown']
        )
        
        # Calculate PnL
        fwd_log = future_k_bar_return(ratio_clean, k=K_AHEAD)
        fwd_log.index = time_clean
        pnl_components = pnl_from_positions(positions, fwd_log, vol_series)
        backtest_results = summarize_performance(pnl_components, positions, fwd_log)
        
        # Get metrics
        sharpe = backtest_results.get('Sharpe', -999)
        trade_count = backtest_results.get('TradeCount', 0)
        
        # Calculate balance
        long_pct = (positions == 1).sum() / len(positions)
        short_pct = (positions == -1).sum() / len(positions)
        
        # Penalties
        penalties = 0
        
        # 1. Balance penalty: Strongly penalize if not 40-60% long/short
        if long_pct < 0.35 or long_pct > 0.65:
            balance_penalty = abs(long_pct - 0.5) * 20  # Heavy penalty
            penalties += balance_penalty
        elif long_pct < 0.40 or long_pct > 0.60:
            balance_penalty = abs(long_pct - 0.5) * 10  # Moderate penalty
            penalties += balance_penalty
        
        # 2. Trade count penalty: Target 100-300 trades
        if trade_count < 50:
            trade_penalty = (50 - trade_count) / 50 * 2  # Too few trades
            penalties += trade_penalty
        elif trade_count > 500:
            trade_penalty = (trade_count - 500) / 500 * 2  # Too many trades
            penalties += trade_penalty
        
        # 3. Zero trading penalty
        if trade_count <= 1:
            penalties += 10  # Massive penalty for getting stuck
        
        final_score = sharpe - penalties
        
        # Log results
        trial.set_user_attr('Sharpe', sharpe)
        trial.set_user_attr('TotalPnL', backtest_results.get('TotalPnL', 0))
        trial.set_user_attr('TradeCount', trade_count)
        trial.set_user_attr('MaxDrawdown', backtest_results.get('MaxDrawdown', 0))
        trial.set_user_attr('HitRate', backtest_results.get('HitRate', 0))
        trial.set_user_attr('LongPct', long_pct * 100)
        trial.set_user_attr('ShortPct', short_pct * 100)
        trial.set_user_attr('Penalties', penalties)
        trial.set_user_attr('FinalScore', final_score)
        
        return final_score
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return -999.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=200,
                       help='Number of optimization trials (default: 200)')
    parser.add_argument('--objective', type=str, default='both', choices=['mse', 'pseudohuber', 'both'],
                       help='Which objective to test: mse, pseudohuber, or both')
    args = parser.parse_args()
    
    print("="*80)
    print("OPTUNA OPTIMIZATION: BALANCED LONG/SHORT STRATEGY")
    print("="*80)
    print(f"\n🎯 Goal: Find optimal hyperparameters for FORCED 50/50 long/short")
    print(f"📊 Testing: {args.objective.upper()}")
    print(f"🔬 Trials per objective: {args.trials}")
    print(f"\n⚠️  CRITICAL DIFFERENCE:")
    print("   v4.9.3: Allows neutral positions, goes neutral ~95% of time (Sharpe 3.85)")
    print("   v4.11: FORCES long/short 100% of time (requires different params)")
    print(f"\n📈 Target Metrics:")
    print("   • Long/Short Balance: 40-60% each")
    print("   • Trade Count: 100-300 (moderate frequency)")
    print("   • Sharpe: > 1.0 (positive risk-adjusted returns)")
    print("   • Max Drawdown: < -30%")
    print("="*80)
    
    # Load data
    print("\n📥 Loading data...")
    df = pd.read_parquet("data/processed/btc_eth_ratio_1m.parquet")
    X, y_smoothed, kept_idx, meta = build_features(df, k_ahead=K_AHEAD, label="bps")
    y_smoothed = y_smoothed.astype("float32")
    
    time_clean = as_dtindex(meta.get("time_clean"))
    ratio_clean = df.loc[kept_idx, "R"].reset_index(drop=True)
    
    print(f"✓ Data loaded: {X.shape[0]:,} samples, {X.shape[1]} features")
    
    objectives_to_test = []
    if args.objective in ['mse', 'both']:
        objectives_to_test.append(('MSE', False))
    if args.objective in ['pseudohuber', 'both']:
        objectives_to_test.append(('Pseudo-Huber', True))
    
    all_results = {}
    
    for obj_name, use_pseudohuber in objectives_to_test:
        print("\n" + "="*80)
        print(f"🚀 OPTIMIZING WITH {obj_name.upper()}")
        print("="*80)
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=15),
            study_name=f'v4_balanced_longshort_{obj_name.lower()}'
        )
        
        # Optimize
        print(f"\n⚡ Starting {args.trials} trials for {obj_name}...")
        print(f"⏱️  Estimated time: ~{args.trials * 2.5 / 60:.1f} hours")
        print("="*80 + "\n")
        
        study.optimize(
            lambda trial: objective(trial, X, y_smoothed, ratio_clean, time_clean, use_pseudohuber),
            n_trials=args.trials,
            show_progress_bar=True,
        )
        
        # Results
        print("\n" + "="*80)
        print(f"✅ {obj_name.upper()} OPTIMIZATION COMPLETE")
        print("="*80)
        
        best_trial = study.best_trial
        print(f"\n🏆 Best Trial: #{best_trial.number}")
        print(f"   Final Score: {best_trial.value:.4f} (Sharpe - Penalties)")
        print(f"   Sharpe: {best_trial.user_attrs.get('Sharpe', 0):.4f}")
        print(f"   Penalties: {best_trial.user_attrs.get('Penalties', 0):.4f}")
        print(f"   Total PnL: {best_trial.user_attrs.get('TotalPnL', 0):.2%}")
        print(f"   Trades: {best_trial.user_attrs.get('TradeCount', 0):.0f}")
        print(f"   Max DD: {best_trial.user_attrs.get('MaxDrawdown', 0):.2%}")
        print(f"   Hit Rate: {best_trial.user_attrs.get('HitRate', 0):.2%}")
        print(f"   Long%: {best_trial.user_attrs.get('LongPct', 0):.1f}% | Short%: {best_trial.user_attrs.get('ShortPct', 0):.1f}%")
        
        print(f"\n📊 Best Parameters:")
        for key, value in best_trial.params.items():
            print(f"   {key}: {value}")
        
        # Save results
        all_results[obj_name] = {
            'best_trial': best_trial.number,
            'best_score': best_trial.value,
            'best_params': best_trial.params,
            'best_metrics': best_trial.user_attrs,
            'objective': 'reg:pseudohubererror' if use_pseudohuber else 'reg:squarederror',
        }
        
        # Save study
        output_dir = Path(f"reports/v4_balanced_longshort_{obj_name.lower()}")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        study_df = study.trials_dataframe()
        study_df.to_csv(output_dir / 'all_trials.csv', index=False)
        print(f"\n✓ Trials saved to {output_dir}/all_trials.csv")
        
        # Top 5 trials
        print("\n🎯 TOP 5 TRIALS:")
        top_5 = study_df.nlargest(5, 'value')[['number', 'value', 'user_attrs_Sharpe', 'user_attrs_TradeCount', 'user_attrs_LongPct', 'user_attrs_ShortPct']]
        print(top_5.to_string(index=False))
    
    # Final comparison
    print("\n" + "="*80)
    print("🏁 FINAL COMPARISON")
    print("="*80)
    
    for obj_name, results in all_results.items():
        sharpe = results['best_metrics'].get('Sharpe', 0)
        trades = results['best_metrics'].get('TradeCount', 0)
        long_pct = results['best_metrics'].get('LongPct', 0)
        short_pct = results['best_metrics'].get('ShortPct', 0)
        print(f"\n{obj_name}:")
        print(f"  Sharpe: {sharpe:.2f}")
        print(f"  Trades: {trades:.0f}")
        print(f"  Balance: {long_pct:.1f}% long / {short_pct:.1f}% short")
    
    # Determine winner
    if len(all_results) == 2:
        mse_sharpe = all_results['MSE']['best_metrics'].get('Sharpe', -999)
        ph_sharpe = all_results['Pseudo-Huber']['best_metrics'].get('Sharpe', -999)
        
        print("\n" + "="*80)
        if ph_sharpe > mse_sharpe + 0.1:
            winner = 'Pseudo-Huber'
            diff = ph_sharpe - mse_sharpe
            print(f"🏆 WINNER: {winner} (+{diff:.2f} Sharpe)")
        elif mse_sharpe > ph_sharpe + 0.1:
            winner = 'MSE'
            diff = mse_sharpe - ph_sharpe
            print(f"🏆 WINNER: {winner} (+{diff:.2f} Sharpe)")
        else:
            winner = 'Tie'
            print(f"🤝 TIE: Both perform similarly (within 0.1 Sharpe)")
    else:
        winner = list(all_results.keys())[0]
    
    # Save combined results
    output_dir = Path("reports/v4_balanced_longshort_optimization")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    final_results = {
        'optimization_type': 'balanced_long_short',
        'date': pd.Timestamp.now().isoformat(),
        'trials_per_objective': args.trials,
        'winner': winner,
        'all_results': all_results,
        'comparison_with_v4_9_3': {
            'v4_9_3_sharpe': 3.85,
            'v4_9_3_strategy': 'Neutral-allowed (95% neutral)',
            'v4_11_strategy': 'Forced long/short (100% exposure)',
            'note': 'Lower Sharpe expected due to forced market exposure'
        }
    }
    
    with open(output_dir / 'optimization_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✓ Final results saved to {output_dir}/optimization_results.json")
    
    print("\n" + "="*80)
    print("🎉 OPTIMIZATION COMPLETE!")
    print("="*80)
    print(f"\n📝 Next steps:")
    print(f"1. Review best parameters in {output_dir}/optimization_results.json")
    print(f"2. Update scripts/run_v4_11_optimized.py with best params")
    print(f"3. Run backtest to validate performance")
    print("="*80)


if __name__ == "__main__":
    main()

