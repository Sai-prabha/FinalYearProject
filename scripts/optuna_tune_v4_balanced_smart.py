"""Optuna optimization for SMART Balanced Long/Short Strategy.

KEY DIFFERENCE from forced approach:
- Allows neutral positions (don't force bad trades)
- Encourages balanced long/short exposure WHEN TRADING
- Optimizes for high Sharpe and low drawdown
- Only trade when conditions are favorable

Target: Like v4.9.3's selectivity (high Sharpe) but with balanced long/short
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
    adaptive_tau, future_k_bar_return, generate_positions,
    pnl_from_positions, ratio_returns_1m, summarize_performance,
)
from src.utils import as_dtindex


K_AHEAD = 10
VOL_MULTIPLIER = 2.5


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series,
              ratio_clean: pd.Series, time_clean: pd.DatetimeIndex, 
              use_pseudohuber: bool) -> float:
    """Optimization for smart balanced strategy - neutral allowed but balanced when trading."""
    
    # XGBoost hyperparameters
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 600),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.3, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
        'subsample': trial.suggest_float('subsample', 0.65, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.55, 0.95),
        'gamma': trial.suggest_float('gamma', 0.0, 7.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 40.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 25.0, log=True),
    }
    
    if use_pseudohuber:
        xgb_params['huber_slope'] = trial.suggest_float('huber_slope', 0.3, 2.5)
    
    # Strategy parameters - allow neutral but encourage good trades
    strategy_params = {
        'prob_threshold': trial.suggest_float('prob_threshold', 0.50, 0.75),
        'cooldown': trial.suggest_int('cooldown', 3, 30),
        'tau_mult': trial.suggest_float('tau_mult', 0.8, 3.5),
        'tau_bps_base': trial.suggest_float('tau_bps_base', 0.5, 5.0),
        'min_hold': trial.suggest_int('min_hold', 3, 60),
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
        
        # Backtest with STANDARD strategy (allows neutral)
        decision_times = time_clean[np.array(xgb_te_idx, dtype=int)]
        forecast_series = pd.Series(xgb_pred / 10_000.0, index=decision_times)
        confidence_series = pd.Series(xgb_conf, index=decision_times)
        
        vol_series = ratio_returns_1m(ratio_clean)
        vol_series.index = time_clean
        
        tau_dynamic = adaptive_tau(vol_series, multiplier=VOL_MULTIPLIER)
        tau_base = strategy_params['tau_bps_base'] / 10_000.0
        tau_series = np.maximum(tau_base, tau_dynamic.reindex(decision_times)).ffill()
        tau_series = tau_series.fillna(tau_base) * strategy_params['tau_mult']
        
        # Use STANDARD generate_positions (allows neutral)
        positions = generate_positions(
            forecast_series, confidence_series, tau_series,
            prob_threshold=strategy_params['prob_threshold'],
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
        max_dd = backtest_results.get('MaxDrawdown', 0)
        trade_count = backtest_results.get('TradeCount', 0)
        total_pnl = backtest_results.get('TotalPnL', 0)
        
        # Calculate position statistics
        long_mask = positions == 1
        short_mask = positions == -1
        neutral_mask = positions == 0
        
        long_pct = long_mask.sum() / len(positions)
        short_pct = short_mask.sum() / len(positions)
        neutral_pct = neutral_mask.sum() / len(positions)
        
        # Calculate balance WHEN TRADING (excluding neutral)
        trading_time = long_pct + short_pct
        if trading_time > 0.01:  # At least 1% time trading
            long_ratio_when_trading = long_pct / trading_time
        else:
            long_ratio_when_trading = 0.5  # No penalty if not trading
        
        # Penalties and bonuses
        penalties = 0
        
        # 1. Imbalance penalty: Penalize if long/short is too skewed WHEN TRADING
        # Want 40-60% long when trading, allow neutral anytime
        if trading_time > 0.05:  # Only if trading >5% of time
            if long_ratio_when_trading < 0.25 or long_ratio_when_trading > 0.75:
                # Heavily skewed - moderate penalty
                imbalance = abs(long_ratio_when_trading - 0.5)
                penalties += imbalance * 3
            elif long_ratio_when_trading < 0.35 or long_ratio_when_trading > 0.65:
                # Somewhat skewed - small penalty
                imbalance = abs(long_ratio_when_trading - 0.5)
                penalties += imbalance * 1
        
        # 2. Drawdown penalty: Penalize large drawdowns
        if max_dd < -0.30:  # Worse than -30%
            dd_penalty = abs(max_dd + 0.30) * 10
            penalties += dd_penalty
        
        # 3. No trading penalty: Discourage completely avoiding trades
        if trade_count < 5:
            penalties += (5 - trade_count) * 0.5
        
        # 4. Overtrading penalty: Discourage excessive trading
        if trade_count > 1000:
            penalties += (trade_count - 1000) / 1000 * 2
        
        # Bonuses
        bonuses = 0
        
        # 1. Selectivity bonus: Reward trading only when confident
        if 10 <= trade_count <= 200:
            bonuses += 0.5  # Good trade count
        
        # 2. Low drawdown bonus
        if max_dd > -0.15:  # Better than -15%
            bonuses += 1.0
        
        # 3. Positive PnL bonus
        if total_pnl > 0.05:  # More than 5% return
            bonuses += 0.5
        
        final_score = sharpe + bonuses - penalties
        
        # Log comprehensive results
        trial.set_user_attr('Sharpe', sharpe)
        trial.set_user_attr('TotalPnL', total_pnl)
        trial.set_user_attr('TradeCount', trade_count)
        trial.set_user_attr('MaxDrawdown', max_dd)
        trial.set_user_attr('HitRate', backtest_results.get('HitRate', 0))
        trial.set_user_attr('LongPct', long_pct * 100)
        trial.set_user_attr('ShortPct', short_pct * 100)
        trial.set_user_attr('NeutralPct', neutral_pct * 100)
        trial.set_user_attr('LongRatioWhenTrading', long_ratio_when_trading * 100)
        trial.set_user_attr('Penalties', penalties)
        trial.set_user_attr('Bonuses', bonuses)
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
                       help='Which objective to test')
    args = parser.parse_args()
    
    print("="*80)
    print("OPTUNA OPTIMIZATION: SMART BALANCED LONG/SHORT STRATEGY")
    print("="*80)
    print(f"\n🎯 Goal: High Sharpe + Low DD + Balanced long/short WHEN TRADING")
    print(f"📊 Testing: {args.objective.upper()}")
    print(f"🔬 Trials per objective: {args.trials}")
    print(f"\n💡 KEY STRATEGY:")
    print("   ✅ Allow neutral positions (don't force bad trades)")
    print("   ✅ Trade long AND short (not just one direction)")
    print("   ✅ Balanced when trading (40-60% long/short ratio)")
    print("   ✅ High Sharpe through selectivity")
    print("   ✅ Low drawdown through risk control")
    print(f"\n📈 Target Metrics:")
    print("   • Sharpe: > 2.5 (high risk-adjusted returns)")
    print("   • Max Drawdown: < -20% (controlled risk)")
    print("   • Trade Count: 20-200 (selective but active)")
    print("   • Long/Short when trading: 40-60% each")
    print("   • Can be neutral when uncertain")
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
            study_name=f'v4_smart_balanced_{obj_name.lower()}'
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
        print(f"   Final Score: {best_trial.value:.4f}")
        print(f"   Sharpe: {best_trial.user_attrs.get('Sharpe', 0):.4f}")
        print(f"   Total PnL: {best_trial.user_attrs.get('TotalPnL', 0):.2%}")
        print(f"   Max DD: {best_trial.user_attrs.get('MaxDrawdown', 0):.2%}")
        print(f"   Trades: {best_trial.user_attrs.get('TradeCount', 0):.0f}")
        print(f"   Hit Rate: {best_trial.user_attrs.get('HitRate', 0):.2%}")
        print(f"   Long: {best_trial.user_attrs.get('LongPct', 0):.1f}% | Short: {best_trial.user_attrs.get('ShortPct', 0):.1f}% | Neutral: {best_trial.user_attrs.get('NeutralPct', 0):.1f}%")
        print(f"   Long ratio when trading: {best_trial.user_attrs.get('LongRatioWhenTrading', 0):.1f}%")
        
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
        output_dir = Path(f"reports/v4_smart_balanced_{obj_name.lower()}")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        study_df = study.trials_dataframe()
        study_df.to_csv(output_dir / 'all_trials.csv', index=False)
        print(f"\n✓ Trials saved to {output_dir}/all_trials.csv")
        
        # Top 5 trials
        print("\n🎯 TOP 5 TRIALS:")
        top_5 = study_df.nlargest(5, 'value')[['number', 'value', 'user_attrs_Sharpe', 'user_attrs_MaxDrawdown', 'user_attrs_TradeCount']]
        print(top_5.to_string(index=False))
    
    # Final comparison
    print("\n" + "="*80)
    print("🏁 FINAL COMPARISON")
    print("="*80)
    
    for obj_name, results in all_results.items():
        sharpe = results['best_metrics'].get('Sharpe', 0)
        pnl = results['best_metrics'].get('TotalPnL', 0)
        dd = results['best_metrics'].get('MaxDrawdown', 0)
        trades = results['best_metrics'].get('TradeCount', 0)
        long_ratio = results['best_metrics'].get('LongRatioWhenTrading', 0)
        print(f"\n{obj_name}:")
        print(f"  Sharpe: {sharpe:.2f} | PnL: {pnl:.1%} | DD: {dd:.1%}")
        print(f"  Trades: {trades:.0f} | Long ratio when trading: {long_ratio:.1f}%")
    
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
            print(f"🤝 TIE: Both perform similarly")
    else:
        winner = list(all_results.keys())[0]
    
    # Save combined results
    output_dir = Path("reports/v4_smart_balanced_optimization")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    final_results = {
        'optimization_type': 'smart_balanced_long_short',
        'date': pd.Timestamp.now().isoformat(),
        'trials_per_objective': args.trials,
        'winner': winner,
        'all_results': all_results,
        'strategy': {
            'allows_neutral': True,
            'encourages_balance_when_trading': True,
            'target_sharpe': '>2.5',
            'target_drawdown': '<-20%',
            'target_trades': '20-200'
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
    print(f"3. Run backtest to validate")
    print("="*80)


if __name__ == "__main__":
    main()

