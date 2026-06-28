#!/usr/bin/env python3
"""
Generate Final V4.12 Optimization Report

Analyzes all optimization results and creates comprehensive report.
Run this after optimization completes.

Usage:
    python scripts/generate_final_report.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_results(jsonl_path='reports/v4_12_optimization_log.jsonl'):
    """Load all optimization results."""
    results = []
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    except FileNotFoundError:
        print(f"❌ Results file not found: {jsonl_path}")
        return []
    return results


def print_report(results):
    """Print comprehensive final report."""
    
    print("\n" + "="*80)
    print("V4.12 OPTIMIZATION - FINAL REPORT")
    print("="*80)
    print(f"Generated: {datetime.now().isoformat()}")
    print(f"Total Iterations: {len(results)}")
    print("="*80 + "\n")
    
    # Summary table
    print("📊 COMPLETE RESULTS SUMMARY")
    print("="*80)
    print(f"{'Iter':<6} {'Configuration':<30} {'Sharpe':<10} {'Trades':<8} {'MaxDD':<10} {'Overfit':<8}")
    print("-"*80)
    
    best_sharpe = 0
    best_config = None
    
    for result in results:
        iter_num = result['iteration']
        name = result['config_name']
        sharpe = result['backtest_metrics'].get('Sharpe', 0)
        trades = result['backtest_metrics'].get('TradeCount', 0)
        max_dd = result['backtest_metrics'].get('MaxDrawdown', 0) * 100
        overfit = result['overfitting_check'].get('is_overfitting', False) if result['overfitting_check'] else None
        overfit_str = "YES⚠️" if overfit else "NO✅" if overfit is not None else "N/A"
        
        print(f"{iter_num:<6} {name:<30} {sharpe:<10.4f} {trades:<8} {max_dd:<9.2f}% {overfit_str:<8}")
        
        # Track best (excluding overfitted ones)
        if sharpe > best_sharpe and not overfit:
            best_sharpe = sharpe
            best_config = result
    
    print("="*80 + "\n")
    
    # Best configuration details
    if best_config:
        print("🏆 BEST CONFIGURATION (No Overfitting)")
        print("="*80)
        print(f"Name: {best_config['config_name']}")
        print(f"Iteration: {best_config['iteration']}")
        print(f"\n📊 XGBoost Parameters:")
        for key, value in best_config['xgb_params'].items():
            print(f"   {key:<20s} {value}")
        
        print(f"\n🎯 Strategy Parameters:")
        for key, value in best_config['strategy_params'].items():
            print(f"   {key:<20s} {value}")
        
        print(f"\n💰 Performance Metrics:")
        metrics = best_config['backtest_metrics']
        print(f"   Sharpe Ratio:       {metrics.get('Sharpe', 0):.4f} ⭐")
        print(f"   Sortino Ratio:      {metrics.get('Sortino', 0):.4f}")
        print(f"   Total PnL:          {metrics.get('TotalPnL', 0)*100:.2f}%")
        print(f"   Max Drawdown:       {metrics.get('MaxDrawdown', 0)*100:.2f}%")
        print(f"   Trade Count:        {metrics.get('TradeCount', 0)}")
        print(f"   Hit Rate:           {metrics.get('HitRate', 0):.1%}")
        print(f"   Win Rate:           {metrics.get('WinRate', 0):.1%}")
        print(f"   Profit Factor:      {metrics.get('ProfitFactor', 0):.2f}")
        
        if best_config['overfitting_check']:
            ov = best_config['overfitting_check']
            print(f"\n🔍 Overfitting Check:")
            print(f"   Train RMSE:         {ov.get('train_rmse', 0):.2f}")
            print(f"   Val RMSE:           {ov.get('val_rmse', 0):.2f}")
            print(f"   RMSE Ratio:         {ov.get('rmse_ratio', 0):.2f} {'⚠️' if ov.get('rmse_ratio', 0) > 1.3 else '✅'}")
            print(f"   Train Hit:          {ov.get('train_hit', 0):.1%}")
            print(f"   Val Hit:            {ov.get('val_hit', 0):.1%}")
            print(f"   Hit Difference:     {ov.get('hit_diff', 0):+.1%} {'⚠️' if ov.get('hit_diff', 0) > 0.10 else '✅'}")
        
        print("="*80 + "\n")
    
    # Comparison with Trial 7
    trial7 = next((r for r in results if 'trial7' in r['config_name'].lower()), None)
    if trial7 and best_config:
        print("📈 IMPROVEMENT OVER TRIAL 7")
        print("="*80)
        trial7_sharpe = trial7['backtest_metrics'].get('Sharpe', 0)
        best_sharpe_val = best_config['backtest_metrics'].get('Sharpe', 0)
        improvement = ((best_sharpe_val - trial7_sharpe) / trial7_sharpe * 100) if trial7_sharpe > 0 else 0
        
        trial7_trades = trial7['backtest_metrics'].get('TradeCount', 0)
        best_trades = best_config['backtest_metrics'].get('TradeCount', 0)
        
        print(f"Sharpe Ratio:")
        print(f"   Trial 7:            {trial7_sharpe:.4f}")
        print(f"   Best V4.12:         {best_sharpe_val:.4f}")
        print(f"   Improvement:        {improvement:+.1f}%")
        print(f"\nTrade Activity:")
        print(f"   Trial 7:            {trial7_trades} trades")
        print(f"   Best V4.12:         {best_trades} trades")
        print(f"   Increase:           {best_trades - trial7_trades:+d} trades")
        print("="*80 + "\n")
    
    # Key insights
    print("💡 KEY INSIGHTS")
    print("="*80)
    
    # Find patterns
    depth_5_results = [r for r in results if r['xgb_params'].get('max_depth') == 5]
    depth_4_results = [r for r in results if r['xgb_params'].get('max_depth') == 4]
    
    if depth_5_results and depth_4_results:
        avg_sharpe_5 = sum(r['backtest_metrics'].get('Sharpe', 0) for r in depth_5_results) / len(depth_5_results)
        avg_sharpe_4 = sum(r['backtest_metrics'].get('Sharpe', 0) for r in depth_4_results) / len(depth_4_results)
        print(f"1. max_depth=5 avg Sharpe: {avg_sharpe_5:.4f}")
        print(f"   max_depth=4 avg Sharpe: {avg_sharpe_4:.4f}")
        if avg_sharpe_5 > avg_sharpe_4:
            print(f"   → Deeper trees (5) performed better by {((avg_sharpe_5/avg_sharpe_4 - 1)*100):.1f}%")
        else:
            print(f"   → Shallower trees (4) performed better by {((avg_sharpe_4/avg_sharpe_5 - 1)*100):.1f}%")
    
    # Check overfitting
    overfit_count = sum(1 for r in results if r['overfitting_check'] and r['overfitting_check'].get('is_overfitting', False))
    print(f"\n2. Overfitting detected in {overfit_count}/{len(results)} configurations")
    if overfit_count > 0:
        print(f"   → Strong regularization is essential")
    else:
        print(f"   → All configurations generalize well ✅")
    
    # Trade frequency
    low_trade_configs = [r for r in results if r['backtest_metrics'].get('TradeCount', 0) < 100]
    high_trade_configs = [r for r in results if r['backtest_metrics'].get('TradeCount', 0) >= 100]
    
    if low_trade_configs:
        avg_sharpe_low = sum(r['backtest_metrics'].get('Sharpe', 0) for r in low_trade_configs) / len(low_trade_configs)
        print(f"\n3. Low trade configs (<100 trades) avg Sharpe: {avg_sharpe_low:.4f}")
    if high_trade_configs:
        avg_sharpe_high = sum(r['backtest_metrics'].get('Sharpe', 0) for r in high_trade_configs) / len(high_trade_configs)
        print(f"   High trade configs (≥100 trades) avg Sharpe: {avg_sharpe_high:.4f}")
        if high_trade_configs and low_trade_configs:
            if avg_sharpe_high > avg_sharpe_low:
                print(f"   → More active trading improves Sharpe ✅")
            else:
                print(f"   → Quality over quantity - fewer trades performed better")
    
    print("="*80 + "\n")
    
    # Recommendations
    print("🎯 RECOMMENDATIONS")
    print("="*80)
    
    if best_sharpe >= 3.0:
        print("✅ EXCELLENT RESULTS - Ready for production deployment!")
        print(f"   Achieved Sharpe {best_sharpe:.4f} ≥ 3.0 target")
        print(f"   Deploy: {best_config['config_name']}")
    elif best_sharpe >= 2.0:
        print("✅ GOOD RESULTS - Strong performance, ready for testing")
        print(f"   Achieved Sharpe {best_sharpe:.4f} ≥ 2.0 realistic target")
        print(f"   Recommend: Forward test {best_config['config_name']} on new data")
    elif best_sharpe >= 1.5:
        print("⚠️  MODERATE RESULTS - Better than baseline")
        print(f"   Achieved Sharpe {best_sharpe:.4f}")
        print(f"   Consider: Additional parameter tuning or ensemble approaches")
    else:
        print("⚠️  LIMITED IMPROVEMENT - Further work needed")
        print(f"   Achieved Sharpe {best_sharpe:.4f}")
        print(f"   Recommend: Revisit feature engineering or try different models")
    
    print("\nNext Steps:")
    if best_config:
        print(f"1. Save best config as production V4.12: {best_config['config_name']}")
        print(f"2. Review reports in: reports/4.12/xgboost/")
        print(f"3. Examine equity curve and trade breakdown")
        print(f"4. Run forward test on most recent data")
        print(f"5. Deploy if forward test confirms performance")
    
    print("="*80 + "\n")
    
    # Save best config
    if best_config:
        best_config_file = Path('reports/v4_12_best_config.json')
        with open(best_config_file, 'w') as f:
            json.dump(best_config, f, indent=2)
        print(f"💾 Best configuration saved to: {best_config_file}\n")


def main():
    results = load_results()
    
    if not results:
        print("❌ No results found. Optimization may not have completed yet.")
        print("\nCheck status with: bash scripts/monitor_v4_12_optimization.sh")
        return
    
    print(f"\n✅ Found {len(results)} completed iterations\n")
    print_report(results)
    
    print("\n" + "="*80)
    print("Report generation complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

