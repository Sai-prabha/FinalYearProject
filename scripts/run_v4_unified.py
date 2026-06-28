#!/usr/bin/env python3
"""
Unified V4 Runner Script

This script provides a single entry point to run any V4 strategy version
through the unified pipeline.

Usage:
    python scripts/run_v4_unified.py v4.10
    python scripts/run_v4_unified.py v4.11
    python scripts/run_v4_unified.py v4.11_original
    python scripts/run_v4_unified.py v4.11_adjusted
"""

import sys
from pathlib import Path

# Add project root to path (not src/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.v4_config import list_versions
from src.v4_pipeline import run_v4_version


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_v4_unified.py <version>")
        print(f"\nAvailable versions: {', '.join(list_versions())}")
        sys.exit(1)
    
    version = sys.argv[1]
    
    try:
        results = run_v4_version(version)
        
        # Print summary
        print("\n" + "="*80)
        print("EXECUTION SUMMARY")
        print("="*80)
        print(f"\n✅ Successfully completed {version}")
        print(f"\n📊 Model Performance:")
        print(f"   RMSE: {results['model_metrics']['RMSE']:.2f}")
        print(f"   MAE: {results['model_metrics']['MAE']:.2f}")
        print(f"   Hit Rate: {results['model_metrics']['HitRate']:.1%}")
        print(f"\n💰 Backtest Performance:")
        print(f"   Sharpe: {results['backtest_metrics']['Sharpe']:.2f}")
        print(f"   Total PnL: {results['backtest_metrics']['TotalPnL']*100:.1f}%")
        print(f"   Max Drawdown: {results['backtest_metrics']['MaxDrawdown']*100:.1f}%")
        print(f"   Trade Count: {results['backtest_metrics']['TradeCount']}")
        print(f"   Long/Short/Neutral: {results['backtest_metrics'].get('LongTime%', 0):.1f}% / "
              f"{results['backtest_metrics'].get('ShortTime%', 0):.1f}% / "
              f"{results['backtest_metrics'].get('NeutralTime%', 100):.1f}%")
        print()
        
    except Exception as e:
        print(f"\n❌ Error running {version}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

