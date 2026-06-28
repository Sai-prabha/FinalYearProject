"""Data Quality Control for V4 Historical Data.

Checks and fixes:
1. Timestamp alignment between BTC and ETH
2. Missing values and gaps
3. Duplicate entries
4. Outliers and anomalies
5. Ratio calculation validation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime


def check_timestamps(df: pd.DataFrame, symbol: str) -> dict:
    """Check timestamp quality and consistency."""
    issues = {
        'symbol': symbol,
        'total_rows': len(df),
        'duplicates': df['open_time'].duplicated().sum(),
        'missing_timestamps': df['open_time'].isna().sum(),
        'out_of_order': (df['open_time'].diff() < pd.Timedelta(0)).sum(),
    }
    
    # Check for gaps (should be 1-minute intervals)
    df_sorted = df.sort_values('open_time')
    time_diffs = df_sorted['open_time'].diff()
    expected_diff = pd.Timedelta(minutes=1)
    
    gaps = time_diffs[time_diffs > expected_diff]
    issues['gaps_count'] = len(gaps)
    issues['max_gap_minutes'] = time_diffs.max().total_seconds() / 60 if len(time_diffs) > 0 else 0
    
    # Time range
    issues['start_time'] = df['open_time'].min()
    issues['end_time'] = df['open_time'].max()
    issues['time_span_days'] = (issues['end_time'] - issues['start_time']).days
    
    return issues


def check_values(df: pd.DataFrame, symbol: str) -> dict:
    """Check data value quality."""
    issues = {
        'symbol': symbol,
        'null_close': df['close'].isna().sum(),
        'null_volume': df['volume'].isna().sum(),
        'zero_volume': (df['volume'] == 0).sum(),
        'negative_close': (df['close'] <= 0).sum(),
        'negative_volume': (df['volume'] < 0).sum(),
    }
    
    # Outlier detection (>5 std from mean for returns)
    if 'close' in df.columns and len(df) > 1:
        returns = df['close'].pct_change()
        mean_ret = returns.mean()
        std_ret = returns.std()
        outliers = np.abs(returns - mean_ret) > 5 * std_ret
        issues['outlier_returns'] = outliers.sum()
        issues['max_return'] = returns.max()
        issues['min_return'] = returns.min()
    
    return issues


def align_timestamps(btc_df: pd.DataFrame, eth_df: pd.DataFrame) -> tuple:
    """Align BTC and ETH timestamps to ensure matching."""
    print("\n" + "="*80)
    print("TIMESTAMP ALIGNMENT")
    print("="*80)
    
    print(f"Before alignment:")
    print(f"  BTC rows: {len(btc_df):,}")
    print(f"  ETH rows: {len(eth_df):,}")
    
    # Ensure timestamps are datetime
    btc_df['open_time'] = pd.to_datetime(btc_df['open_time'])
    eth_df['open_time'] = pd.to_datetime(eth_df['open_time'])
    
    # Remove duplicates
    btc_df = btc_df.drop_duplicates(subset=['open_time'], keep='first')
    eth_df = eth_df.drop_duplicates(subset=['open_time'], keep='first')
    
    print(f"\nAfter removing duplicates:")
    print(f"  BTC rows: {len(btc_df):,}")
    print(f"  ETH rows: {len(eth_df):,}")
    
    # Sort by time
    btc_df = btc_df.sort_values('open_time').reset_index(drop=True)
    eth_df = eth_df.sort_values('open_time').reset_index(drop=True)
    
    # Find common timestamps
    common_times = pd.Index(btc_df['open_time']).intersection(pd.Index(eth_df['open_time']))
    print(f"\nCommon timestamps: {len(common_times):,}")
    
    # Keep only common timestamps
    btc_aligned = btc_df[btc_df['open_time'].isin(common_times)].reset_index(drop=True)
    eth_aligned = eth_df[eth_df['open_time'].isin(common_times)].reset_index(drop=True)
    
    print(f"\nAfter alignment:")
    print(f"  BTC rows: {len(btc_aligned):,}")
    print(f"  ETH rows: {len(eth_aligned):,}")
    
    # Verify alignment
    assert len(btc_aligned) == len(eth_aligned), "Length mismatch after alignment"
    assert (btc_aligned['open_time'].values == eth_aligned['open_time'].values).all(), "Timestamps not aligned"
    
    print("✓ Timestamps perfectly aligned")
    
    return btc_aligned, eth_aligned


def validate_ratio(btc_close: pd.Series, eth_close: pd.Series) -> dict:
    """Validate ratio calculations."""
    ratio = btc_close / eth_close
    
    issues = {
        'null_ratios': ratio.isna().sum(),
        'inf_ratios': np.isinf(ratio).sum(),
        'mean_ratio': ratio.mean(),
        'std_ratio': ratio.std(),
        'min_ratio': ratio.min(),
        'max_ratio': ratio.max(),
    }
    
    # Check for suspicious values
    issues['suspiciously_low'] = (ratio < ratio.quantile(0.001)).sum()
    issues['suspiciously_high'] = (ratio > ratio.quantile(0.999)).sum()
    
    return issues


def main():
    print("="*80)
    print("DATA QUALITY CONTROL FOR V4 MODELS")
    print("="*80)
    
    data_dir = Path("data/legacy_v4/binance_spot")
    
    # Load raw data
    print("\nLoading raw data...")
    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'qvol', 'ntrades', 'taker_base', 'taker_quote', 'ignore']
    
    btc_df = pd.read_csv(data_dir / "BTCFDUSD.csv", header=None, names=cols)
    eth_df = pd.read_csv(data_dir / "ETHFDUSD.csv", header=None, names=cols)
    
    btc_df['open_time'] = pd.to_datetime(btc_df['open_time'], unit='ms', utc=True).dt.tz_convert(None)
    eth_df['open_time'] = pd.to_datetime(eth_df['open_time'], unit='ms', utc=True).dt.tz_convert(None)
    
    print(f"✓ Loaded BTC: {len(btc_df):,} rows")
    print(f"✓ Loaded ETH: {len(eth_df):,} rows")
    
    # Check timestamps
    print("\n" + "="*80)
    print("TIMESTAMP QUALITY CHECK")
    print("="*80)
    
    btc_ts_issues = check_timestamps(btc_df, "BTCFDUSD")
    eth_ts_issues = check_timestamps(eth_df, "ETHFDUSD")
    
    print("\nBTC Timestamp Issues:")
    for key, value in btc_ts_issues.items():
        print(f"  {key}: {value}")
    
    print("\nETH Timestamp Issues:")
    for key, value in eth_ts_issues.items():
        print(f"  {key}: {value}")
    
    # Check values
    print("\n" + "="*80)
    print("VALUE QUALITY CHECK")
    print("="*80)
    
    btc_val_issues = check_values(btc_df, "BTCFDUSD")
    eth_val_issues = check_values(eth_df, "ETHFDUSD")
    
    print("\nBTC Value Issues:")
    for key, value in btc_val_issues.items():
        print(f"  {key}: {value}")
    
    print("\nETH Value Issues:")
    for key, value in eth_val_issues.items():
        print(f"  {key}: {value}")
    
    # Align timestamps
    btc_clean, eth_clean = align_timestamps(btc_df, eth_df)
    
    # Validate ratio
    print("\n" + "="*80)
    print("RATIO VALIDATION")
    print("="*80)
    
    ratio_issues = validate_ratio(btc_clean['close'], eth_clean['close'])
    
    print("\nRatio Statistics:")
    for key, value in ratio_issues.items():
        print(f"  {key}: {value}")
    
    # Clean data
    print("\n" + "="*80)
    print("DATA CLEANING")
    print("="*80)
    
    # Remove rows with null or invalid values
    initial_len = len(btc_clean)
    
    # Filter out invalid closes
    valid_mask = (
        (btc_clean['close'] > 0) &
        (eth_clean['close'] > 0) &
        (btc_clean['volume'] >= 0) &
        (eth_clean['volume'] >= 0) &
        (btc_clean['close'].notna()) &
        (eth_clean['close'].notna())
    )
    
    btc_clean = btc_clean[valid_mask].reset_index(drop=True)
    eth_clean = eth_clean[valid_mask].reset_index(drop=True)
    
    removed = initial_len - len(btc_clean)
    print(f"✓ Removed {removed} invalid rows ({removed/initial_len*100:.2f}%)")
    print(f"✓ Final clean data: {len(btc_clean):,} rows")
    
    # Save cleaned data
    print("\n" + "="*80)
    print("SAVING CLEANED DATA")
    print("="*80)
    
    output_dir = Path("data/legacy_v4/binance_spot")
    
    # Save cleaned CSVs
    btc_clean.to_csv(output_dir / "BTCFDUSD_clean.csv", index=False)
    eth_clean.to_csv(output_dir / "ETHFDUSD_clean.csv", index=False)
    
    print(f"✓ Saved cleaned BTC data: {output_dir / 'BTCFDUSD_clean.csv'}")
    print(f"✓ Saved cleaned ETH data: {output_dir / 'ETHFDUSD_clean.csv'}")
    
    # Generate quality report
    report = {
        'timestamp': datetime.now().isoformat(),
        'original_btc_rows': len(btc_df),
        'original_eth_rows': len(eth_df),
        'final_rows': len(btc_clean),
        'rows_removed': removed,
        'removal_rate_pct': removed/initial_len*100,
        'btc_timestamp_issues': btc_ts_issues,
        'eth_timestamp_issues': eth_ts_issues,
        'btc_value_issues': btc_val_issues,
        'eth_value_issues': eth_val_issues,
        'ratio_validation': ratio_issues,
    }
    
    import json
    with open(output_dir.parent.parent / "reports" / "data_quality_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✓ Quality report saved")
    
    print("\n" + "="*80)
    print("DATA QUALITY CONTROL COMPLETE")
    print("="*80)
    print("\n✅ Data is now clean, aligned, and validated")
    print(f"✅ Ready for model training with {len(btc_clean):,} high-quality samples")


if __name__ == "__main__":
    main()

