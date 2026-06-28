#!/usr/bin/env python3
"""
Simple diagnostic: Are V4.12 predictions too weak?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from src.features import build_features
from src.v4_config import V4_REGISTRY

print("\n" + "="*80)
print("SIMPLE PREDICTION STRENGTH DIAGNOSTIC")
print("="*80)

# Load data
print("\n📊 Loading data...")
df = pd.read_parquet("data/processed/btc_eth_ratio_1m.parquet")
X_all, y_all, kept_idx, meta = build_features(df, k_ahead=4, label="bps")
y_all = y_all.astype("float32")

# Split
split_idx = int(len(X_all) * 0.8)
X_train, X_test = X_all[:split_idx], X_all[split_idx:]
y_train, y_test = y_all[:split_idx], y_all[split_idx:]

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

# Test V4.12 Conservative
print("\n" + "="*80)
print("V4.12 CONSERVATIVE")
print("="*80)

v4_config = V4_REGISTRY['v4.12_conservative']
xgb_params = v4_config.xgb_params.to_dict()
strategy_params = v4_config.strategy_params.to_dict()

# Train
print("\n🚀 Training...")
model = XGBRegressor(**xgb_params, random_state=42)
model.fit(X_train, y_train)

# Predict
print("🔮 Predicting...")
y_pred = model.predict(X_test)

# Analyze predictions
print(f"\n📊 PREDICTION ANALYSIS:")
print(f"\nForecast distribution (bps):")
print(f"  Min:     {y_pred.min():8.4f}")
print(f"  25%:     {np.percentile(y_pred, 25):8.4f}")
print(f"  Median:  {np.median(y_pred):8.4f}")
print(f"  75%:     {np.percentile(y_pred, 75):8.4f}")
print(f"  Max:     {y_pred.max():8.4f}")
print(f"  Mean:    {y_pred.mean():8.4f}")
print(f"  Std:     {y_pred.std():8.4f}")

# Simulate confidence as abs(prediction)
confidence = np.abs(y_pred)
conf_normalized = confidence / confidence.max()

print(f"\nConfidence (abs value):")
print(f"  Min:     {conf_normalized.min():8.4f}")
print(f"  25%:     {np.percentile(conf_normalized, 25):8.4f}")
print(f"  Median:  {np.median(conf_normalized):8.4f}")
print(f"  75%:     {np.percentile(conf_normalized, 75):8.4f}")
print(f"  Max:     {conf_normalized.max():8.4f}")
print(f"  Mean:    {conf_normalized.mean():8.4f}")

# Test different thresholds
print(f"\n📈 THRESHOLD ANALYSIS:")
prob_threshold = strategy_params.get('prob_threshold', 0.55)
print(f"\nCurrent threshold: {prob_threshold:.2f}")

for thresh in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
    above = (conf_normalized > thresh).sum()
    pct = 100.0 * above / len(conf_normalized)
    marker = " ← CURRENT" if abs(thresh - prob_threshold) < 0.01 else ""
    marker = " ← OPTUNA MIN" if abs(thresh - 0.48) < 0.01 else marker
    print(f"  {thresh:.2f}: {above:6,} / {len(conf_normalized):,} = {pct:5.2f}%{marker}")

# Check how many strong signals
strong_signals = (conf_normalized > 0.55).sum()
print(f"\n💡 INSIGHT:")
print(f"  Only {strong_signals:,} / {len(conf_normalized):,} predictions ({100.0*strong_signals/len(conf_normalized):.2f}%) exceed current threshold of 0.55")

if strong_signals < len(conf_normalized) * 0.10:
    print(f"  ❌ Model predictions are TOO WEAK!")
    print(f"  ✅ Recommendation: Lower prob_threshold to 0.35-0.45")
else:
    print(f"  ✅ Reasonable signal strength")

# Compare to a simpler model
print(f"\n" + "="*80)
print("COMPARISON: LESS REGULARIZED MODEL")
print("="*80)

# Reduce regularization
xgb_params_loose = xgb_params.copy()
xgb_params_loose['max_depth'] = 6  # vs 4
xgb_params_loose['learning_rate'] = 0.08  # vs 0.04
xgb_params_loose['gamma'] = 0.1  # vs 0.5
xgb_params_loose['reg_alpha'] = 0.1  # vs 0.5
xgb_params_loose['reg_lambda'] = 0.5  # vs 3.0

print("\n🚀 Training less regularized model...")
model_loose = XGBRegressor(**xgb_params_loose, random_state=42)
model_loose.fit(X_train, y_train)

print("🔮 Predicting...")
y_pred_loose = model_loose.predict(X_test)

confidence_loose = np.abs(y_pred_loose)
conf_norm_loose = confidence_loose / confidence_loose.max()

print(f"\nConfidence distribution:")
print(f"  Mean:    {conf_norm_loose.mean():8.4f}  (vs {conf_normalized.mean():.4f} conservative)")
print(f"  Median:  {np.median(conf_norm_loose):8.4f}  (vs {np.median(conf_normalized):.4f} conservative)")
print(f"  Std:     {conf_norm_loose.std():8.4f}  (vs {conf_normalized.std():.4f} conservative)")

strong_loose = (conf_norm_loose > 0.55).sum()
pct_loose = 100.0 * strong_loose / len(conf_norm_loose)
pct_conservative = 100.0 * strong_signals / len(conf_normalized)

print(f"\nStrong signals (> 0.55):")
print(f"  Less regularized:  {strong_loose:6,} ({pct_loose:5.2f}%)")
print(f"  Conservative:      {strong_signals:6,} ({pct_conservative:5.2f}%)")

if pct_loose > pct_conservative * 2:
    print(f"\n💡 INSIGHT: Reducing regularization increases strong signals by {pct_loose/pct_conservative:.1f}x!")
    print(f"   Consider using less aggressive regularization.")

print(f"\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n🎯 ROOT CAUSE:")
if strong_signals < 5000:
    print(f"  ❌ V4.12 Conservative model predictions are TOO WEAK")
    print(f"  ❌ Only {strong_signals:,} strong signals in {len(conf_normalized):,} predictions")
    print(f"  ❌ This explains why Optuna found ZERO valid trials!")
    
    print(f"\n💊 RECOMMENDED FIX:")
    print(f"  Option 1: Lower prob_threshold from 0.55 to 0.35-0.40")
    print(f"  Option 2: Reduce regularization (gamma, reg_alpha, reg_lambda)")
    print(f"  Option 3: Increase max_depth from 4 to 5-6")
    print(f"  Option 4: Combination of above")
else:
    print(f"  ✅ Prediction strength looks reasonable")
    print(f"  🔍 Problem might be elsewhere (TAU thresholds, strategy logic)")

print(f"\n" + "="*80 + "\n")

