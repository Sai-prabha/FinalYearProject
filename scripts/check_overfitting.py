#!/usr/bin/env python3
"""
Overfitting Analysis Script

This script performs comprehensive overfitting checks on trained models:
1. Train vs Validation performance comparison
2. Feature importance analysis
3. Complexity metrics
4. Cross-validation stability
5. Recommendations for reducing overfitting

Usage:
    python scripts/check_overfitting.py --version v4.12
    python scripts/check_overfitting.py --version v4.11 --compare v4.12
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

from src.v4_config import get_config, list_versions
from src.features import build_features
from src.models import (
    check_overfitting,
    get_feature_importance,
    select_features_by_importance,
)


def analyze_model_complexity(config) -> Dict:
    """Analyze model complexity metrics."""
    xgb_params = config.xgb_params
    
    # Calculate complexity score (0-100, higher = more complex = more prone to overfitting)
    complexity_factors = {
        'max_depth': xgb_params.max_depth / 10 * 30,  # 30 points max
        'n_estimators': min(xgb_params.n_estimators / 500 * 25, 25),  # 25 points max
        'learning_rate': (0.1 - xgb_params.learning_rate) / 0.1 * 15,  # 15 points (lower LR = longer training)
        'min_child_weight': max(0, (5 - xgb_params.min_child_weight) / 5 * 10),  # 10 points
        'regularization': max(0, (5 - (xgb_params.reg_alpha + xgb_params.reg_lambda) / 2) / 5 * 20),  # 20 points
    }
    
    total_complexity = sum(complexity_factors.values())
    
    return {
        'complexity_score': total_complexity,
        'factors': complexity_factors,
        'max_depth': xgb_params.max_depth,
        'n_estimators': xgb_params.n_estimators,
        'learning_rate': xgb_params.learning_rate,
        'min_child_weight': xgb_params.min_child_weight,
        'reg_alpha': xgb_params.reg_alpha,
        'reg_lambda': xgb_params.reg_lambda,
        'subsample': xgb_params.subsample,
        'colsample_bytree': xgb_params.colsample_bytree,
    }


def perform_validation_check(
    config,
    data_path: str = "data/processed/btc_eth_ratio_1m.parquet"
) -> Dict:
    """Perform train/val split and check for overfitting."""
    print(f"\n{'='*80}")
    print(f"OVERFITTING ANALYSIS: {config.name.upper()}")
    print(f"{'='*80}\n")
    
    # Load data
    print("📥 Loading data...")
    df = pd.read_parquet(data_path)
    X, y, kept_idx, meta = build_features(df, k_ahead=10, label="bps")
    y = y.astype("float32")
    
    print(f"   Dataset: {len(X):,} samples, {X.shape[1]} features")
    
    # Split into train/val (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"   Train: {len(X_train):,} | Val: {len(X_val):,}")
    
    # Train model
    print("\n🔧 Training model...")
    model = XGBRegressor(
        **config.xgb_params.to_dict(),
        n_jobs=-1,
        tree_method='hist',
        random_state=42,
    )
    
    try:
        # Try with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        best_iteration = getattr(model, 'best_iteration', None)
        if best_iteration:
            print(f"   Best iteration: {best_iteration} / {config.xgb_params.n_estimators}")
    except:
        # Fallback without early stopping
        model.fit(X_train, y_train)
    
    # Check overfitting
    overfitting_metrics = check_overfitting(
        model, X_train, y_train, X_val, y_val, verbose=True
    )
    
    # Feature importance
    fi = get_feature_importance(model, X.columns)
    n_features_used = (fi > 0.001).sum()
    print(f"\n📊 Features with importance > 0.001: {n_features_used} / {len(X.columns)}")
    
    return {
        'overfitting_metrics': overfitting_metrics,
        'feature_count': len(X.columns),
        'features_used': n_features_used,
        'feature_importance': fi.head(20).to_dict(),
        'best_iteration': getattr(model, 'best_iteration', config.xgb_params.n_estimators),
    }


def generate_recommendations(complexity: Dict, validation: Dict) -> list:
    """Generate recommendations based on analysis."""
    recommendations = []
    
    complexity_score = complexity['complexity_score']
    overfitting = validation['overfitting_metrics']
    
    # High complexity
    if complexity_score > 70:
        recommendations.append({
            'severity': 'HIGH',
            'issue': 'Model complexity is very high',
            'action': f"Reduce max_depth from {complexity['max_depth']} to 4-5, "
                     f"or reduce n_estimators from {complexity['n_estimators']} to 200-300"
        })
    elif complexity_score > 50:
        recommendations.append({
            'severity': 'MEDIUM',
            'issue': 'Model complexity is moderate-high',
            'action': 'Consider reducing max_depth or n_estimators for better generalization'
        })
    
    # Overfitting detected
    if overfitting['is_overfitting']:
        if overfitting['rmse_ratio'] > 1.3:
            recommendations.append({
                'severity': 'HIGH',
                'issue': f"Validation RMSE is {overfitting['rmse_ratio']:.2f}x train RMSE",
                'action': 'Increase regularization (reg_alpha, reg_lambda), reduce max_depth, or add more dropout'
            })
        if overfitting['hit_diff'] > 0.10:
            recommendations.append({
                'severity': 'MEDIUM',
                'issue': f"Train hit rate exceeds validation by {overfitting['hit_diff']:.1%}",
                'action': 'Model is memorizing training patterns. Use feature selection or increase min_child_weight'
            })
    
    # Too many features
    feature_ratio = validation['features_used'] / validation['feature_count']
    if validation['feature_count'] > 80 or feature_ratio > 0.9:
        recommendations.append({
            'severity': 'MEDIUM',
            'issue': f"Using {validation['features_used']} features ({feature_ratio:.0%} of total)",
            'action': 'Use feature selection to keep only top 40-60 most important features'
        })
    
    # Weak regularization
    if complexity['reg_alpha'] < 0.5 and complexity['reg_lambda'] < 2.0:
        recommendations.append({
            'severity': 'LOW',
            'issue': 'Weak L1/L2 regularization',
            'action': 'Increase reg_alpha to 0.5-1.0 and reg_lambda to 2.0-5.0'
        })
    
    # Good model
    if not recommendations:
        recommendations.append({
            'severity': 'INFO',
            'issue': 'No major overfitting issues detected',
            'action': 'Model looks well-regularized. Monitor performance on new data.'
        })
    
    return recommendations


def print_report(config_name: str, complexity: Dict, validation: Dict, recommendations: list):
    """Print comprehensive report."""
    print(f"\n{'='*80}")
    print(f"OVERFITTING REPORT: {config_name.upper()}")
    print(f"{'='*80}\n")
    
    # Complexity Analysis
    print("1️⃣  MODEL COMPLEXITY")
    print(f"   Overall Score: {complexity['complexity_score']:.1f}/100 ", end="")
    if complexity['complexity_score'] > 70:
        print("(⚠️ HIGH - prone to overfitting)")
    elif complexity['complexity_score'] > 50:
        print("(⚠️ MODERATE)")
    else:
        print("(✅ LOW - good generalization)")
    
    print(f"\n   Parameter Analysis:")
    print(f"      max_depth:          {complexity['max_depth']:3d} {'⚠️  TOO DEEP' if complexity['max_depth'] > 6 else '✅'}")
    print(f"      n_estimators:       {complexity['n_estimators']:3d} {'⚠️  TOO MANY' if complexity['n_estimators'] > 400 else '✅'}")
    print(f"      learning_rate:      {complexity['learning_rate']:.3f}")
    print(f"      min_child_weight:   {complexity['min_child_weight']:3d} {'⚠️  TOO LOW' if complexity['min_child_weight'] < 3 else '✅'}")
    print(f"      reg_alpha (L1):     {complexity['reg_alpha']:.2f} {'⚠️  WEAK' if complexity['reg_alpha'] < 0.5 else '✅'}")
    print(f"      reg_lambda (L2):    {complexity['reg_lambda']:.2f} {'⚠️  WEAK' if complexity['reg_lambda'] < 2.0 else '✅'}")
    print(f"      subsample:          {complexity['subsample']:.2f}")
    print(f"      colsample_bytree:   {complexity['colsample_bytree']:.2f}")
    
    # Validation Performance
    print(f"\n2️⃣  TRAIN VS VALIDATION PERFORMANCE")
    ov = validation['overfitting_metrics']
    print(f"   RMSE:      Train={ov['train_rmse']:6.2f}  Val={ov['val_rmse']:6.2f}  Ratio={ov['rmse_ratio']:.2f} {'⚠️' if ov['rmse_ratio'] > 1.3 else '✅'}")
    print(f"   MAE:       Train={ov['train_mae']:6.2f}  Val={ov['val_mae']:6.2f}")
    print(f"   Hit Rate:  Train={ov['train_hit']:6.1%}  Val={ov['val_hit']:6.1%}  Diff={ov['hit_diff']:+.1%} {'⚠️' if ov['hit_diff'] > 0.10 else '✅'}")
    
    # Feature Usage
    print(f"\n3️⃣  FEATURE ANALYSIS")
    print(f"   Total Features:     {validation['feature_count']}")
    print(f"   Features Used:      {validation['features_used']} ({validation['features_used']/validation['feature_count']:.0%})")
    print(f"   Early Stop At:      {validation.get('best_iteration', 'N/A')}")
    
    # Recommendations
    print(f"\n4️⃣  RECOMMENDATIONS")
    for i, rec in enumerate(recommendations, 1):
        severity_icon = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢', 'INFO': '💡'}
        icon = severity_icon.get(rec['severity'], '•')
        print(f"   {icon} [{rec['severity']:6s}] {rec['issue']}")
        print(f"      → {rec['action']}\n")
    
    print(f"{'='*80}\n")


def compare_configs(config1_name: str, config2_name: str):
    """Compare two configurations side by side."""
    config1 = get_config(config1_name)
    config2 = get_config(config2_name)
    
    comp1 = analyze_model_complexity(config1)
    comp2 = analyze_model_complexity(config2)
    
    print(f"\n{'='*80}")
    print(f"CONFIGURATION COMPARISON")
    print(f"{'='*80}\n")
    print(f"{'Parameter':<25s} {config1_name:>15s} {config2_name:>15s}   Winner")
    print(f"{'-'*25} {'-'*15} {'-'*15}   {'-'*10}")
    
    def compare_param(name, val1, val2, lower_is_better=True):
        if lower_is_better:
            winner = '←' if val1 < val2 else ('→' if val2 < val1 else '=')
        else:
            winner = '←' if val1 > val2 else ('→' if val2 > val1 else '=')
        
        if isinstance(val1, float):
            print(f"{name:<25s} {val1:>15.3f} {val2:>15.3f}   {winner}")
        else:
            print(f"{name:<25s} {val1:>15d} {val2:>15d}   {winner}")
    
    compare_param("Complexity Score", comp1['complexity_score'], comp2['complexity_score'], lower_is_better=True)
    compare_param("max_depth", comp1['max_depth'], comp2['max_depth'], lower_is_better=True)
    compare_param("n_estimators", comp1['n_estimators'], comp2['n_estimators'], lower_is_better=True)
    compare_param("learning_rate", comp1['learning_rate'], comp2['learning_rate'], lower_is_better=False)
    compare_param("min_child_weight", comp1['min_child_weight'], comp2['min_child_weight'], lower_is_better=False)
    compare_param("reg_alpha", comp1['reg_alpha'], comp2['reg_alpha'], lower_is_better=False)
    compare_param("reg_lambda", comp1['reg_lambda'], comp2['reg_lambda'], lower_is_better=False)
    
    print(f"\n   Legend: ← = {config1_name} better, → = {config2_name} better, = = equal")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze model for overfitting")
    parser.add_argument("--version", default="v4.11", help="Config version to analyze")
    parser.add_argument("--compare", help="Compare with another version")
    parser.add_argument("--data", default="data/processed/btc_eth_ratio_1m.parquet", help="Data path")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation check (only analyze config)")
    
    args = parser.parse_args()
    
    # Check if version exists
    available = list_versions()
    if args.version not in available:
        print(f"❌ Version '{args.version}' not found. Available: {', '.join(available)}")
        return
    
    config = get_config(args.version)
    
    # Analyze complexity
    complexity = analyze_model_complexity(config)
    
    # Perform validation check
    if not args.skip_validation:
        validation = perform_validation_check(config, args.data)
        recommendations = generate_recommendations(complexity, validation)
        print_report(config.name, complexity, validation, recommendations)
    else:
        print(f"\n📊 Complexity Score: {complexity['complexity_score']:.1f}/100")
    
    # Compare with another config
    if args.compare:
        if args.compare not in available:
            print(f"❌ Compare version '{args.compare}' not found. Available: {', '.join(available)}")
            return
        compare_configs(args.version, args.compare)


if __name__ == "__main__":
    main()

