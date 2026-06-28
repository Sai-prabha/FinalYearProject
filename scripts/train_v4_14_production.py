#!/usr/bin/env python3
"""
Train V4.14 XGBClassifier for PRODUCTION deployment.

Unlike the backtest (which uses walk-forward CV), this trains on ALL available
data to produce the best possible model for live prediction.

Outputs:
    models/v4_14_production/model.json       - XGBoost native model
    models/v4_14_production/feature_names.json - Ordered feature list
    models/v4_14_production/config.json       - Strategy + model parameters

Usage:
    python scripts/train_v4_14_production.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import roc_auc_score

from src.features import build_features
from src.models import select_features_by_importance


# ============================================================================
# V4.14 PARAMETERS (same as backtest)
# ============================================================================

K_AHEAD = 10
MAX_FEATURES = 50

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 3,
    "learning_rate": 0.04,
    "min_child_weight": 7,
    "subsample": 0.75,
    "colsample_bytree": 0.5589686229585592,
    "gamma": 2.5,
    "reg_alpha": 0.05828398920416312,
    "reg_lambda": 0.5,
    "scale_pos_weight": 1.0,
}

STRATEGY_PARAMS = {
    "entry_threshold": 0.525,
    "exit_threshold": 0.51,
    "cooldown": 15,
    "min_hold": 25,
    "cb_lookback": 500,
    "cb_threshold": -0.03,
}

DATA_PATH = "data/processed/btc_eth_ratio_1m.parquet"
MODEL_DIR = Path("models/v4_14_production")


def main():
    print("=" * 80)
    print("V4.14 PRODUCTION MODEL TRAINING")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data: {DATA_PATH}")
    print(f"Output: {MODEL_DIR}")
    print()

    # ------------------------------------------------------------------
    # 1. LOAD DATA & BUILD FEATURES
    # ------------------------------------------------------------------
    print("STEP 1: Loading data and building features...")
    df = pd.read_parquet(DATA_PATH)
    X, y_bps, kept_idx, meta = build_features(df, k_ahead=K_AHEAD, label="bps")
    y_bps = y_bps.astype("float32")

    print(f"  Samples: {X.shape[0]:,}")
    print(f"  Raw features: {X.shape[1]}")
    print()

    # ------------------------------------------------------------------
    # 2. FEATURE SELECTION
    # ------------------------------------------------------------------
    print("STEP 2: Feature selection...")

    def make_simple_reg():
        return XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
        )

    X_selected, feature_importance = select_features_by_importance(
        X,
        y_bps,
        make_model=make_simple_reg,
        max_features=MAX_FEATURES,
        verbose=True,
    )

    feature_names = list(X_selected.columns)
    print(f"\n  Selected {len(feature_names)} features for production model")
    print()

    # ------------------------------------------------------------------
    # 3. TRAIN PRODUCTION CLASSIFIER ON ALL DATA
    # ------------------------------------------------------------------
    print("STEP 3: Training production XGBClassifier on ALL data...")

    # Binary target: 1 if future return is positive
    y_dir = (y_bps > 0).astype(int)

    # Use 90% train, 10% validation for early stopping
    val_size = max(int(len(y_dir) * 0.1), 1000)
    X_train = X_selected.iloc[:-val_size]
    y_train = y_dir.iloc[:-val_size]
    X_val = X_selected.iloc[-val_size:]
    y_val = y_dir.iloc[-val_size:]

    cls = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        early_stopping_rounds=30,
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
        **XGB_PARAMS,
    )

    cls.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate on validation set
    val_proba = cls.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)
    val_hit = float(((val_proba > 0.5).astype(int) == y_val).mean())

    # Now retrain on ALL data (no early stopping, use best_iteration from above)
    best_n = cls.best_iteration + 1 if hasattr(cls, 'best_iteration') and cls.best_iteration else XGB_PARAMS["n_estimators"]
    print(f"  Best iteration from early stopping: {best_n}")
    print(f"  Validation AUC: {val_auc:.4f}")
    print(f"  Validation Hit Rate: {val_hit:.1%}")
    print()

    # Final model trained on all data with the optimal number of trees
    final_params = XGB_PARAMS.copy()
    final_params["n_estimators"] = best_n

    cls_final = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
        **final_params,
    )
    cls_final.fit(X_selected, y_dir)

    # Quick sanity check
    full_proba = cls_final.predict_proba(X_selected)[:, 1]
    full_auc = roc_auc_score(y_dir, full_proba)
    print(f"  Full-data AUC: {full_auc:.4f}")
    print(f"  Proba mean: {full_proba.mean():.4f}, std: {full_proba.std():.4f}")
    print(f"  Pct above entry ({STRATEGY_PARAMS['entry_threshold']}): "
          f"{(full_proba >= STRATEGY_PARAMS['entry_threshold']).mean() * 100:.1f}%")
    print(f"  Pct below entry ({1 - STRATEGY_PARAMS['entry_threshold']:.3f}): "
          f"{(full_proba <= (1 - STRATEGY_PARAMS['entry_threshold'])).mean() * 100:.1f}%")
    print()

    # ------------------------------------------------------------------
    # 4. SAVE MODEL + METADATA
    # ------------------------------------------------------------------
    print("STEP 4: Saving production model...")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save XGBoost model in native JSON format
    model_path = MODEL_DIR / "model.json"
    cls_final.save_model(str(model_path))
    print(f"  Model saved: {model_path}")

    # Save feature names (ordered)
    feature_names_path = MODEL_DIR / "feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"  Feature names saved: {feature_names_path}")

    # Save full config (model params + strategy params)
    config = {
        "version": "v4.14",
        "trained_at": datetime.now().isoformat(),
        "data_path": DATA_PATH,
        "data_rows": int(df.shape[0]),
        "data_time_range": {
            "start": str(df["open_time"].iloc[0]),
            "end": str(df["open_time"].iloc[-1]),
        },
        "k_ahead": K_AHEAD,
        "max_features": MAX_FEATURES,
        "n_features_selected": len(feature_names),
        "best_n_estimators": best_n,
        "xgb_params": {k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in final_params.items()},
        "strategy_params": STRATEGY_PARAMS,
        "validation_metrics": {
            "auc": float(val_auc),
            "hit_rate": float(val_hit),
            "val_size": val_size,
        },
        "full_data_metrics": {
            "auc": float(full_auc),
            "proba_mean": float(full_proba.mean()),
            "proba_std": float(full_proba.std()),
        },
    }
    config_path = MODEL_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved: {config_path}")

    # ------------------------------------------------------------------
    # 5. VERIFICATION
    # ------------------------------------------------------------------
    print()
    print("STEP 5: Verification - loading saved model...")

    # Load and verify
    cls_loaded = XGBClassifier()
    cls_loaded.load_model(str(model_path))

    with open(feature_names_path, "r") as f:
        loaded_features = json.load(f)

    # Verify predictions match
    test_input = X_selected.iloc[-10:]
    original_pred = cls_final.predict_proba(test_input)[:, 1]
    loaded_pred = cls_loaded.predict_proba(test_input[loaded_features])[:, 1]

    max_diff = float(np.max(np.abs(original_pred - loaded_pred)))
    print(f"  Max prediction difference: {max_diff:.2e}")
    assert max_diff < 1e-6, f"Predictions don't match! Max diff: {max_diff}"
    print(f"  Predictions match perfectly")

    print()
    print("=" * 80)
    print("PRODUCTION MODEL READY")
    print("=" * 80)
    print(f"  Model:    {model_path}")
    print(f"  Features: {feature_names_path} ({len(feature_names)} features)")
    print(f"  Config:   {config_path}")
    print(f"  Val AUC:  {val_auc:.4f}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()

