"""Cost-adjusted shadow accounting + dual-model shadow slot (V4183_CANDIDATE.md §4)."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import model_server as ms
from api.version_config import V418_CONFIG, register_version


def test_apply_cost_model_math_and_idempotency():
    # position notional $500, +0.40% gross => +$2.00 gross
    t = {"pnl_pct": 0.40, "pnl_dollar": 2.00}
    out = ms._apply_cost_model(t)
    # 4 fills x 4.5 bps = 0.18% of the $500 notional = $0.90
    assert abs(out["fee_dollar"] - 0.90) < 1e-9
    assert abs(out["pnl_pct_net"] - (0.40 - 0.18)) < 1e-9
    assert abs(out["pnl_dollar_net"] - 1.10) < 1e-9
    # Idempotent: a second pass never double-charges
    again = ms._apply_cost_model(dict(out))
    assert again["pnl_dollar_net"] == out["pnl_dollar_net"]


def test_cost_model_flips_marginal_winner_to_net_loser():
    t = ms._apply_cost_model({"pnl_pct": 0.10, "pnl_dollar": 0.50})
    assert t["pnl_pct"] > 0 and t["pnl_pct_net"] < 0


def test_net_view_feeds_stats_engine_with_net_numbers():
    trades = [
        {"pnl_pct": 0.40, "pnl_dollar": 2.00, "entry_time": 1, "exit_time": 2},
        {"pnl_pct": -0.30, "pnl_dollar": -1.50, "entry_time": 3, "exit_time": 4},
    ]
    net = ms._net_view(trades)
    assert abs(net[0]["pnl_pct"] - 0.22) < 1e-9        # 0.40 - 0.18
    assert abs(net[1]["pnl_pct"] - (-0.48)) < 1e-9     # -0.30 - 0.18
    assert trades[0]["pnl_pct"] == 0.40                # originals untouched


def _train_tiny_model(feature_names):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(feature_names))).astype("float32")
    y = (X[:, 0] > 0).astype(int)
    m = XGBClassifier(n_estimators=3, max_depth=2)
    m.fit(X, y)
    return m


def test_dual_model_shadow_loads_own_weights_and_gates_on_features(tmp_path):
    version = "v9.9-testdual"
    register_version(version, V418_CONFIG)
    fnames = [f"f{i}" for i in range(8)]
    mdir = Path(ms.__file__).resolve().parent.parent / "models" / version.replace(".", "_")
    saved_fnames = ms.feature_names
    try:
        mdir.mkdir(parents=True, exist_ok=True)
        _train_tiny_model(fnames).save_model(str(mdir / "model.json"))
        (mdir / "feature_names.json").write_text(json.dumps(fnames))

        # Matching feature set -> own weights loaded
        ms.feature_names = fnames
        ms._activate_shadow(version)
        assert ms.shadow_model is not None
        assert ms.SHADOW_MODEL_VERSION == version

        # Mismatched feature set -> own weights REFUSED, falls back to shared probas
        ms.feature_names = fnames + ["extra"]
        ms._activate_shadow(version)
        assert ms.shadow_model is None

        # Deactivation clears the slot completely
        ms.feature_names = fnames
        ms._activate_shadow(version)
        assert ms.shadow_model is not None
        ms._deactivate_shadow()
        assert ms.shadow_model is None and ms.shadow_signal_gen is None
    finally:
        ms.feature_names = saved_fnames
        ms._deactivate_shadow()
        shutil.rmtree(mdir, ignore_errors=True)


def test_no_model_dir_means_shared_probas():
    version = "v9.9-testshared"
    register_version(version, V418_CONFIG)
    try:
        ms._activate_shadow(version)
        assert ms.shadow_model is None          # no models/v9_9-testshared/ on disk
        assert ms.shadow_signal_gen is not None
    finally:
        ms._deactivate_shadow()


def test_top1_removed_sign_stability_gate():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_v4183 import top1_removed_sign_ok

    # One monster win carrying an otherwise-losing book -> gate fails
    lucky = [{"pnl_pct_net": 5.0}] + [{"pnl_pct_net": -0.2}] * 10
    assert top1_removed_sign_ok(lucky) is False
    # Evenly earned edge -> gate passes
    steady = [{"pnl_pct_net": 0.3}, {"pnl_pct_net": 0.2}, {"pnl_pct_net": 0.25}, {"pnl_pct_net": -0.1}]
    assert top1_removed_sign_ok(steady) is True
