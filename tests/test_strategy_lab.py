"""
Strategy lab tests — search space, walk-forward ranking, deploy-candidate
gating, job state machine, scheduler windows, shadow registration.

The evaluation test runs the REAL replay engine over a synthetic proba stream
engineered so a short-side conviction config must win — the pipeline is
validated end-to-end, not around stubs.
"""

import dataclasses
import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import api.strategy_lab as sl
from api.version_config import StrategyConfig, get_strategy_config, register_version


@pytest.fixture
def lab_dirs(tmp_path, monkeypatch):
    """Isolate every lab artifact under tmp so tests never touch data/."""
    research = tmp_path / "research"
    monkeypatch.setattr(sl, "RESEARCH_DIR", research)
    monkeypatch.setattr(sl, "CANDIDATES_JSON", research / "candidates.json")
    monkeypatch.setattr(sl, "STATE_JSON", research / "state.json")
    monkeypatch.setattr(sl, "AUDIT_JSONL", research / "audit.jsonl")
    monkeypatch.setattr(sl, "SHADOW_REGISTRATION_JSON", tmp_path / "live" / "shadow_registration.json")
    return tmp_path


# ── Search space ────────────────────────────────────────────────────────────

class TestSearchSpace:
    def test_space_is_small_and_stable(self):
        specs = sl.build_candidate_specs()
        candidates = [s for s in specs if s["role"] == "candidate"]
        # 2 sides × 3 tails × 3 holds + 3 entries × 2 exits × 2 holds = 30
        assert len(candidates) == 30
        assert len(specs) == 30 + len(sl.BASELINE_VERSIONS)
        # ids are content-addressed: same params → same id, forever
        again = sl.build_candidate_specs()
        assert [s["id"] for s in specs] == [s["id"] for s in again]

    def test_conviction_config_short_only_disables_longs(self):
        spec = next(s for s in sl.build_candidate_specs()
                    if s["family"] == "conviction" and s["params"]["side"] == "short")
        cfg = sl.spec_config(spec)
        assert cfg.entry_threshold_short == spec["params"]["tail"]
        assert cfg.entry_threshold_long > 1.0  # unreachable — longs off
        assert cfg.max_hold_bars == spec["params"]["max_hold"]

    def test_conviction_both_sides_mirrors_the_tail(self):
        spec = next(s for s in sl.build_candidate_specs()
                    if s["family"] == "conviction" and s["params"]["side"] == "both"
                    and s["params"]["tail"] == 0.45)
        cfg = sl.spec_config(spec)
        assert cfg.entry_threshold_long == 0.55

    def test_band_config_overrides_only_named_params(self):
        spec = next(s for s in sl.build_candidate_specs() if s["family"] == "band")
        cfg = sl.spec_config(spec)
        base = get_strategy_config("v4.15")
        assert cfg.entry_threshold == spec["params"]["entry"]
        assert cfg.min_hold == spec["params"]["min_hold"]
        assert cfg.tp_sl_mode == base.tp_sl_mode  # untouched


# ── Ranking math ────────────────────────────────────────────────────────────

class TestRankingMath:
    def test_deflated_sharpe_bar_rises_with_trial_count(self):
        few = sl.expected_max_sharpe([1.0, 0.5, -0.2])
        many = sl.expected_max_sharpe([1.0, 0.5, -0.2] * 10)
        assert many > few > 0

    def test_deflated_sharpe_degenerate_cases(self):
        assert sl.expected_max_sharpe([]) == 0.0
        assert sl.expected_max_sharpe([1.0]) == 0.0
        assert sl.expected_max_sharpe([0.7, 0.7, 0.7]) < 1e-9  # no variance → nothing to deflate

    def test_score_shrinks_small_samples_toward_zero(self):
        small = sl._score(0.10, n=5, consistency=1.0, dd_pct=0.0)
        large = sl._score(0.10, n=500, consistency=1.0, dd_pct=0.0)
        assert 0 < small < large

    def test_score_punishes_inconsistency_and_drawdown(self):
        base = sl._score(0.10, 100, 1.0, 0.0)
        assert sl._score(0.10, 100, 0.4, 0.0) < base
        assert sl._score(0.10, 100, 1.0, -50.0) < base
        assert sl._score(None, 100, 1.0, 0.0) == 0.0

    def test_folds_slice_by_exit_time_and_flag_empties(self):
        trades = [
            {"exit_time": 100, "pnl_pct_net": 0.5},
            {"exit_time": 150, "pnl_pct_net": -0.1},
            {"exit_time": 450, "pnl_pct_net": 0.2},
        ]
        folds = sl._fold_slices(trades, 0, 500, 5)
        # edges are half-open [lo, hi): 100 and 150 both land in fold 2
        assert [f["n"] for f in folds] == [0, 2, 0, 0, 1]
        assert folds[1]["positive"] is True   # (0.5 - 0.1)/2 > 0
        assert folds[0]["positive"] is False  # empty fold is never evidence

    def test_neighbors_are_one_step_in_one_dimension(self):
        specs = sl.build_candidate_specs()
        spec = next(s for s in specs if s["family"] == "conviction"
                    and s["params"] == {"side": "short", "tail": 0.45, "max_hold": 240})
        neigh_ids = sl._neighbors(spec, specs)
        by_id = {s["id"]: s for s in specs}
        for nid in neigh_ids:
            p, q = by_id[nid]["params"], spec["params"]
            assert sum(1 for k in p if p[k] != q[k]) == 1
        # side flip + tail up/down + hold up/down = 5 neighbors
        assert len(neigh_ids) == 5


# ── Walk-forward evaluation on the real replay engine ──────────────────────

def synthetic_dataset(n_bars: int, seed: int = 7) -> pd.DataFrame:
    """Proba stream with engineered short-tail edge: proba dips below 0.44
    precede genuine ratio declines, everything else is noise around 0.5."""
    rng = np.random.default_rng(seed)
    proba = np.clip(rng.normal(0.5, 0.01, n_bars), 0.05, 0.95)
    drift = np.zeros(n_bars)
    for start in range(200, n_bars - 400, 300):
        proba[start] = 0.42  # conviction short entry
        drift[start:start + 240] -= 0.00002  # ratio decays over the hold
    ratio = 30.0 * np.exp(np.cumsum(drift + rng.normal(0, 0.000004, n_bars)))
    t0 = 1_700_000_000
    return pd.DataFrame({
        "time": np.arange(t0, t0 + n_bars * 60, 60),
        "ratio": ratio,
        "proba": proba,
        "valid": np.ones(n_bars, dtype=bool),
    })


@pytest.fixture
def tiny_lab(monkeypatch):
    """Shrink the lab so the real engine runs in test time."""
    monkeypatch.setattr(sl, "WARMUP", 50)
    monkeypatch.setattr(sl, "HOLDOUT_DAYS", 2)
    monkeypatch.setattr(sl, "FEE_GRID", (4.5, 2.0))
    monkeypatch.setattr(sl, "FAMILIES", {
        "conviction": {
            "base": "v4.18", "rationale": "test",
            "grid": {"side": ["short"], "tail": [0.44, 0.45], "max_hold": [240]},
        },
        "band": {
            "base": "v4.15", "rationale": "test control",
            "grid": {"entry": [0.525], "exit": [0.51], "min_hold": [25]},
        },
    })
    monkeypatch.setattr(sl, "BASELINE_VERSIONS", ["v4.15"])


class TestEvaluateRun:
    def test_pipeline_end_to_end_on_engineered_edge(self, tiny_lab):
        ds = synthetic_dataset(16 * 1440)  # 16 synthetic days
        result = sl.evaluate_run(ds, sl.build_candidate_specs(), live_version="v4.15")

        assert result["trials"] == 3  # candidates only, baselines don't count
        ranked = result["candidates"]
        assert [r["rank"] for r in ranked] == list(range(1, len(ranked) + 1))
        assert ranked == sorted(ranked, key=lambda r: r["score"], reverse=True)

        conv = [r for r in ranked if r["family"] == "conviction"]
        assert all(r["metrics"]["n_trades"] > 0 for r in conv), "engineered dips must trade"
        # The engineered short edge must outrank the churn control family
        assert ranked[0]["family"] == "conviction"

        for r in ranked:
            if r["role"] != "candidate":
                assert r["lifecycle"] == "BASELINE"
                continue
            assert set(r["gates"].keys()) == {
                "edge", "trades", "consistency", "worst_fold",
                "concentration", "dsr", "neighborhood"}
            assert r["lifecycle"] in ("DEPLOY_CANDIDATE", "MATCH", "REJECTED")
            if r["lifecycle"] == "REJECTED":
                assert r["status_reasons"], "rejections must be explainable"

    def test_holdout_only_for_gate_passers(self, tiny_lab):
        ds = synthetic_dataset(16 * 1440)
        result = sl.evaluate_run(ds, sl.build_candidate_specs(), live_version="v4.15")
        for r in result["candidates"]:
            if r["role"] != "candidate":
                continue
            if r["holdout"]["evaluated"]:
                assert r["pre_holdout_pass"], "holdout is never mined by gate failures"

    def test_fee_sensitivity_reported_per_level(self, tiny_lab):
        ds = synthetic_dataset(16 * 1440)
        result = sl.evaluate_run(ds, sl.build_candidate_specs(), live_version="v4.15")
        cand = next(r for r in result["candidates"] if r["role"] == "candidate")
        assert set(cand["fees"].keys()) == {"4.5", "2"}


# ── Job state machine ───────────────────────────────────────────────────────

class TestResearchJob:
    def _wait_done(self, timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            snap = sl.job_snapshot()
            if snap["state"] in ("done", "failed"):
                return snap
            time.sleep(0.02)
        raise TimeoutError(f"job stuck: {sl.job_snapshot()}")

    def test_run_completes_and_persists(self, lab_dirs, tiny_lab, monkeypatch):
        monkeypatch.setattr(sl, "build_dataset",
                            lambda *a, **k: synthetic_dataset(16 * 1440))
        sl.start_research_job(trigger="test", depth="daily", live_version="v4.15")
        snap = self._wait_done()
        assert snap["state"] == "done", snap.get("error")

        store = sl.load_candidates()
        assert store["trigger"] == "test"
        assert len(store["candidates"]) == 4
        assert sl.load_state()["last_success"]["run_id"] == snap["run_id"]
        events = [json.loads(l)["event"] for l in sl.AUDIT_JSONL.read_text().splitlines()]
        assert events[0] == "run_started" and events[-1] == "run_completed"

    def test_single_job_guard(self, lab_dirs, monkeypatch):
        started = {"n": 0}

        def slow_dataset(*a, **k):
            started["n"] += 1
            time.sleep(0.3)
            raise RuntimeError("stop here")

        monkeypatch.setattr(sl, "build_dataset", slow_dataset)
        sl.start_research_job(trigger="one", live_version="v4.15")
        with pytest.raises(RuntimeError, match="already running"):
            sl.start_research_job(trigger="two", live_version="v4.15")
        snap = self._wait_done()
        assert snap["state"] == "failed" and started["n"] == 1

    def test_failure_is_recorded_not_swallowed(self, lab_dirs, monkeypatch):
        monkeypatch.setattr(sl, "build_dataset",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no data")))
        sl.start_research_job(trigger="test", live_version="v4.15")
        snap = self._wait_done()
        assert snap["state"] == "failed" and "no data" in snap["error"]
        events = [json.loads(l)["event"] for l in sl.AUDIT_JSONL.read_text().splitlines()]
        assert "run_failed" in events

    def test_bad_depth_rejected_before_thread_spawn(self):
        with pytest.raises(ValueError, match="depth"):
            sl.start_research_job(trigger="test", depth="bogus")


# ── Lifecycle stickiness across refreshes ───────────────────────────────────

class TestLifecycleMerge:
    def test_shadow_running_survives_a_metrics_refresh(self, lab_dirs):
        first = {
            "trials": 1, "sr_benchmark": 0, "baseline_version": "v4.15",
            "baseline_score": 0, "dataset": {},
            "candidates": [{"id": "conviction-abc", "role": "candidate",
                            "lifecycle": "SHADOW_RUNNING", "score": 1,
                            "shadow": {"since": "2026-07-01T00:00:00+00:00"}}],
        }
        sl._persist_run(dict(first), "r1", "test", "deep")
        refreshed = dict(first)
        refreshed["candidates"] = [{"id": "conviction-abc", "role": "candidate",
                                    "lifecycle": "REJECTED", "score": -1}]
        store = sl._persist_run(refreshed, "r2", "test", "deep")
        cand = store["candidates"][0]
        assert cand["lifecycle"] == "SHADOW_RUNNING"     # sticky — still evaluating
        assert cand["backtest_verdict"] == "REJECTED"    # but the new verdict is visible
        assert cand["shadow"]["since"] == "2026-07-01T00:00:00+00:00"


# ── Scheduler windows ───────────────────────────────────────────────────────

class TestScheduler:
    def test_weekday_next_run_is_daily_0430(self):
        # Monday 2026-07-13 12:00 UTC → Tuesday 04:30 daily
        nxt, kind = sl.next_scheduled_run(datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc))
        assert (nxt.hour, nxt.minute, kind) == (4, 30, "daily")
        assert nxt.date() == datetime(2026, 7, 14).date()

    def test_early_morning_run_is_same_day(self):
        nxt, kind = sl.next_scheduled_run(datetime(2026, 7, 13, 3, 0, tzinfo=timezone.utc))
        assert nxt.date() == datetime(2026, 7, 13).date() and kind == "daily"

    def test_friday_night_rolls_to_saturday_deep_scan(self):
        nxt, kind = sl.next_scheduled_run(datetime(2026, 7, 17, 23, 0, tzinfo=timezone.utc))
        assert kind == "deep"
        assert nxt.weekday() == 5 and (nxt.hour, nxt.minute) == (5, 0)

    def test_saturday_has_no_0430_daily(self):
        # Saturday 04:45 UTC: the 04:30 daily never fires on Saturday — next is 05:00 deep
        nxt, kind = sl.next_scheduled_run(datetime(2026, 7, 18, 4, 45, tzinfo=timezone.utc))
        assert kind == "deep" and nxt.hour == 5 and nxt.date() == datetime(2026, 7, 18).date()

    def test_boundary_is_strictly_future(self):
        at = datetime(2026, 7, 14, 4, 30, tzinfo=timezone.utc)
        nxt, _ = sl.next_scheduled_run(at)
        assert nxt > at


# ── Shadow registration + promotion readiness ───────────────────────────────

class TestShadowRegistration:
    def _store_with(self, lifecycle="DEPLOY_CANDIDATE"):
        spec = next(s for s in sl.build_candidate_specs() if s["family"] == "conviction")
        cand = {**spec, "lifecycle": lifecycle, "score": 0.1, "rank": 1,
                "status_reasons": [], "metrics": {}, "folds": [], "consistency": 1.0,
                "fees": {}, "concentration": 0.1}
        sl.RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
        sl.CANDIDATES_JSON.write_text(json.dumps({"candidates": [cand]}))
        return cand

    def test_registration_roundtrip_rebuilds_identical_config(self, lab_dirs):
        cand = self._store_with()
        reg = sl.write_shadow_registration(cand, trigger="test")
        assert reg["version"] == f"lab-{cand['id']}"
        restored = sl.read_shadow_registration()
        assert sl.registration_config(restored) == sl.spec_config(cand)

    def test_clear_registration_audits_and_removes(self, lab_dirs):
        cand = self._store_with()
        sl.write_shadow_registration(cand, trigger="test")
        sl.clear_shadow_registration(reason="test cleanup")
        assert sl.read_shadow_registration() is None
        events = [json.loads(l)["event"] for l in sl.AUDIT_JSONL.read_text().splitlines()]
        assert "shadow_registration_cleared" in events

    def test_set_lifecycle_records_shadow_since(self, lab_dirs):
        cand = self._store_with()
        sl.set_lifecycle(cand["id"], "SHADOW_RUNNING", reason="test")
        assert sl.get_candidate(cand["id"])["lifecycle"] == "SHADOW_RUNNING"
        assert sl.get_candidate(cand["id"])["shadow"]["since"]

    def test_register_version_protects_static_versions(self):
        with pytest.raises(ValueError, match="static"):
            register_version("v4.18", get_strategy_config("v4.15"))
        register_version("lab-test-xyz", get_strategy_config("v4.18"))
        assert get_strategy_config("lab-test-xyz") is not None


class TestPromotionReadiness:
    NOW = 1_800_000_000

    def _stats(self, n=10, exp=0.05):
        return {"n": n, "expectancy_pct": exp}

    def test_not_ready_before_observation_window(self):
        since = self.NOW - 3 * 86400
        r = sl.promotion_readiness(since, self._stats(), self._stats(exp=0.01), now_ts=self.NOW)
        assert not r["ready"]
        obs = next(c for c in r["checks"] if c["name"] == "observation")
        assert not obs["pass"] and obs["value"] == 3.0

    def test_ready_when_all_checks_pass(self):
        since = self.NOW - 20 * 86400
        r = sl.promotion_readiness(since, self._stats(n=8, exp=0.05),
                                   self._stats(exp=0.01), now_ts=self.NOW)
        assert r["ready"] and all(c["pass"] for c in r["checks"])

    def test_never_ready_while_losing_to_live(self):
        since = self.NOW - 20 * 86400
        r = sl.promotion_readiness(since, self._stats(n=8, exp=0.01),
                                   self._stats(exp=0.05), now_ts=self.NOW)
        assert not r["ready"]
        assert not next(c for c in r["checks"] if c["name"] == "vs_live")["pass"]

    def test_no_trades_is_never_ready(self):
        since = self.NOW - 60 * 86400
        r = sl.promotion_readiness(since, {"n": 0, "expectancy_pct": None},
                                   self._stats(), now_ts=self.NOW)
        assert not r["ready"]
