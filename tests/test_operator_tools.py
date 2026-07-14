"""Operator tools: audit readers, timing advisor, multi-pair search,
simulation campaigns (RESEARCH_TOOLS.md).

Run: .venv/bin/python -m pytest tests/test_operator_tools.py -q
"""
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api.ops_audit as oa
import api.simulation_lab as sim
import api.strategy_lab as sl
import api.timing_advisor as ta
from api import model_server as ms

from tests.test_strategy_lab import lab_dirs, synthetic_dataset, tiny_lab  # noqa: F401


def _run(coro):
    return asyncio.run(coro)


# ── Audit reader ────────────────────────────────────────────────────────────

@pytest.fixture
def audit_file(tmp_path):
    path = tmp_path / "audit.jsonl"
    rows = [
        {"ts": "2026-07-14T00:01:00+00:00", "event": "auto_execute_changed",
         "actor": "adm1nFYP", "via": "meridian", "outcome": "applied", "version": 2,
         "prev": False, "new": True, "instance": "r-1"},
        {"ts": "2026-07-14T00:02:00+00:00", "event": "auto_execute_changed",
         "actor": "anonymous", "via": "vercel", "outcome": "applied", "version": 3,
         "prev": True, "new": False, "instance": "r-1"},
        {"ts": "2026-07-14T00:03:00+00:00", "event": "auto_execute_conflict",
         "actor": "adm1nFYP", "via": "meridian", "outcome": "conflict",
         "expected_version": 2, "current_version": 3, "instance": "r-1"},
        {"ts": "2026-07-14T00:04:00+00:00", "event": "auto_execute_changed",
         "actor": "adm1nFYP", "via": "api", "outcome": "applied", "version": 4,
         "prev": False, "new": True, "instance": "r-1"},
    ]
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write("NOT-JSON\n")  # torn write must be tolerated
    return path


class TestAuditReader:
    def test_newest_first_with_stable_ids(self, audit_file):
        out = oa.read_audit(audit_file)
        assert out["total_matched"] == 4
        assert [r["ts"][-8:-6] for r in out["rows"]] == ["00"] * 4
        assert [r["version"] for r in out["rows"] if "version" in r] == [4, 3, 2]
        assert out["rows"][0]["id"] == "3"  # line index is the stable id
        assert out["next_cursor"] is None

    def test_filters(self, audit_file):
        assert oa.read_audit(audit_file, actor="adm1nFYP")["total_matched"] == 3
        assert oa.read_audit(audit_file, via="vercel")["total_matched"] == 1
        assert oa.read_audit(audit_file, outcome="conflict")["total_matched"] == 1
        assert oa.read_audit(audit_file, since="2026-07-14T00:02:30",
                             until="2026-07-14T00:03:30")["total_matched"] == 1
        assert oa.read_audit(audit_file, actor="nobody")["total_matched"] == 0

    def test_cursor_pagination(self, audit_file):
        p1 = oa.read_audit(audit_file, limit=2)
        assert len(p1["rows"]) == 2 and p1["next_cursor"] == "2"
        p2 = oa.read_audit(audit_file, limit=2, cursor=p1["next_cursor"])
        assert len(p2["rows"]) == 2 and p2["next_cursor"] is None
        ids = [r["id"] for r in p1["rows"] + p2["rows"]]
        assert len(set(ids)) == 4  # no overlap, no loss

    def test_missing_file_is_empty_not_error(self, tmp_path):
        out = oa.read_audit(tmp_path / "nope.jsonl")
        assert out == {"rows": [], "next_cursor": None, "total_matched": 0}

    def test_summary_counts_and_conflict_rate(self, audit_file):
        s = oa.audit_summary(audit_file)
        assert s["total"] == 4
        assert s["by_via"] == {"meridian": 2, "vercel": 1, "api": 1}
        assert s["by_outcome"] == {"applied": 3, "conflict": 1}
        assert s["conflict_rate"] == 0.25
        assert s["first_ts"] < s["last_ts"]

    def test_execution_audit_endpoint(self, audit_file, monkeypatch):
        from api import execution_control as ec
        monkeypatch.setattr(ec, "AUDIT_PATH", audit_file)
        out = _run(ms.get_execution_audit(limit=50, authorization=None))
        assert out["total_matched"] == 4
        out = _run(ms.get_execution_audit(limit=50, via="meridian", authorization=None))
        assert out["total_matched"] == 2


# ── Timing advisor ──────────────────────────────────────────────────────────

class TestTimingAdvisor:
    NOW = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)  # Tuesday noon

    def test_staleness_triggers_recommendation(self):
        old = {"ts": "2026-07-01T00:00:00+00:00"}
        out = ta.advise(now=self.NOW, last_success=old, vol=None)
        assert out["now"]["recommended"] is True
        assert any("days ago" in r for r in out["now"]["reasons"])
        assert out["now"]["confidence"] >= 0.8

    def test_fresh_run_quiet_regime_no_nagging(self):
        fresh = {"ts": "2026-07-14T04:35:00+00:00"}
        vol = {"ratio": 1.0, "regime": "normal"}
        out = ta.advise(now=self.NOW, last_success=fresh, vol=vol)
        assert out["now"]["recommended"] is False
        assert out["now"]["reasons"] == []

    def test_regime_shift_creates_immediate_window(self):
        fresh = {"ts": "2026-07-13T04:35:00+00:00"}
        vol = {"ratio": 1.8, "regime": "elevated"}
        out = ta.advise(now=self.NOW, last_success=fresh, vol=vol)
        assert out["now"]["recommended"] is True
        assert out["windows"][0]["kind"] == "regime-shift"
        assert "1.8×" in out["windows"][0]["reason"]

    def test_recurring_windows_daily_and_saturday(self):
        sat = datetime(2026, 7, 17, 23, 0, tzinfo=timezone.utc)  # Friday night
        out = ta.advise(now=sat, last_success={"ts": sat.isoformat()}, vol=None)
        kinds = [w["kind"] for w in out["windows"]]
        assert "daily-close" in kinds
        assert "weekly-trough" in kinds  # Saturday is within 48h

    def test_inside_daily_window_recommends(self):
        inside = datetime(2026, 7, 14, 0, 30, tzinfo=timezone.utc)
        out = ta.advise(now=inside, last_success={"ts": "2026-07-13T05:00:00+00:00"}, vol=None)
        assert out["now"]["recommended"] is True
        assert any("complete" in r for r in out["now"]["reasons"])

    def test_scheduler_proximity_damps_the_nag(self):
        near = datetime(2026, 7, 14, 3, 0, tzinfo=timezone.utc)
        out = ta.advise(now=near, last_success={"ts": "2026-07-01T00:00:00+00:00"},
                        next_scheduled={"next_run": "2026-07-14T04:30:00+00:00"}, vol=None)
        assert out["now"]["recommended"] is False  # scheduler covers it
        assert any("scheduler" in r for r in out["now"]["reasons"])

    def test_vol_state_from_synthetic_cache(self, tmp_path, monkeypatch):
        import scripts.fast_backtest as fb
        monkeypatch.setattr(fb, "CACHE_DIR", tmp_path)
        now_s = int(time.time())
        n = 10 * 1440  # 10 days of minutes
        t = np.arange(now_s - n * 60, now_s, 60)
        rng = np.random.default_rng(3)
        rets = rng.normal(0, 0.0001, n)
        rets[-1440:] *= 5  # last day: 5× the baseline vol
        ratio = 30 * np.exp(np.cumsum(rets))
        pd.DataFrame({"time": t, "ratio": ratio}).to_parquet(
            tmp_path / f"probas_{t[0] * 1000}_{t[-1] * 1000}.parquet")
        vol = ta.ratio_vol_state(now_ts=float(now_s))
        assert vol is not None
        assert vol["regime"] == "elevated"
        assert vol["ratio"] > 1.5

    def test_vol_state_degrades_without_cache(self, tmp_path, monkeypatch):
        import scripts.fast_backtest as fb
        monkeypatch.setattr(fb, "CACHE_DIR", tmp_path)
        assert ta.ratio_vol_state() is None


# ── Multi-pair search ───────────────────────────────────────────────────────

class TestMultiPair:
    def test_build_dataset_rejects_unknown_pair(self):
        with pytest.raises(ValueError, match="unknown pair"):
            sl.build_dataset(120, fetch_missing=False, pair_key="DOGE-SHIB")

    def test_pair_cache_is_namespaced(self, tmp_path, monkeypatch):
        import scripts.fast_backtest as fb
        monkeypatch.setattr(fb, "CACHE_DIR", tmp_path)
        ds = synthetic_dataset(16 * 1440)
        t0, t1 = int(ds["time"].iloc[0]) * 1000, int(ds["time"].iloc[-1]) * 1000
        ds.to_parquet(tmp_path / f"probas_SOL-ETH_{t0}_{t1}.parquet")
        now = datetime.fromtimestamp(t1 / 1000, tz=timezone.utc)
        monkeypatch.setattr(sl, "MIN_DATASET_DAYS", 10)
        out = sl.build_dataset(14, fetch_missing=False, now=now, pair_key="SOL-ETH")
        assert len(out) > 0
        # the default pair must NOT see the SOL-ETH cache
        with pytest.raises(RuntimeError, match="No proba data"):
            sl.build_dataset(14, fetch_missing=False, now=now, pair_key="BTC-ETH")

    def test_evaluate_run_pools_pairs_into_one_dsr(self, tiny_lab):
        ds1 = synthetic_dataset(16 * 1440, seed=7)
        ds2 = synthetic_dataset(16 * 1440, seed=11)
        specs = sl.build_candidate_specs()
        single = sl.evaluate_run(ds1, specs, live_version="v4.15")
        multi = sl.evaluate_run({"BTC-ETH": ds1, "SOL-ETH": ds2}, specs, live_version="v4.15")
        assert single["trials"] == 3
        assert multi["trials"] == 6  # trials multiply across pairs
        assert multi["pairs"] == ["BTC-ETH", "SOL-ETH"]
        ids = [r["id"] for r in multi["candidates"]]
        assert any(i.endswith("@SOL-ETH") for i in ids)
        sol_rows = [r for r in multi["candidates"] if r.get("pair") == "SOL-ETH"]
        assert all(r["role"] == "candidate" for r in sol_rows), "baselines stay on the default pair"
        # per-pair dataset meta is exposed; legacy `dataset` stays default-pair
        assert set(multi["datasets"]) == {"BTC-ETH", "SOL-ETH"}
        assert multi["dataset"] == multi["datasets"]["BTC-ETH"]

    def test_default_pair_is_mandatory(self, tiny_lab):
        ds = synthetic_dataset(16 * 1440)
        with pytest.raises(ValueError, match="default pair"):
            sl.evaluate_run({"SOL-ETH": ds}, sl.build_candidate_specs(), live_version="v4.15")

    def test_multi_symbol_store_written(self, lab_dirs, tiny_lab, monkeypatch):
        monkeypatch_path = lab_dirs / "research" / "multi_symbol.json"
        monkeypatch.setattr(sl, "MULTI_SYMBOL_JSON", monkeypatch_path)
        ds1 = synthetic_dataset(16 * 1440, seed=7)
        ds2 = synthetic_dataset(16 * 1440, seed=11)
        result = sl.evaluate_run({"BTC-ETH": ds1, "SOL-ETH": ds2},
                                 sl.build_candidate_specs(), live_version="v4.15")
        sl._persist_run(result, "r-test", "test", "daily")
        data = json.loads(monkeypatch_path.read_text())
        assert data["pairs"] == ["BTC-ETH", "SOL-ETH"]
        assert set(data["per_pair"]) == {"BTC-ETH", "SOL-ETH"}
        for entry in data["matrix"]:
            assert "@" not in entry["id"]
            assert set(entry["per_pair"]) == {"BTC-ETH", "SOL-ETH"}

    def test_start_research_job_validates_pairs(self, lab_dirs):
        with pytest.raises(ValueError, match="unknown pairs"):
            sl.start_research_job("test", depth="daily", pairs=["DOGE-SHIB"])

    def test_shadow_refuses_non_default_pair(self, lab_dirs, monkeypatch):
        store = {"candidates": [{
            "id": "conviction-abc@SOL-ETH", "role": "candidate", "pair": "SOL-ETH",
            "lifecycle": "DEPLOY_CANDIDATE", "family": "conviction",
            "base_version": "v4.18", "params": {}, "label": "x · SOL-ETH",
        }]}
        sl.CANDIDATES_JSON.parent.mkdir(parents=True, exist_ok=True)
        sl.CANDIDATES_JSON.write_text(json.dumps(store))
        req = ms.ShadowRegisterRequest(confirm=True)
        with pytest.raises(HTTPException) as exc:
            _run(ms.research_register_shadow("conviction-abc@SOL-ETH", req, authorization=None))
        assert exc.value.status_code == 400
        assert "robustness evidence" in exc.value.detail


# ── Simulation campaigns ────────────────────────────────────────────────────

@pytest.fixture
def sim_dirs(tmp_path, monkeypatch, lab_dirs):
    simdir = tmp_path / "simulation"
    monkeypatch.setattr(sim, "SIM_DIR", simdir)
    monkeypatch.setattr(sim, "CAMPAIGNS_JSON", simdir / "campaigns.json")
    return tmp_path


class TestSimulation:
    def test_validation_errors(self, sim_dirs):
        with pytest.raises(ValueError, match="unknown strategy ids"):
            sim.create_campaign(["nope"])
        with pytest.raises(ValueError, match="not be empty"):
            sim.create_campaign([])
        ok_id = sl.build_candidate_specs()[0]["id"]
        with pytest.raises(ValueError, match="unknown pairs"):
            sim.create_campaign([ok_id], pairs=["DOGE-SHIB"])
        with pytest.raises(ValueError, match="window_days"):
            sim.create_campaign([ok_id], window_days=5)

    def test_mutual_exclusion_with_research(self, sim_dirs, monkeypatch):
        ok_id = sl.build_candidate_specs()[0]["id"]
        monkeypatch.setattr(sl, "_job", {"state": "running"})
        with pytest.raises(RuntimeError, match="research job"):
            sim.create_campaign([ok_id])
        # and the reverse: research refuses while a campaign runs
        monkeypatch.setattr(sl, "_job", {"state": "idle"})
        monkeypatch.setattr(sim, "_job", {"state": "running"})
        with pytest.raises(RuntimeError, match="simulation campaign"):
            sl.start_research_job("test", depth="daily")

    def test_campaign_end_to_end(self, sim_dirs, tiny_lab, monkeypatch):
        ds = synthetic_dataset(24 * 1440)  # 24 days → 3 weekly buckets
        monkeypatch.setattr(sim, "build_dataset",
                            lambda *a, **k: ds.copy())
        monkeypatch.setattr(sl, "MIN_DATASET_DAYS", 10)
        specs = sl.build_candidate_specs()
        ids = [s["id"] for s in specs if s["role"] == "candidate"][:2]
        meta = sim.create_campaign(ids, label="test campaign", window_days=24,
                                   created_by="tester", created_via="meridian")
        assert meta["status"] == "running"
        for _ in range(200):
            if sim.job_snapshot().get("state") in ("done", "failed"):
                break
            time.sleep(0.1)
        assert sim.job_snapshot()["state"] == "done"

        stored = sim.get_campaign(meta["id"])
        assert stored["status"] == "done"
        results = sim.load_results(meta["id"])
        assert results is not None
        scen_names = {s["name"] for s in results["scenarios"]["BTC-ETH"]}
        assert "full-window" in scen_names
        assert {"vol-spike-week", "quiet-week"} <= scen_names
        assert len(results["verdicts"]) == 2
        for v in results["verdicts"]:
            assert v["verdict"] in ("ROBUST", "MIXED", "CONFIRMS_REJECT")
        taker_cells = [c for c in results["cells"] if c["fee_bps"] == 4.5]
        assert all(c["equity"] is not None for c in taker_cells)
        # audited into the research trail
        events = [json.loads(l)["event"] for l in sl.AUDIT_JSONL.read_text().splitlines()]
        assert "simulation_started" in events and "simulation_completed" in events

    def test_campaign_never_touches_broker_or_control(self):
        """Structural isolation: the simulation module cannot import the
        broker or the control plane's write path."""
        import api.simulation_lab as module
        src = Path(module.__file__).read_text()
        assert "broker_client" not in src
        assert "from .broker" not in src and "from api.broker" not in src
        assert "execution_control" not in src

    def test_status_endpoints(self, sim_dirs):
        with pytest.raises(HTTPException) as exc:
            _run(ms.simulation_status(id="nope", authorization=None))
        assert exc.value.status_code == 404
        out = _run(ms.list_simulation_campaigns(authorization=None))
        assert out["campaigns"] == []
