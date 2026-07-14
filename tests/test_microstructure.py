"""Microstructure layer tests: normalization, 1s aggregation, segment
storage round-trip, retention, OFI/feature math, determinism, env gate."""

import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api import microstructure_ingest as mi
from api import microstructure_features as mf


@pytest.fixture(autouse=True)
def _reset_state():
    for s in mi.SYMBOLS:
        mi._buf_lob[s] = []
        mi._pending_lob[s] = None
        mi._acc_trades[s] = {}
        mi._buf_tape[s] = []
        mi._state[s].update({"last_lob_ms": None, "last_trade_ms": None,
                             "rows_flushed_today": 0, "segments_today": 0})
    yield


class TestNormalization:
    def test_lob_row_shape_and_derived(self):
        bids = [["100.0", "5"], ["99.5", "2"]]
        asks = [["100.5", "7"], ["101.0", "3"]]
        row = mi.normalize_lob("BTCUSDT", 1_700_000_000_000, bids, asks, levels=3)
        assert row["best_bid"] == 100.0 and row["best_ask"] == 100.5
        assert row["mid"] == pytest.approx(100.25)
        assert row["spread"] == pytest.approx(0.5)
        assert row["bid_depth20"] == 7 and row["ask_depth20"] == 10
        # padding beyond available levels
        assert np.isnan(row["bid_px_2"]) and row["bid_sz_2"] == 0.0

    def test_lob_defensive_resort(self):
        row = mi.normalize_lob("X", 0, [["99", "1"], ["100", "2"]],
                               [["102", "1"], ["101", "2"]], levels=2)
        assert row["best_bid"] == 100.0 and row["best_ask"] == 101.0

    def test_trade_maker_flag(self):
        buy = mi.parse_trade({"T": 1000, "p": "50.5", "q": "2", "m": False})
        sell = mi.parse_trade({"T": 1000, "p": "50.5", "q": "2", "m": True})
        assert buy["is_buy"] and not sell["is_buy"]

    def test_second_aggregation_math(self):
        acc = {}
        mi.agg_trade_into(acc, {"price": 10.0, "qty": 2.0, "is_buy": True})
        mi.agg_trade_into(acc, {"price": 20.0, "qty": 2.0, "is_buy": False})
        row = mi.finalize_second(123, acc)
        assert row["n_trades"] == 2 and row["buy_qty"] == 2.0 and row["sell_qty"] == 2.0
        assert row["vwap"] == pytest.approx(15.0)
        assert row["high"] == 20.0 and row["low"] == 10.0 and row["last_px"] == 20.0


class TestStorage:
    def test_segment_roundtrip_and_day_partition(self, tmp_path):
        day_edge = 1_784_073_600  # 2026-07-15T00:00:00Z
        rows = [{"ts_s": day_edge - 1, "n_trades": 1, "buy_qty": 1.0, "sell_qty": 0.0,
                 "buy_n": 1, "sell_n": 0, "vwap": 1.0, "last_px": 1.0, "high": 1.0, "low": 1.0},
                {"ts_s": day_edge + 1, "n_trades": 2, "buy_qty": 0.0, "sell_qty": 2.0,
                 "buy_n": 0, "sell_n": 2, "vwap": 2.0, "last_px": 2.0, "high": 2.0, "low": 2.0}]
        written = mi.write_segment(rows, "trades1s", "BTCUSDT", root=tmp_path)
        assert len(written) == 2  # midnight boundary -> two day dirs
        days = {p.parent.name for p in written}
        assert days == {"2026-07-14", "2026-07-15"}
        assert not list(tmp_path.rglob("*.tmp"))  # atomic rename left no temps
        back = mf.load_trades_1s("BTCUSDT", day_edge - 10, day_edge + 10, root=tmp_path)
        assert len(back) == 2 and list(back["ts_s"]) == [day_edge - 1, day_edge + 1]

    def test_empty_rows_write_nothing(self, tmp_path):
        assert mi.write_segment([], "lob", "BTCUSDT", root=tmp_path) == []

    def test_retention_prunes_only_old_days(self, tmp_path):
        old = tmp_path / "lob" / "BTCUSDT" / "2025-01-01"
        new = tmp_path / "lob" / "BTCUSDT" / "2026-07-14"
        old.mkdir(parents=True)
        new.mkdir(parents=True)
        removed = mi.prune_old_days(root=tmp_path, today="2026-07-14")
        assert old in removed and not old.exists() and new.exists()


class TestMessageRouting:
    def _depth_msg(self, sym, event_ms):
        return json.dumps({"stream": f"{sym.lower()}@depth20@500ms",
                           "data": {"e": "depthUpdate", "E": event_ms, "s": sym,
                                    "b": [["100", "1"]], "a": [["101", "1"]]}})

    def test_lob_sampled_at_1hz(self):
        # three updates in second 1, one in second 2 -> exactly one buffered row
        for ms in (1000, 1200, 1900):
            mi._handle_message(self._depth_msg("BTCUSDT", ms))
        assert mi._buf_lob["BTCUSDT"] == []
        mi._handle_message(self._depth_msg("BTCUSDT", 2100))
        assert len(mi._buf_lob["BTCUSDT"]) == 1
        assert mi._buf_lob["BTCUSDT"][0]["event_ms"] == 1900  # last book of sec 1

    def test_unknown_symbol_ignored(self):
        mi._handle_message(self._depth_msg("SOLUSDT", 1000))
        assert all(not v for v in mi._buf_lob.values())

    def test_drain_keeps_current_second_hot(self):
        msg = json.dumps({"stream": "btcusdt@trade",
                          "data": {"s": "BTCUSDT", "T": 5_000, "p": "1", "q": "1", "m": False}})
        mi._handle_message(msg)
        out = {s: (l, t) for s, l, t, _ in mi._drain_buffers(now_s=5)}
        assert out["BTCUSDT"][1] == []          # second 5 not finalized at now_s=5
        out = {s: (l, t) for s, l, t, _ in mi._drain_buffers(now_s=6)}
        assert len(out["BTCUSDT"][1]) == 1      # finalized once the second passed


class TestFeatures:
    def test_ofi_hand_computed(self):
        bb = np.array([100.0, 100.0, 99.0])
        bs = np.array([5.0, 8.0, 2.0])
        ba = np.array([101.0, 101.0, 102.0])
        asz = np.array([7.0, 4.0, 3.0])
        ofi = mf.compute_ofi(bb, bs, ba, asz)
        # t1: bid same px 5->8 (+3); ask same px 7->4 (-3) => ofi = 3-(-3) = 6
        # t2: bid down a level (-8); ask retreats up (-4) => ofi = -8-(-4) = -4
        assert ofi[0] == 0.0
        assert ofi[1] == pytest.approx(6.0)
        assert ofi[2] == pytest.approx(-4.0)

    def test_feature_frame_determinism(self, tmp_path):
        rng = np.random.default_rng(3)
        t0 = 1_784_073_600
        lob_rows, tr_rows = [], []
        px = 100.0
        for i in range(700):
            px *= float(np.exp(rng.normal(0, 1e-4)))
            lob_rows.append(mi.normalize_lob("BTCUSDT", (t0 + i) * 1000,
                                             [[str(px - 0.5), "5"]], [[str(px + 0.5), "6"]]))
            tr_rows.append({"ts_s": t0 + i, "n_trades": 3, "buy_qty": 1.0 + i % 3,
                            "sell_qty": 1.0, "buy_n": 2, "sell_n": 1, "vwap": px,
                            "last_px": px, "high": px, "low": px})
        mi.write_segment(lob_rows, "lob", "BTCUSDT", root=tmp_path)
        mi.write_segment(tr_rows, "trades1s", "BTCUSDT", root=tmp_path)
        f1 = mf.load_microstructure_features("BTCUSDT", t0, t0 + 700, root=tmp_path)
        f2 = mf.load_microstructure_features("BTCUSDT", t0, t0 + 700, root=tmp_path)
        pd.testing.assert_frame_equal(f1, f2)
        assert {"mid", "spread_bps", "imb_top1", "imb_top20", "ofi_1s",
                "signed_flow_1s", "aggression", "rv_60s", "rv_300s"} <= set(f1.columns)
        assert f1["rv_300s"].notna().sum() > 0
        assert f1["imb_top1"].iloc[0] == pytest.approx(5 / 11)

    def test_regime_conservative_on_unknown(self, tmp_path):
        idx = range(100)
        df = pd.DataFrame({"rv_300s": [1e-5] * 100, "spread_bps": [1.0] * 100}, index=idx)
        assert (mf.label_regime(df) == "toxic").all()  # < min_periods everywhere

    def test_empty_range_returns_empty(self, tmp_path):
        out = mf.load_microstructure_features("BTCUSDT", 0, 100, root=tmp_path)
        assert out.empty


class TestEnvGate:
    def test_enabled_logic(self, monkeypatch):
        monkeypatch.setenv("MICRO_INGEST", "1")
        assert mi.enabled()
        monkeypatch.setenv("MICRO_INGEST", "0")
        assert not mi.enabled()
        monkeypatch.delenv("MICRO_INGEST")
        monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
        for k in [k for k in list(__import__("os").environ) if k.startswith("RAILWAY_")]:
            monkeypatch.delenv(k)
        assert not mi.enabled()
        monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
        assert mi.enabled()

    def test_status_snapshot_shape(self):
        snap = mi.status_snapshot()
        assert set(snap) >= {"enabled", "running", "stale", "symbols", "last_error"}
        assert set(snap["symbols"]) == set(mi.SYMBOLS)
