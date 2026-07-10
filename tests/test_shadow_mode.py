"""Shadow-mode isolation checks.

The invariant that matters: the shadow path writes ONLY its own file and
never touches the live trade history or the primary generator's state.

Run: .venv/bin/python -m pytest tests/test_shadow_mode.py -q
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import model_server as ms


def test_shadow_off_by_default():
    # No SHADOW_MODEL_VERSION env in the test run → feature fully off
    assert ms.SHADOW_MODEL_VERSION == ""
    assert ms.SHADOW_TRADES_JSON is None
    assert ms.shadow_signal_gen is None
    assert ms._shadow_snapshot() is None


def test_make_signal_gen_matches_versions():
    g16 = ms._make_signal_gen("v4.16")
    g18 = ms._make_signal_gen("v4.18")
    assert type(g16).__name__ == "V416SignalGenerator"
    assert g18.cfg.entry_threshold_short == 0.45  # v4.18 conviction gate
    try:
        ms._make_signal_gen("v9.99")
        assert False, "unknown version must fail fast"
    except ValueError:
        pass


def test_shadow_persist_isolated(tmp_path, monkeypatch):
    shadow_file = tmp_path / "shadow_trades.json"
    primary_file = tmp_path / "trade_history.json"
    primary_file.write_text(json.dumps([{"direction": "LONG", "pnl_pct": 1.0}]))
    before = primary_file.read_text()

    monkeypatch.setattr(ms, "SHADOW_MODEL_VERSION", "v4.17")
    monkeypatch.setattr(ms, "SHADOW_TRADES_JSON", shadow_file)
    monkeypatch.setattr(ms, "TRADE_HISTORY_JSON", primary_file)
    monkeypatch.setattr(ms, "shadow_signal_gen", ms._make_signal_gen("v4.17"))
    monkeypatch.setattr(ms, "_shadow_last_trade_count", 0)

    ms.shadow_signal_gen.trades.append(
        {"direction": "SHORT", "pnl_pct": -0.1, "pnl_dollar": -1.0}
    )
    ms._check_and_persist_shadow_trades()

    written = json.loads(shadow_file.read_text())
    assert len(written) == 1 and written[0]["direction"] == "SHORT"
    # the real trade history is byte-identical
    assert primary_file.read_text() == before
    # idempotent: no new trades → no rewrite growth
    ms._check_and_persist_shadow_trades()
    assert len(json.loads(shadow_file.read_text())) == 1


def test_shadow_status_contract_has_stats_and_matched_window(tmp_path, monkeypatch):
    """New additive keys: per-side stats, primary.window_stats, window_since."""
    import asyncio

    shadow_file = tmp_path / "shadow_trades.json"
    primary_file = tmp_path / "trade_history.json"
    # one old primary trade (before window) and one recent (inside window)
    primary_file.write_text(json.dumps([
        {"direction": "LONG", "pnl_pct": 1.0, "pnl_dollar": 10.0, "entry_time": 100, "exit_time": 200},
        {"direction": "SHORT", "pnl_pct": -0.5, "pnl_dollar": -5.0, "entry_time": 5000, "exit_time": 5100},
    ]))
    shadow_file.write_text(json.dumps([
        {"direction": "SHORT", "pnl_pct": 0.2, "pnl_dollar": 2.0, "entry_time": 4000, "exit_time": 4600},
    ]))

    monkeypatch.setattr(ms, "SHADOW_MODEL_VERSION", "v4.18")
    monkeypatch.setattr(ms, "SHADOW_TRADES_JSON", shadow_file)
    monkeypatch.setattr(ms, "TRADE_HISTORY_JSON", primary_file)
    monkeypatch.setattr(ms, "signal_gen", ms._make_signal_gen("v4.16"))
    monkeypatch.setattr(ms, "shadow_signal_gen", ms._make_signal_gen("v4.18"))

    out = asyncio.run(ms.shadow_status(authorization=None))

    assert out["enabled"] is True
    # window anchored at the earliest shadow trade's entry_time
    assert out["window_since"] == 4000
    for side in ("primary", "shadow"):
        assert out[side]["stats"]["n"] == out[side]["total_trades"]
        assert "expectancy_pct" in out[side]["stats"]
    # lifetime primary has both trades; matched window keeps only the recent one
    assert out["primary"]["stats"]["n"] == 2
    assert out["primary"]["window_stats"]["n"] == 1
    assert out["primary"]["window_stats"]["total_pnl_dollar"] == -5.0
    assert out["shadow"]["stats"]["n"] == 1
    # small samples suppress sharpe rather than faking precision
    assert out["primary"]["stats"]["sharpe_per_trade"] is None
