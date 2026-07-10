"""Unit tests for the session max-loss kill switch (api/kill_switch.py)."""

import json

import pytest

from api.kill_switch import KillSwitch


# ── transitions ─────────────────────────────────────────────────────────


def test_default_is_disarmed_and_allows_entries():
    ks = KillSwitch()
    assert ks.state == "disarmed"
    assert ks.entries_allowed


def test_arm_sets_limit_and_resets_session():
    ks = KillSwitch()
    ks.session_pnl_usdt = -50.0
    ks.arm(200.0)
    assert ks.state == "armed"
    assert ks.limit_usdt == 200.0
    assert ks.session_pnl_usdt == 0.0
    assert ks.armed_at is not None


def test_arm_rejects_bad_states_and_limits():
    ks = KillSwitch()
    with pytest.raises(ValueError):
        ks.arm(-5)
    ks.arm(100)
    with pytest.raises(ValueError):
        ks.arm(100)  # already armed


def test_disarm_from_armed_and_tripped_but_not_disarmed():
    ks = KillSwitch()
    with pytest.raises(ValueError):
        ks.disarm()
    ks.arm(100)
    ks.disarm()
    assert ks.state == "disarmed"
    ks.arm(10)
    ks.record_pnl(-10)
    assert ks.state == "tripped"
    ks.disarm()  # explicit escape hatch
    assert ks.state == "disarmed"


def test_rearm_only_from_tripped_resets_session():
    ks = KillSwitch()
    with pytest.raises(ValueError):
        ks.rearm()
    ks.arm(10)
    ks.record_pnl(-15)
    assert ks.state == "tripped"
    ks.rearm()
    assert ks.state == "armed"
    assert ks.session_pnl_usdt == 0.0
    assert ks.tripped_at is None


# ── tripping logic ──────────────────────────────────────────────────────


def test_trips_exactly_at_limit():
    ks = KillSwitch()
    ks.arm(100)
    assert ks.record_pnl(-99.99) is False
    assert ks.entries_allowed
    assert ks.record_pnl(-0.01) is True
    assert ks.state == "tripped"
    assert not ks.entries_allowed


def test_profits_offset_losses():
    ks = KillSwitch()
    ks.arm(100)
    ks.record_pnl(-80)
    ks.record_pnl(50)
    assert ks.record_pnl(-60) is False  # net -90, still above -100
    assert ks.state == "armed"


def test_record_pnl_ignored_when_not_armed():
    ks = KillSwitch()
    assert ks.record_pnl(-1000) is False
    assert ks.state == "disarmed"
    ks.arm(10)
    ks.record_pnl(-10)
    ks.record_pnl(-1000)  # already tripped, no double transition
    assert ks.state == "tripped"


# ── persistence ─────────────────────────────────────────────────────────


def test_state_round_trips_through_file(tmp_path):
    f = tmp_path / "kill_switch.json"
    ks = KillSwitch(path=f)
    ks.arm(75)
    ks.record_pnl(-80)
    assert ks.state == "tripped"

    restored = KillSwitch.load(f)
    assert restored.state == "tripped"
    assert restored.limit_usdt == 75
    assert restored.session_pnl_usdt == -80
    assert not restored.entries_allowed


def test_corrupt_or_missing_file_yields_safe_default(tmp_path):
    missing = KillSwitch.load(tmp_path / "nope.json")
    assert missing.state == "disarmed"

    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    corrupt = KillSwitch.load(bad)
    assert corrupt.state == "disarmed"

    weird = tmp_path / "weird.json"
    weird.write_text(json.dumps({"state": "exploded"}))
    assert KillSwitch.load(weird).state == "disarmed"
