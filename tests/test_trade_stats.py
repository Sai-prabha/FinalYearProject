"""Hand-computed fixtures for the baseline-vs-candidate stats engine."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.trade_stats import MIN_SHARPE_N, compute_trade_stats


def _t(pnl_dollar, pnl_pct, exit_time):
    return {
        "pnl_dollar": pnl_dollar,
        "pnl_pct": pnl_pct,
        "entry_time": exit_time - 600,
        "exit_time": exit_time,
    }


def test_empty_sample_suppresses_everything():
    s = compute_trade_stats([])
    assert s["n"] == 0
    assert s["win_rate"] is None
    assert s["expectancy_pct"] is None
    assert s["profit_factor"] is None
    assert s["sharpe_per_trade"] is None
    assert s["max_drawdown_pct"] is None
    assert s["last_trade_time"] is None
    assert s["total_pnl_dollar"] == 0.0


def test_known_sample_hand_computed():
    # exit-time order: +10, -5, -5, +20  (W L L W), start balance 1000
    trades = [
        _t(20.0, 2.0, 400),  # deliberately out of order in the file
        _t(10.0, 1.0, 100),
        _t(-5.0, -0.5, 200),
        _t(-5.0, -0.5, 300),
    ]
    s = compute_trade_stats(trades, starting_balance=1000.0)
    assert s["n"] == 4 and s["wins"] == 2 and s["losses"] == 2
    assert s["win_rate"] == 0.5
    assert s["total_pnl_dollar"] == 20.0
    assert math.isclose(s["total_return_pct"], 2.0)
    assert math.isclose(s["expectancy_pct"], (1.0 - 0.5 - 0.5 + 2.0) / 4)  # 0.5
    assert math.isclose(s["avg_win_pct"], 1.5)
    assert math.isclose(s["avg_loss_pct"], -0.5)
    assert math.isclose(s["profit_factor"], 30.0 / 10.0)
    # equity: 1010 (peak), 1005, 1000 → dd = 10/1010; then 1020 recovers
    assert math.isclose(s["max_drawdown_pct"], 10.0 / 1010.0 * 100.0)
    assert s["sharpe_per_trade"] is None  # n=4 < MIN_SHARPE_N
    assert s["current_streak"] == 1  # ends on a win
    assert s["max_win_streak"] == 1 and s["max_loss_streak"] == 2
    assert s["first_trade_time"] == 100 and s["last_trade_time"] == 400


def test_zero_pnl_counts_as_loss_and_all_win_profit_factor_none():
    s = compute_trade_stats([_t(0.0, 0.0, 100), _t(5.0, 0.5, 200)])
    assert s["wins"] == 1 and s["losses"] == 1
    # zero-dollar loss → gross_loss == 0 → profit factor suppressed, not inf
    assert s["profit_factor"] is None

    all_wins = compute_trade_stats([_t(5.0, 0.5, i) for i in range(1, 4)])
    assert all_wins["profit_factor"] is None
    assert all_wins["win_rate"] == 1.0
    assert all_wins["avg_loss_pct"] is None
    assert all_wins["max_drawdown_pct"] == 0.0


def test_sharpe_gated_then_computed():
    # alternating +1% / -0.5% for MIN_SHARPE_N trades → stable positive sharpe
    trades = [
        _t(10.0 if i % 2 == 0 else -5.0, 1.0 if i % 2 == 0 else -0.5, 100 * (i + 1))
        for i in range(MIN_SHARPE_N)
    ]
    s = compute_trade_stats(trades)
    assert s["sharpe_per_trade"] is not None
    mean = 0.25
    var = sum(((1.0 if i % 2 == 0 else -0.5) - mean) ** 2 for i in range(MIN_SHARPE_N)) / (
        MIN_SHARPE_N - 1
    )
    assert math.isclose(s["sharpe_per_trade"], mean / var**0.5)

    one_short = compute_trade_stats(trades[:-1])
    assert one_short["sharpe_per_trade"] is None


def test_malformed_rows_skipped():
    s = compute_trade_stats([{"garbage": True}, "not a dict", _t(5.0, 0.5, 100)])
    assert s["n"] == 1 and s["wins"] == 1
