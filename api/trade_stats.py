"""Per-side trade statistics for the baseline-vs-candidate comparison.

One pure function over a list of closed-trade dicts (the shape persisted in
trade_history.json / shadow_*_trades.json). Both the primary and the shadow
side go through the same code path so the comparison is methodologically
identical — no side gets a friendlier calculation.

Honesty rules encoded here:
- A metric that cannot be computed from the sample is None, never 0 or a guess.
- sharpe_per_trade is per-trade (NOT annualized) and suppressed below
  MIN_SHARPE_N closed trades — a Sharpe on 5 trades is noise.
- profit_factor requires at least one win and one loss; an all-win sample
  returns None rather than infinity.
- expectancy_pct is the mean per-trade pnl_pct — identical to "average trade
  return", so only one field is exposed.
"""

from typing import Dict, List, Optional

MIN_SHARPE_N = 20


def compute_trade_stats(trades: List[Dict], starting_balance: float = 1000.0) -> Dict:
    """Stats over closed trades. A win is pnl_dollar > 0 (zero counts as loss).

    Trades are sorted by exit_time so streaks and drawdown are deterministic
    regardless of file ordering.
    """
    ts = sorted(
        (t for t in trades if isinstance(t, dict) and "pnl_dollar" in t),
        key=lambda t: t.get("exit_time") or 0,
    )
    n = len(ts)
    empty: Dict[str, Optional[float]] = {
        "n": n,
        "wins": 0,
        "losses": 0,
        "win_rate": None,
        "total_pnl_dollar": 0.0,
        "total_return_pct": 0.0,
        "expectancy_pct": None,
        "avg_win_pct": None,
        "avg_loss_pct": None,
        "profit_factor": None,
        "max_drawdown_pct": None,
        "sharpe_per_trade": None,
        "current_streak": 0,
        "max_win_streak": 0,
        "max_loss_streak": 0,
        "first_trade_time": None,
        "last_trade_time": None,
    }
    if n == 0:
        return empty

    pnl_d = [float(t["pnl_dollar"]) for t in ts]
    pnl_p = [float(t.get("pnl_pct", 0.0)) for t in ts]
    win_flags = [d > 0 for d in pnl_d]
    wins = sum(win_flags)
    losses = n - wins

    win_pcts = [p for p, w in zip(pnl_p, win_flags) if w]
    loss_pcts = [p for p, w in zip(pnl_p, win_flags) if not w]
    gross_win = sum(d for d in pnl_d if d > 0)
    gross_loss = -sum(d for d in pnl_d if d <= 0)

    # equity curve → max drawdown as % below the running peak
    equity = starting_balance
    peak = starting_balance
    max_dd = 0.0
    for d in pnl_d:
        equity += d
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100.0)

    # streaks over the exit-time-ordered sequence
    cur = max_w = max_l = 0
    for w in win_flags:
        if w:
            cur = cur + 1 if cur > 0 else 1
            max_w = max(max_w, cur)
        else:
            cur = cur - 1 if cur < 0 else -1
            max_l = max(max_l, -cur)

    mean_p = sum(pnl_p) / n
    sharpe = None
    if n >= MIN_SHARPE_N:
        var = sum((p - mean_p) ** 2 for p in pnl_p) / (n - 1)
        std = var ** 0.5
        if std > 0:
            sharpe = mean_p / std

    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / n,
        "total_pnl_dollar": sum(pnl_d),
        "total_return_pct": (sum(pnl_d) / starting_balance * 100.0) if starting_balance else 0.0,
        "expectancy_pct": mean_p,
        "avg_win_pct": (sum(win_pcts) / len(win_pcts)) if win_pcts else None,
        "avg_loss_pct": (sum(loss_pcts) / len(loss_pcts)) if loss_pcts else None,
        "profit_factor": (gross_win / gross_loss) if (wins and losses and gross_loss > 0) else None,
        "max_drawdown_pct": max_dd,
        "sharpe_per_trade": sharpe,
        "current_streak": cur,
        "max_win_streak": max_w,
        "max_loss_streak": max_l,
        "first_trade_time": ts[0].get("exit_time"),
        "last_trade_time": ts[-1].get("exit_time"),
    }
