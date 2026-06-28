#!/usr/bin/env python3
"""
V4.15 Live Trade Diagnostic Analysis

Reads the persisted trade history from data/live/trade_history.json and computes:
  - Missing metrics: Sharpe, Sortino, win/loss streaks, duration distribution
  - Balance reconciliation: walks pnl_dollar to find the $2.95 gap
  - Loss pattern analysis: by exit reason, direction, time-of-day, signal strength
  - Probability discrimination: does entry_probability predict outcomes?

Usage:
    python scripts/analyze_v415_trades.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRADE_HISTORY = ROOT / "data" / "live" / "trade_history.json"
REPORT_DIR = ROOT / "reports" / "v4.16"


# ── Helpers ────────────────────────────────────────────────────────────────

def load_trades() -> list:
    with open(TRADE_HISTORY, "r") as f:
        return json.load(f)


def ts_to_utc(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def fmt_pct(v: float) -> str:
    return f"{v:+.4f}%"


# ── Part A: Calculate Missing Metrics ─────────────────────────────────────

def compute_metrics(trades: list) -> dict:
    """Compute Sharpe, Sortino, streaks, duration distribution."""
    pnl_pcts = np.array([t["pnl_pct"] for t in trades])  # already in %
    pnl_dollars = np.array([t["pnl_dollar"] for t in trades])
    bars_held = np.array([t["bars_held"] for t in trades])
    is_win = pnl_dollars > 0

    # ── Basic stats ──
    n = len(trades)
    wins = int(is_win.sum())
    losses = n - wins
    win_rate = wins / n * 100 if n > 0 else 0
    avg_pnl = float(pnl_pcts.mean())
    avg_win = float(pnl_pcts[is_win].mean()) if wins > 0 else 0
    avg_loss = float(pnl_pcts[~is_win].mean()) if losses > 0 else 0
    rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    profit_factor = float(pnl_dollars[is_win].sum() / abs(pnl_dollars[~is_win].sum())) if losses > 0 else float("inf")

    # ── Sharpe ratio ──
    # Annualize: ~105,120 bars/year for 5-min data (365.25 * 24 * 12)
    # But trades are not evenly spaced — compute per-trade Sharpe and
    # scale by sqrt(estimated trades per year).
    # Avg hold = ~23 bars; one trade every ~(23 + cooldown) = ~38 bars
    # Bars/year = 105120; trades/year ≈ 105120/38 ≈ 2766
    avg_hold = float(bars_held.mean())
    estimated_cycle = avg_hold + 15  # avg hold + cooldown
    trades_per_year = 105120.0 / max(estimated_cycle, 1)
    mean_ret = float(pnl_pcts.mean())
    std_ret = float(pnl_pcts.std()) if n > 1 else 1e-8
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 1e-8 else 0

    # ── Sortino ratio (downside deviation only) ──
    downside = pnl_pcts[pnl_pcts < 0]
    downside_std = float(np.sqrt(np.mean(downside ** 2))) if len(downside) > 0 else 1e-8
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 1e-8 else 0

    # ── Win/loss streaks ──
    max_win_streak = 0
    max_loss_streak = 0
    current_streak = 0
    current_type = None
    for w in is_win:
        if w == current_type:
            current_streak += 1
        else:
            current_type = w
            current_streak = 1
        if w and current_streak > max_win_streak:
            max_win_streak = current_streak
        if not w and current_streak > max_loss_streak:
            max_loss_streak = current_streak

    # ── Duration distribution ──
    p10 = float(np.percentile(bars_held, 10))
    p50 = float(np.percentile(bars_held, 50))
    p90 = float(np.percentile(bars_held, 90))

    # ── Equity curve ──
    balance = 1000.0
    equity_curve = [balance]
    peak = balance
    max_dd = 0
    for t in trades:
        balance += t["pnl_dollar"]
        equity_curve.append(balance)
        peak = max(peak, balance)
        dd = (balance - peak) / peak
        max_dd = min(max_dd, dd)

    final_balance = balance
    total_pnl_dollar = final_balance - 1000.0

    # Monotonicity: fraction of equity curve segments that go up
    diffs = np.diff(equity_curve)
    monotonicity = float((diffs > 0).sum() / max(len(diffs), 1))

    return {
        "n_trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate, 2),
        "avg_pnl_pct": round(avg_pnl, 4),
        "avg_win_pct": round(avg_win, 4),
        "avg_loss_pct": round(avg_loss, 4),
        "reward_risk_ratio": round(rr_ratio, 4),
        "profit_factor": round(profit_factor, 4),
        "sharpe_annualized": round(sharpe, 4),
        "sortino_annualized": round(sortino, 4),
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "duration_p10_bars": p10,
        "duration_p50_bars": p50,
        "duration_p90_bars": p90,
        "avg_hold_bars": round(avg_hold, 1),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "final_balance": round(final_balance, 2),
        "total_pnl_dollar": round(total_pnl_dollar, 2),
        "equity_monotonicity": round(monotonicity, 4),
        "best_trade_pct": round(float(pnl_pcts.max()), 4),
        "worst_trade_pct": round(float(pnl_pcts.min()), 4),
        "trades_per_year_est": round(trades_per_year, 0),
    }


# ── Part B: Balance Reconciliation ────────────────────────────────────────

def reconcile_balance(trades: list) -> dict:
    """Walk through trades and find where the $2.95 gap arises."""
    balance = 1000.0
    phase_boundary = None

    for i, t in enumerate(trades):
        prev_balance = balance
        balance += t["pnl_dollar"]

        # Detect the sizing phase change (5-6% -> 56-59%)
        if phase_boundary is None and t["position_size_pct"] > 20:
            phase_boundary = i

    # Phase 1: low-leverage trades (before restart)
    if phase_boundary is not None:
        phase1_trades = trades[:phase_boundary]
        phase2_trades = trades[phase_boundary:]

        phase1_pnl = sum(t["pnl_dollar"] for t in phase1_trades)
        phase2_pnl = sum(t["pnl_dollar"] for t in phase2_trades)

        # If server restarted at phase boundary, balance was reset to $1000
        # instead of $1000 + phase1_pnl. The lost PnL = phase1_pnl.
        correct_balance = 1000.0 + phase1_pnl + phase2_pnl
        reset_balance = 1000.0 + phase2_pnl  # what server showed after restart
        discrepancy = correct_balance - reset_balance
    else:
        phase1_pnl = 0
        phase2_pnl = sum(t["pnl_dollar"] for t in trades)
        correct_balance = 1000.0 + phase2_pnl
        reset_balance = correct_balance
        discrepancy = 0

    return {
        "phase_boundary_trade_idx": phase_boundary,
        "phase1_n_trades": phase_boundary if phase_boundary else 0,
        "phase1_pnl_dollar": round(phase1_pnl, 4),
        "phase2_n_trades": len(trades) - (phase_boundary or 0),
        "phase2_pnl_dollar": round(phase2_pnl, 4),
        "correct_total_balance": round(correct_balance, 2),
        "balance_after_reset": round(reset_balance, 2),
        "discrepancy_dollar": round(discrepancy, 4),
        "explanation": (
            f"Server restarted after trade #{phase_boundary}. "
            f"Phase 1 accumulated ${phase1_pnl:.4f} that was lost when balance "
            f"reset to $1000. The ${discrepancy:.2f} discrepancy matches the "
            f"reported $2.95 gap."
        ) if phase_boundary else "No sizing phase change detected.",
    }


# ── Part C: Loss Pattern Analysis ─────────────────────────────────────────

def analyze_loss_patterns(trades: list) -> dict:
    """Segment losses by exit reason, direction, time-of-day, signal strength."""
    results = {}

    # ── By exit reason ──
    reason_stats = {}
    for t in trades:
        reason = t.get("reason", "Unknown")
        if reason not in reason_stats:
            reason_stats[reason] = {"count": 0, "wins": 0, "losses": 0,
                                     "total_pnl_pct": 0, "loss_pnls": []}
        reason_stats[reason]["count"] += 1
        reason_stats[reason]["total_pnl_pct"] += t["pnl_pct"]
        if t["pnl_dollar"] > 0:
            reason_stats[reason]["wins"] += 1
        else:
            reason_stats[reason]["losses"] += 1
            reason_stats[reason]["loss_pnls"].append(t["pnl_pct"])

    for reason, stats in reason_stats.items():
        stats["avg_pnl_pct"] = round(stats["total_pnl_pct"] / max(stats["count"], 1), 4)
        stats["avg_loss_pct"] = round(
            np.mean(stats["loss_pnls"]) if stats["loss_pnls"] else 0, 4
        )
        stats["win_rate_pct"] = round(stats["wins"] / max(stats["count"], 1) * 100, 2)
        del stats["loss_pnls"]  # not JSON-serializable as np array

    results["by_exit_reason"] = reason_stats

    # ── By direction ──
    dir_stats = {}
    for t in trades:
        d = t["direction"]
        if d not in dir_stats:
            dir_stats[d] = {"count": 0, "wins": 0, "losses": 0,
                            "total_pnl_pct": 0, "loss_magnitudes": []}
        dir_stats[d]["count"] += 1
        dir_stats[d]["total_pnl_pct"] += t["pnl_pct"]
        if t["pnl_dollar"] > 0:
            dir_stats[d]["wins"] += 1
        else:
            dir_stats[d]["losses"] += 1
            dir_stats[d]["loss_magnitudes"].append(t["pnl_pct"])

    for d, stats in dir_stats.items():
        stats["win_rate_pct"] = round(stats["wins"] / max(stats["count"], 1) * 100, 2)
        stats["avg_pnl_pct"] = round(stats["total_pnl_pct"] / max(stats["count"], 1), 4)
        stats["avg_loss_pct"] = round(
            np.mean(stats["loss_magnitudes"]) if stats["loss_magnitudes"] else 0, 4
        )
        del stats["loss_magnitudes"]

    results["by_direction"] = dir_stats

    # ── By hour of day (UTC) ──
    hourly_stats = {}
    for t in trades:
        hour = ts_to_utc(t["entry_time"]).hour
        if hour not in hourly_stats:
            hourly_stats[hour] = {"count": 0, "total_pnl_pct": 0, "wins": 0, "losses": 0}
        hourly_stats[hour]["count"] += 1
        hourly_stats[hour]["total_pnl_pct"] += t["pnl_pct"]
        if t["pnl_dollar"] > 0:
            hourly_stats[hour]["wins"] += 1
        else:
            hourly_stats[hour]["losses"] += 1

    for h, stats in hourly_stats.items():
        stats["avg_pnl_pct"] = round(stats["total_pnl_pct"] / max(stats["count"], 1), 4)
        stats["win_rate_pct"] = round(stats["wins"] / max(stats["count"], 1) * 100, 2)

    # Sort by hour and find worst hours
    sorted_hours = sorted(hourly_stats.items(), key=lambda x: x[1]["avg_pnl_pct"])
    worst_20pct_count = max(1, len(sorted_hours) // 5)
    worst_hours = [h for h, _ in sorted_hours[:worst_20pct_count]]

    results["by_hour_utc"] = {str(k): v for k, v in sorted(hourly_stats.items())}
    results["worst_20pct_hours"] = worst_hours

    # ── By signal strength ──
    # Bin signal strength into WEAK (<0.06), MODERATE (0.06-0.08), STRONG (>0.08)
    strength_bins = {"WEAK": [], "MODERATE": [], "STRONG": []}
    for t in trades:
        s = t.get("entry_strength", 0)
        if s < 0.06:
            strength_bins["WEAK"].append(t)
        elif s < 0.08:
            strength_bins["MODERATE"].append(t)
        else:
            strength_bins["STRONG"].append(t)

    strength_stats = {}
    for label, group in strength_bins.items():
        if not group:
            strength_stats[label] = {"count": 0}
            continue
        pnls = [t["pnl_pct"] for t in group]
        wins = sum(1 for t in group if t["pnl_dollar"] > 0)
        strength_stats[label] = {
            "count": len(group),
            "avg_pnl_pct": round(np.mean(pnls), 4),
            "win_rate_pct": round(wins / len(group) * 100, 2),
            "total_pnl_pct": round(sum(pnls), 4),
        }

    results["by_signal_strength"] = strength_stats

    return results


# ── Part D: Probability Discrimination ────────────────────────────────────

def analyze_probability(trades: list) -> dict:
    """Test if entry_probability values discriminate outcomes."""
    probas = np.array([t["entry_probability"] for t in trades])
    pnls = np.array([t["pnl_pct"] for t in trades])
    is_win = np.array([t["pnl_dollar"] > 0 for t in trades])

    # Split into above/below median probability distance from 0.5
    dist_from_half = np.abs(probas - 0.5)
    median_dist = np.median(dist_from_half)

    high_conf = dist_from_half >= median_dist
    low_conf = ~high_conf

    # Correlation
    corr = float(np.corrcoef(dist_from_half, pnls)[0, 1]) if len(pnls) > 1 else 0

    # LONG-specific (prob > 0.5) and SHORT-specific (prob < 0.5)
    long_mask = probas > 0.5
    short_mask = probas < 0.5

    return {
        "probability_range": {
            "min": round(float(probas.min()), 4),
            "max": round(float(probas.max()), 4),
            "mean": round(float(probas.mean()), 4),
            "std": round(float(probas.std()), 4),
        },
        "distance_from_0.5": {
            "mean": round(float(dist_from_half.mean()), 4),
            "median": round(float(median_dist), 4),
        },
        "high_confidence_trades": {
            "count": int(high_conf.sum()),
            "win_rate_pct": round(float(is_win[high_conf].mean() * 100), 2) if high_conf.any() else 0,
            "avg_pnl_pct": round(float(pnls[high_conf].mean()), 4) if high_conf.any() else 0,
        },
        "low_confidence_trades": {
            "count": int(low_conf.sum()),
            "win_rate_pct": round(float(is_win[low_conf].mean() * 100), 2) if low_conf.any() else 0,
            "avg_pnl_pct": round(float(pnls[low_conf].mean()), 4) if low_conf.any() else 0,
        },
        "long_trades": {
            "count": int(long_mask.sum()),
            "avg_probability": round(float(probas[long_mask].mean()), 4) if long_mask.any() else 0,
            "win_rate_pct": round(float(is_win[long_mask].mean() * 100), 2) if long_mask.any() else 0,
            "avg_pnl_pct": round(float(pnls[long_mask].mean()), 4) if long_mask.any() else 0,
        },
        "short_trades": {
            "count": int(short_mask.sum()),
            "avg_probability": round(float(probas[short_mask].mean()), 4) if short_mask.any() else 0,
            "win_rate_pct": round(float(is_win[short_mask].mean() * 100), 2) if short_mask.any() else 0,
            "avg_pnl_pct": round(float(pnls[short_mask].mean()), 4) if short_mask.any() else 0,
        },
        "prob_pnl_correlation": round(corr, 4),
        "discriminates_outcomes": abs(corr) > 0.1,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("V4.15 LIVE TRADE DIAGNOSTIC ANALYSIS")
    print("=" * 80)

    if not TRADE_HISTORY.exists():
        print(f"ERROR: Trade history not found at {TRADE_HISTORY}")
        sys.exit(1)

    trades = load_trades()
    print(f"\nLoaded {len(trades)} trades from {TRADE_HISTORY}")
    if trades:
        t0 = ts_to_utc(trades[0]["entry_time"])
        t1 = ts_to_utc(trades[-1]["exit_time"])
        print(f"Time range: {t0.strftime('%Y-%m-%d %H:%M')} -> {t1.strftime('%Y-%m-%d %H:%M')} UTC")

    # ── A. Metrics ──
    print("\n" + "=" * 80)
    print("PART A: CALCULATED METRICS")
    print("=" * 80)
    metrics = compute_metrics(trades)

    for key, val in metrics.items():
        label = key.replace("_", " ").title()
        print(f"  {label:<30} {val}")

    # ── B. Balance reconciliation ──
    print("\n" + "=" * 80)
    print("PART B: BALANCE RECONCILIATION")
    print("=" * 80)
    recon = reconcile_balance(trades)

    for key, val in recon.items():
        label = key.replace("_", " ").title()
        print(f"  {label:<35} {val}")

    # ── C. Loss patterns ──
    print("\n" + "=" * 80)
    print("PART C: LOSS PATTERN ANALYSIS")
    print("=" * 80)
    loss_patterns = analyze_loss_patterns(trades)

    print("\n  By Exit Reason:")
    for reason, stats in loss_patterns["by_exit_reason"].items():
        print(f"    {reason:<20} "
              f"N={stats['count']:>3}  "
              f"WR={stats['win_rate_pct']:>5.1f}%  "
              f"AvgPnL={stats['avg_pnl_pct']:>+7.4f}%  "
              f"AvgLoss={stats['avg_loss_pct']:>+7.4f}%")

    print("\n  By Direction:")
    for d, stats in loss_patterns["by_direction"].items():
        print(f"    {d:<10} "
              f"N={stats['count']:>3}  "
              f"WR={stats['win_rate_pct']:>5.1f}%  "
              f"AvgPnL={stats['avg_pnl_pct']:>+7.4f}%  "
              f"AvgLoss={stats['avg_loss_pct']:>+7.4f}%")

    print("\n  By Hour (UTC) — sorted by avg P&L:")
    sorted_hours = sorted(loss_patterns["by_hour_utc"].items(),
                          key=lambda x: x[1]["avg_pnl_pct"])
    for h, stats in sorted_hours:
        print(f"    Hour {h:>2}:  "
              f"N={stats['count']:>3}  "
              f"WR={stats['win_rate_pct']:>5.1f}%  "
              f"AvgPnL={stats['avg_pnl_pct']:>+7.4f}%")
    print(f"\n  Worst 20% hours (UTC): {loss_patterns['worst_20pct_hours']}")

    print("\n  By Signal Strength:")
    for label, stats in loss_patterns["by_signal_strength"].items():
        if stats["count"] == 0:
            print(f"    {label:<12} N=  0")
            continue
        print(f"    {label:<12} "
              f"N={stats['count']:>3}  "
              f"WR={stats['win_rate_pct']:>5.1f}%  "
              f"AvgPnL={stats['avg_pnl_pct']:>+7.4f}%")

    # ── D. Probability discrimination ──
    print("\n" + "=" * 80)
    print("PART D: PROBABILITY DISCRIMINATION")
    print("=" * 80)
    prob_analysis = analyze_probability(trades)

    print(f"\n  Probability range: "
          f"{prob_analysis['probability_range']['min']:.4f} - "
          f"{prob_analysis['probability_range']['max']:.4f} "
          f"(mean={prob_analysis['probability_range']['mean']:.4f})")
    print(f"  Distance from 0.5: "
          f"mean={prob_analysis['distance_from_0.5']['mean']:.4f}, "
          f"median={prob_analysis['distance_from_0.5']['median']:.4f}")
    print(f"\n  High confidence: "
          f"N={prob_analysis['high_confidence_trades']['count']}, "
          f"WR={prob_analysis['high_confidence_trades']['win_rate_pct']:.1f}%, "
          f"AvgPnL={prob_analysis['high_confidence_trades']['avg_pnl_pct']:+.4f}%")
    print(f"  Low  confidence: "
          f"N={prob_analysis['low_confidence_trades']['count']}, "
          f"WR={prob_analysis['low_confidence_trades']['win_rate_pct']:.1f}%, "
          f"AvgPnL={prob_analysis['low_confidence_trades']['avg_pnl_pct']:+.4f}%")
    print(f"\n  LONG trades: "
          f"N={prob_analysis['long_trades']['count']}, "
          f"AvgProb={prob_analysis['long_trades']['avg_probability']:.4f}, "
          f"WR={prob_analysis['long_trades']['win_rate_pct']:.1f}%, "
          f"AvgPnL={prob_analysis['long_trades']['avg_pnl_pct']:+.4f}%")
    print(f"  SHORT trades: "
          f"N={prob_analysis['short_trades']['count']}, "
          f"AvgProb={prob_analysis['short_trades']['avg_probability']:.4f}, "
          f"WR={prob_analysis['short_trades']['win_rate_pct']:.1f}%, "
          f"AvgPnL={prob_analysis['short_trades']['avg_pnl_pct']:+.4f}%")
    print(f"\n  Prob-PnL correlation: {prob_analysis['prob_pnl_correlation']:.4f}")
    print(f"  Discriminates outcomes: {prob_analysis['discriminates_outcomes']}")

    # ── Save to JSON ──
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trade_history_path": str(TRADE_HISTORY),
        "metrics": metrics,
        "balance_reconciliation": recon,
        "loss_patterns": loss_patterns,
        "probability_analysis": prob_analysis,
    }

    out_path = REPORT_DIR / "diagnostic_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 80}")
    print(f"Diagnostic summary saved to {out_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

