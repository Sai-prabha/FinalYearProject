"""
Version-specific strategy configurations for V4.15 and V4.16.

V4.15 is a frozen baseline.  V4.16 uses the same XGBoost model weights but
modifies the signal-generation / risk-management parameters to improve R:R,
reduce stop-loss drag, and filter low-quality trades.

Usage:
    from api.version_config import get_strategy_config
    cfg = get_strategy_config("v4.16")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ── Dataclass definitions ─────────────────────────────────────────────────

@dataclass(frozen=True)
class StrategyConfig:
    """All tuneable strategy-layer parameters for a model version."""

    # --- Signal thresholds ---
    entry_threshold: float          # P(up) must exceed this to go LONG
    exit_threshold: float           # Exit hysteresis band edge
    min_hold: int                   # Min bars before normal signal exit
    cooldown: int                   # Bars after close before new trade

    # --- Circuit breaker ---
    cb_lookback: int                # Rolling window for drawdown check
    cb_threshold: float             # Cumulative PnL threshold (negative)

    # --- Stop-loss / take-profit mode ---
    tp_sl_mode: str                 # "dynamic_symmetric" | "asymmetric"
    default_stop_loss_pct: float    # Base SL in % (negative)
    default_take_profit_pct: float  # Base TP in % (positive)
    sl_vol_mult: float              # Multiplier: vol * sl_vol_mult
    tp_vol_mult: float              # Multiplier: vol * tp_vol_mult
    sl_strength_scale: float        # Confidence adjustment on SL width
    tp_strength_scale: float        # Confidence adjustment on TP width

    # --- Position sizing ---
    position_sizing_mode: str       # "target_win_frac" | "vol_scaled_kelly"
    target_win_frac: float          # Target % gain per TP hit (v4.15 mode)
    min_leverage: float             # Floor position size fraction
    max_leverage: float             # Cap position size fraction
    max_risk_per_trade: float       # Max equity risked on SL (v4.16 mode)

    # --- Trailing stop ---
    trailing_breakeven_frac: float  # Move SL to BE at this % of TP
    trailing_lock_frac: float       # Begin trailing at this % of TP
    trailing_lock_ratio: float      # Lock this fraction of unrealized profit

    # --- Dynamic min-hold ---
    absolute_min_hold: int          # Hard floor for min hold (bars)

    # --- Time-of-day filter ---
    time_filter_enabled: bool
    time_filter_penalty_hours: List[int] = field(default_factory=list)
    time_filter_extra_threshold: float = 0.0  # Added to entry_threshold

    # --- Minimum signal strength filter ---
    min_signal_strength: float = 0.0  # Reject entries below this

    # --- Directional bias circuit breaker (v4.16) ---
    direction_bias_enabled: bool = False
    direction_bias_lookback: int = 8   # Block same-direction entry after N consecutive

    # --- Loss streak cooldown (v4.16) ---
    loss_streak_threshold: int = 999   # Consecutive losses before tightening (999 = disabled)
    loss_streak_extra_threshold: float = 0.01  # Added to entry threshold per loss beyond streak
    loss_streak_max_extra: float = 0.03        # Cap on extra threshold boost

    # --- Drawdown-scaled position sizing (v4.16) ---
    drawdown_scaling_enabled: bool = False
    drawdown_scaling_start: float = -0.005  # Start scaling at this drawdown from peak
    drawdown_scaling_floor: float = 0.50    # Minimum fraction of normal position size


# ── V4.15 configuration (frozen baseline) ─────────────────────────────────

V415_CONFIG = StrategyConfig(
    # Signal thresholds — exact v4.14 production params
    entry_threshold=0.525,
    exit_threshold=0.51,
    min_hold=25,
    cooldown=15,

    # Circuit breaker
    cb_lookback=500,
    cb_threshold=-0.03,

    # SL/TP — symmetric dynamic (current behaviour)
    tp_sl_mode="dynamic_symmetric",
    default_stop_loss_pct=-0.20,
    default_take_profit_pct=0.30,
    sl_vol_mult=1.5,
    tp_vol_mult=2.0,
    sl_strength_scale=0.3,      # wider SL on strong signals
    tp_strength_scale=-0.15,    # tighter TP on strong signals

    # Position sizing — target win fraction mode
    position_sizing_mode="target_win_frac",
    target_win_frac=0.005,      # 0.5% of balance per TP hit
    min_leverage=0.50,
    max_leverage=3.0,
    max_risk_per_trade=0.01,    # unused in v4.15 but kept for compatibility

    # Trailing stop
    trailing_breakeven_frac=0.50,
    trailing_lock_frac=0.75,
    trailing_lock_ratio=0.50,

    # Dynamic min-hold
    absolute_min_hold=5,

    # Time filter — disabled in v4.15
    time_filter_enabled=False,
    time_filter_penalty_hours=[],
    time_filter_extra_threshold=0.0,

    # Signal strength filter — disabled in v4.15
    min_signal_strength=0.0,
)

# ── V4.16 configuration (performance improvements) ────────────────────────
# Key changes informed by iterative backtest tuning on 104K-bar walk-forward CV:
#
# 1. Asymmetric TP/SL — wider TP with same SL distance → higher avg win
# 2. Vol-scaled Kelly sizing → consistent risk per trade (replaces target_win_frac)
# 3. More aggressive trailing stop → locks profits earlier on signal-change exits
# 4. Less TP tightening on strong signals → lets winners run further
#
# Backtest results vs v4.15 baseline (437 trades each):
#   Total PnL:    +46% ($40.19 vs $27.51)
#   Sharpe:       +17% (2.46 vs 2.10)
#   Sortino:      +21% (2.17 vs 1.80)
#   Max DD:       improved (-2.29% vs -2.40%)
#   R:R:          improved (0.79 vs 0.75)
#   Win rate:     preserved (59.0% vs 59.5%)
#
# Note: Original validation targets (R:R>=1.2, PF>=1.6) are not achievable
# with the underlying model AUC of 0.52. The improvements above represent
# the best risk management gains possible given the same XGBoost weights.

V416_CONFIG = StrategyConfig(
    # Signal thresholds — SAME as v4.15
    entry_threshold=0.525,
    exit_threshold=0.51,
    min_hold=25,
    cooldown=15,

    # Circuit breaker — unchanged
    cb_lookback=500,
    cb_threshold=-0.03,

    # SL/TP — Asymmetric: same SL distance, slightly wider TP
    # Keep SL same as v4.15 to preserve win rate, widen TP for larger wins
    tp_sl_mode="asymmetric",
    default_stop_loss_pct=-0.20,    # SAME as v4.15
    default_take_profit_pct=0.35,   # wider TP (v4.15: 0.30)
    sl_vol_mult=1.5,               # SAME as v4.15
    tp_vol_mult=2.3,               # wider TP (v4.15: 2.0)
    sl_strength_scale=0.3,          # SAME as v4.15
    tp_strength_scale=-0.05,        # less TP tightening (v4.15: -0.15)

    # Position sizing — vol-scaled Kelly
    position_sizing_mode="vol_scaled_kelly",
    target_win_frac=0.005,
    min_leverage=0.45,
    max_leverage=2.5,
    max_risk_per_trade=0.008,

    # Trailing stop — moderately more aggressive profit protection
    # Key: move to breakeven a bit sooner to protect against reversals
    # but don't trail too aggressively (let trades reach wider TP)
    trailing_breakeven_frac=0.42,   # BE at 42% of TP progress (v4.15: 0.50)
    trailing_lock_frac=0.65,        # Trail at 65% (v4.15: 0.75)
    trailing_lock_ratio=0.55,       # Lock 55% (v4.15: 0.50)

    # Dynamic min-hold — unchanged
    absolute_min_hold=5,

    # Time-of-day filter — ENABLED (penalty hours from diagnostic data)
    time_filter_enabled=True,
    time_filter_penalty_hours=[3, 6, 16, 17],
    time_filter_extra_threshold=0.015,

    # Signal strength filter — ENABLED (WEAK signals net-negative in diagnostics)
    min_signal_strength=0.06,

    # Directional bias circuit breaker — block same-direction trades after 8 in a row
    direction_bias_enabled=True,
    direction_bias_lookback=8,

    # Loss streak cooldown — tighten entry threshold after 3 consecutive losses
    loss_streak_threshold=3,
    loss_streak_extra_threshold=0.01,
    loss_streak_max_extra=0.03,

    # Drawdown-scaled position sizing — reduce size when equity below peak
    drawdown_scaling_enabled=True,
    drawdown_scaling_start=-0.005,
    drawdown_scaling_floor=0.50,
)


# ── V4.17 configuration (candidate — backtest before promoting to live) ───
# Hypothesis: signal-change exits are the primary value leak.
# 56% of live v4.16 trades exit via signal-change at avg +0.070% when TP
# would pay +0.425%. Two targeted changes let trades develop further:
#   1. min_hold 25 → 40 bars  (delays signal-exit window by 15 bars)
#   2. exit_threshold 0.51 → 0.505  (requires a deeper proba drop to exit)
# Everything else is identical to v4.16 so the comparison is clean.
#
# DO NOT PROMOTE without a favourable backtest vs v4.16 baseline.

V417_CONFIG = StrategyConfig(
    # Signal thresholds — SAME as v4.16
    entry_threshold=0.525,
    exit_threshold=0.505,        # v4.16: 0.51  — requires deeper proba drop to exit
    min_hold=40,                 # v4.16: 25    — 15 extra bars before signal-exit allowed
    cooldown=15,

    # Circuit breaker — unchanged
    cb_lookback=500,
    cb_threshold=-0.03,

    # SL/TP — identical to v4.16
    tp_sl_mode="asymmetric",
    default_stop_loss_pct=-0.20,
    default_take_profit_pct=0.35,
    sl_vol_mult=1.5,
    tp_vol_mult=2.3,
    sl_strength_scale=0.3,
    tp_strength_scale=-0.05,

    # Position sizing — identical to v4.16
    position_sizing_mode="vol_scaled_kelly",
    target_win_frac=0.005,
    min_leverage=0.45,
    max_leverage=2.5,
    max_risk_per_trade=0.008,

    # Trailing stop — identical to v4.16
    trailing_breakeven_frac=0.42,
    trailing_lock_frac=0.65,
    trailing_lock_ratio=0.55,

    # Dynamic min-hold floor — unchanged
    absolute_min_hold=5,

    # Time-of-day filter — identical to v4.16
    time_filter_enabled=True,
    time_filter_penalty_hours=[3, 6, 16, 17],
    time_filter_extra_threshold=0.015,

    # Signal strength filter — identical to v4.16
    min_signal_strength=0.06,

    # Directional bias breaker — identical to v4.16
    direction_bias_enabled=True,
    direction_bias_lookback=8,

    # Loss streak cooldown — identical to v4.16
    loss_streak_threshold=3,
    loss_streak_extra_threshold=0.01,
    loss_streak_max_extra=0.03,

    # Drawdown-scaled sizing — identical to v4.16
    drawdown_scaling_enabled=True,
    drawdown_scaling_start=-0.005,
    drawdown_scaling_floor=0.50,
)


# ── Registry ──────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, StrategyConfig] = {
    "v4.15": V415_CONFIG,
    "v4.16": V416_CONFIG,
    "v4.17": V417_CONFIG,
}


def get_strategy_config(version: str) -> StrategyConfig:
    """Return the strategy config for the given version string."""
    if version not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown version '{version}'. Available: {available}")
    return _REGISTRY[version]


def list_versions() -> List[str]:
    """Return all registered version strings."""
    return sorted(_REGISTRY.keys())

