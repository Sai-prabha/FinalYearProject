"""
Broker-level pre-trade safety guards.

Called once per position transition, *after* the signal generator has
decided to move but *before* any order is sent to the broker.

These guards protect the execution layer from bad config, stale data,
and runaway positions.  They are intentionally separate from the signal
generator's own cooldown/circuit-breaker, which operate at the model layer.

All thresholds are read from environment variables at construction time
so they can be tuned without code changes.

Env vars (all optional — defaults shown):
  GUARD_MAX_NOTIONAL_USDT   max USD value per leg (price × qty)  default: 500
  GUARD_STALE_SECONDS       reject signals older than this        default: 120
  GUARD_MIN_CONFIDENCE      min model probability for entry       default: 0.0 (disabled)
  GUARD_MAX_POSITIONS       max concurrent open positions         default: 1
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    allowed: bool
    reason: str  # "ok" | "stale_signal" | "notional_limit" | "low_confidence" | "max_positions"


@dataclass
class ExecutionGuards:
    max_notional_usdt: float = field(
        default_factory=lambda: float(os.environ.get("GUARD_MAX_NOTIONAL_USDT", "500"))
    )
    stale_seconds: int = field(
        default_factory=lambda: int(os.environ.get("GUARD_STALE_SECONDS", "120"))
    )
    min_confidence: float = field(
        default_factory=lambda: float(os.environ.get("GUARD_MIN_CONFIDENCE", "0.0"))
    )
    max_positions: int = field(
        default_factory=lambda: int(os.environ.get("GUARD_MAX_POSITIONS", "1"))
    )

    def check_entry(
        self,
        *,
        confidence: float,
        candle_ts: int,           # unix seconds (from kline["time"])
        leg_price: float,         # reference price for the larger leg (BTC close)
        leg_qty: float,           # quantity for the larger leg (default_btc_qty)
        open_position_count: int = 0,
    ) -> GuardResult:
        """
        Run pre-entry checks.  Returns on the first failure.

        Exits to flat (new_pos == 0) bypass this entirely — callers should
        only invoke check_entry when the target position is non-zero.

        ``open_position_count`` should reflect positions *before* this trade.
        For the single ratio-strategy path, pass 0 on fresh entries and
        reversals (the old position is being replaced, not accumulated).
        """
        # 1. Stale signal: candle closed too long ago for a safe entry
        age = int(time.time()) - candle_ts
        if age > self.stale_seconds:
            logger.warning(
                "GUARD BLOCKED stale_signal: signal is %ds old (limit %ds)",
                age, self.stale_seconds,
            )
            return GuardResult(allowed=False, reason="stale_signal")

        # 2. Notional cap: protect against misconfigured qty
        notional = leg_price * leg_qty
        if notional > self.max_notional_usdt:
            logger.warning(
                "GUARD BLOCKED notional_limit: %.2f USDT > max %.2f USDT (price=%.2f qty=%.6f)",
                notional, self.max_notional_usdt, leg_price, leg_qty,
            )
            return GuardResult(allowed=False, reason="notional_limit")

        # 3. Confidence floor (disabled when min_confidence == 0.0)
        if self.min_confidence > 0.0 and confidence < self.min_confidence:
            logger.warning(
                "GUARD BLOCKED low_confidence: %.4f < min %.4f",
                confidence, self.min_confidence,
            )
            return GuardResult(allowed=False, reason="low_confidence")

        # 4. Position cap
        if open_position_count >= self.max_positions:
            logger.warning(
                "GUARD BLOCKED max_positions: %d open >= limit %d",
                open_position_count, self.max_positions,
            )
            return GuardResult(allowed=False, reason="max_positions")

        return GuardResult(allowed=True, reason="ok")

    def to_dict(self) -> dict:
        return {
            "max_notional_usdt": self.max_notional_usdt,
            "stale_seconds": self.stale_seconds,
            "min_confidence": self.min_confidence,
            "max_positions": self.max_positions,
        }
