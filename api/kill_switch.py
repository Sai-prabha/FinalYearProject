"""
Session max-loss kill switch.

A reversible runtime gate, not a delete. Three states:

  disarmed  no effect at all (default; zero behavior change when unused)
  armed     realized session PnL is tracked against a max-loss limit
  tripped   limit breached; new entries are blocked, exits stay allowed

State transitions (all explicit, all logged, all persisted):
  disarmed -> armed      arm(limit_usdt)
  armed    -> disarmed   disarm()
  armed    -> tripped    automatic, when session loss reaches the limit
  tripped  -> armed      rearm()   resets session PnL, fresh session
  tripped  -> disarmed   disarm()  explicit escape hatch

State survives server restarts via a small JSON file next to trade history.
Rollback path: disarm() or delete the state file; disarmed is a no-op.

Env vars:
  KILL_SWITCH_LIMIT_USDT   default arm limit suggestion (default: 100)

Design note: JARVIS/design/session-kill-switch-panel-design.md
Runbook:     JARVIS/ops/kill-switch-runbook.md
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

VALID_STATES = ("disarmed", "armed", "tripped")


@dataclass
class KillSwitch:
    state: str = "disarmed"
    limit_usdt: float = field(
        default_factory=lambda: float(os.environ.get("KILL_SWITCH_LIMIT_USDT", "100"))
    )
    session_pnl_usdt: float = 0.0
    armed_at: Optional[float] = None
    tripped_at: Optional[float] = None
    path: Optional[Path] = None  # persistence target; None disables persistence

    # ── state queries ──────────────────────────────────────────────────

    @property
    def entries_allowed(self) -> bool:
        """New entries are blocked only while tripped."""
        return self.state != "tripped"

    # ── transitions ────────────────────────────────────────────────────

    def arm(self, limit_usdt: Optional[float] = None) -> None:
        if self.state != "disarmed":
            raise ValueError(f"cannot arm from state {self.state!r}")
        if limit_usdt is not None:
            if limit_usdt <= 0:
                raise ValueError("limit_usdt must be positive")
            self.limit_usdt = float(limit_usdt)
        self.state = "armed"
        self.session_pnl_usdt = 0.0
        self.armed_at = time.time()
        self.tripped_at = None
        self._log_and_save("ARMED", f"limit={self.limit_usdt:.2f} USDT")

    def disarm(self) -> None:
        if self.state == "disarmed":
            raise ValueError("already disarmed")
        prev = self.state
        self.state = "disarmed"
        self._log_and_save("DISARMED", f"from={prev} session_pnl={self.session_pnl_usdt:.2f}")

    def rearm(self) -> None:
        if self.state != "tripped":
            raise ValueError(f"cannot rearm from state {self.state!r}")
        self.state = "armed"
        self.session_pnl_usdt = 0.0
        self.armed_at = time.time()
        self.tripped_at = None
        self._log_and_save("REARMED", f"limit={self.limit_usdt:.2f} USDT, fresh session")

    def record_pnl(self, pnl_usdt: float) -> bool:
        """Accumulate realized PnL for a closed trade. Returns True if this
        trade tripped the switch."""
        if self.state != "armed":
            return False
        self.session_pnl_usdt += float(pnl_usdt)
        if self.session_pnl_usdt <= -self.limit_usdt:
            self.state = "tripped"
            self.tripped_at = time.time()
            self._log_and_save(
                "TRIPPED",
                f"session_pnl={self.session_pnl_usdt:.2f} <= -{self.limit_usdt:.2f} USDT; entries blocked",
            )
            return True
        self._save()
        return False

    # ── persistence ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "limit_usdt": self.limit_usdt,
            "session_pnl_usdt": round(self.session_pnl_usdt, 4),
            "armed_at": self.armed_at,
            "tripped_at": self.tripped_at,
        }

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.to_dict(), indent=2))
        except OSError as exc:  # persistence failure must never block trading logic
            logger.error("KILL_SWITCH persist failed: %s", exc)

    def _log_and_save(self, event: str, detail: str) -> None:
        logger.warning("KILL_SWITCH %s | %s", event, detail)
        self._save()

    @classmethod
    def load(cls, path: Path) -> "KillSwitch":
        """Restore persisted state; corrupt or missing file yields a fresh
        disarmed switch (safe default)."""
        ks = cls(path=path)
        try:
            data = json.loads(path.read_text())
            if data.get("state") in VALID_STATES:
                ks.state = data["state"]
                ks.limit_usdt = float(data.get("limit_usdt", ks.limit_usdt))
                ks.session_pnl_usdt = float(data.get("session_pnl_usdt", 0.0))
                ks.armed_at = data.get("armed_at")
                ks.tripped_at = data.get("tripped_at")
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        return ks
