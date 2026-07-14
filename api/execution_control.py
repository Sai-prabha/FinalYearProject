"""Execution control plane — versioning, audit, and writer identity for the
shared auto-execute setting.

The *value* stays ``BrokerConfig.auto_execute`` in
``data/live/broker_config.json`` (the execution loop keeps reading the field
it always has). This module owns the control *metadata*:

- ``data/execution/control.json`` — ``{version, updated_at, updated_by,
  updated_via, request_id}``, atomic writes.
- ``data/execution/audit.jsonl`` — append-only trail of every change and
  every rejected stale write.

Writer identity: only the canonical Railway deployment may place demo
(broker) orders. ``is_writer()`` is env-derived — ``EXECUTION_WRITER=1/0``
overrides, otherwise "am I running on Railway". See EXECUTION_CONTROL.md.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

EXEC_DIR = Path(__file__).resolve().parent.parent / "data" / "execution"
CONTROL_PATH = EXEC_DIR / "control.json"
AUDIT_PATH = EXEC_DIR / "audit.jsonl"

# Serializes read-modify-write of the control state across concurrent PATCHes.
# ponytail: process-wide lock — fine for a single-replica control plane;
# multi-replica would need storage-level compare-and-swap instead.
LOCK = threading.Lock()

_META_DEFAULTS = {
    "version": 1,
    "updated_at": None,
    "updated_by": None,
    "updated_via": None,
    "request_id": None,
}


# ── Writer identity ────────────────────────────────────────────────────────


def _on_railway() -> bool:
    return bool(os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("RAILWAY_PROJECT_ID"))


def writer_backend() -> str:
    return "railway" if _on_railway() else "local"


def instance_id() -> str:
    return os.environ.get("RAILWAY_REPLICA_ID") or f"{socket.gethostname()}-{os.getpid()}"


def is_writer() -> bool:
    """Whether THIS instance is the designated broker-execution writer.

    ``EXECUTION_WRITER`` env wins when set; otherwise Railway is the writer
    and everything else (local dev, forks) is an observer.
    """
    override = os.environ.get("EXECUTION_WRITER")
    if override is not None:
        return override.strip().lower() in ("1", "true", "yes")
    return _on_railway()


def writer_info() -> dict:
    w = is_writer()
    return {
        "backend": writer_backend(),
        "instance_id": instance_id(),
        "is_writer": w,
        "role": "writer" if w else "observer",
    }


# ── Metadata persistence ───────────────────────────────────────────────────


def load_meta() -> dict:
    if not CONTROL_PATH.exists():
        return dict(_META_DEFAULTS)
    try:
        with open(CONTROL_PATH, "r") as f:
            data = json.load(f)
        return {**_META_DEFAULTS, **{k: data.get(k) for k in _META_DEFAULTS}}
    except Exception as e:
        logger.warning(f"execution control: failed to read {CONTROL_PATH} ({e}); using defaults")
        return dict(_META_DEFAULTS)


def _save_meta(meta: dict) -> None:
    EXEC_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="control_", suffix=".tmp", dir=str(EXEC_DIR))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(meta, indent=2))
        os.replace(tmp, CONTROL_PATH)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _audit(row: dict) -> None:
    EXEC_DIR.mkdir(parents=True, exist_ok=True)
    row = {"ts": datetime.now(timezone.utc).isoformat(), "instance": instance_id(), **row}
    with open(AUDIT_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")


def commit_change(
    prev: bool,
    new: bool,
    actor: str,
    via: str,
    request_id: Optional[str] = None,
) -> dict:
    """Record an applied auto-execute change: bump version, persist meta, audit.

    Caller holds LOCK and has already persisted the new broker config value.
    """
    meta = load_meta()
    meta = {
        "version": int(meta["version"]) + 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "updated_by": actor,
        "updated_via": via,
        "request_id": request_id,
    }
    _save_meta(meta)
    _audit({
        "event": "auto_execute_changed",
        "prev": prev,
        "new": new,
        "actor": actor,
        "via": via,
        "request_id": request_id,
        "version": meta["version"],
        "outcome": "applied",
    })
    logger.info(
        "execution control: auto_execute %s→%s by %s via %s (v%d)",
        prev, new, actor, via, meta["version"],
    )
    return meta


def audit_conflict(
    expected_version: int,
    current_version: int,
    desired: bool,
    actor: str,
    via: str,
    request_id: Optional[str] = None,
) -> None:
    """Log a rejected stale write — the id/seen-vs-current context needed to
    debug racing surfaces."""
    _audit({
        "event": "auto_execute_conflict",
        "expected_version": expected_version,
        "current_version": current_version,
        "desired": desired,
        "actor": actor,
        "via": via,
        "request_id": request_id,
        "outcome": "conflict",
    })


def control_state(auto_execute: bool, mode: str) -> dict:
    """Authoritative control document for GET/PATCH responses."""
    meta = load_meta()
    return {
        "auto_execute": auto_execute,
        "version": meta["version"],
        "updated_at": meta["updated_at"],
        "updated_by": meta["updated_by"],
        "updated_via": meta["updated_via"],
        "request_id": meta["request_id"],
        "writer": writer_info(),
        "mode": mode,
    }
