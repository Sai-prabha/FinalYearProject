"""
Runtime broker config — persisted across restarts.

Shape:

    {
        "auto_execute": false,
        "default_symbol": "BTCUSDT",
        "default_qty": 0.001,
        "default_btc_qty": 0.001,
        "default_eth_qty": 0.05
    }

``default_qty`` / ``default_symbol`` control the single-leg manual order
form in the UI.  ``default_btc_qty`` / ``default_eth_qty`` control the
sizes of the two legs that auto-execute places when the ratio model opens
or closes a position.

Stored at ``data/live/broker_config.json`` so a Railway container restart
keeps the user's last toggle state.  Atomic writes (tmp file + os.replace)
guard against corruption mid-write.

If the file is missing on first boot, defaults come from environment vars
(``BINANCE_DEFAULT_SYMBOL``, ``BINANCE_DEFAULT_QTY``,
``BINANCE_DEFAULT_BTC_QTY``, ``BINANCE_DEFAULT_ETH_QTY``);
``auto_execute`` defaults to ``False`` so we never auto-trade until the
user explicitly flips the switch.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

LIVE_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "live"
CONFIG_PATH = LIVE_DATA_DIR / "broker_config.json"


class BrokerConfig(BaseModel):
    auto_execute: bool = False
    default_symbol: str = "BTCUSDT"
    default_qty: float = Field(default=0.001, gt=0)
    # Per-leg sizes used by the dual-leg auto-execute path
    default_btc_qty: float = Field(default=0.001, gt=0)
    default_eth_qty: float = Field(default=0.05, gt=0)


def _defaults_from_env() -> BrokerConfig:
    return BrokerConfig(
        auto_execute=False,
        default_symbol=os.environ.get("BINANCE_DEFAULT_SYMBOL", "BTCUSDT").upper(),
        default_qty=float(os.environ.get("BINANCE_DEFAULT_QTY", "0.001")),
        default_btc_qty=float(os.environ.get("BINANCE_DEFAULT_BTC_QTY", "0.001")),
        default_eth_qty=float(os.environ.get("BINANCE_DEFAULT_ETH_QTY", "0.05")),
    )


def load_broker_config() -> BrokerConfig:
    """Return the persisted config, or env-derived defaults on first boot."""
    if not CONFIG_PATH.exists():
        cfg = _defaults_from_env()
        logger.info(f"Broker config: no file at {CONFIG_PATH}; using defaults {cfg.model_dump()}")
        return cfg
    try:
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
        cfg = BrokerConfig(**data)
        logger.info(f"Broker config loaded: {cfg.model_dump()}")
        return cfg
    except Exception as e:
        logger.warning(f"Broker config: failed to read {CONFIG_PATH} ({e}); falling back to defaults")
        return _defaults_from_env()


def save_broker_config(cfg: BrokerConfig) -> None:
    """Atomically persist the broker config."""
    LIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(cfg.model_dump(), indent=2)
    fd, tmp_path = tempfile.mkstemp(prefix="broker_config_", suffix=".tmp", dir=str(LIVE_DATA_DIR))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(payload)
        os.replace(tmp_path, CONFIG_PATH)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def apply_partial_update(current: BrokerConfig, partial: dict) -> BrokerConfig:
    """Build a new BrokerConfig by merging a partial update over the current one.

    Unknown keys are ignored. Validation runs through Pydantic so callers get
    consistent error messages on bad input.
    """
    base = current.model_dump()
    allowed = set(BrokerConfig.model_fields.keys())
    for key, value in partial.items():
        if key in allowed and value is not None:
            base[key] = value
    if "default_symbol" in base and isinstance(base["default_symbol"], str):
        base["default_symbol"] = base["default_symbol"].upper().strip()
    return BrokerConfig(**base)
