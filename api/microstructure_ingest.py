"""Microstructure ingestion — L2 partial-book + trade tape (read-only).

Design record: MICROSTRUCTURE.md. Production PUBLIC market data from
wss://fstream.binance.com — keyless, subscribe-only. This module contains no
signed request, no POST, no broker import, and no order path of any kind.
The broker host allowlist in broker_client.py is untouched by construction.

Storage: immutable parquet segments (tmp+rename) under
data/live/micro/{lob,trades1s[,tape]}/<SYMBOL>/<YYYY-MM-DD>/<epoch>.parquet.
A crash loses at most one flush interval.
"""

import asyncio
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import websockets

logger = logging.getLogger(__name__)

MICRO_DIR = Path(__file__).resolve().parent.parent / "data" / "live" / "micro"
WS_URL_BASE = "wss://fstream.binance.com/stream"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
LEVELS = 20
STALE_AFTER_S = 120.0

FLUSH_SECONDS = int(os.environ.get("MICRO_FLUSH_SECONDS", "600"))
RETENTION_DAYS = int(os.environ.get("MICRO_RETENTION_DAYS", "180"))
TAPE_RETENTION_DAYS = int(os.environ.get("MICRO_TAPE_RETENTION_DAYS", "14"))
RAW_TAPE = os.environ.get("MICRO_RAW_TAPE", "0") == "1"


def enabled() -> bool:
    """MICRO_INGEST=1 forces on, =0 forces off; unset -> on only on Railway
    (mirrors the writer-gate auto-detection so local dev servers stay quiet)."""
    v = os.environ.get("MICRO_INGEST")
    if v == "1":
        return True
    if v == "0":
        return False
    return any(k.startswith("RAILWAY_") for k in os.environ)


# ── Normalization (pure, tested) ───────────────────────────────────────────

def normalize_lob(symbol: str, event_ms: int, bids: List, asks: List,
                  levels: int = LEVELS) -> Dict:
    """Flatten a partial-book snapshot into one flat row (missing levels:
    px=NaN, sz=0). Sides are re-sorted defensively (bids desc, asks asc)."""
    b = sorted(((float(p), float(q)) for p, q in bids), key=lambda x: -x[0])[:levels]
    a = sorted(((float(p), float(q)) for p, q in asks), key=lambda x: x[0])[:levels]
    row: Dict = {"ts_ms": int(time.time() * 1000), "event_ms": int(event_ms)}
    for i in range(levels):
        bp, bs = b[i] if i < len(b) else (float("nan"), 0.0)
        ap, asz = a[i] if i < len(a) else (float("nan"), 0.0)
        row[f"bid_px_{i}"], row[f"bid_sz_{i}"] = bp, bs
        row[f"ask_px_{i}"], row[f"ask_sz_{i}"] = ap, asz
    best_bid = b[0][0] if b else float("nan")
    best_ask = a[0][0] if a else float("nan")
    row["best_bid"], row["best_ask"] = best_bid, best_ask
    row["mid"] = (best_bid + best_ask) / 2.0
    row["spread"] = best_ask - best_bid
    row["bid_depth20"] = sum(q for _, q in b)
    row["ask_depth20"] = sum(q for _, q in a)
    return row


def parse_trade(d: Dict) -> Dict:
    """@trade payload -> normalized trade (p/q/m/T; live-verified 2026-07-14 —
    @aggTrade is silently absent on fstream combined streams). m=True means the BUYER was the
    maker, i.e. the aggressor SOLD."""
    return {
        "ts_ms": int(d["T"]),
        "price": float(d["p"]),
        "qty": float(d["q"]),
        "is_buy": not bool(d["m"]),
    }


def agg_trade_into(acc: Dict, trade: Dict) -> None:
    """Fold one trade into its 1-second accumulator (keyed externally)."""
    q, px, buy = trade["qty"], trade["price"], trade["is_buy"]
    if not acc:
        acc.update({"n_trades": 0, "buy_qty": 0.0, "sell_qty": 0.0, "buy_n": 0,
                    "sell_n": 0, "pv": 0.0, "qsum": 0.0, "last_px": px,
                    "high": px, "low": px})
    acc["n_trades"] += 1
    acc["pv"] += px * q
    acc["qsum"] += q
    acc["last_px"] = px
    acc["high"] = max(acc["high"], px)
    acc["low"] = min(acc["low"], px)
    if buy:
        acc["buy_qty"] += q
        acc["buy_n"] += 1
    else:
        acc["sell_qty"] += q
        acc["sell_n"] += 1


def finalize_second(ts_s: int, acc: Dict) -> Dict:
    return {
        "ts_s": ts_s, "n_trades": acc["n_trades"],
        "buy_qty": acc["buy_qty"], "sell_qty": acc["sell_qty"],
        "buy_n": acc["buy_n"], "sell_n": acc["sell_n"],
        "vwap": acc["pv"] / acc["qsum"] if acc["qsum"] else acc["last_px"],
        "last_px": acc["last_px"], "high": acc["high"], "low": acc["low"],
    }


# ── Segment storage (pure paths in/out, tested) ────────────────────────────

def write_segment(rows: List[Dict], kind: str, symbol: str,
                  root: Path = MICRO_DIR) -> List[Path]:
    """Write rows as immutable day-partitioned parquet segments (tmp+rename).
    Rows spanning a UTC day boundary land in their own day's segment."""
    if not rows:
        return []
    df = pd.DataFrame(rows)
    # canonical time axis: 1s aggregates -> ts_s; LOB -> exchange event time
    ts_col = "ts_s" if "ts_s" in df.columns else (
        "event_ms" if "event_ms" in df.columns else "ts_ms")
    secs = df[ts_col] if ts_col == "ts_s" else df[ts_col] // 1000
    df["_day"] = pd.to_datetime(secs, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    written = []
    for day, part in df.groupby("_day"):
        part = part.drop(columns="_day").sort_values(ts_col)
        d = root / kind / symbol / str(day)
        d.mkdir(parents=True, exist_ok=True)
        start = int(part[ts_col].iloc[0]) if ts_col == "ts_s" else int(part[ts_col].iloc[0] // 1000)
        final = d / f"{start}.parquet"
        tmp = d / f".{start}.parquet.tmp"
        part.to_parquet(tmp, index=False)
        tmp.rename(final)
        written.append(final)
    return written


def prune_old_days(root: Path = MICRO_DIR, today: Optional[str] = None) -> List[Path]:
    """Delete day directories past retention. Returns what was removed."""
    today_d = datetime.now(timezone.utc).date() if today is None else \
        datetime.strptime(today, "%Y-%m-%d").date()
    removed = []
    for kind, keep_days in (("lob", RETENTION_DAYS), ("trades1s", RETENTION_DAYS),
                            ("tape", TAPE_RETENTION_DAYS)):
        base = root / kind
        if not base.exists():
            continue
        for sym_dir in base.iterdir():
            if not sym_dir.is_dir():
                continue
            for day_dir in sym_dir.iterdir():
                try:
                    age = (today_d - datetime.strptime(day_dir.name, "%Y-%m-%d").date()).days
                except ValueError:
                    continue
                if age > keep_days:
                    shutil.rmtree(day_dir, ignore_errors=True)
                    removed.append(day_dir)
    return removed


# ── Runtime state ──────────────────────────────────────────────────────────

_state: Dict[str, Dict] = {s: {"last_lob_ms": None, "last_trade_ms": None,
                               "rows_flushed_today": 0, "segments_today": 0}
                           for s in SYMBOLS}
_last_error: Optional[str] = None
_started = False
_buf_lob: Dict[str, List[Dict]] = {s: [] for s in SYMBOLS}
_pending_lob: Dict[str, Optional[tuple]] = {s: None for s in SYMBOLS}  # (sec, row)
_acc_trades: Dict[str, Dict[int, Dict]] = {s: {} for s in SYMBOLS}
_buf_tape: Dict[str, List[Dict]] = {s: [] for s in SYMBOLS}
_state_day: str = ""


def status_snapshot() -> Dict:
    now_ms = time.time() * 1000
    per_symbol = {}
    stale = False
    for s in SYMBOLS:
        st = _state[s]
        lob_age = (now_ms - st["last_lob_ms"]) / 1000 if st["last_lob_ms"] else None
        tr_age = (now_ms - st["last_trade_ms"]) / 1000 if st["last_trade_ms"] else None
        if _started and (lob_age is None or lob_age > STALE_AFTER_S):
            stale = True
        per_symbol[s] = {
            "last_lob_age_s": round(lob_age, 1) if lob_age is not None else None,
            "last_trade_age_s": round(tr_age, 1) if tr_age is not None else None,
            "rows_flushed_today": st["rows_flushed_today"],
            "segments_today": st["segments_today"],
        }
    return {"enabled": enabled(), "running": _started, "stale": stale if _started else None,
            "symbols": per_symbol, "last_error": _last_error,
            "flush_seconds": FLUSH_SECONDS, "retention_days": RETENTION_DAYS,
            "raw_tape": RAW_TAPE}


# ── Message routing ────────────────────────────────────────────────────────

def _handle_message(raw: str) -> None:
    global _last_error
    msg = json.loads(raw)
    stream = msg.get("stream", "")
    d = msg.get("data", {})
    symbol = d.get("s")
    if symbol not in _state:
        return
    if "@depth" in stream:
        event_ms = int(d.get("E", time.time() * 1000))
        row = normalize_lob(symbol, event_ms, d.get("b", []), d.get("a", []))
        sec = event_ms // 1000
        pending = _pending_lob[symbol]
        if pending is not None and pending[0] != sec:
            _buf_lob[symbol].append(pending[1])   # 1 Hz sample: last book of each second
        _pending_lob[symbol] = (sec, row)
        _state[symbol]["last_lob_ms"] = event_ms
    elif "@trade" in stream:
        tr = parse_trade(d)
        acc = _acc_trades[symbol].setdefault(tr["ts_ms"] // 1000, {})
        agg_trade_into(acc, tr)
        _state[symbol]["last_trade_ms"] = tr["ts_ms"]
        if RAW_TAPE:
            _buf_tape[symbol].append(tr)


async def _ws_loop() -> None:
    global _last_error
    streams = "/".join(f"{s.lower()}@depth{LEVELS}@500ms/{s.lower()}@trade"
                       for s in SYMBOLS)
    url = f"{WS_URL_BASE}?streams={streams}"
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(url, ping_interval=180, ping_timeout=60) as ws:
                logger.info("Microstructure WS connected (%d streams)", 2 * len(SYMBOLS))
                _last_error = None
                backoff = 1.0
                async for message in ws:
                    try:
                        _handle_message(message)
                    except Exception as e:  # malformed frame: log, keep streaming
                        _last_error = f"parse: {e}"
                        logger.warning("micro parse error: %s", e)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _last_error = f"ws: {e}"
            logger.warning("Microstructure WS error: %s — reconnect in %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)


def _drain_buffers(now_s: Optional[int] = None):
    """Swap out everything flushable. Current second's trade acc stays hot."""
    now_s = int(time.time()) if now_s is None else now_s
    out = []
    for s in SYMBOLS:
        lob_rows, _buf_lob[s] = _buf_lob[s], []
        done = {k: v for k, v in _acc_trades[s].items() if k < now_s}
        for k in done:
            del _acc_trades[s][k]
        tr_rows = [finalize_second(k, v) for k, v in sorted(done.items())]
        tape_rows, _buf_tape[s] = _buf_tape[s], []
        out.append((s, lob_rows, tr_rows, tape_rows))
    return out


async def _flush_loop() -> None:
    global _last_error, _state_day
    while True:
        await asyncio.sleep(FLUSH_SECONDS)
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if today != _state_day:
                _state_day = today
                for s in SYMBOLS:
                    _state[s]["rows_flushed_today"] = 0
                    _state[s]["segments_today"] = 0
            for s, lob_rows, tr_rows, tape_rows in _drain_buffers():
                segs = []
                segs += await asyncio.to_thread(write_segment, lob_rows, "lob", s)
                segs += await asyncio.to_thread(write_segment, tr_rows, "trades1s", s)
                if RAW_TAPE and tape_rows:
                    segs += await asyncio.to_thread(write_segment, tape_rows, "tape", s)
                _state[s]["rows_flushed_today"] += len(lob_rows) + len(tr_rows)
                _state[s]["segments_today"] += len(segs)
        except Exception as e:
            _last_error = f"flush: {e}"
            logger.error("micro flush error: %s", e, exc_info=True)


async def _janitor_loop() -> None:
    while True:
        try:
            removed = await asyncio.to_thread(prune_old_days)
            for p in removed:
                logger.info("micro retention: removed %s", p)
        except Exception as e:
            logger.error("micro janitor error: %s", e)
        await asyncio.sleep(24 * 3600)


_tasks: List[asyncio.Task] = []


def start_microstructure_ingestion() -> List[asyncio.Task]:
    """Spawn ws + flush + janitor tasks. Caller (lifespan) owns cancellation."""
    global _started, _tasks
    if _started:
        return _tasks
    _started = True
    _tasks = [asyncio.create_task(_ws_loop()),
              asyncio.create_task(_flush_loop()),
              asyncio.create_task(_janitor_loop())]
    logger.info("Microstructure ingestion started (%s, flush=%ss, retention=%sd)",
                ",".join(SYMBOLS), FLUSH_SECONDS, RETENTION_DAYS)
    return _tasks


async def stop_microstructure_ingestion() -> None:
    global _started
    _started = False
    for t in _tasks:
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    # final drain so a clean shutdown persists the last partial interval
    for s, lob_rows, tr_rows, tape_rows in _drain_buffers(now_s=int(time.time()) + 1):
        write_segment(lob_rows, "lob", s)
        write_segment(tr_rows, "trades1s", s)
        if RAW_TAPE and tape_rows:
            write_segment(tape_rows, "tape", s)
