"""Trade reconciliation engine — design record: TRADE_RECONCILIATION.md.

Invariants implemented here:
  * broker history is source of truth for realized execution,
  * the model ledger is source of truth for intent/expectation,
  * position parity is the primary (always-on) invariant,
  * per-trade economics are an attribution layer: realized − expected is
    decomposed into slippage / fees / funding / residual, and a trade is
    "broken" only when linkage fails or the residual/sign cannot be explained.

Everything in this module is read-only with respect to the broker: the sync
path calls GET userTrades / GET income and appends to a local JSONL cache.
No code path here can place, amend, or cancel an order.

Pure functions take plain dicts so tests can drive synthetic scenarios; the
only stateful piece is BrokerHistoryStore (cache file + sync cursor).
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

RECON_SYNC_SECONDS = int(os.environ.get("RECON_SYNC_SECONDS", "300"))

# Status taxonomy (TRADE_RECONCILIATION.md §7)
MATCHED = "MATCHED"
EXPLAINED_COSTS = "EXPLAINED_COSTS"
SIZE_DRIFT = "SIZE_DRIFT"
PARTIAL_EXECUTION = "PARTIAL_EXECUTION"
TIMING_DRIFT = "TIMING_DRIFT"
MODEL_ONLY = "MODEL_ONLY"
MODEL_ONLY_BREAK = "MODEL_ONLY_BREAK"
FILLS_MISSING = "FILLS_MISSING"
BROKER_ONLY = "BROKER_ONLY"
UNEXPLAINED_DELTA = "UNEXPLAINED_DELTA"
SIGN_MISMATCH = "SIGN_MISMATCH"
PENDING = "PENDING"
UNVERIFIED_PAPER = "UNVERIFIED_PAPER"

INFO, WARNING, CRITICAL = "info", "warning", "critical"

_SEVERITY = {
    MATCHED: INFO,
    EXPLAINED_COSTS: INFO,
    TIMING_DRIFT: INFO,          # escalated to warning when the gap is large
    MODEL_ONLY: INFO,
    PENDING: INFO,
    UNVERIFIED_PAPER: INFO,
    SIZE_DRIFT: WARNING,
    PARTIAL_EXECUTION: WARNING,
    UNEXPLAINED_DELTA: WARNING,  # escalated to critical past the hard cap
    MODEL_ONLY_BREAK: CRITICAL,
    FILLS_MISSING: CRITICAL,
    SIGN_MISMATCH: CRITICAL,
    BROKER_ONLY: CRITICAL,       # downgraded to warning for manual API orders
}

_EXEC_SYMBOLS = ("BTCUSDT", "ETHUSDT")
_WINDOW_S = 180          # fallback matcher window
_TIMING_WARN_S = 60      # window match beyond this ⇒ warning
_SIZE_DRIFT_FRAC = 0.25  # filled vs planned qty deviation ⇒ SIZE_DRIFT
_HARD_CAP_MULT = 5       # residual beyond 5× tolerance ⇒ critical


def _tolerance(entry_notional: float) -> float:
    """Explained-vs-unexplained band on the *residual after attribution*:
    max($0.02, 2 bps of entry notional)."""
    return max(0.02, 0.0002 * abs(entry_notional))


# ── Normalization ─────────────────────────────────────────────────────────

def normalize_fill(raw: dict) -> Optional[dict]:
    """Binance userTrades row → canonical BrokerFill. None if malformed."""
    try:
        return {
            "kind": "fill",
            "id": int(raw["id"]),
            "symbol": str(raw["symbol"]).upper(),
            "order_id": str(raw["orderId"]),
            "side": str(raw.get("side", "")),
            "price": float(raw["price"]),
            "qty": float(raw["qty"]),
            "quote_qty": float(raw.get("quoteQty", 0.0) or 0.0),
            "realized_pnl": float(raw.get("realizedPnl", 0.0) or 0.0),
            "commission": float(raw.get("commission", 0.0) or 0.0),
            "commission_asset": str(raw.get("commissionAsset", "")),
            "maker": bool(raw.get("maker", False)),
            "time_ms": int(raw["time"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def normalize_funding(raw: dict) -> Optional[dict]:
    """Binance income row (FUNDING_FEE) → canonical FundingEvent."""
    try:
        t = int(raw["time"])
        income = float(raw.get("income", 0.0) or 0.0)
        symbol = str(raw.get("symbol", "")).upper()
        return {
            "kind": "funding",
            "id": raw.get("tranId") or f"{symbol}-{t}-{income}",
            "symbol": symbol,
            "income": income,
            "time_ms": t,
        }
    except (KeyError, TypeError, ValueError):
        return None


# ── Broker history store (cache + cursor; restart- and late-data-safe) ────

class BrokerHistoryStore:
    """Append-only local cache of broker fills + funding with a sync cursor.

    Reconciliation is a pure recomputation over the cache, so a fill that
    arrives late simply lands in the cache and the next computation uses it.
    """

    def __init__(self, live_dir: Path):
        self.fills_path = live_dir / "broker_fills.jsonl"
        self.cursor_path = live_dir / "recon_sync.json"

    # -- read --

    def load(self) -> Tuple[List[dict], List[dict]]:
        fills: Dict[tuple, dict] = {}
        funding: Dict[tuple, dict] = {}
        if not self.fills_path.exists():
            return [], []
        try:
            with open(self.fills_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = (row.get("symbol"), str(row.get("id")))
                    if row.get("kind") == "fill":
                        fills[key] = row
                    elif row.get("kind") == "funding":
                        funding[key] = row
        except OSError as e:
            logger.warning(f"broker history read failed: {e}")
        fs = sorted(fills.values(), key=lambda r: r["time_ms"])
        fu = sorted(funding.values(), key=lambda r: r["time_ms"])
        return fs, fu

    def cursor(self) -> dict:
        if self.cursor_path.exists():
            try:
                with open(self.cursor_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"fills": {}, "income_ms": None, "last_sync": None}

    def last_sync_age_s(self) -> Optional[float]:
        ls = self.cursor().get("last_sync")
        if not ls:
            return None
        try:
            dt = datetime.fromisoformat(ls)
            return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
        except ValueError:
            return None

    # -- write --

    def _append(self, rows: List[dict]) -> None:
        if not rows:
            return
        self.fills_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.fills_path, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def sync(self, broker) -> dict:
        """Pull new fills + funding from the broker into the cache.

        Blocking (requests) — call via asyncio.to_thread. Never raises.
        Returns {"added_fills", "added_funding"} or {"error": str} /
        {"unsupported": True} (paper mode).
        """
        cursor = self.cursor()
        existing_fills, existing_funding = self.load()
        seen = {("fill", r["symbol"], str(r["id"])) for r in existing_fills}
        seen |= {("funding", r["symbol"], str(r["id"])) for r in existing_funding}
        now_ms = int(time.time() * 1000)
        added_fills = 0
        added_funding = 0
        supported = False

        for symbol in _EXEC_SYMBOLS:
            last_id = cursor["fills"].get(symbol)
            from_id = int(last_id) + 1 if last_id is not None else None
            start_ms = None if from_id is not None else now_ms - int(6.5 * 86_400_000)
            for _ in range(10):  # page cap; 10k fills per symbol per sync is plenty
                raw = broker.get_user_trades(symbol, start_ms=start_ms, from_id=from_id, limit=1000)
                if raw is None:
                    break
                supported = True
                batch = [n for n in (normalize_fill(r) for r in raw) if n]
                new_rows = [r for r in batch if ("fill", r["symbol"], str(r["id"])) not in seen]
                self._append(new_rows)
                for r in new_rows:
                    seen.add(("fill", r["symbol"], str(r["id"])))
                added_fills += len(new_rows)
                if batch:
                    cursor["fills"][symbol] = max(
                        int(cursor["fills"].get(symbol) or 0),
                        max(r["id"] for r in batch),
                    )
                if len(raw) < 1000:
                    break
                from_id = max(r["id"] for r in batch) + 1
                start_ms = None

        income_start = cursor.get("income_ms")
        raw_income = broker.get_income(
            income_type="FUNDING_FEE",
            start_ms=(income_start + 1) if income_start else now_ms - int(6.5 * 86_400_000),
            limit=1000,
        )
        if raw_income is not None:
            supported = True
            batch = [n for n in (normalize_funding(r) for r in raw_income) if n]
            new_rows = [r for r in batch if ("funding", r["symbol"], str(r["id"])) not in seen]
            self._append(new_rows)
            added_funding = len(new_rows)
            if batch:
                cursor["income_ms"] = max(int(income_start or 0), max(r["time_ms"] for r in batch))

        if not supported:
            return {"unsupported": True}

        cursor["last_sync"] = datetime.now(timezone.utc).isoformat()
        try:
            tmp = self.cursor_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(cursor, f)
            os.replace(tmp, self.cursor_path)
        except OSError as e:
            logger.warning(f"recon cursor write failed: {e}")
        logger.info(f"recon sync: +{added_fills} fills, +{added_funding} funding rows")
        return {"added_fills": added_fills, "added_funding": added_funding}


# ── Eligibility windows (was auto-execute ON at time t?) ─────────────────

def load_eligibility(audit_path: Path) -> List[Tuple[float, bool]]:
    """Sorted (epoch_s, enabled) change points from the control-plane audit.

    Conservative: with no audit trail we assume NOT eligible, so a missing
    file can never manufacture critical breaks.
    """
    points: List[Tuple[float, bool]] = []
    if not audit_path.exists():
        return points
    try:
        with open(audit_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("event") != "auto_execute_changed":
                    continue
                try:
                    ts = datetime.fromisoformat(str(row["ts"])).timestamp()
                except (KeyError, ValueError):
                    continue
                points.append((ts, bool(row.get("new"))))
    except OSError as e:
        logger.warning(f"eligibility read failed: {e}")
    points.sort()
    return points


def eligible_at(points: List[Tuple[float, bool]], t_s: float) -> bool:
    state = False
    for ts, enabled in points:
        if ts > t_s:
            break
        state = enabled
    return state


# ── Linkage ───────────────────────────────────────────────────────────────

def parse_client_id(client_id: str) -> Optional[int]:
    """v415-<signal_epoch_s>-<label>-<SYM> → signal_epoch_s, else None."""
    parts = (client_id or "").split("-")
    if len(parts) >= 3 and parts[0] == "v415":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def _event_signal_ts(event: dict) -> Optional[float]:
    ts = event.get("signal_ts")
    if ts is not None:
        try:
            return float(ts)
        except (TypeError, ValueError):
            pass
    for leg in event.get("legs", []) or []:
        got = parse_client_id(leg.get("client_id", ""))
        if got is not None:
            return float(got)
    return None


def _event_wall_ts(event: dict) -> Optional[float]:
    try:
        return datetime.fromisoformat(str(event.get("timestamp"))).timestamp()
    except (TypeError, ValueError):
        return None


def _dir_pos(direction: str) -> int:
    return 1 if direction == "LONG" else -1


def find_event(events: List[dict], t_s: float, role: str, direction: str) -> Tuple[Optional[dict], Optional[str]]:
    """Locate the transition event for a model trade boundary.

    role "entry": event must open the trade direction (new_pos == dir).
    role "exit":  event must leave it (prev_pos == dir, new_pos != dir).
    Returns (event, method) where method ∈ {"signal_ts", "window", None}.
    """
    pos = _dir_pos(direction)

    def _role_ok(ev: dict) -> bool:
        if role == "entry":
            return ev.get("new_pos") == pos
        return ev.get("prev_pos") == pos and ev.get("new_pos") != pos

    # Exact: stamped signal_ts or client_id-derived
    for ev in events:
        sig = _event_signal_ts(ev)
        if sig is not None and abs(sig - t_s) < 1 and _role_ok(ev):
            return ev, "signal_ts"

    # Fallback: nearest role-consistent event within the window (wall clock
    # or signal ts) — pre-upgrade rows have neither signal_ts nor client_id.
    best, best_gap = None, None
    for ev in events:
        if not _role_ok(ev):
            continue
        ref = _event_signal_ts(ev)
        if ref is None:
            ref = _event_wall_ts(ev)
        if ref is None:
            continue
        gap = abs(ref - t_s)
        if gap <= _WINDOW_S and (best_gap is None or gap < best_gap):
            best, best_gap = ev, gap
    if best is not None:
        return best, "window"
    return None, None


# ── Attribution ───────────────────────────────────────────────────────────

def _vwap(fills: List[dict]) -> Tuple[float, float]:
    """(vwap, total_qty) over fills; (0, 0) when empty."""
    q = sum(f["qty"] for f in fills)
    if q <= 0:
        return 0.0, 0.0
    return sum(f["price"] * f["qty"] for f in fills) / q, q


def _legs_for(event: Optional[dict], key: str) -> List[dict]:
    if not event:
        return []
    return [l for l in (event.get(key) or []) if l.get("status") != "SKIPPED"]


def attribute(
    trade: dict,
    entry_event: Optional[dict],
    exit_event: Optional[dict],
    fills_by_order: Dict[str, List[dict]],
    funding: List[dict],
) -> dict:
    """Per-round-trip economics at broker size (TRADE_RECONCILIATION.md §6).

    Sign convention: every bucket is a signed USDT contribution to realized
    gross (slippage negative = cost). fees ≤ 0, funding signed as paid.
    """
    entry_legs = _legs_for(entry_event, "entry_legs")
    exit_legs = _legs_for(exit_event, "exit_legs")
    flags: List[str] = []

    expected_gross = 0.0
    expected_known = True
    slippage = 0.0
    realized_gross = 0.0
    fees = 0.0
    entry_notional = 0.0
    fill_count = 0
    legs_out: List[dict] = []

    for symbol in _EXEC_SYMBOLS:
        en = next((l for l in entry_legs if l.get("symbol") == symbol), None)
        ex = next((l for l in exit_legs if l.get("symbol") == symbol), None)
        en_fills = fills_by_order.get(str(en.get("order_id"))) if en else None
        ex_fills = fills_by_order.get(str(ex.get("order_id"))) if ex else None
        en_fills = en_fills or []
        ex_fills = ex_fills or []
        fill_count += len(en_fills) + len(ex_fills)

        s = 0
        if en:
            s = 1 if en.get("side") == "BUY" else -1
        elif ex:
            s = -1 if ex.get("side") == "BUY" else 1  # exit side reverses the leg

        f_en, q_en = _vwap(en_fills)
        f_ex, q_ex = _vwap(ex_fills)
        d_en = (en or {}).get("decision_price") or ((en or {}).get("sizing") or {}).get("price")
        d_ex = (ex or {}).get("decision_price")

        for f in en_fills + ex_fills:
            realized_gross += f["realized_pnl"]
            if f["commission_asset"] in ("USDT", ""):
                fees -= f["commission"]
            else:
                flags.append("non-usdt-commission")
        entry_notional += f_en * q_en

        if s and q_ex > 0 and d_en and d_ex:
            expected_gross += s * q_ex * (d_ex - d_en)
        elif en or ex:
            expected_known = False
            flags.append("no-decision-price")
        if s and q_en > 0 and d_en:
            slippage += s * q_en * (d_en - f_en)
        if s and q_ex > 0 and d_ex:
            slippage += s * q_ex * (f_ex - d_ex)

        planned_en = float((en or {}).get("qty") or 0.0)
        if en and planned_en > 0 and abs(q_en - planned_en) / planned_en > _SIZE_DRIFT_FRAC:
            flags.append("size-drift")
        if en and q_en > 0 and ex and q_ex > 0 and abs(q_en - q_ex) / max(q_en, q_ex) > _SIZE_DRIFT_FRAC:
            flags.append("entry-exit-qty-gap")
        if (en and q_en < planned_en * 0.999 and en.get("status") not in ("REJECTED", "ERROR")):
            if q_en > 0:
                flags.append("partial-fill")
        if en or ex:
            legs_out.append({
                "symbol": symbol,
                "side": (en or {}).get("side") or ("SELL" if (ex or {}).get("side") == "BUY" else "BUY"),
                "planned_qty": planned_en or float((ex or {}).get("qty") or 0.0),
                "entry": {"order_id": (en or {}).get("order_id"), "decision_price": d_en,
                          "fill_avg": f_en or None, "filled_qty": q_en,
                          "fills": en_fills},
                "exit": {"order_id": (ex or {}).get("order_id"), "decision_price": d_ex,
                         "fill_avg": f_ex or None, "filled_qty": q_ex,
                         "fills": ex_fills},
            })

    lo_ms = int(float(trade["entry_time"]) * 1000)
    hi_ms = int(float(trade["exit_time"]) * 1000)
    funding_total = sum(
        r["income"] for r in funding
        if r["symbol"] in _EXEC_SYMBOLS and lo_ms <= r["time_ms"] <= hi_ms
    )

    residual = (realized_gross - expected_gross - slippage) if expected_known else None
    realized_net = realized_gross + fees + funding_total
    tol = _tolerance(entry_notional)
    return {
        "expected_gross": round(expected_gross, 6) if expected_known else None,
        "realized_gross": round(realized_gross, 6),
        "slippage": round(slippage, 6),
        "fees": round(fees, 6),
        "funding": round(funding_total, 6),
        "residual": round(residual, 6) if residual is not None else None,
        "realized_net": round(realized_net, 6),
        "tolerance": round(tol, 6),
        "entry_notional": round(entry_notional, 4),
        "fill_count": fill_count,
        "legs": legs_out,
        "flags": sorted(set(flags)),
    }


# ── Status decision ───────────────────────────────────────────────────────

def _sign(x: float) -> int:
    return 0 if abs(x) < 1e-12 else (1 if x > 0 else -1)


def decide_status(attr: dict, entry_method: str, exit_method: str,
                  entry_gap_s: float, exit_gap_s: float) -> Tuple[str, str, str]:
    """(status, severity, explanation) for a fully-linked trade with fills."""
    exp = attr["expected_gross"]
    net = attr["realized_net"]
    gross = attr["realized_gross"]
    res = attr["residual"]
    tol = attr["tolerance"]
    costs = attr["fees"] + attr["funding"] + attr["slippage"]
    flags = attr["flags"]

    def _money(v: Optional[float]) -> str:
        return "n/a" if v is None else f"{v:+.2f}"

    base_line = (
        f"expected {_money(exp)} → realized net {_money(net)} USDT "
        f"(slippage {_money(attr['slippage'])}, fees {_money(attr['fees'])}, "
        f"funding {_money(attr['funding'])}, residual {_money(res)})"
    )

    if exp is None:
        # Legacy linkage without decision prices — judge on sign only, softly.
        if _sign(gross) != 0 and _sign(gross) != _sign(net) and abs(net) > tol:
            pass  # fall through to cost note below
        if _sign(net) == 0 or _sign(gross) == _sign(net):
            return MATCHED, INFO, f"linked without decision-price baseline; {base_line}"
        return UNEXPLAINED_DELTA, WARNING, f"no decision-price baseline to attribute against; {base_line}"

    if res is not None and abs(res) <= tol:
        sign_flip = _sign(net) != _sign(exp) and _sign(exp) != 0
        costs_material = abs(costs) > max(tol, 0.25 * abs(exp))
        if sign_flip or costs_material:
            reason = "costs flipped the outcome" if sign_flip else "costs are a material share of the edge"
            status = EXPLAINED_COSTS
            sev = INFO
            expl = f"healthy: {reason} — {base_line}"
        else:
            status, sev, expl = MATCHED, INFO, f"execution matches intent — {base_line}"
        # Structural modifiers outrank a clean economic verdict
        if "partial-fill" in flags or "entry-exit-qty-gap" in flags:
            return PARTIAL_EXECUTION, WARNING, f"partial execution ({', '.join(flags)}); {base_line}"
        if "size-drift" in flags:
            return SIZE_DRIFT, WARNING, f"filled size drifted >25% from plan; {base_line}"
        if entry_method == "window" or exit_method == "window":
            gap = max(entry_gap_s, exit_gap_s)
            sev2 = WARNING if gap > _TIMING_WARN_S else INFO
            return TIMING_DRIFT, sev2, f"linked by time-window fallback (gap {gap:.0f}s); {base_line}"
        return status, sev, expl

    # Residual outside tolerance — unexplained territory
    if _sign(net) != _sign(exp) and _sign(exp) != 0:
        return SIGN_MISMATCH, CRITICAL, (
            f"realized sign contradicts model expectation and costs do not explain it — {base_line}"
        )
    if res is not None and abs(res) > _HARD_CAP_MULT * tol:
        return UNEXPLAINED_DELTA, CRITICAL, f"residual {res:+.2f} USDT far outside tolerance ±{tol:.2f}; {base_line}"
    return UNEXPLAINED_DELTA, WARNING, f"residual {res:+.2f} USDT outside tolerance ±{tol:.2f}; {base_line}"


# ── Top-level reconcile ───────────────────────────────────────────────────

def reconcile(
    model_trades: List[dict],
    events: List[dict],
    fills: List[dict],
    funding: List[dict],
    *,
    now_s: Optional[float] = None,
    eligibility: Optional[List[Tuple[float, bool]]] = None,
    manual_order_ids: Optional[set] = None,
    sync_age_s: Optional[float] = None,
    position_check: Optional[dict] = None,
) -> dict:
    """Pure recomputation: model ledger × exec events × broker history →
    reconciled trades, orphan broker activity, breaks, summary."""
    now_s = now_s or time.time()
    eligibility = eligibility or []
    manual_order_ids = manual_order_ids or set()

    fills_by_order: Dict[str, List[dict]] = {}
    for f in fills:
        fills_by_order.setdefault(str(f["order_id"]), []).append(f)

    rows: List[dict] = []
    referenced_orders: set = set()
    for ev in events:
        for leg in ev.get("legs", []) or []:
            if leg.get("order_id"):
                referenced_orders.add(str(leg["order_id"]))

    pending_window = max(2 * RECON_SYNC_SECONDS, 600)

    for trade in model_trades:
        try:
            entry_t = float(trade["entry_time"])
            exit_t = float(trade["exit_time"])
            direction = str(trade["direction"])
        except (KeyError, TypeError, ValueError):
            continue

        entry_ev, entry_m = find_event(events, entry_t, "entry", direction)
        exit_ev, exit_m = find_event(events, exit_t, "exit", direction)

        row = {
            "id": f"rt-{int(entry_t)}-{int(exit_t)}",
            "opened_at": int(entry_t),
            "closed_at": int(exit_t),
            "direction": direction,
            "model": {
                "entry_price": trade.get("entry_price"),
                "exit_price": trade.get("exit_price"),
                "pnl_pct": trade.get("pnl_pct"),
                "pnl_dollar": trade.get("pnl_dollar"),
                "entry_probability": trade.get("entry_probability"),
                "entry_strength": trade.get("entry_strength"),
                "reason": trade.get("reason"),
                "model_version": trade.get("model_version"),
                "position_size_pct": trade.get("position_size_pct"),
                "bars_held": trade.get("bars_held"),
            },
            "linkage": {
                "entry": {"method": entry_m, "outcome": (entry_ev or {}).get("outcome"),
                          "event_ts": (entry_ev or {}).get("timestamp")},
                "exit": {"method": exit_m, "outcome": (exit_ev or {}).get("outcome"),
                         "event_ts": (exit_ev or {}).get("timestamp")},
            },
        }

        if entry_ev is None and exit_ev is None:
            if eligible_at(eligibility, entry_t):
                row["status"] = MODEL_ONLY_BREAK
                row["explanation"] = (
                    "auto-execute was ON at entry time but no execution event exists — "
                    "the broker was never asked to trade this signal"
                )
            else:
                row["status"] = MODEL_ONLY
                row["explanation"] = (
                    "no execution attempt recorded (auto-execute off, not eligible, "
                    "or this instance is not the writer) — model-only trade, expected"
                )
            row["severity"] = _SEVERITY[row["status"]]
            row["attribution"] = None
            rows.append(row)
            continue

        entry_legs = _legs_for(entry_ev, "entry_legs")
        exit_legs = _legs_for(exit_ev, "exit_legs")
        all_leg_ids = [str(l.get("order_id") or "") for l in entry_legs + exit_legs]
        if any(oid.startswith("paper-") for oid in all_leg_ids if oid):
            row["status"] = UNVERIFIED_PAPER
            row["severity"] = INFO
            row["explanation"] = "paper mode — simulated fills carry no broker economics to verify"
            row["attribution"] = None
            rows.append(row)
            continue

        attr = attribute(trade, entry_ev, exit_ev, fills_by_order, funding)
        row["attribution"] = attr

        if attr["fill_count"] == 0:
            has_real_orders = any(oid for oid in all_leg_ids)
            if not has_real_orders:
                # Events exist but produced no orders (skipped / reconciled-drift)
                oc = (entry_ev or exit_ev or {}).get("outcome", "")
                row["status"] = MODEL_ONLY
                row["severity"] = INFO
                row["explanation"] = f"transition recorded without orders (outcome {oc or 'n/a'})"
            elif now_s - exit_t < pending_window or sync_age_s is None:
                row["status"] = PENDING
                row["severity"] = INFO
                row["explanation"] = "orders recorded; broker fill history not yet synced for this trade"
            else:
                row["status"] = FILLS_MISSING
                row["severity"] = CRITICAL
                row["explanation"] = (
                    "orders were recorded for this trade but broker history has no fills — "
                    "verify on the exchange; possible rejected/ghost orders or sync gap"
                )
            rows.append(row)
            continue

        def _gap(ev, t):
            ref = _event_signal_ts(ev) or _event_wall_ts(ev)
            return abs((ref or t) - t)

        status, severity, explanation = decide_status(
            attr,
            entry_m or "none", exit_m or "none",
            _gap(entry_ev, entry_t) if entry_ev else 0.0,
            _gap(exit_ev, exit_t) if exit_ev else 0.0,
        )
        # One-sided linkage (e.g. exit event missing) is partial execution
        if entry_ev is None or exit_ev is None:
            missing = "entry" if entry_ev is None else "exit"
            status, severity = PARTIAL_EXECUTION, WARNING
            explanation = f"no {missing} transition event linked; economics incomplete — {explanation}"
        row["status"] = status
        row["severity"] = severity
        row["explanation"] = explanation
        rows.append(row)

    # ── Orphan broker activity: fills no execution event accounts for ──
    orphans: List[dict] = []
    by_order: Dict[str, List[dict]] = {}
    for f in fills:
        if str(f["order_id"]) not in referenced_orders:
            by_order.setdefault(str(f["order_id"]), []).append(f)
    for oid, ofills in sorted(by_order.items(), key=lambda kv: kv[1][0]["time_ms"], reverse=True):
        vw, q = _vwap(ofills)
        manual = oid in manual_order_ids
        orphans.append({
            "order_id": oid,
            "symbol": ofills[0]["symbol"],
            "side": ofills[0]["side"],
            "qty": round(q, 8),
            "avg_price": round(vw, 4),
            "time_ms": ofills[0]["time_ms"],
            "realized_pnl": round(sum(f["realized_pnl"] for f in ofills), 6),
            "commission": round(sum(f["commission"] for f in ofills), 6),
            "status": BROKER_ONLY,
            "severity": WARNING if manual else CRITICAL,
            "origin": "manual-api" if manual else "unknown",
            "explanation": (
                "manual order placed through the API (POST /trade) — outside model intent by design"
                if manual else
                "broker fill with no matching model intent or manual record — investigate immediately"
            ),
        })

    rows.sort(key=lambda r: r["closed_at"], reverse=True)

    # ── Breaks & summary ──
    breaks = [r for r in rows if r["severity"] in (WARNING, CRITICAL)] + \
             [o for o in orphans]
    if position_check and position_check.get("status") == "DRIFT":
        breaks.insert(0, {
            "id": "position-drift",
            "status": "POSITION_DRIFT",
            "severity": CRITICAL,
            "explanation": (
                f"position parity broken: model={position_check.get('model_pos')} "
                f"broker-tracked={position_check.get('broker_pos')} "
                f"exchange={position_check.get('exchange_pos')}"
            ),
        })

    by_status: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        by_severity[r["severity"]] = by_severity.get(r["severity"], 0) + 1
    for o in orphans:
        by_status[BROKER_ONLY] = by_status.get(BROKER_ONLY, 0) + 1
        by_severity[o["severity"]] = by_severity.get(o["severity"], 0) + 1

    linked = [r for r in rows if r.get("attribution") and r["attribution"]["fill_count"] > 0]
    explained = [r for r in linked if r["status"] in (MATCHED, EXPLAINED_COSTS, TIMING_DRIFT, SIZE_DRIFT, PARTIAL_EXECUTION)]
    tot_notional = sum(r["attribution"]["entry_notional"] for r in linked)
    tot_slippage = sum(r["attribution"]["slippage"] for r in linked)
    tot_fees = sum(r["attribution"]["fees"] for r in linked)
    tot_funding = sum(r["attribution"]["funding"] for r in linked)

    summary = {
        "total_model_trades": len(rows),
        "linked_with_fills": len(linked),
        "explained": len(explained),
        "explained_rate": round(len(explained) / len(linked), 4) if linked else None,
        "by_status": by_status,
        "by_severity": by_severity,
        "critical_count": by_severity.get(CRITICAL, 0),
        "warning_count": by_severity.get(WARNING, 0),
        "sign_mismatch_count": by_status.get(SIGN_MISMATCH, 0),
        "unmatched_model": by_status.get(MODEL_ONLY_BREAK, 0) + by_status.get(FILLS_MISSING, 0),
        "unmatched_broker": len(orphans),
        "avg_slippage_bps": round(tot_slippage / tot_notional * 10_000, 2) if tot_notional else None,
        "total_slippage": round(tot_slippage, 4),
        "total_fees": round(tot_fees, 4),
        "total_funding": round(tot_funding, 4),
        "fee_drag_bps": round(tot_fees / tot_notional * 10_000, 2) if tot_notional else None,
        "expected_gross_total": round(sum(r["attribution"]["expected_gross"] or 0.0 for r in linked), 4),
        "realized_net_total": round(sum(r["attribution"]["realized_net"] for r in linked), 4),
        "position_check": position_check,
        "sync": {
            "age_s": round(sync_age_s, 1) if sync_age_s is not None else None,
            "interval_s": RECON_SYNC_SECONDS,
            "stale": (sync_age_s is not None and sync_age_s > 3 * RECON_SYNC_SECONDS),
        },
    }
    return {"trades": rows, "orphans": orphans, "breaks": breaks, "summary": summary}


# ── Manual-order recognition (activity log read) ─────────────────────────

def manual_order_ids_from_activity(jsonl_path: Path) -> set:
    """Order ids placed through the API *without* an auto-exec client_id —
    i.e. operator-manual /trade calls. Used to downgrade BROKER_ONLY severity."""
    ids: set = set()
    if not jsonl_path.exists():
        return ids
    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("action") != "place_order":
                    continue
                req = row.get("request") or {}
                cid = str(req.get("client_id") or "")
                if cid.startswith("v415-"):
                    continue
                resp = row.get("response") or {}
                oid = resp.get("broker_order_id") or (resp.get("raw") or {}).get("orderId") if isinstance(resp, dict) else None
                if oid:
                    ids.add(str(oid))
    except OSError as e:
        logger.warning(f"activity read failed: {e}")
    return ids
