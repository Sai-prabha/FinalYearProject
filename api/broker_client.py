"""
Broker abstraction for the v4.15 model server.

Exposes a small interface (`BrokerClient`) that the rest of the server uses to
talk to a broker, with two concrete implementations:

  * `BinanceFuturesBroker` — signed REST against Binance USDM Futures Testnet.
  * `PaperBroker`          — fully simulated; used when credentials are
                             missing or the live broker fails to initialize.

All Binance-specific concerns (HMAC signing, recvWindow, time sync, error
envelope parsing) live inside `BinanceFuturesBroker`. Callers only see the
`OrderRequest` / `OrderResponse` Pydantic models.

Methods are synchronous (`requests`); async callers should wrap them with
`await asyncio.to_thread(...)` so the FastAPI event loop is never blocked.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import urlencode

import requests
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# JSONL trade log lives next to the existing trade history files.
LIVE_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "live"
JSONL_LOG_PATH = LIVE_DATA_DIR / "binance_trades_demo.jsonl"

# Fields that must never reach disk in cleartext.
_REDACTED_KEYS = {"apiKey", "apikey", "X-MBX-APIKEY", "secret", "signature"}

# ── Hard safety rails (demo/testnet-only project) ─────────────────────────
# Only these hosts may ever receive a signed order. Production hosts
# (fapi.binance.com etc.) are structurally unreachable from this codebase.
ALLOWED_BASE_HOSTS = {"testnet.binancefuture.com"}

# Per-order quantity caps and symbol allowlist. Env-overridable but with
# tiny defaults; the gate rejects anything outside them before signing.
def _safety_limits() -> dict:
    allowlist = os.environ.get("BROKER_SYMBOL_ALLOWLIST", "BTCUSDT,ETHUSDT")
    return {
        "symbols": {s.strip().upper() for s in allowlist.split(",") if s.strip()},
        "max_qty": {
            "BTCUSDT": float(os.environ.get("BROKER_MAX_BTC_QTY", "0.01")),
            "ETHUSDT": float(os.environ.get("BROKER_MAX_ETH_QTY", "0.5")),
        },
        "default_max_qty": float(os.environ.get("BROKER_MAX_QTY", "0.01")),
    }


# Flip-at-runtime kill switch: create this file to block all order placement
# (touch data/live/KILL_SWITCH). Checked on every order, no restart needed.
KILL_SWITCH_PATH = LIVE_DATA_DIR / "KILL_SWITCH"


def _dry_run_enabled() -> bool:
    return os.environ.get("BROKER_DRY_RUN", "").strip().lower() in ("1", "true", "yes")


def check_order_safety(req: "OrderRequest") -> Optional[str]:
    """Return a rejection reason if the order violates any safety rail, else None."""
    if KILL_SWITCH_PATH.exists():
        return f"KILL SWITCH active ({KILL_SWITCH_PATH}) — delete the file to re-enable trading"
    limits = _safety_limits()
    sym = req.symbol.upper()
    if sym not in limits["symbols"]:
        return f"symbol {sym} not in allowlist {sorted(limits['symbols'])}"
    max_qty = limits["max_qty"].get(sym, limits["default_max_qty"])
    if req.quantity > max_qty:
        return f"quantity {req.quantity} exceeds max {max_qty} for {sym}"
    return None


class BrokerConfigError(RuntimeError):
    """Raised when a real broker cannot be constructed from the environment."""


# ── Pydantic models ────────────────────────────────────────────────────────


class OrderRequest(BaseModel):
    symbol: str
    side: Literal["BUY", "SELL"]
    order_type: Literal["MARKET", "LIMIT"] = "MARKET"
    quantity: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    client_id: Optional[str] = None
    reduce_only: bool = False


class OrderResponse(BaseModel):
    broker_order_id: str
    status: str
    filled_qty: float = 0.0
    avg_price: float = 0.0
    message: Optional[str] = None
    raw: Optional[dict] = Field(default=None, exclude=False)


# ── Abstract base ─────────────────────────────────────────────────────────


class BrokerClient(ABC):
    """Generic broker interface. Implementations must set ``mode``."""

    mode: str  # "demo" | "paper"
    # Why the real broker was NOT constructed (e.g. expired testnet keys).
    # None when this broker is running as intended; surfaced via /broker/*
    # endpoints so operators can see degraded state instead of a silent
    # paper fallback.
    init_error: Optional[str] = None

    @abstractmethod
    def place_order(self, req: OrderRequest) -> OrderResponse: ...

    @abstractmethod
    def place_test_order(self, req: OrderRequest) -> OrderResponse: ...

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool: ...

    @abstractmethod
    def get_open_positions(self) -> list[dict]: ...

    @abstractmethod
    def get_balance(self) -> dict: ...

    def try_get_open_positions(self) -> Optional[list[dict]]:
        """Like ``get_open_positions`` but None when the exchange truth is
        unavailable (query error, or a broker with no exchange behind it).

        Callers use None to mean "fall back to internally tracked state" —
        an empty list is a *positive* statement that the account is flat.
        """
        return None

    def get_symbol_filters(self, symbol: str) -> Optional[dict]:
        """Exchange trading filters for one symbol, or None when unknown.

        Shape: ``{"step_size": float, "min_qty": float, "min_notional": float}``.
        """
        return None

    @abstractmethod
    def get_order_status(self, symbol: str, order_id: str) -> Optional[dict]:
        """Return fill info for a submitted order, or None if unavailable.

        On success returns ``{"status": str, "filled_qty": float, "avg_price": float}``.
        Returning None signals the broker cannot provide this (e.g. paper mode).
        Never raises.
        """
        ...

    # ── Logging shared by all implementations ──

    def _log_interaction(
        self,
        action: str,
        request: Any,
        response: Any,
        error: Optional[str] = None,
    ) -> None:
        """Append one JSONL row describing a broker interaction."""
        try:
            LIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "mode": self.mode,
                "action": action,
                "request": _redact(request),
                "response": _redact(response),
                "error": error,
            }
            with open(JSONL_LOG_PATH, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:  # pragma: no cover - logging must never crash the caller
            logger.warning(f"Broker JSONL log write failed: {e}")


def _redact(payload: Any) -> Any:
    """Recursively strip secrets from a payload before persisting."""
    if isinstance(payload, dict):
        return {
            k: ("<redacted>" if k in _REDACTED_KEYS else _redact(v))
            for k, v in payload.items()
        }
    if isinstance(payload, list):
        return [_redact(v) for v in payload]
    return payload


# ── Binance Futures Testnet implementation ───────────────────────────────


class BinanceFuturesBroker(BrokerClient):
    """USDM Futures Testnet client using HMAC-SHA256 signed REST.

    Reads credentials from the environment at construction time:
      * BINANCE_API_KEY
      * BINANCE_API_SECRET
      * BINANCE_BASE_URL  (e.g. ``https://testnet.binancefuture.com``)
    """

    mode = "demo"

    def __init__(self) -> None:
        api_key = os.environ.get("BINANCE_API_KEY", "").strip()
        api_secret = os.environ.get("BINANCE_API_SECRET", "").strip()
        base_url = os.environ.get("BINANCE_BASE_URL", "").strip().rstrip("/")
        if not api_key or not api_secret or not base_url:
            raise BrokerConfigError(
                "BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_BASE_URL all required"
            )

        # Hard testnet pin: refuse to construct against any non-testnet host,
        # regardless of what BINANCE_ENV claims. This makes production
        # endpoints structurally unreachable even with valid live keys.
        host = base_url.split("//", 1)[-1].split("/", 1)[0].lower()
        if host not in ALLOWED_BASE_HOSTS:
            raise BrokerConfigError(
                f"BINANCE_BASE_URL host {host!r} is not an approved testnet host "
                f"{sorted(ALLOWED_BASE_HOSTS)} — refusing to initialize"
            )

        self._api_key = api_key
        self._api_secret = api_secret.encode("utf-8")
        self._base_url = base_url
        self._session = requests.Session()
        self._session.headers["X-MBX-APIKEY"] = self._api_key
        self._time_offset_ms = 0
        self._sync_time()

    # -- Internal helpers --

    def _sync_time(self) -> None:
        """Compute server-local clock skew so signed requests stay inside recvWindow."""
        try:
            url = f"{self._base_url}/fapi/v1/time"
            resp = self._session.get(url, timeout=10)
            resp.raise_for_status()
            server_ms = int(resp.json()["serverTime"])
            local_ms = int(time.time() * 1000)
            self._time_offset_ms = server_ms - local_ms
            logger.info(f"Binance time offset: {self._time_offset_ms}ms")
        except Exception as e:
            logger.warning(f"Binance time sync failed (using offset 0): {e}")
            self._time_offset_ms = 0

    def _sign(self, params: dict) -> str:
        query = urlencode(params, doseq=True)
        return hmac.new(self._api_secret, query.encode("utf-8"), hashlib.sha256).hexdigest()

    def _signed_request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        retried: bool = False,
    ) -> tuple[int, dict]:
        """Execute a signed REST call.

        Returns ``(http_status, response_json)``. Never raises on Binance error
        envelopes — the caller inspects status and payload.
        """
        params = dict(params or {})
        params["timestamp"] = int(time.time() * 1000) + self._time_offset_ms
        params["recvWindow"] = 5000
        params["signature"] = self._sign(params)
        url = f"{self._base_url}{path}"

        try:
            resp = self._session.request(method, url, params=params, timeout=15)
        except requests.RequestException as e:
            return 0, {"code": -1, "msg": f"http error: {e}"}

        try:
            body = resp.json()
        except Exception:
            body = {"code": -1, "msg": resp.text[:500]}

        # Lazy time-sync recovery on -1021 ("timestamp outside recvWindow")
        if (
            isinstance(body, dict)
            and body.get("code") == -1021
            and not retried
        ):
            logger.warning("Binance -1021 timestamp drift — resyncing and retrying once")
            self._sync_time()
            params.pop("signature", None)
            params.pop("timestamp", None)
            params.pop("recvWindow", None)
            return self._signed_request(method, path, params, retried=True)

        return resp.status_code, body

    @staticmethod
    def _build_order_params(req: OrderRequest) -> dict:
        params = {
            "symbol": req.symbol.upper(),
            "side": req.side,
            "type": req.order_type,
            "quantity": req.quantity,
        }
        if req.order_type == "LIMIT":
            if req.price is None:
                raise ValueError("LIMIT order requires price")
            params["price"] = req.price
            params["timeInForce"] = "GTC"
        if req.reduce_only:
            params["reduceOnly"] = "true"
        if req.client_id:
            params["newClientOrderId"] = req.client_id
        return params

    def _place_bracket_orders(self, req: OrderRequest) -> None:
        """Place STOP_MARKET and/or TAKE_PROFIT_MARKET bracket orders after entry.

        Uses ``closePosition=true`` so Binance closes the full position when
        triggered — no quantity alignment needed.  Both orders are fire-and-
        forget; failures are logged but do NOT affect the already-placed entry.
        """
        # Bracket orders run on the closing side
        close_side = "SELL" if req.side == "BUY" else "BUY"

        def _bracket(order_type: str, stop_price: float) -> None:
            params = {
                "symbol": req.symbol.upper(),
                "side": close_side,
                "type": order_type,
                "stopPrice": round(stop_price, 8),
                "closePosition": "true",
            }
            _, body = self._signed_request("POST", "/fapi/v1/order", params)
            err = None
            if isinstance(body, dict) and isinstance(body.get("code"), int) and body["code"] < 0:
                err = str(body.get("msg", "unknown"))
                logger.warning(f"Bracket {order_type} for {req.symbol} rejected: {err}")
            self._log_interaction(
                f"bracket_{order_type.lower()}",
                {"symbol": req.symbol, "side": close_side, "stop_price": stop_price},
                body,
                error=err,
            )

        if req.stop_loss:
            try:
                _bracket("STOP_MARKET", req.stop_loss)
            except Exception as e:
                logger.warning(f"Bracket STOP_MARKET failed ({req.symbol}): {e}")

        if req.take_profit:
            try:
                _bracket("TAKE_PROFIT_MARKET", req.take_profit)
            except Exception as e:
                logger.warning(f"Bracket TAKE_PROFIT_MARKET failed ({req.symbol}): {e}")

    @staticmethod
    def _parse_order_response(body: dict) -> OrderResponse:
        if isinstance(body, dict) and body.get("code") and int(body.get("code", 0)) < 0:
            return OrderResponse(
                broker_order_id="",
                status="REJECTED",
                message=str(body.get("msg", "unknown error")),
                raw=body,
            )
        order_id = str(body.get("orderId", ""))
        status = str(body.get("status", "NEW"))
        filled = float(body.get("executedQty", 0.0) or 0.0)
        avg_price = float(body.get("avgPrice", 0.0) or 0.0)
        return OrderResponse(
            broker_order_id=order_id,
            status=status,
            filled_qty=filled,
            avg_price=avg_price,
            raw=body,
        )

    # -- Public API --

    def place_order(self, req: OrderRequest) -> OrderResponse:
        reason = check_order_safety(req)
        if reason is not None:
            r = OrderResponse(broker_order_id="", status="REJECTED", message=f"SAFETY: {reason}")
            logger.warning(f"Order blocked by safety gate: {reason}")
            self._log_interaction("place_order", req.model_dump(), r.model_dump(), error=reason)
            return r

        if _dry_run_enabled():
            # Validate signing/params via the test endpoint; never places.
            resp = self.place_test_order(req)
            resp.message = f"DRY RUN — no order placed ({resp.message})"
            return resp

        try:
            params = self._build_order_params(req)
        except ValueError as e:
            r = OrderResponse(broker_order_id="", status="REJECTED", message=str(e))
            self._log_interaction("place_order", req.model_dump(), r.model_dump(), error=str(e))
            return r

        _, body = self._signed_request("POST", "/fapi/v1/order", params)
        result = self._parse_order_response(body)
        self._log_interaction("place_order", req.model_dump(), result.model_dump())

        # Place TP/SL bracket orders when requested on a non-reducing entry
        if result.status not in ("REJECTED",) and not req.reduce_only:
            if req.stop_loss or req.take_profit:
                self._place_bracket_orders(req)

        return result

    def place_test_order(self, req: OrderRequest) -> OrderResponse:
        """Validate signing & params via Binance's test endpoint without placing."""
        try:
            params = self._build_order_params(req)
        except ValueError as e:
            r = OrderResponse(broker_order_id="", status="REJECTED", message=str(e))
            self._log_interaction("place_test_order", req.model_dump(), r.model_dump(), error=str(e))
            return r

        _, body = self._signed_request("POST", "/fapi/v1/order/test", params)
        if isinstance(body, dict) and body.get("code") and int(body.get("code", 0)) < 0:
            result = OrderResponse(
                broker_order_id="",
                status="REJECTED",
                message=str(body.get("msg", "unknown error")),
                raw=body,
            )
        else:
            # Test endpoint returns {} on success.
            result = OrderResponse(
                broker_order_id="",
                status="TEST_OK",
                message="test order accepted",
                raw=body if isinstance(body, dict) else None,
            )
        self._log_interaction("place_test_order", req.model_dump(), result.model_dump())
        return result

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        if not symbol:
            symbol = os.environ.get("BINANCE_DEFAULT_SYMBOL", "BTCUSDT")
        params = {"symbol": symbol.upper(), "orderId": order_id}
        _, body = self._signed_request("DELETE", "/fapi/v1/order", params)
        ok = isinstance(body, dict) and body.get("status") in ("CANCELED", "CANCELLED")
        self._log_interaction(
            "cancel_order",
            {"order_id": order_id, "symbol": symbol},
            body,
            error=None if ok else str(body.get("msg", "")) if isinstance(body, dict) else "unknown",
        )
        return ok

    def get_open_positions(self) -> list[dict]:
        _, body = self._signed_request("GET", "/fapi/v2/positionRisk", {})
        if not isinstance(body, list):
            self._log_interaction("get_open_positions", {}, body, error="non-list response")
            return []
        result: list[dict] = []
        for p in body:
            try:
                size = float(p.get("positionAmt", 0.0))
            except (TypeError, ValueError):
                continue
            if size == 0.0:
                continue
            result.append(
                {
                    "symbol": p.get("symbol", ""),
                    "side": "LONG" if size > 0 else "SHORT",
                    "size": abs(size),
                    "entry_price": float(p.get("entryPrice", 0.0) or 0.0),
                    "mark_price": float(p.get("markPrice", 0.0) or 0.0),
                    "unrealized_pnl": float(p.get("unRealizedProfit", 0.0) or 0.0),
                    "leverage": float(p.get("leverage", 0.0) or 0.0),
                }
            )
        self._log_interaction("get_open_positions", {}, result)
        return result

    def try_get_open_positions(self) -> Optional[list[dict]]:
        """Positions with error visibility: [] means confirmed flat, None means
        the query failed and the caller must not assume anything."""
        _, body = self._signed_request("GET", "/fapi/v2/positionRisk", {})
        if not isinstance(body, list):
            self._log_interaction("try_get_open_positions", {}, body, error="non-list response")
            return None
        result: list[dict] = []
        for p in body:
            try:
                size = float(p.get("positionAmt", 0.0))
            except (TypeError, ValueError):
                continue
            if size == 0.0:
                continue
            result.append(
                {
                    "symbol": p.get("symbol", ""),
                    "side": "LONG" if size > 0 else "SHORT",
                    "size": abs(size),
                    "signed_size": size,
                    "entry_price": float(p.get("entryPrice", 0.0) or 0.0),
                    "mark_price": float(p.get("markPrice", 0.0) or 0.0),
                }
            )
        return result

    _filters_cache: Optional[dict] = None

    def get_symbol_filters(self, symbol: str) -> Optional[dict]:
        """Market-order filters from /fapi/v1/exchangeInfo, cached for the
        process lifetime (filters change ~never). None on fetch failure."""
        if self._filters_cache is None:
            try:
                resp = self._session.get(f"{self._base_url}/fapi/v1/exchangeInfo", timeout=15)
                resp.raise_for_status()
                cache: dict = {}
                for s in resp.json().get("symbols", []):
                    fs = {f.get("filterType"): f for f in s.get("filters", [])}
                    lot = fs.get("MARKET_LOT_SIZE") or fs.get("LOT_SIZE") or {}
                    notional = fs.get("MIN_NOTIONAL") or {}
                    cache[s.get("symbol", "")] = {
                        "step_size": float(lot.get("stepSize", 0.0) or 0.0),
                        "min_qty": float(lot.get("minQty", 0.0) or 0.0),
                        "min_notional": float(notional.get("notional", 0.0) or 0.0),
                    }
                self._filters_cache = cache
                logger.info(f"Exchange filters cached for {len(cache)} symbols")
            except Exception as e:
                logger.warning(f"exchangeInfo fetch failed (filters unavailable): {e}")
                return None
        return self._filters_cache.get(symbol.upper())

    def get_balance(self) -> dict:
        _, body = self._signed_request("GET", "/fapi/v2/balance", {})
        if not isinstance(body, list):
            self._log_interaction("get_balance", {}, body, error="non-list response")
            return {"assets": [], "raw": body}
        assets = []
        for entry in body:
            try:
                bal = float(entry.get("balance", 0.0) or 0.0)
                avail = float(entry.get("availableBalance", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if bal == 0.0 and avail == 0.0:
                continue
            assets.append(
                {
                    "asset": entry.get("asset", ""),
                    "balance": bal,
                    "available": avail,
                }
            )
        # USDT first if present
        assets.sort(key=lambda a: (a["asset"] != "USDT", a["asset"]))
        result = {"assets": assets}
        self._log_interaction("get_balance", {}, result)
        return result

    def get_order_status(self, symbol: str, order_id: str) -> Optional[dict]:
        """Poll GET /fapi/v1/order for confirmed fill status.

        Returns ``{"status", "filled_qty", "avg_price"}`` on success, None on error.
        """
        try:
            params = {"symbol": symbol.upper(), "orderId": order_id}
            _, body = self._signed_request("GET", "/fapi/v1/order", params)
            if not isinstance(body, dict) or (
                isinstance(body.get("code"), int) and body["code"] < 0
            ):
                logger.warning(f"get_order_status {symbol} #{order_id}: {body}")
                return None
            status = str(body.get("status", ""))
            filled_qty = float(body.get("executedQty", 0.0) or 0.0)
            avg_price = float(body.get("avgPrice", 0.0) or 0.0)
            result = {"status": status, "filled_qty": filled_qty, "avg_price": avg_price}
            self._log_interaction(
                "get_order_status",
                {"symbol": symbol, "order_id": order_id},
                result,
            )
            return result
        except Exception as e:
            logger.warning(f"get_order_status {symbol} #{order_id} raised: {e}")
            return None


# ── Paper fallback ────────────────────────────────────────────────────────


class PaperBroker(BrokerClient):
    """Synthetic broker used when no credentials are available."""

    mode = "paper"

    def place_order(self, req: OrderRequest) -> OrderResponse:
        reason = check_order_safety(req)
        if reason is not None:
            r = OrderResponse(broker_order_id="", status="REJECTED", message=f"SAFETY: {reason}")
            self._log_interaction("place_order", req.model_dump(), r.model_dump(), error=reason)
            return r
        oid = f"paper-{uuid.uuid4().hex[:12]}"
        result = OrderResponse(
            broker_order_id=oid,
            status="FILLED",
            filled_qty=req.quantity,
            avg_price=req.price or 0.0,
            message="simulated",
        )
        self._log_interaction("place_order", req.model_dump(), result.model_dump())
        return result

    def place_test_order(self, req: OrderRequest) -> OrderResponse:
        result = OrderResponse(
            broker_order_id="",
            status="TEST_OK",
            message="paper",
        )
        self._log_interaction("place_test_order", req.model_dump(), result.model_dump())
        return result

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        self._log_interaction("cancel_order", {"order_id": order_id, "symbol": symbol}, {"ok": True})
        return True

    def get_open_positions(self) -> list[dict]:
        return []

    def get_balance(self) -> dict:
        return {"assets": []}

    def get_order_status(self, symbol: str, order_id: str) -> Optional[dict]:
        # Paper orders are already FILLED on placement — nothing to poll.
        return None


# ── Factory ───────────────────────────────────────────────────────────────


def make_broker_client() -> BrokerClient:
    """Pick the right broker based on environment.

    Returns a `BinanceFuturesBroker` when ``BINANCE_ENV == "demo"`` and the
    required credentials are present *and* a smoke-test ``get_balance`` call
    succeeds. Otherwise returns a `PaperBroker`. The fallback path never
    raises — the server must keep running even if the broker is unhealthy.
    """
    env = os.environ.get("BINANCE_ENV", "").strip().lower()
    if env != "demo":
        logger.info("Broker: BINANCE_ENV != 'demo'; using PaperBroker")
        return PaperBroker()

    try:
        broker = BinanceFuturesBroker()
        # Smoke test must actually prove auth works: get_balance() swallows
        # Binance error envelopes (e.g. -2015 invalid/expired key), so check
        # for one explicitly instead of treating any response as success.
        bal = broker.get_balance()
        raw = bal.get("raw")
        if isinstance(raw, dict) and isinstance(raw.get("code"), int) and raw["code"] < 0:
            raise BrokerConfigError(
                f"Testnet auth failed ({raw.get('code')}: {raw.get('msg')}) — "
                "keys likely expired; regenerate at https://testnet.binancefuture.com"
            )
        logger.info("=" * 60)
        logger.info("🟢 BROKER: BINANCE FUTURES **TESTNET / DEMO ONLY** 🟢")
        logger.info(f"   host pinned to {sorted(ALLOWED_BASE_HOSTS)}")
        logger.info(f"   dry_run={_dry_run_enabled()} kill_switch={KILL_SWITCH_PATH.exists()}")
        logger.info("=" * 60)
        return broker
    except BrokerConfigError as e:
        reason = str(e)
        logger.warning(f"Broker: config incomplete ({reason}); using PaperBroker")
    except Exception as e:
        reason = f"init failed: {e}"
        logger.warning(f"Broker: live init failed ({e}); using PaperBroker")

    fallback = PaperBroker()
    fallback.init_error = reason
    return fallback
