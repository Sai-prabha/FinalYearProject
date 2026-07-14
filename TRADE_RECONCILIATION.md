# TRADE_RECONCILIATION.md — Broker-Aware Trade History & Reconciliation

> Design record, written 2026-07-14 before implementation. Companion docs:
> `EXECUTION_CONTROL.md` (writer/control plane), `RESEARCH_TOOLS.md` (ops
> surfaces), `STRATEGY_LAB.md`. UI ships in Meridian Operator
> (`tradingview-claude-fable5-hud/app`).

## 1. Problem

Meridian shows the model's internal trade ledger (`/trades`,
`trade_history.json`) and, separately, broker execution events
(`/broker/executions`). Nothing connects them. When the model books a +0.38%
ratio win, the operator cannot tell whether the demo account actually made
money on that trade, lost money, or never traded at all — and when the numbers
differ, cannot tell a healthy difference (fees, slippage) from a real defect
(missed exit, rogue order, sign flip).

## 2. Current state (verified in-repo, 2026-07-14)

**What exists**

| Layer | Storage | Served by | Content |
|---|---|---|---|
| Model ledger | `data/live/trade_history.json` (+CSV) | `GET /trades` | Closed simulated ratio trades: direction, ratio entry/exit, `pnl_pct`, `pnl_dollar` (simulated book), epoch-s times, entry probability/strength, exit reason, model_version |
| Execution events | `data/live/exec_events.jsonl` | `GET /broker/executions` | One event per position transition: prev/new/final pos, outcome (`OK`, `RECONCILED`, `SKIPPED`, partial-failure family), exit/entry/unwind/skipped legs with `order_id`, `filled_qty`, `avg_price`, sizing detail |
| Broker interaction log | `data/live/binance_trades_demo.jsonl` | `GET /broker/activity` | Raw request/response per broker call, secrets redacted |
| Live positions/balance | — | `GET /broker/positions`, `/broker/balance` | Exchange truth, on demand |

**Correlation that already exists:** every auto-exec leg is submitted with
`newClientOrderId = v415-<signal_candle_epoch_s>-<label>-<SYM>` where label ∈
{L, S, X, XL, XS, unwind-…}. The signal candle timestamp equals the model
trade's `entry_time`/`exit_time`. The planner already detects position drift
(`skip_kind: "drift"`, outcome `RECONCILED`).

**What is missing**

1. The `client_id` is computed but **not persisted** on leg results; exec
   events store only wall-clock submit time, not the signal timestamp.
2. No ingestion of broker fills (`/fapi/v1/userTrades`: per-fill
   `realizedPnl`, `commission`) or funding (`/fapi/v1/income`,
   `FUNDING_FEE`) — so no broker-realized PnL, fees, or funding exists
   anywhere in the system.
3. No linkage from a closed model trade to its entry+exit transitions and
   their fills; no reconciliation status, no attribution, no endpoints, no UI.
4. Exit legs carry no decision price (entry legs carry it inside `sizing`),
   so exit slippage cannot be measured today.

**Likely failure modes today:** silent missed executions (model books a trade,
broker never moved — only visible by manually diffing two panels); cost drag
invisible (demo account bleeds fees while the model ledger shows wins); rogue
or manual orders invisible in any trade-shaped view; position drift only
surfaces at the *next* transition.

## 3. Research → chosen invariants

Researched (2026-07-14): industry cash/position/transaction reconciliation
practice, Perold implementation-shortfall decomposition and TCA, Binance
USDM API semantics. Sources in the final report; the conclusions that drive
this design:

1. **Exact PnL equality is the wrong invariant — rejected.** The model books
   PnL on the BTC/ETH *ratio* at candle closes with a simulated percent-of-book
   size and zero costs. The broker realizes PnL on *two USDT-margined legs*
   with taker fees, slippage, funding, and step-size-floored quantities on a
   different notional. The two numbers are structurally never equal; a system
   that flags every inequality trains the operator to ignore it.
2. **Broker is source of truth for realized execution. Model ledger is source
   of truth for intent/expectation.** Neither is "wrong" when they differ —
   the difference itself is the object of interest and must be *attributed*.
3. **Position-first reconciliation is the primary invariant** (industry
   practice: reconcile positions before transactions — one position break
   explains many transaction-level questions). This strategy holds exactly one
   position (FLAT/LONG/SHORT ratio pair), so position parity is cheap,
   continuous, and catches everything catastrophic within one candle:
   `model position == broker tracked position == exchange net position`.
4. **Trade-level reconciliation is an attribution/explain layer** (Perold /
   TCA shape): realized − expected decomposed into named buckets; the trade is
   *explained* when the residual is within tolerance, and *broken* only when
   linkage fails or the residual/sign cannot be explained.
5. **Sizes must be normalized before comparison.** Model `pnl_dollar` lives on
   a simulated book; broker PnL on actual notional. All comparisons are made
   at broker size: "expected" = what the actually-filled quantities would have
   earned at the model's decision prices. Model-book dollars are shown, never
   compared.

## 4. Canonical entities

| Entity | Source | Notes |
|---|---|---|
| `ExpectedTrade` | model ledger row (closed) or open position | Intent + expectation: direction, ratio entry/exit, decision timestamps, probabilities |
| `ExecutionEvent` | `exec_events.jsonl` | The bridge: one per transition; legs carry `order_id`, `client_id`, `decision_price` (new), qty, fills |
| `BrokerFill` | `userTrades` cache (`broker_fills.jsonl`) | Per-fill price, qty, `realizedPnl` (gross), `commission`, orderId, time |
| `FundingEvent` | `income` cache (same file, `kind:"funding"`) | FUNDING_FEE rows per symbol |
| `ReconciledTrade` | computed | ExpectedTrade ⟂ entry event ⟂ exit event ⟂ fills ⟂ funding + attribution + status + severity |
| `ReconciliationBreak` | computed | Any ReconciledTrade or orphan at warn/critical, plus position-parity breaks |
| `PositionCheck` | computed | model vs broker-tracked vs exchange position, now |

Storage is append-only JSONL + a small sync-cursor JSON; reconciliation is a
**pure recomputation** over those files (deterministic, restart-safe,
late-data-safe by construction — a late fill lands in the cache and the next
computation picks it up). No database; volumes are tens of trades.

## 5. Correlation strategy (ordered, deterministic)

1. **Client order ID (primary).** Legs persist `client_id`
   (`v415-<signal_ts>-<label>-<SYM>`); events persist `signal_ts`. A model
   trade links to its entry event by `signal_ts == entry_time` and exit event
   by `signal_ts == exit_time`.
2. **Order ID (fills).** Fills attach to legs by exchange `orderId` — exact,
   survives everything.
3. **Fallback window matcher** (for pre-upgrade events without `signal_ts`):
   nearest event within ±180 s of the model timestamp whose transition
   direction matches the trade (entry: FLAT→dir or reversal into dir; exit:
   dir→FLAT or reversal out). Ties → closest; matches this way are stamped
   `TIMING_DRIFT`, never silently equated.
4. **Orphans.** Fills whose orderId no leg references → `BROKER_ONLY` pool
   (manual `/trade` orders are recognizable via the activity log and
   downgraded to warning "manual"); model trades with no event →
   `MODEL_ONLY`, healthy or break depending on execution eligibility at the
   time (paper mode / auto-exec off / guard-blocked / non-writer ⇒ healthy).

## 6. PnL attribution (all in USDT at broker size)

For a linked round-trip, per leg ℓ with filled qty qℓ, entry/exit fill
averages fℓᵉⁿ/fℓᵉˣ, decision prices dℓᵉⁿ/dℓᵉˣ (candle closes at signal time),
and leg sign sℓ (+1 long leg, −1 short leg):

```
expected_gross  = Σℓ sℓ·qℓ·(dℓᵉˣ − dℓᵉⁿ)      # model's move at actual size
realized_gross  = Σ fills.realizedPnl          # broker truth, gross
slippage        = Σℓ [sℓ·qℓ·(dℓᵉⁿ − fℓᵉⁿ) + sℓ·qℓ·(fℓᵉˣ − dℓᵉˣ)]  # signed cost
fees            = − Σ fills.commission
funding         = Σ FUNDING_FEE while position open (signed)
residual        = realized_gross − expected_gross − slippage
realized_net    = realized_gross + fees + funding
```

`residual` captures what the named buckets don't (fill-timestamp price drift
inside the candle, partial-fill residue, data gaps). Tolerance:
`|residual| ≤ max($0.02, 2 bps of entry notional)` ⇒ explained.

The engine answers the five hard cases explicitly:
same-sign/different-magnitude (buckets sum the gap); model win / broker loss
(costs ≥ edge ⇒ `EXPLAINED_COSTS`, healthy but aggregated into fee-drag
alerting); model loss / broker win (favorable slippage, same treatment);
missing broker closure (open exchange position with no matching intent ⇒
position-parity break); broker trade with no model intent ⇒ `BROKER_ONLY`.

Model-book numbers (`pnl_dollar`, `pnl_pct`) are displayed alongside as
*intent*, plus both returns in bps for shape comparison — never differenced
against broker dollars.

## 7. Status taxonomy & severity

| Status | Meaning | Severity |
|---|---|---|
| `MATCHED` | Linked, residual in tolerance, costs small, same sign | info |
| `EXPLAINED_COSTS` | Linked & explained; fees/slippage/funding flipped or dominated the outcome | info (aggregate alerting) |
| `SIZE_DRIFT` | Linked; filled qty deviates >25% from planned | warning |
| `PARTIAL_EXECUTION` | Leg skipped/not-tradable/partial fill; attribution incomplete | warning |
| `TIMING_DRIFT` | Linked only via fallback window matcher | info (>60 s ⇒ warning) |
| `MODEL_ONLY` | No broker activity; execution wasn't eligible (paper/off/guard/non-writer) | info, cause shown |
| `MODEL_ONLY_BREAK` | No broker activity but execution *was* eligible | **critical** |
| `FILLS_MISSING` | Orders were recorded but broker history shows no fills after sync grace | **critical** |
| `BROKER_ONLY` | Broker fills with no model intent (manual → warning) | **critical** / warning |
| `UNEXPLAINED_DELTA` | Linked but residual outside tolerance | warning (> 5× tol ⇒ critical) |
| `SIGN_MISMATCH` | Realized sign ≠ expected sign, not explained by costs | **critical** |
| `PENDING` | Closed less than one sync interval ago; broker data may lag | info |
| `UNVERIFIED_PAPER` | Paper mode — no broker economics exist to verify | info |

Position parity is reported separately in the summary
(`position_check: OK | DRIFT`) and a drift is always a **critical** break.

**Operator rule of thumb the UI encodes:** *info = execution reality, read it
for cost awareness; warning = look when convenient; critical = something is
wrong with the system, look now.*

## 8. Data flow & endpoints

```
Binance testnet ──userTrades/income (read-only, cursored, 7d windows)──▶
  broker_fills.jsonl + recon_sync.json          (background task, ~5 min +
                                                 on-demand POST /reconciliation/sync)
trade_history.json ─┐
exec_events.jsonl ──┼──▶ ReconciliationEngine (pure recompute) ──▶
broker_fills.jsonl ─┘        GET /reconciliation/summary   (rates, drags, parity, sync age)
                             GET /reconciliation/trades    (reconciled rows, filterable)
                             GET /reconciliation/breaks    (warn+critical only)
                             GET /broker/history           (normalized fills, newest first)
```

All endpoints are auth-gated like the rest of the operator API and strictly
read-only against the broker (`userTrades`/`income` are GETs; the sync writes
only the local cache). **No new order-placing path exists or is created.**
PaperBroker returns "unsupported" and the engine degrades to structural
reconciliation with `UNVERIFIED_PAPER` economics — honestly labeled, never
fabricated.

## 9. Meridian UI (Operator module)

- Trade History panel becomes view-tabbed: **reconciled (default) · model ·
  broker**. Model view = existing table unchanged (also the fallback when the
  backend predates these endpoints). Broker view = normalized fills.
- Reconciled table: closed time, symbol pair, direction, model entry→exit
  (ratio), broker entry/exit (per-leg avg), expected vs realized (USDT at
  broker size), delta, status chip, severity color, one-line explanation.
- Row click → drilldown: signal (probability/strength/reason), planned vs
  filled legs with fill timeline, attribution waterfall (expected → slippage
  → fees → funding → realized), sizing comparison, health verdict + operator
  guidance text.
- Filters: status, severity, symbol, breaks-only toggle; summary strip:
  matched %, explained %, open breaks, sign flips, avg slippage bps, fee
  drag, unmatched counts, last sync age, position parity.
- Ops module: recon breaks join the operations timeline; summary chips get
  break counts. Severity colors follow the design system (loss/warn accents
  reserved for exceptions — healthy explained differences render quiet).

## 10. Better-solution assessment (Phase 8, answered up front)

- **Exact row-by-row parity** — rejected (§3.1): structurally impossible
  (1 model trade ≙ 2 transitions ≙ ≥4 orders), and it criminalizes normal
  execution reality.
- **Tolerance-band parity alone** — rejected: hides *why*; a trade inside the
  band from luck and one inside from offsetting slippage+fees look identical.
- **Attribution-first only** — insufficient alone: per-trade explain without
  a standing position invariant misses "broker never closed" until the next
  trade tries to reconcile.
- **Chosen: position-first invariant + attribution-first trade layer**
  (hybrid). Position parity is the always-on tripwire; the attribution layer
  explains economics per round-trip; tolerance bands only decide
  explained-vs-unexplained on the *residual after attribution*, which is the
  only place a band is honest.

## 11. Testing

Backend (`tests/test_reconciliation.py`): linkage by signal_ts and client_id,
fallback window matcher, orphan classification (incl. manual), attribution
math against hand-computed fixtures, cost-flip (model win → broker net loss
⇒ `EXPLAINED_COSTS`, not break), true sign mismatch ⇒ critical, partial
fills, size drift, paper mode, sync cursor/late-fill idempotency, restart
recompute determinism. Synthetic broker-fill scenario builder lives in the
test module (usable by Simulation later).

Frontend (`src/lib/recon.test.ts`): row shaping, filters, severity/status
chip mapping, explanation lines, empty/degraded (endpoint-missing) states —
house style is lib-level vitest, matched here.
