# Microstructure Data Layer — L2 + Trade-Tape Ingestion and Features

Design record, 2026-07-14. Read this before touching `api/microstructure_ingest.py`
or `api/microstructure_features.py`.

## Why this exists

Four consecutive v4.18.3 gate refusals (`reports/experiments.jsonl`, H1–H4)
established that **every hypothesis family reachable from 1m-kline data is
evidence-closed**. The documented way forward is new information. This layer
captures it — limit-order-book snapshots and trade-tape flow — starting now,
so that future candidates have data that does not exist historically.
**Richer data does not guarantee an edge; it creates the conditions under
which an edge could exist.** Candidates built on it still face the same
frozen gate.

## Source decision (challenged from the mission brief)

The brief said "Binance USDⓈ-M futures **demo**". Demo/testnet **market
data** would be worthless for research: testnet books are thin and
unrepresentative (community-documented; the venue exists to test order
mechanics, not to mirror liquidity). The system already takes its price
truth from production public data (spot klines via `wss://stream.binance.com`).
So this layer ingests **production USDⓈ-M futures public market data** from
`wss://fstream.binance.com` — keyless, unauthenticated, subscribe-only.
The demo account is untouched; the broker host allowlist
(`ALLOWED_BASE_HOSTS = {testnet.binancefuture.com}`) is untouched; this
module imports nothing from `broker_client` and contains no signed request,
no POST, and no order path of any kind.

## Ingestion: WebSocket, not REST polling

- WS market streams carry **zero request weight** — REST `depth` polling at
  1–2 Hz × 2 symbols would burn rate-limit budget shared with the broker's
  REST host and give worse resolution.
- The server is already a long-lived asyncio process with a reconnecting WS
  task (`connect_to_data_stream`); this adds a second, identical-pattern task.
- One combined connection, four streams:
  `btcusdt@depth20@500ms / btcusdt@trade / ethusdt@depth20@500ms / ethusdt@trade`.
  (Live-verified 2026-07-14: `@aggTrade` is silently absent from fstream
  combined streams — the subscription is accepted but never delivers; `@trade`
  delivers the raw tape with the same `p/q/m/T` fields. The 1s aggregation
  absorbs the higher message rate.)
- Reconnect with capped exponential backoff (1s → 60s); Binance recycles
  connections ~24h — the loop treats any close as a normal reconnect.
- Env gate `MICRO_INGEST`: `1` force on, `0` force off, unset → on only when
  `RAILWAY_*` env is present (same auto-detection philosophy as the writer
  gate; local dev servers stay quiet by default and never double-collect).

## Storage

Under the **proven-persistent** Railway volume path (`data/live/` survives
deploys — established 2026-07-14 during the control-plane rollout):

```
data/live/micro/lob/BTCUSDT/2026-07-15/<start_epoch>.parquet
data/live/micro/trades1s/BTCUSDT/2026-07-15/<start_epoch>.parquet
(same for ETHUSDT)
```

- **Segment files, not daily appends**: rows buffer in memory and flush every
  `MICRO_FLUSH_SECONDS` (default 600) as an immutable segment written
  atomically (tmp + rename). Parquet cannot be safely appended in place; an
  open ParquetWriter that dies mid-write corrupts the day. Segments make a
  crash lose ≤ one flush interval and nothing else. Day directories give the
  mission's daily partitioning; readers glob segments by day range.
- **LOB rows** (1 Hz, sampled from the 500ms stream — latest book per second):
  `ts_ms`, `event_ms`, `bid_px_0..19`, `bid_sz_0..19`, `ask_px_0..19`,
  `ask_sz_0..19` (flat float64 columns — the parquet-idiomatic layout of the
  brief's "arrays"), plus precomputed `best_bid`, `best_ask`, `mid`,
  `spread`, `bid_depth20`, `ask_depth20`.
- **Trade rows — 1-second aggregates by default**: `ts_s`, `n_trades`,
  `buy_qty`, `sell_qty` (aggressor side from the trade-stream maker flag `m`:
  `m=true` ⇒ buyer was maker ⇒ SELL aggressor), `buy_n`, `sell_n`, `vwap`,
  `last_px`, `high`, `low`. Rationale: BTCUSDT futures can print millions of
  trades/day (~10²MB/day compressed) — unsustainable on a small volume,
  and per-trade granularity adds nothing at the strategy's 1m–4h horizons.
  `MICRO_RAW_TAPE=1` additionally persists the raw tape
  (`data/live/micro/tape/...`) for short-retention studies.
- **Volume budget** (both symbols): LOB ≈ 86,400 rows × ~90 cols/day ≈
  10–20 MB/day/symbol compressed; trades1s ≤ 86,400 rows/day/symbol ≈ low
  MB. Total ≈ **25–50 MB/day**, ~4–9 GB for the 180-day default retention.
- **Retention**: daily janitor deletes day-directories older than
  `MICRO_RETENTION_DAYS` (default 180; raw tape `MICRO_TAPE_RETENTION_DAYS`
  default 14). Deletions are logged. No aggregation-on-expiry in v1 — the
  1s aggregates already are the long-form.

## Feature pipeline (`api/microstructure_features.py`)

Offline, deterministic, read-only over the parquet layer:

```python
load_lob(symbol, start_ts, end_ts)          # raw 1Hz book frame
load_trades_1s(symbol, start_ts, end_ts)    # raw 1s tape aggregates
load_microstructure_features(symbol, start_ts, end_ts, feature_set="basic")
```

`basic` features (1s grid, forward-fillable to 1m for kline alignment):

| feature | definition |
|---|---|
| `mid`, `spread`, `spread_bps` | best-level book state |
| `imb_top1` | `bid_sz_0 / (bid_sz_0 + ask_sz_0)` |
| `imb_top20` | `bid_depth20 / (bid_depth20 + ask_depth20)` |
| `ofi_1s` | Cont–Kukanov–Stoikov best-level OFI between consecutive snapshots |
| `signed_flow_1s` | `buy_qty − sell_qty` (tape) |
| `aggression` | `buy_qty / (buy_qty + sell_qty)` |
| `rv_60s`, `rv_300s` | realized vol of 1s mid log-returns |
| `regime` | quiet / normal / toxic — `rv_300s` and `spread_bps` vs their own trailing quantiles (same self-normalizing philosophy as the shipped `vol_filter_enabled` RV240/p90 rule) |

Determinism: pure functions of stored rows; no wall-clock, no network.
Same inputs ⇒ same outputs (tested).

## Health

`/status` gains a `microstructure` block: `enabled`, per-symbol
`last_lob_ms` / `last_trade_ms` / age seconds, `rows_flushed_today`,
`segments_today`, `last_error`, `stale` (no LOB event for >120 s while
enabled). Meridian's Ops surface can read this from the existing `/status`
poll — no new endpoint required.

## Strategy Lab / future candidates

- Nothing in v4.15/v4.18 or any live path reads this layer. Offline only.
- Alignment recipe for training: resample features to 1m
  (`last` within minute, then `shift(1)` so the value is known at bar open),
  join on kline `time`; labels/trade records join exactly as in
  `scripts/tune_v4183_calm.py`.
- Candidate ideas this unlocks (each still pre-registered + gated):
  regime-aware entry gates using `imb_top20`/`ofi` instead of RV-only;
  OFI-based trade-acceptance filters (the H_B meta family, but with features
  the primary has never seen and a real information delta); execution-timing
  studies (spread/imbalance state at entry vs realized slippage from the
  reconciliation layer).
- **Constraint that stays true**: no candidate can be backtested on
  microstructure older than the day this layer went live. Sample accrues
  ~30 days/month; a first honest microstructure experiment needs ≥2–3
  months of accrued data plus the usual tune/holdout split.

## Safety summary

Read-only public data; no keys touched; no new broker paths; broker host
allowlist untouched; writer logic untouched; ingest failures degrade to a
logged `/status` error and never affect trading, execution, or the kline
stream (separate task, separate connection).
