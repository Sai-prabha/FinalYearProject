# Autonomous Improvement Run — Progress Log

Session start: 2026-07-08 (Claude Code, autonomous).
Repo: `/Users/sai.p/Projects/FinalYearProject` (moved from `~/VSCode/Workspace(main)/FinalYearProject`; archive on Desktop 2026-06-24).

## Phase 0 — System map (verified against code 2026-07-08)

**Pipeline:** Binance spot WS (BTCUSDT/ETHUSDT 1m klines) → `api/feature_calculator.py:V414FeatureCalculator`
(50 causal rolling features over ≤1500-bar buffer, mirrors training `src/features.py`) →
`models/v4_14_production/model.json` XGBClassifier (fixed weights, AUC ≈ 0.52–0.54) → P(up) →
signal generator (`V414SignalGenerator` = v4.15 baseline; `V416SignalGenerator` drives v4.16/v4.17 configs
from `api/version_config.py`) → hysteresis + SL/TP + circuit breaker → position ∈ {-1,0,+1} →
`api/model_server.py` auto-exec gate (`broker.mode=="demo" and broker_config.auto_execute`) →
dual-leg MARKET orders (LONG ratio = BUY BTC + SELL ETH) via `api/broker_client.py`
(`BinanceFuturesBroker` = signed REST vs `BINANCE_BASE_URL`; `PaperBroker` fallback) →
JSONL order log `data/live/binance_trades_demo.jsonl`, trades `data/live/trade_history.{json,csv}` →
FastAPI WS/REST to React frontend (port 8888).

**Strategy layer state:** v4.15 frozen baseline; v4.16 live default candidate (+46% PnL in prior 104K-bar backtest);
**v4.17 candidate exists but is UNVALIDATED** (`min_hold` 25→40, `exit_threshold` 0.51→0.505) with explicit
"DO NOT PROMOTE without backtest" note and ready harness `scripts/backtest_v417_candidate.py`
(fetches Oct 1–Dec 15 2025 out-of-sample 1m klines via REST; model trained Feb–Oct 2025).

**Environment state at session start:** `.venv` was empty → reinstalled from `api/requirements.txt` (xgboost 3.3.0, py3.13).
`data/live/` empty (fresh clone; live history lives in the Desktop archive). No `binance-env.sh`, no `BINANCE_*` env vars
→ **no credentials available this session**; testnet order path will be built + verified up to the authenticated call.

## Phase 0 — Risk list (ranked)

1. **No transaction costs anywhere.** Sim/backtests execute at candle close with zero fees/slippage
   (documented in `run_v4_16_simulation.py`). A ratio round trip = 4 taker MARKET fills (~0.045%/leg on USDM)
   ≈ 0.18% of leg notional vs avg trade PnL ~+0.07%. Cost-blind eval can flip sign of every conclusion. → Fix in Phase 1.
2. **Base-URL not pinned.** `BinanceFuturesBroker` trusts `BINANCE_BASE_URL` verbatim; `BINANCE_ENV=demo` +
   production URL + production keys would trade real money. No testnet-host allowlist. → Fix in Phase 5.
3. **No kill switch / max-order-rate / symbol allowlist / dry-run flag** in the execution path
   (only `auto_execute` toggle + tiny default qtys 0.001 BTC / 0.05 ETH). → Phase 5.
4. **Sim ↔ broker divergence:** simulated portfolio (ratio PnL at close) is the source of truth for the
   frontend; broker fills are logged but not fed back into strategy PnL. Acceptable for demo; documented.
5. **Model-quality ceiling:** AUC ≈ 0.52 with fixed weights; decision-layer (thresholds/exits/sizing) is
   where realistic gains are. Prior tuning already acknowledges R:R ≥ 1.2 / PF ≥ 1.6 unreachable.
6. **Eval staleness:** prior comparisons end Dec 2025; six more months of true out-of-sample data
   (Jan–Jul 2026) now exist and are unused. → Phase 1 extended holdout.
7. Minor: per-bar backtest recomputes the full feature frame every bar (~hours per run) — motivates a
   cached-proba harness (probas are strategy-independent, so one feature/model pass powers every config sweep).

## Plan of attack

1. **Phase 1:** build `scripts/fast_backtest.py` — vectorized feature pass + batch predict + cached
   (time, proba, ratio) parquet; parity-check probas vs the per-bar live path on sampled bars; add a
   configurable fee/slippage model. Baseline scorecard for v4.15 + v4.16, Oct–Dec 2025 window **and**
   Jan–Jun 2026 holdout, with and without costs. Artifacts under `reports/eval/`.
2. **Phase 2–4:** judge v4.17 candidate on both windows; sweep decision-layer params (entry/exit thresholds,
   min_hold, TP width, trailing, min_signal_strength) tuned on 2025 window, validated on 2026 holdout only;
   target expectancy/PF/avg-win:avg-loss after costs, not hit rate.
3. **Phase 5:** safety hardening (testnet host pin, `EXECUTION_ENABLED` kill switch, dry-run mode, symbol
   allowlist, per-order + open-position caps, demo banner), credential checklist.
4. **Phase 6:** end-to-end loop with PaperBroker + `place_test_order` path; replay proof if no live signal.
5. **Phase 7–8:** before/after artifacts, docs/runbook, vault wiki + log updates.

## Phase 1–4 results (all artifacts under `reports/eval/`)

**Harness:** `scripts/fast_backtest.py` — vectorized feature pass + batch predict + cached
(time, proba, ratio) parquet under `data/backtest/`; any strategy config replays a quarter of
1m bars in ~1s. **Parity vs the live per-bar path: max |Δproba| = 0.00e+00 over 30 sampled bars.**
Configurable fees (`--fee-bps-per-side`, default 4.5 = USDM taker; 4 fills per ratio round trip = 0.18%).

**Baseline scorecards (NET of taker fees):**

| config | window | trades | net exp/trade | net PF | net PnL ($1k start) | max DD |
|---|---|---|---|---|---|---|
| v4.15 | 2025-10-01→12-15 | 1600 | −0.159% | 0.19 | −$766 | −77% |
| v4.15 | 2025-12-15→2026-07-01 | 3329 | −0.145% | 0.19 | −$936 | −94% |
| v4.16 | tune / holdout | 73 / 1534 | −0.220% / −0.135% | 0.07 / 0.26 | −$61 / −$513 | −6% / −52% |
| v4.17 | tune / holdout | 72 / 1280 | −0.210% / −0.139% | 0.10 / 0.26 | −$58 / −$463 | −6% / −47% |
| **v4.18** | tune / holdout | **58 / 78** | **−0.176% / −0.085%** | 0.45 / 0.71 | **−$35 / −$22** | **−3.8% / −3.7%** |

**Headline findings (the truth-first deliverable):**
1. *Every* pre-existing config is deeply net-negative under realistic taker fees. The historical
   "+46% v4.16 vs v4.15" claim was gross-of-fees and does not survive: entries at |p−0.5| ≥ 0.025 held
   ~25–70 min sit where the model edge is +0.01–0.03%/trade — ~10x below the 0.18% cost hurdle.
2. Proba-bucket analysis (`scripts/proba_informativeness.py`, both windows) shows the only
   *bucket-level* edge replicating out-of-sample is p<0.45 SHORT at ~240 bars: +0.25% (t=7.4) /
   +0.22% (t=5.6). Longer horizons invert on the 2026 holdout; long-tail signals flip sign.
3. **Event study kills even that**: entering at *first tail-crossing* (the only implementable entry)
   yields −0.04% (2025) / +0.16% (2026) gross, t ≤ 1.4 — the bucket average was an episode-clustering
   artifact (mid-episode bars condition on persistence you can't know at entry).
4. **Conclusion: with the fixed v4.14 weights (AUC ≈ 0.52), no operating point clears taker costs.**
   Sensitivity sweep `scripts/sweep_v418.py` (3 thresholds × 3 horizons × 2 windows) confirms: all net-negative.

**What was shipped anyway — v4.18 "conviction gate"** (`api/version_config.py:V418_CONFIG`, new
`entry_threshold_long/short` + `max_hold_bars` support in `V416SignalGenerator`): SHORT-only p<0.45
entries, 4h time exit, ≤1x sizing, wide guardrail SL/TP, churn gates removed. It is the honest best
default for the demo product: ~30 trades/quarter (vs 1600–3300), bleed ≈ −1%/month (vs −14%/month for
v4.15 net), max DD −3.7% (vs −77…94%), payoff asymmetry achieved (avg win +0.53…0.60% vs avg loss
−0.34…0.52%, W/L 1.2–1.7 vs 0.72 baseline). Bug fixed en route: `model_server.py` lifespan sent every
non-v4.16 version to the v4.15 legacy generator — v4.17/v4.18 silently ran the wrong strategy.

## Phase 5–6 — execution safety + e2e (all verified by runnable tests)

- **Testnet host pin:** `BinanceFuturesBroker` refuses any `BINANCE_BASE_URL` host outside
  `ALLOWED_BASE_HOSTS = {testnet.binancefuture.com}` — production endpoints structurally unreachable.
- **Safety gate on every order** (`check_order_safety`, both brokers): file kill switch
  (`touch data/live/KILL_SWITCH`), symbol allowlist (BTCUSDT/ETHUSDT), per-order qty caps
  (0.01 BTC / 0.5 ETH, env-overridable), plus `BROKER_DRY_RUN=true` (routes to /order/test only).
- `BINANCE_ENV != demo` (including `live`) always falls back to PaperBroker (pre-existing, verified).
- **Tests:** `tests/test_broker_safety.py` (6 checks) and `tests/test_auto_exec_e2e.py`
  (FLAT→SHORT→LONG→FLAT via the real transition code path: 8 legs placed + JSONL-logged) — both green.
- **Live REST e2e** against a running server: paper fill OK; oversized, off-allowlist, and
  kill-switch orders all REJECTED with reasons; all attempts in `data/live/binance_trades_demo.jsonl`.
- Server boots and streams live data on v4.15 and v4.18 (verified 2026-07-08).

**BLOCKED — final authenticated step (needs the operator):** no testnet credentials on this machine.
Checklist to go live on testnet:
1. Create account/API key at https://testnet.binancefuture.com (independent of real Binance funds).
2. `cp binance-env.sh.example binance-env.sh`, fill `BINANCE_API_KEY/SECRET` (testnet values).
3. `source binance-env.sh && .venv/bin/python start_model_server.py --model-version v4.18`
   — expect the "🟢 TESTNET / DEMO ONLY" banner; first run with `BROKER_DRY_RUN=true` recommended.
4. Sanity: `curl -s -X POST localhost:8888/trade/test -H 'Content-Type: application/json' -d
   '{"symbol":"BTCUSDT","side":"BUY","quantity":0.001}'` → `TEST_OK`; then a real 0.001 BTC testnet order
   via `/trade`; then flip `auto_execute` on via `POST /broker/config {"auto_execute": true}`.

## Recommended next steps (in value order)

1. **Cost reduction before model work:** maker-style entries (LIMIT post-only at first tail-crossing)
   would halve the hurdle to ~0.08%; the harness's `--fee-bps-per-side 2` scenario is the cheap test.
2. **Model refresh (speculative, not started):** retrain the classifier through 2025-12 with the same
   pipeline, evaluate on 2026 holdout with the fee-aware harness *before* any promotion. Do not trust
   gross metrics anywhere in this repo's older scripts.
3. Longer-horizon labels (4h relative move) would align training with the only horizon that showed any
   out-of-sample structure.
4. Keep `reports/eval/` as the ratchet: any future change must beat v4.18 NET on the holdout window.

## Status log

- 2026-07-08: Phase 0 complete. Venv rebuilt.
- 2026-07-08: Phases 1–6 complete (this update). v4.18 shipped + safety rails + e2e; testnet creds are
  the only blocker. Files changed: `api/broker_client.py`, `api/version_config.py`,
  `api/feature_calculator.py`, `api/model_server.py`, `binance-env.sh.example`,
  `scripts/fast_backtest.py`, `scripts/proba_informativeness.py`, `scripts/sweep_v418.py`,
  `tests/test_broker_safety.py`, `tests/test_auto_exec_e2e.py`. Uncommitted (per operating rules —
  review the diff and commit when back).

## Addendum — canonical-copy port + testnet auth status (2026-07-08, later same session)

- Mid-session discovery: this vault copy (`projects/predicting-relative-movement/`) is the **canonical
  working tree** (per its CLAUDE.md); the session had started in the stale git clone
  `/Users/sai.p/Projects/FinalYearProject`. All session work was ported here and re-verified:
  **42/42 checks green** (test_broker_safety 6, test_auto_exec_e2e, pytest guards/kill-switch/leg-sync 34)
  and the holdout replay reproduces identical numbers. The git clone was kept in sync file-for-file
  for the api/scripts/tests files this session touched (commit from there when ready).
- This copy already had `api/execution_guards.py` + `api/kill_switch.py` (session max-loss state
  machine) from a previous session — complementary to the new order-level gate; both remain active.
- **Testnet credentials exist here (`binance-env.sh`) but are EXPIRED**: balance call returns
  `-2015 Invalid API-key` (futures testnet resets/expires keys). Found + fixed a real gap this exposed:
  `make_broker_client()`'s smoke test treated Binance error envelopes as success, so the server would
  claim demo mode with dead keys — it now falls back to PaperBroker with a clear warning (verified).
- **Only remaining blocker (operator, ~2 min):** log in at https://testnet.binancefuture.com →
  API Key tab → regenerate → paste the new key/secret into binance-env.sh. Then:
  `source binance-env.sh && .venv/bin/python start_model_server.py --model-version v4.18`
  (browser session wasn't available from this session — Chrome extension not connected).
