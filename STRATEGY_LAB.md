# Strategy Lab — research → deploy-candidate → shadow → promote

Architecture decision record for the strategy research platform behind Meridian's
Backtests module. Read this before touching `api/strategy_lab.py`.

> **Addendum (operator tools build — RESEARCH_TOOLS.md):**
>
> - **Multi-pair search**: `POST /research/run` accepts `pairs` (whitelist
>   `PAIR_UNIVERSE`; BTC-ETH always included). Candidates are replayed per
>   pair and **all (config × pair) trials pool into one deflated-Sharpe
>   benchmark** — adding pairs raises the bar for everyone, so multi-symbol
>   search strengthens the false-discovery discipline. The frozen model is
>   trained on BTC/ETH; other pairs are cross-pair robustness evidence, and
>   **only default-pair candidates are shadow-eligible** (enforced with a 400).
>   Per-pair verdicts: `data/research/multi_symbol.json` / `GET /research/multi-symbol`.
> - **Simulation campaigns** (`api/simulation_lab.py`, `/simulation/*`):
>   post-hoc scenario replays (full window / vol-spike week / quiet week) at a
>   fee band; verdicts (`ROBUST | MIXED | CONFIRMS_REJECT`) never feed back
>   into ranking or lifecycle; research and simulation are mutually exclusive
>   so only one heavy replay job runs at a time.
> - **Timing advisor** (`api/timing_advisor.py`, `GET /research/timing`):
>   recommends research windows with measurable reasons (UTC-close data
>   completeness, Saturday liquidity trough, realized-vol regime shift vs the
>   30d baseline, staleness). Advisory only.
> - **Audit read APIs**: `GET /research/audit` (this lab's trail) and
>   `GET /execution/audit` (control plane) feed Meridian's Ops timeline.

## What "strategy search" means here

The alpha model is frozen (v4.14 XGBoost weights, AUC≈0.52). Every live version
since v4.15 differs only in its **strategy layer** — a `StrategyConfig` applied to
the shared P(up) stream (thresholds, sides, holds, exits, sizing). Strategy search
is therefore a **disciplined enumeration of StrategyConfig variants** replayed over
cached model probabilities with realistic fees, using the parity-verified
`scripts/fast_backtest.py` engine (max |Δproba| vs live path = 0.0).

It is deliberately NOT a brute-force optimizer:

- **Small, explainable families** (~40 trials/run), each with a written rationale.
- Every trial is counted in a multiple-testing haircut (see Ranking), so adding
  trials automatically raises the bar for all of them.
- Known-bad families are kept as **controls** — the pipeline must reject them.

### Candidate families

1. **`conviction`** (v4.18 lineage) — trade only the probability tails, hold to a
   time horizon. Grid: side ∈ {short, both} × tail ∈ {0.44, 0.45, 0.46} ×
   max_hold ∈ {120, 240, 480} bars. Rationale: the only edge that has ever cleared
   taker fees in this system is the p<0.45 tail at a ~4h horizon
   (reports/eval 2026-07-08: +0.25% t=7.4 tune, +0.22% t=5.6 holdout).
2. **`band`** (v4.15/16 lineage, control group) — symmetric entry/exit band churn.
   Grid: entry ∈ {0.525, 0.535, 0.545} × exit ∈ {0.51, 0.505} × min_hold ∈ {25, 40}.
   Rationale: known cost-hostile (~0.18%/round-trip vs +0.01–0.03% edge). These
   exist to prove the gates reject fragile strategies; a `band` candidate reaching
   DEPLOY_CANDIDATE is a red flag for the pipeline, not a discovery.

Registered baselines (v4.15…v4.18) run through the identical protocol as reference
rows so every ranking is relative to what is actually deployed.

## Evaluation protocol (walk-forward + reserved holdout)

Dataset: cached (time, proba, ratio) parquet, rebuilt incrementally from public
Binance klines; default lookback 270 days ending at the last complete UTC day.

```
|—————————— validation: 5 contiguous folds ——————————|— holdout (56d) —|
```

- **Validation replay** is one continuous run (generator state — compounding,
  circuit breakers — carries across folds, matching deployment). Fold metrics
  slice closed trades by exit time; folds measure **regime stability**, not
  re-fitting (configs are fixed a priori — there is nothing to fit per fold).
- **Holdout replay** starts a fresh generator (cold start, like a real deploy)
  and is only ever run for candidates that already passed validation gates
  (top-5 max). Nobody gets to mine the holdout.
- **Costs**: taker 4.5 bps/side × 4 fills default; every candidate is re-scored
  at 2 bps ("maker-ish") and 0 bps as a **fee-sensitivity band** — a strategy
  whose sign flips between 4.5 and 2 bps is a cost artifact, and the UI says so.

## Ranking + gates ("deploy-candidate" must mean something)

Per-candidate robust score (validation, net of fees):

- `edge` — shrunk net expectancy: `mean_net_exp · n/(n+30)` (shrinks small
  samples toward zero; 30-trade prior).
- `consistency` — fraction of folds with positive net expectancy.
- `dd` — max drawdown of the validation equity path.
- `score = edge · consistency · (1 + maxDD)` — a fragile or inconsistent edge
  scores near zero. The score ranks; the **gates** decide.

**DEPLOY_CANDIDATE gates** (all must pass; each failure is recorded as a
machine-readable reason — the UI's "why rejected" explainer):

| gate | threshold | why |
|---|---|---|
| trades | ≥ 30 closed in validation | statistical floor |
| consistency | ≥ 3/5 folds net-positive | regime stability |
| worst fold | expectancy > −0.10%/trade | no catastrophic regime |
| concentration | best trade < 40% of gross profit | not one lucky fill |
| deflated Sharpe | validation SR > E[max SR] of N trials under the null (Bailey & López de Prado benchmark, stdlib `NormalDist`) | multiple-testing haircut |
| neighborhood | median score of grid neighbors > 0 | parameter plateau, not cliff |
| holdout | net expectancy > 0 AND PF ≥ 1.0 | out-of-sample survival |

- Passing all gates **and** beating the live baseline's same-protocol score →
  `DEPLOY_CANDIDATE`.
- Passing all gates but not beating the baseline → `MATCH`.
- Any gate failed → `REJECTED` with reasons.

## Lifecycle

```
DISCOVERED → BACKTESTING → REJECTED
                         → MATCH
                         → DEPLOY_CANDIDATE → SHADOW_RUNNING → SHADOW_REJECTED
                                                             → PROMOTION_READY → PROMOTED
```

State lives in the candidate manifest (`data/research/candidates.json`) and every
transition is appended to `data/research/audit.jsonl` (who/what triggered it, when,
why — including scheduler-started runs).

## Shadow flow (evaluation-only, structurally)

"Run in shadow" registers the candidate's config as a runtime version
(`lab-<id>`) via `version_config.register_version`, then swaps it into the
**existing** shadow slot — the same machinery `SHADOW_MODEL_VERSION` uses today,
not a second scheme. Everything already proven about that slot applies:

- shadow writes only `data/live/shadow_<ver>_trades.json`;
- the execution path never reads shadow state (test-enforced:
  `tests/test_execution_planner.py` runs a shadow that raises on any read);
- broker orders come from the primary generator alone.

`data/live/shadow_registration.json` records the active lab registration so a
restart restores it (explicit registration wins over the `SHADOW_MODEL_VERSION`
env; with no registration file the env behaves exactly as before).

**Promotion readiness** (evaluated continuously, surfaced in `/shadow/status`):
≥ 14 days in shadow AND ≥ 5 closed shadow trades AND net expectancy ≥ 0 AND
matched-window score ≥ primary's. Readiness is a signal, never an action — going
live remains the deliberate `/model/promote` flow (confirm=true, flat, no drift).

## Scheduler ("optimal research windows")

Crypto has no session close; the natural boundaries are the **UTC day** (daily
kline close = data-completeness point) and the **weekly liquidity trough**.
Research on BTC intraday seasonality places the lowest volume/volatility in the
00:00–06:00 UTC window (realized-vol minimum ≈ 05:00 UTC), with weekends quieter
than weekdays. Therefore:

- **Daily refresh — 04:30 UTC**: after the UTC close with hours of slack for
  data completeness, inside the low-activity window, and finished well before
  the US/EU overlap (14:00–16:00 UTC) where most volume — and most of this
  strategy's action — occurs.
- **Weekly deep scan — Saturday 05:00 UTC**: widest grid, cheapest hours of the
  week.
- **Guards**: one research job at a time (async lock); replay runs in a worker
  thread so the live signal loop never blocks; a run is skipped if the last
  success is < 20h old; `RESEARCH_AUTO=0` disables scheduling entirely.
- **Observability**: `GET /research/status` — job state/progress, last run
  (duration, trials, outcomes), next scheduled run, top candidates.

## API surface (all auth-gated like other admin routes)

- `GET  /research/status` — scheduler + current/last job + leaderboard summary
- `POST /research/run` — start a research job (409 if one is running)
- `GET  /research/candidates` — full ranked candidates with gates/folds/fees
- `POST /research/candidates/{id}/shadow` — confirm-gated shadow registration

## Deliberately not built (and why)

- **Auto-promotion** — promotion stays a human action behind `/model/promote`.
- **Paper portfolio of all candidates** — one shadow slot is enough until a
  candidate has ever earned PROMOTION_READY; more slots = more surface, no
  decision value yet.
- **Unbounded/generative search spaces** — every added trial degrades every
  other trial's deflated-Sharpe bar; families grow only with a written rationale.
