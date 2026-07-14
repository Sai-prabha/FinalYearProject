# Meridian Research & Operator Tools

Design record for the operator-tool suite layered on the PRM backend and
Meridian: execution audit viewer, simulation campaigns, multi-symbol strategy
search, timing advisor, and the unified operations timeline. Written before
implementation; the sections below are the review artifact.

Companion records: `EXECUTION_CONTROL.md` (control plane), `STRATEGY_LAB.md`
(research pipeline). Nothing here weakens either.

---

## 0. Research grounding

What the implementation borrows from serious platforms (sources in the wiki
log entry):

- **Surveillance / audit tooling** (ACA, market-abuse platforms, audit-trail
  practice): every action carries actor, timestamp, action type, and
  before/after state; investigators work from a filterable timeline (date,
  actor, outcome) and drill into single events with full context; alerts are
  enriched by correlating with surrounding activity (here: broker executions
  and research runs on the same axis). Audit views are *investigative* tools —
  dense, filterable, exportable — not decorative history.
- **Simulation / stress platforms** (BlackRock 360°, AlternativeSoft, TS
  Imagine, testfol.io): decisions come from comparing strategies **under the
  same named scenarios** (crisis windows, volatility shocks), not from staring
  at one equity curve; presentation is decision-oriented — "does it survive
  the bad week" — with side-by-side tables and explicit thresholds.
- **Multi-symbol research** (TrendSpider variance testing): the same strategy
  is run across a universe and presented per-symbol *and* aggregated, framed
  as a **robustness test** — an edge that only exists on one symbol in one
  window is treated as suspect, not celebrated.
- **Alerting** (alert-fatigue literature): alert only on *unusual* relative to
  the instrument's own baseline (multiples of normal range, regime shifts);
  few alerts, each with an explicit "why now"; two-stage confirmation beats
  hair-trigger thresholds. Advisory, never auto-acting.
- **Multiple-testing discipline** (Bailey & López de Prado DSR; White's
  reality check lineage): the trial count must include **everything tried in
  the run — configs × symbols**. Adding symbols to a search multiplies
  trials and must raise the deflated-Sharpe bar for every candidate.

## 1. Tools, scope, and lifecycle fit

| Tool | Responsibility | Strategy-lab lifecycle fit | Writer/observer fit |
|---|---|---|---|
| **Execution audit** | Read-only investigation of control-plane changes (auto-execute) | none (control plane) | each backend serves ITS OWN audit file; Meridian labels which backend it is reading |
| **Simulation campaigns** | Named, repeatable replay grids (strategies × pairs × scenarios × fees) for decision support | cross-checks verdicts: CONFIRMS_REJECT reinforces REJECTED; ROBUST flags "research further" — never changes lifecycle by itself | pure computation; runs anywhere; never touches broker or control plane |
| **Multi-symbol search** | The existing research run over a whitelisted pair universe | full lifecycle per (config, pair); only default-pair (BTC-ETH) candidates are shadow-eligible | runs anywhere; shadow registration still writer-agnostic and evaluation-only |
| **Timing advisor** | Recommend research windows + explain why; WS advisory flag | advises when to trigger runs; never starts one | advisory only, no state |
| **Ops timeline** | One time axis: control changes, research runs, executions, campaigns | context/correlation view | read-only composition |
| **Search run log** | How research is actually being used (trigger, duration, verdict counts) | derived from research audit | read-only |

## 2. API surface (all `_require_auth_or_skip`)

### Execution audit (control plane)
- `GET /execution/audit?limit=50&cursor=&actor=&via=&outcome=&since=&until=`
  → `{rows: [...], next_cursor, total_matched}`; rows normalized:
  `{id, ts, event, actor, via, instance, prev, new, version, outcome,
  request_id, expected_version?, current_version?}`. Newest-first; cursor is
  the reverse-index into the filtered set (the file is human-action-scale,
  read whole + filter; revisit if it ever grows past ~10 MB).
- `GET /execution/audit/summary` → counts by actor/via/outcome, conflict
  rate, first/last ts, total.
- Source: `data/execution/audit.jsonl` (written by the control plane since
  `b184ef5`). Same auth as the control endpoints — the role that can change
  execution control can read its audit.

### Research audit (lab events, feeds timeline + run log)
- `GET /research/audit?…` — same reader/filters over
  `data/research/audit.jsonl` (`run_started/run_completed/run_failed/
  lifecycle/shadow_*/simulation_*`).

### Simulation
- `POST /simulation/campaign` `{label?, strategy_ids[], pairs?, window_days?,
  fees?}` → campaign meta (queued/running). 409 while any heavy job
  (research OR simulation) is running — one replay engine user at a time so
  the live signal loop is never starved.
- `GET /simulation/campaigns` → list (newest first).
- `GET /simulation/status?id=` → job state + phase.
- `GET /simulation/results?id=` → per (strategy × pair × scenario × fee)
  metrics + downsampled equity, per-strategy verdict
  (`ROBUST | MIXED | CONFIRMS_REJECT`) and `consistent_with_lab`.
- Persistence: `data/simulation/campaigns.json` (metas) +
  `data/simulation/<id>.json` (results). Audited into the research audit.

### Multi-symbol search
- `POST /research/run` gains `pairs: ["BTC-ETH", "SOL-ETH", …]` (subset of
  the whitelist; default `["BTC-ETH"]`).
- Store rows gain `pair`; non-default-pair row ids are `{spec_id}@{PAIR}`.
- `data/research/multi_symbol.json`: per-pair verdict counts + the
  strategy × pair lifecycle matrix for the latest run.

### Timing advisor
- `GET /research/timing` → `{now: {recommended, reasons[], confidence},
  windows: [{from, to, kind, reason, confidence}], vol: {r24h_pct, r30d_pct,
  ratio, regime}, last_success, next_scheduled}`.
- WS live payload gains `research_timing: {recommended, reason}` (cached,
  recomputed ≤ every 5 min) — Meridian toasts on the rising edge. **No
  auto-start.**

## 3. Engine changes (the one enabler)

`build_dataset` and the replay path are already pair-agnostic in substance —
`compute_probas(leg1, leg2)` treats "btc"/"eth" as leg *roles*. Changes:

- `build_dataset(..., pair=("BTCUSDT","ETHUSDT"))`; per-pair proba caches
  (`probas_{BASE}-{QUOTE}_{start}_{end}.parquet`; legacy unprefixed files
  remain the default pair's cache).
- `PAIR_UNIVERSE` whitelist (4 pairs: BTC-ETH, SOL-ETH, BNB-ETH, SOL-BTC —
  liquid USDT-margined legs with long history). Universe grows only with a
  written rationale, same rule as strategy families.
- `evaluate_run` accepts `{pair_key: dataset}`;
  the replay loop runs per pair, rows pool into ONE ranking with ONE
  deflated-Sharpe benchmark across **all** (config × pair) trials — adding
  pairs raises everyone's bar. Neighborhood gates stay within-pair. Holdout
  stays global top-5 (the holdout is mined once per run, not once per pair).
- Baselines are replayed on the default pair only (they ARE BTC-ETH
  strategies); the baseline score is the bar for every pair — conservative
  and explicit.

**Honest framing:** the frozen model was trained on BTC/ETH. Other pairs are
a **cross-pair robustness/transfer test** — evidence about whether the signal
is pair-specific or spurious — not a deployment menu.

## 4. Safety & governance

1. **No execution backdoors.** New modules never import the broker; the only
   lifecycle-changing surfaces remain the existing shadow/promote flows.
   Shadow registration **refuses non-default-pair candidates** (the live loop
   trades BTC/ETH; a SOL-ETH "deploy candidate" is research evidence, not a
   deployable).
2. **False-discovery discipline strengthened, not diluted.** Multi-pair runs
   multiply the DSR trial count; simulation campaigns are *post-hoc checks*
   of already-gated candidates and never feed scores back into ranking;
   scenario windows are auto-named (worst-vol week, quiet week) rather than
   cherry-picked.
3. **One heavy job at a time.** Research and simulation share a mutual
   exclusion check; both run in worker threads; the live loop is untouched.
4. **Advisory, never automatic.** The timing advisor recommends and explains;
   runs still start from the scheduler or an explicit operator action.
5. **Understanding over automation.** Every verdict carries machine-readable
   reasons; the audit viewer, timeline, and run log exist so the operator can
   reconstruct *why* the system did what it did.

## 5. Meridian surfaces

- **Ops module** (new route): execution-audit summary chips → filterable
  event table (actor/surface/outcome/time) → row drill-down (before/after,
  conflict context) → unified timeline underneath (audit + research runs +
  broker executions + campaigns). Empty/failure states for: endpoint missing
  (backend predates it), observer backend (banner names which backend's
  audit you are reading), auth required.
- **Strategy Lab additions** (Backtests): timing-advisor widget (next
  windows + "now" state + reasons); pair selection on search; leaderboard
  pair column + per-pair grouping; strategy-health matrix (family × pair
  verdicts); simulation panel (create from selected leaderboard rows,
  status, results with gate-echo badges); search run log.
- **Operator control card**: "last change by … via …" links into the Ops
  audit view pre-filtered to that actor.

## 6. Deliberately not built

- No generic "AI recommendations" — every advisor reason is a measurable
  condition (data completeness, vol-regime ratio, staleness, scheduler
  proximity).
- No per-symbol standalone strategies / new model training — the frozen
  model + config families remain the only strategy space.
- No cross-backend audit aggregation (each backend owns its file; Meridian
  labels its source) — revisit only if a real multi-instance need appears.
- No simulation-driven lifecycle changes — campaigns inform, gates decide.
