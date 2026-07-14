# v4.18.3 Next Candidate — H_regime (pre-registered 2026-07-14)

> Written BEFORE any conditional performance is computed. Gate = the frozen
> V4183 gate (`V4183_CANDIDATE.md` §3), unchanged. Prior ledger:
> H1/H2 retrains REJECTED (`reports/experiments.jsonl`). This document
> commits the regime definitions so they cannot be tuned to the outcome.

## 1. Hypothesis families considered and their fate

- **H_execution (maker tactics): REJECTED WITHOUT TRIAL — bounded above.**
  The 2 bps/side sensitivity (assumes 100% passive fills, zero adverse
  selection) already fails tune2025 (−0.076%/trade). The negative-drift
  literature (arXiv:2407.16527, 2502.18625) shows honest fill models are
  strictly worse than this bound: passive fills cluster on adverse moves,
  and simulators overestimate fills. Building a fill model to reject what
  arithmetic already rejects would be theater.
- **H_microstructure (OFI/VPIN/book features): REJECTED — not backtestable
  here.** No historical L2 or trade-tape storage exists; klines-level
  aggression proxies are already in the feature set. Pretending otherwise
  would produce untestable claims.
- **H_regime (CHOSEN):** the short-tail edge (v4.18) exists *conditionally*.
  The documented failure mode of every prior attempt is the 2026 ratio
  uptrend decaying stale shorts. A mechanistic trend condition — do not
  short when the ratio sits above its own daily mean — targets exactly that
  mechanism, is computable from the generator's existing per-bar inputs
  (ratio stream), adds zero features, zero latency, and zero new data
  dependencies.

## 2. Frozen definitions (no tuning — stated once, evaluated once)

Let SMA1440(t) = simple moving average of the ratio over the trailing 1440
one-minute bars (24 h). Let RV240(t) = standard deviation of 1m ratio log
returns over the trailing 240 bars, and P90(t) its trailing 30-day (43 200
bar) 90th percentile (self-normalizing; no tuned constant).

Variants (all declared now; selection among them on tune2025 ONLY):

- **V1 trend-block:** v4.18 unchanged, except a SHORT entry is blocked when
  ratio(t) ≥ SMA1440(t). (Shorts only below the daily mean — never short a
  ratio riding above it.)
- **V2 toxic-vol block:** v4.18 unchanged, except any entry is blocked when
  RV240(t) > P90(t).
- **V3 = V1 ∧ V2.**

Blocking affects ENTRIES only; exits and in-position management are
untouched (a blocked regime never traps an open position).

## 3. Protocol

1. Replay v4.18 (incumbent decision layer, existing cached old-model probas)
   with each variant's regime series on **tune2025**. Selection criterion:
   highest net PnL with net expectancy > v4.18's tune value (−0.176%) and
   n ≥ 15. If none qualifies → experiment over, ledger row, no holdout look.
2. The selected variant gets **one** holdout2026 evaluation through the
   frozen gate: net exp > 0, net PnL > −$22 (v4.18's), n ≥ 20, max DD >
   −7.4%, top-1-trade sign stability. The n ≥ 20 floor explicitly prevents
   "wins by never trading" (an empty book scores $0 > −$22 but fails n).
3. Ablation reporting: all three variants' tune numbers + the blocked-trade
   counts, so the mechanism (which trades got removed) is visible.
4. Fees: taker 4.5 bps/side × 4 fills, the standard harness.
5. If the gate passes: implement the SAME frozen rule inside
   `V416SignalGenerator` (config-driven, internal ratio deque — no feature
   contract change), register `v4.18.3`, shadow-activate. If it fails:
   ledger row, incumbent stands, no weights/config ship.

## 4. Why this is not hidden risk

The filter can only REMOVE trades from a fixed strategy — it cannot size up,
add leverage, widen stops, or extend horizons. Any improvement must come
from *not taking* specific trades. Risk metrics (DD, per-trade risk) are
gate-checked; trade-count transparency is mandatory in the report.

## 5. Results (appended after the runs, 2026-07-14)

Tune2025 (selection window; v4.18 unfiltered reference: −0.176%/trade, −$35):

| variant | n | net exp | net PnL | max DD | qualified? |
|---|---|---|---|---|---|
| V1 trend-block | 10 | −0.162% | −$5.69 | −0.60% | no (n < 15) |
| **V2 toxic-vol** | 50 | **−0.116%** | −$20.49 | −2.46% | **yes** |
| V3 both | 8 | −0.105% | −$2.98 | −0.52% | no (n < 15) |

V2 selected → ONE holdout2026 run:
**n=61, net exp −0.0341%/trade, net PnL −$7.52, PF 0.86, max DD −2.50%**
(robustness: identical to 4 d.p. under the live stride-quantile semantics).

**Gate: FAILED** — `holdout_net_exp_pos` is false. **v4.18.3 is not created**
(third consecutive refusal). However V2 *strictly dominates* the incumbent
v4.18 on holdout (exp −0.034% vs −0.085%, PnL −$7.52 vs −$22, DD −2.5% vs
−3.7%, same protocol, purely by removing entries in toxic-vol regimes) —
satisfying the PROGRESS ratchet ("beat v4.18 NET on holdout") without
reaching a positive edge.

**Disposition:** the filter ships as a config-gated incumbent refinement,
**default OFF** (`vol_filter_enabled` in `StrategyConfig`; live
implementation in `V416SignalGenerator` with parity-tested stride quantile;
entries only, exits untouched, "Toxic-vol regime" surfaces in blocked_by).
Enabling it on the demo default is an OPERATOR decision — it reduces bleed
~66% on holdout evidence but is not, and must not be sold as, an edge.
No weights, no version registration, no shadow change shipped.

## 6. What this experiment adds to the map

Three hypothesis families are now evidence-closed for this model family:
decision-layer tuning (2026-07-08/10), horizon-aligned retrains (H1/H2),
and execution tactics (bounded above by the maker sensitivity + negative-
drift literature). Regime conditioning (this experiment) is the first lever
that MOVED holdout economics materially in the right direction — the edge
problem is now isolated to signal quality *within calm regimes*. The next
credible hypothesis is therefore a model trained (or meta-labeled) ONLY on
calm-regime bars — conditional fitting rather than conditional gating —
through this same protocol.
