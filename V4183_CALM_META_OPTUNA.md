# v4.18.3 Fourth Candidate — H_calm (conditional fitting + disciplined HPO)
## Pre-registered 2026-07-14, BEFORE any conditional number was computed

> Gate = the frozen V4183 gate (`V4183_CANDIDATE.md` §3), unchanged for the
> fourth time. Prior ledger: H1/H2 retrains and H3 regime gating all
> REJECTED (`reports/experiments.jsonl`). H3 isolated the open question this
> experiment attacks: v4.18's gross expectancy in calm regimes is ~+0.15%/trade
> but the 0.18% taker round trip eats it — nobody has yet FIT a model to calm
> bars. This document freezes every definition, search range, trial budget and
> objective so nothing can be tuned to the outcome.

## 0. Current regime + gate note (Phase 0 deliverable)

- **Regime filter as shipped:** `vol_filter_enabled` (default OFF) in
  `StrategyConfig`; calm ⇔ RV240 ≤ trailing-30d p90 of RV (43 200-bar window,
  min_periods 960, self-normalizing). Live implementation in
  `V416SignalGenerator` (entries only, stride-30 quantile, parity-tested to
  4 d.p. vs the offline series). Proven effect (H3): holdout2026 bleed
  −0.085% → −0.034%/trade — **cost avoidance, not edge**.
- **Gate (frozen, 3 refusals):** ONE holdout2026 evaluation; net exp > 0,
  net PnL > −$22 (v4.18's), n ≥ 20, max DD > −7.4%, top-1-trade-removed sign
  stability. Selection strictly on tune2025 (n ≥ 15, exp > −0.176% floors).
  Ledger append-only.
- **HPO history is the local cautionary tale:** `scripts/optuna_tune_v4*.py`
  (legacy, pre-ratchet) tuned XGB + strategy params to maximize *gross*
  Sharpe; that era produced the fee-blind "+46% v4.16" claim demolished in
  PROGRESS Phase 1. Optuna re-enters ONLY with: cost-aware robust objective,
  compact pre-registered space, fixed trial budget, tune-only optimization,
  one gate shot. Optuna is a research-only dependency — it must never enter
  `api/requirements.txt` or the Railway image.

## 1. Research conclusions driving the design (Phase 1)

- **Meta-labeling (López de Prado):** secondary model filters the primary's
  false positives / sizes bets; explicitly suited to high-recall/low-precision
  primaries and regime non-stationarity. Known failure mode (QuantConnect
  practitioners' critique): a meta-model on the SAME features extracts nothing
  new. It earns its keep only when fed **state features the primary never
  saw**. Our primary (frozen v4.14, AUC≈0.52, most trades fail to clear
  costs) fits the useful case; the meta features below (vol headroom, trend
  gap, threshold margin) are absent from the 50-feature contract.
- **Sample-size reality:** the base strategy yields only ~10²  trades on the
  train window ⇒ the meta-model must be a regularized logistic regression
  with ≤3 features and NO hyperparameter search. Anything bigger fits noise.
- **HPO best practice** (GT-Score paper arXiv:2602.00080; multi-objective
  Optuna anti-overfitting write-ups; walk-forward literature): never optimize
  raw single-window PnL; reward consistency across sub-periods; keep spaces
  compact and interpretable; validate on an untouched holdout. Adopted here
  as a **min-over-two-tune-halves net expectancy** objective with hard
  floors — the strongest robustness statement available at n≈50 trades.
- **Optuna vs alternatives:** TPE ≈ random search at ≤8 dims × 40 trials in
  expectation, but Optuna adds a per-trial ledger (SQLite + JSONL), seeded
  reproducibility, and clean define-by-run constraints; Ray Tune adds
  cluster machinery we don't need; Hyperopt is unmaintained relative to
  Optuna; grid is infeasible at 8 dims. **No pruner**: trials are cheap
  (~20 s) and return-based pruning biases selection toward front-loaded
  windows — the exact bias the mission warns about.

## 2. Frozen regime + window definitions

Identical to H3 (`eval_v4183_regime.py`), restated:

- RV240(t) = std of 1m log ratio returns over trailing 240 bars.
- P90(t) = trailing 43 200-bar 90th percentile of RV240 (min_periods 960).
- **calm(t)** ⇔ RV240 ≤ P90 and P90 defined. Unknown ⇒ toxic (conservative).
- Windows (exact cached spans): train = 2025-02-01→2025-10-01 (fetch from
  01-25 for warmup); tune2025 = cache `1759168800000→1765756800000`;
  holdout2026 = cache `1765648800000→1782864000000`. Fees 4.5 bps/side ×
  4 fills. Label horizon 240 bars (matches v4.18 hold ceiling); labels end
  240 bars before train end (no peek).

## 3. Candidate families (both declared now; selection on tune2025 ONLY)

### H_A — calm-conditional retrain + disciplined Optuna HPO
Train an XGBClassifier (same 50-feature contract — REQUIRED for dual-model
shadow loading) on **calm train bars only**, label = sign of 240-bar forward
log ratio return. Deployment semantics: v4.18 decision layer, SHORT-only
(long_thr 1.01), entries gated to calm bars (the shipped V2 filter), exits
untouched.

**Optuna study (frozen):** TPE sampler `seed=42`, `n_trials=40`, no pruner.
Search space:

| param | range | rationale |
|---|---|---|
| max_depth | {2, 3, 4} | v4.14 uses 3; ±1 |
| n_estimators | 20–120 step 10 | v4.14 uses 35 |
| learning_rate | 0.02–0.10 log | v4.14 uses 0.04 |
| min_child_weight | 1–20 log-int | regularization |
| subsample | 0.6–1.0 | regularization |
| colsample_bytree | 0.6–1.0 | regularization |
| reg_lambda | 0.5–10 log | regularization |
| short_q | {0.02, 0.05, 0.10} | short thr = that quantile of the candidate's probas on valid∧calm tune bars (tune-only, as H1) |

### H_B — calm meta-acceptance gate (NO hyperparameter search)
Base = incumbent v4.18 on frozen v4.14 probas, calm entry gating. Meta
training set = trades from replaying that base on the TRAIN window. Meta
features at entry bar (all causal, all computable live, none in the primary's
50): `thr_margin = short_thr − p`, `vol_headroom = RV240/P90`,
`trend_gap = log(ratio/SMA1440)`. Meta label = `pnl_pct_net > 0`. Meta model
= `LogisticRegression(C=1.0)` on z-scored features (train-trade stats only) —
fixed, no search. Acceptance: take the entry iff meta-probability ≥ cutoff,
cutoff grid **{0.40, 0.45, 0.50, 0.55, 0.60}** scored on tune2025.
*Design choice:* the meta-model trains on the calm-gated trade population
(train/serve consistency) with vol-headroom as a feature — not on all-regime
trades — because the deployed candidate only ever acts on calm entries.

## 4. Objective (both families; tune2025 ONLY)

Split tune2025 bars into two contiguous halves. Replay the candidate; score:

```
score = min(net_expectancy_half1, net_expectancy_half2)
hard reject (score = −1000 − shortfall) if:
    n_total < 15, or either half has < 3 trades, or max DD ≤ −6%
```

Rationale: rewards sub-period consistency (GT-Score philosophy), denies wins
concentrated in one lucky cluster, keeps the existing tune floors. The
holdout parquet is never loaded inside the objective or the study — it is
touched exactly once, after family selection, by a separate function.

## 5. Selection + the ONE holdout shot

1. Best H_A trial and best H_B cutoff by the score above.
2. The single overall winner (higher score, floors met) is THE candidate.
   If neither clears the floors → experiment over, ledger row, holdout
   untouched.
3. ONE holdout2026 evaluation through the frozen gate: net exp > 0, net PnL
   > −$22, n ≥ 20, max DD > −7.4%, top-1-removed sign stability.
4. **Commitments:** no re-runs, no second holdout look, no gate edits, no
   widened search after seeing results. Fail ⇒ candidate weights deleted,
   ledger row `v4.18.3-h4-calm-fit`, incumbent stands. Pass ⇒ weights to
   `models/v4_18_3/` with full Optuna provenance (study db, trial JSONL,
   best params), shadow-slot activation (read-only vs broker, existing
   isolation), Meridian metadata; H_B additionally requires the meta gate
   implemented + tested in `V416SignalGenerator` BEFORE any shadow start.
   v4.15/v4.18.3 never share a broker account (split-account rehearsal doc
   stands).

## 6. Anti-overfitting safeguards (summary)

- 40 trials total, pre-registered; 8-dim compact space grounded in v4.14's
  own values; no pruner; fixed seeds (TPE 42, XGB 42).
- Robust min-half objective + hard floors, computed net of fees.
- H_B has zero tunable model hyperparameters and 3 meta features for ~10²
  samples.
- Selection multiplicity is confined to tune2025; the frozen holdout gate is
  the only verdict that counts, and it fires once.

## 7. Results (appended after the runs, 2026-07-14)

Harness integrity anchor: base calm-gated v4.18 on tune reproduced H3
exactly (n=50, −0.1157% vs recorded −0.116%). One clean run; no re-runs.

**H_A (calm retrain + Optuna, 40/40 trials, 36 cleared floors):**
best min-half tune net expectancy across the ENTIRE 8-dim space =
**−0.1201%/trade** (trial 3: depth 4, 60 trees, lr 0.024, short_q 0.02;
median cleared trial −0.20%). Not one of 40 hyperparameter configurations
was positive on the selection window. Training on 315,569 calm bars
(pos_rate 0.511) does not create signal the full-data fits lacked —
**the deficit is information, not hyperparameters.** H_A never reached
the holdout.

**H_B (logit meta-acceptance, 178 train trades, pos_rate 0.365):**
cutoff 0.45 cleared tune floors with min-half **+0.0667%** (n=16, both
halves positive) → selected as THE candidate. ONE holdout2026 run:
**n=26, net exp −0.1752%/trade, net PnL −$16.16, PF 0.347, DD −1.69%**
(meta blocked 261 calm entry signals). The tune-positive score inverted
out-of-sample — worse per-trade than even unfiltered v4.18 (−0.085%) and
far worse than plain calm-gating (−0.034%): the 3-feature logit fit noise
in 178 samples, exactly the small-sample failure mode flagged in §1.

**Gate: FAILED** (`holdout_net_exp_pos` false). **v4.18.3 is not created**
(fourth consecutive refusal). No weights persisted (H_A models lived only
in memory; H_B was a logit over the incumbent). Ledger row
`v4.18.3-h4-calm-fit`; artifacts `reports/eval/v4183/{summary_calm.json,
optuna_calm.db,optuna_trials_calm.jsonl,tune_calm_run.log}`.

## 8. What this closes

Conditional *fitting* — the hypothesis H3 pointed at — is now
evidence-closed in both credible forms: retraining on calm bars (H_A,
exhaustively across hyperparameters) and meta-labeling calm entries (H_B,
which overfits at the trade counts this strategy generates and inverted
OOS). Combined with H1–H3 this closes every family reachable with the
present data: decision-layer tuning, horizon retrains, execution tactics
(bounded above), regime gating, conditional fitting, meta-labeling. The
honest conclusion: **no improvement is currently credible from the
existing 1m-kline feature universe.** The binding constraint is upstream —
either new information (data the features have never seen) or a cost
structure change; more model-fitting on this data is p-hacking with extra
steps. The toxic-vol filter (H3, default OFF) remains the only
operator-facing improvement on record.
