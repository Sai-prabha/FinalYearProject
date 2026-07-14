# v4.18.3 — Candidate Design & Pre-Registered Experiment (2026-07-14)

> **OUTCOME (2026-07-14, same day): GATE FAILED — twice. v4.18.3 is NOT
> activated.** Both pre-registered retrain hypotheses were evaluated and
> rejected by the gate in §3 (results in §7 below; ledger:
> `reports/experiments.jsonl`, artifacts: `reports/eval/v4183/`). The
> infrastructure built for the candidate (dual-model shadow slot,
> net-of-cost shadow accounting, this experiment protocol) shipped anyway
> and applies to every future candidate. v4.18 remains the incumbent.

> Status at creation: **HYPOTHESIS — not validated.** This document is written
> BEFORE training/evaluation so the acceptance gate cannot drift to fit the
> result. Companion evidence: `reports/PROGRESS.md` (the ratchet),
> `reports/eval/*`, `TRADE_RECONCILIATION.md`.

## 1. Why v4.18.3 is a retrain, not a config tweak

The requested framing was "improve gains via balance-based sizing and dynamic
TP/SL." The evidence rejects that framing:

- Broker sizing is **already** balance-based (`sizing_mode: balance`,
  wallet-equity × risk_fraction, step/min-notional floored). Sizing is not the
  bottleneck: live history (n=91) shows gross expectancy **−0.0009%/trade**
  against a **0.18%** taker round-trip hurdle. Scaling size scales the bleed.
- Dynamic TP/SL, trailing, break-even, time filters, loss-streak gates,
  drawdown scaling **all already exist** (v4.16/17/18) and were swept
  exhaustively with the fee-aware harness: every decision-layer operating
  point of the fixed v4.14 weights is net-negative on at least one validation
  window (`reports/eval/sweep_v418.csv`; maker-fee best case still fails
  tune2025). A v4.18.2 was already refused on this evidence (2026-07-10).
- The one structural lead: the model was trained on **10-bar labels**
  (`k_ahead: 10`) but the only OOS-persistent signal structure lives at the
  **240-bar (4h)** horizon — and the live ledger shows churn (22-min median
  hold) whose costs eat everything.

**Hypothesis (the only one this candidate tests):** retraining the same
architecture, same 50 causal features, same data span, with **240-bar
horizon labels** produces a probability stream whose tradeable edge at 4h
holds clears real taker costs, where the 10-bar model's could not.

## 2. Pre-registered training spec (no post-hoc tuning)

| Item | Choice | Rationale |
|---|---|---|
| Features | the identical 50 live features | isolates the label-horizon variable; live parity guaranteed |
| Model | XGBClassifier, **exact v4.14 hyperparams** (depth 3, 35 trees, lr 0.04, heavy regularization) | no architecture search — one variable changes |
| Train window | 2025-02-01 → 2025-10-01 (same span as v4.14) | regime-comparable to the original |
| Label | sign of 240-bar forward log ratio return; rows within 240 bars of window end dropped | the validated horizon; no deadband (no extra knob) |
| Decision layer | v4.18 shape (tail gate + `max_hold_bars=240` + wide guardrails, ≤1× sizing) with thresholds re-derived on **tune2025 only** (2025-10-01→12-15), then frozen | thresholds must match the new proba distribution; selection isolated from the final test |
| Final test | **one** evaluation on holdout2026 (2025-12-15→2026-07-01) at 4.5 bps/side taker | the ratchet window |

### Ablations (all fee-aware, all three windows where applicable)

1. Old weights + v4.18 layer (exists — the incumbent).
2. **New weights + v4.18 layer untouched** — isolates the weights effect.
3. New weights + tune2025-derived thresholds — the actual candidate.
4. Sizing ablation: none. Sizing is already balance-based and evidence says
   it is not the lever; changing it here would blur attribution.

## 3. Pre-registered acceptance gate (the ratchet, extended)

v4.18.3 SHIPS TO SHADOW only if, at taker fees (4.5 bps/side × 4 fills):

1. holdout2026 net expectancy **> 0** and net PnL **> v4.18's −$22**,
2. tune2025 net expectancy **> v4.18's −0.176%**,
3. n ≥ 20 trades on holdout2026 (no small-n edge claims),
4. max DD on holdout no worse than 2× v4.18's (−3.7% → floor −7.4%),
5. no window shows the strategy relying on a single trade for its sign
   (top-1 trade removed ⇒ expectancy sign unchanged).

If any check fails: **v4.18.3 is not activated**; the evidence is filed in
`reports/eval/` + the experiment ledger, v4.18 stays, and the next hypothesis
(maker-entry fill model, longer retraining span) goes through this same gate.
A failed gate is a successful experiment — it is the system refusing to
manufacture improvement.

## 4. Runtime mode decision (Phase 6 — challenged as required)

**Rejected — same-account dual writing** (user's suggested "also place trades
using the same balance/API"): both versions trade the SAME two symbols
(BTCUSDT/ETHUSDT) on a one-way-mode USDM account. Positions net at the
exchange: v4.18.3 opening SHORT while v4.15 holds LONG *closes v4.15's
position on the broker*. Fills, fees, funding and margin become jointly
caused and unattributable; reconciliation (which keys fills → orders →
intent) would classify the cross-contamination as breaks. This is
methodologically destroyed before it starts — and it would also create a
second writer, violating the control-plane invariant.

**Rejected for now — split sub-account / second keys (Option C):** clean
attribution, but at ~30 trades/quarter the live sample cannot reach decision
power for months; testnet Binance does not offer sub-accounts, so this means
a second full account + second writer instance + doubled ops surface, for a
sample that decides nothing the harness has not already decided. Documented
as the **pre-promotion step**: after v4.18.3 passes the harness gate AND
accumulates a clean shadow record, stage it on its own testnet account/keys
as the final rehearsal before any promotion decision.

**Chosen — Option A+B hybrid: shadow slot with broker-realistic accounting.**
- v4.18.3 runs in the existing ONE shadow slot (evaluation-only, structurally
  incapable of broker writes) — extended to support **its own model weights**
  (dual-model shadow: the slot gets its own proba stream from the same
  feature vector; primary stream untouched).
- Shadow trades are recorded with **net-of-cost fields** (taker fee model +
  observed slippage from reconciliation once demo fills exist), so the
  Meridian comparison is broker-realistic, not internal-ledger fantasy.
- Railway stays the sole writer; v4.15 execution and attribution are
  untouched; reconciliation continues to verify the baseline only.

## 5. Comparison metrics (what Meridian shows)

Baseline-vs-candidate on: net expectancy/trade (after modeled costs), net
PnL, win rate, W/L ratio, max DD, trade count, cost drag share, and the
promotion-readiness checks. Any "improvement" that arrives with larger DD or
size is flagged as risk-bought, not edge — DD and per-trade risk are
first-class columns, not footnotes.

## 6. Results (recorded after the pre-registered runs — 2026-07-14)

**Hypothesis #1 — 240-bar labels, train Feb→Oct 2025** (`scripts/train_v4183.py`):
in-sample AUC 0.610 (last-15% 0.630); tune2025 selection picked short-only at
the 2% proba tail (0.4489). Tune beat the incumbent (gate check passed) but
**holdout2026: n=36, net exp −0.243%/trade, PF 0.42, PnL −$30.52** —
`holdout_net_exp_pos` and `holdout_pnl_beats_v418` both failed. The 2026
regime that inverted the raw tail edge also defeats the retrained weights.

**Hypothesis #2 — same labels, train through 2025-12** (`train_v4183_h2.py`,
deterministic threshold rule, zero selection freedom, one holdout run):
**holdout2026: n=104, net exp −0.201%/trade, PF 0.36, PnL −$57.88** — failed.
More recent training data produced more trades, not more edge.

**Verdict:** with this feature set and architecture, no configuration —
including horizon-aligned retrained weights — clears real taker costs on the
2026 holdout. The rejected weights were deleted (`models/v4_18_3/` removed)
so they can never be loaded by the dual-model slot. What would change this
verdict: a genuinely new feature/architecture family, or an honest
maker-fill model (limit-through fill simulation, not a fee discount) —
each through this same protocol. Sizing changes remain off the table until
an edge exists: sizing scales edge, and scales its absence.

## 7. Iteration loop (Phase 8)

Every hypothesis gets a row in `reports/experiments.jsonl`:
`{ts, id, hypothesis, spec, windows, net_metrics, gate_result, verdict,
artifacts}` — appended by the evaluation script, never edited. The loop is:
propose → pre-register spec + gate here → run harness → append row →
keep/reject. The gate always includes the ratchet (beat incumbent NET on
holdout) so risk-inflation cannot masquerade as improvement.
