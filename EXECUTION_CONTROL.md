# Execution Control Plane

Single backend-authoritative auto-execute control shared by Meridian and the
Vercel frontend. Written before implementation; describes current behavior,
risks, and the final design.

## Current behavior (before)

- **Truth already lives in the backend**: `BrokerConfig.auto_execute`, persisted
  atomically at `data/live/broker_config.json`, survives restarts. Good base.
- **Reads**: `GET /broker/config`, `/status`, and WS payloads all serve
  `_broker_summary()`. Meridian polls `/status`; the Vercel frontend fetches on
  mount and merges WS pushes.
- **Writes**: `POST /broker/config` partial update. Both UIs compute the new
  value client-side as `!current` from their own render state — **toggle
  semantics** — and the Vercel frontend applies an **optimistic local update**
  before the server confirms.
- **No version, no audit, no writer identity.** Any locally-run copy of this
  backend with testnet keys and `auto_execute: true` in its own config file
  would place demo orders — a second writer is one env file away.

## Risks

1. **Double-toggle race** — two surfaces clicking near-simultaneously send
   inverted stale values; last write wins, final state is whatever landed last,
   not what either operator intended.
2. **Stale-tab flip** — a tab holding old state inverts the wrong way.
3. **Optimistic drift** — Vercel UI can show ON while backend truth is OFF.
4. **No audit** — "who enabled auto-exec and when" is unanswerable.
5. **No writer identity** — a non-Railway instance can believe it may execute.

## Final design

### Resource

- `GET /execution/control` → `{ auto_execute, version, updated_at, updated_by,
  updated_via, request_id, writer: { backend, instance_id, is_writer, role },
  mode }`
- `PATCH /execution/control` body `{ auto_execute: <bool, required>,
  expected_version?: int, request_id?: str }`, surface identified by the
  `X-Control-Surface` header (`meridian` | `vercel` | anything else → `api`).

Semantics:
- **Explicit SET only.** The desired final value is in the request; the server
  never inverts anything.
- **Idempotent**: `desired == current` → 200, no version bump, no audit spam.
  Retries and duplicate submissions converge. `request_id` is recorded in the
  audit trail for traceability (set semantics + versioning make a dedupe cache
  unnecessary).
- **Optimistic concurrency**: `expected_version != current version` → **409**
  with the authoritative current state embedded, plus an audited conflict row.
  (Integer version compare-on-write — the standard OCC pattern, cf. Kubernetes
  `resourceVersion`, RFC 9110 preconditions.)

### One bit, one store

The value stays `BrokerConfig.auto_execute` in `data/live/broker_config.json`
— the execution loop keeps reading the exact field it always has. The control
plane adds *metadata*, not a second copy of the bit:

- `data/execution/control.json` — `{ version, updated_at, updated_by,
  updated_via, request_id }`, atomic writes.
- `data/execution/audit.jsonl` — append-only: ts, prev, new, actor (JWT
  username), via, instance, request_id, version, outcome
  (`applied` | `conflict`).

Every mutation of the bit — new PATCH endpoint **and** the legacy
`POST /broker/config` path (`via: "legacy-api"`) — goes through one
lock-guarded setter, so no path can bypass versioning or audit.

### Writer identity (one execution instance)

- `writer_backend` = `railway` when `RAILWAY_ENVIRONMENT` is present, else
  `local`. `instance_id` = `RAILWAY_REPLICA_ID` or `host-pid`.
- `is_writer` defaults to "am I on Railway"; `EXECUTION_WRITER=1/0` overrides
  explicitly.
- `_auto_exec_eligible` additionally requires **writer status for demo mode**:
  a non-writer instance never places broker orders merely because
  `auto_execute` is true. **Paper mode stays allowed everywhere** — simulated
  fills are the rehearsal path and touch no broker.
- Writer identity is surfaced in `/execution/control`, `/status`, and WS
  broker summaries.

### Frontends (two views over one resource)

Both Meridian and the Vercel frontend:
- render **backend truth only** (Vercel's optimistic update for auto-execute is
  removed; pending state shows on the switch instead),
- send `PATCH { auto_execute: <desired>, expected_version }` with their
  surface header,
- refresh from the response after every write,
- on 409 show "changed elsewhere — refreshed to latest value" and adopt the
  embedded current state,
- poll the resource every 5 s so both surfaces converge without a reload,
- show the shared control card: value, controlled-by backend, writer instance,
  last changed at/by/via, a warning when the connected backend is **not** the
  execution writer, and a warning when connected to a non-canonical backend.

Non-writer surfaces keep the switch enabled — flipping it is safe by
construction (the writer gate blocks demo orders on non-writers; paper
rehearsal remains useful) and the labeling makes scope explicit.

## Deployment status

**Deployed to Railway 2026-07-14** (`b184ef5`, ~3 min auto-deploy). Verified
live: `/execution/control` auth-gated (401 pre-login); Railway self-identifies
as writer (`backend: railway, is_writer: true`, replica UUID as instance id —
zero env configuration needed); `auto_execute: true` survived the deploy via
`data/live/broker_config.json`. Cross-surface sync, the simultaneous-click
race (exactly one version bump, loser 409s and refreshes), opposing-intent
conflicts, and full actor/surface attribution (`adm1nFYP via meridian/vercel`)
all verified in Chrome against production. A local instance confirmed as
`observer` — it manages only its own local state and can never place demo
orders. Note: the audit trail lives in `data/execution/audit.jsonl` on the
container; there is no read endpoint yet, so production audit rows are
verifiable only indirectly (metadata transitions + unit-tested write path).

## Reading the trail

`GET /execution/audit` (filters: actor/via/outcome/event/since/until, cursor
pagination) and `GET /execution/audit/summary` expose this file read-only,
behind the same auth as the control endpoints. Meridian's **Ops module**
renders it as an investigative timeline — the control card's "last change
by …" line links straight into it. Design: RESEARCH_TOOLS.md §2. Rejected
stale writes (409s) appear as `conflict` rows: that is the race protection
working, not an error.

## Deliberately not built

- No idempotency-key dedupe cache (SET + version already make retries safe).
- No distributed lock/leader election — Railway runs one replica; the writer
  flag is environment-derived, not negotiated.
- No per-field ETag machinery on other broker config fields — only
  auto-execute is a shared operational control.
