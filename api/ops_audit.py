"""Read side of the append-only audit trails (RESEARCH_TOOLS.md §2).

One generic JSONL reader serves both audit files:
- data/execution/audit.jsonl  — control-plane changes (execution_control.py)
- data/research/audit.jsonl   — strategy-lab / simulation events

The files are human-action-scale (one row per operator action or research
run), so the reader loads the file, filters, and paginates by reverse index.
# ponytail: whole-file read — revisit with byte-offset cursors if a file ever
# grows past ~10 MB, which at current write rates is years away.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_LIMIT = 200


def _load_rows(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    try:
        with open(path, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a torn write; the trail keeps going
                row["_i"] = i
                rows.append(row)
    except OSError as e:
        logger.warning(f"audit read failed for {path}: {e}")
    return rows


def _matches(row: Dict, actor: Optional[str], via: Optional[str],
             outcome: Optional[str], event: Optional[str],
             since: Optional[str], until: Optional[str]) -> bool:
    if actor and row.get("actor") != actor:
        return False
    if via and row.get("via") != via:
        return False
    if outcome and row.get("outcome") != outcome:
        return False
    if event and row.get("event") != event:
        return False
    ts = str(row.get("ts", ""))
    # ISO-8601 UTC timestamps compare correctly as strings
    if since and ts < since:
        return False
    if until and ts > until:
        return False
    return True


def read_audit(path: Path, limit: int = 50, cursor: Optional[str] = None,
               actor: Optional[str] = None, via: Optional[str] = None,
               outcome: Optional[str] = None, event: Optional[str] = None,
               since: Optional[str] = None, until: Optional[str] = None) -> Dict:
    """Filtered, newest-first page of audit rows.

    ``cursor`` is the offset into the filtered, newest-first sequence — pass
    back ``next_cursor`` to continue. Rows keep their original fields plus a
    stable ``id`` (file line index).
    """
    limit = max(1, min(int(limit), MAX_LIMIT))
    matched = [r for r in _load_rows(path)
               if _matches(r, actor, via, outcome, event, since, until)]
    matched.reverse()  # newest first (file is append-only)
    try:
        start = max(0, int(cursor)) if cursor else 0
    except ValueError:
        start = 0
    page = matched[start:start + limit]
    rows = []
    for r in page:
        out = {k: v for k, v in r.items() if k != "_i"}
        out["id"] = str(r["_i"])
        rows.append(out)
    next_cursor = str(start + limit) if start + limit < len(matched) else None
    return {"rows": rows, "next_cursor": next_cursor, "total_matched": len(matched)}


def audit_summary(path: Path) -> Dict:
    """Counts by actor/surface/outcome/event + conflict rate, for the header
    chips of an audit view."""
    rows = _load_rows(path)
    by_actor: Dict[str, int] = {}
    by_via: Dict[str, int] = {}
    by_outcome: Dict[str, int] = {}
    by_event: Dict[str, int] = {}
    for r in rows:
        for key, bucket in (("actor", by_actor), ("via", by_via),
                            ("outcome", by_outcome), ("event", by_event)):
            v = r.get(key)
            if v is not None:
                bucket[str(v)] = bucket.get(str(v), 0) + 1
    applied = by_outcome.get("applied", 0)
    conflicts = by_outcome.get("conflict", 0)
    denom = applied + conflicts
    return {
        "total": len(rows),
        "by_actor": by_actor,
        "by_via": by_via,
        "by_outcome": by_outcome,
        "by_event": by_event,
        "conflict_rate": round(conflicts / denom, 4) if denom else None,
        "first_ts": rows[0].get("ts") if rows else None,
        "last_ts": rows[-1].get("ts") if rows else None,
    }
