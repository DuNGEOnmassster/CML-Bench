"""Contract C31 (scale-up audit): stratified sample, targeted pool, global stop rule, orchestrator pause.

Audit verdicts are one JSON line per audited item (written by the evaluator / reviewers):
  {"item_id", "batch_id", "summary_sha1", "auditor", "auditor_family", "audited_at",
   "major": int, "outside": int, "minor": int, "claims": int}
`summary_sha1` = sha1 of the audited summary text; a verdict only acts on that exact abstract.

Rules (evaluator-repilot-report.md 5):
  sample     per block of 10 consecutive main batches, the 2 batches with the lowest sha1(seed+batch_id) are sampled and
             the item with the lowest sha1(seed+item_id) of each is audited (2%; independent of merge order)
  targeted   every merged item routed by merge gate G8 is audited in full, outside the 2% sample
  revoke     a batch with any major or outside verdict on its current abstracts leaves data/ and is rewritten
  stop       issuing stops for everyone when any 50 consecutive verdicts hold >= 2 major/outside, or when, after >= 60
             verdicts, the Wilson 95% upper bound of the major rate exceeds 3%
  pause      an orchestrator with >= 2 revoked batches among its last 20 audited batches is paused
"""
from __future__ import annotations

import hashlib
import math
import re

SEED = "c31-20261009"
WINDOW, WINDOW_MAX_BAD = 50, 2
MIN_FOR_WILSON, MAX_MAJOR_UPPER = 60, 0.03
PAUSE_LAST, PAUSE_REVOKED = 20, 2


def _h(s: str) -> str:
    return hashlib.sha1(f"{SEED}:{s}".encode()).hexdigest()


def batch_number(batch_id: str) -> int | None:
    m = re.search(r"-b(\d+)$", batch_id)
    return int(m.group(1)) if m and "-hold-" not in batch_id else None


def sample_batches(batch_ids: list[str]) -> set[str]:
    blocks: dict[int, list[str]] = {}
    for b in batch_ids:
        n = batch_number(b)
        if n is not None:
            blocks.setdefault((n - 1) // 10, []).append(b)
    return {b for bs in blocks.values() for b in sorted(bs, key=_h)[:2]}


def sample_item(item_ids: list[str]) -> str:
    return min(item_ids, key=_h)


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 1.0
    p = k / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre + margin) / (1 + z * z / n)


def stop_rule(verdicts: list[dict]) -> dict:
    """verdicts in audit order. Returns {"stop": bool, "reason": str, ...}."""
    bad = [1 if (v.get("major", 0) or v.get("outside", 0)) else 0 for v in verdicts]
    worst = max((sum(bad[i : i + WINDOW]) for i in range(max(1, len(bad) - WINDOW + 1))), default=0)
    majors = sum(1 for v in verdicts if v.get("major", 0))
    n = len(verdicts)
    upper = wilson_upper(majors, n)
    reasons = []
    if worst >= WINDOW_MAX_BAD:
        reasons.append(f"{worst} major/outside within {WINDOW} consecutive audited items")
    if n >= MIN_FOR_WILSON and upper > MAX_MAJOR_UPPER:
        reasons.append(f"major-rate Wilson 95% upper bound {upper:.3f} > {MAX_MAJOR_UPPER} after {n} audited items")
    return {"stop": bool(reasons), "reason": "; ".join(reasons), "audited": n, "major": majors,
            "outside": sum(1 for v in verdicts if v.get("outside", 0)),
            "minor": sum(v.get("minor", 0) for v in verdicts), "claims": sum(v.get("claims", 0) for v in verdicts),
            "worst_window_bad": worst, "major_rate_upper95": round(upper, 4) if n else None}


def orchestrator_of(batch_id: str, ranges: list[dict]) -> str | None:
    n = batch_number(batch_id)
    for r in ranges:
        if n is not None and r["first"] <= n <= r["last"]:
            return r["name"]
    return None


def paused_orchestrators(audited_batches: list[tuple[str, bool]], ranges: list[dict]) -> list[str]:
    """audited_batches: (batch_id, revoked) in audit order, one entry per audited batch."""
    per: dict[str, list[bool]] = {}
    for b, revoked in audited_batches:
        o = orchestrator_of(b, ranges)
        if o:
            per.setdefault(o, []).append(revoked)
    return sorted(o for o, flags in per.items() if sum(flags[-PAUSE_LAST:]) >= PAUSE_REVOKED)
