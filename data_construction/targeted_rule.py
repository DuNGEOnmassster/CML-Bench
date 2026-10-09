"""C31 targeted-pool budget rule (G8). Deterministic: anyone can recompute the audit set from targeted.jsonl + writer notes.

Tier A  always: title_mention, ungrounded_names, and writer notes that question the film's identity (IDENTITY_RE).
Tier A4 coverage-only triggers (top_speaker_missing, last_third_not_covered, no other A trigger): audit when h < 1/4
        (coordinator approval 2026-10-09 18:58 after 50 such audits found nothing; revert to Tier A if a sampled
        coverage item ever finds a missing ending).
Tier B  writer_source_issue only, note plausibly affects the abstract (IMPACT_RE: missing/omitted/truncated content,
        scene order or merged scenes, title-page/credits/epigraph scenes, actor names, name inconsistencies, a writer's
        attribution guess): audit when h(item) < 1/3, at most 1 per batch.
Tier C  writer_source_issue only, tagging/OCR noise: audit when h(item) < 1/20 (calibration that noise notes hide nothing).
h(item) = int(sha1("c31-g8:" + item_id)[:8], 16) / 2**32.
Budget guard: if the selected targeted set exceeds 8% of merged items, Tier C is suspended; if it still exceeds 8%,
Tier B drops to h < 1/8. Tier A is never cut.
The evaluator recomputes the selection with its own copy (evaluator/fanout_audit/targeted_rule.py): keep the two identical.
"""
import hashlib, re

IDENTITY_RE = re.compile(r"wrong (film|movie|script)|unrelated (film|movie|script|story)|different (film|movie|script|story)|"
                         r"not (the|this) (film|movie)|another (film|movie)'?s? script|title.{0,20}mismatch|doesn'?t match the title", re.I)
IMPACT_RE = re.compile(r"missing|omitted|omit\b|deleted|cut off|truncat|separate document|ends (with|mid|abruptly)|jumps|"
                       r"out of order|belong(s)? (after|before)|misplaced|no scene heading|merge[sd]? .{0,30}(scene|quarters|location)|"
                       r"title[- ]page|title card|credits|epigraph|boilerplate|logo|\bactor\b|probably|likely .{0,20}(line|speaker|reply)|"
                       r"unclear who|ambiguous|never named|inconsistent|spelled (both|inconsistently)|instead of|called .{0,40} (then|but|in dialogue)|"
                       r"\bbut the character is\b|mislabeled as|labeled both|unexplained|mismatch", re.I)
FRAC_B, FRAC_C, CAP_B_PER_BATCH = 1 / 3, 1 / 20, 1
FRAC_COV = 1 / 4
COVERAGE = {"top_speaker_missing", "last_third_not_covered"}


def h(item_id: str) -> float:
    return int(hashlib.sha1(f"c31-g8:{item_id}".encode()).hexdigest()[:8], 16) / 2 ** 32


def tier(reasons: list[str], note: str) -> str:
    strong = [r for r in reasons if r not in COVERAGE and r != "writer_source_issue"]
    if strong or (note and IDENTITY_RE.search(note)):
        return "A"
    if any(r in COVERAGE for r in reasons):
        return "A4"
    return "B" if note and IMPACT_RE.search(note) else "C"


BUDGET = 0.08


def select(targeted: list[dict], notes: dict[str, str], merged_items: int | None = None) -> list[dict]:
    """targeted: rows of audit/<slug>/targeted.jsonl (item_id, batch_id, reasons, summary_sha1).
    Returns the rows to audit, each with "tier"."""
    out = _select(targeted, notes, FRAC_B, FRAC_C)
    if merged_items and len(out) > BUDGET * merged_items:
        out = _select(targeted, notes, FRAC_B, 0.0)
        if len(out) > BUDGET * merged_items:
            out = _select(targeted, notes, 1 / 8, 0.0)
    return out


def _select(targeted, notes, frac_b, frac_c):
    out, b_used = [], {}
    for t in sorted(targeted, key=lambda r: (r["batch_id"], h(r["item_id"]))):
        k = tier(t.get("reasons", []), notes.get(t["item_id"], ""))
        if k == "A":
            out.append({**t, "tier": "A"})
        elif k == "A4":
            if h(t["item_id"]) < FRAC_COV:
                out.append({**t, "tier": "A4"})
        elif k == "B" and h(t["item_id"]) < frac_b and b_used.get(t["batch_id"], 0) < CAP_B_PER_BATCH:
            b_used[t["batch_id"]] = b_used.get(t["batch_id"], 0) + 1
            out.append({**t, "tier": "B"})
        elif k == "C" and h(t["item_id"]) < frac_c:
            out.append({**t, "tier": "C"})
    return out
