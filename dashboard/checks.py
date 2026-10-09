"""Per-item contract checks for the dashboard, reusing the pipeline's own checkers.

The Space ships copies of the pipeline modules in `pipeline/` (see deploy_space.py); a repo checkout
imports them from `data_construction/`. No Gradio imports here: worker processes import this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _cand in (_HERE / "pipeline", _HERE.parent / "data_construction"):
    if (_cand / "cml_format.py").is_file():
        sys.path.insert(0, str(_cand))
        break

from build_segments import CONFIG, norm_title, overlap, shingles, words_of  # noqa: E402
from check_abstracts import check_one  # noqa: E402
from cml_format import parse_script, segment_stats, speaker_name, validate_cml  # noqa: E402
from contract_checks import FIRST, GT_MEDIAN_TOKENS, LLM_RESIDUE_RE, PROVENANCE  # noqa: E402
from make_abstract_batches import target_center, target_words  # noqa: E402

# Same pattern and duplicate-scene rule as contract_checks.py C12/C13 (defined inline there).
JUNK_RE = re.compile(r"\ufffd|[\x00-\x08\x0b-\x1f]|&amp;amp;|>\(?(CONTINUED|OMITTED)\)?<|>\d+\.?<|-[LR][RSC]B-|\*")
NAME_RE = re.compile(r"^.+_\d{4}$")
IMDB_RE = re.compile(r"^tt\d{7,8}$")
# Evaluator-proposed v2 checks (evaluator-report.md §1, patterns from evaluator/residue.py).
CONTD_RE = re.compile(r"\(CONT'D\.\)|CONT 'D")
QUOTE_INNER_RE = re.compile(r'(?:^|[\s>])" [A-Za-z]|[a-z.!?,] "(?=[\s<])')
REMAKE_TYPES = {"remake", "same_story", "same-story", "reboot_same_story"}


@dataclass(frozen=True)
class Assertion:
    cid: str
    group: str
    text: str
    bar: str
    column: str | None  # per-item boolean column in the items frame, None = release-level / audit only


GROUPS = {
    "A": "Schema & provenance",
    "B": "Content format",
    "C": "Dedupe & leakage",
    "D": "Abstracts (automated)",
    "E": "Abstracts (evaluator audit)",
    "F": "Process & release",
    "V": "Proposed for contract v2 (evaluator report, not adopted yet)",
}

CONTRACT = [
    Assertion("C01", "A", "First four fields are movie_name, imdb_id, script_segment, summary (non-empty)", "100%", "c01"),
    Assertion("C02", "A", "movie_name is Title_YYYY and imdb_id is tt + 7-8 digits", "100%", "c02"),
    Assertion("C03", "A", "Provenance fields present (ids, source, split, url, file, scene range, sha1, prompt, author)", "100%", "c03"),
    Assertion("C04", "A", "item_id and content_sha1 unique; content_sha1 = sha1(script_segment)", "100%", "c04"),
    Assertion("C05", "A", "20 random items re-cut from the source byte-for-byte", "20/20", None),
    Assertion("C06", "A", "info.json has the gt_100_info layout and agrees with the data", "agree", None),
    Assertion("C07", "B", "Valid CML: <script> root, <scene> children, 6 element tags, none empty or nested", "100%", "c07"),
    Assertion("C08", "B", "No LLM residue; starts with <script>, ends with </script>", "100%", "c08"),
    Assertion("C09", "B", "2,000-10,000 cl100k tokens; median within 20% of GT (5,702)", "100%", "c09"),
    Assertion("C10", "B", "12-24 scenes; at least 80% of items at 15-20", "100%", "c10"),
    Assertion("C11", "B", ">= 20 dialogue turns, >= 2 speakers, dialogue share 0.10-0.85", "100%", "c11"),
    Assertion("C12", "B", "<= 2 tokenization artefacts per 1k words; no LRB, *, U+FFFD, CONTINUED, page numbers", ">= 99%", "c12"),
    Assertion("C13", "B", "OCR garble <= 0.5%, mis-tagged speakers <= 5%, headings >= 0.7, no element > 3k chars, no duplicate scenes", "100%", "c13"),
    Assertion("C14", "B", "English: >= 97% ASCII letters in content and abstract", "100%", "c14"),
    Assertion("C15", "C", "Not one of the 100 GT movies (IMDb id or normalized title)", "100%", "c15"),
    Assertion("C16", "C", "13-gram overlap with the union of GT segments <= 2%", "100%", "c16"),
    Assertion("C17", "C", "No duplicate screenplay under another imdb_id (> 30% shared 13-grams)", "100%", None),
    Assertion("C18", "C", "Scene ranges are disjoint within a movie", "100%", "c18"),
    Assertion("C19", "D", "Abstract hard checks: 90-300 words, <= 3 paragraphs, no markdown/meta, grounded names", "100%", "c19"),
    Assertion("C20", "D", "Abstract inside its per-item target length; mean 130-200 words", ">= 85%", "c20"),
    Assertion("C21", "D", "Top speaker named (>= 95%); >= 2 of the top 3 named (>= 85%)", ">= 95%", "c21"),
    Assertion("C22", "D", "All three thirds of the segment covered (>= 90%); last third (>= 95%)", ">= 90%", "c22"),
    Assertion("C23", "D", "Lexical grounding >= 0.35 per item; mean >= 0.55", "100%", "c23"),
    Assertion("C24", "D", "No two abstracts share their first 8 words; no opening 3-gram in > 10%", "100%", "c24"),
    Assertion("C25", "E", "Faithfulness: 0 major hallucinations in >= 10 audited items", "audit", None),
    Assertion("C26", "E", "Coverage: opening, main turns and ending covered (mean >= 4/5)", "audit", None),
    Assertion("C27", "E", "No outside knowledge (cast, crew, title, reception, later events)", "audit", None),
    Assertion("C28", "E", "Blind pairwise vs GT: wins + ties >= 50%", "audit", None),
    Assertion("C29", "F", "State recoverable; rebuild reproduces the same sha1 set", "manual", None),
    Assertion("C30", "F", "No screenplay text in git; HF upload only to a private repo", "manual", None),
    Assertion("C10′", "V", "100% at 12-24 scenes; 15-20 share >= 0.80 judged only at n >= 500, else one-sided binomial test", "p ≥ 0.05", None),
    Assertion("C12b", "V", "No stray backslash, no (CONT'D.) / CONT 'D, <= 1 quote-inner space per 1k words", "100%", "c12b"),
    Assertion("C12c", "V", "No orphan speaker lines (a description that is just a speaker name of the segment)", "100%", "c12c"),
    Assertion("C13b", "V", "No speaker split by noise (e.g. \"BLAKE \\\" next to \"BLAKE\")", "100%", "c13b"),
    Assertion("C15b", "V", "No remake of a GT story; sequels and same-universe items carry gt_related and are outside eval_safe", "100%", "c15b"),
    Assertion("C20b", "V", "Abstract length shape: 35-65% below target_center, p25 <= 150 words, >= 60% single paragraph", "release", None),
]

ABSTRACT_COLUMNS = ("c19", "c20", "c21", "c22", "c23", "c24")
PROPOSED_COLUMNS = ("c12b", "c12c", "c13b", "c15b")
COLUMN_CID = {a.column: a.cid for a in CONTRACT if a.column}


def ascii_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isascii() for c in letters) / max(1, len(letters))


def has_dup_scene(scenes) -> bool:
    bodies = ["\n".join(t for tag, t in sc.elements if tag != "stage_direction") for sc in scenes]
    bodies = [b for b in bodies if len(b) >= 200]
    return len(bodies) != len(set(bodies))


def gt_relation(rec: dict) -> tuple[bool | None, str | None, str | None]:
    """(related, type, gt_movie) from the optional `gt_related` field; related is None when the field is absent."""
    if "gt_related" not in rec:
        return None, None, None
    v = rec["gt_related"]
    if isinstance(v, list):
        v = v[0] if v else None
    if not v:
        return False, None, None
    if isinstance(v, dict):
        kind = v.get("type") or v.get("relation") or v.get("kind") or "related"
        movie = v.get("gt_movie") or v.get("gt_movie_name") or v.get("movie") or v.get("gt_imdb_id")
        return True, str(kind), (str(movie) if movie else None)
    if isinstance(v, str):
        return True, v, rec.get("gt_related_movie")
    return True, str(rec.get("gt_related_type") or "related"), rec.get("gt_related_movie")


def metadata_checks(rec: dict, gt_ids: frozenset, gt_titles: frozenset) -> dict:
    """Checks that need only the record's fields (no content parsing)."""
    related, kind, _ = gt_relation(rec)
    first = list(rec)[:4]
    c01 = first == FIRST and all(isinstance(rec.get(k), str) and rec[k].strip() for k in FIRST)
    name, imdb = str(rec.get("movie_name") or ""), str(rec.get("imdb_id") or "")
    c03 = all(rec.get(k) not in (None, "") for k in PROVENANCE) and (
        isinstance(rec.get("scene_start"), int) and isinstance(rec.get("scene_end"), int) and rec["scene_start"] <= rec["scene_end"]
    )
    return {
        "c01": c01,
        "c02": bool(NAME_RE.match(name) and IMDB_RE.match(imdb)),
        "c03": c03,
        "c15": not (imdb in gt_ids or (name and norm_title(name) in gt_titles)),
        "c15b": None if related is None else not (related and str(kind).lower() in REMAKE_TYPES),
    }


def item_checks(content: str, summary: str, script_tokens: int, content_sha1: str | None, gt_shingles: set | None) -> dict:
    """Content and abstract checks for one item; mirrors contract_checks.py at item granularity."""
    scenes = parse_script(content, detok=False)
    st = segment_stats(scenes, "")  # content="" skips the token count; script_tokens comes from the record
    junk = bool(JUNK_RE.search(content))
    dup = has_dup_scene(scenes)
    ratio = st["dialogue_char_ratio"]
    elements = [(t, x) for s in scenes for t, x in s.elements]
    raw_speakers = {x.strip() for t, x in elements if t == "character"}
    speakers = {speaker_name(x) for x in raw_speakers}
    orphans = sum(1 for t, x in elements if t == "scene_description" and len(x.strip()) < 30
                  and x.strip() == x.strip().upper() and speaker_name(x) in speakers)
    by_key: dict[str, set] = {}
    for x in raw_speakers:
        by_key.setdefault(re.sub(r"[^A-Z0-9]", "", speaker_name(x)), set()).add(speaker_name(x))
    splits = sorted(n for names in by_key.values() if len(names) > 1 for n in names)
    n_words = max(1, len(content.split()))
    backslashes, contd = content.count("\\"), len(CONTD_RE.findall(content))
    quote_1k = round(1000 * len(QUOTE_INNER_RE.findall(content)) / n_words, 2)
    res = {
        "backslashes": backslashes,
        "contd": contd,
        "quote_inner_1k": quote_1k,
        "orphan_lines": orphans,
        "speaker_splits": splits,
        "c12b": backslashes == 0 and contd == 0 and quote_1k <= 1,
        "c12c": orphans == 0,
        "c13b": not splits,
        "dialogue_turns_chk": st["dialogue_turns"],
        "num_speakers": st["num_speakers"],
        "dialogue_ratio": ratio,
        "artefacts_1k": st["tokenization_artefacts_per_1k_words"],
        "garble_rate": st["garble_rate"],
        "bad_char_ratio": st["bad_character_tag_ratio"],
        "max_element_chars": st["max_element_chars"],
        "heading_ratio": st["heading_ratio"],
        "junk": junk,
        "dup_scene": dup,
        "sha_ok": (hashlib.sha1(content.encode()).hexdigest() == content_sha1) if content_sha1 else None,
        "c07": not validate_cml(content),
        "c08": content.startswith("<script>") and content.endswith("</script>") and not LLM_RESIDUE_RE.search(content),
        "c11": st["dialogue_turns"] >= 20 and st["num_speakers"] >= 2 and 0.10 <= ratio <= 0.85,
        "c12": st["tokenization_artefacts_per_1k_words"] <= 2 and not junk,
        "c13": (st["garble_rate"] <= 0.005 and st["bad_character_tag_ratio"] <= 0.05 and st["max_element_chars"] <= 3000
                and st["heading_ratio"] >= 0.7 and not dup),
        "ascii_content": round(ascii_share(content), 4),
        "gt_overlap": round(overlap(shingles(words_of(content), CONFIG["ngram"]), gt_shingles), 5) if gt_shingles is not None else None,
    }
    if res["gt_overlap"] is not None:
        res["c16"] = res["gt_overlap"] <= CONFIG["max_gt_segment_overlap"]
    text = summary.strip()
    res["c14"] = res["ascii_content"] >= 0.97 and (not text or ascii_share(text) >= 0.97)
    if text:
        lo, hi = target_words(script_tokens)
        c = check_one(text, content, [lo, hi])
        thirds = c.get("thirds_covered") or [False, False, False]
        res.update(
            {
                "abs_words": c["words"],
                "paragraphs": c.get("paragraphs"),
                "target_lo": lo,
                "target_hi": hi,
                "target_center": target_center(script_tokens),
                "hard": c["hard"],
                "soft": c["soft"],
                "top1": bool(c.get("top1_speaker_mentioned")),
                "top3": int(c.get("top3_speakers_mentioned") or 0),
                "grounding": c.get("lexical_grounding"),
                "thirds": thirds,
                "ungrounded": c.get("ungrounded_proper_nouns", []),
                "c19": not c["hard"],
                "c20": not c["hard"] and "word_count_outside_target" not in c["soft"],
                "c21": bool(c.get("top1_speaker_mentioned")),
                "c22": all(thirds),
                "c23": (c.get("lexical_grounding") or 0) >= 0.35,
            }
        )
    return res


# --- worker-process entry points -------------------------------------------------------------

_GT_SHINGLES: set | None = None


def init_worker(gt_shingles: set | None) -> None:
    global _GT_SHINGLES
    _GT_SHINGLES = gt_shingles


def check_chunk(task: tuple) -> list[tuple[str, dict]]:
    """task = (path or None, [(key, offset_or_content, script_tokens, content_sha1), ...])."""
    path, entries = task
    out = []
    fh = open(path, "rb") if path else None
    try:
        for key, ref, script_tokens, sha in entries:
            if fh is not None:
                fh.seek(ref)
                rec = json.loads(fh.readline())
                content, summary = record_content(rec), record_abstract(rec)
            else:
                content, summary = ref
            try:
                out.append((key, item_checks(content, summary, script_tokens, sha, _GT_SHINGLES)))
            except Exception as exc:  # one malformed item must not stall the batch
                out.append((key, {"check_error": f"{type(exc).__name__}: {exc}"}))
    finally:
        if fh is not None:
            fh.close()
    return out


def record_content(rec: dict) -> str:
    return rec.get("script_segment") or rec.get("content") or ""


def record_abstract(rec: dict) -> str:
    return rec.get("summary") or rec.get("abstract") or ""


def gt_shingle_set(gt_contents: list[str]) -> set:
    from cml_format import render

    sh: set = set()
    for c in gt_contents:
        sh |= shingles(words_of(render(parse_script(c))), CONFIG["ngram"])
    return sh


# --- release-level verdicts -------------------------------------------------------------------


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0


def release_verdicts(df, info: dict | None) -> dict[str, tuple[bool | None, str]]:
    """Release-level pass/fail for the automated assertions, computed from the items frame.

    `df` holds metadata + check columns; abstract checks only count items with an abstract.
    """
    import pandas as pd

    out: dict[str, tuple[bool | None, str]] = {}
    n = len(df)
    if n == 0:
        return out

    def rate(col, frame=df):
        s = frame[col].dropna() if col in frame else pd.Series(dtype=bool)
        return (float(s.astype(bool).mean()) if len(s) else None), len(s)

    for col in ("c01", "c02", "c03", "c07", "c08", "c11", "c13", "c14", "c15", "c16", "c18", "c12b", "c12c", "c13b", "c15b"):
        r, k = rate(col)
        if r is not None:
            out[COLUMN_CID[col]] = (r == 1.0 and k == n, f"{int(round(r * k))}/{k} items pass")
    if "gt_related" in df and df["gt_related"].notna().any():
        rel = df[df["gt_related"] == True]  # noqa: E712
        kinds = rel["gt_rel_type"].fillna("related").value_counts()
        detail = ", ".join(f"{k} {v}" for k, v in kinds.items()) or "no GT-related items"
        if "C15b" in out:
            out["C15b"] = (out["C15b"][0], f"{len(rel):,} GT-related items ({detail})")

    if "c04" in df:
        dup_ids = int(df["item_id"].duplicated(keep=False).sum())
        dup_sha = int(df["content_sha1"].dropna().duplicated(keep=False).sum())
        mism = int((df["sha_ok"] == False).sum()) if "sha_ok" in df else 0  # noqa: E712
        out["C04"] = (dup_ids == 0 and dup_sha == 0 and mism == 0, f"dup_ids={dup_ids}, dup_sha={dup_sha}, sha_mismatch={mism}")

    toks = df["script_tokens"].dropna()
    if len(toks):
        med = _median(toks.tolist())
        in_range = bool(((toks >= 2000) & (toks <= 10000)).all())
        out["C09"] = (in_range and abs(med - GT_MEDIAN_TOKENS) <= 0.2 * GT_MEDIAN_TOKENS,
                      f"min={int(toks.min()):,}, median={int(med):,}, max={int(toks.max()):,}")
    sc = df["num_scenes"].dropna()
    if len(sc):
        share = float(((sc >= 15) & (sc <= 20)).mean())
        hard_ok = bool(((sc >= 12) & (sc <= 24)).all())
        out["C10"] = (hard_ok and share >= 0.8, f"range {int(sc.min())}-{int(sc.max())}, share at 15-20 = {share:.1%}")
        m, k = len(sc), int(((sc >= 15) & (sc <= 20)).sum())
        if m >= 500:
            out["C10′"] = (hard_ok and share >= 0.8, f"n = {m:,} >= 500: share {share:.1%} vs 80%")
        else:
            p = sum(math.comb(m, i) * 0.8 ** i * 0.2 ** (m - i) for i in range(k + 1))
            out["C10′"] = (hard_ok and p >= 0.05, f"n = {m} < 500: P(X <= {k} | p = 0.8) = {p:.3f}")

    if "artefacts_1k" in df and df["artefacts_1k"].notna().any():
        art = sorted(df["artefacts_1k"].dropna().tolist())
        p99 = art[min(len(art) - 1, int(0.99 * len(art)))]
        junk = int(df["junk"].fillna(False).astype(bool).sum())
        out["C12"] = (p99 <= 2 and junk == 0, f"artefacts p99 = {p99}/1k words, junk items = {junk}")

    ab = df[df["has_abstract"] & df["c19"].notna()] if "c19" in df else df.iloc[0:0]
    m = len(ab)
    if m:
        hard = int((~ab["c19"].astype(bool)).sum())
        out["C19"] = (hard == 0, f"hard failures = {hard}/{m}")
        in_t = float(ab["c20"].astype(bool).mean())
        mean_w = float(ab["abs_words"].mean())
        out["C20"] = (in_t >= 0.85 and 130 <= mean_w <= 200, f"in target {in_t:.0%}, mean {mean_w:.0f} words")
        top1 = float(ab["top1"].astype(bool).mean())
        top3 = float((ab["top3"] >= 2).mean())
        out["C21"] = (top1 >= 0.95 and top3 >= 0.85, f"top-1 {top1:.0%}, >=2 of top-3 {top3:.0%}")
        all3 = float(ab["c22"].astype(bool).mean())
        last = float(ab["thirds"].map(lambda t: bool(t[2]) if isinstance(t, (list, tuple)) and len(t) == 3 else False).mean())
        out["C22"] = (all3 >= 0.90 and last >= 0.95, f"all thirds {all3:.0%}, last third {last:.0%}")
        g = ab["grounding"].astype(float)
        out["C23"] = (g.mean() >= 0.55 and g.min() >= 0.35, f"mean {g.mean():.3f}, min {g.min():.3f}")
        w = ab["abs_words"].astype(float)
        below = float((w < ab["target_center"].astype(float)).mean())
        p25 = float(w.quantile(0.25))
        single = float((ab["paragraphs"].astype(float) == 1).mean())
        out["C20b"] = (0.35 <= below <= 0.65 and p25 <= 150 and single >= 0.6,
                       f"below center {below:.0%}, p25 {p25:.0f} words, single paragraph {single:.0%}")
    with_abs = df[df["has_abstract"]]
    if len(with_abs):
        first8 = with_abs["summary"].map(lambda s: " ".join(s.split()[:8]).lower())
        open3 = with_abs["summary"].map(lambda s: " ".join(s.split()[:3]).lower())
        vc8, vc3 = first8.value_counts(), open3.value_counts()
        share3 = vc3.iloc[0] / len(with_abs)
        out["C24"] = (int(vc8.iloc[0]) == 1 and share3 <= 0.10,
                      f"max shared 8-word opening = {int(vc8.iloc[0])}, top 3-gram \u201c{vc3.index[0]}\u201d {share3:.0%}")

    if info and isinstance(info.get("individual_results"), list):
        ind = {x.get("item_id"): x for x in info["individual_results"]}
        keys_ok = all({"script_tokens", "summary_tokens", "tag_counts", "imdb_rating", "genres"} <= set(x) for x in ind.values())
        agree = all(ind.get(i, {}).get("script_tokens") == t for i, t in zip(df["item_id"], df["script_tokens"]))
        tot_ok = info.get("summary", {}).get("total_script_tokens") == int(df["script_tokens"].sum())
        out["C06"] = (keys_ok and agree and tot_ok and len(ind) == n, f"keys_ok={keys_ok}, agree={agree}, totals_ok={tot_ok}")
    return out
