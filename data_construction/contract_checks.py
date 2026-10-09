"""Check a release folder against the automated assertions of the expansion contract (v2).

  python data_construction/contract_checks.py --release REL [--run_dir RUN] [--content_only] [--out report.json]

REL/data/*.jsonl holds schema-1.0 records. --content_only checks a content set (no abstracts yet): the abstract
assertions (C19-C24, C20b, C21') are skipped and an empty `summary` is allowed. Audit assertions (C25'-C28') need
an evaluator; C29 needs a rebuild; both are reported as manual. C05' and C16b come from verify_alignment.py, which
shares no code with the cleaner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_segments import CONFIG, norm_title, overlap, shingles, words_of  # noqa: E402
from check_abstracts import check_one  # noqa: E402
from cml_format import parse_script, render, segment_stats, validate_cml  # noqa: E402
from dataset_schema import load_gt_related, validate_record  # noqa: E402
from make_abstract_batches import target_center, target_words  # noqa: E402
from residue_scan import HEADING_NO_RE, orphan_speaker_lines  # noqa: E402
from verify_alignment import verify  # noqa: E402

FIRST = ["movie_name", "imdb_id", "script_segment", "summary"]
GT_MEDIAN_TOKENS = 5702
LLM_RESIDUE_RE = re.compile(
    r"here (is|are) the (exact )?(\d+[- ])?(consecutive[- ])?scenes?\b|consecutive[- ]scene segment|"
    r"these \d+ (consecutive )?scenes|from the provided script|```",
    re.I,
)
C12B_RE = re.compile(r"-[LR][RSC]B-|\*|\\|\bCONT\s*['\u2019]?\s*D\b|\(CONTINUED\)|\bOMITTED\b")
QUOTE_SPACE_RE = re.compile(r'(?:^|[\s>])" [A-Za-z]|[a-z.!?,] "(?=[\s<])')
CODE_SUFFIXES = (".py", ".md", ".sh", ".json")


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0


def binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def base_title(movie_name: str) -> str:
    t = re.sub(r"_\d{4}$", "", movie_name).lower()
    t = re.sub(r"^the\s+", "", t)
    t = re.split(r"[:\-–]", t)[0]
    t = re.sub(r"\b(part\s+)?([ivx]+|\d+)\b\s*$", "", t.strip())
    return re.sub(r"[^a-z0-9]", "", t)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True)
    ap.add_argument("--run_dir", help="unused; kept for older command lines")
    ap.add_argument("--content_only", action="store_true")
    ap.add_argument("--moviesum_dir", default="data_construction/work/sources/moviesum")
    ap.add_argument("--gt_path", default="data_construction/work/sources/cml_bench/gt_100.json")
    ap.add_argument("--imdb_meta", default="data_construction/work/sources/imdb_meta.json")
    ap.add_argument("--excluded", default="data_construction/work/build_v3/excluded_movies.jsonl",
                    help="build's excluded_movies.jsonl (C32 duplicate-screenplay keeps)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--out")
    args = ap.parse_args()

    data_dir = os.path.join(args.release, "data")
    lines = [line for fn in sorted(os.listdir(data_dir)) if fn.endswith(".jsonl")
             for line in open(os.path.join(data_dir, fn), encoding="utf-8")]
    results = {}

    def record(cid, ok, detail):
        results[cid] = {"pass": bool(ok), "detail": detail}

    recs, parse_err = [], 0
    for line in lines:
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            parse_err += 1
    n = len(recs)
    need = FIRST if not args.content_only else FIRST[:3]
    bad_first = sum(1 for r in recs if list(r)[:4] != FIRST or not all(isinstance(r[k], str) and r[k].strip() for k in need)
                    or not isinstance(r["summary"], str))
    record("C01", parse_err == 0 and bad_first == 0 and recs, f"{n} items, parse_errors={parse_err}, bad_first_fields={bad_first}")

    ms = {}
    for split in ("train", "val", "test"):
        with open(os.path.join(args.moviesum_dir, f"{split}.jsonl"), encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                ms.setdefault(row["imdb_id"], {})[split] = row
    bad_ids = [r["item_id"] for r in recs if ms.get(r["imdb_id"], {}).get(r["source_split"], {}).get("movie_name") != r["movie_name"]]
    record("C02", not bad_ids, f"mismatches={bad_ids[:5]}")

    schema_bad = Counter(p for r in recs for p in validate_record(r))
    if args.content_only:
        schema_bad = Counter({k: v for k, v in schema_bad.items() if k != "summary_fields_inconsistent"})
    bad_prov = sum(1 for r in recs if r["source_dataset"] != "MovieSum" or r["source_split"] not in ("train", "val", "test"))
    record("C03", not schema_bad and not bad_prov, f"schema_problems={dict(schema_bad)}, bad_values={bad_prov}")

    ids, shas = Counter(r["item_id"] for r in recs), Counter(r["content_sha1"] for r in recs)
    sha_mismatch = sum(1 for r in recs if hashlib.sha1(r["script_segment"].encode()).hexdigest() != r["content_sha1"])
    record("C04", max(ids.values()) == 1 and max(shas.values()) == 1 and not sha_mismatch,
           f"dup_ids={sum(v > 1 for v in ids.values())}, dup_sha={sum(v > 1 for v in shas.values())}, sha_mismatch={sha_mismatch}")

    align, align_fails = verify(recs, args.moviesum_dir, args.gt_path, args.workers)
    record("C05'", align["c05_fail"] == 0,
           f"aligned={align['items']}, fail={align['c05_fail']}, inserted_words={align['inserted_words']}, "
           f"deleted_words={align['deleted_words']}/{align['raw_words']}, kinds={align['problem_kinds']}, "
           f"examples={[f['item_id'] for f in align_fails[:3]]}")

    if args.content_only:
        results["C06"] = {"pass": None, "detail": "content set: no info.json"}
    else:
        with open(os.path.join(args.release, "info.json"), encoding="utf-8") as f:
            info = json.load(f)
        ind = {x["item_id"]: x for x in info.get("individual_results", [])}
        keys_ok = all({"script_tokens", "summary_tokens", "tag_counts", "imdb_rating", "genres"} <= set(x) for x in ind.values())
        agree = all(ind.get(r["item_id"], {}).get("script_tokens") == r["script_tokens"] for r in recs)
        tot_ok = info.get("summary", {}).get("total_script_tokens") == sum(r["script_tokens"] for r in recs)
        record("C06", keys_ok and agree and tot_ok and len(ind) == n, f"keys_ok={keys_ok}, agree={agree}, totals_ok={tot_ok}")

    invalid = [r["item_id"] for r in recs if validate_cml(r["script_segment"])]
    record("C07", not invalid, f"invalid={invalid[:5]}")

    residue = [r["item_id"] for r in recs if not r["script_segment"].startswith("<script>") or not r["script_segment"].endswith("</script>")
               or LLM_RESIDUE_RE.search(r["script_segment"])]
    record("C08", not residue, f"residue={residue[:5]}")

    toks = [r["script_tokens"] for r in recs]
    med = median(toks)
    record("C09", all(2000 <= t <= 10000 for t in toks) and abs(med - GT_MEDIAN_TOKENS) <= 0.2 * GT_MEDIAN_TOKENS,
           f"min={min(toks)}, median={med}, max={max(toks)} (GT median {GT_MEDIAN_TOKENS})")

    sc = [r["num_scenes"] for r in recs]
    k = sum(15 <= s <= 20 for s in sc)
    share = k / n
    if n >= 500:
        pref_ok, how = share >= 0.80, "share>=0.80 (n>=500)"
    else:
        pval = binom_cdf(k, n, 0.80)
        pref_ok, how = pval >= 0.05, f"binomial P(X<={k}|n={n},p=0.8)={pval:.3f} >= 0.05"
    record("C10'", all(12 <= s <= 24 for s in sc) and pref_ok, f"range=[{min(sc)},{max(sc)}], share_15_20={share:.3f}, {how}")

    vocab = Counter()
    for by_split in ms.values():
        for row in by_split.values():
            vocab.update(re.findall(r"[a-z]{3,}", row["script"].lower()))
    parsed = {r["item_id"]: parse_script(r["script_segment"], detok=False) for r in recs}
    stats = {r["item_id"]: segment_stats(parsed[r["item_id"]], r["script_segment"], vocab) for r in recs}
    bad_dlg = [i for i, s in stats.items() if s["dialogue_turns"] < 20 or s["num_speakers"] < 2 or not 0.10 <= s["dialogue_char_ratio"] <= 0.85]
    record("C11", not bad_dlg, f"violations={bad_dlg[:5]}")

    art = sorted(s["tokenization_artefacts_per_1k_words"] for s in stats.values())
    p99 = art[min(len(art) - 1, int(0.99 * len(art)))]
    junk = [r["item_id"] for r in recs if re.search(r"\ufffd|[\x00-\x08\x0b-\x1f]|&amp;amp;|>\(?(CONTINUED|OMITTED)\)?<|>\d+\.?<", r["script_segment"])]
    record("C12", p99 <= 2 and not junk, f"artefacts_p99={p99}/1k words, junk_items={junk[:5]}")

    c12b = [r["item_id"] for r in recs if C12B_RE.search(r["script_segment"])]
    qrate = sorted(1000 * len(QUOTE_SPACE_RE.findall(r["script_segment"])) / max(1, len(r["script_segment"].split())) for r in recs)
    q99 = qrate[min(n - 1, int(0.99 * n))]
    record("C12b", not c12b and q99 <= 1, f"bracket/asterisk/backslash/CONT'D items={len(c12b)} {c12b[:3]}, quote_inner_space_p99={q99:.2f}/1k words")

    orphans = {r["item_id"]: orphan_speaker_lines(r["script_segment"]) for r in recs}
    bad_orph = [i for i, v in orphans.items() if v]
    record("C12c", not bad_orph, f"items_with_orphan_speaker_lines={len(bad_orph)} {bad_orph[:3]}, total={sum(orphans.values())}")

    def has_dup_scene(scenes):
        bodies = ["\n".join(t for tag, t in s.elements if tag != "stage_direction") for s in scenes]
        bodies = [b for b in bodies if len(b) >= 200]
        return len(bodies) != len(set(bodies))

    noise = [i for i, s in stats.items() if s["garble_rate"] > 0.005 or s["rare_word_rate"] > 0.01
             or s["bad_character_tag_ratio"] > 0.05 or s["max_element_chars"] > 3000 or s["heading_ratio"] < 0.7
             or has_dup_scene(parsed[i])]
    record("C13", not noise, f"violations={noise[:5]}")

    headings = sum(r["script_segment"].count("<stage_direction>") for r in recs)
    numbered = sum(len(HEADING_NO_RE.findall(r["script_segment"])) for r in recs)
    record("C13b", numbered <= 0.01 * headings, f"numbered_headings={numbered}/{headings} ({numbered / max(1, headings):.4%})")

    def ascii_share(t):
        letters = [c for c in t if c.isalpha()]
        return sum(c.isascii() for c in letters) / max(1, len(letters))

    non_en = [r["item_id"] for r in recs if ascii_share(r["script_segment"]) < 0.97 or (r["summary"] and ascii_share(r["summary"]) < 0.97)]
    record("C14", not non_en, f"non_english={non_en[:5]}")

    with open(args.gt_path, encoding="utf-8") as f:
        gt = [json.loads(line) for line in f if line.strip()]
    gt_ids, gt_titles = {g["imdb_id"] for g in gt}, {norm_title(g["movie_name"]) for g in gt}
    leak = [r["item_id"] for r in recs if r["imdb_id"] in gt_ids or norm_title(r["movie_name"]) in gt_titles]
    record("C15", not leak, f"gt_movie_items={leak[:5]}")

    table = load_gt_related()
    rel = {x["imdb_id"]: x for x in table["relations"]}
    cleared = set(rel) | {x["imdb_id"] for x in table["unrelated"]}
    remakes = [r["item_id"] for r in recs if rel.get(r["imdb_id"], {}).get("type") == "remake"]
    wrong_flag = [r["item_id"] for r in recs
                  if (r["gt_related"] or None) != ({"type": rel[r["imdb_id"]]["type"], "gt_movie": rel[r["imdb_id"]]["gt_movie"],
                                                     "gt_imdb_id": rel[r["imdb_id"]]["gt_imdb_id"]} if r["imdb_id"] in rel else None)]
    gt_base = {g["imdb_id"]: base_title(g["movie_name"]) for g in gt}
    lookalikes = set()
    for r in recs:
        name, b = r["movie_name"].lower(), base_title(r["movie_name"])
        for gid, gb in gt_base.items():
            kws = table["ip_keywords"].get(gid, [])
            if (len(gb) >= 4 and len(b) >= 4 and (gb in b or b in gb)) or any(re.search(rf"\b{re.escape(kw)}\b", name) for kw in kws):
                if r["imdb_id"] not in cleared:
                    lookalikes.add(r["movie_name"])
    unsafe = sum(1 for r in recs if r["eval_safe"] != (r["gt_related"] is None))
    record("C15b", not remakes and not wrong_flag and not lookalikes and not unsafe,
           f"remake_items={len(remakes)}, wrong_gt_related={len(wrong_flag)} {wrong_flag[:3]}, "
           f"unreviewed_title_lookalikes={sorted(lookalikes)[:8]}, eval_safe_inconsistent={unsafe}, "
           f"gt_related_items={sum(r['gt_related'] is not None for r in recs)}")

    gt_sh = set()
    for g in gt:
        gt_sh |= shingles(words_of(render(parse_script(g["script_segment"]))), CONFIG["ngram"])
    item_sh = {r["item_id"]: shingles(words_of(r["script_segment"]), CONFIG["ngram"]) for r in recs}
    gt_ov = {i: overlap(s, gt_sh) for i, s in item_sh.items()}
    record("C16", max(gt_ov.values()) <= 0.02, f"max_gt_overlap={max(gt_ov.values()):.4f}")
    record("C16b", align["c16b_pass"], f"max_gt_8gram_overlap={align['c16b_max_gt_8gram_overlap']:.5f} (independent tokenizer)")

    owner = defaultdict(set)
    for r in recs:
        for h in item_sh[r["item_id"]]:
            owner[h].add(r["imdb_id"])
    dup_pairs = []
    for r in recs:
        sh = item_sh[r["item_id"]]
        other = Counter(m for h in sh for m in owner[h] if m != r["imdb_id"])
        if other and other.most_common(1)[0][1] / max(1, len(sh)) > 0.30:
            dup_pairs.append((r["item_id"], other.most_common(1)[0][0]))
    record("C17", not dup_pairs, f"duplicate_pairs={dup_pairs[:5]}")

    by_movie = defaultdict(list)
    for r in recs:
        by_movie[r["imdb_id"]].append((r["scene_start"], r["scene_end"]))
    overlaps = [m for m, rs in by_movie.items() if any(a2 <= b1 for (a1, b1), (a2, b2) in zip(sorted(rs), sorted(rs)[1:]))]
    record("C18", not overlaps, f"movies_with_overlapping_ranges={overlaps[:5]}")

    if args.content_only:
        for cid in ("C19", "C20", "C20b", "C21'", "C22", "C23", "C24"):
            results[cid] = {"pass": None, "detail": "content set: no abstracts"}
    else:
        chk = [check_one(r["summary"], r["script_segment"], target_words(r["script_tokens"])) for r in recs]
        hard = [r["item_id"] for r, c in zip(recs, chk) if c["hard"]]
        record("C19", not hard, f"hard_fail={hard[:5]}")
        in_target = sum("word_count_outside_target" not in c["soft"] for c in chk) / n
        mean_words = sum(c["words"] for c in chk) / n
        record("C20", in_target >= 0.85 and 130 <= mean_words <= 200, f"in_target={in_target:.2f}, mean_words={mean_words:.1f}")
        below = sum(c["words"] < target_center(r["script_tokens"]) for r, c in zip(recs, chk)) / n
        wsorted = sorted(c["words"] for c in chk)
        p25 = wsorted[int(0.25 * (n - 1))]
        single = sum(c.get("paragraphs", 1) == 1 for c in chk) / n
        record("C20b", 0.35 <= below <= 0.65 and p25 <= 150 and single >= 0.6,
               f"below_center={below:.2f} (0.35-0.65), p25_words={p25} (<=150), single_paragraph={single:.2f} (>=0.6)")
        top1 = sum(c["top1_speaker_mentioned"] for c in chk) / n
        top3 = sum(c["top3_speakers_mentioned"] >= 2 for c in chk) / n
        record("C21'", top1 >= 0.95 and top3 >= 0.85, f"top1={top1:.2f}, top3>=2={top3:.2f} (spelling variants count)")
        all3 = sum(all(c["thirds_covered"]) for c in chk) / n
        last = sum(c["thirds_covered"][2] for c in chk) / n
        record("C22", all3 >= 0.90 and last >= 0.95, f"all_thirds={all3:.2f}, last_third={last:.2f}")
        g = [c["lexical_grounding"] for c in chk]
        record("C23", sum(g) / n >= 0.55 and min(g) >= 0.35, f"mean={sum(g) / n:.3f}, min={min(g):.3f}")
        first8 = Counter(" ".join(r["summary"].split()[:8]).lower() for r in recs)
        open3 = Counter(" ".join(r["summary"].split()[:3]).lower() for r in recs)
        record("C24", max(first8.values()) == 1 and max(open3.values()) / n <= 0.10,
               f"max_first8_dup={max(first8.values())}, top_opening_3gram={open3.most_common(1)[0]}")

    for cid in ("C25'", "C26'", "C27", "C28'"):
        results[cid] = {"pass": None, "detail": "evaluator audit required"}
    results["C29"] = {"pass": None, "detail": "manual: rerun build_segments.py and compare sha1 sets"}

    try:
        tracked = subprocess.run(["git", "ls-files", "data_construction/work"], capture_output=True, text=True, check=True).stdout.strip()
        grep = subprocess.run(["git", "grep", "-l", "<scene>"], capture_output=True, text=True).stdout.split()
        extra = [p for p in grep if not p.endswith(CODE_SUFFIXES)]
        record("C30", not tracked and not extra, f"tracked_work_files={bool(tracked)}, non_code_files_with_scene_tag={extra}")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        results["C30"] = {"pass": None, "detail": f"git unavailable: {exc}"}

    results["C31"] = {"pass": None, "detail": "scale-up process: per-batch merge gates (validate_batch.py) + evaluator sample audit"}

    year_off = [r["item_id"] for r in recs if r["year"] and abs(int(r["movie_name"][-4:]) - r["year"]) > 1]
    dup_keeps, dup_bad = [], []
    if os.path.exists(args.excluded):
        with open(args.excluded, encoding="utf-8") as f:
            dups = [e for e in map(json.loads, f) if e["reason"] == "duplicate_script_text"]
        by_base = defaultdict(list)
        for r in recs:
            by_base[base_title(r["movie_name"])].append(r)
        for e in dups:
            for r in by_base.get(base_title(e["movie_name"]), []):
                dup_keeps.append(r["movie_name"])
                if r["year"] is None or int(r["movie_name"][-4:]) != r["year"] or r["imdb_rating"] is None:
                    dup_bad.append(r["item_id"])
    record("C32", not dup_bad and len(year_off) <= 0.01 * n,
           f"duplicate_keeps={sorted(set(dup_keeps))}, keep_metadata_mismatch={dup_bad[:3]}, "
           f"items_with_title_year_vs_imdb_year_gap>1={len(year_off)}")

    summary = {"items": n, "content_only": args.content_only, "build_id": recs[0].get("build_id") if recs else None,
               "automated_pass": sum(v["pass"] is True for v in results.values()),
               "automated_fail": sum(v["pass"] is False for v in results.values()),
               "manual": sum(v["pass"] is None for v in results.values())}
    for cid in sorted(results, key=lambda c: (int(re.sub(r"\D", "", c)), c)):
        v = results[cid]
        flag = "PASS" if v["pass"] else ("TODO" if v["pass"] is None else "FAIL")
        print(f"{cid:5} {flag:4} {v['detail']}")
    print(json.dumps(summary))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"build_id": summary["build_id"], "summary": summary, "results": results,
                       "alignment_failures": align_fails[:200]}, f, indent=2)


if __name__ == "__main__":
    main()
