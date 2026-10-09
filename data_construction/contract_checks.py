"""Check a release folder against the automated assertions of the expansion contract (C01-C24, C30).

  python data_construction/contract_checks.py --release REL --run_dir RUN [--out report.json]

Audit assertions (C25-C28) need an evaluator reading items; C29 needs a rebuild and is reported as manual.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_segments import CONFIG, norm_title, overlap, shingles, words_of  # noqa: E402
from check_abstracts import check_one  # noqa: E402
from cml_format import drop_duplicate_scenes, drop_front_matter, parse_script, render, segment_stats, validate_cml  # noqa: E402
from make_abstract_batches import target_words  # noqa: E402

FIRST = ["movie_name", "imdb_id", "script_segment", "summary"]
PROVENANCE = ["item_id", "source_dataset", "source_split", "source_url", "source_file", "scene_start", "scene_end",
              "imdb_url", "content_sha1", "content_normalization", "abstract_prompt_version", "abstract_author"]
GT_MEDIAN_TOKENS = 5702
LLM_RESIDUE_RE = re.compile(
    r"here (is|are) the (exact )?(\d+[- ])?(consecutive[- ])?scenes?\b|consecutive[- ]scene segment|"
    r"these \d+ (consecutive )?scenes|from the provided script|```",
    re.I,
)
CODE_SUFFIXES = (".py", ".md", ".sh")


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--moviesum_dir", default="data_construction/work/sources/moviesum")
    ap.add_argument("--gt_path", default="data_construction/work/sources/cml_bench/gt_100.json")
    ap.add_argument("--trace_sample", type=int, default=20)
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
    bad_first = sum(1 for r in recs if list(r)[:4] != FIRST or not all(isinstance(r[k], str) and r[k].strip() for k in FIRST))
    record("C01", parse_err == 0 and bad_first == 0 and recs, f"{len(recs)} items, parse_errors={parse_err}, bad_first_fields={bad_first}")

    ms = {}
    for split in ("train", "val", "test"):
        with open(os.path.join(args.moviesum_dir, f"{split}.jsonl"), encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                ms.setdefault(row["imdb_id"], {})[split] = row
    bad_ids = [r["item_id"] for r in recs if not re.match(r"^.+_\d{4}$", r["movie_name"]) or not re.match(r"^tt\d{7,8}$", r["imdb_id"])
               or ms.get(r["imdb_id"], {}).get(r["source_split"], {}).get("movie_name") != r["movie_name"]]
    record("C02", not bad_ids, f"mismatches={bad_ids[:5]}")

    missing = Counter(k for r in recs for k in PROVENANCE if r.get(k) in (None, ""))
    bad_prov = sum(1 for r in recs if r.get("source_dataset") != "MovieSum" or r.get("source_split") not in ("train", "val", "test")
                   or not (r.get("scene_start", 0) <= r.get("scene_end", -1)))
    record("C03", not missing and not bad_prov, f"missing={dict(missing)}, bad_values={bad_prov}")

    ids, shas = Counter(r["item_id"] for r in recs), Counter(r["content_sha1"] for r in recs)
    sha_mismatch = sum(1 for r in recs if hashlib.sha1(r["script_segment"].encode()).hexdigest() != r["content_sha1"])
    record("C04", max(ids.values()) == 1 and max(shas.values()) == 1 and not sha_mismatch,
           f"dup_ids={sum(v > 1 for v in ids.values())}, dup_sha={sum(v > 1 for v in shas.values())}, sha_mismatch={sha_mismatch}")

    rng = random.Random(0)
    sample = rng.sample(recs, min(args.trace_sample, len(recs)))
    trace_fail = []
    for r in sample:
        detok = "detok" in r["content_normalization"]
        scenes = drop_duplicate_scenes(drop_front_matter(parse_script(ms[r["imdb_id"]][r["source_split"]]["script"], detok=detok)))
        sl = [s for s in scenes if r["scene_start"] <= s.index <= r["scene_end"]]
        if render(sl) != r["script_segment"]:
            trace_fail.append(r["item_id"])
    record("C05", not trace_fail, f"traced={len(sample)}, failures={trace_fail}")

    with open(os.path.join(args.release, "info.json"), encoding="utf-8") as f:
        info = json.load(f)
    ind = {x["item_id"]: x for x in info.get("individual_results", [])}
    keys_ok = all({"script_tokens", "summary_tokens", "tag_counts", "imdb_rating", "genres"} <= set(x) for x in ind.values())
    agree = all(ind.get(r["item_id"], {}).get("script_tokens") == r["script_tokens"] for r in recs)
    tot_ok = info.get("summary", {}).get("total_script_tokens") == sum(r["script_tokens"] for r in recs)
    record("C06", keys_ok and agree and tot_ok and len(ind) == len(recs), f"keys_ok={keys_ok}, agree={agree}, totals_ok={tot_ok}")

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
    share_pref = sum(15 <= s <= 20 for s in sc) / len(sc)
    record("C10", all(12 <= s <= 24 for s in sc) and share_pref >= 0.8, f"range=[{min(sc)},{max(sc)}], share_15_20={share_pref:.2f}")

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
    junk = [r["item_id"] for r in recs if re.search(r"\ufffd|[\x00-\x08\x0b-\x1f]|&amp;amp;|>\(?(CONTINUED|OMITTED)\)?<|>\d+\.?<|-[LR][RSC]B-|\*", r["script_segment"])]
    record("C12", p99 <= 2 and not junk, f"artefacts_p99={p99}/1k words, junk_items={junk[:5]}")

    def has_dup_scene(scenes):
        bodies = ["\n".join(t for tag, t in sc.elements if tag != "stage_direction") for sc in scenes]
        bodies = [b for b in bodies if len(b) >= 200]
        return len(bodies) != len(set(bodies))

    noise = [i for i, s in stats.items() if s["garble_rate"] > 0.005 or s["rare_word_rate"] > 0.01
             or s["bad_character_tag_ratio"] > 0.05 or s["max_element_chars"] > 3000 or s["heading_ratio"] < 0.7
             or has_dup_scene(parsed[i])]
    record("C13", not noise, f"violations={noise[:5]}")

    def ascii_share(t):
        letters = [c for c in t if c.isalpha()]
        return sum(c.isascii() for c in letters) / max(1, len(letters))

    non_en = [r["item_id"] for r in recs if ascii_share(r["script_segment"]) < 0.97 or ascii_share(r["summary"]) < 0.97]
    record("C14", not non_en, f"non_english={non_en[:5]}")

    with open(args.gt_path, encoding="utf-8") as f:
        gt = [json.loads(line) for line in f if line.strip()]
    gt_ids, gt_titles = {g["imdb_id"] for g in gt}, {norm_title(g["movie_name"]) for g in gt}
    leak = [r["item_id"] for r in recs if r["imdb_id"] in gt_ids or norm_title(r["movie_name"]) in gt_titles]
    record("C15", not leak, f"gt_movie_items={leak[:5]}")

    gt_sh = set()
    for g in gt:
        gt_sh |= shingles(words_of(render(parse_script(g["script_segment"]))), CONFIG["ngram"])
    item_sh = {r["item_id"]: shingles(words_of(r["script_segment"]), CONFIG["ngram"]) for r in recs}
    gt_ov = {i: overlap(s, gt_sh) for i, s in item_sh.items()}
    record("C16", max(gt_ov.values()) <= 0.02, f"max_gt_overlap={max(gt_ov.values()):.4f}")

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

    chk = [check_one(r["summary"], r["script_segment"], target_words(r["script_tokens"])) for r in recs]
    n = len(chk)
    hard = [r["item_id"] for r, c in zip(recs, chk) if c["hard"]]
    record("C19", not hard, f"hard_fail={hard[:5]}")
    in_target = sum("word_count_outside_target" not in c["soft"] for c in chk) / n
    mean_words = sum(c["words"] for c in chk) / n
    record("C20", in_target >= 0.85 and 130 <= mean_words <= 200, f"in_target={in_target:.2f}, mean_words={mean_words:.1f}")
    top1 = sum(c["top1_speaker_mentioned"] for c in chk) / n
    top3 = sum(c["top3_speakers_mentioned"] >= 2 for c in chk) / n
    record("C21", top1 >= 0.95 and top3 >= 0.85, f"top1={top1:.2f}, top3>=2={top3:.2f}")
    all3 = sum(all(c["thirds_covered"]) for c in chk) / n
    last = sum(c["thirds_covered"][2] for c in chk) / n
    record("C22", all3 >= 0.90 and last >= 0.95, f"all_thirds={all3:.2f}, last_third={last:.2f}")
    g = [c["lexical_grounding"] for c in chk]
    record("C23", sum(g) / n >= 0.55 and min(g) >= 0.35, f"mean={sum(g) / n:.3f}, min={min(g):.3f}")
    first8 = Counter(" ".join(r["summary"].split()[:8]).lower() for r in recs)
    open3 = Counter(" ".join(r["summary"].split()[:3]).lower() for r in recs)
    record("C24", max(first8.values()) == 1 and max(open3.values()) / n <= 0.10,
           f"max_first8_dup={max(first8.values())}, top_opening_3gram={open3.most_common(1)[0]}")

    for cid in ("C25", "C26", "C27", "C28"):
        results[cid] = {"pass": None, "detail": "evaluator audit required"}
    results["C29"] = {"pass": None, "detail": "manual: rerun build_segments.py and compare sha1 sets"}

    try:
        tracked = subprocess.run(["git", "ls-files", "data_construction/work"], capture_output=True, text=True, check=True).stdout.strip()
        grep = subprocess.run(["git", "grep", "-l", "<scene>"], capture_output=True, text=True).stdout.split()
        extra = [p for p in grep if not p.endswith(CODE_SUFFIXES)]
        record("C30", not tracked and not extra, f"tracked_work_files={bool(tracked)}, non_code_files_with_scene_tag={extra}")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        results["C30"] = {"pass": None, "detail": f"git unavailable: {exc}"}

    summary = {"items": len(recs), "automated_pass": sum(v["pass"] is True for v in results.values()),
               "automated_fail": sum(v["pass"] is False for v in results.values()),
               "manual": sum(v["pass"] is None for v in results.values())}
    for cid in sorted(results):
        v = results[cid]
        flag = "PASS" if v["pass"] else ("TODO" if v["pass"] is None else "FAIL")
        print(f"{cid} {flag:4} {v['detail']}")
    print(json.dumps(summary))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
