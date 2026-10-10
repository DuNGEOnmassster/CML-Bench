"""Stage 1: MovieSum screenplays -> cleaned, deduplicated, leakage-free CML segments.

Mirrors the CML-Dataset construction (MovieSum script -> contiguous 15-20 scene excerpt), but cuts
segments deterministically and verbatim instead of asking an LLM to copy them out, which in the
original set leaked preambles ("Here is the exact 15-consecutive-scene segment...") into 3/100 items
and altered the text of ~23/100.

Outputs (in --out):
  segments.jsonl          accepted segments (content + metadata, `summary` empty until stage 5)
  rejected_windows.jsonl  candidate windows that failed a quality filter (ids + reasons, no content)
  excluded_movies.jsonl   movies dropped before windowing (ids + reasons)
  build_stats.json        counts, config, distributions
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
import zlib
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cml_format import (  # noqa: E402
    CONTINUOUS_RE,
    HEADING_RE,
    TRANSITION_END_RE,
    clean_text,
    count_tokens,
    drop_duplicate_scenes,
    drop_front_matter,
    parse_script,
    render,
    segment_stats,
    validate_cml,
)
from dataset_schema import gt_relation, load_gt_related, load_identity_table  # noqa: E402
from mislabel_gate import score_items as c34_scores  # noqa: E402

MOVIESUM_URL = "https://huggingface.co/datasets/rohitsaxena/MovieSum/resolve/main/{split}.jsonl"
MOVIESUM_PAGE = "https://huggingface.co/datasets/rohitsaxena/MovieSum"
GT_URL = "https://huggingface.co/datasets/songdj/CML-Bench/resolve/main/ground_truth/gt_100.json"
SPLITS = ("train", "val", "test")
NORMALIZATION_VERSION = "moviesum_clean_detok_v3_2"

CONFIG = {
    "scenes_preferred": [15, 20],
    "scenes_fallback": [12, 24],
    "tokens_min": 2000,
    "tokens_max": 10000,
    "tokens_target": 5500,
    "min_movie_scenes": 30,
    "min_dialogue_turns": 20,
    "min_speakers": 2,
    "dialogue_char_ratio": [0.10, 0.85],
    "max_element_chars": 3000,
    "min_heading_ratio": 0.7,
    "max_bad_character_tag_ratio": 0.05,
    "max_garble_rate": 0.005,
    "max_rare_word_rate": 0.01,
    "ngram": 13,
    "shingle_sample_mod": 16,
    "max_gt_segment_overlap": 0.02,
    "max_gt_movie_overlap": 0.20,
    "max_duplicate_movie_overlap": 0.30,
    # C34: paren_only + cue_as_dlg + bad_cue + fused per window, speakers pooled over the film's accepted windows
    "max_mislabel_score": 3,
}


def download(url: str, path: str) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"downloading {url}", flush=True)
    urllib.request.urlretrieve(url, path + ".part")
    os.replace(path + ".part", path)


def norm_title(movie_name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", movie_name.lower())


def words_of(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def shingles(words: list[str], n: int, sample_mod: int = 1) -> set[int]:
    ids = [zlib.crc32(w.encode()) for w in words]
    out = set()
    for i in range(len(ids) - n + 1):
        h = hash(tuple(ids[i : i + n]))
        if sample_mod == 1 or h % sample_mod == 0:
            out.add(h)
    return out


def overlap(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a) if a else 0.0


def load_moviesum(moviesum_dir: str) -> list[dict]:
    rows = []
    for split in SPLITS:
        path = os.path.join(moviesum_dir, f"{split}.jsonl")
        download(MOVIESUM_URL.format(split=split), path)
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                row["split"] = split
                rows.append(row)
    return rows


def load_gt(gt_path: str) -> list[dict]:
    download(GT_URL, gt_path)
    with open(gt_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def boundary_score(scenes, end: int) -> float:
    """How natural it is to cut right before scenes[end]."""
    if end >= len(scenes):
        return 1.0
    score = 0.0
    nxt = scenes[end].heading
    if HEADING_RE.match(nxt):
        score += 0.5
    if nxt and not CONTINUOUS_RE.search(nxt):
        score += 0.5
    last_text = scenes[end - 1].elements[-1][1] if scenes[end - 1].elements else ""
    if TRANSITION_END_RE.search(last_text):
        score += 0.25
    return score


def choose_windows(scenes, tok_prefix, cfg) -> list[tuple[int, int]]:
    """Greedy non-overlapping windows; prefer 15-20 scenes, a natural cut point, and ~target tokens."""
    windows, i, n = [], 0, len(scenes)
    lo_p, hi_p = cfg["scenes_preferred"]
    lo_f, hi_f = cfg["scenes_fallback"]
    while i < n:
        best = None
        for sizes in (range(lo_p, hi_p + 1), [s for s in range(lo_f, hi_f + 1) if s < lo_p or s > hi_p]):
            for size in sizes:
                j = i + size
                if j > n:
                    break
                toks = tok_prefix[j] - tok_prefix[i]
                if not cfg["tokens_min"] <= toks <= cfg["tokens_max"]:
                    continue
                score = boundary_score(scenes, j) - abs(toks - cfg["tokens_target"]) / 8000
                if best is None or score > best[0]:
                    best = (score, j)
            if best:
                break
        if best is None:
            i += 1
            continue
        windows.append((i, best[1]))
        i = best[1]
    return windows


def segment_rejections(stats: dict, problems: list[str], cfg) -> list[str]:
    reasons = [f"invalid_cml:{p}" for p in problems]
    if not cfg["tokens_min"] <= stats["content_tokens"] <= cfg["tokens_max"]:
        reasons.append("tokens_out_of_range")
    if stats["dialogue_turns"] < cfg["min_dialogue_turns"]:
        reasons.append("few_dialogue_turns")
    if stats["num_speakers"] < cfg["min_speakers"]:
        reasons.append("few_speakers")
    lo, hi = cfg["dialogue_char_ratio"]
    if not lo <= stats["dialogue_char_ratio"] <= hi:
        reasons.append("dialogue_ratio_out_of_range")
    if stats["max_element_chars"] > cfg["max_element_chars"]:
        reasons.append("oversized_element")
    if stats["heading_ratio"] < cfg["min_heading_ratio"]:
        reasons.append("few_scene_headings")
    if stats["bad_character_tag_ratio"] > cfg["max_bad_character_tag_ratio"]:
        reasons.append("bad_character_tags")
    if stats["garble_rate"] > cfg["max_garble_rate"]:
        reasons.append("ocr_garble")
    if stats["rare_word_rate"] > cfg["max_rare_word_rate"]:
        reasons.append("ocr_rare_words")
    return reasons


def load_imdb_meta(path: str | None) -> dict:
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--moviesum_dir", default="data_construction/work/sources/moviesum")
    ap.add_argument("--gt_path", default="data_construction/work/sources/cml_bench/gt_100.json")
    ap.add_argument("--imdb_meta", default="data_construction/work/sources/imdb_meta.json")
    ap.add_argument("--gt_related", default=None, help="GT relation table (default: data_construction/gt_related.json)")
    ap.add_argument("--identity_table", default=None, help="C33 decisions (default: data_construction/identity_table.json)")
    ap.add_argument("--out", default="data_construction/work/build")
    ap.add_argument("--limit_movies", type=int, default=0, help="debug: only process the first N candidate movies")
    ap.add_argument("--no_detok", action="store_true", help="keep MovieSum's PTB tokenization (\"do n't\", \" ,\") verbatim")
    args = ap.parse_args()
    cfg = CONFIG
    normalization = NORMALIZATION_VERSION.replace("_detok", "") if args.no_detok else NORMALIZATION_VERSION
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)

    rows = load_moviesum(args.moviesum_dir)
    gt = load_gt(args.gt_path)
    imdb_meta = load_imdb_meta(args.imdb_meta)
    gt_table = load_gt_related(args.gt_related)
    identity = load_identity_table(args.identity_table)
    gt_ids = {r["imdb_id"] for r in gt}
    gt_titles = {norm_title(r["movie_name"]) for r in gt}

    vocab = Counter()
    for row in rows:
        vocab.update(re.findall(r"[a-z]{3,}", row["script"].lower()))

    gt_seg_shingles = set()
    for r in gt:
        gt_seg_shingles |= shingles(words_of(render(parse_script(r["script_segment"]))), cfg["ngram"])

    excluded, rejected, accepted = [], [], []
    stats = Counter()

    # One row per imdb_id: prefer the train split, then the longer script.
    split_rank = {s: i for i, s in enumerate(SPLITS)}
    by_id: dict[str, dict] = {}
    for row in rows:
        cur = by_id.get(row["imdb_id"])
        if cur is None or (split_rank[row["split"]], -len(row["script"])) < (split_rank[cur["split"]], -len(cur["script"])):
            if cur is not None:
                excluded.append({"imdb_id": cur["imdb_id"], "movie_name": cur["movie_name"], "reason": "duplicate_imdb_id"})
            by_id[row["imdb_id"]] = row
        else:
            excluded.append({"imdb_id": row["imdb_id"], "movie_name": row["movie_name"], "reason": "duplicate_imdb_id"})

    # Movie-level shingles of the GT movies' full scripts catch the same screenplay filed under another id/title.
    gt_movie_shingles = set()
    for imdb_id in gt_ids:
        if imdb_id in by_id:
            gt_movie_shingles |= shingles(words_of(clean_text(by_id[imdb_id]["script"])), cfg["ngram"], cfg["shingle_sample_mod"])

    for e in identity.values():
        if e["decision"] == "relabel":
            t = e["target_imdb_id"]
            if t in gt_ids or t in by_id or gt_relation(t, gt_table) or norm_title(e["target_movie_name"]) in gt_titles:
                sys.exit(f"identity_table: relabel target {t} is a GT movie, GT-related or already a MovieSum row")
        if e["decision"] == "duplicate_keep_other" and e["keep_imdb_id"] not in by_id:
            sys.exit(f"identity_table: {e['imdb_id']} keeps {e['keep_imdb_id']}, which is not a MovieSum row")

    candidates = sorted(by_id.values(), key=lambda r: (split_rank[r["split"]], r["imdb_id"]))
    if args.limit_movies:
        candidates = candidates[: args.limit_movies]
    seen_shingles: set[int] = set()

    for mi, row in enumerate(candidates):
        imdb_id, name = row["imdb_id"], row["movie_name"]
        if imdb_id in gt_ids:
            excluded.append({"imdb_id": imdb_id, "movie_name": name, "reason": "in_cml_bench_gt"})
            continue
        if norm_title(name) in gt_titles:
            excluded.append({"imdb_id": imdb_id, "movie_name": name, "reason": "gt_title_match"})
            continue
        relation = gt_relation(imdb_id, gt_table)
        if relation and relation["type"] == "remake":
            excluded.append({"imdb_id": imdb_id, "movie_name": name, "reason": "gt_remake", "gt_movie": relation["gt_movie"]})
            continue
        # C33: a screenplay MovieSum files under the wrong film is relabelled, dropped, or (duplicates) the row whose
        # cast matches is kept instead; this happens before the duplicate-text check so the right row survives.
        ident = identity.get(imdb_id, {})
        decision = ident.get("decision")
        if decision in ("exclude", "duplicate_keep_other"):
            excluded.append({"imdb_id": imdb_id, "movie_name": name, "reason": f"identity_{decision}",
                             **({"keep_imdb_id": ident["keep_imdb_id"]} if decision == "duplicate_keep_other" else {})})
            continue
        out_id, out_name = imdb_id, name
        source_label = None
        if decision == "relabel":
            out_id, out_name = ident["target_imdb_id"], ident["target_movie_name"]
            source_label = {"imdb_id": imdb_id, "movie_name": name, "split": row["split"]}
            relation = gt_relation(out_id, gt_table)
        dropped: dict[int, str] = {}
        scenes = parse_script(row["script"], detok=not args.no_detok, dropped=dropped)
        scenes = drop_duplicate_scenes(drop_front_matter(scenes), dropped=dropped)
        if len(scenes) < cfg["min_movie_scenes"]:
            excluded.append({"imdb_id": imdb_id, "movie_name": name, "reason": "too_few_scenes", "num_scenes": len(scenes)})
            continue
        movie_text = " ".join(t for s in scenes for _, t in s.elements)
        movie_sh = shingles(words_of(movie_text), cfg["ngram"], cfg["shingle_sample_mod"])
        ov_gt = overlap(movie_sh, gt_movie_shingles)
        if ov_gt > cfg["max_gt_movie_overlap"]:
            excluded.append({"imdb_id": imdb_id, "movie_name": name, "reason": "gt_movie_text_overlap", "overlap": round(ov_gt, 3)})
            continue
        ov_dup = overlap(movie_sh, seen_shingles)
        if ov_dup > cfg["max_duplicate_movie_overlap"]:
            excluded.append({"imdb_id": imdb_id, "movie_name": name, "reason": "duplicate_script_text", "overlap": round(ov_dup, 3)})
            continue
        seen_shingles |= movie_sh

        wrapper_tokens = count_tokens(render([]))
        scene_tokens = [count_tokens(render([s])) - wrapper_tokens for s in scenes]
        prefix = [0]
        for t in scene_tokens:
            prefix.append(prefix[-1] + t)
        windows = choose_windows(scenes, prefix, cfg)
        stats["movies_windowed"] += 1
        meta = imdb_meta.get(out_id, {})
        kept = 0
        film_start = len(accepted)
        for wi, (a, b) in enumerate(windows):
            seg_scenes = scenes[a:b]
            content = render(seg_scenes)
            item_id = f"{out_id}-s{seg_scenes[0].index:04d}-{seg_scenes[-1].index:04d}"
            kept_idx = {s.index for s in seg_scenes}
            gaps = [i for i in range(seg_scenes[0].index, seg_scenes[-1].index + 1) if i not in kept_idx]
            assert all(i in dropped for i in gaps), (item_id, [i for i in gaps if i not in dropped])
            dropped_scenes = [{"scene": i, "reason": dropped[i]} for i in gaps]
            st = segment_stats(seg_scenes, content, vocab)
            reasons = segment_rejections(st, validate_cml(content), cfg)
            gt_ov = overlap(shingles(words_of(content), cfg["ngram"]), gt_seg_shingles)
            if gt_ov > cfg["max_gt_segment_overlap"]:
                reasons.append("gt_segment_overlap")
            stats["windows_total"] += 1
            if reasons:
                rejected.append({"item_id": item_id, "movie_name": out_name, "reasons": reasons, "content_tokens": st["content_tokens"]})
                stats.update(f"reject:{r.split(':')[0]}" for r in reasons)
                continue
            accepted.append(
                {
                    "item_id": item_id,
                    "movie_name": out_name,
                    "imdb_id": out_id,
                    "script_segment": content,
                    "summary": "",
                    "segment_index": kept,
                    "scene_start": seg_scenes[0].index,
                    "scene_end": seg_scenes[-1].index,
                    "dropped_scenes": dropped_scenes,
                    "relative_position": round(a / len(scenes), 4),
                    "source_dataset": "MovieSum",
                    "source_split": row["split"],
                    "source_url": MOVIESUM_PAGE,
                    "source_file": f"{row['split']}.jsonl",
                    "imdb_url": f"https://www.imdb.com/title/{out_id}/",
                    "imdb_rating": meta.get("rating"),
                    "imdb_votes": meta.get("votes"),
                    "genres": meta.get("genres", []),
                    "year": meta.get("year"),
                    "content_normalization": normalization,
                    "content_sha1": hashlib.sha1(content.encode()).hexdigest(),
                    "gt_related": relation,
                    "source_label": source_label,
                    "script_version": "draft" if decision == "accept_as_draft" else None,
                    "identity_decision": decision,
                    "gt_ngram_overlap": round(gt_ov, 5),
                    **{k: st[k] for k in st},
                }
            )
            kept += 1
        if kept:
            sig = c34_scores(accepted[film_start:])
            keep = []
            for it in accepted[film_start:]:
                s = sig[it["item_id"]]
                if s["score"] > cfg["max_mislabel_score"]:
                    rejected.append({"item_id": it["item_id"], "movie_name": out_name, "reasons": ["mislabel_structure"],
                                     "content_tokens": it["content_tokens"], "c34": s})
                    stats["reject:mislabel_structure"] += 1
                else:
                    keep.append({**it, "segment_index": len(keep)})
            accepted[film_start:] = keep
            kept = len(keep)
        if kept:
            stats["movies_with_segments"] += 1
        if (mi + 1) % 200 == 0:
            print(f"[{mi + 1}/{len(candidates)}] accepted={len(accepted)} rejected={len(rejected)} {time.time() - t0:.0f}s", flush=True)

    def dump(name, items):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    dump("segments.jsonl", accepted)
    dump("rejected_windows.jsonl", rejected)
    dump("excluded_movies.jsonl", excluded)

    def dist(xs):
        xs = sorted(xs)
        if not xs:
            return {}
        return {p: xs[min(len(xs) - 1, int(p / 100 * len(xs)))] for p in (0, 5, 10, 25, 50, 75, 90, 95, 100)}

    per_movie = Counter(s["imdb_id"] for s in accepted)
    build_stats = {
        "config": cfg,
        "normalization": normalization,
        "moviesum_rows": len(rows),
        "unique_imdb_ids": len(by_id),
        "gt_items": len(gt),
        "candidate_movies": len(candidates),
        "excluded_movies": dict(Counter(e["reason"] for e in excluded)),
        "movies_windowed": stats["movies_windowed"],
        "movies_with_segments": stats["movies_with_segments"],
        "windows_total": stats["windows_total"],
        "segments_accepted": len(accepted),
        "window_rejections": {k[7:]: v for k, v in stats.items() if k.startswith("reject:")},
        "segments_per_movie": dist(list(per_movie.values())),
        "content_tokens": dist([s["content_tokens"] for s in accepted]),
        "num_scenes": dist([s["num_scenes"] for s in accepted]),
        "dialogue_turns": dist([s["dialogue_turns"] for s in accepted]),
        "split_counts": dict(Counter(s["source_split"] for s in accepted)),
        "gt_related_segments": dict(Counter(s["gt_related"]["type"] for s in accepted if s["gt_related"])),
        "eval_safe_segments": sum(s["gt_related"] is None for s in accepted),
        "identity_decisions": dict(Counter(s["identity_decision"] for s in accepted if s["identity_decision"])),
        "identity_table": "identity_v1",
        "gt_related_table": gt_table["version"],
        "total_content_tokens": sum(s["content_tokens"] for s in accepted),
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "build_stats.json"), "w") as f:
        json.dump(build_stats, f, indent=2)
    print(json.dumps({k: v for k, v in build_stats.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
