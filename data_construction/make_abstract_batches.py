"""Stage 2: select segments for a run and lay them out as abstract-writing batches for agents.

Run directory layout:
  items.jsonl                         selected segments (content + metadata); source of truth for the run
  batches.jsonl                       one line per batch: batch_id, item_ids, content_sha1s (deterministic)
  batches/<batch_id>/manifest.json    work unit for one agent: prompt path + items (content/abstract paths, target words)
  batches/<batch_id>/items/<id>.xml   exact `script_segment` text, one file per item
  batches/<batch_id>/scratch/         private temp space for the agent working this batch
  abstracts/<id>.json                 written by agents (stage 3); existing files are never overwritten here

Abstract writing is done by agents (no API key needed): each agent takes one manifest, reads the prompt
and its items, and writes one JSON file per item. A batch is done when every abstract_path exists and
`check_abstracts.py --batch <dir>` passes. Batch membership depends only on the segment set and --seed,
so rebuilding the same content gives byte-identical manifests.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset_schema import build_id, source_slug  # noqa: E402

DEFAULT_PROMPT = os.path.join(HERE, "prompts", "abstract_prompt_v1_4.md")


def target_center(content_tokens: int) -> int:
    """GT summaries: words ~= 123 + 6.1 per 1k content tokens (r=0.32), median 151.
    Writers overshoot the stated center by ~15 words (pilot v1.1), so the center sits below the GT line."""
    return round(108 + 6 * content_tokens / 1000)


def target_words(content_tokens: int) -> list[int]:
    """Per-item abstract length range, +-35 words around the GT-calibrated center.
    Writers fill whatever ceiling they are given (pilot v1: every abstract landed in the top 15%),
    so the range is kept narrow."""
    center = target_center(content_tokens)
    return [max(90, center - 35), min(300, center + 35)]


def select(segments: list[dict], sample: int, one_per_movie: bool, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if one_per_movie:
        by_movie = defaultdict(list)
        for s in segments:
            by_movie[s["imdb_id"]].append(s)
        movies = sorted(by_movie)
        rng.shuffle(movies)
        picked = [rng.choice(by_movie[m]) for m in movies]
    else:
        picked = list(segments)
        rng.shuffle(picked)
    return picked[:sample] if sample else picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segments", default="data_construction/work/build/segments.jsonl")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--sample", type=int, default=0, help="number of items (0 = all)")
    ap.add_argument("--one_per_movie", action="store_true", help="at most one segment per movie")
    ap.add_argument("--item_ids", help="file with one item_id per line: use exactly these items, in this order")
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--batch_prefix", default=None, help="batch id prefix (default: '<source slug>-b')")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--prompt_version", default="abstract_v1.4")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    with open(args.segments, encoding="utf-8") as f:
        segments = [json.loads(line) for line in f]
    sources = {s["source_dataset"] for s in segments}
    norms = {s["content_normalization"] for s in segments}
    if len(sources) != 1 or len(norms) != 1:
        sys.exit(f"one source and normalization per run, got {sources} / {norms}")
    slug = source_slug(sources.pop())
    build = build_id(slug, norms.pop(), [s["content_sha1"] for s in segments])
    if args.item_ids:
        by_id = {s["item_id"]: s for s in segments}
        with open(args.item_ids, encoding="utf-8") as f:
            wanted = [line.strip() for line in f if line.strip()]
        missing = [i for i in wanted if i not in by_id]
        if missing:
            sys.exit(f"item ids not in the build: {missing[:5]}")
        items = [by_id[i] for i in wanted]
    else:
        items = select(segments, args.sample, args.one_per_movie, args.seed)
    prefix = args.batch_prefix if args.batch_prefix is not None else f"{slug}-b"

    os.makedirs(os.path.join(run_dir, "abstracts"), exist_ok=True)
    index = []
    for n, b in enumerate(range(0, len(items), args.batch_size), start=1):
        batch_id = f"{prefix}{n:04d}"
        bdir = os.path.join(run_dir, "batches", batch_id)
        os.makedirs(os.path.join(bdir, "items"), exist_ok=True)
        os.makedirs(os.path.join(bdir, "scratch"), exist_ok=True)
        entries = []
        for it in items[b : b + args.batch_size]:
            it["batch_id"] = batch_id
            it["build_id"] = build
            content_path = os.path.join(bdir, "items", f"{it['item_id']}.xml")
            with open(content_path, "w", encoding="utf-8") as f:
                f.write(it["script_segment"])
            entries.append(
                {
                    "item_id": it["item_id"],
                    "movie_name": it["movie_name"],
                    "content_path": content_path,
                    "content_sha1": it["content_sha1"],
                    "abstract_path": os.path.join(run_dir, "abstracts", f"{it['item_id']}.json"),
                    "content_tokens": it["content_tokens"],
                    "num_scenes": it["num_scenes"],
                    "target_words": target_words(it["content_tokens"]),
                    "target_center": target_center(it["content_tokens"]),
                }
            )
        manifest = {
            "batch_id": batch_id,
            "build_id": build,
            "run_dir": run_dir,
            "prompt_path": os.path.abspath(args.prompt),
            "prompt_version": args.prompt_version,
            "scratch_dir": os.path.join(bdir, "scratch"),
            "num_items": len(entries),
            "total_content_tokens": sum(e["content_tokens"] for e in entries),
            "items": entries,
        }
        with open(os.path.join(bdir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        index.append({"batch_id": batch_id, "item_ids": [e["item_id"] for e in entries],
                      "content_sha1s": [e["content_sha1"] for e in entries]})

    with open(os.path.join(run_dir, "items.jsonl"), "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(os.path.join(run_dir, "batches.jsonl"), "w", encoding="utf-8") as f:
        for row in index:
            f.write(json.dumps(row) + "\n")
    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump({"build_id": build, "source_slug": slug, "seed": args.seed, "batch_size": args.batch_size,
                   "prompt_version": args.prompt_version, "items": len(items), "batches": len(index)}, f, indent=2)
    print(f"{len(items)} items -> {len(index)} batches of <= {args.batch_size} in {run_dir} (build {build})")


if __name__ == "__main__":
    main()
