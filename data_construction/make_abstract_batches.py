"""Stage 2: select segments for a run and lay them out as abstract-writing batches for agents.

Run directory layout:
  items.jsonl                         selected segments (content + metadata); source of truth for the run
  batches/batch_0001/manifest.json    work unit for one agent: prompt path + items (content/abstract paths, target words)
  batches/batch_0001/items/<id>.xml   exact `script_segment` text, one file per item
  abstracts/<id>.json                 written by agents (stage 3); existing files are never overwritten here

Abstract writing is done by agents (no API key needed): each agent takes one manifest, reads the prompt
and its items, and writes one JSON file per item. A batch is done when every abstract_path exists and
`check_abstracts.py --batch <dir>` passes.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROMPT = os.path.join(HERE, "prompts", "abstract_prompt_v1_1.md")


def target_center(content_tokens: int) -> int:
    """GT summaries: words ~= 123 + 6.1 per 1k content tokens (r=0.32), median 151."""
    return round(120 + 6 * content_tokens / 1000)


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
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--prompt_version", default="abstract_v1.1")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    with open(args.segments, encoding="utf-8") as f:
        segments = [json.loads(line) for line in f]
    items = select(segments, args.sample, args.one_per_movie, args.seed)

    os.makedirs(os.path.join(run_dir, "abstracts"), exist_ok=True)
    with open(os.path.join(run_dir, "items.jsonl"), "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    n_batches = 0
    for b in range(0, len(items), args.batch_size):
        n_batches += 1
        batch_id = f"batch_{n_batches:04d}"
        bdir = os.path.join(run_dir, "batches", batch_id)
        os.makedirs(os.path.join(bdir, "items"), exist_ok=True)
        entries = []
        for it in items[b : b + args.batch_size]:
            content_path = os.path.join(bdir, "items", f"{it['item_id']}.xml")
            with open(content_path, "w", encoding="utf-8") as f:
                f.write(it["script_segment"])
            entries.append(
                {
                    "item_id": it["item_id"],
                    "movie_name": it["movie_name"],
                    "content_path": content_path,
                    "abstract_path": os.path.join(run_dir, "abstracts", f"{it['item_id']}.json"),
                    "content_tokens": it["content_tokens"],
                    "num_scenes": it["num_scenes"],
                    "target_words": target_words(it["content_tokens"]),
                    "target_center": target_center(it["content_tokens"]),
                }
            )
        manifest = {
            "batch_id": batch_id,
            "run_dir": run_dir,
            "prompt_path": os.path.abspath(args.prompt),
            "prompt_version": args.prompt_version,
            "num_items": len(entries),
            "total_content_tokens": sum(e["content_tokens"] for e in entries),
            "items": entries,
        }
        with open(os.path.join(bdir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"{len(items)} items -> {n_batches} batches of <= {args.batch_size} in {run_dir}")


if __name__ == "__main__":
    main()
