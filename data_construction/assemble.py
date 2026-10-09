"""Stage 5: join segments with checked abstracts into the release files.

Output (in --out):
  data/<name>.jsonl  one item per line; first four fields match CML-Bench's gt_100.json
                     (movie_name, imdb_id, script_segment, summary), followed by provenance/metadata
  info.json          same layout as CML-Bench's gt_100_info.json (individual_results + summary)
  stats.json         distributions + comparison with the CML-Bench GT set
  README.md          dataset card (private; CC BY-NC 4.0 inherited from MovieSum)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cml_format import count_tokens  # noqa: E402
from dataset_schema import TAG_KEYS, make_record  # noqa: E402


def dist(xs):
    xs = sorted(xs)
    if not xs:
        return {}
    return {f"p{p}": xs[min(len(xs) - 1, int(p / 100 * len(xs)))] for p in (0, 5, 10, 25, 50, 75, 90, 95, 100)} | {
        "mean": round(sum(xs) / len(xs), 2)
    }


def build_record(item: dict, abstract: dict) -> dict:
    return make_record(item, batch_id=item["batch_id"], build=item["build_id"], abstract=abstract, count_tokens=count_tokens)


def info_json(records: list[dict]) -> dict:
    individual = [
        {
            "movie_name": r["movie_name"],
            "imdb_id": r["imdb_id"],
            "item_id": r["item_id"],
            "script_tokens": r["script_tokens"],
            "summary_tokens": r["summary_tokens"],
            "tag_counts": r["tag_counts"],
            "imdb_rating": r["imdb_rating"],
            "genres": r["genres"],
        }
        for r in records
    ]
    n = len(records) or 1
    total_tags = {k: sum(r["tag_counts"][k] for r in records) for k in TAG_KEYS}
    ratings = [r["imdb_rating"] for r in records if r["imdb_rating"] is not None]
    return {
        "individual_results": individual,
        "summary": {
            "total_script_tokens": sum(r["script_tokens"] for r in records),
            "total_summary_tokens": sum(r["summary_tokens"] for r in records),
            "avg_script_tokens": round(sum(r["script_tokens"] for r in records) / n, 2),
            "avg_summary_tokens": round(sum(r["summary_tokens"] for r in records) / n, 2),
            "total_tag_counts": total_tags,
            "avg_tag_counts": {k: round(v / n, 2) for k, v in total_tags.items()},
            "genre_counts": dict(Counter(g for r in records for g in r["genres"])),
            "avg_imdb_rating": round(sum(ratings) / len(ratings), 3) if ratings else None,
        },
    }


DATASET_CARD = """---
license: cc-by-nc-4.0
language:
- en
pretty_name: {pretty_name}
tags:
- screenplay
- movie-scripts
- summarization
- cml-bench
---

# {pretty_name}

Private expansion of the CML-Dataset used by [CML-Bench](https://github.com/DuNGEOnmassster/CML-Bench)
(arXiv:2510.06231). Each item pairs a contiguous excerpt of a human-written movie screenplay
(`script_segment`, Cinematic Markup Language) with an AI-written abstract (`summary`).

- Items: {n_items} from {n_movies} movies
- Source screenplays: [MovieSum](https://huggingface.co/datasets/rohitsaxena/MovieSum) (CC BY-NC 4.0); provenance per item in
  `source_url`, `source_file`, `scene_start`/`scene_end`, `imdb_url`
- No overlap with CML-Bench's 100 ground-truth movies (excluded by IMDb id, title and 13-gram text overlap)
- Abstracts: prompt `{prompt_versions}`, written by {authors}

## Fields

The first four fields match CML-Bench `ground_truth/gt_100.json`: `movie_name`, `imdb_id`, `script_segment`, `summary`.
The rest are provenance and statistics (see `info.json` for the gt_100_info.json-style summary).

## Copyright

The screenplays remain the property of their rights holders. This dataset is private and for
non-commercial research only; do not make it public or redistribute it.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="cml_expansion")
    ap.add_argument("--pretty_name", default="CML-Dataset Expanded")
    ap.add_argument("--gt_info", default="data_construction/work/sources/cml_bench/gt_100_info.json")
    args = ap.parse_args()

    with open(os.path.join(args.run_dir, "items.jsonl"), encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    checks_path = os.path.join(args.run_dir, "abstract_checks.jsonl")
    if not os.path.exists(checks_path):
        sys.exit("run check_abstracts.py --run_dir first")
    with open(checks_path, encoding="utf-8") as f:
        checks = {c["item_id"]: c for c in map(json.loads, f)}

    records, skipped = [], Counter()
    for it in items:
        chk = checks.get(it["item_id"])
        if chk is None or chk["hard"]:
            skipped["no_check" if chk is None else "hard_fail"] += 1
            continue
        with open(os.path.join(args.run_dir, "abstracts", f"{it['item_id']}.json"), encoding="utf-8") as f:
            records.append(build_record(it, json.load(f)))

    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    with open(os.path.join(args.out, "data", f"{args.name}.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info_json(records), f, indent=1, ensure_ascii=False)

    gt_summary = {}
    if os.path.exists(args.gt_info):
        with open(args.gt_info, encoding="utf-8") as f:
            gt = json.load(f)
        gt_summary = {
            "script_tokens": dist([r["script_tokens"] for r in gt["individual_results"]]),
            "summary_tokens": dist([r["summary_tokens"] for r in gt["individual_results"]]),
        }
    stats = {
        "items": len(records),
        "skipped": dict(skipped),
        "movies": len({r["imdb_id"] for r in records}),
        "script_tokens": dist([r["script_tokens"] for r in records]),
        "summary_tokens": dist([r["summary_tokens"] for r in records]),
        "summary_words": dist([r["summary_words"] for r in records]),
        "num_scenes": dist([r["num_scenes"] for r in records]),
        "dialogue_turns": dist([r["tag_counts"]["<dialogue>"] for r in records]),
        "imdb_rating": dist([r["imdb_rating"] for r in records if r["imdb_rating"] is not None]),
        "year": dist([r["year"] for r in records if r["year"]]),
        "genre_counts": dict(Counter(g for r in records for g in r["genres"]).most_common()),
        "source_split_counts": dict(Counter(r["source_split"] for r in records)),
        "abstract_authors": dict(Counter(r["abstract_author"] for r in records)),
        "cml_bench_gt_reference": gt_summary,
    }
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            DATASET_CARD.format(
                pretty_name=args.pretty_name,
                n_items=len(records),
                n_movies=stats["movies"],
                prompt_versions=", ".join(sorted({str(r["abstract_prompt_version"]) for r in records})),
                authors=", ".join(sorted({str(r["abstract_author"]) for r in records})),
            )
        )
    print(json.dumps({k: stats[k] for k in ("items", "skipped", "movies", "script_tokens", "summary_words")}, indent=2))


if __name__ == "__main__":
    main()
