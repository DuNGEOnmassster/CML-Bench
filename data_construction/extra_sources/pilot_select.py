"""Seeded pilot selection for extra sources: one window per film, a fixed number of films per source.

  python data_construction/extra_sources/pilot_select.py --build data_construction/work/extra_build \
      --per_source IMSDb=14,DailyScript=8,SimplyScripts=8 --out_dir RUN_ROOT

Writes RUN_ROOT/<slug>.ids (one item_id per line) for make_abstract_batches.py --item_ids, one run per source.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--per_source", required=True)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    for spec in args.per_source.split(","):
        name, n = spec.split("=")
        with open(os.path.join(args.build, f"segments_{name.lower()}.jsonl"), encoding="utf-8") as f:
            segs = [json.loads(line) for line in f]
        by_film = defaultdict(list)
        for s in segs:
            by_film[s["imdb_id"]].append(s["item_id"])
        films = sorted(by_film)
        rng.shuffle(films)
        ids = [rng.choice(sorted(by_film[m])) for m in films[: int(n)]]
        with open(os.path.join(args.out_dir, f"{name.lower()}.ids"), "w") as f:
            f.write("\n".join(ids) + "\n")
        print(name, len(ids), "of", len(films), "films")


if __name__ == "__main__":
    main()
