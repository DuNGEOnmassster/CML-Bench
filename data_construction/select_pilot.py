"""Pick a re-pilot sample: carry over audited items (mapped to the current build) plus new seeded items.

  python data_construction/select_pilot.py --segments SEG --carry ITEM_IDS... --previous OLD_PILOT.jsonl \
      --n_new 20 --seed 20261010 --out ids.txt

A carried item keeps its movie; when a rebuild moved the window, the current window with the largest scene-range
overlap is used. New items are one segment per movie, never from a movie in the previous pilot or the carry list.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--carry", nargs="+", default=[], help="item ids from an earlier build")
    ap.add_argument("--previous", help="earlier pilot jsonl; its movies are not sampled again")
    ap.add_argument("--n_new", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.segments, encoding="utf-8") as f:
        segs = [json.loads(line) for line in f]
    by_movie = defaultdict(list)
    for s in segs:
        by_movie[s["imdb_id"]].append(s)
    picked, mapping = [], {}
    for old in args.carry:
        imdb_id, rng = old.split("-s")
        a, b = map(int, rng.split("-"))
        cands = by_movie.get(imdb_id, [])
        if not cands:
            raise SystemExit(f"{old}: movie not in the build")
        best = max(cands, key=lambda s: (min(b, s["scene_end"]) - max(a, s["scene_start"]), -s["scene_start"]))
        picked.append(best["item_id"])
        mapping[old] = best["item_id"]
    used = {i.split("-s")[0] for i in args.carry}
    if args.previous:
        with open(args.previous, encoding="utf-8") as f:
            used |= {json.loads(line)["imdb_id"] for line in f}
    rng = random.Random(args.seed)
    movies = sorted(m for m in by_movie if m not in used)
    rng.shuffle(movies)
    for m in movies[: args.n_new]:
        picked.append(rng.choice(sorted(by_movie[m], key=lambda s: s["item_id"]))["item_id"])
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(picked) + "\n")
    print(json.dumps({"items": len(picked), "carried": mapping}, indent=1))


if __name__ == "__main__":
    main()
