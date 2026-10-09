"""Build imdb_meta.json (rating, votes, genres, year, title) for every MovieSum imdb_id.

Uses the official IMDb non-commercial datasets (https://datasets.imdbws.com/), the same
metadata CML-Bench stores in gt_100_info.json (`imdb_rating`, `genres`).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import urllib.request

csv.field_size_limit(sys.maxsize)
IMDB_URL = "https://datasets.imdbws.com/{name}"


def fetch(name: str, imdb_dir: str) -> str:
    path = os.path.join(imdb_dir, name)
    if not os.path.exists(path):
        os.makedirs(imdb_dir, exist_ok=True)
        print(f"downloading {name}", flush=True)
        urllib.request.urlretrieve(IMDB_URL.format(name=name), path + ".part")
        os.replace(path + ".part", path)
    return path


def read_tsv(path: str, ids: set[str]):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        for row in reader:
            if row[0] in ids:
                yield dict(zip(header, row))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--moviesum_dir", default="data_construction/work/sources/moviesum")
    ap.add_argument("--imdb_dir", default="data_construction/work/sources/imdb")
    ap.add_argument("--out", default="data_construction/work/sources/imdb_meta.json")
    args = ap.parse_args()

    ids = set()
    for split in ("train", "val", "test"):
        with open(os.path.join(args.moviesum_dir, f"{split}.jsonl"), encoding="utf-8") as f:
            for line in f:
                ids.add(json.loads(line)["imdb_id"])

    meta = {i: {} for i in ids}
    for row in read_tsv(fetch("title.basics.tsv.gz", args.imdb_dir), ids):
        genres = [] if row["genres"] == "\\N" else row["genres"].split(",")
        year = None if row["startYear"] == "\\N" else int(row["startYear"])
        meta[row["tconst"]].update({"title": row["primaryTitle"], "year": year, "genres": genres})
    for row in read_tsv(fetch("title.ratings.tsv.gz", args.imdb_dir), ids):
        meta[row["tconst"]].update({"rating": float(row["averageRating"]), "votes": int(row["numVotes"])})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=0)
    found = sum(1 for m in meta.values() if "genres" in m)
    rated = sum(1 for m in meta.values() if "rating" in m)
    print(f"{len(ids)} ids, basics found for {found}, ratings for {rated} -> {args.out}")


if __name__ == "__main__":
    main()
