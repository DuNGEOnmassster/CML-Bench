"""Catalog survey: how many films each extra source adds after dedupe against MovieSum and the CML-Bench GT.

  python -m extra_sources.survey --catalog_dir data_construction/work/sources/extra/catalogs \
      --out data_construction/work/extra_survey

Reads saved index pages only (fetch them with fetch.py / curl first). Writes catalog_matches.jsonl (one row per
catalog entry with its IMDb match and dedupe status) and survey_stats.json (per-source and union counts).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict

from . import catalogs
from .imdb_index import fuzzy_key_buckets, load_index, lookup_any, norm_title, title_variants

SOURCES = {
    "imsdb": ("imsdb_all.html", catalogs.imsdb),
    "dailyscript": (("dailyscript_movie.html", "dailyscript_movie_nz.html"), catalogs.dailyscript),
    "awesomefilm": ("awesomefilm.html", catalogs.awesomefilm),
    "simplyscripts": ("simplyscripts_movies.html", catalogs.simplyscripts),
    "scriptslug": ("scriptslug_scripts.xml", catalogs.scriptslug_sitemap),
}
# Order used when the same film is offered by several sources: clean HTML/TXT first, PDFs last.
SOURCE_PRIORITY = ("imsdb", "dailyscript", "awesomefilm", "simplyscripts", "scriptslug")
FORMAT_PRIORITY = ("html", "txt", "pdf", "rtf", "doc", "docx")


def load_reference(moviesum_dir: str, gt_path: str):
    ms_ids, ms_titles = set(), defaultdict(set)
    for split in ("train", "val", "test"):
        with open(os.path.join(moviesum_dir, f"{split}.jsonl"), encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                title, _, year = row["movie_name"].rpartition("_")
                ms_ids.add(row["imdb_id"])
                ms_titles[norm_title(title)].add(int(year) if year.isdigit() else None)
    gt_ids, gt_titles = set(), set()
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                gt_ids.add(row["imdb_id"])
                gt_titles.add(norm_title(row["movie_name"].rpartition("_")[0]))
    return ms_ids, ms_titles, gt_ids, gt_titles


def read_catalog(catalog_dir: str, name: str) -> list[dict]:
    files, parser = SOURCES[name]
    files = (files,) if isinstance(files, str) else files
    text = ""
    for fn in files:
        path = os.path.join(catalog_dir, fn)
        if os.path.exists(path):
            with open(path, encoding="latin-1") as f:
                text += f.read()
    return parser(text) if text else []


TRANSCRIPT_RE = re.compile(r"transcript", re.I)


def match_entry(e: dict, index: dict, ms_ids, ms_titles, gt_ids, gt_titles, fuzzy_keys=None) -> dict:
    """IMDb match + dedupe status. Year-less matches are only provisional: build_extra verifies them
    against the cast character names before a film is used."""
    m = None
    if e.get("imdb_id"):
        m = lookup_any(index, e["title"], e.get("year"), e.get("year_kind") or "release") or {}
        m = {**m, "imdb_id": e["imdb_id"], "confidence": "source"} if not m or m.get("imdb_id") != e["imdb_id"] else {**m, "confidence": "source"}
    elif e["kind"] == "film":
        m = lookup_any(index, e["title"], e.get("year"), e.get("year_kind") or "release", fuzzy_keys)
    keys = {norm_title(t) for t in title_variants(e["title"])}
    status = "new"
    if m is None:
        status = "unmatched" if e["kind"] == "film" else "tv"
    if (m and m["imdb_id"] in gt_ids) or keys & gt_titles:
        status = "in_gt"
    elif m and m["imdb_id"] in ms_ids:
        status = "in_moviesum"
    elif keys & set(ms_titles):
        years = set().union(*(ms_titles[k] for k in keys & set(ms_titles)))
        y = (m or {}).get("year") or e.get("year")
        status = "in_moviesum_title" if y is None or any(my is None or abs(my - y) <= 3 for my in years) else "remake_of_moviesum_title"
    if e["kind"] == "tv":
        status = "tv"
    elif TRANSCRIPT_RE.search(e["title"]) or TRANSCRIPT_RE.search(e["url"]):
        status = "transcript_title"
    return {**e, "match": m, "status": status}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog_dir", default="data_construction/work/sources/extra/catalogs")
    ap.add_argument("--moviesum_dir", default="data_construction/work/sources/moviesum")
    ap.add_argument("--gt_path", default="data_construction/work/sources/cml_bench/gt_100.json")
    ap.add_argument("--imdb_index", default="data_construction/work/sources/imdb/title_index.json")
    ap.add_argument("--out", default="data_construction/work/extra_survey")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    index = load_index(args.imdb_index)
    fuzzy_keys = fuzzy_key_buckets(index)
    ref = load_reference(args.moviesum_dir, args.gt_path)

    rows = []
    for name in SOURCES:
        for e in read_catalog(args.catalog_dir, name):
            rows.append(match_entry(e, index, *ref, fuzzy_keys=fuzzy_keys))
    with open(os.path.join(args.out, "catalog_matches.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    per_source = {}
    new_by_source = defaultdict(set)
    for r in rows:
        src = r["source"].split(":")[0]
        per_source.setdefault(src, Counter())
        per_source[src]["entries"] += 1
        per_source[src][r["status"]] += 1
        if r["status"] in ("new", "remake_of_moviesum_title") and r["match"]:
            new_by_source[src].add(r["match"]["imdb_id"])
    union, claimed = set(), {}
    for src in SOURCE_PRIORITY:
        for i in sorted(new_by_source.get(src, ())):
            if i not in union:
                claimed[i] = src
            union.add(i)
    usable = {s for s in SOURCE_PRIORITY if s != "scriptslug"}
    stats = {
        "per_source": {s: dict(c) | {"unique_new_films": len(new_by_source[s])} for s, c in per_source.items()},
        "union_new_films_all_sources": len(union),
        "union_new_films_excluding_scriptslug": len({i for i, s in claimed.items() if s in usable} |
                                                   {i for s in usable for i in new_by_source.get(s, ())}),
        "first_claim_by_source": dict(Counter(claimed.values())),
        "simplyscripts_hosts_new": dict(Counter(r["source"].split(":", 1)[1] for r in rows
                                                if r["source"].startswith("simplyscripts:") and r["status"] == "new").most_common(30)),
    }
    with open(os.path.join(args.out, "survey_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
