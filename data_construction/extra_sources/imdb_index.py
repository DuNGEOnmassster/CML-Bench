"""Title (+ year hint) -> IMDb id lookup built from the IMDb non-commercial dumps.

  python -m extra_sources.imdb_index --imdb_dir data_construction/work/sources/imdb   # builds title_index.json

The index keeps feature-length title types only (movie, tvMovie, video) plus series types for TV sources,
keyed by a normalized title (articles, punctuation, accents and "&" folded), with year, type, votes,
rating and genres so ambiguous titles can be resolved by year proximity and popularity.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import sys
import unicodedata

csv.field_size_limit(sys.maxsize)
FILM_TYPES = ("movie", "tvMovie", "video")
SERIES_TYPES = ("tvSeries", "tvMiniSeries")
_ARTICLE_RE = re.compile(r"^(the|a|an)\s+|,\s*(the|a|an)$")


def norm_title(title: str) -> str:
    t = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode().lower().strip()
    t = t.replace("&", " and ")
    t = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9, ]", " ", t)).strip()
    t = _ARTICLE_RE.sub("", t).strip()
    return re.sub(r"[^a-z0-9]", "", t)


def build_index(imdb_dir: str, out_path: str) -> dict:
    ratings = {}
    with gzip.open(os.path.join(imdb_dir, "title.ratings.tsv.gz"), "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader)
        for tconst, rating, votes in reader:
            ratings[tconst] = (float(rating), int(votes))
    index: dict[str, list] = {}
    with gzip.open(os.path.join(imdb_dir, "title.basics.tsv.gz"), "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader)
        for row in reader:
            tconst, ttype, primary, original, is_adult, start = row[:6]
            if ttype not in FILM_TYPES + SERIES_TYPES or is_adult == "1":
                continue
            rating, votes = ratings.get(tconst, (None, 0))
            if ttype in FILM_TYPES and votes < 50:
                continue
            year = int(start) if start.isdigit() else None
            rec = [tconst, year, ttype, votes, rating, row[8] if len(row) > 8 else "", primary]
            for key in {norm_title(primary), norm_title(original)}:
                if key:
                    index.setdefault(key, []).append(rec)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(index, f)
    return index


def load_index(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def lookup(index: dict, title: str, year: int | None = None, year_kind: str = "release", series: bool = False) -> dict | None:
    """Best IMDb match for a title. `year_kind="draft"` means the year is a script draft date, so the
    film is expected 0-4 years later. Returns {imdb_id, year, title, votes, rating, genres, confidence}."""
    types = SERIES_TYPES if series else FILM_TYPES
    cands = [c for c in index.get(norm_title(title), []) if c[2] in types]
    if not cands:
        return None

    def year_fit(c):
        if year is None or c[1] is None:
            return None
        d = c[1] - year
        return d if year_kind == "draft" else abs(d)

    conflict = False
    if year is not None:
        lo, hi = (-1, 5) if year_kind == "draft" else (-1, 1)
        close = [c for c in cands if c[1] is not None and lo <= c[1] - year <= hi]
        if close:
            best = max(close, key=lambda c: (c[3], -abs(year_fit(c) or 0)))
            conf = "high" if len(close) == 1 or best[3] >= 10 * sorted((c[3] for c in close), reverse=True)[1] else "medium"
        else:
            best, conf = max(cands, key=lambda c: c[3]), "low"
            conflict = True  # nothing near the year hint: the most popular same-title film is a guess
    else:
        best = max(cands, key=lambda c: c[3])
        votes = sorted((c[3] for c in cands), reverse=True)
        conf = "medium" if len(votes) == 1 or votes[0] >= 10 * votes[1] else "low"
    tconst, y, ttype, votes, rating, genres, primary = best
    if votes < 1000 and conf == "high":
        conf = "low"  # obscure title sharing the name of the intended film (e.g. a 347-vote video)
    ambiguous = conflict or (year is None and sum(1 for c in cands if c[3] >= 1000) >= 2)
    return {"imdb_id": tconst, "year": y, "title": primary, "type": ttype, "votes": votes, "rating": rating,
            "genres": [g for g in genres.split(",") if g and g != "\\N"], "confidence": conf, "ambiguous": ambiguous}


_ALT_RE = re.compile(r"\((?:was|aka|a\.k\.a\.|originally|written as|formerly|also known as|filmed as|released as|produced as|made as)?\s*\"?([^()\"]+)\"?\)?", re.I)


def _title_forms(title: str) -> tuple[list[str], list[str]]:
    """(primary, secondary) title forms. Primary: the title with and without a trailing parenthetical or
    'script' suffix. Secondary: parenthetical aliases ("was"/"aka" titles) and both parts of 'X: Y'."""
    title = re.sub(r"\s+", " ", title).strip()
    base = re.sub(r"\s*\(.*?(\)|$)", "", title).strip()
    aliases = [m.strip() for m in _ALT_RE.findall(title[len(base):]) if m.strip()]
    base2 = re.sub(r"\s+(script|screenplay|shooting script|transcript)$", "", base, flags=re.I)
    released = re.findall(r"\((?:filmed|released|produced|made) as\s+\"?([^()\"]+)\"?\)", title, re.I)
    primary = released + [title, base, base2]
    secondary = aliases + ([p.strip() for p in base2.split(":", 1)] if ":" in base2 else [])

    def uniq(xs, seen):
        out = []
        for t in xs:
            k = norm_title(t)
            if len(k) >= 2 and k not in seen:
                seen.add(k)
                out.append(t)
        return out

    seen: set[str] = set()
    return uniq(primary, seen), uniq(secondary, seen)


def title_variants(title: str) -> list[str]:
    """'U Turn (Stray Dogs)' -> ['U Turn (Stray Dogs)', 'U Turn', 'Stray Dogs'];
    'Friday the 13th Part 10: Jason X' -> [..., 'Friday the 13th Part 10', 'Jason X']."""
    primary, secondary = _title_forms(title)
    return primary + secondary


def _fuzzy(index: dict, title: str, year, year_kind, fuzzy_keys: dict) -> dict | None:
    from rapidfuzz import fuzz, process

    k = norm_title(title)
    pool = fuzzy_keys.get(k[:2], [])
    best = process.extractOne(k, pool, scorer=fuzz.ratio, score_cutoff=92) if len(k) >= 6 else None
    if not best:
        return None
    m = lookup(index, best[0], year, year_kind)
    if m is None:
        return None
    return {**m, "matched_title": best[0], "confidence": "low", "fuzzy_score": round(best[1], 1)}


def lookup_any(index: dict, title: str, year: int | None = None, year_kind: str = "release",
               fuzzy_keys: dict | None = None) -> dict | None:
    """Exact match on the full title (with/without parenthetical), then a fuzzy match of the full title
    (rapidfuzz ratio >= 92 among keys sharing the first two characters: 'Sweet Smell of Sucess',
    'Lord of the Rings: Fellowship of the Ring, The'), and only then aliases and the parts of an 'X: Y'
    title. Fuzzy and alias/part matches get confidence 'low', so build_extra requires cast verification."""
    primary, secondary = _title_forms(title)
    for t in primary:
        m = lookup(index, t, year, year_kind)
        if m:
            return {**m, "matched_title": t}
    if fuzzy_keys is not None:
        for t in primary:
            m = _fuzzy(index, t, year, year_kind, fuzzy_keys)
            if m:
                return m
    for t in secondary:
        m = lookup(index, t, year, year_kind)
        if m:
            if year is None or m["confidence"] != "high":
                m["confidence"] = "low"
            return {**m, "matched_title": t}
    return None


def id_meta(index: dict, ids: set[str]) -> dict[str, dict]:
    """imdb_id -> {title, year, type, votes, rating, genres} (the metadata of record, whatever matched the title)."""
    out = {}
    for cands in index.values():
        for c in cands:
            if c[0] in ids and c[0] not in out:
                out[c[0]] = {"title": c[6], "year": c[1], "type": c[2], "votes": c[3], "rating": c[4],
                             "genres": [g for g in c[5].split(",") if g and g != "\\N"]}
    return out


def fuzzy_key_buckets(index: dict, min_votes: int = 1000) -> dict:
    buckets: dict[str, list] = {}
    for k, cands in index.items():
        if any(c[2] in FILM_TYPES and c[3] >= min_votes for c in cands):
            buckets.setdefault(k[:2], []).append(k)
    return buckets


def load_characters(imdb_dir: str, ids: set[str]) -> dict[str, set[str]]:
    """Cast character-name tokens per title from title.principals (for verifying a title match against the
    speakers of the parsed script)."""
    out: dict[str, set[str]] = {}
    with gzip.open(os.path.join(imdb_dir, "title.principals.tsv.gz"), "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            tconst = line[: line.find("\t")]
            if tconst not in ids:
                continue
            chars = line.rstrip("\n").rsplit("\t", 1)[-1]
            if chars != "\\N":
                for tok in re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", chars):
                    out.setdefault(tconst, set()).add(tok.upper())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imdb_dir", default="data_construction/work/sources/imdb")
    args = ap.parse_args()
    idx = build_index(args.imdb_dir, os.path.join(args.imdb_dir, "title_index.json"))
    print(f"{len(idx)} normalized titles")


if __name__ == "__main__":
    main()
