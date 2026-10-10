"""Contract C33 (report only): does each film's screenplay belong to the film its imdb_id names?

For every film in a segments/content file, the top-10 speakers of its segments are compared with the character
names IMDb lists for the title's principal cast (`title.principals`). Token rules follow PR #2's `verify_match`
(extra_sources/build_extra.py): speaker-name tokens of >= 3 letters minus generic words, matched against the
character-name tokens of the cast. Per film it reports
  exact      top speakers sharing a token with a cast character name
  fuzzy      top speakers matching only a spelling variant (similarity >= 0.85, e.g. MCCLEOD ~ MACLEOD)
  turn_share share of the top speakers' turns spoken by exactly matched speakers
and flags `suspect` (no exact and no fuzzy match while IMDb lists characters) and `low` (exactly one weak match).
Films whose IMDb principals carry no usable character names ("Self", "Narrator") are `unverifiable`.

  python data_construction/cast_check.py --items data_construction/work/build_v3/segments.jsonl --out c33.json --md c33.md
"""
from __future__ import annotations

import argparse
import difflib
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cml_format import is_bad_character_tag, speaker_name  # noqa: E402

NAME_STOP = {"MR", "MRS", "MS", "DR", "THE", "OLD", "YOUNG", "MAN", "WOMAN", "GIRL", "BOY", "VOICE", "OFFICER", "COP", "GUARD"}
GENERIC_CHARACTERS = {"SELF", "HIMSELF", "HERSELF", "THEMSELVES", "NARRATOR", "VOICE", "UNCREDITED", "ARCHIVE", "FOOTAGE"}
CHAR_RE = re.compile(r"<character>([^<]+)</character>")


def load_characters(imdb_dir: str, ids: set[str]) -> dict[str, set[str]]:
    """Cast character-name tokens per title from title.principals (as in PR #2's imdb_index.load_characters)."""
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


def top_speakers(contents: list[str], k: int = 10) -> list[tuple[str, int, set[str]]]:
    counts = Counter(speaker_name(c) for text in contents for c in CHAR_RE.findall(text))
    out = []
    for name, n in counts.most_common():
        if is_bad_character_tag(name):
            continue
        toks = {t for t in re.findall(r"[A-Z][A-Z'\-]{2,}", name) if t not in NAME_STOP}
        if toks:
            out.append((name, n, toks))
        if len(out) == k:
            break
    return out


def score(speakers, characters: set[str] | None) -> dict:
    usable = {c for c in (characters or set()) if c not in GENERIC_CHARACTERS}
    if not usable:
        return {"status": "unverifiable", "exact": 0, "fuzzy": 0, "n": len(speakers), "turn_share": None}
    exact = [s for s in speakers if s[2] & usable]
    fuzzy = [s for s in speakers if not (s[2] & usable) and any(
        len(t) >= 5 and difflib.SequenceMatcher(None, t, c).ratio() >= 0.85 for t in s[2] for c in usable)]
    turns = sum(s[1] for s in speakers) or 1
    share = round(sum(s[1] for s in exact) / turns, 3)
    if not exact and not fuzzy:
        status = "suspect"
    elif len(exact) <= 1 and share < 0.2:
        status = "low"
    else:
        status = "ok"
    return {"status": status, "exact": len(exact), "fuzzy": len(fuzzy), "n": len(speakers), "turn_share": share}


def identify(imdb_dir: str, flagged: list[dict], min_votes: int = 2000) -> dict[str, dict]:
    """For flagged films, the IMDb movie (>= min_votes) whose principal characters match most top speakers:
    evidence that the screenplay belongs to another film (a remake, the original, a sequel) rather than a draft."""
    need = {t for r in flagged for toks in r["_speaker_tokens"] for t in toks}
    cand: dict[str, set[str]] = {}
    with gzip.open(os.path.join(imdb_dir, "title.principals.tsv.gz"), "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            chars = line.rstrip("\n").rsplit("\t", 1)[-1]
            if chars == "\\N":
                continue
            toks = {t.upper() for t in re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", chars)} & need
            if toks:
                cand.setdefault(line[: line.find("\t")], set()).update(toks)
    votes = {}
    with gzip.open(os.path.join(imdb_dir, "title.ratings.tsv.gz"), "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            tconst, _, v = line.rstrip("\n").split("\t")
            if tconst in cand and int(v) >= min_votes:
                votes[tconst] = int(v)
    names = {}
    with gzip.open(os.path.join(imdb_dir, "title.basics.tsv.gz"), "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.split("\t", 6)
            if p[0] in votes and p[1] in ("movie", "tvMovie", "video"):
                names[p[0]] = f"{p[2]}_{p[5]}"
    out = {}
    for r in flagged:
        best = None
        for tconst, toks in cand.items():
            if tconst not in names or tconst == r["imdb_id"]:
                continue
            hits = sum(1 for st in r["_speaker_tokens"] if st & toks)
            if hits >= 3 and (best is None or (hits, votes[tconst]) > (best[1], votes[best[0]])):
                best = (tconst, hits)
        if best:
            out[r["imdb_id"]] = {"imdb_id": best[0], "movie_name": names[best[0]], "matched_speakers": best[1]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True, help="segments.jsonl / content jsonl (needs imdb_id, movie_name, script_segment)")
    ap.add_argument("--imdb_dir", default="data_construction/work/sources/imdb")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", help="optional markdown summary")
    args = ap.parse_args()

    films = defaultdict(lambda: {"movie_name": None, "items": [], "contents": []})
    with open(args.items, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            film = films[r["imdb_id"]]
            film["movie_name"] = r["movie_name"]
            film["items"].append(r["item_id"])
            film["contents"].append(r["script_segment"])
    chars = load_characters(args.imdb_dir, set(films))
    rows = []
    for imdb_id, film in sorted(films.items()):
        sp = top_speakers(film["contents"])
        res = score(sp, chars.get(imdb_id))
        rows.append({"imdb_id": imdb_id, "movie_name": film["movie_name"], "items": film["items"], **res,
                     "top_speakers": [s[0] for s in sp], "imdb_characters": sorted(chars.get(imdb_id, set()))[:20],
                     "_speaker_tokens": [s[2] for s in sp]})
    flagged = [r for r in rows if r["status"] in ("suspect", "low")]
    best = identify(args.imdb_dir, flagged) if flagged else {}
    for r in rows:
        own = r["exact"]
        alt = best.get(r["imdb_id"])
        r["best_other_match"] = alt if alt and alt["matched_speakers"] >= own + 2 else None
        del r["_speaker_tokens"]
    status = Counter(r["status"] for r in rows)
    summary = {"films": len(rows), "items": sum(len(r["items"]) for r in rows), "status": dict(status),
               "items_by_status": {s: sum(len(r["items"]) for r in rows if r["status"] == s) for s in status}}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "films": rows}, f, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1))
    for r in rows:
        if r["status"] in ("suspect", "low"):
            alt = r["best_other_match"]
            print(r["status"], r["imdb_id"], r["movie_name"], len(r["items"]), f"exact={r['exact']}/{r['n']} fuzzy={r['fuzzy']} "
                  f"share={r['turn_share']}", "| other:", f"{alt['movie_name']} ({alt['imdb_id']}, {alt['matched_speakers']})" if alt else "-")


if __name__ == "__main__":
    main()
