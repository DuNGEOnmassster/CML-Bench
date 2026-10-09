"""Extra-source segment build: catalog matches -> fetch -> text -> CML -> script quality -> dedupe -> windows.

  python data_construction/extra_sources/build_extra.py \
      --matches data_construction/work/extra_survey/catalog_matches.jsonl --out data_construction/work/extra_build

Films are candidates when the survey marked them new (IMDb id not in MovieSum or the CML-Bench GT, title
not a MovieSum/GT title). For each film the offers are tried in source/format priority order until one
passes the script-level quality gates (quality.py), the IMDb-match check and the duplicate checks:

- IMDb match: year-less or ambiguous matches must share a speaker name with the IMDb cast characters;
- 13-gram overlap (sampled) with the GT movies' scripts <= 20%, with every MovieSum script <= 30%,
  and with already accepted extra films <= 30% (remakes, re-titled copies, the same draft on two sites).

Accepted films are windowed with build_segments.choose_windows and filtered per window exactly like
MovieSum (segment_rejections + GT segment overlap <= 2%). Outputs (no screenplay text outside segments.jsonl):

  segments.jsonl        accepted segments, MovieSum segment schema + extra provenance fields
  films.jsonl           one row per candidate film: offers tried, quality metrics, outcome (no text)
  rejected_windows.jsonl, build_stats.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from build_segments import (  # noqa: E402
    CONFIG,
    NORMALIZATION_VERSION as MOVIESUM_NORMALIZATION,
    choose_windows,
    load_gt,
    load_moviesum,
    overlap,
    segment_rejections,
    shingles,
    words_of,
)
from cml_format import count_tokens, drop_duplicate_scenes, drop_front_matter, parse_script, render, segment_stats, speaker_name, validate_cml  # noqa: E402

from extra_sources.fetch import Fetcher, PolicyError  # noqa: E402
from extra_sources.imdb_index import id_meta, load_characters, load_index, lookup, norm_title  # noqa: E402
from extra_sources.quality import QUALITY_CONFIG, absorbed_action_rate, script_quality  # noqa: E402
from extra_sources.text_screenplay import PARSER_VERSION, extract_text, text_to_scenes  # noqa: E402
from extra_sources import catalogs  # noqa: E402
from dataset_schema import gt_relation, load_gt_related  # noqa: E402

# parser version + the shared cml_format cleaning version ("moviesum_clean_detok_v3" -> "clean_detok_v3")
NORMALIZATION_VERSION = f"{PARSER_VERSION.replace('_', '')}_{MOVIESUM_NORMALIZATION.split('_', 1)[1]}"
SOURCE_NAMES = {"imsdb": "IMSDb", "dailyscript": "DailyScript", "awesomefilm": "AwesomeFilm", "simplyscripts": "SimplyScripts"}
SOURCE_LANDING = {"imsdb": "https://imsdb.com", "dailyscript": "https://www.dailyscript.com",
                  "awesomefilm": "https://www.awesomefilm.com", "simplyscripts": "https://www.simplyscripts.com/movie-screenplays.html"}


def _base_title(name: str) -> str:
    """Same rule as contract_checks.base_title (title without year, leading 'the', subtitle, sequel number)."""
    t = re.sub(r"_\d{4}$", "", name).lower()
    t = re.sub(r"^the\s+", "", t)
    t = re.split(r":| - | \u2013 ", t)[0]
    t = re.sub(r"\b(part\s+)?([ivx]+|\d+)\b\s*$", "", t.strip())
    return re.sub(r"[^a-z0-9]", "", t)


def gt_lookalike(title: str, table: dict, gt_names: dict | None = None) -> str | None:
    """GT imdb_id the title looks like under contract C15b: a curated IP keyword in the title, the same base title,
    or a multi-word GT base title (>= 6 chars) inside the film's base title. Such films must be listed in
    gt_related.json (as a relation or as unrelated) before they can be released."""
    low = title.lower()
    for gt_id, kws in table.get("ip_keywords", {}).items():
        if any(re.search(rf"\b{re.escape(kw)}\b", low) for kw in kws):
            return gt_id
    b = _base_title(title)
    for gt_id, gt_name in (gt_names or {}).items():
        gb, plain = _base_title(gt_name), re.sub(r"_\d{4}$", "", gt_name)
        multi = len(plain.split()) > 1 and not re.match(r"^the \S+$", plain.lower())
        if (b == gb and len(gb) >= 3) or (multi and len(gb) >= 6 and gb in b):
            return gt_id
    return None
SOURCE_PRIORITY = ("imsdb", "dailyscript", "awesomefilm", "simplyscripts")
FORMAT_PRIORITY = ("html", "txt", "pdf")
SUPPORTED_FORMATS = set(FORMAT_PRIORITY)
OCR_SYMBOL_RE = re.compile(r"[\\~{}^|]|[_]{2,}[^_\s]|[,.'`]{2,}[_\\]")


_SPEAKER_LED_RE = re.compile(r"^([A-Z][A-Z.'\-]+(?: [A-Z][A-Z.'\-]+)?)\s+(?:\(|[A-Z][a-z']|[!?.,\-])")
_TRAILING_PAGE_NO_RE = re.compile(r"[.!?\"]\s+\d{1,3}\.?$")


def window_noise(seg) -> list[str]:
    """Evaluator R3/C12b/C12c/C13b/P2 on one window: backslash residue, speaker names split by noise
    ("BLAKE \\" next to "BLAKE"), orphan speaker lines, OCR symbol debris."""
    reasons = []
    texts = [(tag, t) for s in seg for tag, t in s.elements]
    if any("\\" in t for _, t in texts):
        reasons.append("backslash_residue")
    names = {speaker_name(t) for tag, t in texts if tag == "character"}
    keys = Counter(re.sub(r"[^A-Z0-9]", "", n) for n in names)
    if any(v > 1 for k, v in keys.items() if k):
        reasons.append("speaker_fission")
    if any(tag == "scene_description" and len(t) < 30 and t.upper() == t and speaker_name(t.rstrip(".:")) in names for tag, t in texts):
        reasons.append("orphan_speaker_line")
    if sum(len(OCR_SYMBOL_RE.findall(t)) for _, t in texts) > 0:
        reasons.append("ocr_symbols")
    if absorbed_action_rate(seg) > 0.10:
        reasons.append("dialogue_action_merged")
    # Limits below are the MovieSum p99 per segment (sample of 1,210 segments), found in the pilot writers' reports.
    talkers = {speaker_name(t) for i, (tag, t) in enumerate(texts)
               if tag == "character" and i + 1 < len(texts) and texts[i + 1][0] in ("dialogue", "parenthetical")}
    descs = [t for tag, t in texts if tag == "scene_description"]
    led = sum(1 for d in descs if (m := _SPEAKER_LED_RE.match(d)) and m.group(1) in talkers)
    if led > 4:
        reasons.append("speaker_in_action")  # "LOIS Superman!": cue and line fused into an action element
    if sum(1 for d in descs if _TRAILING_PAGE_NO_RE.search(d)) > 1:
        reasons.append("page_numbers_in_text")
    if sum(1 for n in talkers if len(n) >= 3 and any(o != n and len(o) == len(n) + 1 and o.endswith(n) for o in talkers)) >= 2:
        reasons.append("drop_cap_names")  # "ENRY" next to "HENRY": first letters lost in a PDF conversion
    if sum(1 for d in descs if len(re.sub(r"[^A-Za-z]", "", d)) <= 2) >= 3:
        reasons.append("margin_letters")
    return reasons


NAME_STOP = {"MR", "MRS", "MS", "DR", "THE", "OLD", "YOUNG", "MAN", "WOMAN", "GIRL", "BOY", "VOICE", "OFFICER", "COP", "GUARD"}


def load_reference(args, cache_path: str):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    rows = load_moviesum(args.moviesum_dir)
    gt = load_gt(args.gt_path)
    gt_ids = {r["imdb_id"] for r in gt}
    vocab = Counter()
    for row in rows:
        vocab.update(re.findall(r"[a-z]{3,}", row["script"].lower()))
    ms_sh, gt_movie_sh, seen = set(), set(), set()
    for row in rows:
        if row["imdb_id"] in seen:
            continue
        seen.add(row["imdb_id"])
        text = " ".join(t for s in parse_script(row["script"]) for _, t in s.elements)
        sh = shingles(words_of(text), CONFIG["ngram"], CONFIG["shingle_sample_mod"])
        ms_sh |= sh
        if row["imdb_id"] in gt_ids:
            gt_movie_sh |= sh
    gt_seg_sh = set()
    for r in gt:
        gt_seg_sh |= shingles(words_of(render(parse_script(r["script_segment"]))), CONFIG["ngram"])
    ref = {"vocab": vocab, "ms_ids": seen, "ms_shingles": ms_sh, "gt_ids": gt_ids, "gt_movie_shingles": gt_movie_sh,
           "gt_segment_shingles": gt_seg_sh}
    with open(cache_path, "wb") as f:
        pickle.dump(ref, f)
    return ref


def candidate_films(matches_path: str, sources: set[str]) -> dict[str, list[dict]]:
    films = defaultdict(list)
    with open(matches_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            src = r["source"].split(":")[0]
            if src not in sources or r["status"] not in ("new", "remake_of_moviesum_title") or not r.get("match"):
                continue
            if r["format"] not in SUPPORTED_FORMATS:
                continue
            films[r["match"]["imdb_id"]].append(r)
    for offers in films.values():
        offers.sort(key=lambda r: (SOURCE_PRIORITY.index(r["source"].split(":")[0]), FORMAT_PRIORITY.index(r["format"]), r["url"]))
    return films


def speaker_tokens(scenes) -> list[set[str]]:
    counts = Counter(speaker_name(t) for s in scenes for tag, t in s.elements if tag == "character")
    out = []
    for name, _ in counts.most_common(10):
        toks = {t for t in re.findall(r"[A-Z][A-Z'\-]{2,}", name) if t not in NAME_STOP}
        if toks:
            out.append(toks)
    return out


def verify_match(scenes, characters: set[str] | None, need: int = 1) -> bool | None:
    """True if >= `need` of the script's top-10 speakers carry a name of the IMDb cast characters."""
    if not characters:
        return None
    return sum(1 for toks in speaker_tokens(scenes) if toks & characters) >= need


def resolve_imsdb(offer: dict, fetcher: Fetcher, index: dict) -> dict:
    """IMSDb detail page -> release year (re-run the IMDb match with it) and the real script link."""
    try:
        page, _ = fetcher.get(offer["page_url"])
    except (RuntimeError, PolicyError):
        return offer
    det = catalogs.imsdb_detail(page.decode("latin-1"))
    offer = dict(offer)
    if det["script_url"]:
        offer["url"] = det["script_url"]
        if norm_title(det["script_title"] or "") != norm_title(offer["title"]):
            offer["mislinked"] = det["script_title"]  # IMSDb links this title to another film's script
    if det["release_year"]:
        offer["release_year"] = det["release_year"]
        m = lookup(index, offer["title"], det["release_year"], "release")
        if m and m["confidence"] == "high":
            offer["match"] = {**m, "matched_title": offer["title"]}
    return offer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matches", default="data_construction/work/extra_survey/catalog_matches.jsonl")
    ap.add_argument("--sources", default=",".join(SOURCE_PRIORITY))
    ap.add_argument("--cache_dir", default="data_construction/work/sources/extra/raw")
    ap.add_argument("--moviesum_dir", default="data_construction/work/sources/moviesum")
    ap.add_argument("--gt_path", default="data_construction/work/sources/cml_bench/gt_100.json")
    ap.add_argument("--imdb_dir", default="data_construction/work/sources/imdb")
    ap.add_argument("--gt_related", default=None, help="GT relation table (default: data_construction/gt_related.json)")
    ap.add_argument("--out", default="data_construction/work/extra_build")
    ap.add_argument("--limit_films", type=int, default=0)
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--offline", action="store_true", help="only use cached downloads")
    args = ap.parse_args()
    cfg = CONFIG
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)

    ref = load_reference(args, os.path.join(args.out, "reference_cache.pkl"))
    gt_table = load_gt_related(args.gt_related)
    gt_names = {r["imdb_id"]: r["movie_name"] for r in load_gt(args.gt_path)}
    reviewed = {r["imdb_id"] for r in gt_table["relations"]} | {r["imdb_id"] for r in gt_table.get("unrelated", [])}
    index = load_index(os.path.join(args.imdb_dir, "title_index.json"))
    films = candidate_films(args.matches, set(args.sources.split(",")))
    order = sorted(films, key=lambda i: (SOURCE_PRIORITY.index(films[i][0]["source"].split(":")[0]), i))
    if args.limit_films:
        order = order[: args.limit_films]
    print(f"{len(order)} candidate films; loading IMDb cast characters", flush=True)
    characters = load_characters(args.imdb_dir, set(order))
    meta_by_id = id_meta(index, set(order))
    fetcher = Fetcher(args.cache_dir, delay=args.delay, offline=args.offline)

    accepted, rejected, film_rows = [], [], []
    stats = Counter()
    extra_sh: set[int] = set()
    for fi, imdb_id in enumerate(order):
        row = {"imdb_id": imdb_id, "offers": [], "outcome": None}
        chosen = None
        for offer in films[imdb_id]:
            src = offer["source"].split(":")[0]
            if src == "imsdb":
                offer = resolve_imsdb(offer, fetcher, index)
            info = {"source": offer["source"], "url": offer["url"], "format": offer["format"]}
            row["offers"].append(info)
            if offer.get("mislinked"):
                info["reasons"] = [f"imsdb_mislinked:{offer['mislinked']}"]
                continue
            try:
                body, rec = fetcher.get(offer["url"])
            except PolicyError as exc:
                info["reasons"] = [f"policy:{exc}"]
                continue
            except RuntimeError as exc:
                info["reasons"] = [f"fetch_failed:{str(exc)[:80]}"]
                continue
            text, pdf_meta = extract_text(body, offer["format"])
            scenes, diag = text_to_scenes(text)
            dropped = {i: "no_body" for i in diag.get("heading_only_scenes", [])}
            scenes = drop_duplicate_scenes(drop_front_matter(scenes), dropped=dropped)
            info["orphan_cues_dropped"] = diag.get("orphan_cues_dropped", 0)
            reasons, metrics = script_quality(text, scenes, diag, pdf_meta, ref["vocab"])
            info.update({"metrics": metrics, "layout": diag.get("layout"), "source_file": rec["file"], "source_sha1": rec["sha1"]})
            match = offer["match"]
            verified = verify_match(scenes, characters.get(match["imdb_id"]), need=2 if match.get("confidence") == "low" else 1)
            info["imdb_match"] = {k: match.get(k) for k in ("imdb_id", "title", "year", "confidence", "matched_title")} | {"verified": verified}
            if not reasons and verified is False and match.get("confidence") not in ("low", "medium"):
                reasons.append("imdb_cast_mismatch")  # e.g. a catalog link labelled with the wrong film
            if not reasons and match.get("confidence") == "low" and match.get("ambiguous"):
                reasons.append("imdb_match_ambiguous")  # several popular films share the title and no year is known
            if not reasons and match.get("confidence") in ("low", "medium") and verified is not True:
                reasons.append("imdb_match_unverified")
            if not reasons and match["imdb_id"] != imdb_id:
                reasons.append("imdb_match_changed")  # the IMSDb release date points to another film
            meta = meta_by_id.get(imdb_id)
            if not reasons and not meta:
                reasons.append("imdb_metadata_missing")
            rel = gt_relation(imdb_id, gt_table)
            if not reasons and rel and rel["type"] == "remake":
                reasons.append("gt_remake")
            look = gt_lookalike(meta["title"], gt_table, gt_names) if not reasons and imdb_id not in reviewed else None
            if look:
                reasons.append(f"gt_lookalike_unreviewed:{look}")
            if not reasons:
                text_all = " ".join(t for s in scenes for _, t in s.elements)
                sh = shingles(words_of(text_all), cfg["ngram"], cfg["shingle_sample_mod"])
                ov = {"gt": overlap(sh, ref["gt_movie_shingles"]), "moviesum": overlap(sh, ref["ms_shingles"]), "extra": overlap(sh, extra_sh)}
                info["overlap"] = {k: round(v, 4) for k, v in ov.items()}
                if ov["gt"] > cfg["max_gt_movie_overlap"]:
                    reasons.append("gt_movie_text_overlap")
                elif ov["moviesum"] > cfg["max_duplicate_movie_overlap"]:
                    reasons.append("duplicate_of_moviesum_text")
                elif ov["extra"] > cfg["max_duplicate_movie_overlap"]:
                    reasons.append("duplicate_extra_text")
            info["reasons"] = reasons
            if not reasons:
                chosen = (offer, rec, scenes, sh, verified, dropped, meta, rel)
                break
        if chosen is None:
            last = row["offers"][-1].get("reasons", []) if row["offers"] else []
            row["outcome"] = "rejected:" + (last[0].split(":")[0] if last else "no_offer")
            stats["films_rejected"] += 1
            stats.update(f"film_reject:{r.split(':')[0]}" for o in row["offers"][-1:] for r in o.get("reasons", []))
            film_rows.append(row)
            continue
        offer, rec, scenes, sh, verified, dropped, meta, gt_rel = chosen
        extra_sh |= sh
        match = offer["match"]
        name = f"{meta['title']}_{meta['year']}"
        wrapper_tokens = count_tokens(render([]))
        prefix = [0]
        for s in scenes:
            prefix.append(prefix[-1] + count_tokens(render([s])) - wrapper_tokens)
        windows = choose_windows(scenes, prefix, cfg)
        kept = 0
        src = offer["source"].split(":")[0]
        for a, b in windows:
            seg = scenes[a:b]
            content = render(seg)
            item_id = f"{imdb_id}-s{seg[0].index:04d}-{seg[-1].index:04d}"
            st = segment_stats(seg, content, ref["vocab"])
            reasons = segment_rejections(st, validate_cml(content), cfg) + window_noise(seg)
            gt_ov = overlap(shingles(words_of(content), cfg["ngram"]), ref["gt_segment_shingles"])
            if gt_ov > cfg["max_gt_segment_overlap"]:
                reasons.append("gt_segment_overlap")
            stats["windows_total"] += 1
            if reasons:
                rejected.append({"item_id": item_id, "movie_name": name, "reasons": reasons, "content_tokens": st["content_tokens"]})
                stats.update(f"reject:{r.split(':')[0]}" for r in reasons)
                continue
            accepted.append({
                "item_id": item_id, "movie_name": name, "imdb_id": imdb_id, "script_segment": content, "summary": "",
                "segment_index": kept, "scene_start": seg[0].index, "scene_end": seg[-1].index,
                "dropped_scenes": [{"scene": i, "reason": dropped[i]} for i in range(seg[0].index, seg[-1].index + 1)
                                   if i not in {s.index for s in seg}],
                "relative_position": round(a / len(scenes), 4),
                "source_dataset": SOURCE_NAMES[src], "source_split": None, "source_url": SOURCE_LANDING[src],
                "source_file": offer["url"], "source_cache_file": rec["file"], "source_sha1": rec["sha1"], "source_format": offer["format"],
                "source_host": offer["source"].split(":", 1)[1] if ":" in offer["source"] else src,
                "source_page_url": offer.get("page_url"), "catalog_title": offer["title"],
                "imdb_url": f"https://www.imdb.com/title/{imdb_id}/", "imdb_rating": meta.get("rating"),
                "imdb_votes": meta.get("votes"), "genres": meta.get("genres", []), "year": meta.get("year"),
                "imdb_match_confidence": match.get("confidence"), "imdb_match_verified": verified,
                "gt_related": gt_rel, "eval_safe": gt_rel is None,
                "content_normalization": NORMALIZATION_VERSION, "content_sha1": hashlib.sha1(content.encode()).hexdigest(),
                "gt_ngram_overlap": round(gt_ov, 5), **st,
            })
            kept += 1
        row.update({"outcome": "accepted" if kept else "accepted_no_windows", "movie_name": name, "source": offer["source"],
                    "gt_related": gt_rel,
                    "scenes": len(scenes), "windows": len(windows), "segments": kept})
        stats["films_accepted"] += 1
        stats["films_with_segments"] += bool(kept)
        film_rows.append(row)
        if (fi + 1) % 25 == 0:
            print(f"[{fi + 1}/{len(order)}] films_with_segments={stats['films_with_segments']} segments={len(accepted)} "
                  f"{time.time() - t0:.0f}s", flush=True)

    def dump(name, items):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    dump("segments.jsonl", accepted)
    for src_name in sorted({s["source_dataset"] for s in accepted}):
        dump(f"segments_{src_name.lower()}.jsonl", [s for s in accepted if s["source_dataset"] == src_name])
    dump("films.jsonl", film_rows)
    dump("rejected_windows.jsonl", rejected)

    def dist(xs):
        xs = sorted(xs)
        return {p: xs[min(len(xs) - 1, int(p / 100 * len(xs)))] for p in (0, 5, 10, 25, 50, 75, 90, 95, 100)} if xs else {}

    per_movie = Counter(s["imdb_id"] for s in accepted)
    build_stats = {
        "config": cfg, "quality_config": QUALITY_CONFIG, "normalization": NORMALIZATION_VERSION,
        "candidate_films": len(order), "films_accepted": stats["films_accepted"], "films_with_segments": stats["films_with_segments"],
        "films_rejected": stats["films_rejected"],
        "film_rejections": {k[12:]: v for k, v in stats.items() if k.startswith("film_reject:")},
        "film_outcomes": dict(Counter(r["outcome"] for r in film_rows)),
        "windows_total": stats["windows_total"], "segments_accepted": len(accepted),
        "window_rejections": {k[7:]: v for k, v in stats.items() if k.startswith("reject:")},
        "segments_by_source": dict(Counter(s["source_dataset"] for s in accepted)),
        "films_by_source": dict(Counter(r["source"].split(":")[0] for r in film_rows if r["outcome"] == "accepted")),
        "segments_by_format": dict(Counter(s["source_format"] for s in accepted)),
        "gt_related_segments": sum(1 for s in accepted if s["gt_related"]),
        "gt_related_films": sorted({(s["movie_name"], s["gt_related"]["gt_movie"]) for s in accepted if s["gt_related"]}),
        "gt_related_table": gt_table["version"],
        "gt_lookalikes_unreviewed": sorted({(r["imdb_id"], o["imdb_match"]["title"], rr.split(":", 1)[1]) for r in film_rows
                                            for o in r["offers"] for rr in o.get("reasons", []) if rr.startswith("gt_lookalike")}),
        "segments_per_movie": dist(list(per_movie.values())),
        "content_tokens": dist([s["content_tokens"] for s in accepted]),
        "num_scenes": dist([s["num_scenes"] for s in accepted]),
        "total_content_tokens": sum(s["content_tokens"] for s in accepted),
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "build_stats.json"), "w") as f:
        json.dump(build_stats, f, indent=2)
    print(json.dumps({k: v for k, v in build_stats.items() if k not in ("config", "quality_config")}, indent=2))


if __name__ == "__main__":
    main()
