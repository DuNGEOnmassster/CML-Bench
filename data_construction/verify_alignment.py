"""Independent source alignment for every item (contract C05') and GT 8-gram overlap (C16b).

  python data_construction/verify_alignment.py --items SEGMENTS_OR_RELEASE.jsonl [--out report.json]

Deliberately imports nothing from cml_format.py / build_segments.py: it re-reads the MovieSum screenplay, takes raw
scenes scene_start..scene_end, and aligns lowercase alphanumeric words of the raw text with the released text
(difflib, no autojunk). An item passes C05' when
  - no word is inserted;
  - every deleted word is whitelisted: page-break/revision markers ((MORE), CONTINUED, CONT'D, OMITTED), scene or
    page numbers (margin/heading numbers, number-only non-dialogue elements), PTB bracket tokens (-LRB-), and bare
    speaker names that never get a line (a <character> whose dialogue is empty/missing, with its parentheticals, or a
    <scene_description> that is exactly the name of a speaker who talks elsewhere in the script);
  - no word of a non-empty <dialogue> is deleted unless it is a marker, or the whole line is a bare number with no
    speaker before it or a "speaker" who never says anything with letters (a page number under a revision stamp);
  - raw scenes in range == num_scenes + len(dropped_scenes), and each dropped scene is really body-less ("no_body")
    or a verbatim repeat of an earlier scene ("duplicate").
The declared normalizations it tolerates: HTML entities, NFKC, and spacing-only differences (the same letters split or
joined differently: "do n't" -> "don't", "gon na" -> "gonna", "J·im" -> "Jim").
"""
from __future__ import annotations

import argparse
import difflib
import html
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from multiprocessing import Pool

SCENE = re.compile(r"<scene>(.*?)</scene>", re.S)
ELEMENT = re.compile(r"<([a-z_]+)>(.*?)</\1>|<([a-z_]+)\s*/>", re.S)
NOISE = "\u2022\u25a0\u25aa\u00b7\u25cf\u25a1\u2023\u2043"
MERGES = {"gonna": ("gon", "na"), "wanna": ("wan", "na"), "gotta": ("got", "ta"), "lemme": ("lem", "me"), "gimme": ("gim", "me")}
MARKER = re.compile(r"\(\s*MORE\s*\)|\(?\b(CONTINUED|OMITTED|OMIT)\b\)?|\bCONT\s*['\u2019`]?\s*D\b")
MARKER_ANY_CASE = re.compile(MARKER.pattern, re.I)
NUMBER_LINE = re.compile(r"^\d{1,4}[A-Z]?\.?$")
MARKER_WORDS = {"more", "continued", "omitted", "omit", "cont", "d", "lrb", "rrb", "lsb", "rsb", "lcb", "rcb"}
PTB = re.compile(r"-[LR][RSC]B-")
DUP_NUMBER = re.compile(r"\b(\d{1,3}-?[A-Z]{0,3}|\d{1,3}pt)\s+(?=.*\b\1\b)")
SCENE_NUMBER = re.compile(r"\b\d{1,3}-?[A-Z]{0,3}\b|\b\d{1,3}pt\b")
HEADING_NUMBERS = re.compile(
    r"^\s*(\d{1,3}-?[A-Z]{0,3})\.?\s+(?=(?:INT|EXT|I/E)\b)|\b(?:DAY|NIGHT|MORNING|EVENING|AFTERNOON|DAWN|DUSK|LATER|SUNSET|"
    r"SUNRISE|CONTINUOUS)\s+(\d{1,3}-?[A-Z]{0,3})\s*\.?\s*$"
)
DUP_LETTERED_NUMBER = re.compile(r"\b(\d{1,3}-?[A-Z]{1,3}|\d{1,3}pt)\s+\1\b")
# Speaker tags that are revision stamps, headings, camera directions or captions, not people.
NOT_A_NAME = re.compile(
    r"[\d/:]|-$|\b(CUT|DISSOLVE|FADE|FLASH|SMASH|MATCH|WIPE|INTERCUT|ANGLE|INT|EXT|LATER|CONTINUOUS|SAME|TITLE|LEGEND|"
    r"SUPER|INSERT|POV|MONTAGE|SERIES|CLOSE|WIDE|SHOT|BACK|DAY|NIGHT|MORNING|EVENING|CONTINUED|OMITTED|MORE)\b|TO$"
)
NUMBER_ONLY = re.compile(r"^[\W\d_]*$")
SUFFIX = re.compile(r"\s*\([^)]*\)\s*|\s+(V\.?O\.?|O\.?S\.?|O\.?C\.?|CONT'?D\.?)\s*$", re.I)


def norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", html.unescape(html.unescape(text)))
    t = re.sub(r"[\u0000-\u0008\u000b-\u001f\u007f-\u009f\ufffd]", "", t)
    t = re.sub(rf"[{NOISE}]+", " ", t)
    return t.replace("n't", " n't").replace("N'T", " N'T")


def words(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", norm(text).lower()):
        out.extend(MERGES.get(w, (w,)))
    return out


def name_of(text: str) -> str:
    return SUFFIX.sub(" ", re.sub(r"[\\*]+", " ", norm(text))).strip().upper()


def raw_scenes(script: str) -> list[list[tuple[str, str]]]:
    return [[(m.group(1) or m.group(3), m.group(2) or "") for m in ELEMENT.finditer(s)] for s in SCENE.findall(script)]


def has_line(text: str) -> bool:
    return bool(words(DUP_LETTERED_NUMBER.sub(" ", MARKER.sub(" ", norm(text)))))


def talking_speakers(scenes) -> set[str]:
    """Names that say at least one line containing letters somewhere in the script."""
    out = set()
    for els in scenes:
        for k, (tag, text) in enumerate(els):
            if tag != "character":
                continue
            j = k + 1
            while j < len(els) and els[j][0] == "parenthetical":
                j += 1
            name = name_of(text)
            if j < len(els) and els[j][0] == "dialogue" and has_line(els[j][1]) and re.search(r"[A-Za-z]", els[j][1]) \
                    and re.search(r"[A-Z]{2}", name) and not NOT_A_NAME.search(name):
                out.add(name)
    return out


def page_number_line(els, k: int, talkers: set[str]) -> bool:
    """A dialogue that is a bare number nobody real says: no speaker before it, or a speaker who never talks."""
    if els[k][0] != "dialogue" or not NUMBER_LINE.match(MARKER.sub(" ", norm(els[k][1])).strip(" :")):
        return False
    i = k - 1
    while i >= 0 and (els[i][0] == "parenthetical" or (els[i][0] != "dialogue" and not re.search(r"[A-Za-z]", els[i][1]))):
        i -= 1
    return i < 0 or els[i][0] != "character" or name_of(els[i][1]) not in talkers


def deletable_words(els, k: int, talkers: set[str]) -> Counter:
    """Multiset of words of element k that the cleaner may delete."""
    tag, raw = els[k]
    text = re.sub(r"[\\*]+", " ", norm(raw))
    allowed = Counter()
    for m in (MARKER if tag == "dialogue" else MARKER_ANY_CASE).finditer(text):
        allowed.update(words(m.group()))
    for m in PTB.finditer(text):
        allowed.update(words(m.group()))
    edge = re.match(r"^\s*(\d{1,3}-?[A-Z]{0,3})\s+(.+)\s+\1\s*$", text)
    if edge:
        allowed.update(words(edge.group(1)) * 2)
        text = edge.group(2)
    for m in DUP_LETTERED_NUMBER.finditer(text):
        allowed.update(words(m.group()))
    if tag == "dialogue":
        return Counter(words(raw)) if page_number_line(els, k, talkers) else allowed
    if NUMBER_ONLY.match(SCENE_NUMBER.sub(" ", MARKER_ANY_CASE.sub(" ", PTB.sub(" ", text)))):
        return Counter(words(raw))
    for m in DUP_NUMBER.finditer(text):
        if not DUP_LETTERED_NUMBER.match(m.group(1) + " " + m.group(1)):
            allowed.update(words(m.group()) * 2)
    if tag == "stage_direction":
        for m in HEADING_NUMBERS.finditer(text):
            allowed.update(words(m.group(1) or m.group(2) or ""))
        lettered = Counter(re.findall(r"\b\d{1,3}-?[A-Z]{1,3}\b", norm(raw)))
        for num, c in lettered.items():
            if c >= 2:
                allowed.update(words(num) * c)
    def skippable(x):  # parentheticals, and letterless non-dialogue debris the cleaner removes first ("911!", "75")
        return els[x][0] == "parenthetical" or (els[x][0] != "dialogue" and not re.search(r"[A-Za-z]", els[x][1]))

    if tag == "character":
        j = k + 1
        while j < len(els) and skippable(j):
            j += 1
        no_line = not (j < len(els) and els[j][0] == "dialogue" and has_line(els[j][1]))
        if no_line or (j < len(els) and page_number_line(els, j, talkers)):
            return Counter(words(raw))
    if tag == "parenthetical":
        i = k - 1
        while i >= 0 and els[i][0] == "parenthetical":
            i -= 1
        j = k + 1
        while j < len(els) and skippable(j):
            j += 1
        if i >= 0 and els[i][0] == "character" and j < len(els) and els[j][0] == "dialogue" \
                and (not has_line(els[j][1]) or page_number_line(els, j, talkers)):
            return Counter(words(raw))
    bare = MARKER_ANY_CASE.sub(" ", re.sub(r"[\\*]+", " ", norm(raw))).strip(" :")
    if tag == "scene_description" and bare.isupper() and bare.upper() == name_of(bare) and bare in talkers:
        return Counter(words(raw))
    return allowed


def body_words(els) -> list[str]:
    """Body words without markers and scene numbers (a revised repeat of a scene differs only in those)."""
    out = []
    for t, x in els:
        if t != "stage_direction":
            out.extend(words(SCENE_NUMBER.sub(" ", MARKER_ANY_CASE.sub(" ", PTB.sub(" ", re.sub(r"[\\*]+", " ", norm(x)))))))
    return out


def verify_item(item: dict, scenes, talkers: set[str]) -> dict:
    a, b = item["scene_start"], item["scene_end"]
    dropped = {d["scene"]: d["reason"] for d in item.get("dropped_scenes") or []}
    problems = []
    if b >= len(scenes):
        return {"item_id": item["item_id"], "pass": False, "problems": ["range_outside_script"]}
    for idx, reason in dropped.items():
        els = scenes[idx]
        if reason == "no_body":
            leftover = [w for k, (t, x) in enumerate(els) if t != "stage_direction"
                        for w in (Counter(words(x)) - deletable_words(els, k, talkers)).elements()]
            if leftover:
                problems.append(f"dropped_scene_has_body:{idx}")
        elif reason == "duplicate":
            body = body_words(els)
            if not any(body_words(scenes[e]) == body for e in range(idx)):
                problems.append(f"dropped_scene_not_duplicate:{idx}")
        else:
            problems.append(f"dropped_scene_reason:{idx}")
    in_range = list(range(a, b + 1))
    released = SCENE.findall(item["script_segment"])
    if len(in_range) != len(released) + len(dropped) or len(released) != item["num_scenes"]:
        problems.append(f"scene_count:raw={len(in_range)},released={len(released)},dropped={len(dropped)}")

    # Allowances are pooled per scene: difflib may attribute a deleted word to a neighbouring element with the same
    # word (deleting the speaker "COLONEL" can show up as the "Colonel?" ending the previous line).
    raw_tokens, owner = [], []
    allowance = {}
    for idx in in_range:
        if idx in dropped:
            continue
        els = scenes[idx]
        allowance[idx] = Counter()
        for k, (tag, text) in enumerate(els):
            ws = words(text)
            raw_tokens.extend(ws)
            owner.extend([(idx, k)] * len(ws))
            allowance[idx].update(deletable_words(els, k, talkers))
    rel_tokens = [w for s in released for m in ELEMENT.finditer(s) for w in words(m.group(2) or "")]
    sm = difflib.SequenceMatcher(None, raw_tokens, rel_tokens, autojunk=False)
    inserted, deleted, unexplained, dialogue_deleted = 0, 0, [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal" or (op == "replace" and "".join(raw_tokens[i1:i2]) == "".join(rel_tokens[j1:j2])):
            continue
        inserted += j2 - j1
        for i in range(i1, i2):
            deleted += 1
            key, tok = owner[i], raw_tokens[i]
            if allowance[key[0]][tok] > 0:
                allowance[key[0]][tok] -= 1
                continue
            # Numbers and markers repeat across scenes ("130" in a heading and in a description): borrow item-wide.
            if re.search(r"\d", tok) or tok in MARKER_WORDS:
                donor = next((s for s, a in allowance.items() if a[tok] > 0), None)
                if donor is not None:
                    allowance[donor][tok] -= 1
                    continue
            tag, text = scenes[key[0]][key[1]]
            (dialogue_deleted if tag == "dialogue" else unexplained).append(f"s{key[0]}:{tag}:'{raw_tokens[i]}' in {text[:60]}")
    if inserted:
        problems.append(f"inserted_words:{inserted}")
    if unexplained:
        problems.append(f"unexplained_deletions:{len(unexplained)}")
    if dialogue_deleted:
        problems.append(f"dialogue_deleted:{len(dialogue_deleted)}")
    return {"item_id": item["item_id"], "pass": not problems, "problems": problems, "raw_words": len(raw_tokens),
            "deleted_words": deleted, "inserted_words": inserted,
            "examples": sorted(set(unexplained))[:5] + sorted(set(dialogue_deleted))[:5]}


def ngrams(ws: list[str], n: int = 8) -> set[tuple]:
    return {tuple(ws[i : i + n]) for i in range(len(ws) - n + 1)}


_GT8: set = set()
_ROWS: dict = {}


def verify_movie(args):
    key, items = args
    row = _ROWS[key]
    scenes = raw_scenes(row["script"])
    talkers = talking_speakers(scenes)
    out = []
    for it in items:
        res = verify_item(it, scenes, talkers)
        if row["movie_name"] != it["movie_name"]:
            res["pass"] = False
            res["problems"].append("movie_name_mismatch")
        g = ngrams(words(re.sub(r"<[^>]+>", " ", it["script_segment"])))
        res["gt_8gram_overlap"] = round(len(g & _GT8) / max(1, len(g)), 5)
        out.append(res)
    return out


def verify(items: list[dict], moviesum_dir: str, gt_path: str, workers: int) -> tuple[dict, list[dict]]:
    global _GT8
    with open(gt_path, encoding="utf-8") as f:
        gt = [json.loads(line) for line in f if line.strip()]
    _GT8 = set()
    for g in gt:
        _GT8 |= ngrams(words(re.sub(r"<[^>]+>", " ", g["script_segment"])))
    groups = defaultdict(list)
    for it in items:
        groups[(it["imdb_id"], it["source_split"])].append(it)
    _ROWS.clear()
    # An imdb_id can occur twice in one split; like the builder, the longer screenplay is the source.
    for split in ("train", "val", "test"):
        with open(os.path.join(moviesum_dir, f"{split}.jsonl"), encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                key = (row["imdb_id"], split)
                if key in groups and (key not in _ROWS or len(row["script"]) > len(_ROWS[key]["script"])):
                    _ROWS[key] = row
    # Workers are forked after _ROWS/_GT8 are filled, so they share them without pickling.
    with Pool(workers) as pool:
        results = [r for rs in pool.imap_unordered(verify_movie, sorted(groups.items()), chunksize=4) for r in rs]
    fails = [r for r in results if not r["pass"]]
    ov = [r["gt_8gram_overlap"] for r in results]
    summary = {
        "items": len(results),
        "c05_pass": len(results) - len(fails),
        "c05_fail": len(fails),
        "inserted_words": sum(r.get("inserted_words", 0) for r in results),
        "deleted_words": sum(r.get("deleted_words", 0) for r in results),
        "raw_words": sum(r.get("raw_words", 0) for r in results),
        "problem_kinds": dict(Counter(p.split(":")[0] for r in fails for p in r["problems"])),
        "c16b_max_gt_8gram_overlap": max(ov) if ov else 0,
        "c16b_pass": (max(ov) if ov else 0) <= 0.01,
    }
    return summary, fails


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True, help="segments.jsonl, items.jsonl or a release jsonl")
    ap.add_argument("--moviesum_dir", default="data_construction/work/sources/moviesum")
    ap.add_argument("--gt_path", default="data_construction/work/sources/cml_bench/gt_100.json")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--out")
    args = ap.parse_args()
    with open(args.items, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    summary, fails = verify(items, args.moviesum_dir, args.gt_path, args.workers)
    print(json.dumps(summary, indent=1))
    for r in fails[:10]:
        print(r["item_id"], r["problems"], r.get("examples"))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "failures": fails}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
