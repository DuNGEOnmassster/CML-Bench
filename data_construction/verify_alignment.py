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
  - words of page furniture (watermarks, running headers/footers, rising page numbers glued to element ends; C12e)
    may be deleted from any element;
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
MARKER = re.compile(r"\(\s*MORE\s*\)|\(?\b(CONTINUED|OMITTED|OMIT)\b\)?\s*:?(?:\s*\(\s*\d+\s*\))?|\bCONT\s*['\u2019`]?\s*D\b")
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


def deletable_words(els, k: int, talkers: set[str], furn=None) -> Counter:
    """Multiset of words of element k that the cleaner may delete. `furn(j)`: page-furniture words of element j."""
    furn = furn or (lambda j: Counter())
    tag, raw = els[k]
    text = re.sub(r"[\\*]+", " ", norm(raw))
    allowed = Counter()
    for m in PTB.finditer(text):
        allowed.update(words(m.group()))
    text = text.replace("-LRB-", "(").replace("-RRB-", ")")
    for m in (MARKER if tag == "dialogue" else MARKER_ANY_CASE).finditer(text):
        allowed.update(words(m.group()))
    edge = re.match(r"^\s*(\d{1,3}-?[A-Z]{0,3})\s+(.+)\s+\1\s*$", text)
    if edge:
        allowed.update(words(edge.group(1)) * 2)
        text = edge.group(2)
    for m in DUP_LETTERED_NUMBER.finditer(text):
        allowed.update(words(m.group()))
    if tag == "dialogue":
        return Counter(words(raw)) if page_number_line(els, k, talkers) else allowed
    if NUMBER_ONLY.match(SCENE_NUMBER.sub(" ", MARKER_ANY_CASE.sub(" ", PTB.sub(" ", FURN_WATERMARK.sub(" ", text))))):
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
        # a line that is nothing but page furniture is no line either
        no_line = not (j < len(els) and els[j][0] == "dialogue" and has_line(els[j][1])
                       and (Counter(words(els[j][1])) - furn(j)))
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
                and (not has_line(els[j][1]) or page_number_line(els, j, talkers) or not (Counter(words(els[j][1])) - furn(j))):
            return Counter(words(raw))
    bare = MARKER_ANY_CASE.sub(" ", re.sub(r"[\\*]+", " ", norm(raw))).strip(" :")
    if tag == "scene_description" and bare.isupper() and bare.upper() == name_of(bare) and bare in talkers:
        return Counter(words(raw))
    return allowed


FURN_WATERMARK = re.compile(
    r"©\s*\d{0,4}\s*DISNEY\s*/?\s*PIXAR(?:\s*-\s*PRIVILEGED AND CONFIDENTIAL)?|©\s*\d{0,4}\s*MARVEL STUDIOS,?\s*INC\.?|"
    r"©\s*MARVEL\b|©\s*\d{0,4}\s*CTMG\.?(?:\s*All Rights Reserved\.?)?|©[^.<]{0,50}?All Rights Reserved\.?|"
    r"\bPRIVILEGED AND CONFIDENTIAL\b|(?:\b\d?FLIX(?:\.COM|\s+INSTITUTE)\s*)?\bSCREENPLAY DATABASE\b(?:\s*(?:\d{8}|\d{2}\.\d{2}\.\d{4}))?|"
    r"\bFOR EDUCATIONAL (?:USE|PURPOSES) ONLY\b|\bNO DUPLICATION WITHOUT [A-Z']+ WRITTEN CONSENT\.?(?:\s*\(\s*\d+\s*\))?|"
    r"=*\s*\bScript\s*Fly\.com\b\s*=*|"
    # revision stamps, e.g. "5/1/91 BLUE (2) 39.", ")P( 5/1/91 BLUE", "Rev. 12/11/00 (Grey) 76.", "Rev. Blue 4/30/01 19.",
    # "Yellow 05/14/2001 ' 62.", "Rev. 02/06/01 (2nd Yellow) 56.", "Cherry Rev. (Nov 16 '17) - 249"
    r"(?:\)[A-Z]\(\s*)?\bRev(?:ised|\.)?\s*(?:(?:\d(?:st|nd|rd|th)\s+)?(?:WHITE|BLUE|PINK|YELLOW|GREEN|GOLDENROD|BUFF|SALMON|CHERRY|TAN|GREY|GRAY|IVORY|LAVENDER)\s+)?\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}"
    r"(?:\s*\((?:\d(?:st|nd|rd|th)\s+)?[A-Za-z]+\))*(?:\s+'?\s*\d{1,3}[A-Z]?\s*\.)?|"
    r"(?:\)[A-Z]\(\s*)?\b(?:WHITE|BLUE|PINK|YELLOW|GREEN|GOLDENROD|BUFF|SALMON|CHERRY|TAN|GREY|GRAY|IVORY|LAVENDER)\s+'?\d{1,2}/\d{1,2}/\d{2,4}(?:\s*')?(?:\s*\(\s*\d+\s*\))?(?:\s+'?\s*\d{1,3}[A-Z]?\s*\.)?|"
    r"(?:\)[A-Z]\(\s*)?\b\d{1,2}/\d{1,2}/\d{2,4}\s+\(?(?:\d(?:st|nd|rd|th)\s+)?(?:WHITE|BLUE|PINK|YELLOW|GREEN|GOLDENROD|BUFF|SALMON|CHERRY|TAN|GREY|GRAY|IVORY|LAVENDER)\)?(?:\s*\(\s*\d+\s*\))?(?:\s+\d{1,3}[A-Z]?\s*\.)?|"
    r"\b(?:WHITE|BLUE|PINK|YELLOW|GREEN|GOLDENROD|BUFF|SALMON|CHERRY|TAN|GREY|GRAY|IVORY|LAVENDER)\s+Rev(?:ision|\.)?\s*\([^)]{3,20}\)(?:\s*-\s*\d{1,3}[A-Z]?\.?)?|©",
    re.I,
)
FURN_SIGNAL_WORDS = {"draft", "rev", "revised", "revision", "final", "script", "screenplay", "adapted", "copyright",
                     "confidential", "duplication"}
NUMBERISH = re.compile(r"\d+[a-z]?")
PAGE_TAIL = re.compile(r"(?<=[.!?\"')])\s+(\d{1,3})\s*\.\s*$")


def _furniture_key(toks: list[str]) -> tuple | None:
    key = tuple("#" if NUMBERISH.fullmatch(t) else t for t in toks)
    if "#" not in key or (key[0] != "#" and key[-1] != "#"):
        return None
    alpha = [t for t in key if t != "#" and not t.isdigit() and len(t) >= 2]
    if len(alpha) < 2 or sum(len(t) for t in alpha) < 8:
        return None
    s = " ".join(key)
    has_date = "# # #" in s
    by_credit = any(key[i] == "by" and i + 2 < len(key) and key[i + 1] != "#" and key[i + 2] != "#" for i in range(len(key)))
    if not (has_date or by_credit or (FURN_SIGNAL_WORDS & set(key)) or ("work" in key and "file" in key)):
        return None
    return key


def furniture_templates(elements: list[tuple[str, str]]) -> set[tuple]:
    """Token templates of running headers/footers: an edge phrase of >= 3 description/dialogue/parenthetical elements
    that differs only in its numbers (>= 2 distinct), has the number at its edge and a header signal."""
    count, digits = defaultdict(set), defaultdict(set)
    for i, (tag, text) in enumerate(elements):
        if tag in ("stage_direction", "character") or not re.search(r"\d", text):
            continue
        toks = words(MARKER_ANY_CASE.sub(" ", FURN_WATERMARK.sub(" ", norm(text))))
        for m in range(3, min(16, len(toks)) + 1):
            for win in (toks[:m], toks[-m:]):
                key = _furniture_key(win)
                if key:
                    count[key].add(i)
                    digits[key].add(tuple(t for t in win if NUMBERISH.fullmatch(t)))
    keys = {k for k in count if len(count[k]) >= 3 and len(digits[k]) >= 2}
    return {k for k in keys if not any(k != o and len(o) > len(k) and _contains(o, k) and len(count[o]) >= len(count[k]) for o in keys)}


def _contains(hay: tuple, needle: tuple) -> bool:
    n = len(needle)
    return any(hay[i : i + n] == needle for i in range(len(hay) - n + 1))


def _template_hits(toks: list[str], templates: set[tuple]) -> list[int]:
    """Token positions covered by a template occurrence (digits match any number)."""
    pos = set()
    norm_toks = ["#" if NUMBERISH.fullmatch(t) else t for t in toks]
    for k in templates:
        n = len(k)
        for i in range(len(toks) - n + 1):
            if tuple(norm_toks[i : i + n]) == k:
                pos.update(range(i, i + n))
    return sorted(pos)


def page_tail_positions(scenes) -> set[tuple[int, int]]:
    tails = [(si, k, int(m.group(1))) for si, els in enumerate(scenes) for k, (_, t) in enumerate(els) if (m := PAGE_TAIL.search(norm(t)))]
    rising = sum(b[2] >= a[2] for a, b in zip(tails, tails[1:]))
    return {(si, k) for si, k, _ in tails} if len(tails) >= 3 and rising >= 0.7 * (len(tails) - 1) else set()


def furniture_words(scenes) -> dict[tuple[int, int], Counter]:
    """Per raw element, the words that belong to page furniture (watermarks, running headers, rising page numbers)."""
    flat = [(si, k, tag, text) for si, els in enumerate(scenes) for k, (tag, text) in enumerate(els)]
    templates = furniture_templates([(tag, text) for _, _, tag, text in flat])
    tails = page_tail_positions(scenes)
    out = {}
    for si, k, tag, text in flat:
        c = Counter()
        for m in FURN_WATERMARK.finditer(re.sub(r"[\\*]+", "", norm(text))):
            c.update(words(m.group()))
        if templates and re.search(r"\d", text):
            toks = words(MARKER_ANY_CASE.sub(" ", FURN_WATERMARK.sub(" ", norm(text))))
            c.update(toks[i] for i in _template_hits(toks, templates))
        if (si, k) in tails:
            c.update(words(PAGE_TAIL.search(norm(text)).group(1)))
        if c:
            out[(si, k)] = c
    return out


def furniture_findings(elements: list[tuple[str, str]]) -> dict:
    """C12e on released text of one film: watermark strings, running-header templates, rising page-number tails."""
    wm = sum(len(FURN_WATERMARK.findall(norm(t))) for _, t in elements)
    templates = furniture_templates(elements)
    tails = page_tail_positions([elements])
    return {"watermarks": wm, "templates": sorted(" ".join(k) for k in templates), "page_tails": len(tails)}


def body_words(els) -> list[str]:
    """Body words without markers and scene numbers (a revised repeat of a scene differs only in those)."""
    out = []
    for t, x in els:
        if t != "stage_direction":
            out.extend(words(SCENE_NUMBER.sub(" ", MARKER_ANY_CASE.sub(" ", PTB.sub(" ", re.sub(r"[\\*]+", " ", norm(x)))))))
    return out


def verify_item(item: dict, scenes, talkers: set[str], furniture: dict | None = None) -> dict:
    furniture = furniture or {}
    a, b = item["scene_start"], item["scene_end"]
    dropped = {d["scene"]: d["reason"] for d in item.get("dropped_scenes") or []}
    problems = []
    if b >= len(scenes):
        return {"item_id": item["item_id"], "pass": False, "problems": ["range_outside_script"]}
    for idx, reason in dropped.items():
        els = scenes[idx]
        if reason == "no_body":
            leftover = [w for k, (t, x) in enumerate(els) if t != "stage_direction"
                        for w in (Counter(words(x)) - deletable_words(els, k, talkers, lambda j, i=idx: furniture.get((i, j), Counter()))
                                  - furniture.get((idx, k), Counter())).elements()]
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
            allowance[idx].update(deletable_words(els, k, talkers, lambda j, i=idx: furniture.get((i, j), Counter())))
            allowance[idx].update(furniture.get((idx, k), Counter()))
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
    furniture = furniture_words(scenes)
    out = []
    for it in items:
        res = verify_item(it, scenes, talkers, furniture)
        if row["movie_name"] != (it.get("source_label") or it)["movie_name"]:
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
        # A relabelled item (C33) is aligned with the MovieSum row it came from.
        groups[((it.get("source_label") or it)["imdb_id"], it["source_split"])].append(it)
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
