"""Contract C05'' (label and speaker preservation). Reference implementation by the independent evaluator
(internal/dataset-expansion/evaluator/repilot/dialogue_preservation.py); shares no code with the cleaner.
Adaptations, each marked "C05'' adaptation" below: relabelled items (C33) are read from their source_label row; words of
page furniture (C12e, found by verify_alignment's independent detector) are removed from raw lines first; long lines
also match when only the spacing differs ("fre·e" -> "free"); a bare-number line under no speaker or under a "speaker"
who never says anything with letters is a page number (expected), as declared in C05'.

For every item: each raw MovieSum <dialogue> in scenes scene_start..scene_end (minus dropped_scenes) that has letters
or digits (not only page/revision markers) must appear, in order, inside a released <dialogue> element, under a
released <character> whose name equals the raw speaker's name (suffixes like (CONT'D)/(V.O.) ignored).
Reports per item: missing lines, lines moved into another tag, speaker changed, and short lines ("More.", "478.", "?").
"""
import html
import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from multiprocessing import Pool

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from verify_alignment import furniture_words  # noqa: E402

SCENE = re.compile(r"<scene>(.*?)</scene>", re.S)
EL = re.compile(r"<([a-z_]+)>(.*?)</\1>", re.S)
# C05'' adaptation: the "(2)" counter after CONTINUED is a marker too.
MARK = re.compile(r"\(\s*MORE\s*\)|\(?\b(CONTINUED|OMITTED|OMIT)\b\)?\s*:?(?:\s*\(\s*\d+\s*\))?|\bCONT\s*['\u2019`]?\s*D\b\.?|-[LR][RSC]B-")
NOTNAME = re.compile(r"\d|\b(CUT|CLOSE|CLOSEUP|ANGLE|SHOT|INT|EXT|REV|REVISED|OMITTED|BACK|INTERCUT|INSERT|POV|DISSOLVE|FADE|WIPE|SCENE|CAMERA|VIEW|MONTAGE|SERIES|TITLE|SUPER|PAN|TRUCK|DOLLY|LATER|CONTINUOUS|DAY|NIGHT|MORNING|EVENING|WIDE|MED|FULL|LONG|MOVING|TWO|THREE|GROUP|FAVORING|ON|INSERT|NEWSPAPER)\b|YELLOW|PINK|BLUE|GREEN|GOLDENROD|SALMON|CHERRY|BUFF")
MERGE = {"gonna": "gon na", "wanna": "wan na", "gotta": "got ta", "lemme": "lem me", "gimme": "gim me"}


# C05'' adaptation: title/caption directions and heading fragments used as "speakers" are not names either.
EXTRA_NOTNAME = re.compile(r"\b(SUPERIMPOSE|SAME|OMIT|THE END|TITLE CARD|LEGEND)\b")
# C05'' adaptation: shooting-script scene numbers printed on both margins ("95-B ... 95-B") are not part of the line.
MARGIN = re.compile(r"^\s*(\d{1,3}-?[A-Z]{0,3})\s+(.+?)\s+\1\s*$", re.S)


# C05'' adaptation: doubled shooting-script scene numbers ("149pt 149pt", "75A 75A") are not part of the line.
DUP_NO = re.compile(r"\b(\d{1,3}-?[A-Z]{1,3}|\d{1,3}pt)\s+\1\b")


def key(t):
    t = DUP_NO.sub(" ", MARGIN.sub(r"\2", re.sub(r"[\\*]+", " ", t)))
    t = unicodedata.normalize("NFKC", html.unescape(html.unescape(t)))
    t = MARK.sub(" ", t).replace("\ufffd", "").replace("\u2022", "'").replace("\u00b7", " ").replace("\u2019", "'").replace("n't", " n't").replace("N'T", " N'T")
    ws = re.findall(r"[a-z0-9]+", t.lower())
    out = []
    for w in ws:
        out.extend(MERGE.get(w, w).split())
    return out


def spk(t):
    t = unicodedata.normalize("NFKC", html.unescape(t))
    t = re.sub(r"\s*\([^)]*\)\s*", " ", t)
    t = re.sub(r"\b(V\.?O|O\.?S|O\.?C|CONT'?D|CONTINUED)\b\.?", " ", t, flags=re.I)
    return " ".join(re.findall(r"[A-Z0-9]+", t.upper()))


def contains(hay, needle):
    if not needle:
        return True
    n = len(needle)
    first = needle[0]
    for i in range(len(hay) - n + 1):
        if hay[i] == first and hay[i:i + n] == needle:
            return True
    return False


def contains_joined(hay, needle):
    """C05'' adaptation: the same letters with different spacing (noise bullets split or join words)."""
    return len(needle) > 2 and "".join(needle) in "".join(hay)


def strip_tokens(tokens, remove):
    """C05'' adaptation: drop page-furniture words (counted multiset), scanning from the end of the line."""
    remove = Counter(remove)
    out = []
    for w in reversed(tokens):
        if remove[w] > 0:
            remove[w] -= 1
            continue
        out.append(w)
    return out[::-1]


_ROWS = {}


def check_movie(args):
    k, items = args
    row = _ROWS[k]
    raw = [[(m.group(1), m.group(2)) for m in EL.finditer(s)] for s in SCENE.findall(row["script"])]
    # C05'' adaptation: furniture words per raw element, and speakers who say something with letters somewhere.
    furniture = furniture_words(raw)
    talkers = set()
    for els in raw:
        for a, (t, x) in enumerate(els):
            if t == "character" and a + 1 < len(els) and els[a + 1][0] == "dialogue" and re.search(r"[A-Za-z]", els[a + 1][1]) \
                    and not NOTNAME.search(spk(x)) and not EXTRA_NOTNAME.search(spk(x)) and not re.search(r"[!?]", x):
                talkers.add(spk(x))  # C05'' adaptation: exclamations tagged as speakers ("STOP!!") are not names
    out = []
    for it in items:
        dropped = {d["scene"] for d in it.get("dropped_scenes") or []}
        rel_scenes = list(ET.fromstring(it["script_segment"]))
        rel = [(el.tag, el.text or "") for sc in rel_scenes for el in sc]
        rel_keys = [(t, key(x)) for t, x in rel]
        # speaker of each released dialogue
        rel_dlg = []
        cur = None
        for t, x in rel:
            if t == "character":
                cur = spk(x)
            elif t == "dialogue":
                rel_dlg.append((key(x), cur))
            elif t not in ("parenthetical",):
                cur = None if t == "stage_direction" else cur
        res = Counter()
        ex = defaultdict(list)
        ptr = 0
        started = False
        for si in range(it["scene_start"], it["scene_end"] + 1):
            if si in dropped or si >= len(raw):
                continue
            speaker = None
            prev = None
            for ei, (t, x) in enumerate(raw[si]):
                if t == "character":
                    speaker = spk(x)
                    prev = t
                    continue
                if t != "dialogue":
                    if t != "parenthetical":
                        prev = t
                    continue
                prev_tag, prev = prev, t  # C05'' adaptation: a line right after another line has no speaker of its own
                kx = strip_tokens(key(x), furniture.get((si, ei), {}))
                if not kx:
                    continue
                res["raw_lines"] += 1
                short = len(kx) <= 2
                if short:
                    res["short_lines"] += 1
                found = None
                for j in range(ptr + (1 if started else 0), min(len(rel_dlg), ptr + 20)):
                    if rel_dlg[j][0] == kx:
                        found = j
                        break
                if found is None:
                    for j in list(range(ptr + (1 if started else 0), min(len(rel_dlg), ptr + 20))) + [ptr]:
                        if j < len(rel_dlg) and (contains(rel_dlg[j][0], kx) or contains_joined(rel_dlg[j][0], kx)):
                            found = j
                            break
                if found is None:
                    for j in range(0, len(rel_dlg)):
                        if contains(rel_dlg[j][0], kx) or contains_joined(rel_dlg[j][0], kx):
                            found = j
                            break
                if found is None:
                    other = [tg for tg, kk in rel_keys if tg != "dialogue" and contains(kk, kx)]
                    kind = "moved_to_" + other[0] if other else "missing"
                    numeric = all(re.fullmatch(r"\d{1,4}[a-z]{0,3}|pt", w) for w in kx) and len(kx) <= 2
                    page_number = numeric and (not speaker or speaker not in talkers or prev_tag != "character")
                    if not speaker or NOTNAME.search(speaker) or EXTRA_NOTNAME.search(speaker) or page_number:  # C05'' adaptation
                        kind = "expected_" + kind
                    res[kind] += 1
                    if short:
                        res[kind + "_short"] += 1
                    ex[kind].append(f"{speaker}: {x.strip()[:60]}")
                    continue
                ptr = found
                started = True
                rs = rel_dlg[found][1]
                if speaker and rs and not NOTNAME.search(speaker) and rs != speaker and not (rs and (rs in speaker or speaker in rs)):
                    res["speaker_changed"] += 1
                    ex["speaker_changed"].append(f"{speaker} -> {rs}: {x.strip()[:50]}")
        out.append({"item_id": it["item_id"], **res, "examples": {k2: v[:3] for k2, v in ex.items()}})
    return out


def check(items, ms_dir, workers=8):
    groups = defaultdict(list)
    for it in items:
        groups[((it.get("source_label") or it)["imdb_id"], it["source_split"])].append(it)  # C05'' adaptation
    for split in ("train", "val", "test"):
        for line in open(f"{ms_dir}/{split}.jsonl"):
            r = json.loads(line)
            k = (r["imdb_id"], split)
            if k in groups and (k not in _ROWS or len(r["script"]) > len(_ROWS[k]["script"])):
                _ROWS[k] = r
    with Pool(workers) as p:
        res = [r for rs in p.imap_unordered(check_movie, sorted(groups.items()), chunksize=4) for r in rs]
    tot = Counter()
    for r in res:
        for k, v in r.items():
            if isinstance(v, int):
                tot[k] += v
    bad = [r for r in res if r.get("missing") or any(k.startswith("moved_to") for k in r) or r.get("speaker_changed") or any(k.startswith("expected_") for k in r)]
    summary = {"items": len(res), "totals": dict(tot), "items_with_problems": len(bad),
               "items_missing": sum(1 for r in res if r.get("missing")),
               "items_moved": sum(1 for r in res if any(k.startswith("moved_to") and not k.endswith("_short") for k in r)),
               "items_speaker_changed": sum(1 for r in res if r.get("speaker_changed"))}
    summary["lost_lines"] = tot["missing"] + sum(v for k2, v in tot.items() if k2.startswith("moved_to") and not k2.endswith("_short"))
    summary["speaker_changed_share"] = round(tot["speaker_changed"] / max(1, tot["raw_lines"]), 6)
    summary["c05pp_pass"] = summary["lost_lines"] == 0 and summary["speaker_changed_share"] <= 0.001
    return summary, bad


def main(seg_path, ms_dir, out_path):
    items = [json.loads(l) for l in open(seg_path)]
    summary, bad = check(items, ms_dir)
    json.dump({"summary": summary, "problems": bad}, open(out_path, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    agg = defaultdict(Counter)
    for r in bad:
        for k, v in r["examples"].items():
            for e in v:
                agg[k][e] += 1
    for k, c in agg.items():
        print(k, sum(c.values()), list(c)[:8])


if __name__ == "__main__":
    main(*sys.argv[1:4])
