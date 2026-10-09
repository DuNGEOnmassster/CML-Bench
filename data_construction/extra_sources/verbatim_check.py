"""Independent verbatim check for extra-source segments (contract C05' style): word-level alignment of each
`script_segment` against its cached raw source file. Deliberately imports nothing from cml_format or
text_screenplay; only pdftotext is shared.

  python data_construction/extra_sources/verbatim_check.py --segments RUN/items.jsonl --out report.json

Per item: words inserted (must be ~0: the pipeline only removes or re-spaces), words deleted inside the
aligned span split into whitelisted page furniture (CONTINUED, MORE, OMITTED, page and scene numbers,
revision words, dates) and other deletions, which are listed with context. Deletions inside a scene the
item lists in `dropped_scenes` are counted separately.
"""
from __future__ import annotations

import argparse
import difflib
import html
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
import unicodedata

WHITELIST = re.compile(
    r"^(continued|cont|contd|d|more|omitted|omit|\d{1,4}[a-z]{0,3}|pt|rev|revised|revision|revisions|draft|pink|blue|yellow|"
    r"green|goldenrod|buff|salmon|cherry|tan|white|page|final|shooting|script|of)$"
)


PAGE_FURNITURE = re.compile(r"\b(continued|revised|revision|rev|draft|omitted|more|progress|pink|blue|yellow|green|goldenrod|buff|salmon|cherry)\b")


def raw_words(path: str, fmt: str) -> list[str]:
    return raw_layout(path, fmt)[0]


def raw_layout(path: str, fmt: str) -> tuple[list[str], list[int], list[str]]:
    """(words, line index of each word, lines) of the raw download, page furniture removed."""
    with open(path, "rb") as f:
        body = f.read()
    if fmt == "pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(body)
            tmp.flush()
            text = subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", tmp.name, "-"], capture_output=True).stdout.decode("utf-8", "replace")
    else:
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            text = body.decode("cp1252", "replace")
        if fmt == "html":
            text = re.sub(r"<[^>]+>", " ", re.sub(r"(?is)<(script|style|head)[^>]*>.*?</\1>", " ", text))
    # declared normalizations: HTML entities (twice), NFKC (ligatures)
    text = unicodedata.normalize("NFKC", html.unescape(html.unescape(text)))
    headers = page_headers(text)
    lines = ["" if (mask(l) in headers or tail_mask(l) in headers or SCENE_NOTE.match(l) or PAGE_OF.match(l)) else l
             for l in text.replace("\r", "").replace("\f", "\n").expandtabs(8).split("\n")]
    ws, wl = [], []
    for k, l in enumerate(lines):
        for w in words(l):
            ws.append(w)
            wl.append(k)
    return ws, wl, lines


SCENE_NOTE = re.compile(r"^\s*SCENES?\s+\d+[A-Z]?(\s*(-|TO|AND|THRU|THROUGH)\s*\d+[A-Z]?)?(\s+(INCORPORATED INTO|MOVED TO|COMBINED WITH|"
                        r"DELETED|OMITTED)(\s+SCENES?\s+\d+[A-Z]?)?)?\s*$")
PAGE_OF = re.compile(r"^\s*page\s+\d{1,3}\s+of\s+\d{1,3}\s*$", re.I)


def tail_mask(line: str) -> str:
    """Only the trailing number masked: 'V12_THE SHOOTING SCRIPT_7/2/16 30.' -> 'v12_the shooting script_7/2/16 #.'"""
    return re.sub(r"\s+", " ", re.sub(r"\d+(?=\D*$)", "#", line.strip().lower()))


def mask(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.strip().lower()))


def page_headers(text: str, min_repeats: int = 5) -> set[str]:
    """Running page headers/footers, found independently of the parser: lines with a number whose text (numbers
    masked) repeats >= min_repeats times and is not a scene heading. They are declared page furniture."""
    def shot_or_cue(l):  # numbered shot headings and cues ("127. CLOSER SHOT") repeat too, but they are content
        letters = [c for c in l if c.isalpha()]
        return len(l.split()) <= 6 and letters and sum(c.isupper() for c in letters) / len(letters) >= 0.9

    counts = Counter(mask(l) for l in text.split("\n")
                     if re.search(r"\d", l) and len(l.strip()) >= 6 and not re.match(r"\s*\d*\s*(int|ext)\b", l, re.I)
                     and not shot_or_cue(l))
    out = {k for k, c in counts.items() if c >= min_repeats and len(re.sub(r"[^a-z]", "", k)) >= 4}
    # an all-caps title with a trailing page number ("WONDERSTRUCK 2.", "V12_THE SHOOTING SCRIPT_7/2/16 30.") also counts
    # when only that page number varies (>= 3 values) and the line does not start with a number (shot slugs do)
    titles, pages = Counter(), {}
    for l in text.split("\n"):
        t = l.strip()
        if len(t) >= 6 and re.search(r"[A-Za-z]{3}", t) and re.search(r"\d+\D{0,2}$", t) and not re.match(r"\d", t) \
                and not re.match(r"(int|ext)\b", t, re.I):
            k = tail_mask(t)
            titles[k] += 1
            pages.setdefault(k, set()).add(re.findall(r"\d+", t)[-1])
    out |= {k for k, c in titles.items() if c >= min_repeats and len(pages[k]) >= 3}
    return out


_CUE_NOT = re.compile(r"^(INT|EXT|I/E|EST|FADE|CUT|DISSOLVE|SMASH|MATCH|ANGLE|CLOSE|WIDE|POV|INSERT|SUPER|TITLE|BACK|LATER|MONTAGE|"
                      r"SERIES|INTERCUT|CONTINUOUS|THE END|END|SOUND|SFX|ON|OVER|UNDER|FLASHBACK|CREDITS|MORE|CONTINUED|OMITTED|"
                      r"DAY|NIGHT|MORNING|EVENING|AFTERNOON|SUNSET|SUNRISE|MOMENTS LATER)\b")  # not DAWN: a name too


def cue_name_key(t: str) -> str:
    return re.sub(r"[^A-Z]", "", re.sub(r"\([^)]*\)", " ", t.replace("\u2019", "'")).upper())


def _cue_like(t: str) -> bool:
    t = re.sub(r"\s+\*+$", "", t.strip()).replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')  # revision mark ("MR. LAWSON   *")
    name = re.sub(r"\s*\([^)]*\)\s*", " ", t).strip()
    letters = [c for c in name if c.isalpha()]
    last = name.split()[-1] if name.split() else ""
    return (2 <= len(letters) and len(name) <= 35 and len(name.split()) <= 4 and not name[:1].isdigit() and not t.startswith("(")
            and sum(c.isupper() for c in letters) / len(letters) >= 0.8 and not name.rstrip().endswith((":", "!", "?"))
            and not (last.endswith(".") and sum(c.isalpha() for c in last) > 3)  # "DOOR." ends a sentence; "DR.", "M.E." are names
            and not _CUE_NOT.match(name) and re.fullmatch(r"[A-Za-z0-9 .,'\-&#/\"]+", name) is not None)


def raw_cues(lines: list[str], lo: int, hi: int) -> list[dict]:
    """Speaker cues in raw lines lo..hi, from layout alone (no parser code): {"name", "words" (first speech words),
    "lines" (first, last line of the speech; None for dual dialogue)}. A cue is a short all-caps line at the file's
    cue column (anywhere, after a blank line, in flush-left files) followed by speech: the next line directly, or
    after blank lines a parenthetical. The speech runs to a blank line (past blank lines that follow only
    parentheticals), the next cue or heading, or, when it is indented, a line back at the action indent. A line
    holding two such names apart is a dual-dialogue cue line; the speech below it is split at the column gap."""
    ind = [len(l) - len(l.lstrip()) for l in lines]
    long_ind = Counter(ind[k] for k, l in enumerate(lines) if len(l.strip()) > 50)
    action = long_ind.most_common(1)[0][0] if long_ind else 0
    flush = not any(ind[k] >= action + 8 and _cue_like(l) for k, l in enumerate(lines) if l.strip())
    # the cue column: mode indent of cue-like lines followed directly by a non-caps line (shouted all-caps dialogue sits
    # at the dialogue indent, left of it)
    cols = Counter(ind[k] for k in range(len(lines) - 1) if lines[k].strip() and ind[k] >= action + 8 and _cue_like(lines[k])
                   and lines[k + 1].strip() and not lines[k + 1].strip().isupper())
    cue_col = cols.most_common(1)[0][0] if cols else action + 8
    # files whose speech sits at the action indent (IMSDb "LOIS" / "Uh-huh" at column 0) vs. indented speech, where a
    # cue-column line over action is a slug ("ANNA'S POV" / "James Hutton lies...")
    under = [ind[k + 1] <= action + 2 for k in range(len(lines) - 1) if ind[k] >= cue_col - 5 and _cue_like(lines[k])
             and lines[k + 1].strip() and not lines[k + 1].strip().startswith("(") and not lines[k + 1].strip().isupper()]
    flat_speech = sum(under) >= 0.3 * max(1, len(under))
    known = {cue_name_key(lines[k]) for k in range(len(lines) - 1) if ind[k] >= cue_col - 5 and _cue_like(lines[k])
             and lines[k + 1].strip()}

    def dual_line(t):
        parts = [p for p in re.split(r"\s{4,}", re.sub(r"\s+\*+$", "", t.strip())) if p]
        return len(parts) == 2 and all(_cue_like(p) for p in parts)
    out = []
    for k in range(max(0, lo), min(len(lines), hi + 1)):
        t = lines[k]
        if not t.strip():
            continue
        parts = [p for p in re.split(r"\s{4,}", re.sub(r"\s+\*+$", "", t.strip())) if p]
        # a speech line right under a cue is never a dual cue line ("MADISON.    OPEN THE DOOR." shouted)
        dual = len(parts) == 2 and all(_cue_like(p) for p in parts) and not (k > 0 and _cue_like(lines[k - 1]))
        # cues sit at the cue column, or (jittery PDF columns) >= 6 right of the speech line under them; flush-left, a
        # cue starts a block
        at_col = ind[k] >= max(action + 8, cue_col - 5) or (
            k + 1 < len(lines) and lines[k + 1].strip() and ind[k] >= max(action + 8, ind[k + 1] + 6) and ind[k + 1] > action + 2) or (
            # a known speaker's cue printed at the action column right above indented speech
            not flat_speech and ind[k] <= action + 2 and cue_name_key(t) in known and k + 1 < len(lines)
            and lines[k + 1].strip() and ind[k + 1] > action + 3)
        if not dual and (not (flush or at_col) or (flush and k > 0 and lines[k - 1].strip()) or not _cue_like(t)):
            continue
        j = k + 1
        if j < len(lines) and lines[j].strip():
            direct = True
        else:
            while j < len(lines) and j - k <= 12 and not lines[j].strip():
                j += 1
            direct = False
        nxt = lines[j].strip() if j < len(lines) else ""
        letters = [c for c in nxt if c.isalpha()]
        if not (nxt.startswith("(") or (direct and letters and sum(c.isupper() for c in letters) / len(letters) < 0.6
                                        and not re.match(r"(INT|EXT)\b", nxt)
                                        and (dual or flat_speech or flush or ind[j] > action + 2))):
            continue
        if dual:
            cols2 = [p for p in re.split(r"\s{3,}", nxt) if p]
            speech = [cols2[0], " ".join(cols2[1:])] if len(cols2) >= 2 else [nxt, ""]
            out += [{"name": cue_name_key(parts[0]), "words": words(speech[0])[:8], "lines": None},
                    {"name": cue_name_key(parts[1]), "words": words(speech[1])[:8], "lines": None}]
            continue
        # key = first speech line after any parenthetical lines ("(shaking his head)" / "You're not gonna...")
        m, in_paren, key, last, spoken, flat, gap, sp_ind = j, False, nxt, j, False, None, False, None
        while m < len(lines) and m - j < 40:
            ln = lines[m].strip()
            if not ln:
                if spoken and not flat:
                    # an unfinished indented speech that goes on after a blank line at the same indent
                    q = m + 1
                    while q < len(lines) and q - m <= 2 and not lines[q].strip():
                        q += 1
                    nq = lines[q].strip() if q < len(lines) else ""
                    if (nq and abs(ind[q] - sp_ind) <= 2 and not _cue_like(nq) and not dual_line(lines[q])
                            and not nq.startswith("(") and not re.match(r"(INT|EXT)\b", nq)
                            and (nq[:1].islower() or not re.search(r"[.!?\"')\-]$", lines[last].strip()))):
                        m = q
                        continue
                if spoken or m - j >= 12:
                    break
                gap = True
                m += 1
                continue
            if m > j and dual_line(lines[m]):
                break  # two-column cue line right under the speech
            if gap and not spoken and not ln.startswith("(") and ind[m] <= action + 2 < ind[j]:
                break  # after an indented parenthetical-only speech and a blank line, action resumes
            if m > j and (re.match(r"(INT|EXT)\b", ln) or (ln.isupper() and ind[m] >= ind[j] + 6 and not in_paren) or (
                    (ind[m] >= cue_col - 5 or ind[m] >= max(action + 8, ind[j] + 6)) and _cue_like(ln) and not in_paren)):
                break  # a heading, a transition right of the speech ("DISSOLVE TO:"), or the next cue
            if in_paren or ln.startswith("("):
                in_paren = ")" not in ln
            else:
                if flat is None:
                    flat, sp_ind = ind[m] <= action + 2, ind[m]
                elif not flat and ind[m] <= action + 2:
                    break  # action resumes right under an indented speech
                if not spoken:
                    key = ln
                spoken = True
            last = m
            m += 1
        out.append({"name": cue_name_key(t), "words": words(key)[:8], "lines": (j, last)})
    return out


def cml_speakers(segment: str) -> list[tuple[str, str, list[str]]]:
    """(tag, speaker key of the element's block, words) for every element of a CML segment."""
    out = []
    for sc in re.findall(r"<scene>(.*?)</scene>", segment, re.S):
        speaker = ""
        for tag, text in re.findall(r"<([a-z_]+)>(.*?)</\1>", sc, re.S):
            text = html.unescape(text)
            if tag == "character":
                speaker = cue_name_key(text)
            elif tag not in ("dialogue", "parenthetical"):
                speaker = ""
            out.append((tag, speaker, words(text)))
    return out


def tag_check(raw_lines: list[str], lo: int, hi: int, segment: str, aligned: dict[int, tuple[str, str]] | None = None) -> dict:
    """Tag-aware raw-vs-CML check: the speech of each raw cue in the window must sit in dialogue/parenthetical
    elements under the same speaker. Otherwise it was flattened into action, carried over to another speaker, or a
    dual-dialogue pair was collapsed. With `aligned` (raw line -> [(tag, speaker)] of the CML words the verbatim
    alignment pairs with that line's words) every speech word is checked, so one-word speeches count too; a speech is
    wrong when most of its aligned words, or >= 5 of them, are mistagged. Without it (and always for dual dialogue)
    the first >= 4 speech words are looked up in the CML elements."""
    els = cml_speakers(segment)
    out = Counter()
    examples = []
    for cue in raw_cues(raw_lines, lo, hi):
        name = cue["name"]
        if not name:
            continue

        def same(a):
            return a == name or (a and (a.startswith(name) or name.startswith(a)))
        if aligned is not None and cue["lines"]:
            labels = [lab for k in range(cue["lines"][0], cue["lines"][1] + 1) for lab in aligned.get(k, [])]
            if not labels:
                continue  # speech outside the window
            out["checked"] += 1
            as_action = [(t, s) for t, s in labels if t not in ("dialogue", "parenthetical")]
            other = [(t, s) for t, s in labels if t in ("dialogue", "parenthetical") and not same(s)]
            if len(as_action) >= max(1, len(labels) / 2) or (len(as_action) >= 5 and len(as_action) >= len(other)):
                out["speech_as_action"] += 1
                examples.append((name, as_action[0][0], " ".join(cue["words"])))
            elif len(other) >= max(1, len(labels) / 2) or len(other) >= 5:
                out["wrong_speaker"] += 1
                examples.append((name, other[0][1], " ".join(cue["words"])))
            continue
        speech = cue["words"]
        if len(speech) < 4:
            continue  # too short to locate reliably
        hits = [(tag, spk_) for tag, spk_, ws in els
                if any(ws[i:i + len(speech)] == speech for i in range(len(ws) - len(speech) + 1))]
        if not hits:
            continue  # speech outside the window (or rewritten by declared cleaning)
        out["checked"] += 1
        if any(tag in ("dialogue", "parenthetical") and same(spk_) for tag, spk_ in hits):
            continue
        tag, spk_ = hits[0]
        if tag not in ("dialogue", "parenthetical"):
            out["speech_as_action"] += 1
            examples.append((name, tag, " ".join(speech)))
        else:
            out["wrong_speaker"] += 1
            examples.append((name, spk_, " ".join(speech)))
    return {"cues_checked": out["checked"], "speech_as_action": out["speech_as_action"], "wrong_speaker": out["wrong_speaker"],
            "tag_examples": examples[:6]}


def words(text: str) -> list[str]:
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return re.findall(r"[a-z0-9]+", text.lower())


def segment_scene_words(segment: str) -> list[list[str]]:
    scenes = re.findall(r"<scene>(.*?)</scene>", segment, re.S)
    return [words(html.unescape(re.sub(r"<[^>]+>", " ", s))) for s in scenes]


def check(item: dict, cache_dir: str) -> dict:
    raw, raw_line, raw_lines = raw_layout(os.path.join(cache_dir, item.get("source_cache_file") or item["source_file"]),
                                          item["source_format"])
    seg_scenes = segment_scene_words(item["script_segment"])
    seg = [w for s in seg_scenes for w in s]
    # anchor on number-free words (margin scene/page numbers interrupt the raw word stream), align in full
    num = re.compile(r"\d+[a-z]{0,3}")
    raw_pos = [i for i, w in enumerate(raw) if not num.fullmatch(w)]
    raw_f = [raw[i] for i in raw_pos]
    seg_f = [w for w in seg if not num.fullmatch(w)]
    anchor, end_anchor = seg_f[:12], seg_f[-12:]
    starts = [i for i in range(len(raw_f) - len(anchor) + 1) if raw_f[i : i + len(anchor)] == anchor]
    if not starts:
        sm = difflib.SequenceMatcher(None, raw_f, anchor, autojunk=False)
        starts = [sm.find_longest_match(0, len(raw_f), 0, len(anchor)).a]
    best = None
    for st in starts:  # tightest span that starts at an anchor occurrence and ends at the end anchor
        end = min(len(raw_f), st + len(seg_f))
        for j in range(st + int(0.8 * len(seg_f)) - len(end_anchor), min(len(raw_f), st + 3 * len(seg_f))):
            if raw_f[j : j + len(end_anchor)] == end_anchor:
                end = j + len(end_anchor)
                break
        if best is None or end - st < best[1] - best[0]:
            best = (st, end)
    start, end = raw_pos[best[0]], raw_pos[best[1] - 1] + 1
    region = raw[start:end]
    near_ws = raw[max(0, start - len(seg)):end + len(seg)]
    # coverage n-grams skip page-furniture tokens ("cont d", page numbers), which interleave the raw words
    near_f = [w for w in near_ws if not WHITELIST.match(w)]
    near4 = {tuple(near_f[i:i + 4]) for i in range(len(near_f) - 3)}
    seg4 = {tuple(t) for t in (lambda f: [f[i:i + 4] for i in range(len(f) - 3)])([w for w in seg if not WHITELIST.match(w)])}
    joined = {"".join(near_ws[i:i + k]) for k in (2, 3) for i in range(len(near_ws) - k + 1)}

    def covered(ws, i, grams):
        if WHITELIST.match(ws[i]):
            return True
        f = [k for k in range(max(0, i - 12), min(len(ws), i + 13)) if not WHITELIST.match(ws[k])]
        p = f.index(i)
        return any(tuple(ws[f[k]] for k in range(s0, s0 + 4)) in grams for s0 in range(max(0, p - 3), min(p, len(f) - 4) + 1))

    speakers = {w for c in re.findall(r"<character>(.*?)</character>", item["script_segment"]) for w in words(html.unescape(c))}
    sm = difflib.SequenceMatcher(None, region, seg, autojunk=False)
    inserted, reordered, del_white, del_declared, del_other, runs, ins_runs = 0, 0, 0, 0, 0, [], []
    for op, a1, a2, b1, b2 in sm.get_opcodes():
        if op == "replace" and "".join(region[a1:a2]) == "".join(seg[b1:b2]):
            continue  # spacing only: the same letters split or joined differently ("mouse- pad" -> "mousepad")
        if op in ("insert", "replace"):
            # words inside a 6-gram that also occurs in the raw text nearby are real text the aligner paired with
            # another copy of a repeated passage; only the rest counts as inserted
            found = sum(covered(seg, j, near4) or seg[j] in joined for j in range(b1, b2))
            reordered += found
            if b2 - b1 - found:
                inserted += b2 - b1 - found
                ins_runs.append(" ".join(seg[max(0, b1 - 3):b1]) + " {{" + " ".join(seg[b1:b2]) + "}} " + " ".join(seg[b2:b2 + 3]))
        if op in ("delete", "replace"):
            run = region[a1:a2]
            moved = [not WHITELIST.match(region[j]) and covered(region, j, seg4) for j in range(a1, a2)]
            reordered += sum(moved)
            run = [w for w, m in zip(run, moved) if not m]
            if not run:
                continue
            other = [w for w in run if not WHITELIST.match(w)]
            del_white += len(run) - len(other)
            if other and all(w in speakers for w in other):
                del_declared += len(other)  # bare speaker-name line without dialogue (removed by design, C12c)
                continue
            if other and len(run) <= 20 and PAGE_FURNITURE.search(" ".join(run)):
                del_declared += len(other)  # page header/footer ("CONTINUED", revision colour and date, page no.)
                continue
            del_other += len(other)
            if other:
                runs.append(" ".join(region[max(0, a1 - 4):a1]) + " [[" + " ".join(run) + "]] " + " ".join(region[a2:a2 + 4]))
    seg_labels = [(tag, spk_) for tag, spk_, ws in cml_speakers(item["script_segment"]) for _ in ws]
    aligned = None
    if len(seg_labels) == len(seg):
        aligned = {}
        for a, b, size in sm.get_matching_blocks():
            for x in range(size):
                aligned.setdefault(raw_line[start + a + x], []).append(seg_labels[b + x])
    tags = tag_check(raw_lines, raw_line[start], raw_line[end - 1], item["script_segment"], aligned) if region else {
        "cues_checked": 0, "speech_as_action": 0, "wrong_speaker": 0, "tag_examples": []}
    return {"item_id": item["item_id"], "segment_words": len(seg), "region_words": len(region), "inserted": inserted,
            "tag_errors": tags["speech_as_action"] + tags["wrong_speaker"], **tags,
            "aligned_elsewhere": reordered,
            "deleted_whitelisted": del_white, "deleted_declared": del_declared, "deleted_other": del_other, "dropped_scenes": item.get("dropped_scenes", []),
            "other_runs": runs[:15], "inserted_runs": ins_runs[:10]}


def ngrams8(ws: list[str]) -> set:
    return {" ".join(ws[i:i + 8]) for i in range(len(ws) - 7)}


def verify_extra(records: list[dict], cache_dir: str, gt_path: str) -> tuple[dict, list[dict]]:
    """C05'/C16b for extra-source records (schema 1.0): `source_file` is the script URL, looked up in the fetch
    cache index. Pass = no inserted word and no unexplained deletion."""
    index = {}
    with open(os.path.join(cache_dir, "index.jsonl"), encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            index[rec["url"]] = rec
    with open(gt_path, encoding="utf-8") as f:
        gt8 = set().union(*(ngrams8(words(re.sub(r"<[^>]+>", " ", json.loads(l)["script_segment"]))) for l in f if l.strip()))
    results, fails = [], []
    for r in records:
        rec = index.get(r["source_file"])
        if not rec or not rec.get("file"):
            res = {"item_id": r["item_id"], "pass": False, "problems": ["source_not_cached"]}
        else:
            fmt = {"htm": "html"}.get(rec["file"].rsplit(".", 1)[-1], rec["file"].rsplit(".", 1)[-1])
            res = check({**r, "source_cache_file": rec["file"], "source_format": fmt}, cache_dir)
            res["problems"] = (["inserted_words"] if res["inserted"] else []) + (["deleted_words"] if res["deleted_other"] else [])
            res["pass"] = not res["problems"]
        g = ngrams8(words(re.sub(r"<[^>]+>", " ", r["script_segment"])))
        res["gt_8gram_overlap"] = round(len(g & gt8) / max(1, len(g)), 5)
        results.append(res)
        if not res["pass"]:
            fails.append(res)
    ov = [x["gt_8gram_overlap"] for x in results]
    summary = {
        "items": len(results), "c05_pass": len(results) - len(fails), "c05_fail": len(fails),
        "inserted_words": sum(x.get("inserted", 0) for x in results),
        "deleted_words": sum(x.get("deleted_other", 0) for x in results),
        "raw_words": sum(x.get("region_words", 0) for x in results),
        "problem_kinds": dict(Counter(p for x in fails for p in x["problems"])),
        "c16b_max_gt_8gram_overlap": max(ov) if ov else 0, "c16b_pass": (max(ov) if ov else 0) <= 0.01,
        # speaker preservation (the extra-source counterpart of C05''): raw cues whose speech left its speaker
        "cues_checked": sum(x.get("cues_checked", 0) for x in results),
        "tag_errors": sum(x.get("tag_errors", 0) for x in results),
        "tag_error_examples": [(x["item_id"], x["tag_examples"][:2]) for x in results if x.get("tag_errors")][:5],
    }
    return summary, fails


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--cache_dir", default="data_construction/work/sources/extra/raw")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    with open(args.segments, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415
        with ProcessPoolExecutor(args.workers) as ex:
            results = list(ex.map(check, items, [args.cache_dir] * len(items), chunksize=4))
    else:
        results = [check(it, args.cache_dir) for it in items]
    tot = {k: sum(r[k] for r in results) for k in ("segment_words", "inserted", "aligned_elsewhere", "deleted_whitelisted",
                                                   "deleted_declared", "deleted_other", "cues_checked", "speech_as_action",
                                                   "wrong_speaker", "tag_errors")}
    summary = {"items": len(results), **tot,
               "items_with_inserted": sum(r["inserted"] > 0 for r in results),
               "items_with_other_deletions": sum(r["deleted_other"] > 0 for r in results),
               "items_with_tag_errors": sum(r["tag_errors"] > 0 for r in results),
               "tag_errors_per_1k_cues": round(1000 * tot["tag_errors"] / max(1, tot["cues_checked"]), 2)}
    print(json.dumps(summary, indent=2))
    for r in results:
        if r["inserted"] or r["deleted_other"]:
            print(r["item_id"], "ins", r["inserted"], "del_other", r["deleted_other"], r["other_runs"][:4])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "items": results}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
