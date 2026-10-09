"""Contract C34: MovieSum speaker/action mislabel window gate (evaluator ruling, evaluator-repilot-report.md top section).

  python data_construction/mislabel_gate.py --items RUN/items.jsonl [--threshold 4] [--out list.json]

A window is dropped when paren_only + cue_as_dlg + bad_cue + fused >= 4, counted on its CML text:
  paren_only   <dialogue> that is only a parenthetical ("(intently)"); the speech usually went to the next action
  cue_as_dlg   <dialogue> that is a bare ALL-CAPS speaker name of the film
  bad_cue      <character> that is a heading, shot or action ("ACROSS THE STREET - MOMENTS LATER", ends with '.', > 5 words)
  fused        fused_cue + paren_speech + dual_collapse: speech fused into an action element
Speaker names are every <character> of the film in the build, so a name not tagged inside the window still counts.
Mirrors the evaluator's ms_mislabel_gate.py; the signature counter is the sources worker's extra_sources/mislabel.py
(PR #2, d6c5186), copied verbatim so a rebuild reproduces the frozen v3.1 list without the stacked branch.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

EL = re.compile(r"<(stage_direction|scene_description|character|parenthetical|dialogue)>(.*?)</\1>", re.S)
HEADING_CUE = re.compile(r"^(INT|EXT)\b|^\d+[A-Z]?\.?\s|\s-\s|\b(SERIES OF|SHOTS?|ANGLE|CLOSE ON|MONTAGE|MOMENTS LATER|CONTINUOUS|"
                         r"FLASHBACK|INTERCUT)\b|(?<![-\w])(DAY|NIGHT|LATER)(?![-\w])")
VOX = re.compile(r"\s+(V\.?\s?O|O\.?\s?S|O\.?\s?C|CONT'?D)\.?$")
THRESHOLD = 4

SPEECH_PAREN = re.compile(r"\((O\.?S|V\.?O|O\.?C|CONT|to |beat|into|on |re:|over|off|then|quietly|sotto|whisper|softly|"
                          r"reading|yelling|shouting|pause|calling|angrily|smiling|laughing|beat)[^)]*\)", re.I)
PRON = re.compile(r"\b(I|I'm|I'll|I've|I'd|you|you're|your|we|we're|my|me|us|our|yeah|yes|no|okay|ok|hey|oh)\b", re.I)
NAME_TOKEN = re.compile(r"[A-Z0-9][A-Z0-9.'\u2019\-&#]*")
NOT_NAMES = {"INT", "EXT", "CUT", "FADE", "ANGLE", "CLOSE", "WIDE", "POV", "INSERT", "SUPER", "TITLE", "BACK", "LATER",
             "MONTAGE", "FLASHBACK", "INTERCUT", "CONTINUOUS", "SOUND", "SFX", "THE", "A", "AN", "WE", "ON", "IN", "AT",
             "DISSOLVE", "SMASH", "MATCH", "END", "OMITTED", "CONTINUED"}


def spk(t: str) -> str:
    return re.sub(r"\s*\(.*?\)\s*", " ", t.replace("\u2019", "'")).strip().upper().rstrip(".:")


def _speech(rest: str) -> bool:
    first = re.split(r"(?<=[.!?])\s", rest, maxsplit=1)[0]
    third_person = re.match(r"^[A-Za-z]+s\b", rest) and not PRON.search(first)
    return bool((PRON.search(first) or first.rstrip().endswith(("?", "!"))) and not third_person)


def _lead_names(text: str, talkers: set[str], limit: int = 2) -> tuple[list[str], str]:
    toks, names, k = text.split(), [], 0
    while k < len(toks) and len(names) < limit:
        best = None
        for j in range(min(len(toks), k + 4), k, -1):
            cand = " ".join(toks[k:j])
            if all(NAME_TOKEN.fullmatch(t) for t in toks[k:j]) and spk(cand) in talkers and spk(cand) not in NOT_NAMES:
                best = j
                break
        if best is None:
            break
        names.append(spk(" ".join(toks[k:best])))
        k = best
    return names, " ".join(toks[k:])


def signatures(elements: list[tuple[str, str]], talkers: set[str]) -> Counter:
    c = Counter()
    for tag, x in elements:
        x = x.strip()
        if tag == "scene_description":
            names, rest = _lead_names(x, talkers)
            if names:
                m = re.match(r"^((?:\([^)]*\)\s*)*)(.*)$", rest)
                parens, rest2 = m.group(1), m.group(2)
                if rest2 and not (rest2[:1].islower() and not rest2.startswith(("i ", "i'"))):
                    if len(names) >= 2 and (rest2[:1].isupper() or parens):
                        c["dual_collapse"] += 1
                    elif (parens and SPEECH_PAREN.search(parens)) or _speech(rest2):
                        c["fused_cue"] += 1
            elif x.startswith("("):
                m = re.match(r"^((?:\([^)]*\)\s*)+)(.+)$", x)
                if m and _speech(m.group(2)):
                    c["paren_speech"] += 1
        elif tag == "dialogue":
            for m in re.finditer(r"(?<=[.!?\-]) ([A-Z][A-Z.'\u2019\-]+(?: [A-Z][A-Z.'\u2019\-]+)?) (?=\(|[A-Z][a-z'])", x):
                if spk(m.group(1)) in talkers and spk(m.group(1)) not in NOT_NAMES:
                    c["name_in_dialogue"] += 1
                    break
            for m in re.finditer(r"(?:^|[.!?]\s+)([A-Z][A-Za-z'\-]+)\s+([a-z]+(?:s|ed))\b", x):
                if m.group(1).upper() in {t.split()[0] for t in talkers if t} and m.group(2) not in (
                        "was", "has", "is", "does", "says", "needs", "wants", "likes", "loves", "knows", "thinks", "gets",
                        "goes", "seems", "looks", "called", "said", "used", "asked", "told"):
                    c["absorbed_action"] += 1
                    break
    return c


def bad_cue(x: str) -> bool:
    s = VOX.sub("", re.sub(r"\s*\([^)]*\)", "", x).strip()).strip()
    w = s.split()
    if not w:
        return False
    initials = all(re.fullmatch(r"(?:[A-Z]\.){1,3}|[A-Z]\.?|SR\.|JR\.|[A-Z][A-Z'\-]+", t) for t in w)
    if HEADING_CUE.search(s) or len(w) > 5:
        return True
    if s.endswith(".") and len(w) >= 3 and not initials:
        return True
    return "," in s and len(w) >= 3


def window_signals(els: list[tuple[str, str]], talkers: set[str]) -> dict:
    c = Counter()
    for t, x in els:
        x = x.strip()
        if t == "dialogue":
            if re.fullmatch(r"\([^()]*\)\.?", x):
                c["paren_only"] += 1
            elif re.fullmatch(r"[A-Z][A-Z .'\-]{1,30}", x) and spk(x) in talkers:
                c["cue_as_dlg"] += 1
        elif t == "character" and bad_cue(x):
            c["bad_cue"] += 1
    sig = signatures(els, talkers)
    c["fused"] = sig["fused_cue"] + sig["paren_speech"] + sig["dual_collapse"]
    out = {k: c[k] for k in ("paren_only", "cue_as_dlg", "bad_cue", "fused")}
    out["score"] = sum(out.values())
    return out


def film_talkers(windows: list[tuple[str, str]]) -> dict[str, set[str]]:
    """(film_id, cml) pairs -> every speaker name tagged in the film's windows."""
    out = defaultdict(set)
    for fid, cml in windows:
        out[fid] |= {spk(x) for t, x in EL.findall(cml) if t == "character" and len(spk(x)) >= 2}
    return out


def score_items(items: list[dict], film_key: str = "imdb_id") -> dict[str, dict]:
    """item_id -> C34 signals for every item of a build (talkers pooled per film over the whole build)."""
    talkers = film_talkers([(r[film_key], r["script_segment"]) for r in items])
    return {r["item_id"]: window_signals(EL.findall(r["script_segment"]), talkers[r[film_key]]) for r in items}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True, help="a build's items.jsonl (all windows, before batching)")
    ap.add_argument("--threshold", type=int, default=THRESHOLD)
    ap.add_argument("--out")
    args = ap.parse_args()
    with open(args.items, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    sig = score_items(items)
    flagged = [r for r in items if sig[r["item_id"]]["score"] >= args.threshold]
    print(f"C34: {len(flagged)} of {len(items)} windows score >= {args.threshold} "
          f"({len({r['imdb_id'] for r in flagged})} films)")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"rule": f"C34 mislabel window gate: drop a window when paren_only + cue_as_dlg + bad_cue + fused >= {args.threshold}",
                       "build_id": items[0].get("build_id"), "threshold": args.threshold, "count": len(flagged),
                       "items": [{"item_id": r["item_id"], "content_sha1": r["content_sha1"], **sig[r["item_id"]]} for r in flagged]},
                      f, indent=1)


if __name__ == "__main__":
    main()
