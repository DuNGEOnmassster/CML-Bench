"""CML-only speaker/action mislabel signatures, measured the same way on any build (MovieSum or extra sources).

  python data_construction/extra_sources/mislabel.py --segments BUILD/segments.jsonl [--sample_mod 1] [--out report.json]

Signatures per window (speaker names are taken from the whole film's <character> tags in the build, so a name that is
not tagged inside the window still counts):
  fused_cue        action element opening with a speaker's name followed by speech ("LOIS Superman!", "M.E. I'm
                   guessing…"); a third-person verb after the name ("LOIS watches…") is action, not speech
  dual_collapse    action element opening with two speaker names ("BARNES TAYLOR Co -- Copy that.")
  paren_speech     action element opening with a parenthetical followed by speech ("(shaking his head) You're not…")
  name_in_dialogue dialogue element with another speaker's ALL-CAPS name inside, followed by speech
                   ("Please come with me. MUFFY I'm sure Chaz is fine.")
  absorbed_action  dialogue element containing a sentence that opens with a speaker's name and a third-person verb
                   ("…Like an announcement. Annie confers with Karin")
The rate is signatures per 1,000 dialogue elements; the share of windows with >= 1 signature is reported too.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cml_format import parse_script  # noqa: E402

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
    """Speaker names at the start of an element (up to `limit`), and the remaining text."""
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


KINDS = ("fused_cue", "dual_collapse", "paren_speech", "name_in_dialogue", "absorbed_action")


def measure(segments: list[dict]) -> dict:
    film_talkers = defaultdict(set)
    parsed = []
    for s in segments:
        els = [(t, x) for sc in parse_script(s["script_segment"], detok=False) for t, x in sc.elements]
        parsed.append(els)
        film_talkers[s["imdb_id"]] |= {spk(x) for t, x in els if t == "character" and len(spk(x)) >= 2}
    per = []
    for s, els in zip(segments, parsed):
        c = signatures(els, film_talkers[s["imdb_id"]])
        per.append({"item_id": s["item_id"], "dialogue": sum(1 for t, _ in els if t == "dialogue"), **{k: c[k] for k in KINDS}})
    dlg = sum(p["dialogue"] for p in per) or 1
    tot = {k: sum(p[k] for p in per) for k in KINDS}
    allsig = [sum(p[k] for k in KINDS) for p in per]
    srt = sorted(allsig)
    n = len(per) or 1
    return {"segments": len(per), "dialogue_elements": dlg, "per_kind": tot,
            "per_1k_dialogue": {k: round(1000 * v / dlg, 3) for k, v in tot.items()},
            "all_per_1k_dialogue": round(1000 * sum(tot.values()) / dlg, 3),
            "mean_per_window": round(sum(allsig) / n, 3), "p99_per_window": srt[min(len(srt) - 1, int(0.99 * len(srt)))] if srt else 0,
            "max_per_window": max(allsig) if allsig else 0, "windows_with_any": round(sum(x > 0 for x in allsig) / n, 4),
            "items": per}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    with open(args.segments, encoding="utf-8") as f:
        segs = [json.loads(line) for line in f]
    rep = measure(segs)
    print(json.dumps({k: v for k, v in rep.items() if k != "items"}, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f)


if __name__ == "__main__":
    main()
