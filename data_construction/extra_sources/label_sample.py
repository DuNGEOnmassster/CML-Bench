"""Blind labeled sample for speaker/action tag errors, measured the same way on MovieSum and extra-source builds.

  python data_construction/extra_sources/label_sample.py make --group moviesum_v3_1=BUILD_A/segments.jsonl \
      --group extra_v7=BUILD_B/segments.jsonl --group extra_v9=BUILD_C/segments.jsonl --out DIR
  (labelers write DIR/labels/<labeler>.json for the windows in DIR/assign/<labeler>.txt, following DIR/RUBRIC.md)
  python data_construction/extra_sources/label_sample.py score --dir DIR [--out report.json]

`make` draws --n windows per group with a fixed seed (skipping windows labeled in --exclude samples), writes each window's CML (no ids, names or source) to
DIR/windows/<blind id>.xml in shuffled order, and splits them over --labelers so every labeler sees a mix of groups.
--double windows per group get a second, different labeler (inter-rater agreement). A window whose content is
byte-identical in two groups is labeled once and counted in both. DIR/key.json unblinds.

`score` reports, per group, labeled errors per 1,000 dialogue elements (primary labels; sure and sure+likely),
the share of windows with >= 1 error, a 95% bootstrap interval over windows, agreement on the doubled windows,
and the CML detector's (mislabel.py) window-level agreement with the labels.
`pool --dir A --dir B --group NAME=segments.jsonl ...` pools primary labels across samples: a labeled window counts
for a build when its exact content is in that build.
DIR holds screenplay text: keep it out of the repository.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cml_format import parse_script  # noqa: E402
from extra_sources.mislabel import KINDS, signatures, spk  # noqa: E402

TYPES = ("speech_as_action", "action_as_dialogue", "wrong_speaker", "other_tag")

RUBRIC = """# Labeling rubric: speaker/action tag errors in CML windows

Each file in `windows/` is one excerpt of a screenplay converted to CML markup:
`<stage_direction>` scene heading, `<scene_description>` action, `<character>` speaker cue, `<parenthetical>`,
`<dialogue>`. Judge from the CML only. Read every element of every window assigned to you.

Report every element whose tag is wrong in one of these ways:

1. `speech_as_action`: words a character speaks sit inside `<scene_description>` or `<stage_direction>`.
   Invented examples: `<scene_description>MARA Get down!</scene_description>` (cue fused into action),
   `<scene_description>(whispering) Don't move.</scene_description>` (cue lost before a parenthetical),
   `<scene_description>MARA OTTO Go! -- Now!</scene_description>` (two-column dialogue collapsed), or a whole
   speech with no cue. Not errors: action naming a character in caps (`MARA enters.`), sound effects, signs,
   inserts, on-screen text, lyrics or quoted words the writer deliberately put in action.
2. `action_as_dialogue`: narration or stage action inside `<dialogue>` (`<dialogue>We leave at dawn. Otto turns
   and walks out.</dialogue>`), a scene heading or transition inside dialogue, or another speaker's cue and line
   merged into this dialogue (`<dialogue>Fine. OTTO No, wait.</dialogue>`).
3. `wrong_speaker`: a `<dialogue>` under the wrong `<character>`: two consecutive speeches under one cue that
   clearly belong to two people, a character answering their own question because a cue was lost, a speaker who
   cannot be right in context.
4. `other_tag`: a `<character>` holding non-name text (action, a transition), a `<parenthetical>` holding spoken
   words, a `<dialogue>` holding only a stage direction, and similar.

Do not count: typos, OCR noise, missing words, page or scene numbers, style, a speech split over two consecutive
`<dialogue>` elements of the same speaker, removed (CONT'D)/(MORE).

Confidence: `sure` when the CML alone makes it certain, `likely` when context strongly suggests it. Skip
anything weaker. One entry per wrong element (a cue fused into action and the speech after it = one entry).

Write `labels/<labeler>.json`:
{"labeler": "L01", "windows": {"W001": [], "W002": [{"tag": "scene_description", "quote": "first ~10 words",
 "type": "speech_as_action", "confidence": "sure", "reason": "one line"}]}}
Every assigned window must appear, with [] when it has no errors.
"""


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def elements(seg: dict) -> list[tuple[str, str]]:
    return [(t, x) for sc in parse_script(seg["script_segment"], detok=False) for t, x in sc.elements]


def make(args) -> None:
    rng = random.Random(args.seed)
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists (samples are write-once)")
    groups = dict(g.split("=", 1) for g in args.group)
    seen = set()
    for prev in args.exclude or []:
        with open(prev) as f:
            seen |= {e["content_sha1"] for ents in json.load(f)["windows"].values() for e in ents}
    windows, key = {}, {}
    by_sha, per_group = {}, defaultdict(list)
    for name, path in groups.items():
        segs = [s for s in load(path) if s.get("content_sha1") not in seen]
        for s in rng.sample(segs, min(args.n, len(segs))):
            sha = s.get("content_sha1") or str(hash(s["script_segment"]))
            ent = {"group": name, "item_id": s["item_id"], "movie_name": s["movie_name"], "imdb_id": s["imdb_id"],
                   "content_sha1": sha, "dialogue_elements": sum(1 for t, _ in elements(s) if t == "dialogue")}
            if sha not in by_sha:
                by_sha[sha] = len(windows)
                windows[len(windows)] = (s, [ent])
            else:
                windows[by_sha[sha]][1].append(ent)
            per_group[name].append(by_sha[sha])
    order = list(windows)
    rng.shuffle(order)
    os.makedirs(os.path.join(args.out, "windows"))
    os.makedirs(os.path.join(args.out, "assign"))
    os.makedirs(os.path.join(args.out, "labels"))
    blind = {}
    for i, w in enumerate(order, 1):
        bid = f"W{i:03d}"
        blind[w] = bid
        seg, ents = windows[w]
        with open(os.path.join(args.out, "windows", bid + ".xml"), "w", encoding="utf-8") as f:
            f.write(seg["script_segment"] + "\n")
        key[bid] = ents
    labelers = [f"L{i:02d}" for i in range(1, args.labelers + 1)]
    assign = defaultdict(list)
    primary = {}
    for i, w in enumerate(order):
        primary[blind[w]] = labelers[i % len(labelers)]
        assign[primary[blind[w]]].append(blind[w])
    second = {}
    for name in groups:
        cands = [blind[w] for w in dict.fromkeys(per_group[name]) if blind[w] not in second]
        for bid in rng.sample(cands, min(args.double, len(cands))):
            others = sorted((x for x in labelers if x != primary[bid]), key=lambda x: (len(assign[x]), x))
            second[bid] = others[0]
            assign[others[0]].append(bid)
    for lab, ids in assign.items():
        rng.shuffle(ids)
        with open(os.path.join(args.out, "assign", lab + ".txt"), "w") as f:
            f.write("\n".join(ids) + "\n")
    with open(os.path.join(args.out, "RUBRIC.md"), "w") as f:
        f.write(RUBRIC)
    with open(os.path.join(args.out, "key.json"), "w") as f:
        json.dump({"seed": args.seed, "n": args.n, "groups": groups, "windows": key, "primary": primary, "second": second,
                   "group_windows": {g: [blind[w] for w in ws] for g, ws in per_group.items()}}, f, indent=1)
    print(json.dumps({"windows": len(order), "assignments": {k: len(v) for k, v in sorted(assign.items())},
                      "doubled": len(second)}, indent=1))


def _boot(rows: list[tuple[int, int]], rng: random.Random, reps: int = 2000) -> list[float]:
    vals = []
    for _ in range(reps):
        pick = [rows[rng.randrange(len(rows))] for _ in rows]
        d = sum(p[1] for p in pick) or 1
        vals.append(1000 * sum(p[0] for p in pick) / d)
    vals.sort()
    return [round(vals[int(0.025 * reps)], 2), round(vals[int(0.975 * reps)], 2)]


def score(args) -> None:
    with open(os.path.join(args.dir, "key.json")) as f:
        key = json.load(f)
    labels = {}
    for path in sorted(glob.glob(os.path.join(args.dir, "labels", "*.json"))):
        with open(path) as f:
            lab = json.load(f)
        labels[lab["labeler"]] = lab["windows"]
    missing = [(bid, lab) for bid, lab in list(key["primary"].items()) + list(key["second"].items())
               if bid not in labels.get(lab, {})]
    if missing:
        sys.exit(f"missing labels: {missing[:10]} ({len(missing)})")

    def errs(bid, lab, sure_only=False):
        return [e for e in labels[lab][bid] if e.get("type") in TYPES and (e.get("confidence") == "sure" or not sure_only)]

    rng = random.Random(1)
    segs, film_talkers = {}, {}
    for g, path in key["groups"].items():
        want = {e["item_id"] for ents in key["windows"].values() for e in ents if e["group"] == g}
        films = {e["imdb_id"] for ents in key["windows"].values() for e in ents if e["group"] == g}
        talkers = defaultdict(set)
        for s in load(path):
            if s["imdb_id"] in films:
                talkers[s["imdb_id"]] |= {spk(x) for t, x in elements(s) if t == "character" and len(spk(x)) >= 2}
            if s["item_id"] in want:
                segs[(g, s["item_id"])] = s
        film_talkers[g] = talkers
    out = {"groups": {}, "agreement": {}, "detector_vs_labels": {}}
    for g, bids in key["group_windows"].items():
        rows, rows_sure, types, any_err, det_rows = [], [], Counter(), 0, []
        talkers = film_talkers[g]
        for bid in bids:
            ent = next(e for e in key["windows"][bid] if e["group"] == g)
            e_all = errs(bid, key["primary"][bid])
            rows.append((len(e_all), ent["dialogue_elements"]))
            rows_sure.append((len(errs(bid, key["primary"][bid], True)), ent["dialogue_elements"]))
            types.update(e["type"] for e in e_all)
            any_err += bool(e_all)
            sig = signatures(elements(segs[(g, ent["item_id"])]), talkers[ent["imdb_id"]])
            det_rows.append((sum(sig[k] for k in KINDS if k != "absorbed_action") > 0, bool(e_all)))
        dlg = sum(r[1] for r in rows) or 1
        out["groups"][g] = {
            "windows": len(bids), "dialogue_elements": dlg,
            "errors_sure_or_likely": sum(r[0] for r in rows), "errors_sure": sum(r[0] for r in rows_sure),
            "per_1k_dialogue": round(1000 * sum(r[0] for r in rows) / dlg, 2), "per_1k_dialogue_ci95": _boot(rows, rng),
            "per_1k_dialogue_sure": round(1000 * sum(r[0] for r in rows_sure) / dlg, 2),
            "windows_with_any": round(any_err / len(bids), 3), "by_type": dict(types),
        }
        tp = sum(1 for d, l in det_rows if d and l)
        out["detector_vs_labels"][g] = {"both": tp, "detector_only": sum(1 for d, l in det_rows if d and not l),
                                        "labels_only": sum(1 for d, l in det_rows if l and not d),
                                        "neither": sum(1 for d, l in det_rows if not d and not l)}
    pairs = [(len(errs(b, key["primary"][b])), len(errs(b, lab))) for b, lab in key["second"].items()]
    a = [x > 0 for x, _ in pairs]
    b = [y > 0 for _, y in pairs]
    n = len(pairs) or 1
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = (sum(a) / n) * (sum(b) / n) + (1 - sum(a) / n) * (1 - sum(b) / n)
    out["agreement"] = {"doubled_windows": len(pairs), "window_any_error_agreement": round(po, 3),
                        "cohen_kappa_any_error": round((po - pe) / (1 - pe), 3) if pe < 1 else None,
                        "error_count_primary": sum(x for x, _ in pairs), "error_count_second": sum(y for _, y in pairs),
                        "pairs": {bid: p for bid, p in zip(key["second"], pairs)}}
    print(json.dumps(out, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


def pool(args) -> None:
    """Pool primary labels from several samples per target build: a labeled window counts for a build when its
    exact content (content_sha1) is in that build."""
    rng = random.Random(1)
    labeled = {}  # content_sha1 -> (n errors, n sure, dialogue elements, types)
    for d in args.dir:
        with open(os.path.join(d, "key.json")) as f:
            key = json.load(f)
        labs = {}
        for path in glob.glob(os.path.join(d, "labels", "*.json")):
            with open(path) as f:
                lab = json.load(f)
            labs[lab["labeler"]] = lab["windows"]
        for bid, ents in key["windows"].items():
            errs = [e for e in labs[key["primary"][bid]][bid] if e.get("type") in TYPES]
            labeled.setdefault(ents[0]["content_sha1"], (len(errs), sum(e.get("confidence") == "sure" for e in errs),
                                                         ents[0]["dialogue_elements"], Counter(e["type"] for e in errs)))
    out = {}
    for g in args.group:
        name, path = g.split("=", 1)
        shas = {s.get("content_sha1") for s in load(path)}
        rows = [v for sha, v in labeled.items() if sha in shas]
        dlg = sum(r[2] for r in rows) or 1
        out[name] = {"windows": len(rows), "dialogue_elements": dlg, "errors": sum(r[0] for r in rows),
                     "per_1k_dialogue": round(1000 * sum(r[0] for r in rows) / dlg, 2),
                     "per_1k_dialogue_ci95": _boot([(r[0], r[2]) for r in rows], rng) if rows else None,
                     "per_1k_dialogue_sure": round(1000 * sum(r[1] for r in rows) / dlg, 2),
                     "windows_with_any": round(sum(r[0] > 0 for r in rows) / max(1, len(rows)), 3),
                     "by_type": dict(sum((r[3] for r in rows), Counter()))}
    print(json.dumps(out, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--group", action="append", required=True, help="NAME=segments.jsonl")
    m.add_argument("--n", type=int, default=30)
    m.add_argument("--double", type=int, default=6, help="windows per group labeled twice")
    m.add_argument("--labelers", type=int, default=12)
    m.add_argument("--seed", type=int, default=20261009)
    m.add_argument("--exclude", action="append", help="key.json of an earlier sample: skip windows labeled there")
    m.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--dir", required=True)
    s.add_argument("--out")
    p = sub.add_parser("pool")
    p.add_argument("--dir", action="append", required=True, help="sample folder (repeatable)")
    p.add_argument("--group", action="append", required=True, help="NAME=segments.jsonl of the build to report")
    p.add_argument("--out")
    args = ap.parse_args()
    {"make": make, "score": score, "pool": pool}[args.cmd](args)


if __name__ == "__main__":
    main()
