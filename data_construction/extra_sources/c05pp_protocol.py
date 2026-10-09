"""C05''-sources measurement (contract-c05pp-sources-amendment.md, P1-P10): blind-labeled speaker/action mislabel rate
on fresh windows of a frozen extra-source build, with a concurrent MovieSum control.

  python data_construction/extra_sources/c05pp_protocol.py draw --sources BUILD/segments.jsonl \
      --control MOVIESUM/segments.jsonl --c34_list mislabel_gate_v3_1.json --exclude_file used.txt \
      --exclude_key DIR1/key.json --exclude_key DIR2/key.json --seed S --out DIR
  (labelers write DIR/labels/<labeler>.json for DIR/assign/<labeler>.txt; the external reader works in --external_dir)
  python data_construction/extra_sources/c05pp_protocol.py score --dir DIR [--external_dir EXT] --out score.json

draw: sources windows stratified by source_dataset in proportion to segment counts (>= --min_per_source each,
>= --n_sources in total, grown until >= --min_dialogue dialogue elements); --n_control MovieSum windows that pass C34
(not in --c34_list) and were never labeled. Every window whose item_id or content_sha1 is in --exclude_file or an
earlier key is skipped. Windows are rendered without ids, titles or source and shuffled; each labeler gets a mix.
--double_frac of the windows get a second read: --external_reads of them by the non-Claude reader (written with the
rubric to --external_dir), the rest by a different Claude labeler. DIR/key.json unblinds.
score: per group, errors per 1,000 <dialogue> elements (sure+likely; higher count on double-read windows), sure-only,
windows with >= 1 error, per type, per source, 95% cluster bootstrap over windows (10,000 resamples); agreement on
double-read windows; decision (P10) against R_ref with the control calibration.
DIR holds screenplay text: keep it out of the repository.
"""
from __future__ import annotations

import argparse
import difflib
import glob
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extra_sources.label_sample import RUBRIC, TYPES, elements, load  # noqa: E402

R_REF, R_REF_CI = 10.7, (5.0, 17.5)
PROVENANCE_NOTE = ('\nAdd your provenance to the label file: "provenance": {"agent_id": "...", "model": "...", '
                   '"family": "..."} (model family, e.g. Claude, GPT, Gemini).\n')


def dialogue_count(seg: dict) -> int:
    return sum(1 for t, _ in elements(seg) if t == "dialogue")


def draw(args) -> None:
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists (samples are write-once)")
    rng = random.Random(args.seed)
    used = set()
    for path in args.exclude_file or []:
        with open(path) as f:
            used |= {line.strip() for line in f if line.strip()}
    for path in args.exclude_key or []:
        with open(path) as f:
            used |= {v for ents in json.load(f)["windows"].values() for e in ents for v in (e["item_id"], e["content_sha1"])}
    c34 = set()
    if args.c34_list:
        with open(args.c34_list) as f:
            data = json.load(f)
        items = data.get("items", data) if isinstance(data, dict) else data
        c34 = {x["item_id"] if isinstance(x, dict) else x for x in items}

    def fresh(s):
        return s["item_id"] not in used and s.get("content_sha1") not in used

    src = [s for s in load(args.sources) if fresh(s)]
    by_src = defaultdict(list)
    for s in src:
        by_src[s["source_dataset"]].append(s)
    total = sum(len(v) for v in by_src.values())
    n = args.n_sources
    while True:
        alloc = {k: max(args.min_per_source, round(n * len(v) / total)) for k, v in by_src.items()}
        picks = {k: rng.sample(sorted(v, key=lambda s: s["item_id"]), min(alloc[k], len(v))) for k, v in by_src.items()}
        if sum(dialogue_count(s) for v in picks.values() for s in v) >= args.min_dialogue or n > total:
            break
        n += 3
    ctl_pool = [s for s in load(args.control) if fresh(s) and s["item_id"] not in c34]
    control = rng.sample(sorted(ctl_pool, key=lambda s: s["item_id"]), args.n_control)
    windows = [("sources", s) for v in picks.values() for s in v] + [("moviesum_control", s) for s in control]
    rng.shuffle(windows)
    os.makedirs(os.path.join(args.out, "windows"))
    os.makedirs(os.path.join(args.out, "assign"))
    os.makedirs(os.path.join(args.out, "labels"))
    key = {}
    for i, (g, s) in enumerate(windows, 1):
        bid = f"W{i:03d}"
        with open(os.path.join(args.out, "windows", bid + ".xml"), "w", encoding="utf-8") as f:
            f.write(s["script_segment"] + "\n")
        key[bid] = {"group": g, "source": s.get("source_dataset"), "item_id": s["item_id"], "movie_name": s["movie_name"],
                    "imdb_id": s["imdb_id"], "content_sha1": s.get("content_sha1"), "dialogue_elements": dialogue_count(s)}
    ids = list(key)
    labelers = [f"L{i:02d}" for i in range(1, args.labelers + 1)]
    primary = {bid: labelers[i % len(labelers)] for i, bid in enumerate(ids)}
    n_double = max(args.external_reads, -(-int(args.double_frac * 100) * len(ids) // 100))
    doubles = rng.sample(ids, n_double)
    second = {}
    for j, bid in enumerate(doubles):
        if j < args.external_reads:
            second[bid] = "XF"
        else:
            others = [x for x in labelers if x != primary[bid]]
            second[bid] = others[j % len(others)]
    assign = defaultdict(list)
    for bid, lab in list(primary.items()) + list(second.items()):
        assign[lab].append(bid)
    for lab, lst in assign.items():
        rng.shuffle(lst)
        target = args.external_dir if lab == "XF" else args.out
        os.makedirs(os.path.join(target, "assign"), exist_ok=True)
        with open(os.path.join(target, "assign", lab + ".txt"), "w") as f:
            f.write("\n".join(lst) + "\n")
    rubric = RUBRIC.replace("label_sample", "c05pp sample") + PROVENANCE_NOTE
    with open(os.path.join(args.out, "RUBRIC.md"), "w") as f:
        f.write(rubric)
    if args.external_dir:
        os.makedirs(os.path.join(args.external_dir, "windows"), exist_ok=True)
        os.makedirs(os.path.join(args.external_dir, "labels"), exist_ok=True)
        for bid in assign["XF"]:
            shutil.copy(os.path.join(args.out, "windows", bid + ".xml"), os.path.join(args.external_dir, "windows", bid + ".xml"))
        with open(os.path.join(args.external_dir, "RUBRIC.md"), "w") as f:
            f.write(rubric.replace("labels/<labeler>.json", "labels/XF.json"))
    with open(os.path.join(args.out, "key.json"), "w") as f:
        json.dump({"seed": args.seed, "freeze": args.freeze, "sources_build": args.sources, "control_build": args.control,
                   "windows": key, "primary": primary, "second": second, "external_dir": args.external_dir}, f, indent=1)
    per = Counter(v["source"] for v in key.values() if v["group"] == "sources")
    print(json.dumps({"windows": len(key), "sources": dict(per), "control": args.n_control,
                      "sources_dialogue": sum(v["dialogue_elements"] for v in key.values() if v["group"] == "sources"),
                      "double": len(second), "external": args.external_reads,
                      "assignments": {k: len(v) for k, v in sorted(assign.items())}}, indent=1))


def _cluster_boot(rows, rng, reps=10000):
    vals = []
    for _ in range(reps):
        pick = [rows[rng.randrange(len(rows))] for _ in rows]
        vals.append(1000 * sum(p[0] for p in pick) / max(1, sum(p[1] for p in pick)))
    vals.sort()
    return [round(vals[int(0.025 * reps)], 2), round(vals[int(0.975 * reps) - 1], 2)]


def score(args) -> None:
    with open(os.path.join(args.dir, "key.json")) as f:
        key = json.load(f)
    labels, prov = {}, {}
    for path in sorted(glob.glob(os.path.join(args.dir, "labels", "*.json")) +
                       (glob.glob(os.path.join(args.external_dir, "labels", "*.json")) if args.external_dir else [])):
        with open(path) as f:
            lab = json.load(f)
        labels[lab["labeler"]] = lab["windows"]
        prov[lab["labeler"]] = lab.get("provenance")
    need = [(b, l) for b, l in list(key["primary"].items()) + list(key["second"].items()) if b not in labels.get(l, {})]
    if need:
        sys.exit(f"missing labels: {need[:10]} ({len(need)})")

    def errs(b, lab, sure=False):
        return [e for e in labels[lab][b] if e.get("type") in TYPES and (not sure or e.get("confidence") == "sure")]

    rng = random.Random(1)
    out = {"provenance": prov, "groups": {}, "per_source": {}, "agreement": {}}
    win = {}
    for b, k in key["windows"].items():
        reads = [key["primary"][b]] + ([key["second"][b]] if b in key["second"] else [])
        best = max(reads, key=lambda lab: len(errs(b, lab)))  # P7: the higher count on a double-read window
        win[b] = {"n": len(errs(b, best)), "sure": len(errs(b, best, True)), "dlg": k["dialogue_elements"],
                  "types": Counter(e["type"] for e in errs(b, best)), "group": k["group"], "source": k["source"]}

    def summarize(bids):
        rows = [(win[b]["n"], win[b]["dlg"]) for b in bids]
        d = sum(r[1] for r in rows) or 1
        return {"windows": len(bids), "dialogue_elements": d, "errors": sum(r[0] for r in rows),
                "per_1k": round(1000 * sum(r[0] for r in rows) / d, 2), "ci95": _cluster_boot(rows, rng) if rows else None,
                "per_1k_sure": round(1000 * sum(win[b]["sure"] for b in bids) / d, 2),
                "windows_with_any": round(sum(win[b]["n"] > 0 for b in bids) / max(1, len(bids)), 3),
                "by_type": dict(sum((win[b]["types"] for b in bids), Counter()))}
    for g in ("sources", "moviesum_control"):
        out["groups"][g] = summarize([b for b in win if win[b]["group"] == g])
    for s in sorted({win[b]["source"] for b in win if win[b]["group"] == "sources"}):
        out["per_source"][s] = summarize([b for b in win if win[b]["group"] == "sources" and win[b]["source"] == s])
    pairs = []
    for b, sec in key["second"].items():
        a, c = errs(b, key["primary"][b]), errs(b, sec)
        qa, qc = [e.get("quote", "")[:60].lower() for e in a], [e.get("quote", "")[:60].lower() for e in c]
        matched = sum(1 for q in qa if any(difflib.SequenceMatcher(None, q, r).ratio() >= 0.6 for r in qc))
        pairs.append({"window": b, "primary": key["primary"][b], "second": sec, "n_primary": len(a), "n_second": len(c),
                      "matched": matched})
    n = len(pairs) or 1
    tot = sum(max(p["n_primary"], p["n_second"]) for p in pairs)
    out["agreement"] = {"double_read": len(pairs),
                        "any_error_agreement": round(sum((p["n_primary"] > 0) == (p["n_second"] > 0) for p in pairs) / n, 3),
                        "counts_within_1": round(sum(abs(p["n_primary"] - p["n_second"]) <= 1 for p in pairs) / n, 3),
                        "matched_errors_share": round(sum(p["matched"] for p in pairs) / tot, 3) if tot else None,
                        "external_reads": sum(1 for p in pairs if p["second"] == "XF"), "pairs": pairs}
    r_src, r_ctl = out["groups"]["sources"]["per_1k"], out["groups"]["moviesum_control"]["per_1k"]
    if r_ctl < R_REF_CI[0]:
        decision, bar = "void (control below the reference CI: labelers too lax)", None
    else:
        bar = r_ctl if r_ctl > R_REF_CI[1] else R_REF
        worst = max(v["per_1k"] for v in out["per_source"].values())
        decision = "pass" if r_src <= bar and worst <= 2 * R_REF else "fail"
    out["decision"] = {"R_sources": r_src, "R_control": r_ctl, "R_ref": R_REF, "bar": bar,
                       "max_source_rate": max(v["per_1k"] for v in out["per_source"].values()), "per_source_cap": 2 * R_REF,
                       "agreement_ok": out["agreement"]["counts_within_1"] >= 0.8, "result": decision}
    print(json.dumps({k: v for k, v in out.items() if k != "agreement"} | {"agreement": {k: v for k, v in out["agreement"].items() if k != "pairs"}}, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--sources", required=True)
    d.add_argument("--control", required=True)
    d.add_argument("--c34_list", help="MovieSum C34 frozen list (windows dropped from the release)")
    d.add_argument("--exclude_file", action="append", help="item_ids / content_sha1s used before, one per line")
    d.add_argument("--exclude_key", action="append", help="key.json of an earlier label sample")
    d.add_argument("--n_sources", type=int, default=36)
    d.add_argument("--min_per_source", type=int, default=8)
    d.add_argument("--min_dialogue", type=int, default=2500)
    d.add_argument("--n_control", type=int, default=24)
    d.add_argument("--labelers", type=int, default=8)
    d.add_argument("--double_frac", type=float, default=0.24)
    d.add_argument("--external_reads", type=int, default=7)
    d.add_argument("--external_dir", required=True)
    d.add_argument("--freeze", required=True, help="parser commit and build ids, recorded in the key")
    d.add_argument("--seed", type=int, required=True)
    d.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--dir", required=True)
    s.add_argument("--external_dir")
    s.add_argument("--out")
    args = ap.parse_args()
    draw(args) if args.cmd == "draw" else score(args)


if __name__ == "__main__":
    main()
