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
    return words(unicodedata.normalize("NFKC", html.unescape(html.unescape(text))))


def words(text: str) -> list[str]:
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return re.findall(r"[a-z0-9]+", text.lower())


def segment_scene_words(segment: str) -> list[list[str]]:
    scenes = re.findall(r"<scene>(.*?)</scene>", segment, re.S)
    return [words(html.unescape(re.sub(r"<[^>]+>", " ", s))) for s in scenes]


def check(item: dict, cache_dir: str) -> dict:
    raw = raw_words(os.path.join(cache_dir, item.get("source_cache_file") or item["source_file"]), item["source_format"])
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
    near6 = {tuple(near_ws[i:i + 6]) for i in range(len(near_ws) - 5)}
    seg6 = {tuple(seg[i:i + 6]) for i in range(len(seg) - 5)}

    def covered(ws, i, grams):
        return any(tuple(ws[k:k + 6]) in grams for k in range(max(0, i - 5), min(i, len(ws) - 6) + 1))
    speakers = {w for c in re.findall(r"<character>(.*?)</character>", item["script_segment"]) for w in words(html.unescape(c))}
    sm = difflib.SequenceMatcher(None, region, seg, autojunk=False)
    inserted, reordered, del_white, del_declared, del_other, runs, ins_runs = 0, 0, 0, 0, 0, [], []
    for op, a1, a2, b1, b2 in sm.get_opcodes():
        if op in ("insert", "replace"):
            # words inside a 6-gram that also occurs in the raw text nearby are real text the aligner paired with
            # another copy of a repeated passage; only the rest counts as inserted
            found = sum(covered(seg, j, near6) for j in range(b1, b2))
            reordered += found
            if b2 - b1 - found:
                inserted += b2 - b1 - found
                ins_runs.append(" ".join(seg[max(0, b1 - 3):b1]) + " {{" + " ".join(seg[b1:b2]) + "}} " + " ".join(seg[b2:b2 + 3]))
        if op in ("delete", "replace"):
            run = region[a1:a2]
            moved = [covered(region, j, seg6) for j in range(a1, a2)]
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
    return {"item_id": item["item_id"], "segment_words": len(seg), "region_words": len(region), "inserted": inserted, "aligned_elsewhere": reordered,
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
    }
    return summary, fails


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--cache_dir", default="data_construction/work/sources/extra/raw")
    ap.add_argument("--out")
    args = ap.parse_args()
    with open(args.segments, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    results = [check(it, args.cache_dir) for it in items]
    tot = {k: sum(r[k] for r in results) for k in ("segment_words", "inserted", "aligned_elsewhere", "deleted_whitelisted",
                                                   "deleted_declared", "deleted_other")}
    summary = {"items": len(results), **tot,
               "items_with_inserted": sum(r["inserted"] > 0 for r in results),
               "items_with_other_deletions": sum(r["deleted_other"] > 0 for r in results)}
    print(json.dumps(summary, indent=2))
    for r in results:
        if r["inserted"] or r["deleted_other"]:
            print(r["item_id"], "ins", r["inserted"], "del_other", r["deleted_other"], r["other_runs"][:4])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "items": results}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
