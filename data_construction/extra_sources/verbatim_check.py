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

WHITELIST = re.compile(
    r"^(continued|cont|contd|d|more|omitted|omit|\d{1,4}[a-z]{0,3}|pt|rev|revised|revision|revisions|draft|pink|blue|yellow|"
    r"green|goldenrod|buff|salmon|cherry|tan|white|page|final|shooting|script|of)$"
)


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
            text = html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"(?is)<(script|style|head)[^>]*>.*?</\1>", " ", text)))
    return words(text)


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
    anchor = seg[:12]
    start = -1
    for i in range(len(raw) - len(anchor)):
        if raw[i : i + len(anchor)] == anchor:
            start = i
            break
    if start < 0:
        sm = difflib.SequenceMatcher(None, raw, anchor, autojunk=False)
        start = sm.find_longest_match(0, len(raw), 0, len(anchor)).a
    end_anchor = seg[-12:]
    end = start + len(seg)
    for j in range(start, min(len(raw), start + 3 * len(seg))):
        if raw[j : j + len(end_anchor)] == end_anchor:
            end = j + len(end_anchor)
    region = raw[start:end]
    speakers = {w for c in re.findall(r"<character>(.*?)</character>", item["script_segment"]) for w in words(html.unescape(c))}
    sm = difflib.SequenceMatcher(None, region, seg, autojunk=False)
    inserted, del_white, del_declared, del_other, runs = 0, 0, 0, 0, []
    for op, a1, a2, b1, b2 in sm.get_opcodes():
        if op in ("insert", "replace"):
            inserted += b2 - b1
        if op in ("delete", "replace"):
            run = region[a1:a2]
            other = [w for w in run if not WHITELIST.match(w)]
            del_white += len(run) - len(other)
            if other and all(w in speakers for w in other):
                del_declared += len(other)  # bare speaker-name line without dialogue (removed by design, C12c)
                continue
            del_other += len(other)
            if other:
                runs.append(" ".join(region[max(0, a1 - 4):a1]) + " [[" + " ".join(run) + "]] " + " ".join(region[a2:a2 + 4]))
    return {"item_id": item["item_id"], "segment_words": len(seg), "region_words": len(region), "inserted": inserted,
            "deleted_whitelisted": del_white, "deleted_declared": del_declared, "deleted_other": del_other, "dropped_scenes": item.get("dropped_scenes", []),
            "other_runs": runs[:15]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--cache_dir", default="data_construction/work/sources/extra/raw")
    ap.add_argument("--out")
    args = ap.parse_args()
    with open(args.segments, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    results = [check(it, args.cache_dir) for it in items]
    tot = {k: sum(r[k] for r in results) for k in ("segment_words", "inserted", "deleted_whitelisted", "deleted_declared", "deleted_other")}
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
