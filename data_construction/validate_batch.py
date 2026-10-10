"""Merge gate for one abstract batch: decides whether a written batch may be merged into the release.

  python data_construction/validate_batch.py --run_dir RUN --batch RUN/batches/moviesum-b0001

Blocking gates (all must pass; contract IDs in brackets):
  G1 complete     every manifest item has a valid abstract JSON (item_id, abstract, prompt_version == manifest) and its
                  author is on the manifest's allowed_authors list
  G2 integrity    the abstract's content_sha1 (required) == sha1(content file) == manifest == run items.jsonl
  G3 hard checks  check_abstracts hard-pass for every item [C19]
  G4 grounding    lexical grounding >= 0.35 for every item [C23]
  G5 length       >= 70% of items inside target_words [C20; release level is >= 85%]
  G6 boilerplate  first 8 words unique within the batch and against every already merged abstract [C24]
  G7 schema       the merged record passes dataset_schema.validate_record
Non-blocking G8 (audit routing, C31): an item goes to the targeted audit pool, outside the 2% random sample, when its
abstract names the film's title (not as words of the excerpt), has 1-2 ungrounded proper nouns, misses the top speaker
or the last third, or the writer reported a source problem for it (batches/<id>/writer_report.json).
Faithfulness/coverage (C25'-C28') are judged by the audit; a major or outside error revokes the batch (C31).
Items on the C35 content-safety list (content_exclusions.json: listed items and every window of a listed film) are not
part of any batch for G1-G7: they need no abstract, an abstract written for one is ignored, and G5 counts without them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_abstracts import check_one  # noqa: E402
from dataset_schema import make_record, validate_record  # noqa: E402

MIN_IN_TARGET = 0.7
MIN_GROUNDING = 0.35
DEFAULT_AUTHORS = ["claude-opus-5.5"]
DEFAULT_CONTENT_EXCLUSIONS = "/cursor/stores/bc-36270f16-3cac-4d49-8f64-7dfb9bff8601/internal/dataset-expansion/content_exclusions.json"
G8_SOFT = {"some_ungrounded_proper_nouns": "ungrounded_names", "top_speaker_missing": "top_speaker_missing",
           "last_third_not_covered": "last_third_not_covered"}


def title_mention(abstract: str, movie_name: str, content: str) -> bool:
    """The abstract uses the film's title (>= 2 words, or one word of >= 5 letters) and the excerpt never does."""
    title = re.sub(r"_\d{4}$", "", movie_name)
    title = re.sub(r"^(the|a|an)\s+", "", title, flags=re.I).split(":")[0].strip()
    words = re.findall(r"[A-Za-z']+", title)
    if not words or (len(words) == 1 and len(words[0]) < 5):
        return False
    pat = r"\b" + r"\W+".join(map(re.escape, words)) + r"\b"
    return bool(re.search(pat, abstract, re.I)) and not re.search(pat, content, re.I)


def load_content_exclusions(path: str | None = DEFAULT_CONTENT_EXCLUSIONS) -> dict:
    """C35 list: {"items": {item_id: content_sha1}, "films": {imdb_id}}; empty when the file is absent."""
    if not path or not os.path.exists(path):
        return {"items": {}, "films": set()}
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    return {"items": {x["item_id"]: x.get("content_sha1") for x in spec.get("items", [])},
            "films": {x["imdb_id"] for x in spec.get("films", [])}}


def content_excluded(item_ids, items_by_id: dict, c35: dict) -> set[str]:
    """The given items that C35 excludes: listed by item_id (whatever their content now is), or a window of a listed film."""
    return {i for i in item_ids if i in c35["items"] or (items_by_id.get(i) or {}).get("imdb_id") in c35["films"]
            or i.split("-s")[0] in c35["films"]}


def writer_issues(batch_dir: str) -> set[str]:
    path = os.path.join(batch_dir, "writer_report.json")
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        rep = json.load(f)
    return {x["item_id"] for x in rep.get("source_issues", []) if x.get("item_id")}


def first8(text: str) -> str:
    return " ".join(text.split()[:8]).lower()


def abstract_set_hash(manifest: dict) -> str:
    """Identifies the exact abstract files a verdict was computed for (cache key; revocations bind to it)."""
    h = hashlib.sha1()
    for e in manifest["items"]:
        h.update(e["item_id"].encode() + b"\0" + e["content_sha1"].encode() + b"\0")
        if os.path.exists(e["abstract_path"]):
            with open(e["abstract_path"], "rb") as f:
                h.update(f.read())
        h.update(b"\1")
    return h.hexdigest()


def validate(manifest: dict, items_by_id: dict, merged_first8: set[str], count_tokens=None, skip: set[str] | None = None) -> dict:
    from make_abstract_batches import target_center  # noqa: PLC0415

    if skip:
        manifest = {**manifest, "items": [e for e in manifest["items"] if e["item_id"] not in skip]}
        if not manifest["items"]:  # every item is C35-listed: nothing to write, nothing to merge
            return {"state": "passed", "failures": [], "soft": [], "records": [], "items": [], "audit_targets": []}
    failures, soft, records, per_item, targets = [], [], [], [], []
    present = [os.path.exists(e["abstract_path"]) for e in manifest["items"]]
    if not any(present):
        return {"state": "pending", "failures": [], "soft": [], "records": [], "items": [], "audit_targets": []}
    allowed = set(manifest.get("allowed_authors") or DEFAULT_AUTHORS)
    issues = writer_issues(os.path.dirname(manifest["scratch_dir"]))
    seen8 = set()
    for e, ok in zip(manifest["items"], present):
        iid = e["item_id"]
        if not ok:
            failures.append(f"G1:{iid}:missing")
            continue
        try:
            with open(e["abstract_path"], encoding="utf-8") as f:
                ab = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            failures.append(f"G1:{iid}:invalid_json")
            continue
        if ab.get("item_id") != iid or not isinstance(ab.get("abstract"), str) or not ab.get("author"):
            failures.append(f"G1:{iid}:bad_fields")
            continue
        if ab.get("prompt_version") != manifest["prompt_version"]:
            failures.append(f"G1:{iid}:prompt_version={ab.get('prompt_version')}")
        if ab.get("author") not in allowed:
            failures.append(f"G1:{iid}:author_not_allowed={ab.get('author')}")
        with open(e["content_path"], encoding="utf-8") as f:
            content = f.read()
        item = items_by_id.get(iid)
        sha = hashlib.sha1(content.encode()).hexdigest()
        if not ab.get("content_sha1"):
            failures.append(f"G2:{iid}:content_sha1_missing")
            continue
        if item is None or not (sha == e["content_sha1"] == item["content_sha1"] == ab["content_sha1"]):
            failures.append(f"G2:{iid}:content_changed")
            continue
        chk = check_one(ab["abstract"], content, e["target_words"])
        if chk["hard"]:
            failures.append(f"G3:{iid}:{','.join(chk['hard'])}")
        if chk.get("lexical_grounding", 0) < MIN_GROUNDING:
            failures.append(f"G4:{iid}:grounding={chk.get('lexical_grounding')}")
        key = first8(ab["abstract"])
        if key in seen8 or key in merged_first8:
            failures.append(f"G6:{iid}:duplicate_opening")
        seen8.add(key)
        rec = make_record(item, batch_id=manifest["batch_id"], build=manifest["build_id"], abstract=ab, count_tokens=count_tokens)
        problems = validate_record(rec)
        if problems:
            failures.append(f"G7:{iid}:{','.join(problems)}")
        soft.extend(f"{iid}:{s}" for s in chk["soft"])
        reasons = [G8_SOFT[x] for x in chk["soft"] if x in G8_SOFT]
        if title_mention(ab["abstract"], item["movie_name"], content):
            reasons.append("title_mention")
        if iid in issues:
            reasons.append("writer_source_issue")
        if reasons:
            targets.append({"item_id": iid, "batch_id": manifest["batch_id"], "reasons": reasons})
        per_item.append({"item_id": iid, "words": chk["words"], "target_center": target_center(item["content_tokens"]),
                         "paragraphs": chk.get("paragraphs"), "hard": chk["hard"], "soft": chk["soft"],
                         "grounding": chk.get("lexical_grounding")})
        records.append(rec)
    if all(present) and per_item:
        in_target = sum("word_count_outside_target" not in p["soft"] for p in per_item) / len(manifest["items"])
        if in_target < MIN_IN_TARGET:
            failures.append(f"G5:in_target={in_target:.2f}")
    state = "passed" if all(present) and not failures else ("incomplete" if not all(present) else "rejected")
    return {"state": state, "failures": failures, "soft": soft, "records": records if state == "passed" else [],
            "items": per_item, "audit_targets": targets}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--content_exclusions", default=DEFAULT_CONTENT_EXCLUSIONS, help="C35 list ('' for none)")
    args = ap.parse_args()
    with open(os.path.join(args.run_dir, "items.jsonl"), encoding="utf-8") as f:
        items = {it["item_id"]: it for it in map(json.loads, f)}
    with open(os.path.join(args.batch, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    skip = content_excluded([e["item_id"] for e in manifest["items"]], items, load_content_exclusions(args.content_exclusions))
    res = validate(manifest, items, set(), skip=skip)
    print(json.dumps({k: res[k] for k in ("state", "failures", "soft", "audit_targets")} | {"c35_skipped": sorted(skip)}, indent=2))
    sys.exit(0 if res["state"] == "passed" else 1)


if __name__ == "__main__":
    main()
