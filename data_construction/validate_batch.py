"""Merge gate for one abstract batch: decides whether a written batch may be merged into the release.

  python data_construction/validate_batch.py --run_dir RUN --batch RUN/batches/moviesum-b0001

Blocking gates (all must pass; contract IDs in brackets):
  G1 complete     every manifest item has a valid abstract JSON (item_id, abstract, prompt_version == manifest, author)
  G2 integrity    sha1(content file) == manifest content_sha1 == run items.jsonl content_sha1 (== abstract's, if given)
  G3 hard checks  check_abstracts hard-pass for every item [C19]
  G4 grounding    lexical grounding >= 0.35 for every item [C23]
  G5 length       >= 70% of items inside target_words [C20; release level is >= 85%]
  G6 boilerplate  first 8 words unique within the batch and against every already merged abstract [C24]
  G7 schema       the merged record passes dataset_schema.validate_record
Soft signals (reported, never blocking): top speaker named, thirds covered, paragraphs, words vs target_center.
Faithfulness/coverage (C25-C28) are judged by the evaluator's sample audit; a failed audit revokes the batch (C31).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_abstracts import check_one  # noqa: E402
from dataset_schema import make_record, validate_record  # noqa: E402

MIN_IN_TARGET = 0.7
MIN_GROUNDING = 0.35


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


def validate(manifest: dict, items_by_id: dict, merged_first8: set[str], count_tokens=None) -> dict:
    from make_abstract_batches import target_center  # noqa: PLC0415

    failures, soft, records, per_item = [], [], [], []
    present = [os.path.exists(e["abstract_path"]) for e in manifest["items"]]
    if not any(present):
        return {"state": "pending", "failures": [], "soft": [], "records": [], "items": []}
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
        with open(e["content_path"], encoding="utf-8") as f:
            content = f.read()
        item = items_by_id.get(iid)
        sha = hashlib.sha1(content.encode()).hexdigest()
        if item is None or not (sha == e["content_sha1"] == item["content_sha1"]) or ab.get("content_sha1", sha) != sha:
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
            "items": per_item}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--batch", required=True)
    args = ap.parse_args()
    with open(os.path.join(args.run_dir, "items.jsonl"), encoding="utf-8") as f:
        items = {it["item_id"]: it for it in map(json.loads, f)}
    with open(os.path.join(args.batch, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    res = validate(manifest, items, set())
    print(json.dumps({k: res[k] for k in ("state", "failures", "soft")}, indent=2))
    sys.exit(0 if res["state"] == "passed" else 1)


if __name__ == "__main__":
    main()
