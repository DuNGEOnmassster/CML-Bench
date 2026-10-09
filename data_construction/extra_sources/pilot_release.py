"""Write-once pilot release for extra sources (one abstract run per source, as make_abstract_batches requires).

  python data_construction/extra_sources/pilot_release.py --runs RUN_IMSDB RUN_DAILYSCRIPT ... --out REL --name extra_sources_v1

REL/data/<slug>.jsonl holds schema-1.0 records (dataset_schema.make_record; every record passes
validate_record), REL/info.json follows gt_100_info.json (assemble.info_json), REL/pilot.json is the pilot
metadata hf_sync.py expects. Only items whose abstract passed check_abstracts' hard checks are included.
Refuses to write into an existing folder (pilots are write-once).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from assemble import info_json  # noqa: E402
from check_abstracts import check_one  # noqa: E402
from cml_format import count_tokens  # noqa: E402
from dataset_schema import SCHEMA_VERSION, make_record, source_slug, validate_record  # noqa: E402
from make_abstract_batches import target_words  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--note", default="extra screenplay sources beyond MovieSum")
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; pilot releases are write-once")

    all_records, skipped = [], Counter()
    by_slug: dict[str, list] = {}
    for run in args.runs:
        with open(os.path.join(run, "items.jsonl"), encoding="utf-8") as f:
            items = [json.loads(line) for line in f]
        for it in items:
            path = os.path.join(run, "abstracts", f"{it['item_id']}.json")
            if not os.path.exists(path):
                skipped["no_abstract"] += 1
                continue
            with open(path, encoding="utf-8") as f:
                abstract = json.load(f)
            if check_one(abstract["abstract"], it["script_segment"], target_words(it["content_tokens"]))["hard"]:
                skipped["hard_fail"] += 1
                continue
            rec = make_record(it, batch_id=it["batch_id"], build=it["build_id"], abstract=abstract, count_tokens=count_tokens)
            problems = validate_record(rec)
            if problems:
                sys.exit(f"{it['item_id']}: schema problems {problems}")
            by_slug.setdefault(source_slug(rec["source_dataset"]), []).append(rec)
            all_records.append(rec)

    os.makedirs(os.path.join(args.out, "data"))
    for slug, recs in by_slug.items():
        with open(os.path.join(args.out, "data", f"{slug}.jsonl"), "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info_json(all_records), f, indent=1, ensure_ascii=False)
    meta = {
        "name": args.name, "schema_version": SCHEMA_VERSION, "items": len(all_records),
        "movies": len({r["imdb_id"] for r in all_records}),
        "sources": dict(Counter(r["source_dataset"] for r in all_records)),
        "build_ids": sorted({r["build_id"] for r in all_records}),
        "prompt_version": ", ".join(sorted({r["abstract_prompt_version"] for r in all_records})),
        "content_normalization": ", ".join(sorted({r["content_normalization"] for r in all_records})),
        "items_eval_safe": sum(r["eval_safe"] for r in all_records), "skipped": dict(skipped), "note": args.note,
    }
    with open(os.path.join(args.out, "pilot.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
