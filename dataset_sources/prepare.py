"""Prepare reviewed LOCAL source assets. This is not a benchmark release command.

The manifest and exclusion ledger are required, versionable inputs. Unknown
rights fail closed. Output never invents summaries, speakers, or movie IDs.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from .common import KINDS, digest, load_json, read_jsonl, registry, write_json, write_jsonl
from .parse import parse, windows
from .rights import license_partition, rights_errors
from .splits import assign


def validate_identity(record):
    required = (
        "record_id",
        "source_id",
        "source_url",
        "work_id",
        "edition_id",
        "story_family_id",
        "title",
        "language",
        "text_path",
        "format",
    )
    for key in required:
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError("missing_identity:" + key)
    if record.get("kind") not in KINDS:
        raise ValueError("unsupported_kind")
    if record.get("identity_reviewed") is not True:
        raise ValueError("identity_and_adaptation_review_required")
    if (
        not isinstance(record.get("creator_ids"), list)
        or not record["creator_ids"]
        or not all(isinstance(x, str) and x for x in record["creator_ids"])
    ):
        raise ValueError("creator_ids_required")
    if not isinstance(record.get("related_family_ids", []), list) or not all(
        isinstance(x, str) and x for x in record.get("related_family_ids", [])
    ):
        raise ValueError("invalid_related_family_ids")
    if not isinstance(record.get("genres"), list):
        raise ValueError("genres_array_required")


def load_exclusions(path):
    spec = load_json(path)  # absent file is an error, never an implicit empty ledger
    if not isinstance(spec, dict) or spec.get("schema_version") != "1.0":
        raise ValueError("unsupported exclusion ledger")
    for name in ("record_ids", "work_ids", "story_family_ids", "asset_sha256"):
        if not isinstance(spec.get(name), list) or not all(isinstance(x, str) for x in spec[name]):
            raise ValueError("invalid exclusion ledger field: " + name)
    return spec


def is_excluded(record, exclusions):
    return (
        record["record_id"] in exclusions["record_ids"]
        or record["work_id"] in exclusions["work_ids"]
        or bool(
            {record["story_family_id"], *record.get("related_family_ids", [])}
            & set(exclusions["story_family_ids"])
        )
        or record.get("rights", {}).get("asset_sha256") in exclusions["asset_sha256"]
    )


def prepare(
    manifest,
    exclusions_path,
    out,
    *,
    allow_fixtures=False,
    max_chars=24000,
    min_chars=2000,
    seed="cml-literary-v1",
):
    if not 0 < min_chars <= max_chars:
        raise ValueError("require 0 < min_chars <= max_chars")
    manifest = Path(manifest)
    records = list(read_jsonl(manifest))
    if not all(isinstance(r, dict) for r in records):
        raise ValueError("manifest rows must be JSON objects")
    ids = [r.get("record_id") for r in records]
    if not all(isinstance(i, str) and i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("missing/duplicate manifest record IDs")
    sources = registry()
    exclusions = load_exclusions(exclusions_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    accepted, rejected, parsed = [], [], {}
    root = manifest.resolve().parent
    for record in sorted(records, key=lambda r: r["record_id"]):
        try:
            validate_identity(record)
            if record["source_id"] not in sources:
                raise ValueError("unregistered_source")
            reasons = rights_errors(
                record, sources[record["source_id"]], allow_fixtures=allow_fixtures
            )
            if reasons:
                raise ValueError(";".join(reasons))
            if is_excluded(record, exclusions):
                raise ValueError("excluded_by_ledger")
            path = (root / record["text_path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("text_path_must_stay_within_manifest_directory")
            if path.stat().st_size > 30_000_000:
                raise ValueError("asset_exceeds_30MB_limit")
            raw = path.read_bytes()
            if digest(raw) != record["rights"]["asset_sha256"]:
                raise ValueError("asset_changed_since_rights_review")
            blocks = parse(raw, record["format"], record["kind"])
            # Text-only fingerprint catches formatting differences. Work/family
            # identity still must resolve translations and different editions.
            normalized = " ".join(b["text"] for b in blocks)
            normalized = re.sub(r"\s+", " ", normalized).strip().casefold()
            if not normalized:
                raise ValueError("empty_normalized_text")
            row = {k: v for k, v in record.items() if k != "text_path"}
            row["normalized_sha256"] = digest(normalized.encode())
            row["license_partition"] = license_partition(record)
            accepted.append(row)
            parsed[row["record_id"]] = blocks
        except (ValueError, OSError, UnicodeError) as exc:
            rejected.append({"record_id": record["record_id"], "reason": str(exc)[:600]})

    # Related adaptations inherit an exclusion through transitive family links.
    all_families = {
        r["story_family_id"] for r in records if isinstance(r.get("story_family_id"), str)
    }
    # Rows excluded directly are not in assign(); preserve their declared family
    # relations in an explicit graph, including bridges via excluded works.
    adjacency = {f: set() for f in all_families}
    for r in records:
        if (
            r.get("identity_reviewed") is not True
            or not isinstance(r.get("story_family_id"), str)
            or not isinstance(r.get("related_family_ids", []), list)
        ):
            continue
        f = r["story_family_id"]
        for rel in r.get("related_family_ids", []):
            if not isinstance(rel, str):
                continue
            adjacency.setdefault(f, set()).add(rel)
            adjacency.setdefault(rel, set()).add(f)
    blocked = set(exclusions["story_family_ids"])
    # Excluding an entire work also excludes its adaptations from this release.
    blocked.update(
        r["story_family_id"]
        for r in records
        if r.get("work_id") in exclusions["work_ids"] and isinstance(r.get("story_family_id"), str)
    )
    queue = list(blocked)
    while queue:
        for linked in adjacency.get(queue.pop(), set()) - blocked:
            blocked.add(linked)
            queue.append(linked)
    # Preserve transitive split links even if an intermediate edition fails a
    # parser/rights gate. Only reviewed identity edges may contribute.
    for row in accepted:
        reachable, todo = set(), [row["story_family_id"]]
        while todo:
            family = todo.pop()
            if family in reachable:
                continue
            reachable.add(family)
            todo.extend(adjacency.get(family, set()) - reachable)
        row["related_family_ids"] = sorted(reachable - {row["story_family_id"]})
    survivors = []
    for row in accepted:
        if row["story_family_id"] in blocked:
            rejected.append({"record_id": row["record_id"], "reason": "excluded_related_family"})
        else:
            survivors.append(row)
    accepted = survivors
    assignments = assign(accepted, seed)
    segments, duplicates, window_rejections, seen = [], [], [], {}
    for row in accepted:
        row.update(assignments[row["record_id"]])
        fp = row["normalized_sha256"]
        if fp in seen:
            duplicates.append({"record_id": row["record_id"], "duplicate_of": seen[fp]})
            continue
        seen[fp] = row["record_id"]
        wins, drops = windows(parsed[row["record_id"]], max_chars, min_chars)
        window_rejections.extend({"record_id": row["record_id"], **d} for d in drops)
        for i, blocks in enumerate(wins):
            payload = json.dumps(blocks, ensure_ascii=False, sort_keys=True).encode()
            content_hash = digest(payload)
            segments.append(
                {
                    "schema_version": "1.0",
                    "segment_id": f"{row['record_id']}:{i}:{content_hash[:12]}",
                    "record_id": row["record_id"],
                    "work_id": row["work_id"],
                    "edition_id": row["edition_id"],
                    "story_family_id": row["story_family_id"],
                    "source_id": row["source_id"],
                    "kind": row["kind"],
                    "language": row["language"],
                    "split": row["split"],
                    "split_group": row["split_group"],
                    "license_partition": row["license_partition"],
                    "synthetic_fixture": row.get("synthetic_fixture", False),
                    "asset_sha256": row["rights"]["asset_sha256"],
                    "content_sha256": content_hash,
                    "parser_version": "literary-blocks-v1",
                    "blocks": blocks,
                    "summary": None,
                    "annotation_status": "pending",
                    "benchmark_ready": False,
                }
            )
    active = {s["record_id"] for s in segments}
    works = [r for r in accepted if r["record_id"] in active]
    write_jsonl(out / "works.jsonl", works)
    write_jsonl(out / "segments.jsonl", segments)
    write_jsonl(out / "rejections.jsonl", rejected)
    write_jsonl(out / "duplicate_editions.jsonl", duplicates)
    write_jsonl(out / "window_rejections.jsonl", window_rejections)
    write_jsonl(
        out / "attribution.jsonl",
        [
            {"record_id": r["record_id"], "source_url": r["source_url"], "rights": r["rights"]}
            for r in works
        ],
    )
    counts = Counter(r["work_id"] for r in works)
    segments_per_work = Counter(s["work_id"] for s in segments)
    creator_splits = {}
    for row in works:
        for creator in row["creator_ids"]:
            creator_splits.setdefault(creator, set()).add(row["split"])

    def distinct_by(field):
        values = {r[field] for r in works}
        return {v: len({r["work_id"] for r in works if r[field] == v}) for v in sorted(values)}

    report = {
        "schema_version": "1.0",
        "benchmark_ready": False,
        "release_ready": False,
        "fixture_mode": allow_fixtures,
        "scope": "prepared corpus; summaries, semantic/rights review and benchmark validation are not certified",
        "input_records": len(records),
        "prepared_editions": len(works),
        "distinct_works": len(counts),
        "story_families": len({r["split_group"] for r in works}),
        "segments": len(segments),
        "rejected_records": len(rejected),
        "duplicate_editions": len(duplicates),
        "dropped_windows_or_blocks": len(window_rejections),
        "by_source_works": distinct_by("source_id"),
        "by_kind_works": distinct_by("kind"),
        "by_language_works": distinct_by("language"),
        "by_split_segments": dict(Counter(s["split"] for s in segments)),
        "by_license_partition": dict(Counter(s["license_partition"] for s in segments)),
        "largest_work_segment_share": max(segments_per_work.values(), default=0)
        / max(1, len(segments)),
        "authors_crossing_splits": sum(len(s) > 1 for s in creator_splits.values()),
        "author_holdout_enforced": False,
        "near_duplicate_review_complete": False,
        "split_seed": seed,
        "split_policy": "family-connected-components-80-10-10-hash",
        "manifest_sha256": digest(manifest.read_bytes()),
        "exclusions_sha256": digest(Path(exclusions_path).read_bytes()),
        "registry_sha256": digest((Path(__file__).parent / "registry.json").read_bytes()),
        "config": {"min_chars": min_chars, "max_chars": max_chars},
        "output_sha256": {p.name: digest(p.read_bytes()) for p in sorted(out.glob("*.jsonl"))},
    }
    write_json(out / "report.json", report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--exclusions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-fixtures", action="store_true")
    ap.add_argument("--min-chars", type=int, default=2000)
    ap.add_argument("--max-chars", type=int, default=24000)
    ap.add_argument("--seed", default="cml-literary-v1")
    args = ap.parse_args()
    try:
        report = prepare(
            args.manifest,
            args.exclusions,
            args.out,
            allow_fixtures=args.allow_fixtures,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            seed=args.seed,
        )
    except (ValueError, OSError) as exc:
        ap.exit(2, f"prepare: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # Exclusions are expected, but require the caller to inspect the report.
    raise SystemExit(2 if report["rejected_records"] or not report["segments"] else 0)


if __name__ == "__main__":
    main()
