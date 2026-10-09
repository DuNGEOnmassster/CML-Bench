"""Idempotent, incremental sync of the expanded dataset to a PRIVATE Hugging Face dataset repo.

Repo layout (`<slug>` = source slug, e.g. moviesum; each source is owned by the worker that builds it):
  README.md                        dataset card, regenerated on every sync
  schema.json                      frozen item schema (dataset_schema.py)
  build_status.json                aggregate status, recomputed from status/*.json on every sync
  control.json                     phase + gates (set with `status --phase/--gate`)
  status/<slug>.json               per-source status
  content/<slug>/part-00000.jsonl  every item of the source's current build, summary "" (config `content`)
  manifests/<slug>/batches.jsonl   batch_id -> item_ids + content_sha1s (abstract work units)
  data/<slug>/<batch_id>.safe.jsonl     merged items with checked abstracts, eval_safe == true
  data/<slug>/<batch_id>.related.jsonl  merged items with checked abstracts, gt_related != null
  pilots/<name>/...                write-once pilot releases

Commands:
  sync   --run_dir RUN            reconcile one source with RUN: content + manifests + every batch that passes
                                  validate_batch.py (merged) + status. Run it again whenever abstracts land.
  pilot  --folder REL --name N    upload a pilot release once; refuses to change an existing pilot
  status [--phase T] [--gate k=v] recompute build_status.json / README (and set phase/gates)
Every run diffs desired vs remote file hashes, commits only changes against the current head (parent_commit,
retried on conflict), and mirrors build_status.json to the project store. Re-running is a no-op.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset_schema import SCHEMA_VERSION, make_record, schema_json, source_slug, validate_record  # noqa: E402

STORE_STATUS = "/cursor/stores/bc-36270f16-3cac-4d49-8f64-7dfb9bff8601/internal/dataset-expansion/build_status.json"
DEFAULT_VERDICTS = "/cursor/stores/bc-36270f16-3cac-4d49-8f64-7dfb9bff8601/internal/dataset-expansion/audit"
SHARD_BATCHES = 100
DEFAULT_REPO = f"{os.environ.get('HF_ACCOUNT', '')}/CML-Dataset-Expanded"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def jsonl_bytes(records) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode()


def json_bytes(obj) -> bytes:
    return (json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=False) + "\n").encode()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class Repo:
    def __init__(self, repo_id: str):
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not token:
            sys.exit("HF_TOKEN is not set; add a write token (Cursor Dashboard -> Cloud Agents -> Secrets).")
        from huggingface_hub import HfApi

        self.api = HfApi(token=token)
        self.repo_id = repo_id

    def ensure_private(self) -> None:
        self.api.create_repo(self.repo_id, repo_type="dataset", private=True, exist_ok=True)
        if not self.api.repo_info(self.repo_id, repo_type="dataset").private:
            sys.exit(f"{self.repo_id} is PUBLIC; refusing to upload copyrighted screenplay text.")

    def head(self) -> str:
        return self.api.repo_info(self.repo_id, repo_type="dataset").sha

    def tree(self, revision: str) -> dict:
        out = {}
        for f in self.api.list_repo_tree(self.repo_id, repo_type="dataset", recursive=True, revision=revision):
            if hasattr(f, "blob_id"):
                lfs = getattr(f, "lfs", None)
                out[f.path] = {"blob": f.blob_id, "sha256": (lfs.sha256 if lfs is not None else None)}
        return out

    def read(self, path: str, revision: str) -> bytes:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(self.repo_id, path, repo_type="dataset", revision=revision, token=self.api.token)
        with open(local, "rb") as f:
            return f.read()

    def commit(self, plan, message: str, attempts: int = 6):
        """plan(head, tree) -> (adds {path: bytes}, deletes [path]); only changed files are sent."""
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        from huggingface_hub.errors import HfHubHTTPError

        for attempt in range(attempts):
            head = self.head()
            tree = self.tree(head)
            adds, deletes = plan(head, tree)
            ops = []
            for path, data in sorted(adds.items()):
                cur = tree.get(path)
                if cur and (cur["sha256"] == hashlib.sha256(data).hexdigest() if cur["sha256"] else cur["blob"] == git_blob_sha1(data)):
                    continue
                ops.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=data))
            ops += [CommitOperationDelete(path_in_repo=p) for p in sorted(set(deletes)) if p in tree]
            if not ops:
                return head, 0
            try:
                info = self.api.create_commit(self.repo_id, ops, repo_type="dataset", commit_message=message, parent_commit=head)
                return info.oid, len(ops)
            except HfHubHTTPError as exc:
                status = getattr(exc.response, "status_code", None)
                if status in (409, 412) and attempt < attempts - 1:
                    time.sleep(2 + 3 * attempt)
                    continue
                raise
        raise RuntimeError("commit kept conflicting")


def stable_status(new: dict, old_bytes: bytes | None) -> dict:
    """Keep the old updated_at when nothing else changed, so re-runs produce identical bytes."""
    if old_bytes:
        old = json.loads(old_bytes)
        if {k: v for k, v in old.items() if k != "updated_at"} == {k: v for k, v in new.items() if k != "updated_at"}:
            new["updated_at"] = old.get("updated_at", new["updated_at"])
    return new


def remote_json(repo: Repo, tree: dict, path: str, head: str):
    return json.loads(repo.read(path, head)) if path in tree else None


def aggregate(repo: Repo, head: str, tree: dict, override: dict[str, bytes]) -> dict:
    """build_status.json from status/*.json + control.json + pilots/*/pilot.json (override = files about to be written)."""
    def get(path):
        if path in override:
            return json.loads(override[path])
        return remote_json(repo, tree, path, head)

    paths = sorted({p for p in list(tree) + list(override) if p.startswith("status/") and p.endswith(".json")})
    sources = {}
    for p in paths:
        st = get(p)
        if st:
            sources[st["source_dataset"]] = {k: v for k, v in st.items() if k != "batch_states"}
    control = get("control.json") or {"phase": "", "gates": {}}
    pilots = {}
    for p in sorted({p for p in list(tree) + list(override) if p.startswith("pilots/") and p.endswith("/pilot.json")}):
        meta = get(p)
        pilots[meta["name"]] = meta
    keys = ("items_total", "movies", "items_eval_safe", "items_gt_related", "items_with_abstract", "items_checks_passed",
            "items_merged", "batches_total", "batches_merged", "batches_rejected", "content_tokens_total")
    totals = {k: sum(s.get(k, 0) for s in sources.values()) for k in keys}
    updated = max([s["updated_at"] for s in sources.values()] + [control.get("updated_at", "")] +
                  [m.get("uploaded_at", "") for m in pilots.values()] or [now()])
    return {
        "status_format": 1,
        "dataset_repo": repo.repo_id,
        "schema_version": SCHEMA_VERSION,
        "updated_at": updated,
        "phase": control.get("phase", ""),
        "gates": control.get("gates", {}),
        "totals": totals,
        "per_source": sources,
        "pilots": pilots,
        "definitions": {
            "items_total": "items in the source's current content build (content/<slug>/)",
            "items_with_abstract": "items with an abstract file written against the current content (content_sha1 matches)",
            "items_checks_passed": "items whose abstract passes the per-item merge gates G1-G4, G7 (validate_batch.py)",
            "items_merged": "items in data/ (their whole batch passed all gates and is not revoked by the audit)",
        },
    }


def card(status: dict, tree_paths: set[str]) -> bytes:
    has_data = any(p.startswith("data/") for p in tree_paths)
    has_safe = any(p.startswith("data/") and p.endswith(".safe.jsonl") for p in tree_paths)
    configs = []
    if has_data:
        configs.append(("release", "data/*/*.jsonl", True))
        if has_safe:
            configs.append(("eval_safe", "data/*/*.safe.jsonl", False))
    configs.append(("content", "content/*/*.jsonl", not has_data))
    for name, meta in sorted(status["pilots"].items()):
        configs.append((f"pilot_{name}", f"pilots/{name}/data/*.jsonl", False))
    yaml_cfg = "\n".join(
        f"- config_name: {n}\n  data_files:\n  - split: train\n    path: \"{p}\"" + ("\n  default: true" if d else "")
        for n, p, d in configs
    )
    t = status["totals"]
    rows = "\n".join(
        f"| {src} | {s['build_id']} | {s['items_total']:,} | {s['movies']:,} | {s['items_eval_safe']:,} | "
        f"{s['items_with_abstract']:,} | {s['items_merged']:,} |"
        for src, s in sorted(status["per_source"].items())
    )
    pilots = "\n".join(f"- `pilot_{n}`: {m.get('items')} items, prompt `{m.get('prompt_version')}`, content `{m.get('content_normalization')}` "
                       f"— {m.get('note', '')}" for n, m in sorted(status["pilots"].items()))
    text = f"""---
license: cc-by-nc-4.0
language:
- en
pretty_name: CML-Dataset Expanded
tags:
- screenplay
- movie-scripts
- summarization
- cml-bench
configs:
{yaml_cfg}
---

# CML-Dataset Expanded (private, work in progress)

Expansion of the CML-Dataset used by [CML-Bench](https://github.com/DuNGEOnmassster/CML-Bench) (arXiv:2510.06231):
contiguous excerpts of human-written screenplays (`script_segment`, Cinematic Markup Language) paired with
AI-written abstracts (`summary`). Built by `data_construction/` in the CML-Bench repo.

**Phase:** {status['phase'] or 'n/a'}

| Source | Build | Items | Movies | Eval-safe | With abstract | Merged |
|---|---|---|---|---|---|---|
{rows}
| **Total** | | {t['items_total']:,} | {t['movies']:,} | {t['items_eval_safe']:,} | {t['items_with_abstract']:,} | {t['items_merged']:,} |

## Configs

- `release`: items whose abstract passed every merge gate (`data/<source>/<batch>.{{safe,related}}.jsonl`).
- `eval_safe`: the `release` items with no story/franchise relation to the 100 CML-Bench GT movies (`gt_related == null`).
- `content`: every item of the current content build; `summary` is empty until its abstract is merged.
{pilots}

Machine-readable progress: `build_status.json`. Item schema (v{SCHEMA_VERSION}): `schema.json`. The first four fields
match CML-Bench `ground_truth/gt_100.json` (`movie_name`, `imdb_id`, `script_segment`, `summary`).
GT segments start with a newline before `<script>` and usually end with one; these items have neither.
Windows with `relative_position == 0` may open with title-page text.

## Provenance and copyright

Source screenplays: [MovieSum](https://huggingface.co/datasets/rohitsaxena/MovieSum) (CC BY-NC 4.0 curation); each item
records `source_url`, `source_file`, `scene_start`/`scene_end`, `dropped_scenes` and `imdb_url`. The CML-Bench GT movies
are excluded (IMDb id, title, 13-gram text overlap, curated remake table). Film identity is checked against IMDb cast
names: screenplays MovieSum files under the wrong film are relabelled (`source_label` keeps MovieSum's label), early
drafts carry `script_version: draft`, and films awaiting a human identity read are in `*-hold-*` batches that are not
written. Abstracts were written to land within a few words of a per-item target length, so their length spread is
narrower than GT's. Audit pools and verdicts: `audit/<source>/`. The screenplays remain the property of their rights
holders: this dataset is private and for non-commercial research only. Do not make it public or redistribute it.
"""
    return text.encode()


def write_store_status(status: dict, path: str) -> None:
    if path and os.path.isdir(os.path.dirname(path)):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=1, ensure_ascii=False)
            f.write("\n")


def source_state(run_dir: str, verdicts_path: str | None = None):
    """Desired repo files for one source + its status, computed from the local run dir (+ the audit verdicts file)."""
    from audit_rules import paused_orchestrators, sample_batches, sample_item, stop_rule
    from cml_format import count_tokens
    from validate_batch import abstract_set_hash, first8, validate

    with open(os.path.join(run_dir, "run.json"), encoding="utf-8") as f:
        run = json.load(f)
    with open(os.path.join(run_dir, "items.jsonl"), encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    with open(os.path.join(run_dir, "batches.jsonl"), encoding="utf-8") as f:
        batches = [json.loads(line) for line in f]
    items_by_id = {it["item_id"]: it for it in items}
    slug, build = run["source_slug"], run["build_id"]
    source = items[0]["source_dataset"]
    revoked_path = os.path.join(run_dir, "revoked.json")
    revoked = json.load(open(revoked_path, encoding="utf-8")) if os.path.exists(revoked_path) else {}
    cache_path = os.path.join(run_dir, "merge_cache.json")
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
    fanout_path = os.path.join(run_dir, "fanout.json")
    ranges = json.load(open(fanout_path, encoding="utf-8"))["orchestrators"] if os.path.exists(fanout_path) else []
    verdicts = []
    paths = []
    if verdicts_path and os.path.isdir(verdicts_path):
        paths = sorted(os.path.join(verdicts_path, fn) for fn in os.listdir(verdicts_path) if re.match(r"verdicts.*\.jsonl$", fn))
    elif verdicts_path and os.path.exists(verdicts_path):
        paths = [verdicts_path]
    for path in paths:  # one file per auditor, so reviewers never write the same file
        with open(path, encoding="utf-8") as f:
            verdicts += [v for v in map(json.loads, filter(str.strip, f)) if v.get("item_id") in items_by_id]
    verdicts.sort(key=lambda v: v.get("audited_at", ""))
    bad_verdict = {(v["item_id"], v.get("summary_sha1")) for v in verdicts if v.get("major", 0) or v.get("outside", 0)}
    sampled = sample_batches([b["batch_id"] for b in batches])
    targeted, sample = [], []

    files: dict[str, bytes] = {}
    content_records = []
    for b in batches:
        for iid in b["item_ids"]:
            rec = make_record(items_by_id[iid], batch_id=b["batch_id"], build=build)
            problems = validate_record(rec)
            if problems:
                sys.exit(f"schema violation in {iid}: {problems}")
            content_records.append(rec)
    per_shard = SHARD_BATCHES * run["batch_size"]
    for k in range(0, len(content_records), per_shard):
        files[f"content/{slug}/part-{k // per_shard:05d}.jsonl"] = jsonl_bytes(content_records[k : k + per_shard])
    with open(os.path.join(run_dir, "batches.jsonl"), "rb") as f:
        files[f"manifests/{slug}/batches.jsonl"] = f.read()

    merged_first8: set[str] = set()
    states, with_abstract, checks_passed, merged_items, prompt_versions = {}, 0, 0, 0, {}
    new_cache = {}
    for b in batches:
        bdir = os.path.join(run_dir, "batches", b["batch_id"])
        with open(os.path.join(bdir, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        key = abstract_set_hash(manifest)
        hit = cache.get(b["batch_id"])
        reusable = (hit and hit["key"] == key and not any(f.startswith("G6") for f in hit["res"]["failures"])
                    and not (set(hit["first8"]) & merged_first8))
        if reusable and hit["res"]["state"] in ("pending", "incomplete", "rejected"):
            res, rec_list = hit["res"], []
        else:
            full = validate(manifest, items_by_id, merged_first8, count_tokens=count_tokens)
            res = {k: full[k] for k in ("state", "failures", "items", "audit_targets")}
            rec_list = full["records"]
        first8s = [first8(r["summary"]) for r in rec_list]
        new_cache[b["batch_id"]] = {"key": key, "res": res, "first8": first8s}
        state = res["state"]
        if state == "passed" and revoked.get(b["batch_id"], {}).get("key") == key:
            state = "revoked"
        sha = {r["item_id"]: hashlib.sha1(r["summary"].encode()).hexdigest() for r in rec_list}
        if state == "passed" and any((i, h) in bad_verdict for i, h in sha.items()):
            state = "revoked"  # C31: a major/outside verdict on these exact abstracts
        if state != "pending":
            states[b["batch_id"]] = state
        failed_items = {f.split(":")[1] for f in res["failures"] if f.count(":") >= 2}
        with_abstract += len(res["items"])
        checks_passed += sum(1 for p in res["items"] if p["item_id"] not in failed_items)
        if state == "passed":
            merged_first8.update(first8s)
            merged_items += len(rec_list)
            targeted += [{**t, "summary_sha1": sha[t["item_id"]]} for t in res.get("audit_targets", [])]
            if b["batch_id"] in sampled:
                pick = sample_item([r["item_id"] for r in rec_list])
                sample.append({"item_id": pick, "batch_id": b["batch_id"], "summary_sha1": sha[pick], "reason": "stratified_2pct"})
            for r in rec_list:
                prompt_versions[r["abstract_prompt_version"]] = prompt_versions.get(r["abstract_prompt_version"], 0) + 1
            safe = [r for r in rec_list if r["eval_safe"]]
            related = [r for r in rec_list if not r["eval_safe"]]
            if safe:
                files[f"data/{slug}/{b['batch_id']}.safe.jsonl"] = jsonl_bytes(safe)
            if related:
                files[f"data/{slug}/{b['batch_id']}.related.jsonl"] = jsonl_bytes(related)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(new_cache, f)
    if targeted:
        files[f"audit/{slug}/targeted.jsonl"] = jsonl_bytes(targeted)
    if sample:
        files[f"audit/{slug}/sample.jsonl"] = jsonl_bytes(sample)
    if verdicts:
        files[f"audit/{slug}/verdicts.jsonl"] = jsonl_bytes(verdicts)
    rule = stop_rule(verdicts)
    audited_batches, seen_b = [], set()
    for v in verdicts:
        if v["batch_id"] not in seen_b:
            seen_b.add(v["batch_id"])
            audited_batches.append((v["batch_id"], any(x["batch_id"] == v["batch_id"] and (x.get("major", 0) or x.get("outside", 0))
                                                        for x in verdicts)))
    audit = {"targeted_items": len(targeted), "sample_items": len(sample), **rule,
             "paused_orchestrators": paused_orchestrators(audited_batches, ranges), "orchestrators": ranges}

    checks_path = os.path.join(run_dir, "content_checks.json")
    content_checks = None
    if os.path.exists(checks_path):
        with open(checks_path, encoding="utf-8") as f:
            rep = json.load(f)
        content_checks = {"report_build_id": rep.get("build_id"), **rep["summary"],
                          "failed": sorted(k for k, v in rep["results"].items() if v["pass"] is False)}
    count = lambda st: sum(1 for v in states.values() if v == st)  # noqa: E731
    status = {
        "source_dataset": source,
        "source_slug": slug,
        "schema_version": SCHEMA_VERSION,
        "build_id": build,
        "content_normalization": items[0]["content_normalization"],
        "prompt_version": run["prompt_version"],
        "items_total": len(items),
        "movies": len({it["imdb_id"] for it in items}),
        "items_eval_safe": sum(it["gt_related"] is None for it in items),
        "items_gt_related": sum(it["gt_related"] is not None for it in items),
        "content_tokens_total": sum(it["content_tokens"] for it in items),
        "items_with_abstract": with_abstract,
        "items_checks_passed": checks_passed,
        "items_merged": merged_items,
        "batches_total": len(batches),
        "batches_merged": count("passed"),
        "batches_rejected": count("rejected"),
        "batches_incomplete": count("incomplete"),
        "batches_revoked": count("revoked"),
        "batch_size": run["batch_size"],
        "merged_prompt_versions": prompt_versions,
        "content_checks": content_checks,
        "items_held": sum(it.get("identity_decision") == "hold" for it in items),
        "identity_decisions": dict(sorted({d: sum(it.get("identity_decision") == d for it in items)
                                           for d in {it.get("identity_decision") for it in items} if d}.items())),
        "audit": audit,
        "updated_at": now(),
        "batch_states": states,
    }
    return slug, files, status


def cmd_sync(repo: Repo, args) -> dict:
    slug, files, status = source_state(args.run_dir, args.verdicts)
    owned = (f"content/{slug}/", f"manifests/{slug}/", f"data/{slug}/", f"audit/{slug}/")
    result = {}

    def plan(head, tree):
        adds = dict(files)
        st_path = f"status/{slug}.json"
        old = repo.read(st_path, head) if st_path in tree else None
        adds[st_path] = json_bytes(stable_status(dict(status), old))
        adds["schema.json"] = json_bytes(schema_json())
        agg = aggregate(repo, head, tree, adds)
        adds["build_status.json"] = json_bytes(agg)
        paths = set(tree) | set(adds)
        deletes = [p for p in tree if p.startswith(owned) and p not in adds]
        paths -= set(deletes)
        adds["README.md"] = card(agg, paths)
        result["status"] = agg
        return adds, deletes

    oid, n = repo.commit(plan, f"sync {slug}: {status['build_id']}, merged {status['items_merged']}/{status['items_total']}")
    print(f"sync {slug}: {n} file operations, head {oid[:8]}")
    return result["status"]


def cmd_pilot(repo: Repo, args) -> dict:
    base = f"pilots/{args.name}/"
    local = {}
    for root, _, fns in os.walk(args.folder):
        for fn in sorted(fns):
            p = os.path.join(root, fn)
            with open(p, "rb") as f:
                local[base + os.path.relpath(p, args.folder).replace(os.sep, "/")] = f.read()
    meta = json.loads(local.get(base + "pilot.json", b"{}") or b"{}")
    if not meta.get("name") == args.name:
        sys.exit(f"{args.folder}/pilot.json must exist and have name={args.name}")
    result = {}

    def same(remote, data):
        return remote["sha256"] == hashlib.sha256(data).hexdigest() if remote["sha256"] else remote["blob"] == git_blob_sha1(data)

    def plan(head, tree):
        existing = {p for p in tree if p.startswith(base)}
        if existing:
            changed = [p for p in local if p in tree and not same(tree[p], local[p])]
            if changed or existing - set(local):
                sys.exit(f"pilot {args.name} already exists on the hub and differs ({(changed or sorted(existing - set(local)))[:3]}); "
                         "pilots are write-once — use a new --name")
        adds = dict(local)
        agg = aggregate(repo, head, tree, adds)
        adds["build_status.json"] = json_bytes(agg)
        adds["README.md"] = card(agg, set(tree) | set(adds))
        result["status"] = agg
        return adds, []

    oid, n = repo.commit(plan, f"pilot {args.name}")
    print(f"pilot {args.name}: {n} file operations, head {oid[:8]}")
    return result["status"]


def cmd_status(repo: Repo, args) -> dict:
    result = {}

    def plan(head, tree):
        adds = {}
        control = remote_json(repo, tree, "control.json", head) or {"phase": "", "gates": {}}
        new = json.loads(json.dumps(control))
        if args.phase is not None:
            new["phase"] = args.phase
        for kv in args.gate or []:
            k, _, v = kv.partition("=")
            new.setdefault("gates", {})[k] = v
        if new != control or "control.json" not in tree:
            new["updated_at"] = now()
            adds["control.json"] = json_bytes(new)
        adds["schema.json"] = json_bytes(schema_json())
        agg = aggregate(repo, head, tree, adds)
        adds["build_status.json"] = json_bytes(agg)
        adds["README.md"] = card(agg, set(tree) | set(adds))
        result["status"] = agg
        return adds, []

    oid, n = repo.commit(plan, "status")
    print(f"status: {n} file operations, head {oid[:8]}")
    return result["status"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_id", default=DEFAULT_REPO)
    ap.add_argument("--store_status", default=STORE_STATUS, help="local mirror of build_status.json ('' to skip)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync")
    s.add_argument("--run_dir", required=True)
    s.add_argument("--verdicts", default=DEFAULT_VERDICTS, help="audit verdicts: a jsonl file, or a dir of verdicts*.jsonl (C31); missing = none yet")
    p = sub.add_parser("pilot")
    p.add_argument("--folder", required=True)
    p.add_argument("--name", required=True)
    st = sub.add_parser("status")
    st.add_argument("--phase")
    st.add_argument("--gate", action="append", help="key=value")
    args = ap.parse_args()
    if args.repo_id.startswith("/"):
        sys.exit("set HF_ACCOUNT or pass --repo_id USER/NAME")

    repo = Repo(args.repo_id)
    repo.ensure_private()
    status = {"sync": cmd_sync, "pilot": cmd_pilot, "status": cmd_status}[args.cmd](repo, args)
    write_store_status(status, args.store_status)
    t = status["totals"]
    print(f"https://huggingface.co/datasets/{args.repo_id} (private): items {t['items_total']}, "
          f"with abstract {t['items_with_abstract']}, merged {t['items_merged']}")


if __name__ == "__main__":
    main()
