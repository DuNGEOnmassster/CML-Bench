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
                                  Merge-time exclusions (merge_exclusions/<build_id>.json, e.g. C34) and hold items
                                  never enter data/; their batches still pass or fail on G1-G7 as usual.
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
DEFAULT_AUDIT_MIRROR = DEFAULT_VERDICTS + "/{slug}"
DEFAULT_EXCLUSIONS = os.path.join(HERE, "merge_exclusions")
DEFAULT_CONTENT_EXCLUSIONS = "/cursor/stores/bc-36270f16-3cac-4d49-8f64-7dfb9bff8601/internal/dataset-expansion/content_exclusions.json"
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
    keys = ("items_total", "items_releasable", "movies", "items_eval_safe", "items_gt_related", "items_with_abstract",
            "items_checks_passed", "items_merged", "batches_total", "batches_merged", "batches_rejected", "content_tokens_total")
    totals = {k: sum(s.get(k, 0) for s in sources.values()) for k in keys}
    totals["items_merged_draft"] = sum(s.get("items_merged", 0) for s in sources.values() if s.get("abstracts_draft"))
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
            "items_releasable": "items_total minus merge-time exclusions (C34 mislabel windows, hold films); the release target",
            "items_merged_draft": "merged items of sources whose abstracts are still drafts (config release_draft)",
        },
    }


def card(status: dict, tree_paths: set[str]) -> bytes:
    draft_slugs = {s["source_slug"] for s in status["per_source"].values() if s.get("abstracts_draft")}
    data_slugs = sorted({p.split("/")[1] for p in tree_paths if p.startswith("data/") and p.count("/") >= 2})
    final = [s for s in data_slugs if s not in draft_slugs]
    drafts = [s for s in data_slugs if s in draft_slugs]
    safe = [s for s in final if any(p.startswith(f"data/{s}/") and p.endswith(".safe.jsonl") for p in tree_paths)]
    configs = []
    if final:
        configs.append(("release", [f"data/{s}/*.jsonl" for s in final], True))
        if safe:
            configs.append(("eval_safe", [f"data/{s}/*.safe.jsonl" for s in safe], False))
    if drafts:
        configs.append(("release_draft", [f"data/{s}/*.jsonl" for s in drafts], False))
    configs.append(("content", "content/*/*.jsonl", not final))
    for name, meta in sorted(status["pilots"].items()):
        configs.append((f"pilot_{name}", f"pilots/{name}/data/*.jsonl", False))

    def yaml_path(p):
        return f"path: \"{p}\"" if isinstance(p, str) else "path:\n" + "\n".join(f"    - \"{x}\"" for x in p)

    yaml_cfg = "\n".join(
        f"- config_name: {n}\n  data_files:\n  - split: train\n    {yaml_path(p)}" + ("\n  default: true" if d else "")
        for n, p, d in configs
    )
    draft_note = "\n".join(f"- {src}: {s['abstracts_draft']}" for src, s in sorted(status["per_source"].items())
                           if s.get("abstracts_draft")) or "- none"
    t = status["totals"]
    rows = "\n".join(
        f"| {src} | {s['build_id']} | {s['items_total']:,} | {s.get('items_releasable', s['items_total']):,} | {s['movies']:,} | "
        f"{s['items_eval_safe']:,} | {s['items_with_abstract']:,} | {s['items_merged']:,} |"
        for src, s in sorted(status["per_source"].items())
    )
    exclusions = "\n".join(
        f"- {src}: " + ", ".join(f"{n:,} `{r}`" for r, n in s["items_excluded"].items())
        + (f" (list `{s['merge_exclusion_list']['file']}`: {s['merge_exclusion_list']['rule']})" if s.get("merge_exclusion_list") else "")
        for src, s in sorted(status["per_source"].items()) if s.get("items_excluded")
    ) or "- none"
    n_gaps = sum((s.get("writer_notes") or {}).get("source_gaps_released", 0) for s in status["per_source"].values())
    gaps = (f"{n_gaps:,} released windows have a gap in the source screenplay reported by the abstract writer (a missing page, an "
            f"omitted sequence, dialogue \"on a separate document\"); they stay in the release, their abstracts are audited "
            f"for invented bridging (C31 Tier B), and they are listed in `audit/<source>/writer_signals.json`.") if n_gaps else ""
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

| Source | Build | Items | Releasable | Movies | Eval-safe | With abstract | Merged |
|---|---|---|---|---|---|---|---|
{rows}
| **Total** | | {t['items_total']:,} | {t.get('items_releasable', t['items_total']):,} | {t['movies']:,} | {t['items_eval_safe']:,} | {t['items_with_abstract']:,} | {t['items_merged']:,} |

Items in the content build that are never released (abstracts may still be written; they are dropped at merge, and
audit sampling and release statistics are computed without them):
{exclusions}

`C34` drops a window whose MovieSum markup is structurally mislabelled (dialogue that is only a parenthetical, a
speaker name tagged as dialogue, a heading/shot/action tagged as a speaker, speech fused into action) at least 4 times.
`identity_hold` films await a human read of their identity (C33) and stay out of the release until it is done.
`C35` removes windows, or whole films, on content-safety grounds; they are also left out of the `content` config.

## Configs

- `release`: items whose abstract passed every merge gate (`data/<source>/<batch>.{{safe,related}}.jsonl`), from
  sources whose abstracts are final.
- `eval_safe`: the `release` items with no story/franchise relation to the 100 CML-Bench GT movies (`gt_related == null`).
- `release_draft`: merged items of sources whose abstracts are still drafts (same gates; kept out of `release` until the
  source passes its remaining audit):
{draft_note}
- `content`: every item of the current content build; `summary` is empty until its abstract is merged.
{pilots}

Machine-readable progress: `build_status.json`. Item schema (v{SCHEMA_VERSION}): `schema.json`. The first four fields
match CML-Bench `ground_truth/gt_100.json` (`movie_name`, `imdb_id`, `script_segment`, `summary`).
GT segments start with a newline before `<script>` and usually end with one; these items have neither.
Windows with `relative_position == 0` may open with title-page text.
{gaps}

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


GAP_NOTE_RE = re.compile(r"PAGE MISSING|pages? (?:are |is )?missing|sequence omitted|omitted from (?:the )?original|separate document|"
                         r"scenes? (?:is |are )?(?:missing|omitted)|cut off mid|excerpt ends (?:with|mid|abruptly)", re.I)
MARKER_NOTE_RE = re.compile(r"\b(?:SCENES? )?(?:DELETED|OMITTED)\b")


def writer_notes(run_dir: str, batches: list[dict]) -> dict[str, str]:
    """item_id -> the writer's source-issue note (batch order; a later note for the same item wins, as in the
    evaluator's harness)."""
    notes = {}
    for b in batches:
        path = os.path.join(run_dir, "batches", b["batch_id"], "writer_report.json")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    rows = json.load(f).get("source_issues", [])
            except (ValueError, OSError):
                continue
            notes.update({x["item_id"]: x.get("note", "") for x in rows if x.get("item_id")})
    return notes


def writer_signals(notes: dict[str, str], items_by_id: dict, released: set[str]) -> dict:
    """Writer notes read as content signals (C31 amendment notes): source gaps for the card, identity doubts for C33,
    marker residue and films with >= 3 tagging/OCR noise notes for the next cleaning pass."""
    from targeted_rule import IDENTITY_RE, IMPACT_RE

    def row(iid):
        return {"item_id": iid, "movie_name": items_by_id[iid]["movie_name"], "released": iid in released, "note": notes[iid]}

    noisy = {}
    for iid, note in notes.items():
        if iid in items_by_id and not IMPACT_RE.search(note) and not IDENTITY_RE.search(note):
            noisy.setdefault(items_by_id[iid]["imdb_id"], []).append(iid)
    return {
        "notes": len(notes),
        "source_gaps": [row(i) for i in sorted(notes) if i in items_by_id and GAP_NOTE_RE.search(notes[i])],
        "identity_doubts": [row(i) for i in sorted(notes) if i in items_by_id and IDENTITY_RE.search(notes[i])],
        "marker_residue": [row(i) for i in sorted(notes) if i in items_by_id and MARKER_NOTE_RE.search(notes[i])],
        "noisy_films": [{"imdb_id": f, "movie_name": items_by_id[ids[0]]["movie_name"], "noise_notes": len(ids), "items": sorted(ids)}
                        for f, ids in sorted(noisy.items(), key=lambda kv: (-len(kv[1]), kv[0])) if len(ids) >= 3],
    }


def load_merge_exclusions(exclusions_dir: str | None, build: str, items_by_id: dict) -> tuple[dict[str, str], dict]:
    """item_id -> rule for the windows that must not be released from this exact build (one frozen list per build_id:
    a list made for another build means nothing for this one). Every listed item must match its content_sha1."""
    path = os.path.join(exclusions_dir, f"{build}.json") if exclusions_dir else ""
    if not path or not os.path.exists(path):
        return {}, {}
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    if spec.get("build_id") != build:
        sys.exit(f"{path}: made for build {spec.get('build_id')}, not {build}")
    rule = spec["rule"].split()[0]
    out = {}
    for x in spec["items"]:
        it = items_by_id.get(x["item_id"])
        if it is None or it["content_sha1"] != x["content_sha1"]:
            sys.exit(f"{path}: {x['item_id']} is not in build {build} with content_sha1 {x['content_sha1']}")
        out[x["item_id"]] = rule
    return out, {"file": os.path.relpath(path, HERE), "rule": spec["rule"], "listed_items": len(out)}


def source_state(run_dir: str, verdicts_path: str | None = None, exclusions_dir: str | None = DEFAULT_EXCLUSIONS,
                 draft_reason: str | None = None, content_exclusions: str | None = DEFAULT_CONTENT_EXCLUSIONS,
                 revalidate_all: bool = False):
    """Desired repo files for one source + its status, computed from the local run dir (+ the audit verdicts file).
    Excluded items (merge exclusion list, hold films) are left out of data/, of the audit pools and of all release
    statistics; the batch that carries them is still judged on all its items."""
    from audit_rules import dedupe_verdicts, orchestrator_audit, orchestrator_of, sample_batches, sample_item, stop_rule
    from cml_format import count_tokens
    from targeted_rule import select as select_targeted
    from targeted_rule import tier as targeted_tier
    from validate_batch import abstract_set_hash, content_excluded, first8, load_content_exclusions, validate

    with open(os.path.join(run_dir, "run.json"), encoding="utf-8") as f:
        run = json.load(f)
    with open(os.path.join(run_dir, "items.jsonl"), encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    with open(os.path.join(run_dir, "batches.jsonl"), encoding="utf-8") as f:
        batches = [json.loads(line) for line in f]
    items_by_id = {it["item_id"]: it for it in items}
    slug, build = run["source_slug"], run["build_id"]
    source = items[0]["source_dataset"]
    excluded, exclusion_spec = load_merge_exclusions(exclusions_dir, build, items_by_id)
    c35 = load_content_exclusions(content_exclusions)
    c35_ids = content_excluded(items_by_id, items_by_id, c35)
    excluded.update({i: "C35" for i in c35_ids})
    for it in items:
        if it.get("identity_decision") == "hold":
            excluded.setdefault(it["item_id"], "identity_hold")
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
    raw_verdicts, verdicts = verdicts, dedupe_verdicts(verdicts)
    bad_verdict = {(v["item_id"], v.get("summary_sha1")) for v in verdicts if v.get("major", 0) or v.get("outside", 0)}
    sampled = sample_batches([b["batch_id"] for b in batches])
    targeted, sample = [], []
    routed: dict[str, list[str]] = {}  # G8 reasons of every validated batch, revoked ones included

    files: dict[str, bytes] = {}
    content_records = []
    for b in batches:
        for iid in b["item_ids"]:
            if iid in c35_ids:
                continue
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
    excluded_from_merged: dict[str, int] = {}
    new_cache = {}
    rec_dir = os.path.join(run_dir, "merge_records")
    os.makedirs(rec_dir, exist_ok=True)
    validation = {"reused": 0, "validated": 0, "full": bool(revalidate_all)}
    for b in batches:
        bdir = os.path.join(run_dir, "batches", b["batch_id"])
        with open(os.path.join(bdir, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        skip = c35_ids & set(b["item_ids"])
        key = abstract_set_hash(manifest) + ("|c35:" + ",".join(sorted(skip)) if skip else "")
        report = os.path.join(bdir, "writer_report.json")
        if os.path.exists(report):  # G8 routing reads the writer's notes: a backfilled report re-validates the batch
            with open(report, "rb") as f:
                key += "|wr:" + hashlib.sha1(f.read()).hexdigest()[:16]
        hit = cache.get(b["batch_id"])
        reusable = (hit and hit["key"] == key and not any(f.startswith("G6") for f in hit["res"]["failures"])
                    and not (set(hit["first8"]) & merged_first8))
        rec_path = os.path.join(rec_dir, f"{b['batch_id']}.jsonl")
        if reusable and hit["res"]["state"] in ("pending", "incomplete", "rejected"):
            res, rec_list = hit["res"], []
        elif reusable and not revalidate_all and hit["res"]["state"] == "passed" and os.path.exists(rec_path):
            # same abstract files, same C35 skip set, no G6 clash: the records validated earlier still hold
            res = hit["res"]
            with open(rec_path, encoding="utf-8") as f:
                rec_list = [json.loads(line) for line in f]
            validation["reused"] += 1
        else:
            full = validate(manifest, items_by_id, merged_first8, count_tokens=count_tokens, skip=skip)
            res = {k: full[k] for k in ("state", "failures", "items", "audit_targets")}
            rec_list = full["records"]
            validation["validated"] += full["state"] != "pending"
            if res["state"] == "passed":
                with open(rec_path, "wb") as f:
                    f.write(jsonl_bytes(rec_list))
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
        routed.update({t["item_id"]: t.get("reasons", []) for t in res.get("audit_targets", [])})
        failed_items = {f.split(":")[1] for f in res["failures"] if f.count(":") >= 2}
        with_abstract += len(res["items"])
        checks_passed += sum(1 for p in res["items"] if p["item_id"] not in failed_items)
        if state == "passed":
            merged_first8.update(first8s)
            for r in rec_list:
                if r["item_id"] in excluded:
                    excluded_from_merged[excluded[r["item_id"]]] = excluded_from_merged.get(excluded[r["item_id"]], 0) + 1
            rec_list = [r for r in rec_list if r["item_id"] not in excluded]
            merged_items += len(rec_list)
            targeted += [{**t, "summary_sha1": sha[t["item_id"]]} for t in res.get("audit_targets", []) if t["item_id"] not in excluded]
            if b["batch_id"] in sampled and rec_list:
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
    notes = {i: n for i, n in writer_notes(run_dir, batches).items() if i not in c35_ids}
    selected = [{**t, "writer_note": notes.get(t["item_id"], "")} for t in select_targeted(targeted, notes, merged_items)]
    released = {r["item_id"] for p, data in files.items() if p.startswith(f"data/{slug}/") for r in map(json.loads, data.decode().splitlines())}
    signals = writer_signals(notes, items_by_id, released)
    if targeted:
        files[f"audit/{slug}/targeted.jsonl"] = jsonl_bytes(targeted)
    if selected:
        files[f"audit/{slug}/targeted_selected.jsonl"] = jsonl_bytes(selected)
    if notes:
        files[f"audit/{slug}/writer_signals.json"] = json_bytes(signals)
    if sample:
        files[f"audit/{slug}/sample.jsonl"] = jsonl_bytes(sample)
    if raw_verdicts:
        files[f"audit/{slug}/verdicts.jsonl"] = jsonl_bytes(raw_verdicts)
    targeted_meta = {i: {"tier": targeted_tier(rs, notes.get(i, "")), "reasons": rs} for i, rs in routed.items()}
    # Random-sample membership is the design, not the current merge state: the sampled batch's pick among its items
    # left after exclusions, whether the batch is passed, revoked or rewritten. A revoked batch drops out of
    # sample.jsonl, but its verdicts stay random-sample verdicts (items in both pools count as random).
    design_sample = set()
    for b in batches:
        if b["batch_id"] in sampled:
            kept = [i for i in b["item_ids"] if i not in excluded]
            if kept:
                design_sample.add(sample_item(kept))
    rule = stop_rule(verdicts, design_sample, targeted_meta)
    files[f"audit/{slug}/targeted_alarm.json"] = json_bytes({**rule["targeted_alarm"], "targeted_pool": rule["targeted_pool"]})
    sample_ids = design_sample
    trig = {}
    for v in verdicts:
        if v.get("major", 0) or v.get("outside", 0):
            trig.setdefault(v["batch_id"], set()).add("sample" if v["item_id"] in sample_ids else "targeted")
    orch = orchestrator_audit(verdicts, sample_ids, ranges)
    paused = orch["paused"]
    ta = rule["targeted_alarm"]
    ta["by_orchestrator"] = orch["targeted_by_orchestrator"]
    if orch["targeted_concentrated"]:
        ta["reason"] = "; ".join(filter(None, [ta["reason"], f"{', '.join(orch['targeted_concentrated'])}: targeted revocations "
                                                               f"concentrated (>= 3 of the last 20 targeted-audited batches, >= 2x the "
                                                               f"other orchestrators' rate); the evaluator decides on a pause"]))
        ta["fired"], ta["orchestrators"] = True, orch["targeted_concentrated"]
    files[f"audit/{slug}/targeted_alarm.json"] = json_bytes({**ta, "targeted_pool": rule["targeted_pool"]})
    revoked_list = [{"batch_id": b, "orchestrator": orchestrator_of(b, ranges), "trigger": sorted(trig.get(b) or {"manual"})}
                    for b, s in sorted(states.items()) if s == "revoked"]
    tiers = {k: sum(t["tier"] == k for t in selected) for k in ("A", "A4", "B", "C")}
    audit = {"targeted_items": len(targeted), "targeted_selected": len(selected), "targeted_tiers": tiers,
             "targeted_selected_share": round(len(selected) / merged_items, 4) if merged_items else None,
             "sample_items": len(sample), **rule,
             "revoked_batch_ids": [r["batch_id"] for r in revoked_list], "revoked_batches": revoked_list,
             "paused_orchestrators": paused, "orchestrators": ranges}

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
        "abstracts_draft": draft_reason or None,
        "items_held": sum(it.get("identity_decision") == "hold" for it in items),
        "items_releasable": len(items) - len(excluded),
        "items_excluded": {r: sum(v == r for v in excluded.values()) for r in sorted(set(excluded.values()))},
        "items_excluded_from_merged": dict(sorted(excluded_from_merged.items())),
        "merge_exclusion_list": exclusion_spec or None,
        "content_exclusion_list": {"file": "content_exclusions.json (project store)", "listed_items": len(c35["items"]),
                                   "listed_films": len(c35["films"]), "excluded_here": len(c35_ids)} if (c35["items"] or c35["films"]) else None,
        "writer_notes": {"notes": signals["notes"],
                         "source_gaps_released": sum(r["released"] for r in signals["source_gaps"]),
                         "identity_doubts": len(signals["identity_doubts"]), "marker_residue": len(signals["marker_residue"]),
                         "noisy_films": len(signals["noisy_films"])},
        "identity_decisions": dict(sorted({d: sum(it.get("identity_decision") == d for it in items)
                                           for d in {it.get("identity_decision") for it in items} if d}.items())),
        "audit": audit,
        "updated_at": now(),
        "batch_states": states,
    }
    print(f"validation {slug}: reused {validation['reused']}, validated {validation['validated']}, full {validation['full']}")
    return slug, files, status


def cmd_sync(repo: Repo, args) -> dict:
    slug, files, status = source_state(args.run_dir, args.verdicts, args.exclusions, args.draft, args.content_exclusions, args.full)
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
    write_local_release(args.run_dir, slug, files)
    if args.audit_mirror:
        mirror_audit(args.audit_mirror.format(slug=slug), slug, files)
    return result["status"]


def _records(files: dict, prefix: str) -> list[dict]:
    return [json.loads(line) for p in sorted(files) if p.startswith(prefix) for line in files[p].decode().splitlines() if line]


def write_local_release(run_dir: str, slug: str, files: dict) -> None:
    """The merged items as a release folder (data/merged.jsonl + info.json) for the release-level contract gates."""
    from assemble import info_json

    out = os.path.join(run_dir, "merged_release")
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    recs = _records(files, f"data/{slug}/")
    with open(os.path.join(out, "data", "merged.jsonl"), "wb") as f:
        f.write(jsonl_bytes(recs))
    with open(os.path.join(out, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info_json(recs), f, ensure_ascii=False)


def _write_if_changed(path: str, data: bytes) -> bool:
    if os.path.exists(path):
        with open(path, "rb") as f:
            if f.read() == data:
                return False
    with open(path, "wb") as f:
        f.write(data)
    return True


def mirror_audit(out_dir: str, slug: str, files: dict) -> None:
    """For auditors without HF access: the audit lists plus every listed item's content and abstract, in the store."""
    os.makedirs(out_dir, exist_ok=True)
    lists = {name: _records(files, f"audit/{slug}/{name}.jsonl") for name in ("sample", "targeted", "targeted_selected")}
    wanted, note = {}, {}
    for name, rows in lists.items():
        for r in rows:
            tag = {"sample": "sample", "targeted": "targeted:" + ",".join(r.get("reasons", []))}.get(name) or f"selected:tier_{r['tier']}"
            wanted.setdefault(r["item_id"], []).append(tag)
            if r.get("writer_note"):
                note[r["item_id"]] = r["writer_note"]
    items = []
    for rec in _records(files, f"data/{slug}/"):
        if rec["item_id"] in wanted:
            items.append({"item_id": rec["item_id"], "batch_id": rec["batch_id"], "movie_name": rec["movie_name"],
                          "summary_sha1": hashlib.sha1(rec["summary"].encode()).hexdigest(), "audit_lists": wanted[rec["item_id"]],
                          "writer_note": note.get(rec["item_id"], ""),
                          "abstract_prompt_version": rec["abstract_prompt_version"], "abstract_author": rec["abstract_author"],
                          "script_segment": rec["script_segment"], "summary": rec["summary"]})
    outputs = [(f"{name}.jsonl", jsonl_bytes(rows)) for name, rows in lists.items()] + [("items.jsonl", jsonl_bytes(items))]
    for name in ("writer_signals.json", "targeted_alarm.json"):
        if files.get(f"audit/{slug}/{name}"):
            outputs.append((name, files[f"audit/{slug}/{name}"]))
    changed = [name for name, data in outputs if _write_if_changed(os.path.join(out_dir, name), data)]
    print(f"audit mirror {out_dir}: sample {len(lists['sample'])}, targeted {len(lists['targeted'])} "
          f"(selected {len(lists['targeted_selected'])}), items {len(items)}, rewrote {changed}")


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
    s.add_argument("--audit_mirror", default=DEFAULT_AUDIT_MIRROR, help="store dir for audit lists + listed items ('' to skip)")
    s.add_argument("--exclusions", default=DEFAULT_EXCLUSIONS, help="dir of <build_id>.json merge-time exclusion lists ('' for none)")
    s.add_argument("--draft", default="", help="mark this source's merged abstracts as drafts, with this reason (config release_draft)")
    s.add_argument("--content_exclusions", default=DEFAULT_CONTENT_EXCLUSIONS, help="C35 content-safety list ('' for none)")
    s.add_argument("--full", action="store_true", help="re-validate every handed-off batch instead of reusing unchanged passed ones")
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
