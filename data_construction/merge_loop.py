"""One iteration of the single merger's loop (fanout-kickoff.md section 6). Run it every ~30 minutes.

  python3 data_construction/merge_loop.py --run_dir data_construction/work/full_v31

1. unpack the orchestrators' hand-offs (store abstracts/<build_id>/<batch_id>/{abstracts.jsonl, writer_report.json})
   into the run dir, leaving unchanged files untouched;
2. hf_sync.py sync (merge gates G1-G8, audit sample/targeted pool/verdicts, C31 gates, status, store audit mirror);
3. when the sync changed the hub, re-export the static dashboard from the dashboard branch's code;
4. release-level contract gates on the merged items (contract_checks.py --release_gates); once >= MIN_FOR_STOP items
   are merged, a failing gate sets mass_abstract_writing to "stop: ...".
Rounds are incremental: only new or changed bundles (abstracts, writer report, C35 skip set) are re-validated and the
rest reuse their stored records. Every FULL_EVERY seconds, on --full, and once more when every main batch is merged,
the round is full: every batch is re-validated and the release gates run. C35 purging, audit mirroring and revocations
run every round. The dashboard export runs in the background.
Then the same for every extra-source run dir under --source_runs (wave 2): own build_id, own hand-off folder, own
audit pools and store mirror (audit/<slug>/). Their release gates only report, and their abstracts are marked drafts
(config release_draft) until the gate c28_sources starts with "pass".
A lock file keeps two iterations from overlapping. Prints one JSON summary line; appends it to work/logs/merge_loop.log.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = "/cursor/stores/bc-36270f16-3cac-4d49-8f64-7dfb9bff8601/internal/dataset-expansion"
DASHBOARD = "/home/ubuntu/cml-dashboard"
MIN_FOR_STOP = 200
DRAFT_GATE = "c28_sources"
FULL_EVERY = 3600


def run(cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _read(path: str) -> str | None:
    """Read a store file; the store occasionally answers EIO/EAGAIN (e.g. while an orchestrator writes the file).
    None after a few retries: the caller skips the bundle this round and retries it next round."""
    for wait in (0.5, 2, None):
        try:
            with open(path, encoding="utf-8") as f:
                return f.read()
        except OSError:
            if wait is None:
                return None
            time.sleep(wait)
    return None


def bundle_stamps(src: str, names: list[str]) -> dict[str, list]:
    """(size, mtime) of each bundle's two files. The store answers each stat slowly, so they are taken in parallel."""
    def stamp(name):
        out = []
        for fn in ("abstracts.jsonl", "writer_report.json"):
            try:
                st = os.stat(os.path.join(src, name, fn))
                out.append([st.st_size, st.st_mtime])
            except OSError:
                out.append(None)
        return out

    with ThreadPoolExecutor(32) as ex:
        return dict(zip(names, ex.map(stamp, names)))


def unpack(run_dir: str, build_id: str, skip: set[str] = frozenset()) -> dict:
    """Copy hand-off bundles into the run dir; a file is only rewritten when its content differs. Items in `skip` (C35)
    are never copied: neither their abstracts nor their writer notes."""
    src = os.path.join(STORE, "abstracts", build_id)
    stats = {"bundles": 0, "abstracts_written": 0, "reports_written": 0, "foreign": 0}
    if not os.path.isdir(src):
        return stats
    batches = {json.loads(l)["batch_id"]: json.loads(l) for l in open(os.path.join(run_dir, "batches.jsonl"), encoding="utf-8")}
    seen_path = os.path.join(run_dir, "unpacked.json")  # (size, mtime) of each bundle file already copied
    seen = json.load(open(seen_path)) if os.path.exists(seen_path) else {}
    names = sorted(os.listdir(src))
    stats["foreign"] = sum(1 for n in names if not n.startswith("_") and n not in batches)
    names = [n for n in names if not n.startswith("_") and n in batches]
    stamps = bundle_stamps(src, names)
    for batch_id in names:
        bdir = os.path.join(src, batch_id)
        stats["bundles"] += 1
        stamp = stamps[batch_id]
        if seen.get(batch_id) == stamp:
            continue
        allowed = set(batches[batch_id]["item_ids"])
        path = os.path.join(bdir, "abstracts.jsonl")
        rep = os.path.join(bdir, "writer_report.json")
        text = _read(path) if stamp[0] else ""
        rep_text = _read(rep) if stamp[1] else None
        try:
            rows = [json.loads(l) for l in (text or "").splitlines() if l.strip()]
            rep_obj = json.loads(rep_text) if rep_text else None
        except ValueError:
            rows = None
        if text is None or rows is None or (stamp[1] and rep_text is None):
            # unreadable or half-written right now: leave the stamp unrecorded so the next round retries this bundle
            stats.setdefault("read_errors", []).append(batch_id)
            continue
        seen[batch_id] = stamp
        for ab in rows:
            if ab.get("item_id") not in allowed or ab.get("item_id") in skip:
                continue
            data = json.dumps(ab, ensure_ascii=False)
            out = os.path.join(run_dir, "abstracts", f"{ab['item_id']}.json")
            if not os.path.exists(out) or open(out, encoding="utf-8").read() != data:
                with open(out, "w", encoding="utf-8") as f:
                    f.write(data)
                stats["abstracts_written"] += 1
        if rep_obj is not None:
            data = rep_text
            if skip & allowed:
                data = json.dumps({**rep_obj, "source_issues": [x for x in rep_obj.get("source_issues", []) if x.get("item_id") not in skip]},
                                  ensure_ascii=False)
            out = os.path.join(run_dir, "batches", batch_id, "writer_report.json")
            if not os.path.exists(out) or open(out, encoding="utf-8").read() != data:
                if os.path.exists(out):  # a backfilled report for a batch already unpacked: its G8 pool is recomputed by the sync
                    stats.setdefault("reports_updated", []).append(batch_id)
                with open(out, "w", encoding="utf-8") as f:
                    f.write(data)
                stats["reports_written"] += 1
    with open(seen_path, "w") as f:
        json.dump(seen, f)
    return stats


def dashboard() -> str:
    """Re-export the static dashboard in the background (it takes minutes); one export at a time, the previous one's
    outcome is reported."""
    log = os.path.join(HERE, "work", "logs", "dashboard_export.log")
    last = open(log, encoding="utf-8").read()[-2000:] if os.path.exists(log) else ""
    previous = "deployed" if "deployed" in last else ("failed" if last.strip() else "none")
    lock = open(os.path.join(HERE, "work", "logs", "dashboard_export.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_UN)
    except BlockingIOError:
        return f"busy (previous: {previous})"
    run(["git", "fetch", "-q", "origin", "cursor/cml-dashboard-a0ca"], cwd=DASHBOARD)
    run(["git", "checkout", "-q", "--detach", "origin/cursor/cml-dashboard-a0ca"], cwd=DASHBOARD)
    with open(log, "w", encoding="utf-8") as f:
        subprocess.Popen(["flock", "-n", lock.name, sys.executable, "dashboard/export_static.py", "--deploy"], cwd=DASHBOARD,
                         stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    return f"started (previous: {previous})"


def ingest_c35(run_dirs: list[str]) -> dict | None:
    """Merge the evaluator's final content-safety scan (content_safety/c35_candidates.json, "final": true) into the C35 list
    content_exclusions.json: windows as item_id + content_sha1, whole-film recommendations as imdb_id. Ids only: no names,
    reasons or descriptions are copied. Entries are only ever added; a scan file is ingested once per content sha1."""
    cand_path = os.path.join(STORE, "content_safety", "c35_candidates.json")
    list_path = os.path.join(STORE, "content_exclusions.json")
    if not os.path.exists(cand_path):
        return None
    raw = open(cand_path, "rb").read()
    try:
        cand = json.loads(raw)
    except ValueError:
        return {"error": "c35_candidates.json is not valid JSON"}
    if cand.get("final") is not True:
        return None
    sha = hashlib.sha1(raw).hexdigest()
    spec = json.load(open(list_path, encoding="utf-8"))
    if any(x.get("sha1") == sha for x in spec.get("ingested", [])):
        return None
    known = {}
    for rd in run_dirs:
        build = json.load(open(os.path.join(rd, "run.json")))["build_id"]
        for line in open(os.path.join(rd, "items.jsonl"), encoding="utf-8"):
            it = json.loads(line)
            known[it["item_id"]] = (it["content_sha1"], it["imdb_id"], build)
    have_items = {x["item_id"] for x in spec["items"]}
    have_films = {x["imdb_id"] for x in spec["films"]}
    now_s = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new_items, unknown, sha_mismatch = [], [], []
    for w in cand.get("exclude_windows", []):
        iid = w["item_id"]
        if iid in have_items:
            continue
        sha1, film, build = known.get(iid, (None, w.get("imdb_id") or w.get("film_id"), None))
        if sha1 is None:
            unknown.append(iid)
            sha1 = w.get("content_sha1")
        elif w.get("content_sha1") and w["content_sha1"] != sha1:
            sha_mismatch.append(iid)
        new_items.append({"item_id": iid, "content_sha1": sha1, "imdb_id": film, "build_id": build, "source": w.get("source"),
                          "added_by": "evaluator scan", "added_at": now_s})
        have_items.add(iid)
    whole = {f.get("imdb_id") or f.get("film_id") for f in cand.get("film_recommendations", []) if f.get("recommend_whole_film_exclusion")}
    whole |= set(cand.get("whole_film_exclusion_imdb_ids") or [])
    source_of = {(f.get("imdb_id") or f.get("film_id")): f.get("source") for f in cand.get("film_recommendations", [])}
    new_films = []
    for fid in sorted(whole - have_films - {None}):
        new_films.append({"imdb_id": fid, "category": "sexual_content_involving_minor", "decided_by": "evaluator scan",
                          "decided_at": now_s[:10], "source": source_of.get(fid)})
        have_films.add(fid)
    spec["items"] += sorted(new_items, key=lambda x: x["item_id"])
    spec["films"] += sorted(new_films, key=lambda x: x["imdb_id"])
    spec["version"] = spec.get("version", 1) + 1
    spec["updated_at"] = now_s
    film_windows = sum(1 for i, (_, f, _) in known.items() if f in {x["imdb_id"] for x in new_films} and i not in {x["item_id"] for x in new_items})
    record = {"source_file": "content_safety/c35_candidates.json", "sha1": sha, "generated_at": cand.get("generated_at"),
              "ingested_at": now_s, "windows_added": len(new_items), "films_added": len(new_films),
              "more_windows_via_films": film_windows, "unknown_item_ids": len(unknown), "content_sha1_mismatch": len(sha_mismatch)}
    spec.setdefault("ingested", []).append(record)
    tmp = list_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)
    os.replace(tmp, list_path)
    return {**record, "listed_items": len(spec["items"]), "listed_films": len(spec["films"])}


def ingest_writer_flags(run_dirs: list[str]) -> dict | None:
    """Standing C35 rule (coordinator, 2026-10-09 16:44 UTC): every window an orchestrator lists in
    content_safety/writer_flags.jsonl ({item_id, batch, timestamp}) that is not on the C35 list yet is added as a window
    exclusion (item_id + content_sha1, category writer_flag). Film-level extension stays with the evaluator."""
    flags_path = os.path.join(STORE, "content_safety", "writer_flags.jsonl")
    list_path = os.path.join(STORE, "content_exclusions.json")
    if not os.path.exists(flags_path):
        return None
    flagged = []
    for line in open(flags_path, encoding="utf-8"):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("item_id"):
            flagged.append(row)
    spec = json.load(open(list_path, encoding="utf-8"))
    have = {x["item_id"] for x in spec["items"]}
    todo = [r for r in flagged if r["item_id"] not in have]
    if not todo:
        return None
    known = {}
    for rd in run_dirs:
        build = json.load(open(os.path.join(rd, "run.json")))["build_id"]
        for line in open(os.path.join(rd, "items.jsonl"), encoding="utf-8"):
            it = json.loads(line)
            known[it["item_id"]] = (it["content_sha1"], it["imdb_id"], build, it["source_dataset"])
    now_s = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    added, unknown = 0, 0
    for r in todo:
        if r["item_id"] in have:
            continue
        sha1, film, build, src = known.get(r["item_id"], (None, r["item_id"].split("-s")[0], None, None))
        unknown += sha1 is None
        spec["items"].append({"item_id": r["item_id"], "content_sha1": sha1, "imdb_id": film, "build_id": build,
                              "batch_id": r.get("batch") or r.get("batch_id"), "source": src, "category": "writer_flag",
                              "added_by": "writer flag", "flagged_at": r.get("timestamp"), "added_at": now_s})
        have.add(r["item_id"])
        added += 1
    spec["version"] = spec.get("version", 1) + 1
    spec["updated_at"] = now_s
    tmp = list_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)
    os.replace(tmp, list_path)
    return {"flags": len(flagged), "windows_added": added, "unknown_item_ids": unknown, "listed_items": len(spec["items"])}


def _rewrite(path: str, data: str) -> None:
    """Replace a file's content. The store mount can refuse a rename while it syncs a file (EAGAIN): retry, then fall
    back to writing in place."""
    tmp = path + ".c35tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
    for wait in (0.5, 1, 2, 4):
        try:
            os.replace(tmp, path)
            return
        except BlockingIOError:
            time.sleep(wait)
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)
    try:
        os.remove(tmp)
    except OSError:
        pass


def purge_c35(run_dir: str) -> tuple[set[str], dict]:
    """Remove every C35-listed item's abstract and writer note from the store hand-off bundles of its batch, from the run
    dir (abstract files, writer reports, merge cache entries) and report the counts. Idempotent; only bundles of batches
    that hold listed items are read. G1-G7 skip listed items, so the rest of each batch merges unchanged."""
    sys.path.insert(0, HERE)
    from validate_batch import content_excluded, load_content_exclusions

    c35 = load_content_exclusions(os.path.join(STORE, "content_exclusions.json"))
    items = {}
    for line in open(os.path.join(run_dir, "items.jsonl"), encoding="utf-8"):
        it = json.loads(line)
        items[it["item_id"]] = it
    listed = content_excluded(items, items, c35)
    stats = {"listed_here": len(listed), "bundle_abstracts_removed": 0, "bundle_notes_removed": 0, "bundles_rewritten": 0,
             "run_abstracts_removed": 0, "run_notes_removed": 0, "cache_entries_dropped": 0}
    if not listed:
        return listed, stats
    build = json.load(open(os.path.join(run_dir, "run.json")))["build_id"]
    cache_path = os.path.join(run_dir, "merge_cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    state_path = os.path.join(run_dir, "purge_state.json")  # per batch: bundle stamps + listed ids after its last purge
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    todo = []
    for line in open(os.path.join(run_dir, "batches.jsonl"), encoding="utf-8"):
        b = json.loads(line)
        ids = sorted(listed & set(b["item_ids"]))
        if ids:
            todo.append((b, ids))
    src = os.path.join(STORE, "abstracts", build)
    stamps = bundle_stamps(src, [b["batch_id"] for b, _ in todo])
    stats["bundles_unchanged"] = 0
    for b, ids_sorted in todo:
        if state.get(b["batch_id"]) == {"stamp": stamps[b["batch_id"]], "ids": ids_sorted}:
            stats["bundles_unchanged"] += 1
            continue
        ids = set(ids_sorted)
        bdir = os.path.join(src, b["batch_id"])
        changed = False
        try:
            for name in ("abstracts.jsonl.c35tmp", "writer_report.json.c35tmp"):
                if os.path.exists(os.path.join(bdir, name)):
                    os.remove(os.path.join(bdir, name))
            p = os.path.join(bdir, "abstracts.jsonl")
            if os.path.exists(p):
                text = _read(p)
                if text is None:
                    raise OSError(f"unreadable {p}")
                lines = [l for l in text.splitlines() if l.strip()]
                keep = [l for l in lines if json.loads(l).get("item_id") not in ids]
                if len(keep) < len(lines):
                    _rewrite(p, "".join(l + "\n" for l in keep))
                    stats["bundle_abstracts_removed"] += len(lines) - len(keep)
                    changed = True
            for p, key in ((os.path.join(bdir, "writer_report.json"), "bundle_notes_removed"),
                           (os.path.join(run_dir, "batches", b["batch_id"], "writer_report.json"), "run_notes_removed")):
                if os.path.exists(p):
                    text = _read(p)
                    if text is None:
                        raise OSError(f"unreadable {p}")
                    rep = json.loads(text)
                    issues = rep.get("source_issues", [])
                    kept = [x for x in issues if x.get("item_id") not in ids]
                    if len(kept) < len(issues):
                        _rewrite(p, json.dumps({**rep, "source_issues": kept}, ensure_ascii=False))
                        stats[key] += len(issues) - len(kept)
                        changed |= key.startswith("bundle")
        except (OSError, ValueError):
            stats.setdefault("read_errors", []).append(b["batch_id"])  # retried next round: its state stays unrecorded
            continue
        stats["bundles_rewritten"] += changed
        for iid in ids:
            p = os.path.join(run_dir, "abstracts", f"{iid}.json")
            if os.path.exists(p):
                os.remove(p)
                stats["run_abstracts_removed"] += 1
        if cache.pop(b["batch_id"], None) is not None:
            stats["cache_entries_dropped"] += 1
        state[b["batch_id"]] = {"stamp": bundle_stamps(src, [b["batch_id"]])[b["batch_id"]], "ids": ids_sorted}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f)
    return listed, stats


def released_shas(run_dir: str) -> dict[str, tuple[str, str]]:
    """item_id -> (batch_id, summary_sha1) of the source's released items (the local copy of data/ from the last sync)."""
    p = os.path.join(run_dir, "merged_release", "data", "merged.jsonl")
    out = {}
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["item_id"]] = (r["batch_id"], hashlib.sha1(r["summary"].encode()).hexdigest())
    return out


def released_ids(run_dir: str) -> set[str]:
    return set(released_shas(run_dir))


def reused_pilot_check(run_dir: str, slug: str) -> dict | None:
    """Wave 2 reuses audited pilot abstracts (evaluator condition 2): each must be byte-identical to its record, with the
    author normalized to the allowlist value."""
    path = os.path.join(STORE, "sources", "wave2_v12", "reused_pilot_abstracts.json")
    if not os.path.exists(path):
        return None
    rows = [r for r in json.load(open(path, encoding="utf-8")) if r.get("source") == slug]
    present = ok = 0
    bad = []
    for r in rows:
        p = os.path.join(run_dir, "abstracts", f"{r['item_id']}.json")
        if not os.path.exists(p):
            continue
        present += 1
        ab = json.load(open(p, encoding="utf-8"))
        if hashlib.sha1(ab["abstract"].encode()).hexdigest() == r["abstract_sha1"] and ab.get("author") == r["author_now"]:
            ok += 1
        else:
            bad.append(r["item_id"])
    return {"listed": len(rows), "merged_in": present, "identical": ok, "mismatch": bad}


def source_round(run_dir: str, draft: str = "", full: bool = False) -> tuple[dict, dict]:
    """Unpack, sync and gate one source. Returns its summary and its build_status per-source entry. Incremental rounds
    reuse unchanged passed batches; a full round re-validates every batch and runs the release-level contract gates."""
    run_info = json.load(open(os.path.join(run_dir, "run.json")))
    build_id = run_info["build_id"]
    listed, purge = purge_c35(run_dir)
    before = released_shas(run_dir)
    in_data_before = len(set(before) & listed)
    out = {"build_id": build_id, "unpack": unpack(run_dir, build_id, listed)}
    if listed:
        out["c35"] = {**purge, "removed_from_data": in_data_before}
    cmd = [sys.executable, os.path.join(HERE, "hf_sync.py"), "sync", "--run_dir", run_dir] + (["--draft", draft] if draft else []) \
        + (["--full"] if full else [])
    r = run(cmd)
    text = r.stdout + r.stderr
    m = re.search(r"sync \S+: (\d+) file operations", text)
    out["sync_ops"] = int(m.group(1)) if m else None
    v = re.search(r"validation \S+: reused (\d+), validated (\d+)", text)
    if v:
        out["validation"] = {"reused": int(v.group(1)), "validated": int(v.group(2)), "full": full}
    if r.returncode != 0 or m is None:
        out["sync_error"] = text[-500:]
    st = json.load(open(os.path.join(STORE, "build_status.json")))
    src = next((s for s in st["per_source"].values() if s.get("build_id") == build_id), None)
    if src is None:
        out["sync_error"] = out.get("sync_error") or f"no status for {build_id}"
        return out, {}
    out.update({k: src[k] for k in ("items_with_abstract", "items_checks_passed", "items_merged", "batches_merged",
                                    "batches_rejected", "batches_incomplete", "batches_revoked")})
    out["items_excluded_from_merged"] = src.get("items_excluded_from_merged")
    out["items_releasable"] = src.get("items_releasable")
    after = released_shas(run_dir)
    replaced = sorted(i for i in set(before) & set(after) if before[i][1] != after[i][1])
    if replaced:  # rewritten abstracts of already-released items (same content); audit verdicts bind to the new sha1
        out["abstracts_replaced"] = {"items": len(replaced), "batches": sorted({after[i][0] for i in replaced}), "item_ids": replaced}
    if listed:
        out["c35"]["still_in_data"] = len(set(after) & listed)
    out["audit"] = {k: src["audit"].get(k) for k in ("targeted_items", "targeted_selected", "sample_items", "audited",
                                                     "major_or_outside", "stop", "reason", "targeted_alarm",
                                                     "paused_orchestrators", "revoked_batch_ids")}
    if src.get("abstracts_draft"):
        out["abstracts_draft"] = src["abstracts_draft"]
    if src["items_merged"] and full:
        run([sys.executable, os.path.join(HERE, "contract_checks.py"), "--release", os.path.join(run_dir, "merged_release"),
             "--release_gates", "--out", os.path.join(run_dir, "release_gates.json")])
        rep = json.load(open(os.path.join(run_dir, "release_gates.json")))["summary"]
        out["release_gates"] = {"pass": rep["automated_pass"], "fail": rep["failed"]}
    return out, src


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default="data_construction/work/full_v31", help="MovieSum run dir (its gates can stop the fan-out)")
    ap.add_argument("--source_runs", default=os.path.join(HERE, "work", "wave2_v12"),
                    help="dir of extra-source run dirs (<slug>/run.json), merged after MovieSum; their gates only report")
    ap.add_argument("--full", action="store_true", help="force a full round: re-validate every batch and run the release gates")
    args = ap.parse_args()
    run_dir = os.path.abspath(args.run_dir)
    lock = open(os.path.join(run_dir, "merge_loop.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({"skipped": "another iteration is running"}))
        return
    t0 = time.time()
    summary = {"at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    srcs = sorted(d for d in os.listdir(args.source_runs) if os.path.exists(os.path.join(args.source_runs, d, "run.json"))) \
        if os.path.isdir(args.source_runs) else []
    all_runs = [run_dir] + [os.path.join(args.source_runs, s) for s in srcs]
    c35 = ingest_c35(all_runs)
    if c35:
        summary["c35_ingest"] = c35
    flags = ingest_writer_flags(all_runs)
    if flags:
        summary["c35_writer_flags"] = flags
    # Incremental by default. A full round (re-validate everything + release-level contract gates) runs hourly, when
    # forced, and once more whenever every main batch is merged (final release check).
    state_path = os.path.join(run_dir, "merge_loop_state.json")
    lstate = json.load(open(state_path)) if os.path.exists(state_path) else {}
    prev = json.load(open(os.path.join(STORE, "build_status.json")))["per_source"].get("MovieSum", {})
    main_batches = json.load(open(os.path.join(run_dir, "run.json"))).get("main_batches", 0)
    all_merged = main_batches and prev.get("batches_merged", 0) >= main_batches
    full = args.full or time.time() - lstate.get("last_full", 0) >= FULL_EVERY or \
        (all_merged and lstate.get("final_full_at") != prev.get("batches_merged"))
    summary["mode"] = "full" if full else "incremental"
    ms, src = source_round(run_dir, full=full)
    summary.update(ms)
    st = json.load(open(os.path.join(STORE, "build_status.json")))
    summary["gate"] = st["gates"].get("mass_abstract_writing")
    gates = (summary.get("release_gates") or {}).get("fail")
    if gates and src.get("items_merged", 0) >= MIN_FOR_STOP and str(summary["gate"]).startswith("go"):
        reason = f"stop: release gates {','.join(gates)} failing at {src['items_merged']} merged items (merger)"
        run([sys.executable, os.path.join(HERE, "hf_sync.py"), "status", "--gate", f"mass_abstract_writing={reason}"])
        summary["gate"] = reason

    # Extra sources: abstracts stay drafts until the evaluator records C28' for them (gate c28_sources = "pass: ...").
    c28 = str(st["gates"].get(DRAFT_GATE, ""))
    draft = "" if c28.startswith("pass") else "accept_as_draft: C28' (blind pairwise vs GT) for extra sources pending"
    if srcs:
        summary["sources"] = {}
        for slug in srcs:
            sd = os.path.join(args.source_runs, slug)
            res, _ = source_round(sd, draft, full=full)
            res["reused_pilot"] = reused_pilot_check(sd, slug)
            summary["sources"][slug] = res
    if full:
        lstate["last_full"] = time.time()
        if all_merged:
            lstate["final_full_at"] = prev.get("batches_merged")
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(lstate, f)
    if summary.get("sync_ops") or any(s.get("sync_ops") for s in summary.get("sources", {}).values()):
        summary["dashboard"] = dashboard()
    summary["seconds"] = round(time.time() - t0)
    line = json.dumps(summary, ensure_ascii=False)
    os.makedirs(os.path.join(HERE, "work", "logs"), exist_ok=True)
    with open(os.path.join(HERE, "work", "logs", "merge_loop.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
