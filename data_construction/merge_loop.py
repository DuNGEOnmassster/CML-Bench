"""One iteration of the single merger's loop (fanout-kickoff.md section 6). Run it every ~30 minutes.

  python3 data_construction/merge_loop.py --run_dir data_construction/work/full_v31

1. unpack the orchestrators' hand-offs (store abstracts/<build_id>/<batch_id>/{abstracts.jsonl, writer_report.json})
   into the run dir, leaving unchanged files untouched;
2. hf_sync.py sync (merge gates G1-G8, audit sample/targeted pool/verdicts, C31 gates, status, store audit mirror);
3. when the sync changed the hub, re-export the static dashboard from the dashboard branch's code;
4. release-level contract gates on the merged items (contract_checks.py --release_gates); once >= MIN_FOR_STOP items
   are merged, a failing gate sets mass_abstract_writing to "stop: ...".
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
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = "/cursor/stores/bc-36270f16-3cac-4d49-8f64-7dfb9bff8601/internal/dataset-expansion"
DASHBOARD = "/home/ubuntu/cml-dashboard"
MIN_FOR_STOP = 200
DRAFT_GATE = "c28_sources"


def run(cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def unpack(run_dir: str, build_id: str) -> dict:
    """Copy hand-off bundles into the run dir; a file is only rewritten when its content differs."""
    src = os.path.join(STORE, "abstracts", build_id)
    stats = {"bundles": 0, "abstracts_written": 0, "reports_written": 0, "foreign": 0}
    if not os.path.isdir(src):
        return stats
    batches = {json.loads(l)["batch_id"]: json.loads(l) for l in open(os.path.join(run_dir, "batches.jsonl"), encoding="utf-8")}
    seen_path = os.path.join(run_dir, "unpacked.json")  # (size, mtime) of each bundle file already copied
    seen = json.load(open(seen_path)) if os.path.exists(seen_path) else {}
    for batch_id in sorted(os.listdir(src)):
        if batch_id.startswith("_") or batch_id not in batches:
            stats["foreign"] += not batch_id.startswith("_")
            continue
        bdir = os.path.join(src, batch_id)
        stats["bundles"] += 1
        stamp = [[os.path.getsize(p), os.path.getmtime(p)] if os.path.exists(p) else None
                 for p in (os.path.join(bdir, "abstracts.jsonl"), os.path.join(bdir, "writer_report.json"))]
        if seen.get(batch_id) == stamp:
            continue
        seen[batch_id] = stamp
        allowed = set(batches[batch_id]["item_ids"])
        path = os.path.join(bdir, "abstracts.jsonl")
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                ab = json.loads(line)
                if ab.get("item_id") not in allowed:
                    continue
                data = json.dumps(ab, ensure_ascii=False)
                out = os.path.join(run_dir, "abstracts", f"{ab['item_id']}.json")
                if not os.path.exists(out) or open(out, encoding="utf-8").read() != data:
                    with open(out, "w", encoding="utf-8") as f:
                        f.write(data)
                    stats["abstracts_written"] += 1
        rep = os.path.join(bdir, "writer_report.json")
        if os.path.exists(rep):
            data = open(rep, encoding="utf-8").read()
            out = os.path.join(run_dir, "batches", batch_id, "writer_report.json")
            if not os.path.exists(out) or open(out, encoding="utf-8").read() != data:
                with open(out, "w", encoding="utf-8") as f:
                    f.write(data)
                stats["reports_written"] += 1
    with open(seen_path, "w") as f:
        json.dump(seen, f)
    return stats


def dashboard() -> str:
    run(["git", "fetch", "-q", "origin", "cursor/cml-dashboard-a0ca"], cwd=DASHBOARD)
    run(["git", "checkout", "-q", "--detach", "origin/cursor/cml-dashboard-a0ca"], cwd=DASHBOARD)
    r = run([sys.executable, "dashboard/export_static.py", "--deploy"], cwd=DASHBOARD)
    return "deployed" if "deployed" in r.stdout + r.stderr else f"failed: {(r.stderr or r.stdout)[-300:]}"


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


def source_round(run_dir: str, draft: str = "") -> tuple[dict, dict]:
    """Unpack, sync and gate one source. Returns its summary and its build_status per-source entry."""
    run_info = json.load(open(os.path.join(run_dir, "run.json")))
    build_id = run_info["build_id"]
    out = {"build_id": build_id, "unpack": unpack(run_dir, build_id)}
    cmd = [sys.executable, os.path.join(HERE, "hf_sync.py"), "sync", "--run_dir", run_dir] + (["--draft", draft] if draft else [])
    r = run(cmd)
    text = r.stdout + r.stderr
    m = re.search(r"sync \S+: (\d+) file operations", text)
    out["sync_ops"] = int(m.group(1)) if m else None
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
    out["audit"] = {k: src["audit"].get(k) for k in ("targeted_items", "targeted_selected", "sample_items", "audited",
                                                     "major_or_outside", "stop", "reason", "targeted_alarm",
                                                     "paused_orchestrators", "revoked_batch_ids")}
    if src.get("abstracts_draft"):
        out["abstracts_draft"] = src["abstracts_draft"]
    if src["items_merged"]:
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
    ms, src = source_round(run_dir)
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
    srcs = sorted(d for d in os.listdir(args.source_runs) if os.path.exists(os.path.join(args.source_runs, d, "run.json"))) \
        if os.path.isdir(args.source_runs) else []
    if srcs:
        summary["sources"] = {}
        for slug in srcs:
            sd = os.path.join(args.source_runs, slug)
            res, _ = source_round(sd, draft)
            res["reused_pilot"] = reused_pilot_check(sd, slug)
            summary["sources"][slug] = res
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
