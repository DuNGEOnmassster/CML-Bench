"""One iteration of the single merger's loop (fanout-kickoff.md section 6). Run it every ~30 minutes.

  python3 data_construction/merge_loop.py --run_dir data_construction/work/full_v31

1. unpack the orchestrators' hand-offs (store abstracts/<build_id>/<batch_id>/{abstracts.jsonl, writer_report.json})
   into the run dir, leaving unchanged files untouched;
2. hf_sync.py sync (merge gates G1-G8, audit sample/targeted pool/verdicts, C31 gates, status, store audit mirror);
3. when the sync changed the hub, re-export the static dashboard from the dashboard branch's code;
4. release-level contract gates on the merged items (contract_checks.py --release_gates); once >= MIN_FOR_STOP items
   are merged, a failing gate sets mass_abstract_writing to "stop: ...".
A lock file keeps two iterations from overlapping. Prints one JSON summary line; appends it to work/logs/merge_loop.log.
"""
from __future__ import annotations

import argparse
import fcntl
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default="data_construction/work/full_v31")
    args = ap.parse_args()
    run_dir = os.path.abspath(args.run_dir)
    lock = open(os.path.join(run_dir, "merge_loop.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({"skipped": "another iteration is running"}))
        return
    t0 = time.time()
    build_id = json.load(open(os.path.join(run_dir, "run.json")))["build_id"]
    summary = {"at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "build_id": build_id}
    summary["unpack"] = unpack(run_dir, build_id)

    r = run([sys.executable, os.path.join(HERE, "hf_sync.py"), "sync", "--run_dir", run_dir])
    out = r.stdout + r.stderr
    m = re.search(r"sync \S+: (\d+) file operations", out)
    summary["sync_ops"] = int(m.group(1)) if m else None
    if r.returncode != 0 or m is None:
        summary["sync_error"] = out[-500:]
    st = json.load(open(os.path.join(STORE, "build_status.json")))
    src = st["per_source"]["MovieSum"]
    summary.update({k: src[k] for k in ("items_with_abstract", "items_checks_passed", "items_merged", "batches_merged",
                                        "batches_rejected", "batches_incomplete", "batches_revoked")})
    summary["items_excluded_from_merged"] = src.get("items_excluded_from_merged")
    summary["audit"] = {k: src["audit"].get(k) for k in ("targeted_items", "targeted_selected", "sample_items", "audited",
                                                         "major_or_outside", "stop", "reason", "targeted_alarm",
                                                         "paused_orchestrators", "revoked_batches")}
    summary["gate"] = st["gates"].get("mass_abstract_writing")
    if summary["sync_ops"]:
        summary["dashboard"] = dashboard()

    rel = os.path.join(run_dir, "merged_release")
    if src["items_merged"]:
        g = run([sys.executable, os.path.join(HERE, "contract_checks.py"), "--release", rel, "--release_gates",
                 "--out", os.path.join(run_dir, "release_gates.json")])
        rep = json.load(open(os.path.join(run_dir, "release_gates.json")))["summary"]
        summary["release_gates"] = {"pass": rep["automated_pass"], "fail": rep["failed"]}
        if rep["failed"] and src["items_merged"] >= MIN_FOR_STOP and str(summary["gate"]).startswith("go"):
            reason = f"stop: release gates {','.join(rep['failed'])} failing at {src['items_merged']} merged items (merger)"
            run([sys.executable, os.path.join(HERE, "hf_sync.py"), "status", "--gate", f"mass_abstract_writing={reason}"])
            summary["gate"] = reason
    summary["seconds"] = round(time.time() - t0)
    line = json.dumps(summary, ensure_ascii=False)
    os.makedirs(os.path.join(HERE, "work", "logs"), exist_ok=True)
    with open(os.path.join(HERE, "work", "logs", "merge_loop.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
