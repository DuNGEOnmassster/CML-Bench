"""Export the dashboard as a static site and deploy it to a PRIVATE static Hugging Face Space.

Gradio and Docker Spaces on free cpu-basic need a PRO subscription (the Hub answers 402); static Spaces do not.
This renders the same views from the same DataStore into index.html + site.json + items-*.json. No screenplay
text is copied: the sample browser fetches one item's byte span from the private dataset at view time, with the
viewer's HF sign-in (OAuth, read-repos) or a pasted read token, and live build progress is read from
build_status.json in the browser. The snapshot itself is refreshed whenever this exporter runs (`--watch`).

  HF_TOKEN=... HF_ACCOUNT=... python dashboard/export_static.py --deploy [--watch 300]
  DATA_DIR=... GT_DIR=... python dashboard/export_static.py --out /tmp/site   # local preview
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import views
from checks import PROPOSED_COLUMNS
from dataio import ITEM_COLUMNS, SUBSETS, DataStore, has_subsets, release_label, selected_view

HERE = Path(__file__).resolve().parent
ITEM_FIELDS = ["item_id", "film", "source", "split", "num_scenes", "script_tokens", "dialogue_turns", "summary_words",
               "prompt_version", "normalization", "author", "imdb_rating", "genres", "scene_start", "scene_end",
               "relative_position", "eval_safe", "gt_related", "gt_rel_type", "gt_rel_movie", "imdb_url", "source_url",
               "path", "ref", "ref_len", "checked", "failed", "failed_v2", "target_lo", "target_hi", "grounding",
               "num_speakers", "dialogue_ratio", "artefacts_1k", "garble_rate", "bad_char_ratio", "gt_overlap",
               "backslashes", "contd", "quote_inner_1k", "orphan_lines", "speaker_splits", "hard", "ungrounded", "thirds",
               "summary", *ITEM_COLUMNS, *PROPOSED_COLUMNS]
CARD = """---
title: CML Dataset Dashboard
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
hf_oauth: true
hf_oauth_scopes:
  - read-repos
pinned: false
short_description: Snapshot dashboard of the private expanded CML-Bench dataset
---

Static snapshot of the private expanded CML-Bench dataset, exported by `dashboard/export_static.py`
(github.com/DuNGEOnmassster/CML-Bench). Screenplay text is fetched from the private dataset at view time
with the viewer's Hugging Face sign-in; it is not stored here. Do not make this Space public.
"""

log = logging.getLogger("cml-dashboard")


def clean(v):
    if v is None:
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) else round(float(v), 4)
    if isinstance(v, (tuple, list, np.ndarray)):
        return [clean(x) for x in v]
    return v


def slug(release: str) -> str:
    return release.replace(":", "-")


def release_payload(snap, release: str, gt, target: int | None) -> tuple[dict, list]:
    full = snap.releases[release]
    subsets = ["all"] + (["eval_safe", "gt_related"] if has_subsets(full) else [])
    out = {}
    for sub in subsets:
        rv = selected_view(snap, release, sub)
        label = release_label(release) + ("" if sub == "all" else f" · {SUBSETS[sub]}")
        figs = views.distribution_figs(rv.df, gt, label)
        figs.update(sources=views.sources_fig(rv.df), films=views.per_film_fig(rv.df), flags=views.flags_fig(rv.df),
                    box=views.box_fig(rv.df, gt, label))
        out[sub] = {
            "banner": views.banner_html(snap, release),
            "kpis": views.kpis_html(rv, gt),
            "progress": views.progress_html(full, target),
            "subsets": views.subsets_html(full),
            "versions": views.versions_html(rv),
            "contract": views.contract_html(rv, gt),
            "abs_summary": views.abstract_summary_html(rv, gt),
            "gt_table": views.gt_table_html(rv.df, gt, label),
            "figs": {k: json.loads(f.to_json()) for k, f in figs.items()},
        }
    df = full.df
    fields = [f for f in ITEM_FIELDS if f in df.columns]
    rows = [[clean(v) for v in r] for r in df[fields].itertuples(index=False, name=None)]
    meta = {"id": release, "label": release_label(release), "subsets": [[s, SUBSETS[s]] for s in subsets], "views": out,
            "items_file": f"items-{slug(release)}.json"}
    return meta, [fields, rows]


def export(store: DataStore, out_dir: Path) -> dict:
    snap = store.snapshot
    gt = store.gt.df if store.gt else None
    out_dir.mkdir(parents=True, exist_ok=True)
    site = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": store.source.kind,
        "dataset_repo": store.source.label if store.source.kind == "hf" else None,
        "rev": snap.rev,
        "header": views.header_html(snap, store.source.label if store.source.kind == "hf" else "local folder", 0),
        "status": snap.status,
        "message": snap.message,
        "releases": [],
    }
    for release in snap.releases:
        meta, items = release_payload(snap, release, gt, store.target_items)
        site["releases"].append(meta)
        (out_dir / meta["items_file"]).write_text(json.dumps(items, separators=(",", ":"), ensure_ascii=False))
    (out_dir / "site.json").write_text(json.dumps(site, separators=(",", ":"), ensure_ascii=False))
    shutil.copy(HERE / "static" / "index.html", out_dir / "index.html")
    shutil.copy(HERE / "style.css", out_dir / "style.css")
    return site


def deploy(out_dir: Path, space_id: str, token: str, message: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(space_id, repo_type="space", space_sdk="static", private=True, exist_ok=True)
    if not api.space_info(space_id).private:
        sys.exit(f"{space_id} is PUBLIC; refusing to deploy.")
    (out_dir / "README.md").write_text(CARD)
    api.upload_folder(folder_path=str(out_dir), repo_id=space_id, repo_type="space", commit_message=message,
                      delete_patterns=["*.json", "*.html", "*.css", "*.py", "pipeline/*"])


def main() -> None:
    account = os.environ.get("HF_ACCOUNT", "")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="write the site here (default: a temp dir)")
    ap.add_argument("--deploy", action="store_true", help="upload to the private static Space")
    ap.add_argument("--space_id", default=f"{account}/cml-dataset-dashboard")
    ap.add_argument("--watch", type=int, default=0, help="poll the dataset every N seconds and re-export on change")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for name in ("httpx", "httpx2"):
        logging.getLogger(name).setLevel(logging.WARNING)

    store = DataStore.from_env()
    while True:
        changed = store.refresh()
        if changed or not args.watch:
            out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="cml-site-"))
            site = export(store, out)
            log.info("exported %s (rev %s, %d releases)", out, (site["rev"] or "")[:7], len(site["releases"]))
            if args.deploy:
                deploy(out, args.space_id, os.environ["HF_TOKEN"], f"Dashboard snapshot of dataset rev {(site['rev'] or '')[:7]}")
                log.info("deployed https://huggingface.co/spaces/%s", args.space_id)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
