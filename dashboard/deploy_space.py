"""Deploy the dashboard as a PRIVATE Hugging Face Space that reads the private dataset through a Space secret.

  HF_TOKEN=... HF_ACCOUNT=... python dashboard/deploy_space.py [--space_id USER/cml-dataset-dashboard]
      [--dataset_repo USER/CML-Dataset-Expanded] [--target_items 13904] [--refresh_seconds 300]

Uploads the dashboard code, a generated Space card and copies of the pipeline modules the checks import
(`pipeline/`). The Space gets HF_TOKEN as a secret (SPACE_HF_TOKEN if set, e.g. a read-only token) and
DATASET_REPO / TARGET_ITEMS / REFRESH_SECONDS as variables. Refuses to deploy to an existing public Space.

The Hub only hosts Gradio Spaces on cpu-basic for PRO accounts (otherwise create_repo returns 402); without
PRO, export_static.py publishes the same views as a static Space under the same id, and running this script
later switches that Space to the live Gradio app.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPELINE_DIR = HERE.parent / "data_construction"
APP_FILES = ("app.py", "checks.py", "dataio.py", "views.py", "itemview.py", "style.css", "requirements.txt")
PIPELINE_MODULES = ("cml_format.py", "check_abstracts.py", "make_abstract_batches.py", "build_segments.py", "contract_checks.py")
GRADIO_VERSION = "6.30.0"
CARD = f"""---
title: CML Dataset Dashboard
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: {GRADIO_VERSION}
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Live view of the private expanded CML-Bench dataset
---

Private dashboard for the expanded CML-Bench dataset (screenplay segments + AI-written abstracts).
It reads the private dataset repo named in the `DATASET_REPO` variable with the `HF_TOKEN` secret,
polls it every `REFRESH_SECONDS` and re-checks only new items. Code: `dashboard/` in
github.com/DuNGEOnmassster/CML-Bench. Do not make this Space public.
"""


def main() -> None:
    account = os.environ.get("HF_ACCOUNT", "")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space_id", default=f"{account}/cml-dataset-dashboard")
    ap.add_argument("--dataset_repo", default=f"{account}/CML-Dataset-Expanded")
    ap.add_argument("--target_items", type=int, default=13904)
    ap.add_argument("--refresh_seconds", type=int, default=300)
    ap.add_argument("--message", default="Deploy CML dataset dashboard")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.space_id, repo_type="space", space_sdk="gradio", private=True, exist_ok=True)
    if not api.space_info(args.space_id).private:
        sys.exit(f"{args.space_id} is PUBLIC; refusing to deploy a dashboard that shows screenplay text.")

    api.add_space_secret(args.space_id, "HF_TOKEN", os.environ.get("SPACE_HF_TOKEN") or token)
    for key, value in (("DATASET_REPO", args.dataset_repo), ("TARGET_ITEMS", str(args.target_items)),
                       ("REFRESH_SECONDS", str(args.refresh_seconds))):
        api.add_space_variable(args.space_id, key, value)

    with tempfile.TemporaryDirectory() as stage:
        for name in APP_FILES:
            shutil.copy(HERE / name, Path(stage) / name)
        (Path(stage) / "pipeline").mkdir()
        for name in PIPELINE_MODULES:
            shutil.copy(PIPELINE_DIR / name, Path(stage) / "pipeline" / name)
        (Path(stage) / "README.md").write_text(CARD)
        api.upload_folder(folder_path=stage, repo_id=args.space_id, repo_type="space", commit_message=args.message,
                          delete_patterns=["*.py", "pipeline/*.py", "*.css"])
    print(f"deployed https://huggingface.co/spaces/{args.space_id} (private), reading {args.dataset_repo}")


if __name__ == "__main__":
    main()
