"""Stage 6: upload an assembled release folder to a PRIVATE Hugging Face dataset repo.

Requires HF_TOKEN (write scope) in the environment. Refuses to upload to an existing public repo.

  HF_TOKEN=... python data_construction/upload_hf.py --folder data_construction/work/release --repo_id <user>/CML-Dataset-Expanded
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--message", default="Upload CML-Dataset expansion")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set; add a write token (Cursor Dashboard -> Cloud Agents -> Secrets).")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(args.repo_id, repo_type="dataset")
    if not info.private:
        sys.exit(f"{args.repo_id} exists and is PUBLIC; refusing to upload copyrighted screenplay text.")
    api.upload_folder(folder_path=args.folder, repo_id=args.repo_id, repo_type="dataset", commit_message=args.message)
    print(f"uploaded {args.folder} -> https://huggingface.co/datasets/{args.repo_id} (private)")


if __name__ == "__main__":
    main()
