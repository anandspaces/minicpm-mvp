#!/usr/bin/env python
# encoding: utf-8
"""
One-time download of MiniCPM-o 4.5 to a local directory.
Model: https://huggingface.co/openbmb/MiniCPM-o-4_5

After this, run the pro app with:
  uv run python pro/main.py --model-path <LOCAL_DIR> --device cuda --dtype bf16 --port 8084

Example:
  python scripts/download_minicpm_o_4_5.py --local-dir ./models/MiniCPM-o-4_5
  uv run python pro/main.py --model-path ./models/MiniCPM-o-4_5 --device cuda --dtype bf16 --port 8084
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="Download MiniCPM-o 4.5 to local disk (https://huggingface.co/openbmb/MiniCPM-o-4_5)"
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default="./models/MiniCPM-o-4_5",
        help="Directory to save the model (default: ./models/MiniCPM-o-4_5)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="openbmb/MiniCPM-o-4_5",
        help="HuggingFace repo (default: openbmb/MiniCPM-o-4_5)",
    )
    args = parser.parse_args()
    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading {args.repo_id} to {local_dir} (this may take a while, ~18GB+)..."
    )
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    print("Done. Run the pro app with:")
    print(f"  uv run python pro/main.py --model-path {local_dir} --device cuda --dtype bf16 --port 8084")


if __name__ == "__main__":
    main()
