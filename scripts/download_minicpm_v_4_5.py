#!/usr/bin/env python
# encoding: utf-8
"""
One-time download of MiniCPM-V 4.5 to a local directory.
After this, run the demo with:  python web_demos/web_demo_4_5.py --model-path <LOCAL_DIR>
Example:  python web_demos/web_demo_4_5.py --model-path ./models/MiniCPM-V-4_5 --device cuda --dtype fp16
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description='Download MiniCPM-V 4.5 to local disk')
    parser.add_argument(
        '--local-dir',
        type=str,
        default='./models/MiniCPM-V-4_5',
        help='Directory to save the model (default: ./models/MiniCPM-V-4_5)',
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        default='openbmb/MiniCPM-V-4_5',
        help='HuggingFace repo (default: openbmb/MiniCPM-V-4_5)',
    )
    args = parser.parse_args()
    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {args.repo_id} to {local_dir} (this may take a while, ~17GB)...')
    snapshot_download(repo_id=args.repo_id, local_dir=str(local_dir), local_dir_use_symlinks=False)
    print(f'Done. Run the demo with:')
    print(f'  python web_demos/web_demo_4_5.py --model-path {local_dir} --device cuda --dtype fp16')


if __name__ == '__main__':
    main()
