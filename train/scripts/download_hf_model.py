#!/usr/bin/env python3
"""
Download a Hugging Face model repo into STORAGE_PATH (or a provided directory).
Includes timeout and retry logic to handle network issues.

Examples:
  export STORAGE_PATH="/path/to/storage"
  python scripts/download_hf_model.py --repo-id "Qwen/Qwen3-4B-Base"

  # Download a finetuned model you've uploaded:
  python scripts/download_hf_model.py --repo-id "yourname/qwen3-4b_solver_v3"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import time

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError


def default_dest(storage_path: str, repo_id: str) -> str:
    # Keep it stable and filesystem-safe.
    repo_dirname = repo_id.replace("/", "_")
    return str(Path(storage_path) / "models" / repo_dirname)


def download_with_retry(repo_id: str, dest_dir: str, token: str = None, max_retries: int = 3, timeout: int = 300):
    """
    Download model with retry logic and timeout handling.
    
    Args:
        repo_id: HuggingFace repository ID
        dest_dir: Destination directory
        token: HuggingFace token (optional)
        max_retries: Maximum number of retry attempts
        timeout: Timeout in seconds for each download attempt
    """
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Download attempt {attempt}/{max_retries} for {repo_id}...")
            print(f"Destination: {dest_dir}")
            
            # Set environment variables for timeout
            os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = str(timeout)
            
            local_dir = snapshot_download(
                repo_id=repo_id,
                local_dir=dest_dir,
                local_dir_use_symlinks=False,
                token=token,
                resume_download=True,  # Enable resume for interrupted downloads
            )
            print(f"✓ Successfully downloaded {repo_id} to {local_dir}")
            return local_dir
            
        except HfHubHTTPError as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                print(f"✗ Download failed (attempt {attempt}/{max_retries}): {e}")
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"✗ Download failed after {max_retries} attempts: {e}")
                raise
        except Exception as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(f"✗ Unexpected error (attempt {attempt}/{max_retries}): {e}")
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"✗ Download failed after {max_retries} attempts: {e}")
                raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. Qwen/Qwen3-4B-Base")
    parser.add_argument("--revision", default=None, help="Optional HF revision (branch/tag/commit)")
    parser.add_argument(
        "--dest",
        default=None,
        help="Destination directory. Default: $STORAGE_PATH/models/<repo_id_with_underscores>/",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
        help="HF token if required (or set HF_TOKEN env var).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each download attempt (default: 300)",
    )
    args = parser.parse_args()

    storage_path = os.getenv("STORAGE_PATH")
    if not storage_path and not args.dest:
        raise SystemExit("STORAGE_PATH is not set and --dest was not provided.")

    dest_dir = args.dest or default_dest(storage_path, args.repo_id)
    os.makedirs(dest_dir, exist_ok=True)

    print(f"Downloading {args.repo_id} -> {dest_dir}")
    try:
        local_dir = download_with_retry(
            repo_id=args.repo_id,
            dest_dir=dest_dir,
            token=args.token,
            max_retries=args.max_retries,
            timeout=args.timeout,
        )
        print(f"Done. Local path: {local_dir}")
    except Exception as e:
        print(f"Failed to download model: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
