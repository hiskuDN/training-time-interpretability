"""
One-time script to cache the tokenized TinyStories dataset to a local directory.
On Modal, this is handled by `modal run modal_app.py::cache_dataset`.
Use this script for local caching (e.g. during development).

Usage:
    python scripts/cache_dataset.py --cache_dir /tmp/tinystories_cache
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data import get_tokenizer, get_tinystories_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    tokenizer = get_tokenizer("gpt2")
    get_tinystories_dataset(
        tokenizer,
        max_length=args.max_length,
        num_workers=4,
        cache_dir=args.cache_dir,
    )
    print(f"Dataset cached to {args.cache_dir}")


if __name__ == "__main__":
    main()
