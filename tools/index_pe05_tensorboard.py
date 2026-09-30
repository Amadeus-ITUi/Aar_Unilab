#!/usr/bin/env python3
"""Index PE05 TensorBoard runs like PE03 without moving models or rewriting events."""

import argparse
from pathlib import Path

from unilab.algos.torch.pe05.run_logging import index_existing_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_root", type=Path, nargs="+", help="PE05 training log roots")
    args = parser.parse_args()
    for root in args.log_root:
        root = root.expanduser()
        for link in index_existing_runs(root):
            print(link.name)
        print(f"TensorBoard --logdir {root / 'tensorboard_runs'}")


if __name__ == "__main__":
    main()
