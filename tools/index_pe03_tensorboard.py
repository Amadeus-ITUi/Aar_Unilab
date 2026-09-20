#!/usr/bin/env python3
"""Give existing PE03 TensorBoard runs descriptive names without moving checkpoints."""

import argparse
from pathlib import Path

from unilab.algos.torch.pe03.run_logging import index_existing_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_root", type=Path, nargs="+", help="PE03 training log roots")
    args = parser.parse_args()
    for root in args.log_root:
        links = index_existing_runs(root)
        for link in links:
            print(link.name)
        print(f"Indexed {len(links)} runs; TensorBoard --logdir {root / 'tensorboard_runs'}")


if __name__ == "__main__":
    main()
