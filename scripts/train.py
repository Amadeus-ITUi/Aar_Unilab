#!/usr/bin/env python3
"""Generic training entrypoint; all six selectors are explicit."""

from unilab.catalog.cli import run

if __name__ == "__main__":
    raise SystemExit(run("train"))
