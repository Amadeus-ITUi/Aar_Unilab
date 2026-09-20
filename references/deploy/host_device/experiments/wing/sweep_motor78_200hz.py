#!/usr/bin/env python3
"""Compatibility entry point for the ESD-Link P7/P8 sweep."""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "src" / "deploy_tools" / "scripts" / "control"
sys.path.insert(0, str(SCRIPT_DIR))

from esd_wing_sweep import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
