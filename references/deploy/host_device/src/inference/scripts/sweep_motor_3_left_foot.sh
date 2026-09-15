#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/sweep_single_motor.py" --motor-index 2 --motor-id 3 --motor-name left_foot_joint "$@"
