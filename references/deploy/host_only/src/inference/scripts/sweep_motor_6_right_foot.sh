#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/sweep_single_motor.py" --motor-index 5 --motor-id 6 --motor-name right_foot_joint "$@"
