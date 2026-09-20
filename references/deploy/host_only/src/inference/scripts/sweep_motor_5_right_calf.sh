#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/sweep_single_motor.py" --motor-index 4 --motor-id 5 --motor-name right_calf_joint "$@"
