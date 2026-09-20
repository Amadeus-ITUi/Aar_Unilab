#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/sweep_single_motor.py" --motor-index 0 --motor-id 1 --motor-name left_thigh_joint --gains-motor-ids 1 "$@"
