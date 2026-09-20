#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export SWEEP_LOG_DIR="$SCRIPT_DIR/logs"
export SWEEP_CONFIG_PATH="$SCRIPT_DIR/sweep_motor36.yaml"

exec "$ROOT_DIR/scripts/sweeps/sweep_motor14/run_sweep_motor14.sh" "$@"
