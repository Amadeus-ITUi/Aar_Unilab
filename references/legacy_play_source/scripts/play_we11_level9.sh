#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
terrain="${1:-rough}"
replay_hz="${2:-3}"
force_scale="${3:-1.0}"

case "${terrain}" in
  flat|uphill|downhill|rough)
    export RL_SAR_WE11_SCENE="scene_level9_noise006_${terrain}_we11"
    ;;
  *)
    echo "Usage: $0 [flat|uphill|downhill|rough] [0|1|2|3 Hz] [wrench_scale]" >&2
    exit 2
    ;;
esac

exec "${repo_root}/scripts/play_we11.sh" "${replay_hz}" upright "${force_scale}"
