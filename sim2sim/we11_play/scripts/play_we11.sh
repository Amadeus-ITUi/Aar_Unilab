#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
replay_bin="${repo_root}/build/bin/rl_sim_mujoco"
python_prefix="$(python -c 'import sys; print(sys.prefix)')"
export LD_LIBRARY_PATH="${python_prefix}/lib:${LD_LIBRARY_PATH:-}"
difficulty=""
stage=""
autostart="${RL_SAR_PLAY_AUTOSTART:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --difficulty)
      difficulty="${2:?--difficulty requires a value in [0,1]}"
      shift 2
      ;;
    --stage)
      stage="${2:?--stage requires 1..5 or a stage name}"
      shift 2
      ;;
    --autostart)
      autostart=1
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--difficulty 0..1] [--stage 1..5|name] [--autostart]"
      echo "No flapping-frequency, wing-angle CSV, or wrench replay is applied."
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "${stage}" in
  "") ;;
  1|stage1|home_to_getup) stage=home_to_getup ;;
  2|stage2|exact_getup) stage=exact_getup ;;
  3|stage3|getup_with_home) stage=getup_with_home ;;
  4|stage4|balance) stage=balance ;;
  5|stage5|mixed) stage=mixed ;;
  *) echo "Invalid --stage: ${stage}" >&2; exit 2 ;;
esac

if [[ -n "${difficulty}" ]] && [[ ! "${difficulty}" =~ ^(0([.][0-9]+)?|[.][0-9]+|1([.]0*)?)$ ]]; then
  echo "Invalid --difficulty '${difficulty}'; expected [0,1]." >&2
  exit 2
fi

required=(
  "${replay_bin}"
  "${repo_root}/policy/dr002/we11/config.yaml"
  "${repo_root}/policy/dr002/we11/policy.onnx"
  "${repo_root}/src/rl_sar_zoo/dr002_description/mjcf/we11/scene_flat_we11.xml"
)
for path in "${required[@]}"; do
  [[ -e "${path}" ]] || { echo "Missing WE11 playback dependency: ${path}" >&2; exit 1; }
done

# The active WE11 Getup playback deliberately has no measured wing/wrench
# injection. Clear inherited variables so a caller cannot enable it by accident.
while IFS='=' read -r name _; do
  case "${name}" in
    RL_SAR_FORCE_*|RL_SAR_WING_*) unset "${name}" ;;
  esac
done < <(env)

unset RL_SAR_DR002_GETUP_DIFFICULTY RL_SAR_DR002_GETUP_STAGE
unset RL_SAR_DR002_GETUP_STAGE_DIFFICULTY RL_SAR_DR002_BALANCE_DIFFICULTY
export RL_SAR_DR002_CONFIG=we11
export RL_SAR_DR002_START_POSE=getup
[[ "${autostart}" == 1 ]] && export RL_SAR_DR002_AUTOSTART=1 || unset RL_SAR_DR002_AUTOSTART

if [[ -n "${stage}" ]]; then
  export RL_SAR_DR002_GETUP_STAGE="${stage}"
  export RL_SAR_DR002_GETUP_STAGE_DIFFICULTY="${difficulty:-1.0}"
elif [[ -n "${difficulty}" ]]; then
  export RL_SAR_DR002_GETUP_DIFFICULTY="${difficulty}"
fi

echo "WE11 Getup Play: no wing/wrench disturbance"
echo "Controls: RB+DPadUp start, LB+X stop, RB+Y reset; LY=vx, RX=yaw, RY=height"
exec "${replay_bin}" dr002 we11/scene_flat_we11
