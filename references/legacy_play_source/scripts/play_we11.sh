#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
replay_bin="${repo_root}/build/bin/rl_sim_mujoco"
selector="we11"
scene_name="${RL_SAR_WE11_SCENE:-scene_flat_we11}"
scene_xml="${repo_root}/src/rl_sar_zoo/dr002_description/mjcf/${selector}/${scene_name}.xml"
positional_args=()
getup_difficulty="${RL_SAR_WE11_GETUP_DIFFICULTY:-}"
balance_difficulty="${RL_SAR_WE11_BALANCE_DIFFICULTY:-}"
getup_stage="${RL_SAR_WE11_GETUP_STAGE:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --difficulty)
      if [[ $# -lt 2 ]]; then
        echo "--difficulty requires a value in [0, 1]" >&2
        exit 2
      fi
      getup_difficulty="$2"
      shift 2
      ;;
    --difficulty=*)
      getup_difficulty="${1#*=}"
      shift
      ;;
    --balance-difficulty)
      if [[ $# -lt 2 ]]; then
        echo "--balance-difficulty requires a value in [0, 1]" >&2
        exit 2
      fi
      balance_difficulty="$2"
      shift 2
      ;;
    --balance-difficulty=*)
      balance_difficulty="${1#*=}"
      shift
      ;;
    --stage)
      if [[ $# -lt 2 ]]; then
        echo "--stage requires one of: 1..5, home_to_getup, exact_getup, getup_with_home, balance, mixed" >&2
        exit 2
      fi
      getup_stage="$2"
      shift 2
      ;;
    --stage=*)
      getup_stage="${1#*=}"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [0|1|2|3] [flat|getup] [wrench_scale] [--stage 1..5] [--difficulty 0..1]"
      echo "Stages: 1=home_to_getup, 2=exact_getup, 3=getup_with_home, 4=balance, 5=mixed"
      echo "Examples: $0 0 --stage 1 --difficulty 0.8"
      echo "          $0 0 --stage mixed"
      exit 0
      ;;
    --)
      shift
      positional_args+=("$@")
      break
      ;;
    --*)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    *)
      positional_args+=("$1")
      shift
      ;;
  esac
done

replay_hz="${positional_args[0]:-0}"
second_arg="${positional_args[1]:-}"
third_arg="${positional_args[2]:-}"
if [[ ${#positional_args[@]} -gt 3 ]]; then
  echo "Too many positional arguments." >&2
  echo "Usage: $0 [0|1|2|3] [flat|getup] [wrench_scale] [--difficulty 0..1 | --balance-difficulty 0..1]" >&2
  exit 2
fi
force_scale="1.0"
start_profile="${RL_SAR_WE11_START_PROFILE:-${RL_SAR_WE11_START_POSE:-getup}}"

# Preferred interface:
#   play_we11.sh <replay_hz> <flat|getup> [wrench_scale]
# Keep the old <replay_hz> <wrench_scale> [upright|getup] form working.
case "${second_arg}" in
  flat|getup|upright)
    start_profile="${second_arg}"
    if [[ -n "${third_arg}" ]]; then
      force_scale="${third_arg}"
    fi
    ;;
  "")
    ;;
  *)
    force_scale="${second_arg}"
    if [[ -n "${third_arg}" ]]; then
      start_profile="${third_arg}"
    fi
    ;;
esac

if [[ -n "${getup_difficulty}" ]]; then
  if [[ ! "${getup_difficulty}" =~ ^(0([.][0-9]+)?|[.][0-9]+|1([.]0*)?)$ ]]; then
    echo "Invalid --difficulty '${getup_difficulty}'; expected a number in [0, 1]." >&2
    exit 2
  fi
  start_profile="difficulty"
fi
if [[ -n "${balance_difficulty}" ]]; then
  if [[ ! "${balance_difficulty}" =~ ^(0([.][0-9]+)?|[.][0-9]+|1([.]0*)?)$ ]]; then
    echo "Invalid --balance-difficulty '${balance_difficulty}'; expected a number in [0, 1]." >&2
    exit 2
  fi
  if [[ -n "${getup_difficulty}" ]]; then
    echo "Choose only one of --difficulty and --balance-difficulty." >&2
    exit 2
  fi
  start_profile="balance"
fi
if [[ -n "${getup_stage}" ]]; then
  case "${getup_stage}" in
    1|stage1|home_to_getup) getup_stage="home_to_getup" ;;
    2|stage2|exact_getup) getup_stage="exact_getup" ;;
    3|stage3|getup_with_home) getup_stage="getup_with_home" ;;
    4|stage4|balance) getup_stage="balance" ;;
    5|stage5|mixed) getup_stage="mixed" ;;
    *)
      echo "Invalid --stage '${getup_stage}'; expected 1..5 or a stage name." >&2
      exit 2
      ;;
  esac
  if [[ -n "${balance_difficulty}" ]]; then
    echo "Choose --stage or --balance-difficulty, not both." >&2
    exit 2
  fi
  start_profile="stage"
fi

case "${start_profile}" in
  flat|upright)
    start_profile="flat"
    start_pose="upright"
    ;;
  getup)
    start_pose="getup"
    ;;
  difficulty)
    start_pose="getup"
    ;;
  balance)
    start_pose="upright"
    ;;
  stage)
    start_pose="getup"
    ;;
  *)
    echo "Invalid WE11 start profile: ${start_profile}" >&2
    echo "Usage: $0 [0|1|2|3] [flat|getup] [wrench_scale] [--difficulty 0..1 | --balance-difficulty 0..1]" >&2
    echo "Legacy: $0 [0|1|2|3] [wrench_scale] [upright|getup]" >&2
    exit 2
    ;;
esac
play_autostart="${RL_SAR_PLAY_AUTOSTART:-0}"
policy_path="${repo_root}/policy/dr002/${selector}/policy.onnx"

case "${replay_hz}" in
  0)
    force_csv=""
    wing_angle_csv=""
    ;;
  1|2|3)
    force_csv="${repo_root}/replay_data/we11/wrench/${replay_hz}hz.csv"
    wing_angle_csv="${repo_root}/replay_data/we11/wing_angle/${replay_hz}hz.csv"
    ;;
  *)
    echo "Usage: $0 [0|1|2|3] [flat|getup] [wrench_scale] [--difficulty 0..1 | --balance-difficulty 0..1]" >&2
    echo "Legacy: $0 [0|1|2|3] [wrench_scale] [upright|getup]" >&2
    exit 2
    ;;
esac

required_paths=(
  "${replay_bin}"
  "${repo_root}/policy/dr002/${selector}/config.yaml"
  "${policy_path}"
  "${scene_xml}"
)
if [[ "${replay_hz}" != "0" ]]; then
  required_paths+=("${force_csv}" "${wing_angle_csv}")
fi
for required_path in "${required_paths[@]}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing WE11 replay dependency: ${required_path}" >&2
    exit 1
  fi
done
if [[ ! -x "${replay_bin}" ]]; then
  echo "WE11 replay binary is not executable; run ./build.sh first." >&2
  exit 1
fi

echo "WE11 Play policy: ${policy_path}"
echo "WE11 Play profile: ${start_profile}"
echo "WE11 Play start pose: ${start_pose}"
if [[ -n "${getup_difficulty}" ]]; then
  echo "WE11 Play getup difficulty: ${getup_difficulty}"
fi
if [[ -n "${balance_difficulty}" ]]; then
  echo "WE11 Play balance difficulty: ${balance_difficulty}"
fi
if [[ -n "${getup_stage}" ]]; then
  echo "WE11 Play training stage: ${getup_stage}"
  echo "WE11 Play stage difficulty: ${getup_difficulty:-1.0}"
fi

unset RL_SAR_FORCE_APPLY_MEASURED_MOMENT
unset RL_SAR_FORCE_CSV
unset RL_SAR_FORCE_PERIOD
unset RL_SAR_FORCE_PRINT
unset RL_SAR_FORCE_ROTATION
unset RL_SAR_FORCE_SCALE
unset RL_SAR_FORCE_SYNC_WING_TIMING
unset RL_SAR_FORCE_TRACE_CSV
unset RL_SAR_FORCE_TRACE_STRIDE
unset RL_SAR_FORCE_VIS
unset RL_SAR_FORCE_VIS_ONLY
unset RL_SAR_FORCE_VIS_SCALE
unset RL_SAR_FORCE_ZERO_FY
unset RL_SAR_WING_ANGLE_CSV
unset RL_SAR_WING_ANGLE_NORMALIZATION_DEG
unset RL_SAR_WING_ANGLE_ZERO_OFFSETS_DEG
unset RL_SAR_WING_VISUAL
unset RL_SAR_DR002_AUTOSTART
unset RL_SAR_DR002_START_POSE
unset RL_SAR_DR002_GETUP_DIFFICULTY
unset RL_SAR_DR002_BALANCE_DIFFICULTY
unset RL_SAR_DR002_GETUP_STAGE
unset RL_SAR_DR002_GETUP_STAGE_DIFFICULTY
unset RL_SAR_DR002_BASE_COM_OFFSET_BODY
unset RL_SAR_DEPLOY_LIMITS
unset RL_SAR_TRACE_ACTIONS

if [[ "${play_autostart}" == "1" ]]; then
  export RL_SAR_DR002_AUTOSTART=1
fi
export RL_SAR_DR002_START_POSE="${start_pose}"
if [[ -n "${getup_stage}" ]]; then
  export RL_SAR_DR002_GETUP_STAGE="${getup_stage}"
  export RL_SAR_DR002_GETUP_STAGE_DIFFICULTY="${getup_difficulty:-1.0}"
elif [[ -n "${getup_difficulty}" ]]; then
  export RL_SAR_DR002_GETUP_DIFFICULTY="${getup_difficulty}"
fi
if [[ -n "${balance_difficulty}" ]]; then
  export RL_SAR_DR002_BALANCE_DIFFICULTY="${balance_difficulty}"
fi

if [[ "${replay_hz}" != "0" ]]; then
  export RL_SAR_FORCE_APPLY_MEASURED_MOMENT=1
  export RL_SAR_FORCE_CSV="${force_csv}"
  export RL_SAR_FORCE_PERIOD=10.0
  export RL_SAR_FORCE_ROTATION="0.7907964138 0.0 -0.6120792693 0.0 1.0 0.0 0.6120792693 0.0 0.7907964138"
  export RL_SAR_FORCE_SCALE="${force_scale}"
  export RL_SAR_FORCE_SYNC_WING_TIMING=1
  export RL_SAR_FORCE_ZERO_FY=1
  export RL_SAR_WING_ANGLE_CSV="${wing_angle_csv}"
  export RL_SAR_WING_ANGLE_NORMALIZATION_DEG=180.0
  export RL_SAR_WING_ANGLE_ZERO_OFFSETS_DEG="0.0 0.0"
fi

export RL_SAR_DR002_CONFIG="${selector}"
exec "${replay_bin}" dr002 "${selector}/${scene_name}"
