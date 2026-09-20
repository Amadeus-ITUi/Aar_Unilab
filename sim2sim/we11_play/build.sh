#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
jobs="${RL_SAR_BUILD_JOBS:-$(nproc 2>/dev/null || echo 4)}"
python_bin="${PYTHON:-$(command -v python)}"
runtime_root="${AAR_WE11_PLAY_RUNTIME_ROOT:-/ssd/conda/cache/aar_unilab-native/we11-play-runtime}"

/usr/bin/cmake -S "${repo_root}" -B "${repo_root}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DPython3_EXECUTABLE="${python_bin}" \
  -DAAR_WE11_RUNTIME_ROOT="${runtime_root}"
/usr/bin/cmake --build "${repo_root}/build" --target rl_sim_mujoco --parallel "${jobs}"

echo "Built: ${repo_root}/build/bin/rl_sim_mujoco"
