#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
jobs="${RL_SAR_BUILD_JOBS:-$(nproc 2>/dev/null || echo 4)}"

/usr/bin/cmake -S "${repo_root}" -B "${repo_root}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=/usr/bin/python3
/usr/bin/cmake --build "${repo_root}/build" --target rl_sim_mujoco --parallel "${jobs}"

echo "Built: ${repo_root}/build/bin/rl_sim_mujoco"
