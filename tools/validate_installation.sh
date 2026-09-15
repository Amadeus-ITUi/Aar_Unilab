#!/usr/bin/env bash
set -euo pipefail

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

ENV_PREFIX=/ssd/conda/envs/aar_unilab
RUN_ALL=false
while (($#)); do
  case "$1" in
    --prefix) ENV_PREFIX=$2; shift 2 ;;
    --all) RUN_ALL=true; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON="$ENV_PREFIX/bin/python"
test -x "$PYTHON"
"$PYTHON" -c 'import mujoco, torch, unilab; assert mujoco.__version__ == "3.8.0"; assert torch.__version__.split("+")[0] == "2.7.0"'
"$PYTHON" -m pytest tests/catalog tests/release tests/pe01 -q
"$PYTHON" "$REPO_ROOT/tools/audit_repository.py"
(
  cd "$REPO_ROOT/models/we11"
  sha256sum --check SHA256SUMS
)
"$PYTHON" -m ruff check src/unilab/catalog scripts/train.py scripts/play.py tests/catalog tests/release tests/pe01
git -C "$REPO_ROOT" diff --check -- . ':(exclude)references/**'

if [[ "$RUN_ALL" == true ]]; then
  "$PYTHON" -m pytest -m "not slow"
  "$PYTHON" -m mypy src/unilab
  for TASK_PROFILE in "flat:we11_legacy_135:model_1500.pt" "rough:we11_legacy_135:model_1500.pt" "getup:we11_v2_145:model_9999.pt"; do
    IFS=: read -r TASK_NAME OBSERVATION_NAME CHECKPOINT_NAME <<<"$TASK_PROFILE"
    AAR_EXPORT_POLICY=0 "$PYTHON" "$REPO_ROOT/scripts/play.py" \
      robot=we11 task="$TASK_NAME" observation="$OBSERVATION_NAME" \
      policy=we11_mlp algorithm=rsl_rl_ppo simulator=mujoco \
      "algo.load_run=$REPO_ROOT/models/we11/$TASK_NAME/$CHECKPOINT_NAME" \
      training.sim2sim_strict=false training.play_render_mode=none \
      training.play_steps=1 training.play_env_num=1 algo.num_envs=1
  done
  (
    cd "$REPO_ROOT"
    "$PYTHON" scripts/train_pe01.py training.steps=1 training.seed=1 training.device=cpu
  )
  MUJOCO_ROOT=$("$PYTHON" -c 'import pathlib, mujoco; print(pathlib.Path(mujoco.__file__).parent)')
  ONNXRUNTIME_ROOT=$("$PYTHON" -c 'import pathlib, onnxruntime; print(pathlib.Path(onnxruntime.__file__).parent / "capi")')
  GLFW_ROOT=$("$PYTHON" -c 'import pathlib, glfw; print(pathlib.Path(glfw.__file__).parent)')
  cmake -S "$REPO_ROOT/sim2sim" -B "$REPO_ROOT/sim2sim/build" \
    -DCMAKE_BUILD_TYPE=Release -DAAR_MUJOCO_ROOT="$MUJOCO_ROOT" \
    -DAAR_ONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT" \
    -DAAR_ONNXRUNTIME_INCLUDE="$REPO_ROOT/references/legacy_play_source/library/inference_runtime/onnxruntime/include" \
    -DAAR_GLFW_ROOT="$GLFW_ROOT" \
    -DAAR_GLFW_INCLUDE="$REPO_ROOT/references/legacy_play_source/library/glfw/include"
  cmake --build "$REPO_ROOT/sim2sim/build" --parallel
  ctest --test-dir "$REPO_ROOT/sim2sim/build" --output-on-failure
  WE11_RELEASE=/ssd/conda/cache/aar_unilab-native/we11-getup-release
  "$PYTHON" "$REPO_ROOT/tools/build_we11_sim2sim_release.py" \
    --destination "$WE11_RELEASE"
  for TRAJECTORY_STEPS in 1 10 100; do
    "$REPO_ROOT/sim2sim/build/aar_sim2sim" "$WE11_RELEASE" \
      --steps "$TRAJECTORY_STEPS" \
      --telemetry "/ssd/conda/cache/aar_unilab-native/we11-$TRAJECTORY_STEPS"
  done
  "$PYTHON" "$REPO_ROOT/tools/compare_we11_sim2sim.py" \
    --binary "$REPO_ROOT/sim2sim/build/aar_sim2sim" \
    --release "$WE11_RELEASE"
  TORCH_EXTENSIONS_DIR=/ssd/conda/cache/aar_unilab-native \
    "$PYTHON" "$REPO_ROOT/tools/compare_pe01_sim2sim.py" \
      --binary "$REPO_ROOT/sim2sim/build/aar_sim2sim" \
      --release "$REPO_ROOT/releases/examples/pe01_flat"
  if [[ -n "${DISPLAY:-}" ]]; then
    "$REPO_ROOT/sim2sim/build/aar_sim2sim" \
      "$REPO_ROOT/releases/examples/pe01_flat" --steps 2 --interactive \
      --telemetry /ssd/conda/cache/aar_unilab-native/pe01-ui-smoke
    "$REPO_ROOT/sim2sim/build/aar_sim2sim" "$WE11_RELEASE" --steps 2 --interactive \
      --telemetry /ssd/conda/cache/aar_unilab-native/we11-ui-smoke
  fi
fi
