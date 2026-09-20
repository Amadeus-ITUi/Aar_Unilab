#!/usr/bin/env bash
set -euo pipefail

unset PYTHONPATH
export PYTHONNOUSERSITE=1

ENV_PREFIX=/ssd/conda/envs/aar_unilab
while (($#)); do
  case "$1" in
    --prefix) ENV_PREFIX=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PIP_CACHE_DIR=/ssd/conda/cache/pip
export TORCH_EXTENSIONS_DIR=/ssd/conda/cache/aar_unilab-native
"$ENV_PREFIX/bin/python" -m pip install --editable "$REPO_ROOT[dev]"
UNILAB_PYTHON="$ENV_PREFIX/bin/python" \
UNILAB_NATIVE_BUILD_CACHE=/ssd/conda/cache/aar_unilab-native/mujoco_uni_mixed_pd \
  "$REPO_ROOT/third_party/mujoco_uni_mixed_pd/build_extension.sh"

if [[ -f "$REPO_ROOT/sim2sim/CMakeLists.txt" ]]; then
  MUJOCO_ROOT=$("$ENV_PREFIX/bin/python" -c 'import pathlib, mujoco; print(pathlib.Path(mujoco.__file__).parent)')
  ONNXRUNTIME_ROOT=$("$ENV_PREFIX/bin/python" -c 'import pathlib, onnxruntime; print(pathlib.Path(onnxruntime.__file__).parent / "capi")')
  if [[ ! -e "$ONNXRUNTIME_ROOT/libonnxruntime.so.1" ]]; then
    ln -s libonnxruntime.so.1.22.0 "$ONNXRUNTIME_ROOT/libonnxruntime.so.1"
  fi
  ONNXRUNTIME_INCLUDE="$REPO_ROOT/references/legacy_play_source/library/inference_runtime/onnxruntime/include"
  GLFW_ROOT=$("$ENV_PREFIX/bin/python" -c 'import pathlib, glfw; print(pathlib.Path(glfw.__file__).parent)')
  if [[ ! -e "$GLFW_ROOT/x11/libglfw.so.3" ]]; then
    ln -s libglfw.so "$GLFW_ROOT/x11/libglfw.so.3"
  fi
  GLFW_INCLUDE="$REPO_ROOT/references/legacy_play_source/library/glfw/include"
  cmake -S "$REPO_ROOT/sim2sim" -B "$REPO_ROOT/sim2sim/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/ssd/conda/cache/aar_unilab-native \
    -DAAR_MUJOCO_ROOT="$MUJOCO_ROOT" \
    -DAAR_ONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT" \
    -DAAR_ONNXRUNTIME_INCLUDE="$ONNXRUNTIME_INCLUDE" \
    -DAAR_GLFW_ROOT="$GLFW_ROOT" \
    -DAAR_GLFW_INCLUDE="$GLFW_INCLUDE"
  cmake --build "$REPO_ROOT/sim2sim/build" --parallel
fi

echo "Bound environment $ENV_PREFIX to $REPO_ROOT"
