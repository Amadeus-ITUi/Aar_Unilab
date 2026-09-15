#!/usr/bin/env bash
set -euo pipefail

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

if [[ -f "$REPO_ROOT/sim2sim/CMakeLists.txt" ]]; then
  cmake -S "$REPO_ROOT/sim2sim" -B "$REPO_ROOT/sim2sim/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/ssd/conda/cache/aar_unilab-native
  cmake --build "$REPO_ROOT/sim2sim/build" --parallel
fi

echo "Bound environment $ENV_PREFIX to $REPO_ROOT"
