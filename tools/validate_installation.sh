#!/usr/bin/env bash
set -euo pipefail

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
"$PYTHON" -m ruff check src/unilab/catalog scripts/train.py scripts/play.py tests/catalog tests/release tests/pe01
git -C "$REPO_ROOT" diff --check -- . ':(exclude)references/**'

if [[ "$RUN_ALL" == true ]]; then
  "$PYTHON" -m pytest -m "not slow"
  "$PYTHON" -m mypy src/unilab
  cmake -S "$REPO_ROOT/sim2sim" -B "$REPO_ROOT/sim2sim/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$REPO_ROOT/sim2sim/build" --parallel
  "$REPO_ROOT/sim2sim/build/aar_contract_check" "$REPO_ROOT/releases/examples/we11_flat/deployment_manifest.json"
fi
