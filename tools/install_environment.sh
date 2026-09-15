#!/usr/bin/env bash
set -euo pipefail

CONDA_BASE=/ssd/conda/miniforge3
ENV_PREFIX=/ssd/conda/envs/aar_unilab
TORCH_VARIANT=cu128
CONDA_ROOT=/ssd/conda
MINIFORGE_VERSION=26.7.2-0
MINIFORGE_SHA256=281b0ac7d550802efc81af633225a5e6116d29ae72f3ab4eae7168c3931a4c05

while (($#)); do
  case "$1" in
    --conda-base) CONDA_BASE=$2; shift 2 ;;
    --prefix) ENV_PREFIX=$2; shift 2 ;;
    --torch) TORCH_VARIANT=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PKGS_DIR="$CONDA_ROOT/pkgs"
PIP_CACHE="$CONDA_ROOT/cache/pip"
NATIVE_CACHE="$CONDA_ROOT/cache/aar_unilab-native"
INSTALLER="$CONDA_ROOT/cache/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh"
mkdir -p "$CONDA_ROOT/cache" "$PKGS_DIR" "$PIP_CACHE" "$NATIVE_CACHE" "$(dirname "$ENV_PREFIX")"

if [[ ! -x "$CONDA_BASE/bin/conda" ]]; then
  if [[ ! -f "$INSTALLER" ]]; then
    curl --fail --location --retry 3 \
      "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
      --output "$INSTALLER"
  fi
  printf '%s  %s\n' "$MINIFORGE_SHA256" "$INSTALLER" | sha256sum --check --status || {
    echo "Miniforge installer checksum mismatch: $INSTALLER" >&2
    exit 1
  }
  bash "$INSTALLER" -b -p "$CONDA_BASE"
fi

export CONDA_PKGS_DIRS="$PKGS_DIR"
export PIP_CACHE_DIR="$PIP_CACHE"
export TORCH_EXTENSIONS_DIR="$NATIVE_CACHE"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA_BASE/bin/conda" create --yes --prefix "$ENV_PREFIX" \
    --channel conda-forge --strict-channel-priority python=3.13 pip cmake ninja pkg-config compilers git curl
fi

case "$TORCH_VARIANT" in
  cu128) TORCH_INDEX=https://download.pytorch.org/whl/cu128 ;;
  cpu) TORCH_INDEX=https://download.pytorch.org/whl/cpu ;;
  *) echo "--torch must be cu128 or cpu" >&2; exit 2 ;;
esac

"$ENV_PREFIX/bin/python" -m pip install --index-url "$TORCH_INDEX" "torch==2.7.0"
"$ENV_PREFIX/bin/python" -m pip install "mujoco-uni==3.8.0"
"$ENV_PREFIX/bin/python" -m pip install --editable "$REPO_ROOT[dev]"
"$REPO_ROOT/tools/rebind_environment.sh" --prefix "$ENV_PREFIX"

echo "Environment ready: $ENV_PREFIX"
echo "Activate explicitly: source $CONDA_BASE/bin/activate $ENV_PREFIX"
