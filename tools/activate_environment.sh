#!/usr/bin/env bash
# Source this file. It activates the SSD Miniforge prefix without requiring
# conda init, then removes ROS Python 3.10 paths while retaining native tools.
_AAR_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
_AAR_CONDA_SH=/ssd/conda/miniforge3/etc/profile.d/conda.sh
_AAR_ENV_PREFIX=/ssd/conda/envs/aar_unilab
if [[ ! -f "$_AAR_CONDA_SH" || ! -x "$_AAR_ENV_PREFIX/bin/python" ]]; then
  echo "Aar_Unilab SSD environment is missing; run tools/install_environment.sh" >&2
  unset _AAR_REPO_ROOT _AAR_CONDA_SH _AAR_ENV_PREFIX
  return 1
fi
source "$_AAR_CONDA_SH"
conda activate "$_AAR_ENV_PREFIX" || {
  unset _AAR_REPO_ROOT _AAR_CONDA_SH _AAR_ENV_PREFIX
  return 1
}
export CONDA_PKGS_DIRS=/ssd/conda/pkgs
export PIP_CACHE_DIR=/ssd/conda/cache/pip
export TORCH_EXTENSIONS_DIR=/ssd/conda/cache/aar_unilab-native
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export AAR_UNILAB_ROOT="$_AAR_REPO_ROOT"
unset _AAR_REPO_ROOT _AAR_CONDA_SH _AAR_ENV_PREFIX
