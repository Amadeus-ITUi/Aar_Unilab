#!/usr/bin/env bash
# Source this file; it deliberately removes ROS Python 3.10 paths while leaving
# ROS executables and native libraries alone.
_AAR_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export CONDA_PREFIX=/ssd/conda/envs/aar_unilab
export PATH="$CONDA_PREFIX/bin:$PATH"
export CONDA_PKGS_DIRS=/ssd/conda/pkgs
export PIP_CACHE_DIR=/ssd/conda/cache/pip
export TORCH_EXTENSIONS_DIR=/ssd/conda/cache/aar_unilab-native
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export AAR_UNILAB_ROOT="$_AAR_REPO_ROOT"
unset _AAR_REPO_ROOT
