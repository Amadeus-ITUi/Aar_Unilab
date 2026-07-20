#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${UNILAB_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" && -n "${CONDA_PREFIX:-}" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python)"
fi

VERSION="3.8.0"
SDIST_NAME="mujoco_uni-${VERSION}.tar.gz"
SDIST_URL="https://files.pythonhosted.org/packages/d7/af/c042c865b78400d1309736903168a07dea67d84872a9b999bc19328de412/${SDIST_NAME}"
SDIST_SHA256="8b1e32dce4f6b7f87b4cb7bfa55ab25a9a9e3191b6bde865196eaa04bf1baac5"
CACHE_DIR="${UNILAB_NATIVE_BUILD_CACHE:-${HOME}/.cache/unilab/mujoco_uni_mixed_pd}"
ARCHIVE_PATH="${CACHE_DIR}/${SDIST_NAME}"
SOURCE_DIR="${CACHE_DIR}/source-${VERSION}"
FETCH_DIR="${CACHE_DIR}/fetchcontent"
PATCH_PATH="${ROOT_DIR}/third_party/mujoco_uni_mixed_pd/batch_env.patch"
DEST_DIR="${ROOT_DIR}/src/unilab/base/backend/mujoco/_native"
CMAKE_ARGS="-DMUJOCO_PYTHON_USE_SYSTEM_EIGEN=ON -DMUJOCO_PYTHON_BUILD_SIMULATE=OFF"
if [[ -n "${UNILAB_ABSEIL_SOURCE:-}" ]]; then
  CMAKE_ARGS+=" -DFETCHCONTENT_SOURCE_DIR_ABSEIL-CPP=${UNILAB_ABSEIL_SOURCE}"
fi
if [[ -n "${UNILAB_PYBIND11_SOURCE:-}" ]]; then
  CMAKE_ARGS+=" -DFETCHCONTENT_SOURCE_DIR_PYBIND11=${UNILAB_PYBIND11_SOURCE}"
fi

INSTALLED_VERSION="$(${PYTHON_BIN} -c 'import mujoco; print(mujoco.__version__)')"
if [[ "${INSTALLED_VERSION}" != "${VERSION}" ]]; then
  echo "Expected mujoco-uni ${VERSION}, found ${INSTALLED_VERSION}" >&2
  exit 1
fi

MUJOCO_DIR="$(${PYTHON_BIN} -c 'import pathlib, mujoco; print(pathlib.Path(mujoco.__file__).resolve().parent)')"
EXT_SUFFIX="$(${PYTHON_BIN} -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
mkdir -p "${CACHE_DIR}" "${FETCH_DIR}" "${DEST_DIR}"

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  curl --fail --location "${SDIST_URL}" --output "${ARCHIVE_PATH}"
fi
printf '%s  %s\n' "${SDIST_SHA256}" "${ARCHIVE_PATH}" | sha256sum --check --status

rm -rf "${SOURCE_DIR}"
mkdir -p "${SOURCE_DIR}"
tar -xzf "${ARCHIVE_PATH}" --strip-components=1 -C "${SOURCE_DIR}"
patch --directory="${SOURCE_DIR}" --strip=1 < "${PATCH_PATH}"

pushd "${SOURCE_DIR}" >/dev/null
MUJOCO_PATH="${MUJOCO_DIR}" \
MUJOCO_PLUGIN_PATH="${MUJOCO_DIR}/plugin" \
MUJOCO_PYTHON_EXTENSIONS="mujoco._batch_env" \
MUJOCO_FETCHCONTENT_BASE_DIR="${FETCH_DIR}" \
MUJOCO_CMAKE_ARGS="${CMAKE_ARGS}" \
  "${PYTHON_BIN}" setup.py build_ext --inplace
popd >/dev/null

SOURCE_SO="${SOURCE_DIR}/mujoco/_batch_env${EXT_SUFFIX}"
DEST_SO="${DEST_DIR}/_unilab_batch_env${EXT_SUFFIX}"
install -m 0755 "${SOURCE_SO}" "${DEST_SO}"

PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - <<'PY'
from unilab.base.backend.mujoco.native_batch import native_mixed_pd_available

if not native_mixed_pd_available():
    raise SystemExit("native mixed-PD extension import check failed")
print("UniLab native mixed-PD extension: ready")
PY
