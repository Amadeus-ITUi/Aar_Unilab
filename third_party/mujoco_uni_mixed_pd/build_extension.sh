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
CACHE_DIR="${UNILAB_NATIVE_BUILD_CACHE:-/ssd/conda/cache/aar_unilab-native/mujoco_uni_mixed_pd}"
ARCHIVE_PATH="${CACHE_DIR}/${SDIST_NAME}"
SOURCE_DIR="${CACHE_DIR}/source-${VERSION}"
FETCH_DIR="${CACHE_DIR}/fetchcontent"
PATCH_PATH="${ROOT_DIR}/third_party/mujoco_uni_mixed_pd/batch_env.patch"
COMMAND_DELAY_PATCH_PATH="${ROOT_DIR}/third_party/mujoco_uni_mixed_pd/command_delay_pd.patch"
JOINT_POSITION_PATCH_PATH="${ROOT_DIR}/third_party/mujoco_uni_mixed_pd/joint_position_pd.patch"
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
DEST_SO="${DEST_DIR}/_unilab_batch_env${EXT_SUFFIX}"

if [[ -f "${DEST_SO}" ]] && \
   PYTHONPATH="${ROOT_DIR}/src" "${PYTHON_BIN}" -c \
     'from unilab.base.backend.mujoco.native_batch import native_command_delay_pd_available, native_mixed_pd_available, native_joint_position_pd_available; from unilab.base.backend.mujoco.native_batch import _unilab_batch_env; assert native_mixed_pd_available() and native_command_delay_pd_available() and native_joint_position_pd_available() and hasattr(_unilab_batch_env.BatchEnvPool, "has_identified_joint_pd")'; then
  echo "UniLab native torque-FIFO, command-delay and joint-position PD extension: already ready"
  exit 0
fi

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  curl --fail --location "${SDIST_URL}" --output "${ARCHIVE_PATH}"
fi
printf '%s  %s\n' "${SDIST_SHA256}" "${ARCHIVE_PATH}" | sha256sum --check --status

rm -rf "${SOURCE_DIR}"
mkdir -p "${SOURCE_DIR}"
tar -xzf "${ARCHIVE_PATH}" --strip-components=1 -C "${SOURCE_DIR}"
patch --directory="${SOURCE_DIR}" --strip=1 < "${PATCH_PATH}"
patch --directory="${SOURCE_DIR}" --strip=1 < "${COMMAND_DELAY_PATCH_PATH}"
patch --directory="${SOURCE_DIR}" --strip=1 < "${JOINT_POSITION_PATCH_PATH}"
patch --directory="${SOURCE_DIR}" --strip=1 < "${ROOT_DIR}/third_party/mujoco_uni_mixed_pd/controlled_forward.patch"
patch --directory="${SOURCE_DIR}" --strip=1 < "${ROOT_DIR}/third_party/mujoco_uni_mixed_pd/joint_pd_telemetry.patch"
patch --directory="${SOURCE_DIR}" --strip=1 < "${ROOT_DIR}/third_party/mujoco_uni_mixed_pd/identified_joint_pd.patch"

pushd "${SOURCE_DIR}" >/dev/null
MUJOCO_PATH="${MUJOCO_DIR}" \
MUJOCO_PLUGIN_PATH="${MUJOCO_DIR}/plugin" \
MUJOCO_PYTHON_EXTENSIONS="mujoco._batch_env" \
MUJOCO_FETCHCONTENT_BASE_DIR="${FETCH_DIR}" \
MUJOCO_CMAKE_ARGS="${CMAKE_ARGS}" \
  "${PYTHON_BIN}" setup.py build_ext --inplace
popd >/dev/null

SOURCE_SO="${SOURCE_DIR}/mujoco/_batch_env${EXT_SUFFIX}"
install -m 0755 "${SOURCE_SO}" "${DEST_SO}"

PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - <<'PY'
from unilab.base.backend.mujoco.native_batch import (
    native_command_delay_pd_available,
    native_mixed_pd_available,
    native_joint_position_pd_available,
    _unilab_batch_env,
)

if not all((native_mixed_pd_available(), native_command_delay_pd_available(), native_joint_position_pd_available())):
    raise SystemExit("native PD extension import check failed")
assert hasattr(_unilab_batch_env.BatchEnvPool, "has_identified_joint_pd")
print("UniLab native torque-FIFO, command-delay and joint-position PD extension: ready")
PY
