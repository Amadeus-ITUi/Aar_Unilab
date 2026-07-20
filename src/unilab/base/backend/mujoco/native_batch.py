from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mujoco
from mujoco.batch_env import BatchEnvPool

try:
    from ._native import _unilab_batch_env
except ImportError as exc:  # pragma: no cover - depends on local native build
    _unilab_batch_env = None
    _NATIVE_IMPORT_ERROR: ImportError | None = exc
else:
    _NATIVE_IMPORT_ERROR = None


def native_mixed_pd_available() -> bool:
    return _unilab_batch_env is not None and hasattr(
        _unilab_batch_env.BatchEnvPool, "step_mixed_pd"
    )


def native_mixed_pd_import_error() -> ImportError | None:
    return _NATIVE_IMPORT_ERROR


class NativeMixedPdBatchEnvPool(BatchEnvPool):
    """BatchEnvPool backed by UniLab's private mixed-PD native extension."""

    def __init__(
        self,
        model: mujoco.MjModel | Sequence[mujoco.MjModel],
        *,
        nbatch: int,
        nthread: int | None = None,
    ) -> None:
        if not native_mixed_pd_available():
            raise RuntimeError("UniLab native mixed-PD extension is unavailable")
        if nbatch <= 0:
            raise ValueError("nbatch must be positive")
        model_arg: Any = model if isinstance(model, mujoco.MjModel) else list(model)
        self._nthread = 0 if nthread is None else int(nthread)
        self._pool = _unilab_batch_env.BatchEnvPool(  # type: ignore[union-attr]
            model=model_arg,
            nbatch=int(nbatch),
            nthread=self._nthread,
        )

    def get_model(self, env_ids):  # noqa: ANN001, ANN201
        raise NotImplementedError(
            "The private mixed-PD pool intentionally does not export C++ model wrappers"
        )

    def get_all_models(self) -> list[mujoco.MjModel]:
        raise NotImplementedError(
            "The private mixed-PD pool intentionally does not export C++ model wrappers"
        )
