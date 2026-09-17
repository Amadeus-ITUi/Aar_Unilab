"""The accelerated controller must preserve the original PE02 rollout contract."""

import numpy as np
import pytest

from unilab.base.backend.mujoco.native_batch import native_joint_position_pd_available
from unilab.envs.locomotion.pe02.config import load_config
from unilab.envs.locomotion.pe02.vector_env import PE02VectorEnv


@pytest.mark.skipif(not native_joint_position_pd_available(), reason="native PD is not built")
@pytest.mark.parametrize("position_difference", [True, False])
@pytest.mark.parametrize("threads", [1, 3])
def test_native_pd_matches_substep_loop_with_delay_push_reset_and_restore(
    position_difference, threads
):
    envs = [
        PE02VectorEnv(
            load_config(
                [
                    "algo.num_envs=7",
                    "algo.num_mini_batches=1",
                    f"training.mujoco_threads={threads}",
                    f"training.native_pd={str(native).lower()}",
                    f"env.dof_vel_use_pos_diff={str(position_difference).lower()}",
                ]
            )
        )
        for native in (False, True)
    ]
    try:
        backends = [env.backend for env in envs]
        for backend in backends:
            backend.delay_steps[:] = [0, 1, 7, 8, 13, 19, 20]
            backend.delay_buffer = np.zeros((7, 21, 6))
        rng = np.random.default_rng(87)
        for step in range(60):
            actions = rng.normal(0, 0.15, (7, 6))
            forces = rng.normal(0, 2, (7, 8, 3)) if step in (8, 19) else None
            if step == 23:
                ids = np.array([1, 4])
                pose = np.tile(backends[0].home, (2, 1))
                velocity = rng.normal(0, 0.02, (2, 12))
                for backend in backends:
                    backend.reset(ids, pose, velocity)
            for backend in backends:
                if step == 31:
                    backend.restore(backend.snapshot())
                backend.external_wrench = (
                    np.full((7, backend.model.nbody, 6), 0.03) if step == 36 else None
                )
                backend.step(actions, 8, push_forces=forces)
            for name in ("state", "sensors", "torque", "control", "joint_velocity", "delay_buffer"):
                np.testing.assert_allclose(
                    getattr(backends[0], name),
                    getattr(backends[1], name),
                    atol=1e-10,
                    rtol=1e-10,
                    err_msg=f"step={step} field={name}",
                )
            np.testing.assert_array_equal(backends[0].steps, backends[1].steps)
    finally:
        for env in envs:
            env.close()
