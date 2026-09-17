"""The standing experiment must isolate initialization without changing the baseline."""

import mujoco
import numpy as np

from unilab.envs.locomotion.pe02.config import load_config
from unilab.envs.locomotion.pe02.vector_env import PE02VectorEnv


def test_standing_resets_are_ground_aligned_and_deterministic():
    cfg = load_config(["+experiment=standing", "algo.num_envs=4", "training.mujoco_threads=1"])
    env = PE02VectorEnv(cfg)
    try:
        initial = env.state.obs
        for _ in range(2):
            for _ in range(5):
                env.step(np.full((4, 6), 0.2))
            restored = env.reset()
            np.testing.assert_array_equal(env.backend.qpos, np.tile(env.home, (4, 1)))
            np.testing.assert_array_equal(env.backend.qvel, 0)
            np.testing.assert_array_equal(env.backend.default_position, env.backend.qpos[:, 7:])
            np.testing.assert_array_equal(env.backend.delay_steps, 0)
            for group, values in initial.items():
                np.testing.assert_array_equal(restored.obs[group], values)
        env._resample_commands(np.arange(4))
        env._resample_gaits(np.arange(4))
        np.testing.assert_array_equal(env.commands, 0)
        np.testing.assert_array_equal(env.gaits, [cfg.play.gait] * 4)
        assert not env.backend.randomization
        data = mujoco.MjData(env.backend.model)
        data.qpos[:] = env.home
        mujoco.mj_forward(env.backend.model, data)
        assert data.ncon > 0
        assert all(contact.dist >= -1e-7 for contact in data.contact)
        assert cfg.reward.scales.tracking_contacts_shaped_force == 0
        assert cfg.reward.scales.tracking_contacts_shaped_vel == 0
        assert set(cfg.env.termination_bodies) == set(cfg.env.body_names) - set(cfg.env.foot_names)
    finally:
        env.close()

    baseline = load_config()
    assert list(baseline.env.joint_reset_range) == [-0.5, 0.5]
    assert baseline.domain_rand.enabled and baseline.noise.enabled
    assert baseline.reward.scales.tracking_contacts_shaped_force == -2.0
    assert baseline.algo.num_envs == 8192
