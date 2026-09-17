"""Fixed-home walking must retain exploration while classifying only soles as feet."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from unilab.envs.locomotion.pe02.config import ROOT, load_config
from unilab.envs.locomotion.pe02.vector_env import PE02VectorEnv


def walking_config(*overrides):
    return load_config(
        ["+experiment=walking", "algo.num_envs=4", "training.mujoco_threads=1", *overrides]
    )


@pytest.mark.parametrize("evaluation", [False, True])
def test_walking_full_and_partial_resets_keep_home_without_randomization(evaluation):
    cfg = walking_config()
    env = PE02VectorEnv(cfg, evaluation=evaluation, evaluation_reset_noise=True)
    try:
        initial_actor = env.state.obs["actor"].copy()
        for _ in range(2):
            for _ in range(8):
                state = env.step(np.full((4, 6), 0.2))
                assert all(np.isfinite(values).all() for values in state.obs.values())
            untouched = env.backend.state[[1, 3]].copy()
            env._reset_ids(np.array([0, 2]))
            np.testing.assert_array_equal(env.backend.state[[1, 3]], untouched)
            np.testing.assert_array_equal(env.backend.qpos[[0, 2]], np.tile(env.home, (2, 1)))
            np.testing.assert_array_equal(env.backend.qvel[[0, 2]], 0)
            state = env.reset()
            np.testing.assert_array_equal(state.obs["actor"], initial_actor)
            np.testing.assert_array_equal(env.backend.qpos, np.tile(env.home, (4, 1)))
            np.testing.assert_array_equal(env.backend.qvel, 0)
            np.testing.assert_array_equal(env.backend.default_position, env.backend.qpos[:, 7:])
            np.testing.assert_array_equal(env.backend.kp, [cfg.control.kp] * 4)
            np.testing.assert_array_equal(env.backend.kd, [cfg.control.kd] * 4)
            np.testing.assert_array_equal(env.backend.torque_scale, 1)
            np.testing.assert_array_equal(env.backend.delay_steps, 0)
            np.testing.assert_array_equal(env.imu_offset, [[1, 0, 0, 0]] * 4)
            np.testing.assert_array_equal(env.gaits, [[2, 0.5, 0.5, 0.06]] * 4)
            assert not env.backend.randomization
            assert not cfg.noise.enabled and not cfg.domain_rand.push_enabled
        env.step(np.zeros((4, 6)))
        np.testing.assert_allclose(env.phase, 0.04)

        data = mujoco.MjData(env.backend.model)
        data.qpos[:] = env.home
        mujoco.mj_forward(env.backend.model, data)
        assert data.ncon > 0
        allowed = {env.backend.model.geom(name).id for name in cfg.env.foot_contact_geoms}
        for contact in data.contact:
            assert contact.dist >= -1e-7
            assert (contact.geom1 in allowed) != (contact.geom2 in allowed)

        baseline = load_config()
        assert cfg.reward == baseline.reward
        assert cfg.commands == baseline.commands
        assert cfg.control == baseline.control
        assert cfg.network == baseline.network
    finally:
        env.close()


def test_only_sole_contacts_are_exempt_with_original_loose_termination(monkeypatch):
    env = PE02VectorEnv(walking_config(), auto_reset=False)
    try:
        # Freeze physics at home to isolate contact classification and the failure timer.
        monkeypatch.setattr(env.backend, "step", lambda *args, **kwargs: None)
        adr = env.backend.contact_adr
        env.backend.sensors[:, adr[:, None] + np.arange(3)] = 0
        env.backend.sensors[:, adr[env.backend.foot_indices]] = 20
        state = env.step(np.zeros((4, 6)))
        assert state.info["reward_terms"]["collision"] == 0
        assert not state.info["nonfoot_contact"].any()
        assert not state.terminated.any()

        # Hip contact used to be exempt in the migration baseline; it is not a sole.
        hip = env.cfg["env"]["body_names"].index("R_hip_Link")
        env.backend.sensors[:, adr[hip]] = 10
        for _ in range(30):
            state = env.step(np.zeros((4, 6)))
        assert state.info["reward_terms"]["collision"] == pytest.approx(-3 * env.dt)
        assert state.info["nonfoot_contact"].all()
        assert not state.terminated.any()

        # Body force must exceed 5 N; failure ticks accumulate across interruptions.
        body = env.cfg["env"]["body_names"].index("body_link")
        env.backend.sensors[:, adr[body]] = 4
        env.step(np.zeros((4, 6)))
        np.testing.assert_array_equal(env.failure_steps, 0)
        env.backend.sensors[:, adr[body]] = 6
        for _ in range(13):
            env.step(np.zeros((4, 6)))
        env.backend.sensors[:, adr[body]] = 0
        env.step(np.zeros((4, 6)))
        np.testing.assert_array_equal(env.failure_steps, 13)
        env.backend.sensors[:, adr[body]] = 6
        for _ in range(12):
            state = env.step(np.zeros((4, 6)))
        assert not state.terminated.any()
        assert env.step(np.zeros((4, 6))).terminated.all()
    finally:
        env.close()


def test_walking_rejects_misnamed_sole_and_unclassified_foot_geometry(tmp_path):
    cfg = walking_config()
    cfg.env.foot_contact_geoms[1] = "R_foot_Link_visual_0"
    with pytest.raises(ValueError, match="foot_contact_geoms"):
        PE02VectorEnv(cfg)

    cfg = walking_config()
    source = ROOT / cfg.env.model_path
    scene = ET.parse(source)
    robot = ET.parse(source.parent / "pe02.xml")
    robot.getroot().find("compiler").set("meshdir", str(source.parent / "runtime_meshes"))
    foot = robot.getroot().find(".//body[@name='R_foot_Link']")
    ET.SubElement(foot, "geom", name="extra_foot_shell", type="sphere", size="0.005")
    robot.write(tmp_path / "pe02.xml")
    scene.write(tmp_path / "scene.xml")
    with pytest.raises(ValueError, match="Additional foot geometry"):
        PE02VectorEnv(cfg, model_path=tmp_path / "scene.xml")
