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
            np.testing.assert_array_equal(env.gaits, 0)
            np.testing.assert_array_equal(env.foot_air_time, 0)
            np.testing.assert_array_equal(env.foot_contact_time, 0)
            assert not env.backend.randomization
            assert not cfg.noise.enabled and not cfg.domain_rand.push_enabled
        env.step(np.zeros((4, 6)))
        np.testing.assert_array_equal(env.phase, 0)

        data = mujoco.MjData(env.backend.model)
        data.qpos[:] = env.home
        mujoco.mj_forward(env.backend.model, data)
        assert data.ncon > 0
        allowed = {env.backend.model.geom(name).id for name in cfg.env.foot_contact_geoms}
        for contact in data.contact:
            assert contact.dist >= -1e-7
            assert (contact.geom1 in allowed) != (contact.geom2 in allowed)

        baseline = load_config()
        assert cfg.reward.scales.base_height < 0
        assert cfg.reward.base_height_std == 0.03
        assert cfg.training.evaluation_height_tolerance == 0.01
        assert cfg.commands.zero_probability == 0.2
        assert cfg.observation == "pe02_v3"
        assert initial_actor.shape == (4, 240)
        assert cfg.control == baseline.control
        assert cfg.network == baseline.network
    finally:
        env.close()


def freeze_contacts(env, monkeypatch, contacts=(True, False)):
    monkeypatch.setattr(env.backend, "step", lambda *args, **kwargs: None)
    env.backend.sensors[:, env.backend.contact_adr[:, None] + np.arange(3)] = 0
    env.backend.sensors[:, env.backend.foot_ground_contact_adr[:, None] + np.arange(3)] = 0
    env.backend.sensors[:, env.backend.foot_ground_contact_adr] = np.array(contacts) * 20


def test_biped_air_time_is_bounded_command_gated_and_requires_single_ground_support(monkeypatch):
    env = PE02VectorEnv(walking_config(), evaluation=True, auto_reset=False)
    try:
        freeze_contacts(env, monkeypatch)
        env.commands[:] = [0.2, 0, 0]
        for _ in range(3):
            state = env.step(np.zeros((4, 6)))
        assert state.info["reward_terms"]["feet_air_time"] == pytest.approx(0.25 * 0.06 * 0.02)
        for _ in range(20):
            state = env.step(np.zeros((4, 6)))
        assert state.info["reward_terms"]["feet_air_time"] == pytest.approx(0.25 * 0.25 * 0.02)
        saved = env.snapshot()
        untouched = env.foot_air_time[1:].copy()
        env._reset_ids(np.array([0]))
        np.testing.assert_array_equal(env.foot_air_time[0], 0)
        np.testing.assert_array_equal(env.foot_contact_time[0], 0)
        np.testing.assert_array_equal(env.foot_air_time[1:], untouched)
        env.restore(saved)
        np.testing.assert_array_equal(env.foot_air_time, saved["arrays"]["foot_air_time"])

        for command, contacts, rewarded in (
            ([0, 0, 0], (True, False), False),
            ([0, 0, 0.5], (True, False), True),
            ([0.2, 0, 0], (True, True), False),
            ([0.2, 0, 0], (False, False), False),
        ):
            freeze_contacts(env, monkeypatch, contacts)
            env.commands[:] = command
            state = env.step(np.zeros((4, 6)))
            assert (state.info["reward_terms"]["feet_air_time"] > 0) == rewarded

        # Foot self-contact is not ground support.
        env.backend.sensors[:, env.backend.contact_adr[env.backend.foot_indices]] = 20
        assert env.step(np.zeros((4, 6))).info["reward_terms"]["feet_air_time"] == 0
        freeze_contacts(env, monkeypatch)
        env.backend.sensors[:, env.backend.contact_adr[0]] = 6
        assert env.step(np.zeros((4, 6))).info["reward_terms"]["feet_air_time"] == 0
    finally:
        env.close()


def test_v3_has_no_clock_objective_and_penalizes_only_ground_sliding(monkeypatch):
    env = PE02VectorEnv(walking_config(), evaluation=True)
    try:
        freeze_contacts(env, monkeypatch)
        frame, _ = env._frame(np.arange(4))
        total, terms = env._rewards()
        assert frame.shape == (4, 24)
        for name in (
            "keep_balance",
            "tracking_contacts_shaped_force",
            "tracking_contacts_shaped_vel",
        ):
            assert name not in terms
        env.phase[:] = 0.37
        env.gaits[:] = [5, 0.3, 0.2, 0.1]
        np.testing.assert_array_equal(env._frame(np.arange(4))[0], frame)
        np.testing.assert_array_equal(env._rewards()[0], total)
        env.foot_velocity[:, 1, :2] = [1, 0]
        assert not env._rewards()[1]["feet_slide"].any()
        env.foot_velocity[:, 0, :2] = [0.3, 0.4]
        np.testing.assert_allclose(env._rewards()[1]["feet_slide"], -0.5 * env.dt)
    finally:
        env.close()


@pytest.mark.parametrize("height_std", [0.03, 0.06])
@pytest.mark.parametrize("height_target", [None, 0.28])
def test_walking_height_penalty_tracks_target_with_configured_std(height_std, height_target):
    cfg = walking_config(f"reward.base_height_std={height_std}", "reward.scales.base_height=-30")
    cfg.reward.base_height_target = height_target
    env = PE02VectorEnv(cfg, evaluation=True)
    try:
        assert env.height_target == (env.home[2] if height_target is None else height_target)
        env.backend.qpos[:, 2] = env.height_target + np.array([0, 0.005, -0.005, 0.01])
        _, terms = env._rewards()
        # At std=3 cm and weight=-30, +/-5 mm costs 1/60 per step; 1 cm costs 1/15.
        expected = np.array([0, -1 / 60, -1 / 60, -1 / 15]) * (0.03 / height_std) ** 2
        np.testing.assert_allclose(terms["base_height"], expected, atol=1e-12)
        env.backend.qpos[:, 2] = env.height_target + 10
        np.testing.assert_allclose(env._rewards()[1]["base_height"], -cfg.reward.clip_single)
    finally:
        env.close()


@pytest.mark.parametrize("height_std", [0, -0.3, float("nan"), float("inf")])
def test_height_std_must_be_positive_and_finite(height_std):
    with pytest.raises(ValueError, match="reward.base_height_std must be positive and finite"):
        walking_config(f"reward.base_height_std={height_std}")


@pytest.mark.parametrize("support_foot", [0, 1])
def test_air_height_uses_airborne_sole_clearance_and_saturates(monkeypatch, support_foot):
    env = PE02VectorEnv(walking_config(), evaluation=True)
    try:
        freeze_contacts(env, monkeypatch, (support_foot == 0, support_foot == 1))
        env.commands[:] = [0.2, 0, 0]
        # Lift the foot pose sensors from home; the backend computes the sole's lowest point.
        home_heights = env.backend.foot_clearances()
        np.testing.assert_allclose(home_heights, 0, atol=1e-7)
        z_addresses = env.backend.sole_pos_adr + 2
        home_z = env.backend.sensors[:, z_addresses].copy()
        for heights, fractions in (
            ([0, 0.01, 0.055, 0.10], [0, 0, 0.5, 1]),
            ([0.005, 0.02, 0.07, 0.15], [0, 1 / 9, 2 / 3, 1]),
        ):
            env.backend.sensors[:, z_addresses] = home_z
            env.backend.sensors[:, z_addresses[1 - support_foot]] += heights
            # Even an elevated foot contributes nothing when its contact sensor reports support.
            env.backend.sensors[:, z_addresses[support_foot]] += 0.2
            np.testing.assert_allclose(
                env.backend.foot_clearances()[:, 1 - support_foot], heights, atol=1e-7
            )
            actual = env._rewards()[1]["feet_air_height"]
            np.testing.assert_allclose(actual, np.array(fractions) * 0.005, atol=1e-8)
    finally:
        env.close()


def test_air_height_requires_moving_command_single_support_and_stable_body(monkeypatch):
    env = PE02VectorEnv(walking_config(), evaluation=True, auto_reset=False)
    try:
        monkeypatch.setattr(env.backend, "foot_clearances", lambda: np.full((4, 2), 0.10))
        for command, contacts, expected in (
            ([0.2, 0, 0], (True, False), 0.005),
            ([0, 0, 0.5], (False, True), 0.005),
            ([0, 0, 0], (True, False), 0),
            ([0.05, 0, 0], (True, False), 0),
            ([0.2, 0, 0], (True, True), 0),
            ([0.2, 0, 0], (False, False), 0),
        ):
            freeze_contacts(env, monkeypatch, contacts)
            env.commands[:] = command
            state = env.step(np.zeros((4, 6)))
            assert state.info["reward_terms"]["feet_air_height"] == pytest.approx(expected)

        freeze_contacts(env, monkeypatch)
        env.commands[:] = [0.2, 0, 0]
        # Different environments: body contact, hip contact, severe tilt, valid single support.
        env.backend.sensors[0, env.backend.contact_adr[0]] = 6
        env.backend.sensors[1, env.backend.contact_adr[1]] = 6
        half_angle = np.deg2rad(65) / 2
        env.backend.qpos[2, 3:7] = [np.cos(half_angle), np.sin(half_angle), 0, 0]
        np.testing.assert_allclose(env._rewards()[1]["feet_air_height"], [0, 0, 0, 0.005])
    finally:
        env.close()


@pytest.mark.parametrize("remove_scale", [False, True])
def test_disabled_or_older_air_height_config_needs_no_height_range(remove_scale):
    cfg = walking_config()
    if remove_scale:
        del cfg.reward.scales.feet_air_height
    else:
        cfg.reward.scales.feet_air_height = 0
    del cfg.reward.feet_air_height_range
    env = PE02VectorEnv(cfg, evaluation=True)
    try:
        state = env.step(np.zeros((4, 6)))
        assert "feet_air_height" not in state.info["reward_terms"]
        assert not len(env.backend.sole_pos_adr)
        assert np.isfinite(state.reward).all()
    finally:
        env.close()


@pytest.mark.parametrize("height_range", ["[0.1,0.1]", "[0.1,0.01]", "[-0.01,0.1]", "null"])
def test_air_height_range_rejects_invalid_active_bounds(height_range):
    with pytest.raises(ValueError, match="reward.feet_air_height_range"):
        walking_config(f"reward.feet_air_height_range={height_range}")


def test_sole_clearance_matches_collision_geometry_at_tilted_poses():
    env = PE02VectorEnv(walking_config(), evaluation=True)
    try:
        b = env.backend
        qpos = np.tile(env.home, (4, 1))
        qpos[:, 2] += [0, 0.05, 0.15, 0.25]
        for index, (axis, angle) in enumerate(((0, 0), (1, 0.3), (2, -0.2), (3, 0.6))):
            qpos[index, 3:7] = [np.cos(angle / 2), 0, 0, 0]
            if axis:
                qpos[index, 3 + axis] = np.sin(angle / 2)
        b.reset(np.arange(4), qpos, np.zeros((4, 12)))
        actual = b.foot_clearances()
        expected = np.empty((4, 2))
        data = mujoco.MjData(b.model)
        for row in range(4):
            data.qpos[:] = qpos[row]
            mujoco.mj_forward(b.model, data)
            for col, name in enumerate(env.config.env.foot_contact_geoms):
                geom = b.model.geom(name).id
                mesh = b.model.geom_dataid[geom]
                start, count = b.model.mesh_vertadr[mesh], b.model.mesh_vertnum[mesh]
                vertices = b.model.mesh_vert[start : start + count]
                world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
                expected[row, col] = max(0, world[:, 2].min())
        np.testing.assert_allclose(actual, expected, atol=1e-9)
    finally:
        env.close()


def test_standing_and_sampled_yaw_commands_ignore_heading_error(monkeypatch):
    env = PE02VectorEnv(walking_config(), auto_reset=False)
    try:
        count = 0
        for _ in range(1000):
            env._resample_commands(np.arange(4))
            count += int(env.standing_command.sum())
        assert 0.18 < count / 4000 < 0.22
        freeze_contacts(env, monkeypatch, (True, True))
        env.standing_command[:] = [True, False, True, False]
        env.headings[:] = np.pi
        env.commands[:] = [0.5, 0.2, 0.37]
        env.commands[3, 2] = -0.42
        env.step(np.zeros((4, 6)))
        np.testing.assert_array_equal(env.commands[[0, 2]], 0)
        np.testing.assert_array_equal(env.commands[[1, 3], 2], [0.37, -0.42])
    finally:
        env.close()


def test_severe_tilt_and_base_contact_penalized_before_weak_failure(monkeypatch):
    env = PE02VectorEnv(walking_config(), evaluation=True, auto_reset=False)
    try:
        freeze_contacts(env, monkeypatch)
        # 65 degrees is penalized immediately, but below the old ~84 degree failure threshold.
        half_angle = np.deg2rad(65) / 2
        env.backend.qpos[:, 3:7] = [np.cos(half_angle), np.sin(half_angle), 0, 0]
        state = env.step(np.zeros((4, 6)))
        assert state.info["reward_terms"]["severe_tilt"] == pytest.approx(-0.4)
        assert not state.terminated.any()
        np.testing.assert_array_equal(env.failure_steps, 0)
        env.backend.qpos[:, 3:7] = [1, 0, 0, 0]
        env.backend.sensors[:, env.backend.contact_adr[0]] = 6
        state = env.step(np.zeros((4, 6)))
        assert state.info["reward_terms"]["base_contact"] == pytest.approx(-0.4)
        assert state.info["reward_terms"]["collision"] == pytest.approx(-0.06)
        assert not state.terminated.any()
        np.testing.assert_array_equal(env.failure_steps, 1)
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
