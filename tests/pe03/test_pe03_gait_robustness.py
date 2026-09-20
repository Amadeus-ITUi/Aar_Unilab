"""Robustness contracts: physical effects, measurement timing and restart isolation."""

import copy

import mujoco
import numpy as np
import pytest
import torch

from unilab.algos.torch.pe03.gait_evaluation import evaluate_gait
from unilab.algos.torch.pe03.policy import PE03EncoderPolicy
from unilab.base.backend.mujoco.collision_geometry import ground_clearances
from unilab.base.backend.mujoco.native_batch import native_joint_position_pd_available
from unilab.envs.locomotion.pe03.config import load_config
from unilab.envs.locomotion.pe03.gait_env import PE03GaitEnv
from unilab.envs.locomotion.pe03.gait_randomization import evaluation_protocol, push_forces
from unilab.envs.locomotion.pe03.math import inverse_rotate


def config(*overrides):
    return load_config(
        [
            "+experiment=gait_fixed",
            "algo.num_envs=7",
            "training.mujoco_threads=1",
            "training.device=cpu",
            "training.logger=none",
            "domain_rand.curriculum.enabled=false",
            *overrides,
        ]
    )


@pytest.mark.parametrize(
    "override",
    [
        "noise.level=-1",
        "domain_rand.delay_ms_range=[50,25]",
        "play.delay_ms=-1",
        "domain_rand.kp_range=[0,1]",
        "domain_rand.base_com_range=[0,0]",
        "env.reset_randomization.sole_clearance=-0.001",
        "env.reset_randomization.max_attempts=0",
        "training.evaluation_conditions=bad",
    ],
)
def test_invalid_robustness_is_rejected(override):
    with pytest.raises(ValueError):
        config(override)


def test_physical_sampling_and_reset_isolation():
    env = PE03GaitEnv(
        config(
            "domain_rand.delay_ms_range=[25,50]",
            "env.reset_randomization.linear_velocity_range=[-0.5,0.5]",
            "env.reset_randomization.angular_velocity_range=[-0.5,0.5]",
        )
    )
    try:
        b = env.backend
        assert np.all((b.delay_steps >= 10) & (b.delay_steps <= 20))
        assert np.all((b.kp / env.config.control.kp >= 0.8) & (b.kp / env.config.control.kp <= 1.2))
        np.testing.assert_allclose(
            env.weight, b.randomization["body_mass"].sum(1) * b.gravity_acceleration
        )
        friction = b.randomization["geom_friction"]
        for geoms in (b.robot_geoms, b.ground_geoms):
            np.testing.assert_allclose(
                friction[:, geoms, 0], np.broadcast_to(env.friction[:, None], (7, len(geoms)))
            )
        np.testing.assert_array_equal(
            friction[:, :, 1:], np.broadcast_to(b.base_friction[:, 1:], friction[:, :, 1:].shape)
        )
        np.testing.assert_allclose(b.foot_clearances().min(1), 0.001, atol=1e-14)
        assert np.max(np.abs(b.qpos[:, 7:] - env.home[7:])) <= 0.1
        assert np.max(np.abs(b.qvel[:, :6])) <= 0.5 and np.any(b.qvel[:, :6])
        np.testing.assert_array_equal(b.qvel[:, 6:], 0)
        np.testing.assert_array_equal(env.reset_attempts, 1)
        np.testing.assert_array_equal(env.reset_fallback, False)
        np.testing.assert_array_equal(
            env.history, np.repeat(env.state.obs["frame"][:, None], 30, axis=1)
        )
        sampled = copy.deepcopy(b.randomization)
        old_pose, old_history, old_noise = (
            b.qpos.copy(),
            env.history.copy(),
            env.measurement_noise.copy(),
        )
        b.delay_buffer[:] = 4
        env._reset_ids(np.array([2]))
        for key in sampled:
            np.testing.assert_array_equal(sampled[key], b.randomization[key])
        others = np.array([0, 1, 3, 4, 5, 6])
        np.testing.assert_array_equal(b.qpos[others], old_pose[others])
        np.testing.assert_array_equal(env.history[others], old_history[others])
        np.testing.assert_array_equal(env.measurement_noise[others], old_noise[others])
        np.testing.assert_array_equal(b.delay_buffer[2], 0)
        np.testing.assert_array_equal(b.delay_buffer[others], 4)
        assert not np.array_equal(b.qpos[2], old_pose[2])
    finally:
        env.close()


def test_signed_cached_geometry_matches_mujoco_at_random_pose_and_rotation():
    env = PE03GaitEnv(config(), evaluation=True)
    try:
        b, rng = env.backend, np.random.default_rng(15)
        qpos = np.tile(env.home, (128, 1))
        qpos[:, 7:] += rng.uniform(-0.1, 0.1, (128, 6))
        qpos[:, 3:7] = rng.normal(size=(128, 4))
        qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1)[:, None]
        actual = b.reset_geometry.heights(qpos)
        data = mujoco.MjData(b.model)
        for pose, heights in zip(qpos, actual, strict=True):
            data.qpos[:] = pose
            mujoco.mj_kinematics(b.model, data)
            expected = ground_clearances(b.model, data)
            np.testing.assert_allclose(
                heights, [expected[name] for name in env.config.env.foot_contact_geoms], atol=1e-12
            )
        b.align_reset_soles(qpos, 0.001)
        np.testing.assert_allclose(b.reset_geometry.heights(qpos).min(1), 0.001, atol=1e-14)
    finally:
        env.close()


def test_low_friction_reaches_actual_mujoco_contact():
    env = PE03GaitEnv(config("domain_rand.friction_range=[0.1,0.1]"))
    try:
        b = env.backend
        model = copy.copy(b.model)
        model.geom_friction[:] = b.randomization["geom_friction"][0]
        data = mujoco.MjData(model)
        data.qpos[:] = env.home
        data.qpos[2] -= 0.002
        mujoco.mj_forward(model, data)
        contacts = [
            c
            for c in data.contact
            if model.geom_bodyid[c.geom1] == 0 or model.geom_bodyid[c.geom2] == 0
        ]
        assert contacts
        np.testing.assert_allclose([c.friction[0] for c in contacts], 0.1)
    finally:
        env.close()


def test_one_measurement_per_instant_and_true_privileged_targets():
    env = PE03GaitEnv(config())
    try:
        rng = copy.deepcopy(env.rng.bit_generator.state)
        first, second = env._observe(), env._observe()
        assert rng == env.rng.bit_generator.state
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        measured = first["frame"][:, :18].copy()
        reward = env._rewards(env.backend.gait_foot_state(env.reference_points))[0]
        env.measurement_noise[:] = 0
        clean = env._observe()
        difference = measured - clean["frame"][:, :18]
        assert np.any(difference)
        assert np.all(np.abs(difference) <= env.noise_amplitude + 1e-6)
        np.testing.assert_array_equal(first["critic"], clean["critic"])
        np.testing.assert_array_equal(
            reward, env._rewards(env.backend.gait_foot_state(env.reference_points))[0]
        )
        np.testing.assert_allclose(
            clean["critic"][:, :3],
            inverse_rotate(env.backend.qpos[:, 3:7], env.backend.qvel[:, :3]),
            atol=1e-7,
        )
        feet = env.backend.gait_foot_state(env.reference_points)
        np.testing.assert_allclose(
            clean["critic"][:, 8:].reshape(7, 2, 3),
            inverse_rotate(env.backend.qpos[:, None, 3:7], feet["ground_force"])
            / env.weight[:, None, None],
            atol=1e-7,
        )
        env.set_command([0.2, 0.1, -0.3])
        changed = env._observe()
        np.testing.assert_array_equal(changed["frame"][:, :18], clean["frame"][:, :18])
        np.testing.assert_allclose(changed["actor"][:, -8:-5], [[0.2, 0.1, -0.3]] * 7)
    finally:
        env.close()


@pytest.mark.parametrize("delay", [0, 20, 25, 50])
def test_delay_applies_to_pd_while_history_records_issued_action(delay):
    env = PE03GaitEnv(config(f"play.delay_ms={delay}"), evaluation=True)
    try:
        action = np.zeros((7, 6))
        action[0] = 0.2
        for _ in range(int(delay // 20)):
            state = env.step(action)
            np.testing.assert_array_equal(env.backend.qpos[0], env.backend.qpos[1])
            np.testing.assert_allclose(state.obs["frame"][0, 18:24], 0.2)
        env.step(action)
        assert np.max(np.abs(env.backend.qpos[0] - env.backend.qpos[1])) > 1e-6
        assert env.backend.delay_steps[0] == delay / 2.5
    finally:
        env.close()


def test_push_single_physics_tick_and_snapshot_across_resets():
    env = PE03GaitEnv(config("domain_rand.push_interval_s=0.04", "env.episode_length_s=0.1"))
    try:
        env.backend.steps[:] = 7
        assert push_forces(env) is None  # 8..15, next trigger is 16.
        env.backend.steps[:] = 15
        forces = push_forces(env)
        assert np.count_nonzero(np.linalg.norm(forces, axis=2)) == 7
        np.testing.assert_array_equal(forces[:, 1:], 0)
        np.testing.assert_allclose(env.last_push_impulse, forces[:, 0] * env.backend.dt)
        env.step(np.zeros((7, 6)))
        snapshot = env.snapshot()
        actions = np.random.default_rng(2).normal(0, 0.1, (9, 7, 6))
        expected = [env.step(action) for action in actions]
        env.restore(snapshot)
        for action, target in zip(actions, expected, strict=True):
            actual = env.step(action)
            np.testing.assert_array_equal(actual.reward, target.reward)
            for key in actual.obs:
                np.testing.assert_array_equal(actual.obs[key], target.obs[key])
        assert env.reset_attempts.max() == 1
    finally:
        env.close()


@pytest.mark.skipif(not native_joint_position_pd_available(), reason="native PD not built")
def test_native_and_python_with_all_randomizations():
    envs = [
        PE03GaitEnv(
            config(
                f"training.native_pd={str(native).lower()}",
                "domain_rand.push_interval_s=0.04",
                "env.episode_length_s=0.1",
            )
        )
        for native in (False, True)
    ]
    try:
        for actions in np.random.default_rng(1).normal(0, 0.2, (16, 7, 6)):
            states = [env.step(actions) for env in envs]
            for key in states[0].obs:
                np.testing.assert_allclose(
                    states[0].obs[key], states[1].obs[key], atol=1e-7, rtol=1e-7
                )
            np.testing.assert_allclose(
                envs[0].backend.state, envs[1].backend.state, atol=1e-10, rtol=1e-10
            )
    finally:
        for env in envs:
            env.close()


def test_evaluation_conditions_are_explicit_and_protocol_changes_with_ranges():
    cfg = config()
    nominal = PE03GaitEnv(cfg, evaluation=True)
    robust = PE03GaitEnv(cfg, evaluation=True, robustness=True)
    repeat = PE03GaitEnv(cfg, evaluation=True, robustness=True)
    try:
        assert not nominal.backend.randomization
        assert robust.backend.randomization
        np.testing.assert_array_equal(nominal.backend.qvel, 0)
        np.testing.assert_array_equal(robust.state.obs["actor"], repeat.state.obs["actor"])
        original = evaluation_protocol(cfg)
        cfg.noise.level = 0.5
        assert evaluation_protocol(cfg)["fingerprint"] != original["fingerprint"]
    finally:
        for env in (nominal, robust, repeat):
            env.close()


def test_scheduled_evaluation_repeats_randomized_conditions():
    cfg = config(
        "training.evaluation_episodes=7",
        "env.episode_length_s=0.08",
        "domain_rand.push_interval_s=0.04",
    )
    policy = PE03EncoderPolicy(cfg)
    torch.set_num_threads(1)
    metrics, report = evaluate_gait(policy, cfg, "cpu")
    repeated, again = evaluate_gait(policy, cfg, "cpu")
    assert metrics == repeated and report == again
    assert metrics["evaluation/randomized_conditions"] == 1
    assert report["protocol"]["conditions"] == "randomized"
    np.testing.assert_array_equal(report["randomization"]["delay_ms"], 20.0)
    assert not report["acceptance"]["passed"]


def test_old_disabled_configuration_and_snapshot_remain_supported():
    cfg = config()
    del cfg.env.reset_randomization
    del cfg.training.evaluation_conditions
    cfg.noise = {"enabled": False}
    cfg.domain_rand = {"enabled": False, "push_enabled": False, "delay_ms_range": [0.0, 0.0]}
    env = PE03GaitEnv(cfg)
    try:
        np.testing.assert_array_equal(env.backend.qpos, np.tile(env.home, (7, 1)))
        np.testing.assert_array_equal(env.backend.qvel, 0)
        snapshot = env.snapshot()
        for key in (
            "weight",
            "push_mass",
            "imu_offset",
            "friction",
            "noise_amplitude",
            "measurement_noise",
            "push_count",
            "last_push_impulse",
            "reset_height_correction",
            "reset_attempts",
            "reset_fallback",
        ):
            snapshot["arrays"].pop(key)
        env.step(np.zeros((7, 6)))
        env.restore(snapshot)
        np.testing.assert_array_equal(env.backend.qpos, np.tile(env.home, (7, 1)))
        assert np.isfinite(env.step(np.zeros((7, 6))).obs["critic"]).all()
    finally:
        env.close()
