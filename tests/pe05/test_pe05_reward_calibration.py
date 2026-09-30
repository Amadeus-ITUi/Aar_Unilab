"""Check physical reward sensitivities, not just YAML literals."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from unilab.envs.locomotion.pe05.config import load_config
from unilab.envs.locomotion.pe05.contracts import (
    asset_fingerprint,
    observation_layout,
    resume_contract,
    validate_checkpoint,
)
from unilab.envs.locomotion.pe05.vector_env import PE05VectorEnv


@pytest.fixture
def env():
    instance = PE05VectorEnv(
        load_config(["algo.num_envs=2", "training.mujoco_threads=1"]), evaluation=True
    )
    try:
        yield instance
    finally:
        instance.close()


@pytest.mark.parametrize("error", [0.001, 0.01, 0.02, 0.03])
def test_height_physical_coefficient_and_source_local_derivative(env, error):
    env.backend.qpos[:, 2] = env.height_target + error
    term = env._rewards()[1]["base_height"]
    np.testing.assert_allclose(term, -150 * error**2 * env.dt, atol=1e-12)
    # PE03 P0 * exp(-dt * 100 * e^2 / sigma_negative).
    eps = 1e-5
    source_loss = 0.03 * np.expm1(-env.dt * 100 * eps**2 / 0.02)
    env.backend.qpos[:, 2] = env.height_target + eps
    np.testing.assert_allclose(env._rewards()[1]["base_height"], source_loss, rtol=1e-6)


def test_tracking_curvature_zero_crossing_and_signed_tail(env):
    env.base_velocity[:] = 0
    env.backend.qvel[:, 3:6] = 0
    for error in (1e-5, 0.5, 1.0):
        env.commands[:] = [error, 0, error]
        total, terms = env._rewards()
        for key, peak in (("tracking_lin_vel", 1), ("tracking_ang_vel", 0.5)):
            np.testing.assert_allclose(terms[key], peak * (1 - error**2 / 0.25) * env.dt)
            if error < 1e-4:
                source_loss = peak * env.dt * np.expm1(-(error**2) / 0.25)
                np.testing.assert_allclose(terms[key] - peak * env.dt, source_loss, rtol=1e-5)
        np.testing.assert_allclose(total, sum(terms.values()), atol=1e-8)
    assert (terms["tracking_lin_vel"] < 0).all()


def test_collision_is_additive_and_failure_only_once(env):
    env.backend.contact_history[:] = 0
    env.backend.ground_contact_history[:] = 0
    base, calf = env.penalized[0], env.cfg["env"]["body_names"].index("L_calf_Link")
    env.backend.contact_history[:, 0, base, 2] = 2
    env.backend.contact_history[:, 4, calf, 2] = 2
    reward, terms = env._rewards()
    np.testing.assert_allclose(terms["collision"], -0.2)
    final, terms = env.adaptation.apply_failure_cost(reward, terms, np.array([True, False]))
    np.testing.assert_allclose(terms["termination"], [-2, 0])
    np.testing.assert_allclose(final - reward, [-2, 0], atol=1e-7)
    # Timeout/nonfailure sample has no terminal cost.
    assert final[1] == reward[1]


def test_home_is_inside_calibrated_soft_bounds(env):
    assert (env.home[7:] >= env.soft_limits[:, 0]).all()
    assert (env.home[7:] <= env.soft_limits[:, 1]).all()
    width = np.diff(env.backend.joint_range, axis=1)[:, 0]
    np.testing.assert_allclose(env.soft_limits[:, 0], env.backend.joint_range[:, 0] + 0.05 * width)


def test_saved_old_reward_config_loads_but_cannot_resume_new_objective():
    current = load_config()
    old = OmegaConf.merge(
        current,
        {"reward": {"tracking_velocity_std": 0.4, "scales": {"base_height": -1.0}}},
    )
    payload = dict(
        schema="pe05.training.v1",
        robot_id="pe05",
        asset_sha256=asset_fingerprint(old),
        observation_layout=observation_layout(),
        training_config=OmegaConf.to_container(old, resolve=True),
        training_contract=resume_contract(old),
    )
    saved = validate_checkpoint(payload)
    assert saved.reward.scales.base_height == -1
    assert saved.reward.tracking_velocity_std == 0.4
    with pytest.raises(ValueError, match="contract"):
        validate_checkpoint(payload, current)
