"""Numerical gait contracts, compatibility and live-only evaluation diagnostics."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from unilab.algos.torch.pe05.gait_metrics import GaitMetrics
from unilab.envs.locomotion.pe05.config import ROOT, load_config, validate_config
from unilab.envs.locomotion.pe05.contracts import (
    asset_fingerprint,
    observation_layout,
    resume_contract,
    validate_checkpoint,
)
from unilab.envs.locomotion.pe05.gait_rewards import desired_contacts, foot_costs
from unilab.envs.locomotion.pe05.vector_env import PE05VectorEnv


@pytest.fixture
def reward():
    return OmegaConf.to_container(load_config().reward, resolve=True)


def feet(n=1):
    return dict(
        ground_force=np.zeros((n, 2, 3)),
        reference_velocity=np.zeros((n, 2, 3)),
        clearance=np.zeros((n, 2)),
    )


def test_phase_periodicity_alternation_and_smooth_boundaries():
    phase = np.linspace(0, 1, 1000, endpoint=False)
    gaits = np.tile([2, 0.5, 0.5, 0.03], (len(phase), 1))
    d = desired_contacts(phase, gaits, 0.05)
    np.testing.assert_allclose(d.sum(1), 1, atol=1e-12)
    np.testing.assert_allclose(d[:, 0], np.roll(d[:, 1], 500), atol=1e-12)
    assert d[250, 0] > 0.99999 and d[250, 1] < 0.00001
    edge = desired_contacts(np.array([1 - 1e-7, 1e-7, 0.5 - 1e-7, 0.5 + 1e-7]), gaits[:4], 0.05)
    np.testing.assert_allclose(edge[0], edge[1], atol=2e-6)
    np.testing.assert_allclose(edge[2], edge[3], atol=2e-6)


def test_force_and_velocity_physical_denominators_and_phase_masks(reward):
    f = feet(4)
    f["ground_force"][:, 1, 2] = [0, 1, 2, 10]
    f["reference_velocity"][:, 0, 2] = [0, 0.1, 0.2, 0.5]
    costs = foot_costs(f, np.tile([1.0, 0.0], (4, 1)), reward)
    np.testing.assert_allclose(
        costs["tracking_contacts_shaped_force"][:, 1],
        0.5 * (1 - np.exp(-np.array([0, 1, 4, 100]) / 5)),
    )
    np.testing.assert_allclose(
        costs["tracking_contacts_shaped_vel"][:, 0],
        0.5 * (1 - np.exp(-np.array([0, 0.01, 0.04, 0.25]) / 0.2)),
    )
    assert (np.diff(costs["tracking_contacts_shaped_force"][:, 1]) > 0).all()
    assert (np.diff(costs["tracking_contacts_shaped_vel"][:, 0]) > 0).all()
    f["ground_force"][:, 0] = 100
    f["reference_velocity"][:, 1] = 100
    costs = foot_costs(f, np.tile([1.0, 0.0], (4, 1)), reward)
    np.testing.assert_allclose(costs["tracking_contacts_shaped_force"][:, 0], 0)
    np.testing.assert_allclose(costs["tracking_contacts_shaped_vel"][:, 1], 0)


def test_regulation_penetration_and_landing_boundaries(reward):
    f = feet(6)
    f["clearance"][:] = np.array([-0.001, 0, 0.00025, 0.01, 0.05, 0.049])[:, None]
    f["reference_velocity"][..., 0] = 0.2
    f["reference_velocity"][..., 2] = -0.3
    f["ground_force"][0, :, 2] = 0.1
    f["ground_force"][1, :, 2] = 0.100001
    f["reference_velocity"][2, :, 2] = 0
    f["reference_velocity"][3, :, 2] = 0.3
    costs = foot_costs(f, np.zeros((6, 2)), reward)
    np.testing.assert_allclose(costs["feet_regulation"][:3, 0], [0.04, 0.04, 0.04 / np.e])
    np.testing.assert_allclose(costs["foot_landing_vel"][:, 0], [0.09, 0, 0, 0, 0, 0.09])
    assert costs["feet_regulation"][3, 0] < 1e-15


def test_new_terms_clearance_cycle_and_single_dt(monkeypatch):
    cfg = load_config(["algo.num_envs=1000", "training.mujoco_threads=1"])
    # Use four real environments, sampling phase sequentially for the cycle integral.
    cfg.algo.num_envs = 4
    env = PE05VectorEnv(cfg, evaluation=True)
    try:
        f = feet(4)
        real = env.backend.gait_foot_state(np.array(env.cfg["env"]["reference_points"]))
        real.update(f)
        monkeypatch.setattr(env.backend, "gait_foot_state", lambda points: real)
        env.commands[:] = 0
        env.base_velocity[:] = 0
        env.backend.qvel[:, 3:6] = 0
        env.phase[:] = 0.25
        _, terms = env._rewards()
        assert len(terms) == 19
        assert "contact_schedule" not in terms and "feet_slip" not in terms
        np.testing.assert_allclose(terms["foot_clearance"], -0.0045)
        for error in (0.005, 0.01, 0.02):
            real["clearance"][:, 1] = 0.03 - error
            np.testing.assert_allclose(
                env._rewards()[1]["foot_clearance"], -0.1 * (error / 0.02) ** 2 * 0.02
            )
        real["clearance"][:] = 0
        cycle = []
        for phase in np.linspace(0, 1, 1000, endpoint=False).reshape(-1, 4):
            env.phase[:] = phase
            total, terms = env._rewards()
            np.testing.assert_allclose(total, sum(terms.values()), atol=1e-8)
            cycle.extend(terms["foot_clearance"])
        assert np.mean(cycle) == pytest.approx(-0.0016875)
        env.phase[:] = 0.25
        real["ground_force"][:] = [0, 0, 10]
        np.testing.assert_allclose(
            env._rewards()[1]["tracking_contacts_shaped_force"],
            -0.02 * (1 - np.exp(-20)),
            atol=1e-12,
        )
        # Negative totals remain signed, and term scaling is linear in dt.
        env.commands[:] = [2, 0, 2]
        total, terms = env._rewards()
        assert (total < 0).all()
        env.dt *= 2
        _, doubled = env._rewards()
        for name in ("tracking_contacts_shaped_force", "foot_clearance", "tracking_lin_vel"):
            np.testing.assert_allclose(doubled[name], terms[name] * 2)
    finally:
        env.close()


@pytest.mark.parametrize(
    "override",
    [
        "reward.gait_reward_version=unknown",
        "reward.gait_kappa=0",
        "reward.gait_force_sigma=-1",
        "reward.gait_vel_sigma=nan",
        "reward.feet_regulation_height_scale=0",
        "reward.about_landing_threshold=0",
        "reward.landing_force_threshold=0",
        "reward.scales.tracking_contacts_shaped_force=2",
        "reward.scales.tracking_contacts_shaped_vel=2",
        "reward.zero_command_stance=true",
    ],
)
def test_invalid_shaped_configuration(override):
    with pytest.raises(ValueError):
        load_config([override])


def test_saved_17_term_contract_is_preserved_and_resume_rejected():
    current = load_config()
    old = OmegaConf.merge(current)
    old.reward = OmegaConf.load(ROOT / "docs/assets/pe05_pe01_gait/before_task.yaml").reward
    validate_config(old)
    payload = dict(
        schema="pe05.training.v1",
        robot_id="pe05",
        asset_sha256=asset_fingerprint(old),
        observation_layout=observation_layout(),
        training_config=OmegaConf.to_container(old, resolve=True),
        training_contract=resume_contract(old),
    )
    saved = validate_checkpoint(payload, old)
    assert len(saved.reward.scales) == 17
    assert "gait_reward_version" not in saved.reward
    with pytest.raises(ValueError, match="start a new run"):
        validate_checkpoint(payload, current)
    payload.update(
        training_config=OmegaConf.to_container(current, resolve=True),
        training_contract=resume_contract(current),
    )
    validate_checkpoint(payload, current)
    with pytest.raises(ValueError, match="start a new run"):
        validate_checkpoint(payload, old)


def test_gait_metrics_ignore_dead_environments_and_count_liftoffs():
    collector = GaitMetrics(0.02)
    behavior = dict(
        foot_contact=np.ones((2, 2), bool),
        swing=np.array([[0, 1], [0, 1]], bool),
        foot_clearance=np.zeros((2, 2)),
        target_clearance=np.array([[0, 0.03], [0, 0.03]]),
        foot_reference_velocity=np.zeros((2, 2, 3)),
        reward_terms={"action_rate": np.array([-0.1, -100.0])},
        foot_reward_terms={},
    )
    collector.update(behavior, np.array([True, False]))
    behavior["foot_contact"][0, 1] = False
    behavior["foot_clearance"][0, 1] = 0.03
    collector.update(behavior, np.array([True, False]))
    result = collector.result()
    assert result["reward/action_rate"] == pytest.approx(-0.1)
    assert result["gait/right/liftoffs_per_second"] == 25
    assert result["gait/right/mid_swing_drag_fraction"] == 0.5
    assert result["gait/left/liftoffs_per_second"] == 0


def test_migration_preserves_all_nonreward_task_settings():
    before = OmegaConf.load(ROOT / "docs/assets/pe05_pe01_gait/before_task.yaml")
    after = OmegaConf.load(ROOT / "conf/pe05/task/pe05_flat.yaml")
    del before.reward, after.reward
    assert OmegaConf.to_container(before, resolve=True) == OmegaConf.to_container(
        after, resolve=True
    )
