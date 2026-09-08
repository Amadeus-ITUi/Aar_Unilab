from __future__ import annotations

from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from scripts.visualize_task_env import _build_env_cfg_override, _parse_args

from unilab.base import registry
from unilab.envs.locomotion.dr002.getup import (
    _RESET_BALANCE,
    _RESET_EXACT_GETUP,
    _RESET_HOME,
    _RESET_HOME_TO_GETUP,
    _STAGE_BALANCE,
    _STAGE_ENHANCE,
    _STAGE_EXACT_GETUP,
    _STAGE_GETUP_WITH_HOME,
    _STAGE_HOME_TO_GETUP,
    _STAGE_MIXED,
    DR002JoystickGetupEnv,
    DR002JoystickGetupWE11Cfg,
)
from unilab.envs.locomotion.dr002.joystick import (
    DR002JoystickEnv,
    DR002JoystickFlatWE11Cfg,
)
from unilab.training import BackendAdapter, create_env, ensure_registries

REPO_ROOT = Path(__file__).parents[4]
CONFIG_DIR = REPO_ROOT / "conf/ppo"
FLAT_SELECTOR = "dr002_joystick_flat_we11/mujoco"
GETUP_SELECTOR = "dr002_joystick_getup_we11/mujoco"
TASK_NAME = "DR002JoystickGetupWE11"


def _compose_task(selector: str) -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=[f"task={selector}"])


def _make_getup_env(
    *,
    num_envs: int = 2,
    difficulty: float | None = None,
    forced_stage: str | None = None,
):
    overrides = [f"task={GETUP_SELECTOR}"]
    if difficulty is not None:
        overrides.append(f"env.getup_curriculum.forced_difficulty={difficulty}")
    if forced_stage is not None:
        overrides.append(f"env.getup_curriculum.forced_stage={forced_stage}")
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=overrides)
    override = BackendAdapter(
        cfg, root_dir=REPO_ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()
    return create_env(cfg, num_envs=num_envs, env_cfg_override=override)


def _current_qpos(env) -> np.ndarray:
    snapshot = env.get_physics_state_snapshot()
    nq = int(env._backend.model.nq)
    return np.asarray(snapshot[:, 1 : 1 + nq])


def test_getup_registration_and_configuration_contract() -> None:
    ensure_registries()
    assert DR002JoystickGetupWE11Cfg.__bases__ == (DR002JoystickFlatWE11Cfg,)
    assert DR002JoystickGetupEnv.__bases__ == (DR002JoystickEnv,)
    assert registry.list_registered_envs()[TASK_NAME]["available_backends"] == ["mujoco"]

    cfg = _compose_task(GETUP_SELECTOR)
    assert cfg.env.commands.lin_vel_x == [-1.0, 1.0]
    assert cfg.env.commands.ang_vel_z == [-1.0, 1.0]
    assert cfg.env.commands.rel_standing_envs == 0.30
    assert cfg.env.commands.startup_stand_seconds == 0.0
    assert cfg.env.commands.standing_envs_episode_persistent is True
    assert cfg.env.commands.curriculum is False
    assert cfg.reward.getup_success_gravity_z_threshold == 0.90
    assert cfg.reward.getup_success_base_height_range[0] == 0.20
    assert np.isinf(cfg.reward.getup_success_base_height_range[1])
    assert cfg.reward.getup_tracking_gate_min_height == 0.14285
    assert cfg.reward.getup_tracking_gate_full_height == 0.20
    assert cfg.reward.getup_lin_vel_z_min_multiplier == 0.10
    assert cfg.reward.getup_leg_regularization_min_multiplier == 0.20
    assert cfg.reward.scales.getup_success_bonus == 5.0
    assert cfg.reward.track_ang_vel_z_term_clip == 2.0
    assert cfg.reward.orientation_term_clip == 3.0
    assert cfg.reward.scales.zero_cmd_stationary_vx == -4
    assert cfg.reward.scales.zero_cmd_stationary_yaw == -12
    assert cfg.reward.zero_cmd_stationary_vx_term_clip == 2.0
    assert cfg.reward.zero_cmd_stationary_yaw_term_clip == 2.0


def test_getup_keeps_domain_randomization_unchanged() -> None:
    cfg = _compose_task(GETUP_SELECTOR)
    assert cfg.env.domain_rand.randomize_base_mass is True
    assert cfg.env.domain_rand.body_mass_multiplier_range == [0.9, 1.2]
    assert cfg.env.domain_rand.com_offset_x == [-0.04, 0.04]
    assert cfg.env.domain_rand.ground_friction_multiplier_range == [0.8, 1.2]
    assert cfg.env.domain_rand.kp_multiplier_range == [0.8, 1.2]
    assert cfg.env.domain_rand.kd_multiplier_range == [0.8, 1.2]
    assert cfg.env.domain_rand.push_force_limit == [10.0, 10.0, 0.0]


def test_home_to_getup_endpoints_match_xml_except_randomized_wings() -> None:
    for difficulty, keyframe in ((0.0, "home"), (1.0, "getup_start_v2")):
        env = _make_getup_env(num_envs=8, difficulty=difficulty)
        try:
            env.init_state()
            actual = _current_qpos(env)
            expected = env._backend.get_keyframe_qpos(keyframe)
            non_wing = np.setdiff1d(np.arange(actual.shape[1]), env._wing_qpos_indices)
            np.testing.assert_allclose(
                actual[:, non_wing],
                np.broadcast_to(expected[non_wing], actual[:, non_wing].shape),
                atol=1.0e-7,
            )
            assert np.all(env._episode_reset_category == _RESET_HOME_TO_GETUP)
        finally:
            env.close()


def test_balance_reset_uses_home_with_bounded_pitch_and_rate() -> None:
    env = _make_getup_env(num_envs=100, difficulty=1.0, forced_stage="balance")
    try:
        env.init_state()
        home = env._backend.get_keyframe_qpos("home")
        leg_qpos = _current_qpos(env)[:, [7, 8, 10, 11]]
        np.testing.assert_allclose(
            leg_qpos,
            np.broadcast_to(home[[7, 8, 10, 11]], leg_qpos.shape),
        )
        assert np.count_nonzero(env._episode_balance_direction == -1) == 50
        assert np.count_nonzero(env._episode_balance_direction == 1) == 50
        assert np.max(np.abs(np.rad2deg(env._episode_balance_pitch_rad))) <= 16.25
        assert np.max(np.abs(env._episode_balance_pitch_rate)) <= 0.13
        assert np.all(env._episode_reset_category == _RESET_BALANCE)
    finally:
        env.close()


def test_command_sampling_is_thirty_percent_standing_and_deployment_vx_range() -> None:
    env = _make_getup_env(num_envs=20_000)
    try:
        env.init_state()
        standing = env._episode_standing_mask
        commands = np.asarray(env.state.info["commands"])
        assert 0.29 < float(np.mean(standing)) < 0.31
        np.testing.assert_allclose(commands[standing, :2], 0.0)
        moving = commands[~standing]
        assert np.all((-1.0 <= moving[:, 0]) & (moving[:, 0] <= 1.0))
        assert np.all((-1.0 <= moving[:, 1]) & (moving[:, 1] <= 1.0))
        assert np.std(moving[:, 0]) > 0.50
        assert np.std(moving[:, 1]) > 0.50
    finally:
        env.close()


def test_enhance_starts_stationary_then_resamples_hardware_safe_commands() -> None:
    env = _make_getup_env(num_envs=2_000, difficulty=1.0, forced_stage="enhance")
    try:
        state = env.init_state()
        assert env._getup_curriculum_stage == _STAGE_ENHANCE
        np.testing.assert_allclose(state.info["commands"][:, :2], 0.0)

        startup_steps = int(
            round(env._cfg.getup_curriculum.enhance_startup_stand_seconds / env._cfg.ctrl_dt)
        )
        state.info["steps"].fill(startup_steps)
        env._update_commands(state.info)
        commands = np.asarray(state.info["commands"])
        moving = commands[~env._episode_standing_mask]
        assert moving.size > 0
        assert np.all((-1.0 <= moving[:, 0]) & (moving[:, 0] <= 1.0))
        assert np.std(moving[:, 0]) > 0.50
    finally:
        env.close()


def test_enhance_push_sampling_is_bounded_and_multi_step() -> None:
    env = _make_getup_env(num_envs=1)
    try:
        env._getup_curriculum_stage = _STAGE_ENHANCE
        scales = env.sample_interval_push_force_scale(10_000)
        durations = env.sample_interval_push_duration_steps(10_000)
        assert np.all((0.5 <= scales) & (scales <= 1.0))
        assert np.all((2 <= durations) & (durations <= 5))
        assert np.any(durations > 1)
    finally:
        env.close()


def test_enhance_interval_push_persists_across_control_steps() -> None:
    env = _make_getup_env(num_envs=1, difficulty=1.0, forced_stage="enhance")
    try:
        state = env.init_state()
        env._episode_enhance_push_mask[:] = True
        env._cfg.domain_rand.push_interval = 100
        env._cfg.domain_rand.push_randomize_within_interval = False
        startup = env.interval_push_startup_steps()
        provider = env._dr_manager._provider

        state.info["steps"][:] = startup + 100
        first = provider.build_interval_randomization_plan(env, step_counter=0)
        assert first is not None and first.body_force is not None
        assert np.linalg.norm(first.body_force[0, 0, :3]) > 0.0

        state.info["steps"][:] = startup + 101
        second = provider.build_interval_randomization_plan(env, step_counter=1)
        assert second is not None and second.body_force is not None
        np.testing.assert_allclose(second.body_force, first.body_force)
    finally:
        env.close()


def test_stage_reset_mixtures_exclude_front_and_back() -> None:
    env = _make_getup_env(num_envs=1)
    try:
        expected = {
            _STAGE_EXACT_GETUP: [0.20, 0.80, 0.0, 0.0],
            _STAGE_GETUP_WITH_HOME: [0.10, 0.60, 0.30, 0.0],
            _STAGE_BALANCE: [0.10, 0.40, 0.20, 0.30],
            _STAGE_MIXED: [0.10, 0.50, 0.20, 0.20],
            _STAGE_ENHANCE: [0.10, 0.20, 0.30, 0.40],
        }
        for stage, fractions in expected.items():
            env._getup_curriculum_stage = stage
            _, frontier, _, category = env._sample_reset_plan(50_000)
            actual = [np.mean(category == family) for family in range(4)]
            np.testing.assert_allclose(actual, fractions, atol=0.01)
            assert not np.any(frontier)
            assert np.all((category >= _RESET_HOME_TO_GETUP) & (category <= _RESET_BALANCE))
        assert env._front_pose_indices.size > 0
        assert env._back_pose_indices.size > 0
    finally:
        env.close()


def test_curriculum_stage_progression_and_no_fixed_stage_demotion() -> None:
    env = _make_getup_env(num_envs=4)
    try:
        env.init_state()
        c = env._cfg.getup_curriculum
        c.window_episodes = 4
        rows = np.arange(4, dtype=np.int32)
        env._episode_frontier[:] = True
        env._episode_reset_category[:] = _RESET_HOME_TO_GETUP
        env._getup_curriculum_difficulty = 1.0
        env._getup_succeeded[:] = True
        env._record_curriculum_outcomes(rows)
        assert env._getup_curriculum_stage == _STAGE_EXACT_GETUP

        env._episode_reset_category[:] = _RESET_EXACT_GETUP
        env._getup_succeeded[:] = [True, True, True, True]
        env._record_curriculum_outcomes(rows)
        assert env._getup_curriculum_stage == _STAGE_GETUP_WITH_HOME

        env._episode_reset_category[:] = [_RESET_EXACT_GETUP] * 2 + [_RESET_HOME] * 2
        env._getup_succeeded[:] = True
        env._record_curriculum_outcomes(rows)
        assert env._getup_curriculum_stage == _STAGE_BALANCE

        env._episode_reset_category[:] = _RESET_BALANCE
        env._episode_balance_direction[:] = [-1, -1, 1, 1]
        env._getup_success_hold_steps[:] = 50
        env._record_curriculum_outcomes(rows)
        assert env._getup_curriculum_stage == _STAGE_MIXED

        env._episode_reset_category[:] = [
            _RESET_EXACT_GETUP,
            _RESET_HOME,
            _RESET_BALANCE,
            _RESET_HOME_TO_GETUP,
        ]
        env._getup_success_hold_steps[:] = 50
        env._record_curriculum_outcomes(rows)
        assert env._getup_curriculum_stage == _STAGE_ENHANCE
        assert env._getup_curriculum_mastered
    finally:
        env.close()


def test_home_to_getup_does_not_credit_old_batch_to_new_difficulty() -> None:
    env = _make_getup_env(num_envs=8)
    try:
        env.init_state()
        env._cfg.getup_curriculum.window_episodes = 4
        rows = np.arange(8, dtype=np.int32)
        env._episode_frontier[:] = True
        env._episode_reset_category[:] = _RESET_HOME_TO_GETUP
        env._getup_succeeded[:] = True
        env._record_curriculum_outcomes(rows)
        assert env._getup_curriculum_difficulty == 0.05
        assert env._getup_curriculum_window_completed == 0
    finally:
        env.close()


def test_unified_success_ignores_tracking_and_uses_physical_state_only() -> None:
    env = _make_getup_env(num_envs=5)
    try:
        env.init_state()
        env._getup_current_commands[:] = [[0.5, 1.0, 0.24]] * 5
        gravity = np.zeros((5, 3), dtype=np.float32)
        gravity[:, 2] = [0.91, 0.90, 0.91, 0.91, 0.91]
        height = np.asarray([0.21, 0.21, 0.20, 0.21, 0.21], dtype=np.float32)
        linvel = np.full((5, 3), 100.0, dtype=np.float32)
        gyro = np.full((5, 3), -100.0, dtype=np.float32)
        result = env._getup_success_kinematics_clear(gravity, height, linvel, gyro)
        np.testing.assert_array_equal(result, [True, False, False, True, True])
    finally:
        env.close()


def test_success_truncation_and_balance_time_limit_are_stage_specific() -> None:
    env = _make_getup_env(num_envs=2)
    try:
        state = env.init_state()
        env._getup_succeeded[:] = [True, False]
        for stage in (_STAGE_HOME_TO_GETUP, _STAGE_EXACT_GETUP, _STAGE_GETUP_WITH_HOME):
            env._getup_curriculum_stage = stage
            np.testing.assert_array_equal(env._compute_truncated(state), [True, False])

        env._getup_curriculum_stage = _STAGE_BALANCE
        state.info["steps"][:] = [249, 250]
        np.testing.assert_array_equal(env._compute_truncated(state), [False, True])
        env._getup_curriculum_stage = _STAGE_MIXED
        np.testing.assert_array_equal(env._compute_truncated(state), [False, False])
        env._getup_curriculum_stage = _STAGE_ENHANCE
        np.testing.assert_array_equal(env._compute_truncated(state), [False, False])
    finally:
        env.close()


def test_first_success_latches_but_live_hold_streak_can_fail_again() -> None:
    env = _make_getup_env(num_envs=1)
    try:
        env.init_state()
        env._episode_getup_mask[:] = True
        env._getup_current_commands[:] = 0.0
        env._reward_base_height_values = lambda _n: np.asarray([0.21], dtype=np.float32)
        env._undesired_contact_values = lambda _n: np.zeros((1, 1), dtype=np.float32)
        env.get_local_linvel = lambda: np.zeros((1, 3), dtype=np.float32)
        env.get_gyro = lambda: np.zeros((1, 3), dtype=np.float32)
        gravity = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)

        for step in range(50):
            env._state.info["steps"][:] = step
            env._update_getup_success(gravity)
        assert env._getup_succeeded[0]
        assert env._getup_just_succeeded[0]
        assert env._getup_success_hold_steps[0] == 50
        assert env._getup_success_time_sum_s == 1.0

        env._update_getup_success(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32))
        assert env._getup_succeeded[0]
        assert not env._getup_just_succeeded[0]
        assert env._getup_success_hold_steps[0] == 0
    finally:
        env.close()


def test_reward_gate_and_dynamic_regularizers_follow_height_and_contact() -> None:
    env = _make_getup_env(num_envs=3)
    try:
        state = env.init_state()
        env._reward_base_height_values = lambda _n: np.asarray(
            [0.14285, (0.14285 + 0.20) / 2.0, 0.20], dtype=np.float32
        )
        contacts = np.zeros((3, 9), dtype=np.float32)
        contacts[2, 0] = 0.2
        env._undesired_contact_values = lambda _n: contacts
        linvel = np.zeros((3, 3), dtype=np.float32)
        gyro = np.zeros((3, 3), dtype=np.float32)
        dof_pos = env.get_dof_pos()
        dof_vel = env.get_dof_vel()
        env._compute_reward(state.info, linvel, gyro, np.zeros((3, 3)), dof_pos, dof_vel)
        np.testing.assert_allclose(env._getup_tracking_gate, [0.0, 0.5, 0.0], atol=1.0e-6)
        np.testing.assert_allclose(env._getup_lin_vel_z_multiplier, [0.1, 0.55, 0.1], atol=1.0e-6)
        np.testing.assert_allclose(
            env._getup_leg_regularization_multiplier, [0.2, 0.6, 0.2], atol=1.0e-6
        )
    finally:
        env.close()


def test_curriculum_state_v6_round_trip_and_old_layout_restart() -> None:
    env = _make_getup_env(num_envs=1)
    restored = _make_getup_env(num_envs=1)
    old = _make_getup_env(num_envs=1)
    try:
        env._getup_curriculum_stage = _STAGE_BALANCE
        env._getup_curriculum_difficulty = 1.0
        env._family_episode_counts[:] = [1, 2, 3, 4]
        env._balance_window_completed[:] = [10, 11]
        env._post_success_evaluation_count = 7
        env._success_then_failure_count = 2
        state = env.training_state_dict()
        restored.load_training_state_dict(state)
        assert restored._getup_curriculum_stage == _STAGE_BALANCE
        np.testing.assert_array_equal(restored._family_episode_counts, [1, 2, 3, 4])
        np.testing.assert_array_equal(restored._balance_window_completed, [10, 11])
        assert restored._post_success_evaluation_count == 7
        assert restored._success_then_failure_count == 2

        state["getup_pose_curriculum"]["layout_version"] = 5
        state["getup_pose_curriculum"]["stage"] = _STAGE_MIXED
        restored.load_training_state_dict(state)
        assert restored._getup_curriculum_stage == _STAGE_MIXED

        state["getup_pose_curriculum"]["layout_version"] = 4
        state["getup_pose_curriculum"]["stage"] = 5
        old.load_training_state_dict(state)
        assert old._getup_curriculum_stage == _STAGE_HOME_TO_GETUP
        assert old._getup_curriculum_difficulty == 0.0
    finally:
        env.close()
        restored.close()
        old.close()


def test_getup_constructs_resets_and_steps_with_finite_values() -> None:
    env = _make_getup_env(num_envs=2)
    try:
        state = env.init_state()
        assert state.obs["obs"].shape == (2, 145)
        assert state.obs["critic"].shape == (2, 141)
        reset_obs, _ = env.reset(np.asarray([0, 1], dtype=np.int32))
        assert all(np.isfinite(value).all() for value in reset_obs.values())
        state = env.step(np.zeros((2, 6), dtype=np.float32))
        assert all(np.isfinite(value).all() for value in state.obs.values())
        assert np.isfinite(state.reward).all()
    finally:
        env.close()


def test_getup_play_and_visualizer_force_verified_pose_paths() -> None:
    cfg = _compose_task(GETUP_SELECTOR)
    cfg.training.play_only = True
    override = BackendAdapter(
        cfg, root_dir=REPO_ROOT, algo_name="ppo"
    ).build_play_env_cfg_override()
    assert override["getup_curriculum"]["forced_difficulty"] == 1.0
    assert override["getup_curriculum"]["forced_stage"] == "home_to_getup"

    viz = _build_env_cfg_override(TASK_NAME, getup_difficulty=0.0)
    assert viz["getup_curriculum"]["forced_stage"] == "home_to_getup"
    balance = _build_env_cfg_override(TASK_NAME, balance_difficulty=0.8)
    assert balance["getup_curriculum"]["forced_stage"] == "balance_recovery"
    args = _parse_args(["--task", TASK_NAME, "--balance-difficulty", "1.0"])
    assert args.balance_difficulty == 1.0


def test_flat_configuration_is_not_modified_by_getup_reward_fields() -> None:
    flat = OmegaConf.to_container(_compose_task(FLAT_SELECTOR), resolve=True)
    assert isinstance(flat, dict)
    assert flat["env"]["commands"]["startup_stand_seconds"] == 3.0
    assert flat["env"]["commands"]["rel_standing_envs"] == 0.2
    assert "getup_tracking_gate_min_height" not in flat["reward"]
    assert "getup_success_bonus" not in flat["reward"]["scales"]
