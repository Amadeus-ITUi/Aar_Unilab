from __future__ import annotations

from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from scripts.visualize_task_env import _build_env_cfg_override, _parse_args

from unilab.base import registry
from unilab.envs.locomotion.dr002.getup import (
    _RESET_BACK,
    _RESET_FRONT,
    _RESET_ORIGINAL,
    _STAGE_BACK,
    _STAGE_FRONT,
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


def test_we11_getup_is_registered_as_an_isolated_flat_compatible_task() -> None:
    ensure_registries()

    assert DR002JoystickGetupWE11Cfg.__bases__ == (DR002JoystickFlatWE11Cfg,)
    assert DR002JoystickGetupEnv.__bases__ == (DR002JoystickEnv,)
    assert registry.list_registered_envs()[TASK_NAME] == {
        "config_class": "DR002JoystickGetupWE11Cfg",
        "available_backends": ["mujoco"],
    }
    assert DR002JoystickGetupEnv._init_reward_functions is DR002JoystickEnv._init_reward_functions
    assert DR002JoystickGetupEnv._compute_terminated is DR002JoystickEnv._compute_terminated
    assert DR002JoystickGetupEnv._compute_truncated is DR002JoystickEnv._compute_truncated
    assert DR002JoystickGetupEnv._update_getup_success is DR002JoystickEnv._update_getup_success


def test_we11_getup_matches_flat_except_for_reset_contract() -> None:
    flat = OmegaConf.to_container(_compose_task(FLAT_SELECTOR), resolve=True)
    getup = OmegaConf.to_container(_compose_task(GETUP_SELECTOR), resolve=True)
    assert isinstance(flat, dict)
    assert isinstance(getup, dict)
    assert flat["training"]["task_name"] == "DR002JoystickFlatWE11"
    assert getup["training"]["task_name"] == TASK_NAME

    assert getup["algo"] == flat["algo"]
    getup_reward = dict(getup["reward"])
    flat_reward = dict(flat["reward"])
    termination_overrides = {
        "termination_contact_fail_steps": 50,
        "termination_gravity_z_threshold": 0.4,
        "termination_fail_time_s": 1.0,
    }
    for key, expected in termination_overrides.items():
        assert getup_reward.pop(key) == expected
        flat_reward.pop(key)
    assert getup_reward == flat_reward
    getup_env = dict(getup["env"])
    flat_env = dict(flat["env"])
    getup_reset = getup_env.pop("reset_pose")
    flat_env.pop("reset_pose")
    getup_curriculum = getup_env.pop("getup_curriculum")
    getup_domain_rand = dict(getup_env["domain_rand"])
    flat_domain_rand = dict(flat_env["domain_rand"])
    for reset_key in ("randomize_init_yaw", "init_xy_range", "init_qvel_range"):
        getup_domain_rand.pop(reset_key, None)
        flat_domain_rand.pop(reset_key, None)
    getup_env["domain_rand"] = getup_domain_rand
    flat_env["domain_rand"] = flat_domain_rand
    assert getup_env == flat_env
    assert getup_reset["mode"] == "getup"
    assert getup_reset["getup_probability"] == 1.0
    assert getup_reset["getup_termination_grace_seconds"] == 1.0
    assert getup_curriculum["initial_difficulty"] == 0.0
    assert getup_curriculum["front_stage_front_fraction"] == 0.40
    assert getup_curriculum["back_stage_back_fraction"] == 0.40
    assert getup_curriculum["mixed_home_fraction"] == 0.30
    assert getup_curriculum["mixed_front_fraction"] == 0.20
    assert getup_curriculum["mixed_back_fraction"] == 0.20
    assert getup["env"]["domain_rand"]["randomize_init_yaw"] is False
    assert getup["env"]["domain_rand"]["init_xy_range"] == [0.0, 0.0]
    assert getup["env"]["getup_curriculum"]["initial_difficulty"] == 0.0


def _make_getup_env(*, num_envs: int = 2, difficulty: float | None = None):
    overrides = [f"task={GETUP_SELECTOR}"]
    if difficulty is not None:
        overrides.append(f"env.getup_curriculum.forced_difficulty={difficulty}")
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=overrides)
    override = BackendAdapter(
        cfg, root_dir=REPO_ROOT, algo_name="ppo"
    ).build_task_env_cfg_override()
    return create_env(cfg, num_envs=num_envs, env_cfg_override=override)


def _current_qpos(env) -> np.ndarray:
    state = env.get_physics_state_snapshot()
    nq = int(env._backend.model.nq)
    return np.asarray(state[:, 1 : 1 + nq])


def test_getup_forced_curriculum_endpoints_match_xml_except_randomized_wings() -> None:
    for difficulty, keyframe in ((0.0, "home"), (1.0, "getup_start_v2")):
        env = _make_getup_env(difficulty=difficulty)
        try:
            state = env.init_state()
            expected = env._backend.get_keyframe_qpos(keyframe)
            actual = _current_qpos(env)
            non_wing = np.ones((expected.size,), dtype=np.bool_)
            non_wing[env._wing_qpos_indices] = False
            np.testing.assert_allclose(
                actual[:, non_wing],
                np.broadcast_to(expected[non_wing], (2, int(np.count_nonzero(non_wing)))),
                atol=1.0e-7,
            )
            lower = env._wing_joint_limits[:, 0]
            upper = env._wing_joint_limits[:, 1]
            assert np.all(actual[:, env._wing_qpos_indices] >= lower)
            assert np.all(actual[:, env._wing_qpos_indices] <= upper)
            np.testing.assert_array_equal(state.info["steps"], np.zeros(2, dtype=np.uint32))
            np.testing.assert_allclose(state.info["commands"], [[0.0, 0.0, 0.24]] * 2)
        finally:
            env.close()


def test_getup_midpoint_is_symmetric_and_inside_joint_limits() -> None:
    env = _make_getup_env(difficulty=0.5)
    try:
        env.init_state()
        qpos = _current_qpos(env)
        np.testing.assert_allclose(qpos[:, 7], qpos[:, 10])
        np.testing.assert_allclose(qpos[:, 8], qpos[:, 11])
        np.testing.assert_allclose(qpos[:, 9], qpos[:, 12])
        assert np.all((qpos[:, 7] >= -0.13) & (qpos[:, 7] <= 0.8))
        assert np.all((qpos[:, 8] >= -2.60) & (qpos[:, 8] <= -1.60))
        assert np.all((qpos[:, 2] > 0.1428) & (qpos[:, 2] < 0.3))
    finally:
        env.close()


def test_getup_wing_reset_is_independently_uniform_within_mechanical_limits() -> None:
    env = _make_getup_env(num_envs=1, difficulty=1.0)
    try:
        np.random.seed(23)
        values = env._sample_getup_wing_qpos(20000)

        lower = env._wing_joint_limits[:, 0]
        upper = env._wing_joint_limits[:, 1]
        assert np.all(values >= lower)
        assert np.all(values <= upper)
        np.testing.assert_allclose(np.mean(values, axis=0), (lower + upper) / 2.0, atol=0.015)
        assert np.any(np.abs(values[:, 0] - values[:, 1]) > 1.0e-3)
        np.testing.assert_allclose(upper, [0.0, 0.0], atol=1.0e-7)
        np.testing.assert_allclose(lower, [-1.5708, -1.5708], atol=1.0e-7)
    finally:
        env.close()


def test_getup_sampling_uses_frontier_replay_mixture_and_hard_anchors() -> None:
    env = _make_getup_env(num_envs=1)
    try:
        np.random.seed(7)
        env._getup_curriculum_difficulty = 0.5
        shared, frontier, exact = env._sample_episode_progress(20000)
        assert 0.78 < float(np.mean(frontier)) < 0.82
        assert np.all((shared >= 0.0) & (shared <= 0.5))
        assert np.all(shared[frontier] >= 0.45)
        assert not np.any(exact)

        env._getup_curriculum_difficulty = 1.0
        shared, frontier, exact = env._sample_episode_progress(20000)
        assert 0.09 < float(np.mean(exact)) < 0.11
        assert np.all(shared[exact] == 1.0)
        assert np.all(frontier[exact])
    finally:
        env.close()


def test_getup_staged_sampling_uses_configured_front_back_and_mixed_mixtures() -> None:
    env = _make_getup_env(num_envs=1)
    try:
        np.random.seed(17)
        env._getup_curriculum_difficulty = 1.0
        expected = {
            _STAGE_FRONT: [0.30, 0.20, 0.10, 0.40, 0.0],
            _STAGE_BACK: [0.30, 0.10, 0.10, 0.10, 0.40],
            _STAGE_MIXED: [0.30, 0.20, 0.10, 0.20, 0.20],
        }
        for stage, fractions in expected.items():
            env._getup_curriculum_stage = stage
            shared, frontier, exact_hard, category = env._sample_reset_plan(50_000)
            home = (category == _RESET_ORIGINAL) & (shared == 0.0)
            hard = (category == _RESET_ORIGINAL) & exact_hard
            interpolated = (category == _RESET_ORIGINAL) & ~(home | hard)
            actual = [
                np.mean(home),
                np.mean(hard),
                np.mean(interpolated),
                np.mean(category == _RESET_FRONT),
                np.mean(category == _RESET_BACK),
            ]
            np.testing.assert_allclose(actual, fractions, atol=0.01)
            assert not np.any(frontier)
    finally:
        env.close()


def test_getup_curriculum_state_round_trips() -> None:
    env = _make_getup_env(num_envs=1)
    restored = _make_getup_env(num_envs=1)
    try:
        env._getup_curriculum_difficulty = 0.35
        env._getup_curriculum_promotions = 7
        env._getup_curriculum_demotions = 2
        env._getup_curriculum_mastered = True
        env._getup_curriculum_stage = _STAGE_BACK
        env._getup_curriculum_window_completed = 123
        env._getup_curriculum_window_successes = 101
        env._family_episode_counts[:] = [456, 321]
        env._family_success_counts[:] = [400, 250]
        state = env.training_state_dict()
        restored.load_training_state_dict(state)
        assert restored._getup_curriculum_difficulty == 0.35
        assert restored._getup_curriculum_promotions == 7
        assert restored._getup_curriculum_demotions == 2
        assert restored._getup_curriculum_mastered is True
        assert restored._getup_curriculum_stage == _STAGE_BACK
        assert restored._getup_curriculum_window_completed == 123
        assert restored._getup_curriculum_window_successes == 101
        np.testing.assert_array_equal(restored._family_episode_counts, [456, 321])
        np.testing.assert_array_equal(restored._family_success_counts, [400, 250])
    finally:
        env.close()
        restored.close()


def test_getup_curriculum_promotes_and_demotes_on_frontier_window() -> None:
    env = _make_getup_env(num_envs=4)
    try:
        rows = np.arange(4, dtype=np.int32)
        env._cfg.getup_curriculum.window_episodes = 4
        env._episode_frontier[:] = True
        env._getup_succeeded[:] = True
        env._record_curriculum_outcomes(rows)
        assert env._getup_curriculum_difficulty == 0.05
        assert env._getup_curriculum_promotions == 1

        env._getup_curriculum_difficulty = 0.5
        env._getup_succeeded[:] = False
        env._record_curriculum_outcomes(rows)
        assert env._getup_curriculum_difficulty == 0.45
        assert env._getup_curriculum_demotions == 1
    finally:
        env.close()


def test_getup_curriculum_enters_mastered_mixture_after_hard_window() -> None:
    env = _make_getup_env(num_envs=4)
    try:
        rows = np.arange(4, dtype=np.int32)
        env._cfg.getup_curriculum.window_episodes = 4
        env._getup_curriculum_difficulty = 1.0
        env._episode_frontier[:] = True
        env._getup_succeeded[:] = [True, True, True, False]

        env._record_curriculum_outcomes(rows)

        assert env._getup_curriculum_mastered is False
        assert env._getup_curriculum_difficulty == 1.0

        env._episode_frontier[:] = True
        env._getup_succeeded[:] = [True, True, True, True]
        env._record_curriculum_outcomes(rows)

        assert env._getup_curriculum_mastered is True
        assert env._getup_curriculum_stage == _STAGE_FRONT
        assert env._getup_curriculum_difficulty == 1.0
        assert env._getup_curriculum_last_success_rate == 1.0
    finally:
        env.close()


def test_getup_front_and_back_stages_advance_on_their_own_success_windows() -> None:
    env = _make_getup_env(num_envs=4)
    try:
        rows = np.arange(4, dtype=np.int32)
        env._cfg.getup_curriculum.window_episodes = 4
        env._getup_curriculum_mastered = True
        env._getup_curriculum_stage = _STAGE_FRONT
        env._episode_reset_category[:] = _RESET_FRONT
        env._getup_succeeded[:] = [True, True, True, False]
        env._record_curriculum_outcomes(rows)
        assert env._getup_curriculum_stage == _STAGE_FRONT
        assert env._getup_curriculum_last_success_rate == 0.75

        env._getup_succeeded[:] = True
        env._record_curriculum_outcomes(rows)
        assert env._getup_curriculum_stage == _STAGE_BACK

        # Front replay in the back stage is logged, but cannot qualify back.
        env._episode_reset_category[:] = _RESET_FRONT
        env._record_curriculum_outcomes(rows)
        assert env._getup_curriculum_window_completed == 0
        env._episode_reset_category[:] = _RESET_BACK
        env._record_curriculum_outcomes(rows)
        assert env._getup_curriculum_stage == _STAGE_MIXED
        np.testing.assert_array_equal(env._family_episode_counts, [12, 4])
        np.testing.assert_array_equal(env._family_success_counts, [11, 4])
    finally:
        env.close()


def test_getup_back_pose_sampling_balances_thigh_groups() -> None:
    env = _make_getup_env(num_envs=1)
    try:
        np.random.seed(29)
        selected = env._sample_pose_bank_indices(_RESET_BACK, 40_000)
        counts = np.asarray(
            [np.count_nonzero(np.isin(selected, group)) for group in env._back_pose_groups]
        )
        np.testing.assert_allclose(counts / counts.sum(), np.full((4,), 0.25), atol=0.01)
    finally:
        env.close()


def test_getup_staged_reset_draws_legal_front_and_back_bank_poses() -> None:
    env = _make_getup_env(num_envs=1)
    try:
        np.random.seed(31)
        env._getup_curriculum_stage = _STAGE_MIXED
        qpos = env.sample_getup_reset_qpos(2000)
        category = env._pending_reset_category
        front = category == _RESET_FRONT
        back = category == _RESET_BACK
        assert np.count_nonzero(front) > 300
        assert np.count_nonzero(back) > 300
        np.testing.assert_allclose(qpos[front, 7], 1.57, atol=1.0e-7)
        np.testing.assert_allclose(qpos[front, 10], 1.57, atol=1.0e-7)
        assert np.all((qpos[back, 7] >= -0.13) & (qpos[back, 7] <= 0.60))
        np.testing.assert_allclose(qpos[back, 7], qpos[back, 10], atol=1.0e-7)
        lower = env._wing_joint_limits[:, 0]
        upper = env._wing_joint_limits[:, 1]
        assert np.all(qpos[:, env._wing_qpos_indices] >= lower)
        assert np.all(qpos[:, env._wing_qpos_indices] <= upper)
    finally:
        env.close()


def test_getup_reset_records_completed_episode_after_np_env_clears_steps() -> None:
    env = _make_getup_env(num_envs=1)
    try:
        state = env.init_state()
        state.info["steps"][0] = 0
        env._episode_alive_steps[0] = 12
        env._episode_initialized[0] = True
        env._episode_frontier[0] = True
        env._getup_succeeded[0] = True

        env.reset(np.asarray([0], dtype=np.int32))

        assert env._getup_curriculum_window_completed == 1
        assert env._getup_curriculum_window_successes == 1
    finally:
        env.close()


def test_we11_getup_constructs_resets_and_steps() -> None:
    cfg = _compose_task(GETUP_SELECTOR)
    override = BackendAdapter(
        cfg,
        root_dir=REPO_ROOT,
        algo_name="ppo",
    ).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=2, env_cfg_override=override)

    try:
        state = env.init_state()
        assert state.obs["obs"].shape == (2, 145)
        assert state.obs["critic"].shape == (2, 141)
        assert state.obs["privileged_target"].shape == (2, 3)

        reset_obs, _ = env.reset(np.asarray([0, 1], dtype=np.int32))
        assert all(np.isfinite(value).all() for value in reset_obs.values())

        state = env.step(np.zeros((2, 6), dtype=np.float32))
        assert all(np.isfinite(value).all() for value in state.obs.values())
        assert np.isfinite(state.reward).all()
    finally:
        env.close()


def test_getup_play_override_forces_exact_full_difficulty() -> None:
    cfg = _compose_task(GETUP_SELECTOR)
    cfg.training.play_only = True
    override = BackendAdapter(
        cfg, root_dir=REPO_ROOT, algo_name="ppo"
    ).build_play_env_cfg_override()
    assert override["reset_pose"]["mode"] == "getup"
    assert override["getup_curriculum"]["forced_difficulty"] == 1.0


def test_getup_visualizer_uses_flat_behavior_with_deterministic_reset_placement() -> None:
    override = _build_env_cfg_override(TASK_NAME, getup_difficulty=0.0)
    assert override["domain_rand"]["init_xy_range"] == [0.0, 0.0]
    assert override["domain_rand"]["randomize_init_yaw"] is False
    assert override["commands"]["lin_vel_x"] == [-0.5, 0.5]
    assert override["commands"]["ang_vel_z"] == [-1.0, 1.0]
    assert override["wing_velocity_cmd"]["enabled"] is True
    assert override["noise_config"]["level"] == 1.0
    assert override["getup_curriculum"]["forced_difficulty"] == 0.0

    args = _parse_args(
        [
            "--task",
            TASK_NAME,
            "--getup-difficulty",
            "1.0",
            "--freeze-initial-pose",
        ]
    )
    assert args.freeze_initial_pose is True
