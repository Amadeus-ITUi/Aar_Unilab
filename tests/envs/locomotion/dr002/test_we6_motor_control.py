from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from hydra import compose, initialize_config_dir

from unilab.base import registry
from unilab.envs.locomotion.dr002.joystick import (
    DR002JoystickEnv,
    DR002JoystickFlatWE6Cfg,
)


class _ChangingJointSensorBackend:
    """Expose a different joint state at every physics substep."""

    def __init__(self) -> None:
        self.physics_substep = 0
        self.position_reads: list[int] = []
        self.velocity_reads: list[int] = []

    def get_sensor_data_batch(self, names: tuple[str, ...]) -> np.ndarray:
        if names[0].endswith("_pos"):
            self.position_reads.append(self.physics_substep)
            value = float(self.physics_substep)
        else:
            self.velocity_reads.append(self.physics_substep)
            value = 2.0 * float(self.physics_substep)
        return np.full((1, 6), value, dtype=np.float64)


def test_we6_clock_and_delay_semantics_contract() -> None:
    cfg = DR002JoystickFlatWE6Cfg()

    assert 1.0 / cfg.sim_dt == 400.0
    assert cfg.control_config.motor_control_hz == 400.0
    assert 1.0 / cfg.ctrl_dt == 50.0
    assert cfg.sim_substeps == 8
    assert cfg.critic_obs_mode == "isaaclab"
    assert (1.0 / cfg.sim_dt) / cfg.control_config.motor_control_hz == 1.0
    assert cfg.control_config.action_delay_semantics == "pre_controller_command_fifo"
    assert cfg.control_config.torque_delay_steps == 0
    assert cfg.control_config.action_delay_steps_by_joint == [14, 6, 10, 14, 6, 10]
    assert cfg.control_config.action_delay_min_steps == 6
    assert cfg.control_config.action_delay_max_steps == 14
    assert cfg.control_config.Kp == [3.75, 4.04, 0.0, 3.75, 4.04, 0.0]
    assert cfg.control_config.Kd == [0.145, 0.2, 0.202, 0.145, 0.2, 0.202]
    assert cfg.control_config.use_native_batched_pd
    assert cfg.noise_config.scale_gyro == 0.1
    assert cfg.noise_config.scale_joint_angle == 0.001
    assert cfg.noise_config.scale_joint_vel == 0.2
    assert cfg.noise_config.scale_wheel_vel == 0.3
    assert cfg.noise_config.scale_gravity == 0.0
    assert cfg.noise_config.gravity_noise_mode == "tilt"
    assert cfg.noise_config.gravity_installation_bias_max_deg == 5.0
    assert cfg.noise_config.gravity_dynamic_noise_max_deg == 1.0
    assert cfg.noise_config.gravity_dynamic_noise_time_constant_s == 3.0
    assert registry.list_registered_envs()["DR002JoystickFlatWE6"]["available_backends"] == [
        "mujoco"
    ]


def test_we6_registry_resolves_and_constructs_mujoco_model() -> None:
    assert registry.find_available_sim_backend("DR002JoystickFlatWE6") == "mujoco"
    env = registry.make(
        "DR002JoystickFlatWE6",
        sim_backend="mujoco",
        env_cfg_override={"reward_config": {"scales": {}}},
        num_envs=1,
    )
    try:
        assert isinstance(env, DR002JoystickEnv)
        assert env._motor_control_hz == 400.0
        assert env._motor_control_decimation == 1
    finally:
        env.close()


def test_we6_xml_contains_mirrored_identified_joint_dynamics() -> None:
    cfg = DR002JoystickFlatWE6Cfg()
    model = mujoco.MjModel.from_xml_path(cfg.scene.model_file)
    joint_names = (
        "left_thigh_joint",
        "left_calf_joint",
        "left_foot_joint",
        "right_thigh_joint",
        "right_calf_joint",
        "right_foot_joint",
    )
    dof_ids = np.asarray(
        [
            model.jnt_dofadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            ]
            for joint_name in joint_names
        ]
    )

    np.testing.assert_allclose(
        model.dof_armature[dof_ids],
        [
            0.0030211368456830893,
            0.020428934066981888,
            0.00034739571346796935,
            0.0030211368456830893,
            0.020428934066981888,
            0.00034739571346796935,
        ],
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        model.dof_damping[dof_ids],
        [0.0, 0.028598203906208486, 0.06467854502373717] * 2,
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        model.dof_frictionloss[dof_ids],
        [0.0, 0.09300014369144381, 0.0] * 2,
        rtol=0.0,
        atol=1.0e-15,
    )


def test_we6_hydra_config_preserves_the_identified_runtime_contract() -> None:
    config_dir = Path(__file__).parents[4] / "conf" / "ppo"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="config",
            overrides=["task=dr002_joystick_flat_we6/mujoco"],
        )

    assert cfg.training.task_name == "DR002JoystickFlatWE6"
    assert cfg.env.sim_dt == 0.0025
    assert cfg.env.ctrl_dt == 0.02
    assert cfg.env.control_config.motor_control_hz == 400.0
    assert list(cfg.env.control_config.action_delay_steps_by_joint) == [
        14,
        6,
        10,
        14,
        6,
        10,
    ]
    assert list(cfg.env.control_config.Kp) == [3.75, 4.04, 0.0, 3.75, 4.04, 0.0]
    assert list(cfg.env.control_config.Kd) == [
        0.145,
        0.2,
        0.202,
        0.145,
        0.2,
        0.202,
    ]
    assert cfg.env.control_config.torque_delay_steps == 0
    assert cfg.env.control_config.use_native_batched_pd
    assert cfg.env.noise_config.scale_gyro == 0.1
    assert cfg.env.noise_config.scale_joint_angle == 0.001
    assert cfg.env.noise_config.scale_joint_vel == 0.2
    assert cfg.env.noise_config.scale_wheel_vel == 0.3
    assert cfg.env.noise_config.scale_gravity == 0.0
    assert cfg.env.noise_config.gravity_noise_mode == "tilt"
    assert cfg.env.noise_config.gravity_installation_bias_max_deg == 5.0
    assert cfg.env.noise_config.gravity_dynamic_noise_max_deg == 1.0
    assert cfg.env.noise_config.gravity_dynamic_noise_time_constant_s == 3.0


def test_per_joint_command_fifo_uses_motor_control_ticks() -> None:
    env = object.__new__(DR002JoystickEnv)
    env._action_delay_enabled = True
    env._action_delay_indices = np.asarray([[0, 1, 2, 3, 1, 0]], dtype=np.int32)
    neutral = np.arange(6, dtype=np.float64) - 20.0
    env._action_delay_buffer = np.broadcast_to(neutral, (1, 4, 6)).copy()

    commands = [np.arange(6, dtype=np.float64)[None, :] + 100.0 * tick for tick in range(5)]
    outputs = [env._delayed_policy_ctrl(command).copy() for command in commands]

    delays = env._action_delay_indices[0]
    for tick, output in enumerate(outputs):
        for joint, delay in enumerate(delays):
            expected = neutral[joint] if tick < delay else commands[tick - delay][0, joint]
            assert output[0, joint] == expected


def test_lower_rate_pd_updates_once_per_decimation_and_strictly_holds_torque() -> None:
    env = object.__new__(DR002JoystickEnv)
    env.default_angles = np.zeros(6, dtype=np.float64)
    env._motor_control_substep_index = 0
    env._motor_control_decimation = 2
    env._action_delay_enabled = False
    env._motor_kp = np.asarray([[1.0, 1.0, 0.0, 1.0, 1.0, 0.0]])
    env._motor_kd = np.asarray([[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]])
    env._ctrl_lower = np.full(6, -100.0)
    env._ctrl_upper = np.full(6, 100.0)
    env._last_motor_ctrl = np.zeros((1, 6), dtype=np.float64)
    env._motor_torque_scale = np.ones((1, 6), dtype=np.float64)

    backend = _ChangingJointSensorBackend()
    policy_ctrl = np.full((1, 6), 10.0, dtype=np.float64)
    torques: list[np.ndarray] = []
    for physics_substep in range(6):
        backend.physics_substep = physics_substep
        torques.append(env._pre_step_motor_control(backend, policy_ctrl).copy())

    assert backend.position_reads == [0, 2, 4]
    assert backend.velocity_reads == [0, 2, 4]
    for update_substep in (0, 2, 4):
        np.testing.assert_array_equal(torques[update_substep], torques[update_substep + 1])
    assert not np.array_equal(torques[1], torques[2])
    assert not np.array_equal(torques[3], torques[4])
