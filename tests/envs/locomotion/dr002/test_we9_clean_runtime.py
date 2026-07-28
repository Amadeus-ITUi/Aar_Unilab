from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from unilab.envs.locomotion.dr002.joystick import DR002JoystickFlatWE9Cfg

REPO_ROOT = Path(__file__).parents[4]
U9_ROOT = REPO_ROOT / "src/unilab/assets/robots/dr002/u9"
WRENCH_ROOT = (
    REPO_ROOT
    / "src/unilab/assets/robots/dr002/measured_wrench/sweep_20260728_u9_skin"
)


def _compose_we9():
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(REPO_ROOT / "conf/ppo"),
    ):
        return compose(
            config_name="config",
            overrides=["task=dr002_joystick_flat_we9/mujoco"],
        )


def test_we9_runtime_owner_is_plain_we9_and_u9_scene_compiles() -> None:
    owner = DR002JoystickFlatWE9Cfg()
    scene = U9_ROOT / "scene_flat_u9_pace.xml"

    assert Path(owner.scene.model_file) == scene
    assert owner.control_config.action_delay_semantics == "pre_controller_command_fifo"
    assert owner.control_config.torque_delay_steps == 0
    assert owner.control_config.action_delay_steps_by_joint is None
    assert owner.control_config.action_delay_min_steps == 2
    assert owner.control_config.action_delay_max_steps == 8
    assert owner.control_config.motor_control_hz == pytest.approx(200.0)
    assert owner.control_config.Kp == [4.11, 3.91, 0.0, 4.11, 3.91, 0.0]
    assert owner.control_config.Kd == [0.160, 0.193, 0.05, 0.160, 0.193, 0.05]

    model = mujoco.MjModel.from_xml_path(str(scene))
    for sensor_name in owner.sensor.undesired_contacts:
        assert (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name) >= 0
        ), sensor_name
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "push_site") >= 0


def test_we9_hydra_contract_has_no_torque_fifo_or_4hz_replay() -> None:
    cfg = _compose_we9()

    assert cfg.training.task_name == "DR002JoystickFlatWE9"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.env.sim_dt == pytest.approx(0.0025)
    assert cfg.env.ctrl_dt == pytest.approx(0.02)
    assert cfg.env.control_config.torque_delay_steps == 0
    assert "torque_delay_steps_by_joint" not in cfg.env.control_config
    assert list(cfg.env.control_config.Kp) == [4.11, 3.91, 0.0, 4.11, 3.91, 0.0]
    assert list(cfg.env.control_config.Kd) == [
        0.160,
        0.193,
        0.05,
        0.160,
        0.193,
        0.05,
    ]
    assert cfg.env.control_config.wheel_action_scale == pytest.approx(10.0)
    assert cfg.env.control_config.wheel_clip_actions == pytest.approx(2.5)

    assert list(cfg.env.domain_rand.csv_force_curriculum_hz) == [0, 1, 2, 3]
    assert [
        Path(path).name for path in cfg.env.domain_rand.csv_force_curriculum_paths
    ] == ["1hz.csv", "2hz.csv", "3hz.csv"]
    assert list(cfg.env.domain_rand.csv_force_amplitude_scale_range) == [0.5, 1.5]
    assert list(cfg.env.wing_angle_obs.csv_amplitude_scale_range) == [0.8, 1.2]
    assert cfg.env.wing_angle_obs.gaussian_noise_relative_std == pytest.approx(0.05)
    assert cfg.env.commands.curriculum_allow_demotion is False


@pytest.mark.parametrize("frequency_hz", (1, 2, 3))
def test_we9_wrench_csv_is_finite_and_exactly_closed(frequency_hz: int) -> None:
    samples = np.genfromtxt(
        WRENCH_ROOT / f"{frequency_hz}hz.csv",
        delimiter=",",
        names=True,
        dtype=np.float64,
    )
    columns = np.column_stack([samples[name] for name in samples.dtype.names or ()])

    assert columns.shape == (10001, 7)
    assert np.all(np.isfinite(columns))
    assert columns[0, 0] == pytest.approx(0.0)
    assert columns[-1, 0] == pytest.approx(10.0)
    assert np.all(np.diff(columns[:, 0]) > 0.0)
    np.testing.assert_array_equal(columns[-1, 1:], columns[0, 1:])
