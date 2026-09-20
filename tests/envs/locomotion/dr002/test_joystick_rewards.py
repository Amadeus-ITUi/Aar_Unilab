from types import SimpleNamespace

import numpy as np

from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.dr002.joystick import DR002JoystickEnv


class _BodyPositionBackend:
    def __init__(self, wheel_pos_b: np.ndarray) -> None:
        self._wheel_pos_b = wheel_pos_b
        self.requested_sensor_names: tuple[str, ...] | None = None

    def get_sensor_data_batch(self, names: tuple[str, ...]) -> np.ndarray:
        self.requested_sensor_names = names
        return self._wheel_pos_b.reshape(self._wheel_pos_b.shape[0], -1)


def test_nominal_state_penalizes_wheel_fore_aft_misalignment() -> None:
    wheel_pos_b = np.asarray(
        [
            [[0.12, 0.20, -0.10], [0.12, -0.20, -0.10]],
            [[0.17, 0.25, -0.08], [0.11, -0.19, -0.13]],
        ],
        dtype=np.float32,
    )
    env = object.__new__(DR002JoystickEnv)
    env._backend = _BodyPositionBackend(wheel_pos_b)
    env._cfg = SimpleNamespace(ctrl_dt=0.02)
    env._reward_cfg = SimpleNamespace(scales={"nominal_state_lingzu": -3.0})
    ctx = RewardContext(
        info={},
        linvel=np.zeros((2, 3), dtype=np.float32),
        gyro=np.zeros((2, 3), dtype=np.float32),
        # Deliberately asymmetric joint angles: this reward now depends only
        # on the resulting wheel-axle geometry.
        dof_pos=np.asarray(
            [[0.8, -1.6, 0.0, 0.2, -2.0, 0.0], [0.1, -1.3, 0.0, 1.2, -2.2, 0.0]],
            dtype=np.float32,
        ),
        num_envs=2,
    )

    reward = env._reward_nominal_state_lingzu(ctx)

    assert env._backend.requested_sensor_names == (
        "left_wheel_axis_pos_b",
        "right_wheel_axis_pos_b",
    )
    np.testing.assert_allclose(reward, [0.0, 0.06**2], atol=1.0e-7)
