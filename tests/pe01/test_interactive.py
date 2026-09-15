import numpy as np

from unilab.playback import joystick_command


def test_gamepad_deadzone_and_axis_mapping():
    assert np.array_equal(joystick_command(np.array([0.01, 0.01, 0.0, 0.01])), np.zeros(3))
    assert np.allclose(
        joystick_command(np.array([0.5, -0.25, 0.75, -0.5])),
        [0.25, -0.75, -0.5],
    )
