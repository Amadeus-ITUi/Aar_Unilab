"""The experimental actuator bundle must reach both integrators unchanged."""

import ctypes

import mujoco
import numpy as np
import pytest

from unilab.base.backend.mujoco.actuator_parameters import (
    _constant_refresh_library,
    apply_actuator_parameters,
)
from unilab.base.backend.mujoco.native_batch import native_identified_pd_available
from unilab.envs.locomotion.pe03.config import load_config
from unilab.envs.locomotion.pe03.gait_env import PE03GaitEnv


def config(native=False, *overrides):
    return load_config(
        [
            "+experiment=gait_identified",
            "algo.num_envs=3",
            "training.mujoco_threads=2",
            f"training.native_pd={str(native).lower()}",
            *overrides,
        ]
    )


def test_actuator_refresh_preserves_timer_and_matches_recompiled_constants():
    # The public getter cannot detect the native timer installed by MjData.
    timer = ctypes.c_void_p.in_dll(_constant_refresh_library(), "mjcb_time")
    saved_pointer = timer.value
    settings = dict(
        armature=[0.017],
        damping=[0.003],
        frictionloss=[0.15],
        delay_ms=[20],
        saturation="sum_abs_pd",
    )
    xml = '<mujoco><worldbody><body><joint name="j" {}/><geom size=".1" mass="1"/></body></worldbody></mujoco>'
    expected = mujoco.MjModel.from_xml_string(
        xml.format('armature=".017" damping=".003" frictionloss=".15"')
    )
    callback_type = ctypes.CFUNCTYPE(ctypes.c_double)
    custom_timer = callback_type(lambda: 0.0)
    try:
        for pointer in (None, ctypes.cast(custom_timer, ctypes.c_void_p).value):
            timer.value = pointer
            model = mujoco.MjModel.from_xml_string(xml.format('armature=".01"'))
            apply_actuator_parameters(model, ["j"], settings)
            assert timer.value == pointer
            for name in (
                "dof_armature",
                "dof_damping",
                "dof_frictionloss",
                "dof_M0",
                "dof_invweight0",
                "body_invweight0",
            ):
                np.testing.assert_allclose(
                    getattr(model, name), getattr(expected, name), rtol=0, atol=1e-14
                )
    finally:
        timer.value = saved_pointer


@pytest.mark.skipif(not native_identified_pd_available(), reason="identified native PD not built")
def test_identified_native_matches_python_with_joint_delays_and_reset():
    envs = [PE03GaitEnv(config(native), robustness=False) for native in (False, True)]
    try:
        rng = np.random.default_rng(573)
        for env in envs:
            b = env.backend
            np.testing.assert_allclose(
                b.model.dof_armature[6:], env.config.control.actuator_model.armature
            )
            np.testing.assert_allclose(
                b.model.dof_damping[6:], env.config.control.actuator_model.damping
            )
            np.testing.assert_allclose(
                b.model.dof_frictionloss[6:], env.config.control.actuator_model.frictionloss
            )
            expected = np.tile(
                np.rint(np.array(env.config.control.actuator_model.delay_ms) / 2.5), (3, 1)
            )
            np.testing.assert_array_equal(b.delay_steps, expected)
            assert not b.position_difference
            # Exercise both joint-specific and environment-specific differences,
            # independent of which measured delay model is selected for training.
            b.delay_steps[:] = [[0, 1, 4, 2, 3, 5], [4, 3, 2, 1, 0, 5], [2] * 6]
        for step in range(45):
            actions = rng.normal(0, 0.8, (3, 6))
            for env in envs:
                b = env.backend
                if step == 18:
                    b.reset(np.array([1]), b.home[None].copy(), np.zeros((1, 12)))
                if step == 25:
                    b.restore(b.snapshot())
                b.step(actions, 8)
            for name in ("state", "sensors", "torque", "joint_velocity", "delay_buffer"):
                np.testing.assert_allclose(
                    getattr(envs[0].backend, name),
                    getattr(envs[1].backend, name),
                    atol=1e-9,
                    rtol=1e-9,
                )
    finally:
        for env in envs:
            env.close()


@pytest.mark.parametrize("native", [False, True])
def test_opposed_pd_terms_are_scaled_before_summing(native):
    if native and not native_identified_pd_available():
        pytest.skip("identified native PD not built")
    env = PE03GaitEnv(config(native), robustness=False)
    try:
        b = env.backend
        qvel = np.zeros((3, 12))
        # P=10, D=-8: net clipping would incorrectly leave 2 Nm unchanged.
        qvel[:, 6:] = 8 / b.kd
        b.reset(np.arange(3), np.tile(b.home, (3, 1)), qvel)
        b.delay_steps[:] = 0
        actions = 10 / b.kp
        b.step(actions, 1)
        np.testing.assert_allclose(
            b.torque, np.tile(2 * np.minimum(1, b.torque_limits / 18), (3, 1)), atol=1e-12
        )
    finally:
        env.close()


@pytest.mark.parametrize("native", [False, True])
def test_shared_delay_randomization_reaches_both_integrators(native):
    if native and not native_identified_pd_available():
        pytest.skip("identified native PD not built")
    env = PE03GaitEnv(config(native, "algo.num_envs=32"), robustness=True)
    try:
        b = env.backend
        # Only one shared offset per environment; no six independent draws,
        # no curriculum attenuation and no extra copy of the old fixed delay.
        steps = b.delay_steps.copy()
        np.testing.assert_array_equal(steps, np.repeat(steps[:, :1], 6, axis=1))
        assert set(np.unique(steps)) == {8, 9, 10, 11, 12}
        assert b.delay_buffer.shape[1] == 13
        b.step(np.zeros((32, 6)), 8)
        assert np.isfinite(b.state).all()
        env.reset()
        np.testing.assert_array_equal(b.delay_steps, steps)
        np.testing.assert_array_equal(b.delay_buffer, 0)
    finally:
        env.close()


@pytest.mark.parametrize(
    "override",
    [
        "env.dof_vel_use_pos_diff=true",
        "control.actuator_model.delay_ms=[-1,0,0,0,0,0]",
        "control.actuator_model.armature=[0.01]",
    ],
)
def test_invalid_identified_contract_rejected(override):
    with pytest.raises(ValueError):
        config(False, override)
