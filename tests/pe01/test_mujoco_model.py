import numpy as np

from unilab.envs.locomotion.pe01 import PE01Env


def test_pe01_model_reset_and_step_contract():
    env = PE01Env()
    reset = env.reset()
    assert env.model.nu == 6
    assert reset.actor.shape == (300,)
    assert reset.critic.shape == (33,)
    step, reward, terminated, info = env.step(np.zeros(6, dtype=np.float32))
    assert np.isclose(env.data.time, 1.0 / env.policy_hz)
    assert step.actor.shape == (300,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert np.isfinite(info["base_height"])
