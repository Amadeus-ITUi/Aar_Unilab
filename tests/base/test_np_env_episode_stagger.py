import gymnasium as gym
import numpy as np

from unilab.base.base import EnvCfg
from unilab.base.np_env import NpEnv, NpEnvState


class _EpisodeStaggerEnv(NpEnv):
    def __init__(self, *, num_envs: int, max_episode_steps: int):
        cfg = EnvCfg(
            ctrl_dt=0.01,
            max_episode_seconds=max_episode_steps * 0.01,
        )
        super().__init__(cfg=cfg, backend=object(), num_envs=num_envs)  # type: ignore[arg-type]
        self.reset_calls: list[np.ndarray] = []

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 1}

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        return actions

    def update_state(self, state: NpEnvState) -> NpEnvState:
        return state

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        self.reset_calls.append(env_indices.copy())
        assert self._state is not None
        self._state.terminated[env_indices] = False
        self._state.truncated[env_indices] = False
        obs = np.zeros((len(env_indices), 1), dtype=self._state.obs["obs"].dtype)
        return {"obs": obs}, {}


class _LegacyResetOverrideEnv(_EpisodeStaggerEnv):
    def __init__(self, *, num_envs: int, max_episode_steps: int):
        super().__init__(num_envs=num_envs, max_episode_steps=max_episode_steps)
        self.reset_override_calls = 0

    def _reset_done_envs(self) -> None:
        self.reset_override_calls += 1
        super()._reset_done_envs()


def test_init_state_preserves_randomized_episode_steps(monkeypatch) -> None:
    staggered_steps = np.array([0, 17, 54, 99], dtype=np.uint32)

    def _fake_randint(low, high=None, size=None, dtype=int):
        assert low == 0
        assert high == 100
        assert size == (4,)
        return staggered_steps.astype(dtype, copy=True)

    monkeypatch.setattr(np.random, "randint", _fake_randint)

    env = _EpisodeStaggerEnv(num_envs=4, max_episode_steps=100)
    state = env.init_state()

    np.testing.assert_array_equal(state.info["steps"], staggered_steps)
    np.testing.assert_array_equal(env.reset_calls[0], np.arange(4, dtype=np.int32))


def test_init_state_remains_compatible_with_no_argument_reset_override(monkeypatch) -> None:
    staggered_steps = np.array([3, 27, 61, 88], dtype=np.uint32)
    monkeypatch.setattr(
        np.random,
        "randint",
        lambda *args, **kwargs: staggered_steps.copy(),
    )

    env = _LegacyResetOverrideEnv(num_envs=4, max_episode_steps=100)
    state = env.init_state()

    assert env.reset_override_calls == 1
    np.testing.assert_array_equal(state.info["steps"], staggered_steps)


def test_regular_done_reset_still_zeroes_only_finished_episode_steps(monkeypatch) -> None:
    initial_steps = np.array([11, 22, 33, 44], dtype=np.uint32)
    monkeypatch.setattr(
        np.random,
        "randint",
        lambda *args, **kwargs: initial_steps.copy(),
    )

    env = _EpisodeStaggerEnv(num_envs=4, max_episode_steps=100)
    state = env.init_state()
    state.terminated[[1]] = True
    state.truncated[[3]] = True

    env._reset_done_envs()

    np.testing.assert_array_equal(
        state.info["steps"],
        np.array([11, 0, 33, 0], dtype=np.uint32),
    )
    np.testing.assert_array_equal(env.reset_calls[-1], np.array([1, 3], dtype=np.int32))
