# WE11 Archive Agent Rules

Always use `uv run`, not a bare Python interpreter.

This branch has one supported product path:

- robot: WE11;
- algorithm: RSL-RL PPO;
- simulator: MuJoCo;
- tasks: `dr002_joystick_flat_we11/mujoco` and
  `dr002_joystick_rough_we11/mujoco`.

## Invariants

1. Preserve `NpEnvState.obs` as a dictionary and preserve reset/step contracts.
2. Keep policy/critic observation groups, history length, action order, action
   scale, command delay, PD gains, and network dimensions synchronized between
   configuration, checkpoints, export, and deployment.
3. Put task/reward/randomization choices in Hydra owner YAML whenever possible.
4. Keep MuJoCo-specific behavior in `src/unilab/base/backend/mujoco/` or the
   WE11 MuJoCo configuration; environment code calls only declared backend
   interfaces.
5. Access model/XML/asset metadata only on cold paths. Never parse assets in
   `step()`, `reset()`, command sampling, or domain-randomization hot loops.
6. Keep task keyframes in task/scene XML; `we11.xml` remains a reusable robot
   description.
7. Preserve the 400 Hz physics, 200 Hz motor-control, and existing shared
   command-delay semantics unless the task explicitly changes that contract.
8. Do not reintroduce other robot assets, algorithms, Motrix/Viser routes, or
   stale compatibility copies into this archive.

## Important paths

- Training: `scripts/train_rsl_rl.py`
- Flat task: `conf/ppo/task/dr002_joystick_flat_we11/`
- Rough task: `conf/ppo/task/dr002_joystick_rough_we11/`
- Environment: `src/unilab/envs/locomotion/dr002/`
- Robot and data: `src/unilab/assets/robots/dr002/we11/`
- PPO integration: `src/unilab/algos/torch/`
- MuJoCo backend: `src/unilab/base/backend/mujoco/`
- Handover: `WE11_HANDOVER.md`

## Required validation

Run focused tests nearest the changed contract, then:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src/unilab
uv run pyright
uv run pytest -m "not slow"
git diff --check
```

For asset, control, observation, reward, or randomization changes, compose,
initialize, and step both WE11 tasks before considering the change complete.
