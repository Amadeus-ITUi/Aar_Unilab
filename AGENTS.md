# Aar_Unilab Development Rules

Create the supported SSD Conda environment with `bash tools/install_environment.sh`.
Do not run `conda init`; invoke tools from `/ssd/conda/envs/aar_unilab` explicitly.

The protected baseline product path is:

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
8. New robots and algorithms enter through `unilab.catalog` contracts. Never
   specialize a common entrypoint or the C++ release loader to WE11 dimensions.
9. Treat `references/` as read-only provenance: production modules and build
   targets must not import, include, link or execute reference code.

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
python -m ruff format --check .
python -m ruff check .
python -m mypy src/unilab
python -m pytest -m "not slow"
git diff --check
```

For asset, control, observation, reward, or randomization changes, compose,
initialize, and step both WE11 tasks before considering the change complete.
