# Contributing

This branch is intentionally limited to the WE11 robot, RSL-RL PPO, and the
MuJoCo backend. Do not add another robot, algorithm, or simulator without first
changing the archive scope.

## Setup

```bash
git clone --branch Walking_Eagle-Unilab_final \
  git@git.esdyn.cn:walking-eagle/sar_unilab.git
cd sar_unilab
bash install_conda_environment.txt unilab_cuda cu128
conda activate unilab_cuda
```

Run Python entrypoints directly after activating this Conda environment. Do not
mix packages from the base environment, a local virtual environment, or a
different checkout.

## Ownership rules

- Robot assets and measured training data live under
  `src/unilab/assets/robots/dr002/we11/`.
- WE11 flat and rough task configuration lives under
  `conf/ppo/task/dr002_joystick_{flat,rough}_we11/`.
- Environment behavior belongs in `src/unilab/envs/locomotion/dr002/`; scripts
  should only assemble the workflow.
- PPO integration belongs in `src/unilab/algos/torch/` and
  `scripts/train_rsl_rl.py`.
- MuJoCo-specific behavior must stay inside the MuJoCo backend or its owner
  configuration.
- Keep XML/model lookup out of `step()`, `reset()`, and other hot paths.
- Task keyframes belong in task/scene XML, not the reusable robot XML.

## Validation

Run the narrowest relevant test while editing, then run the complete archive
checks before handoff:

```bash
make lint
make type
make test
git diff --check
```

Changes to WE11 assets, control timing, observations, rewards, randomization,
or PPO network dimensions must also initialize and step both WE11 task configs.

## Commits

Use concise Conventional Commit prefixes such as `feat:`, `fix:`, `test:`,
`docs:`, `refactor:`, and `chore:`. Only the two selected handover checkpoints
under `models/we11/` may be committed; never commit additional checkpoints,
run logs, local environments, caches, or temporary exports.
