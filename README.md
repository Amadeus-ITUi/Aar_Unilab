# Walking Eagle WE11 UniLab

**算法交付负责人：汪成浩**

**正式分支：`Walking_Eagle-Unilab_final`**

交接范围、接口、安全边界和验证记录见：

- [`docs/handover.md`](docs/handover.md)
- [`docs/interface.md`](docs/interface.md)
- [`docs/test_record.md`](docs/test_record.md)
- [`CHANGELOG.md`](CHANGELOG.md)

Minimal, reproducible training repository for the WE11 robot. The supported
runtime surface is intentionally limited to:

- Robot: WE11
- Algorithm: RSL-RL PPO
- Simulator: MuJoCo
- Tasks: `dr002_joystick_flat_we11` and `dr002_joystick_rough_we11`

All other robot assets, algorithm implementations, task owners, and simulation
backends have been removed from this archive.

## Repository layout

- `conf/ppo/task/dr002_joystick_flat_we11/`: policy network, observations,
  control, disturbances, rewards, and domain randomization.
- `conf/ppo/task/dr002_joystick_rough_we11/`: rough-terrain curriculum layered
  on the flat WE11 contract.
- `src/unilab/assets/robots/dr002/we11/`: MJCF, meshes, URDF provenance, PACE
  parameters, measured wrench, and wing-angle data.
- `src/unilab/envs/locomotion/dr002/`: WE11 flat and rough environments.
- `src/unilab/algos/torch/`: retained PPO and WE11 adaptation-network code.
- `third_party/mujoco_uni_mixed_pd/`: reproducible native command-delay PD
  extension patches and build script.
- `scripts/data/prepare_u9_skin_wrench.py`: reproducible conversion of the
  retained hardware acquisition into WE11 wrench replay assets.
- `tests/`: focused WE11, terrain, resume, and environment-contract tests.

The full physical/training contract is documented in
[`WE11_HANDOVER.md`](WE11_HANDOVER.md).

## Setup

```bash
cd /home/esd_wch/lsaac_lab_ws/Walking_Eagle-UniLab-we11-clean
uv sync
./third_party/mujoco_uni_mixed_pd/build_extension.sh
```

Always run repository commands through `uv run`.

## Train

Flat WE11 is the default Hydra task:

```bash
uv run python -u scripts/train_rsl_rl.py \
  training.device=cuda:0 \
  training.no_play=true \
  training.play_render_mode=none
```

Rough WE11:

```bash
env -u PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 \
PYTHONNOUSERSITE=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
UNILAB_MUJOCO_NTHREADS=16 \
UV_CACHE_DIR=/tmp/unilab_uv_cache \
uv run python -u scripts/train_rsl_rl.py \
  task=dr002_joystick_rough_we11/mujoco \
  training.device=cuda:0 \
  training.no_play=true \
  training.play_render_mode=none \
  training.logger=tensorboard \
  algo.num_envs=4096 \
  algo.num_steps_per_env=24 \
  algo.max_iterations=99999 \
  algo.save_interval=500 \
  algo.algorithm.enable_compile=false
```

## Resume and evaluate

```bash
# Resume a run; use its directory name for <RUN>
uv run python -u scripts/train_rsl_rl.py \
  task=dr002_joystick_rough_we11/mujoco \
  algo.resume=true \
  algo.load_run=<RUN> \
  algo.checkpoint=-1 \
  training.no_play=true

# MuJoCo checkpoint playback
uv run eval --algo ppo \
  --task dr002_joystick_rough_we11 \
  --sim mujoco \
  --load-run <RUN> \
  --render-mode interactive
```

## Inspect and validate

```bash
# Zero-action rough environment visualization
uv run python scripts/visualize_task_env.py \
  --task DR002JoystickRoughWE11 \
  --num_envs 16

# TensorBoard
uv run tensorboard --logdir logs/rsl_rl_ppo --port 6006 --reload_interval 5

# Tests and lint
uv run pytest -q
uv run ruff check .
```

The native shared library, checkpoints, logs, videos, and exported deployment
models are generated artifacts and are intentionally not tracked in Git.
