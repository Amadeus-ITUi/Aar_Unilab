# Walking Eagle WE11 UniLab

**算法交付负责人：熊铭煊**

**交付日期：2026-08-31**

**正式分支：`pheonix-we11/unilab-bringup`**

本次交付目的是实现新需求：手动控制上肢机翼扑打角度。倒地自启方面，
已在仿真中成功实现腿部收拢状态下的倒地自启；前倾和后倾状态下的倒地
自启正在实现。

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
- `models/we11/`: final Flat and Rough `model_1500.pt` checkpoints and hashes.
- `install_conda_environment.txt`: one-command Conda/pip installation and
  Flat/Rough smoke validation.
- `scripts/data/prepare_we11_skin_wrench.py`: reproducible conversion of the
  retained hardware acquisition into WE11 wrench replay assets.
- `tests/`: focused WE11, terrain, resume, and environment-contract tests.

The full physical/training contract is documented in
[`WE11_HANDOVER.md`](WE11_HANDOVER.md).

## Setup

```bash
git clone --branch pheonix-we11/unilab-bringup \
  https://git.esdyn.cn/pheonix-eagle/sar_unilab.git
cd sar_unilab
bash install_conda_environment.txt unilab_cuda cu128
conda activate unilab_cuda
```

The installer creates a Python 3.13 Conda environment, installs the project and
test dependencies with pip, builds native command-delay PD, runs regression
tests, and executes one Flat and one Rough PPO smoke. See
[`WE11_CONDA_INSTALL_GUIDE.txt`](WE11_CONDA_INSTALL_GUIDE.txt).

## Final checkpoints

- Flat: `models/we11/flat/model_1500.pt`
- Rough: `models/we11/rough/model_1500.pt`

Verify both with:

```bash
(cd models/we11 && sha256sum -c SHA256SUMS)
```

## Train

Flat WE11 is the default Hydra task:

```bash
python -u scripts/train_rsl_rl.py \
  training.device=cuda:0 \
  training.no_play=true \
  training.play_render_mode=none
```

Rough WE11:

```bash
CUDA_VISIBLE_DEVICES=0 \
UNILAB_MUJOCO_NTHREADS=16 \
python -u scripts/train_rsl_rl.py \
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
python -u scripts/train_rsl_rl.py \
  task=dr002_joystick_rough_we11/mujoco \
  algo.resume=true \
  algo.load_run=<RUN> \
  algo.checkpoint=-1 \
  training.no_play=true

# MuJoCo checkpoint playback
eval --algo ppo \
  --task dr002_joystick_rough_we11 \
  --sim mujoco \
  --load-run <RUN> \
  --render-mode interactive
```

## Inspect and validate

```bash
# Zero-action rough environment visualization
python scripts/visualize_task_env.py \
  --task DR002JoystickRoughWE11 \
  --num_envs 16

# TensorBoard
tensorboard --logdir logs/rsl_rl_ppo --port 6006 --reload_interval 5

# Tests and lint
python -m pytest -q
python -m ruff check .
```

The two selected final checkpoints are tracked in `models/we11/`. Native shared
libraries, other checkpoints, logs, videos, and generated deployment exports
remain machine-specific artifacts and are not tracked in Git.
