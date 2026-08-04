# WE11 UniLab handover

This branch is the self-contained WE11 training archive. Its added task surface
contains only:

- `dr002_joystick_flat_we11/mujoco`
- `dr002_joystick_rough_we11/mujoco`

The rough task inherits only the flat WE11 task. The flat task owns its network,
actor/critic observation contract, measured wrench replay, wing-angle replay,
domain randomization, reward, PD, and command-delay configuration locally; it
does not inherit a WE9 or WE10 task.

## Runtime contract

- Joint order: `[left thigh, left calf, left wheel, right thigh, right calf, right wheel]`
- Physics / motor / policy: `400 / 200 / 50 Hz`
- Integrator: MuJoCo `RK4`
- Kp: `[2.0, 7.59, 0.0, 2.0, 7.59, 0.0]`
- Kd: `[0.080, 0.682, 0.05, 0.080, 0.682, 0.05]`
- Leg action scale: `0.5`
- Wheel action scale / raw clip: `10.0 / ±3.5`
- Shared command FIFO: reset-sampled `2..8` motor ticks (`10..40 ms`)
- Actor / rough critic dimensions: `135 / 334`
- Measured replay: discrete `0/1/2/3 Hz`; 4 Hz is excluded
- Wrench amplitude: one reset-owned scale in `[0.5, 1.5]`
- Wing-position amplitude: independent reset-owned scale in `[0.8, 1.2]`
- Wrench data: `src/unilab/assets/robots/dr002/we11/training_data/measured_wrench_20260728_skin/`
- Wing-angle data: `src/unilab/assets/robots/dr002/we11/training_data/wing_angle_20260713/`

## Reproducible Conda installation

```bash
git clone --branch Walking_Eagle-Unilab_final \
  git@git.esdyn.cn:walking-eagle/sar_unilab.git
cd sar_unilab
bash install_conda_environment.txt unilab_cuda cu128
conda activate unilab_cuda
```

The installer uses Conda + pip and also builds and validates native
command-delay PD. The repository intentionally does not ship `uv.lock`.

## Final checkpoints

| Task | File | SHA-256 |
|---|---|---|
| Flat | `models/we11/flat/model_1500.pt` | `53aa1e447568ebf9de22ab04967b43ad9c7d9a38d8bd76627e080c3f75be28b7` |
| Rough | `models/we11/rough/model_1500.pt` | `f7f80dfae9718584e31fc66d66a4821e95ff4510f6aa37f835e39a983066e232` |

## Build the private MuJoCo extension

```bash
bash third_party/mujoco_uni_mixed_pd/build_extension.sh
```

The generated shared library is a local build artifact and is intentionally not
committed.

## Rebuild the measured wrench replay

```bash
python scripts/data/prepare_we11_skin_wrench.py \
  --source-dir /path/to/measured_skin_wrench_data
```

The checked-in manifest preserves raw-file hashes and preprocessing metadata.

## Train rough WE11

```bash
CUDA_VISIBLE_DEVICES=0 \
UNILAB_MUJOCO_NTHREADS=16 \
python -u scripts/train_rsl_rl.py \
  task=dr002_joystick_rough_we11/mujoco \
  training.device=cuda:0 \
  training.no_play=true \
  training.play_render_mode=none \
  training.logger=tensorboard \
  training.nan_guard.enabled=false \
  algo.resume=false \
  'algo.load_run="-1"' \
  algo.checkpoint=-1 \
  algo.num_envs=4096 \
  algo.num_steps_per_env=24 \
  algo.max_iterations=99999 \
  algo.save_interval=500 \
  algo.run_name=we11_rough_self_contained_16t \
  algo.algorithm.num_learning_epochs=5 \
  algo.algorithm.num_mini_batches=4 \
  algo.algorithm.enable_compile=false
```

## Focused validation

```bash
python -m pytest -q \
  tests/envs/locomotion/dr002/test_we11_pace_training.py \
  tests/envs/locomotion/dr002/test_we11_rough_training.py \
  tests/envs/locomotion/common/test_terrain_spawn.py \
  tests/terrains/test_mujoco_heightfield.py
```
