# Aar_Unilab

A self-contained multi-robot MuJoCo training and Sim2Sim workspace. The current
WE11 Flat/Rough/Getup implementation is preserved as the protected training
baseline; PE01 is a separate robot/observation/policy/algorithm profile.

```bash
bash tools/install_environment.sh

/ssd/conda/envs/aar_unilab/bin/python scripts/train.py \
  robot=we11 task=flat observation=we11_default \
  policy=we11_mlp algorithm=rsl_rl_ppo simulator=mujoco

/ssd/conda/envs/aar_unilab/bin/python scripts/play.py \
  robot=we11 task=flat observation=we11_default \
  policy=we11_mlp algorithm=rsl_rl_ppo simulator=mujoco
```

The repository can be moved: run `tools/rebind_environment.sh` after cloning.
Historical WE11 notes below may mention predecessor paths; they are not runtime
dependencies. Maintained C++ code is in `sim2sim/`, while imported Play and
Deploy implementations are isolated under `references/`.

## Preserved WE11 handover

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
- Tasks: `dr002_joystick_flat_we11`, `dr002_joystick_getup_we11`, and
  `dr002_joystick_rough_we11`

All other robot assets, algorithm implementations, task owners, and simulation
backends have been removed from this archive.

## Repository layout

- `conf/ppo/task/dr002_joystick_flat_we11/`: policy network, observations,
  control, disturbances, rewards, and domain randomization.
- `conf/ppo/task/dr002_joystick_getup_we11/`: 倒地起身课程、重置姿态与奖励配置。
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

## 常用训练命令

以下命令均在 UniLab 仓库根目录执行。首次使用先运行安装脚本并激活
`unilab_cuda` 环境；训练默认不启动交互式画面，日志写入 TensorBoard。

Flat：

```bash
CUDA_VISIBLE_DEVICES=0 UNILAB_MUJOCO_NTHREADS=16 \
python -u scripts/train_rsl_rl.py \
  task=dr002_joystick_flat_we11/mujoco \
  training.device=cuda:0 \
  training.no_play=true \
  training.play_render_mode=none \
  training.logger=tensorboard
```

Getup：

```bash
CUDA_VISIBLE_DEVICES=0 UNILAB_MUJOCO_NTHREADS=16 \
python -u scripts/train_rsl_rl.py \
  task=dr002_joystick_getup_we11/mujoco \
  training.device=cuda:0 \
  training.no_play=true \
  training.play_render_mode=none \
  training.logger=tensorboard \
  algo.num_envs=2048 \
  algo.num_steps_per_env=24 \
  algo.max_iterations=4000 \
  algo.save_interval=100 \
  algo.algorithm.enable_compile=false
```

Rough（长时间训练常用配置）：

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

恢复 Flat 或 Getup 时，将上面命令中的 `task` 换成对应 selector；`<RUN>` 是
`logs/rsl_rl_ppo/` 下的运行目录名，`checkpoint=-1` 表示加载最新 checkpoint。

## 从指定权重续训到板端发布

下面以 Getup 为例串起完整流程。先将 `SOURCE_WEIGHT` 换成要继续训练的
`model_N.pt` 绝对路径；续训会创建新的时间戳 run 目录，不会覆盖原目录。

```bash
cd /home/angela/ssd/Pheonix/UniLab
conda activate unilab_cuda

SOURCE_WEIGHT=/home/angela/ssd/Pheonix/UniLab/logs/rsl_rl_ppo/DR002JoystickGetupWE11/xxx/model_N.pt

CUDA_VISIBLE_DEVICES=0 \
UNILAB_MUJOCO_NTHREADS=16 \
python -u scripts/train_rsl_rl.py \
  task=dr002_joystick_getup_we11/mujoco \
  training.device=cuda:0 \
  training.no_play=true \
  training.play_render_mode=none \
  training.logger=tensorboard \
  algo.resume=true \
  algo.load_run="$SOURCE_WEIGHT" \
  algo.checkpoint=-1 \
  algo.num_envs=2048 \
  algo.num_steps_per_env=24 \
  algo.max_iterations=4000 \
  algo.save_interval=100 \
  algo.algorithm.enable_compile=false
```

训练完成后，将 `RUN_DIR` 指向要发布的 `xxx` run 目录。下面的
`algo.checkpoint=-1` 会按 `model_N.pt` 中的数字选择该目录最新一轮，并在
同一目录生成 `policy.onnx` 和 `policy_export_manifest.json`。

```bash
RUN_DIR=/home/angela/ssd/Pheonix/UniLab/logs/rsl_rl_ppo/DR002JoystickGetupWE11/xxx

python -u scripts/train_rsl_rl.py \
  task=dr002_joystick_getup_we11/mujoco \
  training.device=cuda:0 \
  training.play_only=true \
  training.play_render_mode=none \
  training.play_env_num=1 \
  training.export_jit=false \
  algo.load_run="$RUN_DIR" \
  algo.checkpoint=-1

ls -lh "$RUN_DIR"/policy.onnx "$RUN_DIR"/policy_export_manifest.json
sha256sum "$RUN_DIR"/policy.onnx
```

先 dry-run 校验 ONNX 的来源、哈希和 `obs[1,145] -> act[1,6]` 接口，通过后再
原子发布到同级 `Play` 仓库：

```bash
python scripts/publish_we11_to_play.py \
  --run "$RUN_DIR" \
  --task DR002JoystickGetupWE11

python scripts/publish_we11_to_play.py \
  --run "$RUN_DIR" \
  --task DR002JoystickGetupWE11 \
  --apply
```

完成 Play 回放验收后，从 Play 根目录在 x86 主机上将 ONNX 转换为 MNN。
`--prepare-only` 只做本地转换、接口检查和有限值推理，不连接板端；转换用的
临时发布目录会在命令结束后清理。

```bash
cd /home/angela/ssd/Pheonix/Play
python scripts/publish_we11_from_play.py --prepare-only
```

最后先执行板端只读预检，确认 SSH、目标目录、磁盘空间和控制进程状态；
通过后才使用 `--apply` 重新转换并发布 `policy.onnx`、`lab_policy.mnn` 和
`lab_policy_manifest.json`。脚本会备份旧模型，不会启停节点或使能电机。

```bash
python scripts/publish_we11_from_play.py \
  --host esd@192.168.8.202 \
  --remote-root /home/esd/Pheonix/Deploy

python scripts/publish_we11_from_play.py \
  --host esd@192.168.8.202 \
  --remote-root /home/esd/Pheonix/Deploy \
  --apply
```

Flat 流程将 task selector 改为 `dr002_joystick_flat_we11/mujoco`，并将发布脚本的
`--task` 改为 `DR002JoystickFlatWE11`。checkpoint 的 task、观测维度、历史长度、
动作顺序和 Play 配置必须一致；发布脚本会拒绝不匹配的模型。

## TensorBoard 后端

前台启动，仅允许本机访问：

```bash
tensorboard \
  --logdir logs/rsl_rl_ppo \
  --host 127.0.0.1 \
  --port 6006 \
  --reload_interval 5
```

服务器上后台启动，并监听所有网卡：

```bash
nohup tensorboard \
  --logdir "$PWD/logs/rsl_rl_ppo" \
  --host 0.0.0.0 \
  --port 6006 \
  --reload_interval 5 \
  > /tmp/we11_tensorboard.log 2>&1 &
```

浏览器访问 `http://<服务器IP>:6006`，运行日志可用
`tail -f /tmp/we11_tensorboard.log` 查看。`0.0.0.0` 会对网络开放端口，只应在
可信网络或有防火墙限制的服务器上使用。

## Play 常用命令

以下命令从 UniLab 切换到同级 `Play` 仓库执行；首次运行或 C++ 代码变更后先
构建一次：

```bash
cd ../Play
./build.sh

# Flat / Getup / Getup 混合阶段
./scripts/play_we11.sh 0 flat
./scripts/play_we11.sh 0 getup
./scripts/play_we11.sh 0 --stage mixed

# 加载 1/2/3 Hz 实测 wrench 与翼角数据
./scripts/play_we11.sh 1
./scripts/play_we11.sh 2
./scripts/play_we11.sh 3

# Level-9 rough（默认 rough、3 Hz、1.0 倍外力）
./scripts/play_we11_level9.sh
./scripts/play_we11_level9.sh rough 3 1.0
```

没有手柄、仅做自动化冒烟时，可在 Play 命令前添加
`RL_SAR_PLAY_AUTOSTART=1`；正常手柄回放不要设置该变量。完整参数见
Play 仓库根目录的 `README.md`。

## Inspect and validate

```bash
# Zero-action rough environment visualization
python scripts/visualize_task_env.py \
  --task DR002JoystickRoughWE11 \
  --num_envs 16

# Tests and lint
python -m pytest -q
python -m ruff check .
```

The two selected final checkpoints are tracked in `models/we11/`. Native shared
libraries, other checkpoints, logs, videos, and generated deployment exports
remain machine-specific artifacts and are not tracked in Git.
