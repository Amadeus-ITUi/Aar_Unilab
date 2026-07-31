# Walking Eagle WE11 UniLab

这是 WE11 机器人的最小可复现训练仓库。当前明确支持的范围只有：

- 机器人：WE11
- 算法：RSL-RL PPO
- 仿真器：MuJoCo
- 任务：`dr002_joystick_flat_we11`、`dr002_joystick_rough_we11`

其他机器人资产、算法实现、任务配置和仿真后端均已从本归档移除。

## 目录说明

- `conf/ppo/task/dr002_joystick_flat_we11/`：网络、观测、控制、外力、奖励和域随机化。
- `conf/ppo/task/dr002_joystick_rough_we11/`：在平地 WE11 契约上增加 rough 地形课程。
- `src/unilab/assets/robots/dr002/we11/`：MJCF、网格、URDF 原始资料、PACE 参数、六维力和翼角数据。
- `src/unilab/envs/locomotion/dr002/`：WE11 平地与 rough 环境。
- `src/unilab/algos/torch/`：保留的 PPO 和 WE11 adaptation 网络代码。
- `third_party/mujoco_uni_mixed_pd/`：command-delay 批量 PD 补丁与可复现构建脚本。
- `scripts/data/prepare_u9_skin_wrench.py`：将保留的实测数据可复现转换为 WE11 六维力回放数据。
- `tests/`：WE11、地形、续训和环境契约测试。

完整参数和训练契约见 [`WE11_HANDOVER.md`](WE11_HANDOVER.md)。

## 安装

```bash
cd /home/esd_wch/lsaac_lab_ws/Walking_Eagle-UniLab-we11-clean
uv sync
./third_party/mujoco_uni_mixed_pd/build_extension.sh
```

仓库内命令统一使用 `uv run`。

## 开始训练

默认 Hydra 任务已经是 WE11 平地：

```bash
uv run python -u scripts/train_rsl_rl.py \
  training.device=cuda:0 \
  training.no_play=true \
  training.play_render_mode=none
```

WE11 rough 完整训练命令：

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

## 续训与回放

```bash
# <RUN> 填训练目录名
uv run python -u scripts/train_rsl_rl.py \
  task=dr002_joystick_rough_we11/mujoco \
  algo.resume=true \
  algo.load_run=<RUN> \
  algo.checkpoint=-1 \
  training.no_play=true

# MuJoCo checkpoint 回放
uv run eval --algo ppo \
  --task dr002_joystick_rough_we11 \
  --sim mujoco \
  --load-run <RUN> \
  --render-mode interactive
```

## 查看和验证

```bash
# rough 环境零动作可视化
uv run python scripts/visualize_task_env.py \
  --task DR002JoystickRoughWE11 \
  --num_envs 16

# TensorBoard，5 秒自动刷新
uv run tensorboard --logdir logs/rsl_rl_ppo --port 6006 --reload_interval 5

# 测试与静态检查
uv run pytest -q
uv run ruff check .
```

本地编译的 `.so`、checkpoint、日志、视频和部署导出模型属于生成物，不提交到 Git。
