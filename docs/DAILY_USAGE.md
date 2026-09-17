# 日常使用指南

[返回 README](../README.md)

`logs/` 已于 2026-09-17 清理开发调试产物；实际训练与已验证的站立模型保留，
目录用途见 [logs/README.md](../logs/README.md)。历史报告中的开发日志路径可能已删除。

本页按日常操作排列：复制命令，再查紧邻的参数表。除安装和构建脚本外，
完整 Python 命令假设已激活 SSD 环境，并在仓库根目录执行。
`bash tools/train.sh` 会自行选择 SSD 环境，不需要激活。
示例中的 `<run_id>`、`model_N.pt`、`/path/to/...` 需要换成实际值。

| 操作 | 跳转 |
| --- | --- |
| 环境与命令写法 | [1](#environment) |
| 简短训练命令与默认参数文件 | [1.1](#train-defaults) |
| 入口参数与样机组合 | [2](#selectors) |
| WE11 训练、续训、TensorBoard | [3](#we11-training) |
| WE11 Python 回放与录像 | [4](#we11-play) |
| PE01/PE02 训练与回放 | [5](#pe-training) |
| ONNX 导出与 WE11 模型切换 | [6](#export) |
| C++ Sim2Sim | [7](#cpp) |
| 输出文件位置 | [8](#outputs) |
| 验证与排错 | [9](#checks) |

<a id="environment"></a>

## 1. 环境与命令写法

| 场景 | 命令 | 说明 |
| --- | --- | --- |
| 首次安装 | `bash tools/install_environment.sh` | 默认安装 CUDA 12.8 版 PyTorch，环境位于 `/ssd/conda/envs/aar_unilab` |
| 只使用 CPU 的安装 | `bash tools/install_environment.sh --torch cpu` | `--torch` 支持 `cu128`（默认）和 `cpu` |
| 每次打开终端 | `source tools/activate_environment.sh` | 激活 SSD 环境，清理旧 ROS Python 路径 |
| clone、移动或切换副本后 | `bash tools/rebind_environment.sh` | 重新绑定 editable install、本地 PD 扩展及通用 C++ 构建 |
| 确认 Python 路径 | `which python` | 应为 `/ssd/conda/envs/aar_unilab/bin/python` |

日常开始工作：

```bash
cd /ssd/Aar_Unilab
source tools/activate_environment.sh
```

不运行 `conda init`，也不按环境名执行 `conda activate aar_unilab`。
可显式激活 `conda activate /ssd/conda/envs/aar_unilab`。
自动化脚本可将下文的 `python` 替换为 `/ssd/conda/envs/aar_unilab/bin/python`。

命令写法：

- Python 训练/回放参数写成 `key=value`，如 `algo.num_envs=2048`，不加 `--`。
- C++、安装和发布工具使用 `--参数 值`，如 `--steps 100`。
- 多行命令末尾的 `\` 后不要再加空格或注释。
- 路径包含空格时，将整个参数加引号：`"algo.load_run=/path with spaces/model_100.pt"`。
- Hydra 列表写成 `network.actor_hidden_dims='[256,256]'`；布尔值用 `true/false`，空值用 `null`。

<a id="train-defaults"></a>

### 1.1 简短训练命令与默认参数文件

常用启动参数集中在 [conf/train_defaults.yaml](../conf/train_defaults.yaml)，可直接编辑：

| 文件区域 | 内容 |
| --- | --- |
| `environment` | GPU 可见性、MuJoCo 工作线程、BLAS/OpenMP 线程数 |
| `unset_environment` | 仅对子进程移除的变量，默认清除 `PYTHONPATH` |
| `common` | 4096 环境、24 步、1000 轮、每 100 轮保存、TensorBoard |
| `profiles.<名称>.selectors` | 通过 catalog 选择机器、任务、观测、网络、算法及仿真器 |
| `profiles.<名称>.overrides` | 该预设独立的覆盖参数，可追加 `algo.max_iterations` 等 |

```bash
# WE11 Getup
bash tools/train.sh we11_getup

# PE01 原任务
bash tools/train.sh pe01

# PE02 固定初态行走（2 Hz / 6 cm，关闭物理随机化和噪声）
bash tools/train.sh pe02_walking

# 只查看最终参数，不开始训练
bash tools/train.sh pe02_walking --dry-run

# 仅这次改为 2048 环境、2000 轮
bash tools/train.sh pe02_walking algo.num_envs=2048 algo.max_iterations=2000
```

启动器会在 NumPy、PyTorch、MuJoCo 导入前设置子进程环境，直接使用
`/ssd/conda/envs/aar_unilab/bin/python`，因此无需 `env -u PYTHONPATH` 或激活 Conda。
打印的 `Environment` 和 `Command` 是本次实际传给训练进程的设置。

Hydra 参数优先级是：**命令行 > profile.overrides > common > 原任务 YAML**。
这些只是启动预设，不会修改 PE01、PE02 或 WE11 的原始任务默认配置；直接使用旧
Python 入口时，不会读取这个文件。PE01 和 PE02 的环境、奖励、网络仍分别维护。

环境变量沿用常见的默认值规则：**终端已有的同名环境变量 > 文件 environment**。
例如 `CUDA_VISIBLE_DEVICES=1 bash tools/train.sh pe02_walking` 临时使用 GPU 1；
`CUDA_VISIBLE_DEVICES="" bash tools/train.sh pe02_walking` 隐藏 CUDA 设备。
PE01/PE02 的 `training.device=auto` 会选择可用设备。
`UNILAB_MUJOCO_NTHREADS=24 bash tools/train.sh pe01` 临时使用 24 个仿真线程，
PE01/PE02 的线程参数会跟随这个值；也可在其 profile 中单独写数字。
若修改文件后发现终端环境变量仍覆盖它，可先 `unset` 对应变量，再用 `--dry-run` 核对。

这些环境变量本来并不是统一的项目默认值：未设置 CUDA 可见性时会暴露可用 GPU；
WE11 未指定线程数时按 `min(环境数, CPU 逻辑线程数 × 2)` 选择；PE01/PE02 原任务
各有自己的线程配置；数学库线程数受环境和库版本影响。这个文件把本机常用设置明确保存。

`bash tools/train.sh --list` 列出预设，`--config /path/to/defaults.yaml` 使用其他默认文件。
新增预设也必须使用 catalog 已声明的合法组合。奖励、控制、网络等任务细节继续在
各机器人 owner YAML 中修改；本文件不复制这些训练实现。

<a id="selectors"></a>

## 2. 入口参数与支持的组合

`scripts/train.py` 和 `scripts/play.py` 都要求显式给出下面六个参数。
它们选择对应实现；其余参数再传给该样机的训练或回放脚本。

| 参数 | 含义 | WE11 示例 |
| --- | --- | --- |
| `robot` | 机器人 | `we11` |
| `task` | 任务 | `flat`、`rough`、`getup` 三选一 |
| `observation` | 策略观测定义与历史布局 | `we11_v2_145` |
| `policy` | 策略网络实现 | `we11_mlp` |
| `algorithm` | 训练算法实现 | `rsl_rl_ppo` |
| `simulator` | 仿真后端 | 当前使用 `mujoco` |

常用组合如下，同一行的参数配套使用：

| robot | task | observation | policy | algorithm |
| --- | --- | --- | --- | --- |
| `we11` | `flat` / `rough` / `getup` | `we11_v2_145` | `we11_mlp` | `rsl_rl_ppo` |
| `pe01` | `pe01_flat` | `pe01_v2`（旧模型 `pe01_legacy`） | `pe01_encoder_mlp` | `pe01_custom_ppo` |
| `pe02` | `pe02_flat` | `pe02_v2`（旧模型 `pe02_v1`） | `pe02_encoder_mlp` | `pe02_custom_ppo` |

旧 WE11 入口仍可使用：`python scripts/train_rsl_rl.py task=dr002_joystick_getup_we11/mujoco ...`。
完整任务名用于这个旧入口；统一入口使用 `task=getup`。

<a id="we11-training"></a>

## 3. WE11 训练、续训与日志

### 3.1 开始训练

以下示例选择 GPU 0，训练 Getup 4000 轮，每 100 轮保存一次。
2048 个环境是示例设置，可按显存与 CPU 负载调整。

```bash
CUDA_VISIBLE_DEVICES=0 UNILAB_MUJOCO_NTHREADS=16 \
python scripts/train.py \
  robot=we11 task=getup observation=we11_v2_145 \
  policy=we11_mlp algorithm=rsl_rl_ppo simulator=mujoco \
  training.no_play=true \
  training.logger=tensorboard \
  algo.num_envs=2048 algo.num_steps_per_env=24 \
  algo.max_iterations=4000 algo.save_interval=100
```

切换任务只需改 `task=flat` 或 `task=rough`。以下默认值来自当前三套任务合成后的配置。

| 参数 | 默认值 | 用法 |
| --- | --- | --- |
| `algo.num_envs` | `4096` | 并行环境数；显存不足时先减小，例如 `2048`、`1024` |
| `algo.num_steps_per_env` | `24` | 每轮每个环境采样的策略步数；一轮样本数约为此值 × 环境数 |
| `algo.max_iterations` | `50000` | 本次执行的学习轮数；续训时表示追加轮数 |
| `algo.save_interval` | `100` | 每隔多少轮保存 checkpoint |
| `algo.seed` | `1` | 随机种子，例如 `algo.seed=42` |
| `training.device` | `null`（配置项） | 当前 WE11 脚本未读取此字段，实际自动选择设备；选择 GPU 使用下表的环境变量 |
| `training.no_play` | `false` | 设为 `true`，训练结束后直接退出，不自动进入回放/录像 |
| `training.logger` | `tensorboard` | 常用 `tensorboard`；也支持 `wandb`，`none` 关闭该日志后端 |
| `training.log_root` | `logs/rsl_rl_ppo` | 自定义日志根目录，例如 `training.log_root=logs/we11_trial`；任务名和 run 目录仍会追加 |
| `algo.algorithm.learning_rate` | `0.001` | PPO 初始学习率，例 `0.0003`；后续可能按自适应设置调整 |
| `algo.algorithm.num_learning_epochs` | `5` | 一轮采样数据执行多少遍学习 |
| `algo.algorithm.num_mini_batches` | `4` | 每遍学习拆分的数据批数 |
| `algo.algorithm.enable_compile` | `false` | 是否启用编译优化；日常先保持默认值 |

环境变量写在 `python` 前，不属于 Hydra 参数：

| 环境变量 | 示例 | 含义 |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | `CUDA_VISIBLE_DEVICES=0` | 限制可见 GPU；选物理 GPU 1 时写 `CUDA_VISIBLE_DEVICES=1`，它会成为进程内首个 CUDA 设备 |
| `UNILAB_MUJOCO_NTHREADS` | `UNILAB_MUJOCO_NTHREADS=16` | MuJoCo 批量仿真线程数，按 CPU 资源调整 |

WE11 会按 CUDA、XPU、MPS、CPU 的顺序自动选择可用设备，以终端
`Using device: ...` 为准。本机 CPU 试跑可在命令前设置 `CUDA_VISIBLE_DEVICES=""`
隐藏 CUDA，同时将 `algo.num_envs` 调小；单独传 `training.device=cpu` 不会强制切换。
PE01/PE02 的 `training.device` 正常生效，见第 5 节。

需要修改奖励、课程、观测或网络时，优先编辑对应任务 YAML。
当前 WE11 actor 网络层宽字段是 `algo.actor.actor_hidden_dims`，编码部分是
`algo.actor.mlp_hidden_dims`，critic 是 `algo.critic.hidden_dims`；
不要把公共配置里的 `algo.policy.actor_hidden_dims` 当作当前 WE11 actor 的实际配置。
修改观测、网络维度或控制参数后，需重新核对 checkpoint、导出和回放是否匹配。

### 3.2 从 checkpoint 续训

先将路径替换为实际文件；续训使用与源模型相同的任务和网络配置。

```bash
WE11_CHECKPOINT='/path/to/run/model_1000.pt'

CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  robot=we11 task=getup observation=we11_v2_145 \
  policy=we11_mlp algorithm=rsl_rl_ppo simulator=mujoco \
  "algo.load_run=$WE11_CHECKPOINT" \
  algo.num_envs=2048 algo.max_iterations=2000 \
  training.no_play=true
```

| 参数/写法 | 作用 |
| --- | --- |
| `algo.load_run=/path/to/run/model_1000.pt` | 精确加载这个 checkpoint；直接文件路径优先于 `algo.checkpoint` |
| `algo.load_run=/path/to/run algo.checkpoint=-1` | 加载该目录中编号最大的 `model_N.pt` |
| `algo.load_run=/path/to/run algo.checkpoint=1000` | 加载该目录的 `model_1000.pt` |
| `algo.load_run=<run_id>` | 在当前任务日志目录下查找；完整路径更容易核对来源 |
| `algo.max_iterations=2000` | 在载入的迭代状态上再运行 2000 轮，不是“训练到第 2000 轮” |

续训会建立新的时间戳 run 目录。确认终端打印 `Resuming from ...` 且路径正确。
当前 WE11 训练是否加载权重取决于 `algo.load_run`；单独设置 `algo.resume=true`
不会指定源模型。新训练不传 `algo.load_run`，使用配置里的默认字符串 `"-1"`，
不要带入上一次的加载路径。

### 3.3 看训练曲线

```bash
tensorboard --logdir logs/rsl_rl_ppo --port 6006
```

浏览器打开 `http://localhost:6006`。`--logdir` 可以缩小到某个任务或某个 run，
`--port` 可改为未占用端口。如果设置了 `training.log_root`，这里使用相应目录。

<a id="we11-play"></a>

## 4. WE11 Python 回放与录像

### 4.1 实时操作

以下模型随仓库保留，可以直接使用。回放自己训练的模型时替换 `algo.load_run`，
并将 `task` 设为训练时的任务。

```bash
AAR_EXPORT_POLICY=0 python scripts/play.py \
  robot=we11 task=getup observation=we11_v2_145 \
  policy=we11_mlp algorithm=rsl_rl_ppo simulator=mujoco \
  algo.load_run=models/we11/getup/model_9999.pt \
  training.play_render_mode=interactive training.play_env_num=1 \
  training.play_steps=null
```

| 参数 | 用法 |
| --- | --- |
| `algo.load_run` / `algo.checkpoint` | 与续训的模型选择规则相同 |
| `training.play_render_mode=interactive` | 打开实时窗口；需要按启动键 |
| `training.play_render_mode=record` | 离线渲染视频，需要有限的 `training.play_steps` |
| `training.play_render_mode=none` | 加载模型并执行可选导出，跳过回放循环；不是无窗口 rollout |
| `training.play_render_mode=auto` | 当前 MuJoCo 下等同于 `record`，不是实时窗口 |
| `training.play_steps` | 默认 `500` 个策略步；`null` 用于实时窗口持续运行，录像须填正整数 |
| `training.play_env_num` | 默认 `1`；实时操作保持 `1`，录像可使用多个环境 |
| `training.play_start_pose` | `upright` / `getup` / `mixed`，仅覆盖回放初始姿态；当前默认 `getup` |
| `training.play_getup_difficulty=0.8` | Getup 专用，取值 `0..1`；固定到 home-to-getup 姿态路径的对应难度 |
| `training.sim2sim_strict` | 默认 `true`，检查与训练记录的关键参数是否一致；报错时核对配置来源 |
| `AAR_EXPORT_POLICY=0` | 环境变量，跳过 ONNX 导出；默认值为 `1`，会在模型目录生成/更新导出文件 |

### 4.2 录制视频

```bash
AAR_EXPORT_POLICY=0 python scripts/play.py \
  robot=we11 task=getup observation=we11_v2_145 \
  policy=we11_mlp algorithm=rsl_rl_ppo simulator=mujoco \
  algo.load_run=models/we11/getup/model_9999.pt \
  training.play_render_mode=record training.play_env_num=1 \
  training.play_steps=500
```

视频写入所选 checkpoint 目录的 `play_video.mp4`，再次录制会覆盖同名文件。
WE11 策略频率为 50 Hz，500 步对应约 10 秒仿真时间。
录像摄像机可设置 `training.cam_tracking=true`、`training.cam_distance=6.0`、
`training.cam_elevation=-20.0`、`training.cam_azimuth=90.0`；这些参数用于录像，
实时窗口通过 MuJoCo 原生鼠标操作调整视角。

### 4.3 WE11 实时操作速查

以下适用于 WE11 Python 实时窗口及 `sim2sim/we11_play/`：

| 操作 | 键盘 | 手柄 |
| --- | --- | --- |
| 启动策略 | `1` 或 Enter | `RB+DPadUp` |
| 停止策略 | `P` | `LB+X` |
| 重置 | `R` | `RB+Y` |
| 前进速度 | — | 左摇杆纵轴 |
| yaw 转向 | — | 右摇杆横轴 |
| 机身高度 | — | 右摇杆纵轴，范围 0.20–0.30 m |

MuJoCo 原生鼠标操作用于旋转、缩放、平移和选中物体后拖动施力。
当前 WE11 实时回放不注入翼角 CSV、wrench CSV，也没有扑动频率参数；
训练和离线录像仍由对应任务配置决定，不能由实时窗口行为推断训练扰动已关闭。

<a id="pe-training"></a>

## 5. PE01 与 PE02

PE01 `pe01_v2` 与 PE02 `pe02_v2` 都已迁入完整多环境 PPO、独立速度估计器、
平地任务与 checkpoint 续训，两套代码和配置独立维护。旧最小适配器保留兼容。
基准对照与差异见 [PE01 迁移说明](PE01_TRAINING_MIGRATION.md)和
[PE02 迁移说明](PE02_TRAINING_MIGRATION.md)。

### 5.1 训练命令

PE01：

```bash
python scripts/train.py \
  robot=pe01 task=pe01_flat observation=pe01_v2 \
  policy=pe01_encoder_mlp algorithm=pe01_custom_ppo simulator=mujoco \
  algo.num_envs=4096 algo.num_steps_per_env=24 algo.max_iterations=15000 \
  training.seed=1 training.device=auto
```

PE02：

```bash
python scripts/train.py \
  robot=pe02 task=pe02_flat observation=pe02_v2 \
  policy=pe02_encoder_mlp algorithm=pe02_custom_ppo simulator=mujoco \
  algo.num_envs=256 algo.num_steps_per_env=24 algo.max_iterations=15000 \
  training.seed=1 training.device=auto
```

| 参数 | 默认值 | 适用范围与含义 |
| --- | --- | --- |
| `training.steps` | `128` | 仅旧 PE01 / PE02 v1 最小适配器 |
| `algo.num_envs` | `8192` | 两者：并行环境数；PE02 本机建议显式设 256，模型大小与 PE01 不同 |
| `algo.num_steps_per_env` | `24` | 两者：每个环境每轮采样步数 |
| `algo.max_iterations` | `15000` | 两者：本次学习轮数，续训时表示追加轮数 |
| `algo.save_interval` | `400` | 两者：checkpoint 保存周期 |
| `algo.num_learning_epochs` / `algo.num_mini_batches` | `5` / `4` | 两者：每轮 PPO 与 encoder 各执行 20 次小批次更新 |
| `training.seed` | `1` | 两者：随机种子；注意 WE11 用的是 `algo.seed` |
| `training.device` | 正式版均为 `auto`（旧 PE01 `cpu`） | 策略计算设备，可指定 `cpu`、`cuda:0`；物理仿真在 CPU |
| `training.resume` | `null` | 两者：用于续训的完整 checkpoint 路径 |
| `algo.learning_rate` | `0.001` | 两者：PPO 初始学习率，KL 自适应 |
| `algo.encoder_learning_rate` | `0.001` | 两者：速度估计器学习率 |
| `algo.clip_ratio` | `0.2` | 两者：PPO 概率比裁剪范围 |
| `algo.value_loss_coefficient` | `1.0` | 两者：价值损失权重 |
| `network.encoder_hidden_dims` | `[256,128]` | 两者：历史观测编码网络层宽 |
| `network.latent_dim` | `3` | 两者：监督估计机身线速度，固定 3 维 |
| `network.actor_hidden_dims` | `[512,256,128]` | 两者：actor 隐藏层 |
| `network.critic_hidden_dims` | `[512,256,128]` | 两者：critic 隐藏层 |
| `network.initial_std` | `1.0` | 两者：初始动作分布标准差 |

PE02 日常调参也可用专用入口，省略六个选择项：

```bash
python scripts/train_pe02.py \
  algo.num_envs=256 algo.max_iterations=1000 \
  algo.learning_rate=0.001
```

PE02 配置文件：

- [conf/pe02/config.yaml](../conf/pe02/config.yaml)：训练、网络和回放参数。
- [conf/pe02/task/pe02_flat.yaml](../conf/pe02/task/pe02_flat.yaml)：环境、控制和奖励。
- 默认 `env.reset_keyframe=home`、`env.initial_height=null` 使用已求解的站姿；见[站姿说明](../src/unilab/assets/robots/pe02/analysis/STANDING_POSE.md)。

PE01/PE02 正式版的完整配置写入 checkpoint 和 `training_config.yaml`。回放/导出从 checkpoint
恢复环境和网络配置，修改当前 YAML 不会重新定义旧 checkpoint 的网络与环境。
续训指定 `training.resume=<checkpoint>`，并保持环境数等训练合同一致；
`algo.max_iterations` 为追加轮数。TensorBoard 日志在 run 的 `tensorboard/`，
逐轮指标在 `metrics.jsonl`。`training.export=false` 可跳过最终导出。
PE01 正式版使用同名的独立参数，配置位于 `conf/pe01/config.yaml` 和
`conf/pe01/task/pe01_flat.yaml`。PE01、PE02 与 WE11 的网络参数所有权仍各自独立。

PE01/PE02 完整 PPO 每轮会像 WE11 一样打印对齐的多行训练摘要：当前/目标轮数、
累计采样步数、采样速度、采样和更新耗时、PPO/encoder 损失、KL、学习率、动作
标准差、逐项 `reward/*` 和当轮执行的 `evaluation/*`，末尾显示耗时与 ETA。
这项显示默认启用，与 `training.logger` 的 TensorBoard 开关独立，不需要改启动命令。
两台样机分别维护自己的输出模块，训练、环境、奖励和网络配置仍独立。

- `Mean step reward` 和 `reward/*` 是本轮采样中的单步均值；`Mean reward` 是最近
  最多 100 个已完成回合的累计奖励均值。尚无完整回合时显示 `n/a`。
- `Mean episode length` 以策略步为单位，同时打印 `Mean episode time`（秒）。
- 续训显示累计轮数及本次目标，例如第 600 轮追加 1000 轮时显示 `601/1600`。
  `Time elapsed` 和 ETA 从本次学习循环开始计算，计入评估与周期保存的耗时，
  不包含历史训练、启动初始化或训练后的导出时间。吞吐仍只统计采样和更新。
- 已经启动的训练进程不会热更新输出样式；下一次启动或续训自动使用新样式。

PE02 的 TensorBoard 采用 WE11/RSL-RL 的分组方式，横轴仍为训练轮数：

| 分组 | 指标 |
| --- | --- |
| `Loss/` | `value`、`surrogate`、`entropy`、`kl`、`clip_fraction`、`encoder`、`learning_rate` |
| `Policy/` | `mean_std`，平均动作标准差 |
| `Perf/` | `total_fps`、`collection_time`、`learning_time` |
| `Train/` | 平均回合回报、回合长度（策略步）、回合时间（秒）、单步回报、累计样本与更新次数 |
| `reward/` | 各奖励项的加权单步贡献，保留原数值，包括 `keep_balance=0.02` |
| `Eval/` | 独立评估的回报、失败率、实际高度、倾角、非足端接触和站立比例 |

`Train/mean_reward` 使用最近最多 100 个完整回合；没有回合结束时不写该曲线。
`Train/mean_episode_length` 与 WE11 一样以策略步计，另用 `Train/mean_episode_time`
保留秒数。事件每 10 秒刷新；控制台和 `metrics.jsonl` 保留原指标名称。
PE02 的 `reward/*` 已乘策略周期，WE11 的同名日志在乘周期之前记录，跨样机比较
奖励绝对值时仍需统一单位。`Loss/encoder` 是 PE02 的独立监督损失，不冒充 WE11
的 adaptation loss。

已结束的 PE02 训练可从 `metrics.jsonl` 重建同样的分组，保留原事件的轮数和时间戳：

```bash
env -u PYTHONPATH /ssd/conda/envs/aar_unilab/bin/python tools/rebuild_pe02_tensorboard.py \
  logs/pe02_walking/<run_id>
```

只对已停止训练的 run 执行。工具替换该 run 的 `tensorboard/`，原事件保存在 run 下的
`tensorboard_before_we11_style_<时间戳>.zip`，不会重写检查点或 JSONL。TensorBoard 若
仍缓存旧分组，重启其服务后刷新页面。

### 5.2 回放命令

将 `<run_id>` 换成训练结束时打印的目录。PE01：

```bash
python scripts/play.py \
  robot=pe01 task=pe01_flat observation=pe01_v2 \
  policy=pe01_encoder_mlp algorithm=pe01_custom_ppo simulator=mujoco \
  'checkpoint=logs/pe01_custom_ppo/pe01/pe01_flat/<run_id>/model_400.pt' \
  play.render=interactive play.plot=true play.steps=1000
```

PE02：

```bash
python scripts/play.py \
  robot=pe02 task=pe02_flat observation=pe02_v2 \
  policy=pe02_encoder_mlp algorithm=pe02_custom_ppo simulator=mujoco \
  'checkpoint=logs/pe02_custom_ppo/pe02/pe02_flat/<run_id>/model_400.pt'
```

| 参数 | 默认值 | 用法 |
| --- | --- | --- |
| `checkpoint` | 必填 | 指向该样机的 `.pt` 文件；PE02 也支持 `-1` 自动选取最新训练目录内编号最大的模型 |
| `play.render` | `interactive` | `interactive` 打开窗口；`none` 不开窗口但仍运行仿真 |
| `play.plot` | PE01：`true`；PE02 v2：`false` | 是否打开实时曲线窗口；无显示器时设为 `false` |
| `play.steps` | PE01：`1000`；PE02 v2：`-1` | 正整数限制回放策略步数；`-1` 持续运行，直到关闭仿真窗口或按 `Ctrl+C`；此处不使用 WE11 的 `training.play_steps=null` |
| `play.telemetry` | `logs/play/<robot>/<task>/latest` | CSV/JSONL 输出目录；重复使用同一目录会覆盖同名文件 |
| `training.device` | 正式版均为 `auto`（旧 PE01 `cpu`） | 策略推理设备 |

PE02 walking 自动回放最新模型：

```bash
bash tools/train.sh pe02_walking mode=play checkpoint=-1
```

`-1` 在当前 `training.log_root` 下按训练目录名中的时间排序，再按 `model_<轮次>.pt`
的数字轮次选择最新模型；不使用目录修改时间，忽略临时文件和非训练目录。
例如 walking 的搜索根目录为 `logs/pe02_walking`，standing 为 `logs/pe02_standing`。
若最新训练目录尚未保存模型，会明确报错，不会悄悄加载更早的训练。启动时打印最终路径。
显式指定 `.pt` 路径仍优先按该路径加载；`training.resume` 仍要求显式 checkpoint 路径。

无窗口短回放：在命令末尾添加或覆盖
`play.render=none play.plot=false play.steps=10`。
PE01/PE02 窗口启动后直接运行，不使用 WE11 的 Getup 启停组合键。
画面左上角显示 base link 的实测 X/Y 速度（机身坐标系，m/s）、绕机身 Z 轴的
yaw 角速度（rad/s）和 link 原点的世界 Z 高度（m）；这些量也写入 CSV/JSONL。
`play.plot=false` 只关闭额外曲线窗口，画面上的数值仍会显示。
相机仅在启动时对准机器人，之后由 MuJoCo 界面控制；Free 模式不会随机身移动，
需要跟随时使用原生 Tracking 模式。实时回放按策略周期限速（PE01/PE02 为 20 ms）。

<a id="export"></a>

## 6. ONNX 导出与 WE11 模型切换

### 6.1 WE11：checkpoint 导出 ONNX

```bash
WE11_CHECKPOINT='/path/to/run/model_1000.pt'

AAR_EXPORT_POLICY=1 python scripts/play.py \
  robot=we11 task=getup observation=we11_v2_145 \
  policy=we11_mlp algorithm=rsl_rl_ppo simulator=mujoco \
  "algo.load_run=$WE11_CHECKPOINT" \
  training.play_render_mode=none training.play_env_num=1
```

`task` 与训练任务一致。输出在 checkpoint 所在目录：

- `policy.onnx`：导出策略，重复导出覆盖同名文件。
- `policy_export_manifest.json`：Flat/Getup 导出时生成的来源和校验记录。

当前 Rough 也导出 `policy.onnx`，但此入口不会为它生成上述 Flat/Getup 专用 manifest。
`training.export_jit` 默认 `false`，日常使用 ONNX。

### 6.2 将 Flat/Getup 模型切换到同仓 WE11 C++ Play

先完成上面的导出。发布工具默认只检查；必须显式指定本仓库目标目录，
因为它的旧默认值仍然指向相邻的 `Play`。

```bash
WE11_RUN='/path/to/run'

python scripts/publish_we11_to_play.py \
  --run "$WE11_RUN" --checkpoint model_1000.pt \
  --task DR002JoystickGetupWE11 \
  --play-root sim2sim/we11_play
```

检查输出的 checkpoint、ONNX 哈希和目标路径后，实际切换：

```bash
python scripts/publish_we11_to_play.py \
  --run "$WE11_RUN" --checkpoint model_1000.pt \
  --task DR002JoystickGetupWE11 \
  --play-root sim2sim/we11_play --apply
```

| 参数 | 用法 |
| --- | --- |
| `--run` | 含 checkpoint、`policy.onnx` 和导出记录的 run 目录；必填 |
| `--checkpoint` | 该 run 内文件名或 checkpoint 绝对路径；显式填写便于核对来源 |
| `--task` | 使用完整任务名；短 run 名的查找依赖此值，Getup 填 `DR002JoystickGetupWE11` |
| `--play-root` | 同仓目标填 `sim2sim/we11_play` |
| `--apply` | 写入策略和来源记录；不带此参数只检查、不切换 |

切换目标是 `sim2sim/we11_play/policy/dr002/we11/policy.onnx`；Flat/Getup 共用这个位置。
旧文件备份到 `sim2sim/artifacts/we11-policy-backups/`。重新启动 C++ Play 加载新模型；
仅更换 ONNX 不需要重编译。该脚本不发布到真机，也不生成通用 `aar_sim2sim` release。

### 6.3 PE01/PE02 导出

两套训练脚本结束时自动导出 ONNX 并生成 release，终端打印交付目录。
`policy.onnx` 同时留在 checkpoint 目录。日常无需额外转换命令，
将生成的 release 目录交给下一节的通用 C++ 运行时。

<a id="cpp"></a>

## 7. C++ Sim2Sim

### 7.1 选择入口

| 入口 | 输入模型 | 用途 | MuJoCo |
| --- | --- | --- | --- |
| `sim2sim/we11_play/scripts/play_we11.sh` | 固定读取同仓 WE11 Play 策略位置，切换方法见上一节 | 保留原界面和 Getup 手柄操作 | 暂用 `3.2.7` |
| `sim2sim/build/aar_sim2sim` | 第一个位置参数指定 release 目录 | 通用回放、日志采集、Python/C++ 一致性验证 | `3.8.0`，与 Python 一致 |

### 7.2 WE11 原界面回放

源码变更或首次使用时构建：

```bash
sim2sim/we11_play/build.sh
```

日常运行：

```bash
sim2sim/we11_play/scripts/play_we11.sh --difficulty 1.0
```

| 参数 | 用法 |
| --- | --- |
| `--difficulty 0.8` | 指定 `0..1` 的课程难度；未指定 stage 时使用 home-to-getup 路径 |
| `--stage exact_getup` | 指定阶段；也支持 `home_to_getup`、`getup_with_home`、`balance`、`mixed`，或编号 `1..5` |
| `--stage mixed --difficulty 0.8` | 同时选择阶段与难度；指定 stage 后未填 difficulty 时使用 `1.0` |
| `--autostart` | 自动启动策略，用于冒烟；日常通过键盘或手柄启动 |
| `--help` | 查看脚本支持的参数 |

手柄/键盘见[操作速查](#we11-play)。此入口不接受 checkpoint 路径和扑动频率参数。

### 7.3 通用 release 回放

完成环境安装/重绑定后，增量构建和 C++ 测试：

```bash
cmake --build sim2sim/build --parallel
ctest --test-dir sim2sim/build --output-on-failure
```

仓库 PE01 示例可直接用于无窗口检查：

```bash
sim2sim/build/aar_sim2sim releases/examples/pe01_flat \
  --steps 100 --telemetry logs/sim2sim/pe01-check
```

自己的 PE02 release 实时回放：

```bash
sim2sim/build/aar_sim2sim 'releases/pe02/pe02_flat/<run_id>' \
  --interactive --steps 1000000 --telemetry logs/sim2sim/pe02-play
```

| 参数 | 默认值 | 用法 |
| --- | --- | --- |
| 第一个位置参数 | 必填 | release 目录，包含 `deployment_manifest.json`、ONNX 和资产；不能只传 `.pt` 或 `.onnx` |
| `--steps` | `1` | 策略步数，正整数；日常窗口回放显式设置较大值 |
| `--interactive` | 关闭 | 打开窗口，支持摄像机、鼠标拖动力、手柄和曲线 |
| `--telemetry` | `<release>/telemetry` | 输出目录，写入 `sim2sim.csv` 和 `sim2sim.jsonl`；建议显式放到 `logs/` |

### 7.4 WE11 基线打包与数值对比

下面的工具只打包仓库保留的 Getup 基线 `models/we11/getup/policy.onnx`，
不用于选择任意新 checkpoint。`--destination` 必填，脚本会删除并重建该目录，
使用下面这样的专用缓存目录。

```bash
python tools/build_we11_sim2sim_release.py \
  --destination /ssd/conda/cache/aar_unilab-native/we11-getup-release

python tools/compare_we11_sim2sim.py \
  --binary sim2sim/build/aar_sim2sim \
  --release /ssd/conda/cache/aar_unilab-native/we11-getup-release
```

PE02 使用训练生成的 release：

```bash
python tools/compare_pe02_sim2sim.py \
  --binary sim2sim/build/aar_sim2sim \
  --release 'releases/pe02/pe02_flat/<run_id>' \
  --steps 1 10 100
```

两种对比工具均要求 `--binary` 和 `--release`；PE02 还支持
`--steps`（默认 `1 10 100`）及 `--atol`（默认 `0.00002`）。
更多运行时细节见 [sim2sim/README.md](../sim2sim/README.md)。

<a id="outputs"></a>

## 8. 输出文件位置

| 产物 | 默认位置 |
| --- | --- |
| WE11 Flat run | `logs/rsl_rl_ppo/DR002JoystickFlatWE11/<run_id>/` |
| WE11 Rough run | `logs/rsl_rl_ppo/DR002JoystickRoughWE11/<run_id>/` |
| WE11 Getup run | `logs/rsl_rl_ppo/DR002JoystickGetupWE11/<run_id>/` |
| PE01 run | `logs/pe01_custom_ppo/pe01/pe01_flat/<run_id>/` |
| PE02 run | `logs/pe02_custom_ppo/pe02/pe02_flat/<run_id>/` |
| WE11 checkpoint | 对应 run 下 `model_N.pt` |
| PE01 checkpoint | 正式版 `model_<iteration>.pt`；旧最小适配器 `model_1.pt` |
| PE02 checkpoint | 对应 run 下 `model_<iteration>.pt`，包含完整续训状态 |
| 导出 ONNX | 所选 checkpoint 目录下 `policy.onnx` |
| WE11 视频 | 所选 checkpoint 目录下 `play_video.mp4` |
| PE01/PE02 release | `releases/<robot>/<task>/<run_id>/` |
| PE01/PE02 Python 回放数据 | `play.telemetry` 指定目录下 `telemetry.csv`、`telemetry.jsonl` |
| 通用 C++ 回放数据 | `--telemetry` 指定目录下 `sim2sim.csv`、`sim2sim.jsonl` |

WE11 的 `run_config.json` 保存训练配置及关键参数记录，`run_summary.json` 保存运行摘要。
PE02 的 `training_config.yaml` 与 checkpoint 一起保存；迁移 run 时保留这些配置记录。
WE11 当前正式模型为 `models/we11/getup/model_9999.pt`；旧 Flat/Rough 135D 模型已移除。

<a id="checks"></a>

## 9. 验证与排错

### 9.1 安装与开发检查

基础安装自检（依赖版本、入口/release/PE01 测试、仓库审计和模型校验）：

```bash
bash tools/validate_installation.sh
```

完整安装验证会增加非 slow 测试、类型检查、模型加载、PE01 短训练/导出、
C++ 构建与轨迹对比，并生成相应日志和缓存：

```bash
bash tools/validate_installation.sh --all
```

提交代码前按仓库规则检查：

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy src/unilab
python -m pytest -m "not slow"
git diff --check
```

### 9.2 常见问题

| 现象 | 处理方法 |
| --- | --- |
| 找不到 `unilab` 或仍导入旧仓库 | 先激活 SSD 环境；移动/切换仓库后运行 `bash tools/rebind_environment.sh` |
| 提示 `missing selectors` | 统一 Train/Play 入口必须填写第 2 节的六个参数 |
| task/profile 不匹配 | 按第 2 节同一行组合选择；WE11 用 `task=getup`，PE02 用 `task=pe02_flat` |
| WE11 显存不足 | 减小 `algo.num_envs`；用 `CUDA_VISIBLE_DEVICES` 选择可用 GPU |
| WE11 设置 `training.device=cpu` 仍使用 GPU | 当前脚本自动选择设备，未读取此配置项；本机可设置 `CUDA_VISIBLE_DEVICES=""` 隐藏 CUDA |
| 找不到 checkpoint | 使用实际 `.pt` 完整路径；核对任务与 run 目录，`model_N.pt` 是待替换示例 |
| WE11 窗口中机器人不动 | 按 `1`/Enter 或 `RB+DPadUp` 启动策略 |
| WE11 窗口运行一会儿自动退出 | 将 `training.play_steps` 改为 `null`；录像仍须有限步数 |
| WE11 没弹窗而是生成视频 | 将 `training.play_render_mode=auto` 改为 `interactive` |
| WE11 `none` 模式没有运行轨迹 | 这是加载/导出模式；验证轨迹可用通用 C++ `--steps` 或 Python 录像 |
| PE01/PE02 无窗口仍弹出图表 | 同时设置 `play.render=none play.plot=false` |
| 旧 WE11 模型维度不匹配 | 当前只支持 145D；使用当前 Getup 模型或重新训练 Flat/Rough |
| 回放提示训练参数不一致 | 对照源 run 的配置和所选 task/网络，保留 `training.sim2sim_strict=true` 进行核对 |
| WE11 C++ 仍播放旧模型 | 导出 ONNX 后按第 6 节切换同仓 Play 位置，重新启动程序 |
| C++ 找不到构建目录 | 先执行 `bash tools/rebind_environment.sh` 配置并构建通用运行时 |

实测手柄、摄像机和拖动力体验需要在有显示器和手柄的环境中检查。
