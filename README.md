# Aar_Unilab

多样机 MuJoCo 训练、Python 回放与 C++ Sim2Sim 仓库。
WE11 Flat/Rough/Getup 是当前正式主线。PE02 已按原始 PE01 训练系统迁移独立的
多环境 PPO、速度估计 encoder、任务环境和续训；PE01 也已完成独立的完整训练迁移。
PE02 的迁移范围与物理差异见[迁移说明](docs/PE02_TRAINING_MIGRATION.md)，尚未验收稳定步态。
PE03 CNC 点足样机已从当前 PE02 复制为独立训练线，使用新质量、碰撞体和默认站姿；
训练、回放及验证结果见 [PE03 说明](docs/PE03_TRAINING_MIGRATION.md)。
点足新基线为 [PE05](docs/PE05_BASELINE.md)：冻结 PE03 最新本体，参考原始 PE01
紧凑网络；当前使用[加权奖励与本体适配](docs/PE05_WEIGHTED_TASK.md)，暂关动力学随机化，启用独立速度命令课程，20% 零命令。PE04/TRON1 暂缓，既有训练和回放入口保留。

## 按操作查命令

日常命令、参数取值、输出位置和排错集中在 **[日常使用指南](docs/DAILY_USAGE.md)**。

| 我要做什么 | 文档入口 |
| --- | --- |
| 安装环境、打开新终端、移动仓库 | [环境与命令写法](docs/DAILY_USAGE.md#environment) |
| 选择机器人、任务和算法 | [入口参数与支持的组合](docs/DAILY_USAGE.md#selectors) |
| WE11 训练、续训、查看 TensorBoard | [WE11 训练参数](docs/DAILY_USAGE.md#we11-training) |
| WE11 实时回放、录制视频、调整起始姿态 | [WE11 Python 回放](docs/DAILY_USAGE.md#we11-play) |
| PE01/PE02 训练、回放、修改网络 | [PE01 与 PE02](docs/DAILY_USAGE.md#pe-training) |
| PE03 CNC 站立、行走与模型检查 | [PE03 独立训练线](docs/PE03_TRAINING_MIGRATION.md) |
| PE05 训练、回放、TensorBoard 与日志目录 | [PE05 操作指南](docs/PE05_WORKFLOW.md) |
| PE05 当前训练规则、奖励公式与改造讨论 | [训练规则与讨论基准](docs/PE05_TRAINING_RULES.md) |
| 导出 ONNX、把新模型切换到 WE11 C++ Play | [导出与模型切换](docs/DAILY_USAGE.md#export) |
| C++ 回放、读取 release、对比 Python/C++ | [C++ Sim2Sim](docs/DAILY_USAGE.md#cpp) |
| 找 checkpoint、视频和日志 | [产物位置](docs/DAILY_USAGE.md#outputs) |
| 检查安装或排查常见问题 | [验证与排错](docs/DAILY_USAGE.md#checks) |

## 快速开始

所有命令从仓库根目录执行。首次安装：

```bash
bash tools/install_environment.sh
```

每次打开新终端：

```bash
cd /ssd/Aar_Unilab
source tools/activate_environment.sh
```

环境位于 `/ssd/conda/envs/aar_unilab`。新 clone、移动仓库或切换副本后，执行
`bash tools/rebind_environment.sh` 重新绑定。不要使用旧的 `unilab_cuda` 环境。

### WE11 训练

本机常用设置保存在 [conf/train_defaults.yaml](conf/train_defaults.yaml)：GPU 0、
32 个仿真线程、4096 环境、每环境 24 步、1000 轮、TensorBoard，每 100 轮保存。
这个启动方式自动使用 SSD Python 并清理 ROS 的 Python 路径，无需激活环境或逐条写环境变量。

```bash
bash tools/train.sh we11_getup
```

PE01 和 PE02 行走分别使用 `bash tools/train.sh pe01`、
`bash tools/train.sh pe02_walking`。追加 `algo.max_iterations=2000` 等参数可临时覆盖文件，
追加 `--dry-run` 只预览最终参数。各预设和环境变量优先级见
[启动默认参数说明](docs/DAILY_USAGE.md#train-defaults)。
完整六项选择器入口仍支持 `task=flat` / `rough` / `getup`，见日常使用指南。

PE03 站立和无步态时钟行走分别使用 `bash tools/train.sh pe03_standing`、
`bash tools/train.sh pe03_walking`，默认均为 4096 环境、24 步、1000 轮、32 个仿真线程。

`algo.num_envs` 是并行环境数，`algo.max_iterations` 是本次训练轮数，
`algo.save_interval` 是 checkpoint 保存间隔。续训时追加
`algo.load_run=<run目录或model_N.pt路径>`；完整示例见[续训说明](docs/DAILY_USAGE.md#we11-training)。

### WE11 实时回放

回放仓库保留的 Getup 模型，保持窗口运行直到手动关闭：

```bash
AAR_EXPORT_POLICY=0 python scripts/play.py \
  robot=we11 task=getup observation=we11_v2_145 \
  policy=we11_mlp algorithm=rsl_rl_ppo simulator=mujoco \
  algo.load_run=models/we11/getup/model_9999.pt \
  training.play_render_mode=interactive training.play_env_num=1 \
  training.play_steps=null
```

按 `1`/Enter 或 `RB+DPadUp` 启动，`P`/`LB+X` 停止，`R`/`RB+Y` 重置。
`AAR_EXPORT_POLICY=0` 表示本次只回放；导出 ONNX 时改为 `1`。
当前 WE11 全部使用 `we11_v2_145`，旧 135D 模型已不再支持。

### PE01 / PE02 多环境训练

PE01 已按原始完整训练系统迁移，使用独立环境、网络与配置：

```bash
python scripts/train_pe01.py algo.num_envs=4096 algo.max_iterations=15000
```

参数对照、续训和旧 checkpoint 兼容见 [PE01 迁移说明](docs/PE01_TRAINING_MIGRATION.md)。
PE02 使用另一套独立实现。当前无时钟步态的 walking 训练：

```bash
bash tools/train.sh pe02_walking algo.max_iterations=500
```

walking 的奖励、24 维单帧观测及弱失败机制见
[配置说明](docs/PE02_WALKING_VALIDATION.md)。原始迁移基线仍可独立启动：

```bash
python scripts/train.py \
  robot=pe02 task=pe02_flat observation=pe02_v2 \
  policy=pe02_encoder_mlp algorithm=pe02_custom_ppo simulator=mujoco \
  algo.num_envs=256 algo.num_steps_per_env=24 algo.max_iterations=15000
```

结束时打印 checkpoint 和 release 路径。原版默认环境数为 8192；本机示例显式使用
256 用于快速调试。碰撞 v2 的大规模吞吐和内存见
[PE02 性能优化报告](docs/PE02_COLLISION_OPTIMIZATION.md)。`num_steps_per_env` 与 `max_iterations`
沿用 WE11 的参数含义。续训、回放和源代码核对见[迁移说明](docs/PE02_TRAINING_MIGRATION.md)。

### WE11 C++ Getup 回放

```bash
sim2sim/we11_play/build.sh
sim2sim/we11_play/scripts/play_we11.sh --difficulty 1.0
```

这条入口保留原 MuJoCo `simulate` 界面，暂用 MuJoCo **3.2.7**；
通用 `aar_sim2sim` 与 Python 训练使用 **3.8.0**。
两条 C++ 路径的模型选择方法不同，见[C++ 使用说明](docs/DAILY_USAGE.md#cpp)。

## 修改配置与查资料

| 内容 | 位置 |
| --- | --- |
| WE11 公共训练参数 | `conf/ppo/config.yaml` |
| WE11 任务、奖励、控制与网络 | `conf/ppo/task/dr002_joystick_{flat,rough,getup}_we11/`；任务配置会覆盖公共默认值 |
| WE11 内存与线程测速 | [2048/4096 环境、16/24/32 线程测试](docs/WE11_TRAINING_BENCHMARK.md) |
| PE01 训练、网络与环境 | [conf/pe01/config.yaml](conf/pe01/config.yaml)、[任务配置](conf/pe01/task/pe01_flat.yaml) |
| PE02 训练与网络 | [conf/pe02/config.yaml](conf/pe02/config.yaml) |
| PE02 环境、控制与奖励 | [conf/pe02/task/pe02_flat.yaml](conf/pe02/task/pe02_flat.yaml) |
| 训练输出 | `logs/`；[各样机路径](docs/DAILY_USAGE.md#outputs) |
| WE11 正式模型 | [models/we11/README.md](models/we11/README.md) |
| C++ 通用运行时 | [sim2sim/README.md](sim2sim/README.md) |
| PE02 资产与站姿 | [资产说明](src/unilab/assets/robots/pe02/README.md)、[站姿计算](src/unilab/assets/robots/pe02/analysis/STANDING_POSE.md) |
| WE11 控制与观测约定 | [WE11_HANDOVER.md](WE11_HANDOVER.md)；日常安装命令以本 README 为准 |
| 仓库改造背景 | [改造记录](docs/仓库规整与训练复用性改造计划.md) |

`references/` 保存 Deploy、旧 Play 和 PE01 前身的源码参考；实际真机部署仍在
Deploy 工程完成。安装自检使用 `bash tools/validate_installation.sh`，完整验证见
[验证命令](docs/DAILY_USAGE.md#checks)。
