# PE04：PE03 本体与 TRON1 PF BlindFlat 训练设计

状态（2026-09-24）：**TRON1 基线暂缓**。保留 PE04 训练、checkpoint 回放和
部署入口；后续新基线转向 PE05（PE03 本体、原始 PE01 训练设计）。

2026-09-24 提交前重新验证：PE04 专项与 WE11 flat/rough 共 28 passed；
本次 29 个 Python 文件 Ruff 格式和 lint 通过，mypy 与 `git diff --check` 通过。
全仓非 slow 测试为 533 passed、4 failed、2 xfailed；失败仍为下文记录的
三个 PE03 随机化课程期望与配置不一致，以及只读参考目录内嵌套 Git 元数据。
全仓 Ruff 仍有原有 11 个格式问题和 7 个 lint 问题。原始输出位于
`logs/pe05_validation/before_*.log`（不纳入提交）。

PE04 是独立训练环境。机器人资产冻结自 PE03 最新的 `pe03-cnc-joint-limits-v3`；观测、速度估计网络、非对称 Actor/Critic、PPO、命令与步态采样、奖励和随机化参考 TRON1 PF BlindFlat。它不导入或继承 PE01/PE02/PE03 的环境、算法、配置或 C++ runtime，也不在运行或构建时使用 `references/` 中的代码。

## 使用

2026-09-23 的训练速度优化及可选 BF16 更新见 [PE04_TRAINING_PERFORMANCE.md](PE04_TRAINING_PERFORMANCE.md)。默认仍为 FP32 更新。

在仓库根目录运行。环境由 `bash tools/install_environment.sh` 安装，Python 使用 `/ssd/conda/envs/aar_unilab/bin/python`，不需要 `conda init`。

```bash
# 查看正式训练配置；默认 4096 环境、24 步 rollout、2000 轮
bash tools/train.sh pe04_tron1 --dry-run
bash tools/train.sh pe04_tron1

# 快速验收：32 环境、24 步、10 轮
bash tools/train.sh pe04_smoke

# max_iterations 表示本次额外训练轮数；checkpoint 使用实际路径
bash tools/train.sh pe04_smoke algo.max_iterations=2 \
  training.resume=logs/pe04_smoke/<run>/model_10.pt

# 正式训练最近一次 checkpoint 的交互回放，默认游戏手柄
bash tools/train.sh pe04_tron1 mode=play checkpoint=-1

# 无窗口回放；smoke 与正式训练使用各自的日志目录
bash tools/train.sh pe04_smoke mode=play checkpoint=-1 \
  play.render=headless play.steps=100 play.command_source=fixed \
  'play.command=[0.2,-0.1,0.3]'
```

训练结束默认导出自包含 release 到 `releases/pe04/pe04_flat/<run>/`，包括 ONNX、完整运行配置、资产、manifest 和非零命令的 golden 输入输出。续训保存优化器、随机数、环境和历史帧状态；机器人标识、资产指纹、观测布局和训练契约不一致时拒绝续训。Python checkpoint 加载也校验资产与布局。release 可以移动到其他路径。

```bash
sim2sim/build/aar_sim2sim releases/pe04/pe04_flat/<run> --steps 100
env -u PYTHONPATH PYTHONNOUSERSITE=1 /ssd/conda/envs/aar_unilab/bin/python \
  tools/compare_pe04_sim2sim.py --binary sim2sim/build/aar_sim2sim \
  --release releases/pe04/pe04_flat/<run> --steps 1 10 100
```

## 独立文件边界

| 内容 | PE04 所有者 |
| --- | --- |
| 任务、控制、观测、奖励、随机化 | `conf/pe04/task/pe04_flat.yaml` |
| 网络、PPO、训练与回放设置 | `conf/pe04/config.yaml` |
| 环境、观测、奖励、随机化、Python 回放 | `src/unilab/envs/locomotion/pe04/` |
| 网络、PPO、runner、日志 | `src/unilab/algos/torch/pe04/` |
| checkpoint 适配 | `src/unilab/adapters/pe04_ppo.py` |
| 训练、回放、ONNX 导出入口 | `scripts/train_pe04.py` |
| 独立机器人文件及来源哈希 | `src/unilab/assets/robots/pe04/` |
| C++ 观测、历史、动作与 PD | `sim2sim/include/aar/pe04_runtime.hpp` |
| 专项测试 | `tests/pe04/` |

复用的公共设施限于 catalog、启动器、MuJoCo backend、交互窗口、release 容器和 C++ 通用加载器。新机器人经 catalog 注册，通用入口不硬编码 PE04 的网络维数。

## 观测与网络

| 项目 | 实现 |
| --- | --- |
| Actor 单帧 | 30 维：机体角速度 3、投影重力 3、相对默认关节位置 6、关节速度 6、上一次原始动作 6、相位 sin/cos 2、步态参数 4 |
| 历史帧 | 10 × 30 = 300 维，逐帧排列，旧帧在前、当前帧在后；reset 清除对应环境历史 |
| 命令 | `[vx, vy, yaw_rate]` 独立 3 维输入，不混入历史，不加噪声 |
| Encoder | `300 → 256 → 128 → 3`，ELU；监督目标是真实机体线速度 |
| Actor | `估计速度 3 + 当前观测 30 + 命令 3 = 36`，`512 → 256 → 128 → 6`，ELU |
| Critic | 特权观测 267 + 命令 3 = 270，`512 → 256 → 128 → 1`，ELU |
| 梯度 | Actor 使用 detach 后的速度估计；Encoder 使用独立优化器和速度 MSE |
| 噪声 | 角速度/重力/关节位置/关节速度高斯标准差分别为 `0.05/0.025/0.01/0.01`；独立采样当前观测组和历史观测组的噪声 |
| 缩放 | 先加噪、裁剪到 ±100，再对角速度乘 0.25、关节速度乘 0.05；动作历史保存原始网络输出 |

Critic 的 267 维按 `observations.py::critic_layout()` 固定顺序：真实机体线速度 3、无噪声单帧 30、关节力矩 6、关节加速度 6、全 9 个刚体的 4 帧三维接触力 108、名义质量 9、名义完整惯量矩阵 81、名义 Kp/Kd 各 6、世界系根位置 3、世界系根速度 6、重复根位置 3。

这里保留了参考代码的名义物理参数观测及重复位置字段；接触力覆盖所有刚体。MuJoCo 不支持与 PhysX 等价的静/动摩擦拆分及 restitution，因此完全删去这类特权通道，不用常数占位，也不伪造随机化。

接触观测以 200 Hz 更新，保留最近 4 个采样，最新采样在前；策略 50 Hz、物理和 PD 均为 400 Hz。WE11 原有的 400/200 Hz 控制不变。

## 保留本体与适配项

| 本体项 | PE04 设置 |
| --- | --- |
| 资产 | 独立复制 PE03 的 joint-limits-v3 XML、碰撞网格、视觉网格与 URDF；不使用软链接 |
| 关节顺序 | `L_hip_, L_thigh_, L_calf_, R_hip_, R_thigh_, R_calf_` |
| 站姿与高度 | PE03 最新 `home` 关键帧；高度目标从该关键帧读取，约 0.29138 m |
| PD | 每侧 Kp `[4.3, 4.3, 4.9]`，Kd `[0.34, 0.34, 0.24]` |
| 力矩限制 | 每侧 `[5.5, 5.5, 14.0]` Nm |
| 动作 | 缩放 0.25，叠加默认姿势，再按 PE03 物理关节范围限制目标 |
| 关节速度 | 保留 PE03 位置差分观测；动力学惩罚使用实际关节速度、实际加速度与力矩 |
| 小本体步态参数 | 摆脚高 0.02–0.10 m，最小脚距 0.08 m，落脚阈值 0.05 m，接触力/速度奖励 sigma 为 5.0/0.20 |

训练是平地盲走；未引入 PE03 的 WTW 阶段切换。步态频率 1.5–2.5 Hz、左右相位偏移 0.5、支撑比 0.5，每 5 秒采样一次。相位沿用 TRON1 的 `episode_time × 当前频率` 计算方式。速度命令范围为前后 ±1.5 m/s、横向 ±1.0 m/s、偏航 ±0.5 rad/s，采样间隔 0–5 秒，20% 概率站立命令。

PPO 使用 5 epochs、4 minibatches、γ=0.99、λ=0.95、clip=0.2、初始学习率 1e-3、KL=0.01 自适应学习率、entropy=0.01、梯度范数上限 1.0。奖励按 0.02 秒策略步长积分，保留参考源码的具体公式，例如 base-height 实际为绝对误差，尽管参考函数名称带 `l2`。奖励权重见任务 YAML。

随机化在启动时采样质量、惯量、COM、PD 和单一滑动摩擦系数；保留参考中先质量、再质量与惯量联合缩放的顺序。reset 随机化关节、平移、偏航和根速度。推扰以每策略步 0.002 的概率采样并按本体质量/惯量换算为力和力矩。

## 后端与构建

PE04 使用通用 `TrainingRobotSimulation` 扩展提供实际动力学加速度及接触采样。原生 pool 原有的 post-step sensor forward 会清空控制和外力，因此新增可选 `forward_controlled` 接口，用实际控制与外力在最终状态计算观测；不推进额外物理时间。仅 PE04 调用该接口，旧环境保持原来的路径。扩展补丁随受支持环境安装脚本构建。

C++ 构建所需的 ONNX Runtime 1.22.0 / GLFW 3.4 公开 SDK 头文件改为下载到 SSD 依赖缓存；构建不再读取 `references/` 中的头文件。

## 来源与验证

设计依据是只读参考目录 `references/tron1-rl-isaaclab/` 中：

- `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/PF/limx_base_env_cfg.py`
- 同目录树 `robots/limx_pointfoot_env_cfg.py`、`agents/limx_rsl_rl_ppo_cfg.py`、`mdp/observations.py`、`mdp/rewards.py`、`mdp/events.py`、`mdp/commands/`
- `rsl_rl/rsl_rl/modules/` 和 `rsl_rl/rsl_rl/algorithm/ppo.py`

2026-09-23 验证记录：

- PE04 专项覆盖独立性、观测噪声/历史与命令、接触/奖励、真实加速度、Native/Python PD、随机化、环境快照、Encoder 梯度隔离、GAE、三网络参数更新、续训、搬移 release 和 ONNX 一致性。
- 完整短训 `32 × 24 × 10`：`logs/pe04_smoke/2026-09-23_13-36-14_599213_mujoco/model_10.pt`。
- 续训 2 轮：`logs/pe04_smoke/2026-09-23_13-36-39_021445_mujoco/model_12.pt`。
- Python 无窗口回放 100 步成功；Python/C++ 闭环 1/10/100 步的动作、控制力矩与高度最大误差均为 0。
- 额外 60 步 C++/Python 全历史、单帧和控制对比覆盖 25 ms 延迟与超限动作。
- 最终专项及 WE11 基线回归：`tests/pe04` 加 WE11 PACE/rough 测试共 **21 passed**，包含两个 WE11 任务的 compose、初始化与 step。
- `mypy src/unilab` 通过（159 个源文件），本次修改的 Python 文件格式与 lint 通过，`git diff --check` 通过。
- 全仓 `pytest -m "not slow"`：**526 passed、4 failed、2 xfailed**。三个失败来自未修改的 `test_pe03_robustness_curriculum.py`：测试期望最高随机化等级 1.0，现有 `conf/pe03/task/pe03_gait_flat.yaml` 配置为 0.7。另一个是仓库审计发现两个参考项目已有的嵌套 `.git`。PE04 不修改这些测试、PE03 配置或只读参考目录。
- 全仓 Ruff 仍有已有问题：11 个文件格式不符、7 个 lint 错误，位于 PACE、WE11 工具及旧环境文件；不属于 PE04 代码。

这些短训验证链路和数值契约，尚未验证长期训练的步行收敛或实机效果。
