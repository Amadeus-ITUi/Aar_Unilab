# PE05：PE03 CNC 本体与原始 PE01 训练基线

状态：当前点足新基线。PE04/TRON1 已冻结并暂缓，保留其独立训练和回放入口。
PE05 完成训练/续训/部署链路验收，尚未验证长期步行收敛或实机效果。
WE11 Flat/Rough 的物理、控制及观测契约不变。

## 来源与边界

主要核对源为 `/home/angela/ssd/Point_Phoenix/point-legged`，提交
`17a4b93a7480233de07d52ccb5c790cfdb1d0286`。核对了原始 Dragon 配置、
环境和 base task、Actor/Critic、Encoder、PPO、rollout storage、runner 与训练入口。
`docs/PE05_SOURCE_MANIFEST.json` 保存这些文件的工作区/HEAD SHA256、原始配置字面值和
源码本地差异。核心配置、环境和算法未修改；原仓库训练入口有本地修改，差异已记录。
该修改删除了 HEAD 训练入口强制设置的 50000 轮，恢复配置中的 15000 轮；PE05 使用 15000 轮。
原仓库保持只读，没有导入或执行它的代码。

PE05 以本仓库已迁移的 PE01 实现为代码起点，重新核对原始源码并独立适配。
配置、环境、算法、checkpoint 适配器、训练入口、Python 回放和 C++ runtime
归 PE05 所有，不导入/继承 PE01–PE04 实现。保留 BSD-3-Clause 署名和
`LICENSE.pe01`。仅共享 catalog、启动器、MuJoCo backend、交互窗口与 release 容器。

本体直接冻结自 PE03 `pe03-cnc-joint-limits-v3`，没有软链接。
资产 `provenance.json` 记录源文件哈希；质量、惯量、网格、自碰撞、机械零位和
关节范围保持 PE03，不使用 PE01 的简化本体。全机质量约 3.26689 kg。
机器人 XML 保持可复用；home 关键帧归任务 scene。

## 原始行为与适配

| 项目 | 原始 PE01 | PE05 与原因 |
| --- | --- | --- |
| 本体与站姿 | DRAGON_3，reset 高度 0.32 m，高度目标 0.25 m | PE03 最新 home，reset 与奖励目标均约 0.29138291668 m |
| 关节顺序 | 左髋/大腿/小腿，右髋/大腿/小腿 | `L_hip_, L_thigh_, L_calf_, R_hip_, R_thigh_, R_calf_`，保留真实装配角度 |
| 控制频率 | 物理/PD/策略 400/400/50 Hz | 相同；不影响 WE11 400/200 Hz |
| PD、力矩 | 每侧 Kp `[4.3,4.3,4.9]`、Kd `[0.34,0.34,0.24]`、限矩 `[5.5,5.5,14]` | 相同，仍是初始参数，未经 CNC 实机辨识 |
| 动作 | scale 0.25；用平均 PD 增益和 14 Nm 限制目标偏移 | 保留；额外裁剪原始动作至 ±100，并将最终目标裁剪到 PE03 物理关节范围；随机零位纳入目标 |
| 动作历史 | 保存限幅后的目标偏移/action scale | 相同，训练/导出/回放保持一致 |
| 命令 | 每 5 s 采样；`vx ±1`、`vy ±0.6`、yaw 采样 ±1；heading 模式按包裹航向误差计算 yaw | 相同；heading 计算结果不额外限制到 yaw 采样范围 |
| 零命令比例 | 配置为 0.1，但 `random > 0.1` 实际约 90% 置零 | 显式 `zero_probability=0.9`；按用户选择保留实际行为，零命令仍保留航向保持 |
| 课程 | plane 下 terrain curriculum 被关闭，命令课程依赖该开关 | 不启用地形或命令课程，不引入 WTW/TRON1 阶段 |
| 步态 | 2 Hz；offset 最终强制 0.5；duration 0.5；swing height 0.06 m | 相同，每 5 s 重采样，连续相位积分 |
| Episode | 20 s；累计失败 tick 超过 0.5 s；接触力 >5 N 或重力 z >−0.1 | 相同，不要求连续失败；功率阈值原版未参与终止，PE05 也不启用 |
| 观测 | 30 维单帧、10 帧历史、独立 3 维命令 | 相同；当前帧与历史末帧一致，Critic 使用当前无噪声帧，修正原版旧帧引用及 IMU 更新顺序 |
| Reset | 原版子集命令置零索引和时序有问题 | 使用真实 env ids，重置对应历史/FIFO，reset 不额外推进物理；保留其他环境状态 |
| GAE | 原版 timeout/bootstrap 与 reset 边界不完整 | 从 final observation 引导 timeout，优势不跨 episode 传播 |
| 足端高度 | 球心高度减 0.025 m 半径 | 使用 PE03 实际足底几何的最低点；网格/元数据仅冷路径缓存 |
| 动力学观测 | 位置差分关节速度、策略步差分关节加速度和机体/足端速度 | 保留差分公式，足端采用 link frame；接触从公共 backend 获取受实际控制作用的最终采样 |
| 18 项奖励 | 原版权重与公式，按 0.02 s 积分、单项/总值裁剪 | 保留；高度误差平方、机体系横向脚距、落脚检测使用向上垂直接触力；不是 TRON1 奖励 |
| 随机化 | 质量、惯量、COM、摩擦、PD、力矩系数、零位、延迟、IMU、推扰 | 保留有效范围与启动采样；延迟 25–50 ms，每物理 tick FIFO；每 5 s 施加一次推扰 |
| 接触求解 | PhysX 静/动摩擦、restitution，禁用自碰撞 | MuJoCo 单滑动摩擦、保留 PE03 自碰撞；同时随机化地面和本体，防止名义地面摩擦通过 max 合成掩盖低摩擦采样；不伪造 restitution，也不声称求解器等价 |

任务、控制、奖励和随机化参数归 `conf/pe05/task/pe05_flat.yaml`；
网络、PPO、训练与回放参数归 `conf/pe05/config.yaml`。

网络契约：Encoder `300→256→128→3`；Actor `36→512→256→128→6`；
Critic `39→512→256→128→1`，ELU。Actor 输入为估计速度、当前帧、命令；
Critic 输入为真实速度、当前无噪声帧、命令、估计速度。
Encoder 在 PPO 更新期间冻结，再以真实机体速度独立 Adam/MSE 更新。
PPO 5 epochs、4 minibatches、γ=0.99、λ=0.95、clip=0.2、初始 LR=1e-3、
KL=0.01 自适应学习率、entropy=0.01、梯度范数上限 1.0。
迁移实现用总体标准差归一化优势，并在 batch 不能整除时保留剩余样本；原版分别使用
样本标准差和丢弃余数。默认训练规模可整除。随机采样使用 NumPy/Torch，
不声明与 Isaac Gym 原训练逐样本或逐参数更新一致。

## 使用

首次安装或扩展缺失时运行 `bash tools/install_environment.sh`；所有命令从仓库根目录执行，
使用 `/ssd/conda/envs/aar_unilab`，无需 `conda init`。

```bash
# 本机正式预设：4096 环境、24 步、15000 轮，每 400 轮保存
bash tools/train.sh pe05 --dry-run
bash tools/train.sh pe05

# 快速验收：32 环境、24 步、10 轮
bash tools/train.sh pe05_smoke

# max_iterations 为本次追加轮数；保留原训练契约/环境数
bash tools/train.sh pe05_smoke algo.max_iterations=2 \
  training.resume=logs/pe05_smoke/<run>/model_10.pt

# 正式预设最近 checkpoint，默认手柄、窗口持续运行
bash tools/train.sh pe05 mode=play checkpoint=-1

# smoke 最近 checkpoint，固定命令，无窗口回放
bash tools/train.sh pe05_smoke mode=play checkpoint=-1 \
  play.render=headless play.steps=100 play.command_source=fixed \
  'play.command=[0.2,-0.1,0.3]'
```

直接运行训练脚本时，owner 默认值为原版 8192 环境、24 步、15000 轮。
统一选择器是 `robot=pe05 task=pe05_flat observation=pe05_v1
policy=pe05_encoder_mlp algorithm=pe05_custom_ppo simulator=mujoco`。
观测仍通过 `NpEnvState.obs` 字典提供；catalog 的 Critic 维数是环境输出 33，
网络拼接命令和估计速度后才是 39。

checkpoint 保存三个网络、两个优化器、计数、RNG、随机化、环境、历史和延迟 FIFO。
训练/回放加载校验 PE05 标识、资产指纹、观测布局和训练配置契约；拒绝旧基线模型。
`checkpoint=-1` 仅在当前 profile 的日志根目录选择最近修改的 checkpoint。

训练自动导出完整 Encoder+Actor ONNX、自包含机器人资产、runtime JSON、Hydra 配置、
manifest 和非零 golden 输入。可搬移 release，不依赖原始仓库或其他 PE 基线资产。
golden 命令固定为 `[0.2,-0.1,0.3]`，用于 C++ 默认回放和一致性验证；
Python 交互回放命令由手柄或 `play.command` 控制。

```bash
sim2sim/build/aar_sim2sim releases/pe05/pe05_flat/<run> --steps 100
env -u PYTHONPATH PYTHONNOUSERSITE=1 /ssd/conda/envs/aar_unilab/bin/python \
  tools/compare_pe05_sim2sim.py --binary sim2sim/build/aar_sim2sim \
  --release releases/pe05/pe05_flat/<run> --steps 1 10 100
```

## 验证记录（2026-09-24）

- PE05 23 项专项通过：来源参数、资产快照、运行独立性、启动配置、观测/历史、
  零命令比例、子集 reset、目标限位、随机化/延迟、奖励/终止、GAE、网络更新、
  精确续训、checkpoint 拒绝不兼容，以及异地 ONNX/C++ 回放。
- `32×24×10` 短训生成
  `logs/pe05_smoke/2026-09-24_19-00-08_548357_mujoco/model_10.pt`，7680 样本。
- 追加 2 轮生成
  `logs/pe05_smoke/2026-09-24_19-00-13_391439_mujoco/model_12.pt`，累计 9216 样本；
  两次均输出 TensorBoard、JSONL、checkpoint 和 release。最近 checkpoint 已完成
  非零命令下 100 步无窗口 Python 回放。
- 异地 release 在 25/50 ms 延迟下各完成 100 步 Torch/ONNX 对照，
  1/10/100 步 Python/C++ 闭环通过绝对误差 2e-5 检查；另有 60 步完整历史、
  单帧和控制力矩对照，覆盖原始动作 1000 的极端输入。
- 最终续训 release 的 Python/C++ 1/10/100 步动作、控制力矩和高度最大误差均为 0。
  两个优化器累计各更新 240 次，TensorBoard 续训事件步数为 11–12。
- C++ 构建及 6 项 CTest 通过；20 个新增/修改 Python 文件的 Ruff 格式和 lint 通过；
  mypy 通过（172 个源文件），暂存区 `git diff --check` 通过。
- 最终全仓非 slow 测试：**556 passed、4 failed、2 xfailed**。包含 PE04 回归及
  WE11 flat/rough 的配置组合、初始化与 step。失败与实施前相同：三个 PE03 课程测试
  期望随机化上限 1.0，而原配置为 0.7；仓库审计发现只读参考目录中两个已有 `.git`。
  没有新增失败原因。全仓 Ruff 仍有原有 11 个格式问题、7 个 lint 问题，均未扩入本次修改。

原始检查输出位于 `logs/pe05_validation/`，生成产物不提交 Git。
短训只验证系统与数值契约，不代表策略已稳定站立或行走。
