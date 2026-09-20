# PE02 原始 PE01 训练系统迁移

PE02 现在有独立的多环境 MuJoCo 训练环境、PPO runner、速度估计 encoder、
配置、checkpoint 续训和 Python/C++ 部署合同。本文记录原始 `pe02_v2` 迁移基线。
当前 `+experiment=walking` 使用去掉时钟步态的 `pe02_v3`，见
[walking 配置说明](PE02_WALKING_VALIDATION.md)；下文基线数值不代表当前 walking。
原 `pe02_v1` 单环境最小适配器保留用于旧 checkpoint 兼容，不能与正式训练混用。

最新短训、初始化诊断及长训前待办见
[PE02_TRAINING_READINESS.md](PE02_TRAINING_READINESS.md)。

基准是 `references/pe01/legacy_training` 保存的原始 PE01 / DRAGON 源码，来源提交
`17a4b93a7480233de07d52ccb5c790cfdb1d0286`。核对过程只读取参考源码，生产模块
不导入、不执行 `references/` 或 PE01 模块。派生代码保留 BSD-3-Clause 署名及
`LICENSE.pe01`。本次没有修改 PE01、WE11 的训练实现、配置或机器人资产。

## 启动正式训练

使用仓库支持的环境。以下为本机 32 GB 内存的起步配置，显式使用 256 个环境：

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 \
  /ssd/conda/envs/aar_unilab/bin/python scripts/train_pe02.py \
  algo.num_envs=256 algo.num_steps_per_env=24 algo.max_iterations=15000
```

通过统一 catalog 入口启动同一实现：

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 \
  /ssd/conda/envs/aar_unilab/bin/python scripts/train.py \
  robot=pe02 task=pe02_flat observation=pe02_v2 \
  policy=pe02_encoder_mlp algorithm=pe02_custom_ppo simulator=mujoco \
  algo.num_envs=256 algo.num_steps_per_env=24 algo.max_iterations=15000
```

`training.device=auto` 在有 CUDA 时将网络放到 GPU，否则使用 CPU；MuJoCo 物理仍在
CPU 上。`training.mujoco_threads=16` 控制物理线程，`training.cpu_threads=1` 控制
PyTorch CPU 线程。`training.export=false` 可跳过训练结束后的 ONNX/release 导出。

默认 `algo.num_envs=8192` 保留原始 PE01 参数，没有自动改成单环境。当前 MuJoCo
pool 支持各环境独立质量、惯量和摩擦。碰撞 v2 优化后约需 0.666 MB/模型，
8192 份约为 5.46 GB（十进制），尚未计入轨迹、网络和其他内存；碰撞 v1 的
31.2 GB 模型占用已不适用于新版。启动仍检查可用内存，不会静默减少环境数。
完整吞吐、内存实测见 [性能优化报告](PE02_COLLISION_OPTIMIZATION.md)。

正式训练默认 `training.native_pd=true`，使用独立的 C++ 关节位置 PD 批量入口，
保持原角度偏移 FIFO、位置差分速度、力矩裁剪和末步传感器语义。旧扩展缺少此
入口时会警告并回退到 Python 子步循环；运行
`UNILAB_PYTHON=/ssd/conda/envs/aar_unilab/bin/python bash third_party/mujoco_uni_mixed_pd/build_extension.sh`
即可构建。PE01 和 WE11 的控制入口未切换到这个新路径。

每轮样本数为 `num_envs × num_steps_per_env`。256 × 24 是 6144 条；8192 × 24
是 196608 条。24 是环境/策略步，每个策略步包含 8 个物理步。

## 源码核对与迁移结果

| 内容 | 原始 PE01 | PE02 正式版 |
| --- | --- | --- |
| 环境数 / rollout / 迭代数 | 8192 / 24 / 15000 | 同默认值，可独立覆盖 |
| PPO epochs / minibatches | 5 / 4 | 每轮 20 次 PPO 更新 |
| GAE | gamma 0.99、lambda 0.95 | 保留，明确区分失败与超时 |
| PPO 损失 | clip 0.2、value coef 1、entropy coef 0.01、梯度范数 1 | 保留 |
| 学习率 | 0.001，KL 自适应，目标 KL 0.01 | 保留；上下界 1e-5 / 1e-2 |
| Encoder | 300 → 256 → 128 → 3，ELU | 同结构，估计机身坐标系线速度 |
| Encoder 训练 | 与 actor 解耦，单独 Adam/MSE，lr 0.001 | 每轮另有 20 次监督更新 |
| Actor | 36 → 512 → 256 → 128 → 6，无输出 tanh | 同结构与无界高斯均值 |
| Critic | 39 → 512 → 256 → 128 → 1 | 同结构，输入包含真实线速度与 encoder 输出 |
| 动作标准差 | 可学习 logstd，初始 std 1 | 保留 |
| Actor 单帧 / 历史 | 30 维 / 10 帧 | 保留布局、缩放和长度 |
| 控制 | 400 Hz 位置 PD、50 Hz 策略、action scale 0.25 | 保留；角度参考改为 PE02 home |
| Kp / Kd | 髋、大腿、小腿 4.3/4.3/4.9；0.34/0.34/0.24 | 对应映射至 PE02 六关节 |
| 力矩限制 | 每腿 5.5 / 5.5 / 14 N·m | 保留，与 MJCF gear 对应 |
| 动作延迟 | 随环境初始化抽样 25–50 ms | 10–20 个物理步，FIFO 存角度偏移 |
| 速度命令 | 5 秒重采样，heading 模式 | 保留有效行为，见下方问题处理 |
| 步态 | 2 Hz、左右相位差 0.5、占空比 0.5、摆高输入 0.06 | 保留相位、时钟和软接触目标 |
| 奖励 | 18 项，权重乘策略 dt，各项裁剪 ±5，总量 ±100 | 全部迁移并逐项记录日志 |
| 随机化 | 摩擦、质量、质心、惯量、Kp/Kd、力矩系数、零位偏移、延迟、IMU、推力 | 已迁移，参数独立归 PE02 所有 |
| 超时 / 失败 | 20 秒；机身接触/倾倒累计超过 0.5 秒 | 保留原累计失败语义；可显式改为连续判定 |
| 续训 | 网络、优化器及训练计数 | 额外保存环境状态、FIFO、随机数与 encoder 优化器 |
| 模型与初始站姿 | DRAGON_3 | 使用已校核的 PE02 模型、质量、碰撞体和 home |

Actor 单帧顺序为：机身角速度 3、重力方向 3、相对默认角度 6、关节速度 6、
执行动作 6、相位 sin/cos 2、步态参数 4。Critic 单帧是“真实机身线速度 3 +
无噪声 actor 单帧 30”；命令 3 作为网络独立输入。关节速度默认使用每个物理步
位置差分，速度缩放 0.1。训练采用 `NpEnvState`，`obs` 始终是字典。

物理与控制实现在 `src/unilab/base/backend/mujoco/batched_robot.py`；任务环境
只使用其声明的接口。模型编译、sensor 地址、足底网格顶点和随机化基准数据在
构造时缓存，step/reset/命令采样不读取 XML/STL。每个环境有独立状态、物理参数、
动作延迟队列、命令、历史和回合计数。

## 明确处理的原版问题与物理差异

- 原配置 `zero_command_prob=0.1`，实现却使用 `random > 0.1`，实际约 90% 零命令。
  PE02 用含义明确的 `commands.zero_probability=0.9` 保留有效概率。修正原先
  子集 reset 时将局部索引误用于全局环境的错误。需要 10% 零命令时显式改为 0.1。
- 原 gait offsets 配置 `[0, 1]` 随后被实现覆盖成 0.5；PE02 默认直接写 `[0.5, 0.5]`。
- 原 critic 拼接旧 `self.obs_buf`，导致时间错位；PE02 critic 使用当前无噪声状态。
- 原 IMU 扰动在历史写入之后覆盖当前观测，造成 frame 与 history 不一致；PE02
  在写入历史之前应用 IMU 扰动和观测噪声，当前 frame 与 history 末帧一致。
- 原 timeout bootstrap 使用当前 value；PE02 使用 reset 前终止状态的 value，
  并阻断跨回合 GAE 递推。小批量包含全部样本，不丢弃不能整除 batch 数的尾部。
- PE02 机械限位不同，随机初始角度在原 ±0.5 rad 扰动后裁至其真实限位。
  默认站姿接地高度约 0.293616 m；高度奖励目标从 PE01 专属的 0.25 m 改为 PE02
  home 高度。机身质量、惯量和碰撞资产沿用已校核版本。
- 原足端是半径 0.025 m 的球，PE02 足端为曲面凸包；足底高度改为实际凸包对地面
  的支撑距离。保留当前 PE02 自碰撞设置，没有恢复原 Isaac Gym 的禁用自碰撞。
- 摩擦、质量/质心/惯量、增益、延迟、IMU 安装误差与原版一样在环境初始化时采样，
  同一环境重置不重抽物理参数。非根连杆的质量/惯量同步缩放，根部只缩放惯量并
  单独添加质量。推力按原公式从基座质量与 max_push_vel 换算为单物理步力脉冲。
- **PhysX 的 restitution、contact_offset、求解器迭代数等不能与 MuJoCo 逐项等价。**
  沿用 PE02 的 MuJoCo 接触求解设置；未伪造 restitution 的一一映射，
  `domain_rand.restitution_mapping=null` 明确标记这一差异。摩擦合成规则、接触力
  与积分器也不同，需在实机接触标定后调整。这不是 Isaac Gym 轨迹逐步复现。
- 原 flat 配置使用平地，terrain curriculum 在源码解析时关闭，命令 curriculum
  更新又受 terrain curriculum 控制，因此平地实际不更新这些课程；本次迁移的
  `pe02_flat` 保留这个有效行为，没有额外承诺粗糙地形任务。
- 原配置中未激活的外力课程、关节摩擦随机化、功率终止和高度接触奖励，没有
  被当成已启用功能强行加入 PE02 的平地任务。

## 续训、回放与日志

续训例子（其余网络、任务和 PPO 配置应与 checkpoint 一致）：

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 \
  /ssd/conda/envs/aar_unilab/bin/python scripts/train_pe02.py \
  algo.num_envs=256 algo.max_iterations=1000 \
  training.resume=logs/pe02_custom_ppo/pe02/pe02_flat/<run_id>/model_400.pt
```

`max_iterations` 表示追加轮数。续训恢复网络、两个优化器、KL 学习率、训练计数、
物理状态、随机化、动作延迟、观测历史以及随机数状态。改变观测、动作、网络、
环境、环境数、PPO 超参数或资产时拒绝直接续训，以免悄悄改变旧 checkpoint 合同。
保存采用临时文件替换，周期由 `algo.save_interval` 控制，结束时额外保存最后一轮。

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 \
  /ssd/conda/envs/aar_unilab/bin/python scripts/train_pe02.py mode=play \
  checkpoint=logs/pe02_custom_ppo/pe02/pe02_flat/<run_id>/model_400.pt \
  play.render=interactive play.plot=true
```

无窗口回放使用 `play.render=none play.plot=false`。回放关闭随机化、噪声和训练
推力，默认零速度命令；`play.command='[0.2,0,0]'` 可给定速度，手柄可实时控制。
回放不因训练终止条件自动 reset，与原生 C++ player 一致。`play.delay_ms` 显式
指定确定性的回放延迟，默认 0；正式训练仍按 25–50 ms 随机延迟。

每个 run 有 `training_config.yaml`、`metrics.jsonl`、`tensorboard/` 和
`model_<iteration>.pt`。查看曲线：

```bash
/ssd/conda/envs/aar_unilab/bin/tensorboard --logdir logs/pe02_custom_ppo/pe02/pe02_flat
```

评估使用独立环境和固定种子的小幅初始扰动，不更新网络。除回报与终止率外，还
记录基座高度、倾角、非足端接触和符合站立条件的帧比例。站立条件默认是高度误差
小于 3 cm、倾角小于 10°、水平速度小于 0.1 m/s、没有受罚部位接触。
**回合达到 20 秒不等于站稳**：原版的宽松终止规则允许低姿态和部分非足端接触。

导出保存完整 encoder+actor 的 ONNX，并附带 `robot/pe02_runtime.json`，记录
默认角度、PD 增益、力矩限制、差分速度、动作延迟、相位和观测缩放。
原生 `pe02_v2` builder 与控制器位于独立的 `sim2sim/include/aar/pe02_runtime.hpp`，
公共入口只按合同选择，不把 PE02 尺寸写入其他机器人实现。

旧 PE02 checkpoint 仍按 `pe02_v1` 恢复旧网络、观测和力矩控制；旧最小训练入口须
显式指定 `observation=pe02_v1 training.steps=128`。正式版不再使用 `training.steps`。

## 验证记录

迁移验证包括原配置常量对照、GAE 失败/超时边界、encoder 梯度隔离、完整网络、
观测时间对齐、稀疏 reset、物理延迟、独立随机化、多轮更新及 CPU 精确续训。
旧 PE02 资产/站姿/导出测试继续按显式 legacy 配置执行。

- 256 环境 × 24 步，20 轮 CPU 短训完成：122880 条样本，400 次 PPO 更新与
  400 次 encoder 更新，TensorBoard、周期 checkpoint 和最终 release 已生成。
- CUDA 网络计算完成 32 环境、2 轮测试，并完成网络在 CUDA 上的 ONNX 导出；
  MuJoCo 物理仍使用 CPU。
- 新版 Python/C++ 1/10/100 步闭环轨迹最大误差为 0；默认配置及额外的
  25 ms 延迟、非零速度命令、非默认观测缩放组合均通过。
- PE02 专项 31 项通过；全仓库非 slow 测试 229 项通过、2 项预期失败。
  包含禁止导入 PE01/reference 模块后的独立训练、导出和迁移发布包回放。
- 原生 C++ 编译及 6 项 CTest 通过。
- WE11 flat/rough 均已重新组合配置、初始化并执行 3 步，观测字典且数值有限。
- 新增实现通过 Ruff，mypy 未增加错误；仓库已有的 `playback.py:35` mypy 问题
  及无关文件的 7 项 Ruff / 13 个文件格式问题未在本任务中修改。

256 环境短训 checkpoint：
`logs/pe02-migration/validation/2026-09-16_17-40-37_100974_mujoco/model_20.pt`。
对应发布包位于 `releases/pe02/pe02_flat/2026-09-16_17-40-37_100974_mujoco/`。
详细检查日志保存在 `logs/pe02-migration/verification/`。

对该 checkpoint 的 4 环境独立小扰动评估：平均回合 10.685 秒、失败率 50%、
平均基座高度 0.216 m、平均倾角 20.98°，符合站立条件的帧约 0.70%，受罚部位
接触比例约 93.31%。因此这份短训 checkpoint **尚不能稳定站立**。
原始指标保存在 `logs/pe02-migration/final-evaluation/report.json`，样本量很小，
用于排除“仅回合跑满即成功”的误判，不用于估计最终策略性能。

这些是训练系统与部署合同验证，不是稳定站立或行走策略的验收。迁移完成后仍需
正式长训、观察上述行为指标，并结合 PE02 实机标定判断策略质量。原先
`PERFORMANCE_BENCHMARK.md` 的 24%–30% 差距只适用于 v1 最小适配器，不能作为
本次 v2 完整 PPO 的性能结论。
