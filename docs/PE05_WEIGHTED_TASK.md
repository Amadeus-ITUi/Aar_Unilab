# PE05 weighted_v2：本体适配与加权奖励

> 2026-09-28 更新：当前默认采用 PE01 足端步态机制并保留 PE05 摆高跟踪，见 [迁移与短训报告](PE05_PE01_GAIT.md)。此前的 [PE03 尺度校准](PE05_REWARD_CALIBRATION.md) 及本文旧奖励表保留为历史记录。

当前训练规则的完整快照及后续改造讨论入口见
[PE05 训练规则与改造讨论基准](PE05_TRAINING_RULES.md)。本文保留适配经过与历史验证记录。

当前默认任务为 `conf/pe05/task/pe05_flat.yaml` 的 `weighted_v2`。
本次处理的是 PE03 本体搭配 PE01 奖励时，低机身和小腿支撑缺乏有效约束的问题。
短训只验收训练与部署链路；尚未证明步行收敛、小腿触地问题已经在策略上消失或实机可用。

## 保留的契约

- PE03 CNC v3 冻结资产、home 站姿和约 0.29138291668 m 高度目标，资产指纹不变。
- 单帧 30 维、历史 10 帧、独立 3 维命令；Encoder `300→256→128→3`、
  Actor `36→512→256→128→6`、Critic `39→512→256→128→1`，独立速度监督和梯度隔离。
- 400 Hz 物理/PD、50 Hz 策略、action scale 0.25、原 PD/力矩参数、关节目标物理限位。
- 2026-09-28：暂时关闭动力学随机化、观测噪声、推扰及随机化课程；单独启用速度命令课程。
  训练延迟为 0 ms，reset 关节无扰动、初速度为零；评估延迟和 reset 扰动也设为零。
- 参考 PE03，20% 零命令、80% 非零命令采样；每 10 秒采样，直接控制 yaw rate，关闭 heading 控制；
  2 Hz、相位偏移 0.5、支撑比 0.5。命令从小范围逐步扩展到 vx ±1、vy ±0.6 m/s、yaw ±1 rad/s。
- 2026-09-28：默认摆脚高度由 0.06 m 调整为 0.03 m，默认回放步态同步；旧 checkpoint 仍使用保存的配置。
- 独立 PE05 实现，不导入 PE03、WTW 或其他基线代码；仅复用公共 MuJoCo 接口。

## 奖励：有符号加权和

每策略步 `R = 0.02 × Σ(weight_i × raw_i)`。
没有指数项、乘积式总奖励、正值截断或无条件存活奖励。时间截断不扣失败分。
权重和归一化尺度均在任务 YAML；误差尺度表示“多大物理误差对应 raw=1”，便于解释权重。
所有分项都写入 TensorBoard，包含原始量与加权后量。

| 项目 | 原始量 raw | 权重 |
| --- | --- | ---: |
| 平面速度跟踪 | `1 − (‖v−cmd‖/0.4 m/s)²` | 1.0 |
| yaw 跟踪 | `1 − (yaw_rate−cmd_yaw)²/(0.5 rad/s)²` | 0.5 |
| 机身高度 | `((z−home_z)/0.03 m)²` | −1.0 |
| 摆脚净空 | 摆动脚 `((clearance−target)/0.02 m)²` 之和 | −0.5 |
| 触地时序 | 两脚实际/期望接触布尔值误差平方的均值 | −0.5 |
| 非足端接触 | 每个身体部件本策略步是否发生接触；小腿乘 2，其余乘 1 | −3.0 |
| 支撑脚滑移 | 有实际触地的脚 `‖v_contact_xy‖²/(0.2 m/s)²` 之和 | −0.2 |
| 姿态 | projected gravity 的 xy 分量平方和 / 0.2² | −0.2 |
| 垂直速度 / roll、pitch 角速度 | 速度平方 / 角速度平方和 | −0.5 / −0.01 |
| 力矩 / 关节加速度 | 六关节平方和 | −0.0002 / −2.5e−7 |
| 动作变化 / 二阶变化 | 一阶 / 二阶动作差分平方和 | −0.01 / −0.005 |
| 软限位 | 超出 95% 关节范围的距离之和 / 0.1 rad | −0.2 |
| 脚距 | 机体系横向脚距不足 0.08 m 的误差 / 0.03 m，再平方 | −0.5 |
| 失败 | 真正终止为 1；正常 timeout 为 0 | −100，即每次失败 −2 |

例如机身低 3 cm，高度扣分为 −0.02/步；低 8 cm 为约 −0.1422/步。
此前未归一化的 `−3 × height_error² × dt` 对 8 cm 仅扣 −0.000384/步，
容易被跟踪/存活奖励抵消。一次小腿接触现在至少扣 −0.12/步。
失败扣分用于抑制通过提前倒地逃避持续负奖励的倾向；这些权重仍需要正式训练验证。

摆脚目标为摆动相位上的 `0.03 × sin(π × swing_progress)²`，使用实际足底最低点净空。
2026-09-28：默认 `reward.zero_command_stance=false`，取消零命令双脚持续支撑特例。
所有速度命令下，接触目标与摆脚净空目标均跟随步态相位；零命令也要求原地交替踏步。
摆脚高度为 3 cm。旧 weighted checkpoint 缺少此字段时保留原双脚支撑规则；
奖励合同变化后使用新 run，不能将旧 checkpoint 直接续训为新规则。
`commands/standing_fraction` 仍统计低速指令占比，不表示双脚支撑模式。
滑移取刚体在实际接触点的速度，仅作用于实际支撑脚，不再惩罚空中正常摆动速度。

## 接触、终止和 reset

公共后端新增可选地面接触历史，PE05 每个 400 Hz 物理步采样；每个策略步包含 8 帧。
逐部件记录机身、髋、大腿、小腿的触地力，以及净接触力减地面力得到的其他接触力。
按 1 N 阈值判定。一个部件同时有触地与其他接触时只计一次碰撞惩罚。
这是净合力判定，方向相反的多个接触可能抵消；不将其描述成逐接触点几何计数。

任一非足端部件连续触地达到 0.1 s 即失败，按物理 tick 计时；部件分开累计，
离地就清零。短暂擦碰仍扣分，自碰撞扣分但不累计地面支撑时间。
高度低于 0.15 m 或倾斜超过 60° 连续 0.1 s 也失败。
这是对旧版“累计失败超过 0.5 s”的明确修订，旧配置保留原语义。
新任务 timeout 恰好发生在 20 s；GAE 仍使用 final observation 引导，不穿过 reset。

训练 reset 从 home 出发；当前随机化关闭，关节无扰动。启用随机化后关节扰动最多 ±0.1 rad，随等级增加，物理限位裁剪；
默认初始基座速度为零。根据缓存足底几何调整 z 到最小净空 1 mm，最多 8 次重采样，
仍存在净接触力则回退 home；home 仍有异常接触时直接报错。
对子集 reset 只清理相应历史、FIFO、接触计时和 episode 统计。
无扰动回放保持精确 home，不使用训练 reset 的净空修正，Python/C++ 保持相同初态。

## 渐进训练与诊断

### 当前速度命令课程

`commands.curriculum=true`、`curriculum_strategy=velocity_bins` 启用 PE05 独立实现的
PE03 固定步态速度课程；不导入 PE03 模块，不启用动力学随机化或可变步态课程。

- 初始范围：vx ±0.3、vy ±0.1 m/s、yaw rate ±0.5 rad/s。
- 最大范围：vx ±1、vy ±0.6 m/s、yaw rate ±1 rad/s；三轴网格宽度均为 0.1。
- 初始区域权重为 1，其余为 0；按权重选三维单元，再在单元内均匀采样。
- 每 10 秒结算窗口；成功单元及其 26 个相邻单元权重增加 0.2，最高 1。
  同批次重复成功的单元合并更新，不因并行环境数量增加单次增量。
- 零命令独立占 20%，不会推动课程扩展；真实失败不能晋级。
  提前结束的窗口仍除以完整 500 步，不因回合缩短虚增得分。
- 先用旧指令计算奖励、末帧和课程成绩，再采样新指令；timeout bootstrap 保留旧指令。
- 评估使用七类固定指令，不更新课程。权重、每环境单元、窗口成绩和计时随 checkpoint 保存，支持精确续训。

晋级评分沿用 PE03 四项公式与阈值，**只用于课程判定，不修改加权奖励**：
平面跟踪 `exp(-||速度误差||²/0.25)` ≥0.8，yaw 跟踪 `exp(-yaw误差²/0.25)` ≥0.7；
摆动期接触力评分、支撑期足端速度评分均 ≥0.9。接触目标使用平滑相位（kappa=0.07），
力尺度为本环境体重的 8%，速度平方分母为 10，与 PE03 固定步态课程一致。
完整公式位于 `src/unilab/envs/locomotion/pe05/command_curriculum.py`。

TensorBoard 的 `command_curriculum/*` 记录有效区间、有效单元比例、完成/成功窗口数和四项评分。
`commands.curriculum=false` 可关闭命令课程并直接采样完整范围。

### 保留的随机化课程

当前动力学随机化及其课程关闭。以下说明保留的可选机制；开关重新启用后才生效。
随机化范围仍保存在 YAML，其中 25–50 ms 延迟范围仅在启用动力学随机化后用于训练。
零命令采样概率为 20%，不是按实际轨迹时间保证的比例。

每个环境有独立 level，初始 0，成功 episode 后增加 0.1，最多 1，仅在 reset 时应用。
level 插值质量、惯量、COM、摩擦、PD、力矩系数、零位、IMU 偏置，缩放观测噪声、
reset 扰动和推扰。满级观测噪声系数为 1.0（旧版 1.5）；其余随机化终点保持原配置。
随机化目标仍按环境在启动时采样，并完整保存在 checkpoint。
PhysX restitution 仍无等价映射，不伪造随机化。

旧 weighted checkpoint 未指定 `curriculum_strategy` 时保留原来与随机化 level 绑定的
30%→100% 命令课程及 heading 规则；新速度网格课程不依赖该 level，且要求关闭 heading 控制。

晋级必须满足自然结束且 episode 达到 95% 时长、非足端接触占比 ≤1%、
平均高度误差 ≤1 cm、平面跟踪误差 ≤0.15 m/s、yaw 误差 ≤0.2 rad/s。
单纯存活 20 秒、蹲着支撑或发生失败都不会晋级。若站稳后课程仍停在 0，先检查这些指标，
再调门槛；不通过扩大网络掩盖环境约束问题。

TensorBoard 建议同时看：

- 与 PE03 相同的基础分组：`Train/*`（回报、回合长度/秒数、样本数）、
  `Loss/*`、`Policy/*`、`Perf/*`、`Eval/*`。回合长度按 50 Hz 换算为策略步。
  JSONL 和控制台沿用原字段；`reward/*` 仍是乘 0.02 后的每步贡献，不改变数值。
- `reward/*`、`raw_reward/*`，尤其高度、碰撞、净空、失败及正负分项之和；
- `contact/L_calf_Link/*`、`contact/R_calf_link/*`，区分触地、自碰撞、连续触地秒数；
- `behavior/height_error_m`、`behavior/clearance_error_m`、`behavior/slip_speed_mps`；
- `curriculum/*`、`reset/fallback_fraction`、`commands/standing_fraction`；
- `Eval/<command>/*`：站立、前后、左右、左右转共 7 种固定命令，当前默认 0 ms 延迟，
  包括高度误差 P95、触地率、跟踪误差和综合成功率。综合成功要求存活到期且满足质量阈值，
  不能只看 failure_rate。总指标对各命令等权平均。

## 启动与兼容

奖励/终止/reset/课程改变后应开启新 run；**旧 checkpoint 不能直接续训新版任务**，
训练契约检查会拒绝。旧 checkpoint 加载、回放和导出仍使用其中保存的旧配置。
`task=pe05_legacy` 保留旧任务用于回归；它不会自动恢复旧 run 自定义的其他训练配置。
CLI 回放默认保留 checkpoint 的 gait/delay，显式 `play.gait=` / `play.delay_ms=` 才覆盖。

```bash
cd /ssd/Aar_Unilab
# 正式训练需后续手动启动。本次未启动长期训练。
# 当前启动预设每 100 轮保存。
bash tools/train.sh pe05 algo.save_interval=100

# 短训验收
bash tools/train.sh pe05_smoke
bash tools/train.sh pe05_smoke algo.max_iterations=2 \
  training.resume=logs/pe05_weighted_smoke/<run>/model_10.pt

# 查看新版正式训练与验收日志
/ssd/conda/envs/aar_unilab/bin/python -m tensorboard.main \
  --logdir logs/pe05_weighted/tensorboard_runs \
  --host 127.0.0.1 --port 6006

# 无窗口回放；旧模型也支持给出其明确路径
PYTHONPATH= /ssd/conda/envs/aar_unilab/bin/python scripts/train_pe05.py \
  mode=play checkpoint=<model.pt> play.render=none play.steps=100 \
  play.command_source=fixed 'play.command=[0.2,0,0]' play.delay_ms=25
```

新训练自动建立 `tensorboard_runs/PE05__任务-模拟器__实验__启动时间` 相对链接。
旧训练或已运行的旧日志可通过 `tools/sync_pe05_tensorboard.py <run目录> --watch`
生成实时分组副本；原始事件、JSONL、checkpoint 不变。副本位于 run 内的
`tensorboard_pe03_style/`，同样通过 `tensorboard_runs` 展示。
增加 `--until-pid <训练进程PID>` 可在训练退出并完成最后同步后自动结束跟随。

## 本次验证

验证日期：2026-09-26。使用 `/ssd/conda/envs/aar_unilab/bin/python`；pytest 清空
shell 注入的 `PYTHONPATH`，避免 ROS Python 3.10 的 pytest 插件进入 Python 3.13 环境。

- 完成 `32×24×10` 短训，以及保持同一训练契约的追加 2 轮续训，累计 9216 样本；
  PPO/Encoder 各完成 240 次更新。TensorBoard 已读回第 11、12 轮的接触、奖励和课程指标。
- 短训 checkpoint：`logs/pe05_weighted_smoke/2026-09-26_14-05-36_549294_mujoco/model_10.pt`。
  续训 checkpoint：`logs/pe05_weighted_smoke/2026-09-26_14-06-04_363819_mujoco/model_12.pt`。
- 对应自包含 release：`releases/pe05/pe05_flat/2026-09-26_14-06-04_363819_mujoco`。
  该 release 的 Python/C++ 1、10、100 步闭环最大误差均为 0（现有容差 `2e−5`）。
- 独立部署测试完成异地加载、25/50 ms 延迟、非零命令、超限动作以及 100 步回放。
  新模型和原 run 的 `model_500.pt` 均另外完成 CLI 100 步无窗口回放。
- 新增用例覆盖归一化有符号总奖励、物理步中间擦碰、逐部件连续触地、失败扣分、
  支撑滑移/摆脚净空、无穿透 reset、90% 零命令、子集隔离、课程门槛、原生/非融合后端一致性。
  精确续训覆盖已经晋级的环境：网络参数、课程 level、接触计时和 reset 统计逐值相同。
- 7 类固定命令的实际评估完成并写入 `logs/pe05_weighted_validation/evaluation.json`。
  **12 轮模型成功率为 0，平均存活约 1.07 秒，高度平均误差约 7.86 cm**。
  这说明诊断和失败判定已有效工作，不代表策略问题已通过短训解决。
  不将本次短训模型当作收敛模型；本次未启动长期训练。
- 最终全仓非 slow pytest：**572 passed、4 failed、2 xfailed**，PE05 的 39 项测试全部通过。
  4 个失败已在本次重新执行：PE03 的 3 个课程测试期待上限 1.0，但既有配置为 0.7；
  仓库审计发现 `references/tron1-rl-isaaclab/.git` 与
  `references/walk-these-ways/.git` 嵌套元数据。相关测试/源码与 HEAD 一致，未修改这些其他基线。
- WE11 flat/rough 的 compose、初始化和 step 回归通过，控制/观测契约未变。
- 本次修改的 Python 文件 Ruff 格式和 lint 通过；mypy 174 个源文件通过；`git diff --check` 通过。
  全仓 Ruff 仍有 11 个既有未格式化文件、7 个既有 lint 问题，报错文件与 HEAD 逐字节一致。

训练输出、checkpoint、release 均为忽略的运行产物；未提交或推送本次修复。
