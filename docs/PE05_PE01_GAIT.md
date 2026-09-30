# PE05：PE01 足端步态迁移与 100 轮短训

本次采用 PE01 的足端相位约束，保留 PE05 的显式摆高跟踪和 signed weighted_sum。
配置标识为 `reward.gait_reward_version: pe01_shaped_v1`，共 19 项奖励。
本报告与旧的 [PE03 尺度校准报告](PE05_REWARD_CALIBRATION.md) 是两个独立实验。

## 奖励合同

| 项目 | 权重 | 乘权重、dt 前的原始代价 |
| --- | ---: | --- |
| tracking_contacts_shaped_force | -2 | mean((1-d) × (1-exp(-norm(F)²/5))) |
| tracking_contacts_shaped_vel | -2 | mean(d × (1-exp(-norm(v)²/0.20))) |
| feet_regulation | -0.08 | sum(exp(-max(h,0)/0.00025) × norm(v_xy)²) |
| foot_landing_vel | -0.15 | sum(landing × v_z²) |
| foot_clearance | -0.1 | sum(swing × ((h-target)/0.02)²) |
| action_rate | -0.01 | 原有动作一阶差分平方和 |
| action_smooth | -0.005 | 原有动作二阶差分平方和 |

用前两个接触项替换 contact_schedule，用 feet_regulation 替换 feet_slip，新增
foot_landing_vel；不重复保留旧滑移处罚。其他 12 项保留当前 PE05 配置。
总奖励只有逐项加法，每项只乘一次策略步长 0.02 s，没有正值截断、总奖励指数聚合或常数存活奖励。
两项接触原始值取两足平均，其他足端项取两足之和。

- d 使用 PE01 周期高斯 CDF 接触曲线，kappa=0.05；按支撑时长映射相位。
- 5 N² 与 0.20 (m/s)² 是指数分母，不是标准差，不再平方；使用绝对 SI 单位，不按体重归一化。
- F 来自 PE05 后端足端对地反力，不把自碰撞当作地面支撑。
- v 统一使用固定足底参考点的世界系解析刚体速度 reference_velocity。
  PE01 原实现使用足端位置在策略步上的差分；这里不差分移动接触点，也不新增速度历史状态。
- h 是 PE05 实际足底对地净空，不采用 PE01 球心高度。近地移动长度尺度 0.00025 m
  固定取自 PE01 的 0.25×0.001；不更改 PE05 home 身体高度。
- 仅 feet_regulation 对负净空按零处理，避免轻微接触穿透导致指数放大。
  摆高奖励层不再额外截断；当前后端的净空已经裁剪至 [0,1] m，因此无法用它测量负穿透量。
- landing：h<0.05 m、norm(F)≤0.1 N、v_z<0，三个条件同时满足。
  该 0.1 N 不改变碰撞、失败或接触诊断的 1 N 阈值。
- 摆高继续使用硬摆动掩码和 3 cm 正弦平方曲线，std=2 cm。
  完全不抬脚时峰值扣 0.0045/步，周期平均扣 0.0016875/步。
- 2 Hz、半周期错相、50% 支撑比以及零指令交替保持不变。
  观测、网络、控制、课程、终止和随机化配置不变。

当前命令课程仍用此前 PE03 风格的评分参数（力尺度为体重比例、速度指数分母 10、
kappa=0.07），不改为新奖励的参数；课程分数与新奖励不应当混为同一指标。

## 兼容与复现

旧 checkpoint 缺少 gait_reward_version 时仍执行旧 17 项奖励。配置不补写新默认值，
旧的零指令站立兼容分支保持原义。未知版本拒绝，新旧合同之间不能续训；新合同内部
按现有规则续训。checkpoint schema 和部署观测接口不变，无前代环境或 reference 运行时依赖。

默认配置及独立实验在 `conf/pe05/task/pe05_flat.yaml`、
`conf/pe05/experiment/pe01_gait.yaml`。历史配置冻结在
[before_task.yaml](assets/pe05_pe01_gait/before_task.yaml)。旧校准工具默认读取这个 17 项快照，
不会因当前默认奖励变化而把两轮校准混在一起。

工程冒烟命令：

```bash
bash tools/train.sh pe05 +experiment=pe01_gait algo.num_envs=64 algo.max_iterations=5 algo.save_interval=5 training.evaluation_interval=5 training.log_root=logs/pe05_pe01_gait_smoke training.export=false
```

独立短训命令：

```bash
bash tools/train.sh pe05 +experiment=pe01_gait algo.num_envs=4096 algo.max_iterations=100 algo.save_interval=25 training.evaluation_interval=25 training.evaluation_episodes=8 training.seed=1 training.log_root=logs/pe05_pe01_gait training.export=false
```

新 run 从随机初始化开始，不加载 PE01 或旧 PE05 权重。短训不自动导出可发布 release，
保留 checkpoint 和实际配置；不自动延长至 1000 轮。

## 数值与行为核查

运行 `/ssd/conda/envs/aar_unilab/bin/python tools/audit_pe05_shaped_gait.py` 可复现
[完整周期预算](assets/pe05_pe01_gait/cycle_audit.json)。这些是人为指定的运动学场景，
保持身体/力矩等其他项相同，并单列动作代价；不表示每组足端状态和动作都满足真实动力学。

| 场景 | 平均总奖励/步 |
| --- | ---: |
| 正确交替 | 0.027824 |
| 快速落地 | 0.021749 |
| 静止双支撑 | 0.008313 |
| 拖脚 | 0.007954 |
| 反相触地 | 0.002109 |

交替场景在切换附近仍有平滑相位带来的接触/速度代价，这不是公式计算错误。
合成动作幅度固定为 0.2，仅用于展示对应代价，不代表实际策略的动作成本。

七组评估命令为零、前后 ±0.3 m/s、横向 ±0.1 m/s、yaw ±0.4 rad/s，每组 8 个 episode，
最多 20 s。死亡后的样本不参与分项统计。报告原有基础指标及左右脚承重比例、支撑足速、
接触占空比、双支撑比例、离地次数/秒、摆高误差 P50/P90/P95，以及摆高目标>2 cm 时的
净空<5 mm 比例。左右脚指标按后端 foot_names 顺序，即左、右。

离地次数是 1 N 接触阈值的原始跨越次数，可能包含接触抖动，不能单独证明稳定交替。
没有摆动样本时数值占位为零，同时报告样本数，不能将该零值当作步态合格。
全机器人动作项和足端各项同时记录；回放另存逐步奖励、左右脚贡献、接触与净空轨迹。

短训只检查链路和学习趋势。基础行走门槛仍为每组存活率≥95%、平面误差≤0.15 m/s、
yaw 误差≤0.2 rad/s、高度误差≤2 cm、非足端接触比例≤1%；回报上涨不等于通过这些门槛。

## 实际验证与短训结果

实现和训练链路通过；**100 轮结束时没有形成稳定交替，七组行走验收全部失败。**

Run：`logs/pe05_pe01_gait/2026-09-28_16-51-56_802881_mujoco`。模型：`model_100.pt`，共 9,830,400 条样本，约 3 分 29 秒。
SHA256：`7afa6e10694344630b14f03f70c218547cf493ab6de412c694d2f4e7a33d3deb`。
实际配置见 [training_config.yaml](assets/pe05_pe01_gait/training_config.yaml)，
详细验收见 [evaluation.json](assets/pe05_pe01_gait/evaluation.json)。本轮没有追加训练或调权重。

| 指令 | 平均存活 s | 平面误差 m/s | yaw 误差 rad/s | 高度误差 cm | 非足端接触 |
| --- | ---: | ---: | ---: | ---: | ---: |
| standing | 1.08 | 0.138 | 0.025 | 1.66 | 11.1% |
| forward | 1.20 | 0.413 | 0.033 | 1.75 | 10.0% |
| backward | 1.08 | 0.254 | 0.037 | 1.60 | 11.1% |
| left | 1.08 | 0.171 | 0.027 | 1.63 | 11.1% |
| right | 1.08 | 0.184 | 0.023 | 1.64 | 11.1% |
| turn_left | 1.16 | 0.144 | 0.057 | 1.66 | 10.3% |
| turn_right | 1.08 | 0.133 | 0.081 | 1.40 | 11.1% |

所有组失败率均为 100%，失败原因为左大腿持续触地；不是低高度或倾斜阈值触发。
以上为独立 CPU、8 环境评估。无随机化且 reset 噪声为零，8 个 episode 是同一 nominal 条件的重复，不能当作随机扰动鲁棒性证据。
视频为单环境回放；接触图和连续抽帧显示约前 0.7 秒主要保持双脚着地，随后下蹲/失去支撑，没有稳定的左右交替周期。
末次评估左脚摆动中段拖脚比例约 80%–100%，右脚为 100%；个别阈值跨越事件不足以说明形成交替。

| 轮数 | 七组平均存活 s | 失败率 | 左/右摆动中段拖脚比例 |
| --- | ---: | ---: | --- |
| 25 | 1.003 | 100% | 92.9% / 100.0% |
| 50 | 1.220 | 100% | 97.6% / 96.9% |
| 75 | 1.214 | 100% | 92.6% / 100.0% |
| 100 | 1.109 | 100% | 87.1% / 100.0% |

最终独立评估的分项均值（各命令组等权，每组只计算存活阶段及终止步）：

| 项目 | 平均贡献/步 |
| --- | ---: |
| tracking_lin_vel | 0.01384152 |
| tracking_ang_vel | 0.00984960 |
| tracking_contacts_shaped_force | -0.01754087 |
| tracking_contacts_shaped_vel | -0.00058026 |
| foot_clearance | -0.00160833 |
| feet_regulation | -0.00000435 |
| foot_landing_vel | -0.00000984 |
| action_rate | -0.00014912 |
| action_smooth | -0.00009667 |
| collision | -0.01084291 |
| termination | -0.03614304 |

接触力处罚仍接近双脚承重的 -0.02/步，说明大部分应摆动时段未完成卸载。
动作一阶/二阶合计仅约 -0.000246/步，不能沿用上一轮“动作成本超过迈步直接收益”的数值论据来解释本轮结果。
终止处罚均值较大是约 1 秒就失败所致，并不是把 -2 每步重复施加；每个真实失败只扣一次。
100 轮不足以判断该机制最终能否收敛。本轮能确认的是公式和链路正确、已有短训未出现稳定交替，不能据此保证长训可用。

### 工程检查

- PE05 初次聚焦测试 93 项通过；补充非奖励配置不变测试后，步态专项 17 项通过。
- 完整非 slow pytest：626 passed、2 xfailed、4 个既有失败。补充测试在其收集结束后单独通过。
- 既有失败与前轮一致：PE03 robustness 的 3 项 max_level 预期不一致；只读 references 中嵌套 Git 元数据导致 1 项独立性检查失败。
- mypy：179 个源文件通过。改动文件 Ruff/format 通过，git diff --check 通过。
- 全仓库 Ruff 仍为 7 个既有问题，format 仍为 11 个既有文件，不涉及本次修改。
- WE11 flat/rough 各 2 环境 compose、初始化并 step 5 次通过。
- 旧真实 model_1000.pt 按保存的 17 项配置加载成功，向新合同续训被拒绝。

验证日志索引：[validation.json](assets/pe05_pe01_gait/validation.json)。

### 回放与复现

```bash
/ssd/conda/envs/aar_unilab/bin/python tools/evaluate_pe05_calibration.py --metrics logs/pe05_pe01_gait/2026-09-28_16-51-56_802881_mujoco/metrics.jsonl --output docs/assets/pe05_pe01_gait
MUJOCO_GL=egl /ssd/conda/envs/aar_unilab/bin/python tools/evaluate_pe05_calibration.py --checkpoint logs/pe05_pe01_gait/2026-09-28_16-51-56_802881_mujoco/model_100.pt --output logs/reports/pe05_pe01_gait/final --render
```

七组 MP4、逐步奖励/接触/净空 JSON 保存在 `logs/reports/pe05_pe01_gait/final/replays/`。
回放截图：[motion_sheet.jpg](assets/pe05_pe01_gait/motion_sheet.jpg)。
训练评估曲线：[training_evaluation.png](assets/pe05_pe01_gait/training_evaluation.png)。
