# PE02 无时钟步态行走配置（v3）

修改前仓库快照：`c105eb6a`。通过 `+experiment=walking` 启用
[walking.yaml](../conf/pe02/experiment/walking.yaml)。原始迁移基线和 standing
仍使用 `pe02_v2`，新版 walking 使用 `pe02_v3`。

## 训练目标和初态

- 以速度跟踪、机体姿态稳定、动作平滑和合理地面接触为主。
- 跟踪 home 站姿高度，`base_height_std=0.03 m`、权重 -3；评估高度容差为 1 cm。
- 关闭固定步态接触奖励和常数存活奖励；无规定步频、相位或周期足端轨迹。
- 新增离地高度奖励：1–10 cm 线性增长，10 cm 后封顶，仅奖励运动命令下的稳定单脚支撑。
- 每次从 `scene.xml` 的 `home` 站姿出发，零初速、零关节扰动。
- 保持 400 Hz 物理/PD、50 Hz 策略、原 PD 参数、力矩上限和动作尺度。
- 噪声、物理随机化、推扰、动作延迟均关闭，未新增课程。
- 每 5 秒重采样命令，20% 概率选择静止；静止标记在下次采样前持续强制全部命令为零。
- 运动命令范围为 vx ±0.5 m/s、vy ±0.3 m/s、yaw ±0.5 rad/s，均直接均匀随机采样。
  `pe02_flat.yaml` 中 `heading_command: false`，不再根据目标航向计算转向命令。
- 保留 PPO 探索、原优化参数和网络隐藏层；本机启动预设为 4096 环境、每轮 24 步。

## 观测、网络和模型兼容

每帧 24 维：角速度 3、重力投影 3、相对关节位置 6、关节速度 6、已处理动作 6。
没有相位 sin/cos 或四个步态参数；10 帧历史为 240 维。

- 速度估计器：240 → 256 → 128 → 3，仍由真实机体线速度监督，PPO 梯度与其分离。
- actor：估计速度 3 + 当前帧 24 + 命令 3 = 30 → 512 → 256 → 128 → 6。
- critic：真实速度 3 + 干净当前帧 24 + 命令 3 + 估计速度 3 = 33 → 512 → 256 → 128 → 1。
- 环境 obs 仍为字典；其中 actor 是 240 维历史、frame 是 24 维、critic 是 27 维、command 是 3 维。

checkpoint 保存完整配置；导出为 `pe02.runtime.v3`，Python/C++ 使用同一 24 维布局。
旧 v1/v2 模型仍按 checkpoint 原有契约播放，v3 启动器不会把旧模型的步态输入清空。
新旧训练契约不同，需要从头训练；禁止通过严格 resume 混用。

## 奖励

下表为 YAML 权重，单步贡献为 `权重 × 原始项 × 0.02`，再执行已有裁剪。
零权重项不进入 v3 奖励日志。高度原始项为
`((base_z - target) / base_height_std)^2`，walking 的 `std=0.03 m`、权重 -3；
`base_height_target: null` 使用 home 站姿高度（约 0.293616 m）。
例如高度误差 5 mm、1 cm、2 cm 时，每策略步高度惩罚分别约为 -0.00167、-0.00667、-0.02667。
`std` 是误差归一化尺度，不是目标高度或允许误差阈值。
基线和 standing 的 `std=1.0`；缺少该字段的旧 checkpoint 也按 1.0 计算，保持旧奖励。
启用高度奖励后，评估的 `standing_fraction` 同时检查高度绝对误差严格小于
`training.evaluation_height_tolerance=0.01 m`。此判定还包含速度、倾斜和接触条件，
不能单独解释为行走高度达标率；是否在行走中达到 1 cm 目标仍需训练后测量。

| 项 | 权重 |
| --- | ---: |
| tracking_lin_vel / tracking_ang_vel | 1.5 / 0.75 |
| base_height | -3（std=0.03 m） |
| lin_vel_z / ang_vel_xy | -5 / -0.05 |
| orientation | -5 |
| torques / dof_acc | -0.0002 / -2.5e-7 |
| action_rate / action_smooth | -0.01 / -0.005 |
| dof_pos_limits | -2 |
| collision | -3 |
| feet_distance | -150 |
| foot_landing_vel | -0.15 |
| feet_slide | -1 |
| feet_air_time | 0.25 |
| feet_air_height | 0.25 |
| base_contact / severe_tilt | -20 / -20 |

`feet_slide` 只在实际脚/地面接触时计算水平足速的模长，两脚求和。
地面接触阈值为 1 N。独立的脚/world 接触传感器排除两脚互碰和脚与机体自碰撞，
几何与传感器地址只在初始化时解析。足端仍必须是仅有一个指定 sole 碰撞体的末端连杆。

双足腾空奖励参考本地 Isaac Lab 的 `feet_air_time_positive_biped` 思路，独立实现：

```text
moving = norm(command_xy) > 0.1 m/s OR abs(command_yaw) > 0.1 rad/s
single_support = 恰好一只脚接触地面
mode_time[i] = 接触时的持续支撑时间，否则为持续腾空时间
r_air_raw = min(min(mode_time), 0.25 s)
```

仅在 moving、single_support、没有非足端接触、倾角不超过 60° 时发放奖励。
最大单步贡献为 `0.25 × 0.25 × 0.02 = 0.00125`；双脚支撑、双脚离地、静止命令均为零。
它不规定左右顺序或步频；静止时不会为了领奖励而必须踏步。
接触时长按策略步采样，分辨率为 20 ms；部分重置会清空对应环境计时，checkpoint 会保存计时。

参考来源（只读，不导入或执行）：本地 IsaacLab `82a0ab2d8`，
`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py`。
该双足函数不是 Go1 默认启用项。本版额外加入转向门控及非足端接触/倾斜门控。

足端高度奖励 `feet_air_height` 使用脚底碰撞几何最低点相对 z=0 平地的高度，
由后端的足端连杆坐标系传感器和缓存几何计算，资产几何只在初始化时解析。
新奖励使用 `xbody` 坐标系，避免把惯性坐标系用于变换碰撞顶点；
旧高度接口和旧 checkpoint 的传感器布局保持原有语义。高度区间在
`reward.feet_air_height_range: [0.01, 0.10]` 中配置：

```text
r_height_raw = clip((h - 0.01) / (0.10 - 0.01), 0, 1)
r_height = 0.25 * r_height_raw * 0.02
```

与腾空时间项使用相同门控，只有运动命令、恰好单脚支撑、无非足端触地、
倾斜不超过 60° 时，才给离地脚奖励。触地脚不贡献奖励；静止、双脚支撑、
双脚腾空时均为零。高度不超过 1 cm 时为零，5.5 cm 时每步 0.0025，
10 cm 及以上每步 0.005。抬脚保持时会持续发放，没有额外的左右交替约束。
TensorBoard 自动记录 `reward/feet_air_height`；旧 checkpoint 缺少该项时不启用。

## 即时惩罚与弱失败

- 任意非足连杆接触合力超过 1 N：每个连杆 collision 贡献 -0.06/步。
- 机体接触合力超过 1 N：额外 base_contact 贡献 -0.4/步。
- 倾角超过 60°（包括翻倒）：额外 severe_tilt 贡献 -0.4/步。
- 终止仍保持旧机制：机体接触力 > 5 N 或投影重力 z > -0.1（约 84.26°），
  累计失败计数超过 25 步才终止，即第 26 个失败步；中间恢复正常不清空累计。
- 腿部擦碰、60°～84° 倾斜会扣分，但不会单独造成即时终止。
- episode_length_s 保持 20；沿用旧的 `steps > 1000` 超时判定，实际在第 1001 步超时。

## 启动和回放

```bash
bash tools/train.sh pe02_walking algo.max_iterations=500
bash tools/train.sh pe02_walking mode=play checkpoint=-1
```

训练日志继续位于 `logs/pe02_walking/<时间戳>/`。默认回放 interactive、关闭 plot、
不限步数。`checkpoint=-1` 在最新训练目录里选编号最大的模型，不区分 v2/v3；
播放时以选中 checkpoint 的配置识别版本。

从 walking 启动预设临时切换旧 standing 时，须同时指定
`+experiment=standing observation=pe02_v2`，或直接使用
`python scripts/train_pe02.py +experiment=standing`。

## 本版验证（2026-09-17）

- 全仓非 slow 测试：287 passed、2 xfailed；29 项针对性测试覆盖奖励门控、
  20% 静止采样、弱失败累计、部分重置、精确续训及 v2/v3 导出后播放。
- WE11 平地和粗糙地形分别组合配置、初始化 2 个环境并步进，观测有限且字典契约保持。
- v2/v3 分别完成 4 环境 × 4 步 × 2 轮流程验证、checkpoint 和 ONNX 导出。
- C++ 播放器编译成功；v2/v3 的 Python/C++ 闭环分别检查 1、10、100 步，
  动作、控制量及机体高度的最大差值均为 0。
- walking v3 启动器分别回放旧 `model_500.pt` 和新 v3 验证 checkpoint，各 3 步无窗口运行通过。
- 本次修改文件 Ruff 格式/规则检查通过，`git diff --check` 通过。
  全仓仍有存档前已有的 12 个格式文件、7 项 Ruff 报错和
  `src/unilab/base/backend/mujoco/playback.py:35` 的 1 项 mypy 报错。

流程验证目录：`logs/pe02_walking/validation/clock_free_20260917_172329/`。
对应导出包：`releases/pe02/pe02_flat/clock_free_validation_20260917_172329_v3/`。
两轮仅用于验证训练和部署流程，不能代表已学会行走；正式效果需从头训练评估。

下面记录的旧版训练结果只作为历史资料。

## 历史 v2 验证（2026-09-16，不代表本版效果）

- 新增 4 项行为测试通过：训练/评估的全量和部分重置、无穿地 home、零物理随机化、
  sole 接触豁免、髋部接触惩罚、宽松累计终止，以及足端几何变更时拒绝错误分类。
- 全仓非 slow 测试 255 项通过、2 项预期失败；其中包括 WE11 平地和粗糙地形的
  配置组合、初始化及步进检查。
- 正式入口完成 256 环境 × 24 步 × 3 轮，共 18432 条样本、60 次 PPO 更新和
  60 次 encoder 更新，保存 checkpoint、TensorBoard 和 ONNX 导出包。
- 新导出包的 Python/C++ 闭环在 1、10、100 步上的动作、力矩和高度最大差值均为 0。
- 本次修改的 Python 文件 Ruff 格式/规则检查通过，`git diff --check` 通过。
  全仓静态检查仍有之前的 12 个格式文件、7 项 Ruff 报错及
  `src/unilab/base/backend/mujoco/playback.py:35` 的 1 项 mypy 报错。

流程验证检查点：
`logs/pe02_walking/2026-09-16_21-30-54_633529_mujoco/model_3.pt`。
导出包：`releases/pe02/pe02_flat/2026-09-16_21-30-54_633529_mujoco/`。
检查日志和计数报告统一位于 `logs/pe02_walking/validation/`。

这 3 轮只验证新配置的完整训练流程，未完成行走长训。该早期策略在零命令评估中
虽能存活至回合上限，但非足端接触帧约 95.8%，不算站稳或行走成功。
后续验收应结合足底支撑、速度跟踪、姿态和抬脚情况，不能仅看宽松条件下的存活时长。
