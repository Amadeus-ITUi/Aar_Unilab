# ESD-Link 台架运动与 RL 放行记录

## 测试边界

- 样机固定在台架，P1～P8 运动无干涉，现场人员可立即硬件断电。
- 冻结下位机的八路使能为一个整体；“单端口测试”仅表示只有该目标变化。
- 每一阶段从 `DISABLED` 开始，完成后必须收到 lower `DISABLED` 确认。
- 上位机不再用 ROS 消息年龄或频率设置第二套运行期 deadline。端口离线、IMU 无效、
  非有限数、越位、超出冻结协议能力、下位机 fault/watchdog 或 CONTROL 故障仍终止
  当前运动阶段，不自动继续下一阶段。

## 放行顺序

### A0：只读基线（已通过）

- 30 分钟下位机源状态 `250.000 Hz`，bridge 平均 `249.992 Hz`。
- 源状态序号缺口、COBS、CRC、offline、fault、watchdog 均为 `0`。
- P1～P8 `valid_mask=3`，IMU mask `7`；本阶段没有控制命令。

### A1：5 秒当前位低增益保持

1. 验证手柄、USB CDC、schema/fingerprint、P1～P8、IMU、250 Hz 与 `DISABLED`。
2. 工具打印当前八路上位机坐标和低增益，现场释放后按一次 A。
3. 进入 `MANUAL_TEST`，以 200 Hz 发送完整 P1～P8 当前位目标，持续 5 秒。
4. 位置型端口偏移不得超过 `0.08 rad`，任意端口速度不得超过 `2 rad/s`。
5. 完成或异常后整体失能，保存状态/命令/应用序号 CSV。

```bash
ros2 run deploy_tools esd_hold_current
```

### A2：安全失能与小动作辨向

- 确认显式失能和维护命令停止刷新都进入 `SAFE_DAMPING/DISABLED`。
- P1、P2、P4、P5 依次执行低增益、低速、小于等于 `0.03 rad` 的正负位置动作。
- P3、P6 依次执行小于等于 `0.5 rad/s` 的正负速度动作。
- P7、P8 依次执行小于等于 `0.03 rad` 的镜像位置动作。
- 逐端口记录物理方向、反馈方向、零点、软限位、其他端口保持和失能结果。

一次仅选择一个端口，并按 `P1 → P2 → P4 → P5 → P3 → P6 → P7 → P8`
顺序执行：

```bash
ros2 run deploy_tools esd_small_motion --port 1
```

正向和反向分别在 DISABLED 下等待手柄 A 确认；每个方向都会短时整体低增益使能，
完成后自动回到该次测试的初始目标并整体失能。任意物理方向不符、异响或非选中端口
运动均不得继续下一方向。

### A3：维护与扫频

- 先执行小幅、短时间单/双端口扫频，再执行既定 1/4、2/5、3/6 和 P7/P8 扫频。
- 扫频发令为 200 Hz，每个新 `LowerState` 记录一行；检查状态序号、来源序号、
  命令序号和 `last_applied_command_seq`。
- 命令拒绝、watchdog、CONTROL 故障、端口离线和持续跟踪误差均必须为 `0`。

### A4：零策略与 STANDBY

- 先完成 3 秒 STANDBY 斜坡，再以强制零动作进入 POLICY；验证 145 维观测、腿轮坐标、
  翼观测和 IMU/projected gravity。
- 核对 `source_state_sample_seq`、命令序号和下位机应用确认；策略输出到应用确认 p99 < `12 ms`。
- 强制零策略必须发布原生 `PolicyCommand`，不能用停止发令来伪装零动作；退出时依次
  回 STANDBY、整体失能并确认 lower `DISABLED`。

```bash
./scripts/rl/run_zero_policy_test.sh
```

### A5：POLICY 与既定步态

- 首次 POLICY 仅短时运行，确认手柄命令、翼控制、观测时效、动作时效和退回 STANDBY。
- 从秒级逐步延长到 30 分钟；在 A0～A4 全部通过前，不放行既定步态。
- 坐标或观测错误必须修正上位机接口，不允许通过修改模型掩盖。

## 阶段记录

| 时间 | 阶段 | 结果 | 证据/备注 |
|---|---|---|---|
| 2026-09-09 14:03 | A0 30 分钟 READ_ONLY | PASS | `logs/read_only_30min_20260909_140314/`；下位机源序号零缺口 |
| 2026-09-09 14:43 | 右翼手动辨向 | PASS | P8=右翼，下压时 lower P8 减小 |
| 2026-09-09 14:46 | 左翼手动辨向 | PASS | P7=左翼，下压时 lower P7 增大 |
| 2026-09-09 15:29 | 翼坐标/限位修正后 READ_ONLY | PASS | 250 Hz，P1～P8/IMU 有效，无故障，零命令 |
| 2026-09-09 15:35 | A1 前置 | READY | 手柄 `/dev/input/js0` 与 ESD-SLAVE by-id 路径均存在 |
| 2026-09-09 15:42 | A1 首次保持 | SAFE ABORT | 发送约 202 条有效命令后，维护工具遇到约 22 ms ROS 状态年龄，超过其写死的 20 ms 门限；整体失能已确认 |
| 2026-09-09 15:43 | A1 首次诊断 | PASS | 命令应用延迟约 1.94 ms，零拒绝、零 watchdog、零 CONTROL 故障；bridge 内部 20 ms 硬门控未触发 |
| 2026-09-09 15:46 | A1 工具修正 | PASS | 维护 ROS 监视端独立改为 40 ms，bridge 20 ms 门控不变；新增具体失效原因和失败 CSV；74/74 测试通过 |
| 2026-09-09 15:50 | A1 五秒保持 | MOTION PASS / CLOSEOUT FAIL | 现场确认无位移；位置型端口最大偏移 <0.001 rad，八路最大速度 <0.24 rad/s，1241 条状态；但先写 CSV 后失能触发上位机 maintenance timeout |
| 2026-09-09 15:53 | 维护工具收尾修正 | PASS | 当前位保持、单/双电机扫频、配对扫频和翼扫频全部改为先确认 DISABLED、后写 CSV；74/74 测试通过 |
| 2026-09-09 15:55 | A1 第二次五秒保持 | MOTION PASS / CLOSEOUT FAIL | `logs/esd_hold_current_20260909_155501.csv`；工具确认 1260 条状态且整体失能，现场再次确认“无位移”；bridge 在正常失能事务期间仍误报 maintenance timeout |
| 2026-09-09 16:18 | bridge 失能确认修正 | PASS | 失能事务期间暂停活动控制超时和动作发送；SET_ENABLE 必须在 ACK 后收到新的目标控制状态才算确认；含 120 ms SAFE_DAMPING 的 PTY 测试连续 5 次及 74/74 迁移测试通过 |
| 2026-09-09 16:22 | A1 最终复跑 | PASS | `logs/esd_hold_current_20260909_162258.csv`；5 秒控制段、1381 条状态、整体 DISABLED 确认；最大位置变化 0.00115 rad、最大速度 0.181 rad/s；零链路/命令/看门狗故障，bridge 日志无 control fault；现场确认无位移、无异常声响 |
| 2026-09-09 16:32 | A2 工具准备 | READY | 新增逐端口 `esd_small_motion`：位置 ±0.03 rad、轮速 ±0.5 rad/s，低增益、完整八路命令、分段 A 键确认、实时越位/超速保护和自动失能；安装入口及 52/52 deploy_tools 测试通过 |
| 2026-09-09 16:36 | A2 P1 首次尝试 | SAFE ABORT / NO MOTION | 首次 A 只进入当前位保持；等待第二次 A 时 ROS 监视端超时，bridge 维护期限触发整体安全失能；P1 范围仅 0.000383 rad，未执行小动作；无下位机拒绝/watchdog/CONTROL fault |
| 2026-09-09 16:40 | A2 工具流程修正 | PASS | 移除使能状态下的人工等待；改为每个方向都在 DISABLED 下确认，单方向短时使能、回位后自动整体失能；52/52 deploy_tools 测试通过 |
| 2026-09-09 16:47 | DISABLED 人工行程记录 | PASS | `logs/manual_range_20260909_164704/`；180.002 s、45004 帧，记录端与 bridge 源序号零缺口，零 COBS/CRC/reject/watchdog/fault，命令序号始终为 0；获得 P1～P8 范围 |
| 2026-09-09 17:06 | 人工限位终端界面复测 | PASS | `logs/manual_range_20260909_170608/`；153.645 s、38395 帧，bridge 源序号零缺口、命令序号 0；现场确认 P1/P2/P4/P5/P7/P8 两侧机械硬限位 |
| 2026-09-09 17:26 | 硬限位安全固化 | PASS / POSE BLOCKED | 两个独立记录器得到相同端点；软件限位保留至少 0.075 rad 裕量；bridge 新增使能前全位置检查，PTY 连续 5/5 通过，迁移包 92/92；现场当前 P2/P5/P8 在软件上限外，主动测试保持禁止 |
| 2026-09-09 19:44 | P1/P4 单次扫频流程 | FIELD PASS / CLOSURE FIXED | 完成回收、8 秒低幅、40 秒正式 chirp 和整体失能；bridge 无 control fault；修复客户端 5 秒等待短于 bridge 最长 11.5 秒失能服务窗口导致的结束误报；相关 91 项测试通过 |
| 2026-09-09 19:53 | P3/P6 速度扫频首次运行 | SAFE ABORT / LIMIT FIXED | 0.5 rad/s 低幅通过；正式目标越过约 2 rad/s 后冻结下位机拒绝命令并安全失能；累计拒绝 11、最后拒绝码 INVALID_STATE，链路及端口正常；正式幅值改为 1.8 rad/s 并增加启动前硬拒绝 |
| 2026-09-10 13:54 | 精简后 READ_ONLY | PASS | 15.0018 s、3750 条记录、设备 250.000 Hz；源序号/COBS/CRC/reject/watchdog/fault 零增量，命令序号始终为 0 |
| 2026-09-10 14:01 | 精简后 A1 当前位保持 | SENSOR PASS / ONSITE SOUND PENDING | 5 s、约 199.4 Hz 唯一来源命令；不回位 P2/P5；位置最大差 <0.00085 rad、最大速度 <0.168 rad/s，事后 lower DISABLED 且零故障 |
| 2026-09-10 14:09 | 下位机单一 deadline 实机注入 | PASS | 低增益当前位保持期间暂停上位机 200 ms；下位机 watchdog=1、CONTROL fault=0、device reject=0，工具通过锁存 bridge FAULT 自行退出并确认 DISABLED |
| 2026-09-10 14:12 | A4 原生策略隔离 dry-run | COMPUTE PASS / DEVICE ISOLATED | 500 条唯一来源命令、50.001 Hz；推理 p99 1.257 ms、最大 3.914 ms；bridge 输入已重映射，下位机始终 DISABLED 且命令序号为 0 |
| 2026-09-10 14:12 | 策略/旧下位机轮速合同核对 | HISTORICAL BLOCK | 当前策略 P3/P6 约 -18.6/-16.4 rad/s；旧下位机所有端口只接受 ±2 rad/s，当时不允许直接运行真实 MNN POLICY |
| 2026-09-10 18:26 | 更新下位机源码静态核对 | SOURCE PASS / DEVICE PENDING | P3/P6=±50 rad/s、RS05/RS00 effort=5.5/14 Nm、watchdog=200 ms；主机测试通过，新静态指纹0x4ca27910；bridge恢复±35 rad/s训练合同，等待真实设备只读会话复核 |
| 2026-09-10 18:51 | 更新下位机只读会话与P3/P6原版扫频恢复 | DEVICE PASS / SWEEP READY | 板端实际指纹0x4ca27910、P1～P8/IMU有效、零offline/fault/reject/watchdog、零命令；P3/P6扫频恢复±5 rad/s，保持Kp=0、Kd=0.1、200 Hz、0.1→5 Hz、40 s |
| 2026-09-10 19:02～20:32 | A3 全组扫频与七组固定增益采集 | FIELD PASS | 1/4、2/5、3/6、P7/P8 流程均完成；七组辨识数据按当前 ESD CSV 格式落盘，包括 1/4 的 2/0.1、4/0.2，2/5 的 8/0.8、4/0.2，3/6 的 Kd 0.05/0.1/0.2 |
| 2026-09-10 22:03 | A4 强制零策略实现与自动回归 | CODE PASS / FIELD PENDING | 修正“强制零动作时不发布原生 PolicyCommand”的语义错误；新增 `--force-zero-policy` 与 `scripts/rl/run_zero_policy_test.sh`；inference 1/1、deploy_tools 79/79 通过，等待本轮实机交接 |
| 2026-09-10 22:08 | A4 前真实设备只读预检 | PASS | `logs/rl_preflight_20260910_2208/`；10 秒、2500 条 LowerState、设备严格 250 Hz、指纹 0x4ca27910、P1～P8/IMU 有效，COBS/CRC/序号缺口/reject/watchdog/fault 为 0，命令序号为 0 |
| 2026-09-10 22:12 | supervisor DISABLED 完整预检 | PASS | `--no-rgb --preflight-only` 完成 bridge、模型 hash、手柄 `/joy` 检查；P1～P8 始终 DISABLED；生命周期服务保持 active，未抢占或重启 |
| 2026-09-10 22:30～22:36 | A4 强制零策略实机主路径 | CORE PASS / EXIT ISSUE FOUND | `logs/zero_policy_clean_20260910_2230/`；13729 条原生零动作，来源序号唯一递增，拒绝/COBS/CRC/源缺口/设备非法帧/设备拒绝均为 0；命令到应用 p99 5.14 ms、最大 7.27 ms。Ctrl-C 时 rclpy 先失效造成 watchdog=1，随后已确认 DISABLED；禁止据此放行 A5，先修正常退出 |
| 2026-09-10 22:42 | supervisor 正常退出修复实测 | PASS | `logs/shutdown_handoff_20260910_2242/`；已使能 STANDBY 后 Ctrl-C，supervisor 先确认 lower DISABLED 再退出；最后 211 帧 DISABLED，fault/watchdog/reject/COBS/CRC/序号缺口均零增量，无 ROS 异常栈 |
| 2026-09-10 22:48 | A4 现场观察签署 | PASS | 现场确认 STANDBY/强制零策略期间无异常位移、抖动或异响；A4 完整放行 |
| 2026-09-10 22:54 | A5 三秒实机入口准备 | CODE PASS / FIELD READY | 新增 `scripts/rl/run_policy_smoke_test.sh` 与 `--policy-smoke-seconds 3.0`；到期自动先让 bridge 回 STANDBY、再停策略，且自动记录运行 CSV。正常启动默认不限制 POLICY；deploy_tools 80/80 通过 |
| 2026-09-10 22:55 | A5 真实 STANDBY 姿态隔离 MNN 预演 | PASS | `logs/a5_isolated_preview_20260910_2255/`；原生策略话题重映射，设备仅执行 STANDBY。记录到 94 条非零 MNN 输出，全部有限且位于四路位置和两路轮速合同内：P1 `[0.098,0.556]`、P2 `[-0.417,-0.106]`、P3 `[-22.705,-0.158] rad/s`、P4 `[0.077,0.487]`、P5 `[-0.287,-0.150]`、P6 `[-19.794,-2.525] rad/s`；链路/拒绝/watchdog/fault 零增量，最终 DISABLED |
| 2026-09-10 23:03 | A5 三秒真实 MNN POLICY | PASS | `logs/policy_smoke_20260910_230336/`；bridge 实际 POLICY 约 3 秒并自动回 STANDBY，随后整体失能，最终 lower DISABLED。305 条原生命令全部有限、来源序号唯一递增且未越界；P3/P6 曾触及部署裁剪 `+35 rad/s`，实际速度峰值约 `28.54/28.12 rad/s`。策略读取观测年龄 p99 `4.82 ms`，命令到应用约 `3.12 ms`；COBS/CRC/源缺口/host reject/device reject/watchdog/fault 全为 0。现场确认本次运动不构成异常，无需按异常处置；A5 秒级真实策略验收放行 |
| 2026-09-10 23:27～23:28 | 下地 POLICY 两次回归 | FAIL / GROUND POLICY BLOCKED | 两次均在进入 POLICY 后约一秒内报告 `RUNTIME_DEVICE_LOST: lower controller fault`。第二次会话链路保持 250 Hz，COBS/CRC/序号缺口/离线端口/持久 fault 均为 0；bridge host reject 累计 6、device reject 为 0，随后出现一次 watchdog。推理日志捕获 P4 目标 `+0.874 rad`，超过 bridge 软件上限 `+0.840 rad`，当前 bridge 因单关节越界拒绝整套 P1～P8 帧；旧 `motors_node` 对同类策略目标逐关节裁剪后继续发送。台架 A5 结论保留，但修复该兼容语义前撤销下地 POLICY 放行 |
| 2026-09-10 23:44 | POLICY 逐关节裁剪修复 | CODE PASS / FIELD PENDING | bridge 已按旧 `motors_node` 语义将 P1/P2/P4/P5 的有限位置目标分别裁剪到现有配置边界，不再因单关节越界拒绝整个 P1～P8 帧；P3/P6 `±35 rad/s` 速度合同不变。PTY 集成测试同时越界四路位置，验证完整八路帧正常下发、坐标和裁剪值正确且 reject 不增加；`esd_link_bridge` 18/18 测试通过。已编译到 install，等待下地短时 POLICY 复测 |

## 当前放行点

- 机械硬限位和带裕量的软件运动目标限位已锁定；它们不再规定启动姿势。
- A3 已完成：1/4、2/5、3/6、P7/P8 实机扫频均正常结束，七组固定增益数据已采集。
- 新固件实际会话指纹与全八路反馈只读复核已经通过；P3/P6原版扫频入口已恢复。
  STANDBY、POLICY和既定步态仍按零策略交接、短时真实策略的顺序继续验收。
- A4 已完整通过：原生观测、50 Hz 调度、树莓派 MNN 算力、真实 STANDBY/零命令交接、
  正常退出和现场运动/声音观察均已签署。
- A5 台架路径已完整通过；下地回归暴露的 POLICY 整帧越界拒绝已经按旧
  `motors_node` 语义修复并通过自动测试。完成下地短时复测前，日常开机自启仍只
  放行到 STANDBY。
- 原轮速合同冲突已由下位机 P3/P6 `±50 rad/s` 和 bridge `±35 rad/s` 解决。
  新固件只读复核后，先做强制零动作交接，再短时放行真实 MNN POLICY/既定步态。
