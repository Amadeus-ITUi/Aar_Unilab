# ESD-Link 上下位机操作入口

台架运动、扫频与 RL 分阶段放行记录见
[`ESD_LINK_BENCH_ACCEPTANCE.md`](ESD_LINK_BENCH_ACCEPTANCE.md)。

本页只描述当前默认的 `hardware_backend=esd_link`。旧 SocketCAN 源码保留用于
历史日志和回退排障，不属于当前实机启动链路。

> A4 的 STANDBY、强制零策略、命令应用确认和正常退出已经通过实机验收；A5 的
> 3 秒真实 MNN POLICY 也已完成台架验收。2026-09-10 下地回归发现的策略位置整帧
> 拒绝已经改为逐关节裁剪，并通过伪终端自动测试；完成下地短时复测前，日常手柄入口
> 仍只允许到 STANDBY，不得在下地状态进入 POLICY。

## 安全边界

- bridge 启动后固定为 `DISABLED/READ_ONLY`，不会自动使能。
- P1～P8 是冻结固件的整体使能域；“单电机”只表示该端口目标变化。
- 任何置零、保持、扫频和策略动作都必须有人在场、样机固定在台架上，并可立即硬件断电。
- 本项目自动测试最多运行真实设备只读阶段；自动测试中的使能只发往临时伪终端。

## 已锁定的翼坐标

- 2026-09-09 DISABLED 下手动辨向确认：P7=物理左翼，P8=物理右翼。
- bridge 对冻结下位机的 P7/P8 位置、速度和力矩均取反。上位机统一坐标为
  P7 左翼 `[-0.7,+2.0]`、P8 右翼 `[-2.0,+0.7]`；维护和策略命令下发时自动做逆变换。
- 上述边界来自手动测得的冻结下位机原始 P8 `[-0.7,+2.0]` 和镜像 P7
  `[-2.0,+0.7]`。这是 ESD-Link 命令允许范围；扫频默认仍只在原有小角度区间运行。
- DISABLED 时受重力作用停在软件命令区外、机械硬限位内，不再禁止低增益使能。
  bridge 首帧保持实际当前位置；后续只允许保持或向软件安全区内回收，禁止命令比
  本次使能入口位置更靠近机械端。

## 正常和只读启动

```bash
./start_robot.sh --build
./start_robot.sh --preflight-only
./start_readonly_foxglove.sh
```

正常启动由 supervisor 执行 `BOOT → LINK_CHECK → SYSTEM_READY → WAIT_STANDBY`。
LINK_CHECK 读取 `/lower/link_status` 和 `/lower/state`，检查 schema、配置指纹、
IMU 和 P1～P8 的完整有效状态。250 Hz、帧间隔和 ROS 观察年龄保留在
`LinkStatus` 用于诊断与验收，不再作为上位机的第二套运行期超时门限。
更新后指纹基线为 `0x4ca27910`；只有成功的受控置零流程才会写入
`config/esd_link_identity.txt` 并在后续启动时优先使用新指纹。

## RL 台架入口

强制零策略用于先验证完整的 `STANDBY → POLICY → bridge → 下位机 → 应用确认`
链路。它仍运行真实 MNN 观测和推理，但进入 POLICY 后发给 bridge 的六维动作固定为
零；与 STANDBY 不同，它会持续发布原生 `PolicyCommand`，因此真实覆盖命令序号和
下位机应用确认路径：

```bash
./scripts/rl/run_zero_policy_test.sh
```

若 `we11-deploy.service` 正在持有 RGB，这个测试入口会自动使用 `--no-rgb`，不需要
sudo 停止服务。此时后台绿色呼吸灯只表示手柄已连接，测试状态以终端输出为准；测试
期间不要按方向键上+X/A组合键，以免触发生命周期服务的另一套启停请求。

终端到达 `WAIT_STANDBY` 后按 Y；白灯常亮并完成 STANDBY 后按 X 进入零策略 POLICY。
观察完成后按 X 回 STANDBY，再按 Y 整体失能，最后 Ctrl-C 退出。每次按键均需按下后
释放，不要长按。

首次真实 MNN 使用 A5 专用入口，它会自动记录全部运行话题，并在进入 POLICY 3 秒后
自动按 `POLICY → STANDBY` 顺序交回控制权：

```bash
./scripts/rl/run_policy_smoke_test.sh --no-fault-latch
```

到达 WAIT_STANDBY 后按 Y，STANDBY 完成后按一次 X。3 秒到期后无需再按 X；终端出现
`[SMOKE] ... returning to STANDBY` 后按 Y 整体失能，最后 Ctrl-C 结束。测试期间不要
推动摇杆。记录自动保存到 `logs/policy_smoke_时间/`。

日常正常入口如下；逐关节裁剪修复已经编译，但下地短时复测尚未完成，因此当前只
允许进入 STANDBY 做检查：

```bash
./start_robot.sh
```

需要在生命周期服务运行时从维护终端直接做只读预检，可执行：

```bash
./start_robot.sh --no-rgb --preflight-only --no-fault-latch
```

## 维护入口

直接使用 `ros2 run` 的运动客户端要求 bridge 和 joy 已经启动，且初始模式为
`DISABLED`。`scripts/sweeps/` 下的扫频启动脚本会自动启动缺失的 bridge 和 joy，
并且退出时只清理本次脚本启动的进程：

```bash
# 第一次使能：手柄 A 确认后，P1～P8 低增益保持当前位 5 秒并自动失能
ros2 run deploy_tools esd_hold_current

# 现场人员已明确授权时，也可用 3 秒倒计时取代 A 键
ros2 run deploy_tools esd_hold_current --operator-approved

# A2 单端口正/反小动作；一次只运行一个端口，从 P1 开始
ros2 run deploy_tools esd_small_motion --port 1

# 仅在现场人员已经明确授权且可立即硬件断电时，使用倒计时替代 A 键
ros2 run deploy_tools esd_small_motion --port 1 --operator-approved

# 经配置审核的左右配对 chirp
./scripts/sweeps/sweep_motor14/run_sweep_motor14.sh
./scripts/sweeps/sweep_motor25/run_sweep_motor25.sh
./scripts/sweeps/sweep_motor36/run_sweep_motor36.sh

# 推荐现场流程：一次使能内收至正式姿态、低幅测试、第三次 A 后正式扫频
./scripts/sweeps/run_sweep_test_flow.sh 14
./scripts/sweeps/run_sweep_test_flow.sh 25
./scripts/sweeps/run_sweep_test_flow.sh 36
./scripts/sweeps/run_sweep_test_flow.sh 78

# 补齐当前格式辨识数据中尚缺的四组固定增益；也可将参数换成 all 依次运行
./scripts/sweeps/run_remaining_gain_sweeps.sh 14-kp4-kd0p2
./scripts/sweeps/run_remaining_gain_sweeps.sh 25-kp4-kd0p2
./scripts/sweeps/run_remaining_gain_sweeps.sh 36-kp0-kd0p05
./scripts/sweeps/run_remaining_gain_sweeps.sh 36-kp0-kd0p2

# 任意一个或两个 P1～P6；完整 P1～P8 命令，只有选中目标变化
ros2 run deploy_tools esd_selected_motor_sweep --motor-ids 1
ros2 run deploy_tools esd_selected_motor_sweep --motor-ids 2,5

# P7/P8 镜像扫频
./scripts/sweeps/sweep_motor78/run_sweep_motor78.sh

# P1～P6 置零；P7/P8 必须在线但不会被置零
./set_zeros_1_6_joy.sh
```

失能状态下人工测量全部反馈行程不需要手柄，也不会发布任何命令或调用服务：

```bash
ros2 run deploy_tools esd_manual_range_recorder --duration 180
```

在交互终端中脚本默认显示 P1～P8 实时表格，每路同时给出策略/base_link 坐标与
下位机原始编码器侧坐标的当前值、历史最小值和历史最大值，并显示跨度与当前速度；
P3/P6 会标记为连续旋转轮。界面默认以 10 Hz 刷新，可用 `--ui-hz 2` 降低终端刷新
频率，或用 `--plain` 关闭动态界面。`--duration 0` 表示持续记录，直到 Ctrl-C，退出
时仍会保存 CSV 和摘要。

脚本只允许 upper/lower 都为 DISABLED，并在端口/IMU 无效、链路故障或命令序号变化时
停止。P3/P6 是连续旋转轮，摘要中的位置跨度只表示本次转动量，不是机械软限位。
人工触及的端点不可直接作为控制限位；必须预留机械裕量并检查模型实际动作范围。

本机 2026-09-09 确认后的策略坐标软件限位为：P1 `[-0.89,+0.84]`、
P2 `[-0.89,+0.24]`、P4 `[-0.88,+0.84]`、P5 `[-0.89,+0.22]`、
P7 `[-0.70,+2.00]`、P8 `[-2.00,+0.70]`。P3/P6 只检查速度。以上范围约束主动
运动目标，不再规定启动姿势。当前反馈在范围外时允许以实际当前位置低增益接管；
维护命令在回到安全区之前只能保持或向内运动，扫频工具会限速回收后才开始 chirp。
P3/P6 的上位机速度边界为训练合同的 `±35 rad/s`；更新后的下位机保留
`±50 rad/s` 设备侧余量。

扫频目标以 200 Hz 更新，最大允许 250 Hz；发令由新 `LowerState` 驱动，同一
来源状态序号最多发送一次。CSV 由每个新 `LowerState` 触发记录，
包含状态序号、动作来源序号和下位机已应用命令序号；冻结协议没有温度，字段写为
`unavailable`。

剩余增益采集入口只覆盖当次被扫端口的 Kp/Kd 和输出文件名，不修改基础 YAML、轨迹
或 CSV 字段。当前合同继续采用 1/4 `±0.20 rad`、2/5 `±0.15 rad`、3/6
`±5 rad/s`。直接给 `run_sweep_test_flow.sh` 传入 `--sweep-kp/--sweep-kd` 时，输出
文件也会根据电机组和最终有效增益自动命名；显式 `--output-file` 仅用于覆盖默认名称。

`esd_small_motion` 对 P1/P2/P4/P5/P7/P8 使用最大 `±0.03 rad` 位置动作，
对 P3/P6 使用最大 `±0.5 rad/s` 速度动作。它仍会整体使能并以 200 Hz 发送完整
P1～P8 命令；正向和反向分别在 DISABLED 下等待 A 键确认，每个单方向动作都会
短时使能、回位后自动整体失能。物理方向不正确、出现异响或其他端口移动时不要继续
下一方向，应立即 Ctrl-C 或硬件断电并记录现象。

运行时增益继续使用 `/set_motor_gains`。离线编辑辅助脚本已经改为修改 bridge
配置：

```bash
python3 src/inference/scripts/set_sweep_gains.py --motor-id 1 --sweep-kp 2 --sweep-kd 0.1
```

## 失能和故障

正常退出通过 bridge 请求 P1～P8 整体失能，并允许下位机完成最长约 5 秒的
`SAFE_DAMPING`，最终必须收到 `DISABLED`。bridge 已停止时可独占串口运行：

```bash
./install/esd_link_bridge/lib/esd_link_bridge/esd_link_emergency_disable
```

USB、MCU 或主机已经失效时，软件无法保证发送失能，最终保护必须使用硬件急停或断电。

`/lower/link_status` 会记录主机 COBS/CRC/序号统计、下位机有效/无效/
拒绝命令计数、命令年龄、watchdog 和综合 CONTROL 故障事件。冻结的
schema 1 只把控制周期超时归入 CONTROL 故障位，不提供独立 deadline-miss
计数。

## 自动测试

```bash
colcon test --packages-select motors esd_link_bridge inference deploy_tools
colcon test-result --verbose
```

伪终端测试覆盖 250 Hz、断连、重新枚举、重新建会话、置零后配置身份记录、
来源序号去重、无上位机心跳/超时和整体失能，不访问真实 ESD-SLAVE。
