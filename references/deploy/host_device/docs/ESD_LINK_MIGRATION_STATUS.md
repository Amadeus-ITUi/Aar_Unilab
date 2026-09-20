# ESD-Link 上下位机迁移状态

当前台架运动、扫频和 RL 放行顺序及逐项结果记录在
[`ESD_LINK_BENCH_ACCEPTANCE.md`](ESD_LINK_BENCH_ACCEPTANCE.md)。

更新时间：2026-09-10

## 冻结边界

- 2026-09-10 更新后的 `Phoenix-Slave` 继续使用 schema 1、P1～P8 和 250 Hz 周期；
  P3/P6 速度上限为 50 rad/s，八路 watchdog 为 200 ms，端口力矩上限按 RS05/RS00
  物理范围设为 5.5/14 Nm。完成本轮核对后再次冻结。
- Deploy 仅通过稳定设备路径访问 ESD-SLAVE：
  `/dev/serial/by-id/usb-Sirin_Systems___OmniX_Robotics_ESD-SLAVE_314137523333-if00`。
- 自动化验收只允许 READ_ONLY/DISABLED。使能、零点和运动测试必须由现场人员确认。

## 当前阶段

- [x] `esd_link_msgs` 原生状态、策略、维护、翼、链路状态与控制服务。
- [x] 独立 C++ COBS/CRC32/schema 1 协议库，不依赖 Phoenix-Slave 构建。
- [x] Python codec 的会话/使能/零点/全 8 端口命令黄金帧，以及分片、合帧、噪声、超长帧、坏 CRC 和序号回绕测试。
- [x] 单生产者/单消费者最新帧 mailbox 并发完整性测试；中间状态允许覆盖，最终 generation 必达。
- [x] mailbox 并发回归修复并连续重复 20 轮通过；协议计数器跨线程读取改为原子操作。
- [x] 事件驱动串口线程、单槽最新状态、普通优先级 ROS 发布线程。
- [x] 串口独占锁、`SCHED_FIFO 70`、可选 CPU 亲和性、断线后禁止自动恢复使能。
- [x] READ_ONLY 实机短测：schema 1、掩码 `0x1FE`、约 250 Hz、无 CRC/COBS/序号缺口。
- [x] 推理节点原生 `LowerState` 单快照输入及带来源序号的 `PolicyCommand` 输出。
- [x] supervisor 默认启动 bridge，不再启动 SocketCAN 电机、独立 IMU或直接 CAN 翼节点。
- [x] supervisor 可验证状态机已移除不可达的旧 CAN/IMU/翼定位状态；
  正常路径为 `BOOT → LINK_CHECK → SYSTEM_READY → WAIT_STANDBY → STANDBY/POLICY`。
- [x] Foxglove 只读启动脚本切换为 ESD-Link READ_ONLY。
- [x] 1/4、2/5、3/6 引导扫频入口切换到完整 P1～P8 `MaintenanceCommand`；旧直连 CAN 实现仅保留为 legacy 源码。
- [x] 任意单/双 P1～P6 扫频提供 `esd_selected_motor_sweep`；仅选中目标变化，命令仍完整覆盖 P1～P8。
- [x] P7/P8 扫频迁移到 ESD-Link、全端口在线与三次手柄确认；旧直接 CAN 实现移入 `experiments/wing/legacy`。
- [x] 扫频仍为 200 Hz，反馈 CSV 由每个新 `LowerState` 驱动，并记录状态/来源/应用命令序号；温度标记 `unavailable`。
- [x] 运行记录器加入原生状态、策略来源序号、命令应用确认和 ESD-Link 延迟/错误统计，移除运行时 candump。
- [x] 运行记录器中不再包含 candump/CAN 接口隐藏参数或不可达死代码。
- [x] `LinkStatus` 透传下位机有效命令、无效帧、拒绝命令和命令年龄，
  并分别累计 watchdog 与综合 CONTROL 故障上升沿。schema 1 不单独暴露
  deadline-miss 计数，因此不返回伪造的精确值。
- [x] POLICY 只接受该模式下发布的原生动作；本次模式尚未收到首条有效策略前，翼控制不得复用旧策略缓存。
- [x] POLICY、SWEEP、MANUAL_TEST 不再设上位机命令超时；每个新状态序号最多产生一条命令，
  断流和命令停止统一由下位机 200 ms 看门狗处理。
- [x] 腿部策略/下位机坐标合同抽成共用实现并单测；维护、手动和 `read_motors` 使用同一坐标定义。
- [x] 策略、维护和手动目标执行有限性、来源序号归属、关节/翼位置及轮速限制检查。
- [x] SET_ENABLE/SET_DISABLE 除事务响应外还等待下位机状态确认；失能允许冻结固件完成最长 5 秒 SAFE_DAMPING。
- [x] STANDBY 三秒斜坡完成后持续固定 P1～P6 默认姿态，P7/P8 固定为进入模式时的实际角度，不跟随反馈漂移。
- [x] 下位机主动报告 SAFE_DAMPING、FAULT_LATCHED、故障或端口离线时，通信线程异步排队整体失能，不在自身线程同步等待。
- [x] P1～P6 手柄置零脚本改为只读 bridge 流程；成功后原子保存新配置指纹并重建会话。
- [x] 更新后启动基线指纹为 `0x4ca27910`；受控置零后的本机 identity 记录优先于 YAML 基线。
- [x] `set_sweep_gains.py` 改为编辑 bridge 配置；`motion_player.py` 明确标记 legacy。
- [x] 原 400 Hz 扫频工具的默认值和强制上限降为 250 Hz；新 ESD 扫频默认仍为 200 Hz。
- [x] 伪终端集成：250 Hz 状态、只读零控制输出、断连/重新枚举/重建会话、置零后身份指纹、STANDBY 斜坡、越限丢弃、来源序号去重和整体失能。
- [x] 通信线程不调用 ROS/文件 I/O；串口和 TX eventfd 都立即唤醒 `poll()`，静默时不做运行期超时轮询。
- [x] 维护命令、置零脚本与直连 CAN 工具的安全入口已迁移；ESD bridge 活动时直连 CAN 工具明确拒绝运行。
- [x] 首次使能提供独立的 P1～P8 当前位低增益保持工具；手柄确认后进入
  MANUAL_TEST，越位/超速/下位机显式故障会中止，正常完成后也等待 DISABLED 确认。
- [x] 首次实机发控安全中止路径已验证；约 202 条命令全部有效，命令应用延迟约
  `1.94 ms`，零拒绝/watchdog/CONTROL 故障。当时由过严的上位机 20/40 ms 门限导致的误中止
  已在 2026-09-10 精简中删除。
- [x] 复位、清错、识别 ID、扫描、主机 ID 修改和旧安全检查服务均明确返回 `unsupported by frozen ESD-Link firmware`。
- [ ] 台架安全失能、保持、小动作、维护和 RL 分阶段人工验收。

## 上一版固件已测实机事实（历史基线）

- lower boot id：`2048817021`
- config fingerprint：`0x6fc12897`
- active mask：`0x000001fe`
- schema：`1`
- 状态频率短测：`249.7～250.0 Hz`
- 短测错误：COBS `0`、CRC `0`、sequence gap `0`、offline `0`、fault `0`

这些值仅是本次运行记录。正式启动仍由 LINK_CHECK 动态验证，不把 boot id 或 session id 写死。

### 2026-09-10 下位机配置更新与静态核对

- 已解压 `Phoenix-Slave/esd-h7-slave.zip`；压缩包 SHA-256 为
  `39f8da9bdc2a6561f62d8ba30fa010aec0e2934419ee2871a7a7be5e3c025de8`。
- 新端口表确认 P3/P6 `velocity_max_rad_s=50`，其余端口保持 `2`；RS05端口
  `effort_max_nm=5.5`，RS00端口为 `14`；P1～P8 `watchdog_ms=200`。
- 端口注册、端口指纹库和 RobStride 主机测试全部通过。使用与固件完全相同的
  `esd_port_compute_fingerprint()` 交叉计算：旧端口表得到既有 `0x6fc12897`，新端口表
  得到 `0x4ca27910`，因此 Deploy 基线已同步到 `0x4ca27910`。
- bridge P3/P6 上位机边界已恢复为训练合同 `±35 rad/s`，相对下位机
  `±50 rad/s` 保留余量。下一步必须在真实设备上只读确认会话实际报告
  `0x4ca27910` 后，才继续 STANDBY/零策略/真实策略验收。
- Deploy 五个相关包增量编译通过；定向测试为 bridge `22/22`、deploy_tools
  `70/70`、motors `9/9`、inference `1/1`，合计 `102/102`。当前稳定设备路径未
  枚举，因此尚未进行真实串口会话或任何使能/运动操作。
- 下位机端口表检查点为 `Phoenix-Slave/esd-h7-slave` 的 `9cf431c`；Deploy 的指纹、
  轮速合同、伪终端回归和操作文档检查点为 `4738156`。
- 本机没有 `arm-none-eabi-gcc` 和 Ninja，压缩包自带 CMake cache 又绑定原开发机
  `/home/angela/ssd/...`，因此这里未宣称完成整套MCU交叉构建；已完成的是端口源码
  的主机编译/测试、真实CRC实现的指纹计算以及Deploy全套定向回归。
- P3/P6 ESD-Link扫频已从旧固件约束下的 `±1.8 rad/s` 恢复到原版
  `±5.0 rad/s`；入口校验同步以 `5.0` 为上限。该幅值低于bridge的 `35 rad/s` 和
  更新下位机合同的 `50 rad/s`，保持原 `Kp=0`、`Kd=0.1`、200 Hz、0.1→5 Hz配置。
- 设备重新枚举为 `/dev/ttyACM2` 后完成只读会话复核：板端实际报告
  `config_fingerprint=0x4ca27910`、schema 1、active mask `0x1FE`；P1～P8
  `valid_mask=3`、IMU mask `7`、offline/fault/reject/watchdog均为0。lower保持
  `DISABLED`，命令发送/应用序号均为0，本次没有调用使能或产生运动。

### 2026-09-09 再验证状态

- 稳定路径已从早先的 `ttyACM2` 重新枚举为 `ttyACM0`，bridge 仍通过
  `/dev/serial/by-id/...` 成功建立会话。
- 12 秒只读记录收到 3003 个 `LowerState`，平均约 `249.1 Hz`；链路统计
  为 `250 Hz`、最大帧间隔约 `4.14 ms`、COBS/CRC/序号缺口均为 `0`。
- 当前下位机报告 P1～P8 `valid_mask=0`、`offline_port_mask=0x1FE`、
  `fault_flags=0x8 (ACTUATOR_OFFLINE)`。因此 LINK_CHECK 应当拒绝启动，本次没有
  进行使能、置零或运动。
- 检查点 `8119a3c` 的最终二进制已在真实设备上再次建立会话，通过
  `0x6fc12897` 指纹校验并输出下位机原生计数；链路仍为 `250 Hz`、
  最大帧间隔约 `4.12 ms`、COBS/CRC/序号缺口均为 `0`。
- 恢复八路执行器供电/总线反馈后，需要重新完成 30 分钟 READ_ONLY 验收；
  当前短测不可代替该阶段。

### 2026-09-09 执行器重新供电后的现场复测

- 现场确认执行器已重新供电后，以最终二进制在隔离 ROS domain 中完成约 15 秒
  READ_ONLY 记录；全程没有调用使能、置零或控制服务，
  `last_command_sequence=0`、`last_applied_command_sequence=0`。
- USB CDC、session、schema、配置指纹和 IMU 均正常。稳定采样段为 `250 Hz`，
  最大帧间隔约 `4.48 ms`，COBS/CRC/序号缺口均为 `0`。
- P7、P8 已恢复有效反馈（`valid_mask=3`），但 P1～P6 仍无反馈，设备报告
  `offline_port_mask=0x7e`、`fault_flags=0x8 (ACTUATOR_OFFLINE)`；bridge 正确保持
  `FAULT/DISABLED`，没有进入运动测试。
- 冻结下位机端口映射为 P1～P3→CAN2、P4～P6→CAN1、P7～P8→CAN3；CAN1、
  CAN2 两组同时离线而 CAN3 正常，应优先现场检查腿部公共动力电源、CAN1/CAN2
  收发与线束，以及执行器先上电后下位机重新启动的上电顺序。下位机每个控制周期
  都会重新判定反馈新鲜度，因此上位机不得绕过全端口在线保护。
- 本次记录位于忽略目录 `logs/read_only_power_restored_20260909_135043/`。必须先恢复
  `offline_port_mask=0`、P1～P8 `valid_mask=3`、`fault_flags=0`，之后才开始 30 分钟
  READ_ONLY 和任何使能/保持测试。

### 2026-09-09 下位机重新上电后的八路复测

- 下位机重新上电后 boot id 更新为 `2904551548`，稳定设备路径重新枚举到
  `ttyACM2`，by-id 路径未变化。
- 12 秒 READ_ONLY 记录中 P1～P8 每条记录均为 `valid_mask=3`，
  `offline_port_mask=0`、`fault_flags=0`、IMU valid mask 为 `7`；未调用控制服务，
  上下位机命令序号均为 `0`。
- bridge 接收统计稳定为 `250 Hz`，最大串口帧间隔约 `4.14 ms`，COBS/CRC/下位机
  状态序号缺口均为 `0`。ROS 记录端收到 2923 条状态，p99 接收间隔约
  `4.12 ms`、最大约 `11.93 ms`；记录器/DDS 侧观察到 2 个被覆盖的中间状态，符合
  latest-only 单槽语义，但需在 30 分钟验收中继续计算比例并与链路侧零丢帧分开报告。
- 短测记录位于忽略目录
  `logs/read_only_after_lower_repower_20260909_140047/`。下一阶段为 30 分钟
  READ_ONLY 验收，仍不使能、不置零。

### 2026-09-09 30 分钟 READ_ONLY 验收

- 隔离 ROS domain 连续记录 `1800.05 s`；下位机状态序号从 `55625` 连续到
  `505638`，共 `450014` 个源状态，设备采样时间换算严格为 `250.000 Hz`。
- bridge 链路统计平均 `249.992 Hz`（一秒统计窗为 `249～251 Hz`），下位机状态
  序号缺口、COBS、CRC、offline、fault、拒绝命令、watchdog 和控制故障均为 `0`。
- 已发布状态的接收间隔 p99 约 `4.12 ms`；bridge 原始每秒窗口记录到的最大帧
  间隔为 `12.97 ms`，低于 `20 ms` 上限。最新状态年龄 p99 约 `4.01 ms`、最大
  约 `4.12 ms`。
- P1～P8 在全部 `LowerState` 记录中均为 `valid_mask=3`，IMU valid mask 均为
  `7`；控制状态始终为 lower `DISABLED`、upper `READ_ONLY`。策略、维护和控制命令
  样本均为 `0`，上下位机命令序号均保持 `0`。
- 普通优先级 ROS/CSV 消费端收到 `449073` 条 `/lower/state`，latest-only 单槽覆盖
  了 `941` 个中间状态（`0.209%`），但 bridge 通信线程的源状态缺口为 `0`，且策略
  始终消费最新完整快照。验收报告必须区分“串口链路零丢帧”和“ROS 观察/记录端
  允许覆盖中间帧”这两个指标。
- 记录位于忽略目录 `logs/read_only_30min_20260909_140314/`。30 分钟链路与最新
  观测时效验收通过。

### 2026-09-09 翼物理左右与方向已现场锁定

- 两次辨向全程保持下位机 `DISABLED`，没有发送使能、置零或控制命令。
- 右翼单独手动下压时，P7 保持在 `-1.436 rad`，只有 P8 从约
  `+1.579 rad` 减小到 `+1.048 rad` 后回位。因此 **P8=物理右翼**，下压时
  冻结下位机 P8 坐标减小。原始记录在忽略目录
  `logs/wing_direction_manual_20260909_144305/`。
- 左翼单独手动下压时，P8 保持在 `+1.579 rad`，只有 P7 从约
  `-1.531 rad` 增大到 `-0.995 rad` 后回位。因此 **P7=物理左翼**，下压时
  冻结下位机 P7 坐标增大。原始记录在忽略目录
  `logs/wing_direction_left_20260909_144603/`。
- bridge 因此对 P7/P8 都应用 `wing_sign=-1`：兼容/策略坐标为
  P7 左翼、P8 右翼，而命令下发时做同样的逆变换。允许目标边界由后续手动范围测试锁定。
  训练观测顺序为 `[left,right]`，公式锁定为 `[-P7/π, P8/π]`，竖直向上时
  两个训练观测均约为 `-0.5`。
- 坐标修正已覆盖 `LowerState`、兼容翼话题、运行状态、维护/手动/策略命令与
  推理观测；单元测试和伪终端反馈/命令往返验证已通过。实机使能前仍必须用最终
  二进制再做一次 READ_ONLY 坐标检查，并确认当前角度没有超出软限位。

### 2026-09-09 翼坐标修正后 READ_ONLY 复核

- Git 检查点 `32c1bf2` 的最终二进制已在真实设备上建立会话；整个过程没有调用
  控制服务，lower 保持 `DISABLED`，上下位机命令序号均为 `0`。
- 原生 `/lower/state` 与兼容 `/policy/wing_angles` 一致输出
  P7 左翼约 `+1.531 rad`、P8 右翼约 `-1.579 rad`，证明反馈坐标修正已生效。
- 12 秒 ROS 观察平均约 `249.7 Hz`；P1～P8 `valid_mask` 均为 `3`，
  `offline/fault/COBS/CRC/sequence gap` 均为 `0`。本次复核后 bridge 已干净退出。
- 现场进一步手动测试确认：冻结下位机原始 P8 可用范围约为
  `[-0.7,+2.0] rad`，P7 按机械镜像为 `[-2.0,+0.7] rad`。原先的 `±π/2` 是旧后端
  的常用目标范围，不是本机械的反馈安全边界，不应将正常零点偏差判成越界。
- 换算到 bridge 上位机坐标后，允许目标范围更新为
  P7 左翼 `[-0.7,+2.0]`、P8 右翼 `[-2.0,+0.7]`。当前 P8 `-1.579 rad` 因此合法，
  上一检查点记录的 `0.47°` 假越界阻断已解除。扫频默认中心/幅值与 RL 模型均未改动，
  不会因边界扩大而自动运动到新极限。
- 新配置于真实设备再次只读加载成功；本次 lower boot id 为 `118896887`，
  bridge 统计 `250 Hz`，P1～P8 `valid_mask=3`、IMU mask `7`，
  `offline/fault/COBS/CRC/sequence gap=0`。当前翼角约为 P7 `+1.530 rad`、
  P8 `-1.494 rad`；lower 保持 `DISABLED`，命令序号均为 `0`，复核后 bridge 已干净关闭。

## 安全命令

只读启动：

```bash
./start_readonly_foxglove.sh
```

bridge 停止后独占串口执行整体失能：

```bash
./install/esd_link_bridge/lib/esd_link_bridge/esd_link_emergency_disable
```

协议测试：

```bash
colcon test --packages-select esd_link_bridge
colcon test-result --verbose
```

当前迁移包自动测试基线：`92 tests, 0 errors, 0 failures, 0 skipped`。其中伪终端测试只使用临时 PTY 和随机 ROS domain，不打开真实设备。测试覆盖首次低增益保持的稳定状态、A2 小动作边界/反馈、被动行程统计与终端界面、位置偏移拒绝、任意端口超速拒绝，以及当前位置越过软件限位时不得向下位机发送 enable。

### 2026-09-09 A1 当前位低增益保持进展

- 两次 5 秒 P1～P8 当前位低增益保持均完成运动段，现场人员均确认“无位移”。
  已保存的第二次记录为 `logs/esd_hold_current_20260909_155501.csv`，包含 1260 条
  新状态；前一次记录中位置型端口最大偏移小于 `0.001 rad`，八路最大速度小于
  `0.24 rad/s`，下位机命令拒绝、watchdog 和 CONTROL 故障均为 `0`。
- 第二次测试的正常失能已经由下位机确认，但 bridge 在失能事务等待期间仍按
  `MANUAL_TEST` 的 40 ms 维护命令期限进行检查，产生了一次仅位于上位机日志的
  `maintenance command timeout`。`LinkStatus.control_fault_events=0` 只表示下位机
  CONTROL fault 位没有出现，不能替代 bridge 日志对上位机故障的核查。因此此轮记为
  “运动通过、收尾失败”，尚不将 A1 整体签署为 PASS。
- bridge 已在显式整体失能期间停止接受/发送活动控制命令，并暂停状态、策略和维护
  超时判定；失能成功前保持原模式，失败则锁存 FAULT。另修正 SET_ENABLE 确认竞态：
  事务 ACK 后必须观察到至少一帧新的目标控制状态，禁止用事务前的旧 DISABLED 状态
  提前宣称失能完成。
- 伪终端将 SAFE_DAMPING 人为延长到 `120 ms`（超过维护期限），验证正常失能只有
  一次请求、不会误触发维护故障，并验证维护超时和状态断流仍会触发安全失能。该端到端
  用例连续运行 5 次通过；`motors`、`esd_link_bridge`、`inference`、`deploy_tools`
  合计 `74 tests, 0 errors, 0 failures, 0 skipped`。
- 下一步必须用修正后的最终二进制重跑一次 A1，并同时满足：现场无位移、工具成功、
  下位机 DISABLED 确认、bridge 无 `control fault` 日志。此前不放行 A2 或扫频。

#### 修正后二进制的 A1 最终复跑

- 16:22 使用检查点 `6747a70` 构建的 bridge 完成最终复跑，工具报告控制段
  `5.00 s`、CSV 共 1381 条新状态，并在保存文件前收到 P1～P8 整体 DISABLED 确认。
  记录位于忽略目录 `logs/esd_hold_current_20260909_162258.csv`。
- 八路反馈的最大位置范围为 `0.00115 rad`，最大绝对速度为 `0.181 rad/s`，均远低于
  A1 的 `0.08 rad`、`2 rad/s` 门限。普通 ROS 记录端覆盖了 3 个中间状态，bridge
  串口源状态序号缺口仍为 `0`。
- 测试后的 LinkStatus 为 upper/lower `DISABLED`、250 Hz、offline/fault/COBS/CRC/
  reject/watchdog/CONTROL fault 全部为 `0`；最后命令序号与应用确认序号均为 `1003`，
  最近观测到命令约 `3.25 ms`、命令到应用确认约 `0.75 ms`。
- bridge 本轮 ROS 日志没有 `control fault` 或 `maintenance command timeout`，随后
  bridge 与手柄节点均干净退出。现场人员确认本轮“无位移、无异常声响”，因此 A1
  的自动数据、软件收尾和机械观察均通过，正式签署为 PASS。下一阶段为 A2 低增益、
  分端口小动作辨向；扫频、STANDBY、POLICY 和既定步态仍未放行。

### 2026-09-09 A2 分端口小动作工具就绪

- 新增 `esd_small_motion --port P`，一次只改变一个端口目标，但遵守冻结下位机合同，
  每条命令仍覆盖 P1～P8，且所有端口作为一个整体使能/失能。
- P1/P2/P4/P5/P7/P8 使用上位机坐标 `±0.03 rad` 低速斜坡，P3/P6 使用
  `±0.5 rad/s` 速度命令。低增益与 A1 相同；未选择的位置型端口偏移超过
  `0.03 rad`、测试端口超出包络、任意端口超速、状态过期或链路无效都会中止并请求
  整体失能。
- 当前实现令正向、反向分别在 DISABLED 下等待手柄 A 确认；每个单方向动作短时使能，
  自动回到初始目标并整体失能，且要求反馈达到最小响应。CSV 保存状态/来源/应用序号
  及完整八路命令反馈。
- 安装后的 `ros2 run deploy_tools esd_small_motion --help` 已验证；新增 5 个目标限位、
  非选中端口漂移和双向响应测试，`deploy_tools` 为 52/52，四个迁移包总基线更新为
  80/80。尚未执行 A2 实机动作，第一项固定为 P1。
- P1 首次尝试中，第一次 A 只进入当前位保持；在等待第二次 A 时 ROS 普通优先级
  监视端出现超时，bridge 的 40 ms 维护期限同步触发整体安全失能。P1 实际位置范围
  仅 `0.000383 rad`，可确认没有执行正向动作；下位机 reject/watchdog/CONTROL fault
  均为 `0`。bridge 源状态累计出现 2 个序号缺口，本轮按严格标准记为 SAFE ABORT。
- 小动作工具随后取消“使能后无限等待人工确认”：每个方向都先在 DISABLED 下确认，
  然后短时整体使能、执行单方向、自动回位并立即失能；正反方向成为两个互相隔离的
  有界控制窗口。修改后 `deploy_tools` 52/52 测试通过，尚未再次实机运行。

### 2026-09-09 DISABLED 人工行程测量

- 按现场意见新增 `esd_manual_range_recorder`，不依赖手柄，没有 publisher 或 service
  client；只在 upper/lower 均为 DISABLED、P1～P8 与 IMU 全有效时记录每个新
  `LowerState`。命令序号发生变化或任何状态失效都会停止。
- 2 秒只读冒烟测试收到 501 帧且所有统计为零；随后正式记录 `180.002 s`、45004 帧，
  记录器序号缺口与 bridge 源序号缺口均为 `0`，COBS/CRC/reject/watchdog/CONTROL
  fault 均为 `0`，`last_command_sequence` 与 `last_applied_command_sequence` 始终为
  `0`。正式数据位于忽略目录 `logs/manual_range_20260909_164704/`。
- bridge 上位机策略坐标的人工观测范围如下。P3/P6 为连续旋转轮，其位置跨度只表示
  本次手动转动量，不构成机械限位：

| 端口 | 最小位置 rad | 最大位置 rad | 跨度 rad |
|---|---:|---:|---:|
| P1 | -0.979561 | +0.931779 | 1.911340 |
| P2 | -0.974867 | +0.330283 | 1.305149 |
| P3 | -5.071588 | +0.243556 | 5.315144 |
| P4 | -0.960757 | +0.933709 | 1.894466 |
| P5 | -0.978280 | +0.308539 | 1.286818 |
| P6 | -6.055153 | +6.095572 | 12.150725 |
| P7 | -0.794871 | +2.084028 | 2.878899 |
| P8 | -2.079388 | +0.773510 | 2.852898 |

- 第二次现场记录位于 `logs/manual_range_20260909_170608/`，持续 `153.645 s`、
  收到 38395 帧；同时运行的独立记录位于 `logs/manual_range_20260909_170523/`，持续
  `318.182 s`、收到 79541 帧。两份记录的 bridge 源序号缺口、COBS、CRC、reject、
  watchdog、CONTROL fault 和命令序号均为 `0`，且端点逐路一致。现场人员确认这些是
  本机机械硬限位。
- 基于第二次确认值向内取整，锁定策略坐标软件限位：P1 `[-0.89,+0.84]`、
  P2 `[-0.89,+0.24]`、P4 `[-0.88,+0.84]`、P5 `[-0.89,+0.22]`、
  P7 `[-0.70,+2.00]`、P8 `[-2.00,+0.70]`；各机械端保留不少于 `0.075 rad`
  裕量。P3/P6 继续只做速度限制和位置卷绕。
- bridge、A2、单/双端口维护扫频、配对扫频和翼扫频已统一使用这组运动目标限制。
  旧 2/5 扫频中心 `+0.41338 rad` 超过机械硬限位，
  已改为中心 `0`、幅值 `0.15 rad`；所有配对扫频配置都有启动前范围验证。
- 17:26 版本曾把运动目标软限位错误地用作启动姿态限制；P2/P5/P8 在重力作用下靠近
  机械端时会被拒绝使能。该限制已撤销：允许实际当前位置低增益接管，维护命令在进入
  软件安全区前只能保持或向内回收，任何比本次使能入口更靠机械端的目标仍会整体失能。
- 配对、任意单/双端口和翼扫频会先按配置速度把越界位置关节向内回收到安全目标，再
  开始 chirp。新增 `scripts/sweeps/run_sweep_test_flow.sh`，在一次使能内按“收至正式
  姿态→8 秒低幅扫频→A 键确认→正式扫频→整体 DISABLED”执行。bridge 与
  deploy_tools 相关自动测试通过，包含越界启动保持、合法向内回收和非法向外命令失能。
- 18:41 首次 1/4 低幅现场流程已完成使能和向内回收，但扫频工具仍残留 `20 ms` 的
  ROS 观察端门限，在一次 `24.249 ms` 的 DDS/调度延迟上主动中止；整体失能已确认。
  bridge 日志中的后续 maintenance timeout 是工具退出停止刷新后的安全收尾，不是
  下位机链路断流。
- 18:44 重跑的 1/4 低幅阶段完成并正常整体失能；随后旧脚本另起第二套三确认正式流程，
  既重复操作，也只在第一段把越界关节夹到软边界而没有收至正式扫频姿态。正式 chirp
  中普通 ROS 状态观察再次瞬时超时，工具退出后整体失能已确认。
- 流程已合并为单次使能和三次 A：第一次接管并限速收至正式扫频姿态，第二次开始 8 秒
  低幅 chirp，第三次确认后直接开始正式 chirp，最后整体失能。配对扫频的 200 Hz 发令
  已从 ROS 回调/CSV 线程拆为独立心跳，普通优先级 DDS 观察抖动不再中断命令刷新；
  bridge 仍以 31 帧/约 `124 ms` 来源历史窗口拒绝真正过期的命令，通信线程的原始状态
  断流硬门限保持 `20 ms`。异常退出会保存失败前 CSV 和具体状态原因。
- 19:11 单次流程已完成回收和 8 秒低幅 chirp，但等待第三次 A 时 bridge 因
  `maintenance command timeout` 进入 `control_state=5`。扫频端因此进一步删除了首版
  独立心跳与主线程共享的命令锁；主线程原子替换完整命令快照，200 Hz 心跳只读取快照，
  不会被 ROS 回调、CSV 记录或按键等待持锁阻塞。
- 19:22 无锁版本再次运行并进入正式 chirp 后，同一错误复现。失败 CSV 在 bridge 报错
  时显示 `last_applied_command_seq` 仍每约 `4–8 ms` 递增，证明维护命令正在持续到达并
  被下位机应用。最终根因是 bridge 多线程时间竞态：检查线程先读取 `now`，命令线程随即
  写入比该 `now` 更新的单调时钟时间戳，旧代码的无符号减法下溢为巨大间隔并误判 40 ms
  超时。所有跨线程超时计算已改为防下溢的饱和时间差；真实旧时间戳仍按原门限失能。
- 19:38 修复后 1/4 流程完成回收、8 秒低幅和 40 秒正式 chirp，最后正常进入
  SAFE_DAMPING；bridge 全程无 control fault。扫频客户端只等待模式服务 `5 s`，短于
  bridge 的 `5 s` SET_ENABLE 事务窗口加 `6.5 s` 失能确认窗口，因此在安全阻尼结束前
  误报服务超时，随后收尾重试已确认 P1-P8 整体 DISABLED。所有维护工具的模式服务等待
  已统一为 `13 s`，并增加正式 chirp 开始、完成及安全阻尼等待提示。
- 14、25 配对流程随后均已现场完成。19:53 的 3/6 速度流程通过 `0.5 rad/s` 低幅段，
  正式段目标升至约 `2.1 rad/s` 后下位机停止应用命令，并在约 40 ms 后进入
  SAFE_DAMPING。重开只读 bridge 取得冻结下位机累计诊断：`device_rejected_commands=11`、
  `last_reject_code=9 (INVALID_STATE)`，链路 250 Hz、零 COBS/CRC/序号缺口、端口全在线。
  这表明旧 `5 rad/s` 配置超过当前冻结下位机实际接受范围；3/6 正式幅值已降为
  `1.8 rad/s`（相对约 `2 rad/s` 拒绝边界保留 `0.2 rad/s` 裕量），工具启动前拒绝更大值。
- 被动记录器已增加终端动态表格：P1～P8 同时显示当前策略/base_link 坐标、历史
  min/max/跨度、由现有坐标合同反算的下位机原始编码值和当前速度；P3/P6 明确标为
  连续轮。界面没有 publisher 或 service client，仍由 DISABLED、有效掩码、故障和
  命令序号不变四类条件硬约束。当前 `deploy_tools` 69/69、bridge 22/22，共 91 项
  相关测试通过。
- 当时下一步为重新执行 1/4 单次扫频流程，实机验证无锁心跳能跨过第三次 A 等待并完成
  正式 chirp；失败时仍整体失能并保留失败前 CSV。

### 2026-09-10 上位机保护层精简

- 已将 2026-09-09 完成的实机扫频修复、自然重力位置接管和时间戳竞态修复归档为
  Git 检查点 `56811dc`；精简工作从该可编译、可测试基线继续。
- bridge 已删除运行期 `20 ms` 状态超时和 `40 ms` POLICY/MAINTENANCE 超时，
  `LinkStatus` 中的状态年龄、频率和帧间隔保留为纯诊断数据，不再改变控制模式。
- 策略和维护命令现在必须引用 bridge 确实接收过、且相对上一条已接受命令更新的
  `state_sample_seq`。重复或乱序来源、NaN、非法布局和越限目标只丢弃该条命令并累计
  `rejected_commands`，不再把健康链路切入全局 FAULT。持续没有新有效命令时，由冻结
  下位机的 `50 ms` 看门狗作为唯一运行期 deadline。
- 下位机明确报告 fault、端口 offline 或 FAULT_LATCHED 时仍停止发令并异步请求整体
  失能；USB 断开、会话身份校验、完整 P1～P8 命令、坐标合同和已确认软件目标限位
  均保持不变。SAFE_DAMPING 被视为正常的下位机安全过渡，并使上位机控制源回到
  DISABLED，而不是制造第二个上位机故障。
- 通信线程继续使用 `SCHED_FIFO 70`、`poll()`/eventfd 和预分配缓冲；默认 CPU 亲和
  从 CPU3 改为 `-1`，允许 Linux 在四核间调度，仍可通过参数进行对照测试。
- 第一轮代码已编译；更新后的 250 Hz PTY 集成测试验证：越限命令只丢弃、重复状态
  序号不重复下发、没有维护命令或状态静默时 bridge 不生成心跳也不触发上位机超时。
  尚未启动真实串口或使能实机。
- 推理节点的原生 ESD 路径现在只从单条 `LowerState` 原子构造 IMU、P1～P8 观测；
  每个 `state_sample_seq` 最多推理一次，断流时不会继续用旧观测发新命令。原生路径删除
  分话题 IMU/翼反馈的 ROS 年龄门限；这些参数只保留给 legacy 后端。`inference`
  已编译，新增序号去重/回绕测试并通过。
- P1～P6 配对扫频已删除重复旧状态的独立 200 Hz 心跳线程，改为由新
  `LowerState` 驱动发令；目标更新与发令解耦，每个来源序号最多发送一次。P7/P8
  扫频、当前位保持和小动作也删除了 ROS 年龄门限，且不重复发送同一来源序号。
  脚本语法检查通过，相关 `deploy_tools` 单测 16/16 通过。
- supervisor 不再订阅独立 IMU、关节、翼角和兼容运行状态来拼凑放行条件，改为直接
  验证一条原生 `LowerState`。`state_rate_hz`、`latest_state_age_ms` 和
  `maximum_interarrival_ms` 只作诊断，不再阻断启动或运行。启动仍要求会话、schema、
  指纹、P1～P8/IMU 有效且数值有限。
- STANDBY↔POLICY 交接顺序已调整：进入时先让推理开始产生新命令，再将 bridge 控制权
  切到 POLICY；退出时先恢复 bridge 的 STANDBY 命令，再停止策略输出。交接期间不再
  留出可触发 50 ms 下位机看门狗的空窗。supervisor 定向测试 10/10、`deploy_tools`
  全包回归 69/69 通过。
- `esd_link_msgs`、`motors`、`esd_link_bridge`、`deploy_tools`、`inference` 五包已完整编译。
  bridge 协议/mailbox/坐标/时间工具/PTY 共 17 项、`deploy_tools` 69 项、推理 1 项，
  合计 87 项全部通过。
- 精简后二进制已在真实 ESD-SLAVE 上完成 15.0018 s READ_ONLY 复测；记录 3750 条
  `LowerState`，设备序号和采样时间换算为 `250.000 Hz`。bridge 源序号缺口、COBS、
  CRC、拒绝命令、watchdog 和 CONTROL fault 增量均为 0；P1～P8/IMU 全部有效。
  upper/lower 全程 DISABLED，命令发送/应用序号均为 0。普通 ROS/CSV 记录端覆盖
  1 个中间状态，不属于串口源链路丢帧。记录在忽略目录
  `logs/read_only_simplified_20260910_135427/`。本轮没有使能或运动。
- 现场授权后以当前反馈为目标完成 5 s P1～P8 整体低增益保持；不回位，P2/P5 保持
  重力自然姿态。状态驱动命令流产生约 997 个唯一来源序号，约 `199.4 Hz`，
  无来源序号倒退或超前。位置型端口最大命令/反馈差小于 `0.00085 rad`，八路最大速度
  小于 `0.168 rad/s`；正常整体失能已确认。事后 READ_ONLY 会话显示 lower
  `DISABLED`、P1～P8/IMU 有效、fault/offline/invalid/rejected/watchdog/CONTROL 均为 0。
  记录位于 `logs/simplified_hold_20260910_140113/`。这一步的传感器验收通过；异常声响
  需由现场人员确认。
- 冻结下位机单一 deadline 已实机验证：当前位低增益保持期间将维护进程暂停
  200 ms，下位机产生恰好 1 次 watchdog、0 次 CONTROL fault、0 设备无效帧、0 设备拒绝，
  并进入 SAFE_DAMPING。bridge 没有上位机 20/40 ms 先行超时。维护工具已增加对锁存
  `LinkStatus/FAULT` 的响应；复测时工具自行中止且确认整体 DISABLED，无需强制终止。
  故障交接边缘 bridge 丢弃 2 条上位机维护命令，未下发到设备；下位机累计
  `last_reject_code=9` 来自 2026-09-09 P3/P6 旧测试，本会话 `device_rejected_commands=0`。
  记录位于 `logs/simplified_lower_watchdog_20260910_140946/`。
- 已完成 10 s 原生策略隔离 dry-run：将 bridge 的策略输入重映射到未发布话题，真实
  ESD-SLAVE 全程保持 DISABLED，设备发送/应用命令序号均为 0。推理侧得到 500 条
  `PolicyCommand`，频率 `50.001 Hz`，来源状态序号从 `3120564` 到 `3123059`，无重复、
  倒退或超前。单次 MNN 推理耗时平均 `0.449 ms`、p95 `0.739 ms`、p99 `1.257 ms`、
  最大 `3.914 ms`，树莓派算力满足 20 ms 策略周期。记录位于
  `logs/simplified_policy_dry_run_20260910_141252/`；CSV 普通消费者覆盖中间
  `LowerState` 不影响设备采样时间严格 `250 Hz` 的结论。
- dry-run 曾暴露出 RL 放行阻塞项：旧 P1～P8 配置在
  `ESD_Control_Ports.cpp` 中将每个端口 `dq_des` 限为 `±2 rad/s`，而当前训练/旧
  SocketCAN 合同将轮动作乘以 `10`，本次 P3/P6 稳态输出约为 `-18.6/-16.4 rad/s`。
  该问题已由2026-09-10下位机更新解决：P3/P6提升至 `±50 rad/s`，bridge恢复训练合同
  `±35 rad/s`，没有对策略动作做重新缩放。
- bridge/deploy_tools/inference 最新定向回归均为 0 失败；`colcon test-result` 分包
  汇总分别为 22/22、70/70、1/1。legacy `hipnuc_imu` 目录中已有的 lint 失败不属于
  ESD 默认启动或本轮修改。
- P7/P8 翼扫频新增独立现场启动器
  `scripts/sweeps/sweep_motor78/run_sweep_motor78.sh`：自动 source 工作区、启动或复用
  ESD-Link bridge 与 Xbox `/joy`，并将 CSV/启动日志统一落到 `sweep_motor78/logs/`。
  `run_sweep_test_flow.sh 78` 已成为统一入口；直接执行裸
  `ros2 run deploy_tools esd_wing_sweep` 仍要求操作者预先启动 bridge 和 joy。
- 首次运行专用入口在使能前暴露 P7/P8 端口迭代错误：有序的翼限位区间元组被误当作
  端口号，触发 `tuple - int` 后客户端退出；bridge 和 joy 均由启动器正常清理，未进入
  SWEEP、未发送使能。实现已明确拆分 `WING_PORTS=(7,8)` 与
  `WING_POSITION_LIMITS`，新增回归测试保证 P1～P6 原位保持且仅 P7/P8 收至扫频中心。
- 训练侧已确认可直接消费当前 ESD CSV，不再做 legacy `motor_*_raw` 字段转换。已有
  1/4 `2/0.1`、2/5 `8/0.8`、3/6 `0/0.1` 三组保持原样；新增
  `run_remaining_gain_sweeps.sh`，仅补 1/4 `4/0.2`、2/5 `4/0.2`、3/6
  `0/0.05` 与 `0/0.2` 四组。单次覆盖参数不修改基础 YAML 和 CSV schema，文件名自动
  写入 Kp/Kd；2/5 激励保持当前已验证的 `±0.15 rad`。参数覆盖和轮子 D-only 约束
  已加入回归测试，`deploy_tools` 全包 `78/78` 通过；本阶段尚未自动触发实机运动。
- 增益覆盖命令不再要求人工填写 `--output-file`：采集器按实际电机组、Kp 和 Kd 自动
  生成 `motor{pair}_{timestamp}_kp..._kd....csv`，避免文件标签与实际控制增益不一致；
  显式输出路径仍具有最高优先级。
- 1/4、2/5、3/6、P7/P8 实机扫频均已正常完成；七组固定增益辨识数据已经按当前 ESD
  CSV 格式采集完毕，训练侧确认无需格式转换。
- 修正强制零策略的关键语义：`force_zero_policy_commands` 现在只覆盖六维命令值，
  不再关闭 POLICY 发布权限。进入 POLICY 后会以 50 Hz 发布原生零动作并真实经过
  bridge、下位机命令序号和应用确认；STANDBY 仍不发布原生策略命令。新增
  `./scripts/rl/run_zero_policy_test.sh` 和 supervisor `--force-zero-policy` 入口。
- 本轮 `inference` 1/1、`deploy_tools` 80/80 自动测试通过，启动脚本语法、launch 参数
  和 Git diff 检查通过；DISABLED 预检、STANDBY↔零策略交接及短时真实 MNN POLICY
  均已完成实机验收。
- A4 前真实设备预检已通过：10 秒记录 2500 条 LowerState，设备采样严格 250 Hz，
  指纹 `0x4ca27910`，P1～P8/IMU 有效，链路/命令/故障计数零增量且没有命令。随后
  supervisor 以 `--no-rgb --preflight-only` 完成 bridge、模型 hash 和 `/joy` 检查，
  全程 DISABLED。新增 `--no-rgb` 维护入口，允许无需 sudo 地与只持有待机 RGB 的
  生命周期服务并存；该模式不改变串口和控制权合同。
- A4 强制零策略已完成真实设备主路径验证：`logs/zero_policy_clean_20260910_2230/`
  记录 13729 条原生 `PolicyCommand`，六维动作全部严格为零，来源状态序号唯一且递增；
  bridge 拒绝、COBS、CRC、源序号缺口、设备非法帧和设备拒绝均为零。命令到应用确认
  p99 `5.14 ms`、最大 `7.27 ms`。首次 Ctrl-C 收尾暴露 rclpy 先关闭导致一次下位机
  watchdog 接管；正常退出现改为由 supervisor 独占信号处理并等待整体失能完成。
- 修正后以已使能 STANDBY 复测正常退出，`logs/shutdown_handoff_20260910_2242/`
  最后 211 条状态均为 DISABLED，fault、watchdog、拒绝、COBS、CRC 和序号缺口均为
  零增量；终端明确打印 lower DISABLED 确认且无 ROS 异常栈。A4 技术及现场验收均已
  通过，随后完成了 A5 短时真实 MNN POLICY。
- 现场已确认 A4 的 STANDBY/强制零策略无异常位移、抖动或异响，A4 完整签署通过。
  随后在真实 STANDBY 姿态完成隔离 MNN 预演：94 条非零输出全部有限且落在四路位置
  与两路 `±35 rad/s` 轮速合同内；策略原生话题已重映射，设备没有收到策略动作，链路、
  拒绝、watchdog 和 fault 零增量，最终 DISABLED。记录位于
  `logs/a5_isolated_preview_20260910_2255/`。
- 新增 `scripts/rl/run_policy_smoke_test.sh` 和 supervisor
  `--policy-smoke-seconds 3.0`：首次真实 MNN POLICY 到期自动先回 bridge STANDBY、再停
  策略发布，并自动记录运行 CSV。普通启动的 POLICY 时长不受影响；deploy_tools
  80/80 通过，A5 三秒真实策略窗口已完成。
- A5 三秒真实 MNN POLICY 已完成传感器侧验收：bridge 实际 POLICY 窗口约 3 秒并
  自动回 STANDBY，最终 lower DISABLED。305 条原生命令全部有限、来源序号唯一递增且
  未越界；P3/P6 曾达到裁剪上限 `+35 rad/s`，实际速度峰值约 `28.54/28.12 rad/s`。
  策略读取观测年龄 p99 `4.82 ms`，命令到应用约 `3.12 ms`，链路、拒绝、watchdog 和
  fault 全为零。记录位于 `logs/policy_smoke_20260910_230336/`；现场确认本次运动不构成
  异常，A5 秒级真实策略实机验收已签署通过。
- 23:27 和 23:28 两次下地回归均在进入 POLICY 后约一秒内报告
  `RUNTIME_DEVICE_LOST: lower controller fault`。第二次会话实际保持 250 Hz，COBS、
  CRC、源序号缺口、离线端口和持久 fault 均为零；bridge host reject 累计 6，device
  reject 为零，随后出现一次下位机 watchdog。推理日志捕获 P4 策略目标 `+0.874 rad`，
  超过 bridge 上限 `+0.840 rad`；bridge 当前拒绝整个 P1～P8 帧，而旧 `motors_node`
  会逐关节裁剪并继续下发。`last_reject_code=9` 是此前设备诊断遗留值，本次设备拒绝
  计数为零，不能作为当前根因。台架 A5 仍有效，但修复这一兼容语义并复测前，撤销
  日常下地 POLICY 放行，手柄入口只允许到 STANDBY。
- bridge POLICY 位置越界现已恢复旧 `motors_node` 的逐关节裁剪语义：P1/P2/P4/P5
  的有限目标分别 clamp 到当前配置边界后继续聚合并下发完整 P1～P8 命令；P3/P6
  `±35 rad/s` 速度合同保持不变。新增 PTY 回归同时越界四路位置，验证裁剪后的下位机
  坐标、两路轮速、完整八路发送和 host reject 零增量；`esd_link_bridge` 18/18 通过并
  已编译到 install。等待下地短时 POLICY 复测后再恢复日常 POLICY 放行。
- 2026-09-11 修复 ESD-Link 迁移时引入的倒地自启语义回退：DIRECT_POLICY 不再经过
  3 秒 STANDBY 默认姿态斜坡。bridge 现在允许从 DISABLED 直接选择 POLICY，整体使能后
  不排队当前位置保持帧，等待第一条新鲜策略命令。PTY 回归确认使能后 100 ms 内执行器
  命令数为 0，首条命令为指定的非零 Policy 输出；普通 STANDBY/维护路径保持不变。

## Git 检查点

- `ffbfe26`：导入最新 Deploy，并完成 ESD-Link 第一阶段协议、bridge、只读实机验证及推理接线草稿。
- `e987d9f`：supervisor、构建入口和只读 Foxglove 默认切换为 ESD-Link，22 个状态机/ROS supervisor 测试通过。
- `d1225e3`：迁移 1/4、2/5、3/6 扫频和运行记录到原生 ESD-Link 状态/命令接口。
- `0af4888`：扩充协议黄金向量、流恢复和 mailbox 并发测试。
- `98d3ba8`：加固命令新鲜度、协议统计和 mailbox 并发安全。
- `5d4fce2`：新增 250 Hz ESD-Link 伪终端端到端集成验证。
- `e6ae32c`：固化策略/下位机坐标合同、整体失能确认和命令聚合安全。
- `bb84e0a`：完成 ESD-Link 维护扫频、置零身份记录、直连 CAN 互斥和快速断流保护。
- `da1581b`：从 ESD 运行记录器彻底删除旧 candump 捕获路径。
- `eefcfd4`：删除 supervisor 不可达的旧 CAN/IMU/翼定位状态及测试。
- `b07f787`：收紧配置指纹、STANDBY 固定保持、兼容服务枚举和原生链路诊断合同。
- `32c1bf2`：根据 DISABLED 下的左右翼手动辨向，锁定 P7=左翼、P8=右翼，
  完成 bridge 反馈/命令与推理观测的双向坐标修正及自动测试。
- `a5329f2`：按现场手动范围扩大 ESD-Link 翼目标限位，保持默认扫频和 RL 范围不变。
- `b78543c`：新增手柄确认、低增益、越位/超速监测和自动失能的首次当前位保持工具。
- `6747a70`：修正整体失能期间的误超时及 SET_ENABLE 旧状态确认竞态，并记录
  A1 两次“现场无位移、收尾待修正”结果；迁移路径 74/74 测试通过。
- `51f6f55`：现场确认最终保持无位移、无异常声响，正式签署 A1。
- `5fe8c58`：新增 A2 单端口正反向小动作工具、完整八路安全保持和自动测试。
- `cb4121b`：将 A2 正反动作改成各自有界的短时使能窗口，并记录 P1 零运动安全退出。
- `6cc1e47`：新增无需手柄、全程 DISABLED 的 P1～P8 人工行程记录器。
- `386668e`：记录 180 秒被动人工行程与腿部限位待复核结论。
- `c368f30`：为被动行程记录器增加八路实时终端界面、原始编码坐标和自动测试。
- `76f45be`：固化现场确认硬限位、带机械裕量的软件限位、全路径预检和 bridge
  使能前拦截，并修正越界的 2/5 扫频配置。
- `56811dc`：归档 1/4、2/5 实机扫频、P3/P6 下位机实际速度边界、重力自然姿态
  接管和跨线程时间竞态修复。
- `3f95586`：删除 bridge 运行期 20/40 ms 上位机超时，由下位机 50 ms 看门狗
  统一负责 deadline；默认不绑定 CPU。
- `9e9ebf5`：原生推理每个新 `LowerState` 序号最多消费一次，不复用旧观测。
- `63081ec`：扫频/维护命令改为新状态驱动，删除重复旧来源的定时心跳。
- `95a756c`：supervisor 改为原生 `LowerState` 单入口，删除兼容话题年龄/频率硬门限，
  并修正 STANDBY↔POLICY 交接顺序。
- `d2c851f`：记录五包编译和 87 项迁移测试全通过。
- `d4830d9`：记录精简后真实 ESD-SLAVE 的 15 s READ_ONLY/250 Hz 复测。
- `ad9f4ac`：为当前位保持工具增加明确现场授权的无手柄倒计时入口。
- `7ab5467`：维护工具订阅锁存 bridge FAULT，不增加时效门限即可在下位机故障后立即停止。
- `a8a9d66`：精简后的状态驱动当前位保持通过台架传感器验收，全程不要求 P2/P5 回位。
- `f5956ac`：暂停上位机 200 ms 的实机注入确认冻结下位机看门狗是唯一运行期 deadline。
- `4c158c3`：清理原生路径中遗留的 legacy IMU/电机及“上位机超时”诊断文案。
- `d48fd6c`：记录策略隔离 dry-run，并将 bridge P3/P6 速度边界对齐冻结下位机
  `±2 rad/s`，新增伪终端越界帧不下发验证。
- `4738156`：接入2026-09-10更新下位机合同，基线指纹改为 `0x4ca27910`，bridge
  P3/P6恢复训练合同 `±35 rad/s`，并验证实际dry-run轮速可通过、`35.01 rad/s`仍被拒绝。
- `e67e33e`：清理 STANDBY↔POLICY 正常交接产生的伪拒绝统计，并消除 supervisor
  Ctrl-C 时的 ROS executor 异常栈；bridge 22/22、deploy_tools 79/79 通过。
- `9e184cf`：supervisor 独占 SIGINT/SIGTERM，在关闭 ROS 之前完成策略停发和整体
  失能确认；已使能 STANDBY 实机退出复测零 watchdog、最终 DISABLED。
- `09e88ea`：新增带自动运行记录的 A5 三秒真实策略入口，到期严格按 bridge
  STANDBY 接管、策略停发的顺序退出；deploy_tools 80/80 通过。
- 后续每完成一个可编译、可测试的阶段再提交；不提交 `build/`、`install/`、`log/`、`logs/` 和运行生成数据。
