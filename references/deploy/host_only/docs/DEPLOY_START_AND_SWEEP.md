# Walking Eagle Deploy：start_robot 启动与扫频流程

文档版本：v1.0  
审计日期：2026-08-03  
源码基准：桌面 deploy 完整版  
代码仓库：https://git.esdyn.cn/walking-eagle/sar_unilab.git  
交付分支：Walking_Eagle-Deploy_final  
最终 Commit Hash：发布后以 git rev-parse HEAD 和交接记录为准  
适用机器人：Walking Eagle WE11  
当前状态：源码、配置、模型和三组扫频脚本闭包完整；本次完成静态审计，未重新执行真机运动测试  

> 本文是独立的 Deploy 模块文档，只描述 start_robot 实机启动链路和 1～6 号腿轮电机扫频采集。训练、PACE/Kp/Kd 拟合和 rl_sar 回放不在本文范围内。

## 1. 交接范围

本模块包含：

- ROS 2 Humble 部署工作区；
- 1～6 号腿轮电机控制节点；
- 7/8 号翼电机控制节点；
- IMU、Xbox、MNN 推理节点；
- CAN 配置、状态检查和运行记录工具；
- start_robot.sh 一键启动状态机；
- 1/4、2/5、3/6 三组直接 CAN 扫频；
- 当前部署配置、lab_policy.mnn 和整理后的 2026-07-30 扫频数据。

不包含：

- UniLab/IsaacLab 训练代码；
- PACE、Kp/Kd 参数拟合程序；
- rl_sar MuJoCo 手柄回放；
- 机器人机械、电气设计源文件；
- build/install/log/logs/plots 等本机生成物。

## 2. 目录和关键文件

~~~text
Walking_Eagle-Deploy/
├── README_CN.md
├── env_init.sh
├── setup_dual_can.sh
├── start_robot.sh
├── docs/
│   ├── LAB_DEPLOY_START.md
│   ├── CAN配置说明.md
│   ├── PD_AND_WING_FREQUENCY.md
│   └── troubleshooting/
├── config/system/
├── src/
│   ├── hipnuc_imu/
│   ├── hipnuc_lib_package/
│   ├── motors/
│   ├── inference/
│   ├── xbox/
│   ├── remote_controller/
│   └── deploy_tools/
├── scripts/sweeps/
│   ├── sweep_motor14/
│   ├── sweep_motor25/
│   └── sweep_motor36/
├── experiments/wing/
└── sweep_results_20260730/
~~~

| 文件 | 作用 | 是否必须 |
| --- | --- | --- |
| env_init.sh | 安装 ROS/系统依赖、准备 MNN runtime、构建部署包 | 是 |
| setup_dual_can.sh | 建立并配置 can0 | 是 |
| start_robot.sh | 整条实机启动、安全确认、监控和退出状态机 | 是 |
| src/motors/config/motors.yaml | 1～6 号电机、PD、零位变换、限位、频率 | 是 |
| src/motors/config/wing_motors.yaml | 7/8 号翼电机回零、扑翼和安全边界 | 是 |
| src/inference/config/lab_inference.yaml | 模型、策略周期、IMU 轴向和手柄门控 | 是 |
| src/inference/models/lab_policy.mnn | 当前实机 MNN 策略 | 是 |
| src/hipnuc_imu/config/hipnuc_config.yaml | 串口、波特率、IMU 频率和滤波 | 是 |
| scripts/sweeps/sweep_motor14/ | 大腿 paired sweep | 是 |
| scripts/sweeps/sweep_motor25/ | 小腿 paired sweep | 是 |
| scripts/sweeps/sweep_motor36/ | 轮子 paired sweep | 是 |
| sweep_results_20260730/ | 七组正式扫频结果与 SHA256 | 建议保留 |

## 3. 环境依赖与安装

### 3.1 系统要求

| 项目 | 要求 |
| --- | --- |
| 操作系统 | Ubuntu 22.04 |
| CPU | x86_64 或 aarch64 |
| ROS | ROS 2 Humble |
| Python | 系统 /usr/bin/python3；构建时不要使用 Conda Python |
| RMW | Fast DDS / rmw_fastrtps_cpp |
| CAN | SocketCAN can0，默认 1 Mbps |
| 推理 | MNN runtime |
| 工具 | colcon、rosdep、screen、can-utils、iproute2、ccache |
| 权限 | 安装依赖和配置 CAN 时需要 sudo |

### 3.2 一键安装和构建

在仓库根目录执行：

~~~bash
./env_init.sh
~~~

脚本会：

1. 安装 ROS 2 和系统依赖；
2. 清理 Conda 对 ROS 构建 Python 的干扰；
3. 根据 x86_64/aarch64 选择 MNN runtime；
4. runtime 缺失时下载并构建 MNN；
5. 使用 rosdep 安装包依赖；
6. 构建 hipnuc_imu、motors、inference、xbox、deploy_tools；
7. 检查节点、launch、配置和 lab_policy.mnn 是否进入 install/。

安装后只验证软件闭包，不会自动启动电机。

## 4. 接口、频率和坐标合同

### 4.1 ROS 输入

| 话题 | 类型 | 频率 | 来源 |
| --- | --- | ---: | --- |
| /IMU_data | sensor_msgs/Imu | 200 Hz | hipnuc_imu |
| /joy | sensor_msgs/Joy | 手柄节点频率 | joy_node |
| /cmd_vel | geometry_msgs/Twist | 50 Hz | xbox_vel_publisher |
| /policy/joint_states | sensor_msgs/JointState | 200 Hz | motors_node |
| /policy/wing_angles | sensor_msgs/JointState | 200 Hz | wing_motor_node |

### 4.2 ROS/CAN 输出

| 输出 | 类型 | 频率 | 目标 |
| --- | --- | ---: | --- |
| /policy/commands | std_msgs/Float32MultiArray，6 维 | 50 Hz | motors_node |
| 1～6 号 MIT 控制帧 | SocketCAN | 200 Hz | 腿轮电机 |
| 7/8 号控制帧 | SocketCAN | 200 Hz | 翼电机 |
| 扫频 CSV | 文件 | 200 Hz | scripts/sweeps/*/logs |

### 4.3 坐标和关节顺序

1～6 号策略顺序固定为：

~~~text
[左大腿, 左小腿, 左轮, 右大腿, 右小腿, 右轮]
~~~

正常部署默认角：

~~~text
[-0.92020, +0.98338, 0, -0.92020, +0.98338, 0] rad
~~~

翻转配置：

~~~text
[true, false, true, false, true, false]
~~~

IMU 到 base_link 的固定符号映射为：

~~~text
[x, y, z] -> [-x, -y, +z]
~~~

该映射等价于绕 base_link z 轴旋转 180°，必须与训练侧坐标合同一致。

## 5. 当前关键配置

### 5.1 1～6 号腿轮电机

| 电机 | 关节 | 类型 | Kp | Kd | 模式 | 相对位置限制 |
| --- | --- | --- | ---: | ---: | --- | --- |
| 1 | 左大腿 | RS05 | 2.0 | 0.1 | 位置 | [-1.23, 0.77] rad |
| 2 | 左小腿 | RS00 | 8.0 | 0.8 | 位置 | [-0.98338, 1.51662] rad |
| 3 | 左轮 | RS05 | 0 | 0.25 | 速度 | [-6.28, 6.28] rad |
| 4 | 右大腿 | RS05 | 2.0 | 0.1 | 位置 | [-1.23, 0.77] rad |
| 5 | 右小腿 | RS00 | 8.0 | 0.8 | 位置 | [-0.98338, 1.51662] rad |
| 6 | 右轮 | RS05 | 0 | 0.25 | 速度 | [-6.28, 6.28] rad |

其他关键值：

- control_frequency：200 Hz；
- publish_rate_policy_joint_states：200 Hz；
- policy_frequency：50 Hz；
- enable_interpolation：false；
- offline_threshold：600，约 3 s @ 200 Hz；
- auto_zero_on_start：false；
- policy_motor_warmup_sec：0。

### 5.2 7/8 号翼电机

- CAN：can0；
- publish_frequency：200 Hz；
- 启动目标：motor7=`+90°`、motor8=`-90°`，不写硬件零点；
- 启动定位 Kp/Kd：10/1；
- 启动定位速度：0.35 rad/s；
- 软件目标轨迹走完即放行，不验证实际到位误差或速度；
- 目标完成后保持启动姿态；
- 右摇杆纵向连续控制翼旋转速度和方向，回中后 PD 保持当前翼角；
- 最大遥控速度：`pi/4 rad/s`；
- 遥控 PD：Kp/Kd = 20/1；
- 两侧机械安全边界均为 [-60°, +60°]；
- 运行反馈超时 0.5 s；
- shutdown 时禁用电机。

### 5.3 MNN 推理

- 模型：lab_policy.mnn；
- 默认 height command：0.24；
- 物理 dt：0.0025 s；
- decimation：8；
- 策略周期：0.02 s，即 50 Hz；
- wing angle timeout：0.1 s；
- 翼角超时后回 standby 并发布零动作；
- start_robot 启动时打开手柄门控，默认从 standby 开始。

## 6. start_robot 启动流程

### 6.1 运行前安全检查

1. 机器人架空或可靠固定，腿轮和翅膀运动范围内无人、无障碍物。
2. 急停可触达，电源电压和限流正确。
3. USB-CAN、终端电阻和 can0 接线正确。
4. IMU 和手柄连接正常。
5. 不存在另一个 motors_node、扫频程序或调试程序占用 can0。
6. 摇杆回中，操作者明确 A/B/X/Y 按键功能。
7. motors.yaml 中的 ID、类型、方向、默认角、限位和 PD 已复核。

### 6.2 启动命令

在仓库根目录执行：

~~~bash
./start_robot.sh
~~~

start_robot.sh 必须保持前台。退出时在该终端按 Ctrl+C，让脚本统一清理。

### 6.3 启动状态机

~~~text
Fast DDS SHM 与 ROS 图检查
  -> 检测并人工确认清理残留控制进程
  -> setup_dual_can.sh 配置 can0
  -> 0x02 探测 1～8 号电机
  -> colcon 构建五个部署包
  -> 检查节点、launch、YAML、MNN 产物
  -> 可选启动 Foxglove Bridge
  -> 启动 IMU，检查 /IMU_data 和实际频率
  -> 启动 Xbox，建立 /joy 和 /cmd_vel
  -> 终端输入 y，允许启动 1～6 号 motors_node
  -> 检查 /policy/joint_states，人工核对后按 Enter
  -> 自动启动 7/8 号并将软件目标推进到 +90°/-90°
  -> 启动 MNN 推理并验证零 /policy/commands
  -> 用户按 X 选择普通/倒地自启模式，按 Y 执行（默认 def-pos standby）
  -> 按 X 在 standby 和 policy 之间切换
  -> 持续监控 motors_node 与 wing_motor_node
~~~

### 6.4 阶段说明

#### CAN 与构建

setup_dual_can.sh 默认配置：

~~~text
interface = can0
bitrate = 1000000
txqueuelen = 1000
restart-ms = 100
~~~

原生 gs_usb 不存在时，脚本可从 /dev/ttyACM0 通过 slcand 建立 can0。start_robot 随后用 0x02 逐一探测 1～8 号电机；任一映射失败都不会启动 motors_node。

脚本构建 hipnuc_imu、motors、inference、xbox、deploy_tools，并检查必要安装产物。空构建或 MNN 模型缺失会直接失败。

#### IMU 和手柄

IMU 启动后必须在 15 s 内收到 /IMU_data。脚本短时间统计实际频率，由操作者确认后继续。随后启动 Xbox 手柄链路。

#### 1～6 号腿轮电机

终端输入 y 后启动 motors_node。脚本等待 /policy/joint_states，打印 1～6 号位置、速度和力矩，人工确认后按 Enter。

正常启动：

- 不调用 /set_zeros；
- 不写 1～6 号硬件零点；
- 使用 motors.yaml 中的 joint_default_angle 做策略坐标转换。

#### 7/8 号翼电机

翼电机无 A 确认门。`wing_motor_node` 自动将软件目标推进到
motor7=`+90°`/motor8=`-90°`，然后进入 `POSITION_HOLD`。该过程不写硬件零点，
也不判定实际到位误差或速度；仅保留反馈和安全边界检查。

#### standby 和 policy

翼启动目标完成后，推理节点以 standby 启动并发布零 policy command。
自动检查通过后默认选择普通启动：等待时 1~6 号失能、右摇杆可控机翼，直接按 Y 进入 standby；按 Y 前可用 X 切换到
倒地自启（绿色呼吸），再按 Y 跳过 def-pos 斜坡并直接进入 policy。进入运行阶段后，
X 仍在 standby 与 policy 间切换；任一运行状态按 Y 都会用 3 秒线性卸力 1~6 号，
归零后硬件失能并返回等待。

| 按键 | 作用 |
| --- | --- |
| 右摇杆纵向 | 控制 7/8 号翼电机镜像旋转，符号 `[+1,-1]`，最大 `pi/4 rad/s`；回中保持 |
| Y | 等待阶段执行所选启动模式；Standby/Policy 中三秒卸力、失能 1~6 号并回等待 |
| X | 等待阶段选择普通/倒地自启；运行阶段在 standby 和 policy 之间切换 |
| 方向键上 + A | 运行或故障时长按约 0.25 秒，触发统一安全退出 |
| 方向键上 + X | 程序停止时长按约 0.25 秒并松开，重新执行完整启动和自检 |

### 6.5 日志、监控与退出

motors_node 保持当前终端前台输出；其他组件使用 screen 会话：

~~~bash
screen -r imu_session
screen -r joy_session
screen -r wing_motor_session
screen -r lab_inference_session
screen -r foxglove_bridge_session
~~~

日志位于 logs/，生命周期日志名为 start_robot_lifecycle_*.log。

start_robot 持续监控 motors_node 和 wing_motor_node。任一关键电机节点退出时，会停止推理和其余链路。正常退出方法：

~~~text
在 start_robot.sh 前台终端按 Ctrl+C
~~~

不要关闭终端或使用 kill -9 代替统一清理。

## 7. 1～6 号电机扫频

### 7.1 公共安全条件

1. start_robot.sh、motors_node 和其他 CAN 控制节点必须已完全退出。
2. 机器人必须架空或固定，急停可用。
3. 每次只运行一个 paired sweep。
4. 扫频只访问 1～6 号，不访问 7/8 号。
5. 扫频不调用 /set_zeros，不写机械零点。
6. A 键每次按下后必须释放，才能进入下一确认门。
7. 正常完成和异常退出都必须尝试发送 0x04 禁用 1～6 号。

### 7.2 标准入口

~~~bash
./scripts/sweeps/sweep_motor14/run_sweep_motor14.sh
./scripts/sweeps/sweep_motor25/run_sweep_motor25.sh
./scripts/sweeps/sweep_motor36/run_sweep_motor36.sh
~~~

默认手柄设备是 /dev/input/js0。需要切换时：

~~~bash
SWEEP_JOY_DEV=/dev/input/js1 ./scripts/sweeps/sweep_motor14/run_sweep_motor14.sh
~~~

launcher 会复用有效的 /joy；没有发布者时启动 Xbox。若存在发布者但 3 s 内收不到数据，程序会停止，避免重复启动两个手柄节点。

### 7.3 公共三次 A 状态机

~~~text
打印 sweep pose、零点偏移、raw target、方向和 1～6 初始状态
  -> 0x02 检查 1～6
  -> 0x03 使能并切入 MIT 模式
  -> 第一次 A：低增益回/保持 sweep 专用姿态
  -> 第二次 A：人工确认姿态，进入非扫频关节 hold
  -> 第三次 A：开始正式 chirp
  -> 保存 1～6 号完整目标与反馈 CSV
  -> 回 sweep 姿态；轮子先回 0 rad/s
  -> 0x04 禁用 1～6
~~~

### 7.4 扫频专用姿态

原始 sweep base：

~~~text
[-0.42, +0.50, 0, -0.42, +0.50, 0] rad
~~~

新零点参考偏移：

~~~text
[-0.10020, -0.08662, 0, -0.10020, -0.08662, 0] rad
~~~

最终 policy sweep center：

~~~text
[-0.52020, +0.41338, 0, -0.52020, +0.41338, 0] rad
~~~

最终电机 raw center：

~~~text
[+0.52020, +0.41338, 0, -0.52020, -0.41338, 0] rad
~~~

该姿态只用于扫频，不是正常部署 def-pos。

### 7.5 当前 launcher 默认参数

| 电机对 | 模式 | 激励 | Kp/Kd | 非扫频电机 | hold Kp/Kd |
| --- | --- | --- | --- | --- | --- |
| 1/4 大腿 | MIT 位置 | ±0.20 rad | 2.0/0.1 | 2/3/5/6 | 20/1 |
| 2/5 小腿 | MIT 位置 | ±0.25 rad | 8.0/0.8 | 1/3/4/6 | 40/2 |
| 3/6 轮子 | MIT 速度 D-only | ±5 rad/s | 0/0.1 | 1/2/4/5 | 1/0.1 |

三组公共参数：

| 参数 | 值 |
| --- | --- |
| CAN | can0 |
| 控制频率 | 200 Hz |
| 记录频率 | 200 Hz |
| chirp | 线性 0.1 → 5.0 Hz |
| 正式时长 | 40 s |
| 回位 Kp/Kd | 1.0/0.1 |
| 回位速度 | 0.10 rad/s |
| torque field | 0 N·m |
| 退出 | 恢复 hold/零速度并发送 0x04 |

轮子扫频关系：

~~~text
Kp = 0
torque = Kd * (target_velocity - measured_velocity)
target_velocity in [-5, +5] rad/s
~~~

motor3 与 motor6 原始速度方向互为镜像。退出时先发送 0 rad/s，再禁用。

### 7.6 CSV 字段和输出

默认输出目录：

~~~text
scripts/sweeps/sweep_motor14/logs/
scripts/sweeps/sweep_motor25/logs/
scripts/sweeps/sweep_motor36/logs/
~~~

每行首先记录 timestamp，随后对电机 1～6 依次记录：

~~~text
target_raw
target_vel_raw
pos_raw
vel_raw
torque_raw
temp_c
~~~

正式 40 s、200 Hz 数据约 8000 行。原始 CSV 不应覆盖修改；分析、切分和坐标转换应写到新目录，并保留 SHA256。

### 7.7 已归档正式数据

sweep_results_20260730/ 保存七组数据：

| 电机对 | 参数 |
| --- | --- |
| 1/4 | Kp/Kd=4/0.2、2/0.1 |
| 2/5 | Kp/Kd=4/0.2、8/0.8 |
| 3/6 | Kp=0，Kd=0.05、0.1、0.2 |

注意：上表是已采集数据集合；第 7.5 节是当前 launcher 的默认 YAML。二者不能混为同一配置。

## 8. 最小验证

### 8.1 无硬件静态检查

~~~bash
bash -n env_init.sh
bash -n setup_dual_can.sh
bash -n start_robot.sh
bash -n scripts/sweeps/sweep_motor14/run_sweep_motor14.sh
bash -n scripts/sweeps/sweep_motor25/run_sweep_motor25.sh
bash -n scripts/sweeps/sweep_motor36/run_sweep_motor36.sh
~~~

Python 静态编译可把缓存写到 /tmp：

~~~bash
PYTHONPYCACHEPREFIX=/tmp/deploy_pycache python3 -m py_compile \
  scripts/sweeps/sweep_motor14/sweep_motor14.py
~~~

预期：全部退出码为 0。

### 8.2 构建检查

~~~bash
./env_init.sh
~~~

预期：

- 五个部署包构建成功；
- install/ 中存在 IMU、电机、翼电机、推理、Xbox 和 CAN 探测节点；
- lab_inference.yaml、wing_motors.yaml 和 lab_policy.mnn 已安装；
- 不连接或使能电机。

### 8.3 台架验收顺序

1. 只配置 can0，使用 candump 和状态请求确认 1～8 TX/RX；
2. 核对电机 ID、类型、位置符号、反馈类型和温度；
3. 启动 IMU，核对 200 Hz、静止偏置和 XYZ 轴向；
4. 启动 start_robot，只进入 standby；
5. 验证 A/A、Y、X 门控和 Ctrl+C 清理；
6. 分别对 1/4、2/5、3/6 做低风险短时检查；
7. 再执行正式参数 sweep；
8. 检查退出后的 0x04、轮子 0 rad/s 和残留 CAN 流量。

## 9. 安全限制和禁止操作

- 未确认 joint_position_limits、PD 和 motor_types 前禁止进入 policy。
- 未架空/固定机器人时禁止执行 paired sweep。
- 禁止同时运行 start_robot 和 sweep。
- 禁止用 /set_zeros 掩盖方向、默认角或坐标错误。
- 禁止通过单纯增大 offline_threshold 掩盖 CAN 丢包。
- 禁止在 IMU 无数据或轴向未核对时进入 policy。
- 禁止只凭 /policy/wing_angles 存在就认定翼电机机械回零正确。
- 出现振动、越限、温升、offline、tx buffer 满或异常帧时立即急停并保存日志。

## 10. 常见故障

| 现象 | 优先检查 |
| --- | --- |
| 0x02 探测失败 | 供电、终端电阻、USB-CAN、can0 状态、电机 ID/类型 |
| tx buffer 满 | CAN 负载、重复查询、控制节点冲突、TX/RX 对称性 |
| /policy/joint_states 无数据 | motors_node 日志、合法反馈帧、QoS、offline 锁存 |
| /IMU_data 无数据 | 串口、波特率、权限、节点日志 |
| 位置符号错误 | flipped_motors、joint_default_angle、机械安装方向 |
| 翼电机停机 | 反馈超时、±60° 边界、编码器方向、CAN 反馈 |
| sweep 异常退出 | finally 清理、0 rad/s、0x04、残留 Xbox/CAN 进程 |

排查 CAN 时应同时使用：

- candump 查看原始帧；
- ip -details -statistics link show can0 查看错误和队列；
- CAN 工具统计负载；
- 示波器检查 CAN_H/CAN_L 波形、电平和终端；
- 对比 TX 与合法 RX，确认半双工链路和反馈完整性。

## 11. 日志和交付记录

每次正式部署至少记录：

- 仓库、分支、Commit Hash；
- motors.yaml、wing_motors.yaml、lab_inference.yaml、lab_policy.mnn SHA256；
- Ubuntu、内核、架构、ROS 版本；
- 电机、IMU、主控、USB-CAN、电池/电源版本；
- 1～8 号启动探测结果；
- IMU 和关节话题实测频率；
- standby/policy 和翼电机确认结果；
- 生命周期日志、节点日志和 CAN 统计；
- 正常退出后的安全状态。

每次正式 sweep 至少记录：

- 电机对、YAML SHA256、Kp/Kd、幅值、频率和时长；
- 供电、工装、操作者和环境；
- 原始 CSV、日志和 SHA256；
- 是否正常回 hold/0 rad/s 并发送 0x04；
- 异常现象和处理。

## 12. 当前验收状态

| 检查项 | 状态 | 说明 |
| --- | --- | --- |
| 核心源代码 | 通过 | motors/inference/IMU/xbox/deploy_tools 齐全 |
| 配置和 launch | 通过 | 部署和三组 sweep 配置齐全 |
| 当前 MNN 模型 | 通过 | lab_policy.mnn 存在 |
| 三组 sweep launcher | 通过 | 1/4、2/5、3/6 齐全 |
| 示例数据 | 通过 | 2026-07-30 七组数据和 SHA256 |
| 根启动脚本语法 | 通过 | 本次静态检查记录写入发布交接 |
| 完整构建 | 待在干净分支验证 | 不复用桌面 build/install 作为证明 |
| 真机回归 | 本次未执行 | 发布任务不运行电机 |
| Git 版本追溯 | 发布后完成 | 以 Walking_Eagle-Deploy_final 远端 HEAD 为准 |

## 13. 发布内容边界

最终 Git 分支保留：

- 全部核心源代码、配置、launch、脚本和文档；
- 当前 MNN/ONNX 模型；
- 双架构 MNN runtime 和必要 converter；
- 整理后的 sweep_results_20260730。

不提交：

- 桌面原 .git；
- build/install/log/logs/plots/.ccache-tmp；
- __pycache__、pyc；
- scripts/sweeps/*/logs 临时采集目录；
- 本机 .planning/.learnings；
- 可由 env_init.sh 重新获取的 MNN-master 和未被当前 MNN 部署链路使用的 ONNX Runtime 上游副本。

## 14. 从零运行

~~~bash
git clone git@git.esdyn.cn:walking-eagle/sar_unilab.git
cd sar_unilab
git switch Walking_Eagle-Deploy_final
./env_init.sh
./start_robot.sh
~~~

实机启动前仍必须完成第 6.1 节安全检查。环境安装成功不代表可以跳过人工确认门。

## 15. 交接签署

- 交接人：
- 接收人：
- 交接日期：
- 仓库：https://git.esdyn.cn/walking-eagle/sar_unilab.git
- 分支：Walking_Eagle-Deploy_final
- 最终 Commit Hash：由发布结果填写
- 台架验证结论：
- 真机验证结论：
- 备注：
