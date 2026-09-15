# Walking Eagle WE11 ROS2 Deploy

[![ROS2](https://img.shields.io/badge/ROS2-Humble-silver)](https://docs.ros.org/en/humble/index.html)
![C++](https://img.shields.io/badge/C++-17-blue)
[![Linux platform](https://img.shields.io/badge/platform-linux--x86_64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Linux platform](https://img.shields.io/badge/platform-linux--aarch64-orange.svg)](https://releases.ubuntu.com/22.04/)

## 概述
本仓库参考自roboParty的Atom01部署框架修改与优化，感谢开源。

本仓库提供基于 ROS 2 Humble 的 Walking Eagle WE11 实机部署框架。日常运行由
单一 supervisor 统一管理自动自检、RGB 提示、手柄确认、ROS 子进程、故障锁存
和安全失能。

**维护者**: 马昕阳

**联系方式**: 15929257653  
 <mr.mytheo@sjtu.edu.cn>

**Walking Eagle WE11 算法交付负责人：汪成浩**

**正式分支：`Walking_Eagle-Deploy_final`**

本次算法交接文档：

- [`docs/handover.md`](docs/handover.md)
- [`docs/interface.md`](docs/interface.md)
- [`docs/test_record.md`](docs/test_record.md)
- [`CHANGELOG.md`](CHANGELOG.md)

**主要特性:**

- `易于上手` 提供全部细节代码，便于学习并允许修改代码。
- `隔离性` 不同功能由不同包实现，支持加入自定义功能包。
- `长期支持` 本仓库将随着训练仓库代码的更新而更新，并将长期支持。

## 📚 快速参考（已部署用户）

如果您已经完成部署，可以使用以下快速命令：

### 常用命令

```bash
# 维护阶段：源码变化后，启动前编译一次
cd /home/esd/Pheonix/Deploy
./start_robot.sh --build

# 手动配置当前单路 can0（仅维护/调试；脚本名为历史兼容名称）
cd /home/esd/Pheonix/Deploy
bash setup_dual_can.sh

# 日常启动：无 sudo、无终端输入、无运行时编译
cd /home/esd/Pheonix/Deploy
./start_robot.sh

# 查看 CAN 消息（可选，用于调试）
candump can0

```

当前实机启动流程请参考：`docs/LAB_DEPLOY_START.md`，完整状态机和灯光定义见
`docs/WE11_DAILY_STATE_MACHINE.md`；常用脚本清单请参考：
`docs/troubleshooting/USEFUL_SCRIPTS.txt`。整理后的目录职责见
`docs/WORKSPACE_LAYOUT.md`。

### WE11 日常启动状态机

| 状态 | RGB | 自动动作或人工操作 |
|---|---|---|
| 树莓派已开机、手柄未连接 | 橙色呼吸 | 连接或重连 `/dev/input/js0` |
| 树莓派已开机、手柄可用 | 绿色呼吸 | 长按方向键上 + `X` 并松开启动程序 |
| 启动交接中 | 白色呼吸 | 上+`X` 已接受；生命周期启动器持续显示，直到 supervisor 接管 RGB |
| 开机 | 白色呼吸 | supervisor 建立 ROS 图 |
| CAN 与电机自检 | 蓝色呼吸 | 检查 `can0` 为 UP，读取 1–8 号反馈；不使能电机 |
| IMU 自检 | 紫色呼吸 | 一个 1 秒窗口严格大于 100 Hz，最大断流不超过 200 ms |
| 设备自检 | 青色呼吸 | 检查手柄和 1–6 号只读反馈；日常不校验模型和关节初始位置 |
| 翼启动定位 | 黄色闪烁 | 自动推进 motor7=`+90°`/motor8=`-90°` 软件目标，无 A 确认 |
| 等待启动（普通） | 青色呼吸 | 1–6 号失能、右摇杆可控机翼；按 `X` 可切换倒地自启，扶稳后按 `Y` 执行普通启动 |
| 等待启动（倒地自启） | 绿色呼吸 | 1–6 号失能、右摇杆可控机翼；按 `X` 切回，确认地面和周围安全后按 `Y` |
| Standby 进入中 | 白色闪烁 | 1–6 号用 3 秒位置/PD 斜坡进入默认姿态 |
| 倒地自启进入中 | 绿色闪烁 | 1–6 号只初始化/使能，不执行默认姿态斜坡；随后直接交给 Policy |
| Standby | 白色常亮 | 策略输出为零；`X` 进入 Policy，`Y` 三秒卸力后失能腿轮并回等待 |
| Policy | 绿色常亮 | 策略输出有效；`X` 退回 Standby，`Y` 三秒卸力后失能腿轮并回等待 |

腿部进入 Standby 时会先串行初始化 1–6 号电机（约 5.1 秒），再执行 3 秒斜坡；
supervisor 的转换响应期限为 15 秒。日志会分别报告服务不存在、服务响应超时和
电机节点明确拒绝，不再把三种情况统一显示为 `STANDBY_SERVICE_FAILED`。

等待启动阶段默认选择普通 Standby 启动；每按一次 `X` 在“普通启动/倒地自启”间
切换，RGB 青色/绿色呼吸显示当前选择，`Y` 执行所选模式。倒地自启跳过默认姿态
斜坡，但保留 1–6 号串行初始化、完整运行门控和推理历史重置；电机在收到明确的
Policy 模式公告前会忽略 standby 零动作。进入 Standby 后，`X` 仍是一键切换，
supervisor 会先执行一次机器门控：IMU、手柄、1–8 号反馈、翼角、关节输入和推理
输出必须新鲜、有限且有效；拒绝时保持 Standby。方向键上 + `A` 连续约 0.25 秒可在正常或故障
状态安全停止完整程序；停止后，方向键上 + `X` 连续约 0.25 秒并松开可重新启动。

模型 manifest/hash 校验只在发布验收模式 `./start_robot.sh --preflight-only` 中执行，
不进入日常启动路径。

日常启动只向 7/8 号下发编码器 `+90°/-90°` 软件目标，不会写硬件零点。
流程只等待软件目标轨迹走完，不以实际角度误差或速度作为放行条件。1–6 号
硬件设零仍使用独立的 `set_zeros_1_6_joy.sh`；逐台 1–8 号维护标定不属于当前
日常状态机。

### RGB 故障码

故障后循环显示：**红色 1 秒 → 分类色 1 秒 → 白色短闪子编号 → 熄灭 2 秒**。
分类色固定为：蓝=CAN/电机、紫=IMU、青=手柄/模型/设备、黄=翼、品红=ROS/推理、
橙=维护。完整故障码和原因同时写入日志。故障会锁存且不会自动重启；观察故障码后
可长按方向键上 + `A` 安全退出，修复接线，再长按方向键上 + `X` 重新执行完整
自检。RGB 初始化或运行时失效同样禁止继续使能。

### 开机自启边界

仓库提供 `config/system/we11-deploy.service` 和
`config/system/install-we11-host.sh`，服务以普通用户运行，不保存
或传递 sudo 密码。先在维护终端完成一次编译，再安装并启用：

```bash
./start_robot.sh --build
./config/system/install-we11-host.sh --enable
```

安装命令不会立即启动机器人，只会设置下次开机启动手柄生命周期服务。开机服务
自动加载 `xpad` 并持续等待 `/dev/input/js0`；未连接时 RGB 橙色呼吸，可用时绿色呼吸。
长按方向键上 + `X` 并松开后立即切为白色呼吸；生命周期启动器通过握手保持该提示，
直到 supervisor 已准备好接管 RGB，然后才释放 GPIO。随后
无参数执行 `start_robot.sh` 和完整硬件自检。CAN 或 IMU 未接好时也能显示对应
故障灯，不会阻止手柄启动器本身运行。自启环境通过
`WE11_AUTOSTART=1` 禁止 `--build`，因此开机期间绝不会编译。systemd 标准输入
固定为 `null`，启动流程不读取鼠标、键盘或终端；Y/X 模式操作只来自 Xbox 手柄。
日常统一使用上+X 启动；如需直接运行 `./start_robot.sh`，必须先停止
`we11-deploy.service`，避免两个进程抢占 RGB GPIO。

### 更换手柄后的映射与断触检测

更换手柄后，先在机器人控制链路停止时运行：

```bash
cd /home/esd/Pheonix/Deploy
source /opt/ros/humble/setup.bash
./check_gamepad.py
```

脚本会在隔离话题 `/gamepad_check/joy` 上启动底层手柄驱动，不发布
`/cmd_vel`，也不启动电机或推理节点。它依次检测左右摇杆四个方向、方向键
四个方向、A/B/X/Y、LB/RB/LT/RT；每项需要持续保持至少 0.5 秒，每项结束后
留出 1 秒松开和回正时间。汇总会显示实际 `axes[]`/`buttons[]` 映射、断触率、
断触段数和最长间断。非默认设备可使用 `./check_gamepad.py --dev /dev/input/js1`。
所有正式手柄入口都会在设备缺失时尝试加载 `xpad`，并在设备不可读时提前停止。
`start_robot.sh` 可用 `START_ROBOT_JOY_DEV=/dev/input/jsX` 指定设备，设零脚本可用
`ZERO_JOY_DEV=/dev/input/jsX`，扫频脚本可用 `SWEEP_JOY_DEV=/dev/input/jsX`。

正式运行或故障锁存时，同时按住手柄**方向键上 + A**约 0.25 秒，会触发
`start_robot.sh` 的统一安全退出流程，效果等同于在前台终端按 `Ctrl+C`：依次停止
所有电机控制进程，等待驱动完成正常析构，然后通过 `can0` 向 1–8 号电机重复
发送失能帧，最后清理推理、手柄和 IMU。组合键需要方向键上和 A 同时持续按住，
单独按 A 不会触发退出。RS 协议没有失能确认帧，急停和
切断动力电源仍是最终安全保障。systemd 手柄生命周期服务仍会留在后台；修复问题
后长按**方向键上 + X**约 0.25 秒并松开，即可重新启动整套程序和完整自检。

## ⚡ 一键快速部署

如果你已经克隆好了本仓库，可以先使用环境初始化脚本安装依赖并编译：

```bash
cd /home/esd/Pheonix/Deploy
./env_init.sh
```

脚本会自动完成：

- 安装依赖（编译工具、ROS2 构建工具、常用工具等）
- 编译 `hipnuc_imu`、`motors`、`inference`、`xbox` 等当前实机链路功能包
- 准备当前 CPU 架构所需的 MNN runtime
- 输出后续启动实机链路的命令

除此之外，仍需要你**手动完成**以下准备工作（下面“完整部署指南”都有详细说明）：

- 安装并配置 ROS2 Humble（确保 `ros2 --help` 可以正常运行）
- 在 `~/.bashrc` 中加入 `source /opt/ros/humble/setup.bash`（避免每次手动输入）
- （推荐）在 `/etc/security/limits.conf` 中为当前用户配置实时优先级 `rtprio` 和 `memlock`
- 确认 USB CAN 适配器已经暴露或可创建 `can0`
- 根据实际硬件修改 `src/motors/config/motors.yaml` 与 `src/hipnuc_imu/config/hipnuc_config.yaml` 中的参数
- 如遇编译或运行问题，可参考文末“常见问题排查”小节逐项检查

## 📦 完整部署指南

本指南将帮助您在一台全新的 Ubuntu 22.04 系统上完成项目的完整部署。

### 1. 系统要求

- **操作系统**: Ubuntu 22.04 (Jammy Jellyfish)
- **架构**: x86_64 或 aarch64
- **ROS版本**: ROS2 Humble Hawksbill
- **权限**: 一次性主机配置需要 sudo；日常 `start_robot.sh` 不使用 sudo

### 2. 系统依赖安装

#### 2.1 更新系统包管理器

```bash
sudo apt update && sudo apt upgrade -y
```

#### 2.2 安装编译工具和基础依赖

```bash
# 编译工具链
sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    nano

# C++ 开发库（项目依赖）
sudo apt install -y \
    ccache \
    libfmt-dev \
    libspdlog-dev \
    libeigen3-dev \
    pkg-config

# CAN 工具（用于 CAN 接口配置和调试）
sudo apt install -y \
    can-utils \
    iproute2

# Python 依赖（ROS2 需要）
sudo apt install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-can \
    python3-gpiozero

# 其他有用的工具
sudo apt install -y \
    htop \
    tmux \
    net-tools \
    usbutils
```

#### 2.3 安装 ROS2 Humble

按照 [ROS2 官方安装指南](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html) 进行安装：

```bash
#使用鱼香ros的一键安装
source <(wget -qO- http://fishros.com/install)
#选择humble版本进行安装，安装完整版
```

#### 2.4 配置环境变量

将以下内容添加到 `~/.bashrc`：

```bash
# ROS2 环境
source /opt/ros/humble/setup.bash
```

然后执行：

```bash
source ~/.bashrc
```

#### 2.5 配置实时优先级（可选但推荐）

为了获得更好的实时性能（特别是电机控制节点），需要配置实时优先级：

```bash
# 编辑 limits 配置文件
sudo nano /etc/security/limits.conf
```

在文件末尾添加以下内容（将 `your_username` 替换为您的实际用户名）：

```bash
# 允许用户设置实时优先级（用于电机控制）
your_username   -   rtprio   98
your_username   -   memlock  unlimited
```

保存后，**需要重新登录**才能使配置生效。

**验证实时优先级配置**：

```bash
# 重新登录后，运行以下命令验证
ulimit -r
# 应该输出 98

ulimit -l
# 应该输出 unlimited
```

### 3. 获取项目代码

```bash
# 创建工作目录（根据实际路径调整）
mkdir -p ~/project/deploy_cpp
cd ~/project/
# 克隆或复制项目到 deploy_cpp 目录
git clone https://gitee.com/mytheo-ma/deploy_cpp.git
# 初次部署需要配置git config，即在终端输入：
#git config --global user.name 'XXX' 
#git config --global user.email 'XXX@email.com'（邮箱与姓名仅作提示）
cd deploy_cpp
```

### 4. 编译项目

#### 4.1 编译基础包（不包含推理节点）

```bash
# 确保已 source ROS2 环境
######################!!!!!!IMPORTANT!!!!!#############################
####在初次编译节点之前，注意关闭终端的conda环境，直到使用ubuntu原生python为止####
######################!!!!!!IMPORTANT!!!!!#############################
source /opt/ros/humble/setup.bash

# 编译日常启动链路
colcon build --packages-select motors inference deploy_tools hipnuc_imu xbox
source install/setup.bash
```

#### 4.2 MNN 推理运行时与模型

当前日常节点是 `lab_inference_node`，运行模型为
`src/inference/models/lab_policy.mnn`。仓库已经按 CPU 架构提供 MNN runtime，
不需要为日常链路额外安装 ONNX Runtime。模型启动前会依据
`lab_policy_manifest.json` 校验文件大小和 SHA-256。

```bash
cd /home/esd/Pheonix/Deploy
sha256sum src/inference/models/lab_policy.mnn
cat src/inference/models/lab_policy_manifest.json
source /opt/ros/humble/setup.bash
colcon build --packages-select inference
```

### 5. CANable 设备配置

#### 5.1 检查设备连接

```bash
# 检查 USB 设备
lsusb | grep -i can

# 检查串口设备（CANable 通常显示为 /dev/ttyACM0 或 /dev/ttyUSB0）
ls -la /dev/ttyACM* /dev/ttyUSB*
```

#### 5.2 配置用户权限

将用户添加到 `dialout` 组（访问串口设备需要）：

```bash
sudo usermod -a -G dialout $USER
```

**注意**：需要重新登录后生效。

#### 5.3 配置 CAN 接口

**方法1：使用兼容 CAN 配置脚本（推荐）**

```bash
cd /home/esd/Pheonix/Deploy
./setup_dual_can.sh
```

该脚本的名称保留自旧双 CAN 链路，当前只配置 `can0`。如果原生 SocketCAN
接口不存在，脚本会按配置尝试从 CANable CDC ACM 设备创建 `can0`。这是一次性
主机维护操作，脚本会为需要的命令调用 `sudo`；日常 `start_robot.sh` 不会调用它。

**方法2：手动配置**

```bash
sudo ip link set can0 down || true
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 txqueuelen 1000
sudo ip link set can0 up

```

#### 5.4 验证 CAN 接口（可选）

```bash
# 查看接口状态
ip link show can0

# 查看详细统计信息
ip -s link show can0

# 监听 CAN 消息
candump can0
```

如果能看到消息收发，说明 CAN 接口配置成功。

### 6. 配置项目参数

#### 6.1 配置电机节点

编辑 `src/motors/config/motors.yaml`，根据实际硬件配置调整参数：

```yaml
motors_node:
    ros__parameters:
        can_interfaces: ["can0", "can0", "can0", "can0", "can0", "can0"]
        motor_ids: [1, 2, 3, 4, 5, 6]  # 电机 ID 列表
        # motor_types、PD、默认角和限位以当前 motors.yaml 为准
```
配置完成后重新编译！

#### 6.2 配置 IMU 节点

编辑 `src/hipnuc_imu/config/hipnuc_config.yaml`，根据实际 IMU 配置调整参数。
```yaml
imu_node:
    ros__parameters:
        # 串口配置
        serial_port: "/dev/ttyUSB0"  # 使用 sudo ls /dev/ttyUSB*指令查看IMU端口
        baud_rate: 115200             
        
        # 话题和帧配置
        frame_id: "imu_link"
        imu_topic: "/IMU_data"       # 发布到 /IMU_data（inference 节点订阅）
        publish_rate: 200            # 定时器上限；只在收到新串口姿态帧时发布
        
        # 低通滤波配置（与推理节点一致）
        imu_ang_vel_lpf_alpha: 1.0   # 角速度低通滤波系数 (0-1, 越小越平滑)
        imu_gravity_lpf_alpha: 1.0   # 重力投影低通滤波系数 (0-1, 越小越平滑)
        
        # 数据记录和绘图
        serial_stale_reconnect_seconds: 0.5
        enable_plotting: false      # 生产环境建议关闭数据记录和绘图
        output_file: ""              # 留空则自动生成时间戳文件名
```
#### 6.3 配置 inference 节点
编辑 `src/inference/config/lab_inference.yaml`，根据实际推理节点配置调整参数：
```yaml
lab_inference_node:
    ros__parameters:
        model_name: "lab_policy.mnn"
        motor_ids: [1, 2, 3, 4, 5, 6]
        wing_angle_topic: "/policy/wing_angles"
        wing_angle_timeout_seconds: 0.1
        imu_timeout_seconds: 0.2
        default_height_command: 0.24
        intra_threads: 1
        dt: 0.0025
        decimation: 8              # 0.02 秒，即 50 Hz
        joy_policy_gate_enabled: false  # 按键由 supervisor 独占
        joy_policy_start_enabled: false # 启动时必须为 Standby 零输出
```
### 7. 验证部署

#### 7.1 验证基础包编译

```bash
# 查看电机节点启动参数（不会启动节点）
source install/setup.bash
ros2 launch motors motors_node.launch.py --help

# 查看 IMU 节点启动参数
source install/setup.bash
ros2 launch hipnuc_imu imu_node.launch.py --help

# 查看 inference 节点启动参数
source install/setup.bash
ros2 launch inference lab_inference_node.launch.py --help
```

#### 7.2 运行测试（如果有硬件）

```bash
# 仅限维护：单独启动腿部节点会绕过 supervisor 的整机自检
source install/setup.bash
ros2 launch motors motors_node.launch.py
```

日常整机验收请始终使用 `./start_robot.sh`，不要用上面的单节点命令代替。

### 8. 常见问题排查

#### 8.1 编译问题

**问题：找不到 ROS2 包**

```bash
# 确保已 source ROS2 环境
source /opt/ros/humble/setup.bash

# 检查 ROS2 是否安装
ros2 --help
```

**问题：找不到系统库（fmt, spdlog, eigen3）**

```bash
# 重新安装依赖
sudo apt update
sudo apt install -y libfmt-dev libspdlog-dev libeigen3-dev
```

**问题：编译错误**

```bash
# 直接重新编译；如确需清理构建产物，请先确认目录和正在运行的进程
colcon build --packages-select motors inference deploy_tools hipnuc_imu xbox
```

#### 8.2 CAN 接口问题

**问题：找不到 /dev/ttyACM0**

```bash
# 检查设备连接
lsusb | grep -i can
dmesg | tail -20  # 查看系统日志

# 检查用户权限
groups  # 应该包含 dialout
```

**问题：Permission denied**

```bash
# 将用户添加到 dialout 组
sudo usermod -a -G dialout $USER
# 然后重新登录
```

**问题：设备被占用**

```bash
# 关闭现有 CAN 接口
sudo ip link set can0 down

# 或卸载 slcan 模块
sudo modprobe -r slcan
```

**问题：波特率不匹配**

```bash
# 检查当前波特率
ip -details link show can0

# 重新配置
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

#### 8.3 运行时问题

**问题：Package not found**

```bash
# 确保在工作区根目录编译并 source
cd /home/esd/Pheonix/Deploy
source /opt/ros/humble/setup.bash
source install/setup.bash

# 验证包是否存在
ros2 pkg list | grep motors
```

**问题：实时优先级设置失败**

```bash
# 检查 limits 配置
cat /etc/security/limits.conf | grep your_username

# 检查当前限制
ulimit -r
ulimit -l

# 不要用 root 运行日常部署；按 2.5 节配置后重新登录
```

**问题：IMU无法被检测到**
```bash
# 新安装的ubuntu 22.04LTS会默认启动一个名为brltty（盲人读屏）的进程，占用端口使得电脑无法找到IMU，卸载即可
sudo apt remove brltty
#卸载后重启电脑并插拔USB设备即可
ls /dev/ttyUSB0
```


### 9. 🎆🎆🎆🎆恭喜顺利完成部署🎆🎆🎆🎆
这么长的部署流程，我自己写完都有点难绷。不得不说，Sim2Real确实是一个浩大的工程，而你已经克服重重困难完成了第一步，恭喜你！我打算未来开发一个一键部署，准备好开始享受机器人带来的无限乐趣吧。日拱一卒，功不唐捐！✊
（如果你是通过快速部署一键完成的，当我没说。）

部署完成后，您可以：
1. 查看 `docs/LAB_DEPLOY_START.md` 了解当前实机启动流程
2. 查看 `docs/troubleshooting/USEFUL_SCRIPTS.txt` 了解保留脚本用途

                                                       ———— by Mytheo Ma
                                                              2026/01/20


---





---

## 🔧 代码架构与实现细节

如果你不满足于仅仅使用代码，而是对代码具体实现也有兴趣，那么欢迎你看看接下来的内容。

本章节详细介绍 `motors_node` 的代码架构、数据流、调用顺序、实时性机制、线程处理和数据安全性。

### 1. 整体架构概述

`motors_node` 采用**分层架构**设计，主要包含以下层次：

1. **ROS2 接口层**：处理 ROS2 话题订阅/发布、服务调用
2. **控制逻辑层**：实现 50Hz 推理到 200Hz 电机控制
3. **电机驱动层**：封装不同型号电机（RS05、RS00）的 CAN 通信协议
4. **CAN 总线层**：`SocketCAN` 提供统一的 CAN 接口和后台接收线程

```
┌─────────────────────────────────────────────────┐
│     ROS2 接口层 (MultiThreadedExecutor)          │
│  - policy_command_callback (50Hz 推理指令)       │
│  - control_loop (200Hz 定时器)                   │
│  - 运行状态服务 (DISARMED/STANDBY/ACTIVE/FAULT)   │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│           控制逻辑层 (MotorsNode)                │
│  - CommandInterpolator (线性插值)                │
│  - 在线电机白名单 (is_motor_connected_)           │
│  - 位置限制检查 (joint_position_limits_)          │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│          电机驱动层 (MotorDriver)                 │
│  - RS05MotorDriver / RS00MotorDriver            │
│  - canRxCallback (状态解析与更新)                 │
│  - MotorMitModeCmd (控制指令发送)                 │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│          CAN 总线层 (SocketCAN)                  │
│  - receiver_thread_ (后台接收线程, RT Priority 80)│
│  - transmit() (发送队列, 128帧缓冲区)              │
│  - 回调注册机制 (can_callback_list_)              │
└─────────────────────────────────────────────────┘
                        │
                        ▼
                    CAN 总线 (can0)
```

### 2. 数据流详解

#### 2.1 控制指令流（推理节点 → 电机）

```
推理节点 (50Hz)
    │
    ├─> /policy/commands (Float32MultiArray)
    │       │
    │       ▼
    ├─> policy_command_callback() [ROS2 回调线程]
    │       │
    │       ├─> 解析位置指令（6个电机）
    │       ├─> 应用位置限制检查
    │       ├─> 应用方向翻转 (flipped_motors_)
    │       ├─> 当前正式配置关闭插值，直接更新目标
    │       └─> 保存供控制循环使用的命令
    │               │
    │               ▼
    └─> control_loop() [200Hz 定时器]
            │
            ├─> 获取插值指令 (interpolators_[i].get_next_command())
            │       │
            │       ▼
            ├─> MotorMitModeCmd() [控制层]
            │       │
            │       ├─> 构建 CAN 帧 (MIT 模式命令)
            │       │   - 位置、速度、Kp、Kd、力矩
            │       │
            │       └─> SocketCAN::transmit() [CAN 层]
            │               │
            │               ▼
            └─> CAN 总线 (can0) → 电机硬件
```

**关键特性**：
- **控制解耦**：50Hz 推理更新目标，200Hz 控制循环负责向电机发送命令
- **位置限制**：在 `policy_command_callback` 中检查 `joint_position_limits_`，防止越界
- **白名单机制**：只对 `is_motor_connected_[i] == true` 的电机发送指令

#### 2.2 状态反馈流（电机 → 推理节点）

```
电机硬件
    │
    ├─> CAN 总线 (can0) [反馈帧，通信类型 0x02]
    │       │
    │       ▼
    └─> SocketCAN::receiver_thread_ [后台接收线程, RT Priority 80]
            │
            ├─> select() 监听 CAN socket
            ├─> read() 读取 CAN 帧
            │
            ├─> 提取电机 ID (CAN ID 的 bit 8-15)
            ├─> 查找回调函数 (can_callback_list_)
            │
            └─> MotorDriver::canRxCallback() [回调执行]
                    │
                    ├─> 解析反馈数据 (8字节)
                    │   - 位置、速度、力矩 (uint16, Big Endian)
                    │   - 温度 (uint16)
                    │   - 故障信息 (CAN ID 的 data_field)
                    │
                    ├─> 转换为物理量
                    │   - uintToFloat() 解码位置/速度/力矩
                    │   - 温度 = raw_value * 0.1
                    │
                    ├─> 更新电机状态 (std::atomic<float>)
                    │   - motor_pos_、motor_spd_、motor_current_
                    │   - motor_temperature_、error_id_
                    │
                    └─> response_count_ = 0 [重置离线计数]
                            │
                            ▼
            control_loop() [200Hz 定时器]
                    │
                    ├─> motor->get_motor_pos() [读取状态]
                    ├─> motor->get_motor_spd()
                    ├─> motor->get_motor_current()
                    │
                    ├─> 应用方向翻转 (flipped_motors_)
                    ├─> 计算相对位置 (position - joint_default_angle_)
                    │
                    └─> publish_joint_states_policy() [200Hz 发布]
                            │
                            └─> /policy/joint_states (JointState)
                                    │
                                    ▼
                            推理节点（读取最新观测并以 50Hz 推理）
```

**关键特性**：
- **后台接收**：`receiver_thread_` 独立线程处理所有 CAN 接收，避免阻塞控制循环
- **原子状态**：电机状态使用 `std::atomic<float>` 保证线程安全
- **高频发布**：`/policy/joint_states` 以 200Hz 发布，确保推理节点获取最新观测

### 3. 调用顺序（节点启动流程）

#### 3.1 构造函数执行流程

```cpp
MotorsNode::MotorsNode()
    │
    ├─> 1. 读取参数 (declare_parameter / get_parameter)
    │       - can_interface, motor_ids, motor_types
    │       - kp, kd, joint_default_angle, flipped_motors
    │       - control_frequency (200Hz), policy_frequency (50Hz)
    │
    ├─> 2. 创建 SocketCAN 实例
    │       └─> SocketCAN::get(can_interface)
    │               │
    │               ├─> 打开 CAN socket (socket PF_CAN)
    │               ├─> 绑定到网络接口 (bind)
    │               ├─> 设置为非阻塞模式 (fcntl O_NONBLOCK)
    │               │
    │               └─> 启动 receiver_thread_ [后台线程]
    │                       │
    │                       ├─> pthread_setname_np("can_rx")
    │                       ├─> pthread_setschedparam(SCHED_FIFO, priority=80)
    │                       └─> select() 循环接收 CAN 帧
    │
    ├─> 3. 创建电机驱动实例
    │       └─> MotorDriver::MotorCreate(...)
    │               │
    │               ├─> 创建 RS05MotorDriver 或 RS00MotorDriver
    │               ├─> 注册 CAN 回调 (SocketCAN::add_can_callback)
    │               │       └─> can_callback_list_[motor_id] = canRxCallback
    │               │
    │               └─> 初始化互斥锁 (motor_mutexes_.push_back)
    │
    ├─> 4. 初始化插值器
    │       └─> interpolators_.resize(motor_ids_.size())
    │       └─> current_commands_ 初始化为默认位置
    │
    ├─> 5. 创建 ROS2 订阅/发布
    │       ├─> policy_command_subscription_ (/policy/commands)
    │       ├─> joint_states_policy_publisher_ (/policy/joint_states)
    │       ├─> motor_state_publishers_ (/motor/motor_*/state)
    │       └─> runtime_status_publisher_ (/motors/runtime_status)
    │
    ├─> 6. 创建 ROS2 服务
    │       ├─> control_motor_service_
    │       ├─> reset_motors_service_
    │       ├─> set_zeros_service_
    │       ├─> set_operational_state_service_
    │       └─> ...
    │
    ├─> 7. 按 control_frequency 创建定时器（当前配置 200Hz）
    │       └─> timer_ = create_wall_timer(5ms)
    │               └─> 回调: control_loop()
    │
    └─> 8. 执行启动检测
            │
            ├─> if (debug_mode_)
            │       └─> init_motors() [直接标记已初始化]
            │
            └─> else
                    └─> auto_init_sequence() [只读检测后进入 DISARMED]
```

#### 3.2 DISARMED 启动序列 (`auto_init_sequence`)

```cpp
auto_init_sequence()
    │
    ├─> 步骤1: 检查配置中的 CAN 接口
    │       ├─> check_can_interface_exists(can0)
    │       ├─> check_can_interface_up(can0)
    │       │
    │       └─> if (!exists or !up)
    │               └─> 直接失败退出，不自动改用其他接口
    │
    ├─> 步骤2: 只读检测 1–6 号在线状态 [避免 Socket 竞争]
    │       │
    │       ├─> [Action] 发送查询指令
    │       │       └─> for each motor:
    │       │               └─> motor->refresh_motor_status()
    │       │                       │
    │       │                       └─> response_count_++ (变为非0)
    │       │
    │       ├─> [Wait] 等待后台线程处理 (200ms)
    │       │       └─> receiver_thread_ 接收反馈帧
    │       │               └─> canRxCallback() 执行
    │       │                       └─> response_count_ = 0 (重置)
    │       │
    │       └─> [Observe] 检查状态
    │               └─> for each motor:
    │                       ├─> if (response_count_ == 0)
    │                       │       └─> is_motor_connected_[i] = true [在线]
    │                       └─> else
    │                               └─> is_motor_connected_[i] = false [离线]
    │
    └─> 步骤3: start_disarmed=true
            ├─> 不调用 MotorInit()
            ├─> 不调用 MotorSetZero()
            ├─> 以 20Hz 查询并发布只读反馈
            └─> 发布 /motors/runtime_status = DISARMED
```

**关键设计**：
- **避免 Socket 竞争**：不在 `auto_init_sequence` 中直接调用 `receive_all_frames()`，而是利用 `response_count_` 机制
- **Y 前不发力**：日常配置 `start_disarmed=true`，自检阶段仅查询反馈，不执行 `MotorInit()`
- **显式使能**：只有 supervisor 自动完成翼启动目标和推理零输出检查后收到 Y，
  才按当前选择进入 Standby，或无斜坡进入倒地自启 ACTIVE/Policy
- **安全斜坡**：Standby 进入过程持续检查反馈、实际位置、目标命令和机械限位；失败即失能并锁存故障
- **倒地自启隔离**：ACTIVE 初始化不执行默认姿态斜坡，并在 inference 明确公告 Policy 前忽略零动作

### 4. 节点启动后的现象

#### 4.1 正常启动流程（由 supervisor 管理）

1. **日志输出**：
   ```
   [INFO] CAN接口 'can0' 已就绪
   [INFO] 步骤2: 逐个检测电机在线状态...
   [INFO] 已发送查询指令，等待后台线程同步状态(200ms)...
   [INFO] 电机 1 在线 (response_count=0)
   [INFO] 电机 2 在线 (response_count=0)
   ...
   [INFO] 找到 6 个在线电机
   [INFO] ========== 自动初始化流程完成（DISARMED） ==========
   [STATE] WING_POSITIONING
   ... 7/8 号软件目标自动到 +90°/-90°，等待 X 选择模式、Y 执行 ...
   [STATE] STANDBY_ENTERING
   [INFO] Standby ramp complete
   [STATE] STANDBY
   ```

2. **CAN 总线活动**：
   - DISARMED 阶段：1–6 号仅以低频状态请求维持只读反馈
   - 普通模式 Y 后斜坡及运行阶段：每 5ms（200Hz）发送 MIT 模式控制帧
   - 倒地自启 Y 后：初始化期间不做姿态斜坡，inference 公告 Policy 后才开始发送策略 MIT 帧
   - 反馈帧：电机以相同频率回复状态帧

3. **ROS2 话题活动**：
   - `/policy/joint_states`：200Hz 发布（供推理节点使用）
   - `/motors/runtime_status`：发布 DISARMED/STANDBY/ACTIVE/FAULT 和 1–6 号健康状态
   - `/wing/runtime_status`：发布 POSITIONING/POSITION_HOLD/RC_CONTROL/FAULT
   - `/inference/runtime_mode`：发布 Standby(0)/Policy(1)
   - `/motor/motor_*/state`：50Hz 发布（用于调试/Rviz）
   - `/policy/commands`：50Hz 订阅（接收推理节点指令）

#### 4.2 运行时行为

1. **控制循环**（200Hz）：
   - DISARMED：不发 MIT 控制帧，仅低频 `refresh_motor_status()`
   - STANDBY/Policy：处理最新命令并发送 MIT 模式控制帧
   - ACTIVE（倒地自启）：首次 inference Policy 公告前拒绝零动作，公告后接受策略帧
   - 发布电机状态到 ROS2 话题

2. **离线检测**：
   - 每次发送控制指令后，检查 `response_count_`
   - 如果超过配置的 `offline_threshold`：锁存 FAULT、发送零增益零命令并失能

3. **模式行为**：
   - Standby 下推理节点持续发布六维有限零值，电机保持默认姿态
   - Policy 下发布 MNN 动作；X 切回 Standby；Y 冻结末帧并三秒卸力，然后失能 1–6 号
   - 传感器/推理故障且电机通信健康时先回故障 Standby；电机、CAN、翼或越限故障立即失能

### 5. 实时性机制

#### 5.1 线程优先级设置

| 线程/进程 | 优先级 | 调度策略 | 说明 |
|----------|--------|----------|------|
| **CAN 接收线程** (`receiver_thread_`) | 80 | SCHED_FIFO | 最高优先级，确保及时处理 CAN 反馈 |
| **主线程** (ROS2 Executor) | 70 | SCHED_FIFO | 次高优先级，处理控制循环 |
| **维护任务线程**（仅旧配置/维护服务） | 60 | SCHED_FIFO | 不属于 WE11 日常启动链路 |
| **普通进程** | 0 | SCHED_OTHER | 默认优先级 |

**设置位置**：
```cpp
// CAN 接收线程 (SocketCAN.cpp)
pthread_setschedparam(pthread_self(), SCHED_FIFO, {.sched_priority = 80});

// 主线程 (motors_node.cpp main())
pthread_setschedparam(pthread_self(), SCHED_FIFO, {.sched_priority = 70});

// 维护任务线程（仅在显式启用旧初始化/标定路径时创建）
pthread_setschedparam(pthread_self(), SCHED_FIFO, {.sched_priority = 60});
```

#### 5.2 定时器精度

- **控制频率**：200Hz（5ms 周期）
- **定时器类型**：`create_wall_timer()` (基于系统时钟)
- **Jitter 控制**：使用实时优先级减少调度延迟

#### 5.3 CAN 发送队列

- **队列大小**：128 帧 (`TX_QUEUE_SIZE = 128`)
- **作用**：缓冲突发发送，避免阻塞控制循环
- **发送方式**：非阻塞 `send()`，失败时丢弃帧（避免阻塞）

### 6. 线程处理

#### 6.1 线程架构

```
┌─────────────────────────────────────────────────────┐
│        主进程 (motors_node)                          │
│                                                      │
│  ┌──────────────────────────────────────────────┐  │
│  │  ROS2 MultiThreadedExecutor (4个线程池)      │  │
│  │  - policy_command_callback [回调线程]        │  │
│  │  - control_loop [200Hz 定时器线程]           │  │
│  │  - /motors/set_operational_state [服务线程]  │  │
│  │  - 其他 ROS2 回调                            │  │
│  └──────────────────────────────────────────────┘  │
│                      │                               │
│                      ▼                               │
│  ┌──────────────────────────────────────────────┐  │
│  │  SocketCAN::receiver_thread_ [独立线程]      │  │
│  │  - Priority: 80 (SCHED_FIFO)                 │  │
│  │  - 持续 select() 监听 CAN socket             │  │
│  │  - 调用 MotorDriver::canRxCallback()         │  │
│  └──────────────────────────────────────────────┘  │
│                      │                               │
│                                                      │
│  WE11 日常配置 start_disarmed=true：                  │
│  启动只查询反馈，不创建自动写零任务、不使能电机       │
└─────────────────────────────────────────────────────┘
```

#### 6.2 线程间通信

1. **CAN 接收线程 → 控制线程**：
   - **机制**：回调函数 (`canRxCallback`)
   - **数据**：电机状态 (`std::atomic<float>`)
   - **同步**：原子操作，无需互斥锁

2. **ROS2 回调线程 → 控制线程**：
   - **机制**：`CommandInterpolator` + `std::shared_mutex`
   - **数据**：插值器状态 (`interpolators_`, `current_commands_`)
   - **同步**：`std::shared_mutex`（读写锁）

3. **运行状态服务 → 控制线程**：
   - **机制**：`operational_state_` 原子状态和服务响应
   - **数据**：`DISARMED`、`STANDBY`、`FAULT` 及 3 秒进入斜坡
   - **同步**：状态原子更新；控制循环只按当前状态决定是否发送控制帧

### 7. 数据安全性

#### 7.1 线程安全机制

| 数据结构 | 类型 | 线程安全机制 | 访问模式 |
|---------|------|-------------|---------|
| `motor_pos_`, `motor_spd_`, `motor_current_` | `std::atomic<float>` | 原子操作 | 写：CAN 接收线程；读：控制线程 |
| `response_count_` | `std::atomic<int>` | 原子操作 | 写：CAN 接收线程；读：控制线程 |
| `interpolators_` | `std::vector<CommandInterpolator>` | `std::shared_mutex` | 写：ROS2 回调线程；读：控制线程 |
| `current_commands_` | `std::vector<JointCommand>` | `std::shared_mutex` | 写：控制线程；读：ROS2 回调线程 |
| `is_motor_connected_` | `std::vector<bool>` | 启动检测后只读 | 写：启动检测；读：所有线程 |
| `is_init_` | `std::atomic<bool>` | 原子操作 | 写：运行状态服务；读：所有线程 |
| `operational_state_` | `std::atomic<uint8_t>` | 原子操作 | 写：状态服务/故障路径；读：控制线程和状态发布器 |
| `policy_is_active_` | `std::atomic<bool>` | 原子操作 | 写：ROS2 回调线程；读：控制线程 |

#### 7.2 关键同步点

1. **策略命令更新**（`policy_command_callback` → `control_loop`）：
   ```cpp
   // policy_command_callback：在互斥保护下保存最新目标命令
   // control_loop：读取最新命令并按 operational_state_ 决定是否下发
   // 当前 motors.yaml 中 enable_interpolation=false，策略命令直接更新；
   // 如维护配置显式开启插值，则通过 CommandInterpolator 生成中间命令。
   ```

2. **电机状态读取**（CAN 接收线程 → 控制线程）：
   ```cpp
   // canRxCallback (CAN 接收线程)
   motor_pos_.store(parsed_position);  // 原子写
   response_count_.store(0);  // 原子写
   
   // control_loop (控制线程)
   float pos = motor->get_motor_pos();  // 内部调用 motor_pos_.load() (原子读)
   ```

3. **在线检测**（启动检测 → 控制线程）：
   ```cpp
   // auto_init_sequence（WE11 日常配置只查询，不使能）
   is_motor_connected_[i] = true;  // 初始化阶段写入
   
   // control_loop (控制线程)
   if (!is_motor_connected_[i]) continue;  // 初始化后只读，无需锁
   ```

#### 7.3 数据一致性保证

1. **电机状态**：
   - 使用 `std::atomic<float>` 保证读写原子性
   - CAN 接收线程写入，控制线程读取，无竞争

2. **插值器状态**：
   - 使用 `std::shared_mutex` 保护 `interpolators_` 和 `current_commands_`
   - `policy_command_callback` 获取写锁更新目标
   - `control_loop` 获取写锁读取并更新当前命令

3. **在线电机列表**：
   - `is_motor_connected_` 在初始化完成后不再修改
   - 后续访问为只读，无需同步

#### 7.4 错误处理与安全机制

1. **离线检测**：
   - `response_count_` 在发送指令时自增，收到反馈时清零
   - 如果超过当前配置的 `offline_threshold`：锁存 `FAULT`，发送零增益零命令并失能

2. **位置限制**：
   - 在 `policy_command_callback` 中检查 `joint_position_limits_`
   - 超出限制时截断到边界值

3. **异常处理**：
   - CAN 发送失败时捕获异常，记录警告日志
   - 启动检测失败时不进入 Standby，不发送控制指令

---

## 📊 编译状态

### ✅ 已成功编译的包



1. **motors** - 电机控制节点
   - ✅ motors_node - 电机控制节点
   - ✅ wing_motor_node - 7/8 号翼电机节点
   - ✅ 运行状态消息以及设置运行状态服务

2. **inference** - 推理节点
   - ✅ lab_inference_node - MNN 实机推理节点
   - ✅ sweep_frequency_node - 扫频曲线下发节点

3. **deploy_tools** - WE11 启动与硬件工具
   - ✅ we11_supervisor.py - 日常状态机、故障锁存与安全清理
   - ✅ rgb_led.py - RGB 常亮、呼吸、闪烁与故障时序

4. **hipnuc_imu** - IMU 节点
   - ✅ 编译成功

5. **hipnuc_lib_package / xbox** - IMU 库与手柄节点
   - ✅ 编译成功

## 📁 项目结构

```
Deploy/
├── src/                # ROS 2 功能包
│   ├── deploy_tools/   # CAN、诊断、参数和离线分析工具
│   ├── hipnuc_imu/     # IMU 节点
│   ├── inference/      # 推理节点
│   ├── motors/         # 电机节点
│   └── xbox/           # 手柄节点
├── experiments/        # 不属于正式启动链路的实验
├── scripts/sweeps/     # 1/4、2/5、3/6 成套扫频脚本
├── config/system/      # udev/systemd 主机配置模板
├── docs/               # 部署、排障和硬件资料
└── *.sh                # 稳定的现场操作入口
```

完整目录职责和旧命令兼容策略见 `docs/WORKSPACE_LAYOUT.md`。

## 📚 文档说明

- **docs/LAB_DEPLOY_START.md** - 当前 Lab 实机启动流程
- **docs/WE11_DAILY_STATE_MACHINE.md** - 日常状态机、灯光、按键及故障码
- **docs/troubleshooting/USEFUL_SCRIPTS.txt** - 当前保留脚本清单和用途
- **docs/WORKSPACE_LAYOUT.md** - ROS 2 工作区目录和规范脚本使用说明



## 许可证

[待添加]

## 贡献

欢迎提交 Issue 和 Pull Request。
