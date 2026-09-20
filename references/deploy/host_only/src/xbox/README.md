# xbox - Xbox 手柄速度指令包

将 Xbox 手柄摇杆数据转换为机器人速度指令的 ROS 2 包。

## 功能特性

- 订阅 `/joy` 话题获取 Xbox 手柄原始数据
- 发布 `/cmd_vel` 话题输出速度指令，与部署推理节点直接兼容
- 内置死区处理，避免摇杆漂移
- 支持通过参数调整最大速度、线加速度、死区和指令超时
- 提供 launch 文件一键启动手柄节点和速度转换节点

## 摇杆映射

| 摇杆 | 轴 | 输出 |
|------|-----|------|
| 左摇杆 Y 轴 | `axes[1]` | `linear.x`（前后速度） |
| 左摇杆 X 轴 | `axes[0]` | `angular.z`（转向角速度） |
| 右摇杆 Y 轴 | `axes[4]` | 由 `wing_motor_node` 直接读取，控制翼速度 |

Flydigi Dune Fox（xpad）实测方向为：左摇杆上/左分别是 `axes[1] +`、
`axes[0] +`，右摇杆上是 `axes[4] +`。正式遥控约定为前进和左转分别输出
正值，`linear.y` 固定为 0。LT/RT 共用 `axes[2] -/+`，当前遥控不使用该轴。

## 话题接口

### 订阅

| 话题 | 类型 | 频率 | 说明 |
|------|------|------|------|
| `/joy` | `sensor_msgs/Joy` | 由 joy_node 控制 | Xbox 手柄原始数据 |

### 发布

| 话题 | 类型 | 频率 | 说明 |
|------|------|------|------|
| `/cmd_vel` | `geometry_msgs/Twist` | 50 Hz | 机器人速度指令 |

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_linear_speed` | `double` | `1.5` | 最大线速度 (m/s) |
| `max_angular_speed` | `double` | `1.0` | 最大角速度 (rad/s) |
| `max_linear_accel` | `double` | `3.0` | `linear.x` 最大加/减速度 (m/s^2)，0 表示关闭限幅 |
| `deadzone` | `double` | `0.05` | 死区值 (0.0-1.0) |
| `command_timeout` | `double` | `0.5` | 手柄数据超时后发布零速度（秒） |
| `input_topic` | `string` | `/joy` | 手柄输入话题 |
| `output_topic` | `string` | `/cmd_vel` | 推理速度指令话题 |

### 死区说明

当摇杆输入绝对值小于死区值时，输出为 0。例如死区为 0.05 时：

- 输入 0.04 → 输出 0.0
- 输入 0.5 → 输出 (0.5-0.05)/(1-0.05) ≈ 0.47

`linear.x` 保留缓加/减速限制。使用默认 `max_linear_accel=3.0` 时，从静止到
满推目标 `1.5 m/s` 理论上约需 `1.5 / 3.0 = 0.5 s`；松手和 Joy 超时后也按
相同加速度限制回到 0。

## 快速开始

### 1. 编译包

```bash
cd /home/esd/Pheonix/Deploy
colcon build --paths src/xbox --symlink-install \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source ./install/setup.bash
```

### 2. 启动节点

同时启动 joy_node 和 xbox_vel_publisher：

```bash
ros2 launch xbox xbox.launch.py
```

### 3. 查看输出

```bash
# 查看速度指令话题
ros2 topic echo /cmd_vel

# 查看话题列表
ros2 topic list

# 查看节点连接
ros2 topic info /cmd_vel
```

## 参数配置

### 通过 launch 参数修改

```bash
# 单参数
ros2 launch xbox xbox.launch.py max_linear_speed:=1.2

# 多参数
ros2 launch xbox xbox.launch.py \
  max_linear_speed:=1.5 \
  max_angular_speed:=1.0 \
  max_linear_accel:=3.0 \
  deadzone:=0.05 \
  command_timeout:=0.5
```

### 通过运行时修改

```bash
# 查看当前参数
ros2 param list /xbox_vel_publisher

# 设置参数
ros2 param set /xbox_vel_publisher max_linear_speed 0.8

# 列出参数
ros2 param get /xbox_vel_publisher max_linear_speed
```
