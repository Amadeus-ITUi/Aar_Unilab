# Remote Controller 远程控制器节点

使用 Xbox 手柄控制机器人，发布高层指令（线速度和角速度）到 `/cmd_vel` 话题。

## 功能特性

- 支持 Xbox 手柄（以及兼容的手柄）
- 自动检测和连接蓝牙手柄
- 发布 `geometry_msgs/msg/Twist` 消息到 `/cmd_vel` 话题
- 可配置的发布频率和话题名称
- 支持底盘控制和头部控制模式切换

## 依赖

### 系统依赖

```bash
sudo apt-get update
sudo apt-get install python3-pygame python3-numpy
```

### ROS2 依赖

- `rclpy`
- `geometry_msgs`

## 编译

```bash
cd /home/esd/project/deploy_cpp
colcon build --paths src/remote_controller --symlink-install
source install/setup.sh
```

## 使用方法

### 1. 直接运行节点

```bash
ros2 run remote_controller remote_controller_node
```

### 2. 使用 Launch 文件

```bash
ros2 launch remote_controller remote_controller.launch.py
```

### 3. 带参数运行

```bash
ros2 run remote_controller remote_controller_node \
  --ros-args \
  -p command_rate:=30.0 \
  -p command_topic:=/cmd_vel \
  -p only_head_control:=false
```

### 4. 使用配置文件

```bash
ros2 run remote_controller remote_controller_node \
  --ros-args \
  --params-file install/remote_controller/share/remote_controller/config/remote_controller.yaml
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `command_rate` | double | 20.0 | 命令发布频率 (Hz) |
| `command_topic` | string | `/cmd_vel` | 命令话题名称 |
| `only_head_control` | bool | false | 仅头部控制模式 |

## 手柄控制说明

### 摇杆映射

- **左摇杆 X 轴**: 控制 `angular.z`（左右转向）
- **左摇杆 Y 轴**: 控制 `linear.x`（前后移动）
- **右摇杆 Y 轴**: 由翼电机节点直接读取，控制翼旋转速度和方向

Flydigi Dune Fox（xpad）实测为左摇杆上 `axes[1] +`、左摇杆左
`axes[0] +`、右摇杆上 `axes[4] +`；遥控节点按前进、左转为正处理，侧向为 0。

### 按钮功能

- **Y 按钮**: 切换底盘控制/头部控制模式

### 速度范围

- `linear.x`: [-0.15, 0.15] m/s
- `linear.y`: [-0.2, 0.2] m/s
- `angular.z`: [-1.0, 1.0] rad/s

## 话题

### 发布话题

- `/cmd_vel` (geometry_msgs/msg/Twist): 速度命令

## 与 inference 节点集成

该节点发布到 `/cmd_vel` 话题，与 `inference_node` 订阅的话题兼容。可以替代或配合使用：

```bash
# 启动远程控制器（替代键盘输入）
ros2 launch remote_controller remote_controller.launch.py

# 或同时运行（远程控制器优先级更高，因为发布频率更高）
ros2 launch remote_controller remote_controller.launch.py &
ros2 launch inference inference_node.launch.py
```

## 故障排除

### 手柄未检测到

1. 确保手柄已通过蓝牙配对
2. 检查系统是否识别手柄：
   ```bash
   ls /dev/input/js*
   ```
3. 如果使用 USB 连接，确保权限正确：
   ```bash
   sudo chmod 666 /dev/input/js0
   ```

### 权限问题

如果遇到权限错误，可能需要将用户添加到 `input` 组：

```bash
sudo usermod -a -G input $USER
# 然后重新登录
```

## 开发说明

节点代码位于：`src/remote_controller/remote_controller_node.py`

主要类：
- `XBoxController`: 手柄控制器类
- `RemoteControllerNode`: ROS2 节点类
