# 轮子 D 与翅膀扑动频率

## 1. 修改轮子 D

轮子是 3、6 号电机，当前使用速度控制，通常保持 `Kp=0`。

### 翼电机遥控参数

先确认 `motors_node` 已启动，然后执行：

```bash
cd /home/esd/deploy_ws/Walking_Eagle-Deploy
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run deploy_tools set_foot_pd --kd 0.1
```

该命令同时修改 3、6 号轮子的 D：

```text
motor_3: Kp=0.0, Kd=0.1
motor_6: Kp=0.0, Kd=0.1
```

只修改一个轮子：

```bash
ros2 run deploy_tools set_foot_pd --foot-ids 3 --kd 0.1
ros2 run deploy_tools set_foot_pd --foot-ids 6 --kd 0.1
```

运行时修改只对当前 `motors_node` 进程有效，重启后会恢复配置文件中的值。

### 永久修改

编辑：

```text
src/motors/config/motors.yaml
```

例如将轮子 D 改为 `0.1`：

```yaml
kd: [0.2, 0.2, 0.1, 0.2, 0.2, 0.1]
```

修改后重新编译：

```bash
cd /home/esd/deploy_ws/Walking_Eagle-Deploy
colcon build --packages-select motors --symlink-install
source install/setup.bash
```

恢复启动时配置的默认增益：

```bash
ros2 run deploy_tools set_foot_pd --restore-defaults
```

## 2. 修改翅膀遥控速度

### 查看当前配置

当前正式链路不再使用固定频率往返。右摇杆纵向 `axes[4]` 直接给出带符号翼速度，
节点将速度积分为位置目标并通过 PD 保持。查看参数：

```bash
ros2 param get /wing_motor_node wing_max_velocity_rad_s
ros2 param get /wing_motor_node wing_rc_kp
ros2 param get /wing_motor_node wing_rc_kd
```

默认最大速度与 Play 一致，为 `pi/4 rad/s`：

```bash
wing_max_velocity_rad_s: 0.7853981634
wing_rc_kp: 20.0
wing_rc_kd: 1.0
```

右摇杆回中或 `/joy` 超过 0.5 秒未更新时，速度命令清零，位置目标冻结。

### 永久修改

编辑：

```text
src/motors/config/wing_motors.yaml
```

修改：

```yaml
wing_max_velocity_rad_s: 0.7853981634
```

修改后重新编译并重新启动节点：

```bash
cd /home/esd/deploy_ws/Walking_Eagle-Deploy
colcon build --packages-select motors --symlink-install
source install/setup.bash
```

## 3. 安全提醒

- 修改轮子 D 前先架空机器人或确保急停可用。
- 修改轮子 D 的脚本不会主动发送策略动作，也不会自动设置零点。
- 操作右摇杆前确认 `7/8` 号电机运动范围内没有人员和障碍物。
- 当前实机软件限位暂为两侧 `[-60°, +60°]`；未经机械验证不要扩大。
