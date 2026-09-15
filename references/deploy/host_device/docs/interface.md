# Walking Eagle WE11 Deploy 接口说明

负责人：汪成浩
适用分支：`Walking_Eagle-Deploy_final`

## 1. ROS 输入

| 名称 | 类型 | 频率 | 来源 |
|---|---|---:|---|
| `/IMU_data` | `sensor_msgs/Imu` | 200 Hz | `hipnuc_imu` |
| `/joy` | `sensor_msgs/Joy` | 手柄频率 | ROS joy node |
| `/cmd_vel` | `geometry_msgs/Twist` | 50 Hz | Xbox command 节点 |
| `/policy/joint_states` | `sensor_msgs/JointState` | 200 Hz | `motors_node` |
| `/policy/wing_angles` | `sensor_msgs/JointState` | 200 Hz | `wing_motor_node` |

## 2. ROS/CAN 输出

| 名称 | 类型 | 频率 | 目标 |
|---|---|---:|---|
| `/policy/commands` | `std_msgs/Float32MultiArray`，6D | 50 Hz | `motors_node` |
| 1～6 号 MIT 帧 | SocketCAN `can0` | 200 Hz | 腿轮电机 |
| 7/8 号控制帧 | SocketCAN `can0` | 200 Hz | 翼电机 |
| 扫频 CSV | 文件 | 200 Hz | `scripts/sweeps/*/logs` |

`/motors/set_operational_state` 使用 `motors/srv/SetOperationalState`：
`DISARMED=0` 为只读失能，`STANDBY=1` 为初始化后斜坡到默认姿态，`ACTIVE=2`
为倒地自启专用的无斜坡初始化/使能。ACTIVE 在首次收到
`/inference/runtime_mode=Policy(1)` 前拒绝 `/policy/commands`。

`/motors/soft_disarm` 使用 `std_srvs/Trigger`：仅用于操作员正常按 `Y` 退出。服务冻结
最后一帧电机指令，保持目标位置并在 `soft_disarm_seconds=3.0` 秒内线性降低速度、
力矩、Kp 和 Kd，归零后再 `MotorLock`。`DISARMED=0` 的立即硬失能语义保持不变。

## 3. 坐标与关节合同

```text
策略顺序 = [左大腿, 左小腿, 左轮, 右大腿, 右小腿, 右轮]
单位     = [rad, rad, rad, rad, rad, rad]
默认角   = [-0.92020, +0.98338, 0, -0.92020, +0.98338, 0]
翻转     = [true, false, true, false, true, false]
IMU轴向  = sensor [x,y,z] -> base_link [-x,-y,+z]
```

`/policy/joint_states` 必须是已经按 `flipped_motors` 和 `joint_default_angle` 转换后的策略状态；原始编码器角不得直接代替策略相对角。

## 4. 关键配置

| 配置项 | 当前值 | 文件 |
|---|---|---|
| CAN | 单路 `can0`，1 Mbps | `setup_dual_can.sh` |
| Motor IDs | `[1,2,3,4,5,6]` | `src/motors/config/motors.yaml` |
| Kp | `[2,8,0,2,8,0]` | 同上 |
| Kd | `[0.1,0.8,0.25,0.1,0.8,0.25]` | 同上 |
| position limits | `[-1.23,0.77,-0.98338,1.51662,-6.28,6.28]` 左右对称 | 同上 |
| motor control | 200 Hz | 同上 |
| policy | 50 Hz | `lab_inference.yaml` |
| policy dt/decimation | `0.0025/8` | 同上 |
| height command | `0.24 m` | 同上 |
| offline threshold | 600 次发送未收到合法反馈 | `motors.yaml` |

## 5. 控制时序

```text
IMU + motor feedback + wing feedback + cmd_vel
                    |
                    v
              50 Hz MNN policy
                    |
                    v
              /policy/commands
                    |
                    v
             200 Hz motors_node
                    |
                    v
              MIT CAN command
                    |
                    v
             matched CAN feedback
```

每次 MIT 发送增加响应计数；只有匹配 motor ID 和反馈帧类型的接收帧才能清零。超过阈值触发锁存安全态，不能以增大阈值替代根因诊断。

## 6. 扫频接口

| 分组 | 电机 | 配置目录 |
|---|---|---|
| 大腿 | 1/4 | `scripts/sweeps/sweep_motor14/` |
| 小腿 | 2/5 | `scripts/sweeps/sweep_motor25/` |
| 轮子 | 3/6 | `scripts/sweeps/sweep_motor36/` |

扫频输出必须保存时间、command、实际位置/速度/力矩和配置元数据；采集完成后由 UniLab PACE 拟合脚本消费，Deploy 不负责修改辨识结果。

## 7. 安全接口

- 正常退出：`start_robot.sh` 前台 `Ctrl+C`；
- 异常退出：关键电机节点退出时停止推理和其余控制链路；
- CAN 异常：进入 `0kp0kd` 锁存安全模式；
- 恢复条件：排除通信/供电问题后重新初始化或按当前流程重新设置零位；
- 禁止：并行启动扫频、正常 policy、另一个 motors node 或第三方 CAN 控制程序。
