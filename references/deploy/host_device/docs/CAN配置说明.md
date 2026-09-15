# CAN 配置说明

当前正式部署使用一路 Linux SocketCAN 接口 `can0`：

```text
can0 -> motors 1/2/3/4/5/6/7/8
bitrate = 1000000
```

1~6 号映射来自 [`motors.yaml`](../src/motors/config/motors.yaml)，7/8 号映射
来自 [`wing_motors.yaml`](../src/motors/config/wing_motors.yaml)。

## 接口脚本

根目录 [`setup_dual_can.sh`](../setup_dual_can.sh) 的文件名为历史兼容名称，
当前只配置 `can0`：

- 如果适配器已经由 `gs_usb` 等驱动暴露为原生 SocketCAN，则尝试设置
  `bitrate=1000000`、`restart-ms=100` 和 `txqueuelen=1000` 后拉起接口。
- 如果 `can0` 不存在，则默认用 `slcand` 将 `/dev/ttyACM0` 创建为 `can0`，
  其中 `s8` 表示 1 Mbps。
- 原生参数设置返回 `Operation not supported` 时，脚本仍会尝试拉起接口；
  对 `slcand` 接口，波特率由 `slcand -s8` 决定。

可通过环境变量覆盖：

```bash
CAN_IFACE=can0 CAN_BITRATE=1000000 CAN_TXQUEUELEN=1000 \
SLCAN_TTY=/dev/ttyACM0 SLCAN_SPEED=s8 ./setup_dual_can.sh
```

`setup_canable_can0.sh`、`config/system/99-auto-up-devs.rules` 和
`config/system/canable-can0@.service` 用于 CANable CDC ACM 的主机级自动绑定。
根目录保留了两个系统配置文件的兼容符号链接。

## 节点配置

1~6 号电机：

```yaml
can_interfaces: ["can0", "can0", "can0", "can0", "can0", "can0"]
motor_ids: [1, 2, 3, 4, 5, 6]
```

7/8 号翼电机：

```yaml
can_interface: "can0"
motor_ids: [7, 8]
```

正式启动会先确认 `can0` 存在且为 `UP`，再用 0x02 状态请求检查 1~6 号反馈；
检查失败会退出，不会自动改用 `can1`。

## 静态代码位置

- SocketCAN 封装：[`SocketCAN.cpp`](../src/motors/src/SocketCAN.cpp)、
  [`SocketCAN.hpp`](../src/motors/src/SocketCAN.hpp)
- 1~6 号节点：[`motors_node.cpp`](../src/motors/src/motors_node.cpp)
- 7/8 号节点：[`wing_motor_node.cpp`](../src/motors/src/wing_motor_node.cpp)
- 独立 CAN 工具：`src/deploy_tools/scripts/can/`

## 只读检查

以下命令本身不启动 `motors_node`：

```bash
ip -details -statistics link show can0
ros2 run deploy_tools read_motor_status --channel can0 --motor-id 1 --request
ros2 run deploy_tools read_all_motor_status --request
```

最后两条会按原工具语义发送 0x02 状态请求，但不发送 MIT 0x01 控制帧。
