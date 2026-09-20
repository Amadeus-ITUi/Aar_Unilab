# 工作区结构

本仓库保持 ROS 2 workspace 的常规边界：ROS 包统一位于 `src/`，操作入口、
实验脚本、系统配置和文档分别管理。此次整理只改变代码归属与路径，不改变
控制参数、CAN 协议、话题、服务或启动顺序。

## 目录职责

```text
Walking_Eagle-Deploy/
├── src/                    ROS 2 packages
│   ├── deploy_tools/       CAN 工具、诊断、运行时参数工具和离线分析
│   ├── hipnuc_imu/         IMU 节点
│   ├── inference/          策略推理节点
│   ├── motors/             1~8 号电机节点和接口
│   └── xbox/               手柄节点
├── experiments/            不属于正式启动链路的硬件实验
│   └── wing/               7/8 号翼电机实验
├── scripts/sweeps/         成套扫频脚本、YAML 与运行记录
│   ├── sweep_motor14/      1/4 号扫频流程
│   ├── sweep_motor25/      2/5 号扫频流程
│   └── sweep_motor36/      3/6 号扫频流程
├── config/system/          udev 与 systemd 主机配置模板
├── docs/                   部署、排障和硬件资料
└── *.sh                    稳定的操作入口和兼容启动脚本
```

## 部署工具

`src/deploy_tools` 是一个 `ament_cmake` 包。编译并 source 后，可使用：

```bash
ros2 run deploy_tools <tool_name> [arguments]
```

工具按职责分为：

- `scripts/can/`：SocketCAN 查询、探测和维护工具。
- `scripts/diagnostics/`：ROS 只读诊断和运行数据记录。
- `scripts/control/`：人工明确调用的运行时参数工具。
- `scripts/analysis/`：只处理 CSV 的离线绘图和比较。

分类不等于安全等级。部分 CAN 工具按其原有参数会发送 CAN 帧；使用带电硬件
前仍需检查各脚本的参数说明。

## 兼容策略

`start_robot.sh`、`start_readonly_foxglove.sh`、`record_deploy_runtime.sh` 等操作
入口继续保留在仓库根目录。Python 工具不在根目录保留转发副本；编译并
source 工作区后，统一通过 ROS 2 包执行：

```bash
ros2 run deploy_tools read_motor_status --help
ros2 run deploy_tools plot_policy_pair_tracking --help
```

扫频实现统一位于 `scripts/sweeps/`，直接使用该目录中的正式入口。
`experiments/wing` 中保存不属于正式启动链路的翼电机实验，使用完整实验
路径执行，不在仓库根目录创建副本。

主机配置模板统一位于 `config/system/`，根目录保留同名符号链接，以兼容
原有的 udev/systemd 安装命令；移动源码模板不会改动 `/etc` 中已安装的文件。

## 生成目录

`build/`、`install/`、`log/`、`logs/` 和扫频脚本目录下的运行日志均为生成内容，
不属于源代码结构。整理和静态检查不会启动节点，也不会访问 CAN 总线。
