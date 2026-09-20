# Walking Eagle WE11 Deploy 算法交接说明

## 1. 当前版本

| 项目 | 内容 |
|---|---|
| 项目 | Walking Eagle WE11 ROS2/CAN 真机部署与扫频采集 |
| 算法交付负责人 | 汪成浩 |
| 交付日期 | 2026-08-04 |
| 仓库 | `git@git.esdyn.cn:walking-eagle/sar_unilab.git` |
| 分支 | `Walking_Eagle-Deploy_final` |
| 功能冻结基线 | `0635201a192ce40332edd16dc6237d9532418dfb` |
| 归档提交 | 以该分支最新 `HEAD` 为准，使用 `git rev-parse HEAD` 获取 |
| 机器人 | Walking Eagle WE11 |
| 当前状态 | 源码与配置归档完成；正式真机运行仍需逐机安全验收 |

README 中原部署框架维护者信息保留；“汪成浩”是本次 Walking Eagle 算法交付负责人。

## 2. 交接范围与责任边界

本仓库包含：

- ROS2 Humble 部署工作区和一键环境构建；
- 1～6 号腿轮电机、7/8 号翼电机、IMU、手柄和 MNN 推理节点；
- `start_robot.sh` 生命周期、安全门控和异常清理；
- SocketCAN `can0` 配置与状态诊断；
- 1/4、2/5、3/6 paired sweep 采集程序；
- 2026-07-30 原始扫频 CSV、说明和 SHA-256；
- 当前 `lab_policy.mnn` 及对应部署配置。

本仓库不包含：

- UniLab/IsaacLab 训练代码；
- PACE、Kp/Kd 离线拟合程序；
- rl_sar MuJoCo 回放；
- 电机固件、USB-CAN 固件、机器人 CAD/电气设计；
- 真机测试视频和未筛选的运行日志。

负责人负责算法接口、启动流程、配置和排障资料交付；不代替硬件电气验收、电机固件验收或操作者现场安全责任。

## 3. 环境依赖

| 项目 | 要求 |
|---|---|
| 操作系统 | Ubuntu 22.04 |
| 架构 | x86_64 或 aarch64；MNN runtime 必须匹配架构 |
| ROS | ROS2 Humble |
| 编译 | CMake、colcon、C++17 |
| Python | 系统 Python；构建 ROS2 时禁用 Conda Python |
| 通信 | SocketCAN `can0`，1 Mbps |
| 主要工具 | can-utils、iproute2、screen、ccache |
| 硬件 | WE11、1～8 号电机、USB-CAN、IMU、Xbox 手柄、急停 |

安装与构建：

```bash
git clone -b Walking_Eagle-Deploy_final \
  git@git.esdyn.cn:walking-eagle/sar_unilab.git Walking_Eagle-Deploy_final
cd Walking_Eagle-Deploy_final
./env_init.sh
```

`env_init.sh` 只验证软件闭包，不会自动使能电机。

## 4. 关键文件

| 路径 | 作用 | 必须 |
|---|---|---|
| `start_robot.sh` | 实机启动、安全确认、监控和统一退出 | 是 |
| `setup_dual_can.sh` | 建立/配置 `can0` | 是 |
| `src/motors/config/motors.yaml` | 1～6 号 ID、类型、PD、默认位、方向、限位和频率 | 是 |
| `src/motors/config/wing_motors.yaml` | 7/8 号回零、扑翼和安全边界 | 是 |
| `src/inference/config/lab_inference.yaml` | 模型、观测坐标、command 和策略周期 | 是 |
| `src/inference/models/lab_policy.mnn` | 当前实机策略 | 是 |
| `src/hipnuc_imu/config/hipnuc_config.yaml` | IMU 串口、话题、频率和滤波 | 是 |
| `scripts/sweeps/` | 1/4、2/5、3/6 paired sweep | 是 |
| `sweep_results_20260730/` | 原始采集数据与 SHA-256 | 建议保留 |
| `docs/DEPLOY_START_AND_SWEEP.md` | 完整 start_robot 与扫频流程 | 是 |

当前 MNN：

```text
file: src/inference/models/lab_policy.mnn
sha256: 103b715a592876ab9816d7f97189299f25fdf4d8fb8c5c0dc68299b0dc4f694f
```

## 5. 从零运行

### 5.1 软件与 CAN 检查

```bash
source /opt/ros/humble/setup.bash
./setup_dual_can.sh
ip -details -statistics link show can0
candump can0
```

同一时间只允许一个程序控制 `can0`。完成监听后先退出 `candump`，再启动控制链路。

### 5.2 正常启动

```bash
./start_robot.sh
```

脚本必须保持前台；正常退出在该终端按 `Ctrl+C`，由脚本统一停止推理和电机链路。完整按键与状态机见 `docs/DEPLOY_START_AND_SWEEP.md`。

### 5.3 扫频采集

确保 `start_robot.sh`、`motors_node` 和其他 CAN 控制程序已退出，然后每次只运行一组：

```bash
./scripts/sweeps/sweep_motor14/run_sweep_motor14.sh
./scripts/sweeps/sweep_motor25/run_sweep_motor25.sh
./scripts/sweeps/sweep_motor36/run_sweep_motor36.sh
```

## 6. 当前关键参数

- 关节顺序：`[左大腿, 左小腿, 左轮, 右大腿, 右小腿, 右轮]`；
- 默认角：`[-0.92020,+0.98338,0,-0.92020,+0.98338,0] rad`；
- Kp：`[2,8,0,2,8,0]`；
- Kd：`[0.1,0.8,0.25,0.1,0.8,0.25]`；
- 1～6 号控制频率：200 Hz；策略频率：50 Hz；
- `offline_threshold=600`，约 3 s @ 200 Hz；
- 默认 height command：0.24 m；
- 推理 `dt=0.0025 s`、`decimation=8`；
- IMU 到 base_link：`[x,y,z] -> [-x,-y,+z]`。

所有运行值以仓库 YAML 为最终依据，详见 [`interface.md`](interface.md)。

## 7. 安全限制

1. 首次运行必须架空或可靠固定，急停可触达，运动范围内无人。
2. 启动前核对电源电压、限流、终端电阻、电机 ID/型号和 `can0` 状态。
3. 未确认默认位、方向、关节限位、PD 和模型 hash 前禁止进入 policy。
4. 扫频和正常部署不得同时运行；任一时刻只能有一个 CAN 控制源。
5. 翼启动定位只等待软件目标走完，不代表实际机械到位已验证；正常退出使用上+A。
6. CAN TX/RX 不对称、bus-off、tx buffer full、电机离线、异常振动或过热时立即急停。
7. 禁止只提高 `offline_threshold` 掩盖通信丢包。
8. 未完成架空、台架和低速验收前，不得进入自由地面高速测试。

## 8. 已知问题和后续工作

| 问题 | 影响 | 处理/后续 |
|---|---|---|
| 多电机推理时曾出现 offline/tx buffer full | 可能触发全 CAN 安全态 | 使用 `candump`、示波器、CAN 负载工具和 `ip -s link` 定位供电/布线/负载/驱动问题 |
| MNN 与训练合同可能漂移 | 输出错误或限位裁切 | 每次换模型必须做 observation/action、PD、scale、clip、delay、坐标 parity |
| build/install/log 属于本机构建产物 | 不具备跨机可追溯性 | 新机器重新构建；正式日志另建索引和 SHA-256 |
| 真机完整验收未由 Git 自动证明 | 不能仅凭代码声明安全 | 补充架空、台架、低速、完整场景记录和视频签字 |

## 9. 交付验收清单

- [x] 仓库、分支、功能冻结基线和负责人明确；
- [x] README、交接、接口、测试和 CHANGELOG 已提供；
- [x] 环境安装、构建、启动和扫频命令已提供；
- [x] 输入输出、坐标、频率、参数和单位已说明；
- [x] MNN 路径和 SHA-256 已记录；
- [x] 真机安全、急停和异常退出条件已说明；
- [ ] 接收人在目标硬件完成环境和构建验证；
- [ ] 架空、台架、低速和完整运行验收完成；
- [ ] 真机视频、日志、CAN 统计和验收表完成归档。

## 10. 交接信息

- 交接人：汪成浩
- 接收人：待填写
- 交接日期：2026-08-04
- 备注：本次提交只增加归档文档，不修改冻结运行代码、配置、模型或采集数据。
