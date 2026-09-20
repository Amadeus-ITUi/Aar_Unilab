# Walking Eagle WE11 Play 算法交接说明

## 1. 当前版本

| 项目 | 内容 |
|---|---|
| 项目 | Walking Eagle WE11 rl_sar MuJoCo 手柄回放 |
| 负责人 | 熊铭煊 |
| 交付日期 | 2026-08-31 |
| 仓库 | `https://git.esdyn.cn/pheonix-eagle/sar_unilab.git` |
| 分支 | `pheonix-we11/play-bringup` |
| 交付目的 | 实现手动控制上肢机翼扑打角度的新需求 |
| 功能冻结基线 | `2a42994681e9ead46e0051fc54b1661851b18ee7` |
| 归档提交 | 以该分支最新 `HEAD` 为准，使用 `git rev-parse HEAD` 获取 |
| 当前状态 | WE11 flat 与 level-9 rough 可独立回放；腿部收拢状态下的倒地自启已完成仿真验证，前倾和后倾状态下的倒地自启正在实现；仅用于仿真，不可控制真机 |
| 适用平台 | Ubuntu 22.04 x86_64 |

## 2. 交接范围与责任边界

本仓库包含：

- WE11 当前 `policy.onnx` 策略；
- WE11 MJCF、LOD mesh、flat 和 level-9 地形；
- 0/1/2/3 Hz wrench/翼角回放数据；
- 400 Hz RK4 MuJoCo、200 Hz PD、50 Hz policy 回放合同；
- MuJoCo、GLFW、ONNX Runtime x86_64 运行依赖；
- flat/rough 手柄 launcher、构建脚本和 SHA-256 清单。

本仓库不包含：

- UniLab 训练、续训、PACE/Kp/Kd 拟合；
- ROS2/CAN 真机控制；
- 其他机器人和其他策略；
- checkpoint `.pt`、训练日志和 TensorBoard event；
- 真机安全认证。

## 3. 环境与构建

| 项目 | 要求 |
|---|---|
| 操作系统 | 推荐 Ubuntu 22.04 |
| 架构 | x86_64；仓库内 runtime 与该架构绑定 |
| 编译 | CMake、C++17、build-essential |
| 系统库 | yaml-cpp、TBB、X11/OpenGL 等 |
| 输入 | Linux joystick `/dev/input/js0..js3` |

```bash
git clone -b pheonix-we11/play-bringup \
  https://git.esdyn.cn/pheonix-eagle/sar_unilab.git Play
cd Play
chmod +x build.sh scripts/play_we11.sh scripts/play_we11_level9.sh
./build.sh
```

构建产物：`build/bin/rl_sim_mujoco`。

## 4. 关键文件

| 路径 | 作用 | 必须 |
|---|---|---|
| `policy/dr002/we11/policy.onnx` | 145D -> 6D WE11 策略 | 是 |
| `policy/dr002/we11/config.yaml` | WE11 selector 与控制合同 | 是 |
| `policy/dr002/base.yaml` | 共享时序和关节元数据 | 是 |
| `src/rl_sar_zoo/dr002_description/mjcf/we11/` | MJCF、mesh、flat/level-9 scenes | 是 |
| `replay_data/we11/` | 1/2/3 Hz wrench 与翼角数据 | 是 |
| `scripts/play_we11.sh` | Flat 回放入口 | 是 |
| `scripts/play_we11_level9.sh` | Level-9 回放入口 | 是 |
| `SHA256SUMS` | 关键交付文件完整性校验 | 是 |

模型来源以 `policy/dr002/we11/SOURCE.md` 为准。

```text
ONNX: policy/dr002/we11/policy.onnx
SHA-256: 6256ef725922ff6e6621debda8287988e87fbcb07e500d021cfe7777fdefc1d9
interface: obs[1,145] -> action[1,6]
```

## 5. 从零运行

Flat、无 CSV 外力：

```bash
./scripts/play_we11.sh 0
```

Flat、1/2/3 Hz 实测回放：

```bash
./scripts/play_we11.sh 1
./scripts/play_we11.sh 2
./scripts/play_we11.sh 3 1.0
```

Level-9 rough：

```bash
./scripts/play_we11_level9.sh rough 3 1.0
```

同一地形资产还支持 `flat/uphill/downhill`。没有手柄时，仅做自动化 smoke 可使用：

```bash
RL_SAR_PLAY_AUTOSTART=1 ./scripts/play_we11.sh 0
```

正常手柄回放不要设置 `RL_SAR_PLAY_AUTOSTART=1`。

## 6. 当前回放合同

- 关节顺序：`[左大腿, 左小腿, 左轮, 右大腿, 右小腿, 右轮]`；
- MuJoCo：400 Hz、RK4；
- motor PD：200 Hz；policy：50 Hz；
- Kp：`[2,7.59,0,2,7.59,0]`；
- Kd：`[0.08,0.682,0.05,0.08,0.682,0.05]`；
- action scale：`[0.5,0.5,10,0.5,0.5,10]`；
- wheel raw action clip：`±3.5`；
- shared command-delay：每次 reset 随机 `2..8` 个 200 Hz tick；
- 默认 command：`[0,0,0.24]`；
- wrench/翼角频率：`0/1/2/3 Hz`。

详细接口见 [`interface.md`](interface.md)。

## 7. 最小验证

```bash
sha256sum -c SHA256SUMS
bash -n build.sh scripts/play_we11.sh scripts/play_we11_level9.sh
./build.sh
```

完成构建后再运行 flat 0 Hz 与 rough 3 Hz smoke。无显示设备的机器应使用适合 MuJoCo 的离屏配置；不要把无 GUI 误判为模型错误。

## 8. 安全边界

- 本仓库不含 ROS2、SocketCAN 或真机电机控制入口；
- 回放输出不得绕过 Deploy 的 parity、限幅、急停和低风险验证直接接入真机；
- 更换 ONNX、MJCF、PD、scale、clip、delay 或观测排列后必须更新 `SHA256SUMS` 并重新做 parity；
- launcher 只接受文档列出的频率和地形，不得把缺失 CSV 静默当作有效回放；
- rough policy 的成功回放不等于真机 rough 验收通过。

## 9. 已知问题和后续工作

| 项目 | 当前状态 | 后续动作 |
|---|---|---|
| runtime 架构 | 随仓库的是 x86_64 | aarch64 需重新准备并校验依赖 |
| 翼角 | 注入策略观测，但 WE11 MJCF 无独立可视翼铰链 | 如需可视翼运动，需扩展资产并重新验证动力学 |
| 模型 | 当前 `policy.onnx` 的来源见 `SOURCE.md` | 新模型必须附来源 run、checkpoint 和 hash |
| 真机 parity | 不在本仓库执行 | 在 Deploy 仓库完成逐帧/golden frame 与安全验收 |

## 10. 交付验收清单

- [x] 仓库、分支、功能冻结基线和负责人明确；
- [x] README、交接、接口、测试和 CHANGELOG 已提供；
- [x] 构建与 flat/rough 运行命令已说明；
- [x] ONNX、MJCF、数据和 runtime 依赖闭包已归档；
- [x] `SHA256SUMS` 可校验关键文件；
- [x] 观测、动作、PD、scale、clip、delay 和频率合同已说明；
- [x] 仿真与真机边界已说明；
- [ ] 接收人在目标机完成构建和两类 smoke；
- [ ] 新模型进入真机前完成 Deploy parity 与安全验收。

## 11. 交接信息

- 交接人：熊铭煊
- 接收人：待填写
- 交接日期：2026-08-31
- 备注：本次交付目的是实现手动控制上肢机翼扑打角度；腿部收拢状态下的倒地自启已完成仿真验证，前倾和后倾状态下的倒地自启正在实现。
> 历史交接记录：频率、wrench/翼角 CSV 和 Level-9 启动器不再属于活动回放。
> 当前使用方法见 `../README.md`。
