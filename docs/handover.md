# Walking Eagle WE11 UniLab 算法交接说明

## 1. 当前版本

| 项目 | 内容 |
|---|---|
| 项目 | Walking Eagle WE11 UniLab 训练与仿真 |
| 负责人 | 熊铭煊 |
| 交付日期 | 2026-08-31 |
| 仓库 | `https://git.esdyn.cn/pheonix-eagle/sar_unilab.git` |
| 分支 | `pheonix-we11/unilab-bringup` |
| 交付目的 | 实现手动控制上肢机翼扑打角度的新需求 |
| 功能冻结基线 | `532646089dab4f174f7cdc7374190c11909b5fd2` |
| 归档提交 | 以该分支最新 `HEAD` 为准，使用 `git rev-parse HEAD` 获取 |
| 当前状态 | WE11 MuJoCo flat/rough 可训练、可评估；腿部收拢状态下的倒地自启已完成仿真验证，前倾和后倾状态下的倒地自启正在实现；不代表真机验收通过 |
| 适用平台 | Walking Eagle WE11 |

## 2. 交接范围与责任边界

本仓库包含：

- WE11 URDF/MJCF、mesh、碰撞体、质量/惯量和 PACE 参数；
- MuJoCo flat 与 rough 环境；
- RSL-RL PPO、actor/critic 网络和训练包装器；
- 观测、动作、Reward、课程学习、域随机化和 command-delay；
- 实测 wrench 与翼角回放数据；
- 训练、续训、评估、TensorBoard、资产可视化和测试入口；
- PACE/Kp/Kd 离线拟合脚本与数据合同。

本仓库不包含：

- ROS2/CAN 真机驱动和电机固件；
- rl_sar 手柄回放 runtime；
- CAD 原始设计与机械加工文件；
- 未经筛选的其他机器人资产、其他算法和 IsaacLab 训练链路；
- checkpoint、日志、视频和本机构建的 native `.so`。

负责人承担训练、仿真、资产和接口合同的交付，不代替硬件、电气、底层驱动或最终真机安全验收。

## 3. 核心技术路线

```text
WE11 URDF/MJCF + 实机扫频数据
              |
              v
       PACE 与 Kp/Kd 拟合
              |
              v
 MuJoCo flat/rough + 观测/动作/Reward/随机化
              |
              v
         RSL-RL PPO 训练
              |
              v
 checkpoint -> 导出模型 -> rl_sar -> ROS2/CAN
```

本仓库负责到 checkpoint/策略导出接口为止；回放和真机执行由对应最终仓库负责。

## 4. 环境与安装

| 项目 | 要求 |
|---|---|
| 操作系统 | Ubuntu 22.04 Linux |
| Python | `>=3.10,<3.14` |
| 训练 | PyTorch 2.7.0、RSL-RL PPO |
| 仿真 | mujoco-uni 3.8.0、MuJoCo RK4 |
| GPU | CUDA 12.8 训练环境；CPU 可用于部分静态测试 |

按仓库当前冻结版本安装：

```bash
git clone -b pheonix-we11/unilab-bringup \
  https://git.esdyn.cn/pheonix-eagle/sar_unilab.git sar_unilab
cd sar_unilab
bash install_conda_environment.txt unilab_cuda cu128
conda activate unilab_cuda
```

安装脚本使用 Conda + pip，并自动构建、加载验证 native PD，运行非 slow 测试和
Flat/Rough 最小 PPO。native `.so` 与 Python ABI、CPU 架构绑定，必须在目标机器
本地构建，不提交 Git。完整说明见 `WE11_CONDA_INSTALL_GUIDE.txt`。

## 5. 关键文件

| 路径 | 作用 | 必须 |
|---|---|---|
| `conf/ppo/task/dr002_joystick_flat_we11/` | Flat 网络、PD、观测、动作、Reward、随机化 | 是 |
| `conf/ppo/task/dr002_joystick_rough_we11/` | Rough 地形与课程配置 | 是 |
| `src/unilab/assets/robots/dr002/we11/` | WE11 模型、mesh、PACE 与回放数据 | 是 |
| `src/unilab/envs/locomotion/dr002/joystick.py` | Flat 环境核心逻辑 | 是 |
| `src/unilab/envs/locomotion/dr002/rough.py` | Rough 环境和地形课程 | 是 |
| `scripts/train_rsl_rl.py` | 训练、续训和 checkpoint 回放入口 | 是 |
| `scripts/pace/` | 离线 PACE/Kp/Kd 拟合 | 是 |
| `third_party/mujoco_uni_mixed_pd/` | native command-delay PD 构建源 | 是 |
| `install_conda_environment.txt` | Conda/pip 一键安装与实跑验证 | 是 |
| `WE11_CONDA_INSTALL_GUIDE.txt` | 安装、使用、迁移和排障说明 | 是 |
| `models/we11/` | Flat/Rough 最终 checkpoint 和 SHA-256 | 是 |
| `tests/` | 最小验证与合同测试 | 是 |

## 6. 从零运行

Flat 训练：

```bash
python -u scripts/train_rsl_rl.py \
  task=dr002_joystick_flat_we11/mujoco \
  training.device=cuda:0 \
  training.no_play=true \
  training.play_render_mode=none
```

Rough 训练：

```bash
python -u scripts/train_rsl_rl.py \
  task=dr002_joystick_rough_we11/mujoco \
  training.device=cuda:0 \
  training.no_play=true \
  training.play_render_mode=none
```

TensorBoard：

```bash
tensorboard --logdir logs/rsl_rl_ppo --port 6006 --reload_interval 5
```

最小验证：

```bash
python -m pytest -q \
  tests/envs/locomotion/dr002/test_we11_pace_training.py \
  tests/envs/locomotion/dr002/test_we11_rough_training.py \
  tests/envs/locomotion/common/test_terrain_spawn.py \
  tests/terrains/test_mujoco_heightfield.py
```

## 7. 当前关键合同

- 关节顺序：`[左大腿, 左小腿, 左轮, 右大腿, 右小腿, 右轮]`；
- 物理/电机 PD/策略频率：`400/200/50 Hz`；
- MuJoCo 积分器：RK4；
- Kp：`[2.0,7.59,0,2.0,7.59,0]`；
- Kd：`[0.08,0.682,0.05,0.08,0.682,0.05]`；
- 腿 action scale：`0.5`；轮 action scale/raw clip：`10/±3.5`；
- shared command FIFO：每次 reset 随机 `2..8` 个 200 Hz tick；
- 本节冻结归档 checkpoint 的 actor/rough critic：`135/334` 维；当前正式 Flat `model_499.pt` 发布契约为 actor `145` 维，不与该旧归档混用；
- 冻结基线实测回放等级：`0/1/2/3 Hz`，4 Hz 不在该提交中。

完整输入输出和单位见 [`interface.md`](interface.md)。

## 8. 模型和实验结果

Git 已跟踪两个选定的最终 checkpoint：

| 模型 | 位置/用途 | 状态 |
|---|---|---|
| Flat | `models/we11/flat/model_1500.pt` | 来源 `2026-07-30_21-41-17_mujoco`；SHA-256 见模型 manifest |
| Rough | `models/we11/rough/model_1500.pt` | 来源 `2026-07-31_13-01-10_mujoco`；对应 Play rough ONNX |

关键 PACE/Kp/Kd 参数已固化在 WE11 配置和资产中；原始训练日志、TensorBoard 曲线和视频应在交付介质中另行冻结，并记录 SHA-256。

## 9. 安全边界

- 本仓库训练/评估命令只授权仿真运行，不可直接控制真机；
- 导出模型进入真机前，必须验证 observation/action 维度、关节顺序、坐标系、PD、scale、clip、delay 和控制频率一致；
- 旧模型不得加载到观测维度或语义已变化的任务；
- 真机首次运行必须架空、低速、低力矩，确认急停和通信超时保护；
- 不得用提高限幅或离线阈值掩盖模型/通信错误。

## 10. 已知问题和后续工作

| 项目 | 当前状态 | 后续动作 |
|---|---|---|
| 其他 checkpoint/日志/视频 | 不纳入 Git | 生成独立交付介质、索引和 SHA-256；两个最终 checkpoint 已入库 |
| 固定评估矩阵 | 尚未形成统一一键报告 | 固定 terrain/force/amplitude/command，比较 success、tracking、clip、torque 等指标 |
| 真机验收 | 不属于本仓库完成状态 | 由 Deploy 链路完成架空、台架、低速和完整场景验收 |
| native PD | 本机编译产物不提交 | 在每台目标机器重新构建并运行可用性检查 |

## 11. 交付验收清单

- [x] 仓库、分支和功能冻结基线明确；
- [x] README、交接、接口、测试和 CHANGELOG 已提供；
- [x] 环境依赖、训练和最小测试命令已说明；
- [x] WE11 资产、配置、示例数据和测试均在仓库中；
- [x] 关键参数、单位、频率和坐标合同已说明；
- [x] Flat/Rough 最终 checkpoint、来源和 SHA-256 已入库；
- [x] 仿真/真机安全边界已说明；
- [ ] 接收人在目标机器完成环境安装和最小训练验证；
- [ ] 最终模型、日志、曲线和视频完成离线介质归档；
- [ ] 真机低风险与完整场景验收签字。

## 12. 交接信息

- 交接人：熊铭煊
- 接收人：待填写
- 交接日期：2026-08-31
- 备注：本次交付目的是实现手动控制上肢机翼扑打角度；腿部收拢状态下的倒地自启已完成仿真验证，前倾和后倾状态下的倒地自启正在实现。
