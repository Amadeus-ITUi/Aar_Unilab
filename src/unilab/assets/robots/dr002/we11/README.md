# WE11 third-version MuJoCo asset

WE11 的形态、惯量、碰撞体和传感器基于新版 CAD + 旧版审核参数合成。
运行模型使用 RK4，腿部 PACE 动力学采用 2026-07-30 实机扫频辨识结果。
本目录包含运行所需的 MJCF、visual 网格和 URDF，不依赖 WE9/WE10 资产目录。

## 与旧版 (we11-origin) 的关键差异

- **上肢**：机翼从旧版 `ancient_link` 死重拆出为**两个可动 hinge**：
  `left_wing_joint` / `right_wing_joint`
  - 零位 = 17.5° 上抬（一字型），limit = [-π/2, 0]（水平位为上限、向上翻 90° 为下限）
  - 两翼 axis 互反：同 joint 值 → 翼面镜像对称
- **尾翼**：U1_link/U2_link 保留新版 mesh 几何，固定连接 base_link
- **base_link**：inertial 抄新版 CAD (mass=1.7974)，collision 抄旧版简化 box + 新增 shoulder box
- **腿部**：惯量、armature、damping、frictionloss 沿用旧版 reviewed 值；calf
  限位按实机膝关节内夹角 `35.47°–106.33°` 标定为
  `[-2.522524368, -1.285784060] rad`

## 关节顺序

- 腿部：`[左大腿, 左小腿, 左轮, 右大腿, 右小腿, 右轮]`（同旧版）
- 机翼：`[左翼, 右翼]`（追加在腿部之后）

## Actuator 顺序

- 腿部 6 motor（同旧版）：`[thigh, calf, foot] × [L, R]`，ctrlrange `±5.5/±14/±5.5`
- 机翼 2 velocity actuator：`[left_wing_vel, right_wing_vel]`，kv=0.5，ctrlrange `±10 rad/s`

## PACE 参数（腿部，同旧版）

- Armature: `[0.0045092746608505355, 0.0056654868268008396, 0.0008, 0.0045092746608505355, 0.0056654868268008396, 0.0008]`
- Damping: `[2.3658001235049574e-06, 1.7025403002922656e-05, 0, 2.3658001235049574e-06, 1.7025403002922656e-05, 0]`
- Frictionloss: `[9.003633786813792e-06, 0.20614840564125578, 0, 9.003633786813792e-06, 0.20614840564125578, 0]`
- Kp: `[2.0, 7.59, 0, 2.0, 7.59, 0]`
- Kd: `[0.080, 0.682, 0.05, 0.080, 0.682, 0.05]`

轮子使用 `Kd=0.05`、action scale `10.0` 和 raw clip `±3.5`。辨识时使用固定
4 步 command delay；训练使用每环境共享、reset 随机 2–8 步 command FIFO。

## 机翼参数（估值，未做 PACE 辨识）

沿用大腿关节值：`effort=5.5, velocity=21.0, damping=6.926e-5, frictionloss=3.928e-4, armature=4.509e-3`。
后续如需精确匹配实机机翼动力学，请自行辨识。

## 文件说明

- `we11.xml`: 独立 WE11 MuJoCo 运行模型（nq=15, nu=8）
- `scene_flat_we11.xml`: 平地场景，含 keyframe `home`
- `scene_getup_alignment_we11.xml`: 独立的起立初始姿态对齐场景，不被 Flat/Rough 训练任务引用
- `getup_pose_bank_v3.npz`: 统一的 typed reset 姿态库，包含 `home`、`getup` 锚点，
  `home_to_getup`、`getup_to_front` 过渡，以及 `front`、`back` 离线碰撞验证姿态；
  机翼角不参与姿态身份，使用时独立随机
- `getup_pose_bank_v3.json`: v3 姿态库的场景哈希、类型数量和生成参数
- `rough_locomotion_task.xml`: rough 环境注入片段
- `meshes_lod/`: WE11 自包含 visual 网格（新版 STL + 旧版 base_link 5 分片 LOD）
- `urdf/we11_reviewed.urdf`: 与运行模型物理属性对应的审核版 URDF
- `urdf/we11_source.urdf`: 新 CAD 导出的原始 URDF 归档
- `we11_pace_params.json`: PACE 辨识参数（腿部 6 关节，与旧版完全一致）
- `training_data/measured_wrench_20260728_skin/`: 六维力数据（同旧版）
- `training_data/wing_angle_20260713/`: 翼角观测数据（同旧版）

姿态库可通过统一管理器按类型查看：

```bash
python scripts/manage_we11_getup_pose_bank.py --pose-type home
python scripts/manage_we11_getup_pose_bank.py --pose-type home_to_getup
python scripts/manage_we11_getup_pose_bank.py --pose-type getup
python scripts/manage_we11_getup_pose_bank.py --pose-type getup_to_front
python scripts/manage_we11_getup_pose_bank.py --pose-type front
python scripts/manage_we11_getup_pose_bank.py --pose-type back
```

旧参数 `--family` 仍可使用，但新命令统一推荐 `--pose-type`。
增加 `--simulate-passive` 可让 MuJoCo 以零 actuator command 推进，用于观察仅靠
MJCF damping/friction 的保持能力；窗口内按 `S` 可随时切换冻结/被动物理。

Getup 训练在姿态库课程前先运行 `balance_recovery`。该阶段始终从精确 `home`
开始，pitch 与 pitch-rate 是 reset 时独立施加的训练扰动，不会写入或改动 v3
姿态库。可使用以下命令查看最大难度：

```bash
python scripts/visualize_task_env.py \
  --task DR002JoystickGetupWE11 \
  --balance-difficulty 1.0 \
  --num_envs 2
```
