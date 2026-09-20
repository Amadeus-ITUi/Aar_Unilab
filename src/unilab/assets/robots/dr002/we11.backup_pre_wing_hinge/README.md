# WE11 identified MuJoCo asset

WE11 的形态、惯量、碰撞体和传感器来自已审核的新资产；运行模型使用 RK4，
腿部 PACE 动力学采用 2026-07-30 实机扫频辨识结果。本目录包含运行所需的
MJCF、visual 网格和 URDF，不依赖 WE9/WE10 资产目录。

- 关节顺序：`[左大腿, 左小腿, 左轮, 右大腿, 右小腿, 右轮]`
- Armature:
  `[0.0045092746608505355, 0.0056654868268008396, 0.0008, 0.0045092746608505355, 0.0056654868268008396, 0.0008]`
- Damping:
  `[2.3658001235049574e-06, 1.7025403002922656e-05, 0, 2.3658001235049574e-06, 1.7025403002922656e-05, 0]`
- Frictionloss:
  `[9.003633786813792e-06, 0.20614840564125578, 0, 9.003633786813792e-06, 0.20614840564125578, 0]`
- Kp: `[2.0, 7.59, 0, 2.0, 7.59, 0]`
- Kd: `[0.080, 0.682, 0.05, 0.080, 0.682, 0.05]`

轮子使用 `Kd=0.05`、action scale `10.0` 和 raw clip `±3.5`。辨识时使用固定
4 步 command delay；训练使用每环境共享、reset 随机 2–8 步 command FIFO。

- `we11.xml`: 独立 WE11 MuJoCo 运行模型。
- `scene_flat_we11.xml`: 平地场景。
- `rough_locomotion_task.xml`: rough 环境注入片段。
- `meshes_lod/`: WE11 自包含 visual 网格。
- `urdf/we11_reviewed.urdf`: 与运行模型物理属性对应的审核版 URDF。
- `urdf/we11_source.urdf`: 新 CAD 导出的原始 URDF 归档。
- `training_data/measured_wrench_20260728_skin/`: WE11 训练使用的 1/2/3 Hz 六维力数据。
- `training_data/wing_angle_20260713/`: 与力相位对齐的 1/2/3 Hz 翼角观测。
