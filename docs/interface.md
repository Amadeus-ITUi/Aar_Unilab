# WE11 UniLab 接口说明

负责人：汪成浩
适用分支：`Walking_Eagle-Unilab_final`

## 1. 数据流

```text
MuJoCo state + command + history + wing observation
                         |
                         v
                  actor observation
                         |
                         v
                    PPO policy
                         |
                         v
             6D raw action -> scale/clip
                         |
                         v
          shared command FIFO -> 200 Hz PD
                         |
                         v
                  400 Hz MuJoCo
```

## 2. 关节、坐标和单位

| 项目 | 定义 |
|---|---|
| 关节顺序 | `[left_thigh, left_calf, left_wheel, right_thigh, right_calf, right_wheel]` |
| 位置/角度 | rad |
| 角速度 | rad/s |
| 力矩 | N·m |
| 线速度 | m/s |
| 角速度 command | rad/s |
| 高度 command | m |
| wrench | `[Fx,Fy,Fz,Mx,My,Mz]`，施加前按数据合同转换到 base/push-site 坐标 |

IMU、gyro、关节正方向和默认位必须与 WE11 MJCF 和 Deploy 配置逐项核对，不得仅按字段名称推断轴向。

## 3. 策略输入

| 输入 | 形状/频率 | 来源 |
|---|---|---|
| actor observation | 当前 145D，50 Hz | 29D 单帧 × 5，term-major；本体状态、command、历史动作、翼角/翼速等 |
| rough critic observation | 334D，训练时使用 | actor 信息加特权物理/地形/外力信息 |
| velocity/height command | 3D | 环境 command curriculum |
| measured wrench | 0/1/2/3 Hz CSV | WE11 `training_data` |
| wing position | 与 wrench 独立缩放的 CSV 观测 | WE11 `training_data` |

观测维度或排列变化后旧 checkpoint 视为不兼容，必须重新训练或提供显式迁移程序。

## 4. 策略输出与控制

| 输出 | 形状 | 处理 |
|---|---|---|
| raw action | 6D | 腿用于相对位置目标，轮用于速度目标 |
| 腿 action | indices `[0,1,3,4]` | scale `0.5` 后进入 command FIFO/PD |
| 轮 action | indices `[2,5]` | scale `10.0`，raw clip `±3.5` |

固定时序：

- physics：400 Hz，`dt=0.0025 s`；
- motor PD：200 Hz，ZOH 两个 physics step；
- policy：50 Hz，decimation 8；
- shared command-delay：`2..8` 个 200 Hz motor tick，即 `10..40 ms`。

PD：

```text
tau = Kp * (q_target_delayed - q) - Kd * qd
```

Kp/Kd：

```text
Kp = [2.0, 7.59, 0.0, 2.0, 7.59, 0.0]
Kd = [0.080, 0.682, 0.05, 0.080, 0.682, 0.05]
```

## 5. 配置入口

| 配置 | 路径 |
|---|---|
| Flat PPO/控制/随机化 | `conf/ppo/task/dr002_joystick_flat_we11/base.yaml` |
| Rough 覆盖项 | `conf/ppo/task/dr002_joystick_rough_we11/` |
| WE11 资产 | `src/unilab/assets/robots/dr002/we11/` |
| Flat 环境 | `src/unilab/envs/locomotion/dr002/joystick.py` |
| Rough 环境 | `src/unilab/envs/locomotion/dr002/rough.py` |
| native command-delay PD | `third_party/mujoco_uni_mixed_pd/` |

## 6. 文件输出

| 输出 | 默认位置 |
|---|---|
| checkpoint | `logs/rsl_rl_ppo/<TASK>/<RUN>/model_<N>.pt` |
| TensorBoard event | 同一 run 目录 |
| 视频/回放 | 由 play/eval 参数确定；生成物不跟踪 Git |
| PACE/KpKd 结果 | 命令指定的 output 目录；需另附 manifest/SHA-256 |

## 7. 跨仓库合同

向 Play/Deploy 交付模型时必须同时冻结：模型 hash、actor 输入维度、动作维度、关节顺序、默认位、PD、action scale/clip、delay、控制频率、IMU/翼角/wrench 坐标和 command 定义。任何一项不一致都不得直接上机。
