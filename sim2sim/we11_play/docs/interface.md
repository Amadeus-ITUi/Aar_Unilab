# Walking Eagle WE11 Play 接口说明

负责人：汪成浩
适用分支：`Walking_Eagle-Play_final`

## 1. Launcher 接口

```text
./scripts/play_we11.sh <frequency_hz> <flat|getup> <wrench_scale> [--difficulty 0..1 | --balance-difficulty 0..1]
frequency_hz = 0 | 1 | 2 | 3
启动场景 = flat（原站立姿势）| getup（机械限位倒地姿势）
wrench_scale = 浮点倍率，默认 1.0
difficulty = 连续 Getup 课程姿态；0 为 home，1 为完整倒地姿态
balance-difficulty = 动态平衡恢复；0 为 home 静止，1 为最大倾角/角速度扰动
```

两个启动场景共用 `policy/dr002/we11/policy.onnx` 和平地 XML，只改变 reset 姿势。
`--difficulty` 使用和训练相同的姿态插值，并优先于 `flat/getup`。
旧格式 `./scripts/play_we11.sh <frequency_hz> <wrench_scale>` 继续兼容。

```text
./scripts/play_we11_level9.sh <terrain> <frequency_hz> <wrench_scale>
terrain = flat | uphill | downhill | rough
frequency_hz = 0 | 1 | 2 | 3
wrench_scale = 浮点倍率，默认 1.0
```

可选环境变量：

| 变量 | 作用 | 使用边界 |
|---|---|---|
| `RL_SAR_JOYSTICK` | 指定 `/dev/input/jsN` | 正常手柄回放可用 |
| `RL_SAR_PLAY_AUTOSTART=1` | 无手柄自动进入 policy | 只用于仿真 smoke |
| `RL_SAR_WE11_START_PROFILE=flat|getup` | 未传启动场景参数时选择 reset 姿势 | 命令行参数优先 |
| `RL_SAR_WE11_GETUP_DIFFICULTY=0..1` | 未传 `--difficulty` 时指定连续课程姿态 | 命令行参数优先 |
| `RL_SAR_WE11_BALANCE_DIFFICULTY=0..1` | 未传 `--balance-difficulty` 时指定动态平衡扰动 | 与 Getup difficulty 互斥 |
| `RL_SAR_BUILD_JOBS` | 限制构建线程数 | 构建时使用 |

## 2. 策略接口

| 项目 | 定义 |
|---|---|
| ONNX 输入 | `obs[1,145]`，float32；29D 单帧 × 5，term-major |
| ONNX 输出 | `action[1,6]`，float32 |
| 关节顺序 | `[left_thigh,left_calf,left_wheel,right_thigh,right_calf,right_wheel]` |
| policy | 50 Hz |
| motor PD | 200 Hz |
| physics | 400 Hz，RK4 |

策略输出处理：

```text
action scale = [0.5, 0.5, 10, 0.5, 0.5, 10]
wheel raw clip = +/-3.5
Kp = [2, 7.59, 0, 2, 7.59, 0]
Kd = [0.08, 0.682, 0.05, 0.08, 0.682, 0.05]
```

## 3. 实测数据接口

| 数据 | 路径 | 含义 |
|---|---|---|
| wrench | `replay_data/we11/wrench/{1,2,3}hz.csv` | `[Fx,Fy,Fz,Mx,My,Mz]`，统一乘 `wrench_scale` |
| wing angle | `replay_data/we11/wing_angle/{1,2,3}hz.csv` | 注入 actor 对应翼角观测 |
| 0 Hz | 不读取实测 CSV | 无 CSV 外力和翼角注入 |

wrench 在 `push_site` 施加，使用固定 sensor-to-base 旋转。力和力矩必须使用同一个正确坐标变换；不得只旋转力而遗漏力矩。

## 4. 地形接口

| terrain | scene |
|---|---|
| flat | `scene_level9_noise006_flat_we11.xml` |
| uphill | `scene_level9_noise006_uphill_we11.xml` |
| downhill | `scene_level9_noise006_downhill_we11.xml` |
| rough | `scene_level9_noise006_rough_we11.xml` |

Level-9 heightfield 固定 seed 42，`noise_range=[0,0.006] m`、`noise_step=0.001 m`。

## 5. 输出

- MuJoCo GUI 中的机器人状态和地形；
- 终端中的 command、policy action、PD/力矩和限幅诊断；
- 进程退出码与依赖缺失报错。

本仓库不发布 ROS topic，也不发送 CAN 帧。

## 6. 跨仓库模型交付

替换策略时必须同步更新：

1. `policy_model_*.onnx`；
2. `deployment_manifest.json` 与 `SOURCE.md`；
3. 观测/动作维度和顺序；
4. WE11 selector 的 PD、scale、clip、delay 和 command；
5. `SHA256SUMS`；
6. flat/rough smoke 和 Deploy parity 记录。
> 历史接口记录：频率、wrench 和翼角回放接口已从活动启动器删除。
> 当前接口见 `../README.md`。
