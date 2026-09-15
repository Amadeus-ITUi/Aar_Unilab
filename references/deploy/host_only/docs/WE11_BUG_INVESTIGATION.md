# WE11 "翼上抬 → 前进" Bug 现场调查清单

## 背景

真机上 lab_inference_node 出现 bug：翼从水平向上抬起时机器人开始前移，翼水平时无移动趋势。仿真回放没有复现。

本地代码已经应用以下修改（**主机侧**，`/ssd/Pheonix/Deploy`）：

- **P1** 观测契约 27D→29D，加入 wing_vel 段（`lab_deploy_utils.hpp`、`lab_inference_node.cpp`、`test/lab_deploy_utils_test.cpp`）
- **P5** wing PD 从 kp=20,kd=1.0 → **kp=10, kd=0.5** 匹配训练/Play（`wing_motors.yaml`）
- **P6** 腿电机 Kp/Kd 从 `[2.0,8.0,0.0]/[0.1,0.8,0.25]` → **`[2.0,7.59,0.0]/[0.08,0.682,0.05]`**（`motors.yaml`）
- **P9** 离线对比工具全部标 STALE 警告（`offline_debug_replay_onnx.cpp`、`offline_inference_check.cpp`、`compare_sim_vs_inference.py`）
- **P2** `lab_policy.mnn` 从 135D 换成 145D 新契约版本（来源 `Play/policy/dr002/we11/policy.onnx`，SHA-256 `5ce669a4eca8ba0c917a2e89abaa608f96df2d70809914bf458c44ce3cbb5ef2`）

**还未收敛、必须现场验证**的四项：P3（翼 sign/swap）、P4（翼软限位）、P7（腿 default_angle）、P8（IMU sign）。

---

## 现场测试步骤

### 前置

- 样机架空、急停可达
- `joy_policy_gate_enabled=true` 保持 standby，需要时才切 policy
- `log_policy_outputs=true`, `policy_log_interval_ticks=1` 打开逐帧日志

### T1  验证 mnn 输入维度（回答 CR-1）

**目的**：确认 Deploy 上加载的 mnn 是 145 D，不是 135 D。

```bash
# 在树莓派上
python3 -c "
import MNN
interp = MNN.Interpreter('/home/esd/Pheonix/Deploy/src/inference/models/lab_policy.mnn')
sess = interp.createSession()
t = interp.getSessionInput(sess)
print('input shape =', t.getShape())
t2 = interp.getSessionOutput(sess)
print('output shape =', t2.getShape())
"
```

**期望**：`input shape = (1, 145)`, `output shape = (1, 6)`。若不是，检查 mnn 文件是否覆盖成功、install space 是否符号链接到源码。

启动 `lab_inference_node` 后看日志开头有 `input=145`，也是一样。

---

### T2  Wing sign/swap 验证（回答 CR-2，决定 P3 落地公式）

**目的**：确认 motor_7 / motor_8 分别对应物理左翼还是右翼，以及编码器方向。

在 Deploy 树莓派 shell 里：

```bash
ros2 topic echo /policy/wing_angles --field position
```

分四步操作（每次记录 position[0] 和 position[1] 的变化方向和幅度）：

1. **静止 keyframe**：机器人开机、翼电机没抬起前的位置。填表格 A。
2. **手动扭动物理左翼**（面朝机头方向站，机身左侧的翼）向下（下拍方向），position 是升是降？填表格 B。
3. **手动扭动物理右翼**（机身右侧）向下（下拍方向），position 是升是降？填表格 C。
4. **两翼同步向下拍到极限**，两个 position 的数值区间。填表格 D。

**当前 Deploy 假设**（`lab_deploy_utils.hpp:motor_wing_angles_to_training_obs`）：
`wing_obs = [motor_8/π, -motor_7/π]`
即："motor_7 = 右翼、编码器方向与训练里 right_wing 反号"。

| 观察结果 | motor_7 = ? | motor_7 sign 与训练 | 需要的公式 | 是否要改 P3 |
|---|---|---|---|---|
| 右翼下拍 → position[0] 变负 | 右翼 | 同号 | `{pos[1]/π, pos[0]/π}` | **要改** |
| 右翼下拍 → position[0] 变正 | 右翼 | 反号 | `{pos[1]/π, -pos[0]/π}` | 保持现状 |
| 左翼下拍 → position[0] 变负 | 左翼 | 同号 | `{pos[0]/π, pos[1]/π}` | **要改** |
| 左翼下拍 → position[0] 变正 | 左翼 | 反号 | `{-pos[0]/π, pos[1]/π}` | **要改** |

**判定阶段（重要）**：把手动扭动后的 obs 值代入训练分布 `[-0.5, 0]`。**任何 obs 出现正值就是错的**。

修改点：`/ssd/Pheonix/Deploy/src/inference/src/lab_deploy_utils.hpp` 的 `motor_wing_angles_to_training_obs` 和 `motor_wing_velocities_to_training_obs`（wing_vel 走同样映射）。

---

### T3  bug 复现 + obs 定位

**目的**：确认 bug 到底出在 obs 还是 action 通路。

分两阶段。

**阶段 A：观测→模型隔离**

```yaml
# lab_inference.yaml 临时改
force_zero_policy_commands: true
```

策略输出全部强制为 0 → 腿电机保持 def-pos，轮不转。**这时手动缓慢抬翼，如果轮子仍然动，就不是策略问题**（说明可能是电机 zero 漂移、机械耦合等）。**如果轮子不动，就是策略输入问题**（继续阶段 B）。

**阶段 B：命令→观测→模型链路对齐**

```yaml
force_zero_policy_commands: false
log_policy_outputs: true
policy_log_interval_ticks: 1
```

`/cmd_vel` 保持 `linear.x=0, angular.z=0`，缓慢抬翼，看日志：

- `cmd_in=[vx=? yaw=? h=?]` 应恒为 `[0, 0, 0.24]`
- `wing_rad=[a, b]` 随姿态变化
- `wing_obs=[c, d]` 是 `motor_wing_angles_to_training_obs` 后的 obs
- `wing_vel_obs=[e, f]` 应在手动扭动时体现（缓慢扭时 ~ 0）
- `raw=[..., ?, ..., ..., ..., ?]` 的 `raw[2]` 和 `raw[5]`（左轮、右轮）

**bug 定位判据**：
- `wing_obs` 出现**正值** → P3 sign 错，按 T2 表格改
- `wing_obs` ∈ [-0.5, 0] 但 `raw[2]/raw[5]` 与 `wing_obs` 单调相关 → 可能是 P2 mnn 与训练不一致（重训 policy）或训练里翼位就是耦合前进的（去看训练 reward）

---

### T4  IMU 与翼上抬关联（回答 CR-7 → 决定 P8）

`force_zero_policy_commands=true`，架空样机，手动缓慢抬翼：

```bash
ros2 topic echo /IMU_data --field angular_velocity_covariance
```

（这个字段被 IMU 节点用来传 projected_gravity 的 0..2 分量）

- 翼水平：应 `~[0, 0, -1]`
- 翼抬起 60°：如果 x/y 分量绝对值 > 0.05 → 机身在动（架空不严）或 IMU sign 错

**排除机械耦合后再看**：若 IMU 数据真的有 pitch 变化，先物理固定机身，再手动"绕 z 轴自转"一下机器人，看 `projected_gravity_x/y` 与预期符号是否一致。若相反 → `imu_to_base_signs: [-1,-1,1]` 改成正确的。

修改点：`/ssd/Pheonix/Deploy/src/inference/config/lab_inference.yaml`。

---

### T5  Default pose 差异实测（CR-6 → 决定 P7）

**目的**：真机 zero pose 与训练 zero pose 是否几何一致。

`start_robot.sh` 完成到 "1-6 号电机进入 def-pos standby" 后：

```bash
ros2 topic echo /policy/joint_states --field position --once
```

期望是**相对默认角**（`motors_node` 已减 default），所以静止时应该 `[0, 0, 0, 0, 0, 0]`。

同时人工看真机腿部机械几何：
- 大腿相对机身 x 轴的角度
- 小腿相对大腿的角度

参考 Play 里 keyframe home 的姿态（左右腿：大腿 0.8 rad ≈ 45.8°、小腿 -1.6 rad ≈ -91.7°），对照真机是否一致。

**注意**：Deploy 侧 `motors.yaml` 有 `flipped_motors=[T,F,T,F,T,F]` + `joint_default_angle=[-0.92020, 0.98338, 0.0]×2`。这个默认角是**电机端**的，经过 flip 展开后：

- flip = true 的电机（thigh 和 foot）：**电机报的 pos = -(policy 期望 pos) + default_angle**
- flip = false 的电机（calf）：**电机报的 pos = policy 期望 pos + default_angle**

训练里 policy 期望 pos = `[0.8, -1.6, 0.0]×2`（腿零位）。经 flip 展开：
- L_thigh 电机侧应为 `-0.8 + (-0.92020) = -1.7202`
- L_calf 电机侧应为 `-1.6 + 0.98338 = -0.6166`（**符号有问题**：训练 calf 是 -1.6，Deploy 默认角 +0.98 意味着"电机报 0.98 时策略觉得机器人在 zero"）

**现场需要做**：
1. 让机器人硬件停在 def-pos standby（`/policy/joint_states` position 全 0）
2. 用长尺、量角器测**机械几何角度**（大腿相对机身、小腿相对大腿）
3. 与训练里 keyframe home（大腿 0.8 rad = 45.8°、小腿 -1.6 rad = -91.7°）比较
4. 若几何一致 → default_angle 没问题
5. 若几何差得远 → 重新校准硬件零点，或调整 `joint_default_angle` 和 `flipped_motors` 使真机 def-pos 几何 = 训练 keyframe home

修改点：`/ssd/Pheonix/Deploy/src/motors/config/motors.yaml`。

---

## 打包同步到树莓派

主机侧修改完毕，打包传给树莓派：

```bash
cd /ssd/Pheonix/Deploy
git status              # 确认有 lab_deploy_utils.hpp / lab_inference_node.cpp / motors.yaml / wing_motors.yaml / models/ 改动
git add -A && git commit -m "align WE11 obs contract to 145D and PD to training values"

# 打包（含新 lab_policy.mnn；排除 build/install/log）
tar --exclude='Deploy/build' --exclude='Deploy/install' --exclude='Deploy/log' \
    --exclude='Deploy/.git' \
    -czf /tmp/deploy-fix-$(date +%Y%m%d).tar.gz -C /ssd/Pheonix Deploy/

sha256sum /tmp/deploy-fix-*.tar.gz
```

在树莓派上：

```bash
# 备份
cd ~/Pheonix
cp -a Deploy Deploy.bak-$(date +%Y%m%d)

# 解压覆盖（保留树莓派本地 .git）
mv Deploy/.git /tmp/pi-deploy.git
tar -xzf /tmp/deploy-fix-*.tar.gz -C ~/Pheonix/
mv /tmp/pi-deploy.git Deploy/.git

# 重编（motors config 是运行时读, 只需 inference）
cd ~/Pheonix/Deploy
source /opt/ros/humble/setup.bash
colcon build --packages-select inference --symlink-install
source install/setup.bash
```

启动 `lab_inference_node` 后按 T1-T5 顺序验证。
