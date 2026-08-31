# DR002 Joystick Reward 项完全指南

本文档描述 `base.yaml` 中 `reward:` 段下每一项的**计算方式**、**作用**、**调整为 0 的效果**以及**改大/改小的效果**，用于你从头调 reward 时对照。

代码入口：
- 派发/加权：[rewards.py:run_reward_dispatch](../../../../../UniLab/src/unilab/envs/locomotion/common/rewards.py#L331)
- 项定义与逐项 clip：[joystick.py:_init_reward_functions](../../../../../UniLab/src/unilab/envs/locomotion/dr002/joystick.py#L2619) 以及紧随其后的 `_reward_*` 方法
- 每步 reward 组装：[joystick.py:_compute_reward](../../../../../UniLab/src/unilab/envs/locomotion/dr002/joystick.py#L4699)

---

## 0. 总体架构

每个控制步（`ctrl_dt = 0.02s`）计算：

```
per_step_reward = ( Σ_i clipped_i(raw_reward_i * scale_i) ) * ctrl_dt
```

其中每个项 `raw_reward_i` 由 `_reward_<name>` 计算（一个 float per env），乘以 `scales[name]`，进入 `_clip_lingzu_reward` 做**逐项 clip**（详见 §2），最后所有项相加、乘 `ctrl_dt`。

**关键规则**：
- `scales[name] == 0` → **该项完全跳过**（[rewards.py:355](../../../../../UniLab/src/unilab/envs/locomotion/common/rewards.py#L355) 的 `continue`），不计算、不写入 log。
- `scales[name] > 0` → 视作正奖励；`< 0` → 视作惩罚。项本身返回的是 raw 数值（如平方误差、能量），正负号完全由 scale 决定。
- `only_positive_rewards: true` → 每步总和被 `max(reward, 0)` 截断到非负。当前是 `false`（惩罚可为负）。

---

## 1. Reward 项详解

下面按当前 `base.yaml` 里的顺序列出。**当前值**列出的是 `base.yaml` 现有权重。

### 1.1 `alive` （当前 = 0.0）

- **计算**（[joystick.py:4918](../../../../../UniLab/src/unilab/envs/locomotion/dr002/joystick.py#L4918)）：
  `raw = 1.0 若未触发失败，否则 0.0`。失败条件是：`gravity_z ≤ 0.7` 或 `termination_contact` 触发，且连续时长超过阈值（见 §4）。
- **作用**：给"每一步没摔倒"一个恒定奖励，可以让 agent 学到"活下来本身有价值"。
- **调 0 的影响**：完全不给存活奖励。目前就是 0——训练完全靠 tracking + 惩罚驱动。
- **调正值**（如 `+0.5`）：鼓励拖时间；容易学到"站着不动骗时间"。
- **调负值**：不建议。

### 1.2 `track_lin_vel_x` （当前 = 1.5）

- **计算**（[joystick.py:4769](../../../../../UniLab/src/unilab/envs/locomotion/dr002/joystick.py#L4769)）：
  ```
  err = (cmd_vx - base_vx)²
  raw = exp(-err / std²)     # std = track_lin_vel_x_std = 0.5
  ```
  `raw` ∈ (0, 1]。误差=0 时 raw=1；|误差|=std 时 raw≈0.37；|误差|=2·std 时 raw≈0.018。
- **作用**：任务核心——让机器人的前向线速度追上指令。
- **调 0 的影响**：agent 完全没有"跟指令走"的信号，最容易的策略是站着不动。**这个是任务信号，一般不该关**。
- **调大**（如 2.5）：优先牺牲其他项去追 vx。
- **调小**（如 0.5）：更容易被惩罚项压过去；agent 可能懒得走。
- **相关 meta**：`track_lin_vel_x_std`（曲线宽度），`track_lin_vel_x_term_clip`（该项每步最大贡献，见 §2）。

### 1.3 `track_lin_vel_x_enhance` （当前 = 1.5）

- **计算**（[joystick.py:4780](../../../../../UniLab/src/unilab/envs/locomotion/dr002/joystick.py#L4780)）：
  ```
  err = (cmd_vx - base_vx)²
  raw = exp(-err / std²) - 1     # std = track_lin_vel_x_enhance_std ≈ 1.581
  ```
  注意 `-1`：即使误差是 0，raw = 0；误差越大 raw 越负。这是**惩罚项**（虽然 scale 是正数）。std 更大意味着这个惩罚曲线更宽、更缓——远离目标时也不会一下爆炸。
- **作用**：给 `track_lin_vel_x` 一个**宽而缓的辅助**——`track_lin_vel_x` 在偏差稍大时就迅速衰减到 0（学不到梯度），`enhance` 保证远处也有回归信号。
- **调 0 的影响**：远离目标速度时没有回归梯度，主 track 项的 σ=0.5 之外区域是"平的"。可能训练更慢。
- **调大**（如 3.0）：远误差的回归压力增强，可能压垮其他项。
- **调小**：回退到只有主 tracking 项。

### 1.4 `track_ang_vel_z` （当前 = 1.0）

- **计算**（[joystick.py:4790](../../../../../UniLab/src/unilab/envs/locomotion/dr002/joystick.py#L4790)）：
  ```
  err = (cmd_wz - gyro_z)²
  raw = exp(-err / tracking_sigma²)     # tracking_sigma = 0.25
  ```
- **作用**：追踪偏航角速度指令。任务信号。
- **调 0 的影响**：机器人可以任意方向乱转。**基本不该关**。
- **调大/调小**：控制"直线走"与"跟随转向"的权重比。

### 1.5 `lin_vel_z` （当前 = -2.0）

- **计算**（[joystick.py:4795](../../../../../UniLab/src/unilab/envs/locomotion/dr002/joystick.py#L4795)）：`raw = vz²`
- **作用**：惩罚竖直方向速度——抑制跳跃、垂直抖动。
- **调 0**：允许上下抖动。腿式机器人通常会先学出很难看的蹦跳姿态。
- **调更负**（如 -5）：进一步压平，可能压到不敢下蹲/抬腿。
- **调轻**（如 -0.5）：允许一定摆动，可能出现小幅蹦跳。

### 1.6 `ang_vel_xy` （当前 = -0.05）

- **计算**（[joystick.py:4799](../../../../../UniLab/src/unilab/envs/locomotion/dr002/joystick.py#L4799)）：`raw = wx² + wy²`
- **作用**：惩罚 roll/pitch 方向的角速度——抑制身体前后左右晃动。
- **调 0**：允许摇头晃脑，容易出现"翻滚状"步态。
- **调更负**（-0.5）：过度约束躯干姿态变化，转身/加速时可能不敢发力。

### 1.7 `orientation` （当前 = -10.0）

- **计算**（[joystick.py:4803](../../../../../UniLab/src/unilab/envs/locomotion/dr002/joystick.py#L4803)）：`raw = g_x² + g_y²`
  这里 `g` 是投影到 base 坐标系的重力单位向量。正直立时 `g_x = g_y = 0, g_z = -1`。
- **作用**：惩罚**姿态**（roll/pitch 的**位置**，不是速度）——保证身体正直。
- **调 0**：允许躯干任意倾斜。**几乎肯定学不出直立行走**。
- **调更负**（-20）：更严格立直；正常。**这是保命项之一**。
- **调正**：agent 会想倒下——不要。

### 1.8 `base_height` （当前 = -2.0）

- **计算**（[joystick.py:4810](../../../../../UniLab/src/unilab/envs/locomotion/dr002/joystick.py#L4810)）：
  ```
  err = base_z - cmd_height        # cmd_height 由 commands 给出，当前固定 0.24
  raw = err² / std²                # std = base_height_std = 0.05
  ```
- **作用**：跟踪指令高度。防止蹲太低或跳起。
- **调 0**：不管高度，可能瘫在地上或蹦得很高。
- **调更负**（-5）：更严格贴目标高度。
- **相关 meta**：`base_height_std`（宽度）、`base_height_clip`（每步单项贡献上限，见 §2）。

### 1.9 `joint_torques_l2` （当前 = -1.0e-4）

- **计算**（[joystick.py:4822](../../../../../UniLab/src/unilab/envs/locomotion/dr002/joystick.py#L4822)）：
  `raw = Σ_leg (torque_i)²`（只算腿部关节：`LEG_ACTION_INDICES`）
- **作用**：能耗/扭矩正则化——鼓励用小力气。
- **调 0**：agent 可能学到扭矩最大化的暴力步态，硬件不友好。
- **调更负**：过度节能，可能不敢使劲，加速慢。
- **数量级**：扭矩平方 sum 是大数（几十到几百），所以 scale 用 `1e-4` 量级。

### 1.10 `joint_torques_wheel_l2` （当前 = -1.0e-4）

- 与上一项同，但对象是**轮子关节**（`WHEEL_ACTION_INDICES`）。
- **作用**：抑制轮子扭矩峰值——轮子驱动能耗/发热控制。
- **调 0**：轮子可能被塞满力矩指令。
- **调更负**：轮子发力受限。

### 1.11 `joint_vel_l2` （当前 = -5.0e-5）

- **计算**（[joystick.py:4842](../../../../../UniLab/src/unilab/envs/locomotion/dr002/joystick.py#L4842)）：`raw = Σ_leg dof_vel_i²`
- **作用**：腿部关节速度正则化——抑制关节角速度峰值，让动作更平缓。
- **调 0**：允许腿部关节高速甩动。
- **调更负**：动作过于"慢半拍"，跟不上指令。

### 1.12 `joint_acc_l2` （当前 = -2.5e-7）

- **计算**（[joystick.py:4849](../../../../../UniLab/src/unilab/envs/locomotion/dr002/joystick.py#L4849)）：`raw = Σ_leg qacc_i²`
- **作用**：关节加速度正则化——直接抑制"抽搐"。硬件磨损相关。
- **调 0**：可能出现高频抖动策略（sim 里 reward 还可以，sim2real 会崩）。
- **数量级**：加速度平方 sum 可以到 10⁵ 甚至更高，所以 scale 是 `1e-7`。

### 1.13 `joint_acc_wheel_l2` （当前 = -2.5e-7）

- 同上，作用于轮子关节。**作用**：平滑轮子速度指令。

### 1.14 `joint_pos_limits` （当前 = -1.0）

- **计算**（[joystick.py:4869](../../../../../UniLab/src/unilab/envs/locomotion/dr002/joystick.py#L4869)）：
  ```
  raw = Σ_leg ( max(lower_i - dof_i, 0) + max(dof_i - upper_i, 0) )
  ```
  只在**越界**时非零。
- **作用**：软性关节限位——不撞硬限位。
- **调 0**：allowed to hit joint limits，可能学出机械上不合理的姿态。
- **调更负**：更严格远离限位。

### 1.15 `nominal_state_lingzu` （当前 = -1.0）

- **计算**（[joystick.py:4881](../../../../../UniLab/src/unilab/envs/locomotion/dr002/joystick.py#L4881)）：
  ```
  raw = (dof_pos[0] - dof_pos[3])² + (dof_pos[1] - dof_pos[4])²
  ```
  索引 0/3 = 左右 thigh，1/4 = 左右 calf。**惩罚左右腿不对称**。
- **作用**：保持左右腿姿态基本对称（"标称/nominal"状态）。
- **调 0**：允许不对称姿态，可能学出瘸腿步态。
- **调更负**：强制严格对称，可能损害转向能力（转向本身就需要左右腿有差异）。

### 1.16 `action_rate_l2` （当前 = -0.01）

- **计算**（[joystick.py:4889](../../../../../UniLab/src/unilab/envs/locomotion/dr002/joystick.py#L4889)）：
  `raw = Σ (current_action - last_action)²`
- **作用**：一阶动作平滑——抑制两步之间指令跳变。
- **调 0**：action 可能高频振荡。
- **调更负**：动作变得非常慢/连续，反应迟钝。

### 1.17 `action_smooth_lingzu` （当前 = -0.03）

- **计算**（[joystick.py:4895](../../../../../UniLab/src/unilab/envs/locomotion/dr002/joystick.py#L4895)）：
  `raw = Σ_leg (a_t - 2·a_{t-1} + a_{t-2})²`
  这是**二阶差分**（加速度形式的动作平滑）。仅前两帧不够时置零。
- **作用**：二阶动作平滑——不仅要求指令变化慢，还要求变化的**变化**慢；抑制"顿挫感"。
- **调 0**：允许突变加速度，可能有"抖一下"的动作。
- **调更负**：动作变化更连续但滞后。

### 1.18 `undesired_contacts` （当前 = -20.0）

- **计算**（[joystick.py:4910](../../../../../UniLab/src/unilab/envs/locomotion/dr002/joystick.py#L4910)）：
  `raw = Σ_i 1{contact_force_i > undesired_contact_threshold}`
  即：统计有多少个"不该碰"的传感器触发。传感器列表来自 `cfg.sensor.undesired_contacts`（[joystick.py:244](../../../../../UniLab/src/unilab/envs/locomotion/dr002/joystick.py#L244)），包括 base、thigh、calf 等不该着地的部位。
- **作用**：**惩罚不该接触地面/障碍的部位**。是"保命"项之一。
- **调 0**：agent 可以随便让身体/腿部触地，容易学出爬地步态。
- **调更负**：更严格避免碰撞。**注意**：`_clip_lingzu_reward` 默认 clip=1.0，意味着不管 scale 是 -20 还是 -200，**每步该项贡献都被截到 ±1·ctrl_dt**（见 §2）。所以 -20 已经是"顶格"，再调大**没有额外效果**。
- **相关 meta**：`undesired_contact_threshold`（阈值，当前 0.1 N）。

---

## 2. Reward Clipping 机制（关键，容易踩坑）

代码：[joystick.py:_clip_lingzu_reward](../../../../../UniLab/src/unilab/envs/locomotion/dr002/joystick.py#L4744)

流程：
```python
weighted_dt = raw * scale * ctrl_dt
clipped     = clip(weighted_dt, ±clip_single_reward * ctrl_dt)
return clipped / (scale * ctrl_dt)     # 再除回来
```

派发时又乘以 `scale`，最后 `_compute_reward` 返回时乘以 `ctrl_dt`。**净效果**：

> 每一项每一步对总 reward 的贡献被硬限制在 `±clip_single_reward × ctrl_dt`。

**默认 `clip_single_reward = 1.0`**，即每步 ±0.02。**例外**：
- `track_lin_vel_x` 和 `track_lin_vel_x_enhance` 用的是 `track_lin_vel_x_term_clip`（当前 1.5）。
- `base_height` 用的是 `base_height_clip`（当前 4.0）。

### 实际含义

以 `undesired_contacts` 为例（scale=-20, clip=1.0, ctrl_dt=0.02）：
- 1 处触碰：raw=1，weighted_dt = -20 · 0.02 = -0.4；clip 到 ±0.02 → **实际贡献 -0.02/步**。
- 5 处触碰：raw=5，weighted_dt = -2.0；同样 clip 到 -0.02。
- **scale 从 -20 改到 -200 完全没差别**——已经在 clip 上限。

**要真的想让某项影响力上升，可能要**：
- 提高对应的 clip（如把 `undesired_contact_clip` 加进 reward config，需要改代码）；
- 或者接受"这个项已经饱和"，靠其他项拉大差异。

### 判断是否被 clip

看训练日志里的 `reward/<name>`：如果这个值恒等于 `±clip_single_reward · ctrl_dt`（默认 ±0.02），说明该项常年顶格 clip。

---

## 3. Meta 参数

在 `reward:` 下、`scales:` 同层：

| 参数 | 当前值 | 作用 |
|---|---|---|
| `tracking_sigma` | 0.25 | `track_ang_vel_z` 的高斯宽度（越大越宽容） |
| `track_lin_vel_x_std` | 0.5 | `track_lin_vel_x` 的宽度；`= None` 时回退到 `tracking_sigma` |
| `track_lin_vel_x_enhance_std` | ≈1.581 | `track_lin_vel_x_enhance` 的宽度；越大惩罚曲线越缓 |
| `track_lin_vel_x_term_clip` | 1.5 | 两个 x 追踪项的**逐项**每步 clip |
| `base_height_std` | 0.05 | 高度追踪宽度；越小对高度偏差越敏感 |
| `base_height_clip` | 4.0 | 高度项每步 clip（默认基线是 1.0） |
| `only_positive_rewards` | false | true 时每步总 reward 被 `max(·, 0)` 截断；改 true 意味着惩罚只能"抵消奖励"，不能让 reward 为负 |
| `undesired_contact_threshold` | 0.1 | 触发算作"接触"的力阈值（N） |

调整思路：
- **σ 系列（宽度）越大 → 越宽容**：远误差还能拿到分，但对精确追踪的激励减弱。
- **clip 越大 → 该项越强势**：能盖过其他项，但也更容易主导。

---

## 4. 终止条件相关（也在 reward 段下）

| 参数 | 当前值 | 作用 |
|---|---|---|
| `termination_gravity_z_threshold` | 0.7 | `gravity_z ≤ 0.7` 视为倾覆；连续 `termination_fail_time_s` 秒后 done |
| `termination_fail_time_s` | 0.5 | 倾覆的连续判定窗口 |
| `termination_contact_threshold` | 5.0 | 触发终止判定的接触力阈值（N）（比 `undesired_contact_threshold` 的 0.1 大得多——即"猛烈接触"才算终止） |
| `termination_contact_fail_steps` | 25 | 连续 25 步（0.5s）都在猛撞才 done |

代码：[joystick.py:_compute_terminated](../../../../../UniLab/src/unilab/envs/locomotion/dr002/joystick.py#L3663)

**注意**：`alive` 项的 raw 值由这两组条件共同决定（[joystick.py:_alive_values](../../../../../UniLab/src/unilab/envs/locomotion/dr002/joystick.py#L3846)）。所以即使 `alive: 0`，这两个阈值仍会影响 episode 是否被 done——只是"活着"这件事不再直接给 reward。

---

## 5. 最小起步建议

**"最小"= 只保留任务信号 + 保命项**。可以先从下面这套开始，其他项训练观察到问题（不倒、抖动、扭矩爆表）再加：

```yaml
reward:
  scales:
    # ── 任务信号 ──
    track_lin_vel_x: 1.5
    track_ang_vel_z: 1.0

    # ── 保命 / 姿态基线 ──
    orientation: -5.0          # 直立
    base_height: -1.0          # 别趴/别跳
    undesired_contacts: -1.0   # 别拿身体着地（scale=-1 就够，clip=1 意味着更大也没意义）

    # ── 其余全 0，训练看情况加回 ──
    alive: 0.0
    track_lin_vel_x_enhance: 0.0
    lin_vel_z: 0.0
    ang_vel_xy: 0.0
    joint_torques_l2: 0.0
    joint_torques_wheel_l2: 0.0
    joint_vel_l2: 0.0
    joint_acc_l2: 0.0
    joint_acc_wheel_l2: 0.0
    joint_pos_limits: 0.0
    nominal_state_lingzu: 0.0
    action_rate_l2: 0.0
    action_smooth_lingzu: 0.0
  tracking_sigma: 0.25
  track_lin_vel_x_std: 0.5
  track_lin_vel_x_enhance_std: 1.5811388300841898
  track_lin_vel_x_term_clip: 1.5
  base_height_std: 0.05
  base_height_clip: 4.0
  only_positive_rewards: false
  undesired_contact_threshold: 0.1
  termination_contact_threshold: 5.0
  termination_contact_fail_steps: 25
  termination_gravity_z_threshold: 0.7
  termination_fail_time_s: 0.5
```

**加回的常见节奏**（每次只加 1-2 项，训一段观察）：
1. 训练发现**高频抖动** → 加 `joint_acc_l2` / `action_rate_l2`。
2. 训练发现**扭矩爆表** → 加 `joint_torques_l2`。
3. 训练发现**上下抖动/蹦跳** → 加 `lin_vel_z`。
4. 训练发现**左右腿瘸拐** → 加 `nominal_state_lingzu`。
5. 训练发现**贴近关节限位** → 加 `joint_pos_limits`。
6. `track_lin_vel_x` 追不上远指令（reward 平坦） → 加 `track_lin_vel_x_enhance`。

**别改的**：`termination_*` 和 `undesired_contact_threshold`——它们决定 episode 终止和"什么算接触"，属于任务定义。除非你要重新定义任务，否则保持默认。

---

## 6. Loss 面板：如何解读训练过程健康度

**Reward 告诉你"agent 学到了什么"（结果）；Loss 告诉你"训练过程本身稳不稳"（过程）。两个都好才叫训得好。**

代码入口：[rsl_rl_ppo.py:_compute_ppo_loss](../src/unilab/algos/torch/rsl_rl_ppo.py#L147) 附近。

TensorBoard 面板里能看到的关键指标（rsl-rl 打印时前缀是 `Mean *`，写到 tensorboard 通常是 `Loss/*` 或直接同名）：

### 6.1 各条曲线含义

| 指标（打印名 / tb 名） | 含义 | 训练健康的样子 | 异常信号与应对 |
|---|---|---|---|
| **`Mean surrogate loss`** | PPO 的策略主 loss：`max(-A·ratio, -A·clip(ratio, 1±ε))`。核心的策略更新信号 | 小幅震荡在 0 附近，通常 \|value\| < 0.1；符号可正可负 | 剧烈振荡 / 单向发散 → 策略在崩。检查 `algorithm.clip_param`（默认 0.2）和 `learning_rate` |
| **`Mean value loss`** | Critic 的 MSE：`(V - target)²`，衡量价值函数拟合误差 | 前期高，逐步下降后稳定在一个合理区间 | 持续上升 → critic 追不上 return 变化（reward 剧烈重塑时正常，稳定后应该回落）|
| **`Mean entropy loss`** | 策略分布的熵。多维高斯：`0.5·log(2πe·σ²)` 之和。**注意：这是"熵的值"，被 `-entropy_coef·entropy` 加进总 loss（负号），所以熵越高越有探索** | 缓慢下降（对应 action_std 从 0.5 慢慢降到 0.1~0.2） | 骤降到接近 0 → 过早收敛，探索死了；长期不降 → 学不到东西 |
| **`Mean kl loss`** | 新旧策略的 KL 散度（近似值）。用于自适应 LR 调整 | 稳定在 `desired_kl` 附近（本项目 = `1.2e-2`） | 持续 >> desired → 策略变化太剧烈，LR 会被自动压低；≈0 → 策略停滞 |
| **`Mean adaptation loss`** | WE11 特有：MlpAdaptModel 里 privileged→proprio 适配头的预测 loss（[rsl_rl_ppo.py:207](../src/unilab/algos/torch/rsl_rl_ppo.py#L207)） | 稳步下降后收敛 | 不下降 → adaptation MLP 学不到 privileged 信息，部署时 proprio-only 推理会漂 |
| **`Mean action std`** | 策略输出高斯的 σ（`training.py` [第 49 行](../scripts/train_rsl_rl.py#L49) patch 上去的） | 从 `init_std=0.5` 缓降到 0.1~0.2 附近 | 秒降到 <0.05 → 探索死；久不降 → 一直在乱试 |
| **`Mean reward`** | 每 iter 平均 episode return（不是每步，是累计） | 单调（可能带小波动）上升然后趋于平台 | 崩到 0 / 变负值 → 策略退化，通常伴随 KL 尖峰 |
| **`Mean episode length`** | 每 iter 平均 episode 步数 | 从低（早期摔倒早）逐渐上升到接近 1150（=23s/0.02s，max_episode_steps） | 长期低 → 频繁被 done。用 `termination/contact_done_frac`、`termination/gravity_done_frac` 定位是接触还是姿态问题 |

### 6.2 情景对照表

| 现象 | 诊断 | 应对 |
|---|---|---|
| reward 稳步上升 + surrogate/value 都在合理量级 + KL 稳定 | 完美 | 继续训，加迭代 |
| reward 上升但 KL 频繁 >> desired_kl | 策略变化太猛，后期可能崩 | 看 `learning_rate` 是否被自动压低；若未压低，手动调小 `algo.algorithm.learning_rate` 或 `desired_kl` |
| reward 平坦 + entropy 高 + surrogate 抖 | 学不到，reward 信号太弱 | 检查任务信号项（`track_lin_vel_x`, `track_ang_vel_z`）是否被惩罚项压垮，参考 §2 clip 分析 |
| reward 平坦 + entropy 迅速降到 0 | 过早收敛到次优（比如"站着不动"） | 调高 `algo.algorithm.entropy_coef`（当前 `5e-3`）；或临时开 `alive: 0.1` 作为脚手架 |
| reward 一度上升后突然崩 | 策略更新过头 | 看 `value_loss` 或 `kl` 是否有尖峰；`training.nan_guard` dump 有没有出现 |
| `value_loss` 一直很大不下降 | Critic 容量不够或 return 分布震荡 | 试 critic 网络增大（`[256,128,64]` → `[256,256,128]`）；或减小 reward 项之间的量级差异 |
| `adaptation_loss` 不下降 | privileged→proprio 头学不到 | 检查 `algo.algorithm.adaptation_loss_coef`（当前 `1.0`）；确认 obs_groups 中 privileged 通道正确 |
| reward 项 log 值恒等于 ±0.02（默认 clip · ctrl_dt） | 该项常年顶格 clip | 参考 §2，调 scale 更大**无效**；要么接受，要么改代码提升该项 clip |

### 6.3 你现在应该关注的顺序

1. **`Mean reward` 和 `Mean episode length`** — 学没学到（结果）
2. **`Mean kl loss`** — 更新稳不稳
3. **`Mean action std`** — 探索有没有死
4. **各 `reward/<name>`** — 每一项的贡献（辅助判断是被 clip 顶格还是真的在起作用）
5. **`Mean value loss`, `Mean surrogate loss`** — 只在训不动或崩了时深入排查

---

## 7. 快速排查提示

- **训练不动**：检查 `track_lin_vel_x` 是否被过强惩罚项压死。日志里看 `reward/track_lin_vel_x` 是否接近 0。
- **摔倒率高**：调高 `orientation`（如 -10）；确认 `alive` 是否非 0；确认 `termination_*` 没被误改。
- **步态抖**：加 / 调重 `joint_acc_l2`, `action_rate_l2`, `action_smooth_lingzu`。
- **某项 reward 恒定** → 大概率被 clip（§2）。
- **loss 全 NaN / 训练中断**：`training.nan_guard` 会把出问题时的 env 状态 dump 到 log 目录下，保留 dump 用来定位。

---

## 8. 训练命令（已实测跑通）

### 完整训练（本地实测：`unilab_cuda` 环境, CUDA 12.8, torch 2.7）

```bash
cd /home/angela/ssd/Pheonix/UniLab
conda activate unilab_cuda

CUDA_VISIBLE_DEVICES=0 \
UNILAB_MUJOCO_NTHREADS=16 \
python -u scripts/train_rsl_rl.py \
  training.device=cuda:0 \
  training.no_play=true \
  training.play_render_mode=none \
  training.logger=tensorboard \
  algo.num_envs=4096 \
  algo.num_steps_per_env=24 \
  algo.max_iterations=2000 \
  algo.save_interval=100
```

### 快速 smoke（用来验证 reward 改动语法/量级是否合理，5 iter 约 5 秒）

```bash
CUDA_VISIBLE_DEVICES=0 UNILAB_MUJOCO_NTHREADS=8 \
python -u scripts/train_rsl_rl.py \
  training.device=cuda:0 \
  training.no_play=true \
  training.play_render_mode=none \
  algo.num_envs=512 \
  algo.max_iterations=5 \
  algo.save_interval=5
```

改完 `base.yaml` 先跑这个；能跑到末尾并输出打印表格就说明配置没写错。**这不是训练**，只是烟测。

### 关键 flag 说明

- `training.no_play=true` **必开**。否则训练结束会自动进 play 模式。**本 UniLab 分支 play 不一定稳**，先关掉；单独想看效果用后面的 play 命令。
- `training.play_render_mode=none` 配合 `no_play=true` 一起关掉渲染路径。
- `training.logger=tensorboard`：产出 tb 事件到 `logs/rsl_rl_ppo/DR002JoystickFlatWE11/<timestamp>_mujoco/`。
- `algo.num_envs`：内存吃紧就减半（2048 甚至 1024）。会线性减少每 iter 步数。
- `algo.max_iterations`：调 reward 时**先 300-500** 看趋势，别一开始就 2000+。

### TensorBoard 监控（另开终端）

```bash
cd /home/angela/ssd/Pheonix/UniLab
conda activate unilab_cuda
tensorboard --logdir logs/rsl_rl_ppo --port 6006 --reload_interval 5
```

浏览器打开 `http://localhost:6006`。TensorBoard 会自动列出所有 run，勾选/取消勾选对比不同 reward 版本。

### 用 Play 工程回放（`.pt` → `.onnx` → `Play/rl_sim_mujoco`）

**本 UniLab 分支的内建 play 路径不保证稳定**。稳妥的看效果方式是把训练产物导出 ONNX，然后交给同层的独立 Play 工程（`/ssd/Pheonix/Play`）回放。Play 是一个已编译好的 MuJoCo + ONNX Runtime 独立闭包，当前接口为 `obs[1,145] -> act[1,6]`。

#### 1. UniLab 侧：确认 ONNX 已导出

`train_rsl_rl.py` 顶部有 `EXPORT_POLICY = True`（[第 429 行](../scripts/train_rsl_rl.py#L429)）。导出 ONNX 的实际入口是 [第 227-228 行](../scripts/train_rsl_rl.py#L227) `runner.export_policy_to_onnx(...)`，位于渲染循环**之前**——所以只要进入 `play_rsl_rl()`，ONNX 就会被写出，无论后续是否真的渲染。

触发方式：`play_only=true` 会强制进入 `play_rsl_rl()`（见 [第 414-420 行](../scripts/train_rsl_rl.py#L414-L420) 的 `should_export_play_only_policy` 逻辑），配合 `play_render_mode=none` 让渲染循环被 [`should_run_playback`](../src/unilab/training/run.py#L16) 短路掉，最终效果就是"只导出，不渲染"：

```bash
cd /home/angela/ssd/Pheonix/UniLab
conda activate unilab_cuda

python -u scripts/train_rsl_rl.py \
  training.play_only=true \
  training.play_render_mode=none \
  algo.load_run=<run目录名> \
  algo.checkpoint=-1
```

`<run目录名>` 在 `logs/rsl_rl_ppo/DR002JoystickFlatWE11/`，形如 `2026-08-19_15-30-00_mujoco`。`checkpoint=-1` 表示用最新的 `model_*.pt`。

**不需要**设 `training.play_steps=1`——`play_render_mode=none` 已经让整个渲染循环被跳过，`play_steps` 只在渲染时才有意义。会看到打印 `Skipping playback because training.play_render_mode=none.`，那是正常的。

执行完，会在 run 目录下产出 `policy.onnx` 和 `policy_export_manifest.json`；后者明确记录本次实际加载的 checkpoint、Git 状态、接口和哈希。**验证**：

```bash
ls logs/rsl_rl_ppo/DR002JoystickFlatWE11/<run目录名>/{policy.onnx,policy_export_manifest.json}
```

#### 2. 发布 ONNX 到 Play 工程

不要手工复制 ONNX；发布脚本会同时校验 145D 契约、Play 配置和来源哈希，并原子更新模型、manifest、`SOURCE.md` 和 `SHA256SUMS`。默认仅检查：

```bash
cd /ssd/Pheonix/UniLab
python scripts/publish_we11_to_play.py --run <run目录名>
python scripts/publish_we11_to_play.py --run <run目录名> --apply
```

旧式 run 没有 sidecar 时，必须显式加 `--checkpoint model_N.pt`；成功收编后会补写 sidecar。被替换文件保存在工作区 `artifacts/we11-policy-backups/<release-id>/Play/`。

**注意**：接口必须是 `obs[1,145] -> act[1,6]`，即 29D 单帧、5 帧 term-major 历史。如果改了观测（history 长度、privileged 通道、wing_angle_obs 开关等），维度会变，发布脚本和 `rl_sim_mujoco` 都会拒绝加载。当前 `base.yaml` 的观测契约与 Play 的 `config.yaml` 是匹配的。

#### 3. 用 Play 回放

Play 工程无需 conda，只要系统装了它 README 列出的 apt 依赖，且 `build/bin/rl_sim_mujoco` 已编译（当前仓库自带二进制，一般直接可用）。

```bash
cd /ssd/Pheonix/Play

# Flat 场景，无外力/翼角回放（最纯净，只看基础步态）
./scripts/play_we11.sh 0

# Flat + 1/2/3 Hz 实测外力和翼角 CSV 回放
./scripts/play_we11.sh 1
./scripts/play_we11.sh 2
./scripts/play_we11.sh 3

# 第二个参数是 wrench 振幅倍率（默认 1.0）
./scripts/play_we11.sh 3 1.5

# Level-9 Rough 地形回放（默认 rough + 3Hz + 1.0 倍力）
./scripts/play_we11_level9.sh
./scripts/play_we11_level9.sh flat 3 1.0
./scripts/play_we11_level9.sh uphill 3 1.0
./scripts/play_we11_level9.sh downhill 3 1.0
```

启动后会打开 MuJoCo GUI 窗口。默认**暂停**状态，按空格开始；或者事先 `export RL_SAR_PLAY_AUTOSTART=1` 自动开跑。

#### 4. 恢复旧版本（对比时用）

从 `artifacts/we11-policy-backups/<release-id>/Play/` 成组恢复 ONNX、manifest、`SOURCE.md` 和 `SHA256SUMS`，不要只替换 ONNX，否则来源记录会故意报漂移。

#### 5. 常见坑

- **`obs shape mismatch`**：你改动了观测维度（`_HISTORY_LENGTH`、`_TERM_DIMS`、`wing_angle_obs.enabled` 等）。要么改回来，要么同步更新 Play 里的 `config.yaml`（不推荐，容易忘记同步）。
- **`policy.onnx` 太小 / 加载失败**：说明 UniLab 那边 export 步骤没跑完。回到 §1 确认 run 目录里 `policy.onnx` 大小正常（几百 KB 到 MB 级）。
- **动作看起来不像训好的**：检查 `policy_export_manifest.json` 和 Play `deployment_manifest.json` 记录的 checkpoint 是否就是目标 checkpoint；发布流程不再通过文件时间猜测来源。

### Resume 中断的训练

```bash
cd /home/angela/ssd/Pheonix/UniLab
conda activate unilab_cuda

python -u scripts/train_rsl_rl.py \
  training.no_play=true \
  training.play_render_mode=none \
  algo.resume=true \
  algo.load_run=<run目录名> \
  algo.checkpoint=-1
```
