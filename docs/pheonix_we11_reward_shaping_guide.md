# Pheonix-WE11：Reward Design 与 Reward Shaping 教程

本教程的目标不是讲完 UniLab 框架，而是建立一套可以重复使用的调 reward 方法：

```text
明确希望的行为
  -> 写成可观测的 reward / penalty
  -> 单变量实验
  -> 用 TensorBoard 判断学习信号
  -> 用 Play 验证真实行为
```

适用任务：`dr002_joystick_flat_we11/mujoco` 与
`dr002_joystick_rough_we11/mujoco`。

> Reward 的作用不是直接“写出走路程序”，而是定义策略在每一步中偏好的行为。PPO 会寻找
> 能最大化累计 reward 的动作序列；因此任何 reward 都要同时考虑它可能被策略钻空子的方式。

## 1. 先建立本项目的 reward 心智模型

WE11 在每个 control step 计算：

```text
raw_term_i = reward_function_i(state, command, action)
weighted_term_i = scale_i × raw_term_i
step_reward = ctrl_dt × Σ weighted_term_i
```

当前 `ctrl_dt = 0.02 s`。`scale` 是 YAML 中的权重；正权重鼓励某个 raw term 变大，负权重
惩罚某个 raw term 变大。

例如：

```yaml
track_lin_vel_x: 1.5
orientation: -10.0
```

含义分别是：奖励前向速度跟踪，强烈惩罚机身倾斜。

reward 的实际分发在
[`joystick.py`](../src/unilab/envs/locomotion/dr002/joystick.py) 的 `_compute_reward()`，
通用分发器会遍历 `reward.scales`，计算已注册项、乘权重、记录日志，最后乘 `ctrl_dt`。
因此 TensorBoard 中的 `reward/<name>` 是**加权但尚未乘 `ctrl_dt` 的即时批均值**；episode
return 才是逐步积分后的回报。不要把这两者直接按数值大小比较。

## 2. 现有 reward 在哪里看

### 2.1 首先看 YAML：权重、阈值、开关

WE11 的 owner YAML 在：

```text
conf/ppo/task/dr002_joystick_flat_we11/base.yaml
```

其中 `reward.scales` 决定启用的项和权重；同一段下面的 `*_std`、`*_clip`、contact threshold
与 termination threshold 决定曲线形状和安全边界。

rough task 没有独立 reward 表，而是继承 flat：

```text
conf/ppo/task/dr002_joystick_rough_we11/mujoco.yaml
  -> /task/dr002_joystick_flat_we11/mujoco
  -> /task/dr002_joystick_flat_we11/base
```

所以直接修改 flat `base.yaml` 会同时影响 flat 和 rough。若只想修改 rough，应在 rough 的
`mujoco.yaml` 中添加覆盖项，例如：

```yaml
reward:
  scales:
    orientation: -12.0
```

### 2.2 再看注册表：名字对应哪个函数

WE11 可用 reward 名称在：

```text
src/unilab/envs/locomotion/dr002/joystick.py
_init_reward_functions()
```

例如：

| YAML 名称 | 行为意图 |
| --- | --- |
| `track_lin_vel_x` | 跟踪前后速度指令 |
| `track_ang_vel_z` | 跟踪偏航角速度指令 |
| `orientation` | 保持机身直立 |
| `base_height` | 跟踪机身高度指令 |
| `joint_torques_l2` | 降低腿部力矩 |
| `action_rate_l2` | 降低连续动作突变 |
| `action_smooth_lingzu` | 降低腿部动作的二阶抖动 |
| `undesired_contacts` | 避免非期望部位接触 |

具体公式紧跟在同一文件的 `_reward_*` 方法中。阅读顺序应始终是：**YAML 名称 -> 注册表 ->
对应函数 -> TensorBoard 曲线**，不要从几千行环境代码开头盲读。

## 3. 不写代码时：调权重、禁用和恢复 reward

### 3.1 最快的单次实验：Hydra 命令行覆盖

不要先改文件。一次只改一个量，用不同 `run_name` 保存结果：

```bash
python -u scripts/train_rsl_rl.py \
  task=dr002_joystick_flat_we11/mujoco \
  training.device=cuda:0 \
  training.no_play=true \
  training.play_render_mode=none \
  training.logger=tensorboard \
  algo.max_iterations=100 \
  algo.save_interval=100 \
  algo.run_name=reward_orientation_minus5 \
  reward.scales.orientation=-5.0
```

这不会修改 YAML。下一次不带该 override，就回到 YAML 基线。

### 3.2 改权重

一般规则：

```text
正奖励变大：更强地追求该行为
负惩罚绝对值变大：更强地避免该行为
scale = 0：关闭该项
```

不要一口气把权重改十倍。第一次通常做 `×0.5`、`×2` 或 `×3` 量级实验；观察 reward tag 的
实际量级后再决定。

例如，机器人晃动但尚未翻倒时，可以依次比较：

```text
ang_vel_xy: -0.025 / -0.05 / -0.10
orientation: -5 / -10 / -15
```

一次只改变一项，否则无法归因。

### 3.3 删除或关闭一项

优先设为 `0.0`，不要立刻从 YAML 删除：

```bash
reward.scales.action_smooth_lingzu=0.0
```

这样方便做 ablation（消融对比）和恢复。删除键名本身不会使函数消失，只是该项不再进入
分发器。

### 3.4 一个重要陷阱：拼写错误可能静默失效

当前分发器对“YAML 中存在、但没有注册函数”的名称会跳过，不会报错。因此：

```yaml
reward:
  scales:
    action_smooh_lingzu: -0.03  # 拼错，实际不会生效
```

训练可能照常开始。每次新增或改名后，都要在 TensorBoard 确认出现对应的
`reward/<name>` tag；没有出现就先查注册表和拼写。

## 4. 新增 reward 项：最小完整流程

只有现有项表达不了你的目标时才新增。新增项至少要完成四件事：**计算函数、注册、YAML 权重、
日志验证**。

下面以“进一步抑制轮子速度”为例。它仅作结构示例，不代表现在应该加入该项。

### 4.1 写 raw term，不在函数内乘权重

在 `src/unilab/envs/locomotion/dr002/joystick.py` 中与其他 `_reward_*` 方法相邻添加：

```python
def _reward_wheel_speed_l2(self, ctx: RewardContext) -> np.ndarray:
    assert ctx.dof_vel is not None
    return np.sum(np.square(ctx.dof_vel[:, WHEEL_ACTION_INDICES]), axis=1)
```

约定：函数返回 shape 为 `(num_envs,)` 的有限 `numpy.ndarray`，只返回原始物理量；不要在
函数中写 `-0.001`、不要乘 `ctrl_dt`、不要直接写 TensorBoard。

因为它是一个非负“代价”，YAML 中应给负 scale。

### 4.2 注册名字

在 `_init_reward_functions()` 中加入：

```python
"wheel_speed_l2": self._reward_wheel_speed_l2,
```

YAML 名称和这个字符串必须完全一致。

### 4.3 在 owner YAML 启用并设初始权重

在 `base.yaml`：

```yaml
reward:
  scales:
    wheel_speed_l2: -1.0e-5
```

初始值只是待测假设。先跑 100 iteration，查看 `reward/wheel_speed_l2` 是否量级合理、是否压制了
速度跟踪；再决定增加、减小或设为 0。

### 4.4 何时需要改 `RewardContext`

若新项只使用已有的 `linvel`、`gyro`、`gravity`、`dof_pos`、`dof_vel`、`info`、commands 或
actions，不需要改 context。若确实需要一个新传感器量，应在 `_compute_reward()` 中一次性取得，
放入 `RewardContext`，再由 reward 函数读取；不要在每个 reward 函数里重复访问 backend。

### 4.5 本地方法还是共享方法

只服务 WE11 的设计，放在 `dr002/joystick.py`。确认多个机器人/任务都使用同一数学定义时，
才放入：

```text
src/unilab/envs/locomotion/common/rewards.py
```

新增代码后，至少运行一次 1--5 iteration smoke，确认没有 NaN、shape error 或遗漏的注册项。

## 5. Reward shaping 的设计方法

### 5.1 先写“目标—代价—终止”三层

一个可维护的 locomotion reward 通常有三层：

| 层 | 问题 | WE11 示例 |
| --- | --- | --- |
| 任务目标 | 它要完成什么？ | `track_lin_vel_x`、`track_ang_vel_z` |
| 质量/代价 | 怎样完成才算好？ | 姿态、高度、力矩、动作平滑 |
| 硬失败 | 什么状态不值得继续？ | 倾倒、持续强碰撞 -> termination |

不要用无限增大的 penalty 替代 termination；也不要只写“活着”而没有任务目标。当前 WE11
已经把接触分成普通惩罚阈值和更高的终止阈值，这就是推荐模式。

### 5.2 先保证可行，再追求优雅

常见训练顺序：

```text
能站住 / 不翻倒
  -> 能按 command 前进和偏航
  -> 高度、姿态稳定
  -> 低力矩、低冲击、平滑动作
  -> 扰动与随机化下仍稳定
```

若一开始就把平滑、能耗、对称性惩罚设得太强，策略可能选择“几乎不动”来避免所有代价。
若跟踪奖励太强、姿态和接触太弱，策略可能学会冲刺、跳动或用异常姿态换速度。

### 5.3 用“量级”而非 YAML 小数判断强弱

`joint_acc_l2: -2.5e-7` 看起来很小，但加速度平方可能很大；`orientation: -10` 看起来很大，
但姿态误差可能很小。真正应比较的是 TensorBoard 中各 `reward/<name>` 的实际加权量级。

实用经验：在机器人能够完成任务的区间，任务目标项应提供清晰正信号；质量/正则项不应长期把
总 reward 完全淹没。发生翻倒时，termination 与关键安全项应足够显著。

### 5.4 防止 reward hacking

每写一个项，问三个问题：

1. 策略能否在不完成任务的情况下拿到高分？
2. 它能否用伤害机械/真机不可接受的动作换取高分？
3. 该项与哪一项冲突，冲突发生时谁应优先？

例子：只奖励 `vx` 可能让策略跳动或摔倒前冲；只罚 torque 可能让策略不动；只罚 action rate
可能让策略以平滑但错误的恒定动作失败。解决方式通常是组合目标项、质量项和合理终止，而不是
盲目增大单一权重。

## 6. TensorBoard：看什么，怎么判断

启动：

```bash
cd ~/ssd/Pheonix/UniLab
source /data/miniconda3/etc/profile.d/conda.sh
conda activate unilab_cuda
tensorboard --logdir logs/rsl_rl_ppo --port 6006
```

浏览器打开 `http://localhost:6006`。先在 Scalars 页面搜索以下前缀。

### 6.1 `reward/`：每一项实际在做什么

当前分发器会自动记录已启用且已注册项：

```text
reward/track_lin_vel_x
reward/orientation
reward/action_rate_l2
reward/undesired_contacts
...
```

它们是 `raw_term × scale` 的批均值，尚未乘 `0.02`。因此：

- 正的 tracking 项应在命令被执行时上升；
- 负的 penalty 项接近 0 通常表示该类代价小；
- 负项绝对值突然变大，说明对应的抖动、倾斜、碰撞或能耗在变坏；
- 某项永远为 0，可能是策略真的避免了它，也可能是未注册、拼错、传感器量恒为 0。

### 6.2 `base_height/`、`termination/`：不要只看总 reward

WE11 还记录：

```text
base_height/mean
base_height/target_mean
base_height/abs_error_mean
termination/contact_now_frac
termination/contact_done_frac
termination/gravity_done_frac
```

判断例子：

| 曲线组合 | 含义 / 下一步 |
| --- | --- |
| tracking 上升，termination 下降，episode length 上升 | 正常学习信号 |
| total reward 上升，但 `base_height/abs_error_mean` 变大 | 可能用高度/姿态换取速度，检查权重平衡 |
| `reward/action_rate_l2` 大幅负值且 tracking 差 | 平滑约束可能太强，或控制本身抖动 |
| `termination/gravity_done_frac` 高 | 常发生倾倒；先看姿态、接触和 command 难度，不急于加新 reward |
| `termination/contact_now_frac` 高但 done 低 | 有频繁轻微碰撞；检查 `undesired_contacts` 与 threshold |
| 某 reward 曲线非常大，其他几乎看不见 | 先检查量纲/裁剪，再调 scale |

### 6.3 PPO 训练曲线：把它当“健康监控”

不同 RSL-RL 版本的 tag 前缀可能略有不同，但通常可以找到：mean episode return、mean episode
length、policy loss、value loss、entropy、action std、采样/学习耗时。使用 TensorBoard 的 tag
搜索确认当前名称，不要依赖某个固定前缀。

重点关系：

```text
episode return 上升 + episode length 上升 + termination 下降
  -> 通常是健康改进

episode return 上升 + safety/height/termination 变差
  -> 可能是 reward 权重失衡或 reward hacking

value loss 长期异常尖峰、NaN、action std 突变
  -> 先检查 reward 量纲、终止、观测和数值稳定性
```

## 7. 推荐实验流程

每次 reward 实验建立一个最小记录：

```text
实验名：reward_orientation_minus5
基线 commit：...
task：flat / rough
唯一改动：orientation -10 -> -5
训练预算：100 / 500 / 1500 iteration
seed：...
观察的 TensorBoard tags：...
Play 0 Hz 结果：...
结论：保留 / 回退 / 下一步
```

执行节奏：

1. 先用 5 iteration 确认配置可组合、日志正常；
2. 用 100 iteration 看 reward 曲线、NaN、termination 和明显退化；
3. 用 500 或 1500 iteration 比较行为趋势；
4. 导出 ONNX，在 Play 的 `0 Hz` 手柄回放中看行为；
5. 只有基础行为成立，再检查 1/2/3 Hz 扰动或进入 Deploy。

不要因为一次短跑 return 较高就接受 reward。最终判定必须同时来自：TensorBoard、Play 行为、
安全约束和后续真机候选验证。

## 8. 改动前的检查清单

- [ ] 这项服务于任务目标、质量约束还是终止？
- [ ] raw term 的单位、范围、极值是否清楚？
- [ ] scale 的正负号是否正确？
- [ ] 是否只改变了一个变量？
- [ ] 新名称是否已注册，TensorBoard 是否出现 `reward/<name>`？
- [ ] 是否记录了基线、命令、seed、hash 和 Play 结果？
- [ ] 是否检查了策略可能利用的漏洞？
- [ ] 是否避免把未经 Play 验证的短训模型放入 Deploy？
