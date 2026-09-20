# PE03 机械限位与 PD 目标修正

2026-09-20：用户确认大腿、小腿现有范围为实际限位，左髋机械范围可设为
`[-0.785, 1.57] rad`；随后选择将训练和控制的内收范围收紧到 **11.1°**。
当前左髋为 `[-0.19373154697137057, 1.57] rad`，右髋按镜像使用
`[-1.57, 0.19373154697137057] rad`。外展、大腿、小腿限位保持原值。
当前任务资产版本为 `pe03-cnc-joint-limits-v3`。

新 `pe03_gait_flat` 的固定、可调步态实验使用独立模型文件：

- 场景：`src/unilab/assets/robots/pe03/scene_joint_limits.xml`。
- 机器人：同目录 `pe03_joint_limits.xml`。
- URDF：同目录 `urdf/pe03_joint_limits.urdf`。
- 足端 workspace：`analysis/gait_workspace_joint_limits.npz`。
- 配置：`conf/pe03/task/pe03_gait_flat.yaml`，`control.clip_joint_targets: true`。

旧模型、旧 workspace 和缺少此配置项的旧 checkpoint 保持原有控制行为。
新旧模型不能直接 strict resume，因为模型/控制合同已变化。新的限制范围是仿真模型
与目标控制使用的范围；各项归一化、38×30 观测、网络结构、home、PD 增益、动作尺度和
力矩上限没有改变。

## 最终 PD 目标

原处理只有归一化动作裁剪及依据力矩估计的目标处理。现在在这些处理之后计算：

```
q_target = clip(home + offset, joint_lower, joint_upper)
offset = q_target - home
applied_action = offset / action_scale
```

逐关节边界从后端编译模型读取并缓存；step 不读取 XML 或其他资产文件。历史观测、
动作差分奖励、PD 执行使用同一个 applied_action。真实电机力矩仍受既有上限控制。

目标裁剪约束的是请求角度。MuJoCo 动态关节约束可能有瞬态超限，不能把它和目标越界
混为一谈。TensorBoard 新增：

- `control/target_clipped_fraction`：最终目标发生硬限位裁剪的关节样本比例。
- `control/target_clip_distance`：目标裁剪距离均值，rad。
- `control/hard_limit_excess`：每步批次真实关节角超出硬范围的最大值，rad。

原来的软限位奖励继续使用原有边距规则，合法 home 不受软限位处罚。没有额外修改
奖励权重、终止条件、命令范围或步态课程。

ONNX 继续输出六维原始策略动作。导出包 `pe03_runtime.json` 记录逐关节
`joint_target_limits`，C++ 验证其与模型、home 一致，并在 PD 前执行同样裁剪。
新包使用 `pe03.runtime.v4.joint-limits.v1` schema，让不支持目标限位的旧 C++ 程序
拒绝加载，而不是静默忽略新约束。
旧导出包没有该字段时沿用旧控制语义。足端 workspace 与 checkpoint 的指纹现在跟随
场景实际 include 的机器人文件，旧模型的指纹算法结果保持一致。

## 负向髋余量与耦合碰撞

以下是收紧到 11.1° 之前的机械范围扫描记录，资产指纹对应当时的模型。
该扫描使用同一组碰撞体，将机身抬到 1 m 排除地面，每侧取 25×25 组大腿/小腿角度。
髋从 0 向 −0.785 rad 以 0.005 rad 步长扫描首次干涉，再二分细化。右腿用镜像符号。
另外单独检查精确 home 腿形。只统计当前 MuJoCo 模型启用的自碰撞对，穿透阈值 1 μm。

| 左侧大腿、小腿角度 | 左髋单侧内收，另一腿保持 home | 双腿镜像内收时的左髋 | 首次干涉 |
|---|---:|---:|---|
| home：−0.31239、−1.25413 rad | 约 −0.4293 rad / −24.6° | 约 −0.1930 rad / −11.1° | 两足互碰 |
| −0.1、−0.2 rad | 扫描至机械端点 −0.785 rad 未碰撞 | 约 −0.1965 rad / −11.3° | 双腿时两足互碰 |
| −0.8、−1.4 rad | 约 −0.5782 rad / −33.1° | 约 −0.2146 rad / −12.3° | 单腿小腿碰机身；双腿大腿互碰 |
| −1.0、−0.2 rad | 约 −0.3291 rad / −18.9° | 约 −0.1396 rad / −8.0° | 单腿大腿碰机身；双腿两足互碰 |

左/右单侧结果接近，个别姿势存在原装配差异。图中灰色表示该腿形在髋=0 时就已
有碰撞，不表示换成所有其他髋角也必然碰撞。

这说明实际可行域依赖至少六个关节的组合，不能用一条固定的“安全髋下限”完整表达。
本次硬限位裁剪不会自动投影到全身无碰撞姿态；足端 workspace 的投影也不是动作
空间避碰器。动态运动路径和六自由度任意组合不能由这三个切片保证。
角度边界来自简化碰撞体，不应把二分数值精度当成实机几何测量精度。

复现与结果：

```
env -u PYTHONPATH /ssd/conda/envs/aar_unilab/bin/python tools/audit_pe03_joint_limits.py
```

结果在 `src/unilab/assets/robots/pe03/analysis/joint_limits/`：
`audit.json`、`slices.npz`、`hip_collision_slices.png`、`hip_collision_slices.svg`。

收紧前的工程验证记录在 `logs/pe03-validation/joint_limits_20260920/`；短训仅验证接口与执行链路，
不代表策略已通过站立或步态验收。

收紧前验证：32 项定向测试通过；全仓非 slow 测试 430 passed、2 xfailed；最终版本化
导出测试通过。32 环境短训 3 轮并保存导出；普通策略与极端常量动作策略的 Python/C++
1、10、100 步闭环差异均为 0。旧 checkpoint 的资产指纹不变，100 步录制动作回放
qpos 差异为 0。WE11 两个保护任务均完成 compose/init/step。
修改文件 Ruff 通过；全仓仍有已有的 12 个格式问题文件、7 个 lint 问题和
`src/unilab/base/backend/mujoco/playback.py:35` 的 1 个 mypy 问题。diff 检查通过。

## 内收收紧到 11.1° 的验证

记录：`logs/pe03-validation/hip_inward_11_1_20260920/`。
XML 与 URDF 同步到新边界，按每轴 41 个采样点重建足端 workspace，左右分别保留
42622、42596 个有效样本，资产指纹及样本关节范围检查通过。
32 项定向测试通过，包含目标裁剪、动作历史、短训保存及 ONNX 数值对照。
新导出包记录 `pe03-cnc-joint-limits-v3` 与逐关节限位，Python/C++ 的
1、10、100 步闭环对照最大差异均为 0。全仓非 slow 测试 430 passed、2 xfailed，
WE11 flat/rough 均完成 compose/init/3 steps。本次修改的 Python 文件 Ruff 检查通过；
全仓静态检查仍为上述已有的 12 个格式问题文件、7 个 lint 问题和 1 个 mypy 问题。
