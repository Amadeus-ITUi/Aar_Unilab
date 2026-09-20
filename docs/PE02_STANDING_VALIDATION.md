# PE02 固定初态站立验证

这是 PE02 的第一阶段站立实验，用于验证新样机的模型、控制和完整 PPO 流程。
通过 `+experiment=standing` 显式启用，配置位于
`conf/pe02/experiment/standing.yaml`；不修改 PE01，也不覆盖 PE02 原始行走迁移基准。

## 实验设置

- 使用 `scene.xml` 的 `home`；关节初始扰动和基座初速度均为零。
- 关闭物理参数随机化、零位偏移、观测噪声、推力和动作延迟。
- 速度命令恒为零，冻结步态时钟，关闭交替接触的两项步态奖励。
- 高度误差权重从 -3 调为 -200，倾斜惩罚从 -5 调为 -10。
- 所有非足端连杆纳入接触检查；超过 35° 倾斜或非足端接触连续超过 0.2 秒终止。
- PD、力矩上限、400 Hz 物理/PD、50 Hz 策略、动作顺序与尺度保持原样。
- 网络层宽和观测布局保持原样；初始动作标准差为 0.3，熵权重为 0.005。
- PPO 默认 4096 环境 × 24 步、32 个物理线程，1000 轮，每 50 轮保存和评估。

静态平衡包含重心与支撑面的关系，也包含关节输出力矩与重力、地面支持力的平衡。
PD 在目标角与实际角相同时不输出位置误差力矩；固定的几何站姿需要通过角度偏移
产生支撑力矩。本实验由 PPO 学习该补偿，没有加入外部支撑或固定基座。

## 运行

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /ssd/conda/envs/aar_unilab/bin/python scripts/train_pe02.py \
  +experiment=standing training.device=cuda:0
```

恢复默认行走迁移配置时去掉 `+experiment=standing`。两个实验的奖励、初始化和
探索配置不同，不支持直接把站立 checkpoint 当作原始行走配置的严格续训 checkpoint。

## 验收

评估使用策略均值动作、固定 home、零速度命令。站立帧须同时满足：

- 高度与 home 相差小于 3 cm；
- 基座倾角小于 10°；
- 水平速度小于 0.1 m/s；
- 无非足端连杆接触。

除训练内的 20 秒评估，还应检查不自动重置的连续长时回放，防止把重复重置或
回合存活误当成稳定站立。固定初态实验不证明抗扰动能力、行走能力或实机可部署性。

初始化检查与本次训练记录位于 `logs/pe02-standing-validation/`。

## 2026-09-16 实测结果：通过

从随机网络开始训练，4096 个环境、每轮每环境采集 24 步，在第 600 轮得到通过验收的
策略，累计采集 58,982,400 个样本。中间分段严格续训，未使用 PE01 预训练权重。
已停止继续训练，保留第 600 轮作为本次验证策略。

独立回放使用单个环境、策略均值动作、固定 home，连续运行 60 秒；不自动重置，
无外部支撑。以下统计来自这一次连续轨迹，不代表随机初态或抗扰动成功率。

| 指标 | 结果 |
| --- | --- |
| 连续站立 | 60 秒，无失败、无重置 |
| 满足上述站立帧条件 | 100% |
| 基座高度范围 | 0.28920–0.29272 m（home 为 0.29362 m） |
| 最大基座倾角 | 4.109° |
| 最大水平位移 | 5.374 mm |
| 非足端接触 | 0 帧 |
| 1 秒后双足均接触 | 100%（每足合接触力 > 0.5 N） |

控制参数仍为每腿 Kp `[4.3, 4.3, 4.9]`、Kd `[0.34, 0.34, 0.24]`，
力矩上限 `[5.5, 5.5, 14.0]` N·m。60 秒回放记录的各关节最大绝对力矩如下：

| 关节 | 左侧 / N·m | 右侧 / N·m | 上限 / N·m |
| --- | --- | --- | --- |
| 髋 | 0.673 | 1.022 | 5.5 |
| 大腿 | 0.771 | 0.581 | 5.5 |
| 小腿 | 2.523 | 2.204 | 14.0 |

这些是仿真回放记录值，不是 PE02 实机电机参数辨识结果。初始几何姿态的竖向支撑力
近似计算得到每腿大腿约 0.50 N·m、小腿约 1.9 N·m，仅用于解释支撑需求；没有把
这个近似结果作为前馈施加到训练中。

本次产物（本机 `logs/`、`releases/` 目录）：

- [第 600 轮检查点](../logs/pe02-standing-validation/continue2/2026-09-16_20-33-32_160381_mujoco/model_600.pt)
- [60 秒完整指标](../logs/pe02-standing-validation/evaluation600/report.json)
- [60 秒状态、动作及力矩轨迹](../logs/pe02-standing-validation/evaluation600/trajectory.npz)
- [前 20 秒回放视频](../logs/pe02-standing-validation/evaluation600/standing.mp4)，使用原始完整视觉网格、正常速度
- [ONNX 导出包](../releases/pe02/pe02_flat/standing_600_20260916/)

从仓库根目录打开已训练策略的 60 秒回放：

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /ssd/conda/envs/aar_unilab/bin/python scripts/train_pe02.py \
  +experiment=standing mode=play training.device=cpu play.steps=3000 \
  checkpoint=logs/pe02-standing-validation/continue2/2026-09-16_20-33-32_160381_mujoco/model_600.pt
```

回放也需要 `+experiment=standing`，以使用一致的零频率步态输入。
复核不自动重置的 60 秒评估可运行：

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /ssd/conda/envs/aar_unilab/bin/python \
  logs/pe02-standing-validation/evaluate_standing.py \
  logs/pe02-standing-validation/continue2/2026-09-16_20-33-32_160381_mujoco/model_600.pt \
  --seconds 60 --output logs/pe02-standing-validation/evaluation600-repeat
```

ONNX 与 PyTorch 黄金输入校验通过；导出包的 Python/C++ 闭环在 1、10、100 步上
记录的动作、力矩和基座高度最大差值均为 0，详见
[导出一致性日志](../logs/pe02-standing-validation/sim2sim600.log)。

PE02 测试 37 项通过；仓库测试 251 项通过、2 项预期失败；WE11 平地和粗糙地形任务
均完成配置组合、初始化及步进。新增测试的 Ruff 检查通过，`git diff --check` 通过。
全仓静态检查仍有本次修改前的问题：12 个文件需格式化、7 项 Ruff 报错，以及
`src/unilab/base/backend/mujoco/playback.py:35` 的 1 项 mypy 报错。

当前结论限于名义参数、固定初态、零命令下的站立。原始行走配置中的随机初始化
尚未修复；下一阶段恢复随机性前，应先处理关节扰动后的足端离地或穿地，再逐项
加入观测噪声、物理参数随机化和推力。PE02 实机的电机、摩擦、延迟参数仍需独立辨识。
