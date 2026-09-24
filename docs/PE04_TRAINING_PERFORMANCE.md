# PE04 训练速度优化（2026-09-23）

优化保持 4096 环境、每轮 24 步、5 epochs × 4 minibatches，以及原有网络、观测、接触历史、奖励、随机化和控制频率。新启动的训练自动使用环境侧优化。没有启动长期训练，也没有修改现有正式训练日志或模型。

## 瓶颈和改动

已有正式训练最后 10 轮平均：采样 **3.464 s**，更新 **0.481 s**，合计 **3.944 s**。主要开销在环境采样，日志和 GPU 更新不是第一瓶颈。独立复测原路径在较早训练状态约 3.5 s/轮；cProfile 会额外增加计时，不能将带 profiler 的结果直接用于速度对比。

1. 原来一个 20 ms 策略步拆成四次原生 PD 调用，每次之后先做清空控制的 forward，再做实际控制的 forward。现在一次调用推进完整 8 个物理步，在 200 Hz 接触采样边界计算实际控制下的观测，并直接返回四帧接触历史和关节加速度。
2. 物理积分期间跳过未使用的中间传感器计算；传感器仍在规定采样时刻更新。控制、FIFO、差分速度及力矩限幅保持原顺序。
3. 接触采样的 forward 已经计算了下一物理步所需的位置/速度阶段；非 RK4 路径用 `mj_step2` 复用结果，重新计算受新控制影响的动力学并积分。RK4 保留完整 `mj_step`。
4. 移除采样时对同一物理状态的冗余 reset/copy；求解器 warmstart 仍按原契约清零。
5. 环境 reset 后只为所选环境计算物理观测及拼接 Critic。保留原来的完整随机数消耗顺序，避免改变其他环境的噪声与后续采样。
6. 复用奖励/终止检查的接触力范数，简化三维叉积，并缓存每个 PPO minibatch 的固定目标数据。

原生扩展的旧 API 默认行为不变，PE01/PE02/PE03 和 WE11 不开启新的 telemetry 路径。扩展构建由 `third_party/mujoco_uni_mixed_pd/build_extension.sh` 的能力检查管理。本机已完成构建。

## FP32 与可选 BF16

默认 `training.update_precision=float32`。纯性能路径通过新旧环境逐项一致性比较；不减少训练工作量，也不改变默认数值精度。

为了进一步接近 1.5 s，可显式启用 CUDA BF16 更新：

```bash
cd /ssd/Aar_Unilab
bash tools/train.sh pe04_tron1 training.update_precision=bfloat16
```

BF16 仅用于训练更新中的网络运算。参数、优化器状态、rollout、损失计算和 checkpoint 仍为 FP32；采样推理、Python 回放与 ONNX 导出保持 FP32。关闭 autocast 权重缓存，避免同一次 update 内优化器修改权重、或 Encoder 在 no-grad/训练间切换时复用过期 cast。BF16 需要支持该格式的 CUDA GPU，当前 RTX 4060 支持。

BF16 **不与 FP32 数值等价**。1024 环境、同一 rollout 和随机排列的一轮更新比较：Actor 输出 RMSE 约 0.000528、最大差约 0.002857；Critic 输出 RMSE 约 0.001408、最大差约 0.01546。两者均完成 20 次 PPO 和 20 次 Encoder 更新。这是短期数值检查，不是长期收敛评估。

需要继续使用原数值精度时直接运行：

```bash
bash tools/train.sh pe04_tron1
```

不建议把降低环境数量、rollout 长度或 PPO epochs 得到的耗时，当作本次优化收益。

## 测量方式

最终代码连续计时结果（新测量均为 3 轮预热＋12 轮计时）：

| 模式 | 采样 | 更新 | 合计 / 轮 |
| --- | ---: | ---: | ---: |
| 优化前正式日志，第 29–38 轮 | 3.464 s | 0.481 s | 3.944 s |
| 优化后，默认 FP32 | 1.371 s | 0.465 s | **1.835 s** |
| 优化后，可选 BF16 更新 | 1.377 s | 0.186 s | **1.563 s** |

BF16 的 12 轮范围为 **1.505–1.642 s**，接近 1.5 s 目标，但尚未达到每轮稳定 ≤1.5 s。FP32 范围为 1.783–1.911 s。正式日志与新测量处于不同训练阶段；接触数量、reset 比例、CPU/GPU 时钟和其他进程都会影响实际耗时。

所有对比使用同一台 Ryzen 9 7945HX / RTX 4060 Laptop，32 个 MuJoCo 线程。计时包含采样和完整网络更新，排除启动、定期评估、checkpoint 保存和导出；CUDA 在计时边界同步。线程数 16、扩大原生任务分块均未获得收益，默认仍使用原线程与分块设置。

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /ssd/conda/envs/aar_unilab/bin/python tools/benchmark_pe04_training.py \
  --warmup 3 --iterations 12 --output /tmp/pe04-benchmark.json \
  training.update_precision=bfloat16
```

可将 `bfloat16` 换成 `float32` 比较。`--profile /tmp/pe04.prof` 在正式计时之后额外采样一轮，避免 profiler 污染测量值。结果保留在 `logs/reports/pe04_performance/`。

## 验证范围

- 新旧完整环境 80 步逐项一致，包含随机化、推扰、噪声、稀疏 reset、奖励和物理状态。
- 专项覆盖融合/旧路径、两种关节速度定义、不同延迟、接触历史、稀疏观测 RNG、CPU/CUDA PPO 缓存等价和 BF16 参数更新/FP32 checkpoint。
- PE04 与 WE11 flat/rough 专项共 28 项通过，包含两个 WE11 任务的初始化与 step。
- BF16 短训 10 轮、续训 2 轮及 ONNX 导出；Python/C++ 闭环 1/10/100 步对比。
- 全仓测试 **533 passed、4 failed、2 xfailed**；4 个失败仍是 PE03 curriculum 与只读参考目录嵌套 Git 的已有问题。mypy、本次代码 Ruff 与 `git diff --check` 通过；全仓仍有原来的 11 个格式文件及 7 个 lint 问题。
