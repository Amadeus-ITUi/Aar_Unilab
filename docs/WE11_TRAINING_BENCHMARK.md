# WE11 内存与训练吞吐测试（2026-09-16）

测量任务为当前默认 `dr002_joystick_flat_we11/mujoco`，使用仓库实际的
RSL-RL OnPolicyRunner、FinalObservationAwarePPO、环境包装器及原生 MuJoCo
命令延迟 PD 后端。奖励、随机化、观测和控制参数不变。本报告不涵盖 Rough/Getup。

硬件：AMD Ryzen 9 7945HX（16 核 / 32 逻辑线程）、约 31 GiB 系统内存，
NVIDIA RTX 4060 Laptop GPU（8 GiB）。网络在 CUDA 上计算，物理仿真在 CPU 上。

## 结果

**原六组矩阵中最快组合：4096 环境 × 32 个物理线程，49,464 样本/秒。**
随后补充的 8192 环境可运行性短测也已通过，见下方记录。

| 环境数 | 物理线程 | 采样秒/轮 | 更新秒/轮 | 总秒/轮 | 样本/秒 | 进程峰值 RSS（GiB） |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 16 | 1.179 | 0.093 | 1.273 | 38,625 | 2.359 |
| 2048 | 24 | 1.041 | 0.093 | 1.134 | 43,340 | 2.362 |
| 2048 | 32 | 0.955 | 0.093 | 1.049 | 46,876 | 2.364 |
| 4096 | 16 | 2.310 | 0.169 | 2.479 | 39,651 | 2.507 |
| 4096 | 24 | 1.958 | 0.167 | 2.125 | 46,251 | 2.493 |
| 4096 | 32 | 1.824 | 0.164 | 1.987 | 49,464 | 2.505 |

各项耗时和 RSS 汇总两次运行；RSS 为两次进程峰值的平均值。最快组合两次分别为
49,687 和 49,243 样本/秒，差异约 0.9%。它比 4096 × 16 高约 24.8%，
比 4096 × 24 高约 6.9%，比 2048 × 32 高约 5.5%。

2048 每轮有 49,152 条样本，4096 每轮有 98,304 条；不能仅比较“每轮秒数”来
判断吞吐。当前开销主要在采样/物理阶段，网络更新仅占约 8%–9%。所有运行的
进程交换内存为 0。

| 内存口径 | 4096 环境实测 | 8192 原外推 | 8192 补充短测实测 |
| --- | ---: | ---: | ---: |
| CPU 进程峰值 RSS | 约 2.49–2.51 GiB | 约 2.76–2.80 GiB | 约 2.812 GiB |
| PyTorch 显存缓存峰值 reserved | 0.600 GiB | 约 1.150 GiB | 1.281 GiB |

本次测量时 WE11 编译后的训练模型 buffer 为 35,276 字节，PE02 碰撞 v1 为
3,805,908 字节。PE02 随后的 v2 优化已将其降至 666,388 字节，见
[PE02 性能优化报告](PE02_COLLISION_OPTIMIZATION.md)。
它们的模型规模和训练实现不同，不能把 PE02 的每环境模型内存套用到 WE11。
上述 RSS 已包含训练进程开销，显存列需另外考虑驱动上下文和桌面应用。

按本次结果启动 WE11 平地训练：

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  UNILAB_MUJOCO_NTHREADS=32 \
  /ssd/conda/envs/aar_unilab/bin/python scripts/train.py \
  robot=we11 task=flat observation=we11_v2_145 \
  policy=we11_mlp algorithm=rsl_rl_ppo simulator=mujoco \
  algo.num_envs=4096 algo.num_steps_per_env=24 training.no_play=true
```

该命令保留生产入口的默认 TensorBoard 和保存流程；它们的开销未计入上述计算吞吐。
本次没有自动修改 WE11 的线程默认值或训练超参数。

## 8192 环境补充验证

WE11 平地任务以 8192 环境、32 个物理线程、CUDA PPO 完成 10 轮实际训练，
每环境每轮 24 步，共 1,966,080 条样本。前 5 轮预热，后 5 轮测得约
51,726 样本/秒；平均采样 3.473 秒/轮、更新 0.328 秒/轮，合计 3.801 秒/轮。
进程峰值 RSS 为 2.812 GiB，最终 RSS 约 2.776 GiB，交换内存为 0；PyTorch
显存峰值 allocated 约 1.044 GiB、reserved 为 1.281 GiB。

这确认本机能够运行该配置的完整采样与 PPO 更新。该次只做一轮较短验证，吞吐
供参考；原矩阵的双次、较长重复排名保持原口径。显存缓存实测高于线性外推，
应按实测预算。该检查同样不包含渲染、日志、保存和导出开销。

原始数据：`logs/we11-benchmark-20260916/e8192_t32_check.json`，同名 `.log`、
`.jsonl` 和 `.yaml` 保存运行输出、逐轮记录和配置。上方训练命令将
`algo.num_envs=4096` 改为 `algo.num_envs=8192` 即可使用这一环境数量。

## 原六组矩阵测量方法

- 组合：MuJoCo 物理线程 16 / 24 / 32 × 并行环境 2048 / 4096。
- PyTorch intra-op / inter-op 与 BLAS / OpenMP CPU 线程固定为 1。
  测试里的“线程数”只指 `UNILAB_MUJOCO_NTHREADS`，不是关闭 GPU 后的纯 CPU PPO。
- 每组独立进程执行 40 轮，前 10 轮预热、后 30 轮计时；每组重复两次。
  第一轮交错环境数量，第二轮反向顺序，以减小运行顺序、温度和缓存的影响。
- 每环境每轮采样 24 个策略步，PPO 保留 5 epochs × 4 minibatches；
  物理 400 Hz、PD 200 Hz、策略 50 Hz。保留 NanGuard 及原任务随机化。
- CUDA 在采样/更新边界同步。采样时间包含策略推理、环境物理、奖励、观测和
  样本入库；更新时间包含 GAE、PPO 和 adaptation loss。总吞吐为全部实测样本数
  除以全部实测采样与更新耗时之和。
- 不开启渲染、TensorBoard、episode 日志统计、checkpoint 保存及 ONNX 导出；
  报告比较的是训练计算吞吐，不包含这些 I/O 和日志管理开销。
- CPU 内存取训练进程 `VmHWM`（RSS 峰值），包括模型、状态、Python/PyTorch、
  CUDA 主机端运行库和采样相关缓存；不是整机已用内存。`VmSwap` 同时记录。
- 显存数据为 PyTorch CUDA allocator 的峰值 reserved，不含桌面应用和全部 CUDA
  驱动上下文；不能直接等同于 `nvidia-smi` 的整卡占用。

原始逐轮结果、完整 Hydra 配置及各进程日志保存在
`logs/we11-benchmark-20260916/`。复测命令（使用新输出目录）：

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /ssd/conda/envs/aar_unilab/bin/python tools/benchmark_we11_training.py \
  --matrix --repeats 2 --warmup 10 --iterations 30 \
  --output logs/we11-benchmark-new
```

## 内存外推方法

8192 环境最初没有在六组矩阵中实跑，之后已完成上述补充验证。原外推使用
同线程数下的 2048 与 4096 实测值，分离
固定运行库开销和随环境数量增长的开销：

`M(8192) ≈ 3 × M(4096) − 2 × M(2048)`。

因此不能把整个 4096 训练进程的 RSS 直接乘二。外推仅适用于相同网络、24 步
rollout、任务和缓存策略；长训练、不同任务、模型随机化变体、分配器缓存以及
其他程序会改变实际占用。吞吐高低也不等于学会站立/行走所需样本更少。
