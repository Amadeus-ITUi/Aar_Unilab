# PE02 碰撞简化与训练性能（2026-09-16）

PE02 已采用碰撞 v2 和等价的 C++ 批量位置 PD。在本机 4096 环境、32 个物理线程、
CUDA 网络计算的两轮独立复测中，完整训练吞吐平均为 **53258 样本/秒**，
同条件 WE11 平地任务为 **50594 样本/秒**。本次测量中 PE02 约高 5.3%，达到
接近 WE11 的吞吐目标。8192 环境也已完成正式短训、保存和发布包导出。

后续显示修复：MJCF 视觉网格已从减面 STL 恢复为保留原始 1,108,202 个三角面的 OBJ。
正式训练在编译前移除视觉资产，训练模型 buffer 仍为 666388 字节，碰撞及控制未改动。
下文“视觉网格保持不变”描述的是碰撞 v2 改造当时的对照结果。

## 实际改造

| 项目 | 原 PE02 碰撞 v1 | 当前 PE02 碰撞 v2 |
| --- | ---: | ---: |
| 机器人碰撞体，不含地面 | 59 | 26 |
| 凸包网格 | 35 | 19 |
| 碰撞网格顶点 | 7430 | 1260 |
| 碰撞网格三角面 | 14720 | 2444 |
| 单份正式训练模型 buffer | 3805908 字节 | 666388 字节 |
| 每只足端的顶点 / 面 | 2351 / 4698 | 230 / 456 |

模型 buffer 减少约 **82.5%**。机身支架间隙保留，大腿内部零件合并，小腿使用
经过间隙检查的三段凸包；足端按最大 0.5 mm 几何误差减面。减面保留默认支撑面
1 mm 内的原始顶点，默认姿态在 0.1、0.25、0.5 mm 深度处的截面积保持原值。
详情见[碰撞几何说明](../src/unilab/assets/robots/pe02/analysis/COLLISION_GEOMETRY.md)。

为消除大规模训练中每个策略步 8 次 Python/原生调用的开销，新增通用
`step_joint_position_pd` 后端入口，将角度偏移 FIFO、位置 PD、位置差分速度和
物理子步合并执行。关节地址和执行器 gear 来自冷路径缓存，没有写死 PE02 尺寸。
`training.native_pd=true` 默认启用；缺少新版扩展时警告并回退到 Python 实现。
PE01 和 WE11 沿用各自原控制入口。

PE02 仍有独立的配置、环境和网络。没有改变其网络层宽、PPO/encoder 更新次数、
奖励、随机化、400 Hz 物理/PD、50 Hz 策略、动作尺度或延迟语义。模型改造也未
改变质量 **3.22689 kg**、各连杆质心/惯量、机械零位、限位、阻尼及 home。

## 完整训练吞吐

硬件：Ryzen 9 7945HX，16 核 / 32 逻辑线程，约 31 GiB 内存，RTX 4060 Laptop
8 GiB。两个任务均使用 CPU MuJoCo + CUDA PPO，PyTorch/BLAS CPU 线程固定为 1。

4096 环境 × 每轮 24 个策略步；每轮 5 epochs × 4 minibatches，保留各自网络和
任务语义。PE02 还执行独立 encoder 优化。每个实现测两次，每次 5 轮预热、
20 轮计时，进程串行执行；CUDA 在采样与更新边界同步。采样包括策略推理、物理、
奖励、观测和轨迹存储，更新包含 GAE、PPO 和辅助网络损失。排除渲染、评估、
TensorBoard 写入、checkpoint 保存及导出。

| 项目 | PE02 最终版 | WE11 flat |
| --- | ---: | ---: |
| 第一次吞吐，样本/秒 | 54247 | 50423 |
| 第二次吞吐，样本/秒 | 52304 | 50766 |
| 合并平均吞吐，样本/秒 | **53258** | **50594** |
| 平均采样秒/轮 | 1.311 | 1.781 |
| 平均网络更新秒/轮 | 0.535 | 0.162 |
| CPU 进程峰值 RSS，两次最大值 | **4.178 GiB** | **2.495 GiB** |
| PyTorch 显存缓存峰值 reserved | 0.889 GiB | 0.600 GiB |

PE02 两次吞吐差约 3.7%。5.3% 是本次测量的差值，不是所有任务/训练阶段的固定
优势。两个任务的网络、碰撞、奖励与姿态分布不同，吞吐相近不代表学习样本效率
或最终控制质量相同。显存 reserved 不包含全部驱动上下文和桌面应用。

最终数据：`logs/pe02-optimization/final_e4096_t32_r{1,2}.json`、
`we11_e4096_t32_r{1,2}.json`、`summary.json`。
目录中的 `candidate`、`compact`、`decimated`、`native_e*` 是中间方案，
其碰撞配方与最终小腿分段不同，不应拿来替代上表的最终复测结果。

复测示例，须使用新的输出路径，两条命令依次执行：

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /ssd/conda/envs/aar_unilab/bin/python tools/benchmark_pe02_training.py \
  --envs 4096 --threads 32 --warmup 5 --iterations 20 --output logs/pe02-benchmark-new.json

env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /ssd/conda/envs/aar_unilab/bin/python tools/benchmark_we11_training.py \
  --envs 4096 --threads 32 --warmup 5 --iterations 20 --output logs/we11-benchmark-new.json
```

PE02 测试工具的 `--no-native-pd` 可用于隔离同一模型的控制循环加速效果。

## 8192 环境正式短训

正式入口完成 8192 × 24 × 5 = **983040 条样本**，100 次 PPO 更新和 100 次
encoder 更新，含 TensorBoard、checkpoint 与 ONNX/release 导出。
`/usr/bin/time -v` 测得该进程全程峰值 RSS 为 **7.006 GiB**，退出状态为 0。
这次用于验证可运行性，未做充分预热与重复，不能据此认定 8192 比 4096 更快。

checkpoint：
`logs/pe02-optimization/formal8192/2026-09-16_19-23-08_420132_mujoco/model_5.pt`。
release：`releases/pe02/pe02_flat/2026-09-16_19-23-08_420132_mujoco/`。
命令、运行输出及内存记录见 `logs/pe02-optimization/formal8192.{log,time}`，
计数核对见 `formal8192_report.json`。

本机训练的性能起点：

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /ssd/conda/envs/aar_unilab/bin/python scripts/train_pe02.py \
  algo.num_envs=4096 training.mujoco_threads=32
```

默认环境数仍保留原始基准的 8192，未静默降低；训练启动的可用内存检查保留。

## 验证与限制

- 48 项 PE02 / PE01 相关测试通过，覆盖质量惯量、源资产、镜像、足端保护、
  小腿间隙、训练、续训与部署合同。
- C++ 与 Python PD 在 1/3 个线程、两种速度反馈模式、不同延迟、外力、部分重置
  和恢复状态下逐步对照，状态、传感器、力矩和 FIFO 的容差为 1e-10。
- 直接比较改造前后的质量、惯量、关节、执行器和 home 数组，全部一致；原始及
  视觉 STL 字节不变。碰撞重复构建一致，URDF → MJCF 往返核对通过。
- 同一批 256 个正式训练初始姿态，新旧模型均无超过 0.1 mm 的自碰撞；均有
  140 个姿态穿地，最大约 49.9 mm。性能优化没有修复原初始化采样的问题，
  对照数据见 `logs/pe02-optimization/reset_geometry_comparison.json`。
- 最终 release 的 Python/C++ 1、10、100 步闭环对照最大误差均为 0。
- WE11 flat/rough 均重新组合、初始化、执行 3 步，观测保持字典且数值有限。
- 全仓库非 slow 测试 **250 项通过、2 项预期失败**，原生 CTest 6 项通过。
  全仓库静态检查仍有原有的 12 个格式文件、7 项 Ruff
  问题及 `playback.py:35` 的 1 项 mypy 问题，本次修改未新增静态检查错误。

完整几何和物理核对保存在
`src/unilab/assets/robots/pe02/analysis/collision_optimization.json`；
检查日志在 `logs/pe02-optimization/`。

这次完成的是性能优化。初始化姿态扰动、PD 参数适配、终止/奖励验收仍需独立
处理；5 轮策略不能作为稳定站立/行走策略使用。新资产指纹与碰撞 v1 不同，
旧 checkpoint 不支持直接续训到新资产；旧 release 继续使用各自冻结模型。
