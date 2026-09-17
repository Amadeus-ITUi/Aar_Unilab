# PE01 完整训练迁移

正式入口 `scripts/train_pe01.py` 默认使用独立的 `pe01_v2`。环境、PPO、速度估计器、
配置和部署运行时分别归 PE01 所有，不导入或继承 PE02 实现。共用 catalog、
MuJoCo 批量后端、交互窗口和 release 打包等基础设施。

核对基准是 `references/pe01/legacy_training` 保存的原始 PE01 源码，提交
`17a4b93a7480233de07d52ccb5c790cfdb1d0286`。该目录保持只读，运行时不执行参考代码。
迁移代码保留 BSD-3-Clause 署名及 `LICENSE.pe01`。

## 参数与任务

| 项目 | PE01 正式迁移值 |
| --- | --- |
| 环境数 / 每环境 rollout / 轮数 | 8192 / 24 / 15000 |
| PPO epoch / minibatch | 5 / 4，每轮 20 次 PPO 更新 |
| 速度估计器 | 300 → 256 → 128 → 3，独立 Adam/MSE，每轮另更新 20 次 |
| Actor / Critic | 36 → 512 → 256 → 128 → 6；39 → 512 → 256 → 128 → 1 |
| 物理 / PD / 策略频率 | 400 / 400 / 50 Hz |
| 动作 | 相对默认角度的偏移，action scale 0.25，位置 PD |
| Kp / Kd | 每腿 4.3/4.3/4.9；0.34/0.34/0.24 |
| 动作延迟 | 25–50 ms，每环境初始化抽样 |
| 默认角度 | 左髋 -0.1、左大腿 0.6、左小腿 -1.2；右髋 0.1、右大腿 0.6、右小腿 -1.2 rad |
| reset 基座高度 / 高度奖励目标 | 0.32 m / 0.25 m，均来自原始 PE01 |
| 奖励 / 随机化 | 原版 18 项奖励；质量、惯量、质心、摩擦、PD、零位、延迟、IMU、推力 |
| episode / failure | 20 秒；失败累计超过 0.5 秒 |

观测布局、GAE 超时处理、encoder 梯度隔离、完整 checkpoint 续训和原版有效行为的
核对原则与[PE02 迁移审计](PE02_TRAINING_MIGRATION.md)相同，但两套实现独立维护。
PE01 的姿态、球形足端和 0.25 m 高度目标按 PE01 处理，没有套用 PE02 的 home。

原版的 `zero_command_prob=0.1` 因 `random > 0.1` 实际产生约 90% 零命令；这里
显式配置 `commands.zero_probability=0.9`。原版 gait offset 最终固定为 0.5。
修正观测时序、子集 reset 索引和 timeout bootstrap 等问题，保留有效超参数。

本次沿用仓库现有 PE01 简化 MJCF，并在任务 `scene.xml` 中加入原始姿态的 `home`
关键帧；机械零位、机器人碰撞和惯量没有重新拟合。该 MJCF 部分连杆质量来自
基础碰撞体的密度推导，不能把“训练系统迁移完成”理解为原始 CAD 动力学已精确复现。
PhysX restitution 与 MuJoCo 接触求解器也没有逐项等价映射。PE01 的球形足端支撑高度
用球心对地高度减半径计算，后端在构造时缓存几何数据，不在 step/reset 中读取资产。
沿用当前 MJCF 的自碰撞配置；原始 Isaac Gym 的禁用自碰撞没有直接复现。

## 使用

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 \
  /ssd/conda/envs/aar_unilab/bin/python scripts/train_pe01.py \
  algo.num_envs=4096 algo.num_steps_per_env=24 algo.max_iterations=15000
```

统一入口选择 `robot=pe01 task=pe01_flat observation=pe01_v2
policy=pe01_encoder_mlp algorithm=pe01_custom_ppo simulator=mujoco`。
控制参数归 `conf/pe01/task/pe01_flat.yaml`，网络和训练参数归 `conf/pe01/config.yaml`。
PE01 的模型内存与 PE02 不同，不能用 PE02 的 3.806 MB/模型估计 PE01。

续训用 `training.resume=<model_N.pt>`，保持原环境数、网络、任务及 PPO 合同一致；
`algo.max_iterations` 表示追加轮数。保存网络、两个优化器、计数、环境状态、随机数、
随机化、历史及延迟 FIFO。TensorBoard 与 JSONL 在 run 目录中。

回放：

```bash
/ssd/conda/envs/aar_unilab/bin/python scripts/train_pe01.py mode=play \
  checkpoint=logs/pe01_custom_ppo/pe01/pe01_flat/<run_id>/model_400.pt \
  play.render=interactive play.plot=true
```

训练结束自动导出完整 encoder+actor ONNX 和 `robot/pe01_runtime.json`；
`sim2sim/include/aar/pe01_runtime.hpp` 独立实现位置 PD、延迟及观测。
使用 `training.export=false` 可跳过导出。

旧 checkpoint 的 `pe01_legacy` / `pe01_v1` 部署合同保持原有网络、观测和力矩控制。
`observation=pe01_legacy` 或旧 `training.steps=128` 参数显式进入
`scripts/train_pe01_legacy.py`。PE01 正式训练不再使用 `training.steps`。

短训用于验证训练链路和续训，并不代表策略已经稳定站立或行走。

## 验证记录

- 通过统一 catalog 入口执行 4096 环境 × 24 步 × 5 轮 CUDA 短训：491520 条样本，
  PPO 与 encoder 各更新 100 次，生成 TensorBoard、周期 checkpoint 和最终 release。
- 从第 5 轮继续训练 1 轮，恢复至第 6 轮、589824 条样本，两个优化器均为 120 次更新。
  单元测试另外核对了 CPU 连续训练与分段续训结果完全相同。
- PE01 / PE02 / catalog 专项共 54 项通过；全仓库非 slow 测试 245 项通过，
  2 项已有预期失败。测试包含禁止导入 PE02/reference 后的训练、导出和异地加载。
- C++ 构建和 6 项 CTest 通过；PE01 默认配置及 25 ms 延迟 + 非零速度命令下，
  Python/C++ 1、10、100 步闭环轨迹最大误差均为 0。
- PE02 正式发布包的 Python/C++ 1、10、100 步回归最大误差为 0；WE11 flat/rough
  均重新组合配置、初始化并执行 3 步，观测字典及数值有限。
- 旧 PE01 最小入口仍完成短训导出，正式 checkpoint 完成无窗口 Python 回放。
- 新增/改动的 Python 实现通过 Ruff。全仓库仍有原有的 7 项 Ruff 错误、12 个
  文件格式问题，以及 `playback.py:35` 的 1 处 mypy 错误，本次没有修整无关文件。

短训 checkpoint：
`logs/pe01-migration/validation/2026-09-16_18-22-30_561774_mujoco/model_5.pt`。
续训 checkpoint：
`logs/pe01-migration/resume/2026-09-16_18-23-44_866978_mujoco/model_6.pt`。
原始检查日志与机器可读摘要保存在 `logs/pe01-migration/verification/`。

WE11 的 4096 内存实测、8192 外推与线程/环境数组合结果见
[WE11 测速报告](WE11_TRAINING_BENCHMARK.md)。
