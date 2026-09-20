# PE03 WTW 步态改造验证记录

日期：2026-09-18。使用 `/ssd/conda/envs/aar_unilab` 环境。

本次完成的是新任务、训练和部署链路的工程验证。尚未训练出通过步态验收的策略，也没有将任何工程短训 checkpoint 标记为合格的第一阶段模型。

## 验证结果

| 检查 | 结果 |
|---|---|
| 新步态专项测试 | 19 项通过；包括相位/支撑占比、完整历史路径、速度分支梯度隔离、重置、失败计时、课程、精确续训、阶段转换后的 PPO 更新、ONNX 导出与迁移 |
| 全仓非 slow pytest | 399 passed，2 xfailed；约 70 秒 |
| WE11 flat、rough | 两个任务均完成独立 compose / init / step，观测为字典且数值有限 |
| C++ 编译与 CTest | 编译成功，6 / 6 通过 |
| Python / C++ 闭环 | 固定步态及非默认步态连续过渡，在 1、10、100 步动作与状态对照中最大误差为 0 |
| 小规模固定阶段短训 | 32 环境 × 24 步 × 3 轮；保存 checkpoint、TensorBoard 和工程评估报告 |
| 默认规模训练 | 4096 环境 × 24 步 × 1 轮，共 98,304 样本；完成 20 次 PPO minibatch 更新及 20 次速度估计器更新 |
| 本次涉及 Python 文件 Ruff | 格式、静态检查通过 |
| `git diff --check` | 通过 |

默认规模检查在 RTX 4060 Laptop GPU 上运行，采样约 2.02 秒、网络更新约 1.37 秒，PyTorch 显存分配峰值约 1194 MiB。这是一次工程测量，不包含每 100 轮执行的正式评估耗时。

验证产物保存在 `logs/pe03-validation/gait_v4_20260918/`：

- `fixed/model_3.pt`：小规模固定阶段短训模型。
- `fixed/evaluation_3.json`：8 回合工程评估，不能用于解锁第二阶段。
- `default_4096/validation.json`：默认规模训练耗时、样本量和显存记录。
- `pytest_gait_final.log`、`pytest_full_final.log`：专项及全仓测试结果。
- `we11_smoke_final.log`、`cpp_transition.log`、`ctest.log`：环境和部署检查结果。
- `artifacts.json`：最终工程导出产物索引，明确标记 `gait_qualified=false`。

阶段转换的自动化测试只在 pytest 临时目录内构造合成验收记录，用于检查权重转移、优化器重建和后续更新；这些记录不是实际策略验收结果。

## 全仓已有静态检查问题

全仓 Ruff / mypy 已执行；以下问题与改造前一致：

- Ruff format：12 个已有文件需要格式化，主要位于 PACE、WE11 回放和旧后端代码。
- Ruff check：7 条已有问题，位于 PACE 脚本及 `sim2sim/we11_play/scripts/`。
- mypy：`src/unilab/base/backend/mujoco/playback.py:35` 的 `func-returns-value` 错误。

完整输出分别保存在 `ruff_format_final.log`、`ruff_check_final.log`、`mypy_final.log`，未改动这些无关文件。

## 后续训练验收

从 `bash tools/train.sh pe03_gait_fixed` 开始正式训练。每 100 轮执行固定种子的 64 个、每个最长 20 秒的评估回合；连续三次达到数值标准并完成对应 checkpoint 的回放确认后，才能通过 `training.stage_from` 转入可调阶段。

足端可行域来自离线采样，每条腿在另一条腿保持 home 时检查碰撞。它是静态参考范围，不保证所有动态双腿组合均无自碰撞；投影距离会记录，验收高度误差仍按原始命令计算。

命令、配置所有权和恢复方法见 [使用说明](PE03_WTW_GAIT.md)。

## 2026-09-20：非足端碰撞独立扣分

新训练的碰撞项改为逐部件计算，再在线性形式下加到指数聚合之后。默认每个接触合力超过 1 N 的非足端部件每策略步扣 0.1；多个部件相加，无整项裁剪和最终零截断。其余奖励、控制和终止条件沿用原配置。旧 checkpoint 缺少新配置块时继续使用原指数公式；新旧奖励合同不能混用严格 resume。

本次检查结果：

- 28 项 PE03 步态/导出专项测试通过，包括多碰撞边际代价、其他代价饱和时仍独立扣分、逐部件权重、阈值、旧公式和 resume 合同。
- 全仓非 slow 测试：408 passed、2 xfailed，约 69 秒。
- 两个 WE11 任务均完成 compose / init / step。
- 新配置完成 32 环境 × 24 步 × 3 轮工程短训，共 2304 样本、60 次 PPO minibatch 更新。
- 固定原 `model_500` 动作，原公式与新公式各运行 1000 策略步。观测、物理状态、终止结果逐步完全一致；平均每步奖励分别为 +0.0129543 和 −0.0124329，独立碰撞贡献平均为 −0.0254。这验证了错误触地轨迹在新目标下受到有效扣分，不代表已经训练出改进步态。
- 本次涉及文件的 Ruff 检查通过，`git diff --check` 通过。全仓仍有原有 12 个格式文件、7 条 Ruff 问题、1 条 mypy 错误，没有新增同类问题。

原始结果：`logs/pe03-validation/collision_additive_20260920/`，包括 `focused.log`、`pytest_full.log`、`we11_smoke.log`、`training.log`、`frozen_replay.json` 及静态检查输出。
