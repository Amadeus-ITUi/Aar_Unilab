# WE11 UniLab 测试与验收记录

负责人：汪成浩
归档日期：2026-08-04
功能冻结基线：`532646089dab4f174f7cdc7374190c11909b5fd2`

## 1. 本次归档变更

本次只增加 README 交付入口、交接说明、接口说明、测试记录和 CHANGELOG，不修改训练代码、配置、模型、资产或实验数据。

## 2. 最小验证

| 检查项 | 命令 | 预期结果 | 结果 |
|---|---|---|---|
| 文档引用 | `test -f docs/handover.md -a -f docs/interface.md -a -f docs/test_record.md` | 文件存在 | 通过 |
| Git whitespace | `git diff --check` | 无错误 | 通过 |
| WE11 focused tests | 见下方命令 | 全部通过 | 15 passed |

```bash
uv run pytest -q \
  tests/envs/locomotion/dr002/test_we11_pace_training.py \
  tests/envs/locomotion/dr002/test_we11_rough_training.py \
  tests/envs/locomotion/common/test_terrain_spawn.py \
  tests/terrains/test_mujoco_heightfield.py
```

测试出现 5 条同源 warning：本机未在该临时 worktree 构建 native mixed-PD 扩展，
环境自动回退为每个 physics substep 一次 BatchEnvPool 调用。测试本身全部通过；
正式训练前仍必须按 handover 构建 native 扩展并验证可用性。

## 3. 接收方验收

| 检查项 | 状态 | 备注 |
|---|---|---|
| 代码可拉取 | 待接收人确认 | 克隆正式分支 |
| 环境可安装 | 待接收人确认 | 目标机器执行安装流程 |
| 最小测试可运行 | 待接收人确认 | 执行 focused tests |
| Flat 仿真可启动 | 待接收人确认 | 至少完成短步数 smoke |
| Rough 仿真可启动 | 待接收人确认 | 至少完成短步数 smoke |
| 日志可输出 | 待接收人确认 | TensorBoard 可读取 event |
| 模型导出/parity | 待验收 | 与 Play/Deploy 合同逐项核对 |
| 真机低风险测试 | 不属于本仓库 | 由 Deploy 流程完成 |

## 4. 异常判定

出现观测/动作维度不匹配、NaN、模型加载 shape mismatch、关节方向异常、action 持续 clip、轮速/力矩持续饱和或 delay 语义不一致时，停止评估，不得将模型交给真机链路。
