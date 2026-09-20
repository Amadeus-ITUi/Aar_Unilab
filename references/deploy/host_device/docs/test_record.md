# Walking Eagle WE11 Deploy 测试与验收记录

负责人：汪成浩
归档日期：2026-08-04
功能冻结基线：`0635201a192ce40332edd16dc6237d9532418dfb`

## 1. 本次归档变更

本次只增加 README 交付入口、交接说明、接口说明、测试记录和 CHANGELOG，不修改 ROS2/CAN 代码、YAML、MNN 模型或原始扫频 CSV。

## 2. 软件侧最小验证

| 检查项 | 命令 | 预期结果 | 结果 |
|---|---|---|---|
| 文档引用 | `test -f docs/handover.md -a -f docs/interface.md -a -f docs/test_record.md` | 文件存在 | 通过 |
| Shell 语法 | `bash -n start_robot.sh setup_dual_can.sh env_init.sh` | 无语法错误 | 通过 |
| 扫频 Python | 读取源码并调用 Python `compile()` | 无语法错误 | 通过，1 个 Python 入口 |
| 扫频数据 hash | `(cd sweep_results_20260730 && sha256sum -c SHA256SUMS)` | 7 个 CSV 全部成功 | 通过 |
| Git whitespace | `git diff --check` | 无错误 | 通过 |

## 3. 硬件验收矩阵

| 检查项 | 当前状态 | 通过标准 |
|---|---|---|
| `can0` 配置与负载 | 待目标机复核 | 无 bus-off/error-passive，TX/RX 与协议响应一致 |
| 1～8 号 0x02 探测 | 待目标机复核 | 每个 ID/型号返回合法帧 |
| IMU | 待目标机复核 | `/IMU_data` 稳定约 200 Hz，轴向正确 |
| 1～6 状态 | 待目标机复核 | `/policy/joint_states` 200 Hz，默认位/方向正确 |
| 翼启动定位 | 待目标机复核 | `+90°/-90°` 目标轨迹完成，无撞限位；实际到位误差不作为放行条件 |
| standby | 待目标机复核 | 所有 command 为安全值，无异常抖动 |
| policy 架空 | 待目标机复核 | 无离线、裁切洪泛、过流和异常振动 |
| 扫频采集 | 历史数据已归档 | 单组运行、200 Hz、异常退出可禁用电机 |
| 真机地面运行 | 未在本次文档提交重测 | 低速到目标场景分级验收并留视频/日志 |

## 4. 失败即停止条件

出现 CAN bus-off、TX buffer full、反馈计数持续增长、任一路电机 offline、IMU 超时/轴向错误、位置限位连续裁切、电机过热、异常振动或急停不可用时，本次验收立即失败并退出控制。
