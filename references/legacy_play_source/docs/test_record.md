# Walking Eagle WE11 Play 测试与验收记录

负责人：汪成浩
归档日期：2026-08-04
功能冻结基线：`2a42994681e9ead46e0051fc54b1661851b18ee7`

## 1. 本次归档变更

本次只增加 README 交付入口、交接说明、接口说明、测试记录和 CHANGELOG，不修改 C++ runtime、launcher、ONNX、MJCF、地形或回放 CSV。

## 2. 最小验证

| 检查项 | 命令 | 预期结果 | 结果 |
|---|---|---|---|
| 关键文件 hash | `sha256sum -c SHA256SUMS` | 全部成功 | 已通过（归档前） |
| Shell 语法 | `bash -n build.sh scripts/play_we11.sh scripts/play_we11_level9.sh` | 无语法错误 | 通过 |
| 构建 | 在 `/tmp` 干净副本运行 `RL_SAR_BUILD_JOBS=2 ./build.sh` | 生成 `build/bin/rl_sim_mujoco` | 通过 |
| Git whitespace | `git diff --check` | 无错误 | 通过 |
| Flat smoke | `RL_SAR_PLAY_AUTOSTART=1 ./scripts/play_we11.sh 0` | 模型加载、场景启动、无依赖缺失 | 待接收方有显示/离屏环境验证 |
| Rough smoke | `RL_SAR_PLAY_AUTOSTART=1 ./scripts/play_we11_level9.sh rough 3 1.0` | level-9 场景和 3 Hz 数据正确加载 | 待接收方有显示/离屏环境验证 |

## 3. 接收方验收

| 检查项 | 状态 | 通过标准 |
|---|---|---|
| 代码可拉取 | 待接收人确认 | 正式分支可克隆 |
| 依赖可安装 | 待接收人确认 | Ubuntu 22.04 x86_64 完成构建 |
| ONNX 可加载 | 发布脚本 + Play runtime | 输入 145D、输出 6D，无 shape 错误 |
| Flat 回放 | 待接收人确认 | 0/1/2/3 Hz 均可启动 |
| Level-9 回放 | 待接收人确认 | flat/uphill/downhill/rough 均可启动 |
| 控制合同 | 待接收人确认 | 与模型来源的 PD、scale、clip、delay 一致 |
| 真机验证 | 不属于本仓库 | 转交 Deploy 后执行 |

## 4. 失败判定

SHA-256 失败、模型 shape mismatch、CSV/scene/mesh 缺失、关节顺序不一致、action 持续 clip、PD/delay 与训练不一致时，本回放不通过，不得作为真机模型来源。
