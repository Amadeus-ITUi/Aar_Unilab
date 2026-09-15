# WE11 最终策略 Checkpoint

负责人：汪成浩

本目录冻结一个 Flat 和一个 Rough `model_1500.pt`，用于训练复现、评估和重新导出。
模型文件直接保存在 Git 中；完整性以同目录 `SHA256SUMS` 为准。

## Flat

```text
file: flat/model_1500.pt
source run: DR002JoystickFlatWE11/2026-07-30_21-41-17_mujoco
iteration: 1500
task: dr002_joystick_flat_we11/mujoco
sha256: 53aa1e447568ebf9de22ab04967b43ad9c7d9a38d8bd76627e080c3f75be28b7
```

## Rough

```text
file: rough/model_1500.pt
source run: DR002JoystickRoughWE11/2026-07-31_13-01-10_mujoco
iteration: 1500
task: dr002_joystick_rough_we11/mujoco
sha256: f7f80dfae9718584e31fc66d66a4821e95ff4510f6aa37f835e39a983066e232
```

Rough checkpoint 是 `Walking_Eagle-Play_final` 中
`policy/dr002/we11/policy_model_1500.onnx` 的来源模型。

## 校验

从仓库根目录执行：

```bash
(cd models/we11 && sha256sum -c SHA256SUMS)
```

## 使用边界

- 加载时必须选择与 checkpoint 对应的 Flat/Rough task。
- 本目录冻结的 Flat/Rough checkpoint 使用显式的 `we11_legacy_135`
  回放合同：actor 输入为 135D、action 为 6D；Flat/Rough critic 分别为
  147D/334D。Python Play 只加载 actor，因而不把历史 critic 合同错误地
  带入当前环境。当前训练及 Getup 保持 `we11_v2_145`，两种合同不得混用。
- 关节顺序、PD、action scale/clip、command-delay 和频率合同见
  `WE11_HANDOVER.md`。
- 观测语义、网络结构或动作维度改变后，不得把这两个 checkpoint 当作兼容模型。
