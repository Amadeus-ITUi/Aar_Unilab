# Aar_Unilab

独立的多样机 MuJoCo 训练、Python 回放和 C++ Sim2Sim 仓库。当前正式主线是
WE11 Flat/Rough/Getup；PE01 使用独立的观测、网络和算法适配器。

## 环境

所有命令都从仓库根目录执行。首次安装：

```bash
bash tools/install_environment.sh
```

新 clone、移动仓库或切换副本后，重新绑定并激活环境：

```bash
bash tools/rebind_environment.sh
source tools/activate_environment.sh
```

## 训练

WE11 使用统一入口，通过 `task=flat|rough|getup` 选择任务：

```bash
python scripts/train.py \
  robot=we11 task=flat observation=we11_v2_145 \
  policy=we11_mlp algorithm=rsl_rl_ppo simulator=mujoco \
  algo.num_envs=4096 algo.max_iterations=50000 training.no_play=true
```

训练结果位于 `logs/rsl_rl_ppo/<RuntimeTaskName>/<run_id>/`。续训时在原命令后
追加：

```text
algo.load_run=<run目录或model_N.pt> algo.checkpoint=-1
```

PE01 当前使用独立的 custom PPO 适配器；训练结束会同时生成 checkpoint、ONNX
和 release：

```bash
python scripts/train.py \
  robot=pe01 task=pe01_flat observation=pe01_legacy \
  policy=pe01_encoder_mlp algorithm=pe01_custom_ppo simulator=mujoco \
  training.steps=128 training.device=cpu
```

## Python 回放

回放新 WE11 模型时，`task` 和 `observation` 必须与训练时一致：

```bash
python scripts/play.py \
  robot=we11 task=getup observation=we11_v2_145 \
  policy=we11_mlp algorithm=rsl_rl_ppo simulator=mujoco \
  algo.load_run=logs/rsl_rl_ppo/DR002JoystickGetupWE11/<run_id>/model_N.pt \
  training.play_render_mode=interactive training.play_steps=500
```

当前 WE11 的 Flat、Rough 和 Getup 均统一使用 `we11_v2_145`。无窗口检查使用
`training.play_render_mode=none`。

PE01 回放：

```bash
python scripts/play.py \
  robot=pe01 task=pe01_flat observation=pe01_legacy \
  policy=pe01_encoder_mlp algorithm=pe01_custom_ppo simulator=mujoco \
  checkpoint=logs/pe01_custom_ppo/pe01/pe01_flat/<run_id>/model_1.pt
```

## 模型转换和 C++ Sim2Sim

WE11 通过无窗口 Play 将 checkpoint 导出为 ONNX。文件生成在 checkpoint
所在目录：

```bash
AAR_EXPORT_POLICY=1 python scripts/play.py \
  robot=we11 task=getup observation=we11_v2_145 \
  policy=we11_mlp algorithm=rsl_rl_ppo simulator=mujoco \
  algo.load_run=logs/rsl_rl_ppo/DR002JoystickGetupWE11/<run_id>/model_N.pt \
  training.play_render_mode=none training.play_steps=1
```

构建当前正式 WE11 Getup release，并用 C++ 回放：

```bash
python tools/build_we11_sim2sim_release.py \
  --destination /ssd/conda/cache/aar_unilab-native/we11-getup-release

sim2sim/build/aar_sim2sim \
  /ssd/conda/cache/aar_unilab-native/we11-getup-release \
  --interactive --steps 1000000
```

PE01 release 位于 `releases/pe01/pe01_flat/<run_id>/`，可直接传给同一个
`aar_sim2sim`。C++ 支持渲染、摄像机、鼠标拖动力、手柄、图表和 CSV/JSONL
telemetry，详见 [sim2sim/README.md](sim2sim/README.md)。

## 验证与资料

```bash
bash tools/validate_installation.sh --all
```

- WE11 训练合同与交接：[WE11_HANDOVER.md](WE11_HANDOVER.md)
- 仓库改造记录：[docs/仓库规整与训练复用性改造计划.md](docs/仓库规整与训练复用性改造计划.md)
- Deploy、旧 Play 和 PE01 前身参考：`references/`

`references/` 中的内容只用于对照，不是训练或 Sim2Sim 的运行依赖。
