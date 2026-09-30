# PE05 训练、回放与 TensorBoard

PE05 使用与 PE03 固定步态相同的命名方式：`pe03_gait_fixed` → `pe05_gait_fixed`。
本次只调整使用入口、模型选择和日志组织；19 项奖励、观测、控制、网络、课程及终止条件不变。

## 常用命令

```bash
# 正式训练；现有默认仍为 4096 环境、15000 轮，每 100 轮保存/评估。
bash tools/train.sh pe05_gait_fixed

# 回放最新正式模型；checkpoint 默认就是 -1。
bash tools/train.sh pe05_gait_fixed mode=play

# 启动时暂停。
bash tools/train.sh pe05_gait_fixed mode=play play.paused=true

# 零指令原地踏步。
bash tools/train.sh pe05_gait_fixed mode=play play.command_source=fixed 'play.command=[0,0,0]'

# 无窗口短回放。
bash tools/train.sh pe05_gait_fixed mode=play play.render=none play.plot=false play.steps=10 play.command_source=fixed

# 只查看最终命令，不开始训练。
bash tools/train.sh pe05_gait_fixed --dry-run
```

旧命令 `bash tools/train.sh pe05 ...` 是 `pe05_gait_fixed` 的兼容别名，选择相同配置和日志根目录。
`pe05_smoke` 继续作为独立的 32 环境、10 轮工程测试入口，新输出位于
`logs/pe05_gait_fixed_smoke`，不进入正式模型自动选择。

正式训练和回放的日志根目录统一为 `logs/pe05_gait_fixed`。`checkpoint=-1` 先按训练目录名
中的开始时间选最新 run，再按 `model_<数字>.pt` 的轮次选择模型；不会因复制或触碰旧文件
而选错模型。忽略非训练目录和临时文件，最新 run 尚未保存模型时明确报错，不退回旧 run。
也可显式传 `checkpoint=<完整路径>`。启动回放时打印解析后的绝对路径。

回放采用 checkpoint 保存的训练配置。步态和控制延迟默认仍由 checkpoint 决定，只有显式
`play.gait=...` 或 `play.delay_ms=...` 才覆盖。PE05 的 3 cm 摆高不随 UI 对齐而变成 PE03 的值。

默认手柄、直接运行、不弹曲线窗口、无限步数。快捷键与 PE03 共用：空格/P 暂停，
R/Backspace 重置，N 在暂停时推进一步，C 相机对准，Q 退出；手柄 Start 暂停、RB+Y 重置。
`play.plot=true` 开启曲线窗口。正式 profile 的遥测目录为 `logs/play/pe05/pe05_flat/fixed`。

## TensorBoard

```bash
/ssd/conda/envs/aar_unilab/bin/python -m tensorboard.main \
  --logdir logs/pe05_gait_fixed/tensorboard_runs --host 127.0.0.1 --port 6006
```

在浏览器打开 `http://127.0.0.1:6006`。已有服务正在运行时直接刷新页面即可，不必重复启动。

新 run 名称：`PE05__pe05_flat-mujoco__gait_fixed__<训练开始时间>`。
事件文件仍存放在 `<run>/tensorboard/`，`tensorboard_runs/` 中使用相对链接建立描述性索引，
不复制事件。历史 run 保留原始实验身份，例如 `pe01_gait` 和 `velocity_bins`；不改 checkpoint
或保存的训练配置以重命名历史实验。

指标分组与 PE03 一致：`Loss/`、`Policy/`、`Perf/`、`Train/`、`Eval/`；保留 PE05 自有的
`reward/`、`raw_reward/`、`contact/`、`command_curriculum/` 等诊断。奖励分项仍为乘 dt 后的
每策略步贡献，数值没有为展示重新缩放。

为历史 run 重建描述性索引：

```bash
/ssd/conda/envs/aar_unilab/bin/python tools/index_pe05_tensorboard.py logs/pe05_gait_fixed
```

新训练自动建立索引，不需要启动 `sync_pe05_tensorboard.py`。该旧工具保留给早期未分组
事件日志使用，现代 run 直接使用原始 TensorBoard 事件即可。

## 历史产物整理

2026-09-28 清理前确认无 PE05 训练进程；没有启动新的训练或修改现有策略。
正式根目录保留两个采用新步态奖励的 run：

- `2026-09-28_17-09-58_288435_mujoco`：用户最近正式训练，最高保存轮次 500；全部中间 checkpoint 保留。
- `2026-09-28_16-51-56_802881_mujoco`：100 轮迁移验证，全部中间 checkpoint 保留。

旧奖励实验、早期训练、smoke/validation 目录和旧 release 统一归档到：

`logs/archive/pe05/2026-09-28_19-47-33_workflow_cleanup/`

最新正式 release 保留在原 `releases/pe05/pe05_flat/2026-09-28_17-09-58_288435_mujoco`。
保留两个旧根目录兼容链接 `logs/pe05_weighted`、`logs/pe05_pe01_gait`，使历史报告中的
模型路径继续可用；旧 TensorBoard 服务通过兼容链接读取新的正式索引。
正式索引不再展示归档实验。历史报告和只读 references 未清理。

迁移前后核对了 101 个模型/配置文件的 SHA256，全部一致；删除文件数为 0。
本次是目录整理与归档，不释放这些模型占用的磁盘空间。
完整路径映射、大小与指纹见 [清理清单](assets/pe05_workflow/cleanup_manifest.json)，归档目录
本身也保存 `manifest.json`。恢复时根据映射反向移动目录；保留 run 的归档位置是指向正式目录的链接。

## 验证

- `pe05` 与 `pe05_gait_fixed` 两种 profile 生成相同训练合同；最近真实 model_500.pt 仍可按新 profile 续训。
- 实际执行了最新 checkpoint 的 10 步无窗口回放，启动路径和遥测正常。
- TensorBoard HTTP 查询确认正式索引仅展示保留的两个 run；旧服务无需重启。
- 完整非 slow pytest：634 passed、2 xfailed、4 个既有失败（PE03 robustness 三项、只读 references 的嵌套 Git 检查一项）。
- 另一次并行聚焦测试遇到 PyTorch 导入时的 ApproximateClock 时钟断言，同项在完整测试及单独复跑均通过，未修改实现绕过。
- mypy 179 个源文件通过；改动文件 Ruff/format、git diff --check 通过。全仓库仍有此前 7 个 Ruff 问题和 11 个待格式化文件。

检查结果与日志见 [validation.json](assets/pe05_workflow/validation.json)。本次没有改动环境、奖励、控制或网络，也没有启动新的正式训练。
