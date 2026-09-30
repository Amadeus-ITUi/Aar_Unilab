# PE03 辨识后重新训练

本实验从零训练策略，使用 `pe03_gain_matrix_test01` 的辨识结果替换旧执行器参数。
不补采扫频，不改 WE11 基线，不覆盖 PE03 旧训练配置、旧权重或树莓派当前默认模型。

## 当前训练延迟配置：20 ms + 0–10 ms

按用户要求，`gait_identified` 现在使用 20 ms 基础延迟，叠加 0–10 ms 的随机偏移。
每个并行环境在初始化时抽样一次，六关节共享；reset 不重新抽样，未加入逐帧抖动。
400 Hz 下实际总延迟为 20、22.5、25、27.5、30 ms。延迟范围不随现有课程强度缩小。
普通回放关闭随机化，`play.delay_ms=0`，总延迟为 20 ms；导出也保存该 20 ms 仿真延迟。
实机部署不额外人为等待。惯量、阻尼、摩擦、Kp/Kd 仍沿用统一 24 ms 辨识版本；
这次是训练鲁棒性配置调整，不是重新辨识，原始 candidate 及其哈希保持不变。
左右独立辨识结果仍是单独候选，尚未接入此训练配置。

```bash
bash tools/train.sh pe03_gait_fixed +experiment=gait_identified algo.max_iterations=500
```

新日志目录为 `logs/pe03_gait_identified_delay20_30`，从零训练。此次只更新配置，未启动训练。
下文的 24 ms 辨识结果与 `test01` 运行记录保留为历史依据。

本次变更验证：相关 12 项测试通过，包括 Python/native 的共享随机延迟、reset 保持、
导出及 Python/C++ 回放一致性、WE11 flat/rough 初始化和 step。全量非 slow 测试为
665 passed、2 xfailed、4 个既有失败；失败仍为旧课程上限预期与 references 嵌套 Git 元数据。
mypy 通过，变更测试文件的 Ruff 与 `git diff --check` 通过；全库保留原有 7 个 lint
问题和 11 个格式问题。旧 test01 的配置快照仍是基础 24 ms、额外延迟为零。

## 初始化计时开销修复

最新 `2026-09-30_19-31-34_964539_mujoco` 的普通迭代约 3.23 秒，其中采样约
2.64 秒、策略更新约 0.58 秒。固定延迟的 test01 已有同样开销，非新增延迟随机化导致。
根因是执行器常数刷新创建 Python `MjData`，自动安装了进程全局的原生 `mjcb_time`；
MuJoCo 内部大量读取时钟，使物理回放变慢。公开的 `get_mjcb_time()` 对该原生回调仍
返回 None，不能据此判断内部计时未开启。

刷新逻辑现使用 C API 分配和释放临时 data，保留 `mj_setConst` 更新模型常数，并保持
用户已有的计时回调不变。辨识参数、限幅规则、延迟与物理步长未改变。
计时开关对照中，状态、传感器、力矩、关节速度与 FIFO 数值完全一致。
修复后同为 4096 环境、24 步的 CUDA runner 采样实测约 1.64 秒；加上此前约 0.58 秒
更新耗时，预计普通迭代约 2.2 秒，尚非修复后的完整训练实测。未恢复训练或更新策略权重。
诊断数据与测试说明见 `artifacts/pe03_delay20_30_performance/summary.json`。
修复相关 13 项测试通过；全量测试 666 passed、2 xfailed，仍为上述 4 个既有失败。
mypy 与变更文件 Ruff 通过，全库 Ruff 仍为上述既有问题。

## 统一延迟与分组延迟

共同通信链路和下位机调度通常应建模为共同延迟。扫频拟合的 **等效延迟** 还会吸收
电机内部响应、反馈时间偏差和未建模动力学，不能解释为各电机实测通信延迟。
原先拟合按 hip/thigh/calf 三组共享左右参数，并不是六个独立通信延迟。

使用 A/B 六次扫频重新拟合统一延迟及惯量、摩擦，C 三次只用于冻结后的验证。
坐标搜索两轮，统一延迟为 24 ms。第一轮 calf 达到搜索次数上限，第二轮三个关节组
均收敛；这是有界局部搜索，不是全局最优或统计置信区间。

| C 组验证，左右汇总 | 分组延迟位置 RMSE | 统一延迟位置 RMSE | 分组速度 RMSE | 统一速度 RMSE |
|---|---:|---:|---:|---:|
| hip | 0.02324 rad | 0.02327 rad | 0.5202 rad/s | 0.5369 rad/s |
| thigh | 0.02187 rad | 0.02204 rad | 0.4071 rad/s | 0.4346 rad/s |
| calf | 0.01066 rad | 0.01675 rad | 0.1638 rad/s | 0.3154 rad/s |

统一模型 A/B 的按幅值归一化位置 MSE 从 0.01159 增至 0.01228，约增加 6%。
采用它作为第一次重新训练的较简单模型，接受 calf 拟合精度下降的代价；保留分组模型
供后续对照。不能由此认定真实延迟就是 24 ms，也不能宣称分组模型只是在过拟合。

结果及图表：`artifacts/pe03_gain_matrix_test01/identification/shared_delay_comparison/`。
命令：

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 /ssd/conda/envs/aar_unilab/bin/python \
  -m scripts.identification.compare_delay_models
```

## 辨识来源模型与历史 test01

| 参数 | hip | thigh | calf |
|---|---:|---:|---:|
| 总 armature，kg·m² | 0.00600097 | 0.00729777 | 0.01796255 |
| 被动 damping，N·m·s/rad | 0.00007128 | 0.000000575 | 0.00147941 |
| frictionloss，N·m | 0.00010135 | 0.00010061 | 0.15242735 |
| 统一等效延迟 | 24 ms | 24 ms | 24 ms |
| Kp | 4.3 | 4.3 | 4.9 |
| Kd | 0.34 | 0.34 | 0.24 |

左右镜像关节共用相同数值。armature 是覆盖旧值的总量，不是额外叠加量；CAD 的质量、
惯性仍保留。接近零的被动阻尼和摩擦不是对物理零值的证明，不能与控制器 Kd 混为一谈。

- 保留 PE03 的 400 Hz 物理/PD、50 Hz 策略、38×30 历史、6 个动作、0.25 动作尺度。
- 24 ms 在 400 Hz FIFO 中量化为 25 ms。原来的共同 20 ms 假设被替换，不能叠加。
- 400 Hz/25 ms 回放相对于 500 Hz/24 ms 拟合回放的位置轨迹差异约 0.0007–0.0025 rad。
- 关节速度使用 MuJoCo qvel 对应的编码器速度，而不是位置差分；树莓派适配器使用
  `RobotState.velocity_rad_s`。保持现有 0.1 速度归一化。
- 控制器使用 `min(1, effort_limit/(abs(P)+abs(D)))` 同比缩放，再合并 P/D；训练命令
  `dq_des=0`、`tau_ff=0`。随机化的力矩效率系数在缩放后应用，最后保留力矩上限。
- 物理参数在模型初始化时修改并调用 `mj_setConst`，不在 step/reset 中解析资产。
- 原 gait/reward/curriculum 保留，机械负载与观测噪声随机化保留。这轮不同时修改算法。

默认实验 `conf/pe03/experiment/gait_identified.yaml` 使用统一模型的物理参数，
训练延迟已按上文覆盖为 20 ms + 0–10 ms；
`gait_identified_grouped.yaml` 保留分组模型，使用独立日志目录。

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /ssd/conda/envs/aar_unilab/bin/python scripts/train_pe03.py +experiment=gait_identified
```

历史首轮 test01 使用固定 25 ms 仿真延迟，4096 环境、计划 1000 次迭代，
每 100 次保存和评估，不恢复旧权重。
实际运行路径、源文件快照及启动命令见 `artifacts/pe03_identified_training_test01/launch.json`。
辨识改善执行器响应建模，不保证地面接触、IMU、时序抖动和最终实机行走同时被解决。

该次训练和 `tools.finish_pe03_identified_run` 后处理进程已按用户要求停止。
最后记录到第 548 次迭代，最新保存为 `model_500.pt`；状态见
`artifacts/pe03_identified_training_test01/completion_status.json`。
后处理工具原设计为训练结束后依次执行完整评估、视频回放、
Python/C++ 对照、MNN 转换、候选包上传和树莓派单线程数值检查。任何步骤失败写入
`failed` 及错误原因；行为评估不通过标为 `completed_candidate_not_accepted`，即使
模型转换和上传成功也不标为通过。这个进程不切换默认模型、不重启服务、不发送电机命令。
已使用第 100 次检查点完整走通后处理流程，其状态正确标为“未通过行为评估”。
早期检查点只供流程验证，不是可部署的最终策略。

## 导出、检查与部署候选

导出在复制的 XML 中固化被动参数，不修改源资产；`pe03_runtime.json` 使用
`.identified.v1` 后缀，带共同/分组 FIFO 和缩放规则。旧 C++ 运行时应拒绝新 schema，
不能静默按旧模型回放。新运行时保留旧 release 的行为。

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 MUJOCO_GL=egl OMP_NUM_THREADS=1 \
  /ssd/conda/envs/aar_unilab/bin/python tools/evaluate_pe03_identified.py CHECKPOINT \
  --output OUTPUT --render
```

评估包括最大配置随机化强度的既有 64 回合检查、7 个固定低速命令，以及三个无自动
重置的回放视频。模拟通过与实机通过分开报告，不把训练奖励当作部署验收。

`tools/package_pe03_candidate.py` 生成 MNN、配套 `policy.yaml`、64 组数值对照及哈希。
`tools/check_pe03_mnn_parity.py` 可在树莓派用单线程运行，不调用 ROS、串口或电机服务。
生成的硬件 profile **没有额外延迟**，惯量和被动阻尼只属于仿真模型。

当前 Pi 的旧 policy.yaml 还保留部分旧 target limit（如 calf 1.99 rad），新训练使用
现有 Robot Profile 对齐的 2.02 rad 限位。因此不能只替换 MNN 而沿用未匹配的旧
policy.yaml。候选包保留完整配套配置，当前默认模型及服务不自动切换。

## 验证记录

- Python/native：分环境、分关节延迟、P/D 抵消时的同比缩放、reset/restore 一致。
- 训练模型与导出 XML 的轨迹一致；Python/C++ 1、10、100 步轨迹一致。
- 第 100 次检查点在 Pi 的 64 组 ONNX/MNN 对照通过，最大动作差 1.63e-5；
  最终权重还必须重新执行该检查。
- 保留旧 PE02 native PD 以及 PE03 旧 release 行为。
- WE11 flat/rough 均 compose、initialize、step 通过。
- 全量非 slow 测试：659 passed、2 xfailed、4 个已有失败。三个失败是 PE03 旧课程
  测试期望 max_level=1，而当前配置是 0.7；另一个是 references 中已有嵌套 Git 元数据。
- mypy 180 个源文件通过。全库 Ruff 仍有 7 个已有 lint 问题、11 个已有格式问题；
  本次修改文件的检查通过。不改动这些无关基线问题。
