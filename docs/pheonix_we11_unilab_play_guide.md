# Pheonix-WE11：UniLab 训练与 Play 手柄验证指南

本指南只覆盖日常最快闭环：

```text
UniLab 训练 checkpoint (.pt)
  -> UniLab 导出 actor ONNX (.onnx)
  -> Play 加载同一份 ONNX
  -> MuJoCo + 手柄观察效果
```

**此闭环不需要 MNN、不需要 artifacts 目录，也不需要树莓派。** 只有某个 ONNX 已在
Play 验收并决定上真机时，才进入 Deploy 的 ONNX -> MNN 流程。

## 1. 先明确两个工作区的职责

| 工作区 | 做什么 | 不做什么 |
| --- | --- | --- |
| `~/ssd/Pheonix/UniLab` | PPO 训练、checkpoint 管理、ONNX 导出 | 手柄交互式仿真 |
| `~/ssd/Pheonix/Play` | 加载 ONNX、MuJoCo 可视化、手柄验证 | PPO 训练、真机 CAN |

UniLab 的 MuJoCo playback 没有 `interactive` 渲染模式；不要对 UniLab 使用
`training.play_render_mode=interactive`。可见、可手柄操作的仿真只在 Play 中完成。

当前策略合同必须始终一致：`obs[1,135] -> act[1,6]`，动作关节顺序为
`[左大腿, 左小腿, 左轮, 右大腿, 右小腿, 右轮]`。不要把不同 task、不同观测维度或不同
关节顺序的 ONNX 直接放入 Play。

## 2. UniLab：环境与冻结基线

每个新终端先进入 Conda 环境：

```bash
cd ~/ssd/Pheonix/UniLab
source /data/miniconda3/etc/profile.d/conda.sh
conda activate unilab_cuda
```

冻结 WE11 checkpoint 的完整性检查：

```bash
(cd models/we11 && sha256sum -c SHA256SUMS)
```

输出两个 `成功` 后，说明归档的 flat/rough 基线文件没有损坏。

## 3. 训练：先 smoke，再正式训练

### 3.1 Smoke：只确认训练链路

这是最先运行的命令；它只有 5 个 iteration，不会产生可上真机的策略：

```bash
cd ~/ssd/Pheonix/UniLab
source /data/miniconda3/etc/profile.d/conda.sh
conda activate unilab_cuda

python -u scripts/train_rsl_rl.py \
  task=dr002_joystick_flat_we11/mujoco \
  training.device=cuda:0 \
  training.no_play=true \
  training.play_render_mode=none \
  training.logger=tensorboard \
  algo.num_envs=64 \
  algo.num_steps_per_env=24 \
  algo.max_iterations=5 \
  algo.save_interval=5 \
  algo.run_name=pheonix_smoke \
  algo.algorithm.enable_compile=false
```

查看新 checkpoint（目录名是时间戳，`algo.run_name` 不会出现在目录名中）：

```bash
find logs/rsl_rl_ppo/DR002JoystickFlatWE11 -type f -name 'model_*.pt' \
  -printf '%TY-%Tm-%Td %TT %p\n' | sort
```

### 3.2 正式训练：先选 flat 或 rough

首次熟悉训练建议使用 flat；它不等于 rough/扑翼扰动策略。`50000 iteration` 是当前 YAML
的上限，不是第一次训练的必需目标。当前一次 iteration 从 `4096` 个并行环境各收集
`24` 条 transition，即 `98,304` 条 RL transition；50000 iteration 约为
`4,915,200,000` 条 transition，适合作为长时间训练上限，不适合作为首次闭环的起点。

建议阶段性训练：先 `100`（检查 reward、checkpoint、ONNX/Play 兼容），再 `500` 或
`1500`（观察行为是否已有改进），确认奖励和行为都正常后才决定是否继续增加。下面是
100 iteration 的第一轮 flat：

```bash
python -u scripts/train_rsl_rl.py \
  task=dr002_joystick_flat_we11/mujoco \
  training.device=cuda:0 \
  training.no_play=true \
  training.play_render_mode=none \
  training.logger=tensorboard \
  algo.max_iterations=100 \
  algo.save_interval=100 \
  algo.run_name=pheonix_flat_v0
```

每个 transition 指的是一次 RL environment step（本任务为 policy/control 时间尺度），
不是 MuJoCo 的单个 400 Hz physics substep。PPO 会对这一批 transition 再进行多轮优化，
所以梯度更新次数不等于 `98,304`。

要训练 rough，只替换 task 和 run 名；rough 会采用 rough task 自己的地形、奖励与随机化合同：

```bash
python -u scripts/train_rsl_rl.py \
  task=dr002_joystick_rough_we11/mujoco \
  training.device=cuda:0 \
  training.no_play=true \
  training.play_render_mode=none \
  training.logger=tensorboard \
  algo.max_iterations=100 \
  algo.save_interval=100 \
  algo.run_name=pheonix_rough_v0
```

不要把 5-iteration smoke checkpoint 当作可用策略，也不要在训练期间修改 task 的 action
scale、PD、command delay、观测历史长度或网络尺寸；这些都必须与 ONNX、Play 和 Deploy
同步。

### 3.3 看训练曲线

另开一个 UniLab 终端：

```bash
cd ~/ssd/Pheonix/UniLab
source /data/miniconda3/etc/profile.d/conda.sh
conda activate unilab_cuda
tensorboard --logdir logs/rsl_rl_ppo --port 6006
```

浏览器打开 `http://localhost:6006`。优先看 episode reward、policy/value loss 与采样速度；
不要只凭最后一个 checkpoint 文件存在就判定训练成功。

## 4. 导出选定 checkpoint 为 ONNX

选定一个 checkpoint 后，用与它**相同的 task** 导出。例如归档 rough 基线：

```bash
cd ~/ssd/Pheonix/UniLab
source /data/miniconda3/etc/profile.d/conda.sh
conda activate unilab_cuda

python -u scripts/train_rsl_rl.py \
  task=dr002_joystick_rough_we11/mujoco \
  training.device=cuda:0 \
  training.play_only=true \
  training.play_render_mode=none \
  training.play_env_num=1 \
  training.export_jit=false \
  algo.load_run=models/we11/rough/model_1500.pt
```

这个无界面命令正常时可能只打印 `Using device: cuda` 后返回；它会将 `policy.onnx` 写到
checkpoint 的同级目录。对新模型，把 `algo.load_run` 换为该 `model_N.pt` 的**绝对路径**，
同时保持正确的 `task=`。

导出后确认文件与 hash：

```bash
ls -lh <checkpoint所在目录>/policy.onnx
sha256sum <checkpoint所在目录>/policy.onnx
```

当前 `MlpAdaptModel` 在此环境不能稳定导出 TorchScript；`training.export_jit=false` 是正确
设置，日常 Play/Deploy 只交付 ONNX/MNN。

## 5. Play：替换 ONNX 并手柄验证

### 5.1 一次性构建 Play

Play 使用系统 C++ 依赖，不使用 UniLab Conda 的 C++ 库。首次构建或旧 build 曾绑定 Conda
yaml-cpp 时执行：

```bash
cd ~/ssd/Pheonix/Play
conda deactivate 2>/dev/null || true
/usr/bin/cmake -S . -B build -U yaml-cpp_DIR \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -Dyaml-cpp_DIR=/usr/lib/x86_64-linux-gnu/cmake/yaml-cpp
/usr/bin/cmake --build build --target rl_sim_mujoco --parallel "$(nproc)"
```

### 5.2 日常最快模型切换

Play 固定加载：

```text
~/ssd/Pheonix/Play/policy/dr002/we11/policy.onnx
```

第一次替换时保存归档基线；之后每次只覆盖运行模型即可：

```bash
cd ~/ssd/Pheonix/Play
cp -n policy/dr002/we11/policy.onnx \
  policy/dr002/we11/policy.onnx.baseline
cp <UniLab导出的policy.onnx绝对路径> \
  policy/dr002/we11/policy.onnx
sha256sum policy/dr002/we11/policy.onnx
```

这是本地仿真试验，**不需要**修改 MNN 或树莓派。要恢复归档策略：

```bash
cp policy/dr002/we11/policy.onnx.baseline \
  policy/dr002/we11/policy.onnx
```

### 5.3 运行与按键

先使用 `0 Hz`，即不加载实测 wrench/翼角 CSV 的平地回放：

```bash
cd ~/ssd/Pheonix/Play
./scripts/play_we11.sh 0
```

可指定手柄设备；程序默认搜索 `/dev/input/js0` 到 `/dev/input/js3`：

```bash
RL_SAR_JOYSTICK=/dev/input/js0 ./scripts/play_we11.sh 0
```

DR002 的 Play 按键与真机不同：

| 操作 | 手柄 | 键盘 |
| --- | --- | --- |
| 启动 policy | `RB + 方向键上`，或停止状态下 `RB + X` | `1` 或 `2`；停止状态下 Enter |
| 暂停/passive | `LB + X`，或运行状态下 `RB + X` | `P`；运行状态下 Enter |
| 回到初始姿态并停止 | `A` | `0` |
| MuJoCo reset 并停止 | `RB + Y` | `R` |
| 前进/后退 | 左摇杆 Y | — |
| 偏航 | 右摇杆 X | — |

`./scripts/play_we11.sh 1`、`2`、`3` 会分别加载归档实测的 1/2/3 Hz wrench 与翼角 CSV；
它们用于评估原始扰动合同。新模型的第一轮检查坚持使用 `0`，确认基础行为后再测试这些
扰动模式。

没有手柄而只检查 ONNX 能否启动时才使用：

```bash
RL_SAR_PLAY_AUTOSTART=1 ./scripts/play_we11.sh 0
```

正常手柄测试不要设置 `RL_SAR_PLAY_AUTOSTART=1`。

## 6. 何时才进入 Deploy

只有同时满足以下条件，才值得进入树莓派流程：

1. checkpoint 的 task、来源 commit 与 ONNX hash 已记录；
2. Play 的 `0 Hz` 手柄表现符合预期；
3. 需要时，Play 的 1/2/3 Hz 扰动回放也已检查；
4. 你明确决定将此版本作为真机候选。

届时才执行 ONNX -> MNN、树莓派安装与架空 standby/policy 验证。训练过程中的每一个
checkpoint 都不应自动进入 Deploy。
