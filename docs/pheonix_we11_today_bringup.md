# Pheonix-WE11：今日端到端 bring-up 教程

本文用于接手 WE11 后，在**不重新采集数据、不扫频、不重新做 PACE/Kp/Kd 辨识**的前提下，走通已有的训练、仿真回放和树莓派真机部署闭环。

## 命令验证状态（2026-08-18）

本文不把“可推测的命令”写成已验证步骤。以下命令已在当前 x86_64 训练/开发机
实际执行并成功：冻结 rough checkpoint 的 ONNX 导出、5 iteration 的 flat PPO
smoke 训练、`SHA256SUMS` 校验、Play 的 `rl_sim_mujoco` 重建，以及 ONNX 到 MNN
的转换。树莓派 arm64 的 ROS 构建、CAN、手柄、IMU 和真机动作必须在目标机上验收，
本文将它们明确标为目标机步骤，不能由当前电脑替代验证。

本文的目标是验证交付链路，而不是在一天内产出可自由地面运行的新策略：

```text
已有 WE11 PACE/PD/延迟/扰动数据
  -> UniLab 训练 smoke 或加载冻结 checkpoint
  -> 导出 ONNX
  -> Play 加载 ONNX 回放
  -> 转换 MNN
  -> 树莓派 Deploy 加载 MNN
  -> 真机 standby / 架空低风险 policy 验证
```

## 0. 范围、成功标准与安全边界

### 今天做什么

- 使用归档中已有的 WE11 PACE 参数、Kp/Kd、command-delay、实测 wrench 与翼角数据；
- 用冻结的 `model_1500.pt` 验证成熟策略；
- 跑一次小规模 PPO smoke，确认新的 checkpoint 能产生；
- 将一个确认兼容的 checkpoint 导出为 ONNX、在 Play 加载、转换为 MNN，并让树莓派加载；
- 在真实机器人上只按 `CAN -> 状态 -> standby -> 架空零速 -> 极低速` 的顺序验收。

### 今天明确不做什么

- 不运行 `scripts/sweeps/`；
- 不重新导入真机 CSV；
- 不运行 `scripts/pace/` 的 PACE 或 Kp/Kd 拟合；
- 不修改 `we11_pace_params.json`、训练 PD、关节零位或 CAN 电机 ID；
- 不在未完成架空和低风险验收前进行地面快速运动。

### 成功标准

1. UniLab 能加载冻结 checkpoint；小规模训练能写出 checkpoint。
2. ONNX actor 的接口为 `obs[1,135] -> act[1,6]`。
3. Play 能用该 ONNX 启动并接受手柄输入。
4. 树莓派上的 Deploy 能构建，并加载相同来源的 MNN。
5. 真机 1--8 号电机均可被只读状态探测；IMU、手柄、关节和翼角话题正常。
6. 真机能安全进入 standby；仅在架空、零速度确认后才尝试 policy。

> **硬性停止条件：** 任一电机没有状态反馈、CAN bus-off/错误计数增长、IMU/翼角超时、关节方向或默认位不确定、异常振动/过流/过热时，停止在当前阶段。不得以提高 offline threshold、放宽限位或跳过确认门来继续。

## 1. 工作区与角色

开发机上的两个工作区与树莓派上的 Deploy 工作区是不同产品，不应互相 `git switch`：

```text
开发机
~/ssd/Pheonix/UniLab   pheonix-we11/unilab-bringup
~/ssd/Pheonix/Play     pheonix-we11/play-bringup

树莓派（从零安装；不使用旧的 deploy_ws 仓库）
~/Pheonix/Deploy       pheonix-we11/deploy-bringup
~/Pheonix/artifacts/   模型交付暂存区
```

角色分工：

| 工作区 | 今天的职责 |
| --- | --- |
| UniLab | 使用已有参数训练/评估、生成 checkpoint、导出 ONNX |
| Play | 用 MuJoCo、手柄与 ONNX 验证回放行为 |
| Deploy | 在树莓派上构建 ROS2/CAN 链路、转换/加载 MNN、真机分级验收 |

每个待交付模型都要记录一个版本名，例如 `pheonix-we11-today-v0`，并记录：checkpoint、ONNX、MNN 的 SHA-256，来源 commit，task，PD/scale/clip/delay 与验证结果。

## 2. 首先确认冻结基线可导出，而不是新训练模型

冻结模型是今天真机 bring-up 的首选，因为它已经经过完整训练；小规模 smoke 模型只用于验证训练和导出管线，不应作为真机性能模型。

在 `~/ssd/Pheonix/UniLab`：

```bash
conda activate unilab_cuda
(cd models/we11 && sha256sum -c SHA256SUMS)
```

UniLab 不提供手柄交互式播放；它的 MuJoCo playback 只支持无界面运行或录制视频。
因此在这里仅做冻结 checkpoint 的无界面加载/导出检查：

```bash
python -u scripts/train_rsl_rl.py \
  task=dr002_joystick_rough_we11/mujoco \
  training.device=cuda:0 \
  training.play_only=true \
  training.play_render_mode=none \
  training.play_env_num=1 \
  training.export_jit=false \
  algo.load_run=models/we11/rough/model_1500.pt
```

此步骤确认 checkpoint、环境、actor 135D 输入、6D action 与 ONNX 导出合同；可交互、可见的手柄仿真验证只在第 6 节 Play 工作区执行。

## 3. 训练 smoke：只验证训练闭环

训练机应使用 CUDA；树莓派不承担 PPO 训练。以下命令刻意缩小规模和迭代次数，只验证：环境创建、采样、反向传播、日志和 checkpoint 写入。

```bash
cd ~/ssd/Pheonix/UniLab
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
  algo.run_name=pheonix_today_smoke \
  algo.algorithm.enable_compile=false
```

找到输出目录和 checkpoint：

```bash
find logs/rsl_rl_ppo/DR002JoystickFlatWE11 -type f -name 'model_*.pt' \
  -printf '%TY-%Tm-%Td %TT %p\n' | sort
```

当前 runner 用时间戳创建 run 目录；`algo.run_name` 写入训练日志，不会出现在目录名中。

这个模型通常没有足够训练预算，**只可做仿真/导出验证，不可进入真机 policy**。

`training.logger=none` 在当前固定的 RSL-RL 版本中不是有效 logger，不能使用；
保持 `training.logger=tensorboard`。本节命令已于 2026-08-18 在本工作区完成
5 个 iteration，并生成 `model_0.pt` 与 `model_4.pt`。

## 4. 导出 ONNX

训练脚本在 play 模式加载 checkpoint 时会导出 ONNX。当前 PyTorch/RSL-RL
组合不能稳定将 `MlpAdaptModel` 导出为 TorchScript，因此本分支默认禁用 JIT
导出；部署只使用 ONNX/MNN。对冻结 rough checkpoint 导出，使用：

```bash
cd ~/ssd/Pheonix/UniLab
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

导出文件会写入 checkpoint 所在目录。当前 RSL-RL 导出的 ONNX 文件名为
`policy.onnx`；仍应先确认文件和 hash，再进行交付：

```bash
find models/we11/rough -maxdepth 1 -type f -name '*.onnx' -print
sha256sum models/we11/rough/policy.onnx
```

对新训练 checkpoint，替换 `algo.load_run` 为该 `model_N.pt` 的绝对路径，并使用与其训练任务完全一致的 `task=`。

> 若日志提示 native mixed-PD extension 不可用，程序会使用正确但较慢的 Python
> fallback，因而可以完成本教程的命令验证；训练前仍应按仓库安装流程构建该扩展。

## 5. 建立不可覆盖的交付暂存目录

不要直接覆盖 Play/Deploy 的归档模型。先把选定 ONNX 放入版本化暂存目录：

```bash
mkdir -p ~/ssd/Pheonix/artifacts/pheonix-we11-today-v0
cp ~/ssd/Pheonix/UniLab/models/we11/rough/policy.onnx ~/ssd/Pheonix/artifacts/pheonix-we11-today-v0/policy.onnx
sha256sum ~/ssd/Pheonix/artifacts/pheonix-we11-today-v0/policy.onnx
git -C ~/ssd/Pheonix/UniLab rev-parse HEAD
```

同时记录：来源 checkpoint hash、task、actor 输入/输出维度、导出命令、UniLab commit。只有确认这些信息后，才可以把模型装入 Play 或 Deploy。

## 6. Play：加载 ONNX，并完成唯一的手柄交互仿真

Play 的归档模型位置为：

```text
~/ssd/Pheonix/Play/policy/dr002/we11/policy.onnx
```

在自己的 `pheonix-we11/play-bringup` 分支上，先备份再替换：

```bash
cd ~/ssd/Pheonix/Play
cp policy/dr002/we11/policy.onnx policy/dr002/we11/policy.onnx.baseline
cp ~/ssd/Pheonix/artifacts/pheonix-we11-today-v0/policy.onnx \
  policy/dr002/we11/policy.onnx
sha256sum policy/dr002/we11/policy.onnx
```

随后更新 `policy/dr002/we11/deployment_manifest.json` 中的来源与 SHA-256；不要让新模型沿用旧模型的来源元数据。

构建 Play。必须使用系统 C++ 依赖；若旧的 `build/` 曾在 Conda 下配置，请重新配置到系统 `yaml-cpp`：

```bash
cd ~/ssd/Pheonix/Play
conda deactivate
/usr/bin/cmake -S . -B build \
  -U yaml-cpp_DIR \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -Dyaml-cpp_DIR=/usr/lib/x86_64-linux-gnu/cmake/yaml-cpp
/usr/bin/cmake --build build --target rl_sim_mujoco --parallel "$(nproc)"
```

先运行无外力/翼角 CSV 的平地模式。这是整个流程中首次、也是唯一需要手柄交互的仿真步骤：

```bash
./scripts/play_we11.sh 0
```

正常手柄操作：`RB + 方向键上` 进入 policy；左摇杆上下控制前后，右摇杆左右控制偏航；`A` 回到初始姿态；`LB + X` 停止 policy。Play 的按键语义不同于真机 Deploy。

今日不验证扑翼扰动时，不运行 `1/2/3 Hz` CSV 回放，也不要设置 `RL_SAR_PLAY_AUTOSTART=1` 进行正常手柄测试。

## 7. Deploy：先在 x86 转换 MNN，再在树莓派构建/加载

不要把 ONNX 转 MNN 作为树莓派临场依赖。先在当前 x86_64 `Deploy` 工作区完成转换，
将已验证的 `.mnn` 和 hash 一起传给树莓派。仓库内的 x64 `MNNConvert` 已实际用
`models/we11/rough/policy.onnx` 转换成功，输入 `obs`、输出 `act`。当前归档中该
二进制缺少可执行位，因此先恢复该文件权限：

```bash
cd ~/ssd/Pheonix/Deploy
chmod u+x src/inference/thirdparty/MNNConverter/x64/MNNConvert \
  src/inference/thirdparty/MNNConverter/x64/MNNConvert.sh
mkdir -p ~/ssd/Pheonix/artifacts/pheonix-we11-today-v0
src/inference/thirdparty/MNNConverter/x64/MNNConvert.sh \
  -f ONNX \
  --modelFile ~/ssd/Pheonix/artifacts/pheonix-we11-today-v0/policy.onnx \
  --MNNModel ~/ssd/Pheonix/artifacts/pheonix-we11-today-v0/policy.mnn \
  --bizCode MNN
sha256sum ~/ssd/Pheonix/artifacts/pheonix-we11-today-v0/policy.onnx \
  ~/ssd/Pheonix/artifacts/pheonix-we11-today-v0/policy.mnn
```

`chmod` 会使 Deploy 工作区出现两个转换器文件的 mode 改动；这是归档权限缺失，不是模型
内容改动。要长期保留此转换路径，可在 Deploy 分支单独提交该 mode 修复。若转换
失败，不要改网络结构或在树莓派上临时安装未知版本的 `MNN` Python 包；先保留 ONNX
并解决转换器/版本兼容问题。

以下是**树莓派 arm64 从零安装**步骤，尚不能在当前 x86_64 机器上替代验证。假设树莓派
已安装 Ubuntu 22.04 arm64、网络可用。树莓派不需要、也不应配置 GitLab 凭据；不要使用
旧的 `~/deploy_ws/Walking_Eagle-Deploy`，新工作区唯一位置是 `~/Pheonix/Deploy`。

先在开发机把已经检出的 Deploy 分支打成不含 `.git` 的源码包。开始前确认 Deploy 没有
未提交的功能性改动；`git archive` 只会打包该分支已经提交的内容：

```bash
git -C ~/ssd/Pheonix/Deploy status --short
git -C ~/ssd/Pheonix/Deploy archive --format=tar \
  --prefix=Deploy/ pheonix-we11/deploy-bringup \
  | gzip -c > /tmp/pheonix-we11-deploy-bringup.tar.gz
sha256sum /tmp/pheonix-we11-deploy-bringup.tar.gz
git -C ~/ssd/Pheonix/Deploy rev-parse pheonix-we11/deploy-bringup
```

这个打包命令已在开发机验证，当前生成包约 21 MB。再创建树莓派目标目录并传包（把
`<pi-user>` 和 `<pi-host>` 替换为真实值）：

```bash
ssh esd@192.168.8.200 'mkdir -p /home/esd/Pheonix'
rsync -avP /tmp/pheonix-we11-deploy-bringup.tar.gz \
  esd@192.168.8.200:/home/esd/Pheonix/pheonix-we11-deploy-bringup.tar.gz
```

在树莓派解压一次。压缩包保留在 `~/Pheonix/` 作为这次安装的源码凭据：

```bash
mkdir -p ~/Pheonix
tar -xzf ~/Pheonix/pheonix-we11-deploy-bringup.tar.gz -C ~/Pheonix
mkdir -p ~/Pheonix/artifacts/pheonix-we11-today-v0
test -f ~/Pheonix/Deploy/env_init.sh && echo 'Deploy source ready'
```

Deploy 的构建不使用 Conda。首次初始化会安装 ROS 2 Humble、系统依赖、arm64 MNN
runtime，并编译 ROS 2 workspace；过程中会请求 `sudo`：

```bash
cd ~/Pheonix/Deploy
conda deactivate 2>/dev/null || true
./env_init.sh
source /opt/ros/humble/setup.bash
source install/setup.bash
```

`env_init.sh` 会针对 `aarch64` 选择 `mnn-linux-aarch64` runtime，并构建 ROS 包。它不会自动使能电机。

回到开发机，将产物传到刚刚创建好的树莓派目录（把 `<pi-user>` 和 `<pi-host>` 替换为
真实用户名、主机名或 IP）：

```bash
rsync -avP ~/ssd/Pheonix/artifacts/pheonix-we11-today-v0/policy.mnn \
  <pi-user>@<pi-host>:/home/<pi-user>/Pheonix/artifacts/pheonix-we11-today-v0/policy.mnn
```

在树莓派上仅安装已经转换好的 MNN；先备份归档模型，再复制：

```bash
cd ~/Pheonix/Deploy
cp src/inference/models/lab_policy.mnn src/inference/models/lab_policy.mnn.baseline
cp ~/Pheonix/artifacts/pheonix-we11-today-v0/policy.mnn \
  src/inference/models/lab_policy.mnn
sha256sum src/inference/models/lab_policy.mnn
```

重新构建/安装 inference 包，确保 install space 取得新模型：

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select inference --symlink-install
source install/setup.bash
```

在进入真机前，应使用同一段 135D recorded observation 对比 ONNX 与 MNN 的 6D raw action；没有该 parity 记录时，新转换模型只允许进入 standby，不应作为运动策略使用。

## 8. 真机部署：按状态机逐级推进

### 8.1 进入 `start_robot.sh` 前

确认以下条件：

- 机器人架空或可靠固定；急停可达；活动范围清空；
- 供电、限流、CAN 线束、终端电阻、USB-CAN、IMU、Xbox 正常；
- 没有 `candump`、扫频脚本、旧 `motors_node` 或其他 CAN 控制源；
- `motors.yaml` 的电机 ID、方向、默认位、限位和 PD 已与实际硬件核对；
- 1--8 号电机都可响应只读状态探测。

只读 CAN 检查：

```bash
cd ~/Pheonix/Deploy
source /opt/ros/humble/setup.bash
./setup_dual_can.sh
ip -details -statistics link show can0
candump can0
```

退出 `candump` 后再启动控制链路。若任一电机无反馈，**不要**启动 `motors_node` 或 policy。

### 8.2 正式启动

```bash
./start_robot.sh
```

脚本的人工确认顺序不得跳过：

```text
CAN 探测 1--8
  -> IMU 和手柄
  -> 输入 y，启动 1--6 腿轮节点
  -> 核对 /policy/joint_states 后按 Enter
  -> A：翼部慢速到编码器 0 rad
  -> 观察实际机械位置
  -> A：确认翼部进入保持零位
  -> Y：启动推理但保持 standby
  -> 核对 /policy/commands
  -> X：只在架空、零速度时切入 policy
```

今日不需要扑翼：翼部完成回零后保持零位，不按 `B`。当前部署推理仍需要 `/policy/wing_angles` 连续有效，因此不要擅自停掉翼电机节点。

正常退出始终在 `start_robot.sh` 前台终端按 `Ctrl+C`。

## 9. 本轮不做 PACE，但要保留其边界

今天复用的既有参数来自：

```text
真机 paired chirp/sweep CSV
  -> 坐标转换与 200 Hz 重采样
  -> PACE：armature / damping / frictionloss
  -> 固定 PACE 后的 Kp/Kd 搜索
  -> 写入训练 MJCF、task PD 与 command delay
```

本轮不得修改这些参数，也不得把新真机问题直接归因于 PPO。先完成模型/观测/控制 parity；后续若确认真实频响、延迟、摩擦或左右差异有问题，再单独启动 Deploy 扫频与 UniLab PACE 迭代。

## 10. 当日记录与提交

每次运行至少记录：

```text
日期、操作者、机器名、CPU 架构、Ubuntu/ROS 版本
UniLab / Play / Deploy commit
checkpoint、ONNX、MNN SHA-256
task、PD、scale、clip、delay、频率
CAN 1--8 探测结果
IMU/关节/翼角话题频率
standby、架空 policy、低速测试结果
日志、视频、异常和停止原因
```

在各自工作区提交相应改动，不要跨分支混合提交：

```bash
git -C ~/ssd/Pheonix/UniLab status
git -C ~/ssd/Pheonix/Play status
git -C ~/Pheonix/Deploy status
```

本教程的后续阶段才是：去除翼部扑动扰动、重新定义翼角遥控接口、以该新合同训练正式的 no-wing 策略，并在 Play/Deploy 做新的 parity 与真机验收。
