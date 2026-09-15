# rl_sar WE11 使用命令

本文档对应 GitLab 分支 `Walking_Eagle-Play_final`，用于 WE11 的 MuJoCo flat 与 level-9 rough 手柄回放。所有命令均从仓库根目录执行。

## 1. 获取源码

~~~bash
git clone -b Walking_Eagle-Play_final \
  git@git.esdyn.cn:walking-eagle/sar_unilab.git Walking_Eagle-Play_final
cd Walking_Eagle-Play_final
~~~

如果已经克隆仓库：

~~~bash
git fetch origin
git switch Walking_Eagle-Play_final
~~~

## 2. 安装系统依赖

推荐 Ubuntu 22.04 x86_64：

~~~bash
sudo apt update
sudo apt install -y build-essential cmake libyaml-cpp-dev libtbb-dev \
  python3-dev python3-numpy libgl1 libx11-6 libxrandr2 libxinerama1 \
  libxcursor1 libxi6
~~~

MuJoCo 3.2.7、GLFW 3.4 和 ONNX Runtime 1.22.0 已放在仓库 `library/` 中，不需要再下载。

## 3. 构建

~~~bash
chmod +x build.sh scripts/play_we11.sh scripts/play_we11_level9.sh
./build.sh
~~~

成功后生成：

~~~text
build/bin/rl_sim_mujoco
~~~

需要限制构建线程时：

~~~bash
RL_SAR_BUILD_JOBS=4 ./build.sh
~~~

## 4. Flat 回放

不施加 CSV 外力，也不注入翼角 CSV：

~~~bash
./scripts/play_we11.sh 0 flat
./scripts/play_we11.sh 0 getup
./scripts/play_we11.sh 0 --difficulty 0.8
./scripts/play_we11.sh 0 --balance-difficulty 0.8
./scripts/play_we11.sh 0 --stage 1 --difficulty 0.8
./scripts/play_we11.sh 0 --stage mixed
~~~

加载 1/2/3 Hz 成对 wrench 与翼角观测：

~~~bash
./scripts/play_we11.sh 1
./scripts/play_we11.sh 2
./scripts/play_we11.sh 3
~~~

第二个参数选择启动场景：`flat` 使用原站立姿势，`getup` 使用机械限位倒地姿势；
两者当前共用同一个平地 XML 和 WE11 策略。第三个参数是 wrench 振幅倍率，例如：

命名参数 `--difficulty 0.0～1.0` 可选择连续 Getup 课程姿态：`0` 精确对应 home，`1` 精确对应完整倒地姿态，中间值与训练使用相同的关节/机身插值和轮子落地高度。它可放在位置参数前后，并优先于 `flat/getup`。

命名参数 `--balance-difficulty 0.0～1.0` 用于动态平衡恢复回放；每次 reset 随机选择前倾或后倾，`1` 使用最大 `25°` pitch 和 `1.2 rad/s` pitch-rate。它不能与 `--difficulty` 同时使用。

`--stage` 可按 Getup 训练课程直接回放。阶段号/名称及训练重置分布如下：

| 阶段 | 名称 | 重置分布 |
| --- | --- | --- |
| 1 | `home_to_getup` | Home→Getup 路径课程；`--difficulty` 是课程上限，默认 1.0 |
| 2 | `exact_getup` | 80% 精确 Getup，20% 路径 |
| 3 | `getup_with_home` | 60% Getup，30% Home，10% 路径 |
| 4 | `balance` | 40% Getup，20% Home，30% 动态平衡，10% 路径 |
| 5 | `mixed` | 50% Getup，20% Home，20% 动态平衡，10% 路径 |

每次按 `1`/Enter 启动都会重新采样，终端会打印阶段、实际姿态类别、路径/扰动进度和前后倾方向。阶段回放同时复现训练中的双翼独立随机化；动态平衡上限取训练配置的 `16.25°` 与 `0.13 rad/s`。

~~~bash
./scripts/play_we11.sh 3 flat 0.5
./scripts/play_we11.sh 3 getup 0.5
~~~

旧格式 `./scripts/play_we11.sh 3 0.5` 继续支持。

## 5. Level-9 rough 回放

默认参数为 `rough 3 1.0`：

~~~bash
./scripts/play_we11_level9.sh
~~~

显式启动 level-9 rough、3 Hz、1.0 倍外力：

~~~bash
./scripts/play_we11_level9.sh rough 3 1.0
~~~

同一 level-9 资产还可以查看 flat、上坡和下坡：

~~~bash
./scripts/play_we11_level9.sh flat 3 1.0
./scripts/play_we11_level9.sh uphill 3 1.0
./scripts/play_we11_level9.sh downhill 3 1.0
~~~

不施加 CSV 外力时：

~~~bash
./scripts/play_we11_level9.sh rough 0 1.0
~~~

## 6. 手柄与自动冒烟

正常使用时接入 Linux joystick，程序默认搜索 `/dev/input/js0`～`js3`。也可以明确指定：

~~~bash
RL_SAR_JOYSTICK=/dev/input/js1 ./scripts/play_we11.sh 0
~~~

没有手柄但只想验证 policy 是否能自动启动：

~~~bash
RL_SAR_PLAY_AUTOSTART=1 ./scripts/play_we11.sh 0
RL_SAR_PLAY_AUTOSTART=1 ./scripts/play_we11_level9.sh rough 3 1.0
~~~

`RL_SAR_PLAY_AUTOSTART=1` 仅用于仿真检查；正常手柄操作不要设置。

## 7. 参数含义

`play_we11.sh`：

~~~text
./scripts/play_we11.sh <frequency_hz> <wrench_scale>
frequency_hz: 0、1、2、3
wrench_scale: 外力/力矩六个通道的统一倍率，默认 1.0
~~~

`play_we11_level9.sh`：

~~~text
./scripts/play_we11_level9.sh <terrain> <frequency_hz> <wrench_scale>
terrain: flat、uphill、downhill、rough
frequency_hz: 0、1、2、3
wrench_scale: 默认 1.0
~~~

## 8. 当前回放合同

- policy：`policy/dr002/we11/SOURCE.md` 记录的当前 ONNX 来源
- 输入/输出：`obs[1,145] -> act[1,6]`，历史布局为 29D × 5 的 term-major
- 关节顺序：`[左大腿, 左小腿, 左轮, 右大腿, 右小腿, 右轮]`
- MuJoCo：400 Hz，RK4
- motor PD：200 Hz
- policy：50 Hz
- Kp：`[2,7.59,0,2,7.59,0]`
- Kd：`[0.08,0.682,0.05,0.08,0.682,0.05]`
- 默认 command：`[0,0,0.24]`

1/2/3 Hz 模式会同步加载仓库内的实测 wrench 和翼角观测 CSV。翼角 CSV 写入 policy 观测；当前 WE11 MJCF 没有独立左右翼铰链，因此不会被描述为可视翼关节驱动。

## 9. 常见问题

### 找不到二进制

~~~text
Missing WE11 replay dependency: .../build/bin/rl_sim_mujoco
~~~

执行：

~~~bash
./build.sh
~~~

### 找不到手柄

~~~text
Joystick [/dev/input/js0] open failed
~~~

检查：

~~~bash
ls -l /dev/input/js*
~~~

也可以通过 `RL_SAR_JOYSTICK` 指定设备。

### 找不到 CSV

确认以下文件存在：

~~~bash
ls replay_data/we11/wrench/{1,2,3}hz.csv
ls replay_data/we11/wing_angle/{1,2,3}hz.csv
~~~

### 找不到场景或 mesh

~~~bash
ls src/rl_sar_zoo/dr002_description/mjcf/we11/scene_flat_we11.xml
ls src/rl_sar_zoo/dr002_description/mjcf/we11/scene_level9_noise006_rough_we11.xml
ls src/rl_sar_zoo/dr002_description/mjcf/we11/meshes_lod
~~~

### 校验关键文件

~~~bash
sha256sum -c SHA256SUMS
~~~
