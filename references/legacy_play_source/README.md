# Walking Eagle WE11 Play Final

**算法交付负责人：熊铭煊**

**交付日期：2026-08-31**

**正式分支：`pheonix-we11/play-bringup`**

本次交付目的是实现新需求：手动控制上肢机翼扑打角度。倒地自启方面，
已在仿真中成功实现腿部收拢状态下的倒地自启；前倾和后倾状态下的倒地
自启正在实现。

交接范围、接口、安全边界和验证记录见：

- [`docs/handover.md`](docs/handover.md)
- [`docs/interface.md`](docs/interface.md)
- [`docs/test_record.md`](docs/test_record.md)
- [`CHANGELOG.md`](CHANGELOG.md)

这是 WE11 的独立 MuJoCo 手柄回放闭包，只包含 WE11 flat/rough 回放所需内容：

- 当前 `policy.onnx` ONNX policy，接口 `obs[1,145] -> act[1,6]`；
- WE11 MJCF、LOD mesh、flat scene 和 level-9 noise006 地形；
- DR002 共享 `base.yaml` 时序/关节元数据与 WE11 selector 配置；
- 1/2/3 Hz 实测六维力/力矩与翼角观测 CSV；
- MuJoCo 3.2.7、GLFW 3.4 和 ONNX Runtime 1.22.0 x86_64 runtime；
- 仅构建 DR002 `rl_sim_mujoco` 的最小 CMake 工程。

仓库不依赖 `Walking_Eagle-UniLab` 的源码目录，也不包含训练框架或真机 CAN 链路。

## 发布到板端

在 UniLab 将策略发布到本目录并完成 Play 验收后，从 Play 根目录执行：

~~~bash
python scripts/publish_we11_from_play.py --prepare-only  # 只转换和检查，不连接板端
python scripts/publish_we11_from_play.py                 # 板端只读预检
python scripts/publish_we11_from_play.py --apply         # 备份并更新板端模型
~~~

默认板端为 `esd@192.168.8.202:/home/esd/Pheonix/Deploy`。脚本只发布 `policy.onnx`、`lab_policy.mnn` 和 `lab_policy_manifest.json`，且检测到控制进程运行时会拒绝更新；它不会启动/停止节点或使能电机。x64 转换工具会依次从相邻的 `../Deploy`、`../DeployHostOnly/Deploy`、`../DeployHostDevice/Deploy` 自动查找，也可用 `--deploy-tools-root` 覆盖。

## 1. 平台与系统依赖

当前随仓库携带的 runtime 是 Linux x86_64 版本。推荐 Ubuntu 22.04：

~~~bash
sudo apt update
sudo apt install -y build-essential cmake libyaml-cpp-dev libtbb-dev \
  python3-dev python3-numpy libgl1 libx11-6 libxrandr2 libxinerama1 \
  libxcursor1 libxi6
~~~

## 2. 构建

~~~bash
git clone -b pheonix-we11/play-bringup \
  https://git.esdyn.cn/pheonix-eagle/sar_unilab.git Play
cd Play
chmod +x build.sh scripts/play_we11.sh scripts/play_we11_level9.sh
./build.sh
~~~

产物：`build/bin/rl_sim_mujoco`。

## 3. Flat 回放

0 Hz 表示不加载实测外力和翼角 CSV：

~~~bash
./scripts/play_we11.sh 0 flat
./scripts/play_we11.sh 0 getup
./scripts/play_we11.sh 0 --difficulty 0.8
./scripts/play_we11.sh 0 --balance-difficulty 0.8
./scripts/play_we11.sh 0 --stage 1 --difficulty 0.8
./scripts/play_we11.sh 0 --stage mixed
~~~

1/2/3 Hz 会加载仓库内成对 wrench/wing-angle 数据：

~~~bash
./scripts/play_we11.sh 1
./scripts/play_we11.sh 2
./scripts/play_we11.sh 3
~~~

第二个参数选择启动场景：`flat` 从原站立姿势启动，`getup` 从机械限位倒地姿势启动。
两者当前共用同一个平地 XML 和 WE11 策略。

`--difficulty` 可指定连续的 Getup 课程姿态，范围为 `0.0～1.0`：`0` 精确对应 home，`1` 精确对应完整倒地姿态。中间值使用与训练一致的关节插值、机身四元数插值和轮子落地高度计算。该命名参数可放在位置参数前后；若同时给出 `flat/getup`，以 `--difficulty` 为准。

`--balance-difficulty` 从精确 home 下肢姿态启动，并随机选择前倾或后倾；`1` 对应最大 `25°` pitch 和 `1.2 rad/s` pitch-rate。它与 `--difficulty` 互斥。

`--stage` 按 Getup 训练端的阶段与重置比例回放；每次按 `1`/Enter 启动策略都会重新抽取姿态并在终端打印结果：

| 阶段 | 名称 | 训练姿态分布 |
| --- | --- | --- |
| 1 | `home_to_getup` | Home→Getup 路径课程；`--difficulty` 指定当前课程上限，默认 1.0 |
| 2 | `exact_getup` | 80% 精确 Getup，20% 路径姿态 |
| 3 | `getup_with_home` | 60% Getup，30% Home，10% 路径姿态 |
| 4 | `balance` | 40% Getup，20% Home，30% 动态平衡恢复，10% 路径姿态 |
| 5 | `mixed` | 50% Getup，20% Home，20% 动态平衡恢复，10% 路径姿态 |

阶段号和名称都可以使用。阶段回放还会像训练一样独立随机两侧翼角；动态平衡使用训练配置的最大 `16.25°` pitch 与 `0.13 rad/s` pitch-rate。

第三个参数是 wrench 振幅倍率：

~~~bash
./scripts/play_we11.sh 3 flat 1.0
./scripts/play_we11.sh 3 getup 1.0
~~~

原来的 `./scripts/play_we11.sh 3 1.0` 写法仍然兼容。

## 4. Level-9 Rough 回放

默认是 rough、3 Hz、1.0 倍外力：

~~~bash
./scripts/play_we11_level9.sh
~~~

完整参数：

~~~bash
./scripts/play_we11_level9.sh rough 3 1.0
./scripts/play_we11_level9.sh flat 3 1.0
./scripts/play_we11_level9.sh uphill 3 1.0
./scripts/play_we11_level9.sh downhill 3 1.0
~~~

level-9 地形来自 WE11 rough 训练的 seed 42、`noise_range=[0,0.006] m`、`noise_step=0.001 m` 资产。

## 5. 控制与观测合同

- 关节顺序：`[左大腿, 左小腿, 左轮, 右大腿, 右小腿, 右轮]`
- MuJoCo：400 Hz，RK4
- motor PD：200 Hz
- policy：50 Hz
- policy decimation：8 个 400 Hz physics step
- Kp：`[2, 7.59, 0, 2, 7.59, 0]`
- Kd：`[0.08, 0.682, 0.05, 0.08, 0.682, 0.05]`
- action scale：`[0.5, 0.5, 10, 0.5, 0.5, 10]`
- wheel raw action clip：`±3.5`
- command-delay：每次 reset 共享随机 2～8 个 200 Hz motor tick
- 默认 command：`[0, 0, 0.25]`

2026-09-10 的 PACE 辨识结果仅保留在 `we11_pace_params.json` 作为分析记录，
当前 Play 运行模型不注入该组参数，与当前 Getup 训练 checkpoint 保持一致。

1/2/3 Hz 模式向 policy 注入翼角观测，并在 `push_site` 施加经过固定 sensor-to-base 旋转的实测 wrench。WE11 当前 MJCF 没有独立左右翼铰链，因此这里不把“翼角观测注入”描述为可视翼关节驱动。

## 6. 安全边界

本仓库仅用于 MuJoCo 回放，不包含 ROS2/CAN 真机控制入口。运行前可用 `SHA256SUMS` 校验关键 policy、资产和数据文件。

没有手柄时，可仅为自动化冒烟设置 `RL_SAR_PLAY_AUTOSTART=1`；正常手柄回放不要设置该变量。完整命令说明见 `docs/RL_SAR_WE11_USAGE.md`。
