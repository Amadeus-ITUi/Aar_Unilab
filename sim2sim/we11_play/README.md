# WE11 C++ Sim2Sim

这是 WE11 的活动 C++ 回放实现，保留原 `/ssd/Pheonix/Play` 的 MuJoCo
`simulate` 界面、Getup 状态机、手柄组合键、观测历史、PD、动作延迟和控制频率。
源码已经并入本仓库，运行时不读取旧工作区。

## 构建与运行

```bash
cd /ssd/Aar_Unilab
source tools/activate_environment.sh
sim2sim/we11_play/build.sh
sim2sim/we11_play/scripts/play_we11.sh --difficulty 1.0
```

可选的 Getup 课程入口：

```bash
sim2sim/we11_play/scripts/play_we11.sh --difficulty 0.8
sim2sim/we11_play/scripts/play_we11.sh --stage exact_getup
sim2sim/we11_play/scripts/play_we11.sh --stage mixed
```

`--autostart` 仅用于自动化冒烟。正常回放通过键盘或手柄启动。

## 控制

- `1`/Enter 或 `RB+DPadUp`：从 Getup 初始姿态复位并启动策略；
- `P` 或 `LB+X`：停止策略；
- `R` 或 `RB+Y`：复位；
- 左摇杆纵轴：前进速度 `vx`；
- 右摇杆横轴：yaw；
- 右摇杆纵轴：在 0.20–0.30 m 范围内调整机身高度；
- MuJoCo 原生鼠标操作用于摄像机、选中物体及拖动施力。

当前回放不加载翼角 CSV、wrench CSV，不叠加周期外力，也没有扑动频率参数。
翼相关观测位置为保持 145D actor 合同而保留，但活动输入保持为零。

## 运行库

为原样保留经过验证的 `simulate` 界面，WE11 C++ Play 暂时使用 MuJoCo 3.2.7；
ONNX Runtime 为 1.22.0。二进制放在：

```text
/ssd/conda/cache/aar_unilab-native/we11-play-runtime
```

通用 PE01 Sim2Sim 和 Python 训练环境继续使用 MuJoCo 3.8.0。
