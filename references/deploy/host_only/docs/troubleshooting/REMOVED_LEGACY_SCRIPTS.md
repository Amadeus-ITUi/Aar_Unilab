# 已移除的旧脚本

以下入口不在当前启动链路中，且已有明确替代，因此在本次工作区整理中移除：

| 已移除入口 | 当前替代 |
| --- | --- |
| `back2defpos_45_lowpd.sh` | `scripts/sweeps/sweep_motor14/run_sweep_motor14.sh` |
| `run_sweep_motors_25_1hz.sh` | `scripts/sweeps/sweep_motor25/run_sweep_motor25.sh` |
| `start_lab_model_defpos.sh` | `start_robot.sh` |
| `probe_motor_status_loop.py` | `ros2 run deploy_tools read_motor_status` 或 `read_all_motor_status` |
| `probe_status_0x02_1khz.py` | `ros2 run deploy_tools read_motor_status --request` |
| `zero_and_read_all_motors.py` | `set_zeros_1_6_joy.sh` |
| `motor7_hold_zero_200hz.py` | 正式 `wing_motor_node` 或保留的 7/8 扫频实验 |

移除范围不包含正式启动、CAN 配置、只读监控、运行记录、手柄设零和当前三套
1~6 号电机扫频入口。
