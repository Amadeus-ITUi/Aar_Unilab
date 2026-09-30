# PE03 CNC 点足模型

源目录：`/home/angela/下载/点足CNC`。未经修改的 URDF 存于
`source/original.urdf`；`meshes/` 保留原始 STL 字节，校验值与唯一质量修正在
`asset_manifest.json` 中。右小腿从 0.098 kg 修正为 0.118 kg，左右均为
0.118 kg；全机 3.26689 kg。新 URDF 的质心、惯量、机械零位、关节轴和装配位置保留；限位已按 2026-09-30 样机实测更新。

- `urdf/pe03.urdf`：包名统一后的 URDF，完整视觉与简化碰撞体。
- `pe03.xml`：MuJoCo 机器人，显式惯量，完整 OBJ 视觉与同一组碰撞体。
- `scene.xml`：地面与重新求解的 `home`。浏览器查看时导入整个目录并选择此文件。
- `runtime_meshes/`：原 STL 完整转换的 OBJ，共 1,108,802 面；没有视觉减面。
- `collision_meshes/`、`collision_recipe.json`：26 个几何体的独立配方与凸包。
- `play_visual.xml`：Python 回放使用的棋盘格地面与亮色天空；训练编译不加载视觉。
- `analysis/standing_pose.json`、`standing_pose.png`：站姿、支撑力矩与姿态图。
- `analysis/collision_audit.json`、`collision_comparison.png`：碰撞检查和前后对比图。
- `analysis/component_mapping.json`：新 CAD 组件与碰撞配方的重新匹配记录。
- `analysis/training_source_manifest.json`：复制时的 PE02 工作区文件指纹。

默认 base link 高度 0.2913829166803791 m；六关节顺序为左髋、左大腿、左小腿、
右髋、右大腿、右小腿。角度约为
`[1.326962, -17.898833, -71.856263, -1.326962, 17.899525, 71.855361]` 度。
未强制镜像原装配的微小左右差异。

## 2026-09-30 实测限位：当前 v4

PE03 standing/walking、固定/可调步态及 PE04、PE05 统一使用以下实测机械范围。
单位 rad，沿用原关节零位、关节顺序及轴方向，最终 PD 目标使用同一范围。

| 关节 | 下限 | 上限 |
| --- | ---: | ---: |
| `L_hip_` | -0.20 | 1.50 |
| `L_thigh_` | -1.44 | 0.00 |
| `L_calf_` | -2.02 | 0.00 |
| `R_hip_` | -1.50 | 0.20 |
| `R_thigh_` | 0.00 | 1.44 |
| `R_calf_` | 0.00 | 2.02 |

资产版本分别为 `pe03-cnc-joint-limits-v4`、`pe04-pe03-cnc-joint-limits-v4`、
`pe05-pe03-cnc-joint-limits-v4`。PE03 两套 XML/URDF 入口均已同步。
PE03 v2/v3 也启用 `control.clip_joint_targets`：在动作与力矩估计预裁剪之后，
裁剪包含电机零位偏置的最终绝对目标，动作历史及动作变化奖励使用裁剪后的动作。
各任务软限位奖励比例、home、质量/惯量、网格、PD、动作尺度和时序保持原值。

PE03 两份足端 workspace 均按各自 scene、每轴 41 点重新生成。运行
`/ssd/conda/envs/aar_unilab/bin/python tools/build_pe03_gait_workspace.py` 重建两份，
可用 `--variant gait` 或 `--variant flat` 选择单份。
当前碰撞扫描见 `analysis/joint_limits_v4/`，旧 `analysis/joint_limits/` 保留历史值。
扫描工具从模型读取六关节范围，仍只描述固定其余自由度或双腿镜像的几何切片，
不保证任意六关节组合无碰撞。此前髋内收约 45° 的推定不再作为当前机械范围依据。

新旧资产指纹不同，旧 checkpoint 不能直接 strict resume、回放或导出到当前资产。
需使用旧版本资产/代码复现，或启动新训练；不修改历史 checkpoint、日志和已发布包。
新 PE03 v2/v3/v4 发布包均使用 `pe03.runtime.vN.joint-limits.v1`，导出逐关节边界，
C++ 校验其与模型及 home 一致；旧自包含发布包继续支持原协议。
PE04 保持其现有导出边界校验，PE05 从包内模型读取范围。ONNX 接口和网络维度不变。

足端是曲面，0.5 mm 截面积是几何贴地指标，不是刚性接触面积或材料形变模型。
凸包减面误差上限 0.5 mm 是相对分组后的凸包，不能解读为简化几何与整个凹形 CAD
表面的距离上限。默认支撑平面附近 1 mm 凸包顶点保持原样。

重建顺序（仓库根目录，先激活受支持环境）：

```bash
python -m unilab.base.backend.mujoco.urdf_import \
  src/unilab/assets/robots/pe03/urdf/pe03.urdf \
  --control-template src/unilab/assets/robots/pe03/pe03.xml \
  --full-resolution-visuals --source-name 点足CNC
python tools/build_pe03_collisions.py
MUJOCO_GL=egl python tools/solve_pe03_standing_pose.py --write --render
MUJOCO_GL=egl python tools/audit_pe03_collisions.py --render
```

修改资产后应开始新训练；续训会校验资产指纹。控制参数为 PE02 的初始复制值，
尚未对 CNC 样机做参数辨识。完整训练、回放与验收结果见
[PE03 训练说明](../../../../../docs/PE03_TRAINING_MIGRATION.md)。
