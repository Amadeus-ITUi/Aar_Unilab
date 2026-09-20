# PE03 CNC 点足模型

源目录：`/home/angela/下载/点足CNC`。未经修改的 URDF 存于
`source/original.urdf`；`meshes/` 保留原始 STL 字节，校验值与唯一质量修正在
`asset_manifest.json` 中。右小腿从 0.098 kg 修正为 0.118 kg，左右均为
0.118 kg；全机 3.26689 kg。新 URDF 的质心、惯量、机械零位、关节轴、限位和装配位置保留。

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

2026-09-20 关节限位修正：新 `pe03_gait_flat` 使用
`scene_joint_limits.xml` / `pe03_joint_limits.xml`，对应
`urdf/pe03_joint_limits.urdf`。按用户选择，训练和控制将内收限制到 11.1°：
左髋为 `[-0.19373154697137057, 1.57] rad`，右髋按镜像为
`[-1.57, 0.19373154697137057] rad`，大腿和小腿范围保留。原 `scene.xml`、`pe03.xml`、
`urdf/pe03.urdf` 和旧 workspace 保留，以便旧 checkpoint 精确回放；
`source/original.urdf` 仍是原始来源文件。

新任务开启 `control.clip_joint_targets`，最终 PD 目标按实际加载模型的逐关节范围裁剪。
机械范围内仍可能自碰撞。`analysis/joint_limits/` 保存左腿、右腿和双腿镜像内收的
25×25 腿形扫描（收紧前，向内扫描至 0.785 rad 的机械范围）；这是固定其余自由度的
几何切片，不是全姿态自动避碰控制器。
使用 `python tools/audit_pe03_joint_limits.py` 重建扫描，使用
`python tools/build_pe03_gait_workspace.py` 重建新模型对应的足端 workspace。

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
