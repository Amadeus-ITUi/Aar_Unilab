# PE02 / dianzu23

新样机 PE02 的接入框架。2026-09-16 从用户提供的 `dianzu23` 目录复制，
ROS 包名、URDF 文件名和 `package://` 引用统一改为 `pe02`。
`meshes/` 保存原始 STL，校验值记录在 `asset_manifest.json`。

## 独立训练实现

- robot/task：`pe02` / `pe02_flat`。
- 正式观测 `pe02_v2`、网络 `pe02_encoder_mlp`、算法 `pe02_custom_ppo`。
  `pe02_v1` 仅用于旧 checkpoint 和最小适配器兼容。
- 环境 `src/unilab/envs/locomotion/pe02/`，网络 `src/unilab/algos/torch/pe02/`，
  PPO 适配器 `src/unilab/adapters/pe02_ppo.py`，入口 `scripts/train_pe02.py`。
  这些实现均独立，不导入或继承 PE01 的类和函数。
- 配置位于 `conf/pe02/config.yaml` 和 `conf/pe02/task/pe02_flat.yaml`，
  不继承 PE01 配置；奖励、终止条件、控制频率、历史长度和网络层宽可单独修改。
- 单帧 30 维、10 帧历史 300 维、critic 33 维、command 3 维、action 6 维。
- 以保存的原始 PE01 完整训练系统为基准迁移：400 Hz 物理/位置 PD、50 Hz 策略、
  8192 环境 × 24 步、15000 轮，独立速度估计 encoder、18 项奖励与随机化。
  PE02 的模型、home 和配置单独维护；并行规模实测见性能优化报告。
- 动作顺序：`L_hip_`, `L_thigh_`, `L_calf_`, `R_hip_`, `R_thigh_`, `R_calf_`。
  使用原 URDF 的名称、轴向、限位、位姿、质量和惯量。
- checkpoint、回放日志和 release 分别保存到 PE02 命名空间；载入时校验
  checkpoint 的 robot ID，避免误用 PE01 的动作语义。
- checkpoint 内包含完整训练配置，导出和回放从 checkpoint 恢复网络与环境参数。
- C++ 正式控制与观测实现在 `sim2sim/include/aar/pe02_runtime.hpp`；
  `pe02_observation.hpp` 保留 v1 兼容。导出合同提供输入维度和控制参数。
- 共用基础设施仅包括 catalog、MuJoCo backend、交互窗口和 release 文件打包。

正式多环境 PPO、续训及部署链路已验证，尚未完成稳定步态训练或机械验证。
基准对照、差异和启动命令见[完整训练迁移说明](../../../../../docs/PE02_TRAINING_MIGRATION.md)。
默认站姿已根据 URDF 质量和足端网格求解，见 [站姿计算说明](analysis/STANDING_POSE.md)。
机械标定零位和限位保留；控制参数仍需实机标定。

第一阶段站立验证使用 `+experiment=standing`：固定 home、零初速度和零命令，
暂时关闭随机化、噪声、延迟及推力，并使用站立奖励。配置和验收方式见
[站立验证说明](../../../../../docs/PE02_STANDING_VALIDATION.md)。

下一阶段使用 `+experiment=walking`：固定初态、关闭随机化，恢复 2 Hz / 6 cm
步态和原始宽松奖励/终止，仅左右 sole 碰撞体免于接触惩罚。运行方式见
[行走配置说明](../../../../../docs/PE02_WALKING_VALIDATION.md)。

## 资产

网页查看时导入完整 `pe02/` 目录，再选择 `scene.xml` 或 `pe02.xml`。
视觉模型使用 `group="2"`，碰撞体使用 `group="3"`，可分别切换 Show Visual / Show Collision。
不要在全局 `<default><geom>` 上添加 `size`：2026-09-16 的 URDF Studio 会把
继承的 `size="0.025"` 错当作 mesh 的三轴缩放 `(0.025, 0, 0)`，使视觉网格和
碰撞凸包都不可见，只剩盒体和圆柱。MuJoCo 会忽略 mesh 的 `size`；所有基础碰撞体
已分别指定尺寸，因此移除该默认值不改变物理模型。STL 文件和 `meshdir` 路径无需修改。
网站选择 `pe02.xml` 时显示机械零位；选择 `scene.xml` 时会应用其中的 `home`
关键帧，显示训练默认站姿。站姿角度见 [站姿计算说明](analysis/STANDING_POSE.md)。

- `urdf/pe02.urdf`：保留源 URDF 的运动学、质量、重心、惯量和视觉引用；
  collision 段已替换为专用碰撞体，与 MJCF 同步。
- `meshes/`：未改动的源 STL。
- `pe02.xml`：由 URDF 转换的独立 MuJoCo 模型；控制默认值已复制到本文件。
- `scene.xml`：独立维护的地面、光源、相机和任务站姿 `home` 关键帧。
- `play_visual.xml`：Python 正式回放专用外观，设置灰白棋盘地面、浅蓝亮色天空和补光。
  在回放构造阶段合入场景；可修改其中的颜色和 `texrepeat` 调整棋盘密度。
  训练不加载此文件，已有检查点的续训资产校验不受其修改影响。
- `runtime_meshes/`：原始视觉网格转成的 OBJ，完整保留每个三角面的坐标和绕序，
  不再减面。源机身有 377,070 个面，超过 MuJoCo 单个 STL 的 200,000 面限制；
  OBJ 可正常加载。早期每个最多 20,000 面的 STL 视觉版本已停用。
  这些网格仅用于显示；正式并行训练在编译前移除视觉 geom 和对应网格资产，
  因此恢复外观不会增加训练物理模型的内存，碰撞体与质量惯量也不受影响。
  网页和可视化回放需要加载更多三角面，加载及渲染开销会增加。
- `collision_recipe.json`：PE02 专用的 CAD 部件选择、基础体拟合和小腿分段参数。
- `collision_meshes/`：从原始 CAD 生成的凸包；左右成对部件镜像生成。
  当前 v2 使用有误差上限的减面，并保留足端默认接触面附近的原始顶点。
  盒体与圆柱体直接写在 URDF/MJCF 中。生成方法与限制见
  [碰撞体说明](analysis/COLLISION_GEOMETRY.md)。
- 内存和完整训练吞吐见 [性能优化报告](../../../../../docs/PE02_COLLISION_OPTIMIZATION.md)。
- `analysis/standing_pose.json`：默认姿态、重心、接触几何和来源校验值。
- `analysis/standing_pose.png`：侧视姿态和重心投影图。
- `config/`、`launch/`、`package.xml`、`CMakeLists.txt` 和 CSV：原导出包附属文件，
  已统一包名；训练和回放入口不依赖 ROS launch 文件。

重建 MuJoCo 资产需要额外的离线工具 `trimesh`；减面模式另需 `fast-simplification`。
正常训练不需要安装它们。从仓库根目录执行：

```bash
/ssd/conda/envs/aar_unilab/bin/python -m unilab.base.backend.mujoco.urdf_import \
  src/unilab/assets/robots/pe02/urdf/pe02.urdf \
  --control-template src/unilab/assets/robots/pe02/pe02.xml \
  --full-resolution-visuals --source-name dianzu23
```

上述转换保留专用碰撞网格，不对其减面。如需修改碰撞拟合方案，先编辑
`collision_recipe.json`，再执行 `tools/build_pe02_collisions.py`；该脚本同步更新
URDF 和 MJCF 的碰撞定义，不修改质量惯量。发布包会包含 `collision_meshes/`。
