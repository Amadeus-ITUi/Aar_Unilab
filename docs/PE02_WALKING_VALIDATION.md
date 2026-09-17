# PE02 固定初态行走配置

通过 `+experiment=walking` 启用 `conf/pe02/experiment/walking.yaml`。
这是站立验证之后的行走准备配置，继续由 PE02 独立维护环境和网络。
原始迁移配置和已验证的 `standing` 配置保留，便于分别复现。

## 当前设置

- 每次训练、评估、回放均从 `scene.xml` 的 `home` 出发；初始关节扰动和速度为零。
- 关闭观测噪声、物理参数随机化、默认角偏移、IMU 偏移、动作延迟和推力。
- 步态频率 2 Hz、左右相位差 0.5、支撑相比例 0.5、抬脚高度输入 0.06 m。
- 奖励各项及权重完全恢复 PE02 原始迁移配置，包括两项交替接触奖励；不沿用
  站立实验的高度 -200、姿态 -10 权重，而是恢复高度 -3、姿态 -5。
- 终止恢复原来的宽松条件：机身接触力 > 5 N 或投影重力 z > -0.1，累计失败时长
  超过 0.5 秒才终止；中间恢复正常不清空失败累计。正常回合长度为 20 秒。
- 保留原 PD、力矩上限、动作尺度、网络和 PPO 参数。4096 环境，每轮 24 步，
  32 个物理线程；默认最多训练 15000 轮。

“关闭随机化”指环境初态、观测和物理参数。速度命令仍按原始范围采样：前后
±1 m/s、侧向 ±0.6 m/s，启用朝向控制；命令采样有 90% 概率置零，随后由朝向
控制更新转向命令。PPO 动作探索仍然保留。尚未额外加入低速命令课程。
2 Hz 和 6 cm 是给策略的步态条件，并不意味着未训练策略已能实现该步频或抬脚高度。

## 合法接触与宽松终止

仅以下机器人碰撞体作为正常足端接触，免于碰撞惩罚：

- `L_foot_Link_collision_sole_0`
- `R_foot_Link_collision_sole_0`

其余机身、左右髋、大腿、小腿接触均计入 `collision` 惩罚及 `nonfoot_contact`
指标。碰撞惩罚仍使用原始 -3 权重和每个连杆合力 > 1 N 的阈值；左右髋补入了
受罚列表。偶发腿部擦碰会扣分，但不会单独触发立即终止；终止仍遵循上述机身/
大倾角的宽松规则。合法判定针对命名碰撞体整体，不是网格中特定三角面的法向。

当前每只脚都是没有子连杆的末端连杆，且恰好只有一个 sole 碰撞体，因此其连杆
接触传感器与指定 sole 的接触严格对应。后端初始化时检查名称和这个结构约束，
不新增逐步几何解析或额外传感器。若以后往足端添加其他碰撞体，将明确报错，要求
先补充接触分类，避免把新增外壳接触默认为合法支撑。

## 启动

从仓库根目录运行，结果统一存放在 `logs/pe02_walking/<时间戳>/`：

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /ssd/conda/envs/aar_unilab/bin/python scripts/train_pe02.py \
  +experiment=walking training.device=cuda:0
```

回放新训练出的行走 checkpoint 时使用同一配置：

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /ssd/conda/envs/aar_unilab/bin/python scripts/train_pe02.py \
  +experiment=walking mode=play training.device=cpu play.plot=false \
  checkpoint=<行走检查点路径> 'play.command=[0.2,0.0,0.0]'
```

新行走配置从头训练；不能将站立 `model_600.pt` 通过 `training.resume` 严格续训
到不同的任务配置。站立模型仍使用 `+experiment=standing` 回放。
本阶段以固定初态绕开原来的随机姿态穿地问题，尚未实现随机姿态的接地修正。

## 本次验证（2026-09-16）

- 新增 4 项行为测试通过：训练/评估的全量和部分重置、无穿地 home、零物理随机化、
  sole 接触豁免、髋部接触惩罚、宽松累计终止，以及足端几何变更时拒绝错误分类。
- 全仓非 slow 测试 255 项通过、2 项预期失败；其中包括 WE11 平地和粗糙地形的
  配置组合、初始化及步进检查。
- 正式入口完成 256 环境 × 24 步 × 3 轮，共 18432 条样本、60 次 PPO 更新和
  60 次 encoder 更新，保存 checkpoint、TensorBoard 和 ONNX 导出包。
- 新导出包的 Python/C++ 闭环在 1、10、100 步上的动作、力矩和高度最大差值均为 0。
- 本次修改的 Python 文件 Ruff 格式/规则检查通过，`git diff --check` 通过。
  全仓静态检查仍有之前的 12 个格式文件、7 项 Ruff 报错及
  `src/unilab/base/backend/mujoco/playback.py:35` 的 1 项 mypy 报错。

流程验证检查点：
`logs/pe02_walking/2026-09-16_21-30-54_633529_mujoco/model_3.pt`。
导出包：`releases/pe02/pe02_flat/2026-09-16_21-30-54_633529_mujoco/`。
检查日志和计数报告统一位于 `logs/pe02_walking/validation/`。

这 3 轮只验证新配置的完整训练流程，未完成行走长训。该早期策略在零命令评估中
虽能存活至回合上限，但非足端接触帧约 95.8%，不算站稳或行走成功。
后续验收应结合足底支撑、速度跟踪、姿态和抬脚情况，不能仅看宽松条件下的存活时长。
