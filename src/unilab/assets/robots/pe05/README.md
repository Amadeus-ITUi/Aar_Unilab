# PE05 冻结本体

2026-09-30：当前资产已同步 PE03 实测限位 v4；髋、大腿、小腿范围分别为
左侧 `[-0.20, 1.50]`、`[-1.44, 0]`、`[-2.02, 0] rad`，右侧按镜像符号。
home、PD、力矩、时序及网络不变。`provenance.json` 保留原始迁移来源，
当前哈希对应实测修订；旧 checkpoint 不可直接用于当前资产，历史发布包保留原样。


直接复制自 PE03 `pe03-cnc-joint-limits-v4`。`provenance.json` 保存源文件 SHA256。
没有软链接；保留原质量、惯量、碰撞、网格、关节轴和限位。
`pe05.xml` 是机器人，`scene.xml` 拥有 home 关键帧；`play_visual.xml` 仅提供回放外观。

home 基座高度为 0.2913829166803791 m。关节顺序为
`L_hip_, L_thigh_, L_calf_, R_hip_, R_thigh_, R_calf_`。
训练、导出及部署 PD 目标均按物理关节范围裁剪。
训练参数与物理适配见 [PE05 基线说明](../../../../../docs/PE05_BASELINE.md)。
