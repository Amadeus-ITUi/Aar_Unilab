# PE05 冻结本体

直接复制自 PE03 `pe03-cnc-joint-limits-v3`。`provenance.json` 保存源文件 SHA256。
没有软链接；保留原质量、惯量、碰撞、网格、关节轴和限位。
`pe05.xml` 是机器人，`scene.xml` 拥有 home 关键帧；`play_visual.xml` 仅提供回放外观。

home 基座高度为 0.2913829166803791 m。关节顺序为
`L_hip_, L_thigh_, L_calf_, R_hip_, R_thigh_, R_calf_`。
训练、导出及部署 PD 目标均按物理关节范围裁剪。
训练参数与物理适配见 [PE05 基线说明](../../../../../docs/PE05_BASELINE.md)。
