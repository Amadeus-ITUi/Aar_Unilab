# PE04 physical asset snapshot

2026-09-30：当前资产已同步 PE03 实测限位 v4；髋、大腿、小腿范围分别为
左侧 `[-0.20, 1.50]`、`[-1.44, 0]`、`[-2.02, 0] rad`，右侧按镜像符号。
home、PD、力矩、时序及网络不变。`provenance.json` 保留原始迁移来源，
当前哈希对应实测修订；旧 checkpoint 不可直接用于当前资产，历史发布包保留原样。


Independent copy of PE03 `pe03-cnc-joint-limits-v4` for the TRON1-style PE04 task.
`pe04.xml` comes from `pe03_joint_limits.xml`; `scene.xml` comes from
`scene_joint_limits.xml`. Names and local include/package paths are updated;
physical parameters, limits, meshes and the `home` pose are retained.

`provenance.json` records SHA-256 digests of the original XML and meshes.
All assets are ordinary local files. Runtime and build do not resolve PE03 or
`references/` paths. See `docs/PE04_TRON1_MIGRATION.md` for the training contract.
