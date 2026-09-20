# PE01 / DRAGON_3

`dragon_3/` is the final PE01 mechanical source asset imported from
`Point_Phoenix/point-legged/leggedgym/resources/robots/DRAGON_3` on
2026-09-15. The original URDF and STL files are preserved unchanged.

`pe01.xml` is the current simplified MuJoCo model. It keeps the six URDF joint
names, limits and kinematic offsets. The trunk has explicit source inertia;
some other link masses/inertias are inferred from primitive collision geometry,
so this model is not an exact CAD dynamics reproduction. Original STL files
exceed MuJoCo's 200,000-face loader limit. `scene.xml` owns the floor, lighting,
camera and the original PE01 training `home` keyframe (0.32 m reset height).

Formal `pe01_v2` training now has independently maintained environment, PPO,
velocity encoder and configuration, without PE02 imports or inheritance.
See the [migration audit](../../../../../docs/PE01_TRAINING_MIGRATION.md).
Old `pe01_legacy` checkpoints retain their previous reset and torque control.
