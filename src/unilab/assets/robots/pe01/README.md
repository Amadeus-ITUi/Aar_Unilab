# PE01 / DRAGON_3

`dragon_3/` is the final PE01 mechanical source asset imported from
`Point_Phoenix/point-legged/leggedgym/resources/robots/DRAGON_3` on
2026-09-15. The original URDF and STL files are preserved unchanged.

`pe01.xml` is the MuJoCo training model. It preserves the six URDF joint names,
limits, link masses and kinematic offsets. Collision/preview geometry is
primitive because the source CAD STL files exceed MuJoCo's 200,000-face loader
limit; replacing visuals with decimated meshes must not change this dynamics
model. `scene.xml` adds only the world, floor, light and camera.
