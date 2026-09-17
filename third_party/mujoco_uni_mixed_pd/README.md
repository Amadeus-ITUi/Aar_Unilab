# UniLab MuJoCo mixed-PD extension

This patch adds one private `mujoco-uni 3.8.0` batch step that evaluates the
DR002 leg position PD and wheel velocity PD inside each MuJoCo physics substep,
then advances a persistent per-joint post-clip motor-torque FIFO.
It also adds a separate command-delay entry point for 400 Hz physics with a
decimated 200 Hz motor controller. That path advances the pre-controller FIFO
and recomputes PD only on motor ticks, while holding torque on the intervening
physics step.
The independent `step_joint_position_pd` entry point evaluates joint-coordinate
position PD, a pre-controller angle-offset FIFO, and wrapped finite-difference
velocity in one batch call. PE02 enables this with `training.native_pd=true`.
It uses caller-supplied joint addresses and actuator gears, preserves the final
force-cleared sensor refresh, and imposes no robot-specific dimensions. PE01
continues using its existing path; the WE11 entry points are unchanged.
It does not replace the Conda environment's official `mujoco._batch_env` module.

Build it in the active UniLab environment:

```bash
./third_party/mujoco_uni_mixed_pd/build_extension.sh
```

The generated `_unilab_batch_env` binary is placed under
`src/unilab/base/backend/mujoco/_native/` and is intentionally ignored by Git
because it is Python/architecture specific. V5 uses it when
`env.control_config.use_native_batched_pd=true`. If the binary is absent or
cannot be imported, the backend warns once and uses the legacy per-substep path.
The PE02 backend similarly warns and falls back when the joint-position entry
point is unavailable. Re-run the build script after updating the extension;
its capability check detects binaries that lack the new entry point.

The build script verifies the pinned source archive checksum. Optional
`UNILAB_ABSEIL_SOURCE` and `UNILAB_PYBIND11_SOURCE` variables can point to local
dependency checkouts for an offline build.
