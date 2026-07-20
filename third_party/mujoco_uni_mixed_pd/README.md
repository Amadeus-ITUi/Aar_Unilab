# UniLab MuJoCo mixed-PD extension

This patch adds one private `mujoco-uni 3.8.0` batch step that evaluates the
DR002 leg position PD and wheel velocity PD inside each MuJoCo physics substep.
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

The build script verifies the pinned source archive checksum. Optional
`UNILAB_ABSEIL_SOURCE` and `UNILAB_PYBIND11_SOURCE` variables can point to local
dependency checkouts for an offline build.
