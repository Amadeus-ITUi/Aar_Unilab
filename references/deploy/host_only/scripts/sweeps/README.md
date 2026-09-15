# Paired-motor sweeps

The active 1/4, 2/5, and 3/6 guided sweep workflows are grouped here. Each
folder owns its YAML configuration, README, launcher, and generated logs.

Use the canonical launch paths from the repository root:

```bash
./scripts/sweeps/sweep_motor14/run_sweep_motor14.sh
./scripts/sweeps/sweep_motor25/run_sweep_motor25.sh
./scripts/sweeps/sweep_motor36/run_sweep_motor36.sh
```

These scripts retain their existing hardware behavior and operator confirmation
gates.

## Output handoff

Deploy stops at raw acquisition. Preserve every CSV byte-for-byte and create or
update a `SHA256SUMS` file before handing the data to UniLab. Offline parsing,
coordinate conversion, PACE/Kp-Kd fitting and report generation are owned by
`scripts/pace/` in the UniLab branch `Walking_Eagle-Unilab_final`.

The fitting side accepts an explicit `--csv-dir`, so it may read this checkout
directly or consume a verified copy. It must not import Deploy CAN/ROS2 source.
