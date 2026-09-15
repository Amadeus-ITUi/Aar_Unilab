# ESD-Link guided sweeps

The active 1/4, 2/5, 3/6 and P7/P8 guided sweep workflows are grouped here.
Each folder owns its configuration or client launcher, README, and generated
logs.

Use the canonical launch paths from the repository root:

```bash
./scripts/sweeps/sweep_motor14/run_sweep_motor14.sh
./scripts/sweeps/sweep_motor25/run_sweep_motor25.sh
./scripts/sweeps/sweep_motor36/run_sweep_motor36.sh
./scripts/sweeps/sweep_motor78/run_sweep_motor78.sh
```

For the bench test flow, run one pair at a time.  In one enable window it
performs current-position takeover, inward-only recovery to the configured
pose, an 8-second low-amplitude sweep, an A-button confirmation, and then the
configured sweep before full disable:

```bash
./scripts/sweeps/run_sweep_test_flow.sh 14
./scripts/sweeps/run_sweep_test_flow.sh 25
./scripts/sweeps/run_sweep_test_flow.sh 36
./scripts/sweeps/run_sweep_test_flow.sh 78
```

The launchers start an ESD-Link bridge when needed and start or reuse the Xbox
`/joy` publisher. They stop only processes created by that launcher. The
operator confirmation gates remain inside the sweep clients.

The current-format ESD data set already contains the default 1/4 `2/0.1`, 2/5
`8/0.8`, and 3/6 `0/0.1` runs. Collect the remaining fixed-gain runs one at a
time, or use `all` to advance through all four (each run retains its own
operator confirmations):

```bash
./scripts/sweeps/run_remaining_gain_sweeps.sh 14-kp4-kd0p2
./scripts/sweeps/run_remaining_gain_sweeps.sh 25-kp4-kd0p2
./scripts/sweeps/run_remaining_gain_sweeps.sh 36-kp0-kd0p05
./scripts/sweeps/run_remaining_gain_sweeps.sh 36-kp0-kd0p2
./scripts/sweeps/run_remaining_gain_sweeps.sh all
```

These overrides do not edit the base YAML or change the CSV schema. Supplying
`--sweep-kp` or `--sweep-kd` automatically names the output from the pair and
effective gains, for example `motor36_YYYYMMDD_HHMMSS_kp0_kd0p05.csv`.
An explicit `--output-file` remains available as an override. P2/P5 keep the
currently validated `+/-0.15 rad` trajectory.

## Output handoff

Deploy stops at raw acquisition. Preserve every CSV byte-for-byte and create or
update a `SHA256SUMS` file before handing the data to UniLab. Offline parsing,
coordinate conversion, PACE/Kp-Kd fitting and report generation are owned by
`scripts/pace/` in the UniLab branch `Walking_Eagle-Unilab_final`.

The fitting side accepts an explicit `--csv-dir`, so it may read this checkout
directly or consume a verified copy. It must not import Deploy CAN/ROS2 source.
