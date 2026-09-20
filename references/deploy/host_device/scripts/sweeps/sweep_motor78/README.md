# P7/P8 ESD-Link wing sweep

Run from the Deploy repository root:

```bash
./scripts/sweeps/sweep_motor78/run_sweep_motor78.sh
```

The launcher sources the ROS workspace, starts an ESD-Link bridge in
`DISABLED` when one is not already running, and starts or reuses the Xbox
`/joy` publisher. It then runs `esd_wing_sweep` and writes the CSV and launcher
logs below this directory by default.

The sweep client still requires its three A-button confirmations. Every
maintenance command covers P1-P8; only P7/P8 targets vary during the sine
sweep. Ctrl-C asks the client to disable all eight ports before the launcher
stops only the bridge and joy processes that it created.

For the canonical fresh-build bench entry use:

```bash
./scripts/sweeps/run_sweep_test_flow.sh 78
```
