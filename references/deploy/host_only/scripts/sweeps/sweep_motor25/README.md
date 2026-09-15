# Motor 2/5 Guided Sweep

This folder provides the motor 2/5 configuration and entrypoint for the shared
direct-CAN guided sweep.

Run:

```bash
./scripts/sweeps/sweep_motor25/run_sweep_motor25.sh
```

The wrapper starts or reuses the Xbox `/joy` chain. Xbox A is used for all
three confirmations and must be released between presses:

1. Print the sweep pose, zero offset, raw targets, directions, and one initial
   status result for motors 1..6.
2. First A press: move motors 2/5 slowly to the sweep-specific def-pos using
   `P=1,D=0.1`.
3. Second A press: confirm the pose and hold motors 1/3/4/6.
4. Third A press: start the motor 2/5 chirp at 200 Hz.

Motor 2 uses raw `+sin`; motor 5 uses raw `-sin`. The script never sends a
set-zero command and never accesses motors 7/8. On exit it returns to the
sweep-specific pose, sends `0x04` to disable motors 1..6, and cleans up only
the Xbox process it started.

CSV output is written below `sweep_motor25/logs/`.
