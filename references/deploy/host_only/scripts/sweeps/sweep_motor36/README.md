# Motor 3/6 MIT Velocity Sweep

This folder provides the wheel motor 3/6 velocity-sweep configuration and
entrypoint for the shared direct-CAN guided sweep.

Run:

```bash
./scripts/sweeps/sweep_motor36/run_sweep_motor36.sh
```

The wrapper starts or reuses the Xbox `/joy` chain. Xbox A is used for all
three confirmations and must be released between presses:

1. Print one initial status sample for motors 1..6, then perform the normal
   feedback check and enable sequence. No hardware zero is written.
2. First A press: hold motors 1/2/4/5 at the sweep pose with `P=1,D=0.1`;
   motors 3/6 receive MIT `Kp=0`, `Kd=0.1`, target velocity `0 rad/s`.
3. Second A press: manually confirm the pose and enter hold.
4. Third A press: start the 3/6 velocity chirp at 200 Hz.

During the sweep, motor 3 receives raw velocity `-sin` and motor 6 receives
raw velocity `+sin`. Their target range is `-5..+5 rad/s`; MIT `Kp=0`, so
the position payload is inactive and control comes from `Kd`.

The script never sends a set-zero command and never accesses motors 7/8. On
exit it sends a final `0 rad/s` target, then `0x04` disable frames for motors
1..6. CSV output is written below `sweep_motor36/logs/` and includes target
and measured position, velocity, torque, and temperature.
