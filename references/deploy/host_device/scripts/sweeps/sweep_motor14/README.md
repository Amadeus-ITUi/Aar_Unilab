# Motor 1/4 Guided Sweep

This folder contains the direct-CAN guided sweep for motors 1 and 4 only.

It does not start `motors_node` and does not call `/set_zeros`. This 1..6
script performs the normal initial feedback check and enables motors 1..6;
the no-enable/no-detection restriction for motors 7/8 is separate. It does
not use motors 7/8.

Run:

```bash
./scripts/sweeps/sweep_motor14/run_sweep_motor14.sh
```

The wrapper starts `xbox.launch.py` on `/dev/input/js0`, waits until a real
`/joy` message is received, and cleans up only the Xbox process that it
started. If a working `/joy` publisher already exists, it is reused. Override
the device when needed:

```bash
SWEEP_JOY_DEV=/dev/input/js1 ./scripts/sweeps/sweep_motor14/run_sweep_motor14.sh
```

Xbox A (`/joy.buttons[0]`) is used for all three confirmations. The A button
must be released between presses:

1. The script prints one initial state sample for motors 1..6.
2. First press: sends motor 1/4 to the YAML sweep-specific def-pos using
   `P=1,D=0.1`.
3. Second press: confirms the sweep-specific def-pos and enters the hold stage.
4. Third press: applies sweep gains to motors 1/4 and starts the YAML chirp.

The hold gains are sent only to motors 2/3/5/6. The control loop sends MIT
`0x01` frames to motors 1..6 at 200 Hz. It first sends one `0x02` status
request pass, then sends normal `0x03` enable and MIT-mode parameter frames.
On exit it sends `0x04` disable frames when `disable_on_exit` is true.

The sweep pose is calculated as `sweep_base_def_pos_policy_rad +
zero_reference_offset_policy_rad`. Both vectors are printed together with the
final raw encoder targets before the initial status check and before any motor
is enabled.

The CSV is written below `sweep_motor14/logs/` by default. All sweep,
trajectory, gain, CAN, button, and recording parameters are in
`sweep_motor14.yaml`.
