# Walking Eagle WE6

`we6.xml` is the MuJoCo model carrying the nominal Bode-PACE actuator
dynamics identified with a 1000 Hz simulation and 500 Hz motor controller.
The UniLab environment `DR002JoystickFlatWE6` intentionally keeps the training
contract at 400 Hz physics, 400 Hz inner motor control, and 50 Hz policy control.
Identified command delays are quantized to the nearest 400 Hz runtime tick.

The actuator delay is a per-joint **pre-controller command FIFO**.  It is
applied before the PD calculation; torque delay is always zero.  Nominal PD
gains and delay ticks are defined by `WE6ControlConfig` and mirrored by joint
type.  The machine-readable fitted contract is `we6_pace_params.json`.

Identification conditions:

- Thigh and calf use the 2026-07-14 imported sweep set.  In each sweep only
  the excited joint is free; MuJoCo equality constraints hold the other five
  joints at their measured initial coordinates.  The import manifest points
  to the preserved 2026-06-28 raw recordings.
- Foot dynamics use the 2026-07-19 total-assembly motor-side chirp.  Its
  fitted torque stayed below the identification limit, but operation outside
  that small-signal sweep amplitude is an unvalidated extrapolation.
- Left/right armature, viscous damping, Coulomb friction, and delay are
  mirrored by joint type.  Encoder bias/zero offset is not mirrored and is
  not applied by the WE6 runtime because the Bode AC response cannot identify
  a DC coordinate offset.

The model intentionally reuses geometry from
`../walking_eagle_v5_review/meshes_lod`; keep that sibling asset directory
when moving or packaging WE6.
