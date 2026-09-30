# PE03 WTW-inspired gait training (v4)

当前资产已于 2026-09-30 同步实测限位 v4，详见 [限位修订](PE03_JOINT_TARGET_LIMITS.md)。
本文原有实验、标定和碰撞数据保留当时版本，不作为新限位的验证结果。

工程验证结果（含已有检查问题）见 [验证记录](PE03_WTW_GAIT_VALIDATION.md)。
训练加速、基准测试和续训方式见 [性能记录](PE03_TRAINING_PERFORMANCE.md)。

This is an independent `pe03_gait_flat` task, not a modification of the old
`standing`/`walking` reward sets. WTW provenance: local `walk-these-ways` at
`0e7236b`, especially `CoRLRewards` and `scripts/train.py`. Formulas and curriculum
behavior are implemented locally; production does not import reference code.

## Run

```bash
# Fixed 2 Hz / 60% support / 3 cm clearance. Defaults: 4096 environments,
# 24 steps per environment, 1000 iterations, evaluate every 100 iterations.
bash tools/train.sh pe03_gait_fixed

# A short engineering check, not a gait-convergence test.
bash tools/train.sh pe03_gait_fixed algo.num_envs=32 algo.max_iterations=5 \
  training.mujoco_threads=4 training.evaluation_interval=0

bash tools/train.sh pe03_gait_fixed mode=play checkpoint=-1

# Exact continuation within the same stage.
bash tools/train.sh pe03_gait_fixed training.resume=/absolute/path/model_1000.pt

# Stage transition: weights transferred; optimizers, curriculum and counters reset.
bash tools/train.sh pe03_gait_variable \
  training.stage_from=/absolute/path/model_1000.pt \
  training.visual_review=/absolute/path/visual_review.json
```

Second-stage training is rejected unless the source is a matching v4 fixed-stage
checkpoint with three consecutive passing evaluations, the last evaluation at
that checkpoint's iteration, and a visual review bound to its SHA256. A review
JSON must contain `checkpoint_sha256`, a nonempty `reviewer`, and all four boolean
fields `no_dragging`, `no_crossing`, `no_nonfoot_support`, `no_frequent_limits` set
to true **after reviewing playback**. Obtain the digest with `sha256sum model_N.pt`.
Short training and automated regression fixtures do not constitute this review.

Logs: `logs/pe03_gait_fixed` and `logs/pe03_gait_variable`. TensorBoard aliases
include robot, task/simulator, experiment and training time. Old PE03 checkpoints
remain playable, but cannot resume into v4. `max_iterations` is a run budget;
exhausting it does not automatically promote the policy to the next stage.

## Observation and control contract

Frame order: gyro(3), projected gravity(3), joint offsets(6), joint velocities(6),
previous action(6), second previous action(6), velocity command(3), gait command(3),
phase sin/cos(2). The 38-dimensional frame is stacked oldest-to-newest for 30
frames. Command order is `[vx, vy, yaw_rate, frequency, support_fraction, clearance]`.
Command scales are `[1,1,1,0.5,1,10]`; joint-velocity scale is 0.1.

- Velocity estimator: 1140 → 256 → 128 → 3, physical body-frame m/s labels.
- Actor: history + detached velocity estimate, 1143 → 512 → 256 → 128 → 6.
- Critic: history + 14 privileged values, 1154 → 512 → 256 → 128 → 1.
- Privileged order: actual velocity(3), height(1), clearances(2), contacts(2),
  body-frame ground reactions divided by weight(6).
- Estimator uses only a separate velocity MSE optimizer; no friction/restitution
  target and no PPO gradient into the estimator. Both policy and value networks
  directly receive the entire history.
- Physics/PD 400 Hz, policy 50 Hz, existing PE03 gains, action processing and
  torque limits. New runs enable PE01-style physical randomization, level-1
  observation noise, pushes and 25–50 ms action delay in both stages; see
  [robustness settings and validation](PE03_GAIT_ROBUSTNESS.md).

ONNX includes the estimator. It accepts `observation_history [1,1140]`,
`observation [1,38]`, `command [1,6]` and emits `action [1,6]`. Current commands
replace the latest frame's command fields. Python and C++ maintain two previous
actions and the same phase/transition sequence. Releases contain the runtime
configuration and offline workspace, so relocation does not require source assets.

## Gait and reward ownership

All task choices and coefficients live in `conf/pe03/task/pe03_gait_flat.yaml`;
experiment selectors are `gait_fixed` and `gait_variable`. Left/right phases differ
by 0.5. Contact probability, triangular swing height, and Raibert xy travel share
the support-fraction mapping. Targets use the home support centers, including
their fore/aft offset and assembly differences.

Reward costs follow WTW (contact force/velocity, clearance, placement, body
height/orientation/motion, effort, smoothness, slip, joint limits, collisions).
Weights are multiplied by policy dt. New runs combine them as
`R_pos * exp(R_neg_other / 0.02) + R_collision`.
Each nonfoot body's collision cost is added independently after the exponential,
so another simultaneous contact still deducts its full cost even when other
penalties have already suppressed the positive reward. The final reward can be
negative; there is no shared collision cap or zero clamp.
Height costs are squared errors in meters, not the old normalized-height reward.
The old air-time/air-height/foot-distance rewards are absent.

Collision configuration is owned by `reward.collision`: `force_threshold` is the
body net-force threshold in N (default 1), `body_multipliers` adjusts individual
nonfoot bodies, and `aggregation` is `additive` by default. Each body's default
cost is `scales.collision * dt = -5 * 0.02 = -0.1` per active policy step.
Two colliding bodies therefore deduct 0.2, three deduct 0.3. Multiple contact
points on one body count once. As before, the body sensors include self contacts;
this change does not add a nonfoot-contact termination timer.

TensorBoard retains `raw_reward/collision` (unweighted body count) and
`reward/collision` (sum of weighted body costs). Individual values are under
`collision/<body>/contact_fraction` and `collision/<body>/weighted_penalty`.
`gait/negative_reward` retains all pre-aggregation negative contributions;
`gait/exponential_negative_reward` includes only terms inside the exponential;
`gait/additive_reward` is the independently added collision contribution.
The per-body diagnostics are not extra reward terms and are never counted twice.

Checkpoints without a `reward.collision` block preserve the original all-negative
exponential formula, including playback and export. Reward changes invalidate
strict resume and stage-transfer contracts: start a fresh run to train the new
objective. To continue an old run with its original objective, explicitly remove
the new block with Hydra `~reward.collision` and keep its other settings identical.
Adding `aggregation: exponential` explicitly also supports controlled comparisons.

WE11 comparison: its `undesired_contacts` term counts configured touch sensors
above 0.1 N, uses scale -10 and adds the result to total reward. However its
current per-term clip is also 10/s, so one and multiple active sensors both reach
-0.2 per 50 Hz step. The PE03 change adopts independent additive costs without
that shared cap; WE11 production behavior is unchanged.

The backend reports world contact-point velocity from rigid-body motion, not the
time derivative of the rolling contact location. Ground reactions are opposite
the force-on-world sign of MuJoCo's `body1=foot, body2=world` netforce sensor.
Only actual sole/world contacts count as foot support. Geometry and sensor
metadata are resolved before stepping.

Rebuild the asset-bound workspace after changing physical assets:

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 /ssd/conda/envs/aar_unilab/bin/python \
  tools/build_pe03_gait_workspace.py --resolution 41
```

The workspace samples each leg with the other at home, rejects penetration and
self collision, and preserves a side-specific foot region. Runtime projection
uses nearest collision-checked samples in `(yaw-frame x, y, sole clearance)`.
It is a sampled static reference envelope, not a guarantee against all dynamic
self collisions; the collision penalty and playback acceptance remain necessary.
Projection distance is logged, and clearance acceptance uses the requested
height, not the projected height.

## Curriculum, reset and acceptance

Velocity commands resample every 10 s, with an independent 20% all-zero quota.
Zero speed retains gait timing. Velocity bins start within ±0.3/±0.1/±0.5 and
expand toward ±1/±0.6/±1. Scores must jointly exceed 0.8/0.7/0.9/0.9; failures
cannot improve the score by shortening the window. Zero commands never expand
velocity bins. Second-stage gait cells span 1.5–3 Hz, 50–70% support, 2–10 cm;
20% retain the fixed gait. Effective parameters transition over 0.5 s without
resetting phase and are included in observations.

Training resets sample joints within home ±0.1 rad and hard limits, root linear
and angular velocities within ±0.5 (m/s and rad/s), and zero joint velocity.
Cached batched sole kinematics align the lowest collision sole 1 mm above ground;
the reward height target stays at nominal home. Ordinary play retains nominal home
and zero velocity. Consecutive body-ground contact, tilt >80°, or height <0.08 m
for 0.1 s terminates; recovery clears the counter.
20-second timeouts bootstrap from the final observation before reset/resampling.

`evaluation_N.json` records fixed-seed randomized full-range episodes, zero/nonzero,
velocity bins and gait bins. It reports success, tracking/contact scores,
contact timing, cycle peak error, duty error, actual touchdown frequency, height
P95, slip, crossings, nonfoot contacts, limits, torque saturation, velocity RMSE
and failure causes. The full numerical gate is in `training.acceptance`.
Evaluation uses the same robustness conditions as training (policy mean actions),
with its protocol fingerprint attached to acceptance records. Old clean evaluation
records cannot qualify a checkpoint for the new randomized variable stage.
Passing a finite set of sampled evaluations is not proof of every possible
velocity/gait combination. Engineering tests and gait acceptance are reported
separately; no pretrained successful gait is provided by this change.
