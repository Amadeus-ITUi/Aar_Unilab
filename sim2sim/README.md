# C++ Sim2Sim

This directory is the maintained, robot-neutral C++ runtime boundary. It reads
only `aar-unilab.actor.v1` releases and fails closed on unknown schemas. Tensor
names, shapes, history, hidden state, joints and control rates come from the
release manifest; they are not compile-time WE11 constants.

The UI/interaction implementation imported from the former Play workspace is
kept under `references/legacy_play_source/`. Its MuJoCo viewer integration,
mouse perturbation, joystick input, matplotlibcpp plots, camera controls and
diagnostic logging are the reference for the next runtime host. Training-only
reward, critic, curriculum and PPO code must not enter this target.

Build the dependency-free contract layer with:

```bash
cmake -S sim2sim -B sim2sim/build
cmake --build sim2sim/build --parallel
ctest --test-dir sim2sim/build --output-on-failure
```

MuJoCo 3.8.0 and ONNX Runtime 1.22.0 runtime distributions are installed below
`/ssd/conda/cache/aar_unilab-native`; no runtime binaries are committed.
